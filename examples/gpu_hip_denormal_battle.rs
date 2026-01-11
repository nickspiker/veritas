//! The Ultimate Showdown: IEEE f32 with Denormals vs Spirix F4E4
//!
//! This test creates REAL denormals and forces IEEE to handle them properly.
//! - Spirix: Zero branch divergence (vanished values = integers)
//! - IEEE: Branch divergence hell (320+ cycles per denormal)
//!
//! Strategy:
//! - Use f32::MIN_POSITIVE * fraction to create TRUE denormals
//! - Ensure 1 denormal per 32-thread wave (worst case for IEEE)
//! - Spirix should WIN due to zero divergence

use spirix::{ScalarF4E4, Tensor};
use std::time::Instant;

// Declare external HIP functions
extern "C" {
    fn spirix_matmul_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        c_frac: *mut i16, c_exp: *mut i16,
        m: i32, n: i32, k: i32,
    );

    fn ieee_matmul_denormal_hip(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: i32, n: i32, k: i32,
    );
}

fn matmul_spirix_hip(a: &Tensor<ScalarF4E4>, b: &Tensor<ScalarF4E4>) -> Tensor<ScalarF4E4> {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    // Separate fraction and exponent
    let mut a_frac = Vec::with_capacity(m * k);
    let mut a_exp = Vec::with_capacity(m * k);
    for val in &a.data {
        a_frac.push(val.fraction);
        a_exp.push(val.exponent);
    }

    let mut b_frac = Vec::with_capacity(k * n);
    let mut b_exp = Vec::with_capacity(k * n);
    for val in &b.data {
        b_frac.push(val.fraction);
        b_exp.push(val.exponent);
    }

    let mut c_frac = vec![0i16; m * n];
    let mut c_exp = vec![0i16; m * n];

    unsafe {
        spirix_matmul_hip(
            a_frac.as_ptr(), a_exp.as_ptr(),
            b_frac.as_ptr(), b_exp.as_ptr(),
            c_frac.as_mut_ptr(), c_exp.as_mut_ptr(),
            m as i32, n as i32, k as i32,
        );
    }

    // Reconstruct Spirix values
    let data: Vec<ScalarF4E4> = c_frac.iter().zip(c_exp.iter())
        .map(|(&f, &e)| ScalarF4E4 { fraction: f, exponent: e })
        .collect();

    Tensor::new(data, vec![m, n])
}

// IEEE f32 matmul using HIP GPU with denormal support
fn matmul_ieee_hip(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    unsafe {
        ieee_matmul_denormal_hip(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as i32, n as i32, k as i32,
        );
    }

    c
}

fn is_denormal(x: f32) -> bool {
    x != 0.0 && x.abs() < f32::MIN_POSITIVE
}

fn create_denormal_infested_data(size: usize, denormal_rate: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    let batch_size = 32; // RDNA2 wave size

    for batch in 0..(size / batch_size) {
        for i in 0..batch_size {
            let idx = batch * batch_size + i;

            // Force at least one denormal per wave (thread 0)
            let force_denormal = i == 0;

            let val = if force_denormal || (idx as f32 / size as f32) < denormal_rate {
                // Create TRUE denormal: f32::MIN_POSITIVE * small_fraction
                let fraction = 0.1 + (idx as f32 % 100.0) / 1000.0; // 0.1 to 0.2
                f32::MIN_POSITIVE * fraction
            } else {
                // Normal value
                ((idx % 100) as f32 - 50.0) * 0.01
            };

            data.push(val);
        }
    }

    // Fill remainder
    for i in (size / batch_size * batch_size)..size {
        data.push(((i % 100) as f32 - 50.0) * 0.01);
    }

    data
}

fn main() {
    println!("=== The Ultimate Denormal Battle ===\n");
    println!("IEEE f32 vs Spirix F4E4 with REAL denormals");
    println!("Expected: Spirix WINS due to zero branch divergence\n");

    let size = 512;

    println!("Creating denormal-infested matrices...");

    // Scenario 1: Light denormals (10%)
    let a1_ieee = create_denormal_infested_data(size * size, 0.1);
    let b1_ieee = create_denormal_infested_data(size * size, 0.1);

    let denormal_count_a = a1_ieee.iter().filter(|&&x| is_denormal(x)).count();
    let denormal_count_b = b1_ieee.iter().filter(|&&x| is_denormal(x)).count();

    println!("Scenario 1 (10% denormals):");
    println!("  Matrix A: {} denormals ({:.1}%)",
        denormal_count_a,
        (denormal_count_a as f64 / a1_ieee.len() as f64) * 100.0);
    println!("  Matrix B: {} denormals ({:.1}%)",
        denormal_count_b,
        (denormal_count_b as f64 / b1_ieee.len() as f64) * 100.0);

    // Verify they're REAL denormals
    let sample = a1_ieee.iter().find(|&&x| is_denormal(x)).unwrap();
    println!("  Sample denormal: {:.6e} (bits: {:032b})", sample, sample.to_bits());
    println!("  f32::MIN_POSITIVE: {:.6e}\n", f32::MIN_POSITIVE);

    // Convert to Spirix
    let a1_spirix: Vec<ScalarF4E4> = a1_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b1_spirix: Vec<ScalarF4E4> = b1_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    let a1_tensor = Tensor::new(a1_spirix, vec![size, size]);
    let b1_tensor = Tensor::new(b1_spirix, vec![size, size]);

    println!("Running benchmarks...\n");

    // Warmup
    let _ = matmul_spirix_hip(&a1_tensor, &b1_tensor);
    let _ = matmul_ieee_hip(&a1_ieee, &b1_ieee, size, size, size);

    // Spirix HIP
    let spirix_start = Instant::now();
    let _spirix_result = matmul_spirix_hip(&a1_tensor, &b1_tensor);
    let spirix_time = spirix_start.elapsed();

    // IEEE HIP (with denormals)
    let ieee_start = Instant::now();
    let ieee_result = matmul_ieee_hip(&a1_ieee, &b1_ieee, size, size, size);
    let ieee_time = ieee_start.elapsed();

    // Check if denormals survived
    let output_denormals = ieee_result.iter().filter(|&&x| is_denormal(x)).count();
    let output_zeros = ieee_result.iter().filter(|&&x| x == 0.0).count();

    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    let winner = if ratio < 1.0 { "🚀 SPIRIX WINS!" } else { "IEEE wins" };

    println!("{:<20} Spirix: {:>8.3?}  IEEE: {:>8.3?}  Ratio: {:.2}x  {}",
        "10% Denormal", spirix_time, ieee_time, ratio, winner);
    println!("  IEEE output: {} denormals, {} zeros", output_denormals, output_zeros);

    // Scenario 2: Heavy denormals (30%)
    println!("\nScenario 2 (30% denormals):");
    let a2_ieee = create_denormal_infested_data(size * size, 0.3);
    let b2_ieee = create_denormal_infested_data(size * size, 0.3);

    let denormal_count_a = a2_ieee.iter().filter(|&&x| is_denormal(x)).count();
    println!("  Matrix A: {} denormals ({:.1}%)",
        denormal_count_a,
        (denormal_count_a as f64 / a2_ieee.len() as f64) * 100.0);

    let a2_spirix: Vec<ScalarF4E4> = a2_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b2_spirix: Vec<ScalarF4E4> = b2_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    let a2_tensor = Tensor::new(a2_spirix, vec![size, size]);
    let b2_tensor = Tensor::new(b2_spirix, vec![size, size]);

    // Warmup
    let _ = matmul_spirix_hip(&a2_tensor, &b2_tensor);
    let _ = matmul_ieee_hip(&a2_ieee, &b2_ieee, size, size, size);

    let spirix_start = Instant::now();
    let _spirix_result = matmul_spirix_hip(&a2_tensor, &b2_tensor);
    let spirix_time = spirix_start.elapsed();

    let ieee_start = Instant::now();
    let ieee_result = matmul_ieee_hip(&a2_ieee, &b2_ieee, size, size, size);
    let ieee_time = ieee_start.elapsed();

    let output_denormals = ieee_result.iter().filter(|&&x| is_denormal(x)).count();
    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    let winner = if ratio < 1.0 { "🚀 SPIRIX WINS!" } else { "IEEE wins" };

    println!("{:<20} Spirix: {:>8.3?}  IEEE: {:>8.3?}  Ratio: {:.2}x  {}",
        "30% Denormal", spirix_time, ieee_time, ratio, winner);
    println!("  IEEE output: {} denormals", output_denormals);

    println!("\n=== Analysis ===");
    println!("If Spirix wins:");
    println!("  ✓ Branch divergence is REAL");
    println!("  ✓ Spirix's vanished handling = zero divergence");
    println!("  ✓ Verified computation can be FASTER than IEEE cheating\n");

    println!("If IEEE wins:");
    println!("  - Either using FTZ mode (flushing denormals)");
    println!("  - Or GPU handles denormals better than expected");
    println!("  - Check output_denormals count to verify\n");
}
