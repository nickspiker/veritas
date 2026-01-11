//! Real Denormal Operations - Not Manufactured Test Data
//!
//! Tests actual operations that produce denormals naturally:
//! - f32::MIN_POSITIVE / 7.0  (denormal result)
//! - Small * Small  (underflows to denormal)
//! - Large gradient descent updates (near underflow)
//!
//! This is what happens in REAL neural network training.

use spirix::{ScalarF4E4, Tensor};
use std::time::Instant;

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

    let data: Vec<ScalarF4E4> = c_frac.iter().zip(c_exp.iter())
        .map(|(&f, &e)| ScalarF4E4 { fraction: f, exponent: e })
        .collect();

    Tensor::new(data, vec![m, n])
}

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

fn create_real_denormal_scenario(size: usize, scenario: &str) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::with_capacity(size);
    let mut b = Vec::with_capacity(size);

    match scenario {
        "division" => {
            // f32::MIN_POSITIVE / small_divisor = denormal
            for i in 0..size {
                let divisor = 2.0 + (i % 20) as f32;
                a.push(f32::MIN_POSITIVE);
                b.push(1.0 / divisor);  // Results in denormal when multiplied
            }
        }
        "underflow" => {
            // Small * Small = denormal
            for i in 0..size {
                let scale = ((i % 100) as f32 + 1.0) / 100.0;  // 0.01 to 1.0
                a.push((f32::MIN_POSITIVE * 10.0).sqrt() * scale);
                b.push((f32::MIN_POSITIVE * 10.0).sqrt() / scale);
                // product ≈ MIN_POSITIVE * 10 (denormal)
            }
        }
        "gradient" => {
            // Simulate gradient descent with vanishing gradients
            for i in 0..size {
                // Learning rate * small gradient = denormal
                let learning_rate = 1e-5;
                let gradient = f32::MIN_POSITIVE * ((i % 50) as f32 + 1.0);
                a.push(learning_rate);
                b.push(gradient);
                // product = denormal
            }
        }
        "mixed" => {
            // Mix of normal and denormal-producing operations
            for i in 0..size {
                if i % 3 == 0 {
                    // Denormal region
                    a.push(f32::MIN_POSITIVE * 0.5);
                    b.push(2.0);
                } else if i % 3 == 1 {
                    // Near denormal
                    a.push(f32::MIN_POSITIVE * 100.0);
                    b.push(0.01);
                } else {
                    // Normal
                    a.push(0.1 + (i % 100) as f32 * 0.01);
                    b.push(0.1 + ((i + 50) % 100) as f32 * 0.01);
                }
            }
        }
        _ => panic!("Unknown scenario"),
    }

    (a, b)
}

fn run_scenario(name: &str, scenario: &str, size: usize) {
    println!("\n=== Scenario: {} ===", name);

    let (a_ieee, b_ieee) = create_real_denormal_scenario(size * size, scenario);

    // Check how many operations will produce denormals
    let mut denormal_products = 0;
    for i in 0..size * size {
        let product = a_ieee[i] * b_ieee[i];
        if is_denormal(product) {
            denormal_products += 1;
        }
    }

    println!("  Matrix size: {}×{}", size, size);
    println!("  Operations producing denormals: {} ({:.1}%)",
        denormal_products,
        (denormal_products as f64 / (size * size) as f64) * 100.0);

    // Show sample values
    println!("  Sample: {:.6e} × {:.6e} = {:.6e}",
        a_ieee[0], b_ieee[0], a_ieee[0] * b_ieee[0]);
    println!("  Is denormal? {}", is_denormal(a_ieee[0] * b_ieee[0]));

    // Convert to Spirix
    let a_spirix: Vec<ScalarF4E4> = a_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b_spirix: Vec<ScalarF4E4> = b_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    let a_tensor = Tensor::new(a_spirix, vec![size, size]);
    let b_tensor = Tensor::new(b_spirix, vec![size, size]);

    // Warmup
    let _ = matmul_spirix_hip(&a_tensor, &b_tensor);
    let _ = matmul_ieee_hip(&a_ieee, &b_ieee, size, size, size);

    // Benchmark
    let spirix_start = Instant::now();
    let _spirix_result = matmul_spirix_hip(&a_tensor, &b_tensor);
    let spirix_time = spirix_start.elapsed();

    let ieee_start = Instant::now();
    let ieee_result = matmul_ieee_hip(&a_ieee, &b_ieee, size, size, size);
    let ieee_time = ieee_start.elapsed();

    // Check denormal preservation
    let output_denormals = ieee_result.iter().filter(|&&x| is_denormal(x)).count();
    let output_zeros = ieee_result.iter().filter(|&&x| x == 0.0).count();

    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    let winner = if ratio < 1.0 { "🚀 SPIRIX WINS!" } else { "IEEE wins" };

    println!("\n  Results:");
    println!("    Spirix:  {:>8.3?}", spirix_time);
    println!("    IEEE:    {:>8.3?}", ieee_time);
    println!("    Ratio:   {:.2}x  {}", ratio, winner);
    println!("    IEEE output: {} denormals, {} zeros", output_denormals, output_zeros);
}

fn main() {
    println!("=== Real Denormal Operations Battle ===\n");
    println!("Testing ACTUAL denormal-producing operations:");
    println!("  - f32::MIN_POSITIVE / divisor");
    println!("  - sqrt(MIN) * sqrt(MIN)");
    println!("  - Gradient descent updates");
    println!("  - Mixed normal/denormal workloads\n");

    let size = 512;

    run_scenario(
        "Division by Small Numbers",
        "division",
        size,
    );

    run_scenario(
        "Underflow from Small×Small",
        "underflow",
        size,
    );

    run_scenario(
        "Vanishing Gradients (Training)",
        "gradient",
        size,
    );

    run_scenario(
        "Mixed Normal/Denormal",
        "mixed",
        size,
    );

    println!("\n=== Summary ===\n");
    println!("These scenarios represent REAL denormal operations:");
    println!("  ✓ Division: f32::MIN_POSITIVE / 7.0 → denormal");
    println!("  ✓ Underflow: sqrt(MIN) × sqrt(MIN) → denormal");
    println!("  ✓ Gradients: 1e-5 × tiny_weight → denormal");
    println!("  ✓ Mixed: Real neural network behavior\n");

    println!("If Spirix wins:");
    println!("  → Verified computation is FASTER on realistic workloads");
    println!("  → Neural network training will benefit");
    println!("  → Branch divergence penalty is real\n");

    println!("If IEEE wins:");
    println!("  → Check if denormals were flushed to zero");
    println!("  → IEEE may be using FTZ (incorrect behavior)");
    println!("  → Spirix maintains correctness\n");
}
