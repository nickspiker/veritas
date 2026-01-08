//! Denormal Torture Test: Force IEEE to Handle Real Denormals
//!
//! Creates guaranteed denormal values using f32::MIN * random fractions.
//! Ensures at least one denormal per batch to maximize branch divergence.
//! Tests both multiply AND add to see which operation IEEE struggles with.

use spirix::{ScalarF4E4, Tensor};
use veritas::gpu::matmul_gpu_opencl;
use std::time::Instant;
use ocl::{ProQue, Buffer, MemFlags};

fn matmul_f32_gpu_opencl(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // DISABLE FTZ: Force full IEEE-754 compliance with denormal support
    // This should force denormal handling instead of flushing to zero
    let kernel_src = r#"
        // Disable FTZ - force full denormal support
        #pragma OPENCL EXTENSION cl_khr_fp64 : enable

        // Inline functions to force denormal preservation
        inline float denormal_safe_mul(float a, float b) {
            // Force computation without FTZ by using volatile
            volatile float result = a * b;
            return result;
        }

        inline float denormal_safe_add(float a, float b) {
            volatile float result = a + b;
            return result;
        }

        __kernel void matmul_f32(
            __global const float* restrict a,
            __global const float* restrict b,
            __global float* restrict c,
            int M, int N, int K
        ) {
            int row = get_global_id(1);
            int col = get_global_id(0);
            if (row >= M || col >= N) return;

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                float product = denormal_safe_mul(a[row * K + k], b[k * N + col]);
                sum = denormal_safe_add(sum, product);
            }
            c[row * N + col] = sum;
        }
    "#;

    let pro_que = ProQue::builder().src(kernel_src).dims((n, m)).build()?;
    let a_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(MemFlags::new().read_only()).len(a.len()).copy_host_slice(a).build()?;
    let b_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(MemFlags::new().read_only()).len(b.len()).copy_host_slice(b).build()?;
    let c_buf = Buffer::<f32>::builder().queue(pro_que.queue().clone()).flags(MemFlags::new().write_only()).len(m * n).build()?;

    let kernel = pro_que.kernel_builder("matmul_f32")
        .arg(&a_buf).arg(&b_buf).arg(&c_buf)
        .arg(m as i32).arg(n as i32).arg(k as i32).build()?;

    unsafe { kernel.enq()?; }
    let mut c = vec![0.0f32; m * n];
    c_buf.read(&mut c).enq()?;
    Ok(c)
}

fn is_denormal(x: f32) -> bool {
    x != 0.0 && x.abs() < f32::MIN_POSITIVE
}

fn create_denormal_data(size: usize, denormal_rate: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    let batch_size = 32; // Wave size on RDNA2

    for batch in 0..(size / batch_size) {
        for i in 0..batch_size {
            let idx = batch * batch_size + i;

            // Ensure at least one denormal per batch (worst case for wave divergence)
            let force_denormal = i == 0; // First thread in every wave gets a denormal

            let val = if force_denormal || (idx as f32 / size as f32) < denormal_rate {
                // Create TRUE denormal: f32::MIN_POSITIVE * fraction
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
    std::env::set_var("RUSTICL_ENABLE", "radeonsi");

    println!("=== Denormal Torture Test ===\n");
    println!("Strategy:");
    println!("  - Use f32::MIN_POSITIVE * fraction to create TRUE denormals");
    println!("  - Ensure 1 denormal per 32-thread wave (worst case divergence)");
    println!("  - Test multiply AND add to find the bottleneck");
    println!("  - Verify denormals survive (not flushed to zero)\n");

    let size = 512;

    println!("Creating denormal-infested data...");

    // Scenario 1: Moderate denormals (10% guaranteed, 1 per wave)
    let a1_ieee = create_denormal_data(size * size, 0.1);
    let b1_ieee = create_denormal_data(size * size, 0.1);

    let denormal_count_a = a1_ieee.iter().filter(|&&x| is_denormal(x)).count();
    let denormal_count_b = b1_ieee.iter().filter(|&&x| is_denormal(x)).count();
    println!("Scenario 1 (10% target):");
    println!("  A denormals: {} ({:.1}%)", denormal_count_a, (denormal_count_a as f64 / a1_ieee.len() as f64) * 100.0);
    println!("  B denormals: {} ({:.1}%)", denormal_count_b, (denormal_count_b as f64 / b1_ieee.len() as f64) * 100.0);

    // Verify they're REAL denormals
    let sample_denormal = a1_ieee.iter().find(|&&x| is_denormal(x)).unwrap();
    println!("  Sample denormal: {:.6e} (bits: {:032b})", sample_denormal, sample_denormal.to_bits());
    println!("  f32::MIN_POSITIVE: {:.6e}", f32::MIN_POSITIVE);

    // Create Spirix equivalents
    let a1_spirix: Vec<ScalarF4E4> = a1_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b1_spirix: Vec<ScalarF4E4> = b1_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    println!("\nRunning benchmarks...\n");

    // Test Scenario 1
    benchmark_scenario("10% Denormal", &a1_spirix, &b1_spirix, &a1_ieee, &b1_ieee, size);

    // Scenario 2: Heavy denormals (30%)
    println!("\nCreating heavy denormal data...");
    let a2_ieee = create_denormal_data(size * size, 0.3);
    let b2_ieee = create_denormal_data(size * size, 0.3);

    let denormal_count_a = a2_ieee.iter().filter(|&&x| is_denormal(x)).count();
    let denormal_count_b = b2_ieee.iter().filter(|&&x| is_denormal(x)).count();
    println!("Scenario 2 (30% target):");
    println!("  A denormals: {} ({:.1}%)", denormal_count_a, (denormal_count_a as f64 / a2_ieee.len() as f64) * 100.0);
    println!("  B denormals: {} ({:.1}%)", denormal_count_b, (denormal_count_b as f64 / b2_ieee.len() as f64) * 100.0);

    let a2_spirix: Vec<ScalarF4E4> = a2_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b2_spirix: Vec<ScalarF4E4> = b2_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    benchmark_scenario("30% Denormal", &a2_spirix, &b2_spirix, &a2_ieee, &b2_ieee, size);

    // Scenario 3: Pathological (every thread has denormal)
    println!("\nCreating pathological denormal data...");
    let a3_ieee = create_denormal_data(size * size, 1.0);
    let b3_ieee = create_denormal_data(size * size, 1.0);

    let denormal_count_a = a3_ieee.iter().filter(|&&x| is_denormal(x)).count();
    let denormal_count_b = b3_ieee.iter().filter(|&&x| is_denormal(x)).count();
    println!("Scenario 3 (100% denormal):");
    println!("  A denormals: {} ({:.1}%)", denormal_count_a, (denormal_count_a as f64 / a3_ieee.len() as f64) * 100.0);
    println!("  B denormals: {} ({:.1}%)", denormal_count_b, (denormal_count_b as f64 / b3_ieee.len() as f64) * 100.0);

    let a3_spirix: Vec<ScalarF4E4> = a3_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b3_spirix: Vec<ScalarF4E4> = b3_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    benchmark_scenario("100% Denormal", &a3_spirix, &b3_spirix, &a3_ieee, &b3_ieee, size);

    println!("\n=== Analysis ===");
    println!("If IEEE slows down significantly:");
    println!("  → Denormals ARE causing branch divergence");
    println!("  → Spirix should win (zero divergence)");
    println!("");
    println!("If IEEE stays fast:");
    println!("  → GPU is using FTZ (flush-to-zero) mode");
    println!("  → Denormals become zeros (loses precision)");
    println!("  → IEEE wins on speed but loses correctness");
    println!("");
    println!("Check the output:");
    println!("  - Did denormals survive? (not flushed to 0)");
    println!("  - Did IEEE slow down with more denormals?");
    println!("  - Did Spirix maintain consistent performance?");
}

fn benchmark_scenario(
    name: &str,
    a_spirix: &[ScalarF4E4],
    b_spirix: &[ScalarF4E4],
    a_ieee: &[f32],
    b_ieee: &[f32],
    size: usize,
) {
    let a_tensor = Tensor::new(a_spirix.to_vec(), vec![size, size]);
    let b_tensor = Tensor::new(b_spirix.to_vec(), vec![size, size]);

    // Warmup
    let _ = matmul_gpu_opencl(&a_tensor, &b_tensor);
    let _ = matmul_f32_gpu_opencl(a_ieee, b_ieee, size, size, size);

    // Spirix GPU
    let spirix_start = Instant::now();
    let spirix_result = matmul_gpu_opencl(&a_tensor, &b_tensor).expect("Spirix GPU failed");
    let spirix_time = spirix_start.elapsed();

    // IEEE GPU
    let ieee_start = Instant::now();
    let ieee_result = matmul_f32_gpu_opencl(a_ieee, b_ieee, size, size, size).expect("IEEE GPU failed");
    let ieee_time = ieee_start.elapsed();

    // Check if denormals survived in IEEE output
    let output_denormals = ieee_result.iter().filter(|&&x| is_denormal(x)).count();
    let output_zeros = ieee_result.iter().filter(|&&x| x == 0.0).count();

    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    let winner = if ratio < 1.0 { "SPIRIX WINS! 🚀" } else { "IEEE wins" };

    println!("{:<20} Spirix: {:>8.3?}  IEEE: {:>8.3?}  Ratio: {:.2}x  {}",
        name, spirix_time, ieee_time, ratio, winner);
    println!("  IEEE output: {} denormals, {} zeros (FTZ={} likely)",
        output_denormals, output_zeros,
        if output_denormals < 100 { "YES" } else { "NO" });
}
