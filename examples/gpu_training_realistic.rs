//! Realistic Training Benchmark: Spirix vs IEEE
//!
//! Tests with actual training-like data:
//! - Vanishing gradients (denormals in IEEE)
//! - Wide dynamic range
//! - Mix of normal and tiny values
//!
//! This is where Spirix should shine!

use spirix::{ScalarF4E4, Tensor};
use veritas::gpu::matmul_gpu_opencl;
use std::time::Instant;
use ocl::{ProQue, Buffer, MemFlags};

fn matmul_f32_gpu_opencl(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let kernel_src = r#"
        __kernel void matmul_f32(
            __global const float* a,
            __global const float* b,
            __global float* c,
            int M, int N, int K
        ) {
            int row = get_global_id(1);
            int col = get_global_id(0);
            if (row >= M || col >= N) return;

            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a[row * K + k] * b[k * N + col];
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

fn main() {
    std::env::set_var("RUSTICL_ENABLE", "radeonsi");

    println!("=== Realistic Training Benchmark: Spirix vs IEEE ===\n");

    println!("Simulating real training scenarios:");
    println!("  1. Normal data (no vanishing)");
    println!("  2. Gradient-like data (10% vanishing)");
    println!("  3. Deep network gradients (30% vanishing)");
    println!("  4. Pathological case (50% vanishing)\n");

    let size = 512;

    // Scenario 1: Normal data (baseline)
    println!("--- Scenario 1: Normal Data ---");
    let a1_spirix: Vec<ScalarF4E4> = (0..size * size).map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01)).collect();
    let b1_spirix: Vec<ScalarF4E4> = (0..size * size).map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01)).collect();
    let a1_ieee: Vec<f32> = (0..size * size).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();
    let b1_ieee: Vec<f32> = (0..size * size).map(|i| ((i % 100) as f32 - 50.0) * 0.01).collect();

    benchmark_scenario("Normal", &a1_spirix, &b1_spirix, &a1_ieee, &b1_ieee, size);

    // Scenario 2: 10% vanishing (typical early training)
    println!("\n--- Scenario 2: 10% Vanishing (Early Training) ---");
    let mut a2_spirix: Vec<ScalarF4E4> = (0..size * size).map(|i| {
        if i % 10 == 0 {
            ScalarF4E4::from(1e-50)  // Vanished in Spirix
        } else {
            ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01)
        }
    }).collect();
    let b2_spirix = a2_spirix.clone();

    let a2_ieee: Vec<f32> = (0..size * size).map(|i| {
        if i % 10 == 0 {
            1e-40f32  // Denormal in IEEE
        } else {
            ((i % 100) as f32 - 50.0) * 0.01
        }
    }).collect();
    let b2_ieee = a2_ieee.clone();

    let denormal_count = a2_ieee.iter().filter(|&&x| x.abs() > 0.0 && x.abs() < 1.17549435e-38).count();
    println!("IEEE denormal count: {} ({:.1}%)", denormal_count, (denormal_count as f64 / a2_ieee.len() as f64) * 100.0);

    benchmark_scenario("10% Vanish", &a2_spirix, &b2_spirix, &a2_ieee, &b2_ieee, size);

    // Scenario 3: 30% vanishing (deep network)
    println!("\n--- Scenario 3: 30% Vanishing (Deep Network) ---");
    let a3_spirix: Vec<ScalarF4E4> = (0..size * size).map(|i| {
        if i % 3 == 0 {
            ScalarF4E4::from(1e-100)  // Vanished
        } else {
            ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.001)
        }
    }).collect();
    let b3_spirix = a3_spirix.clone();

    let a3_ieee: Vec<f32> = (0..size * size).map(|i| {
        if i % 3 == 0 {
            5e-39f32  // Denormal
        } else {
            ((i % 100) as f32 - 50.0) * 0.001
        }
    }).collect();
    let b3_ieee = a3_ieee.clone();

    let denormal_count = a3_ieee.iter().filter(|&&x| x.abs() > 0.0 && x.abs() < 1.17549435e-38).count();
    println!("IEEE denormal count: {} ({:.1}%)", denormal_count, (denormal_count as f64 / a3_ieee.len() as f64) * 100.0);

    benchmark_scenario("30% Vanish", &a3_spirix, &b3_spirix, &a3_ieee, &b3_ieee, size);

    // Scenario 4: 50% vanishing (pathological)
    println!("\n--- Scenario 4: 50% Vanishing (Pathological) ---");
    let a4_spirix: Vec<ScalarF4E4> = (0..size * size).map(|i| {
        if i % 2 == 0 {
            ScalarF4E4::from(1e-200)  // Vanished
        } else {
            ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.0001)
        }
    }).collect();
    let b4_spirix = a4_spirix.clone();

    let a4_ieee: Vec<f32> = (0..size * size).map(|i| {
        if i % 2 == 0 {
            1e-39f32  // Denormal
        } else {
            ((i % 100) as f32 - 50.0) * 0.0001
        }
    }).collect();
    let b4_ieee = a4_ieee.clone();

    let denormal_count = a4_ieee.iter().filter(|&&x| x.abs() > 0.0 && x.abs() < 1.17549435e-38).count();
    println!("IEEE denormal count: {} ({:.1}%)", denormal_count, (denormal_count as f64 / a4_ieee.len() as f64) * 100.0);

    benchmark_scenario("50% Vanish", &a4_spirix, &b4_spirix, &a4_ieee, &b4_ieee, size);

    println!("\n=== Analysis ===");
    println!("Expected results:");
    println!("  Scenario 1: IEEE faster (no denormals, hw advantage)");
    println!("  Scenario 2: Spirix closer (denormals start hurting IEEE)");
    println!("  Scenario 3: Spirix WINS (branch divergence dominates)");
    println!("  Scenario 4: Spirix DOMINATES (IEEE thrashing on denormals)");
    println!("");
    println!("If Spirix wins scenarios 3-4, this PROVES the hypothesis:");
    println!("  Zero branch divergence > hardware float advantage");
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

    // Spirix GPU
    let spirix_start = Instant::now();
    let _spirix_result = matmul_gpu_opencl(&a_tensor, &b_tensor).expect("Spirix GPU failed");
    let spirix_time = spirix_start.elapsed();

    // IEEE GPU
    let ieee_start = Instant::now();
    let _ieee_result = matmul_f32_gpu_opencl(a_ieee, b_ieee, size, size, size).expect("IEEE GPU failed");
    let ieee_time = ieee_start.elapsed();

    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    let winner = if ratio < 1.0 { "SPIRIX WINS" } else { "IEEE wins" };

    println!("{:<15} Spirix: {:>8.3?}  IEEE: {:>8.3?}  Ratio: {:.2}x  {}",
        name, spirix_time, ieee_time, ratio, winner);
}
