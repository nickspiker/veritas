//! Spirix F4E4 vs IEEE-754 f32 Benchmark
//!
//! Direct comparison of:
//! - Spirix F4E4 (i16 frac + i16 exp = 32 bits total)
//! - IEEE-754 f32 (32 bits total)
//!
//! Key differences:
//! - Spirix range: ~10^-9864 to ~10^9864 (MASSIVE)
//! - IEEE range: ~10^-38 to ~10^38
//! - Spirix precision: ~4 decimal digits
//! - IEEE precision: ~7 decimal digits
//!
//! Run with: RUSTICL_ENABLE=radeonsi cargo run --release --features opencl --example gpu_spirix_vs_ieee

use spirix::{ScalarF4E4, Tensor};
use veritas::gpu::matmul_gpu_opencl;
use std::time::Instant;

/// CPU matmul for f32 (for comparison)
fn matmul_f32_cpu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }

    c
}

/// GPU matmul for f32 using OpenCL
fn matmul_f32_gpu_opencl(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    use ocl::{ProQue, Buffer, MemFlags};

    // Simple f32 matrix multiply kernel
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

    let pro_que = ProQue::builder()
        .src(kernel_src)
        .dims((n, m))
        .build()?;

    let a_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(a.len())
        .copy_host_slice(a)
        .build()?;

    let b_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(b.len())
        .copy_host_slice(b)
        .build()?;

    let c_buf = Buffer::<f32>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().write_only())
        .len(m * n)
        .build()?;

    let kernel = pro_que.kernel_builder("matmul_f32")
        .arg(&a_buf)
        .arg(&b_buf)
        .arg(&c_buf)
        .arg(m as i32)
        .arg(n as i32)
        .arg(k as i32)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    let mut c = vec![0.0f32; m * n];
    c_buf.read(&mut c).enq()?;

    Ok(c)
}

fn main() {
    std::env::set_var("RUSTICL_ENABLE", "radeonsi");

    println!("=== Spirix F4E4 vs IEEE-754 f32 Benchmark ===\n");

    println!("Format Comparison:");
    println!("  Spirix F4E4: i16 fraction + i16 exponent = 32 bits");
    println!("  IEEE f32:    23-bit mantissa + 8-bit exp + sign = 32 bits");
    println!("");
    println!("  Spirix range:     ~10^-9864 to ~10^9864");
    println!("  IEEE range:       ~10^-38 to ~10^38");
    println!("  Spirix precision: ~4 decimal digits");
    println!("  IEEE precision:   ~7 decimal digits");
    println!("");

    let sizes = vec![64, 128, 256, 512, 1024];

    println!("Matrix Multiply Performance:\n");
    println!("{:<10} {:<15} {:<15} {:<15} {:<10}", "Size", "Spirix GPU", "IEEE GPU", "IEEE CPU", "Spirix/IEEE");
    println!("{:-<70}", "");

    for size in sizes {
        // Create test data
        let a_spirix: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01))
            .collect();
        let b_spirix: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01))
            .collect();

        let a_ieee: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();
        let b_ieee: Vec<f32> = (0..size * size)
            .map(|i| ((i % 100) as f32 - 50.0) * 0.01)
            .collect();

        let a_tensor = Tensor::new(a_spirix, vec![size, size]);
        let b_tensor = Tensor::new(b_spirix, vec![size, size]);

        // Spirix GPU
        let spirix_start = Instant::now();
        let _spirix_result = match matmul_gpu_opencl(&a_tensor, &b_tensor) {
            Ok(r) => r,
            Err(e) => {
                println!("Spirix GPU failed: {}", e);
                continue;
            }
        };
        let spirix_time = spirix_start.elapsed();

        // IEEE GPU
        let ieee_gpu_start = Instant::now();
        let _ieee_gpu_result = match matmul_f32_gpu_opencl(&a_ieee, &b_ieee, size, size, size) {
            Ok(r) => r,
            Err(e) => {
                println!("IEEE GPU failed: {}", e);
                continue;
            }
        };
        let ieee_gpu_time = ieee_gpu_start.elapsed();

        // IEEE CPU (for reference)
        let ieee_cpu_start = Instant::now();
        let _ieee_cpu_result = matmul_f32_cpu(&a_ieee, &b_ieee, size, size, size);
        let ieee_cpu_time = ieee_cpu_start.elapsed();

        let ratio = ieee_gpu_time.as_secs_f64() / spirix_time.as_secs_f64();

        println!(
            "{:<10} {:<15.3?} {:<15.3?} {:<15.3?} {:<10.2}x",
            format!("{}×{}", size, size),
            spirix_time,
            ieee_gpu_time,
            ieee_cpu_time,
            ratio
        );
    }

    println!("\n--- Analysis ---");
    println!("Spirix F4E4 characteristics:");
    println!("  ✓ ZERO branch divergence (10 branches total)");
    println!("  ✓ No denormals (vanished is explicit)");
    println!("  ✓ No NaN propagation (undefined is explicit)");
    println!("  ✓ Massive range (10^±9864 vs IEEE's 10^±38)");
    println!("  ✗ Lower precision (4 digits vs IEEE's 7)");
    println!("");
    println!("IEEE f32 characteristics:");
    println!("  ✗ Branch divergence (denormal handling)");
    println!("  ✗ NaN propagation stalls pipeline");
    println!("  ✗ Special case handling (inf, -0, subnormals)");
    println!("  ✓ Higher precision (7 decimal digits)");
    println!("  ✗ Limited range (10^±38)");
    println!("");
    println!("Use case recommendations:");
    println!("  Spirix: Physics sims, astronomy, financial (wide range needed)");
    println!("  IEEE:   Graphics, audio, ML inference (precision needed)");
    println!("");
    println!("Performance expectation:");
    println!("  Spirix should be 0.8-1.2x IEEE speed (similar or slightly slower)");
    println!("  With optimizations, Spirix should match or beat IEEE");
    println!("  Reason: Zero divergence compensates for integer ops overhead");
}
