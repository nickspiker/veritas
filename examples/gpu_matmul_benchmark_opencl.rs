//! GPU Matrix Multiply Benchmark (OpenCL)
//!
//! Compares Spirix F4E4 on GPU (OpenCL) vs CPU.
//! Measures actual performance to validate hypothesis from gpu_performance_analysis.md
//!
//! Run with: RUSTICL_ENABLE=radeonsi cargo run --release --features opencl --example gpu_matmul_benchmark_opencl

use spirix::{ScalarF4E4, Tensor};
use veritas::gpu::matmul_gpu_opencl;
use std::time::Instant;

fn main() {
    // Enable rusticl for AMD GPU
    std::env::set_var("RUSTICL_ENABLE", "radeonsi");

    println!("=== Spirix GPU Matmul Benchmark (OpenCL) ===\n");

    println!("Testing GPU availability...");

    // Try a small test first
    let test_a = Tensor::new(
        vec![ScalarF4E4::ONE; 4],
        vec![2, 2],
    );
    let test_b = Tensor::new(
        vec![ScalarF4E4::from(2u8); 4],
        vec![2, 2],
    );

    match matmul_gpu_opencl(&test_a, &test_b) {
        Ok(_) => println!("✓ GPU available and working\n"),
        Err(e) => {
            println!("✗ GPU not available: {}", e);
            println!("\nMake sure:");
            println!("  1. RUSTICL_ENABLE=radeonsi is set");
            println!("  2. mesa-libOpenCL is installed");
            println!("  3. GPU is accessible");
            return;
        }
    }

    // Benchmark different matrix sizes
    let sizes = vec![64, 128, 256, 512];

    println!("Benchmarking matrix multiply (CPU vs GPU):\n");
    println!("{:<10} {:<15} {:<15} {:<10}", "Size", "CPU Time", "GPU Time", "Speedup");
    println!("{:-<55}", "");

    for size in sizes {
        // Create matrices
        let a_data: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01))
            .collect();
        let b_data: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from(((i % 100) as f64 - 50.0) * 0.01))
            .collect();

        let a = Tensor::new(a_data.clone(), vec![size, size]);
        let b = Tensor::new(b_data.clone(), vec![size, size]);

        // CPU benchmark
        let cpu_start = Instant::now();
        let cpu_result = spirix::matmul(&a, &b, ScalarF4E4::ZERO);
        let cpu_time = cpu_start.elapsed();

        // GPU benchmark
        let gpu_start = Instant::now();
        let gpu_result = match matmul_gpu_opencl(&a, &b) {
            Ok(result) => result,
            Err(e) => {
                println!("GPU failed: {}", e);
                continue;
            }
        };
        let gpu_time = gpu_start.elapsed();

        // Verify results match
        let max_diff = cpu_result
            .data
            .iter()
            .zip(gpu_result.data.iter())
            .map(|(c, g)| (c.to_f64() - g.to_f64()).abs())
            .fold(0.0f64, f64::max);

        if max_diff > 0.01 {
            println!("WARNING: CPU and GPU results differ by {:.6}", max_diff);
        }

        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();

        println!(
            "{:<10} {:<15.3?} {:<15.3?} {:<10.2}x",
            format!("{}×{}", size, size),
            cpu_time,
            gpu_time,
            speedup
        );
    }

    println!("\n--- Analysis ---");
    println!("Phase 1 (Naive GPU): Expected to be SLOWER than CPU");
    println!("  - No local memory tiling");
    println!("  - No memory coalescing");
    println!("  - High global memory latency");
    println!("");
    println!("But: ZERO branch divergence (all work-items execute same path)");
    println!("");
    println!("GPU Info:");
    println!("  AMD Radeon RX 6500 XT (RDNA2, Navi 24)");
    println!("  OpenCL via Mesa rusticl");
    println!("  16 Compute Units, 4GB GDDR6");
    println!("");
    println!("Next steps:");
    println!("  1. Add local memory tiling (10x speedup expected)");
    println!("  2. Optimize memory access patterns (2x speedup)");
    println!("  3. Register blocking (2x speedup)");
    println!("  4. Compare against IEEE-754 f32");
    println!("");
    println!("Hypothesis: Optimized Spirix GPU will match or beat IEEE-754");
    println!("  Reason: Zero branch divergence > instruction count penalty");
}
