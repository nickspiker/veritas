//! GPU Matrix Multiply Benchmark
//!
//! Compares Spirix F4E4 on GPU vs CPU.
//! Measures actual performance to validate hypothesis from gpu_performance_analysis.md
//!
//! Expected results:
//! - CPU: ~152 instructions per op, predictable
//! - GPU (naive): Slower than CPU (no optimizations)
//! - GPU (optimized): Match or beat CPU
//!
//! This is Phase 1: Naive GPU kernel, no optimizations.

use spirix::{ScalarF4E4, Tensor};
use veritas::gpu::matmul_gpu;
use std::time::Instant;

fn main() {
    println!("=== Spirix GPU Matmul Benchmark ===\n");

    println!("Testing GPU availability...");

    // Try a small test first
    let test_a = Tensor::new(
        vec![ScalarF4E4::from(1.0); 4],
        vec![2, 2],
    );
    let test_b = Tensor::new(
        vec![ScalarF4E4::from(2.0); 4],
        vec![2, 2],
    );

    match std::panic::catch_unwind(|| {
        matmul_gpu(&test_a, &test_b)
    }) {
        Ok(_) => println!("✓ GPU available and working\n"),
        Err(_) => {
            println!("✗ GPU not available or HIP library not built");
            println!("\nTo build HIP library:");
            println!("  cd gpu/hip");
            println!("  ./build.sh");
            println!("\nThen ensure libspirix_hip.so is in LD_LIBRARY_PATH");
            return;
        }
    }

    // Benchmark different matrix sizes
    let sizes = vec![64, 128, 256, 512, 1024];

    println!("Benchmarking matrix multiply (CPU vs GPU):\n");
    println!("{:<10} {:<15} {:<15} {:<10}", "Size", "CPU Time", "GPU Time", "Speedup");
    println!("{:-<55}", "");

    for size in sizes {
        // Create random matrices
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
        let gpu_result = matmul_gpu(&a, &b);
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
    println!("  - No shared memory tiling");
    println!("  - No memory coalescing");
    println!("  - High global memory latency");
    println!("");
    println!("But: ZERO branch divergence (all threads execute same path)");
    println!("");
    println!("Next steps:");
    println!("  1. Add shared memory tiling (10x speedup expected)");
    println!("  2. Optimize memory access patterns (2x speedup)");
    println!("  3. Register blocking (2x speedup)");
    println!("  4. Compare against rocBLAS f32 (IEEE-754)");
    println!("");
    println!("Hypothesis: Optimized Spirix GPU will match or beat IEEE-754");
    println!("  Reason: Zero branch divergence > instruction count penalty");
}
