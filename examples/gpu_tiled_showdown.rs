//! Tiled Kernel Showdown: Optimized Spirix vs Optimized IEEE
//!
//! Fair comparison:
//! - Both use 16x16 shared memory tiling
//! - Both use coalesced memory access
//! - Same grid/block dimensions
//! - Test with realistic denormal-producing data

use spirix::{ScalarF4E4, Tensor};
use std::time::Instant;

extern "C" {
    // Naive kernels
    fn spirix_matmul_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        c_frac: *mut i16, c_exp: *mut i16,
        m: i32, n: i32, k: i32,
    );

    fn ieee_matmul_denormal_hip(
        a: *const f32, b: *const f32, c: *mut f32,
        m: i32, n: i32, k: i32,
    );

    // Tiled kernels
    fn spirix_matmul_tiled_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        c_frac: *mut i16, c_exp: *mut i16,
        m: i32, n: i32, k: i32,
    );

    fn ieee_matmul_tiled_hip(
        a: *const f32, b: *const f32, c: *mut f32,
        m: i32, n: i32, k: i32,
    );
}

fn create_gradient_data(size: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let learning_rate = 1e-5;
        let gradient = f32::MIN_POSITIVE * ((i % 50) as f32 + 1.0);
        data.push(if i % 2 == 0 { learning_rate } else { gradient });
    }
    data
}

fn main() {
    println!("=== Tiled Kernel Showdown ===\n");
    println!("Optimized Spirix vs Optimized IEEE");
    println!("Both using 16x16 shared memory tiling\n");

    let size = 1024;  // Larger for tiling benefits

    println!("Creating realistic gradient data ({}×{})...", size, size);
    let a_ieee = create_gradient_data(size * size);
    let b_ieee = create_gradient_data(size * size);

    let a_spirix: Vec<ScalarF4E4> = a_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b_spirix: Vec<ScalarF4E4> = b_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    // Prepare Spirix data
    let a_frac: Vec<i16> = a_spirix.iter().map(|s| s.fraction).collect();
    let a_exp: Vec<i16> = a_spirix.iter().map(|s| s.exponent).collect();
    let b_frac: Vec<i16> = b_spirix.iter().map(|s| s.fraction).collect();
    let b_exp: Vec<i16> = b_spirix.iter().map(|s| s.exponent).collect();

    let mut spirix_naive_frac = vec![0i16; size * size];
    let mut spirix_naive_exp = vec![0i16; size * size];
    let mut spirix_tiled_frac = vec![0i16; size * size];
    let mut spirix_tiled_exp = vec![0i16; size * size];

    let mut ieee_naive = vec![0.0f32; size * size];
    let mut ieee_tiled = vec![0.0f32; size * size];

    println!("\nWarming up...");

    // Warmup
    unsafe {
        spirix_matmul_hip(
            a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_naive_frac.as_mut_ptr(), spirix_naive_exp.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );

        spirix_matmul_tiled_hip(
            a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_tiled_frac.as_mut_ptr(), spirix_tiled_exp.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );

        ieee_matmul_denormal_hip(
            a_ieee.as_ptr(), b_ieee.as_ptr(), ieee_naive.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );

        ieee_matmul_tiled_hip(
            a_ieee.as_ptr(), b_ieee.as_ptr(), ieee_tiled.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }

    println!("Running benchmarks...\n");

    // Benchmark Spirix Naive
    let start = Instant::now();
    unsafe {
        spirix_matmul_hip(
            a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_naive_frac.as_mut_ptr(), spirix_naive_exp.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let spirix_naive_time = start.elapsed();

    // Benchmark Spirix Tiled
    let start = Instant::now();
    unsafe {
        spirix_matmul_tiled_hip(
            a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_tiled_frac.as_mut_ptr(), spirix_tiled_exp.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let spirix_tiled_time = start.elapsed();

    // Benchmark IEEE Naive
    let start = Instant::now();
    unsafe {
        ieee_matmul_denormal_hip(
            a_ieee.as_ptr(), b_ieee.as_ptr(), ieee_naive.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let ieee_naive_time = start.elapsed();

    // Benchmark IEEE Tiled
    let start = Instant::now();
    unsafe {
        ieee_matmul_tiled_hip(
            a_ieee.as_ptr(), b_ieee.as_ptr(), ieee_tiled.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let ieee_tiled_time = start.elapsed();

    println!("=== Results ({}×{} matmul) ===\n", size, size);

    println!("Naive Kernels:");
    println!("  Spirix:  {:>10.3?}", spirix_naive_time);
    println!("  IEEE:    {:>10.3?}", ieee_naive_time);
    println!("  Ratio:   {:.2}x  {}",
        spirix_naive_time.as_secs_f64() / ieee_naive_time.as_secs_f64(),
        if spirix_naive_time < ieee_naive_time { "🚀 Spirix wins" } else { "IEEE wins" }
    );

    println!("\nTiled Kernels (Optimized):");
    println!("  Spirix:  {:>10.3?}", spirix_tiled_time);
    println!("  IEEE:    {:>10.3?}", ieee_tiled_time);
    println!("  Ratio:   {:.2}x  {}",
        spirix_tiled_time.as_secs_f64() / ieee_tiled_time.as_secs_f64(),
        if spirix_tiled_time < ieee_tiled_time { "🚀 SPIRIX WINS!" } else { "IEEE wins" }
    );

    println!("\nSpeedups from Tiling:");
    println!("  Spirix:  {:.2}x faster (naive → tiled)",
        spirix_naive_time.as_secs_f64() / spirix_tiled_time.as_secs_f64());
    println!("  IEEE:    {:.2}x faster (naive → tiled)",
        ieee_naive_time.as_secs_f64() / ieee_tiled_time.as_secs_f64());

    println!("\n=== Analysis ===");
    println!("Shared memory tiling benefits:");
    println!("  - Reduces global memory traffic");
    println!("  - Increases data reuse");
    println!("  - Improves cache locality");
    println!("\nBoth kernels use same optimization strategy.");
    println!("Performance difference = pure algorithm advantage.\n");
}
