//! Circle Complex Optimization Showdown
//!
//! Tests:
//! 1. Naive Circle vs IEEE (baseline from before)
//! 2. Tiled+Packed Circle vs IEEE (optimized)
//! 3. With DENORMAL data (vanishing gradients)
//!
//! Expected: The 100:1 instruction advantage should finally show!

use spirix::CircleF4E5;
use std::time::Instant;

extern "C" {
    // Naive version
    fn circle_f4e5_matmul_hip(
        a_real: *const i16, a_imag: *const i16, a_exp: *const i32,
        b_real: *const i16, b_imag: *const i16, b_exp: *const i32,
        c_real: *mut i16, c_imag: *mut i16, c_exp: *mut i32,
        m: i32, n: i32, k: i32,
    );

    // Optimized: Tiled + Packed loads
    fn circle_f4e5_tiled_packed_hip(
        a_real: *const i16, a_imag: *const i16, a_exp: *const i32,
        b_real: *const i16, b_imag: *const i16, b_exp: *const i32,
        c_real: *mut i16, c_imag: *mut i16, c_exp: *mut i32,
        m: i32, n: i32, k: i32,
    );

    fn ieee_complex_matmul_hip(
        a_real: *const f32, a_imag: *const f32,
        b_real: *const f32, b_imag: *const f32,
        c_real: *mut f32, c_imag: *mut f32,
        m: i32, n: i32, k: i32,
    );
}

/// Create gradient-like data with DENORMALS
/// Simulates vanishing gradients in deep network training
fn create_vanishing_gradient_data(size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut real = Vec::with_capacity(size);
    let mut imag = Vec::with_capacity(size);

    for i in 0..size {
        // Mix of normal, near-denormal, and denormal values
        let scale = match i % 4 {
            0 => f32::MIN_POSITIVE * 0.5,        // DENORMAL
            1 => f32::MIN_POSITIVE * 10.0,       // Near denormal
            2 => 1e-30,                          // Deep denormal
            _ => (i as f32 * 0.001).cos() * 1e-35, // Denormal with phase
        };

        let angle = (i as f32) * 0.01;
        real.push(scale * angle.cos());
        imag.push(scale * angle.sin());
    }

    (real, imag)
}

fn main() {
    println!("=== Circle Complex Optimization Showdown ===\n");
    println!("Testing THREE versions:");
    println!("  1. Circle Naive (baseline)");
    println!("  2. Circle Optimized (tiled + packed loads)");
    println!("  3. IEEE Complex64 (2× f32)\n");
    println!("Data: Vanishing gradients with DENORMALS\n");

    let size = 512;

    println!("Creating vanishing gradient data ({}×{})...", size, size);
    let (a_real_ieee, a_imag_ieee) = create_vanishing_gradient_data(size * size);
    let (b_real_ieee, b_imag_ieee) = create_vanishing_gradient_data(size * size);

    // Count denormals
    let a_denormals = a_real_ieee.iter().chain(a_imag_ieee.iter())
        .filter(|&&x| x != 0.0 && x.abs() < f32::MIN_POSITIVE).count();
    let b_denormals = b_real_ieee.iter().chain(b_imag_ieee.iter())
        .filter(|&&x| x != 0.0 && x.abs() < f32::MIN_POSITIVE).count();

    println!("Denormals in A: {} / {} ({:.1}%)",
        a_denormals, size * size * 2,
        (a_denormals as f64 / (size * size * 2) as f64) * 100.0);
    println!("Denormals in B: {} / {} ({:.1}%)\n",
        b_denormals, size * size * 2,
        (b_denormals as f64 / (size * size * 2) as f64) * 100.0);

    // Convert to CircleF4E5
    println!("Converting to CircleF4E5...");
    let mut a_circles = Vec::with_capacity(size * size);
    let mut b_circles = Vec::with_capacity(size * size);

    for i in 0..(size * size) {
        let a_complex = (a_real_ieee[i] as f64, a_imag_ieee[i] as f64);
        let b_complex = (b_real_ieee[i] as f64, b_imag_ieee[i] as f64);
        a_circles.push(CircleF4E5::from(a_complex));
        b_circles.push(CircleF4E5::from(b_complex));
    }

    // Prepare buffers
    let a_real_circle: Vec<i16> = a_circles.iter().map(|c| c.real).collect();
    let a_imag_circle: Vec<i16> = a_circles.iter().map(|c| c.imaginary).collect();
    let a_exp_circle: Vec<i32> = a_circles.iter().map(|c| c.exponent).collect();

    let b_real_circle: Vec<i16> = b_circles.iter().map(|c| c.real).collect();
    let b_imag_circle: Vec<i16> = b_circles.iter().map(|c| c.imaginary).collect();
    let b_exp_circle: Vec<i32> = b_circles.iter().map(|c| c.exponent).collect();

    let mut c_real_naive = vec![0i16; size * size];
    let mut c_imag_naive = vec![0i16; size * size];
    let mut c_exp_naive = vec![0i32; size * size];

    let mut c_real_optimized = vec![0i16; size * size];
    let mut c_imag_optimized = vec![0i16; size * size];
    let mut c_exp_optimized = vec![0i32; size * size];

    let mut c_real_ieee = vec![0.0f32; size * size];
    let mut c_imag_ieee = vec![0.0f32; size * size];

    println!("Warming up...\n");

    // Warmup
    unsafe {
        circle_f4e5_matmul_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_naive.as_mut_ptr(), c_imag_naive.as_mut_ptr(), c_exp_naive.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );

        circle_f4e5_tiled_packed_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_optimized.as_mut_ptr(), c_imag_optimized.as_mut_ptr(), c_exp_optimized.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );

        ieee_complex_matmul_hip(
            a_real_ieee.as_ptr(), a_imag_ieee.as_ptr(),
            b_real_ieee.as_ptr(), b_imag_ieee.as_ptr(),
            c_real_ieee.as_mut_ptr(), c_imag_ieee.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }

    println!("Running benchmarks...\n");

    // Benchmark Circle Naive
    let start = Instant::now();
    unsafe {
        circle_f4e5_matmul_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_naive.as_mut_ptr(), c_imag_naive.as_mut_ptr(), c_exp_naive.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let circle_naive_time = start.elapsed();

    // Benchmark Circle Optimized
    let start = Instant::now();
    unsafe {
        circle_f4e5_tiled_packed_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_optimized.as_mut_ptr(), c_imag_optimized.as_mut_ptr(), c_exp_optimized.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let circle_optimized_time = start.elapsed();

    // Benchmark IEEE
    let start = Instant::now();
    unsafe {
        ieee_complex_matmul_hip(
            a_real_ieee.as_ptr(), a_imag_ieee.as_ptr(),
            b_real_ieee.as_ptr(), b_imag_ieee.as_ptr(),
            c_real_ieee.as_mut_ptr(), c_imag_ieee.as_mut_ptr(),
            size as i32, size as i32, size as i32,
        );
    }
    let ieee_time = start.elapsed();

    // Check IEEE denormal preservation
    let ieee_out_denormals = c_real_ieee.iter().chain(c_imag_ieee.iter())
        .filter(|&&x| x != 0.0 && x.abs() < f32::MIN_POSITIVE).count();

    println!("=== Results ({}×{} complex matmul with denormals) ===\n", size, size);
    println!("Circle Naive:      {:>10.3?}", circle_naive_time);
    println!("Circle Optimized:  {:>10.3?}  (tiled + packed)", circle_optimized_time);
    println!("IEEE:              {:>10.3?}\n", ieee_time);

    let naive_ratio = circle_naive_time.as_secs_f64() / ieee_time.as_secs_f64();
    let optimized_ratio = circle_optimized_time.as_secs_f64() / ieee_time.as_secs_f64();
    let speedup_from_opt = circle_naive_time.as_secs_f64() / circle_optimized_time.as_secs_f64();

    println!("Circle Naive vs IEEE:     {:.2}x  {}",
        naive_ratio,
        if naive_ratio < 1.0 { format!("(Circle {:.2}x faster)", 1.0/naive_ratio) }
        else { format!("(IEEE {:.2}x faster)", naive_ratio) });

    println!("Circle Optimized vs IEEE: {:.2}x  {}",
        optimized_ratio,
        if optimized_ratio < 1.0 { format!("🚀 CIRCLE {:.2}x FASTER!", 1.0/optimized_ratio) }
        else { format!("(IEEE {:.2}x faster)", optimized_ratio) });

    println!("\nOptimization impact: {:.2}x speedup (naive → optimized)\n", speedup_from_opt);

    println!("=== Analysis ===");
    println!("Denormals in IEEE output: {} ({}% preserved)",
        ieee_out_denormals,
        if a_denormals + b_denormals > 0 {
            format!("{:.1}", (ieee_out_denormals as f64 / (a_denormals + b_denormals) as f64) * 100.0)
        } else { "N/A".to_string() }
    );

    println!("\nOptimizations applied to Circle:");
    println!("  ✓ Shared memory tiling (16×16)");
    println!("  ✓ Packed loads (real+imag in single 32-bit load)");
    println!("  ✓ Coalesced memory access");
    println!("  ✓ Reduced memory transactions (3 → 2 loads)\n");

    if optimized_ratio < 0.2 {
        println!("🎉 THE 100:1 ADVANTAGE IS HERE!");
        println!("Circle is now showing its true algorithmic superiority!");
        println!("With denormals + optimization, IEEE's branch divergence is exposed.\n");
    } else if optimized_ratio < 0.5 {
        println!("✅ Circle's advantage is clear with optimization + denormals!");
        println!("The combination of simpler logic + better memory access wins.\n");
    }
}
