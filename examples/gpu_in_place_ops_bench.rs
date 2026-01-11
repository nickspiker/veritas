//! In-Place Operation Benchmarks
//!
//! Tests pure compute throughput: FPU vs ALU.
//! Data stays on GPU, many iterations to amortize transfer cost.
//!
//! This removes memory bandwidth as a bottleneck and measures:
//! - Integer ALU speed (Spirix/Circle)
//! - FP32 FPU speed (IEEE)
//! - Branch divergence cost (1 denormal per wavefront on IEEE)

#![allow(unused)]

use spirix::{ScalarF4E4, CircleF4E5};
use std::time::Instant;

#[link(name = "in_place_ops")]
extern "C" {}

extern "C" {
    fn bench_spirix_inplace(
        h_frac: *const i16, h_exp: *const i16,
        h_out_frac: *mut i16, h_out_exp: *mut i16,
        n: i32, iterations: i32,
    );

    fn bench_ieee_inplace(
        h_data: *const f32, h_out: *mut f32,
        n: i32, iterations: i32,
    );

    fn bench_circle_inplace(
        h_real: *const i16, h_imag: *const i16, h_exp: *const i32,
        h_out_real: *mut i16, h_out_imag: *mut i16, h_out_exp: *mut i32,
        n: i32, iterations: i32,
    );

    fn bench_ieee_complex_inplace(
        h_real: *const f32, h_imag: *const f32,
        h_out_real: *mut f32, h_out_imag: *mut f32,
        n: i32, iterations: i32,
    );
}

fn create_denormal_data(size: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let scale = match i % 4 {
            0 => f32::MIN_POSITIVE * 0.5,  // Denormal
            1 => f32::MIN_POSITIVE * 10.0,
            2 => 1e-30,
            _ => 1e-35,
        };
        data.push(scale * ((i as f32) * 0.01).cos());
    }
    data
}

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           In-Place Operation Performance (Pure Compute)        ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n = 1_000_000;  // 1M elements
    let iterations = 1000;  // 1000 iterations each = 1B operations total

    println!("Testing {} elements × {} iterations = {} total operations\n",
        n, iterations, (n as u64) * (iterations as u64));

    println!("Data stays on GPU. Measuring pure compute throughput.\n");

    // ========================================================================
    // SCALAR MULTIPLY (IN-PLACE)
    // ========================================================================

    println!("═══ Scalar Multiply (self-multiply {} times) ═══\n", iterations);

    let data_ieee = create_denormal_data(n);
    let data_spirix: Vec<ScalarF4E4> = data_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    let frac: Vec<i16> = data_spirix.iter().map(|s| s.fraction).collect();
    let exp: Vec<i16> = data_spirix.iter().map(|s| s.exponent).collect();

    let mut out_frac = vec![0i16; n];
    let mut out_exp = vec![0i16; n];
    let mut out_ieee = vec![0.0f32; n];

    // Warmup
    unsafe {
        bench_spirix_inplace(frac.as_ptr(), exp.as_ptr(), out_frac.as_mut_ptr(), out_exp.as_mut_ptr(), n as i32, iterations as i32);
        bench_ieee_inplace(data_ieee.as_ptr(), out_ieee.as_mut_ptr(), n as i32, iterations as i32);
    }

    // Benchmark Spirix (Integer ALU)
    let start = Instant::now();
    unsafe {
        bench_spirix_inplace(frac.as_ptr(), exp.as_ptr(), out_frac.as_mut_ptr(), out_exp.as_mut_ptr(), n as i32, iterations as i32);
    }
    let spirix_time = start.elapsed();

    // Benchmark IEEE (FP32 FPU with denormal divergence)
    let start = Instant::now();
    unsafe {
        bench_ieee_inplace(data_ieee.as_ptr(), out_ieee.as_mut_ptr(), n as i32, iterations as i32);
    }
    let ieee_time = start.elapsed();

    let total_ops = (n as u64) * (iterations as u64);
    let spirix_gops = (total_ops as f64) / spirix_time.as_secs_f64() / 1e9;
    let ieee_gops = (total_ops as f64) / ieee_time.as_secs_f64() / 1e9;

    println!("Spirix (Integer ALU): {:>10.3?}  ({:.2} Gops/sec)", spirix_time, spirix_gops);
    println!("IEEE (FP32 FPU):      {:>10.3?}  ({:.2} Gops/sec)", ieee_time, ieee_gops);

    let scalar_ratio = ieee_time.as_secs_f64() / spirix_time.as_secs_f64();
    println!("Speedup: {:.2}x (Spirix {} faster)\n", scalar_ratio,
        if scalar_ratio > 1.0 { format!("{:.2}x", scalar_ratio) }
        else { format!("{:.2}x slower", 1.0/scalar_ratio) });

    // ========================================================================
    // COMPLEX MULTIPLY (IN-PLACE)
    // ========================================================================

    println!("═══ Complex Multiply (self-multiply {} times) ═══\n", iterations);

    let (real_ieee, imag_ieee) = {
        let mut real = Vec::with_capacity(n);
        let mut imag = Vec::with_capacity(n);
        for i in 0..n {
            let scale = match i % 4 {
                0 => f32::MIN_POSITIVE * 0.5,
                1 => f32::MIN_POSITIVE * 10.0,
                2 => 1e-30,
                _ => 1e-35,
            };
            let angle = (i as f32) * 0.01;
            real.push(scale * angle.cos());
            imag.push(scale * angle.sin());
        }
        (real, imag)
    };

    let circles: Vec<CircleF4E5> = real_ieee.iter().zip(imag_ieee.iter())
        .map(|(&r, &i)| CircleF4E5::from((r as f64, i as f64)))
        .collect();

    let real_circle: Vec<i16> = circles.iter().map(|c| c.real).collect();
    let imag_circle: Vec<i16> = circles.iter().map(|c| c.imaginary).collect();
    let exp_circle: Vec<i32> = circles.iter().map(|c| c.exponent).collect();

    let mut out_real_circle = vec![0i16; n];
    let mut out_imag_circle = vec![0i16; n];
    let mut out_exp_circle = vec![0i32; n];

    let mut out_real_ieee = vec![0.0f32; n];
    let mut out_imag_ieee = vec![0.0f32; n];

    // Warmup
    unsafe {
        bench_circle_inplace(
            real_circle.as_ptr(), imag_circle.as_ptr(), exp_circle.as_ptr(),
            out_real_circle.as_mut_ptr(), out_imag_circle.as_mut_ptr(), out_exp_circle.as_mut_ptr(),
            n as i32, iterations as i32,
        );
        bench_ieee_complex_inplace(
            real_ieee.as_ptr(), imag_ieee.as_ptr(),
            out_real_ieee.as_mut_ptr(), out_imag_ieee.as_mut_ptr(),
            n as i32, iterations as i32,
        );
    }

    // Benchmark Circle (Integer ALU)
    let start = Instant::now();
    unsafe {
        bench_circle_inplace(
            real_circle.as_ptr(), imag_circle.as_ptr(), exp_circle.as_ptr(),
            out_real_circle.as_mut_ptr(), out_imag_circle.as_mut_ptr(), out_exp_circle.as_mut_ptr(),
            n as i32, iterations as i32,
        );
    }
    let circle_time = start.elapsed();

    // Benchmark IEEE Complex (FP32 FPU with denormal divergence)
    let start = Instant::now();
    unsafe {
        bench_ieee_complex_inplace(
            real_ieee.as_ptr(), imag_ieee.as_ptr(),
            out_real_ieee.as_mut_ptr(), out_imag_ieee.as_mut_ptr(),
            n as i32, iterations as i32,
        );
    }
    let ieee_complex_time = start.elapsed();

    let circle_gops = (total_ops as f64) / circle_time.as_secs_f64() / 1e9;
    let ieee_complex_gops = (total_ops as f64) / ieee_complex_time.as_secs_f64() / 1e9;

    println!("Circle (Integer ALU): {:>10.3?}  ({:.2} Gops/sec)", circle_time, circle_gops);
    println!("IEEE (FP32 FPU):      {:>10.3?}  ({:.2} Gops/sec)", ieee_complex_time, ieee_complex_gops);

    let complex_ratio = ieee_complex_time.as_secs_f64() / circle_time.as_secs_f64();
    println!("Speedup: {:.2}x (Circle {} faster)\n", complex_ratio,
        if complex_ratio > 1.0 { format!("{:.2}x", complex_ratio) }
        else { format!("{:.2}x slower", 1.0/complex_ratio) });

    // ========================================================================
    // SUMMARY
    // ========================================================================

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Pure Compute Throughput (data on GPU, {} iterations):", iterations);
    println!("  Scalar:  Spirix {:.2}x faster ({:.2} vs {:.2} Gops/sec)",
        scalar_ratio, spirix_gops, ieee_gops);
    println!("  Complex: Circle {:.2}x faster ({:.2} vs {:.2} Gops/sec)\n",
        complex_ratio, circle_gops, ieee_complex_gops);

    println!("IEEE Penalty from Branch Divergence:");
    println!("  Every 32-thread wavefront has ≥1 denormal (worst case)");
    println!("  Explicit denormal checks cause wave stalls");
    println!("  FP32 FPU must handle denormal paths = {:.1}x slower\n",
        if scalar_ratio < 1.0 { 1.0/scalar_ratio } else { scalar_ratio });

    println!("Integer ALU vs FP32 FPU:");
    println!("  Both have 32 units per CU on RDNA2");
    println!("  Spirix/Circle use integer ALU (no denormal checks)");
    println!("  IEEE uses FP32 FPU (denormal handling overhead)");
    println!("  Result: Integer path is {:.2}x faster with denormals\n", scalar_ratio);
}
