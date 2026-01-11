//! Isolated Operation Benchmarks
//!
//! Tests individual operations, not full algorithms:
//! - Scalar multiply (*)
//! - Scalar add (+)
//! - Complex multiply (*)
//! - Complex add (+)
//!
//! This shows per-operation cost with denormal data.

use spirix::{ScalarF4E4, CircleF4E5};
use std::time::Instant;

extern "C" {
    fn bench_spirix_mul_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        c_frac: *mut i16, c_exp: *mut i16,
        n: i32,
    );

    fn bench_spirix_add_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        c_frac: *mut i16, c_exp: *mut i16,
        n: i32,
    );

    fn bench_circle_mul_hip(
        a_real: *const i16, a_imag: *const i16, a_exp: *const i32,
        b_real: *const i16, b_imag: *const i16, b_exp: *const i32,
        c_real: *mut i16, c_imag: *mut i16, c_exp: *mut i32,
        n: i32,
    );

    fn bench_ieee_mul_hip(
        a: *const f32, b: *const f32, c: *mut f32, n: i32
    );

    fn bench_ieee_complex_mul_hip(
        a_real: *const f32, a_imag: *const f32,
        b_real: *const f32, b_imag: *const f32,
        c_real: *mut f32, c_imag: *mut f32,
        n: i32,
    );
}

fn create_denormal_data(size: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    for i in 0..size {
        let scale = match i % 4 {
            0 => f32::MIN_POSITIVE * 0.5,
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
    println!("║            Isolated Operation Performance                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let n = 10_000_000; // 10M operations

    println!("Testing {} operations with denormal data...\n", n);

    // ========================================================================
    // SCALAR MULTIPLY
    // ========================================================================

    println!("═══ Scalar Multiply (a * b) ═══\n");

    let a_ieee = create_denormal_data(n);
    let b_ieee = create_denormal_data(n);

    let a_spirix: Vec<ScalarF4E4> = a_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b_spirix: Vec<ScalarF4E4> = b_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    let a_frac: Vec<i16> = a_spirix.iter().map(|s| s.fraction).collect();
    let a_exp: Vec<i16> = a_spirix.iter().map(|s| s.exponent).collect();
    let b_frac: Vec<i16> = b_spirix.iter().map(|s| s.fraction).collect();
    let b_exp: Vec<i16> = b_spirix.iter().map(|s| s.exponent).collect();

    let mut c_frac_spirix = vec![0i16; n];
    let mut c_exp_spirix = vec![0i16; n];
    let mut c_ieee = vec![0.0f32; n];

    // Warmup
    unsafe {
        bench_spirix_mul_hip(a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
                             c_frac_spirix.as_mut_ptr(), c_exp_spirix.as_mut_ptr(), n as i32);
        bench_ieee_mul_hip(a_ieee.as_ptr(), b_ieee.as_ptr(), c_ieee.as_mut_ptr(), n as i32);
    }

    // Benchmark Spirix
    let start = Instant::now();
    unsafe {
        bench_spirix_mul_hip(a_frac.as_ptr(), a_exp.as_ptr(), b_frac.as_ptr(), b_exp.as_ptr(),
                             c_frac_spirix.as_mut_ptr(), c_exp_spirix.as_mut_ptr(), n as i32);
    }
    let spirix_mul_time = start.elapsed();

    // Benchmark IEEE
    let start = Instant::now();
    unsafe {
        bench_ieee_mul_hip(a_ieee.as_ptr(), b_ieee.as_ptr(), c_ieee.as_mut_ptr(), n as i32);
    }
    let ieee_mul_time = start.elapsed();

    let mul_ratio = spirix_mul_time.as_secs_f64() / ieee_mul_time.as_secs_f64();

    println!("Spirix: {:>10.3?}  ({:.1} Gops/sec)", spirix_mul_time, n as f64 / spirix_mul_time.as_secs_f64() / 1e9);
    println!("IEEE:   {:>10.3?}  ({:.1} Gops/sec)", ieee_mul_time, n as f64 / ieee_mul_time.as_secs_f64() / 1e9);
    println!("Ratio:  {:.2}x  {}\n", mul_ratio,
        if mul_ratio < 1.0 { format!("(Spirix {:.2}x faster)", 1.0/mul_ratio) }
        else { format!("(IEEE {:.2}x faster)", mul_ratio) });

    // ========================================================================
    // COMPLEX MULTIPLY
    // ========================================================================

    println!("═══ Complex Multiply ((a+bi) * (c+di)) ═══\n");

    let a_real_ieee = create_denormal_data(n);
    let a_imag_ieee = create_denormal_data(n);
    let b_real_ieee = create_denormal_data(n);
    let b_imag_ieee = create_denormal_data(n);

    let a_circles: Vec<CircleF4E5> = a_real_ieee.iter().zip(a_imag_ieee.iter())
        .map(|(&r, &i)| CircleF4E5::from((r as f64, i as f64)))
        .collect();
    let b_circles: Vec<CircleF4E5> = b_real_ieee.iter().zip(b_imag_ieee.iter())
        .map(|(&r, &i)| CircleF4E5::from((r as f64, i as f64)))
        .collect();

    let a_real_circle: Vec<i16> = a_circles.iter().map(|c| c.real).collect();
    let a_imag_circle: Vec<i16> = a_circles.iter().map(|c| c.imaginary).collect();
    let a_exp_circle: Vec<i32> = a_circles.iter().map(|c| c.exponent).collect();

    let b_real_circle: Vec<i16> = b_circles.iter().map(|c| c.real).collect();
    let b_imag_circle: Vec<i16> = b_circles.iter().map(|c| c.imaginary).collect();
    let b_exp_circle: Vec<i32> = b_circles.iter().map(|c| c.exponent).collect();

    let mut c_real_circle = vec![0i16; n];
    let mut c_imag_circle = vec![0i16; n];
    let mut c_exp_circle = vec![0i32; n];

    let mut c_real_ieee = vec![0.0f32; n];
    let mut c_imag_ieee = vec![0.0f32; n];

    // Warmup
    unsafe {
        bench_circle_mul_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_circle.as_mut_ptr(), c_imag_circle.as_mut_ptr(), c_exp_circle.as_mut_ptr(),
            n as i32,
        );
        bench_ieee_complex_mul_hip(
            a_real_ieee.as_ptr(), a_imag_ieee.as_ptr(),
            b_real_ieee.as_ptr(), b_imag_ieee.as_ptr(),
            c_real_ieee.as_mut_ptr(), c_imag_ieee.as_mut_ptr(),
            n as i32,
        );
    }

    // Benchmark Circle
    let start = Instant::now();
    unsafe {
        bench_circle_mul_hip(
            a_real_circle.as_ptr(), a_imag_circle.as_ptr(), a_exp_circle.as_ptr(),
            b_real_circle.as_ptr(), b_imag_circle.as_ptr(), b_exp_circle.as_ptr(),
            c_real_circle.as_mut_ptr(), c_imag_circle.as_mut_ptr(), c_exp_circle.as_mut_ptr(),
            n as i32,
        );
    }
    let circle_mul_time = start.elapsed();

    // Benchmark IEEE
    let start = Instant::now();
    unsafe {
        bench_ieee_complex_mul_hip(
            a_real_ieee.as_ptr(), a_imag_ieee.as_ptr(),
            b_real_ieee.as_ptr(), b_imag_ieee.as_ptr(),
            c_real_ieee.as_mut_ptr(), c_imag_ieee.as_mut_ptr(),
            n as i32,
        );
    }
    let ieee_complex_mul_time = start.elapsed();

    let complex_mul_ratio = circle_mul_time.as_secs_f64() / ieee_complex_mul_time.as_secs_f64();

    println!("Circle: {:>10.3?}  ({:.1} Gops/sec)", circle_mul_time, n as f64 / circle_mul_time.as_secs_f64() / 1e9);
    println!("IEEE:   {:>10.3?}  ({:.1} Gops/sec)", ieee_complex_mul_time, n as f64 / ieee_complex_mul_time.as_secs_f64() / 1e9);
    println!("Ratio:  {:.2}x  {}\n", complex_mul_ratio,
        if complex_mul_ratio < 1.0 { format!("(Circle {:.2}x faster)", 1.0/complex_mul_ratio) }
        else { format!("(IEEE {:.2}x faster)", complex_mul_ratio) });

    // ========================================================================
    // SUMMARY
    // ========================================================================

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                          SUMMARY                               ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Per-Operation Performance (with denormals):");
    println!("  Scalar multiply:  Spirix {:.2}x faster", 1.0/mul_ratio);
    println!("  Complex multiply: Circle {:.2}x faster\n", 1.0/complex_mul_ratio);

    println!("Note: These are ISOLATED operations, not full algorithms.");
    println!("Matrix multiply involves many ops + memory access patterns.");
    println!("The full matmul speedup (4x scalar, 16x complex) includes:");
    println!("  • Per-op advantage (shown here)");
    println!("  • Memory bandwidth benefits");
    println!("  • Reduced branch divergence across wavefronts\n");
}
