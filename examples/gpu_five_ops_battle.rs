//! The Five Operations Battle: Spirix vs IEEE
//!
//! Tests all five basic arithmetic operations with denormals:
//! - Add (+)
//! - Subtract (-)
//! - Multiply (*)
//! - Divide (/)
//! - Modulo (%)
//!
//! Both running on GPU with denormal-infested data.

use spirix::ScalarF4E4;
use std::time::Instant;

extern "C" {
    fn spirix_ops_hip(
        a_frac: *const i16, a_exp: *const i16,
        b_frac: *const i16, b_exp: *const i16,
        add_frac: *mut i16, add_exp: *mut i16,
        sub_frac: *mut i16, sub_exp: *mut i16,
        mul_frac: *mut i16, mul_exp: *mut i16,
        div_frac: *mut i16, div_exp: *mut i16,
        mod_frac: *mut i16, mod_exp: *mut i16,
        n: i32,
    );

    fn ieee_ops_hip(
        a: *const f32,
        b: *const f32,
        add_result: *mut f32,
        sub_result: *mut f32,
        mul_result: *mut f32,
        div_result: *mut f32,
        mod_result: *mut f32,
        n: i32,
    );
}

fn is_denormal(x: f32) -> bool {
    x != 0.0 && x.abs() < f32::MIN_POSITIVE
}

fn create_denormal_data(size: usize, denormal_rate: f32) -> Vec<f32> {
    let mut data = Vec::with_capacity(size);
    let batch_size = 32;

    for batch in 0..(size / batch_size) {
        for i in 0..batch_size {
            let idx = batch * batch_size + i;
            let force_denormal = i == 0;

            let val = if force_denormal || (idx as f32 / size as f32) < denormal_rate {
                let fraction = 0.1 + (idx as f32 % 100.0) / 1000.0;
                f32::MIN_POSITIVE * fraction
            } else {
                ((idx % 100) as f32 - 50.0) * 0.01
            };

            data.push(val);
        }
    }

    for i in (size / batch_size * batch_size)..size {
        data.push(((i % 100) as f32 - 50.0) * 0.01);
    }

    data
}

fn main() {
    println!("=== The Five Operations Battle ===\n");
    println!("Spirix (integer) vs IEEE f32 (float)");
    println!("All operations on GPU with denormals\n");

    let size = 1_000_000;  // 1M elements

    println!("Creating denormal-infested data ({} elements)...", size);
    let a_ieee = create_denormal_data(size, 0.2);  // 20% denormals
    let b_ieee = create_denormal_data(size, 0.2);

    let denormal_count_a = a_ieee.iter().filter(|&&x| is_denormal(x)).count();
    let denormal_count_b = b_ieee.iter().filter(|&&x| is_denormal(x)).count();

    println!("  A: {} denormals ({:.1}%)", denormal_count_a,
        (denormal_count_a as f64 / a_ieee.len() as f64) * 100.0);
    println!("  B: {} denormals ({:.1}%)\n", denormal_count_b,
        (denormal_count_b as f64 / b_ieee.len() as f64) * 100.0);

    // Convert to Spirix
    let a_spirix: Vec<ScalarF4E4> = a_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();
    let b_spirix: Vec<ScalarF4E4> = b_ieee.iter().map(|&x| ScalarF4E4::from(x as f64)).collect();

    // Separate Spirix fractions and exponents
    let a_frac: Vec<i16> = a_spirix.iter().map(|s| s.fraction).collect();
    let a_exp: Vec<i16> = a_spirix.iter().map(|s| s.exponent).collect();
    let b_frac: Vec<i16> = b_spirix.iter().map(|s| s.fraction).collect();
    let b_exp: Vec<i16> = b_spirix.iter().map(|s| s.exponent).collect();

    // Allocate output buffers
    let mut spirix_add_frac = vec![0i16; size];
    let mut spirix_add_exp = vec![0i16; size];
    let mut spirix_sub_frac = vec![0i16; size];
    let mut spirix_sub_exp = vec![0i16; size];
    let mut spirix_mul_frac = vec![0i16; size];
    let mut spirix_mul_exp = vec![0i16; size];
    let mut spirix_div_frac = vec![0i16; size];
    let mut spirix_div_exp = vec![0i16; size];
    let mut spirix_mod_frac = vec![0i16; size];
    let mut spirix_mod_exp = vec![0i16; size];

    let mut ieee_add = vec![0.0f32; size];
    let mut ieee_sub = vec![0.0f32; size];
    let mut ieee_mul = vec![0.0f32; size];
    let mut ieee_div = vec![0.0f32; size];
    let mut ieee_mod = vec![0.0f32; size];

    println!("Running benchmarks...\n");

    // Warmup
    unsafe {
        spirix_ops_hip(
            a_frac.as_ptr(), a_exp.as_ptr(),
            b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_add_frac.as_mut_ptr(), spirix_add_exp.as_mut_ptr(),
            spirix_sub_frac.as_mut_ptr(), spirix_sub_exp.as_mut_ptr(),
            spirix_mul_frac.as_mut_ptr(), spirix_mul_exp.as_mut_ptr(),
            spirix_div_frac.as_mut_ptr(), spirix_div_exp.as_mut_ptr(),
            spirix_mod_frac.as_mut_ptr(), spirix_mod_exp.as_mut_ptr(),
            size as i32,
        );

        ieee_ops_hip(
            a_ieee.as_ptr(),
            b_ieee.as_ptr(),
            ieee_add.as_mut_ptr(),
            ieee_sub.as_mut_ptr(),
            ieee_mul.as_mut_ptr(),
            ieee_div.as_mut_ptr(),
            ieee_mod.as_mut_ptr(),
            size as i32,
        );
    }

    // Benchmark Spirix
    let spirix_start = Instant::now();
    unsafe {
        spirix_ops_hip(
            a_frac.as_ptr(), a_exp.as_ptr(),
            b_frac.as_ptr(), b_exp.as_ptr(),
            spirix_add_frac.as_mut_ptr(), spirix_add_exp.as_mut_ptr(),
            spirix_sub_frac.as_mut_ptr(), spirix_sub_exp.as_mut_ptr(),
            spirix_mul_frac.as_mut_ptr(), spirix_mul_exp.as_mut_ptr(),
            spirix_div_frac.as_mut_ptr(), spirix_div_exp.as_mut_ptr(),
            spirix_mod_frac.as_mut_ptr(), spirix_mod_exp.as_mut_ptr(),
            size as i32,
        );
    }
    let spirix_time = spirix_start.elapsed();

    // Benchmark IEEE
    let ieee_start = Instant::now();
    unsafe {
        ieee_ops_hip(
            a_ieee.as_ptr(),
            b_ieee.as_ptr(),
            ieee_add.as_mut_ptr(),
            ieee_sub.as_mut_ptr(),
            ieee_mul.as_mut_ptr(),
            ieee_div.as_mut_ptr(),
            ieee_mod.as_mut_ptr(),
            size as i32,
        );
    }
    let ieee_time = ieee_start.elapsed();

    // Check denormal preservation in IEEE output
    let check_denormals = |data: &[f32], name: &str| {
        let count = data.iter().filter(|&&x| is_denormal(x)).count();
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        println!("  {:<8} {} denormals, {} zeros", name, count, zeros);
    };

    println!("\n=== Results ===\n");
    println!("{:<15} Spirix: {:>8.3?}  IEEE: {:>8.3?}  Ratio: {:.2}x  {}",
        "ALL FIVE OPS",
        spirix_time,
        ieee_time,
        spirix_time.as_secs_f64() / ieee_time.as_secs_f64(),
        if spirix_time < ieee_time { "🚀 SPIRIX WINS!" } else { "IEEE wins" }
    );

    println!("\nIEEE denormal preservation:");
    check_denormals(&ieee_add, "Add:");
    check_denormals(&ieee_sub, "Sub:");
    check_denormals(&ieee_mul, "Mul:");
    check_denormals(&ieee_div, "Div:");
    check_denormals(&ieee_mod, "Mod:");

    println!("\n=== Analysis ===");
    println!("Spirix operations:");
    println!("  + Add:      Integer add with exponent alignment");
    println!("  - Subtract: Negate + add");
    println!("  * Multiply: Integer multiply + normalize");
    println!("  / Divide:   Integer divide + normalize");
    println!("  % Modulo:   Divide + multiply + subtract");
    println!("\nIEEE operations:");
    println!("  All use IEEE-754 f32 hardware instructions");
    println!("  Denormals trigger microcode slow paths");
    println!("  Division especially slow with denormals\n");

    let ratio = spirix_time.as_secs_f64() / ieee_time.as_secs_f64();
    if ratio < 1.0 {
        println!("✓ Spirix is {:.1}x FASTER despite using more instructions!", 1.0 / ratio);
        println!("✓ Zero branch divergence wins!");
    } else {
        println!("IEEE is {:.1}x faster", ratio);
        println!("But check if denormals were preserved...");
    }
}
