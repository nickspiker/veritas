use spirix::{ScalarF3E3, ScalarF4E4, ScalarF5E5, ScalarF6E6, ScalarF7E7};

fn test_scalar<T>(name: &str)
where
    T: From<i32> + std::ops::Div<Output = T> + Copy + std::fmt::Debug + Into<f64>,
{
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Testing: {}", name);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut passed = 0;
    let mut failed = 0;

    // Helper to test a division
    let mut test = |a: i32, b: i32, expected: f64, tolerance: f64, desc: &str| {
        let x = T::from(a);
        let y = T::from(b);
        let result = x / y;
        let result_f64: f64 = result.into();
        let error = (result_f64 - expected).abs();

        if error <= tolerance {
            passed += 1;
            println!("  ✓ {}: {} / {} = {} (expected {}, error {:.6})",
                     desc, a, b, result_f64, expected, error);
        } else {
            failed += 1;
            println!("  ✗ {}: {} / {} = {} (expected {}, error {:.6}) FAIL",
                     desc, a, b, result_f64, expected, error);
        }
    };

    // 1. PERFECT DIVISIONS (should be exact or very close)
    test(20, 4, 5.0, 0.01, "Perfect: 20/4");
    test(144, 12, 12.0, 0.01, "Perfect: 144/12");
    test(100, 10, 10.0, 0.01, "Perfect: 100/10");
    test(64, 8, 8.0, 0.01, "Perfect: 64/8");
    test(9, 3, 3.0, 0.01, "Perfect: 9/3");

    // 2. IDENTITY OPERATIONS
    test(1, 1, 1.0, 0.01, "Identity: 1/1");
    test(5, 5, 1.0, 0.01, "Identity: 5/5");
    test(17, 17, 1.0, 0.01, "Identity: 17/17");
    test(100, 1, 100.0, 0.1, "Divide by 1: 100/1");

    // 3. POWERS OF 2 (should be exact)
    test(16, 2, 8.0, 0.01, "Power of 2: 16/2");
    test(32, 4, 8.0, 0.01, "Power of 2: 32/4");
    test(64, 16, 4.0, 0.01, "Power of 2: 64/16");
    test(128, 32, 4.0, 0.01, "Power of 2: 128/32");

    // 4. NON-PERFECT DIVISIONS (allow more tolerance)
    test(7, 3, 2.333333, 0.05, "Non-perfect: 7/3");
    test(10, 3, 3.333333, 0.05, "Non-perfect: 10/3");
    test(22, 7, 3.142857, 0.05, "Non-perfect: 22/7 (π approx)");
    test(100, 7, 14.285714, 0.1, "Non-perfect: 100/7");

    // 5. SMALL DIVISIONS
    test(1, 2, 0.5, 0.01, "Small: 1/2");
    test(1, 4, 0.25, 0.01, "Small: 1/4");
    test(3, 5, 0.6, 0.01, "Small: 3/5");
    test(2, 5, 0.4, 0.01, "Small: 2/5");

    // 6. LARGE NUMERATOR
    test(1000, 7, 142.857143, 1.0, "Large: 1000/7");
    test(500, 13, 38.461538, 0.5, "Large: 500/13");

    // 7. SIGN COMBINATIONS
    test(20, -4, -5.0, 0.01, "Sign: 20/-4");
    test(-20, 4, -5.0, 0.01, "Sign: -20/4");
    test(-20, -4, 5.0, 0.01, "Sign: -20/-4");
    test(-144, 12, -12.0, 0.01, "Sign: -144/12");

    // 8. EDGE CASES (division by larger denominator)
    test(3, 7, 0.428571, 0.02, "Fraction: 3/7");
    test(5, 11, 0.454545, 0.02, "Fraction: 5/11");
    test(1, 17, 0.058824, 0.02, "Fraction: 1/17");

    println!("\n  Summary: {} passed, {} failed", passed, failed);

    if failed > 0 {
        println!("  ⚠️  FAILURES DETECTED!");
    } else {
        println!("  ✅ ALL TESTS PASSED!");
    }
}

fn test_edge_cases() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          EDGE CASE TESTING (Zero, Overflow, etc)        ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // Test zero division for F4E4
    println!("\n--- Zero Division Tests (F4E4) ---");

    let zero = ScalarF4E4::from(0);
    let five = ScalarF4E4::from(5);

    // 0/5 should be 0
    let result = zero / five;
    let result_f64: f64 = result.into();
    println!("  0/5 = {} (expected 0)", result_f64);

    // 5/0 should be exploded/infinity
    let result2 = five / zero;
    println!("  5/0 = {:?} (should be exploded/infinity)", result2);

    // 0/0 should be undefined
    let result3 = zero / zero;
    println!("  0/0 = {:?} (should be undefined)", result3);

    // Test very large exponents
    println!("\n--- Overflow Tests ---");
    let large = ScalarF4E4 { fraction: 0x6000, exponent: 100 };
    let small_div = ScalarF4E4 { fraction: 0x4000, exponent: 0 };
    let overflow_result = large / small_div;
    println!("  Large/Normal = {:?} (exponent should be ~100)", overflow_result);

    // Test very small exponents (denormal-ish)
    println!("\n--- Tiny Value Tests ---");
    let tiny = ScalarF4E4 { fraction: 0x4000, exponent: -100 };
    let normal = ScalarF4E4 { fraction: 0x4000, exponent: 0 };
    let tiny_result = tiny / normal;
    println!("  Tiny/Normal = {:?} (exponent should be ~-100)", tiny_result);

    // Test normalization boundaries
    println!("\n--- Normalization Tests ---");
    let denormalized = ScalarF4E4 { fraction: 0x2000, exponent: 5 };  // Not normalized (MSB != 01)
    let normalized = ScalarF4E4 { fraction: 0x4000, exponent: 5 };    // Normalized (MSB = 01)
    println!("  Denormalized input: {:?}", denormalized);
    println!("  Normalized input: {:?}", normalized);

    let result_denorm = denormalized / five;
    let result_norm = normalized / five;
    println!("  Denormalized/5 = {:?}", result_denorm);
    println!("  Normalized/5 = {:?}", result_norm);
}

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        COMPREHENSIVE DIVISION TEST SUITE                 ║");
    println!("║        Testing all bit widths: F3E3, F4E4, F5E5,        ║");
    println!("║        F6E6, F7E7 with full edge case coverage          ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    // Test all bit widths
    test_scalar::<ScalarF3E3>("ScalarF3E3 (8-bit)");
    test_scalar::<ScalarF4E4>("ScalarF4E4 (16-bit)");
    test_scalar::<ScalarF5E5>("ScalarF5E5 (32-bit)");
    test_scalar::<ScalarF6E6>("ScalarF6E6 (64-bit)");
    test_scalar::<ScalarF7E7>("ScalarF7E7 (128-bit)");

    // Test edge cases
    test_edge_cases();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    TESTING COMPLETE                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}
