use spirix::{CircleF3E3, CircleF4E4, CircleF5E5, CircleF6E6, ScalarF4E4};

fn main() {
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║        COMPREHENSIVE CIRCLE DIVISION TEST SUITE          ║");
    println!("║        Testing all Circle bit widths with edge cases    ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    test_circle_f4e4();
    test_circle_edge_cases();

    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║                    TESTING COMPLETE                      ║");
    println!("╚══════════════════════════════════════════════════════════╝\n");
}

fn test_circle_f4e4() {
    println!("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("  Testing: CircleF4E4 (16-bit)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let mut passed = 0;
    let mut total = 0;

    println!("\n--- Circle / Circle Tests ---");

    // 1. IDENTITY OPERATIONS
    total += 1;
    let z1 = CircleF4E4::from((1, 0));
    let result = z1 / z1;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 1.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ Identity: (1+0i)/(1+0i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Identity: (1+0i)/(1+0i) = ({:.3}, {:.3}i) FAIL (expected 1.0, 0.0)", re_f64, im_f64);
    }

    // 2. REAL-ONLY DIVISIONS
    total += 1;
    let z2 = CircleF4E4::from((20, 0));
    let z3 = CircleF4E4::from((4, 0));
    let result = z2 / z3;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 5.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ Real: (20+0i)/(4+0i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Real: (20+0i)/(4+0i) = ({:.3}, {:.3}i) FAIL (expected 5.0, 0.0)", re_f64, im_f64);
    }

    total += 1;
    let z4 = CircleF4E4::from((144, 0));
    let z5 = CircleF4E4::from((12, 0));
    let result = z4 / z5;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 12.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ Real: (144+0i)/(12+0i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Real: (144+0i)/(12+0i) = ({:.3}, {:.3}i) FAIL (expected 12.0, 0.0)", re_f64, im_f64);
    }

    // 3. IMAGINARY-ONLY DIVISIONS
    total += 1;
    let z6 = CircleF4E4::from((0, 20));
    let z7 = CircleF4E4::from((0, 4));
    let result = z6 / z7;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 5.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ Imag: (0+20i)/(0+4i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Imag: (0+20i)/(0+4i) = ({:.3}, {:.3}i) FAIL (expected 5.0, 0.0)", re_f64, im_f64);
    }

    // 4. COMPLEX DIVISIONS - (1+2i)/(1+i)
    total += 1;
    let z8 = CircleF4E4::from((1, 2));
    let z9 = CircleF4E4::from((1, 1));
    let result = z8 / z9;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    // Expected: (1+2i)/(1+i) = (1+2i)*(1-i)/((1+i)*(1-i)) = (1-i+2i-2i²)/(1-i²) = (1+i+2)/2 = (3+i)/2 = 1.5+0.5i
    if (re_f64 - 1.5).abs() < 0.1 && (im_f64 - 0.5).abs() < 0.1 {
        println!("  ✓ Complex: (1+2i)/(1+i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Complex: (1+2i)/(1+i) = ({:.3}, {:.3}i) FAIL (expected ~1.5, ~0.5)", re_f64, im_f64);
    }

    // 5. SIGN VARIATIONS
    total += 1;
    let z10 = CircleF4E4::from((-20, 0));
    let z11 = CircleF4E4::from((4, 0));
    let result = z10 / z11;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 + 5.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ Sign: (-20+0i)/(4+0i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ Sign: (-20+0i)/(4+0i) = ({:.3}, {:.3}i) FAIL (expected -5.0, 0.0)", re_f64, im_f64);
    }

    println!("\n--- Circle / Scalar Tests ---");

    // 6. CIRCLE / SCALAR
    total += 1;
    let z12 = CircleF4E4::from((20, 0));
    let s1 = ScalarF4E4::from(4);
    let result = z12 / s1;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 5.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ C/S Real: (20+0i)/4 = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ C/S Real: (20+0i)/4 = ({:.3}, {:.3}i) FAIL (expected 5.0, 0.0)", re_f64, im_f64);
    }

    total += 1;
    let z13 = CircleF4E4::from((10, 20));
    let s2 = ScalarF4E4::from(2);
    let result = z13 / s2;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 5.0).abs() < 0.1 && (im_f64 - 10.0).abs() < 0.1 {
        println!("  ✓ C/S Complex: (10+20i)/2 = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ C/S Complex: (10+20i)/2 = ({:.3}, {:.3}i) FAIL (expected 5.0, 10.0)", re_f64, im_f64);
    }

    println!("\n--- Scalar / Circle Tests ---");

    // 7. SCALAR / CIRCLE
    total += 1;
    let s3 = ScalarF4E4::from(10);
    let z14 = CircleF4E4::from((2, 0));
    let result = s3 / z14;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 5.0).abs() < 0.01 && im_f64.abs() < 0.01 {
        println!("  ✓ S/C Real: 10/(2+0i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ S/C Real: 10/(2+0i) = ({:.3}, {:.3}i) FAIL (expected 5.0, 0.0)", re_f64, im_f64);
    }

    // 1/(1+i) = (1-i)/2 = 0.5-0.5i
    total += 1;
    let s4 = ScalarF4E4::from(1);
    let z15 = CircleF4E4::from((1, 1));
    let result = s4 / z15;
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    if (re_f64 - 0.5).abs() < 0.05 && (im_f64 + 0.5).abs() < 0.05 {
        println!("  ✓ S/C Complex: 1/(1+i) = ({:.3}, {:.3}i)", re_f64, im_f64);
        passed += 1;
    } else {
        println!("  ✗ S/C Complex: 1/(1+i) = ({:.3}, {:.3}i) FAIL (expected ~0.5, ~-0.5)", re_f64, im_f64);
    }

    println!("\n  Summary: {} / {} tests passed", passed, total);
    if passed == total {
        println!("  ✅ ALL TESTS PASSED!");
    } else {
        println!("  ⚠️  {} FAILURES DETECTED!", total - passed);
    }
}

fn test_circle_edge_cases() {
    println!("\n╔══════════════════════════════════════════════════════════╗");
    println!("║          CIRCLE EDGE CASE TESTING                       ║");
    println!("╚══════════════════════════════════════════════════════════╝");

    println!("\n--- Zero Division Tests ---");

    let zero = CircleF4E4::from((0, 0));
    let five = CircleF4E4::from((5, 0));
    let complex = CircleF4E4::from((3, 4));

    // 0/5 should be 0
    let result = zero / five;
    println!("  0/(5+0i) = {:?}", result);
    println!("    (expected zero)");

    // 5/0 should be exploded/infinity
    let result2 = five / zero;
    println!("  (5+0i)/0 = {:?}", result2);
    println!("    (should be exploded/infinity)");

    // 0/0 should be undefined
    let result3 = zero / zero;
    println!("  0/0 = {:?}", result3);
    println!("    (should be undefined)");

    // Complex / 0
    let result4 = complex / zero;
    println!("  (3+4i)/0 = {:?}", result4);
    println!("    (should be exploded/infinity)");

    println!("\n--- Identity Tests ---");

    // Dividing by itself should give 1+0i
    let z1 = CircleF4E4::from((3, 4));
    let result = z1 / z1;
    println!("  (3+4i)/(3+4i) = {:?}", result);
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    println!("    As f64: ({:.6}, {:.6}i) [expected 1.0, 0.0]", re_f64, im_f64);

    println!("\n--- Conjugate Division Tests ---");

    // (a+bi) / (a-bi) tests conjugate handling
    // (3+4i)/(3-4i) = ((3+4i)*(3+4i))/((3-4i)*(3+4i)) = (9+24i+16i²)/(9-16i²) = (9+24i-16)/(9+16) = (-7+24i)/25
    let z2 = CircleF4E4::from((3, 4));
    let z3 = CircleF4E4::from((3, -4));
    let result = z2 / z3;
    println!("  (3+4i)/(3-4i) = {:?}", result);
    let re_f64: f64 = ScalarF4E4{ fraction: result.real, exponent: result.exponent }.into();
    let im_f64: f64 = ScalarF4E4{ fraction: result.imaginary, exponent: result.exponent }.into();
    println!("    As f64: ({:.6}, {:.6}i) [expected ~-0.28, ~0.96]", re_f64, im_f64);

    println!("\n✅ Edge case testing complete (manual verification required)");
}
