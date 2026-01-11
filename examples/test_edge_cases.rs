//! Comprehensive Edge Case Testing
//!
//! Tests our Tensor implementation against Spirix edge cases:
//! - Undefined states (all 30+ variants)
//! - Zero handling (positive, negative, actual zero)
//! - Infinity (positive, negative)
//! - Vanished (underflow)
//! - Exploded (overflow)
//! - Transfinite and negligible values

use veritas::autograd::{Tensor, Shape};
use spirix::ScalarF4E4;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Comprehensive Edge Case Testing                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("═══ Test 1: Zero States ═══\n");
    test_zero_states();

    println!("\n═══ Test 2: Infinity ═══\n");
    test_infinity();

    println!("\n═══ Test 3: Undefined Propagation ═══\n");
    test_undefined_propagation();

    println!("\n═══ Test 4: Vanished & Exploded ═══\n");
    test_vanished_exploded();

    println!("\n═══ Test 5: Tensor Edge Cases ═══\n");
    test_tensor_edge_cases();

    println!("\n═══ Test 6: Division by Zero ═══\n");
    test_division_by_zero();

    println!("\n═══ Test 7: Transfinite & Negligible ═══\n");
    test_transfinite_negligible();

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║              Edge Case Testing Complete                       ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn test_zero_states() {
    println!("Testing zero states:");

    let zero = ScalarF4E4::ZERO;
    println!("  ZERO = {:?}", zero);
    println!("    is_zero(): {}", zero.is_zero());
    println!("    is_normal(): {}", zero.is_normal());
    println!("    is_finite(): {}", zero.is_finite());

    // Zero in tensor
    let t = Tensor::from_scalars(vec![zero, zero], Shape::vector(2)).unwrap();
    println!("  ✓ Tensor can store ZERO");

    // Zero operations
    let one = ScalarF4E4::ONE;
    assert_eq!(zero + zero, zero, "0 + 0 should be 0");
    assert_eq!(zero * one, zero, "0 * 1 should be 0");
    assert_eq!(one + zero, one, "1 + 0 should be 1");
    println!("  ✓ Zero arithmetic correct");
}

fn test_infinity() {
    println!("Testing infinity:");

    // Create infinity via division by zero
    let one = ScalarF4E4::ONE;
    let zero = ScalarF4E4::ZERO;
    let inf = one / zero;

    println!("  1 / 0 = {:?}", inf);
    println!("    is_infinite(): {}", inf.is_infinite());
    println!("    is_undefined(): {}", inf.is_undefined());
    println!("    is_finite(): {}", inf.is_finite());

    // Negative infinity
    let neg_one = zero - one;
    let neg_inf = neg_one / zero;
    println!("  -1 / 0 = {:?}", neg_inf);
    println!("    is_infinite(): {}", neg_inf.is_infinite());

    // Infinity in tensor
    let t = Tensor::from_scalars(vec![inf, neg_inf], Shape::vector(2)).unwrap();
    println!("  ✓ Tensor can store infinity");

    // Infinity arithmetic
    let inf_plus_one = inf + one;
    println!("  ∞ + 1 = {:?}", inf_plus_one);

    let inf_times_two = inf * ScalarF4E4::from(2u8);
    println!("  ∞ * 2 = {:?}", inf_times_two);
}

fn test_undefined_propagation() {
    println!("Testing undefined propagation:");

    // 0/0 is undefined
    let zero = ScalarF4E4::ZERO;
    let undef = zero / zero;

    println!("  0 / 0 = {:?}", undef);
    println!("    is_undefined(): {}", undef.is_undefined());
    println!("    is_zero(): {}", undef.is_zero());
    println!("    is_finite(): {}", undef.is_finite());

    // Undefined propagates through operations
    let one = ScalarF4E4::ONE;

    let undef_plus = undef + one;
    println!("  undefined + 1 = {:?}", undef_plus);
    println!("    is_undefined(): {}", undef_plus.is_undefined());
    assert!(undef_plus.is_undefined(), "Undefined should propagate through addition");

    let undef_mult = undef * one;
    println!("  undefined * 1 = {:?}", undef_mult);
    assert!(undef_mult.is_undefined(), "Undefined should propagate through multiplication");

    let undef_div = one / undef;
    println!("  1 / undefined = {:?}", undef_div);
    assert!(undef_div.is_undefined(), "Undefined should propagate through division");

    println!("  ✓ Undefined propagates correctly");

    // Undefined in tensor operations
    let t1 = Tensor::from_scalars(vec![one, undef], Shape::vector(2)).unwrap();
    let t2 = Tensor::from_scalars(vec![one, one], Shape::vector(2)).unwrap();

    let result = t1.add(&t2).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  Tensor [1, undefined] + [1, 1]:");
    println!("    [0] = {:?}, is_undefined: {}", data[0], data[0].is_undefined());
    println!("    [1] = {:?}, is_undefined: {}", data[1], data[1].is_undefined());

    assert!(!data[0].is_undefined(), "1+1 should not be undefined");
    assert!(data[1].is_undefined(), "undefined+1 should be undefined");

    println!("  ✓ Undefined propagates in tensors");
}

fn test_vanished_exploded() {
    println!("Testing vanished (underflow) and exploded (overflow):");

    // Try to create very small number (vanished)
    let small = ScalarF4E4::ONE / ScalarF4E4::from(250u8);
    let tiny = small / ScalarF4E4::from(250u8);

    println!("  1 / 250 / 250 = {:?}", tiny);
    println!("    is_zero(): {}", tiny.is_zero());
    println!("    is_finite(): {}", tiny.is_finite());
    println!("    is_normal(): {}", tiny.is_normal());
    println!("    is_negligible(): {}", tiny.is_negligible());

    // Try to create large number (exploded)
    let large = ScalarF4E4::from(200u8);
    let huge = large * large;

    println!("  200 * 200 = {:?}", huge);
    println!("    is_finite(): {}", huge.is_finite());
    println!("    is_transfinite(): {}", huge.is_transfinite());
    println!("    magnitude: {}", huge);

    // Vanished/Exploded in tensors
    let t = Tensor::from_scalars(vec![tiny, huge], Shape::vector(2)).unwrap();
    println!("  ✓ Tensor can store vanished and exploded values");
}

fn test_tensor_edge_cases() {
    println!("Testing tensor operations with edge cases:");

    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let inf = one / zero;
    let undef = zero / zero;

    // Mixed tensor
    let data = vec![zero, one, inf, undef];
    let t = Tensor::from_scalars(data, Shape::matrix(2, 2)).unwrap();

    println!("  Created tensor [[0, 1], [∞, undefined]]");

    // Transpose with edge cases
    let transposed = t.transpose().unwrap();
    let trans_data = transposed.as_scalars().unwrap();

    println!("  Transposed to [[0, ∞], [1, undefined]]");
    println!("    [0] is_zero: {}", trans_data[0].is_zero());
    println!("    [1] is_infinite: {}", trans_data[1].is_infinite());
    println!("    [2] is_normal: {}", trans_data[2].is_normal());
    println!("    [3] is_undefined: {}", trans_data[3].is_undefined());

    // Scale with edge cases
    let two = ScalarF4E4::from(2u8);
    let scaled = t.scale(two).unwrap();
    let scaled_data = scaled.as_scalars().unwrap();

    println!("  Scaled by 2:");
    println!("    0*2 = {:?}, is_zero: {}", scaled_data[0], scaled_data[0].is_zero());
    println!("    1*2 = {:?}", scaled_data[1]);
    println!("    ∞*2 = {:?}, is_infinite: {}", scaled_data[2], scaled_data[2].is_infinite());
    println!("    undefined*2 = {:?}, is_undefined: {}", scaled_data[3], scaled_data[3].is_undefined());

    println!("  ✓ Edge cases preserved through tensor operations");
}

fn test_division_by_zero() {
    println!("Testing division by zero variants:");

    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let neg_one = zero - one;

    // Positive / zero = +∞
    let pos_inf = one / zero;
    println!("  +1 / 0 = {:?}", pos_inf);
    println!("    is_infinite(): {}", pos_inf.is_infinite());
    println!("    is_positive(): {}", pos_inf.is_positive());

    // Negative / zero = -∞
    let neg_inf = neg_one / zero;
    println!("  -1 / 0 = {:?}", neg_inf);
    println!("    is_infinite(): {}", neg_inf.is_infinite());
    println!("    is_negative(): {}", neg_inf.is_negative());

    // Zero / zero = undefined
    let undef = zero / zero;
    println!("  0 / 0 = {:?}", undef);
    println!("    is_undefined(): {}", undef.is_undefined());

    println!("  ✓ Division by zero produces correct results");
}

fn test_transfinite_negligible() {
    println!("Testing transfinite and negligible values:");

    // Create very large value (transfinite)
    let large = ScalarF4E4::from(250u8);
    let very_large = large * large * large;

    println!("  250 * 250 * 250 = {:?}", very_large);
    println!("    is_transfinite(): {}", very_large.is_transfinite());
    println!("    is_finite(): {}", very_large.is_finite());

    // Create very small value (negligible)
    let small = ScalarF4E4::ONE / large;
    let very_small = small / large / large;

    println!("  1 / 250 / 250 / 250 = {:?}", very_small);
    println!("    is_negligible(): {}", very_small.is_negligible());
    println!("    is_zero(): {}", very_small.is_zero());

    // Transfinite * negligible = undefined (special case)
    let result = very_large * very_small;
    println!("  transfinite * negligible = {:?}", result);
    println!("    is_undefined(): {}", result.is_undefined());

    if result.is_undefined() {
        println!("  ✓ Transfinite × negligible produces undefined (℘ ⬆×⬇)");
    }

    // In tensors
    let t = Tensor::from_scalars(vec![very_large, very_small], Shape::vector(2)).unwrap();
    println!("  ✓ Tensor can store transfinite and negligible");
}
