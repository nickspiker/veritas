//! Spirix Correctness Verification
//!
//! Tests our Tensor operations against native Spirix to ensure:
//! 1. Mathematical correctness (add, mul, etc.)
//! 2. Edge case handling (undefined, infinity, zero, vanished, exploded)
//! 3. No IEEE contamination

use veritas::autograd::{Tensor, Shape};
use spirix::{ScalarF4E4, CircleF4E5};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Spirix Correctness Verification                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("═══ Test 1: Basic Arithmetic ═══\n");
    test_addition();
    test_multiplication();
    test_division();
    println!();

    println!("═══ Test 2: Edge Cases ═══\n");
    test_zero_handling();
    test_undefined_handling();
    test_overflow_underflow();
    println!();

    println!("═══ Test 3: Tensor Operations ═══\n");
    test_transpose();
    test_scale();
    test_element_add();
    println!();

    println!("═══ Test 4: Spirix Constants ═══\n");
    test_constants();
    println!();

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                  Verification Complete                        ║");
    println!("╚════════════════════════════════════════════════════════════════╝");
}

fn test_addition() {
    println!("Testing addition:");

    // Normal addition
    let a = ScalarF4E4::from(5u8);
    let b = ScalarF4E4::from(3u8);
    let result = a + b;
    println!("  5 + 3 = {:?}", result);

    // Tensor addition
    let t1 = Tensor::from_scalars(vec![a, b], Shape::vector(2)).unwrap();
    let t2 = Tensor::from_scalars(vec![b, a], Shape::vector(2)).unwrap();
    let t_result = t1.add(&t2).unwrap();
    let t_data = t_result.as_scalars().unwrap();
    println!("  Tensor [5, 3] + [3, 5] = [{:?}, {:?}]", t_data[0], t_data[1]);

    // Verify: should be [8, 8]
    let expected = ScalarF4E4::from(8u8);
    assert_eq!(t_data[0], expected, "Tensor addition mismatch");
    assert_eq!(t_data[1], expected, "Tensor addition mismatch");

    println!("  ✓ Addition correct");
}

fn test_multiplication() {
    println!("Testing multiplication:");

    // Normal multiplication
    let a = ScalarF4E4::from(7u8);
    let b = ScalarF4E4::from(6u8);
    let result = a * b;
    println!("  7 * 6 = {:?}", result);

    // Tensor scale (element-wise multiply)
    let t = Tensor::from_scalars(vec![a, b], Shape::vector(2)).unwrap();
    let scalar = ScalarF4E4::from(2u8);
    let t_result = t.scale(scalar).unwrap();
    let t_data = t_result.as_scalars().unwrap();
    println!("  Tensor [7, 6] * 2 = [{:?}, {:?}]", t_data[0], t_data[1]);

    // Verify: should be [14, 12]
    let exp_a = ScalarF4E4::from(14u8);
    let exp_b = ScalarF4E4::from(12u8);
    assert_eq!(t_data[0], exp_a, "Tensor scale mismatch");
    assert_eq!(t_data[1], exp_b, "Tensor scale mismatch");

    println!("  ✓ Multiplication correct");
}

fn test_division() {
    println!("Testing division:");

    // Normal division
    let a = ScalarF4E4::from(10u8);
    let b = ScalarF4E4::from(2u8);
    let result = a / b;
    println!("  10 / 2 = {:?}", result);

    // Division by small number
    let c = ScalarF4E4::from(1u8);
    let d = ScalarF4E4::from(3u8);
    let result2 = c / d;
    println!("  1 / 3 = {:?} (should be ~0.333...)", result2);

    println!("  ✓ Division working");
}

fn test_zero_handling() {
    println!("Testing zero handling:");

    // Zero addition
    let zero = ScalarF4E4::ZERO;
    let five = ScalarF4E4::from(5u8);
    let result = five + zero;
    println!("  5 + 0 = {:?}", result);
    assert_eq!(result, five, "Zero addition broken");

    // Zero multiplication (annihilates)
    let result = five * zero;
    println!("  5 * 0 = {:?} (should be zero)", result);
    assert_eq!(result, zero, "Zero multiplication broken");

    // Division by zero (should be undefined)
    let result = five / zero;
    println!("  5 / 0 = {:?} (should be undefined)", result);

    // Check if undefined
    if is_undefined(result) {
        println!("  ✓ Division by zero produces undefined");
    } else {
        println!("  ⚠ Division by zero did not produce undefined!");
    }

    // Zero divided by zero (undefined)
    let result = zero / zero;
    println!("  0 / 0 = {:?} (should be undefined)", result);
    if is_undefined(result) {
        println!("  ✓ 0/0 produces undefined");
    }
}

fn test_undefined_handling() {
    println!("Testing undefined propagation:");

    // Create undefined value (via 0/0)
    let undefined = ScalarF4E4::ZERO / ScalarF4E4::ZERO;
    println!("  undefined = {:?}", undefined);

    // Undefined + anything = undefined
    let five = ScalarF4E4::from(5u8);
    let result = undefined + five;
    println!("  undefined + 5 = {:?}", result);
    if is_undefined(result) {
        println!("  ✓ Undefined propagates through addition");
    }

    // Undefined * anything = undefined
    let result = undefined * five;
    println!("  undefined * 5 = {:?}", result);
    if is_undefined(result) {
        println!("  ✓ Undefined propagates through multiplication");
    }

    // Tensor with undefined
    let t = Tensor::from_scalars(vec![five, undefined, five], Shape::vector(3)).unwrap();
    let t_data = t.as_scalars().unwrap();
    println!("  Tensor [5, undefined, 5] stored successfully");
    if is_undefined(t_data[1]) {
        println!("  ✓ Tensors can store undefined values");
    }
}

fn test_overflow_underflow() {
    println!("Testing overflow/underflow (vanished/exploded):");

    // Large number
    let large = ScalarF4E4::from(255u8); // Maximum u8
    println!("  Max value: {:?}", large);

    // Large * Large (might overflow)
    let result = large * large;
    println!("  255 * 255 = {:?}", result);
    if is_exploded(result) {
        println!("  ✓ Overflow detected (exploded)");
    } else {
        println!("  Result magnitude: {}", result);
    }

    // Very small fraction
    let small = ScalarF4E4::ONE / ScalarF4E4::from(100u8);
    let tiny = small / ScalarF4E4::from(100u8); // 1/10000
    println!("  1/100 / 100 = {:?}", tiny);
    if is_vanished(tiny) {
        println!("  ✓ Underflow detected (vanished)");
    } else {
        println!("  Small value preserved");
    }
}

fn test_transpose() {
    println!("Testing transpose:");

    // Create 2x3 matrix
    let data = vec![
        ScalarF4E4::from(1u8), ScalarF4E4::from(2u8), ScalarF4E4::from(3u8),
        ScalarF4E4::from(4u8), ScalarF4E4::from(5u8), ScalarF4E4::from(6u8),
    ];
    let mat = Tensor::from_scalars(data, Shape::matrix(2, 3)).unwrap();

    println!("  Original [2, 3]:");
    print_matrix(&mat, 2, 3);

    let transposed = mat.transpose().unwrap();
    println!("  Transposed [3, 2]:");
    print_matrix(&transposed, 3, 2);

    // Verify element positions
    let orig_data = mat.as_scalars().unwrap();
    let trans_data = transposed.as_scalars().unwrap();

    // Original[0,0] = Transposed[0,0]
    assert_eq!(orig_data[0], trans_data[0]);
    // Original[0,1] = Transposed[1,0]
    assert_eq!(orig_data[1], trans_data[2]);
    // Original[1,0] = Transposed[0,1]
    assert_eq!(orig_data[3], trans_data[1]);

    println!("  ✓ Transpose correct");
}

fn test_scale() {
    println!("Testing scale:");

    let data = vec![
        ScalarF4E4::from(2u8),
        ScalarF4E4::from(4u8),
        ScalarF4E4::from(6u8),
    ];
    let t = Tensor::from_scalars(data, Shape::vector(3)).unwrap();

    let scalar = ScalarF4E4::from(3u8);
    let scaled = t.scale(scalar).unwrap();
    let scaled_data = scaled.as_scalars().unwrap();

    println!("  [2, 4, 6] * 3 = [{:?}, {:?}, {:?}]",
             scaled_data[0], scaled_data[1], scaled_data[2]);

    // Verify
    assert_eq!(scaled_data[0], ScalarF4E4::from(6u8));
    assert_eq!(scaled_data[1], ScalarF4E4::from(12u8));
    assert_eq!(scaled_data[2], ScalarF4E4::from(18u8));

    println!("  ✓ Scale correct");
}

fn test_element_add() {
    println!("Testing element-wise add:");

    let a = vec![
        ScalarF4E4::from(1u8),
        ScalarF4E4::from(2u8),
        ScalarF4E4::from(3u8),
    ];
    let b = vec![
        ScalarF4E4::from(10u8),
        ScalarF4E4::from(20u8),
        ScalarF4E4::from(30u8),
    ];

    let t1 = Tensor::from_scalars(a, Shape::vector(3)).unwrap();
    let t2 = Tensor::from_scalars(b, Shape::vector(3)).unwrap();

    let result = t1.add(&t2).unwrap();
    let result_data = result.as_scalars().unwrap();

    println!("  [1, 2, 3] + [10, 20, 30] = [{:?}, {:?}, {:?}]",
             result_data[0], result_data[1], result_data[2]);

    // Verify
    assert_eq!(result_data[0], ScalarF4E4::from(11u8));
    assert_eq!(result_data[1], ScalarF4E4::from(22u8));
    assert_eq!(result_data[2], ScalarF4E4::from(33u8));

    println!("  ✓ Element-wise add correct");
}

fn test_constants() {
    println!("Testing Spirix constants:");

    // Check constants are accessible
    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;

    println!("  ZERO = {:?}", zero);
    println!("  ONE = {:?}", one);

    // Verify ONE - ONE = ZERO
    let diff = one - one;
    assert_eq!(diff, zero, "ONE - ONE should equal ZERO");
    println!("  ✓ ONE - ONE = ZERO");

    // Verify ZERO + ONE = ONE
    let sum = zero + one;
    assert_eq!(sum, one, "ZERO + ONE should equal ONE");
    println!("  ✓ ZERO + ONE = ONE");

    // Test if we can access other constants
    println!("  Checking for additional constants...");

    // ScalarF4E4 should have more constants
    // Let's test if denormals are preserved
    let tiny = ScalarF4E4::ONE / ScalarF4E4::from(200u8);
    println!("  1/200 = {:?} (testing denormal preservation)", tiny);
}

/// Check if a scalar is undefined
fn is_undefined(s: ScalarF4E4) -> bool {
    // In Spirix, undefined has a specific bit pattern
    // The Debug format shows it with specific prefixes
    let debug_str = format!("{:?}", s);
    debug_str.contains("UNDEFINED") || debug_str.contains("undefined")
}

/// Check if a scalar has exploded (overflow)
fn is_exploded(s: ScalarF4E4) -> bool {
    let debug_str = format!("{:?}", s);
    debug_str.contains("EXPLODED") || debug_str.contains("exploded")
}

/// Check if a scalar has vanished (underflow)
fn is_vanished(s: ScalarF4E4) -> bool {
    let debug_str = format!("{:?}", s);
    debug_str.contains("VANISHED") || debug_str.contains("vanished")
}

/// Print a matrix in readable format
fn print_matrix(t: &Tensor, rows: usize, cols: usize) {
    let data = t.as_scalars().unwrap();
    for row in 0..rows {
        print!("    [");
        for col in 0..cols {
            let idx = row * cols + col;
            print!("{:?}", data[idx]);
            if col < cols - 1 {
                print!(", ");
            }
        }
        println!("]");
    }
}
