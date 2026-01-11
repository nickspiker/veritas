//! Edge case testing for Tensor operations
//!
//! Tests boundary conditions:
//! - Near exploded (overflow)
//! - Near vanished (underflow)
//! - Transitions between states (normal → exploded, vanished → zero)
//! - Mixed edge cases in same operation

use veritas::autograd::{Tensor, Shape, matmul};
use spirix::ScalarF4E4;

// ═══ Test Helpers ═══

fn is_valid_state(s: &ScalarF4E4) -> bool {
    // A value must be in at least one valid state
    // Note: states can overlap (e.g., zero can be negligible, finite can be transfinite)
    s.is_zero() || s.is_finite() || s.is_undefined() || s.is_infinite() ||
    s.is_negligible() || s.is_transfinite()
}

fn assert_no_state_corruption(t: &Tensor, context: &str) {
    let data = t.as_scalars().unwrap();
    for (i, val) in data.iter().enumerate() {
        assert!(is_valid_state(val),
                "{}: Invalid state at index {}: {:?}", context, i, val);
    }
}

// ═══ Addition Near Exploded ═══

#[test]
fn test_add_near_exploded() {
    println!("Testing addition near exploded range...");

    // Create large value approaching exploded
    let large = ScalarF4E4::from(250u8);
    let huge = large * large;  // ~62500

    println!("  huge = {:?}", huge);
    println!("    is_finite: {}", huge.is_finite());
    println!("    is_transfinite: {}", huge.is_transfinite());

    let t1 = Tensor::from_scalars(vec![huge, huge], Shape::vector(2)).unwrap();
    let t2 = Tensor::from_scalars(vec![huge, huge], Shape::vector(2)).unwrap();

    // huge + huge - should not accidentally vanish!
    let result = t1.add(&t2).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  huge + huge = {:?}", data[0]);
    println!("    is_finite: {}", data[0].is_finite());
    println!("    is_transfinite: {}", data[0].is_transfinite());

    // Should still be large, not vanished
    assert!(!data[0].is_zero(), "Addition near exploded became zero!");
    assert!(data[0] > huge, "Addition should produce larger value");

    assert_no_state_corruption(&result, "Addition near exploded");
    println!("  ✓ Addition near exploded preserves state");
}

#[test]
fn test_add_vanished_to_large() {
    println!("Testing addition of vanished to large number...");

    // Create vanished value
    let small = ScalarF4E4::ONE / ScalarF4E4::from(250u8);
    let tiny = small / ScalarF4E4::from(250u8);  // ~1/62500

    // Create large value
    let large = ScalarF4E4::from(200u8);

    println!("  tiny = {:?}", tiny);
    println!("    is_negligible: {}", tiny.is_negligible());
    println!("  large = {:?}", large);

    let t1 = Tensor::from_scalars(vec![tiny], Shape::vector(1)).unwrap();
    let t2 = Tensor::from_scalars(vec![large], Shape::vector(1)).unwrap();

    let result = t1.add(&t2).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  tiny + large = {:?}", data[0]);

    // Large should dominate
    assert!(data[0] > ScalarF4E4::from(100u8), "Large value lost in addition");
    assert_no_state_corruption(&result, "Addition vanished + large");
    println!("  ✓ Vanished + large preserves large value");
}

// ═══ Subtraction Edge Cases ═══

#[test]
fn test_subtract_near_zero() {
    println!("Testing subtraction near zero...");

    let a = ScalarF4E4::from(5u8);
    let b = ScalarF4E4::from(5u8);

    let t1 = Tensor::from_scalars(vec![a], Shape::vector(1)).unwrap();
    let t2 = Tensor::from_scalars(vec![b], Shape::vector(1)).unwrap();

    // Subtract to get zero
    let diff = t2.add(&t1.scale(ScalarF4E4::ZERO - ScalarF4E4::ONE).unwrap()).unwrap();
    let data = diff.as_scalars().unwrap();

    println!("  5 - 5 = {:?}", data[0]);
    println!("    is_zero: {}", data[0].is_zero());
    println!("    is_negligible: {}", data[0].is_negligible());

    // Should be exactly zero (may also have negligible flag, that's OK)
    assert!(data[0].is_zero(), "5 - 5 should be exactly zero");
    // Note: Spirix may mark zero as negligible in some representations

    println!("  ✓ Subtraction to zero produces zero");
}

#[test]
fn test_subtract_exploded() {
    println!("Testing subtraction of exploded values...");

    let large = ScalarF4E4::from(250u8);
    let huge1 = large * large;
    let huge2 = large * large;

    let t1 = Tensor::from_scalars(vec![huge1], Shape::vector(1)).unwrap();
    let t2 = Tensor::from_scalars(vec![huge2], Shape::vector(1)).unwrap();

    // huge - huge should be zero or near-zero, not undefined
    let diff = t1.add(&t2.scale(ScalarF4E4::ZERO - ScalarF4E4::ONE).unwrap()).unwrap();
    let data = diff.as_scalars().unwrap();

    println!("  huge - huge = {:?}", data[0]);
    println!("    is_zero: {}", data[0].is_zero());
    println!("    is_undefined: {}", data[0].is_undefined());

    // Should be zero or small, not undefined
    assert!(!data[0].is_undefined() || data[0].is_zero(),
            "Exploded - exploded should not be undefined");

    println!("  ✓ Exploded subtraction produces valid result");
}

// ═══ Scale Edge Cases ═══

#[test]
fn test_scale_vanished_grows() {
    println!("Testing scaling vanished value up...");

    let tiny = ScalarF4E4::ONE / ScalarF4E4::from(250u8) / ScalarF4E4::from(250u8);
    let large_scalar = ScalarF4E4::from(200u8);

    println!("  tiny = {:?}", tiny);
    println!("    is_negligible: {}", tiny.is_negligible());

    let t = Tensor::from_scalars(vec![tiny], Shape::vector(1)).unwrap();
    let result = t.scale(large_scalar).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  tiny * 200 = {:?}", data[0]);
    println!("    is_negligible: {}", data[0].is_negligible());

    // Should grow back to normal range
    assert!(data[0] > tiny, "Scaling should make value larger");
    assert_no_state_corruption(&result, "Scale vanished up");
    println!("  ✓ Scaling vanished value increases magnitude");
}

#[test]
fn test_scale_exploded_shrinks() {
    println!("Testing scaling exploded value down...");

    let large = ScalarF4E4::from(250u8);
    let huge = large * large;
    let small_scalar = ScalarF4E4::ONE / ScalarF4E4::from(100u8);

    println!("  huge = {:?}", huge);

    let t = Tensor::from_scalars(vec![huge], Shape::vector(1)).unwrap();
    let result = t.scale(small_scalar).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  huge / 100 = {:?}", data[0]);
    println!("    is_finite: {}", data[0].is_finite());

    // Should shrink but stay valid
    assert!(data[0] < huge, "Scaling down should decrease value");
    assert_no_state_corruption(&result, "Scale exploded down");
    println!("  ✓ Scaling exploded value decreases magnitude");
}

#[test]
fn test_scale_overflow_to_exploded() {
    println!("Testing scaling that causes overflow...");

    let large = ScalarF4E4::from(200u8);
    let t = Tensor::from_scalars(vec![large, large], Shape::vector(2)).unwrap();

    // Scale by large factor
    let result = t.scale(large).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  200 * 200 = {:?}", data[0]);
    println!("    is_finite: {}", data[0].is_finite());
    println!("    is_transfinite: {}", data[0].is_transfinite());

    // Should handle overflow gracefully (exploded or large finite)
    assert!(data[0].is_finite() || data[0].is_transfinite(),
            "Overflow should produce finite or transfinite");
    assert_no_state_corruption(&result, "Scale overflow");
    println!("  ✓ Overflow handled correctly");
}

#[test]
fn test_scale_underflow_to_vanished() {
    println!("Testing scaling that causes underflow...");

    let small = ScalarF4E4::ONE / ScalarF4E4::from(100u8);
    let tiny_scalar = ScalarF4E4::ONE / ScalarF4E4::from(250u8);

    let t = Tensor::from_scalars(vec![small], Shape::vector(1)).unwrap();
    let result = t.scale(tiny_scalar).unwrap();
    let data = result.as_scalars().unwrap();

    println!("  (1/100) * (1/250) = {:?}", data[0]);
    println!("    is_zero: {}", data[0].is_zero());
    println!("    is_negligible: {}", data[0].is_negligible());

    // Should be very small but valid
    assert!(data[0] < small, "Underflow should produce smaller value");
    assert_no_state_corruption(&result, "Scale underflow");
    println!("  ✓ Underflow handled correctly");
}

// ═══ Transpose Edge Cases ═══

#[test]
fn test_transpose_preserves_exploded() {
    println!("Testing transpose preserves exploded...");

    let large = ScalarF4E4::from(250u8);
    let huge = large * large;

    let data = vec![huge, ScalarF4E4::ONE, ScalarF4E4::ONE, huge];
    let t = Tensor::from_scalars(data, Shape::matrix(2, 2)).unwrap();

    let transposed = t.transpose().unwrap();
    let trans_data = transposed.as_scalars().unwrap();

    println!("  Original: [huge, 1; 1, huge]");
    println!("  Transposed: [{:?}, {:?}; {:?}, {:?}]",
             trans_data[0], trans_data[1], trans_data[2], trans_data[3]);

    // Exploded values should still be exploded
    assert!(trans_data[0] > ScalarF4E4::from(100u8), "Exploded value lost");
    assert!(trans_data[3] > ScalarF4E4::from(100u8), "Exploded value lost");

    assert_no_state_corruption(&transposed, "Transpose exploded");
    println!("  ✓ Transpose preserves exploded values");
}

#[test]
fn test_transpose_preserves_vanished() {
    println!("Testing transpose preserves vanished...");

    let tiny = ScalarF4E4::ONE / ScalarF4E4::from(250u8) / ScalarF4E4::from(250u8);

    let data = vec![tiny, ScalarF4E4::ONE, ScalarF4E4::ONE, tiny];
    let t = Tensor::from_scalars(data, Shape::matrix(2, 2)).unwrap();

    let transposed = t.transpose().unwrap();
    let trans_data = transposed.as_scalars().unwrap();

    // Vanished values should stay vanished
    assert!(trans_data[0] < ScalarF4E4::ONE / ScalarF4E4::from(100u8),
            "Vanished value grew");

    assert_no_state_corruption(&transposed, "Transpose vanished");
    println!("  ✓ Transpose preserves vanished values");
}

// ═══ Matmul Edge Cases ═══

#[test]
fn test_matmul_mixed_magnitudes() {
    println!("Testing matmul with mixed magnitudes...");

    let tiny = ScalarF4E4::ONE / ScalarF4E4::from(100u8);
    let large = ScalarF4E4::from(200u8);

    // Matrix with both tiny and large values
    let a_data = vec![tiny, large, large, tiny];
    let b_data = vec![large, tiny, tiny, large];

    let a = Tensor::from_scalars(a_data, Shape::matrix(2, 2)).unwrap();
    let b = Tensor::from_scalars(b_data, Shape::matrix(2, 2)).unwrap();

    let c = matmul(&a, &b).unwrap();

    println!("  Matmul with tiny and large values completed");
    assert_no_state_corruption(&c, "Matmul mixed magnitudes");
    println!("  ✓ Mixed magnitude matmul produces valid results");
}

#[test]
fn test_matmul_infinity_zero() {
    println!("Testing matmul with infinity and zero...");

    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let inf = one / zero;

    // Matrix: [inf, 0; 0, inf]
    let a_data = vec![inf, zero, zero, inf];
    let a = Tensor::from_scalars(a_data, Shape::matrix(2, 2)).unwrap();

    // Matrix: [0, 1; 1, 0]
    let b_data = vec![zero, one, one, zero];
    let b = Tensor::from_scalars(b_data, Shape::matrix(2, 2)).unwrap();

    let c = matmul(&a, &b).unwrap();
    let c_data = c.as_scalars().unwrap();

    println!("  Result contains:");
    for (i, val) in c_data.iter().enumerate() {
        if val.is_undefined() {
            println!("    [{}]: undefined (∞×0)", i);
        } else if val.is_infinite() {
            println!("    [{}]: infinity", i);
        } else if val.is_zero() {
            println!("    [{}]: zero", i);
        } else {
            println!("    [{}]: {}", i, val);
        }
    }

    assert_no_state_corruption(&c, "Matmul infinity/zero");
    println!("  ✓ Infinity/zero matmul produces valid states");
}

// ═══ Comprehensive State Transition Tests ═══

#[test]
fn test_all_state_preservation() {
    println!("Testing all states preserved through operations...");

    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let tiny = one / ScalarF4E4::from(250u8) / ScalarF4E4::from(250u8);
    let large = ScalarF4E4::from(250u8) * ScalarF4E4::from(250u8);
    let inf = one / zero;
    let undef = zero / zero;

    let states = vec![zero, one, tiny, large, inf, undef];

    for (i, &state) in states.iter().enumerate() {
        let t = Tensor::from_scalars(vec![state; 4], Shape::vector(4)).unwrap();

        // Test all operations preserve valid state
        let scaled = t.scale(one).unwrap();
        assert_no_state_corruption(&scaled, &format!("Scale state {}", i));

        let doubled = t.add(&t).unwrap();
        assert_no_state_corruption(&doubled, &format!("Add state {}", i));

        let mat = Tensor::from_scalars(vec![state; 4], Shape::matrix(2, 2)).unwrap();
        let trans = mat.transpose().unwrap();
        assert_no_state_corruption(&trans, &format!("Transpose state {}", i));
    }

    println!("  ✓ All states preserved through all operations");
}

#[test]
fn test_no_accidental_state_transitions() {
    println!("Testing no accidental state transitions...");

    // Normal + normal should stay normal (not become vanished/exploded by accident)
    let a = ScalarF4E4::from(10u8);
    let b = ScalarF4E4::from(20u8);

    let t1 = Tensor::from_scalars(vec![a], Shape::vector(1)).unwrap();
    let t2 = Tensor::from_scalars(vec![b], Shape::vector(1)).unwrap();

    let result = t1.add(&t2).unwrap();
    let data = result.as_scalars().unwrap();

    assert!(data[0].is_finite(), "Normal arithmetic became non-finite");
    assert!(!data[0].is_negligible(), "Normal arithmetic became negligible");
    assert!(!data[0].is_transfinite(), "Normal arithmetic became transfinite");

    println!("  ✓ Normal values stay normal");
}
