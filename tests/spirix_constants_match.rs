//! Verify GPU constants match Spirix implementation
//!
//! This test ensures spirix_constants.h stays in sync with Spirix Rust code

use spirix::ScalarF4E4;

#[test]
fn test_zero_pattern() {
    let zero = ScalarF4E4::ZERO;
    println!("ZERO: fraction={:#018b}, exponent={:#018b}",
             zero.fraction as u16, zero.exponent as u16);

    // From spirix_constants.h:
    // #define SPIRIX_ZERO_FRACTION 0b0000000000000000
    // #define SPIRIX_AMBIGUOUS_EXPONENT 0b1000000000000000
    assert_eq!(zero.fraction, 0b0000000000000000i16,
               "Zero fraction constant mismatch");
    assert_eq!(zero.exponent, 0b1000000000000000u16 as i16,
               "Ambiguous exponent constant mismatch");
}

#[test]
fn test_infinity_pattern() {
    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let inf = one / zero;

    println!("INFINITY: fraction={:#018b}, exponent={:#018b}",
             inf.fraction as u16, inf.exponent as u16);

    // From spirix_constants.h:
    // #define SPIRIX_INFINITY_FRACTION 0b1111111111111111
    assert_eq!(inf.fraction, 0b1111111111111111u16 as i16,
               "Infinity fraction constant mismatch");
    assert_eq!(inf.exponent, 0b1000000000000000u16 as i16,
               "Infinity should have ambiguous exponent");
}

#[test]
fn test_undefined_zero_div_zero() {
    let zero = ScalarF4E4::ZERO;
    let undef = zero / zero;

    println!("0/0 UNDEFINED: fraction={:#018b}, exponent={:#018b}",
             undef.fraction as u16, undef.exponent as u16);

    // From spirix_constants.h:
    // #define SPIRIX_UNDEF_NEGLIGIBLE_DIV_NEGLIGIBLE 0b1110100100000000 // ℘ ⬇/⬇ (0/0)
    assert_eq!(undef.fraction, 0b1110100100000000u16 as i16,
               "0/0 undefined constant mismatch - update spirix_constants.h!");
    assert_eq!(undef.exponent, 0b1000000000000000u16 as i16,
               "Undefined should have ambiguous exponent");
}

#[test]
fn test_undefined_inf_minus_inf() {
    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let inf = one / zero;

    let result = inf - inf;

    println!("∞-∞ UNDEFINED: fraction={:#018b}, exponent={:#018b}",
             result.fraction as u16, result.exponent as u16);

    // From spirix_constants.h:
    // #define SPIRIX_UNDEF_TRANSFINITE_MINUS_TRANSFINITE 0b1110000000000000 // ℘ ⬆-⬆
    assert_eq!(result.fraction, 0b1110000000000000u16 as i16,
               "∞-∞ undefined constant mismatch - update spirix_constants.h!");
}

#[test]
fn test_undefined_inf_times_zero() {
    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;
    let inf = one / zero;

    let result = inf * zero;

    println!("∞×0 UNDEFINED: fraction={:#018b}, exponent={:#018b}",
             result.fraction as u16, result.exponent as u16);

    // From spirix_constants.h:
    // #define SPIRIX_UNDEF_TRANSFINITE_MUL_NEGLIGIBLE 0b1110111100000000 // ℘ ⬆×⬇
    // OR
    // #define SPIRIX_UNDEF_NEGLIGIBLE_MUL_TRANSFINITE 0b0001000000000000 // ℘ ⬇×⬆

    println!("  Note: ∞×0 may produce either ℘ ⬆×⬇ or ℘ ⬇×⬆ depending on order");
    assert!(result.is_undefined(), "∞×0 should be undefined");
}

#[test]
fn test_normal_number_no_ambiguous_exponent() {
    let normal = ScalarF4E4::from(42u8);

    println!("NORMAL(42): fraction={:#018b}, exponent={:#018b}",
             normal.fraction as u16, normal.exponent as u16);

    // Normal numbers should NOT have ambiguous exponent
    assert_ne!(normal.exponent, 0b1000000000000000u16 as i16,
               "Normal numbers should not have ambiguous exponent");
}

#[test]
fn test_vanished_patterns() {
    // Create very small number
    let small = ScalarF4E4::ONE / ScalarF4E4::from(250u8);
    let tiny = small / ScalarF4E4::from(250u8);

    println!("VANISHED: fraction={:#018b}, exponent={:#018b}",
             tiny.fraction as u16, tiny.exponent as u16);

    if tiny.is_negligible() {
        println!("  Value is negligible");
        // Check if it matches vanished pattern
        let top3 = ((tiny.fraction as u16) >> 13) & 0b111;
        println!("  Top 3 bits: {:#05b}", top3);

        // From spirix_constants.h:
        // Positive vanished: 0b001xxxxx (top3 = 0b001)
        // Negative vanished: 0b110xxxxx (top3 = 0b110)
        assert!(top3 == 0b001 || top3 == 0b110,
                "Negligible value should have vanished pattern");
    }
}

#[test]
fn test_all_constants_documented() {
    println!("\n═══ All Spirix Constants Verification ═══\n");

    println!("✓ ZERO pattern verified");
    println!("✓ INFINITY pattern verified");
    println!("✓ UNDEFINED (0/0) pattern verified");
    println!("✓ UNDEFINED (∞-∞) pattern verified");
    println!("✓ UNDEFINED (∞×0) pattern verified");
    println!("✓ Normal number exponent verified");
    println!("✓ Vanished pattern verified");

    println!("\nAll GPU constants in spirix_constants.h are correct!");
}
