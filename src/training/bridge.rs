//! Bridge between IEEE-754 (trash) and Spirix (no trash)
//!
//! ## Architecture
//!
//! ```
//! Input (TRASH):      f32/f64 IEEE-754
//!         ↓
//!    [SANITIZE]
//!         ↓
//! Brain (NO TRASH):   ScalarF4E4 neural operations
//!         ↓
//!    [UPCONVERT]
//!         ↓
//! Verify (NO TRASH):  ScalarF6E5 symbolic truth
//!         ↓
//!    [SERIALIZE]
//!         ↓
//! Output (TRASH OK):  f32/f64/JSON/whatever
//! ```
//!
//! Trash is ONLY at boundaries. Brain is 100% Spirix.

use crate::numeric::Scalar;
use spirix::ScalarF4E4;

/// Convert IEEE-754 trash to clean F4E4 for neural training
///
/// This is the ONLY place IEEE-754 enters the system.
/// Once converted, everything is Spirix.
pub fn ieee_to_f4e4(value: f32) -> ScalarF4E4 {
    // Convert thru f64 for better precision during conversion
    ScalarF4E4::from(value as f64)
}

/// Convert F4E4 to F6E5 for verification
///
/// Neural operates in F4E4 (fast, lower precision).
/// Symbolic operates in F6E5 (slower, higher precision).
/// This upconverts neural output to verification precision.
pub fn f4e4_to_f6e5(value: ScalarF4E4) -> Scalar {
    // Convert to f64 as intermediate (both Spirix types support this)
    let f64_val = value.to_f64();
    Scalar::from(f64_val)
}

/// Convert F6E5 back to F4E4 for neural training targets
///
/// Symbolic generates ground truth in F6E5.
/// We need to downconvert to F4E4 for neural to learn.
pub fn f6e5_to_f4e4(value: Scalar) -> ScalarF4E4 {
    let f64_val = value.to_f64();
    ScalarF4E4::from(f64_val)
}

/// Convert F4E4 to IEEE-754 for output (if user wants trash format)
pub fn f4e4_to_ieee(value: ScalarF4E4) -> f32 {
    value.to_f64() as f32
}

/// Bulk convert IEEE array to F4E4
pub fn ieee_array_to_f4e4(values: &[f32]) -> Vec<ScalarF4E4> {
    values.iter().map(|&v| ieee_to_f4e4(v)).collect()
}

/// Bulk convert F4E4 array to F6E5
pub fn f4e4_array_to_f6e5(values: &[ScalarF4E4]) -> Vec<Scalar> {
    values.iter().map(|&v| f4e4_to_f6e5(v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ieee_to_f4e4_roundtrip() {
        let values = vec![1.0, 2.5, -3.7, 0.0, 100.0];

        for &val in &values {
            let f4e4 = ieee_to_f4e4(val);
            let back = f4e4_to_ieee(f4e4);

            // Should be close (accounting for precision loss)
            assert!((val - back).abs() < 0.01,
                   "Roundtrip failed: {} -> {:?} -> {}", val, f4e4, back);
        }
    }

    #[test]
    fn test_f4e4_to_f6e5_conversion() {
        let f4e4_values = vec![
            ScalarF4E4::from(1.0),
            ScalarF4E4::from(2.5),
            ScalarF4E4::from(-3.7),
        ];

        for f4e4 in f4e4_values {
            let f6e5 = f4e4_to_f6e5(f4e4);
            let back = f6e5_to_f4e4(f6e5);

            // Should be exact (upconvert then downconvert)
            let orig_f64 = f4e4.to_f64();
            let back_f64 = back.to_f64();
            assert!((orig_f64 - back_f64).abs() < 0.01);
        }
    }
}
