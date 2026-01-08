//! Numeric foundation for Veritas
//!
//! All arithmetic uses Spirix two's complement floats.
//! NO IEEE-754 ANYWHERE.
//!
//! Key types:
//! - `Scalar`: Real numbers (ScalarF6E5 from Spirix)
//! - `Circle`: Complex numbers (CircleF6E5 from Spirix)
//!
//! Why Spirix?
//! - Two's complement thruout (no sign bit branches)
//! - Continuous math (no discontinuity at zero)
//! - Traceable errors (no generic NaN)
//! - Vanished/exploded values (not silent zero/infinity)
//! - Preserves mathematical identities (a×b=0 iff a|b=0)

pub mod circle;
pub mod conversion;
pub mod scalar;

pub use circle::{Circle, Complex};
pub use scalar::Scalar;

use crate::error::{Result, VeritasError};

/// Standard precision for most computations
/// F6E5 = 64-bit fraction, 32-bit exponent
/// - ~19 decimal digits precision
/// - Range: ~10^±9 billion
pub type DefaultScalar = spirix::ScalarF6E5;
pub type DefaultCircle = spirix::CircleF6E5;

/// Check if a Spirix scalar is in normal state
#[inline]
pub fn is_normal(s: &DefaultScalar) -> bool {
    s.is_normal()
}

/// Check if a Spirix scalar is vanished (underflow but not zero)
#[inline]
pub fn is_vanished(s: &DefaultScalar) -> bool {
    s.vanished()
}

/// Check if a Spirix scalar is exploded (overflow but not infinite)
#[inline]
pub fn is_exploded(s: &DefaultScalar) -> bool {
    s.exploded()
}

/// Check if a Spirix scalar is undefined
#[inline]
pub fn is_undefined(s: &DefaultScalar) -> bool {
    s.is_undefined()
}

/// Convert a Spirix undefined state to error
pub fn undefined_to_error(s: &DefaultScalar) -> VeritasError {
    if s.is_undefined() {
        // Spirix tracks specific undefined causes
        // We'll extract the cause and convert to error
        VeritasError::UndefinedOperation(format!("{:?}", s))
    } else {
        panic!("Called undefined_to_error on non-undefined value");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_value() {
        let x = DefaultScalar::from(42);
        assert!(is_normal(&x));
        assert!(!is_vanished(&x));
        assert!(!is_exploded(&x));
        assert!(!is_undefined(&x));
    }

    #[test]
    fn test_vanished_value() {
        let tiny = DefaultScalar::MIN_POS.square();
        assert!(!tiny.is_zero());
        assert!(is_vanished(&tiny));
        assert!(!is_exploded(&tiny));
    }

    #[test]
    fn test_exploded_value() {
        let huge = DefaultScalar::MAX * DefaultScalar::from(2);
        assert!(is_exploded(&huge));
        assert!(!is_vanished(&huge));
    }

    #[test]
    fn test_undefined_division_by_zero() {
        let zero = DefaultScalar::ZERO;
        let one = DefaultScalar::ONE;
        let undefined = one / zero;
        assert!(is_undefined(&undefined));
    }

    #[test]
    fn test_mathematical_identity_preserved() {
        // IEEE-754 VIOLATES: a×b=0 iff a|b=0
        // Spirix PRESERVES this identity

        let tiny = DefaultScalar::MIN_POS;
        let product = tiny.square();

        // Product is vanished, NOT zero
        assert!(!product.is_zero());
        assert!(is_vanished(&product));

        // Neither factor is zero
        assert!(!tiny.is_zero());

        // Identity preserved: a×b ≠ 0 because neither a nor b is zero
    }
}
