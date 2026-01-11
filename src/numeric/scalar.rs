//! Scalar type wrapper around Spirix
//!
//! Provides a clean API for real number arithmetic

use crate::error::{Result, VeritasError};
use spirix::ScalarF6E5;

/// Real number using Spirix two's complement floats
///
/// This is a thin wrapper around `ScalarF6E5` that provides:
/// - Error handling for undefined operations
/// - Conversion utilities
/// - Integration with the rest of Veritas
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scalar(pub ScalarF6E5);

impl Scalar {
    pub const ZERO: Self = Scalar(ScalarF6E5::ZERO);
    pub const ONE: Self = Scalar(ScalarF6E5::ONE);
    pub const TWO: Self = Scalar(ScalarF6E5::TWO);
    pub const PI: Self = Scalar(ScalarF6E5::PI);
    pub const E: Self = Scalar(ScalarF6E5::E);

    /// Create from inner Spirix scalar
    #[inline]
    pub fn new(inner: ScalarF6E5) -> Self {
        Scalar(inner)
    }

    /// Get inner Spirix scalar
    #[inline]
    pub fn inner(&self) -> ScalarF6E5 {
        self.0
    }

    /// Check if value is normal (finite, non-zero)
    #[inline]
    pub fn is_normal(&self) -> bool {
        self.0.is_normal()
    }

    /// Check if value is zero
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    /// Check if value is vanished (underflow)
    #[inline]
    pub fn is_vanished(&self) -> bool {
        self.0.vanished()
    }

    /// Check if value is exploded (overflow)
    #[inline]
    pub fn is_exploded(&self) -> bool {
        self.0.exploded()
    }

    /// Check if value is undefined
    #[inline]
    pub fn is_undefined(&self) -> bool {
        self.0.is_undefined()
    }

    /// Check result and return error if undefined
    pub fn check(&self) -> Result<Self> {
        if self.is_undefined() {
            Err(VeritasError::UndefinedOperation(format!("{:?}", self.0)))
        } else {
            Ok(*self)
        }
    }

    // Arithmetic operations that return Result

    /// Checked addition
    pub fn checked_add(&self, rhs: Self) -> Result<Self> {
        let result = Scalar(self.0 + rhs.0);
        result.check()
    }

    /// Checked subtraction
    pub fn checked_sub(&self, rhs: Self) -> Result<Self> {
        let result = Scalar(self.0 - rhs.0);
        result.check()
    }

    /// Checked multiplication
    pub fn checked_mul(&self, rhs: Self) -> Result<Self> {
        let result = Scalar(self.0 * rhs.0);
        result.check()
    }

    /// Checked division
    pub fn checked_div(&self, rhs: Self) -> Result<Self> {
        if rhs.is_zero() {
            return Err(VeritasError::DivisionByZero);
        }
        let result = Scalar(self.0 / rhs.0);
        result.check()
    }

    // Mathematical functions

    /// Square root
    pub fn sqrt(&self) -> Result<Self> {
        let result = Scalar(self.0.sqrt());
        result.check()
    }

    /// Natural logarithm
    pub fn ln(&self) -> Result<Self> {
        let result = Scalar(self.0.ln());
        result.check()
    }

    /// Exponential (e^x)
    pub fn exp(&self) -> Result<Self> {
        let result = Scalar(self.0.exp());
        result.check()
    }

    /// Power
    pub fn pow(&self, exp: Self) -> Result<Self> {
        let result = Scalar(self.0.pow(exp.0));
        result.check()
    }

    /// Sine
    pub fn sin(&self) -> Result<Self> {
        let result = Scalar(self.0.sin());
        result.check()
    }

    /// Cosine
    pub fn cos(&self) -> Result<Self> {
        let result = Scalar(self.0.cos());
        result.check()
    }

    /// Absolute value (magnitude)
    pub fn abs(&self) -> Self {
        Scalar(self.0.magnitude())
    }
}

// Implement arithmetic operators (unchecked, for convenience)
impl std::ops::Add for Scalar {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Scalar(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Scalar {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Scalar(self.0 - rhs.0)
    }
}

impl std::ops::Mul for Scalar {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Scalar(self.0 * rhs.0)
    }
}

impl std::ops::Div for Scalar {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Scalar(self.0 / rhs.0)
    }
}

impl std::ops::Neg for Scalar {
    type Output = Self;
    fn neg(self) -> Self {
        Scalar(-self.0)
    }
}

// Conversions
impl From<i32> for Scalar {
    fn from(i: i32) -> Self {
        Scalar(ScalarF6E5::from(i))
    }
}

impl From<f64> for Scalar {
    fn from(f: f64) -> Self {
        Scalar(ScalarF6E5::from(f))
    }
}

impl From<ScalarF6E5> for Scalar {
    fn from(s: ScalarF6E5) -> Self {
        Scalar(s)
    }
}

impl std::fmt::Display for Scalar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_undefined() {
            write!(f, "[undefined: {:?}]", self.0)
        } else if self.is_vanished() {
            write!(f, "[vanished]")
        } else if self.is_exploded() {
            write!(f, "[exploded]")
        } else if self.is_zero() {
            write!(f, "0")
        } else {
            write!(f, "{}", self.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checked_arithmetic() {
        let a = Scalar::from(10);
        let b = Scalar::from(3);

        assert!(a.checked_add(b).is_ok());
        assert!(a.checked_sub(b).is_ok());
        assert!(a.checked_mul(b).is_ok());
        assert!(a.checked_div(b).is_ok());
    }

    #[test]
    fn test_division_by_zero() {
        let a = Scalar::from(10);
        let zero = Scalar::ZERO;

        assert_eq!(
            a.checked_div(zero).unwrap_err(),
            VeritasError::DivisionByZero
        );
    }

    #[test]
    fn test_mathematical_functions() {
        let x = Scalar::from(2);

        assert!(x.sqrt().is_ok());
        assert!(x.ln().is_ok());
        assert!(x.exp().is_ok());
        assert!(x.sin().is_ok());
        assert!(x.cos().is_ok());
    }

    #[test]
    fn test_vanished_detection() {
        let tiny = Scalar::new(ScalarF6E5::MIN_POS);
        let product = tiny * tiny;

        assert!(!product.is_zero());
        assert!(product.is_vanished());
    }
}
