//! Complex number support using Spirix Circle types

use super::Scalar;
use crate::error::{Result, VeritasError};
use spirix::{CircleF6E5, ScalarF6E5};

/// Complex number using Spirix two's complement floats
///
/// Wraps `CircleF6E5` which stores real and imaginary components
/// with a shared exponent for efficiency.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle(pub CircleF6E5);

/// Alias for Circle (more familiar name)
pub type Complex = Circle;

impl Circle {
    pub const ZERO: Self = Circle(CircleF6E5::ZERO);
    pub const ONE: Self = Circle(CircleF6E5::ONE);
    pub const I: Self = Circle(CircleF6E5::POS_I);

    /// Create from inner Spirix circle
    #[inline]
    pub fn new(inner: CircleF6E5) -> Self {
        Circle(inner)
    }

    /// Create from real and imaginary parts
    pub fn from_parts(real: Scalar, imag: Scalar) -> Self {
        Circle(CircleF6E5::from((real.0, imag.0)))
    }

    /// Get inner Spirix circle
    #[inline]
    pub fn inner(&self) -> CircleF6E5 {
        self.0
    }

    /// Real part
    pub fn real(&self) -> Scalar {
        Scalar(self.0.r())
    }

    /// Imaginary part
    pub fn imag(&self) -> Scalar {
        Scalar(self.0.i())
    }

    /// Complex conjugate
    pub fn conjugate(&self) -> Self {
        Circle(self.0.conjugate())
    }

    /// Magnitude (distance from origin)
    pub fn magnitude(&self) -> Scalar {
        Scalar(self.0.magnitude())
    }

    /// Squared magnitude (faster than magnitude)
    pub fn magnitude_squared(&self) -> Scalar {
        Scalar(self.0.magnitude_squared())
    }

    /// Check if value is normal
    pub fn is_normal(&self) -> bool {
        self.0.is_normal()
    }

    /// Check if value is undefined
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

    // Arithmetic operations

    /// Checked addition
    pub fn checked_add(&self, rhs: Self) -> Result<Self> {
        let result = Circle(self.0 + rhs.0);
        result.check()
    }

    /// Checked subtraction
    pub fn checked_sub(&self, rhs: Self) -> Result<Self> {
        let result = Circle(self.0 - rhs.0);
        result.check()
    }

    /// Checked multiplication
    pub fn checked_mul(&self, rhs: Self) -> Result<Self> {
        let result = Circle(self.0 * rhs.0);
        result.check()
    }

    /// Checked division
    pub fn checked_div(&self, rhs: Self) -> Result<Self> {
        if rhs.0.is_zero() {
            return Err(VeritasError::DivisionByZero);
        }
        let result = Circle(self.0 / rhs.0);
        result.check()
    }

    /// Square root
    pub fn sqrt(&self) -> Result<Self> {
        let result = Circle(self.0.sqrt());
        result.check()
    }

    /// Exponential
    pub fn exp(&self) -> Result<Self> {
        let result = Circle(self.0.exp());
        result.check()
    }
}

// Arithmetic operators
impl std::ops::Add for Circle {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Circle(self.0 + rhs.0)
    }
}

impl std::ops::Sub for Circle {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Circle(self.0 - rhs.0)
    }
}

impl std::ops::Mul for Circle {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Circle(self.0 * rhs.0)
    }
}

impl std::ops::Div for Circle {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Circle(self.0 / rhs.0)
    }
}

impl std::ops::Neg for Circle {
    type Output = Self;
    fn neg(self) -> Self {
        Circle(-self.0)
    }
}

// Conversions
impl From<Scalar> for Circle {
    fn from(s: Scalar) -> Self {
        Circle(CircleF6E5::from(s.0))
    }
}

impl From<(Scalar, Scalar)> for Circle {
    fn from((r, i): (Scalar, Scalar)) -> Self {
        Circle::from_parts(r, i)
    }
}

impl From<CircleF6E5> for Circle {
    fn from(c: CircleF6E5) -> Self {
        Circle(c)
    }
}

impl std::fmt::Display for Circle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let r = self.real();
        let i = self.imag();

        if i.is_zero() {
            write!(f, "{}", r)
        } else if r.is_zero() {
            write!(f, "{}i", i)
        } else if i.inner() < spirix::ScalarF6E5::ZERO {
            write!(f, "{} - {}i", r, -i)
        } else {
            write!(f, "{} + {}i", r, i)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Circle::from_parts(Scalar::from(3), Scalar::from(4));
        let b = Circle::from_parts(Scalar::from(1), Scalar::from(-2));

        let sum = a + b;
        assert_eq!(sum.real(), Scalar::from(4));
        assert_eq!(sum.imag(), Scalar::from(2));

        let product = a * b;
        // (3 + 4i)(1 - 2i) = 3 - 6i + 4i - 8i² = 3 - 2i + 8 = 11 - 2i
        assert_eq!(product.real(), Scalar::from(11));
        assert_eq!(product.imag(), Scalar::from(-2));
    }

    #[test]
    fn test_magnitude() {
        let z = Circle::from_parts(Scalar::from(3), Scalar::from(4));
        let mag = z.magnitude();

        // |3 + 4i| = 5
        assert_eq!(mag, Scalar::from(5));
    }

    #[test]
    fn test_conjugate() {
        let z = Circle::from_parts(Scalar::from(3), Scalar::from(4));
        let conj = z.conjugate();

        assert_eq!(conj.real(), Scalar::from(3));
        assert_eq!(conj.imag(), Scalar::from(-4));
    }
}
