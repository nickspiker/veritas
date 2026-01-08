//! Conversion utilities between numeric types

use super::{Circle, Scalar};

/// Convert between different numeric representations
pub trait NumericConversion: Sized {
    fn to_scalar(&self) -> Option<Scalar>;
    fn to_circle(&self) -> Circle;
}

impl NumericConversion for Scalar {
    fn to_scalar(&self) -> Option<Scalar> {
        Some(*self)
    }

    fn to_circle(&self) -> Circle {
        Circle::from(*self)
    }
}

impl NumericConversion for Circle {
    fn to_scalar(&self) -> Option<Scalar> {
        if self.imag().is_zero() {
            Some(self.real())
        } else {
            None
        }
    }

    fn to_circle(&self) -> Circle {
        *self
    }
}
