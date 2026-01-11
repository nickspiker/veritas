//! Symbolic mathematics engine
//!
//! This module provides symbolic manipulation of mathematical expressions.
//! All numeric evaluation uses Spirix (no IEEE-754).
//!
//! Key types:
//! - `Expr`: Symbolic expression tree
//! - `Context`: Variable bindings
//! - `Simplify`: Expression simplification
//!
//! Design principles:
//! - Every expression can be simplified
//! - Every expression can be evaluated (given context)
//! - Simplification preserves mathematical equivalence
//! - Evaluation returns Spirix types (traceable errors)

pub mod context;
pub mod eval;
pub mod expr;
pub mod simplify;
pub mod arithmetic;
pub mod bitwise;

pub use context::Context;
pub use eval::Evaluate;
pub use expr::Expr;
pub use simplify::Simplify;
pub use arithmetic::{ArithOp, ArithProblem, ArithResult, ArithGenerator};
pub use bitwise::{BitwiseOp, BitwiseProblem, BitwiseResult, BitwiseGenerator};

use crate::error::{Result, VeritasError};
use crate::numeric::{Circle, Scalar};

/// Common mathematical constants as expressions
pub mod constants {
    use super::Expr;

    pub fn pi() -> Expr {
        Expr::Constant("π".to_string())
    }

    pub fn e() -> Expr {
        Expr::Constant("e".to_string())
    }

    pub fn i() -> Expr {
        Expr::Constant("i".to_string())
    }
}
