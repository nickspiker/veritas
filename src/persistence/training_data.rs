//! Training example generation and storage

use crate::numeric::Scalar;
use crate::symbolic::Expr;

/// A verified training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// The problem
    pub problem: Expr,

    /// The verified solution
    pub solution: Scalar,

    /// How it was verified
    pub verification: String,
}

impl TrainingExample {
    pub fn new(problem: Expr, solution: Scalar, verification: impl Into<String>) -> Self {
        TrainingExample {
            problem,
            solution,
            verification: verification.into(),
        }
    }
}
