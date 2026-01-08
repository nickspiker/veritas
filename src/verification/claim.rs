//! Claims about computations

use crate::numeric::{Circle, Scalar};
use crate::symbolic::Expr;

/// A claim about a computation
#[derive(Debug, Clone)]
pub struct Claim {
    /// Natural language statement
    pub statement: String,

    /// Symbolic representation (if parseable)
    pub symbolic: Option<Expr>,

    /// Expected result (if known)
    pub expected: Option<ClaimValue>,
}

/// Value that a claim evaluates to
#[derive(Debug, Clone, PartialEq)]
pub enum ClaimValue {
    Scalar(Scalar),
    Circle(Circle),
    Boolean(bool),
}

impl Claim {
    pub fn new(statement: impl Into<String>) -> Self {
        Claim {
            statement: statement.into(),
            symbolic: None,
            expected: None,
        }
    }

    pub fn with_symbolic(mut self, expr: Expr) -> Self {
        self.symbolic = Some(expr);
        self
    }

    pub fn with_expected(mut self, value: ClaimValue) -> Self {
        self.expected = Some(value);
        self
    }
}
