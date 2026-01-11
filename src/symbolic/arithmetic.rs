//! Basic Arithmetic Engine - Bolt-on for verified computation
//!
//! This is NOT a replacement for basecalc - it's a simplified
//! arithmetic evaluator for training neural networks against
//! verified ground truth.
//!
//! For full-featured calculation, use basecalc directly.

use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;

/// Arithmetic operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArithOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Simple arithmetic problem
#[derive(Debug, Clone)]
pub struct ArithProblem {
    pub left: ScalarF4E4,
    pub right: ScalarF4E4,
    pub op: ArithOp,
}

/// Verified arithmetic result
#[derive(Debug, Clone)]
pub struct ArithResult {
    pub answer: ScalarF4E4,
    pub problem: ArithProblem,
    pub expr: String,
}

impl ArithProblem {
    pub fn new(left: ScalarF4E4, right: ScalarF4E4, op: ArithOp) -> Self {
        Self { left, right, op }
    }

    /// Compute the verified answer using Spirix arithmetic
    pub fn solve(&self) -> Result<ArithResult> {
        let answer = match self.op {
            ArithOp::Add => self.left + self.right,
            ArithOp::Sub => self.left - self.right,
            ArithOp::Mul => self.left * self.right,
            ArithOp::Div => {
                if self.right == ScalarF4E4::ZERO {
                    return Err(VeritasError::DivisionByZero);
                }
                self.left / self.right
            }
        };

        let op_str = match self.op {
            ArithOp::Add => "+",
            ArithOp::Sub => "-",
            ArithOp::Mul => "×",
            ArithOp::Div => "÷",
        };

        Ok(ArithResult {
            answer,
            problem: self.clone(),
            expr: format!("{} {} {}", self.left, op_str, self.right),
        })
    }

    /// Parse simple arithmetic expression like "2 + 3"
    pub fn parse(expr: &str) -> Result<Self> {
        let expr = expr.trim();

        // Find operator
        let (op, op_char) = if let Some(idx) = expr.find('+') {
            (ArithOp::Add, idx)
        } else if let Some(idx) = expr.rfind('-') {
            // rfind to handle negative numbers
            if idx == 0 {
                return Err(VeritasError::InvalidInput(
                    "Cannot parse: starts with negative".to_string(),
                ));
            }
            (ArithOp::Sub, idx)
        } else if let Some(idx) = expr.find('*') {
            (ArithOp::Mul, idx)
        } else if let Some(idx) = expr.find('/') {
            (ArithOp::Div, idx)
        } else {
            return Err(VeritasError::InvalidInput(
                "No operator found".to_string(),
            ));
        };

        let left_str = expr[..op_char].trim();
        let right_str = expr[op_char + 1..].trim();

        // Parse using Spirix's From<u8> and division for decimals
        // For now, support simple integers
        let left = parse_scalar(left_str)?;
        let right = parse_scalar(right_str)?;

        Ok(Self::new(left, right, op))
    }
}

/// Parse a scalar from string (simple integer parsing)
fn parse_scalar(s: &str) -> Result<ScalarF4E4> {
    let s = s.trim();

    // Handle negative numbers
    let (negative, s) = if s.starts_with('-') {
        (true, &s[1..])
    } else {
        (false, s)
    };

    // Parse as integer
    let value: u8 = s.parse().map_err(|_| {
        VeritasError::InvalidInput(format!("Cannot parse '{}' as number", s))
    })?;

    let scalar = ScalarF4E4::from(value);

    Ok(if negative {
        ScalarF4E4::ZERO - scalar
    } else {
        scalar
    })
}

/// Arithmetic problem generator for training
pub struct ArithGenerator {
    max_value: u8,
}

impl ArithGenerator {
    pub fn new(max_value: u8) -> Self {
        Self { max_value }
    }

    /// Generate a random arithmetic problem
    pub fn generate(&self, op: ArithOp) -> ArithProblem {
        let left_val = (rand::random::<u32>() % self.max_value as u32) as u8;
        let right_val = (rand::random::<u32>() % self.max_value as u32) as u8;

        let left = ScalarF4E4::from(left_val);
        let right = ScalarF4E4::from(right_val);

        // For division, ensure right is non-zero
        let right = if op == ArithOp::Div && right == ScalarF4E4::ZERO {
            ScalarF4E4::ONE
        } else {
            right
        };

        ArithProblem::new(left, right, op)
    }

    /// Generate a batch of problems
    pub fn generate_batch(&self, count: usize) -> Vec<ArithProblem> {
        let ops = [ArithOp::Add, ArithOp::Sub, ArithOp::Mul, ArithOp::Div];
        let mut problems = Vec::new();

        for i in 0..count {
            let op = ops[i % ops.len()];
            problems.push(self.generate(op));
        }

        problems
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addition() {
        let prob = ArithProblem::new(
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
            ArithOp::Add,
        );
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, ScalarF4E4::from(5u8));
    }

    #[test]
    fn test_subtraction() {
        let prob = ArithProblem::new(
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(3u8),
            ArithOp::Sub,
        );
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, ScalarF4E4::from(2u8));
    }

    #[test]
    fn test_multiplication() {
        let prob = ArithProblem::new(
            ScalarF4E4::from(4u8),
            ScalarF4E4::from(3u8),
            ArithOp::Mul,
        );
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, ScalarF4E4::from(12u8));
    }

    #[test]
    fn test_division() {
        let prob = ArithProblem::new(
            ScalarF4E4::from(12u8),
            ScalarF4E4::from(3u8),
            ArithOp::Div,
        );
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, ScalarF4E4::from(4u8));
    }

    #[test]
    fn test_division_by_zero() {
        let prob = ArithProblem::new(
            ScalarF4E4::from(5u8),
            ScalarF4E4::ZERO,
            ArithOp::Div,
        );
        assert!(prob.solve().is_err());
    }

    #[test]
    fn test_parse() {
        let prob = ArithProblem::parse("2 + 3").unwrap();
        assert_eq!(prob.left, ScalarF4E4::from(2u8));
        assert_eq!(prob.right, ScalarF4E4::from(3u8));
        assert_eq!(prob.op, ArithOp::Add);

        let result = prob.solve().unwrap();
        assert_eq!(result.answer, ScalarF4E4::from(5u8));
    }

    #[test]
    fn test_generator() {
        let gen = ArithGenerator::new(10);
        let problems = gen.generate_batch(8);
        assert_eq!(problems.len(), 8);

        // Verify all problems can be solved
        for prob in problems {
            let result = prob.solve();
            assert!(result.is_ok());
        }
    }
}
