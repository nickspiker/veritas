//! Bitwise Operations Bolt-on
//!
//! Verified bitwise operations for training neural networks.
//! Operations work on integer representations (no floating point).
//!
//! Supports:
//! - AND, OR, XOR, NOT
//! - Shift left/right
//! - Rotate left/right
//! - Count bits (popcount)

use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;

/// Bitwise operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitwiseOp {
    And,      // a & b
    Or,       // a | b
    Xor,      // a ^ b
    Not,      // !a (unary)
    Shl,      // a << b (shift left)
    Shr,      // a >> b (shift right)
    Rotl,     // rotate left
    Rotr,     // rotate right
    Popcount, // count set bits (unary)
}

/// Bitwise problem
#[derive(Debug, Clone)]
pub struct BitwiseProblem {
    pub left: u8,
    pub right: Option<u8>, // None for unary ops
    pub op: BitwiseOp,
}

/// Verified bitwise result
#[derive(Debug, Clone)]
pub struct BitwiseResult {
    pub answer: u8,
    pub problem: BitwiseProblem,
    pub expr: String,
    pub dozenal: String, // Display in base 12
}

impl BitwiseProblem {
    pub fn new(left: u8, right: Option<u8>, op: BitwiseOp) -> Self {
        Self { left, right, op }
    }

    /// Compute verified answer using integer bitwise operations
    pub fn solve(&self) -> Result<BitwiseResult> {
        let answer = match self.op {
            BitwiseOp::And => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "AND requires two operands".to_string()
                ))?;
                self.left & right
            }
            BitwiseOp::Or => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "OR requires two operands".to_string()
                ))?;
                self.left | right
            }
            BitwiseOp::Xor => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "XOR requires two operands".to_string()
                ))?;
                self.left ^ right
            }
            BitwiseOp::Not => !self.left,
            BitwiseOp::Shl => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "SHL requires two operands".to_string()
                ))?;
                self.left.wrapping_shl(right as u32)
            }
            BitwiseOp::Shr => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "SHR requires two operands".to_string()
                ))?;
                self.left.wrapping_shr(right as u32)
            }
            BitwiseOp::Rotl => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "ROTL requires two operands".to_string()
                ))?;
                self.left.rotate_left(right as u32)
            }
            BitwiseOp::Rotr => {
                let right = self.right.ok_or(VeritasError::InvalidInput(
                    "ROTR requires two operands".to_string()
                ))?;
                self.left.rotate_right(right as u32)
            }
            BitwiseOp::Popcount => self.left.count_ones() as u8,
        };

        let (op_str, expr) = match self.op {
            BitwiseOp::And => ("&", format!("{} & {}", self.left, self.right.unwrap())),
            BitwiseOp::Or => ("|", format!("{} | {}", self.left, self.right.unwrap())),
            BitwiseOp::Xor => ("^", format!("{} ^ {}", self.left, self.right.unwrap())),
            BitwiseOp::Not => ("!", format!("!{}", self.left)),
            BitwiseOp::Shl => ("<<", format!("{} << {}", self.left, self.right.unwrap())),
            BitwiseOp::Shr => (">>", format!("{} >> {}", self.left, self.right.unwrap())),
            BitwiseOp::Rotl => ("rotl", format!("rotl({}, {})", self.left, self.right.unwrap())),
            BitwiseOp::Rotr => ("rotr", format!("rotr({}, {})", self.left, self.right.unwrap())),
            BitwiseOp::Popcount => ("popcount", format!("popcount({})", self.left)),
        };

        // Format in dozenal (base 12)
        let dozenal = match self.op {
            BitwiseOp::Not | BitwiseOp::Popcount => {
                format!("{} = {} (base C)", expr, to_dozenal(answer))
            }
            _ => {
                format!("{} = {} (base C)", expr, to_dozenal(answer))
            }
        };

        Ok(BitwiseResult {
            answer,
            problem: self.clone(),
            expr,
            dozenal,
        })
    }

    /// Convert to Spirix scalars for neural training
    pub fn to_scalars(&self) -> (ScalarF4E4, Option<ScalarF4E4>, ScalarF4E4) {
        let left_scalar = ScalarF4E4::from(self.left);
        let right_scalar = self.right.map(|r| ScalarF4E4::from(r));

        // Encode operation as scalar
        let half = ScalarF4E4::ONE / ScalarF4E4::from(2u8);
        let eighth = ScalarF4E4::ONE / ScalarF4E4::from(8u8);
        let quarter = ScalarF4E4::ONE / ScalarF4E4::from(4u8);

        let op_scalar = match self.op {
            BitwiseOp::And => ScalarF4E4::ZERO,
            BitwiseOp::Or => eighth,
            BitwiseOp::Xor => quarter,
            BitwiseOp::Not => eighth + quarter, // 0.375
            BitwiseOp::Shl => half, // 0.5
            BitwiseOp::Shr => half + eighth, // 0.625
            BitwiseOp::Rotl => half + quarter, // 0.75
            BitwiseOp::Rotr => half + eighth + quarter, // 0.875
            BitwiseOp::Popcount => ScalarF4E4::ONE,
        };

        (left_scalar, right_scalar, op_scalar)
    }
}

/// Convert u8 to dozenal (base 12) string
fn to_dozenal(n: u8) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let mut result = String::new();
    let mut value = n;

    while value > 0 {
        let digit = value % 12;
        let ch = match digit {
            0..=9 => (b'0' + digit) as char,
            10 => 'A', // Ten in dozenal
            11 => 'B', // Eleven in dozenal
            _ => unreachable!(),
        };
        result.insert(0, ch);
        value /= 12;
    }

    result
}

/// Bitwise problem generator for training
pub struct BitwiseGenerator {
    max_value: u8,
    max_shift: u8,
}

impl BitwiseGenerator {
    pub fn new(max_value: u8) -> Self {
        Self {
            max_value,
            max_shift: 8, // Max shift for u8
        }
    }

    /// Generate random bitwise problem
    pub fn generate(&self, op: BitwiseOp) -> BitwiseProblem {
        let left = (rand::random::<u32>() % self.max_value as u32) as u8;

        let right = match op {
            BitwiseOp::Not | BitwiseOp::Popcount => None,
            BitwiseOp::Shl | BitwiseOp::Shr | BitwiseOp::Rotl | BitwiseOp::Rotr => {
                Some((rand::random::<u32>() % self.max_shift as u32) as u8)
            }
            _ => Some((rand::random::<u32>() % self.max_value as u32) as u8),
        };

        BitwiseProblem::new(left, right, op)
    }

    /// Generate batch of problems
    pub fn generate_batch(&self, count: usize) -> Vec<BitwiseProblem> {
        let ops = [
            BitwiseOp::And,
            BitwiseOp::Or,
            BitwiseOp::Xor,
            BitwiseOp::Shl,
            BitwiseOp::Shr,
        ];

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
    fn test_and() {
        let prob = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::And);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b1000);
    }

    #[test]
    fn test_or() {
        let prob = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::Or);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b1110);
    }

    #[test]
    fn test_xor() {
        let prob = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::Xor);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b0110);
    }

    #[test]
    fn test_not() {
        let prob = BitwiseProblem::new(0b00001111, None, BitwiseOp::Not);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b11110000);
    }

    #[test]
    fn test_shift_left() {
        let prob = BitwiseProblem::new(0b00000011, Some(2), BitwiseOp::Shl);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b00001100);
    }

    #[test]
    fn test_shift_right() {
        let prob = BitwiseProblem::new(0b00001100, Some(2), BitwiseOp::Shr);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 0b00000011);
    }

    #[test]
    fn test_popcount() {
        let prob = BitwiseProblem::new(0b10101010, None, BitwiseOp::Popcount);
        let result = prob.solve().unwrap();
        assert_eq!(result.answer, 4);
    }

    #[test]
    fn test_dozenal_conversion() {
        assert_eq!(to_dozenal(0), "0");
        assert_eq!(to_dozenal(11), "B"); // Eleven
        assert_eq!(to_dozenal(12), "10"); // One dozen
        assert_eq!(to_dozenal(144), "100"); // One gross
        assert_eq!(to_dozenal(255), "193"); // Max u8 in dozenal
    }

    #[test]
    fn test_generator() {
        let gen = BitwiseGenerator::new(255);
        let problems = gen.generate_batch(10);
        assert_eq!(problems.len(), 10);

        // Verify all can be solved
        for prob in problems {
            let result = prob.solve();
            assert!(result.is_ok());
        }
    }
}
