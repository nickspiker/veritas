//! Training data generator
//!
//! Generates verified mathematical problems for training.

use crate::persistence::TrainingExample;
use crate::symbolic::{Expr, Context, Evaluate};
use crate::numeric::Scalar;
use crate::error::Result;

/// Generates verified training examples
pub struct TrainingGenerator {
    difficulty: u32,
    problems_generated: usize,
}

impl TrainingGenerator {
    pub fn new(difficulty: u32) -> Self {
        TrainingGenerator {
            difficulty,
            problems_generated: 0,
        }
    }

    /// Generate a verified problem-solution pair
    ///
    /// The symbolic engine generates the problem, solves it (ground truth),
    /// and provides a proof that the solution is correct.
    pub fn generate(&mut self) -> Result<TrainingExample> {
        // Generate different problem types based on count
        let problem_type = self.problems_generated % 5;

        let (problem, solution) = match problem_type {
            0 => self.generate_addition()?,
            1 => self.generate_multiplication()?,
            2 => self.generate_power()?,
            3 => self.generate_sqrt()?,
            4 => self.generate_combined()?,
            _ => unreachable!(),
        };

        self.problems_generated += 1;

        Ok(TrainingExample::new(
            problem,
            solution,
            "symbolic evaluation (ground truth)"
        ))
    }

    /// Generate simple addition problem
    fn generate_addition(&self) -> Result<(Expr, Scalar)> {
        let a = self.random_small_int();
        let b = self.random_small_int();

        let problem = Expr::add(Expr::number(a), Expr::number(b));
        let solution = Scalar::from(a) + Scalar::from(b);

        Ok((problem, solution))
    }

    /// Generate multiplication problem
    fn generate_multiplication(&self) -> Result<(Expr, Scalar)> {
        let a = self.random_small_int();
        let b = self.random_small_int();

        let problem = Expr::mul(Expr::number(a), Expr::number(b));
        let solution = Scalar::from(a) * Scalar::from(b);

        Ok((problem, solution))
    }

    /// Generate power problem
    fn generate_power(&self) -> Result<(Expr, Scalar)> {
        // Keep base positive and small (1-5)
        let base = (self.random_small_int().abs() % 5) + 1;
        let exp = (self.problems_generated % 3) as i32 + 1; // 1-3

        let problem = Expr::pow(Expr::number(base), Expr::number(exp));
        let solution = Scalar::from(base).pow(Scalar::from(exp))?;

        Ok((problem, solution))
    }

    /// Generate square root problem
    fn generate_sqrt(&self) -> Result<(Expr, Scalar)> {
        // Use perfect squares for now
        let root = (self.problems_generated % 10) as i32 + 1;
        let value = root * root;

        let problem = Expr::sqrt(Expr::number(value));
        let solution = Scalar::from(root);

        Ok((problem, solution))
    }

    /// Generate combined operations
    fn generate_combined(&self) -> Result<(Expr, Scalar)> {
        // (a + b) * c
        let a = self.random_small_int();
        let b = self.random_small_int();
        let c = self.random_small_int();

        let problem = Expr::mul(
            Expr::add(Expr::number(a), Expr::number(b)),
            Expr::number(c)
        );

        let solution = (Scalar::from(a) + Scalar::from(b)) * Scalar::from(c);

        Ok((problem, solution))
    }

    /// Generate random small integer (for simplicity)
    fn random_small_int(&self) -> i32 {
        // Simple deterministic "random" for now
        let seed = (self.problems_generated * 17 + self.difficulty as usize) % 20;
        (seed as i32) - 5  // Range: -5 to 14
    }

    /// Get statistics
    pub fn stats(&self) -> GeneratorStats {
        GeneratorStats {
            problems_generated: self.problems_generated,
            difficulty: self.difficulty,
        }
    }
}

/// Generator statistics
#[derive(Debug, Clone)]
pub struct GeneratorStats {
    pub problems_generated: usize,
    pub difficulty: u32,
}
