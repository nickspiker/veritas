//! Verification of neural explanations against symbolic ground truth

use crate::numeric::Scalar;
use crate::error::{Result, VeritasError};

/// Result of verifying a neural explanation
#[derive(Debug, Clone)]
pub enum VerificationResult {
    /// Explanation matches ground truth
    Correct {
        explanation: String,
        answer: Scalar,
    },

    /// Explanation contradicts ground truth
    Contradicted {
        explanation: String,
        claimed_answer: Scalar,
        correct_answer: Scalar,
        error: Scalar,
    },
}

impl VerificationResult {
    pub fn is_correct(&self) -> bool {
        matches!(self, VerificationResult::Correct { .. })
    }

    pub fn is_contradicted(&self) -> bool {
        matches!(self, VerificationResult::Contradicted { .. })
    }
}

/// Verifies neural explanations against symbolic ground truth
pub struct Verifier {
    /// Tolerance for numerical comparison
    epsilon: Scalar,
}

impl Verifier {
    pub fn new() -> Self {
        Verifier {
            epsilon: Scalar::from(1e-10),  // Very tight tolerance
        }
    }

    /// Verify that neural explanation matches symbolic solution
    ///
    /// This is the critical verification step:
    /// - Neural generates explanation + answer
    /// - Symbolic has ground truth
    /// - We check if they match
    ///
    /// If they don't match → contradiction detected → training signal
    pub fn verify(
        &self,
        explanation: String,
        neural_answer: Scalar,
        symbolic_answer: Scalar,
    ) -> VerificationResult {
        // Check if answers match (within epsilon)
        let diff = (neural_answer - symbolic_answer).abs();

        if diff.inner() < self.epsilon.inner() {
            VerificationResult::Correct {
                explanation,
                answer: neural_answer,
            }
        } else {
            VerificationResult::Contradicted {
                explanation,
                claimed_answer: neural_answer,
                correct_answer: symbolic_answer,
                error: diff,
            }
        }
    }
}

impl Default for Verifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics tracking verification results
#[derive(Debug, Clone, Default)]
pub struct VerificationStats {
    pub total_verified: usize,
    pub correct: usize,
    pub contradicted: usize,
}

impl VerificationStats {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn record_correct(&mut self) {
        self.total_verified += 1;
        self.correct += 1;
    }

    pub fn record_contradiction(&mut self) {
        self.total_verified += 1;
        self.contradicted += 1;
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_verified == 0 {
            0.0
        } else {
            self.correct as f64 / self.total_verified as f64
        }
    }
}
