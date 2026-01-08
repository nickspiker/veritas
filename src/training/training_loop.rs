//! Main training loop demonstrating verified self-play

use super::{TrainingGenerator, NeuralExplainer, Verifier, VerificationResult, VerificationStats};
use crate::symbolic::{Context, Evaluate};
use crate::error::Result;

/// The main training loop
///
/// This demonstrates the core innovation:
/// 1. Symbolic generates problem + solution (ground truth)
/// 2. Neural generates explanation + answer
/// 3. Verification checks if they match
/// 4. Contradictions create training signal
/// 5. System self-corrects
pub struct TrainingLoop {
    generator: TrainingGenerator,
    explainer: NeuralExplainer,
    verifier: Verifier,
    stats: VerificationStats,
}

impl TrainingLoop {
    pub fn new(introduce_errors: bool) -> Self {
        TrainingLoop {
            generator: TrainingGenerator::new(1),
            explainer: NeuralExplainer::new(introduce_errors),
            verifier: Verifier::new(),
            stats: VerificationStats::new(),
        }
    }

    /// Run one training iteration
    ///
    /// Returns the verification result for this iteration
    pub fn train_one(&mut self) -> Result<TrainingIteration> {
        // 1. Symbolic generates verified problem
        let example = self.generator.generate()?;

        // 2. Neural generates explanation
        let (explanation, neural_answer) = self.explainer.explain(&example.problem);

        // 3. Verify neural answer against symbolic ground truth
        let verification = self.verifier.verify(
            explanation.clone(),
            neural_answer,
            example.solution,
        );

        // 4. Update statistics
        match &verification {
            VerificationResult::Correct { .. } => {
                self.stats.record_correct();
            }
            VerificationResult::Contradicted { .. } => {
                self.stats.record_contradiction();
            }
        }

        Ok(TrainingIteration {
            problem: format!("{}", example.problem),
            verification,
        })
    }

    /// Run multiple iterations
    pub fn train(&mut self, iterations: usize) -> Result<Vec<TrainingIteration>> {
        let mut results = Vec::new();

        for _ in 0..iterations {
            results.push(self.train_one()?);
        }

        Ok(results)
    }

    /// Get current statistics
    pub fn stats(&self) -> &VerificationStats {
        &self.stats
    }

    /// Get generator statistics
    pub fn generator_stats(&self) -> super::GeneratorStats {
        self.generator.stats()
    }
}

/// Result of one training iteration
#[derive(Debug, Clone)]
pub struct TrainingIteration {
    pub problem: String,
    pub verification: VerificationResult,
}

impl TrainingIteration {
    pub fn is_correct(&self) -> bool {
        self.verification.is_correct()
    }

    pub fn is_contradicted(&self) -> bool {
        self.verification.is_contradicted()
    }
}
