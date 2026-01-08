//! Self-generating training data
//!
//! The symbolic engine generates problems, solves them,
//! and creates verified training examples.
//!
//! The neural explainer generates explanations that are
//! verified against symbolic ground truth.
//!
//! Contradictions create training signal for self-correction.

pub mod generator;
pub mod explainer;
pub mod verifier;
pub mod training_loop;
pub mod neural_cpu;
pub mod bridge;

pub use generator::{TrainingGenerator, GeneratorStats};
pub use explainer::NeuralExplainer;
pub use verifier::{Verifier, VerificationResult, VerificationStats};
pub use training_loop::TrainingLoop;
pub use neural_cpu::{SimpleParser, SimpleTrainer, TrainingConfig, TrainingStats};
pub use bridge::{
    ieee_to_f4e4, f4e4_to_f6e5, f6e5_to_f4e4, f4e4_to_ieee,
    ieee_array_to_f4e4, f4e4_array_to_f6e5
};
