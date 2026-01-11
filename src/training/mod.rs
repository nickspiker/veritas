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
pub mod data_gen;
pub mod pid_lr;
pub mod checkpoint;
pub mod diagnostics;
pub mod expression_parser;
pub mod code_module;

pub use generator::{TrainingGenerator, GeneratorStats};
pub use explainer::NeuralExplainer;
pub use verifier::{Verifier, VerificationResult, VerificationStats};
pub use training_loop::TrainingLoop;
pub use neural_cpu::{SimpleParser, SimpleTrainer, TrainingConfig, TrainingStats};
pub use data_gen::{generate_math_examples, generate_text_examples, generate_training_set, TrainingExample};
pub use pid_lr::PIDLearningRate;
pub use checkpoint::Checkpoint;
pub use diagnostics::Diagnostics;
pub use expression_parser::{ParsedExpression, Operation, parse_math_expression, call_basecalc};
pub use code_module::{verify_rust_code, generate_code_examples, generate_test_code_examples, generate_non_code_examples};
