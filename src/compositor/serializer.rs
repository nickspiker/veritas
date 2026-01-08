//! Response serialization
//!
//! Converts structured thoughts to natural language.
//! THIS is where language generation happens.

use super::{ComposedResponse, ThoughtStructure};
use super::thought::{Transformation, Justification, Computation};

/// Trait for serializing thought structures to output formats
pub trait Serializer {
    fn serialize(&self, thoughts: &ThoughtStructure) -> ComposedResponse;
}

/// Natural language serializer
///
/// This is where Structure → English happens.
/// In production, this would be a trained neural network.
/// For now, rule-based templates.
pub struct NaturalLanguageSerializer {
    verbosity: Verbosity,
}

#[derive(Debug, Clone, Copy)]
pub enum Verbosity {
    Concise,
    Detailed,
    VeryDetailed,
}

impl NaturalLanguageSerializer {
    pub fn new(verbosity: Verbosity) -> Self {
        NaturalLanguageSerializer { verbosity }
    }

    /// Serialize a single computation to English
    fn serialize_computation(&self, comp: &Computation) -> String {
        let mut output = String::new();

        // Start with the problem
        output.push_str(&format!("Computing: {}\n", comp.expression));

        // Show steps if detailed
        if matches!(self.verbosity, Verbosity::Detailed | Verbosity::VeryDetailed) {
            for (i, step) in comp.steps.iter().enumerate() {
                output.push_str(&format!("  Step {}: ", i + 1));
                output.push_str(&self.serialize_step(step));
                output.push('\n');
            }
        }

        // Show result
        output.push_str(&format!("Result: {}", comp.result));

        if comp.verified {
            output.push_str(" ✓");
        } else {
            output.push_str(" (unverified)");
        }

        output
    }

    /// Serialize a computation step to English
    fn serialize_step(&self, step: &super::thought::ComputationStep) -> String {
        let transformation_text = match &step.transformation {
            Transformation::Simplify => {
                format!("Simplify {} to {}", step.before, step.after)
            }
            Transformation::Substitute { var, value } => {
                format!("Substitute {} = {} into {}, getting {}",
                    var, value, step.before, step.after)
            }
            Transformation::Evaluate { subexpr, result } => {
                format!("Evaluate {} = {}, giving {}",
                    subexpr, result, step.after)
            }
            Transformation::Identity { rule } => {
                format!("Apply {} to get {}",
                    rule, step.after)
            }
        };

        // Add justification if very detailed
        if matches!(self.verbosity, Verbosity::VeryDetailed) {
            let justification_text = match &step.justification {
                Justification::AlgebraicIdentity(rule) => {
                    format!(" (by {})", rule)
                }
                Justification::ArithmeticEvaluation => {
                    " (verified arithmetic)".to_string()
                }
                Justification::VariableBinding => {
                    " (variable substitution)".to_string()
                }
                Justification::ProvenFact(id) => {
                    format!(" (by proven fact #{})", id)
                }
            };
            format!("{}{}", transformation_text, justification_text)
        } else {
            transformation_text
        }
    }
}

impl Serializer for NaturalLanguageSerializer {
    fn serialize(&self, thoughts: &ThoughtStructure) -> ComposedResponse {
        let mut output = String::new();

        // Serialize each computation
        for comp in &thoughts.computations {
            output.push_str(&self.serialize_computation(comp));
            output.push_str("\n\n");
        }

        // Show proofs if any
        if !thoughts.proofs.is_empty() {
            output.push_str("Proofs:\n");
            for (i, proof) in thoughts.proofs.iter().enumerate() {
                output.push_str(&format!("{}. {}\n", i + 1, proof.claim.statement));
            }
        }

        ComposedResponse::text(output)
    }
}

/// Simple text serializer (fallback)
pub struct TextSerializer;

impl Serializer for TextSerializer {
    fn serialize(&self, thoughts: &ThoughtStructure) -> ComposedResponse {
        ComposedResponse::text(format!("Thoughts: {:?}", thoughts))
    }
}
