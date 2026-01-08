//! Proofs of correctness

use super::Claim;

/// A proof that a claim is correct
#[derive(Debug, Clone)]
pub struct Proof {
    /// The claim being proven
    pub claim: Claim,

    /// Steps in the proof
    pub steps: Vec<ProofStep>,

    /// Whether this proof has been verified
    pub verified: bool,
}

/// A single step in a proof
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Description of this step
    pub description: String,

    /// Justification (axiom, previous step, etc.)
    pub justification: String,
}

impl Proof {
    pub fn new(claim: Claim) -> Self {
        Proof {
            claim,
            steps: Vec::new(),
            verified: false,
        }
    }

    pub fn add_step(&mut self, description: impl Into<String>, justification: impl Into<String>) {
        self.steps.push(ProofStep {
            description: description.into(),
            justification: justification.into(),
        });
    }
}
