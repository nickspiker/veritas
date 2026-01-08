//! Thought structure representation
//!
//! Internal representation of reasoning BEFORE serialization to language.

use crate::symbolic::Expr;
use crate::numeric::Scalar;
use crate::verification::Proof;

/// Internal thought structure (NOT text yet)
///
/// This is the structured representation of reasoning.
/// Language is generated FROM this, not part of it.
#[derive(Debug, Clone)]
pub struct ThoughtStructure {
    /// Computations performed
    pub computations: Vec<Computation>,

    /// Proofs constructed
    pub proofs: Vec<Proof>,

    /// Dependencies between thoughts
    pub dependencies: Vec<Dependency>,
}

/// A single computation with its result
#[derive(Debug, Clone)]
pub struct Computation {
    pub id: usize,
    pub expression: Expr,
    pub result: Scalar,
    pub verified: bool,

    /// HOW it was computed (steps)
    pub steps: Vec<ComputationStep>,
}

/// A step in computation
#[derive(Debug, Clone)]
pub struct ComputationStep {
    /// The transformation applied
    pub transformation: Transformation,

    /// Expression before this step
    pub before: Expr,

    /// Expression after this step
    pub after: Expr,

    /// Why this transformation is valid
    pub justification: Justification,
}

/// Types of transformations
#[derive(Debug, Clone)]
pub enum Transformation {
    /// Algebraic simplification
    Simplify,

    /// Substitution of variable
    Substitute { var: String, value: Scalar },

    /// Evaluation of subexpression
    Evaluate { subexpr: Expr, result: Scalar },

    /// Application of identity (e.g., x + 0 = x)
    Identity { rule: String },
}

/// Why a transformation is valid
#[derive(Debug, Clone)]
pub enum Justification {
    /// Algebraic identity
    AlgebraicIdentity(String),

    /// Arithmetic evaluation (verified by Spirix)
    ArithmeticEvaluation,

    /// Variable binding from context
    VariableBinding,

    /// Previously proven fact
    ProvenFact(usize),
}

/// Dependency between thoughts
#[derive(Debug, Clone)]
pub struct Dependency {
    pub from: usize,
    pub to: usize,
    pub reason: DependencyReason,
}

/// Why one thought depends on another
#[derive(Debug, Clone)]
pub enum DependencyReason {
    /// Result used in computation
    UsesResult,

    /// Proof used as justification
    UsesProof,

    /// Sequential ordering required
    Sequential,
}

impl ThoughtStructure {
    pub fn new() -> Self {
        ThoughtStructure {
            computations: Vec::new(),
            proofs: Vec::new(),
            dependencies: Vec::new(),
        }
    }

    /// Add a computation
    pub fn add_computation(&mut self, comp: Computation) -> usize {
        let id = self.computations.len();
        self.computations.push(comp);
        id
    }

    /// Add dependency
    pub fn add_dependency(&mut self, from: usize, to: usize, reason: DependencyReason) {
        self.dependencies.push(Dependency { from, to, reason });
    }

    /// Get computation by ID
    pub fn get_computation(&self, id: usize) -> Option<&Computation> {
        self.computations.get(id)
    }
}

impl Default for ThoughtStructure {
    fn default() -> Self {
        Self::new()
    }
}
