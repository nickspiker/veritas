//! Error types for Veritas
//!
//! All errors are strongly typed. No string errors, no wildcards.

use thiserror::Error;

pub type Result<T> = std::result::Result<T, VeritasError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum VeritasError {
    // Symbolic errors
    #[error("Division by zero in symbolic expression")]
    DivisionByZero,

    #[error("Variable not found: {0}")]
    VariableNotFound(String),

    #[error("Invalid simplification: {0}")]
    SimplificationError(String),

    #[error("Expression too complex: depth {0} exceeds limit")]
    ComplexityLimit(usize),

    // Verification errors
    #[error("Verification failed: expected {expected}, got {actual}")]
    VerificationFailed { expected: String, actual: String },

    #[error("Proof invalid: {0}")]
    ProofInvalid(String),

    #[error("Claim cannot be verified: {0}")]
    UnverifiableClaim(String),

    #[error("Contradiction detected: {0}")]
    Contradiction(String),

    // Numeric errors
    #[error("Numeric underflow")]
    NumericUnderflow,

    #[error("Numeric overflow")]
    NumericOverflow,

    #[error("Undefined numeric operation: {0}")]
    UndefinedOperation(String),

    // Persistence errors
    #[error("Failed to encode: {0}")]
    EncodingError(String),

    #[error("Failed to decode: {0}")]
    DecodingError(String),

    #[error("Integrity check failed: {0}")]
    IntegrityError(String),

    #[error("Signature verification failed")]
    SignatureVerificationFailed,

    // Compositor errors
    #[error("Response composition failed: {0}")]
    CompositionError(String),

    #[error("Dependency cycle detected in thought structure")]
    DependencyCycle,

    #[error("Missing required thought node: {0}")]
    MissingNode(String),

    // Training errors
    #[error("Training example generation failed: {0}")]
    GenerationFailed(String),

    #[error("Training data corrupted: {0}")]
    CorruptedTrainingData(String),
}

impl VeritasError {
    /// Check if this error represents a mathematical impossibility
    /// (as opposed to a bug or data corruption)
    pub fn is_mathematical(&self) -> bool {
        matches!(
            self,
            VeritasError::DivisionByZero
                | VeritasError::UndefinedOperation(_)
                | VeritasError::NumericUnderflow
                | VeritasError::NumericOverflow
        )
    }

    /// Check if this error represents a verification failure
    pub fn is_verification_failure(&self) -> bool {
        matches!(
            self,
            VeritasError::VerificationFailed { .. }
                | VeritasError::ProofInvalid(_)
                | VeritasError::Contradiction(_)
        )
    }
}
