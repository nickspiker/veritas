//! Verification state tracking

use crate::numeric::{Circle, Scalar};

/// State of a verification
#[derive(Debug, Clone, PartialEq)]
pub enum VerificationState {
    /// Not yet verified
    Unverified,

    /// Verified as correct with proof
    Verified { proof_id: String },

    /// Contradicted - expected vs actual mismatch
    Contradicted {
        expected: String,
        actual: String,
        error: Scalar,
    },

    /// Cannot be verified (unprovable, undecidable, etc.)
    Uncertain { reason: String },
}

impl VerificationState {
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationState::Verified { .. })
    }

    pub fn is_contradicted(&self) -> bool {
        matches!(self, VerificationState::Contradicted { .. })
    }

    pub fn is_uncertain(&self) -> bool {
        matches!(self, VerificationState::Uncertain { .. })
    }
}
