//! Verification framework
//!
//! This module provides the verification layer that ensures
//! symbolic computations match their claimed results.

pub mod claim;
pub mod proof;
pub mod state;

pub use claim::Claim;
pub use proof::Proof;
pub use state::VerificationState;

// TODO: Implement full verification system
