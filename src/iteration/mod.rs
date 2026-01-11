//! Iteration Engine - Core z² + c mechanics
//!
//! This is where intelligence happens:
//! - Iterate computational state until bounded (converged) or escaped (diverged)
//! - Truth stays bounded, bullshit escapes to infinity
//! - No token limits, just mathematical convergence

pub mod engine;
pub mod convergence;

pub use engine::{IterationEngine, IterationState, IterationResult};
pub use convergence::{ConvergenceDetector, ConvergenceConfig};
