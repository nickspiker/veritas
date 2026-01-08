//! VSF-based persistence layer
//!
//! All storage uses VSF for:
//! - Optimal encoding
//! - Cryptographic verification
//! - Data provenance tracking

pub mod cache;
pub mod checkpoint;
pub mod training_data;

pub use cache::ComputationCache;
pub use checkpoint::Checkpoint;
pub use training_data::TrainingExample;

// TODO: Implement VSF persistence
