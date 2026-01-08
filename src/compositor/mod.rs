//! Response composition
//!
//! The compositor takes parallel computation results and assembles
//! them into a coherent, verified response structure.
//!
//! ## Architecture
//!
//! ```text
//! Parallel Engines → ThoughtStructure (internal, verified)
//!                          ↓
//!                    Compositor (validates consistency)
//!                          ↓
//!                    Serializer (structure → language)
//!                          ↓
//!                    Natural Language Output
//! ```
//!
//! Key insight: Language is GENERATED from verified structure,
//! not part of the reasoning process itself.

pub mod response;
pub mod serializer;
pub mod thought;

pub use response::ComposedResponse;
pub use serializer::{Serializer, NaturalLanguageSerializer, Verbosity};
pub use thought::{
    ThoughtStructure, Computation, ComputationStep,
    Transformation, Justification, Dependency, DependencyReason
};
