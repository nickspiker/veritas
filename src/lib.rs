//! # Veritas: Digital Intelligence thru Verified Computation
//!
//! A fundamentally different approach to intelligence:
//! - Symbolic systems generate ground truth
//! - Neural systems learn to explain it
//! - Verification ensures consistency
//! - All computation is verifiable or marked uncertain
//!
//! ## Architecture
//!
//! ```text
//! Query → Intent → Parallel Computation
//!                      ↓
//!                 [Symbolic]  [Code]  [Logic]
//!                      ↓         ↓       ↓
//!                 [Verified] [Verified] [Verified]
//!                      ↓         ↓       ↓
//!                 ThoughtStructure (internal)
//!                      ↓
//!                 Compositor (validates)
//!                      ↓
//!                 Serializer (format)
//!                      ↓
//!                 Output (verified)
//! ```
//!
//! ## Core Principles
//!
//! 1. **No IEEE-754** - Spirix two's complement floats everywhere
//! 2. **Verification-first** - Every claim is verified or marked uncertain
//! 3. **Structured reasoning** - Not token soup
//! 4. **Cryptographic provenance** - Know where data came from
//! 5. **Memory safety** - Rust enforces correctness at compile time

pub mod compositor;
pub mod numeric;
pub mod persistence;
pub mod symbolic;
pub mod training;
pub mod verification;

pub mod error;
pub use error::{Result, VeritasError};

// Re-export key types
pub use compositor::{ComposedResponse, ThoughtStructure};
pub use numeric::{Circle, Complex, Scalar};
pub use symbolic::{Expr, Simplify};
pub use verification::{Claim, Proof, VerificationState};
