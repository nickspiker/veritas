//! Input Encoding - Extract signal from noise
//!
//! ## The Problem
//!
//! Real inputs are messy:
//! - "☙ ❦ ⁂ cosechhhj clarity of bumbled sorts burnt tacos how do I get to rome?"
//! - Unicode chaos, typos, word salad mixed with actual query
//!
//! ## The Solution
//!
//! Iterative compression finds what survives:
//! - High information density = signal (survives compression)
//! - Low information density = noise (compresses to nothing)
//!
//! ## Architecture
//!
//! ```text
//! Raw text → Tokenize → Embed → Compress → Intent
//!                                    ↓
//!                             [What survives?]
//! ```

pub mod tokenizer;
pub mod compression;
pub mod intent;
pub mod preprocess;

pub use tokenizer::{Tokenizer, Token};
pub use compression::{CompressionEncoder, CompressionResult};
pub use intent::{IntentClassifier, Intent, QueryType};
pub use preprocess::{preprocess, inject_result, PreprocessResult, Operator};
