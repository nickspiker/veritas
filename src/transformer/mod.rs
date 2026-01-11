//! Byte-level Transformer Architecture
//!
//! No tokenization - direct UTF-8 byte processing.
//! Pure Spirix arithmetic throughout.
//!
//! Architecture:
//! - Input: Vec<u8> (256 possible byte values)
//! - Embedding: byte → 512-dim ScalarF4E4 vector
//! - 12 transformer layers (attention + FFN)
//! - Output: 256-class next-byte prediction
//!
//! Design principles:
//! - Network learns ROUTING, not computation
//! - Numbers replaced with <MATH_N> placeholders
//! - Intent generation → symbolic execution
//! - Verified results injected back into generation

pub mod embedding;
pub mod attention;
pub mod ffn;
pub mod layer;
pub mod model;
pub mod rnn;

pub use embedding::ByteEmbedding;
pub use attention::Attention;
pub use ffn::FeedForward;
pub use layer::TransformerLayer;
pub use model::{ByteTransformer, TransformerConfig};
pub use rnn::{SimpleRNN, RNNConfig};
