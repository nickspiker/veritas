//! Byte-level transformer model
//!
//! Complete architecture:
//! - Byte embedding
//! - N transformer layers
//! - Output projection to byte logits

use crate::autograd::{Tensor, Shape, matmul};
use crate::error::Result;
use spirix::ScalarF4E4;

use super::{ByteEmbedding, TransformerLayer};

/// Byte-level transformer model
pub struct ByteTransformer {
    /// Byte embedding layer
    pub embedding: ByteEmbedding,

    /// Transformer layers
    pub layers: Vec<TransformerLayer>,

    /// Output projection: [embed_dim, 256]
    /// Maps embeddings to byte logits
    pub output_proj: Tensor,

    /// Model configuration
    pub config: TransformerConfig,
}

/// Model configuration
#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub embed_dim: usize,
    pub num_layers: usize,
    pub ffn_hidden_dim: usize,
    pub max_seq_len: usize,
}

impl Default for TransformerConfig {
    fn default() -> Self {
        Self {
            embed_dim: 512,
            num_layers: 12,
            ffn_hidden_dim: 2048,
            max_seq_len: 2048,
        }
    }
}

impl ByteTransformer {
    /// Create new transformer model
    pub fn new(config: TransformerConfig) -> Result<Self> {
        // Create embedding layer
        let embedding = ByteEmbedding::new(config.embed_dim, config.max_seq_len)?;

        // Create transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config.embed_dim, config.ffn_hidden_dim)?);
        }

        // Create output projection: embed_dim → 256 byte classes
        let output_proj = init_output_projection(config.embed_dim)?;

        Ok(Self {
            embedding,
            layers,
            output_proj,
            config,
        })
    }

    /// Forward pass: bytes → next byte logits
    ///
    /// Input: &[u8] byte sequence
    /// Output: Tensor [seq_len, 256] logits for each next byte
    pub fn forward(&self, bytes: &[u8]) -> Result<Tensor> {
        // Embed bytes
        let mut x = self.embedding.forward(bytes)?;

        // Apply transformer layers
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }

        // Project to byte logits
        let logits = matmul(&x, &self.output_proj)?;

        Ok(logits)
    }

    /// Predict next byte given sequence
    ///
    /// Returns logits for next byte (256 classes)
    pub fn predict_next(&self, bytes: &[u8]) -> Result<Vec<ScalarF4E4>> {
        let logits = self.forward(bytes)?;
        let data = logits.as_scalars()
            .ok_or_else(|| crate::error::VeritasError::InvalidInput(
                "Logits must be CpuScalar".to_string()
            ))?;

        // Get last timestep logits
        let seq_len = bytes.len();
        let start = (seq_len - 1) * 256;
        let end = start + 256;

        Ok(data[start..end].to_vec())
    }

    /// Sample next byte from logits
    pub fn sample(&self, bytes: &[u8]) -> Result<u8> {
        let logits = self.predict_next(bytes)?;

        // Find argmax (greedy sampling for now)
        let mut max_idx = 0;
        let mut max_val = logits[0];

        for (i, &val) in logits.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx as u8)
    }
}

/// Initialize output projection matrix
fn init_output_projection(embed_dim: usize) -> Result<Tensor> {
    let mut data = Vec::with_capacity(embed_dim * 256);
    let fan_avg = ScalarF4E4::from(((embed_dim + 256) / 2) as u32);
    let scale = (ScalarF4E4::ONE / fan_avg).sqrt();

    for _ in 0..(embed_dim * 256) {
        let init = ScalarF4E4::random_gauss() * scale;
        data.push(init);
    }

    Tensor::from_scalars(data, Shape::matrix(embed_dim, 256)).map(|t| t.with_requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() {
        let config = TransformerConfig {
            embed_dim: 64,
            num_layers: 2,
            ffn_hidden_dim: 256,
            max_seq_len: 128,
        };

        let model = ByteTransformer::new(config).unwrap();
        assert_eq!(model.layers.len(), 2);
    }

    #[test]
    fn test_model_forward() {
        let config = TransformerConfig {
            embed_dim: 64,
            num_layers: 2,
            ffn_hidden_dim: 256,
            max_seq_len: 128,
        };

        let model = ByteTransformer::new(config).unwrap();

        let bytes = b"Hello";
        let logits = model.forward(bytes).unwrap();

        // Check output shape: [5, 256]
        assert_eq!(logits.shape().dims(), &[5, 256]);
    }

    #[test]
    fn test_next_byte_prediction() {
        let config = TransformerConfig {
            embed_dim: 64,
            num_layers: 2,
            ffn_hidden_dim: 256,
            max_seq_len: 128,
        };

        let model = ByteTransformer::new(config).unwrap();

        let bytes = b"Hello, ";
        let next_byte = model.sample(bytes).unwrap();

        // Should return some byte (random initialization)
        assert!(next_byte < 256);
    }
}
