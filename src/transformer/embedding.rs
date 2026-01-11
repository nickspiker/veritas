//! Byte-level embedding layer
//!
//! Maps raw bytes (0-255) to dense 512-dim vectors.
//! No tokenization, no vocabulary - just byte → vector.

use crate::autograd::{Tensor, Shape};
use crate::error::Result;
use spirix::ScalarF4E4;

/// Byte embedding layer
///
/// Converts raw bytes to dense vectors in embedding space.
/// 256 possible byte values × 512 dimensions = 131,072 parameters
pub struct ByteEmbedding {
    /// Embedding matrix: [256, 512]
    /// Each row is the embedding for one byte value
    pub embedding: Tensor,

    /// Position encoding matrix: [max_seq_len, 512]
    /// Sinusoidal position encodings
    pub position: Tensor,

    /// Embedding dimension
    pub dim: usize,

    /// Maximum sequence length
    pub max_len: usize,
}

impl ByteEmbedding {
    /// Create new byte embedding layer
    pub fn new(embedding_dim: usize, max_seq_len: usize) -> Result<Self> {
        // Initialize embedding matrix with Xavier/Glorot initialization (pure Spirix)
        let mut embed_data = Vec::with_capacity(256 * embedding_dim);
        let scale = (ScalarF4E4::ONE / ScalarF4E4::from(embedding_dim as u32)).sqrt();

        for _ in 0..(256 * embedding_dim) {
            embed_data.push(ScalarF4E4::random_gauss() * scale);
        }

        let embedding = Tensor::from_scalars(
            embed_data,
            Shape::matrix(256, embedding_dim),
        )?.with_requires_grad();

        // Create sinusoidal position encodings
        let position = create_position_encoding(max_seq_len, embedding_dim)?;

        Ok(Self {
            embedding,
            position,
            dim: embedding_dim,
            max_len: max_seq_len,
        })
    }

    /// Forward pass: bytes → embedded vectors
    ///
    /// Input: [batch_size, seq_len] byte values
    /// Output: [batch_size, seq_len, embed_dim] embedded vectors
    pub fn forward(&self, bytes: &[u8]) -> Result<Tensor> {
        let seq_len = bytes.len();
        if seq_len > self.max_len {
            return Err(crate::error::VeritasError::InvalidInput(
                format!("Sequence length {} exceeds maximum {}", seq_len, self.max_len)
            ));
        }

        // Look up embeddings for each byte
        let mut embedded = Vec::with_capacity(seq_len * self.dim);
        let embed_data = self.embedding.as_scalars()
            .ok_or_else(|| crate::error::VeritasError::InvalidInput(
                "Embedding must be CpuScalar".to_string()
            ))?;

        // Get position encoding data once
        let pos_data = self.position.as_scalars()
            .ok_or_else(|| crate::error::VeritasError::InvalidInput(
                "Position encoding must be CpuScalar".to_string()
            ))?;

        for (pos, &byte) in bytes.iter().enumerate() {
            let byte_idx = byte as usize;

            // Get byte embedding
            let start = byte_idx * self.dim;
            let end = start + self.dim;
            let byte_embed = &embed_data[start..end];

            // Get position encoding
            let pos_start = pos * self.dim;
            let pos_end = pos_start + self.dim;
            let pos_embed = &pos_data[pos_start..pos_end];

            // Add byte embedding + position encoding
            for i in 0..self.dim {
                embedded.push(byte_embed[i] + pos_embed[i]);
            }
        }

        // Reshape to [seq_len, embed_dim]
        Tensor::from_scalars(embedded, Shape::matrix(seq_len, self.dim))
    }

    /// Get embedding dimension
    pub fn dim(&self) -> usize {
        self.dim
    }
}

/// Create sinusoidal position encodings
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
fn create_position_encoding(max_len: usize, dim: usize) -> Result<Tensor> {
    let mut encoding = Vec::with_capacity(max_len * dim);

    for pos in 0..max_len {
        for i in 0..dim {
            // Compute position encoding using Spirix
            let pos_scalar = ScalarF4E4::from(pos as u8);
            let dim_scalar = ScalarF4E4::from(dim as u8);
            let i_scalar = ScalarF4E4::from(i as u8);

            // Approximation: use simple sinusoidal pattern
            // Full implementation would use proper sin/cos from Spirix
            // For now, use alternating pattern scaled by position
            let val = if i % 2 == 0 {
                // Even: sin-like (ascending pattern)
                (pos_scalar * i_scalar) / (dim_scalar * ScalarF4E4::from(10u8))
            } else {
                // Odd: cos-like (descending pattern)
                ScalarF4E4::ONE - (pos_scalar * i_scalar) / (dim_scalar * ScalarF4E4::from(10u8))
            };

            encoding.push(val);
        }
    }

    Tensor::from_scalars(encoding, Shape::matrix(max_len, dim))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_creation() {
        let embed = ByteEmbedding::new(512, 2048).unwrap();
        assert_eq!(embed.dim(), 512);
    }

    #[test]
    fn test_forward_pass() {
        let embed = ByteEmbedding::new(512, 2048).unwrap();

        // Test with simple byte sequence
        let bytes = vec![b'H', b'e', b'l', b'l', b'o'];
        let output = embed.forward(&bytes).unwrap();

        // Check output shape: [5, 512]
        assert_eq!(output.shape().dims(), &[5, 512]);
    }

    #[test]
    fn test_utf8_bytes() {
        let embed = ByteEmbedding::new(512, 2048).unwrap();

        // UTF-8 string with multi-byte characters
        let text = "Hello, 世界!";
        let bytes = text.as_bytes();

        let output = embed.forward(bytes).unwrap();

        // Each UTF-8 byte gets embedded
        assert_eq!(output.shape().dims()[0], bytes.len());
    }
}
