//! Feed-forward network layer
//!
//! Two-layer FFN with ReLU activation:
//! FFN(x) = W2(ReLU(W1(x)))

use crate::autograd::{Tensor, Shape, matmul, relu};
use crate::error::Result;
use spirix::ScalarF4E4;

/// Feed-forward network
pub struct FeedForward {
    /// First layer: [embed_dim, hidden_dim]
    pub w1: Tensor,

    /// Second layer: [hidden_dim, embed_dim]
    pub w2: Tensor,

    /// Embedding dimension
    pub embed_dim: usize,

    /// Hidden dimension (typically 4× embed_dim)
    pub hidden_dim: usize,
}

impl FeedForward {
    /// Create new FFN layer
    pub fn new(embed_dim: usize, hidden_dim: usize) -> Result<Self> {
        let w1 = init_weight_matrix(embed_dim, hidden_dim)?;
        let w2 = init_weight_matrix(hidden_dim, embed_dim)?;

        Ok(Self {
            w1,
            w2,
            embed_dim,
            hidden_dim,
        })
    }

    /// Forward pass
    ///
    /// Input: [seq_len, embed_dim]
    /// Output: [seq_len, embed_dim]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // First layer + ReLU
        // Shape: [seq_len, hidden_dim]
        let hidden = matmul(x, &self.w1)?;
        let activated = relu(&hidden)?;

        // Second layer
        // Shape: [seq_len, embed_dim]
        let output = matmul(&activated, &self.w2)?;

        Ok(output)
    }
}

/// Initialize weight matrix with Xavier initialization (pure Spirix)
fn init_weight_matrix(rows: usize, cols: usize) -> Result<Tensor> {
    // Xavier scale: 1 / sqrt((rows + cols) / 2) in pure Spirix
    let fan_avg = ScalarF4E4::from(((rows + cols) / 2) as u32);
    let scale = (ScalarF4E4::ONE / fan_avg).sqrt();

    let data: Vec<ScalarF4E4> = (0..(rows * cols))
        .map(|_| ScalarF4E4::random_gauss() * scale)
        .collect();

    Tensor::from_scalars(data, Shape::matrix(rows, cols)).map(|t| t.with_requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffn_creation() {
        let ffn = FeedForward::new(512, 2048).unwrap();
        assert_eq!(ffn.embed_dim, 512);
        assert_eq!(ffn.hidden_dim, 2048);
    }

    #[test]
    fn test_ffn_forward() {
        let ffn = FeedForward::new(64, 256).unwrap(); // Smaller for testing

        // Create dummy input: [4, 64]
        let mut input_data = vec![ScalarF4E4::ZERO; 4 * 64];
        for i in 0..input_data.len() {
            input_data[i] = ScalarF4E4::from((i % 10) as u8);
        }
        let input = Tensor::from_scalars(input_data, Shape::matrix(4, 64)).unwrap();

        let output = ffn.forward(&input).unwrap();

        // Check output shape matches input embed_dim
        assert_eq!(output.shape().dims(), &[4, 64]);
    }
}
