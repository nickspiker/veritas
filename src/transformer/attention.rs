//! Single-head attention mechanism
//!
//! Simplified attention (no multi-head for MVP).
//! Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

use crate::autograd::{Tensor, Shape, matmul};
use crate::error::Result;
use spirix::ScalarF4E4;

/// Single-head attention layer
pub struct Attention {
    /// Query projection: [embed_dim, embed_dim]
    pub w_query: Tensor,

    /// Key projection: [embed_dim, embed_dim]
    pub w_key: Tensor,

    /// Value projection: [embed_dim, embed_dim]
    pub w_value: Tensor,

    /// Output projection: [embed_dim, embed_dim]
    pub w_out: Tensor,

    /// Embedding dimension
    pub dim: usize,
}

impl Attention {
    /// Create new attention layer
    pub fn new(embed_dim: usize) -> Result<Self> {
        // Initialize weight matrices with Xavier initialization
        let w_query = init_weight_matrix(embed_dim, embed_dim)?;
        let w_key = init_weight_matrix(embed_dim, embed_dim)?;
        let w_value = init_weight_matrix(embed_dim, embed_dim)?;
        let w_out = init_weight_matrix(embed_dim, embed_dim)?;

        Ok(Self {
            w_query,
            w_key,
            w_value,
            w_out,
            dim: embed_dim,
        })
    }

    /// Forward pass: apply attention
    ///
    /// Input: [seq_len, embed_dim] embedded sequence
    /// Output: [seq_len, embed_dim] attended sequence
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let seq_len = shape.dims()[0];

        // Project to Q, K, V
        // Q = X @ W_Q
        let q = matmul(x, &self.w_query)?;

        // K = X @ W_K
        let k = matmul(x, &self.w_key)?;

        // V = X @ W_V
        let v = matmul(x, &self.w_value)?;

        // Compute attention scores: Q @ K^T
        // Shape: [seq_len, seq_len]
        let k_t = k.transpose()?;
        let scores = matmul(&q, &k_t)?;

        // Scale by 1/sqrt(d_k)
        let scale = ScalarF4E4::ONE / sqrt_approx(self.dim);
        let scaled_scores = scores.scale(scale)?;

        // Apply softmax (simplified: use row-wise normalization)
        let attention_weights = softmax_approx(&scaled_scores)?;

        // Apply attention: weights @ V
        // Shape: [seq_len, embed_dim]
        let attended = matmul(&attention_weights, &v)?;

        // Output projection
        let output = matmul(&attended, &self.w_out)?;

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

/// Approximate sqrt for Spirix scalar
fn sqrt_approx(n: usize) -> ScalarF4E4 {
    // Simple approximation: use lookup table for small values
    match n {
        1 => ScalarF4E4::ONE,
        4 => ScalarF4E4::from(2u8),
        9 => ScalarF4E4::from(3u8),
        16 => ScalarF4E4::from(4u8),
        25 => ScalarF4E4::from(5u8),
        64 => ScalarF4E4::from(8u8),
        256 => ScalarF4E4::from(16u8),
        512 => ScalarF4E4::from(23u8), // ~22.6
        _ => {
            // For other values, use binary search approximation
            let n_scalar = ScalarF4E4::from(n.min(255) as u8);
            // Rough approximation: n^0.5 ≈ n/2 for small n
            n_scalar / ScalarF4E4::from(2u8)
        }
    }
}

/// Approximate softmax using row-wise normalization
///
/// True softmax requires exp() which is expensive in Spirix.
/// Use simpler normalization: x_i / sum(x)
fn softmax_approx(scores: &Tensor) -> Result<Tensor> {
    let data = scores.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Scores must be CpuScalar".to_string()
        ))?;
    let shape = scores.shape();
    let seq_len = shape.dims()[0];

    let mut output = Vec::with_capacity(data.len());

    // For each row, normalize by sum
    for row in 0..seq_len {
        let start = row * seq_len;
        let end = start + seq_len;
        let row_data = &data[start..end];

        // Compute row sum
        let mut sum = ScalarF4E4::ZERO;
        for &val in row_data {
            sum = sum + val.magnitude(); // Use magnitude for stability
        }

        // Avoid division by zero
        if sum == ScalarF4E4::ZERO {
            sum = ScalarF4E4::ONE;
        }

        // Normalize
        for &val in row_data {
            output.push(val / sum);
        }
    }

    Tensor::from_scalars(output, shape.clone())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_creation() {
        let attn = Attention::new(512).unwrap();
        assert_eq!(attn.dim, 512);
    }

    #[test]
    fn test_attention_forward() {
        let attn = Attention::new(64).unwrap(); // Smaller for testing

        // Create dummy input: [4, 64]
        let mut input_data = vec![ScalarF4E4::ZERO; 4 * 64];
        for i in 0..input_data.len() {
            input_data[i] = ScalarF4E4::from((i % 10) as u8);
        }
        let input = Tensor::from_scalars(input_data, Shape::matrix(4, 64)).unwrap();

        let output = attn.forward(&input).unwrap();

        // Check output shape matches input
        assert_eq!(output.shape().dims(), &[4, 64]);
    }
}
