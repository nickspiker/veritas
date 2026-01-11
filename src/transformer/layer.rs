//! Transformer layer
//!
//! Single transformer block: attention + FFN with residual connections

use crate::autograd::Tensor;
use crate::error::Result;
use spirix::ScalarF4E4;

use super::{Attention, FeedForward};

/// Transformer layer (attention + FFN)
pub struct TransformerLayer {
    pub attention: Attention,
    pub ffn: FeedForward,
}

impl TransformerLayer {
    /// Create new transformer layer
    pub fn new(embed_dim: usize, ffn_hidden_dim: usize) -> Result<Self> {
        let attention = Attention::new(embed_dim)?;
        let ffn = FeedForward::new(embed_dim, ffn_hidden_dim)?;

        Ok(Self { attention, ffn })
    }

    /// Forward pass with residual connections
    ///
    /// x = x + Attention(x)
    /// x = x + FFN(x)
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Attention with residual
        let attn_out = self.attention.forward(x)?;
        let x = x.add(&attn_out)?;

        // FFN with residual
        let ffn_out = self.ffn.forward(&x)?;
        let x = x.add(&ffn_out)?;

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::Shape;

    #[test]
    fn test_layer_creation() {
        let layer = TransformerLayer::new(512, 2048).unwrap();
        assert_eq!(layer.attention.dim, 512);
        assert_eq!(layer.ffn.embed_dim, 512);
    }

    #[test]
    fn test_layer_forward() {
        let layer = TransformerLayer::new(64, 256).unwrap();

        // Create dummy input: [4, 64]
        let mut input_data = vec![ScalarF4E4::ZERO; 4 * 64];
        for i in 0..input_data.len() {
            input_data[i] = ScalarF4E4::from((i % 10) as u8);
        }
        let input = Tensor::from_scalars(input_data, Shape::matrix(4, 64)).unwrap();

        let output = layer.forward(&input).unwrap();

        // Check output shape matches input
        assert_eq!(output.shape().dims(), &[4, 64]);
    }
}
