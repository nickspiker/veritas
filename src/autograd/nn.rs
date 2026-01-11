//! Neural network layers built on Spirix
//!
//! All verified arithmetic, GPU-accelerated where beneficial

use super::tensor::{Tensor, Shape};
use super::ops::{matmul, relu, add};
use super::gpu::GpuOps;
use spirix::ScalarF4E4;
use crate::error::Result;

/// Linear layer: y = Wx + b
pub struct Linear {
    /// Weight matrix
    pub weight: Tensor,
    /// Bias vector
    pub bias: Tensor,
    /// Use GPU acceleration
    use_gpu: bool,
}

impl Linear {
    /// Create new linear layer with random initialization
    pub fn new(in_features: usize, out_features: usize, use_gpu: bool) -> Self {
        // Xavier initialization: scale by sqrt(1/in_features) in pure Spirix
        let in_feat_spirix = ScalarF4E4::from(in_features as u32);
        let scale = (ScalarF4E4::ONE / in_feat_spirix).sqrt();

        let weight_data: Vec<ScalarF4E4> = (0..(in_features * out_features))
            .map(|_| ScalarF4E4::random_gauss() * scale)
            .collect();

        let bias_data: Vec<ScalarF4E4> = vec![ScalarF4E4::ZERO; out_features];

        Linear {
            weight: Tensor::from_scalars(weight_data, Shape::matrix(out_features, in_features))
                .unwrap()
                .with_requires_grad(),
            bias: Tensor::from_scalars(bias_data, Shape::vector(out_features))
                .unwrap()
                .with_requires_grad(),
            use_gpu,
        }
    }

    /// Forward pass: y = Wx + b
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // y = Wx
        let y = if self.use_gpu {
            GpuOps::matmul(&self.weight, x)?
        } else {
            matmul(&self.weight, x)?
        };

        // y = y + b (broadcast bias)
        // TODO: Implement proper broadcasting
        Ok(y)
    }

    /// Get parameters for optimization
    pub fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    /// Get mutable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Simple feedforward network
pub struct MLP {
    layers: Vec<Linear>,
}

impl MLP {
    /// Create multi-layer perceptron
    pub fn new(layer_sizes: &[usize], use_gpu: bool) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(Linear::new(layer_sizes[i], layer_sizes[i + 1], use_gpu));
        }

        MLP { layers }
    }

    /// Forward pass through all layers
    pub fn forward(&self, mut x: Tensor) -> Result<Tensor> {
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x)?;

            // ReLU activation on all but last layer
            if i < self.layers.len() - 1 {
                x = relu(&x)?;
            }
        }

        Ok(x)
    }

    /// Get all parameters
    pub fn parameters(&self) -> Vec<&Tensor> {
        self.layers.iter()
            .flat_map(|l| l.parameters())
            .collect()
    }

    /// Get all mutable parameters
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers.iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }
}

/// Mean squared error loss
pub fn mse_loss(predicted: &Tensor, target: &Tensor) -> Result<ScalarF4E4> {
    let pred_data = predicted.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Predicted must be scalar tensor".to_string()
        ))?;

    let target_data = target.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Target must be scalar tensor".to_string()
        ))?;

    if pred_data.len() != target_data.len() {
        return Err(crate::error::VeritasError::InvalidInput(
            "Predicted and target must have same size".to_string()
        ));
    }

    // MSE = mean((pred - target)^2)
    let mut sum = ScalarF4E4::ZERO;
    for (p, t) in pred_data.iter().zip(target_data.iter()) {
        let diff = *p - *t;
        sum = sum + (diff * diff);
    }

    let mean = sum / ScalarF4E4::from(pred_data.len() as u32);
    Ok(mean)
}
