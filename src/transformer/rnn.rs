//! Simple RNN for routing proof-of-concept
//!
//! Minimal architecture:
//! - Single hidden layer (128 units)
//! - Byte-level input (256 possible values)
//! - Next-byte prediction output (256 classes)
//!
//! Goal: Learn to output <MATH_N> tokens when math is detected

use crate::autograd::{Tensor, Shape, matmul};
use crate::error::Result;
use spirix::ScalarF4E4;

/// Simple RNN configuration
#[derive(Debug, Clone)]
pub struct RNNConfig {
    pub input_size: usize,    // 256 (byte vocabulary)
    pub hidden_size: usize,   // 128 (hidden units)
    pub output_size: usize,   // 256 (next byte prediction)
}

impl Default for RNNConfig {
    fn default() -> Self {
        Self {
            input_size: 256,
            hidden_size: 128,
            output_size: 256,
        }
    }
}

/// Simple RNN model
///
/// Architecture:
/// ```text
/// byte[t] → embedding → hidden[t] → output[t]
///                          ↑
///                       hidden[t-1]
/// ```
pub struct SimpleRNN {
    /// Input to hidden: [256, 128]
    pub w_ih: Tensor,

    /// Hidden to hidden: [128, 128]
    pub w_hh: Tensor,

    /// Hidden to output: [128, 256]
    pub w_ho: Tensor,

    /// Hidden state: [128]
    pub hidden: Tensor,

    /// Configuration
    pub config: RNNConfig,
}

impl SimpleRNN {
    /// Create new RNN with random initialization
    pub fn new(config: RNNConfig) -> Result<Self> {
        // Xavier initialization for input-to-hidden
        let w_ih = init_xavier(config.input_size, config.hidden_size)?;

        // Xavier initialization for hidden-to-hidden
        let w_hh = init_xavier(config.hidden_size, config.hidden_size)?;

        // Xavier initialization for hidden-to-output
        let w_ho = init_xavier(config.hidden_size, config.output_size)?;

        // Initialize hidden state to zeros
        let hidden = Tensor::zeros(Shape::vector(config.hidden_size));

        Ok(Self {
            w_ih,
            w_hh,
            w_ho,
            hidden,
            config,
        })
    }

    /// Reset hidden state to zeros
    pub fn reset_hidden(&mut self) {
        self.hidden = Tensor::zeros(Shape::vector(self.config.hidden_size));
    }

    /// Forward pass: single byte → next byte logits
    ///
    /// Updates internal hidden state
    pub fn step(&mut self, byte: u8) -> Result<Vec<ScalarF4E4>> {
        // Create one-hot encoding for input byte: [256]
        let input = one_hot_encode(byte, self.config.input_size)?;

        // Reshape vectors to [1, N] for matmul
        let input_2d = reshape_to_matrix(&input, 1, self.config.input_size)?;
        let hidden_2d = reshape_to_matrix(&self.hidden, 1, self.config.hidden_size)?;

        // hidden = tanh(W_ih * input + W_hh * hidden_prev)
        // matmul: [1, 256] x [256, 128] = [1, 128]
        let ih_contrib = matmul(&input_2d, &self.w_ih)?;
        // matmul: [1, 128] x [128, 128] = [1, 128]
        let hh_contrib = matmul(&hidden_2d, &self.w_hh)?;
        let hidden_next = ih_contrib.add(&hh_contrib)?;
        let hidden_activated = tanh_approx(&hidden_next)?;

        // output = W_ho * hidden
        // matmul: [1, 128] x [128, 256] = [1, 256]
        let output = matmul(&hidden_activated, &self.w_ho)?;

        // Update hidden state - flatten back to vector [128]
        self.hidden = flatten_to_vector(&hidden_activated)?;

        // Return output logits - flatten from [1, 256] to vec
        output.as_scalars()
            .ok_or_else(|| crate::error::VeritasError::InvalidInput(
                "Output must be CpuScalar".to_string()
            ))
            .map(|s| s.to_vec())
    }

    /// Process sequence of bytes, return logits for each position
    ///
    /// Resets hidden state at start of sequence
    pub fn forward(&mut self, bytes: &[u8]) -> Result<Vec<Vec<ScalarF4E4>>> {
        self.reset_hidden();

        let mut outputs = Vec::with_capacity(bytes.len());

        for &byte in bytes {
            let logits = self.step(byte)?;
            outputs.push(logits);
        }

        Ok(outputs)
    }

    /// Predict next byte given sequence
    ///
    /// Returns argmax of output logits
    pub fn predict_next(&mut self, bytes: &[u8]) -> Result<u8> {
        self.reset_hidden();

        let mut last_logits = vec![ScalarF4E4::ZERO; self.config.output_size];

        for &byte in bytes {
            last_logits = self.step(byte)?;
        }

        // Find argmax
        let mut max_idx = 0;
        let mut max_val = last_logits[0];

        for (i, &val) in last_logits.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }

        Ok(max_idx as u8)
    }

    /// Get model parameters (for optimizer)
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.w_ih, &mut self.w_hh, &mut self.w_ho]
    }
}

/// Initialize weight matrix with Xavier initialization (pure Spirix)
///
/// Scale: 1 / sqrt(fan_in + fan_out)
fn init_xavier(rows: usize, cols: usize) -> Result<Tensor> {
    // Xavier scale in pure Spirix
    let fan_sum = ScalarF4E4::from((rows + cols) as u32);
    let scale = (ScalarF4E4::ONE / fan_sum).sqrt();

    let data: Vec<ScalarF4E4> = (0..(rows * cols))
        .map(|_| ScalarF4E4::random_gauss() * scale)
        .collect();

    Tensor::from_scalars(data, Shape::matrix(rows, cols))
        .map(|t| t.with_requires_grad())
}

/// Create one-hot encoding for byte value
///
/// Example: byte=65 ('A') → [0, 0, ..., 1, ..., 0] (256 elements)
fn one_hot_encode(byte: u8, size: usize) -> Result<Tensor> {
    let mut data = vec![ScalarF4E4::ZERO; size];
    data[byte as usize] = ScalarF4E4::ONE;

    Tensor::from_scalars(data, Shape::vector(size))
}

/// Approximate tanh activation
///
/// True tanh requires exp() which is expensive.
/// Use polynomial approximation: tanh(x) ≈ x / (1 + |x|)
fn tanh_approx(x: &Tensor) -> Result<Tensor> {
    let data = x.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Input must be CpuScalar".to_string()
        ))?;

    let activated: Vec<ScalarF4E4> = data.iter()
        .map(|&val| {
            // tanh(x) ≈ x / (1 + |x|)
            let abs_val = if val < ScalarF4E4::ZERO {
                ScalarF4E4::ZERO - val
            } else {
                val
            };
            let denom = ScalarF4E4::ONE + abs_val;
            val / denom
        })
        .collect();

    Tensor::from_scalars(activated, x.shape().clone())
}

/// Reshape vector [N] to matrix [rows, cols]
fn reshape_to_matrix(vec: &Tensor, rows: usize, cols: usize) -> Result<Tensor> {
    let data = vec.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Tensor must be CpuScalar".to_string()
        ))?;

    if data.len() != rows * cols {
        return Err(crate::error::VeritasError::InvalidInput(
            format!("Size mismatch: {} elements cannot reshape to {}x{}", data.len(), rows, cols)
        ));
    }

    Tensor::from_scalars(data.to_vec(), Shape::matrix(rows, cols))
}

/// Flatten matrix [1, N] or [N, 1] to vector [N]
fn flatten_to_vector(mat: &Tensor) -> Result<Tensor> {
    let data = mat.as_scalars()
        .ok_or_else(|| crate::error::VeritasError::InvalidInput(
            "Tensor must be CpuScalar".to_string()
        ))?;

    let shape = mat.shape();
    let dims = shape.dims();

    // Verify it's a matrix with one dimension being 1
    if dims.len() != 2 || (dims[0] != 1 && dims[1] != 1) {
        return Err(crate::error::VeritasError::InvalidInput(
            format!("Cannot flatten shape {:?} to vector", dims)
        ));
    }

    let size = data.len();
    Tensor::from_scalars(data.to_vec(), Shape::vector(size))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rnn_creation() {
        let config = RNNConfig::default();
        let rnn = SimpleRNN::new(config).unwrap();
        assert_eq!(rnn.config.hidden_size, 128);
    }

    #[test]
    fn test_rnn_step() {
        let config = RNNConfig::default();
        let mut rnn = SimpleRNN::new(config).unwrap();

        let byte = b'H';
        let logits = rnn.step(byte).unwrap();

        // Should return 256 logits (one per byte)
        assert_eq!(logits.len(), 256);
    }

    #[test]
    fn test_rnn_forward() {
        let config = RNNConfig::default();
        let mut rnn = SimpleRNN::new(config).unwrap();

        let bytes = b"Hello";
        let outputs = rnn.forward(bytes).unwrap();

        // Should have one output per input byte
        assert_eq!(outputs.len(), 5);
        assert_eq!(outputs[0].len(), 256);
    }

    #[test]
    fn test_predict_next() {
        let config = RNNConfig::default();
        let mut rnn = SimpleRNN::new(config).unwrap();

        let bytes = b"2 + ";
        let next_byte = rnn.predict_next(bytes).unwrap();

        // Should return some byte (random initialization)
        assert!(next_byte < 256);
    }
}
