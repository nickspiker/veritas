//! Tensor operations with automatic differentiation
//!
//! Each op has:
//! - Forward pass: Compute output
//! - Backward pass: Compute gradients
//! - GPU acceleration: Use verified kernels

use super::tensor::{Tensor, Shape, GradFn, TensorData};
use spirix::ScalarF4E4;
use std::sync::Arc;
use crate::error::{Result, VeritasError};

// ============================================================================
// MATRIX MULTIPLY
// ============================================================================

/// Matrix multiply: C = A @ B
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Validate shapes
    let a_shape = a.shape();
    let b_shape = b.shape();

    if !a_shape.is_matrix() || !b_shape.is_matrix() {
        return Err(VeritasError::InvalidInput(
            "matmul requires 2D tensors".to_string()
        ));
    }

    let m = a_shape.rows().unwrap();
    let k1 = a_shape.cols().unwrap();
    let k2 = b_shape.rows().unwrap();
    let n = b_shape.cols().unwrap();

    if k1 != k2 {
        return Err(VeritasError::InvalidInput(
            format!("Inner dimensions don't match: {} vs {}", k1, k2)
        ));
    }

    // CPU implementation for now
    let a_data = a.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let b_data = b.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    let mut c_data = vec![ScalarF4E4::ZERO; m * n];

    // Naive matmul (TODO: use GPU kernel)
    for i in 0..m {
        for j in 0..n {
            let mut sum = ScalarF4E4::ZERO;
            for k in 0..k1 {
                let a_val = a_data[i * k1 + k];
                let b_val = b_data[k * n + j];
                sum = sum + (a_val * b_val);
            }
            c_data[i * n + j] = sum;
        }
    }

    let result = Tensor::from_scalars(c_data, Shape::matrix(m, n))?;
    Ok(result)
}

/// Compute gradients for matrix multiply
///
/// Given grad_output (gradient of loss w.r.t. output C),
/// computes gradients w.r.t. inputs A and B.
///
/// Returns: (grad_a, grad_b)
pub fn matmul_backward(
    grad_output: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let a_shape = a.shape();
    let b_shape = b.shape();

    if !a_shape.is_matrix() || !b_shape.is_matrix() {
        return Err(VeritasError::InvalidInput(
            "matmul_backward requires 2D tensors".to_string()
        ));
    }

    let m = a_shape.rows().unwrap();
    let k = a_shape.cols().unwrap();
    let n = b_shape.cols().unwrap();

    let grad_out = grad_output.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let a_data = a.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let b_data = b.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    // Compute dL/dA = grad_output @ B^T
    let mut grad_a = vec![ScalarF4E4::ZERO; m * k];
    for i in 0..m {
        for j in 0..k {
            let mut sum = ScalarF4E4::ZERO;
            for l in 0..n {
                let grad_val = grad_out[i * n + l];
                let b_val = b_data[j * n + l];  // B^T
                sum = sum + (grad_val * b_val);
            }
            grad_a[i * k + j] = sum;
        }
    }

    // Compute dL/dB = A^T @ grad_output
    let mut grad_b = vec![ScalarF4E4::ZERO; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut sum = ScalarF4E4::ZERO;
            for l in 0..m {
                let a_val = a_data[l * k + i];  // A^T
                let grad_val = grad_out[l * n + j];
                sum = sum + (a_val * grad_val);
            }
            grad_b[i * n + j] = sum;
        }
    }

    Ok((
        Tensor::from_scalars(grad_a, a_shape.clone())?,
        Tensor::from_scalars(grad_b, b_shape.clone())?,
    ))
}

// ============================================================================
// RELU ACTIVATION
// ============================================================================

/// ReLU activation: f(x) = max(0, x)
pub fn relu(x: &Tensor) -> Result<Tensor> {
    let data = x.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    let output: Vec<ScalarF4E4> = data.iter()
        .map(|&v| if v > ScalarF4E4::ZERO { v } else { ScalarF4E4::ZERO })
        .collect();

    let result = Tensor::from_scalars(output, x.shape().clone())?;
    Ok(result)
}

/// Compute gradients for ReLU activation
///
/// Given grad_output and the input to ReLU,
/// computes gradient w.r.t. input.
///
/// ReLU derivative: 1 if x > 0, else 0
pub fn relu_backward(
    grad_output: &Tensor,
    x: &Tensor,
) -> Result<Tensor> {
    let grad_out = grad_output.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let x_data = x.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    let grad_input: Vec<ScalarF4E4> = x_data.iter()
        .zip(grad_out.iter())
        .map(|(&x, &grad)| {
            if x > ScalarF4E4::ZERO {
                grad
            } else {
                ScalarF4E4::ZERO
            }
        })
        .collect();

    Tensor::from_scalars(grad_input, x.shape().clone())
}

// ============================================================================
// ELEMENTWISE ADD
// ============================================================================

/// Elementwise addition: C = A + B
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(VeritasError::InvalidInput(
            "Shapes must match for elementwise add".to_string()
        ));
    }

    let a_data = a.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let b_data = b.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    let output: Vec<ScalarF4E4> = a_data.iter()
        .zip(b_data.iter())
        .map(|(&a, &b)| a + b)
        .collect();

    let result = Tensor::from_scalars(output, a.shape().clone())?;
    Ok(result)
}

/// Compute gradients for elementwise addition
///
/// Gradient flows equally to both inputs
pub fn add_backward(
    grad_output: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // Both inputs get the same gradient
    Ok((grad_output.clone(), grad_output.clone()))
}

// ============================================================================
// ELEMENTWISE MULTIPLY
// ============================================================================

/// Elementwise multiplication: C = A * B
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(VeritasError::InvalidInput(
            "Shapes must match for elementwise mul".to_string()
        ));
    }

    let a_data = a.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let b_data = b.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    let output: Vec<ScalarF4E4> = a_data.iter()
        .zip(b_data.iter())
        .map(|(&a, &b)| a * b)
        .collect();

    let result = Tensor::from_scalars(output, a.shape().clone())?;
    Ok(result)
}

/// Compute gradients for elementwise multiplication
///
/// dL/dA = grad_output * B
/// dL/dB = grad_output * A
pub fn mul_backward(
    grad_output: &Tensor,
    a: &Tensor,
    b: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let grad_out = grad_output.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let a_data = a.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;
    let b_data = b.as_scalars().ok_or_else(||
        VeritasError::InvalidInput("Expected scalar data".to_string()))?;

    // dL/dA = grad_output * B
    let grad_a: Vec<ScalarF4E4> = grad_out.iter()
        .zip(b_data.iter())
        .map(|(&g, &b)| g * b)
        .collect();

    // dL/dB = grad_output * A
    let grad_b: Vec<ScalarF4E4> = grad_out.iter()
        .zip(a_data.iter())
        .map(|(&g, &a)| g * a)
        .collect();

    Ok((
        Tensor::from_scalars(grad_a, a.shape().clone())?,
        Tensor::from_scalars(grad_b, b.shape().clone())?,
    ))
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            shape: self.shape.clone(),
            data: self.data.clone(),
            grad: None,  // Don't clone gradients
            requires_grad: self.requires_grad,
        }
    }
}
