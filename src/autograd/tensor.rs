//! Tensor abstraction for Spirix
//!
//! Core design:
//! - Data stored in Spirix format (ScalarF4E4 or CircleF4E5)
//! - GPU-resident when possible
//! - Lazy evaluation for graph construction
//! - Verified arithmetic (no IEEE surprises)

use spirix::{ScalarF4E4, CircleF4E5};
use std::sync::Arc;
use crate::error::{Result, VeritasError};

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn scalar() -> Self {
        Shape { dims: vec![] }
    }

    pub fn vector(len: usize) -> Self {
        Shape { dims: vec![len] }
    }

    pub fn matrix(rows: usize, cols: usize) -> Self {
        Shape { dims: vec![rows, cols] }
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }

    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }

    pub fn rows(&self) -> Option<usize> {
        if self.is_matrix() {
            Some(self.dims[0])
        } else {
            None
        }
    }

    pub fn cols(&self) -> Option<usize> {
        if self.is_matrix() {
            Some(self.dims[1])
        } else {
            None
        }
    }
}

/// Tensor data storage
#[derive(Debug, Clone)]
pub enum TensorData {
    /// CPU-resident Spirix scalars
    CpuScalar(Vec<ScalarF4E4>),
    /// CPU-resident Circle complex numbers
    CpuComplex(Vec<CircleF4E5>),
    /// GPU-resident data (opaque handle)
    GpuScalar(GpuHandle),
    /// GPU-resident complex data
    GpuComplex(GpuHandle),
}

/// Opaque handle to GPU memory
#[derive(Debug, Clone)]
pub struct GpuHandle {
    // This will hold device pointers when we implement GPU backend
    // For now, just a placeholder
    id: usize,
}

impl GpuHandle {
    pub fn new(id: usize) -> Self {
        GpuHandle { id }
    }
}

/// Tensor with automatic differentiation
pub struct Tensor {
    /// Shape of the tensor
    pub(crate) shape: Shape,

    /// Data storage
    pub(crate) data: TensorData,

    /// Gradient (computed during backward pass)
    pub(crate) grad: Option<Box<Tensor>>,

    /// Whether this tensor requires gradient computation
    pub(crate) requires_grad: bool,
}

/// Trait for gradient computation (used internally by operations)
pub trait GradFn: Send + Sync {
    /// Compute gradients for inputs given output gradient
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor>;
}

impl Tensor {
    /// Create tensor from CPU scalar data
    pub fn from_scalars(data: Vec<ScalarF4E4>, shape: Shape) -> Result<Self> {
        if data.len() != shape.num_elements() {
            return Err(VeritasError::InvalidInput(
                format!("Data length {} doesn't match shape {:?}", data.len(), shape.dims)
            ));
        }

        Ok(Tensor {
            shape,
            data: TensorData::CpuScalar(data),
            grad: None,
            requires_grad: false,
        })
    }

    /// Create tensor from CPU complex data
    pub fn from_complex(data: Vec<CircleF4E5>, shape: Shape) -> Result<Self> {
        if data.len() != shape.num_elements() {
            return Err(VeritasError::InvalidInput(
                format!("Data length {} doesn't match shape {:?}", data.len(), shape.dims)
            ));
        }

        Ok(Tensor {
            shape,
            data: TensorData::CpuComplex(data),
            grad: None,
            requires_grad: false,
        })
    }

    /// Create tensor filled with zeros
    pub fn zeros(shape: Shape) -> Self {
        let data = vec![ScalarF4E4::ZERO; shape.num_elements()];
        Tensor {
            shape,
            data: TensorData::CpuScalar(data),
            grad: None,
            requires_grad: false,
        }
    }

    /// Create tensor filled with ones
    pub fn ones(shape: Shape) -> Self {
        let data = vec![ScalarF4E4::ONE; shape.num_elements()];
        Tensor {
            shape,
            data: TensorData::CpuScalar(data),
            grad: None,
            requires_grad: false,
        }
    }

    /// Create random tensor with Gaussian distribution (pure Spirix)
    pub fn randn(shape: Shape) -> Self {
        let data: Vec<ScalarF4E4> = (0..shape.num_elements())
            .map(|_| ScalarF4E4::random_gauss())
            .collect();

        Tensor {
            shape,
            data: TensorData::CpuScalar(data),
            grad: None,
            requires_grad: false,
        }
    }

    /// Enable gradient computation for this tensor
    pub fn with_requires_grad(mut self) -> Self {
        self.requires_grad = true;
        self
    }

    /// Get tensor shape
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Get scalar data (CPU only for now)
    pub fn as_scalars(&self) -> Option<&[ScalarF4E4]> {
        match &self.data {
            TensorData::CpuScalar(data) => Some(data),
            _ => None,
        }
    }

    /// Get mutable scalar data
    pub fn as_scalars_mut(&mut self) -> Option<&mut Vec<ScalarF4E4>> {
        match &mut self.data {
            TensorData::CpuScalar(data) => Some(data),
            _ => None,
        }
    }

    /// Get complex data (CPU only for now)
    pub fn as_complex(&self) -> Option<&[CircleF4E5]> {
        match &self.data {
            TensorData::CpuComplex(data) => Some(data),
            _ => None,
        }
    }

    /// Get gradient
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    /// Set gradient
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(Box::new(grad));
    }

    /// Accumulate gradient (for multi-path backprop)
    pub fn accumulate_grad(&mut self, grad: Tensor) -> Result<()> {
        if let Some(existing_grad) = &mut self.grad {
            // Add new gradient to existing
            if let (Some(e), Some(g)) = (existing_grad.as_scalars_mut(), grad.as_scalars()) {
                if e.len() != g.len() {
                    return Err(VeritasError::InvalidInput(
                        "Gradient shapes don't match".to_string()
                    ));
                }
                for (e_val, g_val) in e.iter_mut().zip(g.iter()) {
                    *e_val = *e_val + *g_val;
                }
            }
        } else {
            self.grad = Some(Box::new(grad));
        }
        Ok(())
    }

    /// Zero out gradients
    pub fn zero_grad(&mut self) {
        self.grad = None;
    }

    /// Check if requires gradient
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Transpose a 2D matrix
    ///
    /// For matrix [M, N], produces [N, M] by swapping rows and columns
    pub fn transpose(&self) -> Result<Tensor> {
        // Only works on 2D matrices
        if !self.shape.is_matrix() {
            return Err(VeritasError::InvalidInput(
                format!("transpose() requires 2D matrix, got {:?}", self.shape.dims)
            ));
        }

        let rows = self.shape.dims[0];
        let cols = self.shape.dims[1];

        match &self.data {
            TensorData::CpuScalar(data) => {
                let mut transposed = Vec::with_capacity(data.len());

                // For each column in original (becomes row in result)
                for col in 0..cols {
                    // For each row in original (becomes column in result)
                    for row in 0..rows {
                        let idx = row * cols + col;
                        transposed.push(data[idx]);
                    }
                }

                Ok(Tensor {
                    shape: Shape::matrix(cols, rows),
                    data: TensorData::CpuScalar(transposed),
                    grad: None,
                    requires_grad: self.requires_grad,
                })
            }
            _ => Err(VeritasError::InvalidInput(
                "transpose() only supports CpuScalar tensors currently".to_string()
            ))
        }
    }

    /// Scale tensor by a scalar value (element-wise multiplication)
    ///
    /// result[i] = self[i] * scalar
    pub fn scale(&self, scalar: ScalarF4E4) -> Result<Tensor> {
        match &self.data {
            TensorData::CpuScalar(data) => {
                let scaled: Vec<ScalarF4E4> = data.iter()
                    .map(|&val| val * scalar)
                    .collect();

                Ok(Tensor {
                    shape: self.shape.clone(),
                    data: TensorData::CpuScalar(scaled),
                    grad: None,
                    requires_grad: self.requires_grad,
                })
            }
            _ => Err(VeritasError::InvalidInput(
                "scale() only supports CpuScalar tensors currently".to_string()
            ))
        }
    }

    /// Element-wise addition of two tensors
    ///
    /// result[i] = self[i] + other[i]
    /// Tensors must have the same shape
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        // Check shape compatibility
        if self.shape != other.shape {
            return Err(VeritasError::InvalidInput(
                format!("Shape mismatch: {:?} vs {:?}", self.shape.dims, other.shape.dims)
            ));
        }

        match (&self.data, &other.data) {
            (TensorData::CpuScalar(a), TensorData::CpuScalar(b)) => {
                let sum: Vec<ScalarF4E4> = a.iter()
                    .zip(b.iter())
                    .map(|(&x, &y)| x + y)
                    .collect();

                Ok(Tensor {
                    shape: self.shape.clone(),
                    data: TensorData::CpuScalar(sum),
                    grad: None,
                    requires_grad: self.requires_grad || other.requires_grad,
                })
            }
            _ => Err(VeritasError::InvalidInput(
                "add() only supports CpuScalar tensors currently".to_string()
            ))
        }
    }

}

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor({:?}, requires_grad={})", self.shape.dims, self.requires_grad)
    }
}
