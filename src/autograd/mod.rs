//! Autograd engine built on Spirix
//!
//! Unlike IEEE-based frameworks (PyTorch, Candle), this:
//! - Preserves denormals (no FTZ violations)
//! - Uses integer ALU (8.86x faster than FP32 FPU with denormals)
//! - Maintains additive identity (x + 0 = x, always)
//! - Enables symbolic verification of gradients

pub mod tensor;
pub mod ops;
pub mod graph;
pub mod backward;
pub mod gpu;
pub mod nn;
pub mod optimizer;

pub use tensor::{Tensor, Shape};
pub use ops::{matmul, relu, add, mul, matmul_backward, relu_backward, add_backward, mul_backward};
pub use graph::ComputeGraph;
pub use gpu::GpuOps;
pub use nn::{Linear, MLP, mse_loss};
pub use optimizer::{SGD, Adam};
