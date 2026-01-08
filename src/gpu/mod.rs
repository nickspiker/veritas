//! GPU acceleration for Spirix operations
//!
//! Uses ROCm/HIP for AMD GPUs.
//! ZERO IEEE-754 in kernels - all integer operations.

pub mod hip;

pub use hip::matmul_gpu;
