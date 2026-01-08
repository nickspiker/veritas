//! GPU acceleration for Spirix operations
//!
//! Supports:
//! - OpenCL (Mesa rusticl) - Feature: `opencl`
//! - ROCm/HIP - External library (libspirix_hip.so)
//!
//! ZERO IEEE-754 in kernels - all integer operations.

#[cfg(feature = "opencl")]
pub mod opencl;

#[cfg(feature = "opencl")]
pub use opencl::matmul_gpu_opencl;

pub mod hip;
pub use hip::matmul_gpu;
