#!/bin/bash
# Build and test OpenCL Spirix kernel
#
# This script compiles the OpenCL kernel and runs a simple test.

set -e

echo "=== Spirix OpenCL Build and Test ==="
echo ""

# Enable rusticl for AMD GPU
export RUSTICL_ENABLE=radeonsi

# Check if GPU is available
echo "Checking for OpenCL devices..."
clinfo -l || {
    echo "ERROR: No OpenCL devices found"
    echo "Make sure RUSTICL_ENABLE=radeonsi is set"
    exit 1
}

echo ""
echo "✓ OpenCL GPU detected"
echo ""

# The OpenCL kernel is compiled at runtime (no build step needed)
# OpenCL uses JIT compilation

echo "Building Rust OpenCL wrapper..."
cd ../..
cargo build --release --features opencl

echo ""
echo "✓ Rust code built"
echo ""

# Run benchmark
echo "Running OpenCL benchmark..."
echo ""

RUSTICL_ENABLE=radeonsi cargo run --release --features opencl --example gpu_matmul_benchmark_opencl

echo ""
echo "=== Test Complete ==="
echo ""
echo "Next steps:"
echo "  1. Check results above"
echo "  2. Phase 1 (naive) expected to be slower than CPU"
echo "  3. Implement Phase 2 optimizations (local memory, coalescing)"
echo ""
