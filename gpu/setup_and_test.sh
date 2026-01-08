#!/bin/bash
# Quick setup and test script for Spirix GPU kernels
#
# Builds HIP library and runs benchmark

set -e

echo "=== Spirix GPU Setup and Test ==="
echo ""

# Check if ROCm is installed
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: ROCm/HIP not found"
    echo ""
    echo "Install ROCm:"
    echo "  Fedora: sudo dnf install rocm-hip-sdk"
    echo "  Ubuntu: Follow https://rocm.docs.amd.com/"
    echo ""
    exit 1
fi

echo "✓ ROCm found: $(hipcc --version | head -1)"
echo ""

# Build HIP kernels
echo "Building HIP kernels..."
cd hip
./build.sh

if [ ! -f libspirix_hip.so ]; then
    echo "ERROR: Failed to build libspirix_hip.so"
    exit 1
fi

echo ""
echo "✓ HIP library built"
echo ""

# Go back to project root
cd ../..

# Build Rust code
echo "Building Rust code..."
cargo build --release --example gpu_matmul_benchmark

echo ""
echo "✓ Rust code built"
echo ""

# Run benchmark
echo "Running benchmark..."
echo ""

export LD_LIBRARY_PATH=$PWD/gpu/hip:$LD_LIBRARY_PATH
cargo run --release --example gpu_matmul_benchmark

echo ""
echo "=== Test Complete ==="
echo ""
echo "Next steps:"
echo "  1. Check results above"
echo "  2. Phase 1 (naive) expected to be slower than CPU"
echo "  3. Implement Phase 2 optimizations (shared memory, coalescing)"
echo "  4. Compare against rocBLAS f32 for IEEE-754 baseline"
echo ""
