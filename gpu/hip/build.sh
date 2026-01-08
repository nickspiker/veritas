#!/bin/bash
# Build script for HIP kernels
#
# Compiles .hip files to shared library that Rust can link against.

set -e

echo "=== Building Spirix HIP Kernels ==="

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. Install ROCm first."
    echo "See: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    exit 1
fi

# Detect ROCm version
ROCM_VERSION=$(hipcc --version | grep -oP 'HIP version: \K[0-9.]+' || echo "unknown")
echo "ROCm/HIP version: $ROCM_VERSION"

# Detect GPU architecture
GPU_ARCH=$(rocminfo | grep -oP 'Name:\s+gfx\K[0-9]+' | head -n1 || echo "1030")
echo "Target GPU: gfx$GPU_ARCH"

# Compile HIP kernel to object file
echo "Compiling spirix_matmul.hip..."
hipcc -c spirix_matmul.hip \
    -o spirix_matmul.o \
    --offload-arch=gfx$GPU_ARCH \
    -O3 \
    -fPIC \
    -std=c++17

echo "✓ Compiled spirix_matmul.o"

# Create shared library
echo "Creating libspirix_hip.so..."
hipcc spirix_matmul.o \
    -o libspirix_hip.so \
    -shared \
    -fPIC

echo "✓ Created libspirix_hip.so"

# Verify no IEEE-754 instructions in assembly
echo ""
echo "=== Verifying Zero IEEE-754 ==="
echo "Disassembling to check for IEEE instructions..."

# Extract assembly
llvm-objdump -d spirix_matmul.o > spirix_matmul.asm || true

# Check for IEEE-754 instructions (should find ZERO)
IEEE_COUNT=$(grep -c "v_mul_f32\|v_add_f32\|v_fma_f32\|v_mad_f32" spirix_matmul.asm || echo "0")

if [ "$IEEE_COUNT" -eq 0 ]; then
    echo "✓ VERIFIED: Zero IEEE-754 instructions found"
else
    echo "WARNING: Found $IEEE_COUNT IEEE-754 instructions"
    echo "Investigate spirix_matmul.asm for details"
fi

# Check for integer instructions (should find MANY)
INT_COUNT=$(grep -c "v_mul_i32\|v_add_i32\|v_lshl_b32" spirix_matmul.asm || echo "0")
echo "✓ Found $INT_COUNT integer instructions (expected)"

echo ""
echo "=== Build Complete ==="
echo "Output: libspirix_hip.so"
echo "Assembly: spirix_matmul.asm"
echo ""
echo "Next: Run Rust test with: cargo test --release hip_matmul"
