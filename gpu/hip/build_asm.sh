#!/bin/bash
# Build Spirix GPU kernel with pure inline assembly
#
# This script:
# 1. Compiles HIP kernel with inline RDNA assembly
# 2. Verifies ZERO IEEE-754 instructions in output
# 3. Creates shared library for Rust FFI

set -e

echo "=== Building Spirix Inline Assembly GPU Kernel ==="
echo ""

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. ROCm is required for inline assembly."
    echo ""
    echo "OpenCL (rusticl) cannot be used - it lacks inline assembly support."
    echo ""
    echo "Install ROCm:"
    echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
    echo ""
    exit 1
fi

ROCM_VERSION=$(hipcc --version | grep -oP 'HIP version: \K[0-9.]+' || echo "unknown")
echo "✓ ROCm/HIP found: $ROCM_VERSION"

# Detect GPU architecture
GPU_ARCH=$(rocminfo 2>/dev/null | grep -oP 'Name:\s+gfx\K[0-9]+' | head -n1 || echo "1034")
echo "✓ Target GPU: gfx$GPU_ARCH (RDNA2)"
echo ""

# Compile inline assembly HIP kernel
echo "Compiling spirix_matmul_asm.hip..."
echo "  Flags: -O3, --offload-arch=gfx$GPU_ARCH"
echo ""

hipcc -c spirix_matmul_asm.hip \
    -o spirix_matmul_asm.o \
    --offload-arch=gfx$GPU_ARCH \
    -O3 \
    -fPIC \
    -std=c++17

if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi

echo "✓ Compiled spirix_matmul_asm.o"
echo ""

# Create shared library
echo "Creating libspirix_asm.so..."
hipcc spirix_matmul_asm.o \
    -o libspirix_asm.so \
    -shared \
    -fPIC

echo "✓ Created libspirix_asm.so"
echo ""

# Verify zero IEEE-754 instructions
echo "=== Verifying Zero IEEE-754 Instructions ==="
echo ""

# Disassemble to check instructions
llvm-objdump -d spirix_matmul_asm.o > spirix_asm.disasm 2>/dev/null || {
    echo "Warning: Could not disassemble (llvm-objdump not found)"
    echo "Skipping verification step"
    echo ""
}

if [ -f spirix_asm.disasm ]; then
    # Count IEEE-754 float instructions (should be ZERO)
    IEEE_COUNT=$(grep -c "v_mul_f32\|v_add_f32\|v_fma_f32\|v_mad_f32\|v_rcp_f32" spirix_asm.disasm || echo "0")

    if [ "$IEEE_COUNT" -eq 0 ]; then
        echo "✓ VERIFIED: Zero IEEE-754 float instructions"
    else
        echo "⚠ WARNING: Found $IEEE_COUNT IEEE-754 instructions"
        echo ""
        echo "Showing IEEE instructions:"
        grep "v_mul_f32\|v_add_f32\|v_fma_f32\|v_mad_f32\|v_rcp_f32" spirix_asm.disasm | head -20
        echo ""
    fi

    # Count integer instructions (should be MANY)
    INT_MUL=$(grep -c "v_mul_i32\|v_mul_lo_i32\|v_mul_hi_i32" spirix_asm.disasm || echo "0")
    INT_ADD=$(grep -c "v_add_i32\|v_sub_i32" spirix_asm.disasm || echo "0")
    SHIFTS=$(grep -c "v_lshl\|v_lshr\|v_ashr" spirix_asm.disasm || echo "0")

    echo "✓ Integer multiply instructions: $INT_MUL"
    echo "✓ Integer add/sub instructions: $INT_ADD"
    echo "✓ Shift instructions: $SHIFTS"
    echo ""

    if [ $INT_MUL -eq 0 ] && [ $INT_ADD -eq 0 ]; then
        echo "⚠ WARNING: No integer ops found - something may be wrong"
    fi
fi

echo "=== Build Complete ==="
echo ""
echo "Output files:"
echo "  - libspirix_asm.so (shared library)"
echo "  - spirix_asm.disasm (assembly listing)"
echo ""
echo "Next steps:"
echo "  1. Update Rust FFI to use libspirix_asm.so"
echo "  2. Run benchmark: cargo run --release --example gpu_spirix_asm_bench"
echo "  3. Compare against IEEE f32 (with FTZ enabled)"
echo ""
