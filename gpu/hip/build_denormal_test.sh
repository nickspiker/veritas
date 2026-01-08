#!/bin/bash
# Build HIP kernel with denormal support ENABLED
# This disables FTZ (flush-to-zero) mode

set -e

echo "=== Building IEEE f32 kernel with DENORMAL support ==="
echo ""

# Check if hipcc is available
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found. ROCm needs to be installed."
    echo ""
    echo "For this test, we need REAL ROCm/HIP, not OpenCL."
    echo "OpenCL (rusticl) always uses FTZ mode."
    echo ""
    exit 1
fi

echo "ROCm found: $(hipcc --version | head -1)"
echo ""

# Create IEEE f32 test kernel with denormals ENABLED
cat > ieee_denormal_test.hip <<'EOF'
#include <hip/hip_runtime.h>
#include <stdint.h>

// Matrix multiply kernel with FULL denormal support
// Compile with -fno-fast-math to disable FTZ
__global__ void matmul_f32_denorm(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        // Force denormal preservation with volatile
        volatile float a_val = a[row * K + k];
        volatile float b_val = b[k * N + col];
        volatile float product = a_val * b_val;
        sum += product;
    }

    c[row * N + col] = sum;
}

// Host wrapper
extern "C" void matmul_f32_denorm_hip(
    const float* h_a,
    const float* h_b,
    float* h_c,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    hipMalloc(&d_a, a_size);
    hipMalloc(&d_b, b_size);
    hipMalloc(&d_c, c_size);

    hipMemcpy(d_a, h_a, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, b_size, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(
        matmul_f32_denorm,
        gridDim, blockDim, 0, 0,
        d_a, d_b, d_c, M, N, K
    );

    hipMemcpy(h_c, d_c, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
EOF

echo "Compiling with denormal support..."
echo "Flags: -fno-fast-math -fno-finite-math-only"
echo ""

# Compile with denormal support explicitly enabled
hipcc -c ieee_denormal_test.hip \
    -o ieee_denormal_test.o \
    -O3 \
    -fPIC \
    -fno-fast-math \
    -fno-finite-math-only \
    --offload-arch=gfx1034

echo "✓ Compiled ieee_denormal_test.o"

# Create shared library
hipcc ieee_denormal_test.o \
    -o libieee_denormal.so \
    -shared \
    -fPIC

echo "✓ Created libieee_denormal.so"
echo ""

# Check assembly for denormal handling
echo "=== Checking assembly for denormal instructions ==="
llvm-objdump -d ieee_denormal_test.o > ieee_denormal_test.asm 2>/dev/null || true

# Look for denormal-related instructions
if [ -f ieee_denormal_test.asm ]; then
    echo "Assembly saved to ieee_denormal_test.asm"

    # Count float ops (should have special denormal paths if enabled)
    FLOAT_OPS=$(grep -c "v_mul_f32\|v_add_f32\|v_fma_f32" ieee_denormal_test.asm || echo "0")
    echo "Float operations found: $FLOAT_OPS"
fi

echo ""
echo "=== Build Complete ==="
echo "Now run Rust benchmark to test if denormals are preserved"
