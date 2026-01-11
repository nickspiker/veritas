/**
 * IEEE f32 Matrix Multiplication with Denormal Support
 *
 * This kernel uses IEEE-754 f32 with denormal handling ENABLED.
 * Compile with -fno-fast-math to preserve denormals.
 *
 * Expected: This will be SLOW due to branch divergence on denormals.
 */

#include <hip/hip_runtime.h>

/**
 * Matrix multiply kernel - IEEE f32 with denormal support
 *
 * Uses volatile to force denormal preservation.
 */
__global__ void ieee_matmul_denormal_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Use volatile to force denormal preservation
    volatile float sum = 0.0f;

    for (int k = 0; k < K; k++) {
        volatile float a_val = a[row * K + k];
        volatile float b_val = b[k * N + col];
        volatile float product = a_val * b_val;
        sum += product;
    }

    c[row * N + col] = sum;
}

/**
 * Host wrapper
 */
extern "C" void ieee_matmul_denormal_hip(
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
        ieee_matmul_denormal_kernel,
        gridDim, blockDim, 0, 0,
        d_a, d_b, d_c, M, N, K
    );

    hipMemcpy(h_c, d_c, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
