/**
 * IEEE Complex64 Matrix Multiplication - HIP Kernel
 *
 * Standard C++ complex using 2×f32 = 64 bits per complex number
 *
 * Complex multiplication formula:
 * (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 *
 * IEEE challenges:
 * - Requires 6 floating-point operations per complex multiply
 * - Branch divergence on NaN/Inf handling
 * - Denormal handling in all 6 operations
 * - More memory traffic (64 bits vs Circle's 48 bits)
 */

#include <hip/hip_runtime.h>

/**
 * Naive IEEE complex matrix multiply kernel
 *
 * Uses std::complex equivalent (two floats)
 */
__global__ void ieee_complex_matmul_kernel(
    const float* __restrict__ a_real,
    const float* __restrict__ a_imag,
    const float* __restrict__ b_real,
    const float* __restrict__ b_imag,
    float* __restrict__ c_real,
    float* __restrict__ c_imag,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Accumulator
    volatile float acc_real = 0.0f;
    volatile float acc_imag = 0.0f;

    // Dot product
    for (int k = 0; k < K; k++) {
        int a_idx = row * K + k;
        int b_idx = k * N + col;

        volatile float a_r = a_real[a_idx];
        volatile float a_i = a_imag[a_idx];
        volatile float b_r = b_real[b_idx];
        volatile float b_i = b_imag[b_idx];

        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // This is 6 float operations: 4 muls, 2 adds/subs
        volatile float prod_real = a_r * b_r - a_i * b_i;
        volatile float prod_imag = a_r * b_i + a_i * b_r;

        acc_real += prod_real;
        acc_imag += prod_imag;
    }

    int c_idx = row * N + col;
    c_real[c_idx] = acc_real;
    c_imag[c_idx] = acc_imag;
}

/**
 * Host wrapper
 */
extern "C" void ieee_complex_matmul_hip(
    const float* h_a_real, const float* h_a_imag,
    const float* h_b_real, const float* h_b_imag,
    float* h_c_real, float* h_c_imag,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    float *d_a_real, *d_a_imag;
    float *d_b_real, *d_b_imag;
    float *d_c_real, *d_c_imag;

    hipMalloc(&d_a_real, a_size);
    hipMalloc(&d_a_imag, a_size);
    hipMalloc(&d_b_real, b_size);
    hipMalloc(&d_b_imag, b_size);
    hipMalloc(&d_c_real, c_size);
    hipMalloc(&d_c_imag, c_size);

    hipMemcpy(d_a_real, h_a_real, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_imag, h_a_imag, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_real, h_b_real, b_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_imag, h_b_imag, b_size, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(
        ieee_complex_matmul_kernel,
        gridDim, blockDim, 0, 0,
        d_a_real, d_a_imag,
        d_b_real, d_b_imag,
        d_c_real, d_c_imag,
        M, N, K
    );

    hipMemcpy(h_c_real, d_c_real, c_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_imag, d_c_imag, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a_real); hipFree(d_a_imag);
    hipFree(d_b_real); hipFree(d_b_imag);
    hipFree(d_c_real); hipFree(d_c_imag);
}
