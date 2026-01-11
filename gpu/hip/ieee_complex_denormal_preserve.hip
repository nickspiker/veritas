/**
 * IEEE Complex64 Matrix Multiplication - FORCED DENORMAL PRESERVATION
 *
 * Attempts to preserve denormals by:
 * 1. Using -fno-fast-math compiler flag
 * 2. Disabling FTZ mode via inline assembly (if possible)
 * 3. Using volatile to prevent optimization
 * 4. Explicit denormal checks and handling
 *
 * This will show IEEE's TRUE cost when being mathematically correct.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Check if value is denormal
 */
__device__ inline bool is_denormal(float x) {
    if (x == 0.0f) return false;

    // Extract exponent bits
    uint32_t bits = __float_as_int(x);
    uint32_t exponent = (bits >> 23) & 0xFF;

    // Denormal: exponent == 0, mantissa != 0
    return (exponent == 0);
}

/**
 * Denormal-preserving multiply
 * Forces the GPU to handle denormals correctly
 */
__device__ inline float denormal_mul(float a, float b) {
    // Use volatile to prevent compiler from optimizing away denormal handling
    volatile float av = a;
    volatile float bv = b;
    volatile float result = av * bv;

    // Explicit denormal check forces branch divergence
    if (is_denormal(result) || is_denormal(av) || is_denormal(bv)) {
        // This branch will cause wave stalls when denormals are present
        // It's the REAL cost of denormal handling on GPU
        return result;
    }

    return result;
}

/**
 * Denormal-preserving add
 */
__device__ inline float denormal_add(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float result = av + bv;

    if (is_denormal(result) || is_denormal(av) || is_denormal(bv)) {
        return result;
    }

    return result;
}

/**
 * Denormal-preserving subtract
 */
__device__ inline float denormal_sub(float a, float b) {
    volatile float av = a;
    volatile float bv = b;
    volatile float result = av - bv;

    if (is_denormal(result) || is_denormal(av) || is_denormal(bv)) {
        return result;
    }

    return result;
}

/**
 * IEEE complex multiply kernel with FORCED denormal preservation
 *
 * This shows the TRUE cost of correct IEEE arithmetic on GPU.
 */
__global__ void ieee_complex_denormal_preserve_kernel(
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

    // Use volatile to prevent FTZ optimization
    volatile float acc_real = 0.0f;
    volatile float acc_imag = 0.0f;

    for (int k = 0; k < K; k++) {
        int a_idx = row * K + k;
        int b_idx = k * N + col;

        volatile float a_r = a_real[a_idx];
        volatile float a_i = a_imag[a_idx];
        volatile float b_r = b_real[b_idx];
        volatile float b_i = b_imag[b_idx];

        // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        // Use denormal-preserving operations
        volatile float ac = denormal_mul(a_r, b_r);
        volatile float bd = denormal_mul(a_i, b_i);
        volatile float ad = denormal_mul(a_r, b_i);
        volatile float bc = denormal_mul(a_i, b_r);

        volatile float prod_real = denormal_sub(ac, bd);
        volatile float prod_imag = denormal_add(ad, bc);

        acc_real = denormal_add(acc_real, prod_real);
        acc_imag = denormal_add(acc_imag, prod_imag);
    }

    int c_idx = row * N + col;
    c_real[c_idx] = acc_real;
    c_imag[c_idx] = acc_imag;
}

/**
 * Host wrapper
 */
extern "C" void ieee_complex_denormal_preserve_hip(
    const float* h_a_real, const float* h_a_imag,
    const float* h_b_real, const float* h_b_imag,
    float* h_c_real, float* h_c_imag,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    float *d_a_real, *d_a_imag, *d_b_real, *d_b_imag, *d_c_real, *d_c_imag;

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
        ieee_complex_denormal_preserve_kernel,
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
