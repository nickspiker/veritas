/**
 * Circle Complex Matrix Multiplication - HIP Kernel
 *
 * CircleF4E4: i16 real + i16 imaginary + i16 shared exponent = 48 bits
 * Compare to IEEE complex64: 2×f32 = 64 bits
 *
 * Complex multiplication formula:
 * (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 *
 * Circle advantages:
 * - Shared exponent for both components
 * - Pure integer arithmetic (zero branch divergence)
 * - Simpler normalization than IEEE complex multiply
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Circle complex multiply device function
 *
 * Implements (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 * with shared exponent handling
 */
__device__ inline void circle_mul(
    int16_t a_real, int16_t a_imag, int16_t a_exp,
    int16_t b_real, int16_t b_imag, int16_t b_exp,
    int16_t* c_real, int16_t* c_imag, int16_t* c_exp
) {
    // Check for zero inputs
    if ((a_real == 0 && a_imag == 0) || (b_real == 0 && b_imag == 0)) {
        *c_real = 0;
        *c_imag = 0;
        *c_exp = 0;
        return;
    }

    // Widen to 32-bit for multiplication
    int32_t a_r = (int32_t)a_real;
    int32_t a_i = (int32_t)a_imag;
    int32_t b_r = (int32_t)b_real;
    int32_t b_i = (int32_t)b_imag;

    // Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    // Divide by 2 (>>1) to prevent overflow in intermediate products
    int32_t real_product = (a_r * b_r >> 1) - (a_i * b_i >> 1);
    int32_t imag_product = (a_r * b_i >> 1) + (a_i * b_r >> 1);

    if (real_product == 0 && imag_product == 0) {
        *c_real = 0;
        *c_imag = 0;
        *c_exp = 0;
        return;
    }

    // Find leading zeros/ones for normalization
    // For negative numbers, count leading ones; for positive, count leading zeros
    int32_t leading_r = real_product < 0 ? __clz(~real_product) : __clz(real_product);
    int32_t leading_i = imag_product < 0 ? __clz(~imag_product) : __clz(imag_product);

    // Use minimum shift to keep both components normalized
    int32_t shift = (leading_r < leading_i ? leading_r : leading_i) - 3;
    if (shift < 0) shift = 0;

    int32_t shift_amount = shift + 2;

    // Normalize
    int32_t normalized_real = real_product << shift_amount;
    int32_t normalized_imag = imag_product << shift_amount;

    // Extract high 16 bits
    *c_real = (int16_t)(normalized_real >> 16);
    *c_imag = (int16_t)(normalized_imag >> 16);

    // Compute exponent: exp_a + exp_b - normalization_shift
    int32_t expo_adjust = shift - 3;
    int32_t exp_sum = (int32_t)a_exp + (int32_t)b_exp - expo_adjust;

    // Clamp exponent to valid range
    if (exp_sum > 32767) exp_sum = 32767;
    if (exp_sum < -32768) exp_sum = -32768;

    *c_exp = (int16_t)exp_sum;
}

/**
 * Circle complex add device function
 *
 * Adds two complex numbers with shared exponents
 */
__device__ inline void circle_add(
    int16_t a_real, int16_t a_imag, int16_t a_exp,
    int16_t b_real, int16_t b_imag, int16_t b_exp,
    int16_t* c_real, int16_t* c_imag, int16_t* c_exp
) {
    // Zero checks
    if (a_real == 0 && a_imag == 0) {
        *c_real = b_real;
        *c_imag = b_imag;
        *c_exp = b_exp;
        return;
    }
    if (b_real == 0 && b_imag == 0) {
        *c_real = a_real;
        *c_imag = a_imag;
        *c_exp = a_exp;
        return;
    }

    // Align exponents
    if (a_exp > b_exp) {
        int shift = a_exp - b_exp;
        if (shift > 15) {
            *c_real = a_real;
            *c_imag = a_imag;
            *c_exp = a_exp;
            return;
        }
        *c_real = a_real + (b_real >> shift);
        *c_imag = a_imag + (b_imag >> shift);
        *c_exp = a_exp;
    } else if (b_exp > a_exp) {
        int shift = b_exp - a_exp;
        if (shift > 15) {
            *c_real = b_real;
            *c_imag = b_imag;
            *c_exp = b_exp;
            return;
        }
        *c_real = (a_real >> shift) + b_real;
        *c_imag = (a_imag >> shift) + b_imag;
        *c_exp = b_exp;
    } else {
        *c_real = a_real + b_real;
        *c_imag = a_imag + b_imag;
        *c_exp = a_exp;
    }
}

/**
 * Naive Circle complex matrix multiply kernel
 *
 * Computes C = A × B where A, B, C are matrices of complex numbers
 * Each complex number is represented as (real, imag, exponent)
 */
__global__ void circle_matmul_kernel(
    const int16_t* __restrict__ a_real,
    const int16_t* __restrict__ a_imag,
    const int16_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_real,
    const int16_t* __restrict__ b_imag,
    const int16_t* __restrict__ b_exp,
    int16_t* __restrict__ c_real,
    int16_t* __restrict__ c_imag,
    int16_t* __restrict__ c_exp,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Accumulator for this output element
    int16_t acc_real = 0;
    int16_t acc_imag = 0;
    int16_t acc_exp = 0;

    // Dot product of row from A with column from B
    for (int k = 0; k < K; k++) {
        int a_idx = row * K + k;
        int b_idx = k * N + col;

        // Load A[row,k] and B[k,col]
        int16_t a_r = a_real[a_idx];
        int16_t a_i = a_imag[a_idx];
        int16_t a_e = a_exp[a_idx];

        int16_t b_r = b_real[b_idx];
        int16_t b_i = b_imag[b_idx];
        int16_t b_e = b_exp[b_idx];

        // Skip if either is zero
        if ((a_r == 0 && a_i == 0) || (b_r == 0 && b_i == 0)) {
            continue;
        }

        // Multiply: prod = A[row,k] * B[k,col]
        int16_t prod_real, prod_imag, prod_exp;
        circle_mul(a_r, a_i, a_e, b_r, b_i, b_e, &prod_real, &prod_imag, &prod_exp);

        // Add to accumulator: acc += prod
        int16_t new_real, new_imag, new_exp;
        circle_add(acc_real, acc_imag, acc_exp, prod_real, prod_imag, prod_exp,
                   &new_real, &new_imag, &new_exp);

        acc_real = new_real;
        acc_imag = new_imag;
        acc_exp = new_exp;
    }

    // Write result
    int c_idx = row * N + col;
    c_real[c_idx] = acc_real;
    c_imag[c_idx] = acc_imag;
    c_exp[c_idx] = acc_exp;
}

/**
 * Host wrapper function
 */
extern "C" void circle_matmul_hip(
    const int16_t* h_a_real, const int16_t* h_a_imag, const int16_t* h_a_exp,
    const int16_t* h_b_real, const int16_t* h_b_imag, const int16_t* h_b_exp,
    int16_t* h_c_real, int16_t* h_c_imag, int16_t* h_c_exp,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(int16_t);
    size_t b_size = K * N * sizeof(int16_t);
    size_t c_size = M * N * sizeof(int16_t);

    int16_t *d_a_real, *d_a_imag, *d_a_exp;
    int16_t *d_b_real, *d_b_imag, *d_b_exp;
    int16_t *d_c_real, *d_c_imag, *d_c_exp;

    hipMalloc(&d_a_real, a_size);
    hipMalloc(&d_a_imag, a_size);
    hipMalloc(&d_a_exp, a_size);
    hipMalloc(&d_b_real, b_size);
    hipMalloc(&d_b_imag, b_size);
    hipMalloc(&d_b_exp, b_size);
    hipMalloc(&d_c_real, c_size);
    hipMalloc(&d_c_imag, c_size);
    hipMalloc(&d_c_exp, c_size);

    hipMemcpy(d_a_real, h_a_real, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_imag, h_a_imag, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_real, h_b_real, b_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_imag, h_b_imag, b_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, b_size, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(
        circle_matmul_kernel,
        gridDim, blockDim, 0, 0,
        d_a_real, d_a_imag, d_a_exp,
        d_b_real, d_b_imag, d_b_exp,
        d_c_real, d_c_imag, d_c_exp,
        M, N, K
    );

    hipMemcpy(h_c_real, d_c_real, c_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_imag, d_c_imag, c_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_exp, d_c_exp, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a_real); hipFree(d_a_imag); hipFree(d_a_exp);
    hipFree(d_b_real); hipFree(d_b_imag); hipFree(d_b_exp);
    hipFree(d_c_real); hipFree(d_c_imag); hipFree(d_c_exp);
}
