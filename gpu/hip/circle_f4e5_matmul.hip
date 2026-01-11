/**
 * Circle F4E5 Complex Matrix Multiplication - HIP Kernel
 *
 * CircleF4E5: i16 real + i16 imaginary + i32 exponent = 64 bits
 * Compare to IEEE complex64: f32 real + f32 imaginary = 64 bits
 *
 * TRUE APPLES-TO-APPLES COMPARISON: Same 64 bits per complex number
 *
 * Complex multiplication formula:
 * (a+bi)(c+di) = (ac-bd) + (ad+bc)i
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Circle F4E5 complex multiply device function
 *
 * Uses i32 exponent for extended range (same total bits as IEEE)
 */
__device__ inline void circle_f4e5_mul(
    int16_t a_real, int16_t a_imag, int32_t a_exp,
    int16_t b_real, int16_t b_imag, int32_t b_exp,
    int16_t* c_real, int16_t* c_imag, int32_t* c_exp
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
    int32_t real_product = (a_r * b_r >> 1) - (a_i * b_i >> 1);
    int32_t imag_product = (a_r * b_i >> 1) + (a_i * b_r >> 1);

    if (real_product == 0 && imag_product == 0) {
        *c_real = 0;
        *c_imag = 0;
        *c_exp = 0;
        return;
    }

    // Find leading zeros/ones for normalization
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

    // Compute exponent with i32 range: exp_a + exp_b - normalization_shift
    int32_t expo_adjust = shift - 3;
    int64_t exp_sum = (int64_t)a_exp + (int64_t)b_exp - expo_adjust;

    // Clamp exponent to i32 range
    if (exp_sum > 2147483647LL) exp_sum = 2147483647LL;
    if (exp_sum < -2147483648LL) exp_sum = -2147483648LL;

    *c_exp = (int32_t)exp_sum;
}

/**
 * Circle F4E5 complex add device function
 */
__device__ inline void circle_f4e5_add(
    int16_t a_real, int16_t a_imag, int32_t a_exp,
    int16_t b_real, int16_t b_imag, int32_t b_exp,
    int16_t* c_real, int16_t* c_imag, int32_t* c_exp
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
        int32_t shift = a_exp - b_exp;
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
        int32_t shift = b_exp - a_exp;
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
 * Circle F4E5 complex matrix multiply kernel
 */
__global__ void circle_f4e5_matmul_kernel(
    const int16_t* __restrict__ a_real,
    const int16_t* __restrict__ a_imag,
    const int32_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_real,
    const int16_t* __restrict__ b_imag,
    const int32_t* __restrict__ b_exp,
    int16_t* __restrict__ c_real,
    int16_t* __restrict__ c_imag,
    int32_t* __restrict__ c_exp,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Accumulator
    int16_t acc_real = 0;
    int16_t acc_imag = 0;
    int32_t acc_exp = 0;

    // Dot product
    for (int k = 0; k < K; k++) {
        int a_idx = row * K + k;
        int b_idx = k * N + col;

        int16_t a_r = a_real[a_idx];
        int16_t a_i = a_imag[a_idx];
        int32_t a_e = a_exp[a_idx];

        int16_t b_r = b_real[b_idx];
        int16_t b_i = b_imag[b_idx];
        int32_t b_e = b_exp[b_idx];

        // Skip if either is zero
        if ((a_r == 0 && a_i == 0) || (b_r == 0 && b_i == 0)) {
            continue;
        }

        // Multiply
        int16_t prod_real, prod_imag;
        int32_t prod_exp;
        circle_f4e5_mul(a_r, a_i, a_e, b_r, b_i, b_e, &prod_real, &prod_imag, &prod_exp);

        // Add to accumulator
        int16_t new_real, new_imag;
        int32_t new_exp;
        circle_f4e5_add(acc_real, acc_imag, acc_exp, prod_real, prod_imag, prod_exp,
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
 * Host wrapper
 */
extern "C" void circle_f4e5_matmul_hip(
    const int16_t* h_a_real, const int16_t* h_a_imag, const int32_t* h_a_exp,
    const int16_t* h_b_real, const int16_t* h_b_imag, const int32_t* h_b_exp,
    int16_t* h_c_real, int16_t* h_c_imag, int32_t* h_c_exp,
    int M, int N, int K
) {
    size_t a_real_size = M * K * sizeof(int16_t);
    size_t a_imag_size = M * K * sizeof(int16_t);
    size_t a_exp_size = M * K * sizeof(int32_t);
    size_t b_real_size = K * N * sizeof(int16_t);
    size_t b_imag_size = K * N * sizeof(int16_t);
    size_t b_exp_size = K * N * sizeof(int32_t);
    size_t c_real_size = M * N * sizeof(int16_t);
    size_t c_imag_size = M * N * sizeof(int16_t);
    size_t c_exp_size = M * N * sizeof(int32_t);

    int16_t *d_a_real, *d_a_imag, *d_b_real, *d_b_imag, *d_c_real, *d_c_imag;
    int32_t *d_a_exp, *d_b_exp, *d_c_exp;

    hipMalloc(&d_a_real, a_real_size);
    hipMalloc(&d_a_imag, a_imag_size);
    hipMalloc(&d_a_exp, a_exp_size);
    hipMalloc(&d_b_real, b_real_size);
    hipMalloc(&d_b_imag, b_imag_size);
    hipMalloc(&d_b_exp, b_exp_size);
    hipMalloc(&d_c_real, c_real_size);
    hipMalloc(&d_c_imag, c_imag_size);
    hipMalloc(&d_c_exp, c_exp_size);

    hipMemcpy(d_a_real, h_a_real, a_real_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_imag, h_a_imag, a_imag_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, a_exp_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_real, h_b_real, b_real_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_imag, h_b_imag, b_imag_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, b_exp_size, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(
        circle_f4e5_matmul_kernel,
        gridDim, blockDim, 0, 0,
        d_a_real, d_a_imag, d_a_exp,
        d_b_real, d_b_imag, d_b_exp,
        d_c_real, d_c_imag, d_c_exp,
        M, N, K
    );

    hipMemcpy(h_c_real, d_c_real, c_real_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_imag, d_c_imag, c_imag_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_exp, d_c_exp, c_exp_size, hipMemcpyDeviceToHost);

    hipFree(d_a_real); hipFree(d_a_imag); hipFree(d_a_exp);
    hipFree(d_b_real); hipFree(d_b_imag); hipFree(d_b_exp);
    hipFree(d_c_real); hipFree(d_c_imag); hipFree(d_c_exp);
}
