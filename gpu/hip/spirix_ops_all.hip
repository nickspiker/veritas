/**
 * Spirix Five Operations - HIP Kernel
 *
 * Implements all five basic arithmetic operations using pure integer math:
 * - Add (+)
 * - Subtract (-)
 * - Multiply (*)
 * - Divide (/)
 * - Modulo (%)
 *
 * ZERO IEEE-754. All integer ops.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Spirix Add: c = a + b
 */
__device__ inline void spirix_add(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    if (a_frac == 0) { *c_frac = b_frac; *c_exp = b_exp; return; }
    if (b_frac == 0) { *c_frac = a_frac; *c_exp = a_exp; return; }

    if (a_exp > b_exp) {
        int shift = a_exp - b_exp;
        if (shift > 15) { *c_frac = a_frac; *c_exp = a_exp; return; }
        int32_t sum = (int32_t)a_frac + ((int32_t)b_frac >> shift);
        *c_frac = (int16_t)sum;
        *c_exp = a_exp;
    } else if (b_exp > a_exp) {
        int shift = b_exp - a_exp;
        if (shift > 15) { *c_frac = b_frac; *c_exp = b_exp; return; }
        int32_t sum = ((int32_t)a_frac >> shift) + (int32_t)b_frac;
        *c_frac = (int16_t)sum;
        *c_exp = b_exp;
    } else {
        int32_t sum = (int32_t)a_frac + (int32_t)b_frac;
        *c_frac = (int16_t)sum;
        *c_exp = a_exp;
    }
}

/**
 * Spirix Subtract: c = a - b
 */
__device__ inline void spirix_sub(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    // Negate b and add
    spirix_add(a_frac, a_exp, -b_frac, b_exp, c_frac, c_exp);
}

/**
 * Spirix Multiply: c = a * b
 */
__device__ inline void spirix_mul(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    int32_t frac_product = (int32_t)a_frac * (int32_t)b_frac;
    int32_t exp_sum = (int32_t)a_exp + (int32_t)b_exp;

    if (frac_product == 0) {
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    int leading = __clz(frac_product < 0 ? ~frac_product : frac_product);
    int shift = leading - 1;
    int32_t normalized = frac_product << shift;

    *c_frac = (int16_t)(normalized >> 16);
    int expo_adjust = leading - 2;
    *c_exp = (int16_t)(exp_sum - expo_adjust);

    if (exp_sum < -32768) { *c_frac = 0; *c_exp = -32768; }
    if (exp_sum > 32767) { *c_frac = 0; *c_exp = 32767; }
}

/**
 * Spirix Divide: c = a / b
 *
 * Algorithm:
 * 1. Divide fractions (using integer division on normalized values)
 * 2. Subtract exponents
 * 3. Normalize result
 */
__device__ inline void spirix_div(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    // Check for division by zero
    if (b_frac == 0) {
        *c_frac = 0;
        *c_exp = 32767;  // Infinity/exploded
        return;
    }

    if (a_frac == 0) {
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    // Extend to 32-bit for division
    int32_t a_wide = (int32_t)a_frac << 16;  // Shift left to preserve precision
    int32_t b_wide = (int32_t)b_frac;

    // Integer division
    int32_t quotient = a_wide / b_wide;

    // Subtract exponents
    int32_t exp_diff = (int32_t)a_exp - (int32_t)b_exp;

    // Normalize
    if (quotient == 0) {
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    int leading = __clz(quotient < 0 ? ~quotient : quotient);
    int shift = leading - 1;
    int32_t normalized = quotient << shift;

    *c_frac = (int16_t)(normalized >> 16);
    int expo_adjust = leading - 2;
    *c_exp = (int16_t)(exp_diff - expo_adjust);

    if (*c_exp < -32768) { *c_frac = 0; *c_exp = -32768; }
    if (*c_exp > 32767) { *c_frac = 0; *c_exp = 32767; }
}

/**
 * Spirix Modulo: c = a % b
 *
 * Algorithm:
 * 1. Compute a / b
 * 2. Truncate to integer
 * 3. Compute a - (truncated * b)
 */
__device__ inline void spirix_mod(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    if (b_frac == 0) {
        *c_frac = 0;
        *c_exp = 32767;
        return;
    }

    // Compute a / b
    int16_t div_frac, div_exp;
    spirix_div(a_frac, a_exp, b_frac, b_exp, &div_frac, &div_exp);

    // Truncate to integer (zero the fraction bits)
    // For now, simple version: if exp >= 0, it's already integer-like
    // Full version would properly truncate

    // Compute truncated * b
    int16_t prod_frac, prod_exp;
    spirix_mul(div_frac, div_exp, b_frac, b_exp, &prod_frac, &prod_exp);

    // Compute a - prod
    spirix_sub(a_frac, a_exp, prod_frac, prod_exp, c_frac, c_exp);
}

/**
 * Kernel: Test all five operations
 */
__global__ void spirix_ops_kernel(
    const int16_t* __restrict__ a_frac,
    const int16_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_frac,
    const int16_t* __restrict__ b_exp,
    int16_t* __restrict__ add_frac, int16_t* __restrict__ add_exp,
    int16_t* __restrict__ sub_frac, int16_t* __restrict__ sub_exp,
    int16_t* __restrict__ mul_frac, int16_t* __restrict__ mul_exp,
    int16_t* __restrict__ div_frac, int16_t* __restrict__ div_exp,
    int16_t* __restrict__ mod_frac, int16_t* __restrict__ mod_exp,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int16_t af = a_frac[idx];
    int16_t ae = a_exp[idx];
    int16_t bf = b_frac[idx];
    int16_t be = b_exp[idx];

    // Perform all five operations
    spirix_add(af, ae, bf, be, &add_frac[idx], &add_exp[idx]);
    spirix_sub(af, ae, bf, be, &sub_frac[idx], &sub_exp[idx]);
    spirix_mul(af, ae, bf, be, &mul_frac[idx], &mul_exp[idx]);
    spirix_div(af, ae, bf, be, &div_frac[idx], &div_exp[idx]);
    spirix_mod(af, ae, bf, be, &mod_frac[idx], &mod_exp[idx]);
}

/**
 * Host wrapper
 */
extern "C" void spirix_ops_hip(
    const int16_t* h_a_frac, const int16_t* h_a_exp,
    const int16_t* h_b_frac, const int16_t* h_b_exp,
    int16_t* h_add_frac, int16_t* h_add_exp,
    int16_t* h_sub_frac, int16_t* h_sub_exp,
    int16_t* h_mul_frac, int16_t* h_mul_exp,
    int16_t* h_div_frac, int16_t* h_div_exp,
    int16_t* h_mod_frac, int16_t* h_mod_exp,
    int N
) {
    size_t size = N * sizeof(int16_t);

    int16_t *d_a_frac, *d_a_exp, *d_b_frac, *d_b_exp;
    int16_t *d_add_frac, *d_add_exp, *d_sub_frac, *d_sub_exp;
    int16_t *d_mul_frac, *d_mul_exp, *d_div_frac, *d_div_exp;
    int16_t *d_mod_frac, *d_mod_exp;

    hipMalloc(&d_a_frac, size); hipMalloc(&d_a_exp, size);
    hipMalloc(&d_b_frac, size); hipMalloc(&d_b_exp, size);
    hipMalloc(&d_add_frac, size); hipMalloc(&d_add_exp, size);
    hipMalloc(&d_sub_frac, size); hipMalloc(&d_sub_exp, size);
    hipMalloc(&d_mul_frac, size); hipMalloc(&d_mul_exp, size);
    hipMalloc(&d_div_frac, size); hipMalloc(&d_div_exp, size);
    hipMalloc(&d_mod_frac, size); hipMalloc(&d_mod_exp, size);

    hipMemcpy(d_a_frac, h_a_frac, size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_frac, h_b_frac, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(
        spirix_ops_kernel,
        dim3(gridSize), dim3(blockSize), 0, 0,
        d_a_frac, d_a_exp, d_b_frac, d_b_exp,
        d_add_frac, d_add_exp, d_sub_frac, d_sub_exp,
        d_mul_frac, d_mul_exp, d_div_frac, d_div_exp,
        d_mod_frac, d_mod_exp, N
    );

    hipMemcpy(h_add_frac, d_add_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_add_exp, d_add_exp, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_sub_frac, d_sub_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_sub_exp, d_sub_exp, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mul_frac, d_mul_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mul_exp, d_mul_exp, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_div_frac, d_div_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_div_exp, d_div_exp, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mod_frac, d_mod_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mod_exp, d_mod_exp, size, hipMemcpyDeviceToHost);

    hipFree(d_a_frac); hipFree(d_a_exp); hipFree(d_b_frac); hipFree(d_b_exp);
    hipFree(d_add_frac); hipFree(d_add_exp); hipFree(d_sub_frac); hipFree(d_sub_exp);
    hipFree(d_mul_frac); hipFree(d_mul_exp); hipFree(d_div_frac); hipFree(d_div_exp);
    hipFree(d_mod_frac); hipFree(d_mod_exp);
}
