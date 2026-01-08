/**
 * Spirix Matrix Multiplication - Pure Inline Assembly
 *
 * ALL operations use inline RDNA assembly.
 * Zero reliance on compiler optimizations.
 * Complete control over instruction sequence.
 *
 * RDNA2 ISA Reference:
 * - v_mul_i32: 32-bit integer multiply
 * - v_add_i32: 32-bit integer add
 * - v_sub_i32: 32-bit integer subtract
 * - v_lshlrev_b32: Logical shift left
 * - v_lshrrev_b32: Logical shift right
 * - v_ashrrev_i32: Arithmetic shift right (preserves sign)
 * - s_flbit_i32: Find leading bit (count leading zeros/ones)
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

/**
 * Spirix multiply using pure inline assembly
 *
 * ALL instructions explicitly written - no compiler interference
 */
__device__ inline void spirix_mul_asm(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    int32_t frac_product;
    int32_t exp_sum;
    int32_t leading;
    int32_t shift;
    int32_t normalized;
    int32_t result_frac;
    int32_t result_exp;

    // Extend i16 to i32 and multiply
    asm volatile(
        "v_cvt_i32_i16 %0, %2\n"      // Sign-extend a_frac to i32
        "v_cvt_i32_i16 %1, %3\n"      // Sign-extend b_frac to i32
        "v_mul_i32 %0, %0, %1\n"      // Multiply
        : "=v"(frac_product)
        : "v"(0), "v"(a_frac), "v"(b_frac)
    );

    // Add exponents
    asm volatile(
        "v_cvt_i32_i16 %0, %2\n"      // Sign-extend a_exp to i32
        "v_cvt_i32_i16 %1, %3\n"      // Sign-extend b_exp to i32
        "v_add_i32 %0, %0, %1\n"      // Add exponents
        : "=v"(exp_sum)
        : "v"(0), "v"(a_exp), "v"(b_exp)
    );

    // Check for zero
    if (frac_product == 0) {
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    // Count leading ones/zeros
    // RDNA doesn't have a single "clz" that handles sign, so we do it manually
    int32_t abs_product = frac_product < 0 ? -frac_product : frac_product;

    asm volatile(
        "v_ffbh_u32 %0, %1\n"         // Find first bit high (unsigned)
        : "=v"(leading)
        : "v"(abs_product)
    );

    // expo_adjust = leading - 2
    // shift = leading - 1
    int32_t expo_adjust;
    asm volatile(
        "v_sub_i32 %0, %1, 2\n"       // expo_adjust = leading - 2
        "v_sub_i32 %1, %2, 1\n"       // shift = leading - 1
        : "=v"(expo_adjust), "=v"(shift)
        : "v"(leading), "v"(leading)
    );

    // Normalize: frac_product << shift
    asm volatile(
        "v_lshlrev_b32 %0, %1, %2\n"  // normalized = frac_product << shift
        : "=v"(normalized)
        : "v"(shift), "v"(frac_product)
    );

    // Extract top 16 bits: normalized >> 16
    asm volatile(
        "v_ashrrev_i32 %0, 16, %1\n"  // Arithmetic shift right (preserves sign)
        : "=v"(result_frac)
        : "v"(normalized)
    );

    // Adjust exponent: exp_sum - expo_adjust
    asm volatile(
        "v_sub_i32 %0, %1, %2\n"      // result_exp = exp_sum - expo_adjust
        : "=v"(result_exp)
        : "v"(exp_sum), "v"(expo_adjust)
    );

    // Truncate to i16
    *c_frac = (int16_t)result_frac;
    *c_exp = (int16_t)result_exp;

    // Check for vanished/exploded (scalar comparison, not worth asm)
    if (result_exp < -32768) {
        *c_frac = 0;
        *c_exp = -32768;
    } else if (result_exp > 32767) {
        *c_frac = 0;
        *c_exp = 32767;
    }
}

/**
 * Spirix add using pure inline assembly
 */
__device__ inline void spirix_add_asm(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    // Handle zero cases (scalar comparisons are fine)
    if (a_frac == 0) {
        *c_frac = b_frac;
        *c_exp = b_exp;
        return;
    }
    if (b_frac == 0) {
        *c_frac = a_frac;
        *c_exp = a_exp;
        return;
    }

    int32_t a_frac_32, b_frac_32, a_exp_32, b_exp_32;
    int32_t exp_diff, shift, b_shifted, sum;
    int32_t result_frac, result_exp;

    // Sign-extend to i32
    asm volatile(
        "v_cvt_i32_i16 %0, %4\n"      // a_frac_32
        "v_cvt_i32_i16 %1, %5\n"      // b_frac_32
        "v_cvt_i32_i16 %2, %6\n"      // a_exp_32
        "v_cvt_i32_i16 %3, %7\n"      // b_exp_32
        : "=v"(a_frac_32), "=v"(b_frac_32), "=v"(a_exp_32), "=v"(b_exp_32)
        : "v"(a_frac), "v"(b_frac), "v"(a_exp), "v"(b_exp)
    );

    // Align exponents (shift smaller fraction)
    if (a_exp > b_exp) {
        // Shift b down
        asm volatile(
            "v_sub_i32 %0, %1, %2\n"      // shift = a_exp - b_exp
            "v_ashrrev_i32 %1, %0, %3\n"  // b_shifted = b_frac >> shift
            "v_add_i32 %2, %4, %1\n"      // sum = a_frac + b_shifted
            : "=v"(shift), "=v"(b_shifted), "=v"(sum)
            : "v"(a_exp_32), "v"(b_exp_32), "v"(b_frac_32), "v"(a_frac_32)
        );
        result_frac = sum;
        result_exp = a_exp_32;
    } else if (b_exp > a_exp) {
        // Shift a down
        int32_t a_shifted;
        asm volatile(
            "v_sub_i32 %0, %1, %2\n"      // shift = b_exp - a_exp
            "v_ashrrev_i32 %1, %0, %3\n"  // a_shifted = a_frac >> shift
            "v_add_i32 %2, %1, %4\n"      // sum = a_shifted + b_frac
            : "=v"(shift), "=v"(a_shifted), "=v"(sum)
            : "v"(b_exp_32), "v"(a_exp_32), "v"(a_frac_32), "v"(b_frac_32)
        );
        result_frac = sum;
        result_exp = b_exp_32;
    } else {
        // Same exponent, just add
        asm volatile(
            "v_add_i32 %0, %1, %2\n"      // sum = a_frac + b_frac
            : "=v"(result_frac)
            : "v"(a_frac_32), "v"(b_frac_32)
        );
        result_exp = a_exp_32;
    }

    *c_frac = (int16_t)result_frac;
    *c_exp = (int16_t)result_exp;
}

/**
 * Spirix subtract using pure inline assembly
 */
__device__ inline void spirix_sub_asm(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    // Subtract is just add with negated b
    int16_t neg_b_frac;

    asm volatile(
        "v_sub_i32 %0, 0, %1\n"       // neg_b_frac = 0 - b_frac
        : "=v"(neg_b_frac)
        : "v"(b_frac)
    );

    spirix_add_asm(a_frac, a_exp, neg_b_frac, b_exp, c_frac, c_exp);
}

/**
 * Matrix multiply kernel using inline assembly Spirix ops
 */
__global__ void spirix_matmul_asm_kernel(
    const int16_t* __restrict__ a_frac,
    const int16_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_frac,
    const int16_t* __restrict__ b_exp,
    int16_t* __restrict__ c_frac,
    int16_t* __restrict__ c_exp,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    // Accumulator
    int16_t acc_frac = 0;
    int16_t acc_exp = 0;

    for (int k = 0; k < K; k++) {
        int a_idx = row * K + k;
        int b_idx = k * N + col;

        int16_t a_f = a_frac[a_idx];
        int16_t a_e = a_exp[a_idx];
        int16_t b_f = b_frac[b_idx];
        int16_t b_e = b_exp[b_idx];

        // Multiply using inline assembly
        int16_t prod_frac, prod_exp;
        spirix_mul_asm(a_f, a_e, b_f, b_e, &prod_frac, &prod_exp);

        // Add to accumulator using inline assembly
        int16_t new_acc_frac, new_acc_exp;
        spirix_add_asm(acc_frac, acc_exp, prod_frac, prod_exp, &new_acc_frac, &new_acc_exp);

        acc_frac = new_acc_frac;
        acc_exp = new_acc_exp;
    }

    int c_idx = row * N + col;
    c_frac[c_idx] = acc_frac;
    c_exp[c_idx] = acc_exp;
}

/**
 * Host wrapper - same as before
 */
extern "C" void spirix_matmul_asm_hip(
    const int16_t* h_a_frac, const int16_t* h_a_exp,
    const int16_t* h_b_frac, const int16_t* h_b_exp,
    int16_t* h_c_frac, int16_t* h_c_exp,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(int16_t);
    size_t b_size = K * N * sizeof(int16_t);
    size_t c_size = M * N * sizeof(int16_t);

    int16_t *d_a_frac, *d_a_exp, *d_b_frac, *d_b_exp, *d_c_frac, *d_c_exp;

    hipMalloc(&d_a_frac, a_size);
    hipMalloc(&d_a_exp, a_size);
    hipMalloc(&d_b_frac, b_size);
    hipMalloc(&d_b_exp, b_size);
    hipMalloc(&d_c_frac, c_size);
    hipMalloc(&d_c_exp, c_size);

    hipMemcpy(d_a_frac, h_a_frac, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_frac, h_b_frac, b_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, b_size, hipMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    hipLaunchKernelGGL(
        spirix_matmul_asm_kernel,
        gridDim, blockDim, 0, 0,
        d_a_frac, d_a_exp, d_b_frac, d_b_exp, d_c_frac, d_c_exp,
        M, N, K
    );

    hipMemcpy(h_c_frac, d_c_frac, c_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_exp, d_c_exp, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a_frac); hipFree(d_a_exp);
    hipFree(d_b_frac); hipFree(d_b_exp);
    hipFree(d_c_frac); hipFree(d_c_exp);
}
