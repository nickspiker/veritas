/**
 * Spirix Matrix Multiplication - OpenCL Kernel
 *
 * Pure integer arithmetic, ZERO IEEE-754.
 *
 * Matrix multiply: C = A * B
 * Where A is M×K, B is K×N, C is M×N
 *
 * Each Spirix scalar is represented as:
 * - fraction: short (i16, two's complement)
 * - exponent: short (i16, two's complement)
 *
 * This is Phase 1: Naive OpenCL implementation.
 * No local memory, no optimizations yet.
 * Just prove it works and measure baseline.
 */

/**
 * Device function: Spirix multiply
 *
 * Multiplies two Spirix scalars: c = a * b
 *
 * Algorithm:
 * 1. Multiply fractions (i16 × i16 → i32)
 * 2. Add exponents
 * 3. Normalize result (shift fraction, adjust exponent)
 * 4. Check for vanished/exploded
 *
 * This is the CORE operation. Zero IEEE-754.
 */
inline void spirix_mul(
    short a_frac, short a_exp,
    short b_frac, short b_exp,
    short* c_frac, short* c_exp
) {
    // Multiply fractions (i16 × i16 → i32)
    int frac_product = (int)a_frac * (int)b_frac;

    // Add exponents
    int exp_sum = (int)a_exp + (int)b_exp;

    // Normalize: Find leading bit position
    if (frac_product == 0) {
        // Zero result
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    // Count leading zeros to normalize
    int leading_zeros = clz(frac_product < 0 ? ~frac_product : frac_product);

    // Shift to normalize (align MSB)
    int normalized = frac_product << leading_zeros;

    // Extract top 16 bits as fraction
    *c_frac = (short)(normalized >> 16);

    // Adjust exponent for normalization
    *c_exp = (short)(exp_sum - leading_zeros);

    // Check for vanished (exponent underflow)
    if (exp_sum < -32768) {
        *c_frac = 0;
        *c_exp = -32768;  // Vanished marker
    }

    // Check for exploded (exponent overflow)
    if (exp_sum > 32767) {
        *c_frac = 0;
        *c_exp = 32767;  // Exploded marker
    }
}

/**
 * Device function: Spirix add
 *
 * Adds two Spirix scalars: c = a + b
 *
 * Algorithm:
 * 1. Align exponents (shift smaller fraction)
 * 2. Add fractions
 * 3. Normalize result
 */
inline void spirix_add(
    short a_frac, short a_exp,
    short b_frac, short b_exp,
    short* c_frac, short* c_exp
) {
    // Handle zero cases
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

    // Align exponents (shift fraction with smaller exponent)
    short result_frac;
    short result_exp;

    if (a_exp > b_exp) {
        // a is larger, shift b
        int shift = a_exp - b_exp;
        if (shift > 15) {
            // b is negligible
            *c_frac = a_frac;
            *c_exp = a_exp;
            return;
        }
        int b_shifted = (int)b_frac >> shift;
        int sum = (int)a_frac + b_shifted;
        result_frac = (short)sum;
        result_exp = a_exp;
    } else if (b_exp > a_exp) {
        // b is larger, shift a
        int shift = b_exp - a_exp;
        if (shift > 15) {
            // a is negligible
            *c_frac = b_frac;
            *c_exp = b_exp;
            return;
        }
        int a_shifted = (int)a_frac >> shift;
        int sum = a_shifted + (int)b_frac;
        result_frac = (short)sum;
        result_exp = b_exp;
    } else {
        // Same exponent, just add
        int sum = (int)a_frac + (int)b_frac;
        result_frac = (short)sum;
        result_exp = a_exp;
    }

    // Normalize if needed
    *c_frac = result_frac;
    *c_exp = result_exp;
}

/**
 * Kernel: Naive matrix multiply
 *
 * Each work-item computes one element of C.
 * No local memory, no tiling (yet).
 *
 * Layout: Row-major (same as our Rust tensors)
 */
__kernel void spirix_matmul_kernel(
    __global const short* restrict a_frac,
    __global const short* restrict a_exp,
    __global const short* restrict b_frac,
    __global const short* restrict b_exp,
    __global short* restrict c_frac,
    __global short* restrict c_exp,
    int M, int N, int K
) {
    // Calculate output position
    int row = get_global_id(1);  // Y dimension
    int col = get_global_id(0);  // X dimension

    if (row >= M || col >= N) {
        return;  // Out of bounds
    }

    // Accumulate dot product: sum of A[row,k] * B[k,col]
    short acc_frac = 0;
    short acc_exp = 0;

    for (int k = 0; k < K; k++) {
        // Load A[row, k]
        int a_idx = row * K + k;
        short a_f = a_frac[a_idx];
        short a_e = a_exp[a_idx];

        // Load B[k, col]
        int b_idx = k * N + col;
        short b_f = b_frac[b_idx];
        short b_e = b_exp[b_idx];

        // Multiply: prod = A[row,k] * B[k,col]
        short prod_frac, prod_exp;
        spirix_mul(a_f, a_e, b_f, b_e, &prod_frac, &prod_exp);

        // Add to accumulator: acc += prod
        short new_acc_frac, new_acc_exp;
        spirix_add(acc_frac, acc_exp, prod_frac, prod_exp, &new_acc_frac, &new_acc_exp);

        acc_frac = new_acc_frac;
        acc_exp = new_acc_exp;
    }

    // Write result
    int c_idx = row * N + col;
    c_frac[c_idx] = acc_frac;
    c_exp[c_idx] = acc_exp;
}

/**
 * Performance notes:
 *
 * This is a NAIVE implementation. Expected performance is poor.
 *
 * Problems:
 * 1. No local memory (thrashes global memory)
 * 2. No memory coalescing (unaligned accesses)
 * 3. No loop unrolling (compiler might help)
 * 4. No register tiling (each work-item does one element)
 *
 * But: ZERO BRANCH DIVERGENCE in the inner loop.
 * All work-items execute same path (spirix_mul, spirix_add).
 *
 * Next optimization:
 * - Local memory tiling (16×16 or 32×32 tiles)
 * - Memory coalescing (stride access patterns)
 * - Register blocking (each work-item computes 4×4 tile)
 *
 * Expected speedup from optimizations: 10-50x
 * Expected final performance: Match or beat IEEE-754
 */
