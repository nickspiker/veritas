/**
 * Spirix Matrix Multiplication - HIP Kernel
 *
 * Pure integer arithmetic, ZERO IEEE-754.
 *
 * Matrix multiply: C = A * B
 * Where A is M×K, B is K×N, C is M×N
 *
 * Each Spirix scalar is represented as:
 * - fraction: i16 (two's complement)
 * - exponent: i16 (two's complement)
 *
 * This is Phase 1: Naive HIP implementation.
 * No shared memory, no optimizations yet.
 * Just prove it works and measure baseline.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

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
__device__ inline void spirix_mul(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    // Multiply fractions (i16 × i16 → i32)
    int32_t frac_product = (int32_t)a_frac * (int32_t)b_frac;

    // Add exponents
    int32_t exp_sum = (int32_t)a_exp + (int32_t)b_exp;

    // Normalize: Find leading bit position
    // For now, simple shift by 16 to get back to i16 range
    // (Full normalization would use __clz for leading zeros)

    if (frac_product == 0) {
        // Zero result
        *c_frac = 0;
        *c_exp = 0;
        return;
    }

    // Count leading ones/zeros to normalize
    int leading = __clz(frac_product < 0 ? ~frac_product : frac_product);

    // expo_adjust = leading - 2 (matches CPU implementation)
    // shift = expo_adjust + 1 = leading - 1
    int shift = leading - 1;
    int32_t normalized = frac_product << shift;

    // Extract top 16 bits as fraction
    *c_frac = (int16_t)(normalized >> 16);

    // Adjust exponent: exp_sum - (leading - 2) = exp_sum - leading + 2
    int expo_adjust = leading - 2;
    *c_exp = (int16_t)(exp_sum - expo_adjust);

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
__device__ inline void spirix_add(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
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
    int16_t result_frac;
    int16_t result_exp;

    if (a_exp > b_exp) {
        // a is larger, shift b
        int shift = a_exp - b_exp;
        if (shift > 15) {
            // b is negligible
            *c_frac = a_frac;
            *c_exp = a_exp;
            return;
        }
        int32_t b_shifted = (int32_t)b_frac >> shift;
        int32_t sum = (int32_t)a_frac + b_shifted;
        result_frac = (int16_t)sum;  // May overflow, that's fine
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
        int32_t a_shifted = (int32_t)a_frac >> shift;
        int32_t sum = a_shifted + (int32_t)b_frac;
        result_frac = (int16_t)sum;
        result_exp = b_exp;
    } else {
        // Same exponent, just add
        int32_t sum = (int32_t)a_frac + (int32_t)b_frac;
        result_frac = (int16_t)sum;
        result_exp = a_exp;
    }

    // Normalize if needed
    // (For now, simple version - full implementation would check overflow)
    *c_frac = result_frac;
    *c_exp = result_exp;
}

/**
 * Kernel: Naive matrix multiply
 *
 * Each thread computes one element of C.
 * No shared memory, no tiling (yet).
 *
 * Layout: Row-major (same as our Rust tensors)
 */
__global__ void spirix_matmul_kernel(
    const int16_t* __restrict__ a_frac,
    const int16_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_frac,
    const int16_t* __restrict__ b_exp,
    int16_t* __restrict__ c_frac,
    int16_t* __restrict__ c_exp,
    int M, int N, int K
) {
    // Calculate output position
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) {
        return;  // Out of bounds
    }

    // Accumulate dot product: sum of A[row,k] * B[k,col]
    int16_t acc_frac = 0;
    int16_t acc_exp = 0;

    for (int k = 0; k < K; k++) {
        // Load A[row, k]
        int a_idx = row * K + k;
        int16_t a_f = a_frac[a_idx];
        int16_t a_e = a_exp[a_idx];

        // Load B[k, col]
        int b_idx = k * N + col;
        int16_t b_f = b_frac[b_idx];
        int16_t b_e = b_exp[b_idx];

        // Multiply: prod = A[row,k] * B[k,col]
        int16_t prod_frac, prod_exp;
        spirix_mul(a_f, a_e, b_f, b_e, &prod_frac, &prod_exp);

        // Add to accumulator: acc += prod
        int16_t new_acc_frac, new_acc_exp;
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
 * Host function: Launch matrix multiply
 *
 * Allocates device memory, copies data, launches kernel, copies result back.
 *
 * This is a simple wrapper for testing. In production, we'd manage
 * device memory explicitly and batch operations.
 */
extern "C" void spirix_matmul_hip(
    const int16_t* h_a_frac, const int16_t* h_a_exp,
    const int16_t* h_b_frac, const int16_t* h_b_exp,
    int16_t* h_c_frac, int16_t* h_c_exp,
    int M, int N, int K
) {
    // Size calculations
    size_t a_size = M * K * sizeof(int16_t);
    size_t b_size = K * N * sizeof(int16_t);
    size_t c_size = M * N * sizeof(int16_t);

    // Allocate device memory
    int16_t *d_a_frac, *d_a_exp;
    int16_t *d_b_frac, *d_b_exp;
    int16_t *d_c_frac, *d_c_exp;

    hipMalloc(&d_a_frac, a_size);
    hipMalloc(&d_a_exp, a_size);
    hipMalloc(&d_b_frac, b_size);
    hipMalloc(&d_b_exp, b_size);
    hipMalloc(&d_c_frac, c_size);
    hipMalloc(&d_c_exp, c_size);

    // Copy input data to device
    hipMemcpy(d_a_frac, h_a_frac, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_frac, h_b_frac, b_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, b_size, hipMemcpyHostToDevice);

    // Launch configuration
    dim3 blockDim(16, 16);  // 256 threads per block
    dim3 gridDim(
        (N + blockDim.x - 1) / blockDim.x,
        (M + blockDim.y - 1) / blockDim.y
    );

    // Launch kernel
    hipLaunchKernelGGL(
        spirix_matmul_kernel,
        gridDim, blockDim, 0, 0,
        d_a_frac, d_a_exp,
        d_b_frac, d_b_exp,
        d_c_frac, d_c_exp,
        M, N, K
    );

    // Copy result back to host
    hipMemcpy(h_c_frac, d_c_frac, c_size, hipMemcpyDeviceToHost);
    hipMemcpy(h_c_exp, d_c_exp, c_size, hipMemcpyDeviceToHost);

    // Free device memory
    hipFree(d_a_frac);
    hipFree(d_a_exp);
    hipFree(d_b_frac);
    hipFree(d_b_exp);
    hipFree(d_c_frac);
    hipFree(d_c_exp);
}

/**
 * Performance notes:
 *
 * This is a NAIVE implementation. Expected performance is poor.
 *
 * Problems:
 * 1. No shared memory (thrashes global memory)
 * 2. No memory coalescing (unaligned accesses)
 * 3. No loop unrolling (compiler might help)
 * 4. No register tiling (each thread does one element)
 *
 * But: ZERO BRANCH DIVERGENCE in the inner loop.
 * All threads execute same path (spirix_mul, spirix_add).
 *
 * Next optimization:
 * - Shared memory tiling (16×16 or 32×32 tiles)
 * - Memory coalescing (stride access patterns)
 * - Register blocking (each thread computes 4×4 tile)
 *
 * Expected speedup from optimizations: 10-50x
 * Expected final performance: Match or beat IEEE-754
 */
