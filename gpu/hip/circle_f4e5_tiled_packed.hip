/**
 * Circle F4E5 Complex Matrix Multiplication - OPTIMIZED
 *
 * Optimizations:
 * 1. Shared memory tiling (16×16)
 * 2. PACKED LOADS: real+imag in single 32-bit load
 * 3. Coalesced memory access
 *
 * Memory layout:
 * - packed_ri: int32_t with real in low 16 bits, imag in high 16 bits
 * - exp: int32_t exponent
 * Total: 64 bits, but only 2 loads instead of 3!
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TILE_SIZE 16

/**
 * Circle F4E5 complex multiply - inline for shared memory use
 */
__device__ inline void circle_f4e5_mul(
    int16_t a_real, int16_t a_imag, int32_t a_exp,
    int16_t b_real, int16_t b_imag, int32_t b_exp,
    int16_t* c_real, int16_t* c_imag, int32_t* c_exp
) {
    if ((a_real == 0 && a_imag == 0) || (b_real == 0 && b_imag == 0)) {
        *c_real = 0;
        *c_imag = 0;
        *c_exp = 0;
        return;
    }

    int32_t a_r = (int32_t)a_real;
    int32_t a_i = (int32_t)a_imag;
    int32_t b_r = (int32_t)b_real;
    int32_t b_i = (int32_t)b_imag;

    int32_t real_product = (a_r * b_r >> 1) - (a_i * b_i >> 1);
    int32_t imag_product = (a_r * b_i >> 1) + (a_i * b_r >> 1);

    if (real_product == 0 && imag_product == 0) {
        *c_real = 0;
        *c_imag = 0;
        *c_exp = 0;
        return;
    }

    int32_t leading_r = real_product < 0 ? __clz(~real_product) : __clz(real_product);
    int32_t leading_i = imag_product < 0 ? __clz(~imag_product) : __clz(imag_product);
    int32_t shift = (leading_r < leading_i ? leading_r : leading_i) - 3;
    if (shift < 0) shift = 0;

    int32_t shift_amount = shift + 2;
    int32_t normalized_real = real_product << shift_amount;
    int32_t normalized_imag = imag_product << shift_amount;

    *c_real = (int16_t)(normalized_real >> 16);
    *c_imag = (int16_t)(normalized_imag >> 16);

    int32_t expo_adjust = shift - 3;
    int64_t exp_sum = (int64_t)a_exp + (int64_t)b_exp - expo_adjust;

    if (exp_sum > 2147483647LL) exp_sum = 2147483647LL;
    if (exp_sum < -2147483648LL) exp_sum = -2147483648LL;

    *c_exp = (int32_t)exp_sum;
}

/**
 * Circle F4E5 complex add - inline for accumulation
 */
__device__ inline void circle_f4e5_add(
    int16_t a_real, int16_t a_imag, int32_t a_exp,
    int16_t b_real, int16_t b_imag, int32_t b_exp,
    int16_t* c_real, int16_t* c_imag, int32_t* c_exp
) {
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
 * OPTIMIZED: Tiled kernel with PACKED LOADS
 *
 * Each complex number is TWO loads:
 * - int32_t packed_ri (real in low 16, imag in high 16)
 * - int32_t exp
 */
__global__ void circle_f4e5_tiled_packed_kernel(
    const int32_t* __restrict__ a_packed_ri,  // real+imag packed
    const int32_t* __restrict__ a_exp,
    const int32_t* __restrict__ b_packed_ri,  // real+imag packed
    const int32_t* __restrict__ b_exp,
    int32_t* __restrict__ c_packed_ri,        // real+imag packed
    int32_t* __restrict__ c_exp,
    int M, int N, int K
) {
    // Shared memory tiles - unpacked for computation
    __shared__ int16_t tile_a_real[TILE_SIZE][TILE_SIZE];
    __shared__ int16_t tile_a_imag[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t tile_a_exp[TILE_SIZE][TILE_SIZE];

    __shared__ int16_t tile_b_real[TILE_SIZE][TILE_SIZE];
    __shared__ int16_t tile_b_imag[TILE_SIZE][TILE_SIZE];
    __shared__ int32_t tile_b_exp[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator
    int16_t acc_real = 0;
    int16_t acc_imag = 0;
    int32_t acc_exp = 0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A - PACKED LOAD then unpack
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            int a_idx = a_row * K + a_col;
            int32_t packed = a_packed_ri[a_idx];  // SINGLE LOAD!
            tile_a_real[ty][tx] = (int16_t)(packed & 0xFFFF);         // Low 16 bits
            tile_a_imag[ty][tx] = (int16_t)((packed >> 16) & 0xFFFF); // High 16 bits
            tile_a_exp[ty][tx] = a_exp[a_idx];
        } else {
            tile_a_real[ty][tx] = 0;
            tile_a_imag[ty][tx] = 0;
            tile_a_exp[ty][tx] = 0;
        }

        // Load tile of B - PACKED LOAD then unpack
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            int b_idx = b_row * N + b_col;
            int32_t packed = b_packed_ri[b_idx];  // SINGLE LOAD!
            tile_b_real[ty][tx] = (int16_t)(packed & 0xFFFF);
            tile_b_imag[ty][tx] = (int16_t)((packed >> 16) & 0xFFFF);
            tile_b_exp[ty][tx] = b_exp[b_idx];
        } else {
            tile_b_real[ty][tx] = 0;
            tile_b_imag[ty][tx] = 0;
            tile_b_exp[ty][tx] = 0;
        }

        __syncthreads();

        // Compute partial dot product from this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            int16_t a_r = tile_a_real[ty][k];
            int16_t a_i = tile_a_imag[ty][k];
            int32_t a_e = tile_a_exp[ty][k];

            int16_t b_r = tile_b_real[k][tx];
            int16_t b_i = tile_b_imag[k][tx];
            int32_t b_e = tile_b_exp[k][tx];

            if ((a_r == 0 && a_i == 0) || (b_r == 0 && b_i == 0)) continue;

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

        __syncthreads();
    }

    // Write result - PACKED STORE
    if (row < M && col < N) {
        int c_idx = row * N + col;
        // Pack real+imag into single 32-bit word
        int32_t packed = ((int32_t)(uint16_t)acc_real) | (((int32_t)(uint16_t)acc_imag) << 16);
        c_packed_ri[c_idx] = packed;  // SINGLE STORE!
        c_exp[c_idx] = acc_exp;
    }
}

/**
 * Host wrapper - takes unpacked host arrays, packs for GPU
 */
extern "C" void circle_f4e5_tiled_packed_hip(
    const int16_t* h_a_real, const int16_t* h_a_imag, const int32_t* h_a_exp,
    const int16_t* h_b_real, const int16_t* h_b_imag, const int32_t* h_b_exp,
    int16_t* h_c_real, int16_t* h_c_imag, int32_t* h_c_exp,
    int M, int N, int K
) {
    size_t a_count = M * K;
    size_t b_count = K * N;
    size_t c_count = M * N;

    // Pack on host
    int32_t* h_a_packed = new int32_t[a_count];
    int32_t* h_b_packed = new int32_t[b_count];

    for (size_t i = 0; i < a_count; i++) {
        h_a_packed[i] = ((int32_t)(uint16_t)h_a_real[i]) | (((int32_t)(uint16_t)h_a_imag[i]) << 16);
    }
    for (size_t i = 0; i < b_count; i++) {
        h_b_packed[i] = ((int32_t)(uint16_t)h_b_real[i]) | (((int32_t)(uint16_t)h_b_imag[i]) << 16);
    }

    // Allocate GPU memory
    int32_t *d_a_packed, *d_a_exp, *d_b_packed, *d_b_exp, *d_c_packed, *d_c_exp;

    hipMalloc(&d_a_packed, a_count * sizeof(int32_t));
    hipMalloc(&d_a_exp, a_count * sizeof(int32_t));
    hipMalloc(&d_b_packed, b_count * sizeof(int32_t));
    hipMalloc(&d_b_exp, b_count * sizeof(int32_t));
    hipMalloc(&d_c_packed, c_count * sizeof(int32_t));
    hipMalloc(&d_c_exp, c_count * sizeof(int32_t));

    // Copy to GPU
    hipMemcpy(d_a_packed, h_a_packed, a_count * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_a_exp, h_a_exp, a_count * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_b_packed, h_b_packed, b_count * sizeof(int32_t), hipMemcpyHostToDevice);
    hipMemcpy(d_b_exp, h_b_exp, b_count * sizeof(int32_t), hipMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(
        circle_f4e5_tiled_packed_kernel,
        gridDim, blockDim, 0, 0,
        d_a_packed, d_a_exp,
        d_b_packed, d_b_exp,
        d_c_packed, d_c_exp,
        M, N, K
    );

    // Copy result back
    int32_t* h_c_packed = new int32_t[c_count];
    hipMemcpy(h_c_packed, d_c_packed, c_count * sizeof(int32_t), hipMemcpyDeviceToHost);
    hipMemcpy(h_c_exp, d_c_exp, c_count * sizeof(int32_t), hipMemcpyDeviceToHost);

    // Unpack on host
    for (size_t i = 0; i < c_count; i++) {
        h_c_real[i] = (int16_t)(h_c_packed[i] & 0xFFFF);
        h_c_imag[i] = (int16_t)((h_c_packed[i] >> 16) & 0xFFFF);
    }

    // Cleanup
    delete[] h_a_packed;
    delete[] h_b_packed;
    delete[] h_c_packed;

    hipFree(d_a_packed); hipFree(d_a_exp);
    hipFree(d_b_packed); hipFree(d_b_exp);
    hipFree(d_c_packed); hipFree(d_c_exp);
}
