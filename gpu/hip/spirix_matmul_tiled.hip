/**
 * Spirix Matrix Multiplication - Tiled with Shared Memory
 *
 * Phase 2 Optimizations:
 * - 16x16 tile size (matches RDNA2 wave dimensions)
 * - Shared memory for A and B tiles
 * - Reuse loaded data across multiple threads
 * - Coalesced global memory access
 *
 * Expected: 5-10x faster than naive kernel
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

#define TILE_SIZE 16

/**
 * Spirix multiply - same as before
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
 * Spirix add - same as before
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
 * Tiled matrix multiply kernel
 *
 * Uses shared memory to cache tiles of A and B.
 * Each 16x16 block computes a tile of C.
 */
__global__ void spirix_matmul_tiled_kernel(
    const int16_t* __restrict__ a_frac,
    const int16_t* __restrict__ a_exp,
    const int16_t* __restrict__ b_frac,
    const int16_t* __restrict__ b_exp,
    int16_t* __restrict__ c_frac,
    int16_t* __restrict__ c_exp,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ int16_t tile_a_frac[TILE_SIZE][TILE_SIZE];
    __shared__ int16_t tile_a_exp[TILE_SIZE][TILE_SIZE];
    __shared__ int16_t tile_b_frac[TILE_SIZE][TILE_SIZE];
    __shared__ int16_t tile_b_exp[TILE_SIZE][TILE_SIZE];

    // Thread coordinates
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    // Accumulator for this thread
    int16_t acc_frac = 0;
    int16_t acc_exp = 0;

    // Loop over tiles
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            int a_idx = a_row * K + a_col;
            tile_a_frac[ty][tx] = a_frac[a_idx];
            tile_a_exp[ty][tx] = a_exp[a_idx];
        } else {
            tile_a_frac[ty][tx] = 0;
            tile_a_exp[ty][tx] = 0;
        }

        // Load tile of B into shared memory
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            int b_idx = b_row * N + b_col;
            tile_b_frac[ty][tx] = b_frac[b_idx];
            tile_b_exp[ty][tx] = b_exp[b_idx];
        } else {
            tile_b_frac[ty][tx] = 0;
            tile_b_exp[ty][tx] = 0;
        }

        // Synchronize to ensure tiles are loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            int16_t a_f = tile_a_frac[ty][k];
            int16_t a_e = tile_a_exp[ty][k];
            int16_t b_f = tile_b_frac[k][tx];
            int16_t b_e = tile_b_exp[k][tx];

            // Skip if either is zero
            if (a_f == 0 || b_f == 0) continue;

            // Multiply
            int16_t prod_frac, prod_exp;
            spirix_mul(a_f, a_e, b_f, b_e, &prod_frac, &prod_exp);

            // Add to accumulator
            int16_t new_acc_frac, new_acc_exp;
            spirix_add(acc_frac, acc_exp, prod_frac, prod_exp, &new_acc_frac, &new_acc_exp);

            acc_frac = new_acc_frac;
            acc_exp = new_acc_exp;
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        int c_idx = row * N + col;
        c_frac[c_idx] = acc_frac;
        c_exp[c_idx] = acc_exp;
    }
}

/**
 * Host wrapper
 */
extern "C" void spirix_matmul_tiled_hip(
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

    // Launch with TILE_SIZE x TILE_SIZE threads per block
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(
        spirix_matmul_tiled_kernel,
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
