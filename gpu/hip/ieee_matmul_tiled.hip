/**
 * IEEE f32 Matrix Multiplication - Tiled with Shared Memory
 *
 * Same optimizations as Spirix tiled version:
 * - 16x16 tile size
 * - Shared memory for A and B tiles
 * - Coalesced memory access
 *
 * Fair comparison: Both use same optimization strategy
 */

#include <hip/hip_runtime.h>

#define TILE_SIZE 16

/**
 * Tiled matrix multiply kernel - IEEE f32
 */
__global__ void ieee_matmul_tiled_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int M, int N, int K
) {
    // Shared memory for tiles
    __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    volatile float acc = 0.0f;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load tile of A
        int a_row = row;
        int a_col = t * TILE_SIZE + tx;
        if (a_row < M && a_col < K) {
            tile_a[ty][tx] = a[a_row * K + a_col];
        } else {
            tile_a[ty][tx] = 0.0f;
        }

        // Load tile of B
        int b_row = t * TILE_SIZE + ty;
        int b_col = col;
        if (b_row < K && b_col < N) {
            tile_b[ty][tx] = b[b_row * N + b_col];
        } else {
            tile_b[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        for (int k = 0; k < TILE_SIZE; k++) {
            volatile float av = tile_a[ty][k];
            volatile float bv = tile_b[k][tx];
            volatile float prod = av * bv;
            acc += prod;
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = acc;
    }
}

/**
 * Host wrapper
 */
extern "C" void ieee_matmul_tiled_hip(
    const float* h_a,
    const float* h_b,
    float* h_c,
    int M, int N, int K
) {
    size_t a_size = M * K * sizeof(float);
    size_t b_size = K * N * sizeof(float);
    size_t c_size = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;

    hipMalloc(&d_a, a_size);
    hipMalloc(&d_b, b_size);
    hipMalloc(&d_c, c_size);

    hipMemcpy(d_a, h_a, a_size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, b_size, hipMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    hipLaunchKernelGGL(
        ieee_matmul_tiled_kernel,
        gridDim, blockDim, 0, 0,
        d_a, d_b, d_c, M, N, K
    );

    hipMemcpy(h_c, d_c, c_size, hipMemcpyDeviceToHost);

    hipFree(d_a);
    hipFree(d_b);
    hipFree(d_c);
}
