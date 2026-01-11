/**
 * IEEE f32 Five Operations - HIP Kernel
 *
 * Implements all five basic arithmetic operations using IEEE-754 f32:
 * - Add (+)
 * - Subtract (-)
 * - Multiply (*)
 * - Divide (/)
 * - Modulo (%)
 *
 * Uses volatile to preserve denormals.
 * Compiled with -fno-fast-math to disable FTZ.
 */

#include <hip/hip_runtime.h>
#include <cmath>

/**
 * Kernel: Test all five IEEE operations with denormal support
 */
__global__ void ieee_ops_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ add_result,
    float* __restrict__ sub_result,
    float* __restrict__ mul_result,
    float* __restrict__ div_result,
    float* __restrict__ mod_result,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    volatile float av = a[idx];
    volatile float bv = b[idx];

    // Perform all five operations (volatile forces denormal preservation)
    volatile float add_val = av + bv;
    volatile float sub_val = av - bv;
    volatile float mul_val = av * bv;
    volatile float div_val = av / bv;
    volatile float mod_val = fmodf(av, bv);

    add_result[idx] = add_val;
    sub_result[idx] = sub_val;
    mul_result[idx] = mul_val;
    div_result[idx] = div_val;
    mod_result[idx] = mod_val;
}

/**
 * Host wrapper
 */
extern "C" void ieee_ops_hip(
    const float* h_a,
    const float* h_b,
    float* h_add,
    float* h_sub,
    float* h_mul,
    float* h_div,
    float* h_mod,
    int N
) {
    size_t size = N * sizeof(float);

    float *d_a, *d_b, *d_add, *d_sub, *d_mul, *d_div, *d_mod;

    hipMalloc(&d_a, size);
    hipMalloc(&d_b, size);
    hipMalloc(&d_add, size);
    hipMalloc(&d_sub, size);
    hipMalloc(&d_mul, size);
    hipMalloc(&d_div, size);
    hipMalloc(&d_mod, size);

    hipMemcpy(d_a, h_a, size, hipMemcpyHostToDevice);
    hipMemcpy(d_b, h_b, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(
        ieee_ops_kernel,
        dim3(gridSize), dim3(blockSize), 0, 0,
        d_a, d_b, d_add, d_sub, d_mul, d_div, d_mod, N
    );

    hipMemcpy(h_add, d_add, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_sub, d_sub, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mul, d_mul, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_div, d_div, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_mod, d_mod, size, hipMemcpyDeviceToHost);

    hipFree(d_a); hipFree(d_b);
    hipFree(d_add); hipFree(d_sub); hipFree(d_mul); hipFree(d_div); hipFree(d_mod);
}
