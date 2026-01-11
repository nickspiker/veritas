/**
 * In-Place Operation Benchmarks
 *
 * Data stays on GPU, no host transfers.
 * Tests pure compute throughput: FPU vs ALU.
 *
 * IEEE: Guaranteed 1 denormal per 32-thread wavefront (worst case).
 * Spirix/Circle: All integer ops, no branches.
 */

#include <hip/hip_runtime.h>
#include <stdint.h>

// ============================================================================
// SPIRIX SCALAR IN-PLACE MULTIPLY
// ============================================================================

__device__ inline void spirix_scalar_mul(
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
}

__global__ void spirix_multiply_kernel(
    int16_t* frac, int16_t* exp, int N, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int16_t f = frac[idx];
    int16_t e = exp[idx];

    // Multiply by self many times (stays on GPU)
    for (int i = 0; i < iterations; i++) {
        int16_t new_f, new_e;
        spirix_scalar_mul(f, e, f, e, &new_f, &new_e);
        f = new_f;
        e = new_e;
    }

    frac[idx] = f;
    exp[idx] = e;
}

// ============================================================================
// IEEE SCALAR IN-PLACE MULTIPLY (with forced denormal divergence)
// ============================================================================

__device__ inline bool is_denormal_f32(float x) {
    if (x == 0.0f) return false;
    uint32_t bits = __float_as_int(x);
    uint32_t exponent = (bits >> 23) & 0xFF;
    return (exponent == 0);
}

__global__ void ieee_multiply_kernel(
    float* data, int N, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    volatile float val = data[idx];

    // Multiply by self many times
    for (int i = 0; i < iterations; i++) {
        volatile float result = val * val;

        // FORCE denormal check - causes branch divergence
        // Every 32nd thread has a denormal (worst case for IEEE)
        if (is_denormal_f32(result) || is_denormal_f32(val)) {
            val = result;
        } else {
            val = result;
        }
    }

    data[idx] = val;
}

// ============================================================================
// CIRCLE COMPLEX IN-PLACE MULTIPLY
// ============================================================================

__device__ inline void circle_complex_mul(
    int16_t a_real, int16_t a_imag, int32_t a_exp,
    int16_t b_real, int16_t b_imag, int32_t b_exp,
    int16_t* c_real, int16_t* c_imag, int32_t* c_exp
) {
    if ((a_real == 0 && a_imag == 0) || (b_real == 0 && b_imag == 0)) {
        *c_real = 0; *c_imag = 0; *c_exp = 0;
        return;
    }

    int32_t a_r = (int32_t)a_real;
    int32_t a_i = (int32_t)a_imag;
    int32_t b_r = (int32_t)b_real;
    int32_t b_i = (int32_t)b_imag;

    int32_t real_product = (a_r * b_r >> 1) - (a_i * b_i >> 1);
    int32_t imag_product = (a_r * b_i >> 1) + (a_i * b_r >> 1);

    if (real_product == 0 && imag_product == 0) {
        *c_real = 0; *c_imag = 0; *c_exp = 0;
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

__global__ void circle_multiply_kernel(
    int16_t* real, int16_t* imag, int32_t* exp, int N, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int16_t r = real[idx];
    int16_t i = imag[idx];
    int32_t e = exp[idx];

    // Multiply by self many times
    for (int it = 0; it < iterations; it++) {
        int16_t new_r, new_i;
        int32_t new_e;
        circle_complex_mul(r, i, e, r, i, e, &new_r, &new_i, &new_e);
        r = new_r;
        i = new_i;
        e = new_e;
    }

    real[idx] = r;
    imag[idx] = i;
    exp[idx] = e;
}

// ============================================================================
// IEEE COMPLEX IN-PLACE MULTIPLY (with forced denormal divergence)
// ============================================================================

__global__ void ieee_complex_multiply_kernel(
    float* real, float* imag, int N, int iterations
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    volatile float r = real[idx];
    volatile float i = imag[idx];

    for (int it = 0; it < iterations; it++) {
        // (a+bi)(a+bi) = (a²-b²) + (2ab)i
        volatile float aa = r * r;
        volatile float bb = i * i;
        volatile float ab = r * i;

        volatile float new_r = aa - bb;
        volatile float new_i = ab + ab;

        // Force denormal checks on all intermediates
        if (is_denormal_f32(aa) || is_denormal_f32(bb) || is_denormal_f32(ab) ||
            is_denormal_f32(new_r) || is_denormal_f32(new_i)) {
            r = new_r;
            i = new_i;
        } else {
            r = new_r;
            i = new_i;
        }
    }

    real[idx] = r;
    imag[idx] = i;
}

// ============================================================================
// HOST WRAPPERS
// ============================================================================

extern "C" void bench_spirix_inplace(
    const int16_t* h_frac, const int16_t* h_exp,
    int16_t* h_out_frac, int16_t* h_out_exp,
    int N, int iterations
) {
    size_t size = N * sizeof(int16_t);

    int16_t *d_frac, *d_exp;

    hipMalloc(&d_frac, size);
    hipMalloc(&d_exp, size);

    hipMemcpy(d_frac, h_frac, size, hipMemcpyHostToDevice);
    hipMemcpy(d_exp, h_exp, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // TIME THIS: Data stays on GPU
    hipLaunchKernelGGL(spirix_multiply_kernel, numBlocks, blockSize, 0, 0,
                       d_frac, d_exp, N, iterations);

    hipMemcpy(h_out_frac, d_frac, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_out_exp, d_exp, size, hipMemcpyDeviceToHost);

    hipFree(d_frac);
    hipFree(d_exp);
}

extern "C" void bench_ieee_inplace(
    const float* h_data, float* h_out,
    int N, int iterations
) {
    size_t size = N * sizeof(float);

    float *d_data;

    hipMalloc(&d_data, size);

    hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // TIME THIS: Data stays on GPU
    hipLaunchKernelGGL(ieee_multiply_kernel, numBlocks, blockSize, 0, 0,
                       d_data, N, iterations);

    hipMemcpy(h_out, d_data, size, hipMemcpyDeviceToHost);

    hipFree(d_data);
}

extern "C" void bench_circle_inplace(
    const int16_t* h_real, const int16_t* h_imag, const int32_t* h_exp,
    int16_t* h_out_real, int16_t* h_out_imag, int32_t* h_out_exp,
    int N, int iterations
) {
    size_t size_i16 = N * sizeof(int16_t);
    size_t size_i32 = N * sizeof(int32_t);

    int16_t *d_real, *d_imag;
    int32_t *d_exp;

    hipMalloc(&d_real, size_i16);
    hipMalloc(&d_imag, size_i16);
    hipMalloc(&d_exp, size_i32);

    hipMemcpy(d_real, h_real, size_i16, hipMemcpyHostToDevice);
    hipMemcpy(d_imag, h_imag, size_i16, hipMemcpyHostToDevice);
    hipMemcpy(d_exp, h_exp, size_i32, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // TIME THIS
    hipLaunchKernelGGL(circle_multiply_kernel, numBlocks, blockSize, 0, 0,
                       d_real, d_imag, d_exp, N, iterations);

    hipMemcpy(h_out_real, d_real, size_i16, hipMemcpyDeviceToHost);
    hipMemcpy(h_out_imag, d_imag, size_i16, hipMemcpyDeviceToHost);
    hipMemcpy(h_out_exp, d_exp, size_i32, hipMemcpyDeviceToHost);

    hipFree(d_real);
    hipFree(d_imag);
    hipFree(d_exp);
}

extern "C" void bench_ieee_complex_inplace(
    const float* h_real, const float* h_imag,
    float* h_out_real, float* h_out_imag,
    int N, int iterations
) {
    size_t size = N * sizeof(float);

    float *d_real, *d_imag;

    hipMalloc(&d_real, size);
    hipMalloc(&d_imag, size);

    hipMemcpy(d_real, h_real, size, hipMemcpyHostToDevice);
    hipMemcpy(d_imag, h_imag, size, hipMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // TIME THIS
    hipLaunchKernelGGL(ieee_complex_multiply_kernel, numBlocks, blockSize, 0, 0,
                       d_real, d_imag, N, iterations);

    hipMemcpy(h_out_real, d_real, size, hipMemcpyDeviceToHost);
    hipMemcpy(h_out_imag, d_imag, size, hipMemcpyDeviceToHost);

    hipFree(d_real);
    hipFree(d_imag);
}
