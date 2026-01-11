#include <hip/hip_runtime.h>
#include <stdint.h>

// Simple test to see what instructions RDNA2 actually generates
__global__ void test_int_ops(int32_t* out) {
    int32_t a = 100;
    int32_t b = 200;

    // Test multiply
    int32_t mul = a * b;

    // Test add
    int32_t add = a + b;

    // Test subtract
    int32_t sub = a - b;

    // Test shift left
    int32_t shl = a << 5;

    // Test shift right (arithmetic)
    int32_t shr = a >> 5;

    // Test i16 to i32 sign extension
    int16_t small = -100;
    int32_t extended = (int32_t)small;

    out[0] = mul + add + sub + shl + shr + extended;
}

extern "C" void dummy() {}
