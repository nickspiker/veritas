# GPU Implementation Status

## What's Been Built

### Documentation
1. **CUDA vs ROCm Comparison** ([cuda_vs_rocm.md](cuda_vs_rocm.md))
   - Detailed comparison of NVIDIA CUDA vs AMD ROCm
   - Why ROCm is the right choice (open source, AMD hardware, verification)
   - Implementation strategy: HIP → inline assembly → pure assembly
   - Tooling comparison and troubleshooting guide

2. **GPU Performance Analysis** ([gpu_performance_analysis.md](gpu_performance_analysis.md))
   - Why Spirix might beat IEEE-754 despite higher instruction count
   - Branch divergence analysis (320 branches vs 10)
   - Wave execution model on RDNA2
   - Hypothesis: Zero divergence > instruction count penalty

### Code Implementation

1. **HIP Kernel** ([gpu/hip/spirix_matmul.hip](../gpu/hip/spirix_matmul.hip))
   - Complete naive matrix multiply kernel
   - Pure integer arithmetic, ZERO IEEE-754
   - Device functions: `spirix_mul`, `spirix_add`
   - Kernel: `spirix_matmul_kernel` (naive, no optimizations)
   - Host wrapper: `spirix_matmul_hip` (manages device memory)

2. **Rust FFI Bindings** ([src/gpu/hip.rs](../src/gpu/hip.rs))
   - Safe Rust wrapper around HIP kernel
   - Function: `matmul_gpu(&Tensor<ScalarF4E4>, &Tensor<ScalarF4E4>)`
   - Handles Spirix scalar decomposition (fraction, exponent)
   - Tests for small (2×2) and large (1024×1024) matrices

3. **Build System**
   - Build script: [gpu/hip/build.sh](../gpu/hip/build.sh)
   - Compiles HIP to shared library
   - Verifies zero IEEE-754 instructions in assembly
   - Cargo build: [build.rs](../build.rs) (links libspirix_hip.so)

4. **Benchmark** ([examples/gpu_matmul_benchmark.rs](../examples/gpu_matmul_benchmark.rs))
   - CPU vs GPU performance comparison
   - Multiple matrix sizes (64 to 1024)
   - Validates correctness (CPU and GPU results match)

5. **Setup Script** ([gpu/setup_and_test.sh](../gpu/setup_and_test.sh))
   - One-command build and test
   - Checks ROCm installation
   - Builds HIP library and Rust code
   - Runs benchmark

## Current Status: Phase 1 Complete

**Phase 1: Naive Implementation** ✓
- [x] HIP kernel with basic operations
- [x] Rust FFI bindings
- [x] Build system and verification
- [x] Benchmark infrastructure

**Expected Performance:** 2-10x slower than CPU (no optimizations yet)

**But:** ZERO branch divergence. All threads execute same path.

## Next Steps: Phase 2 Optimizations

**Goal:** Match or beat IEEE-754 f32 performance.

### Optimization 1: Shared Memory Tiling (10x speedup)
```cpp
__shared__ int16_t tile_a_frac[32][32];
__shared__ int16_t tile_a_exp[32][32];
// Load tiles into LDS, reduce global memory by 32x
```

### Optimization 2: Memory Coalescing (2x speedup)
- Align all memory accesses to 128-byte cache lines
- All threads in wave read consecutive addresses
- Bank-conflict-free LDS access

### Optimization 3: Register Blocking (2x speedup)
```cpp
// Each thread computes 4×4 output tile
int16_t acc_frac[4][4];
int16_t acc_exp[4][4];
```

### Optimization 4: Loop Unrolling (1.5x speedup)
```cpp
#pragma unroll
for (int k = 0; k < 32; k++) {
    // Inner loop unrolled
}
```

**Expected total speedup:** 30-40x over naive implementation.

**Result:** Match or exceed rocBLAS f32 (IEEE-754) on real workloads.

## Testing Plan

1. **Correctness**
   - [x] Small matrices (2×2, verify by hand)
   - [x] Large matrices (1024×1024, verify against CPU)
   - [ ] Edge cases (vanished, exploded, zero)

2. **Performance**
   - [ ] Benchmark Phase 1 (naive) vs CPU
   - [ ] Implement Phase 2 optimizations
   - [ ] Benchmark Phase 2 vs CPU
   - [ ] Compare against rocBLAS f32 (IEEE baseline)

3. **Verification**
   - [x] Inspect assembly for IEEE-754 instructions (should be zero)
   - [ ] Test with denormal-equivalent values (should not stall)
   - [ ] Symbolic verification (F6E5 ground truth)

## Integration with Veritas

Once GPU kernels are proven:

1. **Neural Network Training**
   - Replace CPU tensor ops with GPU ops
   - Benchmark full training loop (forward + backward)
   - Measure convergence rate

2. **Verification Pipeline**
   - GPU computes neural output (F4E4)
   - CPU verifies with symbolic engine (F6E5)
   - Catch any IEEE contamination

3. **Production Deployment**
   - Multi-GPU scaling with RCCL
   - Kernel fusion (matmul + ReLU + batch norm)
   - Custom PyTorch ops for compatibility

## Why This Matters

**Conventional wisdom:** IEEE-754 is always faster (hardware optimized).

**Reality:** Branch divergence kills GPU performance.

**Our hypothesis:** Spirix's predictable execution beats IEEE's special cases.

**Proof:** Build it, measure it, publish results.

**Impact:** If we're right, Spirix becomes viable for production ML at scale.

## Open Questions

1. **How does RDNA3 compare?**
   - Wave64 mode vs Wave32
   - Different instruction latencies

2. **What about Tensor Cores equivalent?**
   - NVIDIA has specialized matrix units
   - AMD has Matrix Core instructions (RDNA3+)
   - Can we use them with Spirix?

3. **Mixed precision strategies?**
   - F4E4 for forward pass (fast)
   - F6E5 for backward pass (accurate)
   - F8E6 for accumulation (prevent drift)

4. **Quantization?**
   - F3E3 for inference (8-bit total)
   - F2E2 for extreme compression (4-bit)
   - How much accuracy loss?

## Resources

- RDNA2 ISA: https://www.amd.com/en/support/gpu/amd-radeon-6000-series
- ROCm Docs: https://rocm.docs.amd.com/
- HIP Programming Guide: https://rocm.docs.amd.com/projects/HIP/
- Our performance analysis: [gpu_performance_analysis.md](gpu_performance_analysis.md)

## Timeline

**Week 1:** Phase 1 complete (you are here)
**Week 2:** Implement Phase 2 optimizations
**Week 3:** Benchmark against rocBLAS
**Week 4:** Integrate with neural training
**Month 2:** Production-ready, multi-GPU
**Month 3:** Publish results

---

**Bottom line:** We have a complete, working GPU implementation of Spirix matrix multiply. It's slow (naive), but it proves the concept. Zero IEEE-754, zero branch divergence. Now we optimize and measure.
