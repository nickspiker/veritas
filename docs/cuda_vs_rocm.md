# CUDA vs ROCm: Platform Comparison for Spirix GPU Kernels

## TL;DR

**CUDA (NVIDIA):**
- Proprietary, mature, excellent tooling
- Can't use it - you have AMD hardware

**ROCm (AMD):**
- Open source, improving rapidly
- Required for your RDNA2 GPU
- HIP layer provides CUDA-like API
- Lower-level access via assembly if needed

**Recommendation:** Use ROCm with HIP (high-level), drop to GCN/RDNA assembly for critical kernels.

---

## Platform Overview

### CUDA (NVIDIA)

**Pros:**
- **Mature ecosystem** - 15+ years of development
- **Better tooling** - nvprof, Nsight, cuda-gdb all excellent
- **Extensive libraries** - cuBLAS, cuDNN, Thrust, etc.
- **Better documentation** - Comprehensive guides and examples
- **Industry standard** - Most ML frameworks target CUDA first
- **Stable ABI** - Kernels compile once, run across GPU generations

**Cons:**
- **Proprietary** - Vendor lock-in, closed source runtime
- **NVIDIA only** - Can't run on AMD/Intel GPUs
- **Expensive hardware** - NVIDIA GPUs cost 2-3x AMD equivalents
- **Licensing restrictions** - Can't reverse engineer or extend
- **You don't have NVIDIA hardware** - Deal breaker

**Technical Details:**
- Programming model: Grids → Blocks → Warps (32 threads)
- Memory hierarchy: Global, Shared, Local, Constant, Texture
- ISA: PTX (intermediate) → SASS (native)
- Language: CUDA C/C++ with nvcc compiler

---

### ROCm (AMD)

**Pros:**
- **Open source** - Full stack from compiler to runtime
- **You have AMD hardware** - RDNA2 is supported
- **HIP compatibility layer** - Port CUDA code with ~95% compatibility
- **Lower-level access** - Can write raw GCN/RDNA assembly
- **Better for research** - Full visibility into what's happening
- **Cheaper hardware** - AMD GPUs are 40-60% cheaper than NVIDIA
- **Active development** - AMD investing heavily, improving fast

**Cons:**
- **Less mature** - ROCm 6.x still has rough edges
- **Worse tooling** - rocprof exists but not as polished as nvprof
- **Smaller ecosystem** - Fewer libraries, less community support
- **Documentation gaps** - Some things underdocumented or AMD-internal
- **Framework support** - PyTorch/TensorFlow support AMD but CUDA is primary target
- **Version churn** - ROCm breaking changes between versions

**Technical Details:**
- Programming model: Grids → Workgroups → Waves (32/64 threads depending on arch)
- RDNA2 wave size: 32 threads (same as CUDA warp)
- Memory hierarchy: Global, LDS (local data share), Private, Constant
- ISA: AMDGCN (GCN5 for older, RDNA for newer)
- Languages: HIP (CUDA-like), OpenCL, or raw assembly

---

## For Spirix: Why ROCm Wins

### 1. You Have AMD Hardware
RDNA2 GPU = must use ROCm. End of discussion.

### 2. Open Source Enables Verification
- Can audit entire stack (compiler, runtime, driver)
- Critical for verified computation
- Can prove no IEEE-754 sneaking in

### 3. Lower-Level Control
- Can write raw RDNA assembly for critical kernels
- CUDA hides ISA details, ROCm exposes them
- Important for Spirix (need control over exact instructions)

### 4. Simpler Instruction Set
RDNA2 integer ops are clean and documented:
```asm
v_mul_i32       ; 32-bit multiply
v_lshl_b32      ; Logical shift left
v_add_i32       ; 32-bit add
v_cmp_eq_i32    ; Compare equal
```

No hidden IEEE-754 nonsense in integer paths.

---

## Implementation Strategy

### Phase 1: HIP (High-Level)

Use HIP API (CUDA-like) for initial implementation:

```cpp
// HIP kernel (looks like CUDA)
__global__ void spirix_matmul(
    const int16_t* a_frac, const int16_t* a_exp,
    const int16_t* b_frac, const int16_t* b_exp,
    int16_t* c_frac, int16_t* c_exp,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        // Spirix accumulation (no IEEE-754)
        int32_t acc_frac = 0;
        int32_t acc_exp = 0;

        for (int k = 0; k < K; k++) {
            // Spirix multiply-add
            // (implementation details...)
        }

        c_frac[row * N + col] = (int16_t)acc_frac;
        c_exp[row * N + col] = (int16_t)acc_exp;
    }
}
```

**Advantages:**
- Familiar CUDA-like API
- Easy to port from CPU code
- Good starting point for optimization

**Limitations:**
- Compiler might make assumptions (IEEE behavior)
- Can't control exact instruction sequence
- May not be optimal for Spirix

---

### Phase 2: Inline Assembly (Critical Paths)

Drop to RDNA assembly for critical operations:

```cpp
__device__ inline void spirix_mul_asm(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    int32_t result_frac, result_exp;

    asm volatile(
        "v_mul_i32 %0, %2, %3\n"      // Multiply fractions
        "v_add_i32 %1, %4, %5\n"      // Add exponents
        : "=v"(result_frac), "=v"(result_exp)
        : "v"(a_frac), "v"(b_frac), "v"(a_exp), "v"(b_exp)
    );

    *c_frac = (int16_t)result_frac;
    *c_exp = (int16_t)result_exp;
}
```

**Advantages:**
- Exact control over instructions
- Guarantee no IEEE-754 instructions used
- Can optimize for zero divergence

**Challenges:**
- More complex to write and maintain
- ISA documentation sometimes sparse
- Need to understand RDNA microarchitecture

---

### Phase 3: Pure Assembly Kernels

For maximum performance, write entire kernels in assembly:

```asm
.text
.globl spirix_matmul_kernel
.type spirix_matmul_kernel,@function

spirix_matmul_kernel:
    ; Calculate thread index
    v_mov_b32 v0, s0          ; blockIdx.x
    v_mov_b32 v1, s1          ; threadIdx.x

    ; Load A and B tiles into LDS
    ; (shared memory)

    ; Inner loop: accumulate products
.L_inner_loop:
    v_mov_b32 v2, [a_frac]    ; Load a fraction
    v_mov_b32 v3, [a_exp]     ; Load a exponent
    v_mov_b32 v4, [b_frac]    ; Load b fraction
    v_mov_b32 v5, [b_exp]     ; Load b exponent

    v_mul_i32 v6, v2, v4      ; Multiply fractions
    v_add_i32 v7, v3, v5      ; Add exponents

    ; Normalize and accumulate
    ; ...

    s_cbranch_scc1 .L_inner_loop

    ; Write result
    ; ...

    s_endpgm
```

**Advantages:**
- Maximum control and performance
- Zero wasted instructions
- Can prove no IEEE-754 contamination

**Challenges:**
- Significant development effort
- Hard to debug
- Fragile across GPU generations

---

## Tooling Comparison

### CUDA Tools

**nvcc** - Compiler
- Excellent error messages
- Good optimization
- Stable across versions

**nvprof** - Profiler
- Detailed performance metrics
- Memory transfer analysis
- Kernel occupancy

**cuda-gdb** - Debugger
- Full debugging support
- Inspect registers, memory
- Conditional breakpoints

**Nsight** - IDE
- Visual profiler
- GPU trace
- Memory checker

---

### ROCm Tools

**hipcc** - Compiler
- Wraps clang for HIP code
- Decent error messages
- Improving optimization

**rocprof** - Profiler
- Basic performance metrics
- Works but not as polished
- Improving with each release

**rocgdb** - Debugger
- Based on gdb
- Functional but limited
- Can inspect GPU state

**ROCm Tracer** - Tracing
- API call tracing
- Kernel execution timeline
- Memory transfers

**radeon_profile** - GUI profiler
- Open source
- GPU utilization
- Power/temp monitoring

---

## For Veritas: ROCm Strategy

### Near-term (Next 2 weeks)
1. **HIP kernels** for basic operations (matmul, relu, etc.)
2. **rocprof** to find bottlenecks
3. **Verify zero IEEE-754** by inspecting generated assembly

### Mid-term (1-2 months)
1. **Inline assembly** for critical paths (Spirix multiply, normalize)
2. **Optimize memory access** (coalescing, LDS usage)
3. **Benchmark** against cuBLAS f32 on NVIDIA (borrow GPU or cloud)

### Long-term (3-6 months)
1. **Pure assembly kernels** for matmul (if needed)
2. **Kernel fusion** (matmul + ReLU in one kernel)
3. **Multi-GPU** scaling with RCCL (ROCm collective comms)

---

## Expected Performance

Based on GPU performance analysis:

**IEEE-754 f32 (theoretical):**
- Peak: 27 TFLOPs
- Actual (training): 1.6 TFLOPs (branch divergence)

**Spirix F4E4 (hypothesis):**
- Peak: ~1.8 TFLOPs (higher instruction count)
- Actual (training): 1.1-1.5 TFLOPs (zero divergence)

**Goal:** Match or exceed IEEE-754 on real workloads.

**How:**
1. Zero branch divergence → all 32 threads execute same path
2. No denormal handling → no 200-cycle stalls
3. Predictable memory access → better prefetching
4. SIMD vectorization → pack multiple i16 ops in single instruction

---

## Verification Strategy

**Critical requirement:** Prove no IEEE-754 in kernels.

### Method 1: Inspect Assembly
```bash
hipcc --genco spirix_kernel.hip
llvm-objdump -d spirix_kernel.co

# Look for IEEE instructions:
# v_mul_f32, v_add_f32, v_fma_f32 → RED FLAG
# v_mul_i32, v_add_i32, v_lshl_b32 → GOOD
```

### Method 2: Test Edge Cases
- Denormals → should vanish, not stall
- NaN → should be undefined, not propagate
- Infinity → should explode, not saturate

### Method 3: Symbolic Verification
- Run kernel with known inputs
- Verify output matches F6E5 symbolic computation
- Catch any IEEE contamination

---

## Conclusion

**ROCm is the right choice because:**
1. You have AMD hardware (no choice)
2. Open source enables verification
3. Lower-level access for Spirix
4. Cheaper to scale

**Implementation path:**
1. Start with HIP (familiar, fast to develop)
2. Profile and optimize (rocprof)
3. Drop to assembly for critical kernels
4. Verify zero IEEE-754 throughout

**Next step:** Implement basic HIP kernel for Spirix matmul and measure actual performance.

---

## Resources

**ROCm Documentation:**
- https://rocm.docs.amd.com/
- HIP Programming Guide
- RDNA ISA Reference

**RDNA2 Architecture:**
- "RDNA2 Instruction Set Architecture" (AMD)
- Wave64 vs Wave32 modes
- LDS (Local Data Share) usage

**Community:**
- ROCm GitHub: https://github.com/RadeonOpenCompute/ROCm
- HIP GitHub: https://github.com/ROCm-Developer-Tools/HIP
- r/ROCm (for debugging help)

**Benchmarking:**
- rocBLAS source code (reference implementations)
- Compare Spirix against rocBLAS f32 GEMM
