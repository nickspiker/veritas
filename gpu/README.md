# Spirix GPU Acceleration

GPU kernels for Spirix operations using ROCm/HIP.

**Goal:** Prove that Spirix F4E4 can match or exceed IEEE-754 f32 performance on real workloads, despite higher instruction count, due to zero branch divergence.

## Status

- [x] Phase 1: Naive HIP kernel (this directory)
- [ ] Phase 2: Optimized kernel (shared memory, coalescing)
- [ ] Phase 3: Production kernel (fusion, async)

## Hardware Requirements

- AMD GPU with ROCm support
- Tested on: RDNA2 (gfx1030)
- ROCm 5.0+ recommended

## Build Instructions

### 1. Install ROCm

Follow AMD's instructions: https://rocm.docs.amd.com/

Quick check:
```bash
hipcc --version
rocminfo
```

### 2. Build HIP Kernels

```bash
cd gpu/hip
./build.sh
```

This compiles `spirix_matmul.hip` to `libspirix_hip.so`.

**Verification:** The build script checks that ZERO IEEE-754 instructions are in the compiled kernel. Only integer ops allowed.

### 3. Build Rust Code

```bash
cargo build --release
```

The `build.rs` script automatically links against `libspirix_hip.so`.

### 4. Run Benchmark

```bash
export LD_LIBRARY_PATH=$PWD/gpu/hip:$LD_LIBRARY_PATH
cargo run --release --example gpu_matmul_benchmark
```

## Expected Performance (Phase 1)

**Naive kernel** (no optimizations):
- 64×64: ~2x slower than CPU
- 1024×1024: ~10x slower than CPU

**Why slow?**
- No shared memory (thrashes global memory)
- No memory coalescing (unaligned reads)
- Each thread does one element (no register blocking)

**But:** ZERO branch divergence. All threads execute same path.

## Phase 2: Optimizations

Next steps to reach parity with IEEE-754:

1. **Shared memory tiling** (10x speedup)
   - Load 32×32 tiles into LDS
   - Reduce global memory traffic by 32x

2. **Memory coalescing** (2x speedup)
   - Align access patterns to 128-byte cache lines
   - All threads in wave read consecutive addresses

3. **Register blocking** (2x speedup)
   - Each thread computes 4×4 output tile
   - Better instruction-level parallelism

4. **Loop unrolling** (1.5x speedup)
   - Compiler hint or manual unroll
   - Reduce branch overhead

**Expected result:** Match or beat rocBLAS f32 (IEEE-754) on training workloads.

## Verification

To verify ZERO IEEE-754 in kernels:

```bash
cd gpu/hip
llvm-objdump -d spirix_matmul.o > asm.txt

# Should find ZERO:
grep "v_mul_f32\|v_add_f32\|v_fma_f32" asm.txt

# Should find MANY:
grep "v_mul_i32\|v_add_i32\|v_lshl_b32" asm.txt
```

## Troubleshooting

**Error: "hipcc not found"**
- Install ROCm: https://rocm.docs.amd.com/

**Error: "HSA_STATUS_ERROR_INVALID_ISA"**
- Wrong GPU architecture. Check with `rocminfo`, update `--offload-arch` in build.sh

**Error: "libspirix_hip.so not found"**
- Build HIP library first: `cd gpu/hip && ./build.sh`
- Add to path: `export LD_LIBRARY_PATH=$PWD/gpu/hip:$LD_LIBRARY_PATH`

**Results don't match CPU**
- Check for overflow in normalization (Phase 1 has simplified logic)
- Verify input data is same for CPU and GPU

## Performance Analysis

See [../docs/gpu_performance_analysis.md](../docs/gpu_performance_analysis.md) for detailed analysis of why Spirix might be faster than IEEE-754 on GPU.

Key insight: **Branch divergence kills performance more than instruction count.**

IEEE-754:
- 10 instructions per op (best case)
- 320 branches → frequent divergence
- Denormals stall for 200+ cycles

Spirix:
- 50 instructions per op (all cases)
- 10 branches → minimal divergence
- No denormals → no stalls

**Result:** On real training workloads with 15% divergence rate, Spirix should match or exceed IEEE.

## Next Steps

1. Implement Phase 2 optimizations
2. Benchmark against rocBLAS f32 GEMM
3. Test on actual neural network training (backprop)
4. Scale to multi-GPU with RCCL
5. Integrate with PyTorch via custom ops

## Architecture Notes

**RDNA2 specifics:**
- Wave size: 32 threads (lockstep execution)
- LDS (shared memory): 64 KB per workgroup
- L0 cache: 16 KB per CU (vector cache)
- L1 cache: 128 KB (shared across 2 CUs)
- Memory bandwidth: 448 GB/s

**Optimal configuration for 1024×1024:**
- Block size: 16×16 (256 threads)
- Shared tile: 32×32 (4 KB per tile)
- Register blocking: 2×2 per thread
- Waves per CU: 8-16 (occupancy target)

**Memory access pattern:**
- Coalesced reads: 128-byte aligned
- Bank conflict free: Stride access in LDS
- Prefetch next tile while computing current
