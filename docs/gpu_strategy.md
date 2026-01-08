# GPU Strategy: Pure ROCm/HIP with Inline Assembly

## Why Drop OpenCL

**OpenCL (rusticl/Mesa) has fatal limitations:**
1. **Always uses FTZ mode** - flushes denormals to zero (incorrect)
2. **LLVM backend controls everything** - can't disable optimizations
3. **No inline assembly access** - stuck with compiler decisions
4. **Can't verify IEEE compliance** - black box compilation

**ROCm/HIP gives us:**
1. **Inline RDNA assembly** - complete instruction control
2. **No FTZ** - we control every optimization
3. **Direct hardware access** - no abstraction layers
4. **Verifiable** - inspect generated ISA

## Architecture

### Pure Inline Assembly for All Operations

Every Spirix operation written in RDNA ISA:

**Multiply:**
```cpp
__device__ inline void spirix_mul_asm(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t* c_frac, int16_t* c_exp
) {
    asm volatile(
        "v_cvt_i32_i16 %0, %2\n"      // Extend a_frac
        "v_cvt_i32_i16 %1, %3\n"      // Extend b_frac
        "v_mul_i32 %0, %0, %1\n"      // Multiply (pure integer)
        : "=v"(frac_product)
        : "v"(0), "v"(a_frac), "v"(b_frac)
    );
    // ... normalization in asm ...
}
```

**Add/Subtract:**
```cpp
asm volatile(
    "v_add_i32 %0, %1, %2\n"          // Add (pure integer)
    "v_sub_i32 %0, %1, %2\n"          // Subtract (pure integer)
    : "=v"(result)
    : "v"(a), "v"(b)
);
```

**Shifts:**
```cpp
asm volatile(
    "v_lshlrev_b32 %0, %1, %2\n"      // Logical shift left
    "v_ashrrev_i32 %0, %1, %2\n"      // Arithmetic shift right (sign-preserving)
    : "=v"(result)
    : "v"(shift_amount), "v"(value)
);
```

### Zero IEEE-754 Instructions

**What we USE:**
- `v_mul_i32` - Integer multiply
- `v_add_i32` / `v_sub_i32` - Integer add/subtract
- `v_lshlrev_b32` / `v_ashrrev_i32` - Bit shifts
- `v_ffbh_u32` - Find first bit (leading zeros)
- `v_cmp_*` - Integer comparisons

**What we NEVER use:**
- ❌ `v_mul_f32` - Float multiply (IEEE)
- ❌ `v_add_f32` - Float add (IEEE)
- ❌ `v_fma_f32` - Fused multiply-add (IEEE)
- ❌ `v_rcp_f32` - Reciprocal (IEEE)

### Verification

After compilation, inspect assembly:
```bash
llvm-objdump -d spirix_matmul_asm.o | grep "v_mul_f32\|v_add_f32"
# Should return ZERO matches
```

## Implementation Files

1. **spirix_matmul_asm.hip** - All operations in inline assembly
2. **build_asm.sh** - Compiles HIP, verifies zero IEEE instructions
3. **Rust FFI** - Direct binding to HIP (no OpenCL feature)

## Performance Expectations

**Phase 1 (Naive inline asm):**
- Expected: 0.2-0.3x IEEE speed (2-5x slower)
- Reason: No optimizations, but zero divergence

**Phase 2 (Optimized asm):**
- Shared memory tiling
- Register blocking
- Memory coalescing
- Expected: 0.5-1.0x IEEE speed (parity)

**Phase 3 (Production):**
- SIMD packing (pack 2x i16 in i32 registers)
- Kernel fusion
- Async transfers
- Expected: Match or beat IEEE

## Why This Matters

**Spirix maintains correctness** while IEEE cheats:
- IEEE FTZ: denormal → 0 (WRONG per spec)
- Spirix: vanished → preserved (CORRECT)

**When IEEE is compliant** (FTZ disabled):
- Branch divergence on denormals
- 200+ cycle stalls per denormal
- Spirix should WIN (zero divergence)

**Our approach:**
- Inline assembly = verifiable correctness
- Zero compiler tricks
- Predictable performance
- Suitable for verified computation

## Next Steps

1. ✅ Write inline asm for all Spirix ops (multiply, add, subtract)
2. ⏳ Build with hipcc, verify zero IEEE instructions
3. ⏳ Benchmark vs IEEE with FTZ enabled
4. ⏳ Phase 2: Add optimizations
5. ⏳ Compare against compliant IEEE (if we can find it)
