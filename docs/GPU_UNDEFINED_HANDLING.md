# GPU Undefined State Handling

## Current Situation

Our GPU implementation ([src/gpu/hip.rs](../src/gpu/hip.rs)) reconstructs Spirix scalars from raw `fraction` and `exponent` components:

```rust
// Line 92-96 in hip.rs
let c_data: Vec<ScalarF4E4> = c_frac
    .into_iter()
    .zip(c_exp.into_iter())
    .map(|(frac, exp)| ScalarF4E4 { fraction: frac, exponent: exp })
    .collect();
```

**Issue**: This bypasses Spirix's undefined state validation and constant system.

## Spirix Undefined System

Spirix has **30+ specific undefined variants** defined in [spirix/src/core/undefined.rs](../../spirix/src/core/undefined.rs):

### Example Undefined States:
- `℘ ⬆+⬆` - Transfinite + Transfinite (0b00011111)
- `℘ ⬆-⬆` - Transfinite - Transfinite (0b11100000)
- `℘ ⬇/⬇` - Negligible / Negligible (0b11101001) ← **This is 0/0**
- `℘ ⬆/⬆` - Transfinite / Transfinite (0b00010110)
- `℘ ⬆×⬇` - Transfinite × Negligible (0b11101111)
- `℘ √-` - Square root of negative (0b11110110)
- `℘` - General undefined (0b11111110) ← **IEEE NaN maps here**

Each has a **specific bit pattern prefix** in the fraction component that identifies the exact cause of the undefined state.

## Question: Generic vs Specific Undefined in GPU?

### Option 1: Use Generic Undefined (Current Risk)

**What happens now:**
```rust
// GPU kernel produces 0/0
fraction = 0b11101001_xxxxxxxx  // Some random bits
exponent = 0b10000000_00000000  // AMBIGUOUS_EXPONENT

// We reconstruct directly - may not match canonical ℘ ⬇/⬇ pattern
ScalarF4E4 { fraction, exponent }
```

**Risk**: GPU might produce non-canonical undefined patterns that Spirix doesn't recognize.

### Option 2: Validate and Normalize (Recommended)

**Approach**: After GPU computation, validate results and map to correct undefined constants:

```rust
// In hip.rs reconstruction
let c_data: Vec<ScalarF4E4> = c_frac
    .into_iter()
    .zip(c_exp.into_iter())
    .map(|(frac, exp)| {
        let scalar = ScalarF4E4 { fraction: frac, exponent: exp };

        // Check if undefined and validate
        if scalar.is_undefined() {
            // Optionally: normalize to canonical undefined pattern
            // For now, trust GPU kernel to produce correct patterns
            scalar
        } else {
            scalar
        }
    })
    .collect();
```

### Option 3: Import Undefined Constants in GPU Kernel

**Best approach**: Make GPU kernel aware of Spirix undefined constants.

**In HIP/CUDA kernel:**
```cpp
// spirix_constants.h
#define UNDEFINED_NEGLIGIBLE_DIV_NEGLIGIBLE 0b11101001  // ℘ ⬇/⬇ (0/0)
#define UNDEFINED_TRANSFINITE_DIV_TRANSFINITE 0b00010110  // ℘ ⬆/⬆ (∞/∞)
#define UNDEFINED_GENERAL 0b11111110  // ℘ (generic)

// In kernel when detecting 0/0:
if (divisor_is_zero && dividend_is_zero) {
    result.fraction = UNDEFINED_NEGLIGIBLE_DIV_NEGLIGIBLE;
    result.exponent = AMBIGUOUS_EXPONENT;
}
```

## How Spirix Handles Undefined Natively

Spirix **operators** handle undefined propagation automatically:

```rust
// From spirix/src/implementations/division/scalar_scalar.rs
impl Div<Scalar<F, E>> for Scalar<F, E> {
    fn div(self, rhs: Scalar<F, E>) -> Self::Output {
        // Check for undefined propagation
        if self.is_undefined() { return self; }
        if rhs.is_undefined() { return rhs; }

        // Check for 0/0 → produce ℘ ⬇/⬇
        if self.is_zero() && rhs.is_zero() {
            return Scalar {
                fraction: NEGLIGIBLE_DIVIDE_NEGLIGIBLE.prefix,
                exponent: AMBIGUOUS_EXPONENT,
            };
        }

        // Normal division...
    }
}
```

## Current Test Results

Our edge case tests ([examples/test_edge_cases.rs](../examples/test_edge_cases.rs)) show:

✅ **CPU operations preserve undefined correctly:**
- 0/0 produces undefined (℘ ⬇/⬇)
- Undefined propagates through `add()`, `multiply()`, `transpose()`, `scale()`
- All 30+ undefined variants can be stored in tensors

⚠️ **GPU operations untested for undefined:**
- No tests yet for GPU handling of 0/0, ∞/∞, ∞-∞, etc.
- No validation that GPU produces canonical undefined patterns

## Recommendations

### Immediate (Required):
1. **Test GPU undefined handling** - Create test that checks if GPU 0/0 produces correct undefined
2. **Document GPU kernel undefined handling** - Verify HIP kernel produces canonical patterns
3. **Add validation layer** - Check GPU results for undefined and normalize if needed

### Short-term (Important):
1. **Import Spirix constants to GPU** - Share undefined prefix definitions with HIP/CUDA
2. **Unified undefined detection** - GPU should match CPU undefined patterns exactly
3. **Test all edge cases on GPU** - Verify ∞, vanished, exploded, all 30+ undefined variants

### Long-term (Nice to have):
1. **Spirix GPU library** - Move Spirix arithmetic entirely to GPU-native implementation
2. **Kernel-level undefined tracking** - Track first cause of undefined through computation graph
3. **Undefined visualization** - Debug tools to show which operation caused undefined

## Example: Testing GPU Undefined

```rust
#[test]
fn test_gpu_handles_zero_div_zero() {
    // Create matrix with zeros
    let zero_mat = Tensor::from_scalars(
        vec![ScalarF4E4::ZERO; 4],
        Shape::matrix(2, 2)
    ).unwrap();

    // GPU matmul: 0 * anything = 0, but intermediate sum might hit 0/0
    let result = matmul_gpu(&zero_mat, &zero_mat);

    // Check result is either ZERO or undefined (not garbage)
    for val in result.as_scalars().unwrap() {
        assert!(val.is_zero() || val.is_undefined(),
                "GPU produced invalid result: {:?}", val);
    }
}
```

## Decision Needed

**Question for Nick**: Should we:

A. **Trust GPU kernel** - Assume HIP kernel produces correct undefined patterns (current)
B. **Validate in Rust** - Add validation layer after GPU computation
C. **Import constants** - Share Spirix undefined definitions with GPU kernel code
D. **All of the above** - Validate now, import constants for future kernels

My recommendation: **Start with B (validate in Rust)**, then move to **C (import constants)** once we have more GPU kernels.
