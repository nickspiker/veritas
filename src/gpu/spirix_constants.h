// Spirix Constants for GPU Kernels
//
// Extracted from spirix/src/core/undefined.rs
// All undefined prefixes for GPU detection and creation
//
// CRITICAL: These must stay in sync with Spirix Rust implementation

#ifndef SPIRIX_CONSTANTS_H
#define SPIRIX_CONSTANTS_H

#include <stdint.h>

// ═══ Special Exponent Values ═══

// Ambiguous exponent marker (used for Zero, Infinity, undefined, exploded, vanished)
#define SPIRIX_AMBIGUOUS_EXPONENT ((int16_t)0b1000000000000000)

// ═══ Level 0: Zero and Infinity ═══

#define SPIRIX_ZERO_FRACTION      ((int16_t)0b0000000000000000)
#define SPIRIX_INFINITY_FRACTION  ((int16_t)0b1111111111111111)

// ═══ Level 2: Vanished (Effectively Zero) ═══

#define SPIRIX_VANISHED_POS_PREFIX ((int16_t)0b0010000000000000)  // □□■xxxxx Positive vanished [↓]
#define SPIRIX_VANISHED_NEG_PREFIX ((int16_t)0b1101111111111111)  // ■■□xxxxx Negative vanished [↓]

// ═══ Level 3: Basic Operators (Most Common Undefined States) ═══

// These are the ones we MUST handle correctly in GPU kernels

#define SPIRIX_UNDEF_TRANSFINITE_PLUS_TRANSFINITE  ((int16_t)0b0001111100000000)  // □□□■■■■■ ℘ ⬆+⬆
#define SPIRIX_UNDEF_TRANSFINITE_MINUS_TRANSFINITE ((int16_t)0b1110000000000000)  // ■■■□□□□□ ℘ ⬆-⬆
#define SPIRIX_UNDEF_VANISHED_PLUS_VANISHED        ((int16_t)0b0001111000000000)  // □□□■■■■□ ℘ ↓+↓
#define SPIRIX_UNDEF_VANISHED_MINUS_VANISHED       ((int16_t)0b1110000100000000)  // ■■■□□□□■ ℘ ↓-↓

#define SPIRIX_UNDEF_TRANSFINITE_PLUS_FINITE       ((int16_t)0b0001110000000000)  // □□□■■■□□ ℘ ⬆+
#define SPIRIX_UNDEF_TRANSFINITE_MINUS_FINITE      ((int16_t)0b1110001100000000)  // ■■■□□□■■ ℘ ⬆-

#define SPIRIX_UNDEF_FINITE_PLUS_TRANSFINITE       ((int16_t)0b0001100000000000)  // □□□■■□□□ ℘ +⬆
#define SPIRIX_UNDEF_FINITE_MINUS_TRANSFINITE      ((int16_t)0b1110011100000000)  // ■■■□□■■■ ℘ -⬆

// Division undefined states (0/0, ∞/∞)
#define SPIRIX_UNDEF_TRANSFINITE_DIV_TRANSFINITE   ((int16_t)0b0001011000000000)  // □□□■□■■□ ℘ ⬆/⬆
#define SPIRIX_UNDEF_NEGLIGIBLE_DIV_NEGLIGIBLE     ((int16_t)0b1110100100000000)  // ■■■□■□□■ ℘ ⬇/⬇ (0/0)

// Multiplication undefined states (∞×0)
#define SPIRIX_UNDEF_NEGLIGIBLE_MUL_TRANSFINITE    ((int16_t)0b0001000000000000)  // □□□■□□□□ ℘ ⬇×⬆
#define SPIRIX_UNDEF_TRANSFINITE_MUL_NEGLIGIBLE    ((int16_t)0b1110111100000000)  // ■■■□■■■■ ℘ ⬆×⬇

// Indeterminate sign/direction
#define SPIRIX_UNDEF_SIGN_INDETERMINATE            ((int16_t)0b1110010000000000)  // ■■■□□■□□ ℘ ±∅

// ═══ Level 4: Powers and Roots ═══

#define SPIRIX_UNDEF_TRANSFINITE_POWER             ((int16_t)0b0000111100000000)  // □□□□■■■■ ℘ ⬆^
#define SPIRIX_UNDEF_NEGLIGIBLE_POWER              ((int16_t)0b1111000000000000)  // ■■■■□□□□ ℘ ⬇^
#define SPIRIX_UNDEF_POWER_TRANSFINITE             ((int16_t)0b0000111000000000)  // □□□□■■■□ ℘ ^⬆
#define SPIRIX_UNDEF_POWER_NEGLIGIBLE              ((int16_t)0b1111000100000000)  // ■■■■□□□■ ℘ ^⬇
#define SPIRIX_UNDEF_NEGATIVE_POWER                ((int16_t)0b0000110100000000)  // □□□□■■□■ ℘ -^
#define SPIRIX_UNDEF_SQRT_NEGATIVE                 ((int16_t)0b1111011000000000)  // ■■■■□■■□ ℘ √-

// ═══ Level 7: General ═══

// Generic undefined (IEEE NaN maps here)
#define SPIRIX_UNDEF_GENERAL                       ((int16_t)0b1111111000000000)  // ■■■■■■■□ ℘

// ═══ Detection Helpers ═══

// Check if a value is zero
static inline int spirix_is_zero(int16_t fraction, int16_t exponent) {
    return fraction == SPIRIX_ZERO_FRACTION && exponent == SPIRIX_AMBIGUOUS_EXPONENT;
}

// Check if a value is infinity
static inline int spirix_is_infinity(int16_t fraction, int16_t exponent) {
    return fraction == SPIRIX_INFINITY_FRACTION && exponent == SPIRIX_AMBIGUOUS_EXPONENT;
}

// Check if a value is undefined (any prefix starting with 000 or 111 except zero/inf)
static inline int spirix_is_undefined(int16_t fraction, int16_t exponent) {
    if (exponent != SPIRIX_AMBIGUOUS_EXPONENT) return 0;

    // Check for level 3+ undefined patterns (top 3 bits are 000 or 111, but not all 0s or all 1s)
    uint16_t top3 = (uint16_t)fraction >> 13;
    return (top3 == 0b000 || top3 == 0b111) &&
           fraction != SPIRIX_ZERO_FRACTION &&
           fraction != SPIRIX_INFINITY_FRACTION;
}

// Check if a value is finite (not zero, not infinity, not undefined)
static inline int spirix_is_finite(int16_t fraction, int16_t exponent) {
    if (exponent == SPIRIX_AMBIGUOUS_EXPONENT) {
        return !spirix_is_zero(fraction, exponent) &&
               !spirix_is_infinity(fraction, exponent) &&
               !spirix_is_undefined(fraction, exponent);
    }
    return 1;  // Normal exponent = finite
}

// ═══ Creation Helpers ═══

// Create zero
static inline void spirix_create_zero(int16_t *fraction, int16_t *exponent) {
    *fraction = SPIRIX_ZERO_FRACTION;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
}

// Create infinity (positive or negative based on sign)
static inline void spirix_create_infinity(int16_t *fraction, int16_t *exponent, int positive) {
    *fraction = SPIRIX_INFINITY_FRACTION;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
    // Note: Infinity sign is determined by other means in Spirix
}

// Create undefined (generic)
static inline void spirix_create_undefined(int16_t *fraction, int16_t *exponent) {
    *fraction = SPIRIX_UNDEF_GENERAL;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
}

// Create specific undefined for division by zero (0/0)
static inline void spirix_create_undefined_zero_div_zero(int16_t *fraction, int16_t *exponent) {
    *fraction = SPIRIX_UNDEF_NEGLIGIBLE_DIV_NEGLIGIBLE;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
}

// Create specific undefined for infinity minus infinity
static inline void spirix_create_undefined_inf_minus_inf(int16_t *fraction, int16_t *exponent) {
    *fraction = SPIRIX_UNDEF_TRANSFINITE_MINUS_TRANSFINITE;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
}

// Create specific undefined for infinity times zero
static inline void spirix_create_undefined_inf_times_zero(int16_t *fraction, int16_t *exponent) {
    *fraction = SPIRIX_UNDEF_TRANSFINITE_MUL_NEGLIGIBLE;
    *exponent = SPIRIX_AMBIGUOUS_EXPONENT;
}

// ═══ Common Patterns for GPU Kernels ═══

// Detect and handle division edge cases
static inline void spirix_gpu_safe_divide(
    int16_t a_frac, int16_t a_exp,
    int16_t b_frac, int16_t b_exp,
    int16_t *result_frac, int16_t *result_exp
) {
    int a_is_zero = spirix_is_zero(a_frac, a_exp);
    int b_is_zero = spirix_is_zero(b_frac, b_exp);
    int a_is_inf = spirix_is_infinity(a_frac, a_exp);
    int b_is_inf = spirix_is_infinity(b_frac, b_exp);

    // 0 / 0 = ℘ ⬇/⬇
    if (a_is_zero && b_is_zero) {
        spirix_create_undefined_zero_div_zero(result_frac, result_exp);
        return;
    }

    // ∞ / ∞ = ℘ ⬆/⬆
    if (a_is_inf && b_is_inf) {
        *result_frac = SPIRIX_UNDEF_TRANSFINITE_DIV_TRANSFINITE;
        *result_exponent = SPIRIX_AMBIGUOUS_EXPONENT;
        return;
    }

    // x / 0 = ∞ (for x ≠ 0)
    if (b_is_zero && !a_is_zero) {
        spirix_create_infinity(result_frac, result_exp, 1);  // Sign from a
        return;
    }

    // 0 / x = 0 (for x ≠ 0)
    if (a_is_zero) {
        spirix_create_zero(result_frac, result_exp);
        return;
    }

    // Otherwise: perform actual division (implement Spirix division algorithm)
    // ... (kernel-specific implementation)
}

#endif // SPIRIX_CONSTANTS_H
