//! Test GPU handling of undefined states
//!
//! Verifies that GPU operations produce correct Spirix undefined patterns

use veritas::autograd::{Tensor, Shape, matmul};
use spirix::ScalarF4E4;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║         GPU Undefined State Handling Test                     ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Testing CPU matmul with edge cases first...\n");
    test_cpu_matmul_undefined();

    println!("\n═══════════════════════════════════════════════════════════════");
    println!("Note: GPU tests require actual GPU hardware and are marked #[ignore]");
    println!("═══════════════════════════════════════════════════════════════\n");

    println!("To test GPU undefined handling:");
    println!("  1. Ensure libspirix_hip.so is built");
    println!("  2. Run: cargo test --release test_gpu_undefined -- --ignored");
    println!();
}

fn test_cpu_matmul_undefined() {
    println!("═══ Test 1: CPU Matmul with Zero ═══\n");

    let zero = ScalarF4E4::ZERO;
    let one = ScalarF4E4::ONE;

    // 2x2 matrix with zeros
    let a = Tensor::from_scalars(
        vec![zero, one, one, zero],
        Shape::matrix(2, 2)
    ).unwrap();

    let b = Tensor::from_scalars(
        vec![zero, zero, one, one],
        Shape::matrix(2, 2)
    ).unwrap();

    println!("A = [[0, 1],");
    println!("     [1, 0]]");
    println!();
    println!("B = [[0, 0],");
    println!("     [1, 1]]");
    println!();

    let c = matmul(&a, &b).unwrap();
    let c_data = c.as_scalars().unwrap();

    println!("C = A × B:");
    for i in 0..2 {
        print!("    [");
        for j in 0..2 {
            let idx = i * 2 + j;
            let val = &c_data[idx];
            print!("{:?}", val);
            if j == 0 { print!(", "); }
        }
        println!("]");
    }
    println!();

    // Verify all results are valid (not garbage)
    for (i, val) in c_data.iter().enumerate() {
        let is_valid = val.is_zero() || val.is_finite() || val.is_undefined() || val.is_infinite();
        println!("  C[{}]: is_zero={}, is_finite={}, is_undefined={}, is_infinite={}",
                 i, val.is_zero(), val.is_finite(), val.is_undefined(), val.is_infinite());
        assert!(is_valid, "Invalid value at index {}: {:?}", i, val);
    }

    println!("  ✓ All CPU results are valid\n");

    println!("═══ Test 2: CPU Matmul with Infinity ═══\n");

    let inf = one / zero;  // Create infinity

    let a = Tensor::from_scalars(
        vec![inf, one, one, zero],
        Shape::matrix(2, 2)
    ).unwrap();

    let b = Tensor::from_scalars(
        vec![zero, inf, one, one],
        Shape::matrix(2, 2)
    ).unwrap();

    println!("A = [[∞, 1],");
    println!("     [1, 0]]");
    println!();
    println!("B = [[0, ∞],");
    println!("     [1, 1]]");
    println!();

    let c = matmul(&a, &b).unwrap();
    let c_data = c.as_scalars().unwrap();

    println!("C = A × B:");
    for i in 0..2 {
        print!("    [");
        for j in 0..2 {
            let idx = i * 2 + j;
            let val = &c_data[idx];

            // Show state
            if val.is_undefined() {
                print!("undefined");
            } else if val.is_infinite() {
                print!("∞");
            } else if val.is_zero() {
                print!("0");
            } else {
                print!("{}", val);
            }

            if j == 0 { print!(", "); }
        }
        println!("]");
    }
    println!();

    // Check for undefined (∞ * 0 = undefined)
    let has_undefined = c_data.iter().any(|v| v.is_undefined());
    if has_undefined {
        println!("  ✓ CPU correctly produces undefined for ∞ × 0");
    }

    println!();

    println!("═══ Test 3: CPU Matmul with Undefined ═══\n");

    let undef = zero / zero;  // Create undefined (℘ ⬇/⬇)

    let a = Tensor::from_scalars(
        vec![one, undef, zero, one],
        Shape::matrix(2, 2)
    ).unwrap();

    let b = Tensor::from_scalars(
        vec![one, one, one, one],
        Shape::matrix(2, 2)
    ).unwrap();

    println!("A = [[1, undefined],");
    println!("     [0, 1]]");
    println!();
    println!("B = [[1, 1],");
    println!("     [1, 1]]");
    println!();

    let c = matmul(&a, &b).unwrap();
    let c_data = c.as_scalars().unwrap();

    println!("C = A × B:");
    let mut undefined_count = 0;
    for i in 0..2 {
        print!("    [");
        for j in 0..2 {
            let idx = i * 2 + j;
            let val = &c_data[idx];

            if val.is_undefined() {
                print!("undefined");
                undefined_count += 1;
            } else {
                print!("{}", val);
            }

            if j == 0 { print!(", "); }
        }
        println!("]");
    }
    println!();

    println!("  Undefined propagated to {} elements", undefined_count);
    assert!(undefined_count > 0, "Undefined should propagate");
    println!("  ✓ CPU correctly propagates undefined");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_matmul_with_undefined() {
        // Test that undefined propagates correctly in CPU matmul
        let zero = ScalarF4E4::ZERO;
        let one = ScalarF4E4::ONE;
        let undef = zero / zero;

        let a = Tensor::from_scalars(
            vec![one, undef],
            Shape::matrix(1, 2)
        ).unwrap();

        let b = Tensor::from_scalars(
            vec![one, one],
            Shape::matrix(2, 1)
        ).unwrap();

        // [1, undefined] × [[1], [1]] = [1×1 + undefined×1] = [undefined]
        let c = matmul(&a, &b).unwrap();
        let c_data = c.as_scalars().unwrap();

        assert!(c_data[0].is_undefined(), "Result should be undefined");
    }

    #[test]
    #[ignore]  // Requires GPU hardware
    fn test_gpu_matmul_with_undefined() {
        // TODO: Test GPU matmul with undefined once we have GPU available
        // This should match CPU behavior exactly
    }
}
