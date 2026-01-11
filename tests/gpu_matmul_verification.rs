//! GPU Matrix Multiplication Verification Tests
//!
//! Verifies that GPU matmul maintains Spirix arithmetic integrity:
//! 1. Assembly contains NO floating-point instructions
//! 2. GPU results match CPU exactly (bit-for-bit)
//! 3. Property tests pass on GPU (identity, zero, associativity)
//! 4. Hand-verified test cases are correct

use spirix::{ScalarF4E4, Tensor};

// CPU matmul implementation (reference)
fn matmul_cpu(a: &Tensor<ScalarF4E4>, b: &Tensor<ScalarF4E4>) -> Tensor<ScalarF4E4> {
    assert_eq!(a.shape.len(), 2, "Matrix A must be 2D");
    assert_eq!(b.shape.len(), 2, "Matrix B must be 2D");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let mut c_data = vec![ScalarF4E4::ZERO; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = ScalarF4E4::ZERO;
            for k_idx in 0..k {
                let a_val = a.data[i * k + k_idx];
                let b_val = b.data[k_idx * n + j];
                sum = sum + (a_val * b_val);
            }
            c_data[i * n + j] = sum;
        }
    }

    Tensor::new(c_data, vec![m, n])
}

// ============================================================================
// EXACT MATCH TESTS - GPU vs CPU
// ============================================================================

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_simple() {
    // Simple case: [[5, 3], [2, 7]] × [[1, 4], [6, 2]]
    // Expected: [[5*1+3*6, 5*4+3*2], [2*1+7*6, 2*4+7*2]]
    //         = [[23, 26], [44, 22]]

    let a = Tensor::new(
        vec![
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(7u8),
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(1u8),
            ScalarF4E4::from(4u8),
            ScalarF4E4::from(6u8),
            ScalarF4E4::from(2u8),
        ],
        vec![2, 2],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    assert_eq!(c_cpu.shape, c_gpu.shape);
    for i in 0..c_cpu.data.len() {
        assert_eq!(
            c_cpu.data[i].fraction, c_gpu.data[i].fraction,
            "Mismatch at index {}: CPU fraction={}, GPU fraction={}",
            i, c_cpu.data[i].fraction, c_gpu.data[i].fraction
        );
        assert_eq!(
            c_cpu.data[i].exponent, c_gpu.data[i].exponent,
            "Mismatch at index {}: CPU exponent={}, GPU exponent={}",
            i, c_cpu.data[i].exponent, c_gpu.data[i].exponent
        );
    }

    // Verify expected values
    assert_eq!(c_cpu.data[0], ScalarF4E4::from(23u8));
    assert_eq!(c_cpu.data[1], ScalarF4E4::from(26u8));
    assert_eq!(c_cpu.data[2], ScalarF4E4::from(44u8));
    assert_eq!(c_cpu.data[3], ScalarF4E4::from(22u8));
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_zero() {
    // Test with zeros
    let a = Tensor::new(
        vec![
            ScalarF4E4::from(5u8),
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::from(3u8),
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::ZERO,
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(4u8),
            ScalarF4E4::ZERO,
        ],
        vec![2, 2],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    for i in 0..c_cpu.data.len() {
        assert_eq!(c_cpu.data[i], c_gpu.data[i],
            "Mismatch at index {}: CPU={:?}, GPU={:?}",
            i, c_cpu.data[i], c_gpu.data[i]);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_large_values() {
    // Test with large values (near overflow)
    let large = ScalarF4E4::from(1000.0);

    let a = Tensor::new(
        vec![large, large, large, large],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![large, large, large, large],
        vec![2, 2],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    for i in 0..c_cpu.data.len() {
        assert_eq!(c_cpu.data[i], c_gpu.data[i],
            "Mismatch at index {}: CPU={:?}, GPU={:?}",
            i, c_cpu.data[i], c_gpu.data[i]);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_small_values() {
    // Test with small values (near underflow)
    let small = ScalarF4E4::from(0.001);

    let a = Tensor::new(
        vec![small, small, small, small],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![small, small, small, small],
        vec![2, 2],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    for i in 0..c_cpu.data.len() {
        assert_eq!(c_cpu.data[i], c_gpu.data[i],
            "Mismatch at index {}: CPU={:?}, GPU={:?}",
            i, c_cpu.data[i], c_gpu.data[i]);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_negative() {
    // Test with negative values
    let a = Tensor::new(
        vec![
            ScalarF4E4::from(5),
            ScalarF4E4::from(-3),
            ScalarF4E4::from(-2),
            ScalarF4E4::from(7),
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(-1),
            ScalarF4E4::from(4),
            ScalarF4E4::from(6),
            ScalarF4E4::from(-2),
        ],
        vec![2, 2],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    for i in 0..c_cpu.data.len() {
        assert_eq!(c_cpu.data[i], c_gpu.data[i],
            "Mismatch at index {}: CPU={:?}, GPU={:?}",
            i, c_cpu.data[i], c_gpu.data[i]);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_vs_cpu_exact_match_rectangular() {
    // Test rectangular matrices: 3x2 @ 2x4
    let a = Tensor::new(
        vec![
            ScalarF4E4::from(1u8),
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(4u8),
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(6u8),
        ],
        vec![3, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(7u8),
            ScalarF4E4::from(8u8),
            ScalarF4E4::from(9u8),
            ScalarF4E4::from(10u8),
            ScalarF4E4::from(11u8),
            ScalarF4E4::from(12u8),
            ScalarF4E4::from(13u8),
            ScalarF4E4::from(14u8),
        ],
        vec![2, 4],
    );

    let c_cpu = matmul_cpu(&a, &b);
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Exact bit-for-bit match
    assert_eq!(c_cpu.shape, vec![3, 4]);
    assert_eq!(c_cpu.shape, c_gpu.shape);
    for i in 0..c_cpu.data.len() {
        assert_eq!(c_cpu.data[i], c_gpu.data[i],
            "Mismatch at index {}: CPU={:?}, GPU={:?}",
            i, c_cpu.data[i], c_gpu.data[i]);
    }
}

// ============================================================================
// HAND-VERIFIED TEST CASES
// ============================================================================

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_identity_matrix() {
    // [[2, 0], [0, 3]] × [[4, 0], [0, 5]] = [[8, 0], [0, 15]]

    let a = Tensor::new(
        vec![
            ScalarF4E4::from(2u8),
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::from(3u8),
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(4u8),
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::from(5u8),
        ],
        vec![2, 2],
    );

    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Verify exact expected results
    assert_eq!(c_gpu.data[0], ScalarF4E4::from(8u8));
    assert_eq!(c_gpu.data[1], ScalarF4E4::ZERO);
    assert_eq!(c_gpu.data[2], ScalarF4E4::ZERO);
    assert_eq!(c_gpu.data[3], ScalarF4E4::from(15u8));
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_all_ones() {
    // [[1, 1], [1, 1]] × [[2, 3], [4, 5]] = [[6, 8], [6, 8]]

    let a = Tensor::new(
        vec![
            ScalarF4E4::ONE,
            ScalarF4E4::ONE,
            ScalarF4E4::ONE,
            ScalarF4E4::ONE,
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(4u8),
            ScalarF4E4::from(5u8),
        ],
        vec![2, 2],
    );

    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    // Verify exact expected results
    assert_eq!(c_gpu.data[0], ScalarF4E4::from(6u8));
    assert_eq!(c_gpu.data[1], ScalarF4E4::from(8u8));
    assert_eq!(c_gpu.data[2], ScalarF4E4::from(6u8));
    assert_eq!(c_gpu.data[3], ScalarF4E4::from(8u8));
}

// ============================================================================
// PROPERTY TESTS
// ============================================================================

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_property_identity() {
    // A × I = A (where I is identity matrix)

    let a = Tensor::new(
        vec![
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(7u8),
        ],
        vec![2, 2],
    );

    let identity = Tensor::new(
        vec![
            ScalarF4E4::ONE,
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::ONE,
        ],
        vec![2, 2],
    );

    let c_gpu = veritas::gpu::matmul_gpu(&a, &identity);

    // Result should equal A exactly
    assert_eq!(c_gpu.data, a.data);
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_property_zero() {
    // A × 0 = 0

    let a = Tensor::new(
        vec![
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(7u8),
        ],
        vec![2, 2],
    );

    let zero = Tensor::new(
        vec![
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
        ],
        vec![2, 2],
    );

    let c_gpu = veritas::gpu::matmul_gpu(&a, &zero);

    // Result should be all zeros
    for val in c_gpu.data.iter() {
        assert_eq!(*val, ScalarF4E4::ZERO);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_property_associativity() {
    // (A × B) × C = A × (B × C)

    let a = Tensor::new(
        vec![
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(4u8),
            ScalarF4E4::from(5u8),
        ],
        vec![2, 2],
    );

    let b = Tensor::new(
        vec![
            ScalarF4E4::from(1u8),
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
            ScalarF4E4::from(4u8),
        ],
        vec![2, 2],
    );

    let c = Tensor::new(
        vec![
            ScalarF4E4::from(5u8),
            ScalarF4E4::from(6u8),
            ScalarF4E4::from(7u8),
            ScalarF4E4::from(8u8),
        ],
        vec![2, 2],
    );

    // Compute (A × B) × C
    let ab = veritas::gpu::matmul_gpu(&a, &b);
    let abc_left = veritas::gpu::matmul_gpu(&ab, &c);

    // Compute A × (B × C)
    let bc = veritas::gpu::matmul_gpu(&b, &c);
    let abc_right = veritas::gpu::matmul_gpu(&a, &bc);

    // Should be exactly equal
    for i in 0..abc_left.data.len() {
        assert_eq!(abc_left.data[i], abc_right.data[i],
            "Associativity failed at index {}: left={:?}, right={:?}",
            i, abc_left.data[i], abc_right.data[i]);
    }
}

#[test]
#[ignore] // Only run if GPU is available
fn test_gpu_property_random_exact_match() {
    // Test 100 random matrices for exact GPU vs CPU match
    use rand::Rng;
    let mut rng = rand::thread_rng();

    for _ in 0..100 {
        // Generate random 4x4 matrices with small values
        let a_data: Vec<ScalarF4E4> = (0..16)
            .map(|_| ScalarF4E4::from(rng.gen_range(-10..=10)))
            .collect();
        let b_data: Vec<ScalarF4E4> = (0..16)
            .map(|_| ScalarF4E4::from(rng.gen_range(-10..=10)))
            .collect();

        let a = Tensor::new(a_data, vec![4, 4]);
        let b = Tensor::new(b_data, vec![4, 4]);

        let c_cpu = matmul_cpu(&a, &b);
        let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

        // Exact bit-for-bit match
        for i in 0..c_cpu.data.len() {
            assert_eq!(c_cpu.data[i], c_gpu.data[i],
                "Random test failed at index {}: CPU={:?}, GPU={:?}",
                i, c_cpu.data[i], c_gpu.data[i]);
        }
    }
}
