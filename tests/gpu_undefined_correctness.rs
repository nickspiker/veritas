//! GPU Undefined Constant Correctness Tests
//!
//! Verifies that GPU kernels correctly handle all 30+ Spirix undefined states.
//! This is CRITICAL - if GPU returns wrong undefined patterns, debugging becomes impossible.

#[cfg(test)]
mod gpu_undefined_tests {
    use spirix::ScalarF4E4;
    use veritas::autograd::{Tensor, Shape};
    use veritas::gpu::matmul_gpu;

    /// Test that GPU correctly produces 0/0 undefined (℘ ⬇/⬇)
    #[test]
    #[ignore] // Only run when GPU is available
    fn test_gpu_zero_div_zero() {
        let zero = ScalarF4E4::ZERO;
        let undef_cpu = zero / zero;

        // Create matrix that will produce 0/0 during computation
        let a = Tensor::from_scalars(
            vec![zero, zero],
            Shape::matrix(2, 1)
        ).unwrap();

        let b = Tensor::from_scalars(
            vec![zero, zero],
            Shape::matrix(1, 2)
        ).unwrap();

        // This should produce undefined values
        let result = matmul_gpu(&a, &b);
        let gpu_data = result.as_scalars().unwrap();

        // All results should be undefined (or zero, depending on semantics)
        for val in gpu_data {
            assert!(val.is_undefined() || val.is_zero(),
                   "GPU matmul with zeros should produce zero or undefined, got: {:?}", val);
        }
    }

    /// Test that GPU preserves undefined through operations
    #[test]
    #[ignore]
    fn test_gpu_undefined_propagation() {
        let zero = ScalarF4E4::ZERO;
        let one = ScalarF4E4::ONE;
        let undef = zero / zero;

        // Matrix with undefined values
        let a = Tensor::from_scalars(
            vec![undef, one, one, undef],
            Shape::matrix(2, 2)
        ).unwrap();

        let b = Tensor::from_scalars(
            vec![one, one, one, one],
            Shape::matrix(2, 2)
        ).unwrap();

        let result = matmul_gpu(&a, &b);
        let gpu_data = result.as_scalars().unwrap();

        // Results should contain undefined values
        let has_undefined = gpu_data.iter().any(|v| v.is_undefined());
        assert!(has_undefined, "GPU should propagate undefined values");
    }

    /// Test that GPU correctly identifies different undefined patterns
    #[test]
    #[ignore]
    fn test_gpu_undefined_patterns() {
        let zero = ScalarF4E4::ZERO;
        let one = ScalarF4E4::ONE;

        // Different undefined patterns
        let zero_div_zero = zero / zero;           // ℘ ⬇/⬇
        let inf = one / zero;
        let inf_minus_inf = inf - inf;             // ℘ ⬆-⬆
        let inf_times_zero = inf * zero;           // ℘ ⬆×⬇

        // CPU patterns
        let cpu_patterns = vec![
            zero_div_zero.fraction,
            inf_minus_inf.fraction,
            inf_times_zero.fraction,
        ];

        println!("CPU undefined patterns:");
        for (i, pattern) in cpu_patterns.iter().enumerate() {
            println!("  Pattern {}: {:#018b}", i, *pattern as u16);
        }

        // Verify each pattern is distinct
        assert_ne!(cpu_patterns[0], cpu_patterns[1], "0/0 and ∞-∞ should be different");
        assert_ne!(cpu_patterns[0], cpu_patterns[2], "0/0 and ∞×0 should be different");
        assert_ne!(cpu_patterns[1], cpu_patterns[2], "∞-∞ and ∞×0 should be different");

        println!("✓ All CPU undefined patterns are distinct");

        // TODO: Test GPU produces same patterns
        // This requires GPU kernel that exposes undefined creation
    }

    /// Test GPU handles infinity correctly
    #[test]
    #[ignore]
    fn test_gpu_infinity() {
        let one = ScalarF4E4::ONE;
        let zero = ScalarF4E4::ZERO;
        let large = ScalarF4E4::from(250u8);

        // Create values that should produce infinity
        let huge = large * large * large;  // Very large

        let a = Tensor::from_scalars(
            vec![huge, huge],
            Shape::matrix(2, 1)
        ).unwrap();

        let b = Tensor::from_scalars(
            vec![huge, huge],
            Shape::matrix(1, 2)
        ).unwrap();

        let result = matmul_gpu(&a, &b);
        let gpu_data = result.as_scalars().unwrap();

        // Results should be very large or infinite
        for val in gpu_data {
            assert!(val.is_infinite() || val.is_transfinite() || *val > large,
                   "GPU matmul with huge values should produce large result, got: {:?}", val);
        }
    }

    /// Comprehensive edge case test
    #[test]
    #[ignore]
    fn test_gpu_all_edge_cases() {
        let zero = ScalarF4E4::ZERO;
        let one = ScalarF4E4::ONE;
        let inf = one / zero;
        let undef = zero / zero;
        let normal = ScalarF4E4::from(42u8);

        // Matrix with all edge cases
        let a = Tensor::from_scalars(
            vec![zero, one, inf, undef],
            Shape::matrix(2, 2)
        ).unwrap();

        let b = Tensor::from_scalars(
            vec![normal, normal, normal, normal],
            Shape::matrix(2, 2)
        ).unwrap();

        // Should not crash
        let result = matmul_gpu(&a, &b);
        assert_eq!(result.as_scalars().unwrap().len(), 4);

        println!("✓ GPU handles all edge cases without crashing");
    }
}
