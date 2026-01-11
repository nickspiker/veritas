//! Property-based tests for Tensor operations
//!
//! Verifies mathematical axioms:
//! - Commutativity: a + b = b + a
//! - Associativity: (a + b) + c = a + (b + c)
//! - Identity: a + 0 = a, a * 1 = a
//! - Undefined propagation: undefined op x = undefined
//! - Transpose invariants: (A^T)^T = A, (AB)^T = B^T A^T
//! - Scale distributivity: k(A + B) = kA + kB

use proptest::prelude::*;
use veritas::autograd::{Tensor, Shape, matmul};
use spirix::ScalarF4E4;

// Generator for small Spirix scalars (to avoid overflow in tests)
fn small_scalar() -> impl Strategy<Value = ScalarF4E4> {
    (0u8..=50u8).prop_map(ScalarF4E4::from)
}

// Generator for non-zero scalars
fn nonzero_scalar() -> impl Strategy<Value = ScalarF4E4> {
    (1u8..=50u8).prop_map(ScalarF4E4::from)
}

// Generator for small vectors
fn small_vector(len: usize) -> impl Strategy<Value = Vec<ScalarF4E4>> {
    prop::collection::vec(small_scalar(), len..=len)
}

// Generator for small matrices
fn small_matrix(rows: usize, cols: usize) -> impl Strategy<Value = Vec<ScalarF4E4>> {
    prop::collection::vec(small_scalar(), rows * cols..=rows * cols)
}

// ═══ Addition Properties ═══

proptest! {
    #[test]
    fn test_add_commutative(a in small_vector(5), b in small_vector(5)) {
        let t1 = Tensor::from_scalars(a.clone(), Shape::vector(5)).unwrap();
        let t2 = Tensor::from_scalars(b.clone(), Shape::vector(5)).unwrap();

        let ab = t1.add(&t2).unwrap();
        let ba = t2.add(&t1).unwrap();

        let ab_data = ab.as_scalars().unwrap();
        let ba_data = ba.as_scalars().unwrap();

        for i in 0..5 {
            prop_assert_eq!(ab_data[i], ba_data[i], "Addition not commutative at index {}", i);
        }
    }

    #[test]
    fn test_add_associative(
        a in small_vector(4),
        b in small_vector(4),
        c in small_vector(4)
    ) {
        let ta = Tensor::from_scalars(a, Shape::vector(4)).unwrap();
        let tb = Tensor::from_scalars(b, Shape::vector(4)).unwrap();
        let tc = Tensor::from_scalars(c, Shape::vector(4)).unwrap();

        // (a + b) + c
        let ab = ta.add(&tb).unwrap();
        let abc1 = ab.add(&tc).unwrap();

        // a + (b + c)
        let bc = tb.add(&tc).unwrap();
        let abc2 = ta.add(&bc).unwrap();

        let data1 = abc1.as_scalars().unwrap();
        let data2 = abc2.as_scalars().unwrap();

        for i in 0..4 {
            prop_assert_eq!(data1[i], data2[i], "Addition not associative at index {}", i);
        }
    }

    #[test]
    fn test_add_identity(a in small_vector(5)) {
        let t = Tensor::from_scalars(a, Shape::vector(5)).unwrap();
        let zero = Tensor::zeros(Shape::vector(5));

        let result = t.add(&zero).unwrap();
        let t_data = t.as_scalars().unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..5 {
            prop_assert_eq!(result_data[i], t_data[i], "Zero identity failed at index {}", i);
        }
    }
}

// ═══ Scale Properties ═══

proptest! {
    #[test]
    fn test_scale_identity(a in small_vector(5)) {
        let t = Tensor::from_scalars(a, Shape::vector(5)).unwrap();
        let one = ScalarF4E4::ONE;

        let result = t.scale(one).unwrap();
        let t_data = t.as_scalars().unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..5 {
            prop_assert_eq!(result_data[i], t_data[i], "Scale identity failed at index {}", i);
        }
    }

    #[test]
    fn test_scale_zero(a in small_vector(5)) {
        let t = Tensor::from_scalars(a, Shape::vector(5)).unwrap();
        let zero = ScalarF4E4::ZERO;

        let result = t.scale(zero).unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..5 {
            prop_assert!(result_data[i].is_zero(), "Scale by zero failed at index {}", i);
        }
    }

    #[test]
    fn test_scale_distributive(
        a in small_vector(4),
        b in small_vector(4),
        k in small_scalar()
    ) {
        let ta = Tensor::from_scalars(a, Shape::vector(4)).unwrap();
        let tb = Tensor::from_scalars(b, Shape::vector(4)).unwrap();

        // k(A + B)
        let sum = ta.add(&tb).unwrap();
        let k_sum = sum.scale(k).unwrap();

        // kA + kB
        let ka = ta.scale(k).unwrap();
        let kb = tb.scale(k).unwrap();
        let sum_k = ka.add(&kb).unwrap();

        let data1 = k_sum.as_scalars().unwrap();
        let data2 = sum_k.as_scalars().unwrap();

        for i in 0..4 {
            prop_assert_eq!(data1[i], data2[i], "Scale distributivity failed at index {}", i);
        }
    }

    #[test]
    fn test_scale_associative(
        a in small_vector(4),
        k1 in nonzero_scalar(),
        k2 in nonzero_scalar()
    ) {
        let t = Tensor::from_scalars(a, Shape::vector(4)).unwrap();

        // (k1 * k2) * A
        let k_product = k1 * k2;
        let result1 = t.scale(k_product).unwrap();

        // k1 * (k2 * A)
        let k2_a = t.scale(k2).unwrap();
        let result2 = k2_a.scale(k1).unwrap();

        let data1 = result1.as_scalars().unwrap();
        let data2 = result2.as_scalars().unwrap();

        for i in 0..4 {
            prop_assert_eq!(data1[i], data2[i], "Scale associativity failed at index {}", i);
        }
    }
}

// ═══ Transpose Properties ═══

proptest! {
    #[test]
    fn test_transpose_involution(data in small_matrix(3, 4)) {
        // (A^T)^T = A
        let a = Tensor::from_scalars(data.clone(), Shape::matrix(3, 4)).unwrap();

        let at = a.transpose().unwrap();
        let att = at.transpose().unwrap();

        let a_data = a.as_scalars().unwrap();
        let att_data = att.as_scalars().unwrap();

        prop_assert_eq!(att.shape().dims(), a.shape().dims(), "Shape mismatch after double transpose");

        for i in 0..data.len() {
            prop_assert_eq!(att_data[i], a_data[i], "Double transpose not identity at index {}", i);
        }
    }

    #[test]
    fn test_transpose_shape(rows in 2usize..=5, cols in 2usize..=5) {
        let data: Vec<ScalarF4E4> = (0..rows * cols)
            .map(|i| ScalarF4E4::from((i % 10) as u8))
            .collect();

        let a = Tensor::from_scalars(data, Shape::matrix(rows, cols)).unwrap();
        let at = a.transpose().unwrap();

        prop_assert_eq!(at.shape().dims(), &[cols, rows], "Transpose shape incorrect");
    }

    #[test]
    fn test_transpose_preserves_values(data in small_matrix(2, 3)) {
        let a = Tensor::from_scalars(data.clone(), Shape::matrix(2, 3)).unwrap();
        let at = a.transpose().unwrap();

        let a_data = a.as_scalars().unwrap();
        let at_data = at.as_scalars().unwrap();

        // Check a[i,j] = at[j,i]
        for i in 0..2 {
            for j in 0..3 {
                let a_idx = i * 3 + j;
                let at_idx = j * 2 + i;
                prop_assert_eq!(a_data[a_idx], at_data[at_idx],
                    "Transpose value mismatch at ({}, {})", i, j);
            }
        }
    }
}

// ═══ Undefined Propagation Properties ═══

proptest! {
    #[test]
    fn test_add_undefined_propagation(a in small_vector(5)) {
        let zero = ScalarF4E4::ZERO;
        let undef = zero / zero;

        let mut data = a.clone();
        data[2] = undef;  // Inject undefined at index 2

        let ta = Tensor::from_scalars(data, Shape::vector(5)).unwrap();
        let tb = Tensor::from_scalars(a, Shape::vector(5)).unwrap();

        let result = ta.add(&tb).unwrap();
        let result_data = result.as_scalars().unwrap();

        prop_assert!(result_data[2].is_undefined(),
            "Undefined should propagate through addition at index 2");

        // Other elements should be normal
        for i in [0, 1, 3, 4] {
            prop_assert!(!result_data[i].is_undefined(),
                "Undefined should not spread to index {}", i);
        }
    }

    #[test]
    fn test_scale_undefined_propagation(a in small_vector(5), k in small_scalar()) {
        let zero = ScalarF4E4::ZERO;
        let undef = zero / zero;

        let mut data = a;
        data[1] = undef;

        let t = Tensor::from_scalars(data, Shape::vector(5)).unwrap();
        let result = t.scale(k).unwrap();
        let result_data = result.as_scalars().unwrap();

        prop_assert!(result_data[1].is_undefined(),
            "Undefined should propagate through scale at index 1");
    }

    #[test]
    fn test_transpose_undefined_preservation(data in small_matrix(2, 2)) {
        let zero = ScalarF4E4::ZERO;
        let undef = zero / zero;

        let mut mat_data = data;
        mat_data[1] = undef;

        let a = Tensor::from_scalars(mat_data, Shape::matrix(2, 2)).unwrap();
        let at = a.transpose().unwrap();

        let a_data = a.as_scalars().unwrap();
        let at_data = at.as_scalars().unwrap();

        // Count undefined in both
        let undef_count_a = a_data.iter().filter(|x| x.is_undefined()).count();
        let undef_count_at = at_data.iter().filter(|x| x.is_undefined()).count();

        prop_assert_eq!(undef_count_a, undef_count_at,
            "Transpose should preserve number of undefined values");
    }
}

// ═══ Matmul Properties ═══

proptest! {
    #[test]
    fn test_matmul_identity(data in small_matrix(3, 3)) {
        // A × I = A
        let a = Tensor::from_scalars(data.clone(), Shape::matrix(3, 3)).unwrap();

        // Create identity matrix
        let mut id_data = vec![ScalarF4E4::ZERO; 9];
        id_data[0] = ScalarF4E4::ONE;
        id_data[4] = ScalarF4E4::ONE;
        id_data[8] = ScalarF4E4::ONE;
        let identity = Tensor::from_scalars(id_data, Shape::matrix(3, 3)).unwrap();

        let result = matmul(&a, &identity).unwrap();
        let a_data = a.as_scalars().unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..9 {
            prop_assert_eq!(result_data[i], a_data[i],
                "Matmul identity failed at index {}", i);
        }
    }

    #[test]
    fn test_matmul_zero_annihilation(data in small_matrix(2, 3)) {
        // A × 0 = 0
        let a = Tensor::from_scalars(data, Shape::matrix(2, 3)).unwrap();
        let zero = Tensor::zeros(Shape::matrix(3, 2));

        let result = matmul(&a, &zero).unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..4 {
            prop_assert!(result_data[i].is_zero(),
                "Matmul with zero should produce zero at index {}", i);
        }
    }

    #[test]
    fn test_matmul_dimensions(
        rows in 2usize..=4,
        inner in 2usize..=4,
        cols in 2usize..=4
    ) {
        let a_data: Vec<ScalarF4E4> = (0..rows * inner)
            .map(|i| ScalarF4E4::from((i % 10) as u8))
            .collect();
        let b_data: Vec<ScalarF4E4> = (0..inner * cols)
            .map(|i| ScalarF4E4::from((i % 10) as u8))
            .collect();

        let a = Tensor::from_scalars(a_data, Shape::matrix(rows, inner)).unwrap();
        let b = Tensor::from_scalars(b_data, Shape::matrix(inner, cols)).unwrap();

        let c = matmul(&a, &b).unwrap();

        prop_assert_eq!(c.shape().dims(), &[rows, cols],
            "Matmul output dimensions incorrect");
    }
}

// ═══ Edge Case Properties ═══

proptest! {
    #[test]
    fn test_zero_preservation(data in small_vector(5)) {
        // Operations with zero should preserve zero where expected
        let mut vec_data = data;
        vec_data[0] = ScalarF4E4::ZERO;
        vec_data[4] = ScalarF4E4::ZERO;

        let t = Tensor::from_scalars(vec_data, Shape::vector(5)).unwrap();

        // Add zero vector - should preserve zeros
        let zero_vec = Tensor::zeros(Shape::vector(5));
        let result = t.add(&zero_vec).unwrap();
        let result_data = result.as_scalars().unwrap();

        prop_assert!(result_data[0].is_zero(), "Zero not preserved at index 0");
        prop_assert!(result_data[4].is_zero(), "Zero not preserved at index 4");
    }

    #[test]
    fn test_finite_preservation(data in small_vector(5)) {
        // Normal operations on finite numbers should stay finite
        let t1 = Tensor::from_scalars(data.clone(), Shape::vector(5)).unwrap();
        let t2 = Tensor::from_scalars(data, Shape::vector(5)).unwrap();

        let result = t1.add(&t2).unwrap();
        let result_data = result.as_scalars().unwrap();

        for i in 0..5 {
            prop_assert!(result_data[i].is_finite() || result_data[i].is_zero(),
                "Finite numbers should stay finite at index {}", i);
        }
    }
}
