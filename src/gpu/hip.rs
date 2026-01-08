//! HIP (ROCm) FFI bindings for Spirix GPU kernels
//!
//! Calls into libspirix_hip.so compiled from spirix_matmul.hip

use spirix::{ScalarF4E4, Tensor};

/// Raw FFI binding to HIP kernel
///
/// Safety: Must be called with valid pointers and correct dimensions.
/// The HIP kernel handles device memory allocation internally.
extern "C" {
    fn spirix_matmul_hip(
        a_frac: *const i16,
        a_exp: *const i16,
        b_frac: *const i16,
        b_exp: *const i16,
        c_frac: *mut i16,
        c_exp: *mut i16,
        m: i32,
        n: i32,
        k: i32,
    );
}

/// GPU matrix multiply using HIP kernel
///
/// Computes C = A * B on GPU using Spirix arithmetic.
///
/// # Arguments
/// * `a` - Input matrix A (M×K)
/// * `b` - Input matrix B (K×N)
///
/// # Returns
/// Output matrix C (M×N)
///
/// # Safety
/// This function is safe to call. It manages all GPU memory internally.
///
/// # Example
/// ```no_run
/// use veritas::gpu::matmul_gpu;
/// use spirix::{ScalarF4E4, Tensor};
///
/// let a = Tensor::new(vec![ScalarF4E4::from(1.0); 100], vec![10, 10]);
/// let b = Tensor::new(vec![ScalarF4E4::from(2.0); 100], vec![10, 10]);
/// let c = matmul_gpu(&a, &b);
/// ```
pub fn matmul_gpu(a: &Tensor<ScalarF4E4>, b: &Tensor<ScalarF4E4>) -> Tensor<ScalarF4E4> {
    // Validate shapes
    assert_eq!(a.shape.len(), 2, "Matrix A must be 2D");
    assert_eq!(b.shape.len(), 2, "Matrix B must be 2D");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

    let m = a.shape[0] as i32;
    let k = a.shape[1] as i32;
    let n = b.shape[1] as i32;

    // Separate fractions and exponents
    let (a_frac, a_exp): (Vec<i16>, Vec<i16>) = a
        .data
        .iter()
        .map(|s| (s.fraction, s.exponent))
        .unzip();

    let (b_frac, b_exp): (Vec<i16>, Vec<i16>) = b
        .data
        .iter()
        .map(|s| (s.fraction, s.exponent))
        .unzip();

    // Allocate output
    let c_size = (m * n) as usize;
    let mut c_frac = vec![0i16; c_size];
    let mut c_exp = vec![0i16; c_size];

    // Call HIP kernel
    unsafe {
        spirix_matmul_hip(
            a_frac.as_ptr(),
            a_exp.as_ptr(),
            b_frac.as_ptr(),
            b_exp.as_ptr(),
            c_frac.as_mut_ptr(),
            c_exp.as_mut_ptr(),
            m,
            n,
            k,
        );
    }

    // Reconstruct Spirix scalars
    let c_data: Vec<ScalarF4E4> = c_frac
        .into_iter()
        .zip(c_exp.into_iter())
        .map(|(frac, exp)| ScalarF4E4 { fraction: frac, exponent: exp })
        .collect();

    Tensor::new(c_data, vec![m as usize, n as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirix::{ScalarF4E4, Tensor};

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_gpu_matmul_small() {
        // 2×2 matrix multiply
        let a = Tensor::new(
            vec![
                ScalarF4E4::from(1.0),
                ScalarF4E4::from(2.0),
                ScalarF4E4::from(3.0),
                ScalarF4E4::from(4.0),
            ],
            vec![2, 2],
        );

        let b = Tensor::new(
            vec![
                ScalarF4E4::from(5.0),
                ScalarF4E4::from(6.0),
                ScalarF4E4::from(7.0),
                ScalarF4E4::from(8.0),
            ],
            vec![2, 2],
        );

        let c = matmul_gpu(&a, &b);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0].to_f64() - 19.0).abs() < 0.1);
        assert!((c.data[1].to_f64() - 22.0).abs() < 0.1);
        assert!((c.data[2].to_f64() - 43.0).abs() < 0.1);
        assert!((c.data[3].to_f64() - 50.0).abs() < 0.1);
    }

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_gpu_matmul_large() {
        // 1024×1024 matrix multiply
        let size = 1024;
        let a_data: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from((i % 100) as f64 * 0.01))
            .collect();
        let b_data: Vec<ScalarF4E4> = (0..size * size)
            .map(|i| ScalarF4E4::from((i % 100) as f64 * 0.01))
            .collect();

        let a = Tensor::new(a_data, vec![size, size]);
        let b = Tensor::new(b_data, vec![size, size]);

        let c = matmul_gpu(&a, &b);

        assert_eq!(c.shape, vec![size, size]);
        println!("GPU matmul 1024×1024 completed");
        println!("First element: {:.6}", c.data[0].to_f64());
    }
}
