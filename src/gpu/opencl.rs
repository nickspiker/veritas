//! OpenCL GPU acceleration for Spirix operations
//!
//! Uses Mesa rusticl on AMD GPUs.
//! ZERO IEEE-754 in kernels - all integer operations.

use spirix::{ScalarF4E4, Tensor};
use ocl::{ProQue, Buffer, MemFlags};
use std::fs;

/// GPU matrix multiply using OpenCL kernel
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
/// # Example
/// ```no_run
/// use veritas::gpu::matmul_gpu_opencl;
/// use spirix::{ScalarF4E4, Tensor};
///
/// let a = Tensor::new(vec![ScalarF4E4::from(1.0); 100], vec![10, 10]);
/// let b = Tensor::new(vec![ScalarF4E4::from(2.0); 100], vec![10, 10]);
/// let c = matmul_gpu_opencl(&a, &b).unwrap();
/// ```
pub fn matmul_gpu_opencl(
    a: &Tensor<ScalarF4E4>,
    b: &Tensor<ScalarF4E4>,
) -> Result<Tensor<ScalarF4E4>, Box<dyn std::error::Error>> {
    // Validate shapes
    assert_eq!(a.shape.len(), 2, "Matrix A must be 2D");
    assert_eq!(b.shape.len(), 2, "Matrix B must be 2D");
    assert_eq!(a.shape[1], b.shape[0], "Inner dimensions must match");

    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

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

    // Load OpenCL kernel source
    let kernel_src = fs::read_to_string("gpu/opencl/spirix_matmul.cl")?;

    // Build OpenCL program
    let pro_que = ProQue::builder()
        .src(kernel_src)
        .dims((n, m))  // Global work size: N×M
        .build()?;

    // Create buffers
    let a_frac_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(a_frac.len())
        .copy_host_slice(&a_frac)
        .build()?;

    let a_exp_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(a_exp.len())
        .copy_host_slice(&a_exp)
        .build()?;

    let b_frac_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(b_frac.len())
        .copy_host_slice(&b_frac)
        .build()?;

    let b_exp_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(b_exp.len())
        .copy_host_slice(&b_exp)
        .build()?;

    let c_size = m * n;
    let c_frac_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().write_only())
        .len(c_size)
        .build()?;

    let c_exp_buf = Buffer::<i16>::builder()
        .queue(pro_que.queue().clone())
        .flags(MemFlags::new().write_only())
        .len(c_size)
        .build()?;

    // Build and enqueue kernel
    let kernel = pro_que.kernel_builder("spirix_matmul_kernel")
        .arg(&a_frac_buf)
        .arg(&a_exp_buf)
        .arg(&b_frac_buf)
        .arg(&b_exp_buf)
        .arg(&c_frac_buf)
        .arg(&c_exp_buf)
        .arg(m as i32)
        .arg(n as i32)
        .arg(k as i32)
        .build()?;

    unsafe {
        kernel.enq()?;
    }

    // Read results back
    let mut c_frac = vec![0i16; c_size];
    let mut c_exp = vec![0i16; c_size];

    c_frac_buf.read(&mut c_frac).enq()?;
    c_exp_buf.read(&mut c_exp).enq()?;

    // Reconstruct Spirix scalars
    let c_data: Vec<ScalarF4E4> = c_frac
        .into_iter()
        .zip(c_exp.into_iter())
        .map(|(frac, exp)| ScalarF4E4 { fraction: frac, exponent: exp })
        .collect();

    Ok(Tensor::new(c_data, vec![m, n]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spirix::{ScalarF4E4, Tensor};

    #[test]
    #[ignore] // Only run if GPU is available
    fn test_opencl_matmul_small() {
        // Set environment variable for rusticl
        std::env::set_var("RUSTICL_ENABLE", "radeonsi");

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

        let c = matmul_gpu_opencl(&a, &b).unwrap();

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        //         = [[19, 22], [43, 50]]
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0].to_f64() - 19.0).abs() < 0.1);
        assert!((c.data[1].to_f64() - 22.0).abs() < 0.1);
        assert!((c.data[2].to_f64() - 43.0).abs() < 0.1);
        assert!((c.data[3].to_f64() - 50.0).abs() < 0.1);
    }
}
