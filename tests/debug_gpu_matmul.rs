//! Debug GPU matmul mismatch

use spirix::{ScalarF4E4, Tensor};

// CPU matmul implementation
fn matmul_cpu(a: &Tensor<ScalarF4E4>, b: &Tensor<ScalarF4E4>) -> Tensor<ScalarF4E4> {
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
                println!("  A[{},{}] = {:?}, B[{},{}] = {:?}",
                    i, k_idx, a_val, k_idx, j, b_val);
                let prod = a_val * b_val;
                println!("  Product = {:?}", prod);
                sum = sum + prod;
                println!("  Sum = {:?}", sum);
            }
            c_data[i * n + j] = sum;
            println!("C[{},{}] = {:?}", i, j, sum);
        }
    }

    Tensor::new(c_data, vec![m, n])
}

#[test]
#[ignore]
fn debug_simple_case() {
    // [[5, 3], [2, 7]] × [[1, 4], [6, 2]]
    // Expected: [[23, 26], [44, 22]]

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

    println!("\n=== CPU MATMUL ===");
    let c_cpu = matmul_cpu(&a, &b);

    println!("\n=== GPU MATMUL ===");
    let c_gpu = veritas::gpu::matmul_gpu(&a, &b);

    println!("\n=== RESULTS ===");
    for i in 0..4 {
        println!("Index {}: CPU = {:?}, GPU = {:?}, Match = {}",
            i, c_cpu.data[i], c_gpu.data[i],
            c_cpu.data[i] == c_gpu.data[i]);
    }

    println!("\n=== EXPECTED ===");
    println!("[0] = 23 (5*1 + 3*6)");
    println!("[1] = 26 (5*4 + 3*2)");
    println!("[2] = 44 (2*1 + 7*6)");
    println!("[3] = 22 (2*4 + 7*2)");
}
