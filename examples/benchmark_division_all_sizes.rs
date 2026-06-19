use spirix::{ScalarF3E3, ScalarF4E4, ScalarF5E5, ScalarF6E6, ScalarF7E7};
use std::time::Instant;

fn benchmark<T>(name: &str, iterations: usize, test_pairs: &[(i32, i32)])
where
    T: From<i32> + std::ops::Div<Output = T> + Copy,
{
    let mut results = Vec::with_capacity(iterations);

    // Warm up
    for (a, b) in test_pairs {
        let x = T::from(*a);
        let y = T::from(*b);
        let _ = x / y;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..(iterations / test_pairs.len()) {
        for (a, b) in test_pairs {
            let x = T::from(*a);
            let y = T::from(*b);
            let result = x / y;
            results.push(result);
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let ns_per_op = elapsed.as_nanos() as f64 / iterations as f64;

    println!("{:20} {:10.2} MOPS/s  {:8.2} ns/op",
             name,
             ops_per_sec / 1_000_000.0,
             ns_per_op);
}

fn main() {
    let test_pairs = vec![
        (20, 4),
        (144, 12),
        (100, 7),
        (7, 3),
        (1, 1),
        (10, 2),
        (15, 4),
        (2, 5),
        (50, 25),
        (99, 11),
    ];

    println!("Division Performance Comparison (Newton-Raphson):");
    println!("{:20} {:>15} {:>12}", "Type", "Throughput", "Latency");
    println!("{:-<50}", "");

    benchmark::<ScalarF3E3>("8-bit (F3E3)", 10_000_000, &test_pairs);
    benchmark::<ScalarF4E4>("16-bit (F4E4)", 10_000_000, &test_pairs);
    benchmark::<ScalarF5E5>("32-bit (F5E5)", 5_000_000, &test_pairs);
    benchmark::<ScalarF6E6>("64-bit (F6E6)", 2_000_000, &test_pairs);
    benchmark::<ScalarF7E7>("128-bit (F7E7)", 1_000_000, &test_pairs);
}
