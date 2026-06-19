use spirix::ScalarF7E7;
use std::time::Instant;

fn main() {
    const ITERATIONS: usize = 1_000_000;  // Fewer iterations since 128-bit is slower

    // Prepare test data
    let mut results = Vec::with_capacity(100);
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

    // Warm up
    for (a, b) in &test_pairs {
        let x = ScalarF7E7::from(*a);
        let y = ScalarF7E7::from(*b);
        let _ = x / y;
    }

    // Benchmark
    let start = Instant::now();
    for _ in 0..(ITERATIONS / test_pairs.len()) {
        for (a, b) in &test_pairs {
            let x = ScalarF7E7::from(*a);
            let y = ScalarF7E7::from(*b);
            let result = x / y;
            results.push(result);
        }
    }
    let elapsed = start.elapsed();

    let ops_per_sec = ITERATIONS as f64 / elapsed.as_secs_f64();
    let ns_per_op = elapsed.as_nanos() as f64 / ITERATIONS as f64;

    println!("ScalarF7E7 (128-bit) Division Benchmark:");
    println!("  Iterations: {}", ITERATIONS);
    println!("  Total time: {:.3}s", elapsed.as_secs_f64());
    println!("  Throughput: {:.2} MOPS/s", ops_per_sec / 1_000_000.0);
    println!("  Latency: {:.2} ns/op", ns_per_op);

    // Prevent optimization
    println!("\nSample results: {} values", results.len());
}
