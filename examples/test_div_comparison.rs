use spirix::ScalarF4E4;

fn main() {
    // Test 20 / 4
    let a = ScalarF4E4::from(20);
    let b = ScalarF4E4::from(4);
    let result = a / b;
    let result_f64: f64 = result.into();

    println!("Current Newton-Raphson: 20 / 4 = {:.10}", result_f64);
    println!("Expected: 5.0");
    println!("Error: {:.10}", (result_f64 - 5.0).abs());
    println!("Result equals 5.0? {}", result_f64 == 5.0);

    // Test what the scalar equality operator does
    println!("\nScalar comparison: result == 5 => {}", result == 5);

    // Test 10 / 2
    let c = ScalarF4E4::from(10);
    let d = ScalarF4E4::from(2);
    let result2 = c / d;
    let result2_f64: f64 = result2.into();
    println!("\n10 / 2 = {:.10}", result2_f64);
    println!("result == 5 => {}", result2 == 5);
}
