use spirix::ScalarF4E4;

fn main() {
    let a = ScalarF4E4::from(20usize);
    let b = 4isize;
    let result = a / b;

    println!("20 / 4 (mixed types) = {}", result);
    let result_f64: f64 = result.into();
    println!("As f64: {}", result_f64);
    println!("Expected: 5.0");

    // Also test 20 / 4 where both are ScalarF4E4
    let a2 = ScalarF4E4::from(20);
    let b2 = ScalarF4E4::from(4);
    let result2 = a2 / b2;
    let result2_f64: f64 = result2.into();
    println!("\nScalar / Scalar: {} (as f64: {})", result2, result2_f64);
}
