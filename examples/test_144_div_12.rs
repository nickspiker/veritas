use spirix::ScalarF4E4;

fn main() {
    let a = ScalarF4E4::from(144u64);
    let b = ScalarF4E4::from(12);
    let result = a / b;
    let result_f64: f64 = result.into();

    println!("144 / 12 = {}", result);
    println!("As f64: {:.10}", result_f64);
    println!("Expected: 12.0");
    println!("Error: {:.10}", (result_f64 - 12.0).abs());
    println!("result == 12 => {}", result == 12);

    // Also test some other divisions
    let c = ScalarF4E4::from(100);
    let d = ScalarF4E4::from(4);
    let result2 = c / d;
    let result2_f64: f64 = result2.into();
    println!("\n100 / 4 = {:.10} (expected 25.0)", result2_f64);
}
