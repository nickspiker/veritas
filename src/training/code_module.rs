//! Code Detection and Verification Module
//!
//! CONSTITUTION COMPLIANT:
//! ✓ Pure Spirix training
//! ✓ Rust compiler for verification (external tool, not IEEE-754)
//! ✓ Snap-in architecture proof

use std::fs;
use std::process::Command;

/// Verify Rust code compiles using rustc
///
/// Returns true if code compiles successfully.
/// Uses wasm32 target to avoid platform-specific issues.
pub fn verify_rust_code(code: &str) -> bool {
    // Write to temp file
    let temp_path = "/tmp/verify_code_temp.rs";
    if fs::write(temp_path, code).is_err() {
        return false;
    }

    // Compile check (no execution)
    // Use --crate-type lib for function definitions
    let output = Command::new("rustc")
        .args(&[
            "--crate-type",
            "lib",
            "--target",
            "wasm32-unknown-unknown",
            temp_path,
            "-o",
            "/tmp/verify_code_temp.wasm",
            "--error-format",
            "short",
        ])
        .output();

    // Clean up
    let _ = fs::remove_file(temp_path);
    let _ = fs::remove_file("/tmp/verify_code_temp.wasm");

    match output {
        Ok(result) => result.status.success(),
        Err(_) => false,
    }
}

/// Generate diverse Rust function examples for training
pub fn generate_code_examples() -> Vec<String> {
    let mut examples = Vec::new();

    // Arithmetic functions
    examples.push("fn add(a: u32, b: u32) -> u32 { a + b }".to_string());
    examples.push("fn sub(a: i32, b: i32) -> i32 { a - b }".to_string());
    examples.push("fn mul(a: u32, b: u32) -> u32 { a * b }".to_string());
    examples.push("fn div(a: u32, b: u32) -> u32 { a / b }".to_string());
    examples.push("fn square(x: i32) -> i32 { x * x }".to_string());
    examples.push("fn cube(x: i32) -> i32 { x * x * x }".to_string());
    examples.push("fn double(x: u32) -> u32 { x + x }".to_string());
    examples.push("fn triple(x: u32) -> u32 { x * 3 }".to_string());

    // Boolean functions
    examples.push("fn is_even(n: u32) -> bool { n % 2 == 0 }".to_string());
    examples.push("fn is_odd(n: u32) -> bool { n % 2 == 1 }".to_string());
    examples.push("fn is_zero(n: i32) -> bool { n == 0 }".to_string());
    examples.push("fn is_positive(n: i32) -> bool { n > 0 }".to_string());
    examples.push("fn is_negative(n: i32) -> bool { n < 0 }".to_string());
    examples.push("fn not(b: bool) -> bool { !b }".to_string());
    examples.push("fn and(a: bool, b: bool) -> bool { a && b }".to_string());
    examples.push("fn or(a: bool, b: bool) -> bool { a || b }".to_string());

    // Conditional functions
    examples.push("fn max(a: i32, b: i32) -> i32 { if a > b { a } else { b } }".to_string());
    examples.push("fn min(a: i32, b: i32) -> i32 { if a < b { a } else { b } }".to_string());
    examples.push("fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }".to_string());
    examples.push("fn sign(x: i32) -> i32 { if x > 0 { 1 } else if x < 0 { -1 } else { 0 } }".to_string());
    examples.push("fn clamp(x: i32, lo: i32, hi: i32) -> i32 { if x < lo { lo } else if x > hi { hi } else { x } }".to_string());

    // Simple recursion (for training)
    examples.push("fn factorial(n: u32) -> u32 { if n == 0 { 1 } else { n * factorial(n - 1) } }".to_string());
    examples.push("fn countdown(n: u32) -> u32 { if n == 0 { 0 } else { countdown(n - 1) } }".to_string());
    examples.push("fn sum_to(n: u32) -> u32 { if n == 0 { 0 } else { n + sum_to(n - 1) } }".to_string());

    // String functions
    examples.push("fn len(s: &str) -> usize { s.len() }".to_string());
    examples.push("fn is_empty(s: &str) -> bool { s.is_empty() }".to_string());
    examples.push("fn first_char(s: &str) -> Option<char> { s.chars().next() }".to_string());

    // Tuple functions
    examples.push("fn swap(pair: (i32, i32)) -> (i32, i32) { (pair.1, pair.0) }".to_string());
    examples.push("fn fst(pair: (i32, i32)) -> i32 { pair.0 }".to_string());
    examples.push("fn snd(pair: (i32, i32)) -> i32 { pair.1 }".to_string());

    // Generate variations with different types
    for i in 0..20 {
        examples.push(format!("fn func_{}(x: u32) -> u32 {{ x + {} }}", i, i));
        examples.push(format!("fn calc_{}(a: i32, b: i32) -> i32 {{ a * {} + b }}", i, i));
        examples.push(format!("fn check_{}(n: u32) -> bool {{ n > {} }}", i, i * 10));
    }

    // Generate simple loops (training set)
    examples.push("fn sum_range(n: u32) -> u32 { let mut sum = 0; for i in 0..n { sum += i; } sum }".to_string());
    examples.push("fn count_to(n: u32) -> u32 { let mut i = 0; while i < n { i += 1; } i }".to_string());
    examples.push("fn product_range(n: u32) -> u32 { let mut prod = 1; for i in 1..=n { prod *= i; } prod }".to_string());

    // Array/vector operations
    examples.push("fn sum_array(arr: &[i32]) -> i32 { arr.iter().sum() }".to_string());
    examples.push("fn first(arr: &[i32]) -> Option<&i32> { arr.first() }".to_string());
    examples.push("fn last(arr: &[i32]) -> Option<&i32> { arr.last() }".to_string());

    // Option handling
    examples.push("fn unwrap_or_zero(opt: Option<u32>) -> u32 { opt.unwrap_or(0) }".to_string());
    examples.push("fn is_some(opt: Option<i32>) -> bool { opt.is_some() }".to_string());
    examples.push("fn is_none(opt: Option<i32>) -> bool { opt.is_none() }".to_string());

    // Result handling
    examples.push("fn ok_value(res: Result<u32, ()>) -> Option<u32> { res.ok() }".to_string());
    examples.push("fn is_ok(res: Result<i32, ()>) -> bool { res.is_ok() }".to_string());
    examples.push("fn is_err(res: Result<i32, ()>) -> bool { res.is_err() }".to_string());

    // Pad to 500 with more variations
    while examples.len() < 500 {
        let idx = examples.len();
        examples.push(format!("fn helper_{}(x: u32, y: u32) -> u32 {{ x * {} + y * {} }}", idx, idx % 10, (idx + 1) % 10));
    }

    examples
}

/// Generate test examples with UNSEEN patterns (recursion, pattern matching, closures)
pub fn generate_test_code_examples() -> Vec<String> {
    vec![
        // Advanced recursion (UNSEEN in training)
        "fn fibonacci(n: u32) -> u32 { match n { 0 => 0, 1 => 1, _ => fibonacci(n - 1) + fibonacci(n - 2) } }".to_string(),
        "fn power(base: u32, exp: u32) -> u32 { match exp { 0 => 1, _ => base * power(base, exp - 1) } }".to_string(),
        "fn gcd(a: u32, b: u32) -> u32 { if b == 0 { a } else { gcd(b, a % b) } }".to_string(),

        // Pattern matching (UNSEEN)
        "fn classify(n: i32) -> &'static str { match n { 0 => \"zero\", 1..=10 => \"small\", _ => \"large\" } }".to_string(),
        "fn option_to_result(opt: Option<u32>) -> Result<u32, ()> { match opt { Some(x) => Ok(x), None => Err(()) } }".to_string(),

        // Closures (UNSEEN)
        "fn apply<F>(f: F, x: i32) -> i32 where F: Fn(i32) -> i32 { f(x) }".to_string(),
        "fn make_adder(n: i32) -> impl Fn(i32) -> i32 { move |x| x + n }".to_string(),

        // Iterator methods (UNSEEN)
        "fn filter_even(v: Vec<u32>) -> Vec<u32> { v.into_iter().filter(|x| x % 2 == 0).collect() }".to_string(),
        "fn map_double(v: Vec<u32>) -> Vec<u32> { v.into_iter().map(|x| x * 2).collect() }".to_string(),

        // Struct methods (UNSEEN)
        "struct Point { x: i32, y: i32 } impl Point { fn new(x: i32, y: i32) -> Self { Point { x, y } } }".to_string(),
    ]
}

/// Generate non-code examples
pub fn generate_non_code_examples() -> Vec<String> {
    let mut examples = Vec::new();

    // Natural language
    examples.push("The weather is nice today".to_string());
    examples.push("I have 7 apples in the basket".to_string());
    examples.push("Born in 1984 in California".to_string());
    examples.push("Chapter 3 discusses algorithms".to_string());
    examples.push("Room 42 is empty right now".to_string());
    examples.push("Age 25 years old next month".to_string());
    examples.push("Section 9 begins on page 100".to_string());
    examples.push("Level 8 unlocked after boss".to_string());
    examples.push("Page 15 shows the diagram".to_string());
    examples.push("Year 2024 saw many changes".to_string());

    // Math expressions (not code)
    examples.push("dozenal: 3 + 4 = ".to_string());
    examples.push("Calculate 7 plus 8".to_string());
    examples.push("Sum of 5 and 9".to_string());
    examples.push("Add 2 and 3".to_string());
    examples.push("3 times 4 equals 12".to_string());

    // Instructions/commands
    examples.push("Please compute the total".to_string());
    examples.push("Find the maximum value".to_string());
    examples.push("Sort the list ascending".to_string());
    examples.push("Check if even number".to_string());
    examples.push("Return the result".to_string());

    // Questions
    examples.push("What is the square root?".to_string());
    examples.push("How many items are there?".to_string());
    examples.push("Is this value correct?".to_string());
    examples.push("Can you verify this?".to_string());
    examples.push("Where is the function?".to_string());

    // Partial code-like but invalid
    examples.push("function without body".to_string());
    examples.push("if x greater than 5".to_string());
    examples.push("return the sum".to_string());
    examples.push("loop until done".to_string());
    examples.push("call the helper".to_string());

    // Generate variations
    for i in 0..100 {
        examples.push(format!("This is sentence number {}", i));
        examples.push(format!("The value {} is important", i * 10));
        examples.push(format!("Chapter {} describes the method", i + 1));
        examples.push(format!("Test case {} passed successfully", i));
    }

    // Pad to 500
    while examples.len() < 500 {
        let idx = examples.len();
        examples.push(format!("Random text example number {}", idx));
    }

    examples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_valid_rust() {
        let code = "fn add(a: u32, b: u32) -> u32 { a + b }";
        assert!(verify_rust_code(code));
    }

    #[test]
    fn test_verify_invalid_rust() {
        let code = "this is not valid rust code";
        assert!(!verify_rust_code(code));
    }

    #[test]
    fn test_generate_datasets() {
        let code = generate_code_examples();
        assert_eq!(code.len(), 500);

        let non_code = generate_non_code_examples();
        assert_eq!(non_code.len(), 500);
    }
}
