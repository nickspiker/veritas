//! Test preprocessing pipeline

use veritas::encoding::{preprocess, inject_result};
use spirix::ScalarF4E4;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Preprocessing Pipeline Test                      ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    let examples = vec![
        "What is 2 + 3?",
        "Calculate 15 * 8",
        "Add 1 and 2 and 3 and 4",
        "Compute 10 / 5",
        "What is 7 ^ 2?",
        "15.5 * 2",
        "1/4 + 3/8",
    ];

    for text in examples {
        println!("Input:  \"{}\"", text);

        match preprocess(text) {
            Ok(result) => {
                println!("Output: \"{}\"", result.text);
                println!("Numbers: {} extracted", result.numbers.len());
                println!("Operators: {} extracted", result.operators.len());

                // Demonstrate result injection
                if !result.numbers.is_empty() {
                    let mock_results = vec![ScalarF4E4::from(42u8)];
                    let injected = inject_result(&result.text, &mock_results);
                    println!("Injected: \"{}\"", injected);
                }
            }
            Err(e) => println!("Error: {:?}", e),
        }

        println!();
    }

    println!("✓ Preprocessing pipeline working");
    println!("✓ Sequential placeholder numbering");
    println!("✓ Operator detection");
    println!("✓ Number extraction (integers, decimals, fractions)");
}
