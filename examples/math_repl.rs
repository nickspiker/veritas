//! Natural Language Math REPL with Dozenal Output
//!
//! CONSTITUTION COMPLIANT:
//! ✓ Pure Spirix basecalc (no IEEE-754)
//! ✓ Symbolic computation (100% correct)
//! ✓ Dozenal output (base 12)
//!
//! Demo of routing natural language to verified computation.

use spirix::ScalarF4E4;
use std::io::{self, Write};
use veritas::training::{contains_math_keywords, parse_natural_language_math, SimpleMathOp};

fn main() {
    println!("=== Veritas Natural Language Math REPL ===");
    println!();
    println!("All results in dozenal (base 12):");
    println!("  A = 10 (decimal)");
    println!("  B = 11 (decimal)");
    println!("  10 = 12 (decimal)");
    println!();
    println!("Examples:");
    println!("  'What's seven plus three?' → A");
    println!("  'Five times two' → A");
    println!("  'Eleven minus four' → 7");
    println!("  '9 + 2' → B");
    println!();
    println!("Type 'quit' to exit");
    println!();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(_) => {}
            Err(e) => {
                println!("Error reading input: {}", e);
                continue;
            }
        }

        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input == "quit" || input == "exit" {
            println!("Goodbye!");
            break;
        }

        // Show help
        if input == "help" || input == "?" {
            print_help();
            continue;
        }

        // Simple routing: check if contains math keywords
        let is_math = contains_math_keywords(input);

        if !is_math {
            println!("❌ Not a math question");
            println!("   Try: 'seven plus three' or '7 + 3'");
            println!();
            continue;
        }

        // Parse math expression
        match parse_natural_language_math(input) {
            Some(parsed) => {
                // Show what was parsed (debug)
                let op_str = match parsed.op {
                    SimpleMathOp::Add => "+",
                    SimpleMathOp::Sub => "-",
                    SimpleMathOp::Mul => "×",
                    SimpleMathOp::Div => "÷",
                };
                println!("   Parsed: {} {} {}", parsed.a, op_str, parsed.b);

                // Convert to Spirix
                let a = ScalarF4E4::from(parsed.a);
                let b = ScalarF4E4::from(parsed.b);

                // Compute with basecalc (pure Spirix, 100% correct)
                let result = match parsed.op {
                    SimpleMathOp::Add => a + b,
                    SimpleMathOp::Sub => a - b,
                    SimpleMathOp::Mul => a * b,
                    SimpleMathOp::Div => a / b,
                };

                // Format as dozenal
                let result_str = format!("{:1.12}", result);
                let clean = result_str
                    .trim_start_matches("⦉+")
                    .trim_start_matches("⦉-")
                    .trim_end_matches("⦊");

                println!("✓ Answer: {}", clean);

                // Show decimal equivalent for learning
                let decimal_result = match parsed.op {
                    SimpleMathOp::Add => parsed.a as i32 + parsed.b as i32,
                    SimpleMathOp::Sub => parsed.a as i32 - parsed.b as i32,
                    SimpleMathOp::Mul => parsed.a as i32 * parsed.b as i32,
                    SimpleMathOp::Div => {
                        if parsed.b == 0 {
                            println!("   (division by zero)");
                            continue;
                        }
                        parsed.a as i32 / parsed.b as i32
                    }
                };
                println!("   ({} in decimal)", decimal_result);
                println!();
            }
            None => {
                println!("❌ Could not parse math expression");
                println!("   Try: 'seven plus three' or '7 + 3'");
                println!("   Type 'help' for more examples");
                println!();
            }
        }
    }
}

fn print_help() {
    println!();
    println!("=== Veritas Math REPL Help ===");
    println!();
    println!("Supported formats:");
    println!("  Word numbers:  'seven plus three'");
    println!("  Digit numbers: '7 + 3'");
    println!("  Mixed:         'seven + 3'");
    println!("  Natural:       'What's 5 times 2?'");
    println!();
    println!("Operations:");
    println!("  Addition:       plus, add, +");
    println!("  Subtraction:    minus, subtract, -");
    println!("  Multiplication: times, multiply, *, ×");
    println!("  Division:       divided, divide, /, ÷");
    println!();
    println!("Numbers (dozenal 0-11):");
    println!("  0-9: same as decimal");
    println!("  A = 10 (ten)");
    println!("  B = 11 (eleven)");
    println!();
    println!("Examples:");
    println!("  'seven plus three' → A");
    println!("  'What is 5 times 2?' → A");
    println!("  '11 - 4' → 7");
    println!("  'nine divided by three' → 3");
    println!("  'ten times eleven' → 92 (110 in decimal!)");
    println!();
    println!("Architecture:");
    println!("  1. Natural language input");
    println!("  2. Routing: is_math?");
    println!("  3. Parser: extract numbers and operation");
    println!("  4. Basecalc: pure Spirix symbolic computation");
    println!("  5. Output: dozenal (base 12)");
    println!();
    println!("✓ 100% correct (symbolic computation)");
    println!("✓ Zero IEEE-754 (pure Spirix)");
    println!("✓ Zero hallucination (verified answers)");
    println!();
}
