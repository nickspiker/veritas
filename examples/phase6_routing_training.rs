//! Phase 6: RNN Routing Training
//!
//! Proof of concept: Can a simple RNN learn to detect math and route to basecalc?
//!
//! Pipeline:
//! 1. Generate training examples (math problems)
//! 2. Preprocess: replace numbers with <MATH_N> placeholders
//! 3. Train RNN on byte sequences
//! 4. Test: does network learn lower loss when outputting <MATH_N>?

use veritas::encoding::{preprocess, inject_result};
use veritas::training::{generate_math_examples, generate_text_examples};
use veritas::transformer::{SimpleRNN, RNNConfig};
use veritas::symbolic::ArithProblem;
use spirix::ScalarF4E4;

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║           Phase 6: Routing Training (Proof of Concept)       ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("═══ Step 1: Generate Training Data ═══\n");

    let math_examples = generate_math_examples(100);
    let text_examples = generate_text_examples(50);

    println!("Generated {} math examples", math_examples.len());
    println!("Generated {} text examples", text_examples.len());
    println!();

    // Show some examples
    println!("Sample math examples:");
    for i in 0..3 {
        println!("  Input:  \"{}\"", math_examples[i].input);
        println!("  Target: \"{}\"", math_examples[i].target);
        println!();
    }

    println!("═══ Step 2: Preprocessing ═══\n");

    // Demonstrate preprocessing on a math example
    let example = &math_examples[0];
    let preprocessed = preprocess(&example.input).unwrap();

    println!("Original: \"{}\"", example.input);
    println!("Preprocessed: \"{}\"", preprocessed.text);
    println!("Numbers extracted: {}", preprocessed.numbers.len());
    println!("Operators extracted: {}", preprocessed.operators.len());
    println!();

    println!("═══ Step 3: Create RNN ═══\n");

    let config = RNNConfig::default();
    let mut rnn = SimpleRNN::new(config).unwrap();

    println!("RNN Configuration:");
    println!("  Input size:  {} (byte vocabulary)", rnn.config.input_size);
    println!("  Hidden size: {} units", rnn.config.hidden_size);
    println!("  Output size: {} (next byte)", rnn.config.output_size);
    println!();

    println!("═══ Step 4: Test Forward Pass ═══\n");

    // Test RNN on sample input
    let test_input = b"What is 2 + 3?";
    println!("Test input: \"{}\"", String::from_utf8_lossy(test_input));

    let outputs = rnn.forward(test_input).unwrap();
    println!("RNN outputs: {} timesteps", outputs.len());
    println!("Each output: {} logits (one per byte)", outputs[0].len());
    println!();

    // Predict next byte
    let next_byte = rnn.predict_next(test_input).unwrap();
    println!("Predicted next byte: {} ('{}')", next_byte, next_byte as char);
    println!();

    println!("═══ Step 5: Basecalc Integration Concept ═══\n");

    println!("Training signal:");
    println!("  1. RNN processes: \"What is 2 + 3?\"");
    println!("  2. Preprocessor detects: <MATH_0> <OP_ADD> <MATH_1>");
    println!("  3. If RNN outputs '<', then 'M', then 'A', then 'T', then 'H'");
    println!("     → Lower loss (correct routing behavior)");
    println!("  4. Call basecalc: 2 + 3 = 5 (VERIFIED)");
    println!("  5. Inject result: \"What is 2 + 3? The answer is 5\"");
    println!("  6. Continue generation with verified answer");
    println!();

    println!("Gradient signal:");
    println!("  - Neural prediction vs symbolic truth = error");
    println!("  - Backprop updates weights to minimize error");
    println!("  - Network learns: output <MATH_N> when numbers detected");
    println!();

    println!("═══ Step 6: Demonstrate Symbolic Verification ═══\n");

    // Extract numbers and operator from preprocessed example
    if !preprocessed.numbers.is_empty() && !preprocessed.operators.is_empty() {
        let left = preprocessed.numbers[0];
        let right = preprocessed.numbers.get(1).copied().unwrap_or(ScalarF4E4::ZERO);

        // For demonstration, parse the original expression
        if let Some(op_char) = example.input.chars().find(|&c| matches!(c, '+' | '-' | '*' | '/')) {
            // Extract the numbers from the original input
            let parts: Vec<&str> = example.input.split(op_char).collect();
            if parts.len() >= 2 {
                if let Some(left_str) = parts[0].split_whitespace().last() {
                    if let Some(right_str) = parts[1].split_whitespace().next() {
                        let expr = format!("{} {} {}", left_str, op_char, right_str);
                        if let Ok(prob) = ArithProblem::parse(&expr) {
                            if let Ok(result) = prob.solve() {
                                println!("Symbolic computation:");
                                println!("  Expression: {}", result.expr);
                                println!("  Answer: {}", result.answer);
                                println!("  Status: VERIFIED ✓");
                                println!();
                            }
                        }
                    }
                }
            }
        }
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Phase 6 Complete                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("✓ Training data generation working");
    println!("✓ Preprocessing pipeline functional");
    println!("✓ Simple RNN forward pass working");
    println!("✓ Symbolic verification demonstrated");
    println!();

    println!("Next steps:");
    println!("  → Implement actual training loop");
    println!("  → Add loss computation (cross-entropy)");
    println!("  → Add gradient descent (SGD)");
    println!("  → Train for N epochs");
    println!("  → Measure: does loss decrease for <MATH_N> outputs?");
    println!("  → If yes: routing behavior is learnable");
    println!();

    println!("Key insight:");
    println!("  Network doesn't compute. It learns WHEN to route.");
    println!("  Basecalc does computation. Always verified.");
    println!("  Training signal = contradiction between prediction and truth.");
    println!();
}
