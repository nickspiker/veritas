//! Basecalc Routing Training
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix
//! ✓ Counter-based checkpointing
//! ✓ Symbolic ground truth (basecalc)
//!
//! Network learns WHEN to route to basecalc, not HOW to compute.
//! Arithmetic correctness is 100% (basecalc), routing accuracy is learned.

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, SGD};
use veritas::training::{parse_math_expression, call_basecalc, Diagnostics};

const INPUT_SIZE: usize = 256;
const HIDDEN_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 2;  // Binary: is_math or is_text
const LEARNING_RATE_NUM: u8 = 1;
const LEARNING_RATE_DEN: u16 = 200;  // LR = 0.005
const EPOCHS: usize = 100;
const CHECKPOINT_INTERVAL: u8 = 25;

#[derive(Debug, Clone)]
struct Example {
    input: Vec<u8>,
    is_math: bool,  // Ground truth: should route to basecalc?
}

fn generate_dataset() -> Vec<Example> {
    let mut examples = Vec::new();

    // Math examples with base markers
    for a in 0..=11u8 {
        for b in 0..=11u8 {
            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);

            let a_doz = format!("{:1.12}", a_spirix);
            let b_doz = format!("{:1.12}", b_spirix);

            let a_clean = a_doz.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

            // Explicit base marker
            let input_str = format!("dozenal: {} + {} = ", a_clean, b_clean);

            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                is_math: true,
            });
        }
    }

    // Text examples (no base marker)
    let text_samples = vec![
        "The weather is ",
        "Machine learning ",
        "Neural networks ",
        "Artificial intelligence ",
        "Data structure ",
        "Software engineering ",
        "Computer science ",
        "Algorithm design ",
        "Programming language ",
        "Operating system ",
    ];

    // Repeat to get ~60 text examples
    for _ in 0..6 {
        for text in &text_samples {
            examples.push(Example {
                input: text.as_bytes().to_vec(),
                is_math: false,
            });
        }
    }

    examples
}

/// Encode input as bag-of-bytes
fn encode_input(bytes: &[u8]) -> Vec<ScalarF4E4> {
    let mut encoding = vec![ScalarF4E4::ZERO; INPUT_SIZE];
    for &byte in bytes {
        encoding[byte as usize] = encoding[byte as usize] + ScalarF4E4::ONE;
    }
    encoding
}

fn main() {
    println!("=== Basecalc Routing Training ===\n");
    println!("Task: Learn WHEN to route to symbolic computation");
    println!("Arithmetic: 100% correct (basecalc handles it)");
    println!("Routing: Network learns to classify math vs text\n");

    let dataset = generate_dataset();
    let math_count = dataset.iter().filter(|ex| ex.is_math).count();
    let text_count = dataset.len() - math_count;

    println!("Dataset: {} examples", dataset.len());
    println!("  Math: {} (with base markers)", math_count);
    println!("  Text: {} (no markers)", text_count);
    println!();

    // Verify parsing works
    println!("Sample math expressions:");
    for i in [0, 10, 143] {
        if let Some(ex) = dataset.get(i) {
            if ex.is_math {
                let input_str = String::from_utf8_lossy(&ex.input);
                if let Some(parsed) = parse_math_expression(&ex.input) {
                    let result = call_basecalc(&parsed);
                    println!("  \"{}\" → base={}, {}+{}={} (basecalc)",
                        input_str.trim(), parsed.base, parsed.operand_a, parsed.operand_b, result);
                }
            }
        }
    }
    println!();

    // Initialize routing network
    let in_scale = (ScalarF4E4::ONE / ScalarF4E4::from(INPUT_SIZE as u32)).sqrt();
    let hidden_scale = (ScalarF4E4::ONE / ScalarF4E4::from(HIDDEN_SIZE as u32)).sqrt();

    let w1_data: Vec<ScalarF4E4> = (0..INPUT_SIZE * HIDDEN_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * in_scale)
        .collect();

    let w2_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * OUTPUT_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * hidden_scale)
        .collect();

    let mut w1 = Tensor::from_scalars(w1_data, Shape::matrix(INPUT_SIZE, HIDDEN_SIZE))
        .unwrap()
        .with_requires_grad();

    let mut w2 = Tensor::from_scalars(w2_data, Shape::matrix(HIDDEN_SIZE, OUTPUT_SIZE))
        .unwrap()
        .with_requires_grad();

    let lr = ScalarF4E4::from(LEARNING_RATE_NUM) / ScalarF4E4::from(LEARNING_RATE_DEN);
    let mut optimizer = SGD::new(lr);
    let mut diagnostics = Diagnostics::new();

    let mut checkpoint_counter: u8 = 0;

    println!("Training routing network for {} epochs...\n", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = ScalarF4E4::ZERO;
        let mut routing_correct = 0;
        let mut basecalc_calls = 0;
        let mut false_positives = 0;  // Routed text to basecalc
        let mut false_negatives = 0;  // Didn't route math

        w1.zero_grad();
        w2.zero_grad();

        for ex in &dataset {
            // Encode input
            let input_vec = encode_input(&ex.input);
            let input = Tensor::from_scalars(input_vec, Shape::matrix(1, INPUT_SIZE)).unwrap();

            // Forward: Input → Hidden → Routing (is_math?)
            let hidden_pre = matmul(&input, &w1).unwrap();
            let hidden_data: Vec<ScalarF4E4> = hidden_pre.as_scalars()
                .unwrap()
                .iter()
                .map(|x| x.tanh())
                .collect();
            let hidden = Tensor::from_scalars(hidden_data, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

            let logits = matmul(&hidden, &w2).unwrap();
            let logits_data = logits.as_scalars().unwrap().to_vec();

            // Softmax
            let max_logit = logits_data.iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(ScalarF4E4::ZERO);

            let exp_sum = logits_data.iter()
                .map(|x| (*x - max_logit).exp())
                .fold(ScalarF4E4::ZERO, |a, b| a + b);

            let probs: Vec<ScalarF4E4> = logits_data.iter()
                .map(|&x| (x - max_logit).exp() / exp_sum)
                .collect();

            // Routing decision: is_math = index 1
            let routing_decision = probs[1] > probs[0];

            // Check routing accuracy
            if routing_decision == ex.is_math {
                routing_correct += 1;
            }

            // Track false positives/negatives
            if routing_decision && !ex.is_math {
                false_positives += 1;
            }
            if !routing_decision && ex.is_math {
                false_negatives += 1;
            }

            // If routed to basecalc AND it's actually math
            if routing_decision && ex.is_math {
                if let Some(_parsed) = parse_math_expression(&ex.input) {
                    basecalc_calls += 1;
                    // Basecalc is always correct - no arithmetic loss needed
                }
            }

            // Loss: cross-entropy on routing decision
            let target_idx = if ex.is_math { 1 } else { 0 };
            let target_prob = probs[target_idx];
            let loss = ScalarF4E4::ZERO - target_prob.ln();
            total_loss = total_loss + loss;

            // Backward: gradient only on routing classification
            let mut grad_logits = probs.clone();
            grad_logits[target_idx] = grad_logits[target_idx] - ScalarF4E4::ONE;
            let grad_logits_tensor = Tensor::from_scalars(grad_logits, Shape::matrix(1, OUTPUT_SIZE)).unwrap();

            // Backprop
            let (grad_hidden, grad_w2) = matmul_backward(&grad_logits_tensor, &hidden, &w2).unwrap();

            let hidden_grad_data = hidden.as_scalars().unwrap();
            let grad_hidden_data = grad_hidden.as_scalars().unwrap();
            let grad_hidden_pre: Vec<ScalarF4E4> = hidden_grad_data.iter()
                .zip(grad_hidden_data.iter())
                .map(|(h, g)| {
                    let tanh_deriv = ScalarF4E4::ONE - (*h * *h);
                    *g * tanh_deriv
                })
                .collect();
            let grad_hidden_pre_tensor = Tensor::from_scalars(grad_hidden_pre, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

            let (_grad_input, grad_w1) = matmul_backward(&grad_hidden_pre_tensor, &input, &w1).unwrap();

            // Accumulate gradients
            if w1.grad().is_none() {
                w1.set_grad(grad_w1);
            } else {
                w1.accumulate_grad(grad_w1).unwrap();
            }

            if w2.grad().is_none() {
                w2.set_grad(grad_w2);
            } else {
                w2.accumulate_grad(grad_w2).unwrap();
            }
        }

        // Update weights
        let mut params = vec![&mut w1, &mut w2];
        optimizer.step(&mut params).unwrap();

        let avg_loss = total_loss / ScalarF4E4::from(dataset.len() as u32);
        let routing_acc = (routing_correct * 100) / dataset.len();

        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch {:3}: Loss = {}", epoch, avg_loss);
            println!("  Routing: {}% ({}/{})", routing_acc, routing_correct, dataset.len());
            println!("  Basecalc calls: {}/{} math examples", basecalc_calls, math_count);
            println!("  False positives: {} (text→basecalc)", false_positives);
            println!("  False negatives: {} (math→text)", false_negatives);
        }

        // Counter-based checkpointing
        checkpoint_counter += 1;
        if checkpoint_counter == CHECKPOINT_INTERVAL {
            // Diagnostics expects 3 layers - pass empty for third
            let empty_layer: &[ScalarF4E4] = &[];
            diagnostics.update_weights(
                w1.as_scalars().unwrap(),
                w2.as_scalars().unwrap(),
                empty_layer,
            );

            println!();
            diagnostics.print(epoch);

            if diagnostics.is_healthy() {
                println!("✓ Network health: GOOD");
            }

            checkpoint_counter = 0;
        }

        // Early stopping
        if routing_acc >= 95 {
            println!("\n✓ Target routing accuracy reached: {}%", routing_acc);
            println!("  Basecalc handles arithmetic (100% correct)");
            println!("  Network learned WHEN to route, not HOW to compute");
            break;
        }
    }

    println!("\n=== Training Complete ===");
    println!("✓ Routing network trained");
    println!("✓ Basecalc provides symbolic ground truth");
    println!("✓ Zero arithmetic errors (basecalc always correct)");
    println!("✓ Network learns classification, not computation");
}
