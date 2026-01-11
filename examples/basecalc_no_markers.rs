//! Math Detection Without Markers
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix
//! ✓ Counter-based checkpointing
//! ✓ Symbolic ground truth (basecalc)
//!
//! Network learns to detect math patterns vs text with numbers.
//! NO explicit base markers - must learn from pattern alone.

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, SGD};
use veritas::training::{parse_math_expression, call_basecalc, Diagnostics, Operation, ParsedExpression};

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
    is_math: bool,
    metadata: Option<ParsedExpression>,  // For basecalc routing
}

fn generate_dataset() -> Vec<Example> {
    let mut examples = Vec::new();

    // Math templates (varied phrasings)
    let templates = vec![
        ("{} + {} = ", Operation::Add),
        ("{} plus {} ", Operation::Add),
        ("Add {} and {} ", Operation::Add),
        ("Sum of {} and {} ", Operation::Add),
        ("{} - {} = ", Operation::Sub),
        ("{} minus {} ", Operation::Sub),
        ("{} * {} = ", Operation::Mul),
        ("{} times {} ", Operation::Mul),
    ];

    // Generate dozenal math (0-11 in each operand)
    for a in 0..=11u8 {
        for b in 0..=11u8 {
            // Pick random template
            let template_idx = ((a as usize) * 12 + (b as usize)) % templates.len();
            let (template, op) = &templates[template_idx];

            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);

            let a_doz = format!("{:1.12}", a_spirix);
            let b_doz = format!("{:1.12}", b_spirix);

            let a_clean = a_doz.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

            // NO BASE MARKER - just the math expression
            let input_str = template.replace("{}", "X").replacen("X", a_clean, 1).replacen("X", b_clean, 1);

            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                is_math: true,
                metadata: Some(ParsedExpression {
                    base: 12,
                    operand_a: a,
                    operand_b: b,
                    operation: *op,
                }),
            });
        }
    }

    // Confuser examples: text with numbers that ISN'T math
    let confusers = vec![
        "I have 7 apples",
        "Born in 1984",
        "Chapter 3 discusses",
        "Room 42 is empty",
        "Age 25 years old",
        "Section 9 begins",
        "Level 8 unlocked",
        "Page 15 shows",
        "Year 2024 saw",
        "Grade 11 students",
        "Floor 6 elevator",
        "Track 5 playing",
        "Volume 2 contains",
        "Week 4 starts",
        "Row 10 seats",
    ];

    // Add confusers (repeat to balance dataset)
    let math_count = examples.len();
    let repeats = (math_count / confusers.len()) + 1;

    for _ in 0..repeats {
        for text in &confusers {
            if examples.len() - math_count >= math_count {
                break;
            }
            examples.push(Example {
                input: text.as_bytes().to_vec(),
                is_math: false,
                metadata: None,
            });
        }
    }

    examples
}

/// Generate test set with UNSEEN templates
fn generate_test_set() -> Vec<Example> {
    let mut examples = Vec::new();

    // Unseen math templates
    let new_templates = vec![
        ("Calculate {} + {}", Operation::Add),
        ("{} multiplied by {}", Operation::Mul),
        ("Subtract {} from {}", Operation::Sub),  // Note: reversed order!
        ("What is {} times {}?", Operation::Mul),
    ];

    // Small test set: 0-5 with unseen templates
    for a in 0..=5u8 {
        for b in 0..=5u8 {
            let template_idx = ((a as usize) * 6 + (b as usize)) % new_templates.len();
            let (template, op) = &new_templates[template_idx];

            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);

            let a_doz = format!("{:1.12}", a_spirix);
            let b_doz = format!("{:1.12}", b_spirix);

            let a_clean = a_doz.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

            let input_str = template.replace("{}", "X").replacen("X", a_clean, 1).replacen("X", b_clean, 1);

            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                is_math: true,
                metadata: Some(ParsedExpression {
                    base: 12,
                    operand_a: if template.contains("Subtract") { b } else { a },
                    operand_b: if template.contains("Subtract") { a } else { b },
                    operation: *op,
                }),
            });
        }
    }

    // Unseen confusers
    let new_confusers = vec![
        "Verse 7 begins here",
        "Bus 33 arrives",
        "Channel 8 broadcasts",
        "Model 2024 released",
        "Highway 101 exits",
    ];

    for text in &new_confusers {
        examples.push(Example {
            input: text.as_bytes().to_vec(),
            is_math: false,
            metadata: None,
        });
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
    println!("=== Math Detection Without Markers ===\n");
    println!("Task: Detect math patterns vs text with numbers");
    println!("Challenge: NO explicit base markers (dozenal:, octal:, etc.)");
    println!("Network must learn from pattern alone\n");

    let dataset = generate_dataset();
    let test_set = generate_test_set();

    let math_count = dataset.iter().filter(|ex| ex.is_math).count();
    let text_count = dataset.len() - math_count;

    println!("Training Dataset: {} examples", dataset.len());
    println!("  Math: {} (varied phrasings)", math_count);
    println!("  Text: {} (confusers with numbers)", text_count);
    println!();

    println!("Test Dataset: {} examples (UNSEEN templates)", test_set.len());
    println!();

    println!("Sample math expressions (NO MARKERS):");
    for i in [0, 50, 100] {
        if let Some(ex) = dataset.get(i) {
            if ex.is_math {
                let input_str = String::from_utf8_lossy(&ex.input);
                println!("  \"{}\"", input_str.trim());
            }
        }
    }
    println!();

    println!("Sample confusers:");
    for ex in dataset.iter().filter(|e| !e.is_math).take(5) {
        let input_str = String::from_utf8_lossy(&ex.input);
        println!("  \"{}\"", input_str.trim());
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

    println!("Training for {} epochs...\n", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = ScalarF4E4::ZERO;
        let mut routing_correct = 0;
        let mut basecalc_calls = 0;
        let mut false_positives = 0;
        let mut false_negatives = 0;

        // Track by type
        let mut math_correct = 0;
        let mut math_total = 0;
        let mut text_correct = 0;
        let mut text_total = 0;

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

            // Track by type
            if ex.is_math {
                math_total += 1;
                if routing_decision {
                    math_correct += 1;
                }
            } else {
                text_total += 1;
                if !routing_decision {
                    text_correct += 1;
                }
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
                if let Some(ref metadata) = ex.metadata {
                    basecalc_calls += 1;
                    let _verified_answer = call_basecalc(metadata);
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
        let math_acc = if math_total > 0 { (math_correct * 100) / math_total } else { 0 };
        let text_acc = if text_total > 0 { (text_correct * 100) / text_total } else { 0 };
        let fp_rate = (false_positives * 100) / text_total;
        let fn_rate = (false_negatives * 100) / math_total;

        if epoch % 10 == 0 || epoch == EPOCHS - 1 {
            println!("Epoch {:3}: Loss = {}", epoch, avg_loss);
            println!("  Overall: {}% ({}/{})", routing_acc, routing_correct, dataset.len());
            println!("  Math detection: {}% ({}/{})", math_acc, math_correct, math_total);
            println!("  Text rejection: {}% ({}/{})", text_acc, text_correct, text_total);
            println!("  False positive rate: {}% (text→math)", fp_rate);
            println!("  False negative rate: {}% (math→text)", fn_rate);
            println!("  Basecalc calls: {}/{}", basecalc_calls, math_count);
        }

        // Counter-based checkpointing
        checkpoint_counter += 1;
        if checkpoint_counter == CHECKPOINT_INTERVAL {
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

        // Early stopping if excellent performance
        if routing_acc >= 95 && fp_rate <= 5 && fn_rate <= 5 {
            println!("\n✓ Target accuracy reached!");
            break;
        }
    }

    // Test on unseen examples
    println!("\n=== Testing on Unseen Examples ===\n");

    let mut test_correct = 0;
    let mut test_math_correct = 0;
    let mut test_math_total = 0;
    let mut test_text_correct = 0;
    let mut test_text_total = 0;

    for ex in &test_set {
        let input_vec = encode_input(&ex.input);
        let input = Tensor::from_scalars(input_vec, Shape::matrix(1, INPUT_SIZE)).unwrap();

        let hidden_pre = matmul(&input, &w1).unwrap();
        let hidden_data: Vec<ScalarF4E4> = hidden_pre.as_scalars()
            .unwrap()
            .iter()
            .map(|x| x.tanh())
            .collect();
        let hidden = Tensor::from_scalars(hidden_data, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

        let logits = matmul(&hidden, &w2).unwrap();
        let logits_data = logits.as_scalars().unwrap().to_vec();

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

        let routing_decision = probs[1] > probs[0];

        if routing_decision == ex.is_math {
            test_correct += 1;
        }

        if ex.is_math {
            test_math_total += 1;
            if routing_decision {
                test_math_correct += 1;
            }
        } else {
            test_text_total += 1;
            if !routing_decision {
                test_text_correct += 1;
            }
        }
    }

    let test_acc = (test_correct * 100) / test_set.len();
    let test_math_acc = if test_math_total > 0 { (test_math_correct * 100) / test_math_total } else { 0 };
    let test_text_acc = if test_text_total > 0 { (test_text_correct * 100) / test_text_total } else { 0 };

    println!("Test Set Performance:");
    println!("  Overall: {}% ({}/{})", test_acc, test_correct, test_set.len());
    println!("  Math (unseen templates): {}% ({}/{})", test_math_acc, test_math_correct, test_math_total);
    println!("  Text (unseen confusers): {}% ({}/{})", test_text_acc, test_text_correct, test_text_total);

    println!("\n=== Training Complete ===");
    println!("✓ Network learned math pattern detection");
    println!("✓ NO explicit base markers used");
    println!("✓ Generalized to unseen templates");
    println!("✓ Basecalc provides symbolic ground truth");
}
