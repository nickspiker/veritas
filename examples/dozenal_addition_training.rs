//! Dozenal (Base-12) Addition Training - The ASCII Gap Test
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix arithmetic
//! ✓ Spirix base-12 formatting
//! ✓ Full gradient flow
//!
//! Critical test: ASCII discontinuity between '9' (0x39) and 'A' (0x41)
//! Tests if network learned positional structure vs memorized patterns.
//!
//! Dozenal digits: 0-9, A (10), B (11)
//! ASCII gap: :;<=>?@ (7 bytes between '9' and 'A')

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, SGD};

const INPUT_SIZE: usize = 256;
const HIDDEN_SIZE: usize = 128;
const MATH_CLASSES: usize = 23;  // Dozenal sums 0-22 (decimal) = 0-1A (dozenal)
const TEXT_CLASSES: usize = 256;
const LEARNING_RATE_NUM: u8 = 1;
const LEARNING_RATE_DEN: u8 = 200; // LR = 0.005
const EPOCHS: usize = 150;

#[derive(Debug, Clone)]
struct Example {
    input: Vec<u8>,
    is_math: bool,
    target_value: u8,
    has_letters: bool,  // Track if this example uses A or B
}

fn generate_dataset() -> Vec<Example> {
    let mut examples = Vec::new();

    // Generate all dozenal addition problems (0-B + 0-B)
    for a in 0..=11u8 {
        for b in 0..=11u8 {
            let sum = a + b;

            // Format using Spirix base-12 (C = 12 in base-36)
            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);

            // Spirix format: {:width.base} where base 12 = dozenal
            let a_doz = format!("{:1.12}", a_spirix);
            let b_doz = format!("{:1.12}", b_spirix);

            // Strip ⦉+⦊ wrapper
            let a_clean = a_doz.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

            let input_str = format!("{} + {} = ", a_clean, b_clean);

            // Track if this example contains letters (A or B)
            let has_letters = a >= 10 || b >= 10;

            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                is_math: true,
                target_value: sum,
                has_letters,
            });
        }
    }

    // Add text examples
    let text_samples = vec![
        "The weather is ",
        "Hello world ",
        "Rust language ",
        "Machine learning ",
        "Neural network ",
        "Deep learning ",
        "Artificial intelligence ",
        "Computer science ",
        "Software engineering ",
        "Data structure ",
    ];

    let next_bytes = vec![
        b'n', b't', b'i', b'a', b't', b'm', b's', b'f', b'p', b's',
    ];

    // Repeat to get ~100 text examples (balance dataset)
    for _ in 0..10 {
        for (text, next_byte) in text_samples.iter().zip(next_bytes.iter()) {
            examples.push(Example {
                input: text.as_bytes().to_vec(),
                is_math: false,
                target_value: *next_byte,
                has_letters: false,
            });
        }
    }

    examples
}

fn encode_input(bytes: &[u8]) -> Vec<ScalarF4E4> {
    let mut encoding = vec![ScalarF4E4::ZERO; INPUT_SIZE];
    for &byte in bytes {
        encoding[byte as usize] = encoding[byte as usize] + ScalarF4E4::ONE;
    }
    encoding
}

fn verify_dozenal_sum(input_str: &str) -> Option<u8> {
    let parts: Vec<&str> = input_str.split_whitespace().collect();
    if parts.len() < 3 || parts[1] != "+" {
        return None;
    }

    // Parse dozenal strings to u8
    let a = u8::from_str_radix(parts[0].trim(), 12).ok()?;
    let b = u8::from_str_radix(parts[2].trim(), 12).ok()?;

    // Compute with Spirix
    let a_spirix = ScalarF4E4::from(a);
    let b_spirix = ScalarF4E4::from(b);
    let result_spirix = a_spirix + b_spirix;

    let result_u32: u32 = result_spirix.into();
    Some(result_u32 as u8)
}

fn main() {
    println!("=== Dozenal Addition Training - ASCII Gap Test ===\n");
    println!("Architecture:");
    println!("  Input(256) → Hidden(128)");
    println!("  Hidden → Routing head (binary: is_math?)");
    println!("  Hidden → Value head (math: 23 classes, text: 256 classes)");
    println!();
    println!("Critical test: ASCII discontinuity");
    println!("  '9' = 0x39, 'A' = 0x41 (gap of 7 bytes: :;<=>?@)");
    println!("  Does network learn structure or memorize patterns?");
    println!();

    let dataset = generate_dataset();
    println!("Dataset: {} examples", dataset.len());

    let math_count = dataset.iter().filter(|ex| ex.is_math).count();
    let text_count = dataset.len() - math_count;
    let digit_only = dataset.iter().filter(|ex| ex.is_math && !ex.has_letters).count();
    let with_letters = dataset.iter().filter(|ex| ex.is_math && ex.has_letters).count();

    println!("  Math examples: {} (dozenal 0-B + 0-B)", math_count);
    println!("    - Digit-only (0-9): {}", digit_only);
    println!("    - With letters (A/B): {}", with_letters);
    println!("  Text examples: {}", text_count);
    println!("Learning rate: 0.005");
    println!();

    // Verify examples and show ASCII encoding
    println!("Spirix dozenal verification (sample):");
    for i in [0, 9, 10, 11, 21].iter() {
        if let Some(ex) = dataset.get(*i) {
            if ex.is_math {
                let input_str = String::from_utf8_lossy(&ex.input);
                if let Some(verified) = verify_dozenal_sum(&input_str) {
                    let verified_spirix = ScalarF4E4::from(verified);
                    let verified_doz = format!("{:2.12}", verified_spirix);
                    let verified_clean = verified_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

                    // Show ASCII values for letter examples
                    let ascii_note = if ex.has_letters {
                        let bytes: Vec<String> = ex.input.iter()
                            .take(5)
                            .map(|b| if *b >= 0x41 && *b <= 0x42 {
                                format!("'{}'=0x{:02X}", *b as char, b)
                            } else if *b >= 0x30 && *b <= 0x39 {
                                format!("'{}'=0x{:02X}", *b as char, b)
                            } else {
                                format!("0x{:02X}", b)
                            })
                            .collect();
                        format!(" [ASCII: {}]", bytes.join(" "))
                    } else {
                        String::new()
                    };

                    println!("  \"{}\" → {} (decimal {}){}",
                        input_str.trim(), verified_clean, verified, ascii_note);
                }
            }
        }
    }
    println!();

    // Initialize weights
    let in_scale = (ScalarF4E4::ONE / ScalarF4E4::from(INPUT_SIZE as u32)).sqrt();
    let hidden_scale = (ScalarF4E4::ONE / ScalarF4E4::from(HIDDEN_SIZE as u32)).sqrt();

    let w1_data: Vec<ScalarF4E4> = (0..INPUT_SIZE * HIDDEN_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * in_scale)
        .collect();

    let w_routing_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * 2)
        .map(|_| ScalarF4E4::random_gauss() * hidden_scale)
        .collect();

    let w_math_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * MATH_CLASSES)
        .map(|_| ScalarF4E4::random_gauss() * hidden_scale)
        .collect();

    let w_text_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * TEXT_CLASSES)
        .map(|_| ScalarF4E4::random_gauss() * hidden_scale)
        .collect();

    let mut w1 = Tensor::from_scalars(w1_data, Shape::matrix(INPUT_SIZE, HIDDEN_SIZE))
        .unwrap()
        .with_requires_grad();

    let mut w_routing = Tensor::from_scalars(w_routing_data, Shape::matrix(HIDDEN_SIZE, 2))
        .unwrap()
        .with_requires_grad();

    let mut w_math = Tensor::from_scalars(w_math_data, Shape::matrix(HIDDEN_SIZE, MATH_CLASSES))
        .unwrap()
        .with_requires_grad();

    let mut w_text = Tensor::from_scalars(w_text_data, Shape::matrix(HIDDEN_SIZE, TEXT_CLASSES))
        .unwrap()
        .with_requires_grad();

    let lr = ScalarF4E4::from(LEARNING_RATE_NUM) / ScalarF4E4::from(LEARNING_RATE_DEN);
    let mut optimizer = SGD::new(lr);

    println!("Training for {} epochs...\n", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = ScalarF4E4::ZERO;
        let mut routing_correct = 0;
        let mut math_correct = 0;
        let mut math_digit_correct = 0;
        let mut math_letter_correct = 0;
        let mut text_correct = 0;
        let mut math_count_epoch = 0;
        let mut math_digit_count = 0;
        let mut math_letter_count = 0;
        let mut text_count_epoch = 0;

        w1.zero_grad();
        w_routing.zero_grad();
        w_math.zero_grad();
        w_text.zero_grad();

        for ex in &dataset {
            let input_vec = encode_input(&ex.input);
            let input = Tensor::from_scalars(input_vec, Shape::matrix(1, INPUT_SIZE)).unwrap();

            // Forward: input → hidden
            let hidden_pre = matmul(&input, &w1).unwrap();
            let hidden_data: Vec<ScalarF4E4> = hidden_pre.as_scalars()
                .unwrap()
                .iter()
                .map(|x| x.tanh())
                .collect();
            let hidden = Tensor::from_scalars(hidden_data.clone(), Shape::matrix(1, HIDDEN_SIZE)).unwrap();

            // Routing head
            let routing_logits = matmul(&hidden, &w_routing).unwrap();
            let routing_data = routing_logits.as_scalars().unwrap();

            let max_r = routing_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(ScalarF4E4::ZERO);
            let exp_sum_r = routing_data.iter().map(|x| (*x - max_r).exp()).fold(ScalarF4E4::ZERO, |a, b| a + b);
            let routing_probs: Vec<ScalarF4E4> = routing_data.iter().map(|x| (*x - max_r).exp() / exp_sum_r).collect();

            let predicted_routing = routing_probs[1] > routing_probs[0];
            if predicted_routing == ex.is_math {
                routing_correct += 1;
            }

            let routing_target = if ex.is_math { 1 } else { 0 };
            let routing_target_prob = routing_probs[routing_target];
            let routing_loss = ScalarF4E4::ZERO - routing_target_prob.ln();

            // Value head
            let (value_probs, value_loss, value_head_used) = if ex.is_math {
                math_count_epoch += 1;
                if ex.has_letters {
                    math_letter_count += 1;
                } else {
                    math_digit_count += 1;
                }

                let logits = matmul(&hidden, &w_math).unwrap();
                let logits_data = logits.as_scalars().unwrap();

                let max_v = logits_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(ScalarF4E4::ZERO);
                let exp_sum_v = logits_data.iter().map(|x| (*x - max_v).exp()).fold(ScalarF4E4::ZERO, |a, b| a + b);
                let probs: Vec<ScalarF4E4> = logits_data.iter().map(|x| (*x - max_v).exp() / exp_sum_v).collect();

                let pred = probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u8).unwrap_or(0);
                if pred == ex.target_value {
                    math_correct += 1;
                    if ex.has_letters {
                        math_letter_correct += 1;
                    } else {
                        math_digit_correct += 1;
                    }
                }

                let target_prob = probs[ex.target_value as usize];
                let loss = ScalarF4E4::ZERO - target_prob.ln();

                (probs, loss, true)
            } else {
                text_count_epoch += 1;

                let logits = matmul(&hidden, &w_text).unwrap();
                let logits_data = logits.as_scalars().unwrap();

                let max_v = logits_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(ScalarF4E4::ZERO);
                let exp_sum_v = logits_data.iter().map(|x| (*x - max_v).exp()).fold(ScalarF4E4::ZERO, |a, b| a + b);
                let probs: Vec<ScalarF4E4> = logits_data.iter().map(|x| (*x - max_v).exp() / exp_sum_v).collect();

                let pred = probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u8).unwrap_or(0);
                if pred == ex.target_value {
                    text_correct += 1;
                }

                let target_prob = probs[ex.target_value as usize];
                let loss = ScalarF4E4::ZERO - target_prob.ln();

                (probs, loss, false)
            };

            total_loss = total_loss + routing_loss + value_loss;

            // Backward pass - Routing gradients
            let mut grad_routing_logits = routing_probs.clone();
            grad_routing_logits[routing_target] = grad_routing_logits[routing_target] - ScalarF4E4::ONE;
            let grad_routing_tensor = Tensor::from_scalars(grad_routing_logits, Shape::matrix(1, 2)).unwrap();

            let (grad_hidden_routing, grad_w_routing) = matmul_backward(&grad_routing_tensor, &hidden, &w_routing).unwrap();

            // Value gradients
            let (grad_hidden_value, grad_w_value) = if value_head_used {
                let mut grad_value_logits = value_probs.clone();
                grad_value_logits[ex.target_value as usize] = grad_value_logits[ex.target_value as usize] - ScalarF4E4::ONE;
                let grad_value_tensor = Tensor::from_scalars(grad_value_logits, Shape::matrix(1, MATH_CLASSES)).unwrap();

                let (gh, gw) = matmul_backward(&grad_value_tensor, &hidden, &w_math).unwrap();
                (gh, gw)
            } else {
                let mut grad_value_logits = value_probs.clone();
                grad_value_logits[ex.target_value as usize] = grad_value_logits[ex.target_value as usize] - ScalarF4E4::ONE;
                let grad_value_tensor = Tensor::from_scalars(grad_value_logits, Shape::matrix(1, TEXT_CLASSES)).unwrap();

                let (gh, gw) = matmul_backward(&grad_value_tensor, &hidden, &w_text).unwrap();
                (gh, gw)
            };

            // Accumulate gradients
            if w_routing.grad().is_none() {
                w_routing.set_grad(grad_w_routing);
            } else {
                w_routing.accumulate_grad(grad_w_routing).unwrap();
            }

            if value_head_used {
                if w_math.grad().is_none() {
                    w_math.set_grad(grad_w_value);
                } else {
                    w_math.accumulate_grad(grad_w_value).unwrap();
                }
            } else {
                if w_text.grad().is_none() {
                    w_text.set_grad(grad_w_value);
                } else {
                    w_text.accumulate_grad(grad_w_value).unwrap();
                }
            }

            // Combine hidden gradients
            let grad_hidden_routing_data = grad_hidden_routing.as_scalars().unwrap();
            let grad_hidden_value_data = grad_hidden_value.as_scalars().unwrap();
            let grad_hidden_combined: Vec<ScalarF4E4> = grad_hidden_routing_data.iter()
                .zip(grad_hidden_value_data.iter())
                .map(|(a, b)| *a + *b)
                .collect();

            // Backprop through tanh
            let grad_hidden_pre: Vec<ScalarF4E4> = hidden_data.iter()
                .zip(grad_hidden_combined.iter())
                .map(|(h, g)| {
                    let tanh_deriv = ScalarF4E4::ONE - (*h * *h);
                    *g * tanh_deriv
                })
                .collect();
            let grad_hidden_pre_tensor = Tensor::from_scalars(grad_hidden_pre, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

            // Backprop through w1
            let (_grad_input, grad_w1) = matmul_backward(&grad_hidden_pre_tensor, &input, &w1).unwrap();

            if w1.grad().is_none() {
                w1.set_grad(grad_w1);
            } else {
                w1.accumulate_grad(grad_w1).unwrap();
            }
        }

        // Update all weights
        let mut params = vec![&mut w1, &mut w_routing, &mut w_math, &mut w_text];
        optimizer.step(&mut params).unwrap();

        let avg_loss = total_loss / ScalarF4E4::from(dataset.len() as u32);
        let routing_acc = (routing_correct * 100) / dataset.len();
        let math_acc = if math_count_epoch > 0 { (math_correct * 100) / math_count_epoch } else { 0 };
        let math_digit_acc = if math_digit_count > 0 { (math_digit_correct * 100) / math_digit_count } else { 0 };
        let math_letter_acc = if math_letter_count > 0 { (math_letter_correct * 100) / math_letter_count } else { 0 };
        let text_acc = if text_count_epoch > 0 { (text_correct * 100) / text_count_epoch } else { 0 };

        if epoch % 10 == 0 || epoch < 5 || epoch == EPOCHS - 1 {
            println!("Epoch {:3}: Loss = {:.4}", epoch, avg_loss);
            println!("  Routing: {}% ({}/{})", routing_acc, routing_correct, dataset.len());
            println!("  Math (overall): {}% ({}/{})", math_acc, math_correct, math_count_epoch);
            println!("    - Digits (0-9): {}% ({}/{})", math_digit_acc, math_digit_correct, math_digit_count);
            println!("    - Letters (A/B): {}% ({}/{})", math_letter_acc, math_letter_correct, math_letter_count);
            println!("  Text: {}% ({}/{})", text_acc, text_correct, text_count_epoch);

            // ASCII gap analysis
            if math_digit_acc > 0 && math_letter_acc > 0 {
                let gap_penalty = if math_digit_acc > math_letter_acc {
                    math_digit_acc - math_letter_acc
                } else {
                    math_letter_acc - math_digit_acc
                };
                println!("  ASCII gap penalty: {}% (difference between digits and letters)", gap_penalty);
            }
        }

        if routing_acc >= 95 && math_acc >= 40 {
            println!("\n✓ Target accuracy reached!");
            println!("  Routing: {}% (target: >95%)", routing_acc);
            println!("  Math:    {}% (target: >40%)", math_acc);
            println!("  Digits:  {}%", math_digit_acc);
            println!("  Letters: {}%", math_letter_acc);

            if math_digit_acc > 0 && math_letter_acc > 0 {
                let gap = if math_digit_acc > math_letter_acc {
                    math_digit_acc - math_letter_acc
                } else {
                    math_letter_acc - math_digit_acc
                };

                if gap <= 10 {
                    println!("\n✓ ASCII gap generalization: PASSED");
                    println!("  Network learned positional structure!");
                    println!("  Digit/letter accuracy gap: {}% (<= 10%)", gap);
                } else {
                    println!("\n✗ ASCII gap generalization: FAILED");
                    println!("  Network memorized patterns, didn't learn structure");
                    println!("  Digit/letter accuracy gap: {}% (> 10%)", gap);
                }
            }
            break;
        }
    }

    println!("\n=== Training Complete ===");
    println!("Test: Does ASCII gap (0x39→0x41) break positional learning?");
    println!("Result: Compare digit-only vs letter accuracy for answer");
}
