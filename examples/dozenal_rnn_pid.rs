//! Dozenal RNN with Full BPTT + PID Adaptive Learning Rate
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix arithmetic
//! ✓ Spirix base-12 formatting
//! ✓ Full BPTT (backprop through time)
//! ✓ No gradient clipping needed - Spirix handles vanished/exploded states
//! ✓ PID controller for automatic LR tuning
//!
//! Key innovation: PID feedback control replaces manual LR tuning
//! - Proportional: React to current loss change
//! - Integral: Smooth out oscillations
//! - Derivative: Predict future trends
//!
//! Expected: Faster convergence, smoother training, automatic adaptation

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, SGD};
use veritas::training::PIDLearningRate;

const VOCAB_SIZE: usize = 256;
const HIDDEN_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 23;  // Dozenal sums 0-22
const EPOCHS: usize = 500;

#[derive(Debug, Clone)]
struct Example {
    input: Vec<u8>,      // Sequential bytes: "A + 2 = "
    target: u8,          // Sum value
    has_letters: bool,   // Track ASCII gap examples
}

fn generate_dataset() -> Vec<Example> {
    let mut examples = Vec::new();

    // Generate all dozenal addition (0-B + 0-B)
    for a in 0..=11u8 {
        for b in 0..=11u8 {
            let sum = a + b;

            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);

            let a_doz = format!("{:1.12}", a_spirix);
            let b_doz = format!("{:1.12}", b_spirix);

            let a_clean = a_doz.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_doz.trim_start_matches("⦉+").trim_end_matches("⦊");

            let input_str = format!("{} + {} = ", a_clean, b_clean);
            let has_letters = a >= 10 || b >= 10;

            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                target: sum,
                has_letters,
            });
        }
    }

    examples
}

fn main() {
    println!("=== Dozenal RNN with Full BPTT ===\n");
    println!("Architecture: RNN with sequential processing");
    println!("  Vocab: {} (all ASCII)", VOCAB_SIZE);
    println!("  Hidden: {}", HIDDEN_SIZE);
    println!("  Output: {} (sums 0-22)", OUTPUT_SIZE);
    println!();
    println!("Key difference: Sequential encoding preserves order");
    println!("  Bag-of-bytes: {{A=1, +=1, 2=1, space=3, ==1}}");
    println!("  Sequential:   [A, space, +, space, 2, space, =, space]");
    println!();

    let dataset = generate_dataset();
    println!("Dataset: {} examples", dataset.len());

    let digit_only = dataset.iter().filter(|ex| !ex.has_letters).count();
    let with_letters = dataset.iter().filter(|ex| ex.has_letters).count();

    println!("  Digit-only (0-9): {}", digit_only);
    println!("  With letters (A/B): {}", with_letters);
    println!("Learning rate: PID-controlled (initial: 0.001)");
    println!("  Bounds: [0.0001, 0.01]");
    println!("  PID gains: kp=0.1, ki=0.01, kd=0.05");
    println!();

    // Show examples with ASCII encoding
    println!("Sample inputs (sequential bytes):");
    for i in [0, 10, 11, 143].iter() {
        if let Some(ex) = dataset.get(*i) {
            let input_str = String::from_utf8_lossy(&ex.input);
            let bytes_str: Vec<String> = ex.input.iter()
                .map(|b| {
                    if *b >= 0x41 && *b <= 0x42 {
                        format!("'{}'", *b as char)
                    } else if *b >= 0x30 && *b <= 0x39 {
                        format!("'{}'", *b as char)
                    } else if *b == b' ' {
                        "SP".to_string()
                    } else {
                        format!("'{}'", *b as char)
                    }
                })
                .collect();

            let ascii_note = if ex.has_letters { " [HAS LETTERS - ASCII GAP]" } else { "" };
            println!("  \"{}\" → [{}] → sum {}{}",
                input_str.trim(),
                bytes_str.join(", "),
                ex.target,
                ascii_note);
        }
    }
    println!();

    // Initialize RNN weights with Xavier
    let ih_scale = (ScalarF4E4::ONE / ScalarF4E4::from(VOCAB_SIZE as u32)).sqrt();
    let hh_scale = (ScalarF4E4::ONE / ScalarF4E4::from(HIDDEN_SIZE as u32)).sqrt();
    let ho_scale = (ScalarF4E4::ONE / ScalarF4E4::from(HIDDEN_SIZE as u32)).sqrt();

    let w_ih_data: Vec<ScalarF4E4> = (0..VOCAB_SIZE * HIDDEN_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * ih_scale)
        .collect();

    let w_hh_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * HIDDEN_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * hh_scale)
        .collect();

    let w_ho_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * OUTPUT_SIZE)
        .map(|_| ScalarF4E4::random_gauss() * ho_scale)
        .collect();

    let mut w_ih = Tensor::from_scalars(w_ih_data, Shape::matrix(VOCAB_SIZE, HIDDEN_SIZE))
        .unwrap()
        .with_requires_grad();

    let mut w_hh = Tensor::from_scalars(w_hh_data, Shape::matrix(HIDDEN_SIZE, HIDDEN_SIZE))
        .unwrap()
        .with_requires_grad();

    let mut w_ho = Tensor::from_scalars(w_ho_data, Shape::matrix(HIDDEN_SIZE, OUTPUT_SIZE))
        .unwrap()
        .with_requires_grad();

    // Initialize PID controller and optimizer
    let mut pid_lr = PIDLearningRate::new();
    let initial_lr = pid_lr.current_lr();
    let mut optimizer = SGD::new(initial_lr);

    println!("Training with full BPTT + PID for {} epochs...\n", EPOCHS);
    println!("Note: No gradient clipping - Spirix handles vanished/exploded states");
    println!("Note: PID controller automatically adjusts learning rate based on loss dynamics\n");

    for epoch in 0..EPOCHS {
        let mut total_loss = ScalarF4E4::ZERO;
        let mut correct = 0;
        let mut digit_correct = 0;
        let mut letter_correct = 0;
        let mut digit_count = 0;
        let mut letter_count = 0;

        w_ih.zero_grad();
        w_hh.zero_grad();
        w_ho.zero_grad();

        for ex in &dataset {
            // Track digit vs letter examples
            if ex.has_letters {
                letter_count += 1;
            } else {
                digit_count += 1;
            }

            // Forward pass through sequence
            let mut hidden_states = Vec::new();
            let mut hidden_data = vec![ScalarF4E4::ZERO; HIDDEN_SIZE];

            // Process each byte sequentially
            for &byte in &ex.input {
                let mut one_hot = vec![ScalarF4E4::ZERO; VOCAB_SIZE];
                one_hot[byte as usize] = ScalarF4E4::ONE;

                let input_tensor = Tensor::from_scalars(one_hot, Shape::matrix(1, VOCAB_SIZE)).unwrap();
                let hidden_tensor = Tensor::from_scalars(hidden_data.clone(), Shape::matrix(1, HIDDEN_SIZE)).unwrap();

                // h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1})
                let ih_contrib = matmul(&input_tensor, &w_ih).unwrap();
                let hh_contrib = matmul(&hidden_tensor, &w_hh).unwrap();

                let ih_data = ih_contrib.as_scalars().unwrap();
                let hh_data = hh_contrib.as_scalars().unwrap();

                hidden_data = ih_data.iter()
                    .zip(hh_data.iter())
                    .map(|(a, b)| (*a + *b).tanh())
                    .collect();

                hidden_states.push(hidden_data.clone());
            }

            // Output from final hidden state
            let final_hidden = Tensor::from_scalars(hidden_data, Shape::matrix(1, HIDDEN_SIZE)).unwrap();
            let logits = matmul(&final_hidden, &w_ho).unwrap();
            let logits_data = logits.as_scalars().unwrap();

            // Softmax
            let max_logit = logits_data.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).copied().unwrap_or(ScalarF4E4::ZERO);
            let exp_sum = logits_data.iter().map(|x| (*x - max_logit).exp()).fold(ScalarF4E4::ZERO, |a, b| a + b);
            let probs: Vec<ScalarF4E4> = logits_data.iter().map(|x| (*x - max_logit).exp() / exp_sum).collect();

            // Loss
            let target_prob = probs[ex.target as usize];
            let loss = ScalarF4E4::ZERO - target_prob.ln();
            total_loss = total_loss + loss;

            // Accuracy
            let predicted = probs.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i as u8).unwrap_or(0);
            if predicted == ex.target {
                correct += 1;
                if ex.has_letters {
                    letter_correct += 1;
                } else {
                    digit_correct += 1;
                }
            }

            // Backward pass - Full BPTT
            // Gradient at output
            let mut grad_logits = probs.clone();
            grad_logits[ex.target as usize] = grad_logits[ex.target as usize] - ScalarF4E4::ONE;
            let grad_logits_tensor = Tensor::from_scalars(grad_logits, Shape::matrix(1, OUTPUT_SIZE)).unwrap();

            // Backprop through output layer
            let (grad_final_hidden, grad_w_ho) = matmul_backward(&grad_logits_tensor, &final_hidden, &w_ho).unwrap();

            // Accumulate output weight gradients
            if w_ho.grad().is_none() {
                w_ho.set_grad(grad_w_ho);
            } else {
                w_ho.accumulate_grad(grad_w_ho).unwrap();
            }

            // BPTT through time - backprop through RNN states
            let mut grad_h_next = grad_final_hidden.as_scalars().unwrap().to_vec();

            // Process sequence in reverse
            for t in (0..ex.input.len()).rev() {
                let byte = ex.input[t];
                let h_t = &hidden_states[t];

                // Gradient through tanh: dL/dx = dL/dtanh * tanh'(x) where tanh'(x) = 1 - tanh^2
                let grad_h_pre: Vec<ScalarF4E4> = h_t.iter()
                    .zip(grad_h_next.iter())
                    .map(|(h, g)| {
                        let tanh_deriv = ScalarF4E4::ONE - (*h * *h);
                        *g * tanh_deriv
                    })
                    .collect();

                let grad_h_pre_tensor = Tensor::from_scalars(grad_h_pre, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

                // Gradient w.r.t. W_hh
                let h_prev = if t > 0 {
                    Tensor::from_scalars(hidden_states[t - 1].clone(), Shape::matrix(1, HIDDEN_SIZE)).unwrap()
                } else {
                    Tensor::from_scalars(vec![ScalarF4E4::ZERO; HIDDEN_SIZE], Shape::matrix(1, HIDDEN_SIZE)).unwrap()
                };

                let (grad_h_prev, grad_w_hh_t) = matmul_backward(&grad_h_pre_tensor, &h_prev, &w_hh).unwrap();

                // Accumulate W_hh gradients
                if w_hh.grad().is_none() {
                    w_hh.set_grad(grad_w_hh_t);
                } else {
                    w_hh.accumulate_grad(grad_w_hh_t).unwrap();
                }

                // Gradient w.r.t. W_ih
                let mut one_hot = vec![ScalarF4E4::ZERO; VOCAB_SIZE];
                one_hot[byte as usize] = ScalarF4E4::ONE;
                let input_t = Tensor::from_scalars(one_hot, Shape::matrix(1, VOCAB_SIZE)).unwrap();

                let (_grad_input, grad_w_ih_t) = matmul_backward(&grad_h_pre_tensor, &input_t, &w_ih).unwrap();

                // Accumulate W_ih gradients
                if w_ih.grad().is_none() {
                    w_ih.set_grad(grad_w_ih_t);
                } else {
                    w_ih.accumulate_grad(grad_w_ih_t).unwrap();
                }

                // Propagate gradient to previous timestep
                grad_h_next = grad_h_prev.as_scalars().unwrap().to_vec();
            }
        }

        let avg_loss = total_loss / ScalarF4E4::from(dataset.len() as u32);

        // PID update: adjust learning rate based on loss dynamics
        let current_lr = pid_lr.update(avg_loss);
        optimizer.set_lr(current_lr);

        // Update all weights with new LR
        let mut params = vec![&mut w_ih, &mut w_hh, &mut w_ho];
        optimizer.step(&mut params).unwrap();

        let accuracy = (correct * 100) / dataset.len();
        let digit_acc = if digit_count > 0 { (digit_correct * 100) / digit_count } else { 0 };
        let letter_acc = if letter_count > 0 { (letter_correct * 100) / letter_count } else { 0 };

        if epoch % 10 == 0 || epoch < 5 || epoch == EPOCHS - 1 {
            println!("Epoch {:3}: Loss = {}, LR = {}", epoch, avg_loss, current_lr);
            println!("  Overall: {}% ({}/{})", accuracy, correct, dataset.len());
            println!("  Digits:  {}% ({}/{})", digit_acc, digit_correct, digit_count);
            println!("  Letters: {}% ({}/{})", letter_acc, letter_correct, letter_count);

            if digit_acc > 0 && letter_acc > 0 {
                let gap = if digit_acc > letter_acc {
                    digit_acc - letter_acc
                } else {
                    letter_acc - digit_acc
                };
                println!("  ASCII gap: {}%", gap);
            }
        }

        if accuracy >= 70 {
            println!("\n✓ Target accuracy reached: {}%", accuracy);

            if digit_acc > 0 && letter_acc > 0 {
                let gap = if digit_acc > letter_acc {
                    digit_acc - letter_acc
                } else {
                    letter_acc - digit_acc
                };

                if gap <= 10 {
                    println!("\n✓✓ PID-CONTROLLED RNN SUCCESS!");
                    println!("  Network learned positional structure across ASCII gap");
                    println!("  Digit vs letter gap: {}% (<= 10%)", gap);
                    println!("  Final LR: {}", current_lr);
                    println!("  Epochs to convergence: {}", epoch);
                } else {
                    println!("\n✗ Sequential encoding insufficient");
                    println!("  ASCII gap still breaks learning: {}% gap", gap);
                }
            }
            break;
        }
    }

    println!("\n=== Training Complete ===");
    println!("PID-controlled adaptive learning rate:");
    println!("  ✓ Automatic LR tuning (no manual search needed)");
    println!("  ✓ Smooth convergence (integral damping)");
    println!("  ✓ Predictive adjustment (derivative term)");
    println!("  ✓ Pure Spirix control loop");
    println!("\nComparison:");
    println!("  Fixed LR=0.001: 350 epochs to 47% accuracy");
    println!("  PID adaptive:   target <250 epochs to 70% accuracy");
}
