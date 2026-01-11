//! Binary Addition with Feedforward Network + Backpropagation
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ No Python runtime
//! ✓ Pure Spirix arithmetic
//! ✓ Uses existing autograd infrastructure
//! ✓ Full gradient flow (no BPTT complexity)

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, SGD};

const INPUT_SIZE: usize = 256;  // One-hot vocabulary size
const HIDDEN_SIZE: usize = 64;
const OUTPUT_SIZE: usize = 15;  // Sum 0-14
const LEARNING_RATE_NUM: u8 = 1;
const LEARNING_RATE_DEN: u8 = 100; // LR = 0.01
const EPOCHS: usize = 500;

#[derive(Debug, Clone)]
struct Example {
    input: Vec<u8>,  // UTF-8 bytes of "101 + 011 = "
    target: u8,      // Sum value 0-14
}

fn generate_dataset() -> Vec<Example> {
    let mut examples = Vec::new();
    // Generate all possible 3-bit + 3-bit combinations using pure Spirix
    for a in 0..=7u8 {
        for b in 0..=7u8 {
            let sum = a + b;
            // Format using Spirix (base 2, 3 digits)
            let a_spirix = ScalarF4E4::from(a);
            let b_spirix = ScalarF4E4::from(b);
            let a_bin = format!("{:3.2}", a_spirix);  // Base 2, 3 digits
            let b_bin = format!("{:3.2}", b_spirix);

            // Strip the ⦉+⦊ wrapper to get just the digits
            let a_clean = a_bin.trim_start_matches("⦉+").trim_end_matches("⦊");
            let b_clean = b_bin.trim_start_matches("⦉+").trim_end_matches("⦊");

            let input_str = format!("{} + {} = ", a_clean, b_clean);
            examples.push(Example {
                input: input_str.as_bytes().to_vec(),
                target: sum,
            });
        }
    }
    examples
}

/// Encode input bytes as bag-of-bytes (sum of one-hots)
fn encode_input(bytes: &[u8]) -> Vec<ScalarF4E4> {
    let mut encoding = vec![ScalarF4E4::ZERO; INPUT_SIZE];

    // Bag-of-bytes: add up one-hot for each byte
    for &byte in bytes {
        encoding[byte as usize] = encoding[byte as usize] + ScalarF4E4::ONE;
    }

    encoding
}

fn main() {
    println!("=== Binary Addition: Feedforward Network ===\n");
    println!("Task: Learn 3-bit + 3-bit addition");
    println!("Architecture: Input(256) → Hidden(64) → Output(15)");

    let dataset = generate_dataset();

    println!("Dataset: {} examples", dataset.len());
    let lr_spirix = ScalarF4E4::from(LEARNING_RATE_NUM) / ScalarF4E4::from(LEARNING_RATE_DEN);
    println!("Learning rate: {}/{} = {}", LEARNING_RATE_NUM, LEARNING_RATE_DEN, lr_spirix);
    println!();

    // Check class distribution
    let mut class_counts = vec![0usize; 15];
    for ex in &dataset {
        class_counts[ex.target as usize] += 1;
    }
    println!("Class distribution:");
    for (sum, count) in class_counts.iter().enumerate() {
        if *count > 0 {
            let pct = (ScalarF4E4::from(*count as u32) / ScalarF4E4::from(dataset.len() as u32))
                * ScalarF4E4::from(100u32);
            println!("  Sum {:2}: {:4} samples ({}%)", sum, count, pct);
        }
    }
    println!();

    // Initialize weights with Xavier initialization
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

    // Optimizer
    let lr = ScalarF4E4::from(LEARNING_RATE_NUM) / ScalarF4E4::from(LEARNING_RATE_DEN);
    let mut optimizer = SGD::new(lr);

    println!("Training for {} epochs...\n", EPOCHS);

    for epoch in 0..EPOCHS {
        let mut total_loss = ScalarF4E4::ZERO;
        let mut correct = 0;

        // Zero gradients at epoch level
        w1.zero_grad();
        w2.zero_grad();

        for ex in &dataset {
            // Encode input as bag-of-bytes
            let input_vec = encode_input(&ex.input);
            let input = Tensor::from_scalars(input_vec, Shape::matrix(1, INPUT_SIZE)).unwrap();

            // Forward pass: input → hidden → output
            let hidden_pre = matmul(&input, &w1).unwrap();

            // Tanh activation (pure Spirix)
            let hidden_data: Vec<ScalarF4E4> = hidden_pre.as_scalars()
                .unwrap()
                .iter()
                .map(|x| x.tanh())
                .collect();
            let hidden = Tensor::from_scalars(hidden_data, Shape::matrix(1, HIDDEN_SIZE)).unwrap();

            // Output layer
            let logits = matmul(&hidden, &w2).unwrap();
            let logits_data = logits.as_scalars().unwrap().to_vec();

            // Softmax (pure Spirix)
            let max_logit = logits_data.iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .copied()
                .unwrap_or(ScalarF4E4::ZERO);

            let mut exp_sum = ScalarF4E4::ZERO;
            for &x in &logits_data {
                exp_sum = exp_sum + (x - max_logit).exp();
            }

            let probs: Vec<ScalarF4E4> = logits_data.iter()
                .map(|&x| (x - max_logit).exp() / exp_sum)
                .collect();

            // Cross-entropy loss: -log(p[target])
            let target_prob = probs[ex.target as usize];
            let loss = ScalarF4E4::ZERO - target_prob.ln();
            total_loss = total_loss + loss;

            // Accuracy
            let predicted = probs.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u8)
                .unwrap_or(0);
            if predicted == ex.target {
                correct += 1;
            }

            // Backward pass
            // Gradient of cross-entropy w.r.t. logits: dL/dlogit[i] = p[i] - (i == target ? 1 : 0)
            let mut grad_logits = probs.clone();
            grad_logits[ex.target as usize] = grad_logits[ex.target as usize] - ScalarF4E4::ONE;
            let grad_logits_tensor = Tensor::from_scalars(grad_logits, Shape::matrix(1, OUTPUT_SIZE)).unwrap();

            // Backprop through w2: output layer
            let (grad_hidden, grad_w2) = matmul_backward(&grad_logits_tensor, &hidden, &w2).unwrap();

            // Backprop through tanh: dL/dx = dL/dtanh(x) * tanh'(x) where tanh'(x) = 1 - tanh(x)^2
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

            // Backprop through w1: input → hidden layer
            let (_grad_input, grad_w1) = matmul_backward(&grad_hidden_pre_tensor, &input, &w1).unwrap();

            // Accumulate gradients for both layers
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

        // Update weights (both layers!)
        let mut params = vec![&mut w1, &mut w2];
        optimizer.step(&mut params).unwrap();

        let avg_loss = total_loss / ScalarF4E4::from(dataset.len() as u32);
        let accuracy = (correct * 100) / dataset.len();

        if epoch % 50 == 0 || epoch < 5 || epoch == EPOCHS - 1 {
            println!("Epoch {:3}: Loss = {:.4}, Accuracy = {}% ({}/{})",
                epoch, avg_loss, accuracy, correct, dataset.len());
        }

        if accuracy >= 90 {
            println!("\n✓ Target accuracy reached: {}%", accuracy);
            break;
        }
    }

    println!("\n=== Training Complete ===");
    println!("✓ Pure Spirix backpropagation");
    println!("✓ Full gradient flow through both layers");
    println!("✓ No IEEE-754 contamination");
    println!("✓ No BPTT complexity needed for this task");
}
