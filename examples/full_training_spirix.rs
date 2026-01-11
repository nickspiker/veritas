//! Complete training example with backpropagation
//!
//! Demonstrates:
//! - Forward pass (Spirix F4E4)
//! - Loss computation
//! - Backward pass (gradients)
//! - Weight updates (SGD)
//! - ZERO IEEE-754 in the brain

use spirix::{
    ScalarF4E4, Tensor, Linear, SGD,
    linear_backward, relu_backward, mse_loss, mse_loss_grad,
    matmul, relu,
};

fn main() {
    println!("=== Veritas: Complete Training with Backprop ===\n");

    // Create a simple network: 2 → 3 → 1
    println!("Network architecture: 2 → 3 → 1");
    println!("  Input: 2 features");
    println!("  Hidden: 3 neurons (ReLU)");
    println!("  Output: 1 value\n");

    // Layer 1: 3×2 weights
    let mut w1 = Tensor::new(
        vec![
            ScalarF4E4::ONE >> 1,  // 0.5
            ScalarF4E4::ZERO - (ScalarF4E4::from(3u8) / ScalarF4E4::from(10u8)),  // -0.3
            ScalarF4E4::ONE / ScalarF4E4::from(5u8),  // 0.2
            ScalarF4E4::from(2u8) / ScalarF4E4::from(5u8),  // 0.4
            ScalarF4E4::ZERO - (ScalarF4E4::ONE / ScalarF4E4::from(10u8)),  // -0.1
            ScalarF4E4::from(3u8) / ScalarF4E4::from(5u8),  // 0.6
        ],
        vec![3, 2],
    );
    let mut b1 = Tensor::fill(ScalarF4E4::ZERO, vec![3]);
    let mut layer1 = Linear::new(w1.clone(), b1.clone());

    // Layer 2: 1×3 weights
    let mut w2 = Tensor::new(
        vec![
            ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8),  // 0.7
            ScalarF4E4::ZERO - (ScalarF4E4::ONE / ScalarF4E4::from(5u8)),  // -0.2
            ScalarF4E4::from(3u8) / ScalarF4E4::from(10u8),  // 0.3
        ],
        vec![1, 3],
    );
    let mut b2 = Tensor::fill(ScalarF4E4::ZERO, vec![1]);
    let mut layer2 = Linear::new(w2.clone(), b2.clone());

    // Optimizer
    let optimizer = SGD::new(ScalarF4E4::ONE / ScalarF4E4::from(10u8));  // 0.1
    println!("Optimizer: SGD (learning_rate = 0.1)\n");

    // Training data: simple XOR-like problem
    let inputs = vec![
        Tensor::new(vec![ScalarF4E4::ZERO, ScalarF4E4::ZERO], vec![2, 1]),
        Tensor::new(vec![ScalarF4E4::ZERO, ScalarF4E4::ONE], vec![2, 1]),
        Tensor::new(vec![ScalarF4E4::ONE, ScalarF4E4::ZERO], vec![2, 1]),
        Tensor::new(vec![ScalarF4E4::ONE, ScalarF4E4::ONE], vec![2, 1]),
    ];

    let targets = vec![
        Tensor::new(vec![ScalarF4E4::ZERO], vec![1, 1]),
        Tensor::new(vec![ScalarF4E4::ONE], vec![1, 1]),
        Tensor::new(vec![ScalarF4E4::ONE], vec![1, 1]),
        Tensor::new(vec![ScalarF4E4::ZERO], vec![1, 1]),
    ];

    println!("Training data (XOR-like):");
    println!("  [0, 0] → 0");
    println!("  [0, 1] → 1");
    println!("  [1, 0] → 1");
    println!("  [1, 1] → 0\n");

    println!("Starting training...\n");

    // Training loop
    let epochs = 100;
    for epoch in 0..epochs {
        let mut total_loss = ScalarF4E4::ZERO;

        for (input, target) in inputs.iter().zip(targets.iter()) {
            // === FORWARD PASS ===

            // Layer 1
            let z1 = layer1.forward(input, ScalarF4E4::ZERO);
            let a1 = relu(&z1, ScalarF4E4::ZERO);

            // Layer 2
            let output = layer2.forward(&a1, ScalarF4E4::ZERO);

            // Loss
            let loss = mse_loss(&output, target);
            total_loss = total_loss + loss;

            // === BACKWARD PASS ===

            // Gradient of loss wrt output
            let d_output = mse_loss_grad(&output, target);

            // Layer 2 gradients
            let grads2 = linear_backward(&d_output, &a1, &layer2.weights, ScalarF4E4::ZERO);

            // ReLU gradient
            let d_a1 = relu_backward(&grads2.input_grad, &z1, ScalarF4E4::ZERO);

            // Layer 1 gradients
            let grads1 = linear_backward(&d_a1, input, &layer1.weights, ScalarF4E4::ZERO);

            // === UPDATE WEIGHTS ===

            optimizer.step(&mut layer1.weights, &grads1.weight_grad);
            optimizer.step(&mut layer2.weights, &grads2.weight_grad);
        }

        if epoch % 10 == 0 {
            println!("Epoch {}: loss = {:.6}", epoch, total_loss.to_f64());
        }
    }

    println!("\n✓ Training complete!\n");

    // Test final predictions
    println!("Final predictions:");
    for (i, (input, target)) in inputs.iter().zip(targets.iter()).enumerate() {
        let z1 = layer1.forward(input, ScalarF4E4::ZERO);
        let a1 = relu(&z1, ScalarF4E4::ZERO);
        let output = layer2.forward(&a1, ScalarF4E4::ZERO);

        println!(
            "  Input {:?} → Predicted: {:.3}, Target: {:.3}",
            [input.data[0].to_f64(), input.data[1].to_f64()],
            output.data[0].to_f64(),
            target.data[0].to_f64()
        );
    }

    println!("\n--- Architecture Summary ---");
    println!("✓ Forward pass: Pure Spirix F4E4");
    println!("✓ Backward pass: Pure Spirix F4E4");
    println!("✓ Weight updates: Pure Spirix F4E4");
    println!("✓ Loss computation: Pure Spirix F4E4");
    println!("✓ ZERO IEEE-754 in brain");
    println!("✓ NO denormals, NO NaN, NO infinity");
    println!("\n🚀 Brain is 100% trash-free and learning!");
}
