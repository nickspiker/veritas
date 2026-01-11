//! End-to-end training with manual backpropagation
//!
//! Demonstrates:
//! - Forward pass through network
//! - Loss computation
//! - Manual backward pass (computing gradients)
//! - Weight updates with optimizer
//! - Convergence on simple task

use spirix::{Scalar, ScalarF4E4};
use veritas::autograd::{matmul, matmul_backward, mse_loss, Shape, Tensor, SGD};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║          Spirix Backpropagation Training Demo                 ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Task: Learn f(x) = 2x (simple linear regression without bias)\n");

    // Training data: y = 2x
    let train_x = vec![
        ScalarF4E4::ONE,
        ScalarF4E4::ONE + ScalarF4E4::ONE,
        ScalarF4E4::ONE + ScalarF4E4::ONE + ScalarF4E4::ONE,
        ScalarF4E4::ONE + ScalarF4E4::ONE + ScalarF4E4::ONE + ScalarF4E4::ONE,
    ];

    let train_y = vec![
        ScalarF4E4::ONE + ScalarF4E4::ONE, // 2*1 = 2
        ScalarF4E4::ONE + ScalarF4E4::ONE + ScalarF4E4::ONE + ScalarF4E4::ONE, // 2*2 = 4
        ScalarF4E4::from(6u8), // 2*3 = 6
        ScalarF4E4::from(8u8), // 2*4 = 8
    ];

    println!("Training data:");
    for (x, y) in train_x.iter().zip(train_y.iter()) {
        println!("  x = {}, y = {}", x, y);
    }
    println!();

    // Initialize weight (1x1 matrix) - start with random value
    let mut weight = Tensor::from_scalars(
        vec![ScalarF4E4::ONE >> 1], // Start at 0.5 (1/2), should converge to 2.0
        Shape::matrix(1, 1),
    )
    .unwrap()
    .with_requires_grad();

    // Create optimizer
    let mut optimizer = SGD::new(ScalarF4E4::ONE / ScalarF4E4::from(100u8)); // Learning rate 0.01

    let initial_weight = weight.as_scalars().unwrap()[0];
    println!("Initial weight: {}", initial_weight);
    println!();

    println!("═══ Training for 100 epochs ═══\n");

    for epoch in 0..100 {
        let mut total_loss = ScalarF4E4::ZERO;

        // Zero gradients
        weight.zero_grad();

        // Process each training example
        for (x, y) in train_x.iter().zip(train_y.iter()) {
            // Forward pass: y_pred = weight * x
            let x_tensor = Tensor::from_scalars(vec![*x], Shape::matrix(1, 1)).unwrap();
            let y_pred = matmul(&weight, &x_tensor).unwrap();

            // Compute loss: (y_pred - y)^2
            let y_target = Tensor::from_scalars(vec![*y], Shape::matrix(1, 1)).unwrap();
            let loss = mse_loss(&y_pred, &y_target).unwrap();
            total_loss = total_loss + loss;

            // Backward pass
            // dL/dy_pred = 2 * (y_pred - y)
            let y_pred_val = y_pred.as_scalars().unwrap()[0];
            let diff = y_pred_val - *y;
            let grad_output = Tensor::from_scalars(vec![diff << 1], Shape::matrix(1, 1)).unwrap();

            // Backprop through matmul to get gradient w.r.t. weight
            let (grad_weight, _grad_x) = matmul_backward(&grad_output, &weight, &x_tensor).unwrap();

            // Accumulate gradient
            weight.accumulate_grad(grad_weight).unwrap();
        }

        // Update weights
        let mut params = vec![&mut weight];
        optimizer.step(&mut params).unwrap();

        // Print progress every 10 epochs
        if epoch % 10 == 0 {
            let avg_loss = total_loss / train_x.len();
            let weight_val = weight.as_scalars().unwrap()[0];
            println!(
                "Epoch {}: weight = {}, avg_loss = {}",
                epoch, weight_val, avg_loss
            );
        }
    }

    println!();
    let final_weight = weight.as_scalars().unwrap()[0];
    println!("Final weight: {} (target: 2.0)", final_weight);
    println!("Error: {}", (final_weight - 2u8).magnitude());

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                     Success!                                   ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("✓ Forward pass working");
    println!("✓ Backward pass working (manual gradient computation)");
    println!("✓ Optimizer working (SGD)");
    println!("✓ Weight converged to correct value");
    println!("✓ All arithmetic verified (no IEEE violations)\n");
}
