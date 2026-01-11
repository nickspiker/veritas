//! Phase 3: Arithmetic Bolt-on Training
//!
//! Demonstrates:
//! - Neural network learns basic arithmetic
//! - Symbolic arithmetic engine provides ground truth
//! - Training with verified answers (no guessing)
//! - Pure Spirix, no IEEE contamination

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape, matmul, matmul_backward, relu, SGD};
use veritas::symbolic::{ArithOp, ArithProblem, ArithGenerator};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║         Phase 3: Arithmetic Bolt-on Training                  ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Task: Learn basic arithmetic operations");
    println!("Method: Neural network + Symbolic verification bolt-on\n");

    // Network: 3 inputs (left, right, op_encoding) → 4 hidden → 1 output (answer)
    let mut w1 = Tensor::from_scalars(
        vec![
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8), ScalarF4E4::ONE / ScalarF4E4::from(10u8),
        ],
        Shape::matrix(4, 3) // 4 hidden, 3 inputs
    ).unwrap().with_requires_grad();

    let mut w2 = Tensor::from_scalars(
        vec![
            ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            ScalarF4E4::ONE / ScalarF4E4::from(10u8),
        ],
        Shape::matrix(1, 4) // 1 output, 4 hidden
    ).unwrap().with_requires_grad();

    let mut optimizer = SGD::new(ScalarF4E4::ONE / ScalarF4E4::from(20u8)); // 0.05 learning rate

    // Arithmetic problem generator (basecalc bolt-on)
    let generator = ArithGenerator::new(10); // Problems with values 0-9

    println!("Network: [3 → 4 → 1] (with ReLU activation)");
    println!("Training on addition, subtraction, multiplication, division\n");

    println!("═══ Training ═══\n");

    for epoch in 0..200 {
        let mut total_error = ScalarF4E4::ZERO;
        let mut correct = 0;
        let batch_size = 16;

        w1.zero_grad();
        w2.zero_grad();

        // Generate training batch
        let problems = generator.generate_batch(batch_size);

        for problem in &problems {
            // Solve using symbolic engine (verified ground truth)
            let symbolic_result = problem.solve().unwrap();
            let symbolic_answer = symbolic_result.answer;

            // Encode operation as scalar
            let op_encoding = match problem.op {
                ArithOp::Add => ScalarF4E4::ZERO,
                ArithOp::Sub => ScalarF4E4::ONE / ScalarF4E4::from(3u8), // 0.33
                ArithOp::Mul => ScalarF4E4::ONE / ScalarF4E4::from(3u8) + ScalarF4E4::ONE / ScalarF4E4::from(3u8), // 0.67
                ArithOp::Div => ScalarF4E4::ONE,
            };

            // Neural network forward pass
            let input = Tensor::from_scalars(
                vec![problem.left, problem.right, op_encoding],
                Shape::matrix(3, 1)
            ).unwrap();

            let hidden_pre = matmul(&w1, &input).unwrap();
            let hidden = relu(&hidden_pre).unwrap();
            let output = matmul(&w2, &hidden).unwrap();
            let neural_answer = output.as_scalars().unwrap()[0];

            // Compute error (neural vs symbolic)
            let error = neural_answer - symbolic_answer;
            let error_magnitude = error.magnitude();
            total_error = total_error + error_magnitude;

            // Check if "correct" (within tolerance)
            let tolerance = ScalarF4E4::ONE / ScalarF4E4::from(2u8); // 0.5
            if error_magnitude < tolerance {
                correct += 1;
            }

            // Backprop
            let grad_output = Tensor::from_scalars(
                vec![error << 1], // 2 * error
                Shape::matrix(1, 1)
            ).unwrap();

            // Backprop through W2
            let (grad_w2, grad_hidden) = matmul_backward(&grad_output, &w2, &hidden).unwrap();
            w2.accumulate_grad(grad_w2).unwrap();

            // Backprop through ReLU
            let hidden_pre_data = hidden_pre.as_scalars().unwrap();
            let grad_hidden_data = grad_hidden.as_scalars().unwrap();
            let grad_hidden_pre_data: Vec<ScalarF4E4> = hidden_pre_data
                .iter()
                .zip(grad_hidden_data.iter())
                .map(|(h, g)| if *h > ScalarF4E4::ZERO { *g } else { ScalarF4E4::ZERO })
                .collect();
            let grad_hidden_pre = Tensor::from_scalars(grad_hidden_pre_data, hidden_pre.shape().clone()).unwrap();

            // Backprop through W1
            let (grad_w1, _) = matmul_backward(&grad_hidden_pre, &w1, &input).unwrap();
            w1.accumulate_grad(grad_w1).unwrap();
        }

        // Update weights
        let mut params = vec![&mut w1, &mut w2];
        optimizer.step(&mut params).unwrap();

        if epoch % 20 == 0 {
            let avg_error = total_error / ScalarF4E4::from(batch_size as u8);
            let accuracy = ScalarF4E4::from(correct as u8) / ScalarF4E4::from(batch_size as u8);
            println!("Epoch {}: avg_error = {}, accuracy = {}/{} ({})",
                epoch, avg_error, correct, batch_size, accuracy);
        }
    }

    println!("\n═══ Testing ═══\n");

    // Test on specific problems
    let test_cases = vec![
        ArithProblem::new(ScalarF4E4::from(2u8), ScalarF4E4::from(3u8), ArithOp::Add),
        ArithProblem::new(ScalarF4E4::from(5u8), ScalarF4E4::from(2u8), ArithOp::Sub),
        ArithProblem::new(ScalarF4E4::from(4u8), ScalarF4E4::from(3u8), ArithOp::Mul),
        ArithProblem::new(ScalarF4E4::from(8u8), ScalarF4E4::from(2u8), ArithOp::Div),
        ArithProblem::new(ScalarF4E4::from(7u8), ScalarF4E4::from(7u8), ArithOp::Add),
        ArithProblem::new(ScalarF4E4::from(9u8), ScalarF4E4::from(3u8), ArithOp::Mul),
    ];

    for problem in &test_cases {
        let symbolic_result = problem.solve().unwrap();
        let symbolic_answer = symbolic_result.answer;

        let op_encoding = match problem.op {
            ArithOp::Add => ScalarF4E4::ZERO,
            ArithOp::Sub => ScalarF4E4::ONE / ScalarF4E4::from(3u8),
            ArithOp::Mul => ScalarF4E4::ONE / ScalarF4E4::from(3u8) + ScalarF4E4::ONE / ScalarF4E4::from(3u8),
            ArithOp::Div => ScalarF4E4::ONE,
        };

        let input = Tensor::from_scalars(
            vec![problem.left, problem.right, op_encoding],
            Shape::matrix(3, 1)
        ).unwrap();

        let hidden_pre = matmul(&w1, &input).unwrap();
        let hidden = relu(&hidden_pre).unwrap();
        let output = matmul(&w2, &hidden).unwrap();
        let neural_answer = output.as_scalars().unwrap()[0];

        let error = (neural_answer - symbolic_answer).magnitude();
        let status = if error < (ScalarF4E4::ONE / ScalarF4E4::from(2u8)) {
            "✓"
        } else {
            "✗"
        };

        println!("  {} = {} (neural: {}, symbolic: {}, error: {}) {}",
            symbolic_result.expr,
            symbolic_answer,
            neural_answer,
            symbolic_answer,
            error,
            status);
    }

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Phase 3 Complete                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("✓ Arithmetic bolt-on working:");
    println!("  → Symbolic engine generates verified answers");
    println!("  → Neural network learns to approximate");
    println!("  → Training signal = contradiction between neural and symbolic");
    println!("  → Pure Spirix arithmetic (no IEEE)");
    println!("  → No token limits, just math\n");

    println!("Key insight:");
    println!("  Symbolic engines provide EXACT answers.");
    println!("  Neural learns the patterns, symbolic verifies the truth.");
    println!("  Contradictions drive learning.\n");

    println!("Next: Integrate with basecalc for complex expressions");
    println!("  → Parse full expressions (e.g., \"(2 + 3) * 4\")");
    println!("  → Handle arbitrary precision");
    println!("  → Support trigonometry, logarithms, etc.\n");
}
