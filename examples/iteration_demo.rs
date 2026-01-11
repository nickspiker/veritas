//! Iteration Engine Demo
//!
//! Demonstrates the core z² + c iteration:
//! - Convergence (bounded) = truth
//! - Escape (diverged) = farktogle
//! - No token limits, just mathematical stopping conditions

use spirix::ScalarF4E4;
use veritas::autograd::{Tensor, Shape};
use veritas::iteration::{IterationEngine, IterationResult, ConvergenceConfig};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║              Veritas Iteration Engine Demo                    ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Core primitive: z := z² + c");
    println!("Truth converges (stays bounded).");
    println!("Farktogle escapes (diverges to infinity).\n");

    // Example 1: Converging trajectory (truth)
    println!("═══ Example 1: Converging Trajectory ═══\n");
    println!("Starting with z = 0.5, c = 0");
    println!("Expected: Converges to 0 (bounded)\n");

    let mut engine = IterationEngine::new();

    let z = Tensor::from_scalars(
        vec![ScalarF4E4::ONE >> 1], // 0.5
        Shape::vector(1)
    ).unwrap();

    let c = Tensor::from_scalars(
        vec![ScalarF4E4::ZERO],
        Shape::vector(1)
    ).unwrap();

    let (result, final_z) = engine.iterate_with_progress(z, c, |iter, z_state, mag, change| {
        if iter <= 10 || iter % 10 == 0 {
            let val = z_state.as_scalars().unwrap()[0];
            println!("  Iteration {}: z = {}, magnitude = {}, change = {}",
                iter, val, mag, change);
        }
    }).unwrap();

    match result {
        IterationResult::Converged { iterations } => {
            println!("\n✓ CONVERGED after {} iterations", iterations);
            let final_val = final_z.as_scalars().unwrap()[0];
            println!("  Final value: {}", final_val);
            println!("  Interpretation: This trajectory is BOUNDED (truth)\n");
        }
        IterationResult::Escaped { iterations } => {
            println!("\n✗ ESCAPED after {} iterations", iterations);
            println!("  Interpretation: This trajectory DIVERGED (farktogle)\n");
        }
        IterationResult::MaxIterations { iterations } => {
            println!("\n⚠ Max iterations ({}) reached\n", iterations);
        }
    }

    // Example 2: Escaping trajectory (farktogle)
    println!("═══ Example 2: Escaping Trajectory ═══\n");
    println!("Starting with z = 1.0, c = 1.0");
    println!("Expected: Escapes to infinity (diverges)\n");

    let mut engine = IterationEngine::new();

    let z = Tensor::from_scalars(
        vec![ScalarF4E4::ONE],
        Shape::vector(1)
    ).unwrap();

    let c = Tensor::from_scalars(
        vec![ScalarF4E4::ONE],
        Shape::vector(1)
    ).unwrap();

    let (result, final_z) = engine.iterate_with_progress(z, c, |iter, z_state, mag, change| {
        let val = z_state.as_scalars().unwrap()[0];
        println!("  Iteration {}: z = {}, magnitude = {}, change = {}",
            iter, val, mag, change);
    }).unwrap();

    match result {
        IterationResult::Converged { iterations } => {
            println!("\n✓ CONVERGED after {} iterations", iterations);
            println!("  Interpretation: This trajectory is BOUNDED (truth)\n");
        }
        IterationResult::Escaped { iterations } => {
            println!("\n✗ ESCAPED after {} iterations", iterations);
            let final_val = final_z.as_scalars().unwrap()[0];
            println!("  Final value: {}", final_val);
            println!("  Interpretation: This trajectory DIVERGED (farktogle)\n");
        }
        IterationResult::MaxIterations { iterations } => {
            println!("\n⚠ Max iterations ({}) reached\n", iterations);
        }
    }

    // Example 3: Boundary case (Mandelbrot set boundary)
    println!("═══ Example 3: Boundary Case ═══\n");
    println!("Starting with z = 0, c = -0.5");
    println!("Expected: Converges (just inside Mandelbrot set)\n");

    let mut engine = IterationEngine::new();

    let z = Tensor::from_scalars(
        vec![ScalarF4E4::ZERO],
        Shape::vector(1)
    ).unwrap();

    let c = Tensor::from_scalars(
        vec![ScalarF4E4::ZERO - (ScalarF4E4::ONE >> 1)], // -0.5
        Shape::vector(1)
    ).unwrap();

    let (result, final_z) = engine.iterate_with_progress(z, c, |iter, z_state, mag, change| {
        if iter <= 20 || iter % 20 == 0 {
            let val = z_state.as_scalars().unwrap()[0];
            println!("  Iteration {}: z = {}, magnitude = {}, change = {}",
                iter, val, mag, change);
        }
    }).unwrap();

    match result {
        IterationResult::Converged { iterations } => {
            println!("\n✓ CONVERGED after {} iterations", iterations);
            let final_val = final_z.as_scalars().unwrap()[0];
            println!("  Final value: {}", final_val);
            println!("  Interpretation: This trajectory is BOUNDED (truth)\n");
        }
        IterationResult::Escaped { iterations } => {
            println!("\n✗ ESCAPED after {} iterations", iterations);
            println!("  Interpretation: This trajectory DIVERGED (farktogle)\n");
        }
        IterationResult::MaxIterations { iterations } => {
            println!("\n⚠ Max iterations ({}) reached\n", iterations);
        }
    }

    // Example 4: Multiple elements (vector state)
    println!("═══ Example 4: Vector State (3 elements) ═══\n");
    println!("Three independent trajectories:");
    println!("  [0.3, 0.5, 1.2] with c = [0, 0, 0.5]");
    println!("Expected: First two converge, third escapes\n");

    let mut config = ConvergenceConfig::default();
    config.max_iterations = 100;
    let mut engine = IterationEngine::with_config(config);

    let z = Tensor::from_scalars(
        vec![
            ScalarF4E4::from(3u8) / ScalarF4E4::from(10u8), // 0.3
            ScalarF4E4::ONE >> 1, // 0.5
            ScalarF4E4::ONE + (ScalarF4E4::ONE >> 1) + (ScalarF4E4::ONE / ScalarF4E4::from(10u8)), // 1.6
        ],
        Shape::vector(3)
    ).unwrap();

    let c = Tensor::from_scalars(
        vec![
            ScalarF4E4::ZERO,
            ScalarF4E4::ZERO,
            ScalarF4E4::ONE >> 1, // 0.5
        ],
        Shape::vector(3)
    ).unwrap();

    let (result, final_z) = engine.iterate_with_progress(z, c, |iter, z_state, mag, change| {
        if iter <= 5 || iter % 10 == 0 {
            let vals = z_state.as_scalars().unwrap();
            println!("  Iteration {}: [{}, {}, {}], mag = {}, change = {}",
                iter, vals[0], vals[1], vals[2], mag, change);
        }
    }).unwrap();

    match result {
        IterationResult::Converged { iterations } => {
            println!("\n✓ CONVERGED after {} iterations", iterations);
            let vals = final_z.as_scalars().unwrap();
            println!("  Final values: [{}, {}, {}]", vals[0], vals[1], vals[2]);
            println!("  Interpretation: All trajectories BOUNDED\n");
        }
        IterationResult::Escaped { iterations } => {
            println!("\n✗ ESCAPED after {} iterations", iterations);
            let vals = final_z.as_scalars().unwrap();
            println!("  Final values: [{}, {}, {}]", vals[0], vals[1], vals[2]);
            println!("  Interpretation: At least one trajectory DIVERGED\n");
        }
        IterationResult::MaxIterations { iterations } => {
            println!("\n⚠ Max iterations ({}) reached", iterations);
            let vals = final_z.as_scalars().unwrap();
            println!("  Final values: [{}, {}, {}]\n", vals[0], vals[1], vals[2]);
        }
    }

    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Key Insights                                ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("✓ Convergence = Bounded trajectory = Truth");
    println!("✓ Escape = Diverged trajectory = Farktogle");
    println!("✓ No token limits - iterate until math says \"done\"");
    println!("✓ Pure Spirix arithmetic (no IEEE violations)");
    println!("✓ Scales to vectors/tensors (multiple trajectories)");
    println!("\nThis is the foundation for:");
    println!("  → Encoding queries as c constants");
    println!("  → Neural network learns good initial z");
    println!("  → Iteration determines truth vs farktogle");
    println!("  → Training on symbolic verification\n");
}
