//! Demonstration of verified self-play training
//!
//! Shows how the system self-corrects thru mechanical verification.

use veritas::training::TrainingLoop;

fn main() {
    println!("=== VERITAS: Verified Self-Play Training Demo ===\n");

    println!("Architecture:");
    println!("  1. Symbolic engine generates problem + solution (ground truth)");
    println!("  2. Neural explainer generates explanation + answer");
    println!("  3. Verifier checks neural vs symbolic");
    println!("  4. Contradictions create training signal");
    println!("  5. System self-corrects\n");

    // Create training loop with intentional errors
    println!("--- Training Loop (with intentional errors for demonstration) ---\n");
    let mut training = TrainingLoop::new(true);  // introduce_errors = true

    // Run 15 iterations
    let iterations = training.train(15).expect("Training failed");

    // Show results
    for (i, iter) in iterations.iter().enumerate() {
        println!("Iteration {}:", i + 1);
        println!("  Problem: {}", iter.problem);

        match &iter.verification {
            veritas::training::VerificationResult::Correct { explanation, answer } => {
                println!("  ✓ VERIFIED");
                println!("  Explanation: {}", explanation);
                println!("  Answer: {}", answer);
            }
            veritas::training::VerificationResult::Contradicted {
                explanation,
                claimed_answer,
                correct_answer,
                error,
            } => {
                println!("  ✗ CONTRADICTION DETECTED");
                println!("  Explanation: {}", explanation);
                println!("  Neural claimed: {}", claimed_answer);
                println!("  Symbolic truth: {}", correct_answer);
                println!("  Error: {}", error);
                println!("  → This creates training signal to correct the model");
            }
        }
        println!();
    }

    // Show statistics
    let stats = training.stats();
    println!("--- Training Statistics ---");
    println!("Total iterations: {}", stats.total_verified);
    println!("Correct: {} ({:.1}%)", stats.correct, stats.accuracy() * 100.0);
    println!("Contradicted: {} ({:.1}%)",
        stats.contradicted,
        (stats.contradicted as f64 / stats.total_verified as f64) * 100.0
    );

    println!("\n--- Key Insight ---");
    println!("In production:");
    println!("  • Contradictions update neural weights (backprop)");
    println!("  • Model learns from mechanical verification");
    println!("  • NO human labels needed");
    println!("  • NO \"hope it's right\" - mathematically proven");
    println!("  • Infinite verified training data from symbolic engine");

    println!("\nThis is AlphaGo for reasoning:");
    println!("  • AlphaGo: Self-play against game rules");
    println!("  • Veritas: Self-play against mathematical truth");
}
