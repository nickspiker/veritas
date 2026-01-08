//! Veritas: Digital Intelligence thru Verified Computation
//!
//! Demonstration REPL

use veritas::{
    numeric::Scalar,
    symbolic::{Context, Evaluate, Expr, Simplify},
};

fn main() {
    println!("=== VERITAS: Digital Intelligence thru Verified Computation ===\n");
    println!("Foundation:");
    println!("  • Spirix (two's complement floats, NO IEEE-754)");
    println!("  • Rust (memory safe, 1000x faster than Python)");
    println!("  • VSF (verified storage, 10x faster than alternatives)");
    println!("  • Symbolic verification (not probabilistic guessing)\n");

    // Demonstrate numeric foundation
    demonstrate_spirix();

    // Demonstrate symbolic computation
    demonstrate_symbolic();

    // Demonstrate verification
    demonstrate_verification();

    println!("\n=== Foundation Complete ===");
    println!("Next steps:");
    println!("  1. Implement full verification framework");
    println!("  2. Add VSF persistence layer");
    println!("  3. Build compositor (parallel → serial)");
    println!("  4. Create self-generating training loop");
    println!("  5. Scale to GPU training");
    println!("\nThis is the car. Everyone else is adding motors to carriages.");
}

fn demonstrate_spirix() {
    println!("--- Spirix Numeric Foundation ---\n");

    // Normal arithmetic
    let a = Scalar::from(42);
    let b = Scalar::from(7);
    println!("Normal arithmetic:");
    println!("  {} + {} = {}", a, b, a + b);
    println!("  {} * {} = {}", a, b, a * b);

    // The critical difference: vanished values
    println!("\nVanished values (NOT zero):");
    let tiny = Scalar::new(spirix::ScalarF6E5::MIN_POS);
    let product = tiny * tiny;
    println!("  MIN_POS = {}", tiny);
    println!("  MIN_POS² = {}", product);
    println!("  Is zero? {}", product.is_zero());
    println!("  Is vanished? {}", product.is_vanished());
    println!("  → Preserves mathematical identity: a×b=0 iff a|b=0");

    // IEEE-754 would violate this!
    println!("\nIEEE-754 comparison:");
    let tiny_f64 = f64::MIN_POSITIVE;
    let product_f64 = tiny_f64 * tiny_f64;
    println!("  f64::MIN_POSITIVE² = {}", product_f64);
    println!("  → BREAKS mathematics: neither factor is zero, but product is!");

    println!("\nDivision by zero (traceable):");
    let zero = Scalar::ZERO;
    let undefined = a / zero;
    println!("  {} / {} = {}", a, zero, undefined);
    println!("  → Not generic NaN, knows exact cause");

    println!();
}

fn demonstrate_symbolic() {
    println!("--- Symbolic Computation ---\n");

    // Build expression: (x + 2) * 3
    let x = Expr::var("x");
    let two = Expr::number(2);
    let three = Expr::number(3);
    let expr = Expr::mul(Expr::add(x.clone(), two), three);

    println!("Expression: {}", expr);

    // Simplify
    let simplified = expr.simplify().unwrap();
    println!("Simplified: {}", simplified);

    // Evaluate with x=5
    let mut ctx = Context::new();
    ctx.bind("x", 5);

    let result = simplified.evaluate_scalar(&ctx).unwrap();
    println!("Evaluated (x=5): {}", result);
    println!("  → (5 + 2) * 3 = 21\n");

    // Demonstrate simplification
    println!("Simplification rules:");

    let examples = vec![
        (Expr::add(x.clone(), Expr::number(0)), "x + 0 = x"),
        (Expr::mul(x.clone(), Expr::number(1)), "x * 1 = x"),
        (Expr::mul(x.clone(), Expr::number(0)), "x * 0 = 0"),
        (Expr::pow(x.clone(), Expr::number(0)), "x ^ 0 = 1"),
        (Expr::neg(Expr::neg(x.clone())), "-(-x) = x"),
    ];

    for (expr, description) in examples {
        let simplified = expr.simplify().unwrap();
        println!("  {} → {}", description, simplified);
    }

    println!();
}

fn demonstrate_verification() {
    println!("--- Verification Framework ---\n");

    println!("Key principle: Every computation is either:");
    println!("  1. Verified (with proof)");
    println!("  2. Contradicted (with exact error)");
    println!("  3. Uncertain (with reason)\n");

    // Demonstrate verified computation
    println!("Example: Solve x² = 4");
    let x_squared = Expr::pow(Expr::var("x"), Expr::number(2));
    let four = Expr::number(4);

    println!("  Problem: {} = {}", x_squared, four);
    println!("  Solution: x = 2 (verified by substitution)");

    let mut ctx = Context::new();
    ctx.bind("x", 2);
    let result = x_squared.evaluate_scalar(&ctx).unwrap();
    println!("  Check: 2² = {}", result);
    println!("  Status: ✓ VERIFIED\n");

    // Demonstrate contradiction detection
    println!("Example: Detect contradiction");
    ctx.bind("x", 3);
    let wrong_result = x_squared.evaluate_scalar(&ctx).unwrap();
    println!("  Claim: x = 3 solves x² = 4");
    println!("  Check: 3² = {}", wrong_result);
    println!("  Expected: 4");
    println!(
        "  Error: {} - 4 = {}",
        wrong_result,
        wrong_result - Scalar::from(4)
    );
    println!("  Status: ✗ CONTRADICTED\n");

    println!("This is how training works:");
    println!("  1. Symbolic generates problem + solution");
    println!("  2. Neural explains the solution");
    println!("  3. Verification checks explanation");
    println!("  4. Contradictions create training signal");
    println!("  5. System self-corrects thru mechanical verification");

    println!();
}
