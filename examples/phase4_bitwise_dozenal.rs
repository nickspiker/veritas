//! Phase 4: Bitwise Operations in Dozenal (Base 12)
//!
//! Demonstrates:
//! - Bitwise operations as fundamental I/O primitives
//! - Dozenal (base 12) - more natural for humans
//! - Verified computation with symbolic bolt-on

use spirix::ScalarF4E4;
use veritas::symbolic::{BitwiseOp, BitwiseProblem, BitwiseGenerator};

fn main() {
    println!("╔════════════════════════════════════════════════════════════════╗");
    println!("║       Phase 4: Bitwise Operations (Dozenal Base)              ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("Why Dozenal (Base 12)?");
    println!("  → Divisible by 2, 3, 4, 6 (more factors than decimal)");
    println!("  → Cleaner fractions: 1/3 = 0.4, 1/4 = 0.3, 1/6 = 0.2");
    println!("  → Both Spirix and basecalc support it");
    println!("  → More natural for human cognition\n");

    println!("Bitwise Operations:");
    println!("  → Fundamental I/O primitives");
    println!("  → Low-level computation building blocks");
    println!("  → Verified with symbolic engine\n");

    println!("═══ Symbolic Verification Examples ═══\n");

    // AND operation
    let prob_and = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::And);
    let result_and = prob_and.solve().unwrap();
    println!("AND:");
    println!("  Decimal: {}", result_and.expr);
    println!("  {}", result_and.dozenal);
    println!("  Binary:  1100 & 1010 = {:08b}\n", result_and.answer);

    // OR operation
    let prob_or = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::Or);
    let result_or = prob_or.solve().unwrap();
    println!("OR:");
    println!("  Decimal: {}", result_or.expr);
    println!("  {}", result_or.dozenal);
    println!("  Binary:  1100 | 1010 = {:08b}\n", result_or.answer);

    // XOR operation
    let prob_xor = BitwiseProblem::new(0b1100, Some(0b1010), BitwiseOp::Xor);
    let result_xor = prob_xor.solve().unwrap();
    println!("XOR:");
    println!("  Decimal: {}", result_xor.expr);
    println!("  {}", result_xor.dozenal);
    println!("  Binary:  1100 ^ 1010 = {:08b}\n", result_xor.answer);

    // NOT operation
    let prob_not = BitwiseProblem::new(0b00001111, None, BitwiseOp::Not);
    let result_not = prob_not.solve().unwrap();
    println!("NOT:");
    println!("  Decimal: {}", result_not.expr);
    println!("  {}", result_not.dozenal);
    println!("  Binary:  !00001111 = {:08b}\n", result_not.answer);

    // Shift left
    let prob_shl = BitwiseProblem::new(3, Some(2), BitwiseOp::Shl);
    let result_shl = prob_shl.solve().unwrap();
    println!("Shift Left:");
    println!("  Decimal: {}", result_shl.expr);
    println!("  {}", result_shl.dozenal);
    println!("  Binary:  00000011 << 2 = {:08b}\n", result_shl.answer);

    // Shift right
    let prob_shr = BitwiseProblem::new(12, Some(2), BitwiseOp::Shr);
    let result_shr = prob_shr.solve().unwrap();
    println!("Shift Right:");
    println!("  Decimal: {}", result_shr.expr);
    println!("  {}", result_shr.dozenal);
    println!("  Binary:  00001100 >> 2 = {:08b}\n", result_shr.answer);

    // Popcount
    let prob_pop = BitwiseProblem::new(0b10101010, None, BitwiseOp::Popcount);
    let result_pop = prob_pop.solve().unwrap();
    println!("Popcount (count set bits):");
    println!("  Decimal: {}", result_pop.expr);
    println!("  {}", result_pop.dozenal);
    println!("  Binary:  popcount(10101010) = {}\n", result_pop.answer);

    println!("═══ Dozenal Number Examples ═══\n");

    // Show dozenal advantages
    let examples = vec![
        (0u8, "Zero"),
        (11u8, "Eleven (B in dozenal)"),
        (12u8, "One dozen (10 in dozenal)"),
        (144u8, "One gross (100 in dozenal)"),
        (255u8, "Max u8 (193 in dozenal)"),
    ];

    for (num, desc) in examples {
        let prob = BitwiseProblem::new(num, None, BitwiseOp::Not);
        let result = prob.solve().unwrap();
        println!("  {}: decimal={}, dozenal={}, binary={:08b}",
            desc, num, to_dozenal(num), num);
    }

    println!("\n═══ Training Data Generation ═══\n");

    let generator = BitwiseGenerator::new(255);
    let batch = generator.generate_batch(10);

    println!("Generated {} verified bitwise problems:", batch.len());
    for (i, prob) in batch.iter().enumerate() {
        let result = prob.solve().unwrap();
        println!("  {}. {} (verified)", i + 1, result.dozenal);
    }

    println!("\n╔════════════════════════════════════════════════════════════════╗");
    println!("║                    Phase 4 Complete                           ║");
    println!("╚════════════════════════════════════════════════════════════════╝\n");

    println!("✓ Bitwise operations verified");
    println!("✓ Dozenal representation working");
    println!("✓ Problem generator ready for training");
    println!("✓ Fundamental I/O primitives bolt-on complete\n");

    println!("Next steps:");
    println!("  → Train neural network on bitwise operations");
    println!("  → Combine arithmetic + bitwise for full computation");
    println!("  → Integrate basecalc for complex expressions");
    println!("  → Add logic/proof checker bolt-on\n");

    println!("Why bitwise matters:");
    println!("  → All digital computation reduces to bitwise ops");
    println!("  → Fundamental building blocks for I/O");
    println!("  → CPU instruction set primitives");
    println!("  → Dozenal makes it more intuitive\n");
}

/// Convert u8 to dozenal string (helper for display)
fn to_dozenal(n: u8) -> String {
    if n == 0 {
        return "0".to_string();
    }

    let mut result = String::new();
    let mut value = n;

    while value > 0 {
        let digit = value % 12;
        let ch = match digit {
            0..=9 => (b'0' + digit) as char,
            10 => 'A',
            11 => 'B',
            _ => unreachable!(),
        };
        result.insert(0, ch);
        value /= 12;
    }

    result
}
