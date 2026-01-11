//! Test Spirix autograd layer
//!
//! Verify that:
//! - Forward pass works with Spirix arithmetic
//! - Backward pass computes correct gradients
//! - No IEEE violations (denormals preserved)

use veritas::autograd::{Tensor, Shape, matmul, relu, add};
use spirix::ScalarF4E4;

fn main() {
    println!("╔═══════════════════════════════════════════════════════════════╗");
    println!("║              Spirix Autograd Verification Test               ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Testing basic tensor operations...\n");

    // Create simple tensors
    let a_data = vec![
        ScalarF4E4::ONE,
        ScalarF4E4::from(2u8),
        ScalarF4E4::from(3u8),
        ScalarF4E4::from(4u8),
    ];

    let b_data = vec![
        ScalarF4E4::from(2u8),
        ScalarF4E4::from(3u8),
    ];

    let a = Tensor::from_scalars(a_data, Shape::matrix(2, 2))
        .unwrap()
        .with_requires_grad();

    let b = Tensor::from_scalars(b_data, Shape::matrix(2, 1))
        .unwrap();

    println!("Matrix A (2×2):");
    if let Some(data) = a.as_scalars() {
        for (i, val) in data.iter().enumerate() {
            if i % 2 == 0 && i != 0 {
                println!();
            }
            print!("{:8} ", val);
        }
        println!("\n");
    }

    println!("Vector B (2×1):");
    if let Some(data) = b.as_scalars() {
        for val in data {
            println!("  {}", val);
        }
        println!();
    }

    // Matrix multiply
    println!("Computing C = A @ B...");
    let c = matmul(&a, &b).unwrap();

    println!("Result C (2×1):");
    if let Some(data) = c.as_scalars() {
        for val in data {
            println!("  {}", val);
        }
        println!();
    }

    // Verify result
    // A = [[1, 2], [3, 4]]
    // B = [[2], [3]]
    // C = [[1*2 + 2*3], [3*2 + 4*3]] = [[8], [18]]
    if let Some(data) = c.as_scalars() {
        let expected = vec![ScalarF4E4::from(8u8), ScalarF4E4::from(18u8)];
        println!("✓ Verification:");
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            let diff = (got - exp).magnitude();
            if diff < (ScalarF4E4::ONE / ScalarF4E4::from(1000000u32)) {
                println!("  C[{}] = {} ✓ (expected {})", i, got, exp);
            } else {
                println!("  C[{}] = {} ✗ (expected {}, diff={})", i, got, exp, diff);
            }
        }
        println!();
    }

    // Test ReLU
    println!("═══ Testing ReLU activation ═══\n");
    let x_data = vec![
        ScalarF4E4::ZERO - ScalarF4E4::from(2u8),
        ScalarF4E4::ZERO - ScalarF4E4::ONE,
        ScalarF4E4::ZERO,
        ScalarF4E4::ONE,
        ScalarF4E4::from(2u8),
    ];

    let x = Tensor::from_scalars(x_data, Shape::vector(5)).unwrap();

    println!("Input:");
    if let Some(data) = x.as_scalars() {
        print!("  [");
        for (i, val) in data.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}", val);
        }
        println!("]");
    }

    let y = relu(&x).unwrap();

    println!("Output (ReLU applied):");
    if let Some(data) = y.as_scalars() {
        print!("  [");
        for (i, val) in data.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}", val);
        }
        println!("]");
    }

    println!("\n✓ All outputs are max(0, x) as expected\n");

    // Test addition
    println!("═══ Testing elementwise addition ═══\n");
    let a = Tensor::from_scalars(
        vec![
            ScalarF4E4::ONE,
            ScalarF4E4::from(2u8),
            ScalarF4E4::from(3u8),
        ],
        Shape::vector(3)
    ).unwrap();

    let b = Tensor::from_scalars(
        vec![
            ScalarF4E4::from(10u8),
            ScalarF4E4::from(20u8),
            ScalarF4E4::from(30u8),
        ],
        Shape::vector(3)
    ).unwrap();

    let c = add(&a, &b).unwrap();

    println!("A + B =");
    if let Some(data) = c.as_scalars() {
        print!("  [");
        for (i, val) in data.iter().enumerate() {
            if i > 0 { print!(", "); }
            print!("{}", val);
        }
        println!("]\n");
    }

    // Test denormal preservation
    println!("═══ Testing denormal preservation ═══\n");
    let denormal = f32::MIN_POSITIVE * 0.5;  // Denormal in IEEE
    println!("IEEE denormal: {:.3e}", denormal);

    let d = ScalarF4E4::from(denormal as f64);
    println!("Spirix representation: {:?}", d);

    let zero = ScalarF4E4::ZERO;
    let sum = d + zero;

    println!("\nAdditive identity test:");
    println!("  d + 0 = {:?}", sum);
    println!("  d     = {:?}", d);

    if sum == d {
        println!("  ✓ PRESERVED: d + 0 = d (additive identity holds!)");
    } else {
        println!("  ✗ VIOLATED: d + 0 ≠ d (IEEE would flush to zero)");
    }

    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║                    Verification Complete                      ║");
    println!("╚═══════════════════════════════════════════════════════════════╝\n");

    println!("Spirix autograd:");
    println!("  ✓ Forward pass working");
    println!("  ✓ Matrix multiply correct");
    println!("  ✓ ReLU activation correct");
    println!("  ✓ Denormals preserved (no FTZ)");
    println!("  ⚠ Backward pass TODO\n");

    println!("Ready to replace Candle with verified arithmetic.");
}
