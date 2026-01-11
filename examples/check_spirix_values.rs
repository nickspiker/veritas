//! Check for ambiguous/undefined Spirix values during RNN training

use spirix::{ScalarF4E4, Tensor};
use veritas::training::data_gen::generate_training_set;
use veritas::gpu::matmul_gpu;

const VOCAB_SIZE: usize = 256;
const HIDDEN_SIZE: usize = 32;

struct RNNCell {
    w_ih: Tensor<ScalarF4E4>,
    w_hh: Tensor<ScalarF4E4>,
    w_ho: Tensor<ScalarF4E4>,
}

impl RNNCell {
    fn new() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = 0.01;

        let w_ih_data: Vec<ScalarF4E4> = (0..VOCAB_SIZE * HIDDEN_SIZE)
            .map(|_| ScalarF4E4::from((rng.gen::<f64>() - 0.5) * scale))
            .collect();

        let w_hh_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * HIDDEN_SIZE)
            .map(|_| ScalarF4E4::from((rng.gen::<f64>() - 0.5) * scale))
            .collect();

        let w_ho_data: Vec<ScalarF4E4> = (0..HIDDEN_SIZE * VOCAB_SIZE)
            .map(|_| ScalarF4E4::from((rng.gen::<f64>() - 0.5) * scale))
            .collect();

        RNNCell {
            w_ih: Tensor::new(w_ih_data, vec![VOCAB_SIZE, HIDDEN_SIZE]),
            w_hh: Tensor::new(w_hh_data, vec![HIDDEN_SIZE, HIDDEN_SIZE]),
            w_ho: Tensor::new(w_ho_data, vec![HIDDEN_SIZE, VOCAB_SIZE]),
        }
    }

    fn forward(
        &self,
        input: &Tensor<ScalarF4E4>,
        hidden: &Tensor<ScalarF4E4>,
    ) -> (Tensor<ScalarF4E4>, Tensor<ScalarF4E4>) {
        let input_2d = Tensor::new(input.data.clone(), vec![1, VOCAB_SIZE]);
        let hidden_2d = Tensor::new(hidden.data.clone(), vec![1, HIDDEN_SIZE]);

        let ih_contrib = matmul_gpu(&input_2d, &self.w_ih);
        let hh_contrib = matmul_gpu(&hidden_2d, &self.w_hh);

        let new_hidden_data: Vec<ScalarF4E4> = ih_contrib.data.iter()
            .zip(hh_contrib.data.iter())
            .map(|(a, b)| {
                let sum = *a + *b;
                let val = sum.to_f64();
                ScalarF4E4::from(val.tanh())
            })
            .collect();

        let new_hidden = Tensor::new(new_hidden_data.clone(), vec![1, HIDDEN_SIZE]);
        let logits = matmul_gpu(&new_hidden, &self.w_ho);
        let hidden_flat = Tensor::new(new_hidden_data, vec![HIDDEN_SIZE]);
        (hidden_flat, logits)
    }
}

fn main() {
    println!("=== Checking for Ambiguous Spirix Values ===\n");

    // Generate small dataset
    let examples = generate_training_set(10, 10);
    let rnn = RNNCell::new();

    let mut all_values = Vec::new();
    let mut ambiguous_count = 0;
    let mut undefined_count = 0;
    let mut zero_count = 0;
    let mut infinity_count = 0;
    let mut normal_count = 0;

    println!("Processing {} examples...\n", examples.len());

    for (idx, example) in examples.iter().take(5).enumerate() {
        let bytes = example.target.as_bytes();
        let mut hidden = Tensor::new(
            vec![ScalarF4E4::ZERO; HIDDEN_SIZE],
            vec![HIDDEN_SIZE]
        );

        for &byte in bytes.iter().take(10) {
            let mut one_hot = vec![ScalarF4E4::ZERO; VOCAB_SIZE];
            one_hot[byte as usize] = ScalarF4E4::ONE;
            let input = Tensor::new(one_hot, vec![VOCAB_SIZE]);

            let (new_hidden, logits) = rnn.forward(&input, &hidden);

            // Collect all values
            all_values.extend(new_hidden.data.iter().cloned());
            all_values.extend(logits.data.iter().cloned());

            hidden = new_hidden;
        }
    }

    println!("Collected {} Spirix values from training\n", all_values.len());

    // Analyze each value
    for val in &all_values {
        // Check exponent for special values
        // AMBIGUOUS_EXPONENT = -32768 (i16::MIN)
        if val.exponent == -32768 {
            if val.fraction == 0 {
                zero_count += 1;
            } else {
                ambiguous_count += 1;
                if ambiguous_count <= 5 {
                    println!("Ambiguous value found: frac={}, exp={}", val.fraction, val.exponent);
                }
            }
        } else if val.is_undefined() {
            undefined_count += 1;
            if undefined_count <= 5 {
                println!("Undefined value found: frac={}, exp={}", val.fraction, val.exponent);
            }
        } else if val.fraction == 0x7FFF || val.fraction == -32768 {
            // Check for infinity markers
            infinity_count += 1;
        } else {
            normal_count += 1;
        }
    }

    println!("\n=== Value Distribution ===");
    println!("Total values checked: {}", all_values.len());
    println!("  Normal values:     {} ({:.1}%)", normal_count,
        100.0 * normal_count as f64 / all_values.len() as f64);
    println!("  Zero values:       {} ({:.1}%)", zero_count,
        100.0 * zero_count as f64 / all_values.len() as f64);
    println!("  Ambiguous values:  {} ({:.1}%)", ambiguous_count,
        100.0 * ambiguous_count as f64 / all_values.len() as f64);
    println!("  Undefined values:  {} ({:.1}%)", undefined_count,
        100.0 * undefined_count as f64 / all_values.len() as f64);
    println!("  Infinity values:   {} ({:.1}%)", infinity_count,
        100.0 * infinity_count as f64 / all_values.len() as f64);

    if ambiguous_count > 0 || undefined_count > 0 {
        println!("\n⚠ WARNING: Ambiguous or undefined values detected!");
        println!("This may indicate:");
        println!("  - Division by zero");
        println!("  - Overflow/underflow");
        println!("  - Invalid operations");
    } else if zero_count > all_values.len() / 2 {
        println!("\n⚠ WARNING: >50% zero values - possible dead network");
    } else {
        println!("\n✓ All values are normal or zero - no anomalies detected");
    }

    // Show some sample normal values
    println!("\n=== Sample Normal Values ===");
    let normal_samples: Vec<_> = all_values.iter()
        .filter(|v| v.exponent != -32768 && !v.is_undefined())
        .take(10)
        .collect();

    for (i, val) in normal_samples.iter().enumerate() {
        println!("  [{}] frac={:6}, exp={:6} → {:.6}",
            i, val.fraction, val.exponent, val.to_f64());
    }
}
