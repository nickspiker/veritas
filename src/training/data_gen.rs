//! Training data generation for routing learning
//!
//! Generates simple math problems to train RNN to detect and route math

use crate::encoding::{preprocess, Operator};
use crate::symbolic::ArithProblem;
use spirix::ScalarF4E4;
use rand::Rng;

/// Training example
#[derive(Debug, Clone)]
pub struct TrainingExample {
    /// Raw input text
    pub input: String,

    /// Target output (correct continuation)
    pub target: String,

    /// Whether this example contains math that should be routed
    pub has_math: bool,
}

/// Generate N training examples with simple math problems
///
/// Examples:
/// - "What is 2 + 3? The answer is 5"
/// - "Calculate 7 * 8. The result is 56"
/// - "Add 15 and 23. The sum is 38"
pub fn generate_math_examples(n: usize) -> Vec<TrainingExample> {
    let mut rng = rand::thread_rng();
    let mut examples = Vec::with_capacity(n);

    for _ in 0..n {
        let op_type = rng.gen_range(0..4);

        let (left, right, op) = match op_type {
            0 => {
                // Addition: small numbers
                let a = rng.gen_range(1..50);
                let b = rng.gen_range(1..50);
                (a, b, '+')
            }
            1 => {
                // Subtraction: ensure positive result
                let a = rng.gen_range(10..100);
                let b = rng.gen_range(1..a);
                (a, b, '-')
            }
            2 => {
                // Multiplication: small numbers
                let a = rng.gen_range(2..12);
                let b = rng.gen_range(2..12);
                (a, b, '*')
            }
            _ => {
                // Division: ensure clean division
                let b = rng.gen_range(2..10);
                let quotient = rng.gen_range(2..20);
                let a = b * quotient;
                (a, b, '/')
            }
        };

        // Compute answer using symbolic engine
        let expr = format!("{} {} {}", left, op, right);
        let answer = if let Ok(prob) = ArithProblem::parse(&expr) {
            if let Ok(result) = prob.solve() {
                // Extract numeric answer (Spirix format is verbose)
                // For now, just compute directly
                match op {
                    '+' => left + right,
                    '-' => left - right,
                    '*' => left * right,
                    '/' => left / right,
                    _ => 0,
                }
            } else {
                0
            }
        } else {
            0
        };

        // Generate natural language question
        let question = match rng.gen_range(0..5) {
            0 => format!("What is {} {} {}?", left, op, right),
            1 => format!("Calculate {} {} {}", left, op, right),
            2 => format!("Compute {} {} {}.", left, op, right),
            3 => format!("Solve: {} {} {}", left, op, right),
            _ => format!("{} {} {} = ?", left, op, right),
        };

        let target_answer = match rng.gen_range(0..3) {
            0 => format!(" The answer is {}", answer),
            1 => format!(" The result is {}", answer),
            _ => format!(" {}", answer),
        };

        examples.push(TrainingExample {
            input: question.clone(),
            target: format!("{}{}", question, target_answer),
            has_math: true,
        });
    }

    examples
}

/// Generate non-math examples for contrast
///
/// Examples:
/// - "Hello, world! Nice to meet you."
/// - "The quick brown fox jumps over the lazy dog."
pub fn generate_text_examples(n: usize) -> Vec<TrainingExample> {
    let templates = vec![
        "Hello, world!",
        "The quick brown fox jumps.",
        "Nice to meet you.",
        "How are you today?",
        "This is a test sentence.",
        "Artificial intelligence is fascinating.",
        "The weather is nice today.",
        "I like reading books.",
        "Programming is fun.",
        "Let's go for a walk.",
    ];

    let mut examples = Vec::with_capacity(n);
    let mut rng = rand::thread_rng();

    for _ in 0..n {
        let text = templates[rng.gen_range(0..templates.len())];
        examples.push(TrainingExample {
            input: text.to_string(),
            target: text.to_string(),
            has_math: false,
        });
    }

    examples
}

/// Generate mixed training set (math + text)
pub fn generate_training_set(n_math: usize, n_text: usize) -> Vec<TrainingExample> {
    let mut examples = generate_math_examples(n_math);
    examples.extend(generate_text_examples(n_text));

    // Shuffle examples
    use rand::seq::SliceRandom;
    let mut rng = rand::thread_rng();
    examples.shuffle(&mut rng);

    examples
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_math_examples() {
        let examples = generate_math_examples(100);
        assert_eq!(examples.len(), 100);

        for ex in &examples {
            assert!(ex.has_math);
            assert!(ex.input.len() > 0);
            assert!(ex.target.len() > 0);
        }
    }

    #[test]
    fn test_generate_text_examples() {
        let examples = generate_text_examples(50);
        assert_eq!(examples.len(), 50);

        for ex in &examples {
            assert!(!ex.has_math);
        }
    }

    #[test]
    fn test_generate_training_set() {
        let examples = generate_training_set(70, 30);
        assert_eq!(examples.len(), 100);

        let math_count = examples.iter().filter(|e| e.has_math).count();
        assert_eq!(math_count, 70);
    }
}
