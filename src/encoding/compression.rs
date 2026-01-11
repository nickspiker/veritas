//! Compression Encoder - Iterative compression to find signal
//!
//! ## Theory
//!
//! Information weight = compression survival
//! - High entropy (random noise) compresses to nothing
//! - Low entropy (structured signal) survives compression
//!
//! ## Method
//!
//! Iterate compression until stable:
//! 1. Start with all tokens
//! 2. Compress (remove low-weight tokens)
//! 3. Re-weight remaining tokens
//! 4. Repeat until convergence
//!
//! What survives = the actual query

use super::tokenizer::Token;
use spirix::ScalarF4E4;
use crate::error::Result;

/// Result of compression
#[derive(Debug, Clone)]
pub struct CompressionResult {
    /// Tokens that survived compression (high information density)
    pub signal: Vec<Token>,

    /// Tokens that were filtered out (noise)
    pub noise: Vec<Token>,

    /// Number of compression iterations
    pub iterations: usize,

    /// Final compression ratio (signal_len / original_len)
    pub compression_ratio: ScalarF4E4,
}

/// Compression encoder
pub struct CompressionEncoder {
    /// Minimum weight threshold for survival
    min_weight: ScalarF4E4,

    /// Maximum iterations
    max_iterations: usize,

    /// Convergence threshold (stop when change < this)
    convergence_threshold: ScalarF4E4,
}

impl CompressionEncoder {
    /// Create new compression encoder
    pub fn new() -> Self {
        CompressionEncoder {
            min_weight: ScalarF4E4::ONE / ScalarF4E4::from(2u8), // 0.5
            max_iterations: 10,
            convergence_threshold: ScalarF4E4::ONE / ScalarF4E4::from(100u8), // 0.01
        }
    }

    /// Compress tokens iteratively
    ///
    /// Each iteration:
    /// 1. Filter tokens below weight threshold
    /// 2. Re-weight remaining tokens based on context
    /// 3. Check convergence
    pub fn compress(&self, mut tokens: Vec<Token>) -> Result<CompressionResult> {
        let original_len = tokens.len();
        let mut noise = Vec::new();
        let mut iteration = 0;

        loop {
            iteration += 1;

            let prev_len = tokens.len();

            // Filter out low-weight tokens
            let mut new_tokens = Vec::new();
            for token in tokens {
                if token.weight >= self.min_weight {
                    new_tokens.push(token);
                } else {
                    noise.push(token);
                }
            }

            tokens = new_tokens;

            // Check convergence (no tokens removed)
            if tokens.len() == prev_len || iteration >= self.max_iterations {
                break;
            }

            // Re-weight based on context (simple: boost adjacent high-weight tokens)
            self.boost_context(&mut tokens);
        }

        let compression_ratio = if original_len > 0 {
            ScalarF4E4::from(tokens.len() as u32) / ScalarF4E4::from(original_len as u32)
        } else {
            ScalarF4E4::ZERO
        };

        Ok(CompressionResult {
            signal: tokens,
            noise,
            iterations: iteration,
            compression_ratio,
        })
    }

    /// Boost weights of tokens near other high-weight tokens
    ///
    /// Intuition: "how do I get to Rome" - all words have context
    /// Random symbols don't have context with anything
    fn boost_context(&self, tokens: &mut [Token]) {
        if tokens.len() < 2 {
            return;
        }

        // Simple context boost: average with neighbors
        let original_weights: Vec<ScalarF4E4> = tokens.iter().map(|t| t.weight).collect();

        for i in 0..tokens.len() {
            let left_weight = if i > 0 { original_weights[i - 1] } else { tokens[i].weight };
            let right_weight = if i < tokens.len() - 1 { original_weights[i + 1] } else { tokens[i].weight };

            // Average with neighbors (gives context boost)
            let sum = tokens[i].weight + left_weight + right_weight;
            let avg = sum / ScalarF4E4::from(3u8);
            tokens[i].weight = avg;
        }
    }

    /// Extract the most likely query from compressed tokens
    pub fn extract_query(&self, result: &CompressionResult) -> String {
        result.signal
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Default for CompressionEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_compression() {
        let encoder = CompressionEncoder::new();

        let tokens = vec![
            Token {
                text: "how".to_string(),
                position: 0,
                weight: ScalarF4E4::from(9u8) / ScalarF4E4::from(10u8),
            },
            Token {
                text: "☙".to_string(),
                position: 3,
                weight: ScalarF4E4::ONE / ScalarF4E4::from(10u8),
            },
            Token {
                text: "do".to_string(),
                position: 4,
                weight: ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8),
            },
            Token {
                text: "I".to_string(),
                position: 6,
                weight: ScalarF4E4::from(6u8) / ScalarF4E4::from(10u8),
            },
        ];

        let result = encoder.compress(tokens).unwrap();

        // Should keep high-weight tokens, filter noise
        assert!(result.signal.len() >= 2);
        assert!(result.noise.len() >= 1);
    }

    #[test]
    fn test_extract_query() {
        let encoder = CompressionEncoder::new();

        let result = CompressionResult {
            signal: vec![
                Token {
                    text: "how".to_string(),
                    position: 0,
                    weight: ScalarF4E4::from(9u8) / ScalarF4E4::from(10u8),
                },
                Token {
                    text: "to".to_string(),
                    position: 4,
                    weight: ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8),
                },
                Token {
                    text: "Rome".to_string(),
                    position: 7,
                    weight: ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8),
                },
            ],
            noise: vec![],
            iterations: 1,
            compression_ratio: ScalarF4E4::ONE >> 1,
        };

        let query = encoder.extract_query(&result);
        assert_eq!(query, "how to Rome");
    }
}
