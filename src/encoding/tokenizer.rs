//! Tokenizer - Convert text to tokens
//!
//! Simple byte-pair encoding style tokenizer
//! Handles Unicode, typos, and noise gracefully

use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;

/// A token in the input
#[derive(Debug, Clone)]
pub struct Token {
    /// The token string
    pub text: String,
    /// Position in original text
    pub position: usize,
    /// Information weight (ZERO = noise, ONE = high signal)
    pub weight: ScalarF4E4,
}

/// Simple tokenizer
pub struct Tokenizer {
    // For now, simple whitespace + punctuation splitting
    // TODO: BPE vocabulary
}

impl Tokenizer {
    /// Create new tokenizer
    pub fn new() -> Self {
        Tokenizer {}
    }

    /// Tokenize raw text into tokens
    pub fn tokenize(&self, text: &str) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();
        let mut current = String::new();
        let mut position = 0;

        for (i, ch) in text.chars().enumerate() {
            if ch.is_whitespace() || ch.is_ascii_punctuation() {
                // End current token
                if !current.is_empty() {
                    tokens.push(Token {
                        text: current.clone(),
                        position,
                        weight: ScalarF4E4::ONE, // Initial weight (will be learned)
                    });
                    current.clear();
                }

                // Add punctuation as token
                if ch.is_ascii_punctuation() {
                    tokens.push(Token {
                        text: ch.to_string(),
                        position: i,
                        weight: ScalarF4E4::ONE >> 1, // 0.5 - Punctuation has lower default weight
                    });
                }

                position = i + 1;
            } else {
                current.push(ch);
            }
        }

        // Final token
        if !current.is_empty() {
            tokens.push(Token {
                text: current,
                position,
                weight: ScalarF4E4::ONE,
            });
        }

        Ok(tokens)
    }

    /// Estimate information weight of a token
    ///
    /// Heuristics:
    /// - Random Unicode: low weight
    /// - Dictionary words: high weight
    /// - Common words: medium weight
    /// - Typos: low weight (for now)
    pub fn estimate_weight(&self, token: &str) -> ScalarF4E4 {
        // Check if purely symbols
        if token.chars().all(|c| !c.is_alphanumeric()) {
            return ScalarF4E4::ONE / ScalarF4E4::from(10u8); // 0.1 - Noise
        }

        // Check if all caps random letters (like "AOCRHE")
        if token.len() > 5 && token.chars().all(|c| c.is_uppercase()) {
            return ScalarF4E4::from(2u8) / ScalarF4E4::from(10u8); // 0.2 - Likely noise
        }

        // Check for repeated characters (like "cosechhhj")
        let has_repeated = token.chars()
            .collect::<Vec<_>>()
            .windows(3)
            .any(|w| w[0] == w[1] && w[1] == w[2]);
        if has_repeated {
            return ScalarF4E4::from(3u8) / ScalarF4E4::from(10u8); // 0.3 - Likely typo/noise
        }

        // Common question words = high signal
        match token.to_lowercase().as_str() {
            "how" | "what" | "where" | "when" | "why" | "who" => {
                return ScalarF4E4::from(9u8) / ScalarF4E4::from(10u8); // 0.9
            }
            "do" | "does" | "is" | "are" | "can" | "could" => {
                return ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8); // 0.8
            }
            "get" | "find" | "calculate" | "solve" => {
                return ScalarF4E4::from(85u8) / ScalarF4E4::from(100u8); // 0.85
            }
            _ => {}
        }

        // Location names = high signal (simple heuristic: capitalized)
        if token.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && token.len() > 2 {
            return ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8); // 0.8
        }

        // Default: medium weight
        ScalarF4E4::ONE >> 1 // 0.5
    }

    /// Apply weight estimation to tokens
    pub fn apply_weights(&self, tokens: &mut [Token]) {
        for token in tokens.iter_mut() {
            token.weight = self.estimate_weight(&token.text);
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tokenization() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("hello world").unwrap();

        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, "world");
    }

    #[test]
    fn test_punctuation() {
        let tokenizer = Tokenizer::new();
        let tokens = tokenizer.tokenize("hello, world!").unwrap();

        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].text, "hello");
        assert_eq!(tokens[1].text, ",");
        assert_eq!(tokens[2].text, "world");
        assert_eq!(tokens[3].text, "!");
    }

    #[test]
    fn test_weight_estimation() {
        let tokenizer = Tokenizer::new();

        let threshold_08 = ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8);
        let threshold_02 = ScalarF4E4::from(2u8) / ScalarF4E4::from(10u8);
        let threshold_04 = ScalarF4E4::from(4u8) / ScalarF4E4::from(10u8);
        let threshold_07 = ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8);

        // High weight: question words
        assert!(tokenizer.estimate_weight("how") > threshold_08);
        assert!(tokenizer.estimate_weight("what") > threshold_08);

        // Low weight: symbols
        assert!(tokenizer.estimate_weight("☙❦⁂") < threshold_02);

        // Low weight: repeated chars
        assert!(tokenizer.estimate_weight("cosechhhj") < threshold_04);

        // High weight: location names
        assert!(tokenizer.estimate_weight("Rome") > threshold_07);
    }
}
