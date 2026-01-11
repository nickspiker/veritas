//! Intent Classification - What does the user want?
//!
//! After compression extracts signal, classify intent:
//! - Math query (calculation, integral, etc.)
//! - Travel query (directions, location info)
//! - Code query (program, execute)
//! - General query (explanation, facts)

use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;

/// Type of query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryType {
    /// Mathematical query (calculate, solve, integrate)
    Math,

    /// Travel/navigation query (how to get, where is)
    Travel,

    /// Code execution query (run, execute)
    Code,

    /// Logic/proof query (prove, verify)
    Logic,

    /// General explanation query
    General,

    /// Unknown/ambiguous
    Unknown,
}

/// Parsed intent
#[derive(Debug, Clone)]
pub struct Intent {
    /// Query type
    pub query_type: QueryType,

    /// Extracted entities (numbers, locations, etc.)
    pub entities: Vec<String>,

    /// Confidence (ZERO to ONE)
    pub confidence: ScalarF4E4,
}

/// Intent classifier
pub struct IntentClassifier {
    // For now: rule-based
    // TODO: Train neural classifier
}

impl IntentClassifier {
    /// Create new intent classifier
    pub fn new() -> Self {
        IntentClassifier {}
    }

    /// Classify intent from extracted query
    pub fn classify(&self, query: &str) -> Result<Intent> {
        let query_lower = query.to_lowercase();

        // Math keywords
        if self.contains_any(&query_lower, &["calculate", "solve", "integral", "derivative", "+", "-", "*", "/", "=", "what is"]) {
            return Ok(Intent {
                query_type: QueryType::Math,
                entities: self.extract_numbers(query),
                confidence: ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8), // 0.8
            });
        }

        // Travel keywords
        if self.contains_any(&query_lower, &["how do i get", "how to get", "travel to", "go to", "directions"]) {
            return Ok(Intent {
                query_type: QueryType::Travel,
                entities: self.extract_locations(query),
                confidence: ScalarF4E4::from(8u8) / ScalarF4E4::from(10u8), // 0.8
            });
        }

        // Code keywords
        if self.contains_any(&query_lower, &["run", "execute", "compile", "program"]) {
            return Ok(Intent {
                query_type: QueryType::Code,
                entities: Vec::new(),
                confidence: ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8), // 0.7
            });
        }

        // Logic keywords
        if self.contains_any(&query_lower, &["prove", "verify", "true", "false", "and", "or", "implies"]) {
            return Ok(Intent {
                query_type: QueryType::Logic,
                entities: Vec::new(),
                confidence: ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8), // 0.7
            });
        }

        // Default: general query
        Ok(Intent {
            query_type: QueryType::General,
            entities: Vec::new(),
            confidence: ScalarF4E4::ONE >> 1, // 0.5
        })
    }

    /// Check if query contains any of the keywords
    fn contains_any(&self, text: &str, keywords: &[&str]) -> bool {
        keywords.iter().any(|kw| text.contains(kw))
    }

    /// Extract numbers from query
    fn extract_numbers(&self, query: &str) -> Vec<String> {
        query.split_whitespace()
            .filter(|word| word.chars().any(|c| c.is_numeric()))
            .map(|s| s.to_string())
            .collect()
    }

    /// Extract location names (capitalized words)
    fn extract_locations(&self, query: &str) -> Vec<String> {
        query.split_whitespace()
            .filter(|word| {
                word.chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false)
            })
            .map(|s| s.to_string())
            .collect()
    }
}

impl Default for IntentClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_math_intent() {
        let classifier = IntentClassifier::new();

        let intent = classifier.classify("what is 2 + 3").unwrap();
        assert_eq!(intent.query_type, QueryType::Math);
        let threshold = ScalarF4E4::from(7u8) / ScalarF4E4::from(10u8);
        assert!(intent.confidence > threshold);
    }

    #[test]
    fn test_travel_intent() {
        let classifier = IntentClassifier::new();

        let intent = classifier.classify("how do I get to Rome").unwrap();
        assert_eq!(intent.query_type, QueryType::Travel);
        assert!(intent.entities.contains(&"Rome".to_string()));
    }

    #[test]
    fn test_code_intent() {
        let classifier = IntentClassifier::new();

        let intent = classifier.classify("run this program").unwrap();
        assert_eq!(intent.query_type, QueryType::Code);
    }
}
