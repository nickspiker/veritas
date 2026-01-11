//! Simple Natural Language Math Parser
//!
//! Parses natural language math expressions like:
//! - "seven plus three"
//! - "What's 5 times 2?"
//! - "Eleven minus four"
//!
//! Routes to pure Spirix basecalc for computation.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimpleMathOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy)]
pub struct ParsedMath {
    pub a: u8,
    pub b: u8,
    pub op: SimpleMathOp,
}

/// Parse natural language math expression
///
/// Handles:
/// - Word numbers: "seven", "three"
/// - Digit numbers: "7", "3"
/// - Operation words: "plus", "add", "minus", "times"
/// - Mixed format: "What's seven plus 3?"
pub fn parse_natural_language_math(text: &str) -> Option<ParsedMath> {
    let lower = text.to_lowercase();

    // Extract first number
    let a = extract_first_number(&lower)?;

    // Extract operation
    let op = extract_operation(&lower)?;

    // Extract second number (after operation)
    let b = extract_second_number(&lower, op)?;

    Some(ParsedMath { a, b, op })
}

fn extract_first_number(text: &str) -> Option<u8> {
    // Word numbers (dozenal 0-11)
    let number_words = [
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
    ];

    // Check word numbers first (more specific)
    for (word, value) in &number_words {
        if text.contains(word) {
            return Some(*value);
        }
    }

    // Check digit form (0-11)
    // Look for first digit in text
    for ch in text.chars() {
        if ch.is_ascii_digit() {
            let digit = ch.to_digit(10)? as u8;
            // Check if it's 10 or 11 (two digits)
            let rest = text.split_once(ch)?.1;
            if digit == 1 {
                if rest.starts_with('0') {
                    return Some(10);
                } else if rest.starts_with('1') {
                    return Some(11);
                }
            }
            return Some(digit);
        }
    }

    None
}

fn extract_second_number(text: &str, op: SimpleMathOp) -> Option<u8> {
    // Find position after operation word
    let op_word = match op {
        SimpleMathOp::Add => {
            if text.contains("plus") {
                "plus"
            } else {
                "add"
            }
        }
        SimpleMathOp::Sub => {
            if text.contains("minus") {
                "minus"
            } else {
                "subtract"
            }
        }
        SimpleMathOp::Mul => {
            if text.contains("times") {
                "times"
            } else {
                "multiply"
            }
        }
        SimpleMathOp::Div => {
            if text.contains("divided") {
                "divided"
            } else {
                "divide"
            }
        }
    };

    let op_pos = text.find(op_word)? + op_word.len();
    let after_op = &text[op_pos..];

    // Word numbers
    let number_words = [
        ("zero", 0),
        ("one", 1),
        ("two", 2),
        ("three", 3),
        ("four", 4),
        ("five", 5),
        ("six", 6),
        ("seven", 7),
        ("eight", 8),
        ("nine", 9),
        ("ten", 10),
        ("eleven", 11),
    ];

    for (word, value) in &number_words {
        if after_op.contains(word) {
            return Some(*value);
        }
    }

    // Digit form
    for ch in after_op.chars() {
        if ch.is_ascii_digit() {
            let digit = ch.to_digit(10)? as u8;
            // Check for 10 or 11
            let rest = after_op.split_once(ch)?.1;
            if digit == 1 {
                if rest.starts_with('0') {
                    return Some(10);
                } else if rest.starts_with('1') {
                    return Some(11);
                }
            }
            return Some(digit);
        }
    }

    None
}

fn extract_operation(text: &str) -> Option<SimpleMathOp> {
    // Check in order of specificity
    if text.contains("plus") || text.contains("add") {
        return Some(SimpleMathOp::Add);
    }
    if text.contains("minus") || text.contains("subtract") {
        return Some(SimpleMathOp::Sub);
    }
    if text.contains("times") || text.contains("multiply") {
        return Some(SimpleMathOp::Mul);
    }
    if text.contains("divided") || text.contains("divide") {
        return Some(SimpleMathOp::Div);
    }

    // Check symbol form
    if text.contains('+') {
        return Some(SimpleMathOp::Add);
    }
    if text.contains('-') {
        return Some(SimpleMathOp::Sub);
    }
    if text.contains('*') || text.contains('×') {
        return Some(SimpleMathOp::Mul);
    }
    if text.contains('/') || text.contains('÷') {
        return Some(SimpleMathOp::Div);
    }

    None
}

/// Check if text contains math-related keywords
pub fn contains_math_keywords(text: &str) -> bool {
    let lower = text.to_lowercase();

    // Check for number words
    let has_number = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        "eleven",
    ]
    .iter()
    .any(|&word| lower.contains(word));

    // Check for operation words
    let has_operation = [
        "plus",
        "minus",
        "times",
        "divided",
        "add",
        "subtract",
        "multiply",
        "divide",
    ]
    .iter()
    .any(|&word| lower.contains(word));

    // Check for digits
    let has_digit = lower.chars().any(|c| c.is_ascii_digit());

    // Check for operation symbols
    let has_symbol = lower.contains('+')
        || lower.contains('-')
        || lower.contains('*')
        || lower.contains('/')
        || lower.contains('×')
        || lower.contains('÷');

    (has_number || has_digit) && (has_operation || has_symbol)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_word_numbers() {
        let parsed = parse_natural_language_math("seven plus three").unwrap();
        assert_eq!(parsed.a, 7);
        assert_eq!(parsed.b, 3);
        assert_eq!(parsed.op, SimpleMathOp::Add);
    }

    #[test]
    fn test_parse_digit_numbers() {
        let parsed = parse_natural_language_math("7 + 3").unwrap();
        assert_eq!(parsed.a, 7);
        assert_eq!(parsed.b, 3);
        assert_eq!(parsed.op, SimpleMathOp::Add);
    }

    #[test]
    fn test_parse_mixed() {
        let parsed = parse_natural_language_math("What's seven plus 3?").unwrap();
        assert_eq!(parsed.a, 7);
        assert_eq!(parsed.b, 3);
        assert_eq!(parsed.op, SimpleMathOp::Add);
    }

    #[test]
    fn test_contains_math_keywords() {
        assert!(contains_math_keywords("seven plus three"));
        assert!(contains_math_keywords("7 + 3"));
        assert!(!contains_math_keywords("hello world"));
        assert!(!contains_math_keywords("I have 7 apples"));
    }
}
