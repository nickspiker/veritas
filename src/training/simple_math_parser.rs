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
/// - Fractions: "a third of nine"
pub fn parse_natural_language_math(text: &str) -> Option<ParsedMath> {
    let lower = text.to_lowercase();

    // Extract operation first to find boundaries
    let op = extract_operation(&lower)?;

    // Extract first number (before operation)
    let (a, first_num_end) = extract_first_number_with_pos(&lower)?;

    // Extract second number (after first number position)
    let b = extract_second_number_after_pos(&lower[first_num_end..], op)?;

    Some(ParsedMath { a, b, op })
}

fn extract_first_number_with_pos(text: &str) -> Option<(u8, usize)> {
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

    // Find EARLIEST occurrence (left-to-right parsing)
    let mut earliest_pos = usize::MAX;
    let mut earliest_num = None;
    let mut earliest_end = 0;

    for (word, value) in &number_words {
        if let Some(pos) = text.find(word) {
            if pos < earliest_pos {
                earliest_pos = pos;
                earliest_num = Some(*value);
                earliest_end = pos + word.len();
            }
        }
    }

    if earliest_num.is_some() {
        return Some((earliest_num.unwrap(), earliest_end));
    }

    // Check digit form (0-11) - find first digit
    for (pos, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            let digit = ch.to_digit(10)? as u8;
            // Check if it's 10 or 11 (two digits)
            if digit == 1 && pos + 1 < text.len() {
                let next_char = text.chars().nth(pos + 1)?;
                if next_char == '0' {
                    return Some((10, pos + 2));
                } else if next_char == '1' {
                    return Some((11, pos + 2));
                }
            }
            return Some((digit, pos + 1));
        }
    }

    None
}

fn extract_second_number_after_pos(text: &str, op: SimpleMathOp) -> Option<u8> {
    // Handle fractions with implicit divisors
    if op == SimpleMathOp::Div {
        if text.contains("third") {
            return Some(3);
        }
        if text.contains("half") || text.contains("halves") {
            return Some(2);
        }
        if text.contains("quarter") {
            return Some(4);
        }
        if text.contains("fifth") {
            return Some(5);
        }
    }

    // Word numbers - find earliest in remaining text
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

    let mut earliest_pos = usize::MAX;
    let mut earliest_num = None;

    for (word, value) in &number_words {
        if let Some(pos) = text.find(word) {
            if pos < earliest_pos {
                earliest_pos = pos;
                earliest_num = Some(*value);
            }
        }
    }

    if earliest_num.is_some() {
        return earliest_num;
    }

    // Digit form - find first digit
    for (pos, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            let digit = ch.to_digit(10)? as u8;
            // Check for 10 or 11
            if digit == 1 && pos + 1 < text.len() {
                let next_char = text.chars().nth(pos + 1)?;
                if next_char == '0' {
                    return Some(10);
                } else if next_char == '1' {
                    return Some(11);
                }
            }
            return Some(digit);
        }
    }

    None
}

fn extract_second_number(text: &str, op: SimpleMathOp) -> Option<u8> {
    // Handle fractions with implicit divisors
    if op == SimpleMathOp::Div {
        if text.contains("third") {
            return Some(3);
        }
        if text.contains("half") || text.contains("halves") {
            return Some(2);
        }
        if text.contains("quarter") {
            return Some(4);
        }
        if text.contains("fifth") {
            return Some(5);
        }
    }

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

    // Word numbers - find earliest in text after operation
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

    let mut earliest_pos = usize::MAX;
    let mut earliest_num = None;

    for (word, value) in &number_words {
        if let Some(pos) = after_op.find(word) {
            if pos < earliest_pos {
                earliest_pos = pos;
                earliest_num = Some(*value);
            }
        }
    }

    if earliest_num.is_some() {
        return earliest_num;
    }

    // Digit form - find first digit after operation
    for (pos, ch) in after_op.char_indices() {
        if ch.is_ascii_digit() {
            let digit = ch.to_digit(10)? as u8;
            // Check for 10 or 11
            if digit == 1 && pos + 1 < after_op.len() {
                let next_char = after_op.chars().nth(pos + 1)?;
                if next_char == '0' {
                    return Some(10);
                } else if next_char == '1' {
                    return Some(11);
                }
            }
            return Some(digit);
        }
    }

    None
}

fn extract_operation(text: &str) -> Option<SimpleMathOp> {
    // Check for fractions first (more specific)
    if text.contains("third") || text.contains("thirds") {
        return Some(SimpleMathOp::Div);
    }
    if text.contains("half") || text.contains("halves") {
        return Some(SimpleMathOp::Div);
    }
    if text.contains("quarter") || text.contains("quarters") {
        return Some(SimpleMathOp::Div);
    }
    if text.contains("fifth") || text.contains("fifths") {
        return Some(SimpleMathOp::Div);
    }

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

    // Check for operation words (including fractions)
    let has_operation = [
        "plus",
        "minus",
        "times",
        "divided",
        "add",
        "subtract",
        "multiply",
        "divide",
        "third",
        "half",
        "quarter",
        "fifth",
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
