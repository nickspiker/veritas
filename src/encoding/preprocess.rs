//! Preprocessing pipeline for byte-level routing
//!
//! Key features:
//! - Sequential placeholder numbering: <MATH_0>, <MATH_1>, etc.
//! - Operator replacement: <OP_ADD>, <OP_SUB>, <OP_MUL>, <OP_DIV>, <OP_POW>
//! - Number extraction stored in parallel array
//! - Handles integers, decimals, and fractions

use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;
use regex::Regex;

/// Preprocessing result containing transformed text and extracted data
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// Preprocessed text with placeholders
    pub text: String,

    /// Extracted numbers (as Spirix scalars)
    pub numbers: Vec<ScalarF4E4>,

    /// Extracted operators
    pub operators: Vec<Operator>,
}

/// Mathematical operators
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl Operator {
    /// Convert operator to placeholder token
    pub fn to_token(&self) -> &'static str {
        match self {
            Operator::Add => "<OP_ADD>",
            Operator::Sub => "<OP_SUB>",
            Operator::Mul => "<OP_MUL>",
            Operator::Div => "<OP_DIV>",
            Operator::Pow => "<OP_POW>",
        }
    }

    /// Parse operator from character
    pub fn from_char(ch: char) -> Option<Self> {
        match ch {
            '+' => Some(Operator::Add),
            '-' => Some(Operator::Sub),
            '*' | '×' => Some(Operator::Mul),
            '/' | '÷' => Some(Operator::Div),
            '^' => Some(Operator::Pow),
            _ => None,
        }
    }
}

/// Preprocess text: detect numbers and operators, replace with placeholders
///
/// Examples:
/// - "What is 2 + 3?" → "What is <MATH_0> <OP_ADD> <MATH_1>?"
/// - "Calculate 15.5 * 2" → "Calculate <MATH_0> <OP_MUL> <MATH_1>"
/// - "Add 1/4 and 3/8" → "Add <MATH_0> and <MATH_1>"
pub fn preprocess(text: &str) -> Result<PreprocessResult> {
    let mut result = String::new();
    let mut numbers = Vec::new();
    let mut operators = Vec::new();

    // Regex patterns for different number formats
    let integer_re = Regex::new(r"^\d+").unwrap();
    let decimal_re = Regex::new(r"^\d+\.\d+").unwrap();
    let fraction_re = Regex::new(r"^(\d+)/(\d+)").unwrap();

    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        // Check for operators first
        if let Some(op) = Operator::from_char(ch) {
            operators.push(op);
            result.push(' ');
            result.push_str(op.to_token());
            result.push(' ');
            continue;
        }

        // Check for numbers
        if ch.is_ascii_digit() {
            // Collect remaining characters for regex matching
            let mut num_str = String::new();
            num_str.push(ch);

            // Peek ahead to build complete number string
            while let Some(&next_ch) = chars.peek() {
                if next_ch.is_ascii_digit() || next_ch == '.' || next_ch == '/' {
                    num_str.push(chars.next().unwrap());
                } else {
                    break;
                }
            }

            // Parse number based on format
            let value = if let Some(captures) = fraction_re.captures(&num_str) {
                // Fraction: numerator / denominator
                let num = captures.get(1).unwrap().as_str().parse::<u8>()
                    .map_err(|_| VeritasError::InvalidInput("Invalid fraction numerator".to_string()))?;
                let den = captures.get(2).unwrap().as_str().parse::<u8>()
                    .map_err(|_| VeritasError::InvalidInput("Invalid fraction denominator".to_string()))?;

                if den == 0 {
                    return Err(VeritasError::DivisionByZero);
                }

                ScalarF4E4::from(num) / ScalarF4E4::from(den)
            } else if decimal_re.is_match(&num_str) {
                // Decimal: convert to Spirix
                parse_decimal(&num_str)?
            } else if integer_re.is_match(&num_str) {
                // Integer
                let val = num_str.parse::<u8>()
                    .map_err(|_| VeritasError::InvalidInput(format!("Invalid integer: {}", num_str)))?;
                ScalarF4E4::from(val)
            } else {
                return Err(VeritasError::InvalidInput(format!("Unparseable number: {}", num_str)));
            };

            // Add placeholder with sequential numbering
            let idx = numbers.len();
            numbers.push(value);
            result.push_str(&format!("<MATH_{}>", idx));
            continue;
        }

        // Regular character - pass through
        result.push(ch);
    }

    Ok(PreprocessResult {
        text: result,
        numbers,
        operators,
    })
}

/// Parse decimal string to Spirix scalar
///
/// Example: "15.75" → ScalarF4E4
fn parse_decimal(s: &str) -> Result<ScalarF4E4> {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() != 2 {
        return Err(VeritasError::InvalidInput(format!("Invalid decimal: {}", s)));
    }

    let integer_part = parts[0].parse::<u8>()
        .map_err(|_| VeritasError::InvalidInput(format!("Invalid decimal integer part: {}", parts[0])))?;
    let fractional_part = parts[1].parse::<u8>()
        .map_err(|_| VeritasError::InvalidInput(format!("Invalid decimal fractional part: {}", parts[1])))?;

    // Compute: integer + fractional / 10^(num_digits)
    let num_digits = parts[1].len() as u32;
    let divisor = 10u8.pow(num_digits);

    let integer = ScalarF4E4::from(integer_part);
    let fractional = ScalarF4E4::from(fractional_part) / ScalarF4E4::from(divisor);

    Ok(integer + fractional)
}

/// Inject result back into preprocessed text
///
/// Replaces placeholders with computed results
pub fn inject_result(preprocessed: &str, results: &[ScalarF4E4]) -> String {
    let mut output = preprocessed.to_string();

    // Replace placeholders in reverse order (to avoid index shifting)
    for (idx, result) in results.iter().enumerate().rev() {
        let placeholder = format!("<MATH_{}>", idx);
        let result_str = format!("{:?}", result); // Spirix debug format
        output = output.replace(&placeholder, &result_str);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_addition() {
        let result = preprocess("What is 2 + 3?").unwrap();
        assert_eq!(result.text, "What is <MATH_0> <OP_ADD> <MATH_1>?");
        assert_eq!(result.numbers.len(), 2);
        assert_eq!(result.operators.len(), 1);
    }

    #[test]
    fn test_decimal() {
        let result = preprocess("Calculate 15.5 * 2").unwrap();
        assert_eq!(result.text, "Calculate <MATH_0> <OP_MUL> <MATH_1>");
        assert_eq!(result.numbers.len(), 2);
    }

    #[test]
    fn test_fraction() {
        let result = preprocess("Add 1/4 and 3/8").unwrap();
        assert_eq!(result.text, "Add <MATH_0> and <MATH_1>");
        assert_eq!(result.numbers.len(), 2);
    }

    #[test]
    fn test_sequential_numbering() {
        let result = preprocess("1 + 2 + 3 + 4").unwrap();
        assert_eq!(result.text, "<MATH_0> <OP_ADD> <MATH_1> <OP_ADD> <MATH_2> <OP_ADD> <MATH_3>");
        assert_eq!(result.numbers.len(), 4);
        assert_eq!(result.operators.len(), 3);
    }

    #[test]
    fn test_inject_result() {
        let preprocessed = "The answer is <MATH_0>";
        let results = vec![ScalarF4E4::from(42u8)];
        let output = inject_result(preprocessed, &results);
        assert!(output.contains("42"));
    }
}
