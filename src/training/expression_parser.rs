//! Expression Parser for Basecalc Routing
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix arithmetic
//! ✓ Base-aware parsing
//!
//! Parses math expressions with explicit base markers for symbolic routing.

use spirix::ScalarF4E4;

/// Mathematical operation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add,
    Sub,
    Mul,
    Div,
}

/// Parsed mathematical expression
#[derive(Debug, Clone)]
pub struct ParsedExpression {
    pub base: u8,
    pub operand_a: u8,
    pub operand_b: u8,
    pub operation: Operation,
}

/// Parse math expression with base marker
///
/// Format: "dozenal: A + B = " → base=12, a=10, b=11, op=Add
///         "octal: 7 + 3 = "   → base=8, a=7, b=3, op=Add
pub fn parse_math_expression(input: &[u8]) -> Option<ParsedExpression> {
    let text = String::from_utf8_lossy(input);

    // Check for base markers
    let base = if text.starts_with("dozenal:") {
        12
    } else if text.starts_with("octal:") {
        8
    } else if text.starts_with("binary:") {
        2
    } else if text.starts_with("decimal:") {
        10
    } else {
        return None;
    };

    // Find the colon and skip past it
    let colon_pos = text.find(':')?;
    let math_part = text[colon_pos + 1..].trim();

    // Parse "A + B =" or "7 + 3 ="
    let parts: Vec<&str> = math_part.split_whitespace().collect();
    if parts.len() < 3 {
        return None;
    }

    let operand_a = parse_digit_in_base(parts[0], base)?;
    let operation = parse_operation(parts[1])?;
    let operand_b = parse_digit_in_base(parts[2], base)?;

    Some(ParsedExpression {
        base,
        operand_a,
        operand_b,
        operation,
    })
}

/// Parse digit string in given base
///
/// Examples: "7" in base 12 → 7
///           "A" in base 12 → 10
///           "B" in base 12 → 11
fn parse_digit_in_base(s: &str, base: u8) -> Option<u8> {
    u8::from_str_radix(s, base as u32).ok()
}

/// Parse operation symbol
fn parse_operation(s: &str) -> Option<Operation> {
    match s {
        "+" => Some(Operation::Add),
        "-" => Some(Operation::Sub),
        "×" | "*" => Some(Operation::Mul),
        "÷" | "/" => Some(Operation::Div),
        _ => None,
    }
}

/// Call basecalc (pure Spirix symbolic computation)
///
/// This is the ground truth - always correct, no learning needed.
/// Network learns WHEN to route here, not HOW to compute.
pub fn call_basecalc(expr: &ParsedExpression) -> ScalarF4E4 {
    let a = ScalarF4E4::from(expr.operand_a);
    let b = ScalarF4E4::from(expr.operand_b);

    // Pure Spirix arithmetic - symbolic ground truth
    match expr.operation {
        Operation::Add => a + b,
        Operation::Sub => a - b,
        Operation::Mul => a * b,
        Operation::Div => a / b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_dozenal() {
        let input = b"dozenal: A + B = ";
        let parsed = parse_math_expression(input).unwrap();

        assert_eq!(parsed.base, 12);
        assert_eq!(parsed.operand_a, 10);
        assert_eq!(parsed.operand_b, 11);
        assert_eq!(parsed.operation, Operation::Add);
    }

    #[test]
    fn test_parse_octal() {
        let input = b"octal: 7 + 3 = ";
        let parsed = parse_math_expression(input).unwrap();

        assert_eq!(parsed.base, 8);
        assert_eq!(parsed.operand_a, 7);
        assert_eq!(parsed.operand_b, 3);
        assert_eq!(parsed.operation, Operation::Add);
    }

    #[test]
    fn test_basecalc_dozenal() {
        let expr = ParsedExpression {
            base: 12,
            operand_a: 10,  // A
            operand_b: 11,  // B
            operation: Operation::Add,
        };

        let result = call_basecalc(&expr);
        let expected = ScalarF4E4::from(21u8);  // A + B = 21 (decimal)

        assert_eq!(result, expected);
    }

    #[test]
    fn test_no_base_marker() {
        let input = b"The weather is nice";
        let parsed = parse_math_expression(input);

        assert!(parsed.is_none(), "Should not parse text as math");
    }
}
