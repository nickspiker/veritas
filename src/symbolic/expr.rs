//! Symbolic expression representation
//!
//! Expressions are immutable trees that can be:
//! - Simplified (algebraically)
//! - Evaluated (numerically)
//! - Differentiated (symbolically)
//! - Compared (structurally)

use crate::error::{Result, VeritasError};
use crate::numeric::{Circle, Scalar};
use std::fmt;

/// Symbolic expression
///
/// This is the core type for symbolic mathematics.
/// Expressions are immutable and can be freely cloned/shared.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    // Atomic expressions
    /// Real number value
    Number(Scalar),

    /// Complex number value
    Complex(Circle),

    /// Variable (e.g., "x", "y")
    Variable(String),

    /// Named constant (e.g., "π", "e")
    Constant(String),

    // Binary operations
    /// Addition: a + b
    Add(Box<Expr>, Box<Expr>),

    /// Subtraction: a - b
    Sub(Box<Expr>, Box<Expr>),

    /// Multiplication: a * b
    Mul(Box<Expr>, Box<Expr>),

    /// Division: a / b
    Div(Box<Expr>, Box<Expr>),

    /// Exponentiation: a ^ b
    Pow(Box<Expr>, Box<Expr>),

    // Unary operations
    /// Negation: -a
    Neg(Box<Expr>),

    /// Square root: √a
    Sqrt(Box<Expr>),

    /// Natural logarithm: ln(a)
    Ln(Box<Expr>),

    /// Exponential: e^a
    Exp(Box<Expr>),

    // Trigonometric
    /// Sine: sin(a)
    Sin(Box<Expr>),

    /// Cosine: cos(a)
    Cos(Box<Expr>),

    /// Tangent: tan(a)
    Tan(Box<Expr>),

    // Function application (general)
    /// Function call: f(args...)
    Function(String, Vec<Expr>),
}

impl Expr {
    // Constructors for convenience

    /// Create number expression
    pub fn number<T: Into<Scalar>>(value: T) -> Self {
        Expr::Number(value.into())
    }

    /// Create complex expression
    pub fn complex(real: Scalar, imag: Scalar) -> Self {
        Expr::Complex(Circle::from_parts(real, imag))
    }

    /// Create variable expression
    pub fn var(name: impl Into<String>) -> Self {
        Expr::Variable(name.into())
    }

    /// Create addition
    pub fn add(lhs: Expr, rhs: Expr) -> Self {
        Expr::Add(Box::new(lhs), Box::new(rhs))
    }

    /// Create subtraction
    pub fn sub(lhs: Expr, rhs: Expr) -> Self {
        Expr::Sub(Box::new(lhs), Box::new(rhs))
    }

    /// Create multiplication
    pub fn mul(lhs: Expr, rhs: Expr) -> Self {
        Expr::Mul(Box::new(lhs), Box::new(rhs))
    }

    /// Create division
    pub fn div(lhs: Expr, rhs: Expr) -> Self {
        Expr::Div(Box::new(lhs), Box::new(rhs))
    }

    /// Create power
    pub fn pow(base: Expr, exp: Expr) -> Self {
        Expr::Pow(Box::new(base), Box::new(exp))
    }

    /// Create negation
    pub fn neg(expr: Expr) -> Self {
        Expr::Neg(Box::new(expr))
    }

    /// Create square root
    pub fn sqrt(expr: Expr) -> Self {
        Expr::Sqrt(Box::new(expr))
    }

    // Query methods

    /// Check if expression is a constant (no variables)
    pub fn is_constant(&self) -> bool {
        match self {
            Expr::Number(_) | Expr::Complex(_) | Expr::Constant(_) => true,
            Expr::Variable(_) => false,
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => a.is_constant() && b.is_constant(),
            Expr::Neg(a)
            | Expr::Sqrt(a)
            | Expr::Ln(a)
            | Expr::Exp(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a) => a.is_constant(),
            Expr::Function(_, args) => args.iter().all(|arg| arg.is_constant()),
        }
    }

    /// Get all variables in expression
    pub fn variables(&self) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_variables(&mut vars);
        vars.sort();
        vars.dedup();
        vars
    }

    fn collect_variables(&self, vars: &mut Vec<String>) {
        match self {
            Expr::Variable(name) => vars.push(name.clone()),
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => {
                a.collect_variables(vars);
                b.collect_variables(vars);
            }
            Expr::Neg(a)
            | Expr::Sqrt(a)
            | Expr::Ln(a)
            | Expr::Exp(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a) => {
                a.collect_variables(vars);
            }
            Expr::Function(_, args) => {
                for arg in args {
                    arg.collect_variables(vars);
                }
            }
            _ => {}
        }
    }

    /// Calculate depth of expression tree
    pub fn depth(&self) -> usize {
        match self {
            Expr::Number(_) | Expr::Complex(_) | Expr::Variable(_) | Expr::Constant(_) => 1,
            Expr::Add(a, b)
            | Expr::Sub(a, b)
            | Expr::Mul(a, b)
            | Expr::Div(a, b)
            | Expr::Pow(a, b) => 1 + a.depth().max(b.depth()),
            Expr::Neg(a)
            | Expr::Sqrt(a)
            | Expr::Ln(a)
            | Expr::Exp(a)
            | Expr::Sin(a)
            | Expr::Cos(a)
            | Expr::Tan(a) => 1 + a.depth(),
            Expr::Function(_, args) => 1 + args.iter().map(|arg| arg.depth()).max().unwrap_or(0),
        }
    }

    /// Check complexity limit
    pub fn check_complexity(&self, limit: usize) -> Result<()> {
        let depth = self.depth();
        if depth > limit {
            Err(VeritasError::ComplexityLimit(depth))
        } else {
            Ok(())
        }
    }
}

// Display implementation for pretty printing
impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{}", n),
            Expr::Complex(c) => write!(f, "{}", c),
            Expr::Variable(v) => write!(f, "{}", v),
            Expr::Constant(c) => write!(f, "{}", c),

            Expr::Add(a, b) => write!(f, "({} + {})", a, b),
            Expr::Sub(a, b) => write!(f, "({} - {})", a, b),
            Expr::Mul(a, b) => write!(f, "({} * {})", a, b),
            Expr::Div(a, b) => write!(f, "({} / {})", a, b),
            Expr::Pow(a, b) => write!(f, "({} ^ {})", a, b),

            Expr::Neg(a) => write!(f, "(-{})", a),
            Expr::Sqrt(a) => write!(f, "√({})", a),
            Expr::Ln(a) => write!(f, "ln({})", a),
            Expr::Exp(a) => write!(f, "exp({})", a),
            Expr::Sin(a) => write!(f, "sin({})", a),
            Expr::Cos(a) => write!(f, "cos({})", a),
            Expr::Tan(a) => write!(f, "tan({})", a),

            Expr::Function(name, args) => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_construction() {
        let x = Expr::var("x");
        let two = Expr::number(2);
        let expr = Expr::add(x.clone(), two);

        assert_eq!(format!("{}", expr), "(x + 2)");
    }

    #[test]
    fn test_variables() {
        let x = Expr::var("x");
        let y = Expr::var("y");
        let expr = Expr::mul(x, y);

        let vars = expr.variables();
        assert_eq!(vars, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn test_is_constant() {
        let num = Expr::number(42);
        assert!(num.is_constant());

        let var = Expr::var("x");
        assert!(!var.is_constant());

        let expr = Expr::add(num.clone(), var);
        assert!(!expr.is_constant());
    }

    #[test]
    fn test_depth() {
        let x = Expr::var("x");
        assert_eq!(x.depth(), 1);

        let expr = Expr::add(x.clone(), Expr::number(2));
        assert_eq!(expr.depth(), 2);

        let nested = Expr::mul(expr.clone(), expr);
        assert_eq!(nested.depth(), 3);
    }
}
