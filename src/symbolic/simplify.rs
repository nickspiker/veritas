//! Expression simplification
//!
//! Algebraic simplification preserving mathematical equivalence.

use super::Expr;
use crate::error::Result;
use crate::numeric::Scalar;

/// Trait for simplifying expressions
pub trait Simplify {
    /// Simplify expression algebraically
    fn simplify(&self) -> Result<Expr>;
}

impl Simplify for Expr {
    fn simplify(&self) -> Result<Expr> {
        self.check_complexity(1000)?;

        let simplified = match self {
            // Atomic expressions are already simple
            Expr::Number(_) | Expr::Complex(_) | Expr::Variable(_) | Expr::Constant(_) => {
                self.clone()
            }

            // Addition simplification
            Expr::Add(a, b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;

                match (&a, &b) {
                    // 0 + x = x
                    (Expr::Number(n), _) if n.is_zero() => b,
                    // x + 0 = x
                    (_, Expr::Number(n)) if n.is_zero() => a,
                    // Constant folding: n1 + n2
                    (Expr::Number(n1), Expr::Number(n2)) => Expr::Number((*n1 + *n2)),
                    _ => Expr::add(a, b),
                }
            }

            // Subtraction simplification
            Expr::Sub(a, b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;

                match (&a, &b) {
                    // x - 0 = x
                    (_, Expr::Number(n)) if n.is_zero() => a,
                    // x - x = 0
                    _ if a == b => Expr::Number(Scalar::ZERO),
                    // Constant folding
                    (Expr::Number(n1), Expr::Number(n2)) => Expr::Number((*n1 - *n2)),
                    _ => Expr::sub(a, b),
                }
            }

            // Multiplication simplification
            Expr::Mul(a, b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;

                match (&a, &b) {
                    // 0 * x = 0
                    (Expr::Number(n), _) | (_, Expr::Number(n)) if n.is_zero() => {
                        Expr::Number(Scalar::ZERO)
                    }
                    // 1 * x = x
                    (Expr::Number(n), _) if *n == Scalar::ONE => b,
                    // x * 1 = x
                    (_, Expr::Number(n)) if *n == Scalar::ONE => a,
                    // Constant folding
                    (Expr::Number(n1), Expr::Number(n2)) => Expr::Number((*n1 * *n2)),
                    _ => Expr::mul(a, b),
                }
            }

            // Division simplification
            Expr::Div(a, b) => {
                let a = a.simplify()?;
                let b = b.simplify()?;

                match (&a, &b) {
                    // 0 / x = 0 (x ≠ 0)
                    (Expr::Number(n), _) if n.is_zero() => Expr::Number(Scalar::ZERO),
                    // x / 1 = x
                    (_, Expr::Number(n)) if *n == Scalar::ONE => a,
                    // x / x = 1
                    _ if a == b => Expr::Number(Scalar::ONE),
                    // Constant folding
                    (Expr::Number(n1), Expr::Number(n2)) => {
                        if n2.is_zero() {
                            // Keep division by zero as-is for evaluation to catch
                            Expr::div(a, b)
                        } else {
                            Expr::Number((*n1 / *n2))
                        }
                    }
                    _ => Expr::div(a, b),
                }
            }

            // Power simplification
            Expr::Pow(base, exp) => {
                let base = base.simplify()?;
                let exp = exp.simplify()?;

                match (&base, &exp) {
                    // x ^ 0 = 1
                    (_, Expr::Number(n)) if n.is_zero() => Expr::Number(Scalar::ONE),
                    // x ^ 1 = x
                    (_, Expr::Number(n)) if *n == Scalar::ONE => base,
                    // 0 ^ x = 0 (x > 0)
                    (Expr::Number(n), _) if n.is_zero() => Expr::Number(Scalar::ZERO),
                    // 1 ^ x = 1
                    (Expr::Number(n), _) if *n == Scalar::ONE => Expr::Number(Scalar::ONE),
                    // Constant folding (careful with errors)
                    (Expr::Number(b), Expr::Number(e)) => {
                        // Only fold if it won't error
                        if let Ok(result) = b.pow(*e) {
                            Expr::Number(result)
                        } else {
                            Expr::pow(base, exp)
                        }
                    }
                    _ => Expr::pow(base, exp),
                }
            }

            // Negation simplification
            Expr::Neg(a) => {
                let a = a.simplify()?;

                match &a {
                    // -(-x) = x
                    Expr::Neg(inner) => (**inner).clone(),
                    // -(n) = -n
                    Expr::Number(n) => Expr::Number(-*n),
                    _ => Expr::neg(a),
                }
            }

            // Square root simplification
            Expr::Sqrt(a) => {
                let a = a.simplify()?;

                match &a {
                    // sqrt(0) = 0
                    Expr::Number(n) if n.is_zero() => Expr::Number(Scalar::ZERO),
                    // sqrt(1) = 1
                    Expr::Number(n) if *n == Scalar::ONE => Expr::Number(Scalar::ONE),
                    // Constant folding (only if positive)
                    Expr::Number(n) if n.inner() > spirix::ScalarF6E5::ZERO => {
                        if let Ok(result) = n.sqrt() {
                            Expr::Number(result)
                        } else {
                            Expr::sqrt(a)
                        }
                    }
                    _ => Expr::sqrt(a),
                }
            }

            // Other operations - just simplify children
            Expr::Ln(a) => Expr::Ln(Box::new(a.simplify()?)),
            Expr::Exp(a) => Expr::Exp(Box::new(a.simplify()?)),
            Expr::Sin(a) => Expr::Sin(Box::new(a.simplify()?)),
            Expr::Cos(a) => Expr::Cos(Box::new(a.simplify()?)),
            Expr::Tan(a) => Expr::Tan(Box::new(a.simplify()?)),

            Expr::Function(name, args) => {
                let args: Result<Vec<_>> = args.iter().map(|arg| arg.simplify()).collect();
                Expr::Function(name.clone(), args?)
            }
        };

        Ok(simplified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simplify_add_zero() {
        let x = Expr::var("x");
        let zero = Expr::number(0);
        let expr = Expr::add(x.clone(), zero);

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_simplify_mul_one() {
        let x = Expr::var("x");
        let one = Expr::number(1);
        let expr = Expr::mul(x.clone(), one);

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_simplify_mul_zero() {
        let x = Expr::var("x");
        let zero = Expr::number(0);
        let expr = Expr::mul(x, zero);

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, Expr::number(0));
    }

    #[test]
    fn test_simplify_power_zero() {
        let x = Expr::var("x");
        let zero = Expr::number(0);
        let expr = Expr::pow(x, zero);

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, Expr::number(1));
    }

    #[test]
    fn test_simplify_double_negation() {
        let x = Expr::var("x");
        let expr = Expr::neg(Expr::neg(x.clone()));

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, x);
    }

    #[test]
    fn test_constant_folding() {
        // (2 + 3) * 4 should simplify to 20
        let two = Expr::number(2);
        let three = Expr::number(3);
        let four = Expr::number(4);
        let expr = Expr::mul(Expr::add(two, three), four);

        let simplified = expr.simplify().unwrap();
        assert_eq!(simplified, Expr::number(20));
    }
}
