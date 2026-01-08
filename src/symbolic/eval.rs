//! Expression evaluation
//!
//! Evaluates symbolic expressions to numeric values using Spirix.

use super::context::Value;
use super::{Context, Expr};
use crate::error::{Result, VeritasError};
use crate::numeric::{Circle, Scalar};

/// Trait for evaluating expressions
pub trait Evaluate {
    /// Evaluate expression in given context
    fn evaluate(&self, ctx: &Context) -> Result<Value>;

    /// Evaluate to scalar (error if result is complex)
    fn evaluate_scalar(&self, ctx: &Context) -> Result<Scalar>;

    /// Evaluate to circle (converts scalar if needed)
    fn evaluate_circle(&self, ctx: &Context) -> Result<Circle>;
}

impl Evaluate for Expr {
    fn evaluate(&self, ctx: &Context) -> Result<Value> {
        match self {
            // Atomic values
            Expr::Number(n) => Ok(Value::Scalar(*n)),
            Expr::Complex(c) => Ok(Value::Circle(*c)),

            Expr::Variable(name) => Ok(ctx.get(name)?.clone()),

            Expr::Constant(name) => match name.as_str() {
                "π" | "pi" => Ok(Value::Scalar(Scalar::PI)),
                "e" => Ok(Value::Scalar(Scalar::E)),
                "i" => Ok(Value::Circle(Circle::I)),
                _ => Err(VeritasError::VariableNotFound(format!(
                    "Unknown constant: {}",
                    name
                ))),
            },

            // Binary operations - try scalar first, fallback to circle
            Expr::Add(a, b) => {
                let a_val = a.evaluate(ctx)?;
                let b_val = b.evaluate(ctx)?;

                match (a_val, b_val) {
                    (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.checked_add(b)?)),
                    (Value::Circle(a), Value::Circle(b)) => Ok(Value::Circle(a.checked_add(b)?)),
                    (Value::Scalar(a), Value::Circle(b)) | (Value::Circle(b), Value::Scalar(a)) => {
                        Ok(Value::Circle(Circle::from(a).checked_add(b)?))
                    }
                }
            }

            Expr::Sub(a, b) => {
                let a_val = a.evaluate(ctx)?;
                let b_val = b.evaluate(ctx)?;

                match (a_val, b_val) {
                    (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.checked_sub(b)?)),
                    (Value::Circle(a), Value::Circle(b)) => Ok(Value::Circle(a.checked_sub(b)?)),
                    (Value::Scalar(a), Value::Circle(b)) => {
                        Ok(Value::Circle(Circle::from(a).checked_sub(b)?))
                    }
                    (Value::Circle(a), Value::Scalar(b)) => {
                        Ok(Value::Circle(a.checked_sub(Circle::from(b))?))
                    }
                }
            }

            Expr::Mul(a, b) => {
                let a_val = a.evaluate(ctx)?;
                let b_val = b.evaluate(ctx)?;

                match (a_val, b_val) {
                    (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.checked_mul(b)?)),
                    (Value::Circle(a), Value::Circle(b)) => Ok(Value::Circle(a.checked_mul(b)?)),
                    (Value::Scalar(a), Value::Circle(b)) | (Value::Circle(b), Value::Scalar(a)) => {
                        Ok(Value::Circle(Circle::from(a).checked_mul(b)?))
                    }
                }
            }

            Expr::Div(a, b) => {
                let a_val = a.evaluate(ctx)?;
                let b_val = b.evaluate(ctx)?;

                match (a_val, b_val) {
                    (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a.checked_div(b)?)),
                    (Value::Circle(a), Value::Circle(b)) => Ok(Value::Circle(a.checked_div(b)?)),
                    (Value::Scalar(a), Value::Circle(b)) => {
                        Ok(Value::Circle(Circle::from(a).checked_div(b)?))
                    }
                    (Value::Circle(a), Value::Scalar(b)) => {
                        Ok(Value::Circle(a.checked_div(Circle::from(b))?))
                    }
                }
            }

            Expr::Pow(base, exp) => {
                let base_val = base.evaluate_scalar(ctx)?;
                let exp_val = exp.evaluate_scalar(ctx)?;
                Ok(Value::Scalar(base_val.pow(exp_val)?))
            }

            // Unary operations
            Expr::Neg(a) => match a.evaluate(ctx)? {
                Value::Scalar(s) => Ok(Value::Scalar(-s)),
                Value::Circle(c) => Ok(Value::Circle(-c)),
            },

            Expr::Sqrt(a) => match a.evaluate(ctx)? {
                Value::Scalar(s) => {
                    // sqrt of negative returns complex
                    if s.inner() < spirix::ScalarF6E5::ZERO {
                        let c = Circle::from(s);
                        Ok(Value::Circle(c.sqrt()?))
                    } else {
                        Ok(Value::Scalar(s.sqrt()?))
                    }
                }
                Value::Circle(c) => Ok(Value::Circle(c.sqrt()?)),
            },

            Expr::Ln(a) => {
                let val = a.evaluate_scalar(ctx)?;
                Ok(Value::Scalar(val.ln()?))
            }

            Expr::Exp(a) => match a.evaluate(ctx)? {
                Value::Scalar(s) => Ok(Value::Scalar(s.exp()?)),
                Value::Circle(c) => Ok(Value::Circle(c.exp()?)),
            },

            Expr::Sin(a) => {
                let val = a.evaluate_scalar(ctx)?;
                Ok(Value::Scalar(val.sin()?))
            }

            Expr::Cos(a) => {
                let val = a.evaluate_scalar(ctx)?;
                Ok(Value::Scalar(val.cos()?))
            }

            Expr::Tan(a) => {
                let val = a.evaluate_scalar(ctx)?;
                let sin = val.sin()?;
                let cos = val.cos()?;
                Ok(Value::Scalar(sin.checked_div(cos)?))
            }

            Expr::Function(name, args) => Err(VeritasError::SimplificationError(format!(
                "Unknown function: {}",
                name
            ))),
        }
    }

    fn evaluate_scalar(&self, ctx: &Context) -> Result<Scalar> {
        match self.evaluate(ctx)? {
            Value::Scalar(s) => Ok(s),
            Value::Circle(_) => Err(VeritasError::SimplificationError(
                "Expression evaluates to complex number, not scalar".to_string(),
            )),
        }
    }

    fn evaluate_circle(&self, ctx: &Context) -> Result<Circle> {
        match self.evaluate(ctx)? {
            Value::Scalar(s) => Ok(Circle::from(s)),
            Value::Circle(c) => Ok(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_constant() {
        let expr = Expr::number(42);
        let ctx = Context::new();

        let result = expr.evaluate_scalar(&ctx).unwrap();
        assert_eq!(result, Scalar::from(42));
    }

    #[test]
    fn test_eval_variable() {
        let expr = Expr::var("x");
        let mut ctx = Context::new();
        ctx.bind("x", 42);

        let result = expr.evaluate_scalar(&ctx).unwrap();
        assert_eq!(result, Scalar::from(42));
    }

    #[test]
    fn test_eval_arithmetic() {
        // (x + 2) * 3
        let x = Expr::var("x");
        let two = Expr::number(2);
        let three = Expr::number(3);
        let expr = Expr::mul(Expr::add(x, two), three);

        let mut ctx = Context::new();
        ctx.bind("x", 5);

        let result = expr.evaluate_scalar(&ctx).unwrap();
        assert_eq!(result, Scalar::from(21)); // (5 + 2) * 3 = 21
    }

    #[test]
    fn test_eval_sqrt_negative() {
        // sqrt(-1) should return complex
        let expr = Expr::sqrt(Expr::number(-1));
        let ctx = Context::new();

        let result = expr.evaluate(&ctx).unwrap();
        match result {
            Value::Circle(c) => {
                assert_eq!(c.real(), Scalar::ZERO);
                assert_eq!(c.imag(), Scalar::ONE);
            }
            _ => panic!("Expected complex result"),
        }
    }
}
