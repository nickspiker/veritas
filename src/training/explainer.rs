//! Neural explanation system (stub)
//!
//! In production, this would be a trained neural network.
//! For demonstration, we use rule-based explanations that can be
//! intentionally wrong to show verification catching errors.

use crate::symbolic::Expr;
use crate::numeric::Scalar;
use crate::error::Result;

/// Generates natural language explanations for solutions
pub struct NeuralExplainer {
    /// Intentionally introduce errors for demonstration
    introduce_errors: bool,
    explanations_generated: usize,
}

impl NeuralExplainer {
    pub fn new(introduce_errors: bool) -> Self {
        NeuralExplainer {
            introduce_errors,
            explanations_generated: 0,
        }
    }

    /// Generate explanation for how to solve the problem
    ///
    /// In production: Neural network generates step-by-step explanation
    /// For demo: Rule-based with intentional errors
    pub fn explain(&mut self, problem: &Expr) -> (String, Scalar) {
        self.explanations_generated += 1;

        // Intentionally get some wrong for demonstration
        let should_err = self.introduce_errors && (self.explanations_generated % 3 == 0);

        match problem {
            Expr::Add(a, b) => self.explain_addition(a, b, should_err),
            Expr::Mul(a, b) => self.explain_multiplication(a, b, should_err),
            Expr::Pow(base, exp) => self.explain_power(base, exp, should_err),
            Expr::Sqrt(inner) => self.explain_sqrt(inner, should_err),
            _ => (
                "I don't know how to explain this yet.".to_string(),
                Scalar::ZERO
            ),
        }
    }

    fn explain_addition(&self, a: &Expr, b: &Expr, introduce_error: bool) -> (String, Scalar) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (a, b) {
            let correct_answer = *n1 + *n2;

            if introduce_error {
                // Intentionally wrong - off by one
                let wrong_answer = correct_answer + Scalar::from(1);
                return (
                    format!("To add {} and {}, we sum them getting {} (oops, off by one!)",
                        n1, n2, wrong_answer),
                    wrong_answer
                );
            }

            return (
                format!("To add {} and {}, we sum them to get {}", n1, n2, correct_answer),
                correct_answer
            );
        }

        ("Complex addition".to_string(), Scalar::ZERO)
    }

    fn explain_multiplication(&self, a: &Expr, b: &Expr, introduce_error: bool) -> (String, Scalar) {
        if let (Expr::Number(n1), Expr::Number(n2)) = (a, b) {
            let correct_answer = *n1 * *n2;

            if introduce_error {
                // Intentionally use addition instead of multiplication
                let wrong_answer = *n1 + *n2;
                return (
                    format!("To multiply {} and {}, we add them getting {} (wrong operation!)",
                        n1, n2, wrong_answer),
                    wrong_answer
                );
            }

            return (
                format!("To multiply {} and {}, we get {}", n1, n2, correct_answer),
                correct_answer
            );
        }

        ("Complex multiplication".to_string(), Scalar::ZERO)
    }

    fn explain_power(&self, base: &Expr, exp: &Expr, introduce_error: bool) -> (String, Scalar) {
        if let (Expr::Number(b), Expr::Number(e)) = (base, exp) {
            let correct_answer = b.pow(*e).unwrap_or(Scalar::ZERO);

            if introduce_error {
                // Intentionally use multiplication instead of power
                let wrong_answer = *b * *e;
                return (
                    format!("To raise {} to power {}, we multiply them getting {} (should be exponentiation!)",
                        b, e, wrong_answer),
                    wrong_answer
                );
            }

            return (
                format!("To raise {} to power {}, we get {}", b, e, correct_answer),
                correct_answer
            );
        }

        ("Complex power".to_string(), Scalar::ZERO)
    }

    fn explain_sqrt(&self, inner: &Expr, introduce_error: bool) -> (String, Scalar) {
        if let Expr::Number(n) = inner {
            let correct_answer = n.sqrt().unwrap_or(Scalar::ZERO);

            if introduce_error {
                // Intentionally divide by 2 instead of sqrt
                let wrong_answer = *n / Scalar::from(2);
                return (
                    format!("The square root of {} is {} (wrong: divided by 2 instead of sqrt!)",
                        n, wrong_answer),
                    wrong_answer
                );
            }

            return (
                format!("The square root of {} is {}", n, correct_answer),
                correct_answer
            );
        }

        ("Complex sqrt".to_string(), Scalar::ZERO)
    }
}
