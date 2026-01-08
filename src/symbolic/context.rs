//! Variable context for expression evaluation

use crate::error::{Result, VeritasError};
use crate::numeric::{Circle, Scalar};
use std::collections::HashMap;

/// Value that can be bound to a variable
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(Scalar),
    Circle(Circle),
}

impl From<Scalar> for Value {
    fn from(s: Scalar) -> Self {
        Value::Scalar(s)
    }
}

impl From<Circle> for Value {
    fn from(c: Circle) -> Self {
        Value::Circle(c)
    }
}

impl From<i32> for Value {
    fn from(i: i32) -> Self {
        Value::Scalar(Scalar::from(i))
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Scalar(Scalar::from(f))
    }
}

/// Context for expression evaluation
///
/// Maps variable names to numeric values
#[derive(Debug, Clone)]
pub struct Context {
    bindings: HashMap<String, Value>,
}

impl Context {
    /// Create empty context
    pub fn new() -> Self {
        Context {
            bindings: HashMap::new(),
        }
    }

    /// Bind a variable to a value
    pub fn bind(&mut self, name: impl Into<String>, value: impl Into<Value>) {
        self.bindings.insert(name.into(), value.into());
    }

    /// Get value of a variable
    pub fn get(&self, name: &str) -> Result<&Value> {
        self.bindings
            .get(name)
            .ok_or_else(|| VeritasError::VariableNotFound(name.to_string()))
    }

    /// Get scalar value (error if complex)
    pub fn get_scalar(&self, name: &str) -> Result<Scalar> {
        match self.get(name)? {
            Value::Scalar(s) => Ok(*s),
            Value::Circle(_) => Err(VeritasError::SimplificationError(format!(
                "Variable {} is complex, not scalar",
                name
            ))),
        }
    }

    /// Get circle value (converts scalar if needed)
    pub fn get_circle(&self, name: &str) -> Result<Circle> {
        match self.get(name)? {
            Value::Scalar(s) => Ok(Circle::from(*s)),
            Value::Circle(c) => Ok(*c),
        }
    }

    /// Check if variable is bound
    pub fn contains(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }

    /// Get all bound variables
    pub fn variables(&self) -> Vec<String> {
        self.bindings.keys().cloned().collect()
    }
}

impl Default for Context {
    fn default() -> Self {
        Context::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_binding() {
        let mut ctx = Context::new();
        ctx.bind("x", 42);
        ctx.bind("y", 3.14);

        assert!(ctx.contains("x"));
        assert!(ctx.contains("y"));
        assert!(!ctx.contains("z"));
    }

    #[test]
    fn test_get_scalar() {
        let mut ctx = Context::new();
        ctx.bind("x", 42);

        let val = ctx.get_scalar("x").unwrap();
        assert_eq!(val, Scalar::from(42));
    }

    #[test]
    fn test_variable_not_found() {
        let ctx = Context::new();
        let result = ctx.get("x");

        assert!(matches!(result, Err(VeritasError::VariableNotFound(_))));
    }
}
