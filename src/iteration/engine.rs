//! Iteration Engine - The core z² + c loop
//!
//! This is the heart of Veritas:
//! - Iterate z := z² + c until converged or escaped
//! - Pure Spirix arithmetic (no IEEE violations)
//! - Returns BOUNDED (truth) or ESCAPED (farktogle)

use crate::autograd::{Tensor, Shape};
use crate::error::{Result, VeritasError};
use spirix::ScalarF4E4;
use super::convergence::{ConvergenceDetector, ConvergenceConfig};

/// Result of iteration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterationResult {
    /// State converged (bounded) - this is truth
    Converged { iterations: usize },

    /// State escaped (diverged) - this is farktogle
    Escaped { iterations: usize },

    /// Maximum iterations reached without convergence or escape
    MaxIterations { iterations: usize },
}

/// Current state of iteration
#[derive(Debug, Clone)]
pub struct IterationState {
    /// Current z value
    pub z: Tensor,

    /// Constant c (the input/query)
    pub c: Tensor,

    /// Current iteration number
    pub iteration: usize,
}

/// Iteration engine - performs z² + c until convergence/escape
pub struct IterationEngine {
    detector: ConvergenceDetector,
}

impl IterationEngine {
    /// Create new iteration engine with default convergence config
    pub fn new() -> Self {
        IterationEngine {
            detector: ConvergenceDetector::new(),
        }
    }

    /// Create iteration engine with custom convergence config
    pub fn with_config(config: ConvergenceConfig) -> Self {
        IterationEngine {
            detector: ConvergenceDetector::with_config(config),
        }
    }

    /// Perform one iteration step: z := z² + c
    ///
    /// For scalar tensors, this is element-wise:
    /// z[i] := z[i]² + c[i]
    pub fn step(&self, z: &Tensor, c: &Tensor) -> Result<Tensor> {
        // Get scalar data
        let z_data = z.as_scalars().ok_or_else(|| {
            VeritasError::InvalidInput("z must be scalar tensor".to_string())
        })?;

        let c_data = c.as_scalars().ok_or_else(|| {
            VeritasError::InvalidInput("c must be scalar tensor".to_string())
        })?;

        if z_data.len() != c_data.len() {
            return Err(VeritasError::InvalidInput(
                "z and c must have same length".to_string()
            ));
        }

        // Compute z² + c element-wise
        let new_data: Vec<ScalarF4E4> = z_data
            .iter()
            .zip(c_data.iter())
            .map(|(&z_val, &c_val)| (z_val * z_val) + c_val)
            .collect();

        // Create new tensor with same shape
        Tensor::from_scalars(new_data, z.shape().clone())
    }

    /// Compute magnitude of tensor (max absolute value of elements)
    fn magnitude(&self, z: &Tensor) -> Result<ScalarF4E4> {
        let data = z.as_scalars().ok_or_else(|| {
            VeritasError::InvalidInput("z must be scalar tensor".to_string())
        })?;

        let mut max_mag = ScalarF4E4::ZERO;
        for &val in data {
            let mag = val.magnitude();
            if mag > max_mag {
                max_mag = mag;
            }
        }

        Ok(max_mag)
    }

    /// Compute change between two states (max absolute difference)
    fn change(&self, z_new: &Tensor, z_old: &Tensor) -> Result<ScalarF4E4> {
        let new_data = z_new.as_scalars().ok_or_else(|| {
            VeritasError::InvalidInput("z_new must be scalar tensor".to_string())
        })?;

        let old_data = z_old.as_scalars().ok_or_else(|| {
            VeritasError::InvalidInput("z_old must be scalar tensor".to_string())
        })?;

        if new_data.len() != old_data.len() {
            return Err(VeritasError::InvalidInput(
                "z_new and z_old must have same length".to_string()
            ));
        }

        let mut max_change = ScalarF4E4::ZERO;
        for (&new_val, &old_val) in new_data.iter().zip(old_data.iter()) {
            let diff = (new_val - old_val).magnitude();
            if diff > max_change {
                max_change = diff;
            }
        }

        Ok(max_change)
    }

    /// Iterate z² + c until convergence or escape
    ///
    /// Returns:
    /// - Converged: State is bounded (truth)
    /// - Escaped: State diverged (farktogle)
    /// - MaxIterations: Neither converged nor escaped within iteration limit
    pub fn iterate(&mut self, initial_z: Tensor, c: Tensor) -> Result<(IterationResult, Tensor)> {
        self.detector.reset();

        let mut z_current = initial_z;

        loop {
            self.detector.tick();

            // Perform one iteration: z := z² + c
            let z_new = self.step(&z_current, &c)?;

            // Check for escape (divergence)
            let magnitude = self.magnitude(&z_new)?;
            if self.detector.has_escaped(magnitude) {
                return Ok((
                    IterationResult::Escaped {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            // Check for convergence (stability)
            let change = self.change(&z_new, &z_current)?;
            if self.detector.has_converged(change) {
                return Ok((
                    IterationResult::Converged {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            // Check max iterations
            if self.detector.is_max_iterations() {
                return Ok((
                    IterationResult::MaxIterations {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            z_current = z_new;
        }
    }

    /// Iterate with progress callback
    ///
    /// Callback receives (iteration, z, magnitude, change)
    pub fn iterate_with_progress<F>(
        &mut self,
        initial_z: Tensor,
        c: Tensor,
        mut callback: F,
    ) -> Result<(IterationResult, Tensor)>
    where
        F: FnMut(usize, &Tensor, ScalarF4E4, ScalarF4E4),
    {
        self.detector.reset();

        let mut z_current = initial_z.clone();

        loop {
            self.detector.tick();

            // Perform one iteration
            let z_new = self.step(&z_current, &c)?;

            // Compute metrics
            let magnitude = self.magnitude(&z_new)?;
            let change = self.change(&z_new, &z_current)?;

            // Call progress callback
            callback(self.detector.iterations(), &z_new, magnitude, change);

            // Check for escape
            if self.detector.has_escaped(magnitude) {
                return Ok((
                    IterationResult::Escaped {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            // Check for convergence
            if self.detector.has_converged(change) {
                return Ok((
                    IterationResult::Converged {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            // Check max iterations
            if self.detector.is_max_iterations() {
                return Ok((
                    IterationResult::MaxIterations {
                        iterations: self.detector.iterations(),
                    },
                    z_new,
                ));
            }

            z_current = z_new;
        }
    }
}

impl Default for IterationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_scalar_convergence() {
        let mut engine = IterationEngine::new();

        // z² + c where c = 0, z starts at 0.5
        // Should converge to 0 (0.5² = 0.25, 0.25² = 0.0625, ...)
        let z = Tensor::from_scalars(
            vec![ScalarF4E4::ONE >> 1], // 0.5
            Shape::vector(1)
        ).unwrap();

        let c = Tensor::from_scalars(
            vec![ScalarF4E4::ZERO],
            Shape::vector(1)
        ).unwrap();

        let (result, final_z) = engine.iterate(z, c).unwrap();

        assert!(matches!(result, IterationResult::Converged { .. }));

        let final_val = final_z.as_scalars().unwrap()[0];
        // Should be very close to zero
        assert!(final_val.magnitude() < (ScalarF4E4::ONE / ScalarF4E4::from(100u8)));
    }

    #[test]
    fn test_single_scalar_escape() {
        let mut engine = IterationEngine::new();

        // z² + c where c = 1, z starts at 1
        // Should escape: 1² + 1 = 2, 2² + 1 = 5, ...
        let z = Tensor::from_scalars(
            vec![ScalarF4E4::ONE],
            Shape::vector(1)
        ).unwrap();

        let c = Tensor::from_scalars(
            vec![ScalarF4E4::ONE],
            Shape::vector(1)
        ).unwrap();

        let (result, _) = engine.iterate(z, c).unwrap();

        assert!(matches!(result, IterationResult::Escaped { .. }));
    }

    #[test]
    fn test_multiple_elements() {
        let mut engine = IterationEngine::new();

        // Vector of 3 elements: [0.5, 0.3, 0.1] with c = [0, 0, 0]
        // All should converge to 0
        let z = Tensor::from_scalars(
            vec![
                ScalarF4E4::ONE >> 1, // 0.5
                ScalarF4E4::from(3u8) / ScalarF4E4::from(10u8), // 0.3
                ScalarF4E4::ONE / ScalarF4E4::from(10u8), // 0.1
            ],
            Shape::vector(3)
        ).unwrap();

        let c = Tensor::from_scalars(
            vec![ScalarF4E4::ZERO, ScalarF4E4::ZERO, ScalarF4E4::ZERO],
            Shape::vector(3)
        ).unwrap();

        let (result, final_z) = engine.iterate(z, c).unwrap();

        assert!(matches!(result, IterationResult::Converged { .. }));

        // All values should be near zero
        for &val in final_z.as_scalars().unwrap() {
            assert!(val.magnitude() < (ScalarF4E4::ONE / ScalarF4E4::from(100u8)));
        }
    }
}
