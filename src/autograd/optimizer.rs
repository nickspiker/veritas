//! Optimizers for gradient descent
//!
//! All using verified Spirix arithmetic (no IEEE violations)

use super::tensor::Tensor;
use spirix::ScalarF4E4;
use crate::error::Result;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    /// Learning rate
    learning_rate: ScalarF4E4,
    /// Momentum coefficient (0.0 = no momentum)
    momentum: ScalarF4E4,
    /// Velocity terms for momentum
    velocities: Vec<Vec<ScalarF4E4>>,
}

impl SGD {
    /// Create new SGD optimizer
    pub fn new(learning_rate: ScalarF4E4) -> Self {
        SGD {
            learning_rate,
            momentum: ScalarF4E4::ZERO,
            velocities: Vec::new(),
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(learning_rate: ScalarF4E4, momentum: ScalarF4E4) -> Self {
        SGD {
            learning_rate,
            momentum,
            velocities: Vec::new(),
        }
    }

    /// Perform one optimization step
    ///
    /// Updates weights using: w = w - lr * grad
    /// With momentum: v = momentum * v - lr * grad; w = w + v
    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        // Initialize velocities if needed
        if self.velocities.is_empty() && self.momentum > ScalarF4E4::ZERO {
            for param in parameters.iter() {
                if let Some(data) = param.as_scalars() {
                    self.velocities.push(vec![ScalarF4E4::ZERO; data.len()]);
                }
            }
        }

        // Update each parameter
        for (i, param) in parameters.iter_mut().enumerate() {
            // Get gradient and copy it to avoid borrow checker issues
            let grad_data_copy: Vec<ScalarF4E4> = {
                let grad = param.grad().ok_or_else(|| {
                    crate::error::VeritasError::InvalidInput(
                        "Parameter has no gradient".to_string()
                    )
                })?;

                grad.as_scalars().ok_or_else(|| {
                    crate::error::VeritasError::InvalidInput(
                        "Gradient must be scalar tensor".to_string()
                    )
                })?.to_vec()
            };

            let param_data = param.as_scalars_mut().ok_or_else(|| {
                crate::error::VeritasError::InvalidInput(
                    "Parameter must be scalar tensor".to_string()
                )
            })?;

            if self.momentum > ScalarF4E4::ZERO && i < self.velocities.len() {
                // SGD with momentum
                let velocity = &mut self.velocities[i];

                for j in 0..param_data.len() {
                    // v = momentum * v - lr * grad
                    velocity[j] = self.momentum * velocity[j] - self.learning_rate * grad_data_copy[j];
                    // w = w + v
                    param_data[j] = param_data[j] + velocity[j];
                }
            } else {
                // Vanilla SGD: w = w - lr * grad
                for j in 0..param_data.len() {
                    param_data[j] = param_data[j] - self.learning_rate * grad_data_copy[j];
                }
            }
        }

        Ok(())
    }

    /// Zero all gradients
    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            param.zero_grad();
        }
    }

    /// Get current learning rate
    pub fn get_lr(&self) -> ScalarF4E4 {
        self.learning_rate
    }

    /// Set learning rate (for learning rate schedules)
    pub fn set_lr(&mut self, lr: ScalarF4E4) {
        self.learning_rate = lr;
    }
}

/// Adam optimizer (Adaptive Moment Estimation)
pub struct Adam {
    learning_rate: ScalarF4E4,
    beta1: ScalarF4E4,  // Exponential decay for first moment
    beta2: ScalarF4E4,  // Exponential decay for second moment
    epsilon: ScalarF4E4,  // Small constant for numerical stability

    /// First moment estimates (mean of gradients)
    m: Vec<Vec<ScalarF4E4>>,
    /// Second moment estimates (uncentered variance of gradients)
    v: Vec<Vec<ScalarF4E4>>,
    /// Time step
    t: usize,
}

impl Adam {
    /// Create new Adam optimizer with standard hyperparameters
    pub fn new(learning_rate: ScalarF4E4) -> Self {
        Adam {
            learning_rate,
            beta1: ScalarF4E4::from(9u8) / ScalarF4E4::from(10u8),  // 0.9
            beta2: ScalarF4E4::from(999u16) / ScalarF4E4::from(1000u16),  // 0.999
            epsilon: ScalarF4E4::ONE / ScalarF4E4::from(100000000u32),  // 1e-8
            m: Vec::new(),
            v: Vec::new(),
            t: 0,
        }
    }

    /// Perform one optimization step
    pub fn step(&mut self, parameters: &mut [&mut Tensor]) -> Result<()> {
        self.t += 1;

        // Initialize moments if needed
        if self.m.is_empty() {
            for param in parameters.iter() {
                if let Some(data) = param.as_scalars() {
                    self.m.push(vec![ScalarF4E4::ZERO; data.len()]);
                    self.v.push(vec![ScalarF4E4::ZERO; data.len()]);
                }
            }
        }

        let _t_scalar = ScalarF4E4::from(self.t as u32); // TODO: Use for bias correction

        for (i, param) in parameters.iter_mut().enumerate() {
            // Copy gradient to avoid borrow checker issues
            let grad_data_copy: Vec<ScalarF4E4> = {
                let grad = param.grad().ok_or_else(|| {
                    crate::error::VeritasError::InvalidInput(
                        "Parameter has no gradient".to_string()
                    )
                })?;

                grad.as_scalars().ok_or_else(|| {
                    crate::error::VeritasError::InvalidInput(
                        "Gradient must be scalar tensor".to_string()
                    )
                })?.to_vec()
            };

            let param_data = param.as_scalars_mut().ok_or_else(|| {
                crate::error::VeritasError::InvalidInput(
                    "Parameter must be scalar tensor".to_string()
                )
            })?;

            if i >= self.m.len() {
                continue;
            }

            let m = &mut self.m[i];
            let v = &mut self.v[i];

            for j in 0..param_data.len() {
                let g = grad_data_copy[j];

                // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                m[j] = self.beta1 * m[j] + (ScalarF4E4::ONE - self.beta1) * g;

                // Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
                v[j] = self.beta2 * v[j] + (ScalarF4E4::ONE - self.beta2) * (g * g);

                // Bias correction (simplified - use (1 - beta^t) approximation)
                // For now, skip bias correction to avoid pow/sqrt issues
                // TODO: Implement proper bias correction once Spirix has these ops

                // Update parameter: w = w - lr * m[j] / (v[j] + epsilon)
                // Simplified Adam without bias correction
                let update = self.learning_rate * m[j] / (v[j] + self.epsilon);
                param_data[j] = param_data[j] - update;
            }
        }

        Ok(())
    }

    /// Zero all gradients
    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}
