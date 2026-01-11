//! PID Controller for Adaptive Learning Rate
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure Spirix arithmetic
//! ✓ Verifiable control loop
//!
//! Uses classic PID control theory to automatically adjust learning rate:
//! - Proportional: React to current loss change
//! - Integral: Smooth out oscillations
//! - Derivative: Predict future trends

use spirix::ScalarF4E4;

/// PID controller for adaptive learning rate adjustment
#[derive(Debug, Clone)]
pub struct PIDLearningRate {
    // Current learning rate
    lr: ScalarF4E4,

    // PID state
    integral: ScalarF4E4,
    prev_error: ScalarF4E4,

    // PID gains (tunable)
    kp: ScalarF4E4,  // Proportional gain
    ki: ScalarF4E4,  // Integral gain
    kd: ScalarF4E4,  // Derivative gain

    // Bounds to prevent explosion/stagnation
    min_lr: ScalarF4E4,
    max_lr: ScalarF4E4,

    // History for calculating error
    prev_loss: ScalarF4E4,
    first_update: bool,
}

impl PIDLearningRate {
    /// Create new PID controller with default parameters
    ///
    /// Default configuration:
    /// - Initial LR: 0.001
    /// - kp: 0.1 (proportional gain)
    /// - ki: 0.01 (integral gain)
    /// - kd: 0.05 (derivative gain)
    /// - LR bounds: [0.0001, 0.01]
    pub fn new() -> Self {
        Self {
            lr: ScalarF4E4::from(1u8) / ScalarF4E4::from(1000u16),  // 0.001
            integral: ScalarF4E4::ZERO,
            prev_error: ScalarF4E4::ZERO,
            kp: ScalarF4E4::from(1u8) / ScalarF4E4::from(10u8),     // 0.1
            ki: ScalarF4E4::from(1u8) / ScalarF4E4::from(100u8),    // 0.01
            kd: ScalarF4E4::from(5u8) / ScalarF4E4::from(100u8),    // 0.05
            min_lr: ScalarF4E4::from(1u8) / ScalarF4E4::from(10000u16),  // 0.0001
            max_lr: ScalarF4E4::from(1u8) / ScalarF4E4::from(100u8),     // 0.01
            prev_loss: ScalarF4E4::ZERO,
            first_update: true,
        }
    }

    /// Update learning rate based on current loss
    ///
    /// PID control:
    /// - If loss increasing: reduce LR (too aggressive)
    /// - If loss decreasing: maintain or increase LR
    /// - If loss oscillating: integral term dampens
    /// - If loss trend changing: derivative predicts
    pub fn update(&mut self, current_loss: ScalarF4E4) -> ScalarF4E4 {
        // First update: just record loss
        if self.first_update {
            self.prev_loss = current_loss;
            self.first_update = false;
            return self.lr;
        }

        // Error: how much did loss change?
        // Positive error = loss increased (bad)
        // Negative error = loss decreased (good)
        let error = current_loss - self.prev_loss;

        // Integral: accumulate error over time (smoothing)
        self.integral = self.integral + error;

        // Derivative: rate of change of error (prediction)
        let derivative = error - self.prev_error;

        // PID calculation
        let p_term = self.kp * error;
        let i_term = self.ki * self.integral;
        let d_term = self.kd * derivative;

        let adjustment = p_term + i_term + d_term;

        // Update LR: if error > 0 (loss increased), adjustment > 0, so LR decreases
        //            if error < 0 (loss decreased), adjustment < 0, so LR increases
        self.lr = self.lr * (ScalarF4E4::ONE - adjustment);

        // Clamp to bounds
        if self.lr < self.min_lr {
            self.lr = self.min_lr;
        } else if self.lr > self.max_lr {
            self.lr = self.max_lr;
        }

        // Update state for next iteration
        self.prev_error = error;
        self.prev_loss = current_loss;

        self.lr
    }

    /// Get current learning rate without updating
    pub fn current_lr(&self) -> ScalarF4E4 {
        self.lr
    }

    /// Reset PID state (useful for curriculum learning)
    pub fn reset(&mut self) {
        self.integral = ScalarF4E4::ZERO;
        self.prev_error = ScalarF4E4::ZERO;
        self.first_update = true;
    }
}

impl Default for PIDLearningRate {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pid_increases_lr_on_good_progress() {
        let mut pid = PIDLearningRate::new();
        let initial_lr = pid.current_lr();

        // Simulate decreasing loss (good progress)
        let loss1 = ScalarF4E4::from(3u8);
        let loss2 = ScalarF4E4::from(2u8);
        let loss3 = ScalarF4E4::from(1u8);

        pid.update(loss1);
        let lr2 = pid.update(loss2);
        let lr3 = pid.update(loss3);

        // LR should increase when loss decreases
        assert!(lr2 > initial_lr || lr3 > lr2, "PID should increase LR when loss decreases");
    }

    #[test]
    fn test_pid_decreases_lr_on_divergence() {
        let mut pid = PIDLearningRate::new();

        // Simulate increasing loss (divergence)
        let loss1 = ScalarF4E4::from(1u8);
        let loss2 = ScalarF4E4::from(2u8);
        let loss3 = ScalarF4E4::from(3u8);

        pid.update(loss1);
        let lr2 = pid.update(loss2);
        let lr3 = pid.update(loss3);

        // LR should decrease when loss increases
        assert!(lr3 < lr2, "PID should decrease LR when loss increases");
    }

    #[test]
    fn test_pid_respects_bounds() {
        let mut pid = PIDLearningRate::new();

        // Simulate extreme loss increase
        for _ in 0..100 {
            let huge_loss = ScalarF4E4::from(100u8);
            pid.update(huge_loss);
        }

        // Should not go below min_lr
        assert!(pid.current_lr() >= pid.min_lr, "PID should respect min_lr bound");

        // Reset and simulate extreme loss decrease
        pid.reset();
        let mut loss = ScalarF4E4::from(100u8);
        for _ in 0..100 {
            loss = loss / ScalarF4E4::from(2u8);
            pid.update(loss);
        }

        // Should not go above max_lr
        assert!(pid.current_lr() <= pid.max_lr, "PID should respect max_lr bound");
    }
}
