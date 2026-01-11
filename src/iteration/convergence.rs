//! Convergence detection - when to stop iterating
//!
//! Two stopping conditions:
//! 1. BOUNDED: State converged (stable, this is truth)
//! 2. ESCAPED: State diverged (unstable, this is farktogle)

use spirix::ScalarF4E4;

/// Configuration for convergence detection
#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    /// Magnitude threshold for escape detection
    /// If |z| > escape_threshold, the state has diverged
    pub escape_threshold: ScalarF4E4,

    /// Stability threshold for convergence detection
    /// If |z_new - z_old| < stability_threshold, the state has converged
    pub stability_threshold: ScalarF4E4,

    /// Maximum iterations before giving up
    pub max_iterations: usize,

    /// Minimum iterations before checking convergence
    /// (prevents premature convergence detection)
    pub min_iterations: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        ConvergenceConfig {
            // Escape if magnitude > 2.0 (standard Mandelbrot threshold)
            escape_threshold: ScalarF4E4::from(2u8),

            // Converged if change < 1e-6
            stability_threshold: ScalarF4E4::ONE / ScalarF4E4::from(1000000u32),

            // Maximum 1000 iterations
            max_iterations: 1000,

            // Check convergence after 10 iterations
            min_iterations: 10,
        }
    }
}

/// Detects convergence or divergence of iterative state
pub struct ConvergenceDetector {
    config: ConvergenceConfig,
    iteration_count: usize,
}

impl ConvergenceDetector {
    /// Create new convergence detector with default config
    pub fn new() -> Self {
        Self::with_config(ConvergenceConfig::default())
    }

    /// Create convergence detector with custom config
    pub fn with_config(config: ConvergenceConfig) -> Self {
        ConvergenceDetector {
            config,
            iteration_count: 0,
        }
    }

    /// Reset iteration counter
    pub fn reset(&mut self) {
        self.iteration_count = 0;
    }

    /// Check if state has escaped (diverged to infinity)
    pub fn has_escaped(&self, magnitude: ScalarF4E4) -> bool {
        magnitude > self.config.escape_threshold
    }

    /// Check if state has converged (stable)
    pub fn has_converged(&self, change: ScalarF4E4) -> bool {
        if self.iteration_count < self.config.min_iterations {
            return false;
        }

        change < self.config.stability_threshold
    }

    /// Check if maximum iterations reached
    pub fn is_max_iterations(&self) -> bool {
        self.iteration_count >= self.config.max_iterations
    }

    /// Increment iteration counter
    pub fn tick(&mut self) {
        self.iteration_count += 1;
    }

    /// Get current iteration count
    pub fn iterations(&self) -> usize {
        self.iteration_count
    }
}

impl Default for ConvergenceDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_detection() {
        let detector = ConvergenceDetector::new();

        // Below threshold - not escaped
        assert!(!detector.has_escaped(ScalarF4E4::ONE));

        // Above threshold - escaped
        assert!(detector.has_escaped(ScalarF4E4::from(3u8)));
    }

    #[test]
    fn test_convergence_detection() {
        let mut detector = ConvergenceDetector::new();

        // Before min_iterations - never converged
        for _ in 0..5 {
            detector.tick();
            assert!(!detector.has_converged(ScalarF4E4::ZERO));
        }

        // After min_iterations with small change - converged
        for _ in 5..15 {
            detector.tick();
        }
        let tiny_change = ScalarF4E4::ONE / ScalarF4E4::from(10000000u32);
        assert!(detector.has_converged(tiny_change));
    }

    #[test]
    fn test_max_iterations() {
        let mut config = ConvergenceConfig::default();
        config.max_iterations = 50;
        let mut detector = ConvergenceDetector::with_config(config);

        for i in 0..50 {
            assert!(!detector.is_max_iterations());
            detector.tick();
            assert_eq!(detector.iterations(), i + 1);
        }

        assert!(detector.is_max_iterations());
    }
}
