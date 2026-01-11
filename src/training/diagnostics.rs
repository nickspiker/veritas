//! Training Diagnostics
//!
//! CONSTITUTION COMPLIANT:
//! ✓ No IEEE-754
//! ✓ Pure counting (no percentages, no division)
//! ✓ Spirix state probing
//!
//! Tracks network health through pure counts.

use spirix::ScalarF4E4;

/// Training diagnostics - pure counts only
#[derive(Debug, Clone, Default)]
pub struct Diagnostics {
    /// Layer health: (zero_count, nonzero_count, total_count)
    pub w_ih_state: (usize, usize, usize),
    pub w_hh_state: (usize, usize, usize),
    pub w_ho_state: (usize, usize, usize),

    /// Gradient health: (zero_count, nonzero_count, total_count)
    pub grad_ih: (usize, usize, usize),
    pub grad_hh: (usize, usize, usize),
    pub grad_ho: (usize, usize, usize),
}

impl Diagnostics {
    /// Create new diagnostics tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Probe layer for zero/nonzero values
    ///
    /// Returns: (zero_count, nonzero_count, total_count)
    pub fn probe_layer(data: &[ScalarF4E4]) -> (usize, usize, usize) {
        let mut zeros = 0;
        let total = data.len();

        for val in data {
            if *val == ScalarF4E4::ZERO {
                zeros += 1;
            }
        }

        let nonzeros = total - zeros;
        (zeros, nonzeros, total)
    }

    /// Update layer diagnostics
    pub fn update_weights(
        &mut self,
        w_ih: &[ScalarF4E4],
        w_hh: &[ScalarF4E4],
        w_ho: &[ScalarF4E4],
    ) {
        self.w_ih_state = Self::probe_layer(w_ih);
        self.w_hh_state = Self::probe_layer(w_hh);
        self.w_ho_state = Self::probe_layer(w_ho);
    }

    /// Update gradient diagnostics
    pub fn update_gradients(
        &mut self,
        grad_ih: &[ScalarF4E4],
        grad_hh: &[ScalarF4E4],
        grad_ho: &[ScalarF4E4],
    ) {
        self.grad_ih = Self::probe_layer(grad_ih);
        self.grad_hh = Self::probe_layer(grad_hh);
        self.grad_ho = Self::probe_layer(grad_ho);
    }

    /// Print diagnostics (pure counts, no percentages)
    pub fn print(&self, epoch: usize) {
        println!("\n=== Epoch {} Diagnostics ===", epoch);

        println!("Weight Health:");
        self.print_layer_state("  w_ih", self.w_ih_state);
        self.print_layer_state("  w_hh", self.w_hh_state);
        self.print_layer_state("  w_ho", self.w_ho_state);

        println!("\nGradient Health:");
        self.print_layer_state("  grad_ih", self.grad_ih);
        self.print_layer_state("  grad_hh", self.grad_hh);
        self.print_layer_state("  grad_ho", self.grad_ho);
    }

    /// Print layer state (pure counts)
    fn print_layer_state(&self, name: &str, state: (usize, usize, usize)) {
        let (zeros, nonzeros, total) = state;
        println!("{}: {} nonzero, {} zero (of {})", name, nonzeros, zeros, total);
    }

    /// Check if training is healthy (no widespread zeros)
    pub fn is_healthy(&self) -> bool {
        // Heuristic: >10% nonzero in each layer
        let check = |state: (usize, usize, usize)| -> bool {
            let (_zeros, nonzeros, total) = state;
            if total == 0 {
                return true;
            }
            // Avoid division: use 10× multiplication instead
            nonzeros * 10 > total
        };

        check(self.w_ih_state)
            && check(self.w_hh_state)
            && check(self.w_ho_state)
            && check(self.grad_ih)
            && check(self.grad_hh)
            && check(self.grad_ho)
    }
}
