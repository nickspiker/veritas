//! Training checkpoint management

/// Training checkpoint (stub)
#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub epoch: u32,
    // TODO: Add weights, optimizer state, etc.
}

impl Checkpoint {
    pub fn new(epoch: u32) -> Self {
        Checkpoint { epoch }
    }
}
