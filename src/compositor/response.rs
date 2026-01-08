//! Composed response types

/// A composed response ready for output
#[derive(Debug, Clone)]
pub enum ComposedResponse {
    Text(String),
    Json(serde_json::Value),
    // TODO: Add more output formats
}

impl ComposedResponse {
    pub fn text(s: impl Into<String>) -> Self {
        ComposedResponse::Text(s.into())
    }
}
