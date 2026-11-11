//! AI Agent Framework (AAF) module for gNB

#[cfg(feature = "demo-d446a-agent-conflict")]
pub mod d446a;
pub mod task;
pub use task::AgentTask;
