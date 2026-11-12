//! Semantic Codec module for UE
pub mod task;
pub use task::SemanticCodecTask;

#[cfg(feature = "demo-d444a-ambr-adapt")]
pub mod d444a;
