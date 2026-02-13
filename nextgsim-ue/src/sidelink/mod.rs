//! Sidelink module for UE (Rel-18)
//!
//! Provides NR sidelink relay (UE-to-UE relay), sidelink discovery
//! procedures, PC5 link establishment, and sidelink positioning.

pub mod task;
pub use task::SidelinkTask;
