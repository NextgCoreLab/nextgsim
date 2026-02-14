//! Ranging and Sidelink Positioning module for UE (Rel-18, TS 23.586)
//!
//! Provides UE-to-UE ranging via RTT measurement, carrier phase positioning
//! for cm-level accuracy, and LMF reporting.

pub mod task;
pub use task::RangingTask;
