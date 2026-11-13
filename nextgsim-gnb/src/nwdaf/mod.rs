//! Network Data Analytics Function (NWDAF) module for gNB
//!
//! This module wires the nextgsim-nwdaf crate into the gNB task framework.

#[cfg(feature = "demo-d441d-predictive-ho")]
pub mod d441d;
pub mod task;

pub use task::NwdafTask;
