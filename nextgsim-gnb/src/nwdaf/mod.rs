//! Network Data Analytics Function (NWDAF) module for gNB
//!
//! This module wires the nextgsim-nwdaf crate into the gNB task framework.

#[cfg(feature = "demo-d440a-ai-paging")]
pub mod d440a;
#[cfg(feature = "demo-d441d-predictive-ho")]
pub mod d441d;
pub mod task;

pub use task::NwdafTask;
