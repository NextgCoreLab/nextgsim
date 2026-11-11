//! Network Knowledge Exposure Function (NKEF) module for gNB
//!
//! This module wires the nextgsim-nkef crate into the gNB task framework.

pub mod live;
#[cfg(feature = "nkef-sba")]
pub mod sba;
pub mod task;

pub use task::NkefTask;
