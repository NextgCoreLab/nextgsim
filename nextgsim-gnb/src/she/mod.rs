//! Service Hosting Environment (SHE) module for gNB
//!
//! This module wires the nextgsim-she crate into the gNB task framework.

pub mod task;

pub use task::SheTask;
