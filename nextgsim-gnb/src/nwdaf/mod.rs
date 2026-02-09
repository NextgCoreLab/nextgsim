//! Network Data Analytics Function (NWDAF) module for gNB
//!
//! This module wires the nextgsim-nwdaf crate into the gNB task framework.

pub mod task;

pub use task::NwdafTask;
