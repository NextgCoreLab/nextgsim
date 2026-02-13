//! Energy Savings module for gNB (Rel-18)
//!
//! Provides cell sleep/dormant mode support, wake-up from sleep
//! on paging or traffic, and energy efficiency metrics per cell.

pub mod task;
pub use task::EnergyTask;
