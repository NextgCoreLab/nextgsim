//! MINT (Multi-IMSI/Multi-USIM) module for UE (Rel-18, TS 23.761)
//!
//! Supports multiple IMSI/SUPI per UE (dual-SIM), per-IMSI NAS context
//! management, and IMSI selection for outgoing calls/sessions.

pub mod task;
pub use task::MintTask;
