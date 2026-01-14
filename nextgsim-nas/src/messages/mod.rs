//! NAS 5G Messages
//!
//! This module contains implementations of 5G NAS messages
//! as defined in 3GPP TS 24.501.
//!
//! ## Message Categories
//!
//! - 5GMM (Mobility Management) messages - [`mm`]
//! - 5GSM (Session Management) messages - [`sm`]

pub mod mm;
pub mod sm;

pub use mm::*;
pub use sm::*;
