//! 5GSM (5G Session Management) Procedures
//!
//! This module implements the UE-side 5GSM procedures as defined in 3GPP TS 24.501:
//! - PDU session establishment procedure
//! - PDU session modification procedure
//! - PDU session release procedure
//! - Procedure transaction handling
//!
//! # Procedure Transaction Identity (PTI)
//!
//! PTI is used to identify SM procedures. Each procedure is assigned a unique PTI
//! from the range 1-254. PTI 0 is reserved for network-initiated procedures.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/sm/` implementation.

mod procedure;

pub use procedure::*;
