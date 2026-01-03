//! 5GMM (5G Mobility Management) Procedures
//!
//! This module implements the UE-side 5GMM procedures as defined in 3GPP TS 24.501:
//! - Registration procedure (initial, mobility, periodic)
//! - Deregistration procedure (UE-initiated, network-initiated)
//! - Service request procedure
//! - Authentication procedure
//! - Security mode control procedure
//!
//! # State Machine
//!
//! The MM state machine follows 3GPP TS 24.501 Section 5.1.3:
//! - RM states: RM-DEREGISTERED, RM-REGISTERED
//! - CM states: CM-IDLE, CM-CONNECTED
//! - MM states: MM-NULL, MM-DEREGISTERED, MM-REGISTERED-INITIATED, etc.
//! - U-states: U1-UPDATED, U2-NOT-UPDATED, U3-ROAMING-NOT-ALLOWED
//!
//! The `MmStateMachine` struct manages all state transitions and provides
//! callbacks for state change events.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/nas/mm/` implementation.

mod deregistration;
mod state;

pub use deregistration::*;
pub use state::*;
