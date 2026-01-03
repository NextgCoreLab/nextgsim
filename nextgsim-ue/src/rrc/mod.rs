//! RRC (Radio Resource Control) Module for UE
//!
//! This module implements the RRC protocol handling for the UE, including:
//! - RRC state machine (Idle, Connected, Inactive)
//! - RRC connection management
//! - Cell selection/reselection (TODO)
//! - RRC message handling (TODO)
//!
//! # RRC State Machine (3GPP TS 38.331)
//!
//! ```text
//!                    ┌──────────┐
//!                    │   Idle   │◄────────────────────┐
//!                    └────┬─────┘                     │
//!                         │ RRC Setup                 │
//!                         ▼                           │
//!                    ┌──────────┐                     │
//!              ┌─────│Connected │─────┐               │
//!              │     └──────────┘     │               │
//!              │ RRC Suspend          │ RRC Release   │
//!              ▼                      └───────────────┘
//!         ┌──────────┐
//!         │ Inactive │
//!         └────┬─────┘
//!              │ RRC Resume / Release
//!              ▼
//!         ┌──────────┐
//!         │Connected │ or │Idle│
//!         └──────────┘
//! ```
//!
//! # Reference
//!
//! Based on UERANSIM's UE RRC implementation from `src/ue/rrc/`.

pub mod state;

// Re-export main types
pub use state::{RrcState, RrcStateMachine, RrcStateTransition, RrcStateError};
