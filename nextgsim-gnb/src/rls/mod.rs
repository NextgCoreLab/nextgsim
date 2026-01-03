//! RLS (Radio Link Simulation) Task for gNB
//!
//! This module implements the RLS task for the gNB, which handles:
//! - UE discovery via heartbeat messages
//! - RRC message relay between UE and RRC task
//! - User plane data relay between UE and GTP task
//!
//! # Architecture
//!
//! The RLS task uses UDP for communication with UEs and integrates with:
//! - `GnbCellTracker` for UE discovery and tracking
//! - `RlsTransport` for PDU transmission and acknowledgment
//!
//! # Reference
//!
//! Based on UERANSIM's gNB RLS implementation from `src/gnb/rls/`

mod task;

pub use task::RlsTask;
