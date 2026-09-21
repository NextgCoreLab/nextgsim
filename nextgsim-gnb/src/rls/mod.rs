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

// `pub(crate)` rather than private since issue #57: the RRC task scales a per-UE MAC
// grant ceiling off `task::MAC_GRANT_BYTES`, so it has to be able to name it. Sharing
// the constant is what stops the ceiling and the grant it bounds drifting apart. The
// module stays crate-internal; only `RlsTask` is exported.
pub(crate) mod task;

pub use task::RlsTask;
