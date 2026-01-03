//! GTP Task Module
//!
//! This module implements the GTP-U (User Plane) task for the gNB.
//! It handles:
//! - GTP tunnel management for PDU sessions
//! - User plane data forwarding between UE and UPF
//! - UDP transport for GTP-U messages
//!
//! # Architecture
//!
//! The GTP task receives messages from:
//! - NGAP task: UE context updates, PDU session create/release
//! - RLS task: Uplink data PDUs from UEs
//!
//! The GTP task sends messages to:
//! - RLS task: Downlink data PDUs to UEs
//!
//! # Reference
//!
//! Based on UERANSIM's GtpTask from `src/gnb/gtp/task.cpp`

mod task;

pub use task::GtpTask;
