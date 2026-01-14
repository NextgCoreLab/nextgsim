//! NAS (Non-Access Stratum) Protocol Handling
//!
//! This module implements the UE-side NAS protocol handling including:
//! - 5GMM (Mobility Management) procedures
//! - 5GSM (Session Management) procedures
//!
//! # Architecture
//!
//! The NAS layer is organized into:
//! - `mm`: Mobility Management procedures (registration, deregistration, authentication, etc.)
//! - `sm`: Session Management procedures (PDU session establishment, modification, release)
//!
//! # Reference
//!
//! Based on 3GPP TS 24.501 and UERANSIM's `src/ue/nas/` implementation.

pub mod mm;
pub mod sm;
