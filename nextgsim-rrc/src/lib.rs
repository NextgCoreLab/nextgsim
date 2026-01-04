//! RRC (Radio Resource Control) protocol library
#![allow(missing_docs)]
//!
//! Implements ASN.1 UPER encoding/decoding for RRC messages.
//!
//! # Modules
//!
//! - `codec` - Low-level ASN.1 UPER encoding/decoding
//! - `procedures` - High-level RRC procedure implementations

pub mod codec;
pub mod procedures;
