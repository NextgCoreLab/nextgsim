//! NGAP (NG Application Protocol) library
//!
//! Implements ASN.1 PER encoding/decoding for NGAP messages
//! between gNB and AMF.
//!
//! # Modules
//!
//! - `codec` - Low-level NGAP PDU encoding/decoding using APER
//! - `procedures` - High-level NGAP procedure implementations

pub mod codec;
pub mod procedures;

#[cfg(test)]
mod capture_tests;
