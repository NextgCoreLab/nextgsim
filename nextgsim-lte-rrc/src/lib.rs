//! E-UTRA RRC (Radio Resource Control, TS 36.331) for the nextgsim eNB and LTE UE.
//!
//! The codec is generated at build time from the TS 36.331 V19.4.0 ASN.1
//! (`tools/lte-rrc-19.4.0.asn`, see `tools/LTE-RRC-PROVENANCE.md`). It is its own
//! crate rather than part of `nextgsim-rrc` because the E-UTRA and NR schemas
//! define many same-named types (`MasterInformationBlock`, `RRCConnectionSetup`
//! has no NR twin but `UL-DCCH-Message`, `PhysCellId`, ... do).

pub mod codec;
pub mod msg;

#[cfg(test)]
mod seq_extension_tests;
