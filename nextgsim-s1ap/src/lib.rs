//! S1AP (S1 Application Protocol, TS 36.413) for the nextgsim eNB.
//!
//! The codec is generated at build time from the TS 36.413 V19.2.0 ASN.1
//! (`tools/s1ap-19.2.0.asn`, see `tools/S1AP-PROVENANCE.md`). It is kept
//! independent of nextgcore's S1AP implementation on purpose: nextgsim is the
//! peer that tests nextgcore, and an independent codec is what catches a
//! misreading on either side (nextgsim #45, #205).

pub mod codec;
