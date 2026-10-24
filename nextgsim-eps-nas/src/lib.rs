//! EPS NAS for the nextgsim LTE UE (nextgsim #186, increment 3): the UE side of
//! TS 24.301 EMM and ESM for attach, with EPS AKA (TS 33.401 §6.1, MILENAGE)
//! and NAS security (TS 24.301 §4.4, TS 33.401 §7.2).
//!
//! # Why a crate of its own
//!
//! EPS NAS (TS 24.301) and 5GS NAS (TS 24.501, `nextgsim-nas`) are different
//! protocols that happen to share a header shape: different message types and
//! IEs (EPS mobile identity, UE network capability, ESM message container, the
//! EPS bearer QoS and PDN address of ESM), a different key hierarchy (`K_ASME`,
//! not `K_AMF`) and a different session layer (ESM bearers, not 5GSM PDU
//! sessions). Folding them into one crate would make every type ambiguous about
//! which system it belongs to -- the same reason E-UTRA RRC is
//! `nextgsim-lte-rrc` beside `nextgsim-rrc`.
//!
//! The codecs are written from TS 24.301 and are independent of nextgcore's
//! MME-side ones on purpose (nextgsim #45, #205): this UE is what tests that
//! MME, and a shared codec would agree with it on every misreading.

pub mod aka;
pub mod ie;
pub mod msg;
pub mod security;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod ue;

pub use ue::{DefaultBearer, EmmState, EpsNasConfig, EpsNasUe, NasOutput, PagedWith};
