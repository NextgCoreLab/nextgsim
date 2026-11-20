//! Ranging and Sidelink Positioning module for UE
//!
//! Models the Rel-18 concepts of TS 23.586: UE-to-UE ranging by RTT measurement and
//! carrier-phase positioning. The result is reported to the LMF as an SL-RTT
//! measurement over SLPP (TS 38.355, `crate::nas::slpp`, nextgcore #557). It is wired
//! on the UE side, from an SL-PRS occasion to the UL NAS TRANSPORT, and it measures
//! configured geometry rather than a radio. No two-process run against nextgcore's LMF
//! has happened. [`SPAWN_LOG`] and [`START_LOG`] are the strings the runtime emits,
//! and they name that limit.

pub mod task;
pub use task::{RangingTask, SPAWN_LOG, START_LOG};
