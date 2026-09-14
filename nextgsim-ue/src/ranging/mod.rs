//! Ranging and Sidelink Positioning module for UE
//!
//! **Scaffold, not wired end-to-end.** Models the Rel-18 concepts of TS 23.586
//! — UE-to-UE ranging via RTT measurement, carrier-phase positioning for
//! cm-level accuracy, and LMF reporting — but nothing drives it: no
//! `RangingMessage` producer exists, no SL-PRS stimulus is generated, and
//! there is no UE→LMF (SLPP/RSPP) transport, so the measurement handlers are
//! unreachable at runtime. The module deliberately makes no spec-compliance
//! claim; see [`SPAWN_LOG`] and [`START_LOG`] for the strings the runtime
//! emits, and issue #55 for the increments that would make the claim true.

pub mod task;
pub use task::{RangingTask, SPAWN_LOG, START_LOG};
