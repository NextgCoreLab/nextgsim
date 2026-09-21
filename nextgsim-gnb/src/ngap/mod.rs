//! NGAP Task Module
//!
//! This module implements the NGAP task for the gNB. The NGAP task is responsible for:
//! - Managing AMF connections and state
//! - Handling NG Setup procedure
//! - Managing UE contexts
//! - Routing NAS messages between RRC and AMF
//! - Handling PDU session resources
//!
//! # Architecture
//!
//! ```text
//! SCTP Task <---> NGAP Task <---> RRC Task
//!                    |
//!                    +---------> GTP Task
//! ```
//!
//! The NGAP task receives SCTP events and NGAP PDUs from the SCTP task,
//! processes them according to 3GPP TS 38.413, and routes messages to
//! the appropriate tasks (RRC for NAS, GTP for PDU sessions).
//!
//! MBS (Rel-17) state does not live here: the single MBS state machine is
//! [`crate::mbs_ngap::NgapMbsManager`], which the task drives both from the
//! §9.2.9 session procedures and from the MBS membership IEs on the PDU Session
//! procedures. A second, id-keyed manager under this module was deleted in issue
//! #188 as unreachable.

mod amf_context;
mod task;
pub mod timers;
mod ue_context;
pub mod up_security;

pub use amf_context::{AmfContextInfo, AmfState, NgapAmfContext};
pub use task::NgapTask;
pub use timers::{GuardTimer, GuardTimers};
pub use ue_context::{NgapUeContext, UeState};
