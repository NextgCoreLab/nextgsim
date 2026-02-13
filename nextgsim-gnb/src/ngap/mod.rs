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

mod amf_context;
mod task;
mod ue_context;
pub mod mbs_context;

pub use amf_context::{AmfContextInfo, AmfState, NgapAmfContext};
pub use task::NgapTask;
pub use ue_context::{NgapUeContext, UeState};
pub use mbs_context::{GnbMbsContext, MbsSessionState};
