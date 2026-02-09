//! NGAP Procedures
//!
//! This module contains implementations for NGAP procedures as defined in 3GPP TS 38.413,
//! along with 6G extensions for ISAC, AI-native, and NTN support.

pub mod error_indication;
pub mod handover;
pub mod handover_preparation_failure;
pub mod initial_context_setup;
pub mod initial_ue_message;
pub mod nas_transport;
pub mod ng_setup;
pub mod paging;
pub mod pdu_session_resource;
pub mod ue_context_release;
pub mod amf_status_indication;
pub mod ran_configuration_update;
pub mod isac_reporting;
pub mod ai_native;
pub mod ntn_support;

pub use error_indication::*;
// Note: handover exports UserLocationInfoNr which conflicts with initial_ue_message
// Use explicit imports when both are needed
#[allow(ambiguous_glob_reexports)]
pub use handover::*;
pub use handover_preparation_failure::*;
pub use initial_context_setup::*;
pub use initial_ue_message::*;
pub use nas_transport::*;
pub use ng_setup::*;
pub use paging::*;
pub use pdu_session_resource::*;
pub use ue_context_release::*;
pub use amf_status_indication::*;
pub use ran_configuration_update::*;
pub use isac_reporting::*;
pub use ai_native::*;
pub use ntn_support::*;
