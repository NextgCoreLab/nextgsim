//! SCTP Task Module
//!
//! This module implements the SCTP task for managing SCTP associations
//! used for NGAP transport between gNB and AMF.
//!
//! # Architecture
//!
//! The SCTP task manages connections to AMF(s) and routes messages between
//! the SCTP transport layer and the NGAP task:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                        SCTP Task                             │
//! │  ┌─────────────────────────────────────────────────────┐    │
//! │  │              AMF Connection Manager                  │    │
//! │  │  ┌───────────┐  ┌───────────┐  ┌───────────┐       │    │
//! │  │  │ AMF Conn 1│  │ AMF Conn 2│  │ AMF Conn N│       │    │
//! │  │  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘       │    │
//! │  └────────┼──────────────┼──────────────┼─────────────┘    │
//! │           │              │              │                   │
//! │           └──────────────┴──────────────┘                   │
//! │                          │                                  │
//! │                    Message Router                           │
//! │                          │                                  │
//! └──────────────────────────┼──────────────────────────────────┘
//!                            │
//!                            ▼
//!                       NGAP Task
//! ```
//!
//! # Message Routing
//!
//! The SCTP task routes the following messages to the NGAP task:
//! - `SctpAssociationUp` - When an SCTP association is established
//! - `SctpAssociationDown` - When an SCTP association is closed
//! - `ReceiveNgapPdu` - When an NGAP PDU is received from AMF

mod amf_connection;
mod task;

pub use amf_connection::{AmfConnection, AmfConnectionConfig, AmfConnectionEvent, AmfConnectionState};
pub use task::SctpTask;
