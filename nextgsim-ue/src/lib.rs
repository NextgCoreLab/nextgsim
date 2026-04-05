//! nextgsim UE (User Equipment) Library
#![allow(missing_docs)]
//!
//! This crate provides the UE (User Equipment) implementation for the nextgsim
//! 5G simulator. It includes:
//!
//! - Timer management for NAS procedures
//! - Task framework for async message passing
//! - NAS mobility management (registration, deregistration, authentication)
//! - NAS session management (PDU session establishment, modification, release)
//! - RRC state machine and procedures
//! - RLS transport for cell search and gNB communication
//! - TUN interface for user plane data
//! - Application task for CLI command handling

pub mod app;
pub mod nas;
pub mod rls;
pub mod rrc;
pub mod tasks;
pub mod timer;
pub mod tun;

// 6G AI-native network function client modules
#[cfg(feature = "nextgsim-she")]
pub mod she_client;
#[cfg(feature = "nextgsim-nwdaf")]
pub mod nwdaf_reporter;
#[cfg(feature = "nextgsim-isac")]
pub mod isac_sensor;
#[cfg(feature = "nextgsim-fl")]
pub mod fl_participant;
#[cfg(feature = "nextgsim-semantic")]
pub mod semantic_codec;

// Rel-18 5G-Advanced modules
pub mod ranging;
pub mod mint;
pub mod sidelink;
pub mod ambient_iot;

// Rel-17 protocol extensions
pub mod prose;  // ProSe PC5 proximity services (TS 23.303/23.304)
pub mod uav;    // UAV identification and C2 link management (TS 23.256)
pub mod daps;   // DAPS dual active protocol stack handover (TS 38.331)

// Re-export commonly used types
pub use timer::{GprsTimer2, GprsTimer3, GprsTimer3Unit, UeTimer};

// Re-export NAS types
pub use nas::mm::{
    DeregistrationCause, DeregistrationProcedure, DeregistrationProcedureError,
    MmState, MmSubState, NetworkDeregistrationResult, ProcedureResult, RmState, UpdateStatus,
    T3521_CODE, T3521_DEFAULT_INTERVAL_SECS, T3521_MAX_RETRANSMISSION,
};

// Re-export SM types
pub use nas::sm::{
    ProcedureTransaction, ProcedureTransactionManager, PtState, PtiValidationResult,
    SmMessageType, PTI_MAX, PTI_MIN, PTI_UNASSIGNED, SM_TIMER_T3580, SM_TIMER_T3581,
    SM_TIMER_T3582,
};

// Re-export RRC types
pub use rrc::{RrcState, RrcStateMachine, RrcStateTransition, RrcStateError, RrcTask};

// Re-export RLS types
pub use rls::{RlsTask, RlsTaskConfig, DEFAULT_RLS_PORT};

// Re-export task types
pub use tasks::{
    AppMessage, CmState, NasMessage, RlsMessage, RrcMessage, Task, TaskHandle, TaskId,
    TaskManager, TaskMessage, TaskState, UeCliCommand, UeCliCommandType, UeStatusUpdate,
    UeTaskBase, DEFAULT_CHANNEL_CAPACITY,
};

// Re-export App types
pub use app::{
    parse_ue_cli_command, AppTask, CliCommandResult, CliHandler, DeregistrationRequest, NasAction,
    PduSessionEstablishRequest, PduSessionReleaseRequest, PduSessionType,
};

// Re-export TUN types
pub use tun::{
    is_valid_ip_packet, spawn_tun_reader, IpPacket, IpVersion, TunAppMessage, TunConfig, TunError,
    TunInterface, TunMessage, TunReader, TunTask, TunTaskConfig, TunWriter,
};

// Re-export 6G AI-native network function task types
#[cfg(feature = "nextgsim-she")]
pub use she_client::SheClientTask;
#[cfg(feature = "nextgsim-nwdaf")]
pub use nwdaf_reporter::NwdafReporterTask;
#[cfg(feature = "nextgsim-isac")]
pub use isac_sensor::IsacSensorTask;
#[cfg(feature = "nextgsim-fl")]
pub use fl_participant::FlParticipantTask;
#[cfg(feature = "nextgsim-semantic")]
pub use semantic_codec::SemanticCodecTask;
