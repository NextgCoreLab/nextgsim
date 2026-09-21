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

// Shared unit-test helpers: the `tracing` capture behind the honesty invariants
// (#54, #55) and a `UeTaskBase` builder. Test-only, so it ships in no build.
#[cfg(test)]
mod test_support;

// 6G AI-native network function client modules
#[cfg(feature = "nextgsim-fl")]
pub mod fl_participant;
#[cfg(feature = "nextgsim-isac")]
pub mod isac_sensor;
#[cfg(feature = "nextgsim-nwdaf")]
pub mod nwdaf_reporter;
#[cfg(feature = "nextgsim-semantic")]
pub mod semantic_codec;
#[cfg(feature = "nextgsim-she")]
pub mod she_client;

// Rel-18 5G-Advanced modules
pub mod ambient_iot;
pub mod mint;
pub mod ranging;
// Off by default (issue #54): PC5 is not a capability a default UE should advertise, and
// the gate also keeps the module out of the lean `cargo test --workspace` build. Since
// issue #141 what it gates is working procedures rather than a facade -- see
// `sidelink`'s module docs. `UeConfig::prose_enabled` is the runtime half of the gate.
#[cfg(feature = "sidelink")]
pub mod sidelink;

// Rel-17 protocol extensions
pub mod daps;
// `prose` was REMOVED by issue #141, not gated further. It modelled ProSe contexts,
// bearers, relay contexts and Model A/B discovery status -- and every one of its types had
// zero consumers workspace-wide, so none of it ever ran. The procedures it described are
// now implemented, against the wire, in `sidelink::{pc5s, link, discovery, relay}`:
// `ProseContext`/`ProsePeer` by `Pc5DiscoveryEngine`/`DiscoveredPeer`, `Pc5Bearer` by
// `Pc5LinkContext`, `UeRelayContext` by `RelayForwarder`, and `ProseDiscoveryStatus` by
// the real Model A/B exchange. Keeping a second, inert model of the same procedures beside
// the working one would be two sources of truth with only one of them driven, which is the
// defect #141 exists to remove.
pub mod uav; // UAV identification and C2 link management (TS 23.256) // DAPS dual active protocol stack handover (TS 38.331)

// Re-export commonly used types
pub use timer::{GprsTimer2, GprsTimer3, GprsTimer3Unit, UeTimer};

// Re-export NAS types
pub use nas::mm::{
    // Configuration update
    ConfigUpdateError,
    ConfigUpdateProcedure,
    ConfigUpdateResult,
    ConfigurationUpdateCommand,
    ConfigurationUpdateComplete,
    // Deregistration
    DeregistrationCause,
    DeregistrationProcedure,
    DeregistrationProcedureError,
    // Emergency registration
    EmergencyRegistrationError,
    EmergencyRegistrationProcedure,
    // States
    MmState,
    MmSubState,
    NetworkDeregistrationResult,
    ProcedureResult,
    RmState,
    // Service request
    ServiceAcceptResult,
    ServiceRejectResult,
    ServiceRequestCause,
    ServiceRequestError,
    ServiceRequestProcedure,
    UpdateStatus,
    T3510_CODE,
    T3510_DEFAULT_INTERVAL_SECS,
    T3510_MAX_RETRANSMISSION,
    T3517_CODE,
    T3517_DEFAULT_INTERVAL_SECS,
    T3517_MAX_RETRANSMISSION,
    T3521_CODE,
    T3521_DEFAULT_INTERVAL_SECS,
    T3521_MAX_RETRANSMISSION,
};

// Re-export SM types
pub use nas::sm::{
    ProcedureTransaction, ProcedureTransactionManager, PtState, PtiValidationResult, SmMessageType,
    PTI_MAX, PTI_MIN, PTI_UNASSIGNED, SM_TIMER_T3580, SM_TIMER_T3581, SM_TIMER_T3582,
};

// Re-export RRC types
pub use rrc::{
    // Re-establishment
    ReestablishmentCompleteParams,
    ReestablishmentError,
    ReestablishmentProcedure,
    ReestablishmentRequestParams,
    ReestablishmentState,
    ReestablishmentTrigger,
    // Resume
    ResumeCause,
    ResumeCompleteParams,
    ResumeError,
    ResumeProcedure,
    ResumeProcedureState,
    ResumeRequestParams,
    RlfDetector,
    RrcState,
    RrcStateError,
    RrcStateMachine,
    RrcStateTransition,
    RrcTask,
    N310_DEFAULT,
    N311_DEFAULT,
    T301_DEFAULT_MS,
    T311_DEFAULT_MS,
    T319_DEFAULT_MS,
};

// Re-export RLS types
pub use rls::{RlsTask, RlsTaskConfig, DEFAULT_RLS_PORT};

// Re-export task types
#[cfg(feature = "sidelink")]
pub use tasks::SidelinkMessage;
pub use tasks::{
    AppMessage, CmState, MintMessage, NasMessage, RangingMessage, RlsMessage, RrcMessage, Task,
    TaskHandle, TaskId, TaskManager, TaskMessage, TaskState, UeCliCommand, UeCliCommandType,
    UeRel18Receivers, UeStatusUpdate, UeTaskBase, DEFAULT_CHANNEL_CAPACITY,
};

// Re-export Rel-18 5G-Advanced task types
pub use mint::MintTask;
pub use ranging::RangingTask;
#[cfg(feature = "sidelink")]
pub use sidelink::SidelinkTask;

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
#[cfg(feature = "nextgsim-fl")]
pub use fl_participant::FlParticipantTask;
#[cfg(feature = "nextgsim-isac")]
pub use isac_sensor::IsacSensorTask;
#[cfg(feature = "nextgsim-nwdaf")]
pub use nwdaf_reporter::NwdafReporterTask;
#[cfg(feature = "nextgsim-semantic")]
pub use semantic_codec::SemanticCodecTask;
#[cfg(feature = "nextgsim-she")]
pub use she_client::SheClientTask;
