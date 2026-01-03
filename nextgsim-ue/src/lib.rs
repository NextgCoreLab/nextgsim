//! nextgsim UE (User Equipment) Library
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
// TODO: TUN module has pre-existing compilation errors that need to be fixed
// pub mod tun;

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
pub use rrc::{RrcState, RrcStateMachine, RrcStateTransition, RrcStateError};

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

// TODO: TUN module has pre-existing compilation errors that need to be fixed
// Re-export TUN types
// pub use tun::{IpPacket, IpVersion, TunConfig, TunError, TunInterface, TunTask, TunTaskConfig};
