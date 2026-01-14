//! nextgsim-gnb - 5G gNB (gNodeB) Simulator Library
#![allow(missing_docs)]
//!
//! This crate provides the gNB implementation for the nextgsim 5G simulator.
//! It implements the gNodeB functionality including:
//!
//! - NGAP protocol handling for AMF communication
//! - RRC protocol handling for UE management
//! - GTP-U tunnel management for user plane
//! - RLS (Radio Link Simulation) for UE discovery
//! - SCTP transport for NGAP
//!
//! # Architecture
//!
//! The gNB uses an actor-based task model where each component runs as an
//! independent async task communicating via typed message channels.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                         gNB                                  │
//! │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
//! │  │   App   │  │  NGAP   │  │   RRC   │  │   GTP   │        │
//! │  │  Task   │  │  Task   │  │  Task   │  │  Task   │        │
//! │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘        │
//! │       │            │            │            │              │
//! │       └────────────┴────────────┴────────────┘              │
//! │                         │                                    │
//! │  ┌─────────┐      ┌─────┴─────┐                             │
//! │  │  SCTP   │      │    RLS    │                             │
//! │  │  Task   │      │   Task    │                             │
//! │  └────┬────┘      └─────┬─────┘                             │
//! │       │                 │                                    │
//! └───────┼─────────────────┼────────────────────────────────────┘
//!         │                 │
//!         ▼                 ▼
//!       AMF               UEs
//! ```
//!
//! # Task Lifecycle
//!
//! Tasks are managed by `TaskManager` which handles:
//! - Task spawning and initialization
//! - Health monitoring and state tracking
//! - Graceful shutdown coordination
//! - Error handling and recovery
//!
//! # Configuration Loading
//!
//! The `app` module provides configuration loading and validation:
//!
//! ```rust,ignore
//! use nextgsim_gnb::app::{load_gnb_config, validate_gnb_config};
//!
//! let config = load_gnb_config("config/gnb.yaml")?;
//! validate_gnb_config(&config)?;
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use nextgsim_gnb::tasks::{TaskManager, DEFAULT_CHANNEL_CAPACITY};
//! use nextgsim_gnb::app::load_and_validate_gnb_config;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = load_and_validate_gnb_config("config/gnb.yaml").unwrap();
//!     let (mut manager, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx) =
//!         TaskManager::new(config, DEFAULT_CHANNEL_CAPACITY);
//!
//!     // Spawn tasks with their receivers...
//!     // manager.shutdown().await;
//! }
//! ```

pub mod app;
pub mod gtp;
pub mod ngap;
pub mod rls;
pub mod rrc;
pub mod sctp;
pub mod tasks;

// Re-export NGAP module types
pub use ngap::{AmfContextInfo, AmfState, NgapAmfContext, NgapTask, NgapUeContext, UeState};

// Re-export RRC module types
pub use rrc::{
    RrcConnectionManager, RrcReleaseResult, RrcSetupCompleteResult,
    RrcSetupResult, RrcState, RrcTask, RrcUeContext, RrcUeContextManager,
};

// Re-export SCTP module types
pub use sctp::{AmfConnection, AmfConnectionConfig, AmfConnectionEvent, AmfConnectionState, SctpTask};

// Re-export GTP module types
pub use gtp::GtpTask;

// Re-export RLS module types
pub use rls::RlsTask;

// Re-export app module types
pub use app::{
    load_and_validate_gnb_config, load_gnb_config, load_gnb_config_from_str, validate_gnb_config,
    parse_cli_command, AmfContext, AppTask, CliMessage, CliMessageType, CliResponse, CliServer,
    CliServerError, ConfigError, ConfigValidationError, GnbCmdHandler, GnbStatusInfo,
    StatusReporter, UeContext, CLI_BUFFER_SIZE, CLI_MIN_LENGTH, CLI_RECV_TIMEOUT_MS,
    CLI_VERSION_MAJOR, CLI_VERSION_MINOR, CLI_VERSION_PATCH,
};

// Re-export commonly used types
pub use tasks::{
    AppMessage, CliCommand, GnbCliCommandType, GnbTaskBase, GtpMessage, GtpUeContextUpdate,
    GutiMobileIdentity, NgapMessage, PduSessionResource, RlfCause, RlsMessage, RrcMessage,
    SctpMessage, StatusType, StatusUpdate, Task, TaskHandle, TaskMessage, DEFAULT_CHANNEL_CAPACITY,
    NGAP_PPID,
};

// Re-export lifecycle management types
pub use tasks::{
    TaskError, TaskId, TaskInfo, TaskManager, TaskState, DEFAULT_SHUTDOWN_TIMEOUT_MS,
};
