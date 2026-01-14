//! gNB Application Task Module
//!
//! This module implements the application task for the gNB, which handles:
//! - Configuration loading and validation
//! - CLI command handling
//! - Status reporting
//!
//! # Architecture
//!
//! The App task is the central coordinator for the gNB. It receives status
//! updates from other tasks and handles CLI commands from external clients.
//!
//! # CLI Protocol
//!
//! The CLI uses a UDP-based IPC protocol for communication between the CLI tool
//! and the running gNB instance. Commands are sent as structured messages with
//! version checking for compatibility.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb/app/` implementation.

mod cli_server;
mod cmd_handler;
mod config_loader;
mod status;
mod task;

pub use cli_server::{
    CliMessage, CliMessageType, CliServer, CliServerError, CLI_BUFFER_SIZE, CLI_MIN_LENGTH,
    CLI_RECV_TIMEOUT_MS, CLI_VERSION_MAJOR, CLI_VERSION_MINOR, CLI_VERSION_PATCH,
};

pub use cmd_handler::{
    parse_cli_command, AmfContext, CliResponse, GnbCmdHandler, UeContext,
};

pub use config_loader::{
    load_and_validate_gnb_config, load_gnb_config, load_gnb_config_from_str, validate_gnb_config,
    ConfigError, ConfigValidationError,
};

pub use status::{GnbStatusInfo, StatusReporter};

pub use task::AppTask;
