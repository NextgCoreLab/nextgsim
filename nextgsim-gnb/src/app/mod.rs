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
//! The transport itself lives in [`nextgsim_common::cli_server`] and is shared with
//! the UE. That is the only implementation `nr-cli` can resolve: it registers the
//! node in `PROC_TABLE_DIR`, and that entry is how `nr-cli` turns a node name into
//! a port. This module used to carry a second `CliServer` of its own which bound a
//! port without registering it, so every gNB command was unreachable from outside
//! the process; it also framed messages at protocol version 3.2.7 while `nr-cli`
//! speaks 1.0.0, so even a discovered port would have rejected every datagram.
//! Both defects went away with the duplicate (issue #197).
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb/app/` implementation.

mod cmd_handler;
mod config_loader;
mod status;
mod task;

pub use cmd_handler::{parse_cli_command, AmfContext, CliResponse, GnbCmdHandler, UeContext};

pub use config_loader::{
    load_and_validate_gnb_config, load_gnb_config, load_gnb_config_from_str, validate_gnb_config,
    ConfigError, ConfigValidationError,
};

pub use status::{GnbStatusInfo, StatusReporter};

pub use task::AppTask;
