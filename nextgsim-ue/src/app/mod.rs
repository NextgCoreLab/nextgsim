//! UE Application Task
//!
//! This module implements the application task for the UE, which handles:
//! - Configuration loading and validation
//! - CLI command processing
//! - Status reporting
//! - TUN interface management
//! - Coordination between NAS and user plane
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/app/` implementation.

mod cli_handler;
mod config_loader;
mod status;
mod task;

pub use cli_handler::*;
pub use config_loader::{
    load_and_validate_ue_config, load_ue_config, load_ue_config_from_str, validate_ue_config,
    ConfigError, ConfigValidationError,
};
pub use status::{StatusReporter, TimerInfo, TimersInfo, UeInfo, UeStatusInfo};
pub use task::{parse_ue_cli_command, AppTask};
