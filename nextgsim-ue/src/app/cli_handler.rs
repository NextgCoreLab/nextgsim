//! CLI Command Handler
//!
//! This module implements the CLI command handler for the UE application task.
//! It processes commands received from the CLI tool and triggers the appropriate
//! NAS procedures.
//!
//! # Supported Commands
//!
//! - `info` - Display UE information (SUPI, state, etc.)
//! - `status` - Display UE status
//! - `timers` - Display active timers
//! - `deregister` - Initiate UE deregistration
//! - `ps-establish` - Establish a PDU session
//! - `ps-release` - Release a PDU session
//! - `ps-release-all` - Release all PDU sessions
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/app/cmd_handler.cpp` implementation.

use std::fmt;

use crate::nas::mm::{DeregistrationCause, MmState, MmSubState, RmState};
use crate::nas::sm::{ProcedureTransactionManager, SmMessageType, SM_TIMER_T3580, SM_TIMER_T3582};
use crate::tasks::{UeCliCommand, UeCliCommandType};

/// Result of a CLI command execution.
#[derive(Debug, Clone)]
pub struct CliCommandResult {
    /// Whether the command was successful
    pub success: bool,
    /// Response message to send back to CLI
    pub message: String,
}

impl CliCommandResult {
    /// Creates a successful result with a message.
    pub fn ok(message: impl Into<String>) -> Self {
        Self {
            success: true,
            message: message.into(),
        }
    }

    /// Creates a failure result with an error message.
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            success: false,
            message: message.into(),
        }
    }
}

impl fmt::Display for CliCommandResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.success {
            write!(f, "OK: {}", self.message)
        } else {
            write!(f, "ERROR: {}", self.message)
        }
    }
}

/// PDU session establishment request parameters.
#[derive(Debug, Clone)]
pub struct PduSessionEstablishRequest {
    /// PDU session ID (1-15, 0 means auto-assign)
    pub psi: u8,
    /// Session type (IPv4, IPv6, IPv4v6)
    pub session_type: PduSessionType,
    /// APN/DNN name (optional)
    pub apn: Option<String>,
    /// S-NSSAI (optional)
    pub s_nssai: Option<String>,
}

impl Default for PduSessionEstablishRequest {
    fn default() -> Self {
        Self {
            psi: 0, // Auto-assign
            session_type: PduSessionType::Ipv4,
            apn: None,
            s_nssai: None,
        }
    }
}

/// PDU session type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PduSessionType {
    /// IPv4 only
    #[default]
    Ipv4,
    /// IPv6 only
    Ipv6,
    /// IPv4 and IPv6
    Ipv4v6,
    /// Unstructured
    Unstructured,
    /// Ethernet
    Ethernet,
}

impl PduSessionType {
    /// Parses a session type from a string.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "ipv4" => Some(PduSessionType::Ipv4),
            "ipv6" => Some(PduSessionType::Ipv6),
            "ipv4v6" => Some(PduSessionType::Ipv4v6),
            "unstructured" => Some(PduSessionType::Unstructured),
            "ethernet" => Some(PduSessionType::Ethernet),
            _ => None,
        }
    }
}

impl fmt::Display for PduSessionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PduSessionType::Ipv4 => write!(f, "IPv4"),
            PduSessionType::Ipv6 => write!(f, "IPv6"),
            PduSessionType::Ipv4v6 => write!(f, "IPv4v6"),
            PduSessionType::Unstructured => write!(f, "Unstructured"),
            PduSessionType::Ethernet => write!(f, "Ethernet"),
        }
    }
}

/// PDU session release request parameters.
#[derive(Debug, Clone)]
pub struct PduSessionReleaseRequest {
    /// PDU session ID to release (1-15)
    pub psi: u8,
}

/// Deregistration request parameters.
#[derive(Debug, Clone)]
pub struct DeregistrationRequest {
    /// Whether to perform switch-off (don't wait for response)
    pub switch_off: bool,
}

impl Default for DeregistrationRequest {
    fn default() -> Self {
        Self { switch_off: false }
    }
}

/// Action to be performed by the NAS task after CLI command processing.
#[derive(Debug, Clone)]
pub enum NasAction {
    /// No action needed
    None,
    /// Initiate PDU session establishment
    EstablishPduSession {
        /// PDU session ID (1-15)
        psi: u8,
        /// Procedure Transaction Identity
        pti: u8,
        /// Session type
        session_type: PduSessionType,
        /// APN/DNN
        apn: Option<String>,
    },
    /// Initiate PDU session release
    ReleasePduSession {
        /// PDU session ID (1-15)
        psi: u8,
        /// Procedure Transaction Identity
        pti: u8,
    },
    /// Initiate deregistration
    Deregister {
        /// Deregistration cause
        cause: DeregistrationCause,
    },
}

/// CLI command handler for the UE.
///
/// This struct processes CLI commands and determines the appropriate NAS actions.
#[derive(Debug)]
pub struct CliHandler {
    /// Procedure transaction manager for SM procedures
    pt_manager: ProcedureTransactionManager,
    /// Next available PDU session ID
    next_psi: u8,
    /// Active PDU sessions (PSI -> session info)
    active_sessions: Vec<u8>,
}

impl Default for CliHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CliHandler {
    /// Creates a new CLI handler.
    pub fn new() -> Self {
        Self {
            pt_manager: ProcedureTransactionManager::new(),
            next_psi: 1, // PSI starts at 1
            active_sessions: Vec::new(),
        }
    }

    /// Processes a CLI command and returns the result and any NAS action needed.
    ///
    /// # Arguments
    /// * `command` - The CLI command to process
    /// * `rm_state` - Current RM state
    /// * `mm_state` - Current MM state
    /// * `mm_sub_state` - Current MM sub-state
    ///
    /// # Returns
    /// A tuple of (result, action) where result is the CLI response and action
    /// is the NAS action to perform (if any).
    pub fn handle_command(
        &mut self,
        command: &UeCliCommand,
        rm_state: RmState,
        mm_state: MmState,
        mm_sub_state: MmSubState,
    ) -> (CliCommandResult, NasAction) {
        match &command.command {
            UeCliCommandType::Info => self.handle_info(),
            UeCliCommandType::Status => self.handle_status(rm_state, mm_state, mm_sub_state),
            UeCliCommandType::Timers => self.handle_timers(),
            UeCliCommandType::Deregister { switch_off } => {
                self.handle_deregister(*switch_off, rm_state, mm_state)
            }
            UeCliCommandType::PsEstablish {
                session_type,
                apn,
                s_nssai,
            } => self.handle_ps_establish(
                session_type.as_deref(),
                apn.clone(),
                s_nssai.clone(),
                rm_state,
                mm_state,
            ),
            UeCliCommandType::PsRelease { psi } => {
                self.handle_ps_release(*psi, rm_state, mm_state)
            }
            UeCliCommandType::PsReleaseAll => self.handle_ps_release_all(rm_state, mm_state),
        }
    }

    /// Handles the `info` command.
    fn handle_info(&self) -> (CliCommandResult, NasAction) {
        // TODO: Get actual UE info from config/state
        let info = format!(
            "UE Information:\n  Active PDU Sessions: {:?}\n  Pending Procedures: {}",
            self.active_sessions,
            self.pt_manager.pending_count()
        );
        (CliCommandResult::ok(info), NasAction::None)
    }

    /// Handles the `status` command.
    fn handle_status(
        &self,
        rm_state: RmState,
        mm_state: MmState,
        mm_sub_state: MmSubState,
    ) -> (CliCommandResult, NasAction) {
        use crate::app::UeStatusInfo;
        
        let mut status = UeStatusInfo::new();
        status.set_rm_state(rm_state);
        status.set_mm_state(mm_state);
        status.set_mm_sub_state(mm_sub_state);
        // Add active sessions
        for &psi in &self.active_sessions {
            status.pdu_sessions.push(psi);
        }
        
        match status.to_yaml() {
            Ok(yaml) => (CliCommandResult::ok(yaml), NasAction::None),
            Err(e) => (CliCommandResult::error(format!("Failed to serialize status: {}", e)), NasAction::None),
        }
    }

    /// Handles the `timers` command.
    fn handle_timers(&self) -> (CliCommandResult, NasAction) {
        // TODO: Get actual timer states
        let timers = "Active Timers:\n  (Timer display not yet implemented)";
        (CliCommandResult::ok(timers), NasAction::None)
    }

    /// Handles the `deregister` command.
    fn handle_deregister(
        &mut self,
        switch_off: bool,
        rm_state: RmState,
        mm_state: MmState,
    ) -> (CliCommandResult, NasAction) {
        // Check if UE is registered
        if rm_state != RmState::Registered {
            return (
                CliCommandResult::error(format!(
                    "Cannot deregister: UE is not registered (current state: {})",
                    rm_state
                )),
                NasAction::None,
            );
        }

        // Check if deregistration is already in progress
        if mm_state == MmState::DeregisteredInitiated {
            return (
                CliCommandResult::error("Deregistration already in progress"),
                NasAction::None,
            );
        }

        let cause = if switch_off {
            DeregistrationCause::SwitchOff
        } else {
            DeregistrationCause::ExplicitDeregistration
        };

        (
            CliCommandResult::ok(format!(
                "Initiating deregistration (cause: {})",
                cause
            )),
            NasAction::Deregister { cause },
        )
    }

    /// Handles the `ps-establish` command.
    fn handle_ps_establish(
        &mut self,
        session_type: Option<&str>,
        apn: Option<String>,
        _s_nssai: Option<String>,
        rm_state: RmState,
        mm_state: MmState,
    ) -> (CliCommandResult, NasAction) {
        // Check if UE is registered
        if rm_state != RmState::Registered {
            return (
                CliCommandResult::error(format!(
                    "Cannot establish PDU session: UE is not registered (current state: {})",
                    rm_state
                )),
                NasAction::None,
            );
        }

        // Check MM state
        if mm_state != MmState::Registered {
            return (
                CliCommandResult::error(format!(
                    "Cannot establish PDU session: Invalid MM state ({})",
                    mm_state
                )),
                NasAction::None,
            );
        }

        // Parse session type
        let session_type = session_type
            .and_then(PduSessionType::from_str)
            .unwrap_or(PduSessionType::Ipv4);

        // Allocate PTI
        let pti = match self.pt_manager.allocate() {
            Some(pti) => pti,
            None => {
                return (
                    CliCommandResult::error("No PTI available for new procedure"),
                    NasAction::None,
                );
            }
        };

        // Allocate PSI
        let psi = self.allocate_psi();
        if psi == 0 {
            self.pt_manager.free(pti);
            return (
                CliCommandResult::error("No PSI available for new session"),
                NasAction::None,
            );
        }

        // Start the procedure transaction
        if let Some(pt) = self.pt_manager.get_mut(pti) {
            pt.start(psi, SmMessageType::PduSessionEstablishmentRequest, SM_TIMER_T3580);
        }

        (
            CliCommandResult::ok(format!(
                "Initiating PDU session establishment (PSI: {}, PTI: {}, Type: {})",
                psi, pti, session_type
            )),
            NasAction::EstablishPduSession {
                psi,
                pti,
                session_type,
                apn,
            },
        )
    }

    /// Handles the `ps-release` command.
    fn handle_ps_release(
        &mut self,
        psi: i32,
        rm_state: RmState,
        mm_state: MmState,
    ) -> (CliCommandResult, NasAction) {
        // Validate PSI
        if psi < 1 || psi > 15 {
            return (
                CliCommandResult::error(format!("Invalid PSI: {} (must be 1-15)", psi)),
                NasAction::None,
            );
        }

        let psi = psi as u8;

        // Check if session exists
        if !self.active_sessions.contains(&psi) {
            return (
                CliCommandResult::error(format!("PDU session {} does not exist", psi)),
                NasAction::None,
            );
        }

        // Check if UE is registered
        if rm_state != RmState::Registered {
            return (
                CliCommandResult::error(format!(
                    "Cannot release PDU session: UE is not registered (current state: {})",
                    rm_state
                )),
                NasAction::None,
            );
        }

        // Check MM state
        if mm_state != MmState::Registered {
            return (
                CliCommandResult::error(format!(
                    "Cannot release PDU session: Invalid MM state ({})",
                    mm_state
                )),
                NasAction::None,
            );
        }

        // Allocate PTI
        let pti = match self.pt_manager.allocate() {
            Some(pti) => pti,
            None => {
                return (
                    CliCommandResult::error("No PTI available for release procedure"),
                    NasAction::None,
                );
            }
        };

        // Start the procedure transaction
        if let Some(pt) = self.pt_manager.get_mut(pti) {
            pt.start(psi, SmMessageType::PduSessionReleaseRequest, SM_TIMER_T3582);
        }

        (
            CliCommandResult::ok(format!(
                "Initiating PDU session release (PSI: {}, PTI: {})",
                psi, pti
            )),
            NasAction::ReleasePduSession { psi, pti },
        )
    }

    /// Handles the `ps-release-all` command.
    fn handle_ps_release_all(
        &mut self,
        rm_state: RmState,
        mm_state: MmState,
    ) -> (CliCommandResult, NasAction) {
        if self.active_sessions.is_empty() {
            return (
                CliCommandResult::ok("No active PDU sessions to release"),
                NasAction::None,
            );
        }

        // For now, just release the first session
        // In a full implementation, this would queue releases for all sessions
        let psi = self.active_sessions[0];
        self.handle_ps_release(psi as i32, rm_state, mm_state)
    }

    /// Allocates a new PSI (1-15).
    fn allocate_psi(&mut self) -> u8 {
        // Find an unused PSI
        for psi in 1..=15u8 {
            if !self.active_sessions.contains(&psi) {
                return psi;
            }
        }
        0 // No PSI available
    }

    /// Marks a PDU session as established.
    pub fn on_session_established(&mut self, psi: u8) {
        if !self.active_sessions.contains(&psi) {
            self.active_sessions.push(psi);
        }
    }

    /// Marks a PDU session as released.
    pub fn on_session_released(&mut self, psi: u8) {
        self.active_sessions.retain(|&p| p != psi);
    }

    /// Completes a procedure transaction.
    pub fn complete_procedure(&mut self, pti: u8) {
        self.pt_manager.free(pti);
    }

    /// Aborts a procedure transaction.
    pub fn abort_procedure(&mut self, pti: u8) {
        self.pt_manager.abort(pti);
    }

    /// Returns the number of active PDU sessions.
    pub fn active_session_count(&self) -> usize {
        self.active_sessions.len()
    }

    /// Returns the list of active PDU session IDs.
    pub fn active_sessions(&self) -> &[u8] {
        &self.active_sessions
    }

    /// Returns a reference to the procedure transaction manager.
    pub fn pt_manager(&self) -> &ProcedureTransactionManager {
        &self.pt_manager
    }

    /// Returns a mutable reference to the procedure transaction manager.
    pub fn pt_manager_mut(&mut self) -> &mut ProcedureTransactionManager {
        &mut self.pt_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_command_result_display() {
        let ok = CliCommandResult::ok("Success");
        assert_eq!(format!("{}", ok), "OK: Success");

        let err = CliCommandResult::error("Failed");
        assert_eq!(format!("{}", err), "ERROR: Failed");
    }

    #[test]
    fn test_pdu_session_type_from_str() {
        assert_eq!(PduSessionType::from_str("ipv4"), Some(PduSessionType::Ipv4));
        assert_eq!(PduSessionType::from_str("IPv6"), Some(PduSessionType::Ipv6));
        assert_eq!(
            PduSessionType::from_str("IPV4V6"),
            Some(PduSessionType::Ipv4v6)
        );
        assert_eq!(PduSessionType::from_str("invalid"), None);
    }

    #[test]
    fn test_pdu_session_type_display() {
        assert_eq!(format!("{}", PduSessionType::Ipv4), "IPv4");
        assert_eq!(format!("{}", PduSessionType::Ipv6), "IPv6");
        assert_eq!(format!("{}", PduSessionType::Ipv4v6), "IPv4v6");
    }

    #[test]
    fn test_cli_handler_info() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::Info,
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_status() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::Status,
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(result.message.contains("RM State: RM-REGISTERED"));
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_deregister_not_registered() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::Deregister { switch_off: false },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Deregistered,
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
        );
        assert!(!result.success);
        assert!(result.message.contains("not registered"));
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_deregister_success() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::Deregister { switch_off: false },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(matches!(
            action,
            NasAction::Deregister {
                cause: DeregistrationCause::ExplicitDeregistration
            }
        ));
    }

    #[test]
    fn test_cli_handler_deregister_switch_off() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::Deregister { switch_off: true },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(matches!(
            action,
            NasAction::Deregister {
                cause: DeregistrationCause::SwitchOff
            }
        ));
    }

    #[test]
    fn test_cli_handler_ps_establish_not_registered() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::PsEstablish {
                session_type: None,
                apn: None,
                s_nssai: None,
            },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Deregistered,
            MmState::Deregistered,
            MmSubState::DeregisteredNormalService,
        );
        assert!(!result.success);
        assert!(result.message.contains("not registered"));
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_ps_establish_success() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::PsEstablish {
                session_type: Some("IPv4".to_string()),
                apn: Some("internet".to_string()),
                s_nssai: None,
            },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(result.message.contains("Initiating PDU session establishment"));
        match action {
            NasAction::EstablishPduSession {
                psi,
                pti,
                session_type,
                apn,
            } => {
                assert!(psi >= 1 && psi <= 15);
                assert!(pti >= 1);
                assert_eq!(session_type, PduSessionType::Ipv4);
                assert_eq!(apn, Some("internet".to_string()));
            }
            _ => panic!("Expected EstablishPduSession action"),
        }
    }

    #[test]
    fn test_cli_handler_ps_release_invalid_psi() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::PsRelease { psi: 0 },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(!result.success);
        assert!(result.message.contains("Invalid PSI"));
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_ps_release_not_exists() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::PsRelease { psi: 5 },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(!result.success);
        assert!(result.message.contains("does not exist"));
        assert!(matches!(action, NasAction::None));
    }

    #[test]
    fn test_cli_handler_ps_release_success() {
        let mut handler = CliHandler::new();
        // First establish a session
        handler.on_session_established(5);

        let cmd = UeCliCommand {
            command: UeCliCommandType::PsRelease { psi: 5 },
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(result.message.contains("Initiating PDU session release"));
        match action {
            NasAction::ReleasePduSession { psi, pti } => {
                assert_eq!(psi, 5);
                assert!(pti >= 1);
            }
            _ => panic!("Expected ReleasePduSession action"),
        }
    }

    #[test]
    fn test_cli_handler_session_lifecycle() {
        let mut handler = CliHandler::new();

        assert_eq!(handler.active_session_count(), 0);

        handler.on_session_established(1);
        handler.on_session_established(3);
        assert_eq!(handler.active_session_count(), 2);
        assert_eq!(handler.active_sessions(), &[1, 3]);

        handler.on_session_released(1);
        assert_eq!(handler.active_session_count(), 1);
        assert_eq!(handler.active_sessions(), &[3]);
    }

    #[test]
    fn test_cli_handler_ps_release_all_empty() {
        let mut handler = CliHandler::new();
        let cmd = UeCliCommand {
            command: UeCliCommandType::PsReleaseAll,
            response_addr: None,
        };
        let (result, action) = handler.handle_command(
            &cmd,
            RmState::Registered,
            MmState::Registered,
            MmSubState::RegisteredNormalService,
        );
        assert!(result.success);
        assert!(result.message.contains("No active PDU sessions"));
        assert!(matches!(action, NasAction::None));
    }
}
