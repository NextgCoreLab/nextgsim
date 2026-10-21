//! UE Application Task
//!
//! This module implements the main application task for the UE, which handles:
//! - CLI command processing via UDP server
//! - Status reporting and tracking
//! - TUN interface management
//! - Coordination between NAS and user plane
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue/app/task.cpp` implementation.

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use nextgsim_common::cli_server::{CliCommand, CliServer};

use crate::app::{CliHandler, NasAction};
use crate::nas::mm::{MmState, MmStateMachine, MmSubState, RmState};
use crate::tasks::{
    AppMessage, NasMessage, Task, TaskMessage, UeCliCommand, UeCliCommandType, UeStatusUpdate,
    UeTaskBase,
};

/// Tracks the 5GMM state last published to the App task, so the NAS task
/// sends one [`UeStatusUpdate::MmStateChanged`] per real change rather than
/// one per loop iteration (issue #266).
#[derive(Debug, Default)]
pub struct MmStatePublisher {
    last: Option<(RmState, MmState, MmSubState)>,
}

impl MmStatePublisher {
    /// Creates a publisher that has published nothing yet, so the first
    /// [`Self::poll`] always reports.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the update to send when `state` differs from what was last
    /// published, and records it as published.
    pub fn poll(&mut self, state: &MmStateMachine) -> Option<UeStatusUpdate> {
        let now = (state.rm_state(), state.mm_state(), state.mm_substate());
        if self.last == Some(now) {
            return None;
        }
        self.last = Some(now);
        Some(UeStatusUpdate::MmStateChanged {
            rm_state: now.0,
            mm_state: now.1,
            mm_sub_state: now.2,
        })
    }
}

/// UE Application Task
///
/// Handles CLI commands, status updates, TUN interface, and coordination between tasks.
pub struct AppTask {
    /// Task base for inter-task communication
    task_base: UeTaskBase,
    /// CLI server for receiving commands
    cli_server: Option<CliServer>,
    /// CLI command handler
    cli_handler: CliHandler,
    /// Current RM state
    rm_state: RmState,
    /// Current MM state
    mm_state: MmState,
    /// Current MM sub-state
    mm_sub_state: MmSubState,
    /// Whether CLI is enabled
    cli_enabled: bool,
}

impl AppTask {
    /// Creates a new App task.
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            task_base,
            cli_server: None,
            cli_handler: CliHandler::new(),
            rm_state: RmState::Deregistered,
            mm_state: MmState::Deregistered,
            mm_sub_state: MmSubState::DeregisteredNormalService,
            cli_enabled: true,
        }
    }

    /// Creates a new App task with CLI disabled.
    pub fn new_without_cli(task_base: UeTaskBase) -> Self {
        Self {
            task_base,
            cli_server: None,
            cli_handler: CliHandler::new(),
            rm_state: RmState::Deregistered,
            mm_state: MmState::Deregistered,
            mm_sub_state: MmSubState::DeregisteredNormalService,
            cli_enabled: false,
        }
    }

    /// Initializes the CLI server.
    ///
    /// This should be called before running the task if CLI is enabled.
    pub async fn init_cli_server(&mut self, node_names: Vec<String>) -> std::io::Result<u16> {
        if !self.cli_enabled {
            return Ok(0);
        }

        let mut server = CliServer::new().await?;
        server.register_nodes(node_names)?;
        let port = server.port();
        self.cli_server = Some(server);
        Ok(port)
    }

    /// Returns the CLI server port, or 0 if CLI is disabled.
    pub fn cli_port(&self) -> u16 {
        self.cli_server
            .as_ref()
            .map(nextgsim_common::CliServer::port)
            .unwrap_or(0)
    }

    /// Handles a status update.
    fn handle_status_update(&mut self, update: UeStatusUpdate) {
        debug!("Status update: {:?}", update);
        match update {
            UeStatusUpdate::SessionEstablishment { psi } => {
                self.cli_handler.on_session_established(psi as u8);
            }
            UeStatusUpdate::SessionRelease { psi } => {
                self.cli_handler.on_session_released(psi as u8);
            }
            UeStatusUpdate::CmStateChanged { cm_state } => {
                info!("CM state changed to: {}", cm_state);
            }
            UeStatusUpdate::MmStateChanged {
                rm_state,
                mm_state,
                mm_sub_state,
            } => {
                self.rm_state = rm_state;
                self.mm_state = mm_state;
                self.mm_sub_state = mm_sub_state;
            }
        }
    }

    /// Handles a CLI command.
    async fn handle_cli_command(&mut self, command: UeCliCommand) {
        debug!("CLI command: {:?}", command.command);

        // Process the command
        let (result, action) = self.cli_handler.handle_command(
            &command,
            self.rm_state,
            self.mm_state,
            self.mm_sub_state,
        );

        // Send response if we have a CLI server and destination
        if let (Some(server), Some(addr)) = (&self.cli_server, command.response_addr) {
            let send_result = if result.success {
                server.send_result(addr, &result.message).await
            } else {
                server.send_error(addr, &result.message).await
            };

            if let Err(e) = send_result {
                warn!("Failed to send CLI response: {}", e);
            }
        }

        // Execute the NAS action if any
        self.execute_nas_action(action).await;
    }

    /// Executes a NAS action resulting from a CLI command.
    async fn execute_nas_action(&self, action: NasAction) {
        match action {
            NasAction::None => {}
            NasAction::EstablishPduSession {
                psi,
                pti,
                session_type,
                apn,
            } => {
                info!(
                    "Initiating PDU session establishment: PSI={}, PTI={}, type={:?}, apn={:?}",
                    psi, pti, session_type, apn
                );
                let msg = NasMessage::InitiatePduSessionEstablishment {
                    psi,
                    pti,
                    session_type: format!("{session_type:?}"),
                    apn,
                };
                if let Err(e) = self.task_base.nas_tx.send(msg).await {
                    error!("Failed to send PDU session establishment request: {}", e);
                }
            }
            NasAction::ReleasePduSession { psi, pti } => {
                info!("Initiating PDU session release: PSI={}, PTI={}", psi, pti);
                let msg = NasMessage::InitiatePduSessionRelease { psi, pti };
                if let Err(e) = self.task_base.nas_tx.send(msg).await {
                    error!("Failed to send PDU session release request: {}", e);
                }
            }
            NasAction::ReleaseAllPduSessions { sessions } => {
                info!("Releasing {} PDU session(s) sequentially", sessions.len());
                for (psi, pti) in sessions {
                    info!("Initiating PDU session release: PSI={}, PTI={}", psi, pti);
                    let msg = NasMessage::InitiatePduSessionRelease { psi, pti };
                    if let Err(e) = self.task_base.nas_tx.send(msg).await {
                        error!("Failed to send PDU session release for PSI={}: {}", psi, e);
                        break;
                    }
                    // Small delay between releases to avoid race conditions
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                }
            }
            NasAction::Deregister { cause } => {
                use crate::nas::mm::DeregistrationCause;
                let switch_off = matches!(cause, DeregistrationCause::SwitchOff);
                info!(
                    "Initiating deregistration: cause={}, switch_off={}",
                    cause, switch_off
                );
                let msg = NasMessage::InitiateDeregistration { switch_off };
                if let Err(e) = self.task_base.nas_tx.send(msg).await {
                    error!("Failed to send deregistration request: {}", e);
                }
            }
            NasAction::EmergencyRegister => {
                info!("CLI: triggering emergency registration");
                if let Err(e) = self
                    .task_base
                    .nas_tx
                    .send(NasMessage::InitiateEmergencyRegistration)
                    .await
                {
                    error!("Failed to send emergency registration request: {}", e);
                }
            }
        }
    }

    /// Processes a raw CLI command from the server.
    async fn process_cli_command(&mut self, cmd: CliCommand) {
        // Parse the command string into a UeCliCommandType
        let command_type = match parse_ue_cli_command(&cmd.command) {
            Ok(cmd_type) => cmd_type,
            Err(e) => {
                // Send error response
                if let Some(server) = &self.cli_server {
                    let _ = server.send_error(cmd.client_addr, &e).await;
                }
                return;
            }
        };

        // Create UeCliCommand and handle it
        let ue_cmd = UeCliCommand {
            command: command_type,
            response_addr: Some(cmd.client_addr),
        };

        self.handle_cli_command(ue_cmd).await;
    }

    /// Polls the CLI server for incoming commands.
    ///
    /// Non-blocking, as the gNB's App task is (nextgsim #197): awaiting a
    /// datagram here parked the whole `select!` until a CLI client sent one, so
    /// no inter-task message -- the NAS task's state updates among them -- was
    /// read in between, and a full channel would back-pressure the NAS task.
    async fn poll_cli_server(&mut self) {
        let cmd = match &self.cli_server {
            Some(server) => match server.try_receive_command() {
                Ok(cmd) => cmd,
                Err(e) => {
                    warn!("CLI server error: {}", e);
                    None
                }
            },
            None => None,
        };
        if let Some(cmd) = cmd {
            self.process_cli_command(cmd).await;
        }
    }

    /// Handles TUN data delivery (uplink).
    async fn handle_tun_data(&self, psi: i32, data: nextgsim_common::OctetString) {
        debug!("TUN data delivery: psi={}, len={}", psi, data.len());
        // Forward to NAS for GTP encapsulation
        let msg = NasMessage::UplinkDataDelivery { psi, data };
        if let Err(e) = self.task_base.nas_tx.send(msg).await {
            error!("Failed to send uplink data to NAS: {}", e);
        }
    }

    /// Handles downlink data delivery (to TUN).
    ///
    /// Note: In the current architecture, downlink data forwarding to TUN is handled
    /// directly in main.rs via the `NasMessage::DownlinkDataDelivery` message, as the
    /// TUN task channel is created there.
    fn handle_downlink_data(&self, psi: i32, data: nextgsim_common::OctetString) {
        debug!("Downlink data delivery: psi={}, len={}", psi, data.len());
        // Downlink data forwarding to TUN is handled in main.rs
    }

    /// Feeds one message to the task the way its run loop does, without the
    /// loop or the CLI server. Returns `false` when the message would end the
    /// loop. Lets a test drive the App task with the NAS task's real updates.
    pub async fn handle_message(&mut self, msg: AppMessage) -> bool {
        match msg {
            AppMessage::StatusUpdate(update) => self.handle_status_update(update),
            AppMessage::CliCommand(cmd) => self.handle_cli_command(cmd).await,
            AppMessage::TunDataDelivery { psi, data } => self.handle_tun_data(psi, data).await,
            AppMessage::TunError { error } => error!("TUN error: {}", error),
            AppMessage::DownlinkDataDelivery { psi, data } => self.handle_downlink_data(psi, data),
            AppMessage::PerformSwitchOff => {
                info!("Performing switch off");
                return false;
            }
        }
        true
    }

    /// Updates the RM state.
    pub fn set_rm_state(&mut self, state: RmState) {
        self.rm_state = state;
    }

    /// Updates the MM state.
    pub fn set_mm_state(&mut self, state: MmState) {
        self.mm_state = state;
    }

    /// Updates the MM sub-state.
    pub fn set_mm_sub_state(&mut self, sub_state: MmSubState) {
        self.mm_sub_state = sub_state;
    }
}

/// Parses a CLI command string into a `UeCliCommandType`.
///
/// # Supported Commands
///
/// - `info` - Show UE information
/// - `status` - Show UE status
/// - `timers` - Show active timers
/// - `deregister [normal|switch-off]` - Deregister from network (`--switch-off`/`-s` also accepted)
/// - `ps-establish [--type <type>] [--apn <apn>] [--sst <sst>]` - Establish PDU session
/// - `ps-release <psi>` - Release PDU session
/// - `ps-release-all` - Release all PDU sessions
/// - `emergency-register` - Initiate emergency registration (from DEREGISTERED state)
///
/// # Returns
///
/// * `Ok(UeCliCommandType)` - Successfully parsed command
/// * `Err(String)` - Parse error with description
pub fn parse_ue_cli_command(input: &str) -> Result<UeCliCommandType, String> {
    let tokens: Vec<&str> = input.split_whitespace().collect();

    if tokens.is_empty() {
        return Err("Empty command".to_string());
    }

    match tokens[0].to_lowercase().as_str() {
        "info" => Ok(UeCliCommandType::Info),
        "status" => Ok(UeCliCommandType::Status),
        "timers" => Ok(UeCliCommandType::Timers),
        "deregister" => {
            // UERANSIM's nr-cli form `deregister <type>`, plus the older flag
            // spelling. The type is the TS 24.501 §9.11.3.20 switch-off bit:
            // "normal" sends a de-registration the AMF answers with a
            // DEREGISTRATION ACCEPT, "switch-off" one it does not answer.
            let mut switch_off = None;
            for &t in &tokens[1..] {
                let this = match t.to_lowercase().as_str() {
                    "normal" => false,
                    "switch-off" | "--switch-off" | "-s" => true,
                    "disable-5g" | "remove-sim" => {
                        return Err(format!(
                            "deregister {t} is not supported (supported: normal, switch-off)"
                        ))
                    }
                    other => {
                        return Err(format!(
                            "Unknown de-registration type: {other} (expected normal or switch-off)"
                        ))
                    }
                };
                if switch_off.is_some_and(|s| s != this) {
                    return Err("Conflicting de-registration types".to_string());
                }
                switch_off = Some(this);
            }
            Ok(UeCliCommandType::Deregister {
                switch_off: switch_off.unwrap_or(false),
            })
        }
        "ps-establish" => {
            let mut session_type = None;
            let mut apn = None;
            let mut s_nssai = None;

            let mut i = 1;
            while i < tokens.len() {
                match tokens[i] {
                    "--type" | "-t" if i + 1 < tokens.len() => {
                        session_type = Some(tokens[i + 1].to_string());
                        i += 1;
                    }
                    "--apn" | "-a" if i + 1 < tokens.len() => {
                        apn = Some(tokens[i + 1].to_string());
                        i += 1;
                    }
                    "--sst" | "-s" if i + 1 < tokens.len() => {
                        s_nssai = Some(tokens[i + 1].to_string());
                        i += 1;
                    }
                    _ => {}
                }
                i += 1;
            }

            Ok(UeCliCommandType::PsEstablish {
                session_type,
                apn,
                s_nssai,
            })
        }
        "ps-release" => {
            if tokens.len() < 2 {
                return Err("ps-release requires a PSI argument".to_string());
            }
            let psi = tokens[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid PSI: {}", tokens[1]))?;
            Ok(UeCliCommandType::PsRelease { psi })
        }
        "ps-release-all" => Ok(UeCliCommandType::PsReleaseAll),
        "emergency-register" => Ok(UeCliCommandType::EmergencyRegister),
        "ps-list" => {
            // Alias for status (shows PDU sessions)
            Ok(UeCliCommandType::Status)
        }
        _ => Err(format!("Unknown command: {}", tokens[0])),
    }
}

#[async_trait::async_trait]
impl Task for AppTask {
    type Message = AppMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("App task started");

        if self.cli_enabled {
            if let Some(server) = &self.cli_server {
                info!("CLI server listening on port {}", server.port());
            }
        }

        loop {
            tokio::select! {
                // Handle messages from other tasks
                msg = rx.recv() => {
                    match msg {
                        Some(TaskMessage::Message(app_msg)) => {
                            if !self.handle_message(app_msg).await {
                                break;
                            }
                        }
                        Some(TaskMessage::Shutdown) => {
                            info!("App task received shutdown signal");
                            break;
                        }
                        None => {
                            info!("App task channel closed");
                            break;
                        }
                    }
                }

                // Poll CLI server periodically
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                    self.poll_cli_server().await;
                }
            }
        }

        info!("App task stopped");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{NasMessage, RlsMessage, RrcMessage, TaskHandle};
    use nextgsim_common::config::UeConfig;
    use std::sync::Arc;

    fn create_task_base() -> UeTaskBase {
        let config = UeConfig::default();
        let (app_tx, _) = mpsc::channel::<TaskMessage<AppMessage>>(1);
        let (nas_tx, _) = mpsc::channel::<TaskMessage<NasMessage>>(1);
        let (rrc_tx, _) = mpsc::channel::<TaskMessage<RrcMessage>>(1);
        let (rls_tx, _) = mpsc::channel::<TaskMessage<RlsMessage>>(1);

        UeTaskBase {
            config: Arc::new(config),
            app_tx: TaskHandle::new(app_tx),
            nas_tx: TaskHandle::new(nas_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            rls_tx: TaskHandle::new(rls_tx),
            #[cfg(any(
                feature = "nextgsim-she",
                feature = "nextgsim-nwdaf",
                feature = "nextgsim-isac",
                feature = "nextgsim-fl",
                feature = "nextgsim-semantic",
            ))]
            sixg: None,
            rel18: None,
        }
    }

    #[test]
    fn test_app_task_creation() {
        let task_base = create_task_base();
        let task = AppTask::new(task_base);
        assert!(task.cli_enabled);
    }

    #[test]
    fn test_app_task_without_cli() {
        let task_base = create_task_base();
        let task = AppTask::new_without_cli(task_base);
        assert!(!task.cli_enabled);
    }

    #[test]
    fn test_parse_info_command() {
        let cmd = parse_ue_cli_command("info").unwrap();
        assert!(matches!(cmd, UeCliCommandType::Info));
    }

    #[test]
    fn test_parse_status_command() {
        let cmd = parse_ue_cli_command("status").unwrap();
        assert!(matches!(cmd, UeCliCommandType::Status));
    }

    #[test]
    fn test_parse_timers_command() {
        let cmd = parse_ue_cli_command("timers").unwrap();
        assert!(matches!(cmd, UeCliCommandType::Timers));
    }

    #[test]
    fn test_parse_deregister_command() {
        let cmd = parse_ue_cli_command("deregister").unwrap();
        assert!(matches!(
            cmd,
            UeCliCommandType::Deregister { switch_off: false }
        ));

        let cmd = parse_ue_cli_command("deregister --switch-off").unwrap();
        assert!(matches!(
            cmd,
            UeCliCommandType::Deregister { switch_off: true }
        ));

        let cmd = parse_ue_cli_command("deregister -s").unwrap();
        assert!(matches!(
            cmd,
            UeCliCommandType::Deregister { switch_off: true }
        ));
    }

    #[test]
    fn test_parse_ps_establish_command() {
        let cmd = parse_ue_cli_command("ps-establish").unwrap();
        match cmd {
            UeCliCommandType::PsEstablish {
                session_type,
                apn,
                s_nssai,
            } => {
                assert!(session_type.is_none());
                assert!(apn.is_none());
                assert!(s_nssai.is_none());
            }
            _ => panic!("Expected PsEstablish"),
        }

        let cmd = parse_ue_cli_command("ps-establish --type IPv4 --apn internet").unwrap();
        match cmd {
            UeCliCommandType::PsEstablish {
                session_type,
                apn,
                s_nssai,
            } => {
                assert_eq!(session_type, Some("IPv4".to_string()));
                assert_eq!(apn, Some("internet".to_string()));
                assert!(s_nssai.is_none());
            }
            _ => panic!("Expected PsEstablish"),
        }
    }

    #[test]
    fn test_parse_ps_release_command() {
        let cmd = parse_ue_cli_command("ps-release 5").unwrap();
        assert!(matches!(cmd, UeCliCommandType::PsRelease { psi: 5 }));
    }

    #[test]
    fn test_parse_ps_release_missing_psi() {
        let result = parse_ue_cli_command("ps-release");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ps_release_invalid_psi() {
        let result = parse_ue_cli_command("ps-release abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_ps_release_all_command() {
        let cmd = parse_ue_cli_command("ps-release-all").unwrap();
        assert!(matches!(cmd, UeCliCommandType::PsReleaseAll));
    }

    #[test]
    fn test_parse_ps_list_command() {
        let cmd = parse_ue_cli_command("ps-list").unwrap();
        assert!(matches!(cmd, UeCliCommandType::Status));
    }

    #[test]
    fn test_parse_empty_command() {
        let result = parse_ue_cli_command("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_unknown_command() {
        let result = parse_ue_cli_command("unknown");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert!(parse_ue_cli_command("INFO").is_ok());
        assert!(parse_ue_cli_command("Status").is_ok());
        assert!(parse_ue_cli_command("PS-ESTABLISH").is_ok());
    }

    #[test]
    fn test_status_update_session_establishment() {
        let task_base = create_task_base();
        let mut task = AppTask::new(task_base);

        task.handle_status_update(UeStatusUpdate::SessionEstablishment { psi: 5 });
        assert_eq!(task.cli_handler.active_session_count(), 1);
        assert!(task.cli_handler.active_sessions().contains(&5));
    }

    #[test]
    fn test_status_update_session_release() {
        let task_base = create_task_base();
        let mut task = AppTask::new(task_base);

        task.handle_status_update(UeStatusUpdate::SessionEstablishment { psi: 5 });
        task.handle_status_update(UeStatusUpdate::SessionRelease { psi: 5 });
        assert_eq!(task.cli_handler.active_session_count(), 0);
    }

    #[test]
    fn test_state_setters() {
        let task_base = create_task_base();
        let mut task = AppTask::new(task_base);

        task.set_rm_state(RmState::Registered);
        assert_eq!(task.rm_state, RmState::Registered);

        task.set_mm_state(MmState::Registered);
        assert_eq!(task.mm_state, MmState::Registered);

        task.set_mm_sub_state(MmSubState::RegisteredNormalService);
        assert_eq!(task.mm_sub_state, MmSubState::RegisteredNormalService);
    }

    /// A task base whose NAS receiver is kept, so a test can see what the App
    /// task asks the NAS task to do.
    fn create_task_base_with_nas_rx() -> (UeTaskBase, mpsc::Receiver<TaskMessage<NasMessage>>) {
        let mut base = create_task_base();
        let (nas_tx, nas_rx) = mpsc::channel::<TaskMessage<NasMessage>>(8);
        base.nas_tx = TaskHandle::new(nas_tx);
        (base, nas_rx)
    }

    fn cli(command: UeCliCommandType) -> AppMessage {
        AppMessage::CliCommand(UeCliCommand {
            command,
            response_addr: None,
        })
    }

    #[test]
    fn the_publisher_reports_each_5gmm_change_once() {
        let mut state = MmStateMachine::new();
        let mut publisher = MmStatePublisher::new();
        assert!(
            publisher.poll(&state).is_some(),
            "the first poll always reports"
        );
        assert!(publisher.poll(&state).is_none(), "no change, no update");

        state.switch_mm_state(MmSubState::RegisteredNormalService);
        match publisher.poll(&state) {
            Some(UeStatusUpdate::MmStateChanged {
                rm_state,
                mm_state,
                mm_sub_state,
            }) => {
                assert_eq!(rm_state, RmState::Registered);
                assert_eq!(mm_state, MmState::Registered);
                assert_eq!(mm_sub_state, MmSubState::RegisteredNormalService);
            }
            other => panic!("expected MmStateChanged, got {other:?}"),
        }
        assert!(publisher.poll(&state).is_none());
    }

    /// Issue #266: `deregister` reached the App task, which still believed the
    /// UE DEREGISTERED because nothing ever told it otherwise, and answered
    /// "not registered" without asking the NAS task for anything.
    #[tokio::test]
    async fn deregister_reaches_the_nas_task_once_the_nas_task_reports_registered() {
        let (base, mut nas_rx) = create_task_base_with_nas_rx();
        let mut task = AppTask::new_without_cli(base);

        // What the NAS task publishes after a Registration Accept.
        let mut state = MmStateMachine::new();
        state.switch_mm_state(MmSubState::RegisteredNormalService);
        let update = MmStatePublisher::new().poll(&state).unwrap();
        assert!(task.handle_message(AppMessage::StatusUpdate(update)).await);

        assert!(
            task.handle_message(cli(parse_ue_cli_command("deregister normal").unwrap()))
                .await
        );
        match nas_rx.try_recv() {
            Ok(TaskMessage::Message(NasMessage::InitiateDeregistration { switch_off })) => {
                assert!(!switch_off, "normal de-registration is not switch-off");
            }
            other => panic!("expected InitiateDeregistration, got {other:?}"),
        }

        // De-registration in progress: a second command is refused.
        let mut publisher = MmStatePublisher::new();
        state.switch_mm_state(MmSubState::DeregisteredInitiated);
        let update = publisher.poll(&state).unwrap();
        task.handle_message(AppMessage::StatusUpdate(update)).await;
        task.handle_message(cli(parse_ue_cli_command("deregister switch-off").unwrap()))
            .await;
        assert!(nas_rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn switch_off_reaches_the_nas_task_as_switch_off() {
        let (base, mut nas_rx) = create_task_base_with_nas_rx();
        let mut task = AppTask::new_without_cli(base);
        let mut state = MmStateMachine::new();
        state.switch_mm_state(MmSubState::RegisteredNormalService);
        let update = MmStatePublisher::new().poll(&state).unwrap();
        task.handle_message(AppMessage::StatusUpdate(update)).await;

        task.handle_message(cli(parse_ue_cli_command("deregister switch-off").unwrap()))
            .await;
        assert!(matches!(
            nas_rx.try_recv(),
            Ok(TaskMessage::Message(NasMessage::InitiateDeregistration {
                switch_off: true
            }))
        ));
    }

    #[test]
    fn deregister_takes_the_nr_cli_type_argument() {
        assert!(matches!(
            parse_ue_cli_command("deregister normal").unwrap(),
            UeCliCommandType::Deregister { switch_off: false }
        ));
        assert!(matches!(
            parse_ue_cli_command("deregister switch-off").unwrap(),
            UeCliCommandType::Deregister { switch_off: true }
        ));
        assert!(parse_ue_cli_command("deregister disable-5g").is_err());
        assert!(parse_ue_cli_command("deregister remove-sim").is_err());
        assert!(parse_ue_cli_command("deregister bogus").is_err());
        assert!(parse_ue_cli_command("deregister normal switch-off").is_err());
    }

    #[tokio::test]
    async fn test_cli_server_disabled() {
        let task_base = create_task_base();
        let mut task = AppTask::new_without_cli(task_base);

        let port = task.init_cli_server(vec!["ue1".to_string()]).await.unwrap();
        assert_eq!(port, 0);
        assert_eq!(task.cli_port(), 0);
    }
}
