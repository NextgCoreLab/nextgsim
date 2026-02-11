//! gNB Application Task
//!
//! This module implements the main application task for the gNB, which handles:
//! - CLI command processing via UDP server
//! - Status reporting and tracking
//! - Coordination between tasks
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb/app/task.cpp` implementation.

use std::collections::HashMap;
use std::net::SocketAddr;

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::app::{
    parse_cli_command, AmfContext, CliServer, CliServerError, GnbCmdHandler,
    StatusReporter, UeContext,
};
use crate::tasks::{
    AppMessage, GnbCliCommandType, GnbTaskBase, NgapMessage, StatusUpdate, Task, TaskMessage,
    UeReleaseRequestCause,
};

/// gNB Application Task
///
/// Handles CLI commands, status updates, and coordination between tasks.
pub struct AppTask {
    /// Task base for inter-task communication
    task_base: GnbTaskBase,
    /// CLI server for receiving commands
    cli_server: Option<CliServer>,
    /// Status reporter
    status_reporter: StatusReporter,
    /// UE contexts for CLI display
    ue_contexts: HashMap<i32, UeContext>,
    /// AMF contexts for CLI display
    amf_contexts: HashMap<i32, AmfContext>,
    /// Whether CLI is enabled
    cli_enabled: bool,
}

impl AppTask {
    /// Creates a new App task.
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            cli_server: None,
            status_reporter: StatusReporter::new(),
            ue_contexts: HashMap::new(),
            amf_contexts: HashMap::new(),
            cli_enabled: true,
        }
    }

    /// Creates a new App task with CLI disabled.
    pub fn new_without_cli(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            cli_server: None,
            status_reporter: StatusReporter::new(),
            ue_contexts: HashMap::new(),
            amf_contexts: HashMap::new(),
            cli_enabled: false,
        }
    }

    /// Initializes the CLI server.
    ///
    /// This should be called before running the task if CLI is enabled.
    pub async fn init_cli_server(&mut self, node_name: String) -> Result<u16, CliServerError> {
        if !self.cli_enabled {
            return Ok(0);
        }

        let bind_addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let server = CliServer::new(bind_addr, node_name).await?;
        let port = server.local_addr()?.port();
        self.cli_server = Some(server);
        Ok(port)
    }

    /// Returns the CLI server port, or 0 if CLI is disabled.
    pub fn cli_port(&self) -> u16 {
        self.cli_server
            .as_ref()
            .and_then(|s| s.local_addr().ok())
            .map(|a| a.port())
            .unwrap_or(0)
    }

    /// Handles a status update from another task.
    fn handle_status_update(&mut self, update: StatusUpdate) {
        debug!("Status update: {:?} = {}", update.status_type, update.value);
        self.status_reporter.apply_update(&update);
    }

    /// Handles a CLI command.
    async fn handle_cli_command(
        &mut self,
        command: GnbCliCommandType,
        response_addr: Option<SocketAddr>,
    ) {
        debug!("CLI command: {:?}", command);

        // Create command handler with current state
        let handler = GnbCmdHandler::new(
            &self.task_base,
            self.status_reporter.status(),
            &self.ue_contexts,
            &self.amf_contexts,
        );

        // Handle the command
        let response = handler.handle_command(&command, response_addr);

        // Handle UE release if requested
        if let GnbCliCommandType::UeRelease { ue_id } = command {
            if !response.is_error {
                // Send release request to NGAP task
                info!("UE release requested for UE {}", ue_id);
                let msg = NgapMessage::UeContextReleaseRequest {
                    ue_id,
                    cause: UeReleaseRequestCause::UserTriggered,
                };
                if let Err(e) = self.task_base.ngap_tx.send(msg).await {
                    error!("Failed to send UE release request to NGAP: {}", e);
                }
            }
        }

        // Send response if we have a CLI server and destination
        if let (Some(server), Some(addr)) = (&self.cli_server, response.destination) {
            let result = if response.is_error {
                server.send_error(response.content, addr).await
            } else {
                server.send_result(response.content, addr).await
            };

            if let Err(e) = result {
                warn!("Failed to send CLI response: {}", e);
            }
        }
    }

    /// Processes a raw CLI message from the server.
    async fn process_cli_message(&mut self, msg: crate::app::CliMessage) {
        // Parse the command
        let command = match parse_cli_command(&msg.value) {
            Ok(cmd) => cmd,
            Err(e) => {
                // Send error response
                if let Some(server) = &self.cli_server {
                    if let Some(addr) = msg.client_addr {
                        let _ = server.send_error(e, addr).await;
                    }
                }
                return;
            }
        };

        // Handle the command
        self.handle_cli_command(command, msg.client_addr).await;
    }

    /// Polls the CLI server for incoming messages.
    async fn poll_cli_server(&mut self) {
        let msg = if let Some(server) = &self.cli_server {
            match server.try_receive() {
                Ok(Some(msg)) => Some(msg),
                Ok(None) => None,
                Err(CliServerError::Timeout) => None,
                Err(e) => {
                    warn!("CLI server error: {}", e);
                    None
                }
            }
        } else {
            None
        };

        if let Some(msg) = msg {
            self.process_cli_message(msg).await;
        }
    }

    /// Updates UE context (called when UE state changes).
    pub fn update_ue_context(&mut self, ue_id: i32, ran_ue_ngap_id: i64, amf_ue_ngap_id: Option<i64>) {
        let context = self.ue_contexts.entry(ue_id).or_insert_with(|| {
            UeContext::new(ue_id, ran_ue_ngap_id)
        });
        context.ran_ue_ngap_id = ran_ue_ngap_id;
        context.amf_ue_ngap_id = amf_ue_ngap_id;
    }

    /// Removes a UE context.
    pub fn remove_ue_context(&mut self, ue_id: i32) {
        self.ue_contexts.remove(&ue_id);
    }

    /// Updates AMF context.
    pub fn update_amf_context(&mut self, amf_id: i32, amf_name: Option<String>, is_connected: bool) {
        let context = self.amf_contexts.entry(amf_id).or_insert_with(|| {
            AmfContext::new(amf_id)
        });
        context.amf_name = amf_name;
        context.is_connected = is_connected;
    }

    /// Removes an AMF context.
    pub fn remove_amf_context(&mut self, amf_id: i32) {
        self.amf_contexts.remove(&amf_id);
    }
}

#[async_trait::async_trait]
impl Task for AppTask {
    type Message = AppMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("App task started");

        if self.cli_enabled {
            if let Some(server) = &self.cli_server {
                info!("CLI server listening on port {}", server.local_addr().map(|a| a.port()).unwrap_or(0));
            }
        }

        loop {
            tokio::select! {
                // Handle messages from other tasks
                msg = rx.recv() => {
                    match msg {
                        Some(TaskMessage::Message(app_msg)) => {
                            match app_msg {
                                AppMessage::StatusUpdate(update) => {
                                    self.handle_status_update(update);
                                }
                                AppMessage::CliCommand(cmd) => {
                                    self.handle_cli_command(cmd.command, cmd.response_addr).await;
                                }
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
    use crate::tasks::{
        GtpMessage, NgapMessage, RlsMessage, RrcMessage, SctpMessage, StatusType, TaskHandle,
    };
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Arc;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(310, 410, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            ngap_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            gtp_ip: IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)),
            gtp_advertise_ip: None,
            ignore_stream_ids: false, upf_addr: None, upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
    }

    fn create_task_base(config: GnbConfig) -> GnbTaskBase {
        let (app_tx, _) = mpsc::channel::<TaskMessage<AppMessage>>(1);
        let (ngap_tx, _) = mpsc::channel::<TaskMessage<NgapMessage>>(1);
        let (rrc_tx, _) = mpsc::channel::<TaskMessage<RrcMessage>>(1);
        let (gtp_tx, _) = mpsc::channel::<TaskMessage<GtpMessage>>(1);
        let (rls_tx, _) = mpsc::channel::<TaskMessage<RlsMessage>>(1);
        let (sctp_tx, _) = mpsc::channel::<TaskMessage<SctpMessage>>(1);

        GnbTaskBase {
            config: Arc::new(config),
            app_tx: TaskHandle::new(app_tx),
            ngap_tx: TaskHandle::new(ngap_tx),
            rrc_tx: TaskHandle::new(rrc_tx),
            gtp_tx: TaskHandle::new(gtp_tx),
            rls_tx: TaskHandle::new(rls_tx),
            sctp_tx: TaskHandle::new(sctp_tx),
        }
    }

    #[test]
    fn test_app_task_creation() {
        let config = test_config();
        let task_base = create_task_base(config);
        let task = AppTask::new(task_base);
        assert!(task.cli_enabled);
    }

    #[test]
    fn test_app_task_without_cli() {
        let config = test_config();
        let task_base = create_task_base(config);
        let task = AppTask::new_without_cli(task_base);
        assert!(!task.cli_enabled);
    }

    #[test]
    fn test_status_update() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut task = AppTask::new(task_base);

        let update = StatusUpdate {
            status_type: StatusType::NgapIsUp,
            value: true,
        };
        task.handle_status_update(update);

        assert!(task.status_reporter.status().is_ngap_up);
    }

    #[test]
    fn test_ue_context_management() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut task = AppTask::new(task_base);

        // Add UE context
        task.update_ue_context(1, 100, Some(200));
        assert!(task.ue_contexts.contains_key(&1));
        assert_eq!(task.ue_contexts.get(&1).unwrap().ran_ue_ngap_id, 100);
        assert_eq!(task.ue_contexts.get(&1).unwrap().amf_ue_ngap_id, Some(200));

        // Update UE context
        task.update_ue_context(1, 100, Some(300));
        assert_eq!(task.ue_contexts.get(&1).unwrap().amf_ue_ngap_id, Some(300));

        // Remove UE context
        task.remove_ue_context(1);
        assert!(!task.ue_contexts.contains_key(&1));
    }

    #[test]
    fn test_amf_context_management() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut task = AppTask::new(task_base);

        // Add AMF context
        task.update_amf_context(1, Some("test-amf".to_string()), true);
        assert!(task.amf_contexts.contains_key(&1));
        assert_eq!(task.amf_contexts.get(&1).unwrap().amf_name, Some("test-amf".to_string()));
        assert!(task.amf_contexts.get(&1).unwrap().is_connected);

        // Update AMF context
        task.update_amf_context(1, Some("test-amf".to_string()), false);
        assert!(!task.amf_contexts.get(&1).unwrap().is_connected);

        // Remove AMF context
        task.remove_amf_context(1);
        assert!(!task.amf_contexts.contains_key(&1));
    }

    #[tokio::test]
    async fn test_cli_server_init() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut task = AppTask::new(task_base);

        let port = task.init_cli_server("test-gnb".to_string()).await.unwrap();
        assert!(port > 0);
        assert_eq!(task.cli_port(), port);
    }

    #[tokio::test]
    async fn test_cli_server_disabled() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut task = AppTask::new_without_cli(task_base);

        let port = task.init_cli_server("test-gnb".to_string()).await.unwrap();
        assert_eq!(port, 0);
        assert_eq!(task.cli_port(), 0);
    }
}
