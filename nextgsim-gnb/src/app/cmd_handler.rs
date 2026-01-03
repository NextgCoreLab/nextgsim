//! CLI Command Handler for gNB Application
//!
//! This module implements the command handler for CLI commands sent to the gNB.
//! It processes commands like status queries, UE listing, and UE release requests.
//!
//! # Architecture
//!
//! The command handler receives `CliCommand` messages from the App task and
//! generates appropriate responses. It accesses gNB state through the task base
//! to gather information about connected AMFs, UEs, and overall status.
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb/app/cmd_handler.cpp` implementation.

use std::net::SocketAddr;

use super::status::GnbStatusInfo;
use crate::tasks::{GnbCliCommandType, GnbTaskBase};

/// Response from a CLI command.
#[derive(Debug, Clone)]
pub struct CliResponse {
    /// Response content
    pub content: String,
    /// Whether this is an error response
    pub is_error: bool,
    /// Destination address for the response
    pub destination: Option<SocketAddr>,
}

impl CliResponse {
    /// Creates a successful response.
    pub fn success(content: String, destination: Option<SocketAddr>) -> Self {
        Self {
            content,
            is_error: false,
            destination,
        }
    }

    /// Creates an error response.
    pub fn error(content: String, destination: Option<SocketAddr>) -> Self {
        Self {
            content,
            is_error: true,
            destination,
        }
    }
}

/// CLI command handler for the gNB.
///
/// This handler processes CLI commands and generates responses.
/// It requires access to the gNB task base for querying state.
pub struct GnbCmdHandler<'a> {
    /// Reference to the task base for accessing gNB state
    task_base: &'a GnbTaskBase,
    /// Current status information
    status_info: &'a GnbStatusInfo,
    /// Connected UE contexts (UE ID -> UE info)
    ue_contexts: &'a std::collections::HashMap<i32, UeContext>,
    /// Connected AMF contexts (AMF ID -> AMF info)
    amf_contexts: &'a std::collections::HashMap<i32, AmfContext>,
}

/// UE context information for CLI display.
#[derive(Debug, Clone)]
pub struct UeContext {
    /// UE ID
    pub ue_id: i32,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: i64,
    /// AMF UE NGAP ID (if assigned)
    pub amf_ue_ngap_id: Option<i64>,
}

impl UeContext {
    /// Creates a new UE context.
    pub fn new(ue_id: i32, ran_ue_ngap_id: i64) -> Self {
        Self {
            ue_id,
            ran_ue_ngap_id,
            amf_ue_ngap_id: None,
        }
    }

    /// Formats the UE context as YAML.
    pub fn to_yaml(&self) -> String {
        let mut yaml = format!(
            "ue_id: {}\nran_ue_ngap_id: {}",
            self.ue_id, self.ran_ue_ngap_id
        );
        if let Some(amf_id) = self.amf_ue_ngap_id {
            yaml.push_str(&format!("\namf_ue_ngap_id: {}", amf_id));
        }
        yaml
    }
}

/// AMF context information for CLI display.
#[derive(Debug, Clone)]
pub struct AmfContext {
    /// AMF ID
    pub amf_id: i32,
    /// AMF name (if provided)
    pub amf_name: Option<String>,
    /// Whether the AMF is connected
    pub is_connected: bool,
}

impl AmfContext {
    /// Creates a new AMF context.
    pub fn new(amf_id: i32) -> Self {
        Self {
            amf_id,
            amf_name: None,
            is_connected: false,
        }
    }

    /// Formats the AMF context as YAML.
    pub fn to_yaml(&self) -> String {
        let mut yaml = format!("amf_id: {}\nis_connected: {}", self.amf_id, self.is_connected);
        if let Some(ref name) = self.amf_name {
            yaml.push_str(&format!("\namf_name: {}", name));
        }
        yaml
    }
}

impl<'a> GnbCmdHandler<'a> {
    /// Creates a new command handler.
    pub fn new(
        task_base: &'a GnbTaskBase,
        status_info: &'a GnbStatusInfo,
        ue_contexts: &'a std::collections::HashMap<i32, UeContext>,
        amf_contexts: &'a std::collections::HashMap<i32, AmfContext>,
    ) -> Self {
        Self {
            task_base,
            status_info,
            ue_contexts,
            amf_contexts,
        }
    }

    /// Handles a CLI command and returns a response.
    pub fn handle_command(
        &self,
        command: &GnbCliCommandType,
        response_addr: Option<SocketAddr>,
    ) -> CliResponse {
        match command {
            GnbCliCommandType::Info => self.handle_info(response_addr),
            GnbCliCommandType::Status => self.handle_status(response_addr),
            GnbCliCommandType::AmfList => self.handle_amf_list(response_addr),
            GnbCliCommandType::UeList => self.handle_ue_list(response_addr),
            GnbCliCommandType::UeInfo { ue_id } => self.handle_ue_info(*ue_id, response_addr),
            GnbCliCommandType::UeRelease { ue_id } => self.handle_ue_release(*ue_id, response_addr),
        }
    }

    /// Handles the INFO command - shows gNB configuration.
    fn handle_info(&self, response_addr: Option<SocketAddr>) -> CliResponse {
        let config = &self.task_base.config;
        let yaml = format!(
            "nci: {}\n\
             gnb_id_length: {}\n\
             plmn:\n  mcc: {}\n  mnc: {}\n\
             tac: {}\n\
             link_ip: {}\n\
             ngap_ip: {}\n\
             gtp_ip: {}",
            config.nci,
            config.gnb_id_length,
            config.plmn.mcc,
            config.plmn.mnc,
            config.tac,
            config.link_ip,
            config.ngap_ip,
            config.gtp_ip,
        );
        CliResponse::success(yaml, response_addr)
    }

    /// Handles the STATUS command - shows gNB status.
    fn handle_status(&self, response_addr: Option<SocketAddr>) -> CliResponse {
        match self.status_info.to_yaml() {
            Ok(yaml) => CliResponse::success(yaml, response_addr),
            Err(e) => CliResponse::error(
                format!("Failed to serialize status: {}", e),
                response_addr,
            ),
        }
    }

    /// Handles the AMF_LIST command - shows connected AMFs.
    fn handle_amf_list(&self, response_addr: Option<SocketAddr>) -> CliResponse {
        if self.amf_contexts.is_empty() {
            return CliResponse::success("amfs: []".to_string(), response_addr);
        }

        let mut yaml = String::from("amfs:");
        for amf in self.amf_contexts.values() {
            yaml.push_str(&format!("\n  - id: {}", amf.amf_id));
        }
        CliResponse::success(yaml, response_addr)
    }

    /// Handles the UE_LIST command - shows connected UEs.
    fn handle_ue_list(&self, response_addr: Option<SocketAddr>) -> CliResponse {
        if self.ue_contexts.is_empty() {
            return CliResponse::success("ues: []".to_string(), response_addr);
        }

        let mut yaml = String::from("ues:");
        for ue in self.ue_contexts.values() {
            yaml.push_str(&format!(
                "\n  - ue_id: {}\n    ran_ngap_id: {}",
                ue.ue_id, ue.ran_ue_ngap_id
            ));
            if let Some(amf_id) = ue.amf_ue_ngap_id {
                yaml.push_str(&format!("\n    amf_ngap_id: {}", amf_id));
            }
        }
        CliResponse::success(yaml, response_addr)
    }

    /// Handles the UE_INFO command - shows details for a specific UE.
    fn handle_ue_info(&self, ue_id: i32, response_addr: Option<SocketAddr>) -> CliResponse {
        match self.ue_contexts.get(&ue_id) {
            Some(ue) => CliResponse::success(ue.to_yaml(), response_addr),
            None => CliResponse::error(format!("UE not found with ID: {}", ue_id), response_addr),
        }
    }

    /// Handles the UE_RELEASE command - requests UE context release.
    ///
    /// Note: This only validates the UE exists. The actual release is performed
    /// by sending a message to the NGAP task.
    fn handle_ue_release(&self, ue_id: i32, response_addr: Option<SocketAddr>) -> CliResponse {
        if !self.ue_contexts.contains_key(&ue_id) {
            return CliResponse::error(format!("UE not found with ID: {}", ue_id), response_addr);
        }
        // The actual release will be triggered by the App task after this returns
        CliResponse::success(
            format!("Requesting UE context release for UE {}", ue_id),
            response_addr,
        )
    }
}

/// Parses a CLI command string into a `GnbCliCommandType`.
///
/// # Supported Commands
///
/// - `info` - Show gNB configuration
/// - `status` - Show gNB status
/// - `amf-list` - List connected AMFs
/// - `ue-list` - List connected UEs
/// - `ue-info <ue_id>` - Show UE details
/// - `ue-release <ue_id>` - Release UE context
///
/// # Returns
///
/// * `Ok(GnbCliCommandType)` - Successfully parsed command
/// * `Err(String)` - Parse error with description
pub fn parse_cli_command(input: &str) -> Result<GnbCliCommandType, String> {
    let tokens: Vec<&str> = input.trim().split_whitespace().collect();

    if tokens.is_empty() {
        return Err("Empty command".to_string());
    }

    match tokens[0].to_lowercase().as_str() {
        "info" => Ok(GnbCliCommandType::Info),
        "status" => Ok(GnbCliCommandType::Status),
        "amf-list" => Ok(GnbCliCommandType::AmfList),
        "ue-list" => Ok(GnbCliCommandType::UeList),
        "ue-info" => {
            if tokens.len() < 2 {
                return Err("ue-info requires a UE ID argument".to_string());
            }
            let ue_id = tokens[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid UE ID: {}", tokens[1]))?;
            Ok(GnbCliCommandType::UeInfo { ue_id })
        }
        "ue-release" => {
            if tokens.len() < 2 {
                return Err("ue-release requires a UE ID argument".to_string());
            }
            let ue_id = tokens[1]
                .parse::<i32>()
                .map_err(|_| format!("Invalid UE ID: {}", tokens[1]))?;
            Ok(GnbCliCommandType::UeRelease { ue_id })
        }
        _ => Err(format!("Unknown command: {}", tokens[0])),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{
        AppMessage, GtpMessage, NgapMessage, RlsMessage, RrcMessage, SctpMessage, TaskHandle,
        TaskMessage,
    };
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr};
    use std::sync::Arc;
    use tokio::sync::mpsc;

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
            ignore_stream_ids: false,
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
    fn test_parse_info_command() {
        let cmd = parse_cli_command("info").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::Info));
    }

    #[test]
    fn test_parse_status_command() {
        let cmd = parse_cli_command("status").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::Status));
    }

    #[test]
    fn test_parse_amf_list_command() {
        let cmd = parse_cli_command("amf-list").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::AmfList));
    }

    #[test]
    fn test_parse_ue_list_command() {
        let cmd = parse_cli_command("ue-list").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::UeList));
    }

    #[test]
    fn test_parse_ue_info_command() {
        let cmd = parse_cli_command("ue-info 42").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::UeInfo { ue_id: 42 }));
    }

    #[test]
    fn test_parse_ue_release_command() {
        let cmd = parse_cli_command("ue-release 123").unwrap();
        assert!(matches!(cmd, GnbCliCommandType::UeRelease { ue_id: 123 }));
    }

    #[test]
    fn test_parse_empty_command() {
        let result = parse_cli_command("");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Empty"));
    }

    #[test]
    fn test_parse_unknown_command() {
        let result = parse_cli_command("unknown");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown"));
    }

    #[test]
    fn test_parse_ue_info_missing_id() {
        let result = parse_cli_command("ue-info");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("requires"));
    }

    #[test]
    fn test_parse_ue_info_invalid_id() {
        let result = parse_cli_command("ue-info abc");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid"));
    }

    #[test]
    fn test_handle_info_command() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::Info, None);

        assert!(!response.is_error);
        assert!(response.content.contains("nci: 16"));
        assert!(response.content.contains("mcc: 310"));
    }

    #[test]
    fn test_handle_status_command() {
        let config = test_config();
        let task_base = create_task_base(config);
        let mut status_info = GnbStatusInfo::new();
        status_info.is_ngap_up = true;
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::Status, None);

        assert!(!response.is_error);
        assert!(response.content.contains("is-ngap-up: true"));
    }

    #[test]
    fn test_handle_ue_list_empty() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeList, None);

        assert!(!response.is_error);
        assert!(response.content.contains("ues: []"));
    }

    #[test]
    fn test_handle_ue_list_with_ues() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let mut ue_contexts = HashMap::new();
        ue_contexts.insert(1, UeContext::new(1, 100));
        ue_contexts.insert(2, UeContext::new(2, 200));
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeList, None);

        assert!(!response.is_error);
        assert!(response.content.contains("ue_id: 1"));
        assert!(response.content.contains("ue_id: 2"));
    }

    #[test]
    fn test_handle_ue_info_found() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let mut ue_contexts = HashMap::new();
        let mut ue = UeContext::new(42, 4200);
        ue.amf_ue_ngap_id = Some(9999);
        ue_contexts.insert(42, ue);
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeInfo { ue_id: 42 }, None);

        assert!(!response.is_error);
        assert!(response.content.contains("ue_id: 42"));
        assert!(response.content.contains("ran_ue_ngap_id: 4200"));
        assert!(response.content.contains("amf_ue_ngap_id: 9999"));
    }

    #[test]
    fn test_handle_ue_info_not_found() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeInfo { ue_id: 999 }, None);

        assert!(response.is_error);
        assert!(response.content.contains("not found"));
    }

    #[test]
    fn test_handle_ue_release_found() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let mut ue_contexts = HashMap::new();
        ue_contexts.insert(42, UeContext::new(42, 4200));
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeRelease { ue_id: 42 }, None);

        assert!(!response.is_error);
        assert!(response.content.contains("Requesting"));
    }

    #[test]
    fn test_handle_ue_release_not_found() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::UeRelease { ue_id: 999 }, None);

        assert!(response.is_error);
        assert!(response.content.contains("not found"));
    }

    #[test]
    fn test_handle_amf_list_empty() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let amf_contexts = HashMap::new();

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::AmfList, None);

        assert!(!response.is_error);
        assert!(response.content.contains("amfs: []"));
    }

    #[test]
    fn test_handle_amf_list_with_amfs() {
        let config = test_config();
        let task_base = create_task_base(config);
        let status_info = GnbStatusInfo::new();
        let ue_contexts = HashMap::new();
        let mut amf_contexts = HashMap::new();
        amf_contexts.insert(1, AmfContext::new(1));
        amf_contexts.insert(2, AmfContext::new(2));

        let handler = GnbCmdHandler::new(&task_base, &status_info, &ue_contexts, &amf_contexts);
        let response = handler.handle_command(&GnbCliCommandType::AmfList, None);

        assert!(!response.is_error);
        assert!(response.content.contains("amfs:"));
    }

    #[test]
    fn test_ue_context_to_yaml() {
        let mut ue = UeContext::new(1, 100);
        ue.amf_ue_ngap_id = Some(200);
        let yaml = ue.to_yaml();
        assert!(yaml.contains("ue_id: 1"));
        assert!(yaml.contains("ran_ue_ngap_id: 100"));
        assert!(yaml.contains("amf_ue_ngap_id: 200"));
    }

    #[test]
    fn test_amf_context_to_yaml() {
        let mut amf = AmfContext::new(1);
        amf.amf_name = Some("test-amf".to_string());
        amf.is_connected = true;
        let yaml = amf.to_yaml();
        assert!(yaml.contains("amf_id: 1"));
        assert!(yaml.contains("is_connected: true"));
        assert!(yaml.contains("amf_name: test-amf"));
    }

    #[test]
    fn test_cli_response_success() {
        let response = CliResponse::success("test".to_string(), None);
        assert!(!response.is_error);
        assert_eq!(response.content, "test");
    }

    #[test]
    fn test_cli_response_error() {
        let response = CliResponse::error("error".to_string(), None);
        assert!(response.is_error);
        assert_eq!(response.content, "error");
    }

    #[test]
    fn test_parse_case_insensitive() {
        assert!(parse_cli_command("INFO").is_ok());
        assert!(parse_cli_command("Status").is_ok());
        assert!(parse_cli_command("UE-LIST").is_ok());
    }
}
