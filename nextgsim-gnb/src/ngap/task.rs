//! NGAP Task Implementation
//!
//! This module implements the NGAP task for the gNB. The NGAP task handles:
//! - NG Setup procedure with AMF(s)
//! - UE context management
//! - NAS message routing between RRC and AMF
//! - PDU session resource management
//!
//! # Message Flow
//!
//! ```text
//! SCTP Task ---> NGAP Task ---> RRC Task (NAS delivery)
//!                    |
//!                    +-------> GTP Task (PDU sessions)
//!                    |
//!                    +-------> App Task (status updates)
//! ```

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    AppMessage, GnbTaskBase, GtpMessage, GtpUeContextUpdate, NgapMessage,
    PduSessionResource, RrcMessage, SctpMessage, StatusType, StatusUpdate,
    Task, TaskMessage,
};
use nextgsim_common::OctetString;

use super::amf_context::{AmfState, NgapAmfContext};
use super::ue_context::{NgapPduSession, NgapUeContext};

use nextgsim_ngap::codec::{decode_ngap_pdu, encode_ngap_pdu};
use nextgsim_ngap::procedures::{
    build_ng_setup_request, parse_ng_setup_response, parse_ng_setup_failure,
    is_ng_setup_response, is_ng_setup_failure,
    BroadcastPlmnItem, GnbId, NgSetupRequestParams, PagingDrx, SNssai, SupportedTaItem,
};

/// NGAP Task for managing AMF communication and UE contexts
pub struct NgapTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// AMF contexts indexed by client ID
    amf_contexts: HashMap<i32, NgapAmfContext>,
    /// UE contexts indexed by UE ID
    ue_contexts: HashMap<i32, NgapUeContext>,
    /// Counter for generating RAN UE NGAP IDs
    ran_ue_ngap_id_counter: i64,
    /// Counter for generating downlink TEIDs
    #[allow(dead_code)]
    downlink_teid_counter: u32,
    /// Whether the NGAP task is initialized (at least one AMF ready)
    is_initialized: bool,
}

impl NgapTask {
    /// Creates a new NGAP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            amf_contexts: HashMap::new(),
            ue_contexts: HashMap::new(),
            ran_ue_ngap_id_counter: 0,
            downlink_teid_counter: 0,
            is_initialized: false,
        }
    }

    /// Generates a new RAN UE NGAP ID
    fn next_ran_ue_ngap_id(&mut self) -> i64 {
        self.ran_ue_ngap_id_counter += 1;
        self.ran_ue_ngap_id_counter
    }

    /// Generates a new downlink TEID
    #[allow(dead_code)]
    fn next_downlink_teid(&mut self) -> u32 {
        self.downlink_teid_counter += 1;
        self.downlink_teid_counter
    }

    // ========================================================================
    // AMF Context Management
    // ========================================================================

    /// Creates an AMF context for a new connection
    fn create_amf_context(&mut self, client_id: i32) {
        let ctx = NgapAmfContext::new(client_id);
        self.amf_contexts.insert(client_id, ctx);
        debug!("Created AMF context for client_id: {}", client_id);
    }

    /// Finds an AMF context by client ID
    #[allow(dead_code)]
    fn find_amf_context(&self, client_id: i32) -> Option<&NgapAmfContext> {
        self.amf_contexts.get(&client_id)
    }

    /// Finds a mutable AMF context by client ID
    #[allow(dead_code)]
    fn find_amf_context_mut(&mut self, client_id: i32) -> Option<&mut NgapAmfContext> {
        self.amf_contexts.get_mut(&client_id)
    }

    /// Selects an AMF for a new UE based on capacity and slice support
    fn select_amf(&self, _requested_nssai: Option<i32>) -> Option<i32> {
        // Simple selection: pick the first ready AMF with highest capacity
        self.amf_contexts
            .values()
            .filter(|ctx| ctx.is_ready())
            .max_by_key(|ctx| ctx.relative_capacity)
            .map(|ctx| ctx.ctx_id)
    }

    // ========================================================================
    // UE Context Management
    // ========================================================================

    /// Creates a UE context for a new UE
    fn create_ue_context(&mut self, ue_id: i32, amf_ctx_id: i32) -> Option<i64> {
        let amf_ctx = self.amf_contexts.get_mut(&amf_ctx_id)?;
        let stream_id = amf_ctx.allocate_stream()?;
        let ran_ue_ngap_id = self.next_ran_ue_ngap_id();

        let ctx = NgapUeContext::new(ue_id, ran_ue_ngap_id, amf_ctx_id, stream_id);
        self.ue_contexts.insert(ue_id, ctx);

        debug!(
            "Created UE context: ue_id={}, ran_ue_ngap_id={}, amf_ctx_id={}, stream={}",
            ue_id, ran_ue_ngap_id, amf_ctx_id, stream_id
        );

        Some(ran_ue_ngap_id)
    }

    /// Finds a UE context by UE ID
    #[allow(dead_code)]
    fn find_ue_context(&self, ue_id: i32) -> Option<&NgapUeContext> {
        self.ue_contexts.get(&ue_id)
    }

    /// Finds a mutable UE context by UE ID
    #[allow(dead_code)]
    fn find_ue_context_mut(&mut self, ue_id: i32) -> Option<&mut NgapUeContext> {
        self.ue_contexts.get_mut(&ue_id)
    }

    /// Finds a UE context by RAN UE NGAP ID
    #[allow(dead_code)]
    fn find_ue_by_ran_id(&self, ran_ue_ngap_id: i64) -> Option<&NgapUeContext> {
        self.ue_contexts
            .values()
            .find(|ctx| ctx.ran_ue_ngap_id == ran_ue_ngap_id)
    }

    /// Finds a UE context by AMF UE NGAP ID
    #[allow(dead_code)]
    fn find_ue_by_amf_id(&self, amf_ue_ngap_id: i64) -> Option<&NgapUeContext> {
        self.ue_contexts
            .values()
            .find(|ctx| ctx.amf_ue_ngap_id == Some(amf_ue_ngap_id))
    }

    /// Deletes a UE context
    fn delete_ue_context(&mut self, ue_id: i32) {
        if let Some(ctx) = self.ue_contexts.remove(&ue_id) {
            // Release the stream back to the AMF
            if let Some(amf_ctx) = self.amf_contexts.get_mut(&ctx.amf_ctx_id) {
                amf_ctx.release_stream(ctx.stream_id);
            }
            debug!("Deleted UE context: ue_id={}", ue_id);
        }
    }

    // ========================================================================
    // SCTP Event Handlers
    // ========================================================================

    /// Handles SCTP association up event
    async fn handle_association_up(
        &mut self,
        client_id: i32,
        association_id: i32,
        in_streams: u16,
        out_streams: u16,
    ) {
        info!(
            "SCTP association up: client_id={}, association_id={}, in={}, out={}",
            client_id, association_id, in_streams, out_streams
        );

        // Create or update AMF context
        if !self.amf_contexts.contains_key(&client_id) {
            self.create_amf_context(client_id);
        }

        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            ctx.on_association_up(association_id, in_streams, out_streams);
        }

        // Send NG Setup Request
        self.send_ng_setup_request(client_id).await;
    }

    /// Handles SCTP association down event
    async fn handle_association_down(&mut self, client_id: i32) {
        info!("SCTP association down: client_id={}", client_id);

        // Update AMF context
        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            ctx.on_association_down();
        }

        // Release all UEs associated with this AMF
        let ue_ids: Vec<i32> = self
            .ue_contexts
            .values()
            .filter(|ctx| ctx.amf_ctx_id == client_id)
            .map(|ctx| ctx.ue_id)
            .collect();

        for ue_id in ue_ids {
            self.delete_ue_context(ue_id);
            // Notify RRC of AN release
            self.send_an_release(ue_id).await;
        }

        // Update initialization status
        self.update_initialization_status().await;
    }

    // ========================================================================
    // NG Setup Procedure
    // ========================================================================

    /// Sends NG Setup Request to an AMF
    async fn send_ng_setup_request(&mut self, amf_id: i32) {
        let config = &self.task_base.config;

        // Build PLMN identity bytes
        let plmn_bytes = config.plmn.encode();

        // Build supported TA list
        let slice_support: Vec<SNssai> = config
            .nssai
            .iter()
            .map(|s| SNssai {
                sst: s.sst,
                sd: s.sd,
            })
            .collect();

        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: plmn_bytes,
                gnb_id_value: (config.nci >> (36 - config.gnb_id_length)) as u32,
                gnb_id_length: config.gnb_id_length as u8,
            },
            ran_node_name: Some("nextgsim-gnb".to_string()),
            supported_ta_list: vec![SupportedTaItem {
                tac: [
                    ((config.tac >> 16) & 0xFF) as u8,
                    ((config.tac >> 8) & 0xFF) as u8,
                    (config.tac & 0xFF) as u8,
                ],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: plmn_bytes,
                    slice_support_list: if slice_support.is_empty() {
                        vec![SNssai { sst: 1, sd: None }] // Default slice
                    } else {
                        slice_support
                    },
                }],
            }],
            default_paging_drx: PagingDrx::V128,
        };

        match build_ng_setup_request(&params) {
            Ok(pdu) => {
                match encode_ngap_pdu(&pdu) {
                    Ok(bytes) => {
                        // Mark AMF as waiting for response
                        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                            ctx.on_ng_setup_sent();
                        }

                        // Send via SCTP
                        self.send_ngap_non_ue(amf_id, 0, bytes).await;
                        info!("Sent NG Setup Request to AMF {}", amf_id);
                    }
                    Err(e) => {
                        error!("Failed to encode NG Setup Request: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to build NG Setup Request: {}", e);
            }
        }
    }

    /// Handles NG Setup Response
    fn handle_ng_setup_response(&mut self, amf_id: i32, pdu_bytes: &[u8]) -> bool {
        let pdu = match decode_ngap_pdu(pdu_bytes) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to decode NGAP PDU: {}", e);
                return false;
            }
        };

        if !is_ng_setup_response(&pdu) {
            return false;
        }

        match parse_ng_setup_response(&pdu) {
            Ok(response) => {
                info!(
                    "Received NG Setup Response from AMF {}: name={}",
                    amf_id, response.amf_name
                );

                if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                    ctx.on_ng_setup_response(response);
                }
                true
            }
            Err(e) => {
                error!("Failed to parse NG Setup Response: {}", e);
                false
            }
        }
    }

    /// Handles NG Setup Failure
    fn handle_ng_setup_failure(&mut self, amf_id: i32, pdu_bytes: &[u8]) -> bool {
        let pdu = match decode_ngap_pdu(pdu_bytes) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to decode NGAP PDU: {}", e);
                return false;
            }
        };

        if !is_ng_setup_failure(&pdu) {
            return false;
        }

        match parse_ng_setup_failure(&pdu) {
            Ok(failure) => {
                warn!(
                    "Received NG Setup Failure from AMF {}: cause={:?}",
                    amf_id, failure.cause
                );

                // TODO: Handle time_to_wait for retry
                if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                    ctx.on_association_down(); // Reset to not connected
                }
                true
            }
            Err(e) => {
                error!("Failed to parse NG Setup Failure: {}", e);
                false
            }
        }
    }

    // ========================================================================
    // NAS Message Routing
    // ========================================================================

    /// Handles Initial NAS delivery from RRC (Initial UE Message)
    async fn handle_initial_nas_delivery(
        &mut self,
        ue_id: i32,
        pdu: OctetString,
        rrc_establishment_cause: i64,
        _s_tmsi: Option<crate::tasks::GutiMobileIdentity>,
    ) {
        debug!(
            "Initial NAS delivery: ue_id={}, cause={}, pdu_len={}",
            ue_id,
            rrc_establishment_cause,
            pdu.len()
        );

        // Select an AMF for this UE
        let amf_id = match self.select_amf(None) {
            Some(id) => id,
            None => {
                warn!("No AMF available for UE {}", ue_id);
                return;
            }
        };

        // Create UE context
        let ran_ue_ngap_id = match self.create_ue_context(ue_id, amf_id) {
            Some(id) => id,
            None => {
                error!("Failed to create UE context for UE {}", ue_id);
                return;
            }
        };

        // Build and send Initial UE Message
        // For now, we'll send the NAS PDU directly - full implementation would use
        // nextgsim_ngap::procedures::initial_ue_message
        let stream = self
            .ue_contexts
            .get(&ue_id)
            .map(|ctx| ctx.stream_id)
            .unwrap_or(0);

        // TODO: Build proper Initial UE Message using NGAP procedures
        // For now, log that we would send it
        info!(
            "Would send Initial UE Message: ue_id={}, ran_ue_ngap_id={}, amf_id={}, stream={}",
            ue_id, ran_ue_ngap_id, amf_id, stream
        );

        // Notify GTP task of new UE context
        self.send_gtp_ue_context_update(ue_id, None).await;
    }

    /// Handles Uplink NAS delivery from RRC
    async fn handle_uplink_nas_delivery(&mut self, ue_id: i32, pdu: OctetString) {
        debug!(
            "Uplink NAS delivery: ue_id={}, pdu_len={}",
            ue_id,
            pdu.len()
        );

        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("UE context not found for uplink NAS: ue_id={}", ue_id);
                return;
            }
        };

        if ctx.amf_ue_ngap_id.is_none() {
            warn!("AMF UE NGAP ID not set for UE {}", ue_id);
            return;
        }

        // TODO: Build and send Uplink NAS Transport message
        info!(
            "Would send Uplink NAS Transport: ue_id={}, amf_id={}",
            ue_id, ctx.amf_ctx_id
        );
    }

    /// Delivers downlink NAS to RRC
    #[allow(dead_code)]
    async fn deliver_downlink_nas(&self, ue_id: i32, pdu: OctetString) {
        let msg = RrcMessage::NasDelivery { ue_id, pdu };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to deliver downlink NAS to RRC: {}", e);
        }
    }

    // ========================================================================
    // PDU Session Management
    // ========================================================================

    /// Handles PDU Session Resource Setup from AMF
    #[allow(dead_code)]
    async fn handle_pdu_session_setup(
        &mut self,
        ue_id: i32,
        psi: u8,
        qfi: Option<u8>,
        uplink_teid: u32,
        upf_address: std::net::IpAddr,
    ) {
        let downlink_teid = self.next_downlink_teid();

        // Add session to UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            let session = NgapPduSession {
                psi,
                qfi,
                uplink_teid,
                downlink_teid,
                upf_address,
            };
            ctx.add_pdu_session(session);
        }

        // Notify GTP task
        let resource = PduSessionResource {
            psi: psi as i32,
            qfi,
            uplink_teid,
            downlink_teid,
            upf_address,
        };

        let msg = GtpMessage::SessionCreate { ue_id, resource };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send PDU session create to GTP: {}", e);
        }

        info!(
            "PDU session setup: ue_id={}, psi={}, ul_teid={:08x}, dl_teid={:08x}",
            ue_id, psi, uplink_teid, downlink_teid
        );
    }

    /// Handles PDU Session Resource Release from AMF
    #[allow(dead_code)]
    async fn handle_pdu_session_release(&mut self, ue_id: i32, psi: u8) {
        // Remove session from UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.remove_pdu_session(psi);
        }

        // Notify GTP task
        let msg = GtpMessage::SessionRelease {
            ue_id,
            psi: psi as i32,
        };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send PDU session release to GTP: {}", e);
        }

        info!("PDU session release: ue_id={}, psi={}", ue_id, psi);
    }

    // ========================================================================
    // Radio Link Failure
    // ========================================================================

    /// Handles radio link failure notification from RRC
    async fn handle_radio_link_failure(&mut self, ue_id: i32) {
        info!("Radio link failure: ue_id={}", ue_id);

        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("UE context not found for RLF: ue_id={}", ue_id);
                return;
            }
        };

        // TODO: Send UE Context Release Request to AMF
        info!(
            "Would send UE Context Release Request: ue_id={}, amf_id={}",
            ue_id, ctx.amf_ctx_id
        );

        // Clean up UE context
        self.delete_ue_context(ue_id);

        // Notify GTP task
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }
    }

    // ========================================================================
    // NGAP PDU Handling
    // ========================================================================

    /// Handles received NGAP PDU from SCTP
    async fn handle_ngap_pdu(&mut self, client_id: i32, stream: u16, pdu: OctetString) {
        debug!(
            "Received NGAP PDU: client_id={}, stream={}, len={}",
            client_id,
            stream,
            pdu.len()
        );

        let pdu_bytes = pdu.data();

        // Check AMF state
        let amf_state = self
            .amf_contexts
            .get(&client_id)
            .map(|ctx| ctx.state)
            .unwrap_or(AmfState::NotConnected);

        match amf_state {
            AmfState::WaitingNgSetup => {
                // Expecting NG Setup Response or Failure
                if self.handle_ng_setup_response(client_id, pdu_bytes) {
                    self.update_initialization_status().await;
                } else if self.handle_ng_setup_failure(client_id, pdu_bytes) {
                    // Already handled
                } else {
                    warn!("Unexpected NGAP PDU while waiting for NG Setup response");
                }
            }
            AmfState::Ready | AmfState::Overloaded => {
                // Handle operational messages
                // TODO: Decode and dispatch based on message type
                // For now, just log
                debug!("Received operational NGAP PDU on stream {}", stream);
            }
            _ => {
                warn!(
                    "Received NGAP PDU in unexpected AMF state: {:?}",
                    amf_state
                );
            }
        }
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Sends an NGAP PDU for non-UE-associated signaling (stream 0)
    async fn send_ngap_non_ue(&self, amf_id: i32, stream: u16, data: Vec<u8>) {
        let msg = SctpMessage::SendMessage {
            client_id: amf_id,
            stream,
            buffer: OctetString::from_slice(&data),
        };

        if let Err(e) = self.task_base.sctp_tx.send(msg).await {
            error!("Failed to send NGAP PDU to SCTP: {}", e);
        }
    }

    /// Sends AN release to RRC
    async fn send_an_release(&self, ue_id: i32) {
        let msg = RrcMessage::AnRelease { ue_id };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send AN release to RRC: {}", e);
        }
    }

    /// Sends GTP UE context update
    async fn send_gtp_ue_context_update(&self, ue_id: i32, amf_ue_ngap_id: Option<i64>) {
        let update = GtpUeContextUpdate {
            ue_id,
            amf_ue_ngap_id,
        };
        let msg = GtpMessage::UeContextUpdate { ue_id, update };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context update to GTP: {}", e);
        }
    }

    /// Updates initialization status and notifies App task
    async fn update_initialization_status(&mut self) {
        let any_ready = self.amf_contexts.values().any(|ctx| ctx.is_ready());

        if any_ready != self.is_initialized {
            self.is_initialized = any_ready;

            // Notify App task
            let msg = AppMessage::StatusUpdate(StatusUpdate {
                status_type: StatusType::NgapIsUp,
                value: any_ready,
            });
            if let Err(e) = self.task_base.app_tx.send(msg).await {
                error!("Failed to send status update to App: {}", e);
            }

            // If initialized, notify RRC to power on radio
            if any_ready {
                let msg = RrcMessage::RadioPowerOn;
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to send radio power on to RRC: {}", e);
                }
                info!("NGAP initialized, radio powered on");
            }
        }
    }
}

// ============================================================================
// Task Implementation
// ============================================================================

#[async_trait::async_trait]
impl Task for NgapTask {
    type Message = NgapMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NGAP task started");

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NgapMessage::SctpAssociationUp {
                            client_id,
                            association_id,
                            in_streams,
                            out_streams,
                        } => {
                            self.handle_association_up(
                                client_id,
                                association_id,
                                in_streams,
                                out_streams,
                            )
                            .await;
                        }
                        NgapMessage::SctpAssociationDown { client_id } => {
                            self.handle_association_down(client_id).await;
                        }
                        NgapMessage::ReceiveNgapPdu {
                            client_id,
                            stream,
                            pdu,
                        } => {
                            self.handle_ngap_pdu(client_id, stream, pdu).await;
                        }
                        NgapMessage::InitialNasDelivery {
                            ue_id,
                            pdu,
                            rrc_establishment_cause,
                            s_tmsi,
                        } => {
                            self.handle_initial_nas_delivery(
                                ue_id,
                                pdu,
                                rrc_establishment_cause,
                                s_tmsi,
                            )
                            .await;
                        }
                        NgapMessage::UplinkNasDelivery { ue_id, pdu } => {
                            self.handle_uplink_nas_delivery(ue_id, pdu).await;
                        }
                        NgapMessage::RadioLinkFailure { ue_id } => {
                            self.handle_radio_link_failure(ue_id).await;
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => {
                    info!("NGAP task received shutdown signal");
                    break;
                }
                None => {
                    info!("NGAP task channel closed");
                    break;
                }
            }
        }

        info!(
            "NGAP task stopped, {} AMF contexts, {} UE contexts",
            self.amf_contexts.len(),
            self.ue_contexts.len()
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
        }
    }

    #[test]
    fn test_ngap_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NgapTask::new(task_base);
        assert!(task.amf_contexts.is_empty());
        assert!(task.ue_contexts.is_empty());
        assert!(!task.is_initialized);
    }

    #[test]
    fn test_ngap_task_amf_context_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        assert!(task.amf_contexts.contains_key(&1));
        let ctx = task.find_amf_context(1).unwrap();
        assert_eq!(ctx.ctx_id, 1);
        assert_eq!(ctx.state, AmfState::NotConnected);
    }

    #[test]
    fn test_ngap_task_ue_context_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        // Create AMF context first
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        // Create UE context
        let ran_id = task.create_ue_context(10, 1);
        assert!(ran_id.is_some());
        assert_eq!(ran_id.unwrap(), 1); // First RAN UE NGAP ID

        let ue_ctx = task.find_ue_context(10).unwrap();
        assert_eq!(ue_ctx.ue_id, 10);
        assert_eq!(ue_ctx.amf_ctx_id, 1);
    }

    #[test]
    fn test_ngap_task_select_amf() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        // No AMF available
        assert!(task.select_amf(None).is_none());

        // Add AMF but not ready
        task.create_amf_context(1);
        assert!(task.select_amf(None).is_none());

        // Make AMF ready
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 100;
        }
        assert_eq!(task.select_amf(None), Some(1));
    }

    #[test]
    fn test_ngap_task_ran_ue_ngap_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        let id1 = task.next_ran_ue_ngap_id();
        let id2 = task.next_ran_ue_ngap_id();
        let id3 = task.next_ran_ue_ngap_id();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_ngap_task_ue_context_deletion() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        // Setup
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(10, 1);

        assert!(task.find_ue_context(10).is_some());

        // Delete
        task.delete_ue_context(10);
        assert!(task.find_ue_context(10).is_none());
    }
}
