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
    Task, TaskMessage, UeReleaseRequestCause,
};
use nextgsim_common::OctetString;

use super::amf_context::{AmfState, NgapAmfContext};
use super::ue_context::{NgapPduSession, NgapUeContext};

use nextgsim_ngap::codec::{decode_ngap_pdu, encode_ngap_pdu};
use nextgsim_ngap::procedures::{
    build_ng_setup_request, parse_ng_setup_response, parse_ng_setup_failure,
    is_ng_setup_response, is_ng_setup_failure,
    BroadcastPlmnItem, GnbId, NgSetupRequestParams, PagingDrx, SNssai, SupportedTaItem,
    // Initial UE Message
    encode_initial_ue_message, InitialUeMessageParams,
    RrcEstablishmentCauseValue, UeContextRequestValue,
    // NAS Transport
    decode_downlink_nas_transport, DownlinkNasTransportData,
    encode_uplink_nas_transport, UplinkNasTransportParams,
};
use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi, UserLocationInfoNr, NrCgi, Tai};

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

                // Handle time_to_wait for retry
                // The time_to_wait IE indicates the minimum time the NG-RAN node should wait
                // before re-initiating the NG Setup procedure.
                // Note: In the current implementation, we rely on the SCTP reconnection
                // mechanism to handle retries. A more sophisticated implementation would
                // parse the time_to_wait IE and schedule a retry after that duration.
                if let Some(time_to_wait) = failure.time_to_wait {
                    info!(
                        "AMF {} requested wait time before retry: {:?}",
                        amf_id, time_to_wait
                    );
                }

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

    /// Handles Downlink NAS Transport from AMF
    async fn handle_downlink_nas_transport(
        &mut self,
        _amf_id: i32,
        _stream: u16,
        dl_nas: DownlinkNasTransportData,
    ) {
        info!(
            "Downlink NAS Transport: amf_ue_ngap_id={}, ran_ue_ngap_id={}, nas_pdu_len={}",
            dl_nas.amf_ue_ngap_id,
            dl_nas.ran_ue_ngap_id,
            dl_nas.nas_pdu.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_ctx = self.ue_contexts.values_mut().find(|ctx| {
            ctx.ran_ue_ngap_id == dl_nas.ran_ue_ngap_id as i64
        });

        let ue_id = match ue_ctx {
            Some(ctx) => {
                // Update AMF-UE-NGAP-ID if not set
                if ctx.amf_ue_ngap_id.is_none() {
                    ctx.amf_ue_ngap_id = Some(dl_nas.amf_ue_ngap_id as i64);
                    info!(
                        "Updated UE context: ue_id={}, amf_ue_ngap_id={}",
                        ctx.ue_id, dl_nas.amf_ue_ngap_id
                    );
                }
                ctx.ue_id
            }
            None => {
                warn!(
                    "No UE context found for RAN-UE-NGAP-ID {}",
                    dl_nas.ran_ue_ngap_id
                );
                return;
            }
        };

        // Log the NAS PDU content for debugging
        debug!(
            "NAS PDU (first 16 bytes): {:02x?}",
            &dl_nas.nas_pdu[..dl_nas.nas_pdu.len().min(16)]
        );

        // Forward NAS PDU to RRC for delivery to UE
        let msg = RrcMessage::NasDelivery {
            ue_id,
            pdu: OctetString::from_slice(&dl_nas.nas_pdu),
        };

        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send NAS delivery to RRC: {}", e);
        } else {
            info!(
                "Forwarded NAS PDU to RRC: ue_id={}, nas_len={}",
                ue_id,
                dl_nas.nas_pdu.len()
            );
        }
    }

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

        // Get stream for this UE
        let stream = self
            .ue_contexts
            .get(&ue_id)
            .map(|ctx| ctx.stream_id)
            .unwrap_or(1); // Use stream 1 for UE-associated signaling

        // Build User Location Information
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        // Convert RRC establishment cause to NGAP value
        let rrc_cause = match rrc_establishment_cause {
            0 => RrcEstablishmentCauseValue::Emergency,
            1 => RrcEstablishmentCauseValue::HighPriorityAccess,
            2 => RrcEstablishmentCauseValue::MtAccess,
            3 => RrcEstablishmentCauseValue::MoSignalling,
            4 => RrcEstablishmentCauseValue::MoData,
            5 => RrcEstablishmentCauseValue::MoVoiceCall,
            6 => RrcEstablishmentCauseValue::MoVideoCall,
            7 => RrcEstablishmentCauseValue::MoSms,
            8 => RrcEstablishmentCauseValue::MpsHighPriorityAccess,
            9 => RrcEstablishmentCauseValue::McsHighPriorityAccess,
            _ => RrcEstablishmentCauseValue::MoSignalling, // Default
        };

        // Convert GutiMobileIdentity to FiveGSTmsi if provided
        let five_g_s_tmsi = _s_tmsi.map(|guti| {
            // Convert 5G-TMSI (u32) to bytes
            let tmsi_bytes = guti.tmsi.to_be_bytes();
            FiveGSTmsi {
                amf_set_id: guti.amf_set_id,
                amf_pointer: guti.amf_pointer,
                five_g_tmsi: tmsi_bytes,
            }
        });

        let params = InitialUeMessageParams {
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            nas_pdu: pdu.data().to_vec(),
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: Tai {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
                time_stamp: None,
            },
            rrc_establishment_cause: rrc_cause,
            five_g_s_tmsi,
            amf_set_id: None,
            ue_context_request: Some(UeContextRequestValue::Requested),
            allowed_nssai: None,
        };

        match encode_initial_ue_message(&params) {
            Ok(bytes) => {
                info!(
                    "Sending Initial UE Message: ue_id={}, ran_ue_ngap_id={}, amf_id={}, stream={}, len={}",
                    ue_id, ran_ue_ngap_id, amf_id, stream, bytes.len()
                );
                self.send_ngap_ue_associated(amf_id, stream, bytes).await;
            }
            Err(e) => {
                error!("Failed to encode Initial UE Message: {}", e);
                // Clean up UE context on failure
                self.delete_ue_context(ue_id);
                return;
            }
        }

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

        let amf_ue_ngap_id = match ctx.amf_ue_ngap_id {
            Some(id) => id as u64,
            None => {
                warn!("AMF UE NGAP ID not set for UE {}", ue_id);
                return;
            }
        };

        let ran_ue_ngap_id = ctx.ran_ue_ngap_id as u32;
        let amf_ctx_id = ctx.amf_ctx_id;
        let stream = ctx.stream_id;

        // Build User Location Information
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = UplinkNasTransportParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            nas_pdu: pdu.data().to_vec(),
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: Tai {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
                time_stamp: None,
            },
        };

        match encode_uplink_nas_transport(&params) {
            Ok(bytes) => {
                info!(
                    "Sending Uplink NAS Transport: ue_id={}, ran_ue_ngap_id={}, amf_ue_ngap_id={}, amf_ctx_id={}, stream={}, len={}",
                    ue_id, ran_ue_ngap_id, amf_ue_ngap_id, amf_ctx_id, stream, bytes.len()
                );
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes).await;
            }
            Err(e) => {
                error!("Failed to encode Uplink NAS Transport: {}", e);
            }
        }
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
        self.handle_ue_context_release_request(ue_id, UeReleaseRequestCause::RadioLinkFailure).await;
    }

    /// Handles UE Context Release Request (from App or RRC)
    async fn handle_ue_context_release_request(&mut self, ue_id: i32, cause: UeReleaseRequestCause) {
        info!("UE context release request: ue_id={}, cause={:?}", ue_id, cause);

        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("UE context not found for release: ue_id={}", ue_id);
                return;
            }
        };

        let amf_ctx_id = ctx.amf_ctx_id;
        let ran_ue_ngap_id = ctx.ran_ue_ngap_id;
        let amf_ue_ngap_id = ctx.amf_ue_ngap_id;
        let stream = ctx.stream_id;

        // Send UE Context Release Request to AMF if we have the AMF UE NGAP ID
        if let Some(amf_id) = amf_ue_ngap_id {
            info!(
                "Sending UE Context Release Request: ue_id={}, ran_ue_ngap_id={}, amf_ue_ngap_id={}, amf_ctx_id={}",
                ue_id, ran_ue_ngap_id, amf_id, amf_ctx_id
            );

            // Build and send UE Context Release Request
            // For now, we just clean up locally as the encoding is not yet implemented
            // In a full implementation, we would encode and send the NGAP message here
            debug!(
                "UE Context Release Request would be sent on stream {} (encoding not yet implemented)",
                stream
            );
        } else {
            info!(
                "UE context release without AMF UE NGAP ID: ue_id={}, ran_ue_ngap_id={}",
                ue_id, ran_ue_ngap_id
            );
        }

        // Clean up UE context
        self.delete_ue_context(ue_id);

        // Notify RRC of AN release
        self.send_an_release(ue_id).await;

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
                // Try to decode as Downlink NAS Transport
                if let Ok(dl_nas) = decode_downlink_nas_transport(pdu_bytes) {
                    self.handle_downlink_nas_transport(client_id, stream, dl_nas).await;
                } else {
                    // Other NGAP message types that may be received in Ready state:
                    // - PDU Session Resource Setup Request
                    // - PDU Session Resource Release Command
                    // - UE Context Release Command
                    // - Initial Context Setup Request
                    // - Handover Request
                    // - Paging
                    // - Error Indication
                    // These require additional decoder implementations in nextgsim-ngap
                    debug!(
                        "Received operational NGAP PDU on stream {} (not yet handled, first bytes: {:02x?})",
                        stream,
                        &pdu_bytes[..pdu_bytes.len().min(16)]
                    );
                }
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

    /// Sends an NGAP PDU for UE-associated signaling (stream > 0)
    async fn send_ngap_ue_associated(&self, amf_id: i32, stream: u16, data: Vec<u8>) {
        debug!("Sending UE-associated NGAP PDU: amf_id={}, stream={}, len={}", amf_id, stream, data.len());
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
                        NgapMessage::UeContextReleaseRequest { ue_id, cause } => {
                            self.handle_ue_context_release_request(ue_id, cause).await;
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
            ignore_stream_ids: false, upf_addr: None, upf_port: 2152,
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
