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
    AppMessage, GnbTaskBase, GtpMessage, GtpUeContextUpdate, NgapMessage, PduSessionResource,
    RrcMessage, SctpMessage, StatusType, StatusUpdate, Task, TaskMessage, UeReleaseRequestCause,
};
use nextgsim_common::OctetString;

use super::amf_context::{AmfState, NgapAmfContext};
use super::mbs_context::{GnbMbsContext, MbsSessionManager, MulticastTunnelInfo, Tmgi};
use super::ue_context::{AsSecurityContext, NgapPduSession, NgapUeContext};
use crate::rrc::transaction::RrcProcedure;
use nextgsim_rrc::procedures::rrc_reconfiguration::{
    build_drb_reconfiguration_params, encode_rrc_reconfiguration,
};
use nextgsim_rrc::procedures::security_mode::{
    encode_security_mode_command, CipheringAlgorithmType, IntegrityAlgorithmType,
    SecurityAlgorithms, SecurityModeCommandParams,
};

use nextgsim_ngap::codec::{decode_ngap_pdu, encode_ngap_pdu, NGAP_PDU};
use nextgsim_ngap::procedures::amf_status_indication::{
    decode_amf_status_indication, AmfStatusIndicationData,
};
use nextgsim_ngap::procedures::error_indication::{
    decode_error_indication, encode_error_indication, error_indication_abstract_syntax_error,
    error_indication_transfer_syntax_error, CriticalityDiagnosticsInfo, ErrorIndicationData,
    ErrorIndicationParams, TriggeringMessageValue,
};
use nextgsim_ngap::procedures::handover::{
    decode_handover_command, decode_handover_preparation_failure, decode_handover_request,
    encode_handover_notify, encode_handover_request_acknowledge, encode_handover_required,
    HandoverCause, HandoverCommandData, HandoverNotifyParams, HandoverPreparationFailureData,
    HandoverRequestAcknowledgeParams, HandoverRequestData, HandoverRequiredParams,
    HandoverTypeValue, NrCgiValue as HandoverNrCgiValue, PduSessionResourceAdmittedItem,
    PduSessionResourceHoRequiredItem, TaiValue, TargetIdValue,
    UserLocationInfoNr as HandoverUserLocationInfoNr,
};
use nextgsim_ngap::procedures::initial_context_setup::{
    decode_initial_context_setup_request, encode_initial_context_setup_failure,
    encode_initial_context_setup_response, InitialContextSetupFailureParams,
    InitialContextSetupResponseParams,
};
use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi, NrCgi, Tai, UserLocationInfoNr};
use nextgsim_ngap::procedures::nas_non_delivery_indication::{
    encode_nas_non_delivery_indication, NasNonDeliveryIndicationParams,
};
use nextgsim_ngap::procedures::ng_reset::{
    decode_amf_configuration_update, decode_ng_reset, encode_amf_configuration_update_acknowledge,
    encode_ng_reset_acknowledge, AmfConfigurationUpdateAcknowledgeParams, NgResetAcknowledgeParams,
    NgResetScope, UeAssociation,
};
use nextgsim_ngap::procedures::ng_setup::{
    NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause,
};
use nextgsim_ngap::procedures::overload::{
    decode_overload_start, decode_overload_stop, OverloadData,
};
use nextgsim_ngap::procedures::paging::{decode_paging, PagingData, UePagingIdentityValue};
use nextgsim_ngap::procedures::path_switch::{
    decode_path_switch_request_acknowledge, decode_path_switch_request_failure,
    encode_path_switch_request, PathSwitchRequestAcknowledgeData, PathSwitchRequestFailureData,
    PathSwitchRequestParams, PathSwitchSessionItem, UeSecurityCapabilityBits,
};
use nextgsim_ngap::procedures::pdu_session_resource::{
    // Modify
    decode_pdu_session_resource_modify_request,
    // Release
    decode_pdu_session_resource_release_command,
    decode_pdu_session_resource_setup_request,
    encode_pdu_session_resource_modify_response,
    encode_pdu_session_resource_release_response,
    encode_pdu_session_resource_setup_response,
    PduSessionResourceFailedToModifyItem,
    PduSessionResourceFailedToSetupItem,
    PduSessionResourceModifyRequestData,
    PduSessionResourceModifyResponseItem,
    PduSessionResourceModifyResponseParams,
    PduSessionResourceReleaseCommandData,
    PduSessionResourceReleaseResponseParams,
    PduSessionResourceReleasedItem,
    PduSessionResourceSetupItem,
    PduSessionResourceSetupRequestData,
    PduSessionResourceSetupResponseItem,
    PduSessionResourceSetupResponseParams,
};
use nextgsim_ngap::procedures::transfer::{
    decode_modify_request_transfer, decode_release_command_transfer, decode_setup_request_transfer,
    encode_modify_response_transfer, encode_modify_unsuccessful_transfer,
    encode_release_response_transfer, encode_setup_response_transfer,
    encode_setup_unsuccessful_transfer, GtpTunnelInfo, ModifyResponseTransferParams,
    SetupResponseTransferParams,
};
use nextgsim_ngap::procedures::ue_context_modification::{
    decode_ue_context_modification_request, encode_ue_context_modification_failure,
    encode_ue_context_modification_response, UeContextModificationFailureCause,
    UeContextModificationRequestData,
};
use nextgsim_ngap::procedures::ue_context_release::{
    decode_ue_context_release_command, encode_ue_context_release_complete,
    encode_ue_context_release_request, UeContextReleaseCompleteParams,
    UeContextReleaseRequestParams,
};
use nextgsim_ngap::procedures::{
    build_ng_setup_request,
    // NAS Transport
    decode_downlink_nas_transport,
    // Initial UE Message
    encode_initial_ue_message,
    encode_uplink_nas_transport,
    is_ng_setup_failure,
    is_ng_setup_response,
    parse_ng_setup_failure,
    parse_ng_setup_response,
    BroadcastPlmnItem,
    DownlinkNasTransportData,
    GnbId,
    InitialUeMessageParams,
    NgSetupRequestParams,
    PagingDrx,
    RrcEstablishmentCauseValue,
    SNssai,
    SupportedTaItem,
    UeContextRequestValue,
    UplinkNasTransportParams,
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
    /// MBS session manager (Rel-17)
    mbs_sessions: MbsSessionManager,
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
            mbs_sessions: MbsSessionManager::new(),
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
        // Simple selection: pick the highest-capacity AMF that is Ready and not
        // marked unavailable by an AMF Status Indication (TS 38.413 §8.7.6).
        self.amf_contexts
            .values()
            .filter(|ctx| ctx.is_ready() && !ctx.unavailable)
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
                gnb_id_length: config.gnb_id_length,
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

                // Forward NTN timing config to RRC task if configured
                if let Some(ref ntn) = self.task_base.config.ntn_config {
                    info!(
                        "NTN mode active: satellite_type={}, propagation_delay={}us, k_offset={}",
                        ntn.satellite_type, ntn.propagation_delay_us, ntn.k_offset
                    );
                    let _ = self.task_base.rrc_tx.try_send(
                        crate::tasks::RrcMessage::NtnTimingAdvanceConfig {
                            satellite_type: ntn.satellite_type.clone(),
                            common_ta_us: ntn.common_ta_us,
                            k_offset: ntn.k_offset,
                            max_doppler_hz: ntn.max_doppler_hz,
                            autonomous_ta: ntn.autonomous_ta,
                        },
                    );
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
        amf_id: i32,
        stream: u16,
        dl_nas: DownlinkNasTransportData,
    ) {
        info!(
            "Downlink NAS Transport: amf_ue_ngap_id={}, ran_ue_ngap_id={}, nas_pdu_len={}",
            dl_nas.amf_ue_ngap_id,
            dl_nas.ran_ue_ngap_id,
            dl_nas.nas_pdu.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_ctx = self
            .ue_contexts
            .values_mut()
            .find(|ctx| ctx.ran_ue_ngap_id == dl_nas.ran_ue_ngap_id as i64);

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
                    "No UE context found for RAN-UE-NGAP-ID {}; reporting NAS non-delivery",
                    dl_nas.ran_ue_ngap_id
                );
                // TS 38.413 §8.6.4: the DL NAS message cannot be delivered (no UE
                // context), so report non-delivery to the AMF with the original
                // NAS-PDU rather than dropping it silently.
                self.send_nas_non_delivery_indication(
                    amf_id,
                    stream,
                    &dl_nas,
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownLocalUeNgapId),
                )
                .await;
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
            // TS 38.413 §8.6.4: RRC could not accept the NAS-PDU for delivery to
            // the UE — report non-delivery to the AMF.
            self.send_nas_non_delivery_indication(
                amf_id,
                stream,
                &dl_nas,
                NgSetupFailureCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost),
            )
            .await;
        } else {
            info!(
                "Forwarded NAS PDU to RRC: ue_id={}, nas_len={}",
                ue_id,
                dl_nas.nas_pdu.len()
            );
        }
    }

    /// Encode + send a NAS NON DELIVERY INDICATION (TS 38.413 §8.6.4) carrying
    /// the original NAS-PDU and a cause, UE-associated on `stream`.
    async fn send_nas_non_delivery_indication(
        &self,
        amf_id: i32,
        stream: u16,
        dl_nas: &DownlinkNasTransportData,
        cause: NgSetupFailureCause,
    ) {
        let params = NasNonDeliveryIndicationParams {
            amf_ue_ngap_id: dl_nas.amf_ue_ngap_id,
            ran_ue_ngap_id: dl_nas.ran_ue_ngap_id,
            nas_pdu: dl_nas.nas_pdu.clone(),
            cause,
        };
        match encode_nas_non_delivery_indication(&params) {
            Ok(bytes) => self.send_ngap_ue_associated(amf_id, stream, bytes).await,
            Err(e) => error!("Failed to encode NAS Non Delivery Indication: {}", e),
        }
    }

    /// Handles Initial Context Setup Request from AMF
    ///
    /// Select the NR ciphering + integrity algorithms from the UE Security
    /// Capabilities (TS 38.413 §9.3.1.86 bitmaps: MSB / 0x8000 = 128-xEA1).
    /// gNB priority: prefer algorithm 2 (AES) > 1 (SNOW3G) > 3 (ZUC) > 0 (null).
    /// Integrity never falls back to NIA0 (TS 33.501 §5.11.2): NIA2 is
    /// mandatory-to-support, so an empty integrity bitmap defaults to NIA2.
    fn select_as_algorithms(
        caps: &nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue,
    ) -> ((u8, CipheringAlgorithmType), (u8, IntegrityAlgorithmType)) {
        let ciph = {
            let m = caps.nr_encryption_algorithms;
            if m & 0x4000 != 0 {
                (2, CipheringAlgorithmType::Nea2)
            } else if m & 0x8000 != 0 {
                (1, CipheringAlgorithmType::Nea1)
            } else if m & 0x2000 != 0 {
                (3, CipheringAlgorithmType::Nea3)
            } else {
                (0, CipheringAlgorithmType::Nea0)
            }
        };
        let integ = {
            let m = caps.nr_integrity_algorithms;
            if m & 0x4000 != 0 {
                (2, IntegrityAlgorithmType::Nia2)
            } else if m & 0x8000 != 0 {
                (1, IntegrityAlgorithmType::Nia1)
            } else if m & 0x2000 != 0 {
                (3, IntegrityAlgorithmType::Nia3)
            } else {
                (2, IntegrityAlgorithmType::Nia2)
            }
        };
        (ciph, integ)
    }

    /// Establish AS security at Initial Context Setup (TS 33.501 §6.7 / §6.9,
    /// TS 38.331 §5.3.4): select algorithms from the UE Security Capabilities,
    /// derive the AS keys (K_RRCenc/int, K_UPenc/int) from KgNB (SecurityKey IE)
    /// per TS 33.501 Annex A.8, store the security context, and send the RRC
    /// SecurityModeCommand to the UE on SRB1.
    async fn activate_as_security(
        &mut self,
        ue_id: i32,
        kgnb: [u8; 32],
        caps: &nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue,
    ) {
        use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};

        let ((ciph_id, ciph_alg), (int_id, int_alg)) = Self::select_as_algorithms(caps);

        // Derive the four AS keys from KgNB (TS 33.501 Annex A.8, FC=0x69).
        let sec_ctx = AsSecurityContext {
            kgnb,
            k_rrc_enc: derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcEnc, ciph_id),
            k_rrc_int: derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcInt, int_id),
            k_up_enc: derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpEnc, ciph_id),
            k_up_int: derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpInt, int_id),
            ciphering_alg_id: ciph_id,
            integrity_alg_id: int_id,
        };
        // Wave-6 C4-final: allocate the SecurityModeCommand tid from THIS UE's
        // per-context allocator (TS 38.331 §5.3.4 / §6.3.2). Pinned to 0 on the
        // wire while C5_TYPED_DCCH_DISPATCH is off (the UE DL-DCCH dispatcher is
        // still the legacy nibble matcher, on which a non-zero tid shuffles the
        // routed leading-byte nibble); becomes the per-UE 0..3 cycle once C5
        // types both DL dispatchers.
        let rrc_transaction_id =
            if let Some(ctx) = self.ue_contexts.values_mut().find(|c| c.ue_id == ue_id) {
                ctx.as_security = Some(sec_ctx);
                ctx.transactions.allocate(RrcProcedure::SecurityMode)
            } else {
                0
            };
        info!(
            "AS security established for UE {}: ciphering=NEA{}, integrity=NIA{}",
            ue_id, ciph_id, int_id
        );

        // Encode + send the RRC SecurityModeCommand (SRB1, DL-DCCH).
        let params = SecurityModeCommandParams {
            rrc_transaction_id,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: ciph_alg,
                integrity_algorithm: Some(int_alg),
            },
        };
        match encode_security_mode_command(&params) {
            Ok(pdu) => {
                let msg = RrcMessage::SecurityModeCommand {
                    ue_id,
                    pdu: OctetString::from_slice(&pdu),
                };
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to hand RRC SecurityModeCommand to RRC task: {}", e);
                }
            }
            Err(e) => error!("Failed to encode RRC SecurityModeCommand: {}", e),
        }
    }

    /// Establishes UE security context (KgNB), stores security capabilities,
    /// forwards piggybacked NAS PDU to UE, and sends response back to AMF.
    async fn handle_initial_context_setup_request(
        &mut self,
        amf_id: i32,
        stream: u16,
        ics_req: nextgsim_ngap::procedures::initial_context_setup::InitialContextSetupRequestData,
    ) {
        info!(
            "Initial Context Setup Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, security_key_len={}",
            ics_req.amf_ue_ngap_id,
            ics_req.ran_ue_ngap_id,
            ics_req.security_key.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_ctx = self
            .ue_contexts
            .values_mut()
            .find(|ctx| ctx.ran_ue_ngap_id == ics_req.ran_ue_ngap_id as i64);

        let ue_id = match ue_ctx {
            Some(ctx) => {
                // Update AMF-UE-NGAP-ID
                if ctx.amf_ue_ngap_id.is_none() {
                    ctx.amf_ue_ngap_id = Some(ics_req.amf_ue_ngap_id as i64);
                }
                // Store the AMF-provided UE Security Capabilities and UE-AMBR as
                // the baseline; UE Context Modification may later replace them.
                ctx.ue_security_capabilities = Some(ics_req.ue_security_capabilities.clone());
                if let Some(ambr) = ics_req.ue_aggregate_max_bit_rate.clone() {
                    ctx.ue_ambr = Some(ambr);
                }
                // Transition UE state
                ctx.on_initial_context_setup();
                info!(
                    "UE context updated: ue_id={}, state={}, amf_ue_ngap_id={}",
                    ctx.ue_id, ctx.state, ics_req.amf_ue_ngap_id
                );
                ctx.ue_id
            }
            None => {
                warn!(
                    "No UE context found for RAN-UE-NGAP-ID {}",
                    ics_req.ran_ue_ngap_id
                );
                return;
            }
        };

        // TS 33.501 §6.7 / TS 38.331 §5.3.4: establish AS security — derive the
        // AS keys from KgNB, select algorithms from the UE Security
        // Capabilities, and send the RRC SecurityModeCommand — before
        // completing the context.
        self.activate_as_security(
            ue_id,
            ics_req.security_key,
            &ics_req.ue_security_capabilities,
        )
        .await;

        // Forward piggybacked NAS PDU (e.g., SecurityModeCommand) to UE via RRC
        if let Some(nas_pdu) = &ics_req.nas_pdu {
            debug!(
                "Forwarding piggybacked NAS PDU ({} bytes) to UE {}",
                nas_pdu.len(),
                ue_id
            );
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!(
                    "Failed to forward NAS PDU from InitialContextSetup to RRC: {}",
                    e
                );
            }
        }

        // Transition to Active state (security context established)
        if let Some(ctx) = self.ue_contexts.values_mut().find(|ctx| ctx.ue_id == ue_id) {
            ctx.on_context_setup_complete();
            info!("UE {} context setup complete, state={}", ue_id, ctx.state);
        }

        // amfg-01: set up any PDU sessions carried inside the ICS Request
        // (PDUSessionResourceSetupListCxtReq). Real AMFs (e.g. Open5GS) deliver
        // the first PDU session this way during registration+PDU, so the N3
        // tunnel must be established here, not only on the dedicated PDU Session
        // Resource Setup procedure.
        let carried_sessions = ics_req.pdu_session_setup_list.len();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);
        let mut cxt_res_items = Vec::new();
        let mut cxt_failed_items = Vec::new();
        for item in &ics_req.pdu_session_setup_list {
            match self.setup_one_pdu_session(ue_id, item, gnb_ip).await {
                Ok(resp) => cxt_res_items.push(resp),
                Err(failed) => cxt_failed_items.push(failed),
            }
        }

        // If the ICS Request carried PDU sessions but none could be set up,
        // answer with InitialContextSetupFailure (TS 38.413 §8.3.1.4) instead of
        // a bare success.
        if carried_sessions > 0 && cxt_res_items.is_empty() {
            warn!(
                "ICS carried {} PDU session(s) but none set up; sending InitialContextSetupFailure",
                carried_sessions
            );
            let fail_params = InitialContextSetupFailureParams {
                amf_ue_ngap_id: ics_req.amf_ue_ngap_id,
                ran_ue_ngap_id: ics_req.ran_ue_ngap_id,
                cause: NgSetupFailureCause::RadioNetwork(
                    RadioNetworkCause::RadioResourcesNotAvailable,
                ),
            };
            match encode_initial_context_setup_failure(&fail_params) {
                Ok(bytes) => self.send_ngap_ue_associated(amf_id, stream, bytes).await,
                Err(e) => error!("Failed to encode InitialContextSetupFailure: {}", e),
            }
            return;
        }

        // Send Initial Context Setup Response (with the CxtRes / FailedToSetup
        // lists when the ICS Request carried PDU sessions).
        let response_params = InitialContextSetupResponseParams {
            amf_ue_ngap_id: ics_req.amf_ue_ngap_id,
            ran_ue_ngap_id: ics_req.ran_ue_ngap_id,
            setup_list: if cxt_res_items.is_empty() {
                None
            } else {
                Some(cxt_res_items)
            },
            failed_list: if cxt_failed_items.is_empty() {
                None
            } else {
                Some(cxt_failed_items)
            },
        };
        match encode_initial_context_setup_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!(
                    "Initial Context Setup Response sent for RAN-UE-NGAP-ID {}",
                    ics_req.ran_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Initial Context Setup Response: {}", e);
            }
        }
    }

    /// Handle an inbound UE CONTEXT MODIFICATION REQUEST (TS 38.413 §8.3.4).
    ///
    /// Applies any updated UE Security Capabilities and replaced UE-AMBR to the
    /// stored UE context, and — when a new Security Key IE is present — re-derives
    /// the AS keys (TS 33.501 §6.9.2) by reusing [`Self::activate_as_security`].
    /// Replies with a UE CONTEXT MODIFICATION RESPONSE on success, or a UE CONTEXT
    /// MODIFICATION FAILURE — never an Error Indication — on error. Context state
    /// is mutated only after every failure precondition has passed, so a failed
    /// procedure leaves the UE context unchanged (TS 38.413 §8.3.4).
    ///
    /// Simulator simplification: the AS re-key is signalled to the UE by reusing
    /// [`Self::activate_as_security`]'s RRC Security Mode Command (TS 38.331
    /// §5.3.4), rather than the spec-conformant key-change-on-the-fly carrier — an
    /// RRCReconfiguration with `masterKeyUpdate` (TS 38.331 §5.3.5.3). This mirrors
    /// how Initial Context Setup treats the Security Key IE as the KgNB and is
    /// deliberate: `masterKeyUpdate` is modelled on neither the gNB RRC builder nor
    /// the UE. The four AS keys are still correctly re-derived from the new KgNB.
    async fn handle_ue_context_modification_request(
        &mut self,
        amf_id: i32,
        stream: u16,
        ucm_req: UeContextModificationRequestData,
    ) {
        info!(
            "UE Context Modification Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, rekey={}",
            ucm_req.amf_ue_ngap_id,
            ucm_req.ran_ue_ngap_id,
            ucm_req.security_key.is_some()
        );

        // Phase 1 (read-only): locate the UE by RAN-UE-NGAP-ID and resolve the
        // capabilities a possible AS re-key would use — the request's caps when
        // present, otherwise the capabilities stored at Initial Context Setup. No
        // context state is mutated here so that a failed procedure leaves the UE
        // context unchanged.
        let (ue_id, stored_caps) = match self
            .ue_contexts
            .values()
            .find(|ctx| ctx.ran_ue_ngap_id == ucm_req.ran_ue_ngap_id as i64)
        {
            Some(ctx) => (ctx.ue_id, ctx.ue_security_capabilities.clone()),
            None => {
                warn!(
                    "UE Context Modification for unknown RAN-UE-NGAP-ID {}; replying with UE CONTEXT MODIFICATION FAILURE",
                    ucm_req.ran_ue_ngap_id
                );
                self.send_ue_context_modification_failure(
                    amf_id,
                    stream,
                    ucm_req.amf_ue_ngap_id,
                    ucm_req.ran_ue_ngap_id,
                    UeContextModificationFailureCause::RadioNetwork(
                        RadioNetworkCause::UnknownLocalUeNgapId,
                    ),
                )
                .await;
                return;
            }
        };

        let rekey_caps = ucm_req.ue_security_capabilities.clone().or(stored_caps);

        // Precondition: a Security Key re-key needs the UE Security Capabilities
        // to select the AS algorithms. Fail BEFORE mutating any context state so
        // the procedure is atomic.
        if ucm_req.security_key.is_some() && rekey_caps.is_none() {
            warn!(
                "UE Context Modification for UE {ue_id} carried a Security Key but no UE Security Capabilities are available; replying with UE CONTEXT MODIFICATION FAILURE"
            );
            self.send_ue_context_modification_failure(
                amf_id,
                stream,
                ucm_req.amf_ue_ngap_id,
                ucm_req.ran_ue_ngap_id,
                UeContextModificationFailureCause::Protocol(ProtocolCause::SemanticError),
            )
            .await;
            return;
        }

        // Phase 2 (commit): every failure precondition has passed — apply the
        // non-security updates (replaced UE-AMBR and updated UE Security
        // Capabilities, TS 38.413 §8.3.4.2) to the stored UE context.
        if let Some(ctx) = self.ue_contexts.values_mut().find(|ctx| ctx.ue_id == ue_id) {
            if ctx.amf_ue_ngap_id.is_none() {
                ctx.amf_ue_ngap_id = Some(ucm_req.amf_ue_ngap_id as i64);
            }
            if let Some(ambr) = ucm_req.ue_aggregate_max_bit_rate.clone() {
                ctx.ue_ambr = Some(ambr);
            }
            if let Some(caps) = ucm_req.ue_security_capabilities.clone() {
                ctx.ue_security_capabilities = Some(caps);
            }
        }

        // AS re-keying (TS 33.501 §6.9.2): re-derive the AS key set from the new
        // KgNB. `rekey_caps` is guaranteed present here by the precondition above.
        if let (Some(new_kgnb), Some(caps)) = (ucm_req.security_key, rekey_caps.as_ref()) {
            self.activate_as_security(ue_id, new_kgnb, caps).await;
            info!("AS re-keying completed for UE {ue_id} via UE Context Modification");
        }

        // Success: UE CONTEXT MODIFICATION RESPONSE.
        match encode_ue_context_modification_response(
            ucm_req.amf_ue_ngap_id,
            ucm_req.ran_ue_ngap_id,
        ) {
            Ok(bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, bytes).await;
                info!(
                    "UE Context Modification Response sent for RAN-UE-NGAP-ID {}",
                    ucm_req.ran_ue_ngap_id
                );
            }
            Err(e) => error!("Failed to encode UE Context Modification Response: {e}"),
        }
    }

    /// Encode + send a UE-associated UE CONTEXT MODIFICATION FAILURE.
    async fn send_ue_context_modification_failure(
        &self,
        amf_id: i32,
        stream: u16,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
        cause: UeContextModificationFailureCause,
    ) {
        match encode_ue_context_modification_failure(amf_ue_ngap_id, ran_ue_ngap_id, &cause) {
            Ok(bytes) => self.send_ngap_ue_associated(amf_id, stream, bytes).await,
            Err(e) => error!("Failed to encode UE Context Modification Failure: {e}"),
        }
    }

    /// amfg-01: set up a single PDU session and return its per-session result.
    ///
    /// Decodes the APER PDUSessionResourceSetupRequestTransfer (TS 38.413
    /// §9.3.4.1), allocates the gNB DL F-TEID, stores the session in the UE
    /// context, creates the GTP-U N3 tunnel, forwards any piggybacked NAS, and
    /// builds the PDUSessionResourceSetupResponseTransfer (§9.3.4.2).
    ///
    /// Shared by the dedicated PDU Session Resource Setup procedure and the PDU
    /// sessions carried inside an INITIAL CONTEXT SETUP REQUEST
    /// (`PDUSessionResourceSetupListCxtReq`, TS 38.413 §8.3.1.2). Returns the
    /// success item, or a FailedToSetup item (with an APER unsuccessful-transfer)
    /// on any error.
    /// TS 38.331 §5.3.5.6 / TS 38.300 §16.1.4: establish the user-plane DRB for
    /// a PDU session by sending an RRCReconfiguration carrying a
    /// RadioBearerConfig (one DRB-ToAddMod with an SDAP-Config keyed to the PDU
    /// session id and the accepted QFIs in mappedQoS-FlowsToAdd) plus the
    /// matching CellGroupConfig (one RLC bearer). Requires AS security to be
    /// active (established at Initial Context Setup, C5).
    async fn establish_drb(&mut self, ue_id: i32, psi: u8, qfis: &[u8]) {
        if !self
            .ue_contexts
            .values()
            .any(|c| c.ue_id == ue_id && c.as_security.is_some())
        {
            warn!(
                "Establishing DRB for PDU session {} on UE {} without active AS security",
                psi, ue_id
            );
        }
        // One DRB per PDU session; DRB identity 1..=32, DTCH LCID above the SRBs.
        let drb_id = psi.clamp(1, 32);
        let lcid = (3 + drb_id).min(32);
        // Wave-6 C4-final: allocate the RRCReconfiguration tid from THIS UE's
        // per-context allocator (TS 38.331 §5.3.5 / §6.3.2). Pinned to 0 on the
        // wire while C5_TYPED_DCCH_DISPATCH is off (a non-zero tid shuffles the
        // leading-byte nibble the UE's legacy DL-DCCH dispatcher routes on, e.g.
        // tid 2 -> byte0 0x04 -> misrouted into the DL-information-transfer arm);
        // becomes the per-UE 0..3 cycle once C5 types both DL dispatchers.
        let rrc_transaction_id = self
            .ue_contexts
            .values_mut()
            .find(|c| c.ue_id == ue_id)
            .map(|c| c.transactions.allocate(RrcProcedure::Reconfiguration))
            .unwrap_or(0);
        let params = match build_drb_reconfiguration_params(
            rrc_transaction_id,
            psi,
            drb_id,
            lcid,
            qfis,
            true,
        ) {
            Ok(p) => p,
            Err(e) => {
                error!(
                    "Failed to build DRB reconfiguration for PDU session {}: {}",
                    psi, e
                );
                return;
            }
        };
        match encode_rrc_reconfiguration(&params) {
            Ok(pdu) => {
                let msg = RrcMessage::RrcReconfiguration {
                    ue_id,
                    pdu: OctetString::from_slice(&pdu),
                };
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to hand RRCReconfiguration to RRC task: {}", e);
                } else {
                    info!(
                        "Sent RRCReconfiguration establishing DRB {} for PDU session {} (QFIs {:?})",
                        drb_id, psi, qfis
                    );
                }
            }
            Err(e) => error!(
                "Failed to encode RRCReconfiguration for PDU session {}: {}",
                psi, e
            ),
        }
    }

    async fn setup_one_pdu_session(
        &mut self,
        ue_id: i32,
        item: &PduSessionResourceSetupItem,
        gnb_ip: std::net::IpAddr,
    ) -> Result<PduSessionResourceSetupResponseItem, PduSessionResourceFailedToSetupItem> {
        let psi = item.pdu_session_id;

        // Decode the APER PDUSessionResourceSetupRequestTransfer (TS 38.413 §9.3.4.1)
        let request = match decode_setup_request_transfer(&item.transfer) {
            Ok(req) => req,
            Err(e) => {
                warn!(
                    "PDU Session {}: failed to decode SetupRequestTransfer ({} bytes): {}",
                    psi,
                    item.transfer.len(),
                    e
                );
                let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::Protocol(
                    ProtocolCause::TransferSyntaxError,
                ))
                .unwrap_or_default();
                return Err(PduSessionResourceFailedToSetupItem {
                    pdu_session_id: psi,
                    transfer,
                });
            }
        };

        let upf_teid = request.ul_tunnel.teid;
        let upf_addr = request.ul_tunnel.address;
        // First QoS flow is the default flow for the session
        let qfi = request.qos_flows[0].qfi;

        // Allocate gNB DL TEID for the N3 tunnel
        let gnb_teid = self.next_downlink_teid();

        info!(
            "PDU Session {}: UPF TEID=0x{:08x}, UPF addr={}, QFIs={:?}, gNB TEID=0x{:08x}",
            psi,
            upf_teid,
            upf_addr,
            request.qos_flows.iter().map(|f| f.qfi).collect::<Vec<_>>(),
            gnb_teid
        );

        // Store PDU session in UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.add_pdu_session(NgapPduSession {
                psi,
                qfi: Some(qfi),
                uplink_teid: gnb_teid,
                downlink_teid: upf_teid,
                upf_address: upf_addr,
            });
        }

        // Send GTP SessionCreate to GTP task
        let resource = PduSessionResource {
            psi: psi as i32,
            qfi: Some(qfi),
            // uplink_teid = UPF's N3 TEID (where the gNB sends uplink G-PDUs);
            // downlink_teid = gNB's own TEID (where the UPF sends downlink). See gtp/task.rs:148-149.
            uplink_teid: upf_teid,
            downlink_teid: gnb_teid,
            upf_address: upf_addr,
        };
        let msg = GtpMessage::SessionCreate { ue_id, resource };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send SessionCreate to GTP: {}", e);
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
            }
            let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::RadioResourcesNotAvailable,
            ))
            .unwrap_or_default();
            return Err(PduSessionResourceFailedToSetupItem {
                pdu_session_id: psi,
                transfer,
            });
        }
        info!(
            "GTP session created for PSI={}, gNB TEID=0x{:08x}",
            psi, gnb_teid
        );

        // If NAS PDU present, forward to RRC
        if let Some(ref nas_pdu) = item.nas_pdu {
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!("Failed to forward NAS PDU from Setup Request to RRC: {}", e);
            }
        }

        // TS 38.331 §5.3.5.6: establish the PDU session's user-plane DRB via an
        // RRCReconfiguration carrying the accepted QoS flows (QFIs).
        let accepted_qfis: Vec<u8> = request.qos_flows.iter().map(|f| f.qfi).collect();
        self.establish_drb(ue_id, psi, &accepted_qfis).await;

        // Build the APER PDUSessionResourceSetupResponseTransfer (TS 38.413
        // §9.3.4.2) with the real gNB F-TEID and the accepted QoS flows
        let transfer_params = SetupResponseTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: gnb_ip,
                teid: gnb_teid,
            },
            accepted_qfis,
            failed_qos_flows: vec![],
        };
        match encode_setup_response_transfer(&transfer_params) {
            Ok(response_transfer) => Ok(PduSessionResourceSetupResponseItem {
                pdu_session_id: psi,
                transfer: response_transfer,
            }),
            Err(e) => {
                error!(
                    "Failed to encode SetupResponseTransfer for PSI {}: {}",
                    psi, e
                );
                let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::Misc(
                    nextgsim_ngap::procedures::ng_setup::MiscCause::Unspecified,
                ))
                .unwrap_or_default();
                Err(PduSessionResourceFailedToSetupItem {
                    pdu_session_id: psi,
                    transfer,
                })
            }
        }
    }

    /// Handles PDU Session Resource Setup Request from AMF
    async fn handle_pdu_session_resource_setup(
        &mut self,
        amf_id: i32,
        stream: u16,
        setup_req: PduSessionResourceSetupRequestData,
    ) {
        info!(
            "PDU Session Resource Setup Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            setup_req.amf_ue_ngap_id,
            setup_req.ran_ue_ngap_id,
            setup_req.pdu_session_resource_setup_list.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values_mut()
                .find(|ctx| ctx.ran_ue_ngap_id == setup_req.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => {
                    if ctx.amf_ue_ngap_id.is_none() {
                        ctx.amf_ue_ngap_id = Some(setup_req.amf_ue_ngap_id as i64);
                    }
                    ctx.ue_id
                }
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Resource Setup",
                        setup_req.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        let mut setup_response_items = Vec::new();
        let mut failed_items = Vec::new();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        for item in &setup_req.pdu_session_resource_setup_list {
            match self.setup_one_pdu_session(ue_id, item, gnb_ip).await {
                Ok(resp) => setup_response_items.push(resp),
                Err(failed) => failed_items.push(failed),
            }
        }

        // Build and send PDU Session Resource Setup Response
        let response_params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: setup_req.amf_ue_ngap_id,
            ran_ue_ngap_id: setup_req.ran_ue_ngap_id,
            setup_list: if setup_response_items.is_empty() {
                None
            } else {
                Some(setup_response_items)
            },
            failed_list: if failed_items.is_empty() {
                None
            } else {
                Some(failed_items)
            },
        };

        match encode_pdu_session_resource_setup_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Setup Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Setup Response: {}",
                    e
                );
            }
        }
    }

    /// Handles PDU Session Resource Modify Request from AMF
    async fn handle_pdu_session_resource_modify(
        &mut self,
        amf_id: i32,
        stream: u16,
        modify_req: PduSessionResourceModifyRequestData,
    ) {
        info!(
            "PDU Session Resource Modify Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            modify_req.amf_ue_ngap_id,
            modify_req.ran_ue_ngap_id,
            modify_req.pdu_session_resource_modify_list.len()
        );

        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values()
                .find(|ctx| ctx.ran_ue_ngap_id == modify_req.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => ctx.ue_id,
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Modify",
                        modify_req.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        let mut modify_response_items = Vec::new();
        let mut failed_items = Vec::new();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        for item in &modify_req.pdu_session_resource_modify_list {
            let psi = item.pdu_session_id;

            // Decode the APER PDUSessionResourceModifyRequestTransfer (§9.3.4.3)
            let request = match decode_modify_request_transfer(&item.transfer) {
                Ok(req) => req,
                Err(e) => {
                    warn!(
                        "PDU Session {} modify: failed to decode ModifyRequestTransfer: {}",
                        psi, e
                    );
                    if let Ok(transfer) = encode_modify_unsuccessful_transfer(
                        &NgSetupFailureCause::Protocol(ProtocolCause::TransferSyntaxError),
                    ) {
                        failed_items.push(PduSessionResourceFailedToModifyItem {
                            pdu_session_id: psi,
                            transfer,
                        });
                    }
                    continue;
                }
            };

            // The session must already exist to be modified
            let existing = self
                .ue_contexts
                .get(&ue_id)
                .and_then(|ctx| ctx.get_pdu_session(psi).cloned());
            let Some(existing) = existing else {
                warn!("PDU Session {} modify: session not established", psi);
                if let Ok(transfer) = encode_modify_unsuccessful_transfer(
                    &NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownPduSessionId),
                ) {
                    failed_items.push(PduSessionResourceFailedToModifyItem {
                        pdu_session_id: psi,
                        transfer,
                    });
                }
                continue;
            };

            // Apply UL tunnel modification if requested, otherwise keep current
            let (upf_teid, upf_addr) = match request.new_ul_tunnel {
                Some(tunnel) => (tunnel.teid, tunnel.address),
                None => (existing.downlink_teid, existing.upf_address),
            };
            let qfi = request
                .qos_flows_add_or_modify
                .first()
                .copied()
                .or(existing.qfi)
                .unwrap_or(1);

            // Re-allocate the gNB DL TEID only when the UL tunnel changed
            let gnb_teid = if request.new_ul_tunnel.is_some() {
                self.next_downlink_teid()
            } else {
                existing.uplink_teid
            };

            info!(
                "PDU Session {} modify: UPF TEID=0x{:08x}, gNB TEID=0x{:08x}",
                psi, upf_teid, gnb_teid
            );

            // Update UE context
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
                ctx.add_pdu_session(NgapPduSession {
                    psi,
                    qfi: Some(qfi),
                    uplink_teid: gnb_teid,
                    downlink_teid: upf_teid,
                    upf_address: upf_addr,
                });
            }

            // Send SessionModify to GTP task
            let resource = PduSessionResource {
                psi: psi as i32,
                qfi: Some(qfi),
                // uplink_teid = UPF's N3 TEID (where the gNB sends uplink G-PDUs);
                // downlink_teid = gNB's own TEID (where the UPF sends downlink). See gtp/task.rs:148-149.
                uplink_teid: upf_teid,
                downlink_teid: gnb_teid,
                upf_address: upf_addr,
            };
            let msg = GtpMessage::SessionModify { ue_id, resource };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionModify to GTP: {}", e);
                if let Ok(transfer) =
                    encode_modify_unsuccessful_transfer(&NgSetupFailureCause::RadioNetwork(
                        RadioNetworkCause::RadioResourcesNotAvailable,
                    ))
                {
                    failed_items.push(PduSessionResourceFailedToModifyItem {
                        pdu_session_id: psi,
                        transfer,
                    });
                }
                continue;
            }

            // Forward NAS PDU to RRC if present
            if let Some(ref nas_pdu) = item.nas_pdu {
                let msg = RrcMessage::NasDelivery {
                    ue_id,
                    pdu: OctetString::from_slice(nas_pdu),
                };
                let _ = self.task_base.rrc_tx.send(msg).await;
            }

            // Build the APER PDUSessionResourceModifyResponseTransfer (§9.3.4.4)
            let transfer_params = ModifyResponseTransferParams {
                dl_tunnel: request.new_ul_tunnel.map(|_| GtpTunnelInfo {
                    address: gnb_ip,
                    teid: gnb_teid,
                }),
                ul_tunnel: None,
                modified_qfis: request.qos_flows_add_or_modify.clone(),
                failed_qos_flows: vec![],
            };
            match encode_modify_response_transfer(&transfer_params) {
                Ok(response_transfer) => {
                    modify_response_items.push(PduSessionResourceModifyResponseItem {
                        pdu_session_id: psi,
                        transfer: response_transfer,
                    });
                }
                Err(e) => {
                    error!(
                        "Failed to encode ModifyResponseTransfer for PSI {}: {}",
                        psi, e
                    );
                }
            }
        }

        let response_params = PduSessionResourceModifyResponseParams {
            amf_ue_ngap_id: modify_req.amf_ue_ngap_id,
            ran_ue_ngap_id: modify_req.ran_ue_ngap_id,
            modify_list: if modify_response_items.is_empty() {
                None
            } else {
                Some(modify_response_items)
            },
            failed_list: if failed_items.is_empty() {
                None
            } else {
                Some(failed_items)
            },
        };

        match encode_pdu_session_resource_modify_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Modify Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Modify Response: {}",
                    e
                );
            }
        }
    }

    /// Handles PDU Session Resource Release Command from AMF
    async fn handle_pdu_session_resource_release(
        &mut self,
        amf_id: i32,
        stream: u16,
        release_cmd: PduSessionResourceReleaseCommandData,
    ) {
        info!(
            "PDU Session Resource Release Command: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            release_cmd.amf_ue_ngap_id,
            release_cmd.ran_ue_ngap_id,
            release_cmd.pdu_session_resource_to_release_list.len()
        );

        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values()
                .find(|ctx| ctx.ran_ue_ngap_id == release_cmd.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => ctx.ue_id,
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Release",
                        release_cmd.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        // Forward NAS PDU if present
        if let Some(ref nas_pdu) = release_cmd.nas_pdu {
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            let _ = self.task_base.rrc_tx.send(msg).await;
        }

        let mut released_items = Vec::new();

        for item in &release_cmd.pdu_session_resource_to_release_list {
            let psi = item.pdu_session_id;

            // Decode the APER PDUSessionResourceReleaseCommandTransfer cause (§9.3.4.11)
            match decode_release_command_transfer(&item.transfer) {
                Ok(cause) => {
                    info!("PDU Session {} release cause: {:?}", psi, cause);
                }
                Err(e) => {
                    warn!(
                        "PDU Session {}: failed to decode ReleaseCommandTransfer: {}",
                        psi, e
                    );
                }
            }

            // Remove from UE context
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
            }

            // Release GTP session
            let msg = GtpMessage::SessionRelease {
                ue_id,
                psi: psi as i32,
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionRelease to GTP: {}", e);
            }

            info!("PDU Session {} released for ue_id={}", psi, ue_id);

            // Real APER PDUSessionResourceReleaseResponseTransfer (§9.3.4.12):
            // a single SEQUENCE preamble octet
            let transfer = match encode_release_response_transfer() {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("Failed to encode ReleaseResponseTransfer: {}", e);
                    continue;
                }
            };
            released_items.push(PduSessionResourceReleasedItem {
                pdu_session_id: psi,
                transfer,
            });
        }

        let response_params = PduSessionResourceReleaseResponseParams {
            amf_ue_ngap_id: release_cmd.amf_ue_ngap_id,
            ran_ue_ngap_id: release_cmd.ran_ue_ngap_id,
            released_list: released_items,
        };

        match encode_pdu_session_resource_release_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Release Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Release Response: {}",
                    e
                );
            }
        }
    }

    /// Handles UE Context Release Command from AMF
    async fn handle_ue_context_release_command(
        &mut self,
        amf_id: i32,
        stream: u16,
        release_cmd: nextgsim_ngap::procedures::ue_context_release::UeContextReleaseCommandData,
    ) {
        use nextgsim_ngap::procedures::ue_context_release::UeNgapIds;

        info!(
            "UE Context Release Command: ids={:?}, cause={:?}",
            release_cmd.ue_ngap_ids, release_cmd.cause
        );

        // Find UE by AMF or RAN NGAP ID
        let ue_id = match &release_cmd.ue_ngap_ids {
            UeNgapIds::Pair {
                amf_ue_ngap_id,
                ran_ue_ngap_id,
            } => self
                .ue_contexts
                .values()
                .find(|ctx| {
                    ctx.ran_ue_ngap_id == *ran_ue_ngap_id as i64
                        || ctx.amf_ue_ngap_id == Some(*amf_ue_ngap_id as i64)
                })
                .map(|ctx| ctx.ue_id),
            UeNgapIds::AmfOnly(amf_id_val) => self
                .ue_contexts
                .values()
                .find(|ctx| ctx.amf_ue_ngap_id == Some(*amf_id_val as i64))
                .map(|ctx| ctx.ue_id),
        };

        let ue_id = match ue_id {
            Some(id) => id,
            None => {
                warn!("No UE context found for UE Context Release Command");
                return;
            }
        };

        // Release all PDU sessions for this UE via GTP
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }

        // Notify RRC
        self.send_an_release(ue_id).await;

        // Get NGAP IDs before deletion
        let (amf_ue_ngap_id, ran_ue_ngap_id) = {
            let ctx = self.ue_contexts.get(&ue_id);
            match ctx {
                Some(c) => (
                    c.amf_ue_ngap_id.unwrap_or(0) as u64,
                    c.ran_ue_ngap_id as u32,
                ),
                None => (0, 0),
            }
        };

        // Delete UE context
        self.delete_ue_context(ue_id);

        // Send UE Context Release Complete
        let complete_params = UeContextReleaseCompleteParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
        };

        match encode_ue_context_release_complete(&complete_params) {
            Ok(complete_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, complete_bytes)
                    .await;
                info!(
                    "UE Context Release Complete sent to AMF for ue_id={}",
                    ue_id
                );
            }
            Err(e) => {
                error!("Failed to encode UE Context Release Complete: {}", e);
            }
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
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
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
        self.handle_ue_context_release_request(ue_id, UeReleaseRequestCause::RadioLinkFailure)
            .await;
    }

    /// Handles UE Context Release Request (from App or RRC)
    async fn handle_ue_context_release_request(
        &mut self,
        ue_id: i32,
        cause: UeReleaseRequestCause,
    ) {
        info!(
            "UE context release request: ue_id={}, cause={:?}",
            ue_id, cause
        );

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
            // Map the local trigger to the NGAP cause (TS 38.413 §9.3.1.2)
            let ngap_cause = match cause {
                UeReleaseRequestCause::UserTriggered => {
                    NgSetupFailureCause::Nas(NasCause::NormalRelease)
                }
                UeReleaseRequestCause::RadioLinkFailure => {
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost)
                }
                UeReleaseRequestCause::RanOriginated => NgSetupFailureCause::RadioNetwork(
                    RadioNetworkCause::ReleaseDueToNgranGeneratedReason,
                ),
            };

            let params = UeContextReleaseRequestParams {
                amf_ue_ngap_id: amf_id as u64,
                ran_ue_ngap_id: ran_ue_ngap_id as u32,
                cause: ngap_cause,
            };

            match encode_ue_context_release_request(&params) {
                Ok(bytes) => {
                    info!(
                        "Sending UE Context Release Request: ue_id={}, ran_ue_ngap_id={}, amf_ue_ngap_id={}, amf_ctx_id={}, cause={:?}",
                        ue_id, ran_ue_ngap_id, amf_id, amf_ctx_id, cause
                    );
                    self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                        .await;

                    // Keep the context in Releasing state; cleanup happens when
                    // the AMF answers with UE Context Release Command
                    if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                        ctx.on_context_release();
                    }
                    return;
                }
                Err(e) => {
                    error!("Failed to encode UE Context Release Request: {}", e);
                    // Fall through to local cleanup
                }
            }
        } else {
            info!(
                "UE context release without AMF UE NGAP ID: ue_id={}, ran_ue_ngap_id={}",
                ue_id, ran_ue_ngap_id
            );
        }

        // Local cleanup (no AMF association for this UE, or encoding failed).
        self.local_release_ue(ue_id).await;
    }

    /// Locally release a UE's NG-associated resources without sending any further
    /// NGAP message to the AMF: delete the NGAP UE context, tell RRC to release
    /// the access-network resources, and tell GTP-U to tear down the bearers.
    ///
    /// Used for AMF-less release, and for local release on a received ERROR
    /// INDICATION (TS 38.413 §10.6, Handling of AP ID) — where echoing another
    /// NGAP message back to the AMF would be wrong.
    async fn local_release_ue(&mut self, ue_id: i32) {
        self.delete_ue_context(ue_id);
        self.send_an_release(ue_id).await;
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }
    }

    /// Handle an inbound ERROR INDICATION (TS 38.413 §8.7.5).
    ///
    /// A received Error Indication is processed locally and MUST NOT be answered
    /// with a further Error Indication — doing so risks an on-wire ping-pong. When
    /// it reports AP IDs that identify a UE-associated logical NG-connection, that
    /// connection's resources are released locally (TS 38.413 §10.6, Handling of
    /// AP ID). A non-UE-associated Error Indication (or one whose AP IDs match no
    /// known UE) is logged and dropped.
    async fn handle_error_indication(&mut self, amf_id: i32, err_ind: ErrorIndicationData) {
        info!(
            "Received ERROR INDICATION from AMF[{}]: amf_ue_ngap_id={:?}, ran_ue_ngap_id={:?}, cause={:?}",
            amf_id, err_ind.amf_ue_ngap_id, err_ind.ran_ue_ngap_id, err_ind.cause
        );

        let ue_id = self
            .ue_contexts
            .values()
            .find(|ctx| {
                matches!(err_ind.ran_ue_ngap_id, Some(r) if ctx.ran_ue_ngap_id == r as i64)
                    || matches!(err_ind.amf_ue_ngap_id, Some(a) if ctx.amf_ue_ngap_id == Some(a as i64))
            })
            .map(|ctx| ctx.ue_id);

        match ue_id {
            Some(id) => {
                warn!(
                    "ERROR INDICATION references UE {}; locally releasing its NG connection (TS 38.413 §10.6)",
                    id
                );
                self.local_release_ue(id).await;
            }
            None => {
                debug!(
                    "ERROR INDICATION is not associated with a known UE context; processed with no local release and no reply"
                );
            }
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
                    self.handle_downlink_nas_transport(client_id, stream, dl_nas)
                        .await;
                } else if let Ok(setup_req) = decode_pdu_session_resource_setup_request(pdu_bytes) {
                    self.handle_pdu_session_resource_setup(client_id, stream, setup_req)
                        .await;
                } else if let Ok(modify_req) = decode_pdu_session_resource_modify_request(pdu_bytes)
                {
                    self.handle_pdu_session_resource_modify(client_id, stream, modify_req)
                        .await;
                } else if let Ok(release_cmd) =
                    decode_pdu_session_resource_release_command(pdu_bytes)
                {
                    self.handle_pdu_session_resource_release(client_id, stream, release_cmd)
                        .await;
                } else if let Ok(ue_release_cmd) = decode_ue_context_release_command(pdu_bytes) {
                    self.handle_ue_context_release_command(client_id, stream, ue_release_cmd)
                        .await;
                } else if let Ok(ho_cmd) = decode_handover_command(pdu_bytes) {
                    self.handle_handover_command(client_id, stream, ho_cmd)
                        .await;
                } else if let Ok(ho_req) = decode_handover_request(pdu_bytes) {
                    self.handle_handover_request(client_id, stream, ho_req)
                        .await;
                } else if let Ok(ho_fail) = decode_handover_preparation_failure(pdu_bytes) {
                    self.handle_handover_preparation_failure(client_id, stream, ho_fail)
                        .await;
                } else if let Ok(ps_ack) = decode_path_switch_request_acknowledge(pdu_bytes) {
                    self.handle_path_switch_request_acknowledge(client_id, stream, ps_ack)
                        .await;
                } else if let Ok(ps_fail) = decode_path_switch_request_failure(pdu_bytes) {
                    self.handle_path_switch_request_failure(client_id, stream, ps_fail)
                        .await;
                } else if let Ok(ics_req) = decode_initial_context_setup_request(pdu_bytes) {
                    self.handle_initial_context_setup_request(client_id, stream, ics_req)
                        .await;
                } else if let Ok(ucm_req) = decode_ue_context_modification_request(pdu_bytes) {
                    self.handle_ue_context_modification_request(client_id, stream, ucm_req)
                        .await;
                } else if let Ok(reset) = decode_ng_reset(pdu_bytes) {
                    self.handle_ng_reset(client_id, stream, reset).await;
                } else if let Ok(update) = decode_amf_configuration_update(pdu_bytes) {
                    self.handle_amf_configuration_update(client_id, stream, update)
                        .await;
                } else if let Ok(paging) = decode_paging(pdu_bytes) {
                    self.handle_paging(client_id, paging).await;
                } else if let Ok(overload) = decode_overload_start(pdu_bytes) {
                    self.handle_overload_start(client_id, overload).await;
                } else if decode_overload_stop(pdu_bytes).is_ok() {
                    self.handle_overload_stop(client_id).await;
                } else if let Ok(status) = decode_amf_status_indication(pdu_bytes) {
                    self.handle_amf_status_indication(client_id, status).await;
                } else if let Ok(err_ind) = decode_error_indication(pdu_bytes) {
                    // TS 38.413 §8.7.5: a received Error Indication is processed
                    // locally and MUST NOT be answered with another Error
                    // Indication (which would create an on-wire ping-pong).
                    self.handle_error_indication(client_id, err_ind).await;
                } else {
                    // amfg-05: this operational PDU could not be routed to a
                    // handler. Per TS 38.413 §8.7.5 / §10, answer with an NGAP
                    // Error Indication rather than silently dropping it.
                    self.handle_unroutable_pdu(client_id, pdu_bytes).await;
                }
            }
            _ => {
                warn!("Received NGAP PDU in unexpected AMF state: {:?}", amf_state);
            }
        }
    }

    // ========================================================================
    // NG Reset / AMF Configuration Update (amfg-07)
    // ========================================================================

    /// Handles an inbound NG Reset (TS 38.413 §8.7.4.2).
    ///
    /// Clears the indicated UE-associated logical NG-connections — all of them
    /// for a `NG Interface` reset, or the listed subset for a
    /// `Part of NG Interface` reset — and replies with an NG Reset Acknowledge
    /// echoing the associations actually released.
    async fn handle_ng_reset(
        &mut self,
        client_id: i32,
        stream: u16,
        reset: nextgsim_ngap::procedures::ng_reset::NgResetData,
    ) {
        info!(
            "NG Reset received from AMF[{}]: scope={:?}, cause={:?}",
            client_id, reset.scope, reset.cause
        );

        // Collect the UE ids to release, recording the released associations.
        let mut ue_ids: Vec<i32> = Vec::new();
        let mut released: Vec<UeAssociation> = Vec::new();

        match &reset.scope {
            NgResetScope::All => {
                for ctx in self.ue_contexts.values() {
                    if ctx.amf_ctx_id == client_id {
                        ue_ids.push(ctx.ue_id);
                        released.push(UeAssociation {
                            amf_ue_ngap_id: ctx.amf_ue_ngap_id.map(|v| v as u64),
                            ran_ue_ngap_id: Some(ctx.ran_ue_ngap_id as u32),
                        });
                    }
                }
            }
            NgResetScope::Part(list) => {
                for assoc in list {
                    // Match by RAN UE NGAP ID first, then AMF UE NGAP ID.
                    let found = self.ue_contexts.values().find(|ctx| {
                        ctx.amf_ctx_id == client_id
                            && ((assoc.ran_ue_ngap_id.is_some()
                                && Some(ctx.ran_ue_ngap_id as u32) == assoc.ran_ue_ngap_id)
                                || (assoc.amf_ue_ngap_id.is_some()
                                    && ctx.amf_ue_ngap_id.map(|v| v as u64)
                                        == assoc.amf_ue_ngap_id))
                    });
                    if let Some(ctx) = found {
                        ue_ids.push(ctx.ue_id);
                        released.push(UeAssociation {
                            amf_ue_ngap_id: ctx.amf_ue_ngap_id.map(|v| v as u64),
                            ran_ue_ngap_id: Some(ctx.ran_ue_ngap_id as u32),
                        });
                    } else {
                        // Echo back the requested association even if unknown,
                        // so the AMF can complete its bookkeeping.
                        released.push(assoc.clone());
                    }
                }
            }
        }

        for ue_id in ue_ids {
            self.release_ue_for_reset(ue_id).await;
        }

        // Build the Acknowledge. Always include the (possibly empty) released
        // list so the AMF sees exactly which associations were cleared.
        let params = NgResetAcknowledgeParams {
            released: Some(released),
        };
        match encode_ng_reset_acknowledge(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(client_id, stream, data).await;
                info!("Sent NG Reset Acknowledge to AMF[{}]", client_id);
            }
            Err(e) => error!("Failed to encode NG Reset Acknowledge: {}", e),
        }
    }

    /// Releases a UE context as part of an NG Reset: drop the NGAP context and
    /// notify the RRC and GTP tasks (no UE Context Release Complete is sent —
    /// NG Reset is acknowledged at the interface level).
    async fn release_ue_for_reset(&mut self, ue_id: i32) {
        self.delete_ue_context(ue_id);
        self.send_an_release(ue_id).await;
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }
    }

    /// Handle an inbound OVERLOAD START (TS 38.413 §8.7.7): mark the AMF as
    /// overloaded and record the traffic-load-reduction indication. Because
    /// `select_amf` only picks `Ready` AMFs, an overloaded AMF is automatically
    /// diverted from new UE routing. No reply is sent (Overload Start is a
    /// class-2 procedure), and it is no longer answered with an Error Indication.
    async fn handle_overload_start(&mut self, amf_id: i32, overload: OverloadData) {
        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
            ctx.on_overload_start();
            ctx.traffic_load_reduction = overload.traffic_load_reduction;
            info!(
                "AMF[{}] entered OVERLOAD (state={:?}, traffic_load_reduction={:?})",
                amf_id, ctx.state, ctx.traffic_load_reduction
            );
        } else {
            warn!("OVERLOAD START for unknown AMF association {}", amf_id);
        }
    }

    /// Handle an inbound OVERLOAD STOP (TS 38.413 §8.7.8): clear the overload
    /// state so the AMF is eligible for UE routing again.
    async fn handle_overload_stop(&mut self, amf_id: i32) {
        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
            ctx.on_overload_stop();
            ctx.traffic_load_reduction = None;
            info!("AMF[{}] cleared OVERLOAD (state={:?})", amf_id, ctx.state);
        } else {
            warn!("OVERLOAD STOP for unknown AMF association {}", amf_id);
        }
    }

    /// Handle an inbound AMF STATUS INDICATION (TS 38.413 §8.7.6): mark every AMF
    /// context that serves one of the reported unavailable GUAMIs as unavailable
    /// for UE selection, so `select_amf` reselects toward another (backup) AMF.
    /// No reply is sent, and it is no longer answered with an Error Indication.
    /// Migrating already-connected UEs to the backup AMF is out of scope here;
    /// they reselect on their next Initial UE Message.
    async fn handle_amf_status_indication(&mut self, amf_id: i32, status: AmfStatusIndicationData) {
        if status.unavailable_guami_list.is_empty() {
            warn!("AMF Status Indication from AMF[{amf_id}] with empty unavailable GUAMI list");
            return;
        }
        let mut marked = 0;
        for ctx in self.amf_contexts.values_mut() {
            let hit = status.unavailable_guami_list.iter().any(|u| {
                ctx.served_guami_list.iter().any(|s| {
                    u.guami.plmn_identity == s.guami.plmn_identity
                        && u.guami.amf_region_id == s.guami.amf_region_id
                        && u.guami.amf_set_id == s.guami.amf_set_id
                        && u.guami.amf_pointer == s.guami.amf_pointer
                })
            });
            if hit {
                ctx.unavailable = true;
                marked += 1;
            }
        }
        info!(
            "AMF Status Indication from AMF[{}]: {} unavailable GUAMI(s) -> {} AMF context(s) excluded from selection",
            amf_id,
            status.unavailable_guami_list.len(),
            marked
        );
    }

    /// Handles an inbound AMF Configuration Update (TS 38.413 §8.7.3).
    ///
    /// Updates the stored AMF configuration (name, relative capacity, served
    /// GUAMI count) and replies with an AMF Configuration Update Acknowledge.
    async fn handle_amf_configuration_update(
        &mut self,
        client_id: i32,
        stream: u16,
        update: nextgsim_ngap::procedures::ng_reset::AmfConfigurationUpdateData,
    ) {
        info!(
            "AMF Configuration Update from AMF[{}]: name={:?}, served_guami={}, capacity={:?}, \
             plmn_support={}",
            client_id,
            update.amf_name,
            update.served_guami_count,
            update.relative_amf_capacity,
            update.has_plmn_support_list
        );

        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            if let Some(ref name) = update.amf_name {
                ctx.amf_name = Some(name.clone());
            }
            if let Some(cap) = update.relative_amf_capacity {
                ctx.relative_capacity = cap;
            }
        }

        let params = AmfConfigurationUpdateAcknowledgeParams::default();
        match encode_amf_configuration_update_acknowledge(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(client_id, stream, data).await;
                info!(
                    "Sent AMF Configuration Update Acknowledge to AMF[{}]",
                    client_id
                );
            }
            Err(e) => error!(
                "Failed to encode AMF Configuration Update Acknowledge: {}",
                e
            ),
        }
    }

    // ========================================================================
    // Paging (amfg-06)
    // ========================================================================

    /// Handles an inbound PAGING from the AMF (TS 38.413 §8.6.1 / §9.2.3.1).
    ///
    /// Matches the TAIListForPaging against this gNB's served TAI and, on a
    /// match, triggers RRC Paging toward the air interface carrying the
    /// 5G-S-TMSI. PagingPriority / PagingDRX are honored on a best-effort
    /// (logged) basis.
    async fn handle_paging(&mut self, client_id: i32, paging: PagingData) {
        let config = &self.task_base.config;
        let served_plmn = config.plmn.encode();
        let served_tac = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        // Collect the TAIs from the paging list that this gNB serves.
        let matching: Vec<&nextgsim_ngap::procedures::initial_ue_message::Tai> = paging
            .tai_list_for_paging
            .iter()
            .filter(|tai| tai.plmn_identity == served_plmn && tai.tac == served_tac)
            .collect();

        if matching.is_empty() {
            debug!(
                "PAGING from AMF[{}] not for a served TAI (served plmn={:02x?} tac={:02x?}); ignoring",
                client_id, served_plmn, served_tac
            );
            return;
        }

        if paging.paging_priority.is_some() || paging.paging_drx.is_some() {
            debug!(
                "PAGING QoS hints: priority={:?}, drx={:?}",
                paging.paging_priority, paging.paging_drx
            );
        }

        // Serialize the 5G-S-TMSI for the RRC Paging record.
        let ue_paging_tmsi = serialize_five_g_s_tmsi(&paging.ue_paging_identity);

        // Serialize the matching TAIs (plmn(3) + tac(3) per entry).
        let mut tai_list_for_paging = Vec::with_capacity(matching.len() * 6);
        for tai in &matching {
            tai_list_for_paging.extend_from_slice(&tai.plmn_identity);
            tai_list_for_paging.extend_from_slice(&tai.tac);
        }

        info!(
            "PAGING from AMF[{}]: matched {} served TAI(s); triggering RRC Paging",
            client_id,
            matching.len()
        );

        let msg = RrcMessage::Paging {
            ue_paging_tmsi,
            tai_list_for_paging,
        };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to forward Paging to RRC: {}", e);
        }
    }

    // ========================================================================
    // Handover Procedures
    // ========================================================================

    /// Handles Handover Command from AMF (source gNB side)
    /// AMF sends this after receiving `HandoverRequestAcknowledge` from target gNB.
    /// Source gNB must forward the transparent container to UE via RRC and
    /// release resources after UE completes handover.
    async fn handle_handover_command(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ho_cmd: HandoverCommandData,
    ) {
        info!(
            "Handover Command received: amf_ue_ngap_id={}, ran_ue_ngap_id={}, type={:?}",
            ho_cmd.amf_ue_ngap_id, ho_cmd.ran_ue_ngap_id, ho_cmd.handover_type
        );

        // Find the UE by RAN UE NGAP ID
        let ue_id = self
            .find_ue_by_ran_id(ho_cmd.ran_ue_ngap_id as i64)
            .map(|ctx| ctx.ue_id);

        if let Some(ue_id) = ue_id {
            // Forward the Target-to-Source Transparent Container to UE via RRC
            // This contains the RRC Reconfiguration with mobility control info
            let container = OctetString::from_slice(&ho_cmd.target_to_source_transparent_container);
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: container,
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!("Failed to forward handover command to RRC: {}", e);
            }

            // Release PDU sessions that failed handover
            if let Some(ref release_list) = ho_cmd.pdu_session_resource_to_release_list {
                for item in release_list {
                    let msg = GtpMessage::SessionRelease {
                        ue_id,
                        psi: item.pdu_session_id as i32,
                    };
                    if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                        error!(
                            "Failed to release PDU session {}: {}",
                            item.pdu_session_id, e
                        );
                    }
                }
            }

            info!(
                "Handover Command processed for UE[{}], forwarded to RRC",
                ue_id
            );
        } else {
            warn!(
                "Handover Command for unknown RAN UE NGAP ID: {}",
                ho_cmd.ran_ue_ngap_id
            );
        }
    }

    /// Handles Handover Request from AMF (target gNB side)
    /// AMF sends this to the target gNB to prepare handover resources.
    /// Target gNB must allocate resources and respond with `HandoverRequestAcknowledge`.
    async fn handle_handover_request(
        &mut self,
        client_id: i32,
        stream: u16,
        ho_req: HandoverRequestData,
    ) {
        info!(
            "Handover Request received (target gNB): amf_ue_ngap_id={}, type={:?}",
            ho_req.amf_ue_ngap_id, ho_req.handover_type
        );

        // Allocate a new UE context for the incoming handover
        if let Some(ran_ue_ngap_id) = self.create_ue_context(
            self.ue_contexts.len() as i32 + 1000, // Handover UE IDs start from 1000
            client_id,
        ) {
            // Set AMF UE NGAP ID on the new context
            let ue_id = self.find_ue_by_ran_id(ran_ue_ngap_id).map(|ctx| ctx.ue_id);

            if let Some(ue_id) = ue_id {
                if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                    ctx.amf_ue_ngap_id = Some(ho_req.amf_ue_ngap_id as i64);
                }
            }

            // Build HandoverRequestAcknowledge - admit all PDU sessions
            let admitted_list = vec![PduSessionResourceAdmittedItem {
                pdu_session_id: 1,                         // Default PDU session
                handover_request_ack_transfer: vec![0x00], // Minimal transfer
            }];

            let target_to_source_container = ho_req.source_to_target_transparent_container.clone();

            match encode_handover_request_acknowledge(&HandoverRequestAcknowledgeParams {
                amf_ue_ngap_id: ho_req.amf_ue_ngap_id,
                ran_ue_ngap_id: ran_ue_ngap_id as u32,
                pdu_session_resource_admitted_list: admitted_list,
                pdu_session_resource_failed_list: None,
                target_to_source_transparent_container: target_to_source_container,
            }) {
                Ok(data) => {
                    self.send_ngap_ue_associated(client_id, stream, data).await;
                    info!(
                        "Sent Handover Request Acknowledge: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                        ho_req.amf_ue_ngap_id, ran_ue_ngap_id
                    );
                }
                Err(e) => {
                    error!("Failed to encode Handover Request Acknowledge: {}", e);
                }
            }
        } else {
            error!(
                "Failed to create UE context for handover, amf_ue_ngap_id={}",
                ho_req.amf_ue_ngap_id
            );
        }
    }

    /// Handles Handover Preparation Failure from AMF (source gNB side)
    /// AMF sends this when the target gNB rejected the handover.
    async fn handle_handover_preparation_failure(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ho_fail: HandoverPreparationFailureData,
    ) {
        warn!(
            "Handover Preparation Failure: amf_ue_ngap_id={}, ran_ue_ngap_id={}, cause={:?}",
            ho_fail.amf_ue_ngap_id, ho_fail.ran_ue_ngap_id, ho_fail.cause
        );

        // Handover failed - UE stays on source gNB, no action needed
        if let Some(ctx) = self.find_ue_by_ran_id(ho_fail.ran_ue_ngap_id as i64) {
            info!(
                "Handover preparation failed for UE[{}], staying on source cell",
                ctx.ue_id
            );
        }
    }

    /// Initiates a handover by sending `HandoverRequired` to AMF (source gNB side)
    /// Called when measurement reports indicate better target cell
    #[allow(dead_code)]
    async fn initiate_handover(&mut self, ue_id: i32, target_gnb_id: u32, target_tac: u32) {
        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("Cannot initiate handover for unknown UE[{}]", ue_id);
                return;
            }
        };

        let amf_ue_ngap_id = match ctx.amf_ue_ngap_id {
            Some(id) => id as u64,
            None => {
                warn!("UE[{}] has no AMF UE NGAP ID, cannot handover", ue_id);
                return;
            }
        };

        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();

        let target_tac_bytes = [
            ((target_tac >> 16) & 0xFF) as u8,
            ((target_tac >> 8) & 0xFF) as u8,
            (target_tac & 0xFF) as u8,
        ];

        // Build source-to-target transparent container (simplified)
        let source_container = vec![
            0x00, // Placeholder - real implementation would include
                 // RRC container with UE capabilities and measurement data
        ];

        // Collect PDU sessions for handover
        let pdu_sessions: Vec<PduSessionResourceHoRequiredItem> = ctx
            .pdu_sessions
            .values()
            .map(|sess| PduSessionResourceHoRequiredItem {
                pdu_session_id: sess.psi,
                handover_required_transfer: vec![0x00], // Minimal transfer
            })
            .collect();

        if pdu_sessions.is_empty() {
            // Add at least a default entry
            let pdu_sessions = vec![PduSessionResourceHoRequiredItem {
                pdu_session_id: 1,
                handover_required_transfer: vec![0x00],
            }];

            self.send_handover_required(
                ctx.amf_ctx_id,
                amf_ue_ngap_id,
                ctx.ran_ue_ngap_id as u32,
                &plmn_bytes,
                target_gnb_id,
                &target_tac_bytes,
                &source_container,
                &pdu_sessions,
            )
            .await;
        } else {
            self.send_handover_required(
                ctx.amf_ctx_id,
                amf_ue_ngap_id,
                ctx.ran_ue_ngap_id as u32,
                &plmn_bytes,
                target_gnb_id,
                &target_tac_bytes,
                &source_container,
                &pdu_sessions,
            )
            .await;
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn send_handover_required(
        &self,
        amf_client_id: i32,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
        plmn_bytes: &[u8; 3],
        target_gnb_id: u32,
        target_tac_bytes: &[u8; 3],
        source_container: &[u8],
        pdu_sessions: &[PduSessionResourceHoRequiredItem],
    ) {
        let params = HandoverRequiredParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            handover_type: HandoverTypeValue::Intra5gs,
            cause: HandoverCause::RadioNetwork(
                nextgsim_ngap::procedures::ng_setup::RadioNetworkCause::HandoverDesirableForRadioReason,
            ),
            target_id: TargetIdValue::TargetRanNodeId {
                global_ran_node_id: target_gnb_id.to_be_bytes().to_vec(),
                selected_tai: TaiValue {
                    plmn_identity: *plmn_bytes,
                    tac: *target_tac_bytes,
                },
            },
            direct_forwarding_path_availability: None,
            pdu_session_resource_list: pdu_sessions.to_vec(),
            source_to_target_transparent_container: source_container.to_vec(),
        };

        match encode_handover_required(&params) {
            Ok(data) => {
                self.send_ngap_ue_associated(amf_client_id, 1, data).await;
                info!(
                    "Sent Handover Required: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                    amf_ue_ngap_id, ran_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Handover Required: {}", e);
            }
        }
    }

    /// Sends Handover Notify to AMF (target gNB side)
    /// Called after UE has completed handover to target cell
    #[allow(dead_code)]
    async fn send_handover_notify(
        &self,
        amf_client_id: i32,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
    ) {
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = HandoverNotifyParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            user_location_info: HandoverUserLocationInfoNr {
                nr_cgi: HandoverNrCgiValue {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: TaiValue {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
            },
        };

        match encode_handover_notify(&params) {
            Ok(data) => {
                self.send_ngap_ue_associated(amf_client_id, 1, data).await;
                info!(
                    "Sent Handover Notify: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                    amf_ue_ngap_id, ran_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Handover Notify: {}", e);
            }
        }
    }

    // ========================================================================
    // Path Switch Procedure (TS 38.413 Section 8.4.4)
    // ========================================================================

    /// Sends a Path Switch Request to the AMF (target gNB side, Xn handover).
    ///
    /// Requests the 5GC to switch the DL GTP-U termination point to this gNB
    /// for all PDU sessions of the given UE. The per-session
    /// `PathSwitchRequestTransfer` carries the real DL F-TEID allocated here.
    #[allow(dead_code)]
    async fn send_path_switch_request(&mut self, ue_id: i32, source_amf_ue_ngap_id: u64) {
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        let (amf_ctx_id, ran_ue_ngap_id, stream, sessions) = {
            let ctx = match self.ue_contexts.get(&ue_id) {
                Some(c) => c,
                None => {
                    warn!("Cannot send Path Switch Request for unknown UE[{}]", ue_id);
                    return;
                }
            };
            let sessions: Vec<PathSwitchSessionItem> = ctx
                .pdu_sessions
                .values()
                .map(|sess| PathSwitchSessionItem {
                    pdu_session_id: sess.psi,
                    dl_tunnel: GtpTunnelInfo {
                        address: gnb_ip,
                        teid: sess.uplink_teid,
                    },
                    accepted_qfis: vec![sess.qfi.unwrap_or(1)],
                })
                .collect();
            (ctx.amf_ctx_id, ctx.ran_ue_ngap_id, ctx.stream_id, sessions)
        };

        if sessions.is_empty() {
            warn!(
                "Cannot send Path Switch Request for UE[{}]: no PDU sessions",
                ue_id
            );
            return;
        }

        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = PathSwitchRequestParams {
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            source_amf_ue_ngap_id,
            nr_cgi: NrCgi {
                plmn_identity: plmn_bytes,
                nr_cell_identity: config.nci,
            },
            tai: Tai {
                plmn_identity: plmn_bytes,
                tac: tac_bytes,
            },
            ue_security_capabilities: UeSecurityCapabilityBits::default(),
            sessions,
        };

        match encode_path_switch_request(&params) {
            Ok(bytes) => {
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
                info!(
                    "Sent Path Switch Request: ue_id={}, ran_ue_ngap_id={}, source_amf_ue_ngap_id={}",
                    ue_id, ran_ue_ngap_id, source_amf_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Path Switch Request: {}", e);
            }
        }
    }

    /// Handles Path Switch Request Acknowledge from AMF.
    ///
    /// Switches the UL GTP-U tunnels to the (possibly new) UPF endpoints and
    /// adopts the fresh NH security context.
    async fn handle_path_switch_request_acknowledge(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ack: PathSwitchRequestAcknowledgeData,
    ) {
        info!(
            "Path Switch Request Acknowledge: amf_ue_ngap_id={}, ran_ue_ngap_id={}, ncc={}, {} switched sessions",
            ack.amf_ue_ngap_id,
            ack.ran_ue_ngap_id,
            ack.next_hop_chaining_count,
            ack.switched_sessions.len()
        );

        let ue_id = match self.find_ue_by_ran_id(ack.ran_ue_ngap_id as i64) {
            Some(ctx) => ctx.ue_id,
            None => {
                warn!(
                    "Path Switch Request Acknowledge for unknown RAN-UE-NGAP-ID {}",
                    ack.ran_ue_ngap_id
                );
                return;
            }
        };

        // Adopt the AMF-assigned ID (it can change across the path switch)
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.amf_ue_ngap_id = Some(ack.amf_ue_ngap_id as i64);
        }

        // Update UL tunnels for switched sessions and notify the GTP task
        for session in &ack.switched_sessions {
            let updated = self.ue_contexts.get_mut(&ue_id).and_then(|ctx| {
                ctx.pdu_sessions.get_mut(&session.pdu_session_id).map(|s| {
                    if let Some(tunnel) = session.ul_tunnel {
                        s.downlink_teid = tunnel.teid;
                        s.upf_address = tunnel.address;
                    }
                    s.clone()
                })
            });

            if let Some(s) = updated {
                let resource = PduSessionResource {
                    psi: s.psi as i32,
                    qfi: s.qfi,
                    // NgapPduSession stores UPF TEID in downlink_teid and gNB TEID in
                    // uplink_teid (internal convention); the GTP task expects the opposite
                    // (uplink_teid = UPF dest). Map across the boundary here.
                    uplink_teid: s.downlink_teid,
                    downlink_teid: s.uplink_teid,
                    upf_address: s.upf_address,
                };
                let msg = GtpMessage::SessionModify { ue_id, resource };
                if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                    error!(
                        "Failed to send SessionModify after path switch for PSI {}: {}",
                        session.pdu_session_id, e
                    );
                }
            } else {
                warn!(
                    "Path switch acknowledged unknown PDU session {}",
                    session.pdu_session_id
                );
            }
        }

        info!("Path switch completed for UE[{}]", ue_id);
    }

    /// Handles Path Switch Request Failure from AMF.
    ///
    /// Per TS 38.413 §8.4.4.3 the gNB releases the UE-associated resources
    /// when the path switch is rejected.
    async fn handle_path_switch_request_failure(
        &mut self,
        _client_id: i32,
        _stream: u16,
        failure: PathSwitchRequestFailureData,
    ) {
        warn!(
            "Path Switch Request Failure: amf_ue_ngap_id={}, ran_ue_ngap_id={}, released sessions={:?}",
            failure.amf_ue_ngap_id, failure.ran_ue_ngap_id, failure.released_sessions
        );

        let ue_id = match self.find_ue_by_ran_id(failure.ran_ue_ngap_id as i64) {
            Some(ctx) => ctx.ue_id,
            None => {
                warn!(
                    "Path Switch Request Failure for unknown RAN-UE-NGAP-ID {}",
                    failure.ran_ue_ngap_id
                );
                return;
            }
        };

        // Release the sessions the 5GC reported as released (with cause)
        for (psi, cause) in &failure.released_sessions {
            info!(
                "Releasing PDU session {} after path switch failure (cause: {:?})",
                psi, cause
            );
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(*psi);
            }
            let msg = GtpMessage::SessionRelease {
                ue_id,
                psi: *psi as i32,
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionRelease to GTP: {}", e);
            }
        }

        // The UE context at the target gNB is released after a failed switch
        self.handle_ue_context_release_request(ue_id, UeReleaseRequestCause::RanOriginated)
            .await;
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Handles an operational NGAP PDU that could not be routed to a handler.
    ///
    /// Per TS 38.413 §8.7.5 (Error Indication) and §10 (handling of
    /// unknown/erroneous messages) the NG-RAN node answers with an Error
    /// Indication. We distinguish two cases:
    ///   * the PDU did not APER-decode at all → transfer-syntax-error (no
    ///     CriticalityDiagnostics, since the procedure code is unknown);
    ///   * the PDU decoded but no handler exists for the procedure →
    ///     abstract-syntax-error (ignore-and-notify) with CriticalityDiagnostics
    ///     carrying the offending procedure code.
    async fn handle_unroutable_pdu(&self, amf_id: i32, pdu_bytes: &[u8]) {
        let params = unroutable_error_params(pdu_bytes);
        match decode_ngap_pdu(pdu_bytes) {
            Ok(_) => warn!(
                "Unsupported NGAP procedure from AMF[{}]; sending Error Indication \
                 (abstract syntax error)",
                amf_id
            ),
            Err(e) => warn!(
                "Undecodable NGAP PDU from AMF[{}] ({}); sending Error Indication \
                 (transfer syntax error)",
                amf_id, e
            ),
        }
        self.send_error_indication(amf_id, params).await;
    }

    /// Encodes and sends an NGAP Error Indication to the AMF.
    ///
    /// Error Indication is non-UE-associated here (the offending PDU could not
    /// be tied to a known UE context), so it is sent on stream 0.
    async fn send_error_indication(&self, amf_id: i32, params: ErrorIndicationParams) {
        match encode_error_indication(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(amf_id, 0, data).await;
            }
            Err(e) => {
                error!("Failed to encode Error Indication: {}", e);
            }
        }
    }

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
        debug!(
            "Sending UE-associated NGAP PDU: amf_id={}, stream={}, len={}",
            amf_id,
            stream,
            data.len()
        );
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
        let any_ready = self
            .amf_contexts
            .values()
            .any(super::amf_context::NgapAmfContext::is_ready);

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

    // ========================================================================
    // MBS (Multicast/Broadcast Service) Procedures (Rel-17)
    // ========================================================================

    /// Handles MBS Session Activation Request from AMF
    async fn handle_mbs_session_activation_request(
        &mut self,
        session_id: u32,
        tmgi_bytes: [u8; 6],
        is_broadcast: bool,
        multicast_ip: Option<std::net::IpAddr>,
        qfi: u8,
    ) {
        info!(
            "MBS Session Activation Request: session_id={}, tmgi={:02x?}, broadcast={}, qfi={}",
            session_id, tmgi_bytes, is_broadcast, qfi
        );

        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        let mut mbs_ctx = GnbMbsContext::new(session_id, tmgi, is_broadcast);

        // Create multicast tunnel info if provided
        if let Some(mcast_ip) = multicast_ip {
            let tnl_info = MulticastTunnelInfo {
                multicast_ip: mcast_ip,
                source_ip: None,
                teid: self.next_downlink_teid(),
                qfi,
            };
            mbs_ctx.activate(Some(tnl_info));
        } else {
            mbs_ctx.activate(None);
        }

        // Add session to manager
        if self.mbs_sessions.add_session(mbs_ctx) {
            info!(
                "MBS session {} activated successfully (TMGI={:02x?})",
                session_id, tmgi_bytes
            );
        } else {
            warn!(
                "Failed to activate MBS session {}, already exists",
                session_id
            );
        }
    }

    /// Handles MBS Session Deactivation Request from AMF
    async fn handle_mbs_session_deactivation_request(&mut self, session_id: u32) {
        info!(
            "MBS Session Deactivation Request: session_id={}",
            session_id
        );

        if let Some(mut session) = self.mbs_sessions.remove_session(session_id) {
            session.deactivate();
            info!("MBS session {} deactivated", session_id);
        } else {
            warn!("MBS session {} not found for deactivation", session_id);
        }
    }

    /// Handles Multicast Group Paging from AMF
    async fn handle_multicast_group_paging(&mut self, tmgi_bytes: [u8; 6], area_scope: Vec<u32>) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "Multicast Group Paging: tmgi={:02x?}, area_scope={:?}",
            tmgi_bytes, area_scope
        );

        // Find the MBS session
        if let Some(session) = self.mbs_sessions.get_session_by_tmgi(&tmgi) {
            if session.is_active() {
                info!(
                    "Paging for MBS session {}, {} joined UEs",
                    session.session_id,
                    session.ue_count()
                );
                // In a real implementation, would trigger RRC paging for interested UEs
                // For now, just log the paging request
            } else {
                warn!(
                    "MBS session {} is not active (state: {:?})",
                    session.session_id, session.state
                );
            }
        } else {
            warn!("MBS session with TMGI {:02x?} not found", tmgi_bytes);
        }
    }

    /// Handles MBS UE Join Request from RRC
    async fn handle_mbs_ue_join_request(&mut self, ue_id: i32, tmgi_bytes: [u8; 6]) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "MBS UE Join Request: ue_id={}, tmgi={:02x?}",
            ue_id, tmgi_bytes
        );

        if let Some(session) = self.mbs_sessions.get_session_by_tmgi_mut(&tmgi) {
            if session.add_ue(ue_id) {
                info!(
                    "UE {} joined MBS session {}, total UEs: {}",
                    ue_id,
                    session.session_id,
                    session.ue_count()
                );

                // Send MulticastSessionUpdateRequest to AMF to inform of UE join
                // In a real implementation, would encode and send NGAP message
                debug!(
                    "Would send MulticastSessionUpdateRequest to AMF for session {} (UE {} joined)",
                    session.session_id, ue_id
                );
            } else {
                debug!("UE {} already in MBS session {}", ue_id, session.session_id);
            }
        } else {
            warn!(
                "Cannot join MBS session: TMGI {:02x?} not found",
                tmgi_bytes
            );
        }
    }

    /// Handles MBS UE Leave Request from RRC
    async fn handle_mbs_ue_leave_request(&mut self, ue_id: i32, tmgi_bytes: [u8; 6]) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "MBS UE Leave Request: ue_id={}, tmgi={:02x?}",
            ue_id, tmgi_bytes
        );

        if let Some(session) = self.mbs_sessions.get_session_by_tmgi_mut(&tmgi) {
            if session.remove_ue(ue_id) {
                info!(
                    "UE {} left MBS session {}, remaining UEs: {}",
                    ue_id,
                    session.session_id,
                    session.ue_count()
                );

                // Send MulticastSessionUpdateRequest to AMF to inform of UE leave
                // In a real implementation, would encode and send NGAP message
                debug!(
                    "Would send MulticastSessionUpdateRequest to AMF for session {} (UE {} left)",
                    session.session_id, ue_id
                );
            } else {
                debug!("UE {} not in MBS session {}", ue_id, session.session_id);
            }
        } else {
            warn!(
                "Cannot leave MBS session: TMGI {:02x?} not found",
                tmgi_bytes
            );
        }
    }
}

/// Builds the Error Indication parameters for an NGAP PDU that the gNB could
/// not route (TS 38.413 §8.7.5 / §10).
///
/// * undecodable PDU → transfer-syntax-error (no CriticalityDiagnostics);
/// * decoded but unsupported procedure → abstract-syntax-error
///   (ignore-and-notify) with CriticalityDiagnostics carrying the procedure
///   code and triggering-message kind.
fn unroutable_error_params(pdu_bytes: &[u8]) -> ErrorIndicationParams {
    match decode_ngap_pdu(pdu_bytes) {
        Ok(pdu) => {
            let (proc_code, trigger) = ngap_procedure_code(&pdu);
            let mut p = error_indication_abstract_syntax_error(None, None, false);
            p.criticality_diagnostics = Some(CriticalityDiagnosticsInfo {
                procedure_code: Some(proc_code),
                triggering_message: Some(trigger),
                procedure_criticality: None,
                ies_criticality_diagnostics: Vec::new(),
            });
            p
        }
        Err(_) => error_indication_transfer_syntax_error(None, None),
    }
}

/// Serializes a 5G-S-TMSI UE paging identity into the canonical 48-bit form
/// for an RRC Paging record (TS 23.003 §2.10.1): AMF Set ID (10 bits) +
/// AMF Pointer (6 bits) packed into 2 octets, followed by the 32-bit 5G-TMSI.
fn serialize_five_g_s_tmsi(identity: &UePagingIdentityValue) -> Vec<u8> {
    match identity {
        UePagingIdentityValue::FiveGSTmsi(tmsi) => {
            let mut out = Vec::with_capacity(6);
            // AMF Set ID is 10 bits, AMF Pointer is 6 bits -> 16 bits.
            let packed = ((tmsi.amf_set_id & 0x3FF) << 6) | (tmsi.amf_pointer as u16 & 0x3F);
            out.extend_from_slice(&packed.to_be_bytes());
            out.extend_from_slice(&tmsi.five_g_tmsi);
            out
        }
    }
}

/// Extracts the NGAP procedure code and triggering-message kind from a decoded
/// PDU, for populating CriticalityDiagnostics in an Error Indication
/// (TS 38.413 §9.3.1.3).
fn ngap_procedure_code(pdu: &NGAP_PDU) -> (u8, TriggeringMessageValue) {
    match pdu {
        NGAP_PDU::InitiatingMessage(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::InitiatingMessage,
        ),
        NGAP_PDU::SuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::SuccessfulOutcome,
        ),
        NGAP_PDU::UnsuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::UnsuccessfulOutcome,
        ),
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
                Some(TaskMessage::Message(msg)) => match msg {
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
                    NgapMessage::NtnTimingInfoReceived {
                        satellite_type,
                        satellite_id,
                        propagation_delay_us,
                        common_ta_us,
                        k_offset,
                    } => {
                        info!(
                                "NTN timing info: sat_type={}, sat_id={}, delay={}us, TA={}us, k_offset={}",
                                satellite_type, satellite_id, propagation_delay_us, common_ta_us, k_offset
                            );
                    }
                    NgapMessage::MbsSessionActivationRequest {
                        session_id,
                        tmgi,
                        is_broadcast,
                        multicast_ip,
                        qfi,
                    } => {
                        self.handle_mbs_session_activation_request(
                            session_id,
                            tmgi,
                            is_broadcast,
                            multicast_ip,
                            qfi,
                        )
                        .await;
                    }
                    NgapMessage::MbsSessionDeactivationRequest { session_id } => {
                        self.handle_mbs_session_deactivation_request(session_id)
                            .await;
                    }
                    NgapMessage::MulticastGroupPaging { tmgi, area_scope } => {
                        self.handle_multicast_group_paging(tmgi, area_scope).await;
                    }
                    NgapMessage::MbsUeJoinRequest { ue_id, tmgi } => {
                        self.handle_mbs_ue_join_request(ue_id, tmgi).await;
                    }
                    NgapMessage::MbsUeLeaveRequest { ue_id, tmgi } => {
                        self.handle_mbs_ue_leave_request(ue_id, tmgi).await;
                    }
                },
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
    use nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue;

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
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_select_as_algorithms_prefers_aes_and_derives_distinct_keys() {
        use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
        use nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue;

        // Caps: NEA1|NEA2 (0xC000), NIA1|NIA2 (0xC000) — MSB (0x8000) = 128-xEA1.
        let caps = UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: 0xC000,
            nr_integrity_algorithms: 0xC000,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        };
        let ((ciph_id, ciph), (int_id, integ)) = NgapTask::select_as_algorithms(&caps);
        // gNB prefers AES (NEA2/NIA2).
        assert_eq!(ciph_id, 2);
        assert_eq!(int_id, 2);
        assert_eq!(ciph, CipheringAlgorithmType::Nea2);
        assert_eq!(integ, IntegrityAlgorithmType::Nia2);

        // Empty bitmaps: no encryption -> NEA0, but integrity never falls back
        // to NIA0 (TS 33.501 §5.11.2) — defaults to mandatory-to-support NIA2.
        let caps0 = UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: 0,
            nr_integrity_algorithms: 0,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        };
        let ((c0, _), (i0, _)) = NgapTask::select_as_algorithms(&caps0);
        assert_eq!(c0, 0);
        assert_eq!(i0, 2);

        // AS keys derived from KgNB are distinct per algorithm-type distinguisher
        // and non-zero (TS 33.501 Annex A.8).
        let kgnb = [0x11u8; 32];
        let k_rrc_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcEnc, 2);
        let k_rrc_int = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcInt, 2);
        let k_up_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpEnc, 2);
        assert_ne!(k_rrc_enc, [0u8; 16]);
        assert_ne!(k_rrc_enc, k_rrc_int);
        assert_ne!(k_rrc_enc, k_up_enc);
    }

    #[test]
    fn test_gnb_drb_reconfiguration_is_valid_and_carries_qfis() {
        use nextgsim_rrc::codec::decode_rrc;
        use nextgsim_rrc::codec::generated::DL_DCCH_Message;
        use nextgsim_rrc::procedures::rrc_reconfiguration::is_rrc_reconfiguration;

        // The exact call establish_drb makes: PDU session 5, DRB 5, LCID 8,
        // accepted QFIs 1 & 9. The result must be a decodable DL-DCCH
        // RRCReconfiguration.
        let params = build_drb_reconfiguration_params(0, 5, 5, 8, &[1, 9], true).unwrap();
        let bytes = encode_rrc_reconfiguration(&params).unwrap();
        assert!(!bytes.is_empty());
        let msg: DL_DCCH_Message = decode_rrc(&bytes).unwrap();
        assert!(is_rrc_reconfiguration(&msg));
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

    // ------------------------------------------------------------------
    // amfg-06: inbound Paging
    // ------------------------------------------------------------------

    use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi as PagingTmsi, Tai};
    use nextgsim_ngap::procedures::paging::PagingData;

    fn sample_tmsi() -> PagingTmsi {
        PagingTmsi {
            amf_set_id: 0x155,
            amf_pointer: 0x2A,
            five_g_tmsi: [0xDE, 0xAD, 0xBE, 0xEF],
        }
    }

    #[test]
    fn test_serialize_five_g_s_tmsi_packing() {
        let id = UePagingIdentityValue::FiveGSTmsi(sample_tmsi());
        let bytes = serialize_five_g_s_tmsi(&id);
        assert_eq!(bytes.len(), 6);
        // packed = (0x155 << 6) | 0x2A = 0x556A
        assert_eq!(&bytes[0..2], &[0x55, 0x6A]);
        assert_eq!(&bytes[2..6], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn test_paging_matching_tai_triggers_rrc() {
        let config = test_config();
        let served_plmn = config.plmn.encode();
        let served_tac = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        let paging = PagingData {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(sample_tmsi()),
            tai_list_for_paging: vec![Tai {
                plmn_identity: served_plmn,
                tac: served_tac,
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
        };
        task.handle_paging(1, paging).await;

        match rrc_rx.try_recv() {
            Ok(TaskMessage::Message(RrcMessage::Paging {
                ue_paging_tmsi,
                tai_list_for_paging,
            })) => {
                assert_eq!(ue_paging_tmsi.len(), 6);
                assert_eq!(tai_list_for_paging.len(), 6); // one matching TAI
                assert_eq!(&tai_list_for_paging[0..3], &served_plmn);
                assert_eq!(&tai_list_for_paging[3..6], &served_tac);
            }
            other => panic!("expected RRC Paging, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_paging_non_matching_tai_is_ignored() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        let paging = PagingData {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(sample_tmsi()),
            // A TAI this gNB does not serve.
            tai_list_for_paging: vec![Tai {
                plmn_identity: [0x99, 0x99, 0x99],
                tac: [0x12, 0x34, 0x56],
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
        };
        task.handle_paging(1, paging).await;

        assert!(
            rrc_rx.try_recv().is_err(),
            "no RRC Paging should be emitted for an unserved TAI"
        );
    }

    // ------------------------------------------------------------------
    // amfg-07: NG Reset handler clears exactly the listed UE contexts
    // ------------------------------------------------------------------

    use nextgsim_ngap::procedures::ng_reset::NgResetData;

    #[tokio::test]
    async fn test_ng_reset_partial_clears_listed_contexts() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }

        // Two UE contexts on AMF[1]; give them distinct RAN UE NGAP IDs.
        let ran1 = task.create_ue_context(10, 1).expect("ue1");
        let ran2 = task.create_ue_context(20, 1).expect("ue2");
        assert_ne!(ran1, ran2);

        // Reset only UE[10] by its RAN UE NGAP ID.
        let reset = NgResetData {
            cause: None,
            scope: NgResetScope::Part(vec![UeAssociation {
                amf_ue_ngap_id: None,
                ran_ue_ngap_id: Some(ran1 as u32),
            }]),
        };
        task.handle_ng_reset(1, 0, reset).await;

        assert!(
            task.find_ue_context(10).is_none(),
            "UE[10] should be cleared"
        );
        assert!(task.find_ue_context(20).is_some(), "UE[20] must remain");
    }

    #[tokio::test]
    async fn test_ng_reset_all_clears_all_contexts_for_amf() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(10, 1).expect("ue1");
        task.create_ue_context(20, 1).expect("ue2");

        let reset = NgResetData {
            cause: None,
            scope: NgResetScope::All,
        };
        task.handle_ng_reset(1, 0, reset).await;

        assert!(task.find_ue_context(10).is_none());
        assert!(task.find_ue_context(20).is_none());
    }

    // ------------------------------------------------------------------
    // amfg-05: Error Indication generation for unroutable PDUs
    // ------------------------------------------------------------------

    use nextgsim_ngap::codec::ID_ERROR_INDICATION;
    use nextgsim_ngap::procedures::error_indication::decode_error_indication;
    use nextgsim_ngap::procedures::ng_setup::{NgSetupFailureCause, ProtocolCause};

    /// A garbage byte stream that does not APER-decode as an NGAP PDU must
    /// yield a transfer-syntax-error Error Indication with no
    /// CriticalityDiagnostics (the procedure code is unknown).
    #[test]
    fn test_unroutable_undecodable_yields_transfer_syntax_error() {
        let garbage = [0xFFu8, 0xFE, 0xAB, 0xCD, 0x12, 0x34];
        let params = unroutable_error_params(&garbage);
        assert!(matches!(
            params.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::TransferSyntaxError
            ))
        ));
        assert!(params.criticality_diagnostics.is_none());

        // Round-trip: the produced Error Indication encodes and decodes.
        let bytes = encode_error_indication(&params).expect("encode");
        let decoded = decode_error_indication(&bytes).expect("decode");
        assert!(matches!(
            decoded.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::TransferSyntaxError
            ))
        ));
    }

    /// A well-formed-but-unhandled NGAP PDU must yield an
    /// abstract-syntax-error (ignore-and-notify) Error Indication whose
    /// CriticalityDiagnostics carries the offending procedure code.
    #[test]
    fn test_unroutable_decodable_yields_abstract_syntax_error_with_proc_code() {
        // Use an Error Indication PDU (procedure code 9) as a stand-in for a
        // valid-but-unhandled inbound procedure: it decodes cleanly yet has no
        // gNB inbound handler.
        let valid_pdu_bytes =
            encode_error_indication(&ErrorIndicationParams::default()).expect("encode sample pdu");

        let params = unroutable_error_params(&valid_pdu_bytes);
        assert!(matches!(
            params.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify
            ))
        ));
        let diag = params
            .criticality_diagnostics
            .as_ref()
            .expect("criticality diagnostics present");
        assert_eq!(diag.procedure_code, Some(ID_ERROR_INDICATION));
        assert_eq!(
            diag.triggering_message,
            Some(TriggeringMessageValue::InitiatingMessage)
        );

        // Strict APER round-trip of the generated Error Indication.
        let bytes = encode_error_indication(&params).expect("encode");
        let decoded = decode_error_indication(&bytes).expect("decode");
        assert_eq!(
            decoded
                .criticality_diagnostics
                .and_then(|d| d.procedure_code),
            Some(ID_ERROR_INDICATION)
        );
    }

    /// ngap_procedure_code reports the correct triggering-message kind.
    #[test]
    fn test_ngap_procedure_code_initiating() {
        let bytes = encode_error_indication(&ErrorIndicationParams::default()).expect("encode");
        let pdu = decode_ngap_pdu(&bytes).expect("decode");
        let (code, trigger) = ngap_procedure_code(&pdu);
        assert_eq!(code, ID_ERROR_INDICATION);
        assert_eq!(trigger, TriggeringMessageValue::InitiatingMessage);
    }

    // ---- UE Context Modification (issue #40) --------------------------------

    /// Procedure code for UE Context Modification (TS 38.413).
    const ID_UE_CONTEXT_MODIFICATION: u8 = 40;

    /// Seed an NGAP task with a Ready AMF and one UE context; returns the task
    /// (with `sctp_rx` bound to observe replies) and the UE's RAN-UE-NGAP-ID.
    fn seed_task_with_ue() -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
        i64,
    ) {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        let ran = task.create_ue_context(10, 1).expect("ue context");
        (task, sctp_rx, ran)
    }

    fn caps(nea: u16, nia: u16) -> UeSecurityCapabilitiesValue {
        UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: nea,
            nr_integrity_algorithms: nia,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        }
    }

    fn encode_ucm_request(data: &UeContextModificationRequestData) -> Vec<u8> {
        nextgsim_ngap::procedures::ue_context_modification::encode_ue_context_modification_request(
            data,
        )
        .expect("encode inbound request")
    }

    /// Decode a UE Context Modification RESPONSE and return the echoed
    /// (AMF-UE-NGAP-ID, RAN-UE-NGAP-ID).
    fn response_ids(bytes: &[u8]) -> (u64, u32) {
        use nextgsim_ngap::codec::generated::{
            SuccessfulOutcomeValue, UEContextModificationResponseProtocolIEs_EntryValue as RespIe,
        };
        match decode_ngap_pdu(bytes).expect("decode response") {
            NGAP_PDU::SuccessfulOutcome(o) => match o.value {
                SuccessfulOutcomeValue::Id_UEContextModification(r) => {
                    let mut amf = None;
                    let mut ran = None;
                    for ie in &r.protocol_i_es.0 {
                        match &ie.value {
                            RespIe::Id_AMF_UE_NGAP_ID(id) => amf = Some(id.0),
                            RespIe::Id_RAN_UE_NGAP_ID(id) => ran = Some(id.0),
                            _ => {}
                        }
                    }
                    (amf.expect("amf id"), ran.expect("ran id"))
                }
                other => panic!("expected UEContextModification response, got {other:?}"),
            },
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    /// Decode a UE Context Modification FAILURE and return its Cause.
    fn failure_cause(bytes: &[u8]) -> NgSetupFailureCause {
        use nextgsim_ngap::codec::generated::{
            UEContextModificationFailureProtocolIEs_EntryValue as FailIe, UnsuccessfulOutcomeValue,
        };
        use nextgsim_ngap::procedures::ue_context_release::parse_cause;
        match decode_ngap_pdu(bytes).expect("decode failure") {
            NGAP_PDU::UnsuccessfulOutcome(o) => match o.value {
                UnsuccessfulOutcomeValue::Id_UEContextModification(f) => {
                    for ie in &f.protocol_i_es.0 {
                        if let FailIe::Id_Cause(c) = &ie.value {
                            return parse_cause(c);
                        }
                    }
                    panic!("no Cause IE in failure");
                }
                other => panic!("expected UEContextModification failure, got {other:?}"),
            },
            other => panic!("expected UnsuccessfulOutcome, got {other:?}"),
        }
    }

    /// A UE CONTEXT MODIFICATION REQUEST carrying a new Security Key re-derives
    /// the AS keys and the gNB replies with a RESPONSE (not an Error Indication).
    #[tokio::test]
    async fn test_ue_context_modification_rekeys_and_responds() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();

        // Establish a baseline AS security context.
        task.activate_as_security(10, [0x11u8; 32], &caps(0xC000, 0xC000))
            .await;
        let old = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();

        // Inbound request with a *new* KgNB and updated capabilities.
        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: Some(caps(0xC000, 0xC000)),
            security_key: Some([0x22u8; 32]),
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        // (1) The reply is a UE CONTEXT MODIFICATION RESPONSE, not an Error
        // Indication, and it echoes the request's AMF/RAN IDs.
        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_ne!(stream, 0, "UE-associated reply must not use stream 0");
                let bytes = buffer.data();
                match decode_ngap_pdu(bytes).expect("decode reply") {
                    NGAP_PDU::SuccessfulOutcome(o) => {
                        assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
                    }
                    other => panic!("expected UE Ctx Mod Response, got {other:?}"),
                }
                assert_eq!(response_ids(bytes), (555, ran as u32));
            }
            other => panic!("expected an outbound SCTP SendMessage, got {other:?}"),
        }

        // (2) The AS keys were refreshed from the new KgNB.
        let new = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();
        assert_eq!(new.kgnb, [0x22u8; 32]);
        assert_ne!(new.k_rrc_enc, old.k_rrc_enc, "AS keys must be re-derived");
    }

    /// A UE CONTEXT MODIFICATION REQUEST without a Security Key applies the
    /// replaced UE-AMBR, replies with a RESPONSE, and does not touch AS keys.
    #[tokio::test]
    async fn test_ue_context_modification_updates_ambr_without_rekey() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        task.activate_as_security(10, [0x11u8; 32], &caps(0xC000, 0xC000))
            .await;
        let old = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: Some(
                nextgsim_ngap::procedures::initial_context_setup::UeAggregateMaxBitRate {
                    dl: 300_000_000,
                    ul: 100_000_000,
                },
            ),
            ue_security_capabilities: None,
            security_key: None,
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let pdu = decode_ngap_pdu(buffer.data()).expect("decode reply");
                assert!(matches!(pdu, NGAP_PDU::SuccessfulOutcome(_)));
            }
            other => panic!("expected a Response, got {other:?}"),
        }

        let ctx = task.find_ue_context(10).unwrap();
        let ambr = ctx.ue_ambr.clone().expect("UE-AMBR applied");
        assert_eq!(ambr.dl, 300_000_000);
        assert_eq!(ambr.ul, 100_000_000);
        // No re-key requested → AS keys unchanged.
        let new = ctx.as_security.clone().unwrap();
        assert_eq!(new.kgnb, old.kgnb);
        assert_eq!(new.k_rrc_enc, old.k_rrc_enc);
    }

    /// A UE CONTEXT MODIFICATION REQUEST for an unknown UE is answered with a
    /// UE CONTEXT MODIFICATION FAILURE — never an Error Indication.
    #[tokio::test]
    async fn test_ue_context_modification_unknown_ue_replies_failure() {
        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: 9999, // no UE has this RAN-UE-NGAP-ID
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: None,
            security_key: Some([0x22u8; 32]),
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let bytes = buffer.data();
                match decode_ngap_pdu(bytes).expect("decode reply") {
                    NGAP_PDU::UnsuccessfulOutcome(o) => {
                        assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
                    }
                    // An Error Indication would decode as an InitiatingMessage
                    // (procedure code 9) — explicitly rejected here.
                    other => panic!("expected UE Ctx Mod Failure, got {other:?}"),
                }
                assert_eq!(
                    failure_cause(bytes),
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownLocalUeNgapId)
                );
            }
            other => panic!("expected a Failure, got {other:?}"),
        }
    }

    /// A Security Key with no resolvable UE Security Capabilities is answered
    /// with a FAILURE (Protocol/SemanticError), and — crucially — the procedure
    /// is atomic: a co-carried UE-AMBR is NOT applied and AS keys are untouched.
    #[tokio::test]
    async fn test_ue_context_modification_no_caps_rekey_fails_atomically() {
        // A freshly-created UE context has no stored UE Security Capabilities
        // (they are set at Initial Context Setup, which we deliberately skip).
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        assert!(task
            .find_ue_context(10)
            .unwrap()
            .ue_security_capabilities
            .is_none());

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            // Co-carried UE-AMBR must NOT be committed when the procedure fails.
            ue_aggregate_max_bit_rate: Some(
                nextgsim_ngap::procedures::initial_context_setup::UeAggregateMaxBitRate {
                    dl: 1,
                    ul: 1,
                },
            ),
            ue_security_capabilities: None,
            security_key: Some([0x22u8; 32]),
        };
        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&encode_ucm_request(&data)))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let bytes = buffer.data();
                assert!(matches!(
                    decode_ngap_pdu(bytes).expect("decode reply"),
                    NGAP_PDU::UnsuccessfulOutcome(_)
                ));
                assert_eq!(
                    failure_cause(bytes),
                    NgSetupFailureCause::Protocol(ProtocolCause::SemanticError)
                );
            }
            other => panic!("expected a Failure, got {other:?}"),
        }

        // Atomicity: nothing was applied on the failed procedure.
        let ctx = task.find_ue_context(10).unwrap();
        assert!(
            ctx.ue_ambr.is_none(),
            "UE-AMBR must not be applied on failure"
        );
        assert!(
            ctx.as_security.is_none(),
            "AS keys must be untouched on failure"
        );
    }

    /// Updated UE Security Capabilities carried by the request are stored on the
    /// UE context (TS 38.413 §8.3.4.2).
    #[tokio::test]
    async fn test_ue_context_modification_stores_updated_caps() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: Some(caps(0xA000, 0x8000)),
            security_key: None,
        };
        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&encode_ucm_request(&data)))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                assert!(matches!(
                    decode_ngap_pdu(buffer.data()).expect("decode reply"),
                    NGAP_PDU::SuccessfulOutcome(_)
                ));
            }
            other => panic!("expected a Response, got {other:?}"),
        }

        let stored = task
            .find_ue_context(10)
            .unwrap()
            .ue_security_capabilities
            .clone()
            .expect("UE Security Capabilities applied");
        // Assert the NR fields specifically (parse always fills the E-UTRA
        // fields, so full-struct equality would be brittle).
        assert_eq!(stored.nr_encryption_algorithms, 0xA000);
        assert_eq!(stored.nr_integrity_algorithms, 0x8000);
    }

    // ---- Error Indication + NAS Non Delivery (issue #42) --------------------

    /// A received ERROR INDICATION referencing a known UE locally releases that
    /// UE's NG connection and does NOT echo another Error Indication back to the
    /// AMF (TS 38.413 §8.7.5 / §10.6).
    #[tokio::test]
    async fn test_error_indication_releases_ue_and_does_not_echo() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        assert!(task.find_ue_context(10).is_some());

        let params = ErrorIndicationParams {
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: Some(ran as u32),
            cause: Some(NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::UnknownLocalUeNgapId,
            )),
            criticality_diagnostics: None,
        };
        let inbound = encode_error_indication(&params).expect("encode error indication");

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        // The UE context was locally released ...
        assert!(
            task.find_ue_context(10).is_none(),
            "UE context must be locally released on Error Indication"
        );
        // ... and NO Error Indication (or any NGAP PDU) was echoed to the AMF.
        assert!(
            sctp_rx.try_recv().is_err(),
            "must not reply to a received Error Indication (ping-pong)"
        );
    }

    /// A received ERROR INDICATION whose AP IDs match no UE is processed with no
    /// local release and, crucially, no reply.
    #[tokio::test]
    async fn test_error_indication_unknown_ue_is_silently_processed() {
        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();

        let params = ErrorIndicationParams {
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: Some(4242), // matches no UE
            cause: Some(NgSetupFailureCause::Protocol(ProtocolCause::SemanticError)),
            criticality_diagnostics: None,
        };
        let inbound = encode_error_indication(&params).expect("encode");

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        assert!(task.find_ue_context(10).is_some(), "unrelated UE untouched");
        assert!(sctp_rx.try_recv().is_err(), "no reply to Error Indication");
    }

    /// When RRC cannot accept the NAS-PDU for delivery to the UE (send failure),
    /// the gNB still reports a NAS NON DELIVERY INDICATION (TS 38.413 §8.6.4).
    #[tokio::test]
    async fn test_downlink_nas_rrc_failure_emits_non_delivery() {
        // `seed_task_with_ue` drops the RRC receiver, so `rrc_tx.send` fails —
        // exercising the RRC-send-failure non-delivery path for a *known* UE.
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        let nas_pdu = vec![0x7e, 0x00, 0x55];
        let dl_nas = DownlinkNasTransportData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32, // matches the seeded UE
            nas_pdu: nas_pdu.clone(),
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            allowed_nssai: None,
        };

        task.handle_downlink_nas_transport(1, 3, dl_nas).await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                assert_eq!(nas_non_delivery_pdu(buffer.data()), nas_pdu);
            }
            other => panic!("expected NAS Non Delivery on RRC failure, got {other:?}"),
        }
    }

    /// A DOWNLINK NAS TRANSPORT for an unknown UE is reported to the AMF as a NAS
    /// NON DELIVERY INDICATION carrying the original NAS-PDU (TS 38.413 §8.6.4).
    #[tokio::test]
    async fn test_downlink_nas_unknown_ue_emits_non_delivery() {
        use nextgsim_ngap::procedures::nas_non_delivery_indication::is_nas_non_delivery_indication;

        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();
        let nas_pdu = vec![0x7e, 0x00, 0x42, 0x11, 0x22];
        let dl_nas = DownlinkNasTransportData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: 4242, // no UE has this RAN-UE-NGAP-ID
            nas_pdu: nas_pdu.clone(),
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            allowed_nssai: None,
        };

        task.handle_downlink_nas_transport(1, 3, dl_nas).await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_ne!(
                    stream, 0,
                    "UE-associated non-delivery must not use stream 0"
                );
                let pdu = decode_ngap_pdu(buffer.data()).expect("decode reply");
                assert!(
                    is_nas_non_delivery_indication(&pdu),
                    "expected NAS Non Delivery Indication, got {pdu:?}"
                );
                // The original NAS-PDU is echoed back to the AMF.
                let echoed = nas_non_delivery_pdu(buffer.data());
                assert_eq!(echoed, nas_pdu);
            }
            other => panic!("expected NAS Non Delivery Indication, got {other:?}"),
        }
    }

    /// Extract the NAS-PDU IE from a NAS Non Delivery Indication's bytes.
    fn nas_non_delivery_pdu(bytes: &[u8]) -> Vec<u8> {
        use nextgsim_ngap::codec::generated::{
            InitiatingMessageValue, NASNonDeliveryIndicationProtocolIEs_EntryValue as Ie,
        };
        match decode_ngap_pdu(bytes).expect("decode") {
            NGAP_PDU::InitiatingMessage(m) => match m.value {
                InitiatingMessageValue::Id_NASNonDeliveryIndication(x) => x
                    .protocol_i_es
                    .0
                    .iter()
                    .find_map(|ie| match &ie.value {
                        Ie::Id_NAS_PDU(p) => Some(p.0.clone()),
                        _ => None,
                    })
                    .expect("NAS-PDU IE present"),
                other => panic!("wrong variant: {other:?}"),
            },
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    // ---- Overload Start / Stop (issue #41) ----------------------------------

    /// OVERLOAD START drives the AMF context into `Overloaded` (which excludes it
    /// from `select_amf`) and records the traffic-load-reduction; OVERLOAD STOP
    /// clears it. Neither is echoed with an Error Indication (TS 38.413 §8.7.7/8).
    #[tokio::test]
    async fn test_overload_start_stop_marks_amf_without_echo() {
        use nextgsim_ngap::procedures::overload::{encode_overload_start, encode_overload_stop};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        // OVERLOAD START → Overloaded + reduction stored, no reply.
        let start = encode_overload_start(Some(75)).expect("encode start");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&start))
            .await;
        {
            let ctx = task.find_amf_context_mut(1).unwrap();
            assert_eq!(ctx.state, AmfState::Overloaded);
            assert_eq!(ctx.traffic_load_reduction, Some(75));
        }
        assert!(
            sctp_rx.try_recv().is_err(),
            "Overload Start must not be answered with an Error Indication"
        );

        // OVERLOAD STOP → Ready, reduction cleared, no reply.
        let stop = encode_overload_stop().expect("encode stop");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&stop))
            .await;
        {
            let ctx = task.find_amf_context_mut(1).unwrap();
            assert_eq!(ctx.state, AmfState::Ready);
            assert_eq!(ctx.traffic_load_reduction, None);
        }
        assert!(
            sctp_rx.try_recv().is_err(),
            "Overload Stop must not be answered with an Error Indication"
        );
    }

    /// AMF STATUS INDICATION marking a served GUAMI unavailable excludes that AMF
    /// from `select_amf`, with no Error Indication echo (TS 38.413 §8.7.6).
    #[tokio::test]
    async fn test_amf_status_indication_excludes_amf_from_selection() {
        use nextgsim_ngap::procedures::amf_status_indication::{
            encode_amf_status_indication, AmfStatusIndicationParams, UnavailableGuamiItem,
        };
        use nextgsim_ngap::procedures::initial_context_setup::GuamiValue;
        use nextgsim_ngap::procedures::{Guami, ServedGuamiItem};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 100;
            ctx.served_guami_list = vec![ServedGuamiItem {
                guami: Guami {
                    plmn_identity: [0x00, 0xf1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 3,
                },
                backup_amf_name: None,
            }];
        }
        // Ready + available → this AMF is selectable.
        assert_eq!(task.select_amf(None), Some(1));

        // AMF Status Indication marks that GUAMI unavailable.
        let params = AmfStatusIndicationParams {
            unavailable_guami_list: vec![UnavailableGuamiItem {
                guami: GuamiValue {
                    plmn_identity: [0x00, 0xf1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 3,
                },
                backup_amf_name: Some("backup-amf".into()),
            }],
        };
        let inbound = encode_amf_status_indication(&params).expect("encode");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&inbound))
            .await;

        // The AMF is excluded from selection, and nothing was echoed.
        assert!(task.find_amf_context_mut(1).unwrap().unavailable);
        assert_eq!(
            task.select_amf(None),
            None,
            "an AMF with an unavailable GUAMI must not be selected"
        );
        assert!(
            sctp_rx.try_recv().is_err(),
            "AMF Status Indication must not be answered with an Error Indication"
        );
    }
}
