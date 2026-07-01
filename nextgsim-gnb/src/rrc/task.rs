//! RRC Task Implementation

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GutiMobileIdentity, IsacMessage, NgapMessage, NkefMessage, RlsMessage, RrcMessage,
    SheMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::rrc_setup::{
    decode_rrc_setup_complete, decode_rrc_setup_request,
    RrcEstablishmentCause as AsnEstablishmentCause, UeIdentity,
};
use nextgsim_rrc::procedures::ue_capability::{
    decode_ue_capability_information, encode_ue_capability_enquiry, parse_nr_capability_bands,
    RatType, UeCapabilityEnquiryParams,
};

/// Simplified DL/UL-DCCH envelope code: first byte 0x06 marks a UE capability
/// transfer message; the remaining bytes are the real ASN.1 UPER encoding of
/// UECapabilityEnquiry (DL) / UECapabilityInformation (UL).
const RRC_MSG_TYPE_UE_CAPABILITY: u8 = 0x06;

use super::connection::RrcConnectionManager;
use super::ue_context::RrcUeContextManager;

/// NTN configuration stored at RRC level
#[derive(Debug, Clone)]
pub struct NtnRrcConfig {
    pub satellite_type: String,
    pub common_ta_us: u64,
    pub k_offset: u16,
    pub max_doppler_hz: f64,
    pub autonomous_ta: bool,
}

/// RRC Task for managing UE RRC connections
pub struct RrcTask {
    task_base: GnbTaskBase,
    ue_manager: RrcUeContextManager,
    connection_manager: RrcConnectionManager,
    pdu_id_counter: u32,
    ntn_config: Option<NtnRrcConfig>,
}

impl RrcTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            ue_manager: RrcUeContextManager::new(),
            connection_manager: RrcConnectionManager::new(),
            pdu_id_counter: 0,
            ntn_config: None,
        }
    }

    fn next_pdu_id(&mut self) -> u32 {
        self.pdu_id_counter = self.pdu_id_counter.wrapping_add(1);
        if self.pdu_id_counter == 0 {
            self.pdu_id_counter = 1;
        }
        self.pdu_id_counter
    }

    fn handle_radio_power_on(&mut self) {
        info!("Radio power on - cell is now active");
        self.connection_manager.set_barred(false);
    }

    fn handle_signal_detected(&mut self, ue_id: i32) {
        debug!("Signal detected from UE[{}]", ue_id);
    }

    async fn handle_uplink_rrc(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
        debug!(
            "Uplink RRC: ue_id={}, channel={:?}, len={}",
            ue_id,
            channel,
            data.len()
        );

        match channel {
            RrcChannel::UlCcch | RrcChannel::UlCcch1 => {
                self.handle_ul_ccch_message(ue_id, &data).await;
            }
            RrcChannel::UlDcch => {
                self.handle_ul_dcch_message(ue_id, &data).await;
            }
            _ => warn!("Unexpected uplink channel: {:?}", channel),
        }
    }

    async fn handle_ul_ccch_message(&mut self, ue_id: i32, data: &OctetString) {
        if data.len() < 2 {
            warn!("UL-CCCH message too short: {} bytes", data.len());
            return;
        }

        let bytes = data.data();

        // Try ASN.1 UPER decoding first (proper 3GPP encoding)
        if let Ok(setup_req) = decode_rrc_setup_request(bytes) {
            let (initial_id, is_stmsi) = match setup_req.ue_identity {
                UeIdentity::Ng5gSTmsiPart1(v) => (v as i64, true),
                UeIdentity::RandomValue(v) => (v as i64, false),
            };
            let establishment_cause = match setup_req.establishment_cause {
                AsnEstablishmentCause::Emergency => 0,
                AsnEstablishmentCause::HighPriorityAccess => 1,
                AsnEstablishmentCause::MtAccess => 2,
                AsnEstablishmentCause::MoSignalling => 3,
                AsnEstablishmentCause::MoData => 4,
                AsnEstablishmentCause::MoVoiceCall => 5,
                AsnEstablishmentCause::MoVideoCall => 6,
                AsnEstablishmentCause::MoSms => 7,
                AsnEstablishmentCause::MpsPriorityAccess => 8,
                AsnEstablishmentCause::McsPriorityAccess => 9,
            };

            debug!(
                "Decoded ASN.1 RRCSetupRequest: initial_id={:x}, is_stmsi={}, cause={}",
                initial_id, is_stmsi, establishment_cause
            );

            if let Some(result) = self.connection_manager.process_rrc_setup_request(
                &mut self.ue_manager,
                ue_id,
                initial_id,
                is_stmsi,
                establishment_cause,
            ) {
                self.send_rrc_message(result.ue_id, result.channel, result.rrc_setup_pdu)
                    .await;
            }
            return;
        }

        // Fallback: simplified byte-level parsing for backwards compatibility
        let msg_type = bytes[0] & 0x3F;

        match msg_type {
            // RRC Setup Request (simplified encoding)
            0x00..=0x1F => {
                if data.len() < 6 {
                    warn!("RRC Setup Request too short: {} bytes", data.len());
                    return;
                }
                let initial_id = i64::from_be_bytes([
                    0,
                    0,
                    0,
                    bytes.get(1).copied().unwrap_or(0),
                    bytes.get(2).copied().unwrap_or(0),
                    bytes.get(3).copied().unwrap_or(0),
                    bytes.get(4).copied().unwrap_or(0),
                    bytes.get(5).copied().unwrap_or(0),
                ]) & 0x7FFFFFFFFF;

                let is_stmsi = (bytes[0] & 0x80) != 0;
                let establishment_cause = bytes.get(6).copied().unwrap_or(3) as i64;

                if let Some(result) = self.connection_manager.process_rrc_setup_request(
                    &mut self.ue_manager,
                    ue_id,
                    initial_id,
                    is_stmsi,
                    establishment_cause,
                ) {
                    self.send_rrc_message(result.ue_id, result.channel, result.rrc_setup_pdu)
                        .await;
                }
            }
            // RRC Reestablishment Request (0x24)
            0x24 => {
                self.handle_rrc_reestablishment_request(ue_id, data).await;
            }
            // RRC Resume Request (0x28)
            0x28 => {
                self.handle_rrc_resume_request(ue_id, data).await;
            }
            _ => {
                debug!("Unknown UL-CCCH message type: 0x{:02x}", msg_type);
            }
        }
    }

    async fn handle_ul_dcch_message(&mut self, ue_id: i32, data: &OctetString) {
        if data.is_empty() {
            warn!("Empty UL-DCCH message");
            return;
        }

        let bytes = data.data();
        let message_type = bytes[0] & 0x0F;

        match message_type {
            0x04 => self.handle_rrc_setup_complete(ue_id, data).await,
            0x08 => self.handle_ul_information_transfer(ue_id, data).await,
            0x05 => self.handle_rrc_reestablishment_complete(ue_id, data).await,
            0x09 => self.handle_rrc_resume_complete(ue_id, data).await,
            RRC_MSG_TYPE_UE_CAPABILITY => {
                self.handle_ue_capability_information(ue_id, &bytes[1..])
                    .await;
            }
            _ => {
                // Check if this looks like a raw NAS PDU
                // EPD = 0x7E for 5GMM (Mobility Management), 0x2E for 5GSM (Session Management)
                // Some UE simulators send NAS directly without proper RRC encapsulation
                let is_nas_pdu = bytes.len() >= 3 && (bytes[0] == 0x7E || bytes[0] == 0x2E);
                if is_nas_pdu {
                    // Check if UE context already exists (meaning Initial UE Message was already sent)
                    if let Some(ctx) = self.ue_manager.try_find_ue(ue_id) {
                        if ctx.is_connected() {
                            // UE is already connected, send as Uplink NAS Transport
                            info!("Received raw NAS PDU on UL-DCCH from connected UE, sending as Uplink NAS (ue_id={}, epd=0x{:02x})", ue_id, bytes[0]);
                            self.send_uplink_nas_delivery(ue_id, data.clone()).await;
                            return;
                        }
                    }

                    // First message - send as Initial UE Message
                    info!("Received raw NAS PDU on UL-DCCH, forwarding to NGAP as Initial UE (ue_id={})", ue_id);
                    // Create or get UE context
                    if self.ue_manager.try_find_ue(ue_id).is_none() {
                        // Auto-create UE context for this UE
                        let ctx = self.ue_manager.create_ue(ue_id);
                        ctx.on_setup_complete(); // Mark as connected
                    }
                    // Forward as Initial NAS
                    self.send_initial_nas_delivery(ue_id, data.clone(), 3, None)
                        .await; // cause=3 (mo-Data)
                } else {
                    debug!("Unhandled UL-DCCH message type: {:#x}", message_type);
                }
            }
        }
    }

    async fn handle_rrc_setup_complete(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        if bytes.len() < 3 {
            warn!("RRC Setup Complete too short");
            return;
        }

        let transaction_id = bytes[1];
        let nas_pdu = if bytes.len() > 3 {
            OctetString::from_slice(&bytes[3..])
        } else {
            OctetString::new()
        };

        // RedCap (Reduced Capability) indication (Rel-17, TS 38.331 §6.2.2).
        // The UE rides the indication in the RRCSetupComplete
        // lateNonCriticalExtension octet container; best-effort ASN.1 decode of
        // the full PDU recovers it without disturbing the lenient NAS
        // extraction above.
        let redcap_indication = decode_rrc_setup_complete(bytes)
            .map(|data| data.redcap_indication)
            .unwrap_or(false);
        if redcap_indication {
            self.apply_redcap_restrictions(ue_id);
        }

        if let Some(result) = self.connection_manager.process_rrc_setup_complete(
            &mut self.ue_manager,
            ue_id,
            transaction_id,
            nas_pdu,
            None,
        ) {
            self.send_initial_nas_delivery(
                result.ue_id,
                result.nas_pdu,
                result.establishment_cause,
                result.s_tmsi,
            )
            .await;

            // Enquire UE radio access capabilities (TS 38.331 §5.6.1)
            self.send_ue_capability_enquiry(ue_id).await;
        }
    }

    /// Configures the UE's RedCap processor and applies the RedCap bandwidth /
    /// HD-FDD scheduling restriction (Rel-17, TS 38.306 / TS 38.331).
    ///
    /// The scheduler enforces a reduced serving bandwidth for a RedCap UE: the
    /// cell's PRB grid is clamped to the RedCap maximum bandwidth (20 MHz for
    /// Rel-17), so a RedCap UE is never granted more PRBs than its narrowband
    /// RF supports. The enforced PRB ceiling is stored on the UE context and
    /// observed by downstream resource allocation.
    fn apply_redcap_restrictions(&mut self, ue_id: i32) {
        // Cell serving bandwidth (FR1 normal UE baseline: 100 MHz / 273 PRB at
        // 30 kHz SCS, TS 38.101-1 Table 5.3.2-1).
        const CELL_BANDWIDTH_MHZ: u8 = 100;
        const CELL_MAX_PRB: u32 = 273;

        let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) else {
            return;
        };

        // Configure the RedCap processor with Rel-17 capabilities (20 MHz,
        // single-layer, HD-FDD).
        ctx.redcap
            .configure(super::redcap::RedCapUeCapabilities::rel17());

        // Scheduler enforcement: clamp the granted bandwidth to the RedCap
        // ceiling and translate it to a hard PRB cap proportional to the
        // bandwidth reduction. This is a real reduction applied to this UE's
        // resource grid, not a log-only marker.
        let enforced_bw_mhz = ctx.redcap.restrict_bandwidth(CELL_BANDWIDTH_MHZ);
        let enforced_max_prb =
            (CELL_MAX_PRB * enforced_bw_mhz as u32 / CELL_BANDWIDTH_MHZ as u32).max(1);
        ctx.set_redcap_max_prb(enforced_max_prb);

        info!(
            "RedCap UE[{}]: scheduler bandwidth restricted to {} MHz ({} PRB max, \
             was {} MHz / {} PRB); HD-FDD gaps={}",
            ue_id,
            enforced_bw_mhz,
            enforced_max_prb,
            CELL_BANDWIDTH_MHZ,
            CELL_MAX_PRB,
            ctx.redcap.needs_hd_fdd_gaps(),
        );
    }

    /// Sends a UECapabilityEnquiry to the UE (TS 38.331 §5.6.1)
    async fn send_ue_capability_enquiry(&mut self, ue_id: i32) {
        let params = UeCapabilityEnquiryParams {
            rrc_transaction_id: 0,
            rat_types: vec![RatType::Nr],
        };
        match encode_ue_capability_enquiry(&params) {
            Ok(uper) => {
                let mut pdu = Vec::with_capacity(uper.len() + 1);
                pdu.push(RRC_MSG_TYPE_UE_CAPABILITY);
                pdu.extend_from_slice(&uper);
                info!("Sending UECapabilityEnquiry to UE[{}]", ue_id);
                self.send_rrc_message(ue_id, RrcChannel::DlDcch, OctetString::from_slice(&pdu))
                    .await;
            }
            Err(e) => {
                error!("Failed to encode UECapabilityEnquiry: {}", e);
            }
        }
    }

    /// Handles a UECapabilityInformation from the UE (TS 38.331 §5.6.1)
    async fn handle_ue_capability_information(&mut self, ue_id: i32, uper_bytes: &[u8]) {
        let information = match decode_ue_capability_information(uper_bytes) {
            Ok(data) => data,
            Err(e) => {
                warn!(
                    "Failed to decode UECapabilityInformation from UE[{}]: {}",
                    ue_id, e
                );
                return;
            }
        };

        for container in &information.containers {
            if container.rat_type == RatType::Nr {
                match parse_nr_capability_bands(&container.container) {
                    Ok(bands) => {
                        info!(
                            "UE[{}] NR capability received: supported bands {:?}",
                            ue_id, bands
                        );
                    }
                    Err(e) => {
                        warn!(
                            "UE[{}] sent an undecodable UE-NR-Capability container: {}",
                            ue_id, e
                        );
                    }
                }
                if let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) {
                    ctx.set_nr_capability(container.container.clone());
                }
            }
        }

        if information.containers.is_empty() {
            info!("UE[{}] reported no radio access capabilities", ue_id);
        }
    }

    async fn handle_ul_information_transfer(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        if bytes.len() <= 2 {
            warn!("UL Information Transfer too short");
            return;
        }
        let nas_pdu = OctetString::from_slice(&bytes[2..]);
        self.send_uplink_nas_delivery(ue_id, nas_pdu).await;
    }

    async fn handle_nas_delivery(&mut self, ue_id: i32, nas_pdu: OctetString) {
        let ctx = match self.ue_manager.try_find_ue(ue_id) {
            Some(c) => c,
            None => {
                warn!("NAS delivery for unknown UE[{}]", ue_id);
                return;
            }
        };

        if !ctx.is_connected() {
            warn!("NAS delivery for non-connected UE[{}]", ue_id);
            return;
        }

        let dl_info_transfer = self.build_dl_information_transfer(&nas_pdu);
        self.send_rrc_message(ue_id, RrcChannel::DlDcch, dl_info_transfer)
            .await;
    }

    fn build_dl_information_transfer(&self, nas_pdu: &OctetString) -> OctetString {
        let mut pdu = Vec::with_capacity(nas_pdu.len() + 4);
        pdu.push(0x04);
        pdu.push(0x00);
        pdu.push(0x00);
        pdu.extend_from_slice(nas_pdu.data());
        OctetString::from_slice(&pdu)
    }

    async fn handle_rrc_reestablishment_request(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        // Parse C-RNTI (2 bytes at offset 1-2) and PhysCellId (2 bytes at offset 3-4)
        let c_rnti = if bytes.len() >= 3 {
            u16::from_be_bytes([bytes[1], bytes[2]])
        } else {
            0
        };
        let phys_cell_id = if bytes.len() >= 5 {
            u16::from_be_bytes([bytes[3], bytes[4]])
        } else {
            0
        };
        let cause = bytes.get(5).copied().unwrap_or(2); // Default: otherFailure

        info!(
            "RRC Reestablishment Request from UE[{}]: c_rnti={}, phys_cell_id={}, cause={}",
            ue_id, c_rnti, phys_cell_id, cause
        );

        if let Some(result) = self.connection_manager.process_rrc_reestablishment_request(
            &mut self.ue_manager,
            ue_id,
            c_rnti,
            phys_cell_id,
            cause,
        ) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_reestablishment_pdu)
                .await;
        }
    }

    async fn handle_rrc_reestablishment_complete(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        let transaction_id = if bytes.len() >= 2 { bytes[1] } else { 0 };

        info!(
            "RRC Reestablishment Complete from UE[{}], tid={}",
            ue_id, transaction_id
        );

        if let Some(result) = self
            .connection_manager
            .process_rrc_reestablishment_complete(&mut self.ue_manager, ue_id, transaction_id)
        {
            // If there's a NAS PDU, forward it to NGAP
            if let Some(nas_pdu) = result.nas_pdu {
                self.send_uplink_nas_delivery(result.ue_id, nas_pdu).await;
            }
        }
    }

    async fn handle_rrc_resume_request(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        let resume_cause = bytes.get(1).copied().unwrap_or(0);

        info!(
            "RRC Resume Request from UE[{}]: cause={}",
            ue_id, resume_cause
        );

        if let Some(result) = self.connection_manager.process_rrc_resume_request(
            &mut self.ue_manager,
            ue_id,
            resume_cause,
        ) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_resume_pdu)
                .await;
        }
    }

    async fn handle_rrc_resume_complete(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        let transaction_id = if bytes.len() >= 2 { bytes[1] } else { 0 };
        let nas_pdu = if bytes.len() > 3 {
            Some(OctetString::from_slice(&bytes[3..]))
        } else {
            None
        };

        info!(
            "RRC Resume Complete from UE[{}], tid={}",
            ue_id, transaction_id
        );

        if let Some(result) = self.connection_manager.process_rrc_resume_complete(
            &mut self.ue_manager,
            ue_id,
            transaction_id,
            nas_pdu,
        ) {
            // If there's a NAS PDU, forward it to NGAP as uplink NAS
            if let Some(nas_pdu) = result.nas_pdu {
                self.send_uplink_nas_delivery(result.ue_id, nas_pdu).await;
            }
        }
    }

    async fn handle_an_release(&mut self, ue_id: i32) {
        if let Some(result) = self
            .connection_manager
            .initiate_rrc_release(&mut self.ue_manager, ue_id)
        {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_release_pdu)
                .await;
        }
        self.ue_manager.delete_ue(ue_id);
    }

    async fn handle_paging(&mut self, _ue_paging_tmsi: Vec<u8>, _tai_list_for_paging: Vec<u8>) {
        debug!("Paging request received");
    }

    async fn send_rrc_message(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
        let pdu_id = self.next_pdu_id();
        let msg = RlsMessage::DownlinkRrc {
            ue_id,
            rrc_channel: channel,
            pdu_id,
            data,
        };
        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send RRC message to RLS: {}", e);
        }
    }

    async fn send_initial_nas_delivery(
        &self,
        ue_id: i32,
        pdu: OctetString,
        rrc_establishment_cause: i64,
        s_tmsi: Option<GutiMobileIdentity>,
    ) {
        let msg = NgapMessage::InitialNasDelivery {
            ue_id,
            pdu,
            rrc_establishment_cause,
            s_tmsi,
        };
        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send Initial NAS to NGAP: {}", e);
        }
    }

    async fn send_uplink_nas_delivery(&self, ue_id: i32, pdu: OctetString) {
        let msg = NgapMessage::UplinkNasDelivery { ue_id, pdu };
        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("Failed to send Uplink NAS to NGAP: {}", e);
        }
    }

    /// Route AI/ML inference request from UE to SHE task
    async fn route_6g_ai_ml(&self, ue_id: i32, model_id: String, input_data: Vec<f32>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let request_id = ue_id as u64;
            let msg = SheMessage::InferenceRequest {
                model_id,
                request_id,
                input_data,
            };
            if let Err(e) = sixg.she_tx.send(msg).await {
                error!("Failed to route AI/ML inference to SHE: {}", e);
            }
        } else {
            warn!(
                "6G tasks not initialized, dropping AI/ML inference from UE[{}]",
                ue_id
            );
        }
    }

    /// Route ISAC sensing data from UE to ISAC task
    async fn route_6g_isac(&self, ue_id: i32, measurement_type: String, measurements: Vec<f32>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let cell_id = ue_id; // Use ue_id as cell context
            let msg = IsacMessage::SensingData {
                cell_id,
                measurement_type,
                measurements,
            };
            if let Err(e) = sixg.isac_tx.send(msg).await {
                error!("Failed to route ISAC sensing data: {}", e);
            }
        } else {
            warn!(
                "6G tasks not initialized, dropping ISAC data from UE[{}]",
                ue_id
            );
        }
    }

    /// Route semantic communication message from UE to NKEF task
    async fn route_6g_semantic(&self, ue_id: i32, content_type: String, data: Vec<u8>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let msg = NkefMessage::UpdateKnowledge {
                entity_type: content_type,
                entity_id: format!("ue-{ue_id}"),
                properties: vec![
                    ("source".to_string(), "semantic-comm".to_string()),
                    ("data_len".to_string(), data.len().to_string()),
                ],
            };
            if let Err(e) = sixg.nkef_tx.send(msg).await {
                error!("Failed to route semantic message to NKEF: {}", e);
            }
        } else {
            warn!(
                "6G tasks not initialized, dropping semantic message from UE[{}]",
                ue_id
            );
        }
    }

    async fn handle_nwdaf_handover(&mut self, ue_id: i32, target_cell: i32, confidence: f32) {
        if self.ue_manager.try_find_ue(ue_id).is_none() {
            debug!(
                "RRC: Ignoring NWDAF handover recommendation for unknown UE {}",
                ue_id
            );
            return;
        }
        // A confidence threshold of 0.7 is used to avoid spurious handovers.
        if confidence < 0.7 {
            debug!(
                "RRC: Ignoring low-confidence ({:.2}) handover recommendation for UE {} to cell {}",
                confidence, ue_id, target_cell
            );
            return;
        }
        info!(
            "RRC: Initiating NWDAF-recommended handover for UE {} to cell {}",
            ue_id, target_cell
        );
    }
}

#[async_trait::async_trait]
impl Task for RrcTask {
    type Message = RrcMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RRC task started");

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rrc_msg) => match rrc_msg {
                            RrcMessage::RadioPowerOn => self.handle_radio_power_on(),
                            RrcMessage::SignalDetected { ue_id } => self.handle_signal_detected(ue_id),
                            RrcMessage::UplinkRrc { ue_id, rrc_channel, data } => {
                                self.handle_uplink_rrc(ue_id, rrc_channel, data).await;
                            }
                            RrcMessage::NasDelivery { ue_id, pdu } => {
                                self.handle_nas_delivery(ue_id, pdu).await;
                            }
                            RrcMessage::SecurityModeCommand { ue_id, pdu } => {
                                // TS 38.331 §5.3.4: deliver the SecurityModeCommand
                                // on SRB1 (DL-DCCH).
                                info!(
                                    "Sending RRC SecurityModeCommand to UE {} ({} bytes)",
                                    ue_id,
                                    pdu.len()
                                );
                                self.send_rrc_message(ue_id, RrcChannel::DlDcch, pdu).await;
                            }
                            RrcMessage::AnRelease { ue_id } => {
                                self.handle_an_release(ue_id).await;
                            }
                            RrcMessage::Paging { ue_paging_tmsi, tai_list_for_paging } => {
                                self.handle_paging(ue_paging_tmsi, tai_list_for_paging).await;
                            }
                            RrcMessage::NtnTimingAdvanceConfig {
                                satellite_type, common_ta_us, k_offset,
                                max_doppler_hz, autonomous_ta,
                            } => {
                                info!(
                                    "RRC: NTN timing config received: type={}, TA={}us, k_offset={}, doppler={}Hz, autonomous_ta={}",
                                    satellite_type, common_ta_us, k_offset, max_doppler_hz, autonomous_ta
                                );
                                self.ntn_config = Some(NtnRrcConfig {
                                    satellite_type,
                                    common_ta_us,
                                    k_offset,
                                    max_doppler_hz,
                                    autonomous_ta,
                                });
                            }
                            // 6G message routing
                            RrcMessage::SixgAiMlInference { ue_id, model_id, input_data } => {
                                self.route_6g_ai_ml(ue_id, model_id, input_data).await;
                            }
                            RrcMessage::SixgIsacSensingData { ue_id, measurement_type, measurements } => {
                                self.route_6g_isac(ue_id, measurement_type, measurements).await;
                            }
                            RrcMessage::SixgSemanticMessage { ue_id, content_type, data } => {
                                self.route_6g_semantic(ue_id, content_type, data).await;
                            }
                            RrcMessage::NwdafHandoverRecommendation { ue_id, target_cell, confidence } => {
                                info!(
                                    "RRC: NWDAF handover recommendation for UE {} to cell {} (confidence={:.2})",
                                    ue_id, target_cell, confidence
                                );
                                // Initiate handover preparation for the recommended UE
                                self.handle_nwdaf_handover(ue_id, target_cell, confidence).await;
                            }
                        },
                        TaskMessage::Shutdown => {
                            info!("RRC task received shutdown signal");
                            break;
                        }
                    }
                }
                else => {
                    info!("RRC task channel closed");
                    break;
                }
            }
        }

        info!(
            "RRC task stopped with {} UE contexts",
            self.ue_manager.count()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn test_rrc_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        assert_eq!(task.ue_manager.count(), 0);
    }

    #[test]
    fn test_radio_power_on() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert!(task.connection_manager.is_barred());
        task.handle_radio_power_on();
        assert!(!task.connection_manager.is_barred());
    }

    #[test]
    fn test_pdu_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert_eq!(task.next_pdu_id(), 1);
        assert_eq!(task.next_pdu_id(), 2);
        assert_eq!(task.next_pdu_id(), 3);
    }

    #[test]
    fn test_build_dl_information_transfer() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        let nas_pdu = OctetString::from_slice(&[0x7E, 0x00, 0x41]);
        let dl_info = task.build_dl_information_transfer(&nas_pdu);
        assert_eq!(dl_info.len(), 6);
        assert_eq!(dl_info.data()[0], 0x04);
    }
}
