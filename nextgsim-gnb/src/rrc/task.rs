//! RRC Task Implementation

use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GutiMobileIdentity, IsacMessage, NgapMessage, NkefMessage, RlsMessage, RrcMessage,
    SheMessage, Task, TaskMessage,
};
use nextgsim_common::OctetString;
use nextgsim_common::SNssai;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::dcch_dispatch::{dispatch_ul_dcch, UlDcchMessage};
use nextgsim_rrc::procedures::information_transfer::{
    encode_dl_information_transfer, DlInformationTransferParams,
};
use nextgsim_rrc::procedures::rrc_setup::{
    decode_rrc_setup_complete, decode_rrc_setup_request,
    RrcEstablishmentCause as AsnEstablishmentCause, RrcSetupCompleteData, UeIdentity,
};
use nextgsim_rrc::procedures::ue_capability::{
    decode_ue_capability_information, encode_ue_capability_enquiry, parse_nr_capability_bands,
    RatType, UeCapabilityEnquiryParams, UeCapabilityInformationData,
};

/// Simplified DL/UL-DCCH envelope code: first byte 0x06 marks a UE capability
/// transfer message; the remaining bytes are the real ASN.1 UPER encoding of
/// UECapabilityEnquiry (DL) / UECapabilityInformation (UL).
const RRC_MSG_TYPE_UE_CAPABILITY: u8 = 0x06;

use super::connection::RrcConnectionManager;
use super::transaction::{RrcProcedure, TidVerification, C5_TYPED_DCCH_DISPATCH};
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

    /// Handles `RrcMessage::RadioPowerOn`: activates (unbars) the cell.
    ///
    /// Public because it is a real message-handler entry point also driven
    /// directly by the in-process strict-peer harness
    /// (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub fn handle_radio_power_on(&mut self) {
        info!("Radio power on - cell is now active");
        self.connection_manager.set_barred(false);
    }

    fn handle_signal_detected(&mut self, ue_id: i32) {
        debug!("Signal detected from UE[{}]", ue_id);
    }

    /// Handles `RrcMessage::UplinkRrc`: dispatches an uplink RRC PDU by
    /// logical channel (UL-CCCH / UL-DCCH).
    ///
    /// Public because it is a real message-handler entry point also driven
    /// directly by the in-process strict-peer harness
    /// (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub async fn handle_uplink_rrc(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
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

        // ASN.1-first UL-DCCH dispatch (TS 38.331 §6.2.2, Wave-6 C3): a
        // conformant UE encodes RRCSetupComplete as a real UPER
        // UL-DCCH-Message (c1 CHOICE index 2 → leading byte 0x10..=0x17,
        // low nibble 0x0 for tid 0), which matches NO arm of the legacy
        // nibble dispatcher below and was previously dropped — losing the
        // registration NAS. This narrow RrcSetupComplete-only decode is
        // collision-free against the matched-sim UE's bespoke framing, so it
        // is always on. The BROAD typed dispatch (all c1 variants) is gated
        // separately below.
        if let Ok(complete) = decode_rrc_setup_complete(bytes) {
            self.handle_rrc_setup_complete_asn1(ue_id, complete).await;
            return;
        }

        // Wave-6 C5: full typed UL-DCCH-Message dispatch (TS 38.331 §6.2.1).
        // Gated behind C5_TYPED_DCCH_DISPATCH: the matched-sim UE still emits
        // bespoke framing whose leading bytes COLLIDE with real UPER (e.g. its
        // uplink NAS `[0x08, 0x00, NAS…]` decodes as a well-formed
        // RRCReconfigurationComplete and would swallow the NAS — see
        // nextgsim-rrc dcch_dispatch tests). So the broad typed dispatch is
        // only enabled once the peer UE is a conformant typed peer; until then
        // the legacy nibble arms below preserve the matched-sim path.
        if C5_TYPED_DCCH_DISPATCH && self.try_dispatch_typed_ul_dcch(ue_id, bytes).await {
            return;
        }

        // Legacy nibble fallback (bespoke `bytes[0] & 0x0F` framing + raw-NAS
        // heuristic). This is the sole non-typed dispatch path and is retired
        // in Wave-6 C6 once the UE is a conformant typed peer.
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
                    // Forward as Initial NAS (bespoke UL-DCCH fallback: no
                    // s-NSSAI-List is carried on this path).
                    self.send_initial_nas_delivery(ue_id, data.clone(), 3, None, Vec::new())
                        .await; // cause=3 (mo-Data)
                } else {
                    debug!("Unhandled UL-DCCH message type: {:#x}", message_type);
                }
            }
        }
    }

    /// Wave-6 C5: fully-typed UL-DCCH-Message dispatch (TS 38.331 §6.2.1).
    ///
    /// Decodes the UL-DCCH-Message once (nextgsim-rrc `dispatch_ul_dcch`) and
    /// routes each concrete c1 message type by its typed fields — no
    /// `bytes[0] & 0x0F` nibble matching, no `bytes[N..]` NAS slicing. Returns
    /// `true` if the message was recognised and handled, `false` otherwise (so
    /// the caller can fall through to the legacy path). Enabled on the wire
    /// only behind `C5_TYPED_DCCH_DISPATCH`; unit tests call it directly.
    async fn try_dispatch_typed_ul_dcch(&mut self, ue_id: i32, bytes: &[u8]) -> bool {
        match dispatch_ul_dcch(bytes) {
            Ok(UlDcchMessage::RrcSetupComplete(complete)) => {
                self.handle_rrc_setup_complete_asn1(ue_id, complete).await;
                true
            }
            Ok(UlDcchMessage::SecurityModeComplete(complete)) => {
                self.handle_security_mode_complete(ue_id, complete.rrc_transaction_id);
                true
            }
            Ok(UlDcchMessage::RrcReconfigurationComplete(complete)) => {
                self.handle_rrc_reconfiguration_complete_asn1(ue_id, complete.rrc_transaction_id);
                true
            }
            Ok(UlDcchMessage::UlInformationTransfer(transfer)) => {
                if let Some(nas) = transfer.dedicated_nas_message {
                    info!(
                        "UL InformationTransfer (ASN.1) from UE[{}]: forwarding {} NAS bytes",
                        ue_id,
                        nas.len()
                    );
                    self.send_uplink_nas_delivery(ue_id, OctetString::from_slice(&nas))
                        .await;
                }
                true
            }
            Ok(UlDcchMessage::UeCapabilityInformation(info)) => {
                self.process_ue_capability_information(ue_id, info);
                true
            }
            Ok(UlDcchMessage::Unsupported) => false,
            Err(e) => {
                debug!("UL-DCCH typed decode failed for UE[{}]: {}", ue_id, e);
                false
            }
        }
    }

    /// Handles a typed SecurityModeComplete (TS 38.331 §5.3.4).
    ///
    /// Verifies the echoed tid against the UE's outstanding SecurityMode
    /// transaction on the RRC context. NOTE: AS security is established on the
    /// gNB NGAP task's UE context (the SMC tid is allocated there), so the RRC
    /// context typically reports `NoOutstanding` and this handler confirms
    /// framing/plumbing only. Full cross-task AS-security completion (notifying
    /// the NGAP/AMF side and enforcing PDCP) is the paired UE-AS-security item
    /// (residue I5) and is NOT claimed here.
    fn handle_security_mode_complete(&mut self, ue_id: i32, echoed_tid: u8) {
        if let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) {
            match ctx
                .transactions
                .verify(RrcProcedure::SecurityMode, echoed_tid)
            {
                TidVerification::Mismatch { expected } => {
                    warn!(
                        "Discarding SecurityModeComplete from UE[{}]: echoed tid {} != \
                         outstanding {} (TS 38.331 §5.3.4)",
                        ue_id, echoed_tid, expected
                    );
                    return;
                }
                TidVerification::Match | TidVerification::NoOutstanding => {}
            }
        }
        info!(
            "SecurityModeComplete (ASN.1) from UE[{}], tid={} — AS security \
             activation confirmed at RRC framing level (see residue I5 for full \
             AS-security enforcement)",
            ue_id, echoed_tid
        );
    }

    /// Handles a typed RRCReconfigurationComplete (TS 38.331 §5.3.5).
    ///
    /// Verifies the echoed tid against the UE's outstanding Reconfiguration
    /// transaction. Like SecurityModeComplete, the Reconfiguration tid is
    /// allocated on the NGAP task context, so the RRC context typically reports
    /// `NoOutstanding` (framing/plumbing confirmation only).
    fn handle_rrc_reconfiguration_complete_asn1(&mut self, ue_id: i32, echoed_tid: u8) {
        if let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) {
            match ctx
                .transactions
                .verify(RrcProcedure::Reconfiguration, echoed_tid)
            {
                TidVerification::Mismatch { expected } => {
                    warn!(
                        "Discarding RRCReconfigurationComplete from UE[{}]: echoed tid {} != \
                         outstanding {} (TS 38.331 §5.3.5)",
                        ue_id, echoed_tid, expected
                    );
                    return;
                }
                TidVerification::Match | TidVerification::NoOutstanding => {}
            }
        }
        info!(
            "RRCReconfigurationComplete (ASN.1) from UE[{}], tid={}",
            ue_id, echoed_tid
        );
    }

    /// Handles a typed, ASN.1-decoded RRCSetupComplete (TS 38.331 §5.3.3.4).
    ///
    /// This is the primary path (Wave-6 C3): the UE's real UPER UL-DCCH
    /// encoding carries the transaction id, the dedicated NAS message, and
    /// the RedCap indication as typed fields — no byte-offset extraction.
    async fn handle_rrc_setup_complete_asn1(&mut self, ue_id: i32, complete: RrcSetupCompleteData) {
        // Requested S-NSSAI(s) the UE carried in RRCSetupComplete (TS 38.331
        // §6.2.2), used downstream for slice-aware AMF selection. The RRC codec
        // models SD as a u32; convert to the common S-NSSAI (3-byte SD).
        let s_nssai_list: Vec<SNssai> = complete
            .s_nssai_list
            .as_ref()
            .map(|list| {
                list.iter()
                    .map(|s| match s.sd {
                        Some(sd) => SNssai::with_sd_u32(s.sst, sd),
                        None => SNssai::new(s.sst),
                    })
                    .collect()
            })
            .unwrap_or_default();

        info!(
            "RRC Setup Complete (ASN.1 UL-DCCH) from UE[{}]: tid={}, nas_len={}, redcap={}, s_nssai_count={}",
            ue_id,
            complete.rrc_transaction_id,
            complete.dedicated_nas_message.len(),
            complete.redcap_indication,
            s_nssai_list.len()
        );

        let nas_pdu = OctetString::from_slice(&complete.dedicated_nas_message);
        self.finish_rrc_setup_complete(
            ue_id,
            complete.rrc_transaction_id,
            nas_pdu,
            complete.redcap_indication,
            s_nssai_list,
            "ASN.1",
        )
        .await;
    }

    /// Handles a bespoke (non-ASN.1) RRCSetupComplete. Only reached when the
    /// ASN.1-first decode in `handle_ul_dcch_message` failed — i.e. for the
    /// UE's legacy fallback framing `[0x04, tid, 0x01, NAS...]` (retired in C6).
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

        info!(
            "RRC Setup Complete (bespoke fallback) from UE[{}]: tid={}, nas_len={}",
            ue_id,
            transaction_id,
            nas_pdu.len()
        );

        self.finish_rrc_setup_complete(
            ue_id,
            transaction_id,
            nas_pdu,
            redcap_indication,
            // The bespoke framing carries no s-NSSAI-List; slice-aware selection
            // is only available on the ASN.1 path.
            Vec::new(),
            "bespoke-fallback",
        )
        .await;
    }

    /// Common tail for both RRCSetupComplete decode paths: transitions the UE
    /// context to Connected, forwards the NAS PDU to NGAP as an Initial UE
    /// Message, and enquires UE capabilities. `via` names the decode path in
    /// the log line so the ASN.1 and bespoke-fallback paths are
    /// distinguishable in an E2E log (Wave-6 C3 acceptance).
    async fn finish_rrc_setup_complete(
        &mut self,
        ue_id: i32,
        transaction_id: u8,
        nas_pdu: OctetString,
        redcap_indication: bool,
        s_nssai_list: Vec<SNssai>,
        via: &str,
    ) {
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
            info!(
                "Initial UE Message via {} RRCSetupComplete path (ue_id={}, nas_len={})",
                via,
                result.ue_id,
                result.nas_pdu.len()
            );
            self.send_initial_nas_delivery(
                result.ue_id,
                result.nas_pdu,
                result.establishment_cause,
                result.s_tmsi,
                s_nssai_list,
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
    /// RF supports. This is modelled only: the functional simulator has no PRB
    /// scheduler, so the computed ceiling is logged, not enforced on a resource
    /// grid.
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

        // Model the RedCap bandwidth restriction and derive the equivalent PRB
        // ceiling for logging. This functional simulator has no PRB scheduler,
        // so the ceiling is reported, not enforced on a resource grid.
        let enforced_bw_mhz = ctx.redcap.restrict_bandwidth(CELL_BANDWIDTH_MHZ);
        let enforced_max_prb =
            (CELL_MAX_PRB * enforced_bw_mhz as u32 / CELL_BANDWIDTH_MHZ as u32).max(1);

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
        // Wave-6 C4-final: allocate the UECapabilityEnquiry tid from THIS UE's
        // per-context allocator (TS 38.331 §5.6.1 / §6.3.2), recorded as
        // outstanding so the tid echoed in UECapabilityInformation is verified.
        // The wire value is pinned to 0 while C5_TYPED_DCCH_DISPATCH is off (the
        // UE DL-DCCH dispatcher is still the legacy nibble matcher); it becomes
        // the full per-UE 0..3 cycle when C5 lands.
        let rrc_transaction_id = self
            .ue_manager
            .try_find_ue_mut(ue_id)
            .map(|ctx| ctx.transactions.allocate(RrcProcedure::UeCapability))
            .unwrap_or(0);
        let params = UeCapabilityEnquiryParams {
            rrc_transaction_id,
            rat_types: vec![RatType::Nr],
        };
        match encode_ue_capability_enquiry(&params) {
            Ok(uper) => {
                // Wave-6 C5: `uper` is already a complete UPER DL-DCCH-Message
                // (c1 = ueCapabilityEnquiry, TS 38.331 §6.2.1). When
                // C5_TYPED_DCCH_DISPATCH is on, send it raw — no bespoke `0x06`
                // envelope byte. While off, keep the `0x06` envelope the
                // matched-sim UE dispatcher matches on (dropped in C6).
                let pdu = if C5_TYPED_DCCH_DISPATCH {
                    uper
                } else {
                    let mut framed = Vec::with_capacity(uper.len() + 1);
                    framed.push(RRC_MSG_TYPE_UE_CAPABILITY);
                    framed.extend_from_slice(&uper);
                    framed
                };
                info!("Sending UECapabilityEnquiry to UE[{}]", ue_id);
                self.send_rrc_message(ue_id, RrcChannel::DlDcch, OctetString::from_slice(&pdu))
                    .await;
            }
            Err(e) => {
                error!("Failed to encode UECapabilityEnquiry: {}", e);
            }
        }
    }

    /// Handles a UECapabilityInformation from the UE (TS 38.331 §5.6.1),
    /// decoding the legacy `0x06`-envelope framing. Delegates the tid
    /// verification + capability handling to [`Self::process_ue_capability_information`],
    /// which the C5 typed dispatch also uses.
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
        self.process_ue_capability_information(ue_id, information);
    }

    /// Common core for a decoded UECapabilityInformation (TS 38.331 §5.6.1):
    /// verifies the echoed tid, parses the NR capability bands, and stores the
    /// container. Reached from both the legacy `0x06`-envelope path and the C5
    /// typed UL-DCCH dispatch.
    fn process_ue_capability_information(
        &mut self,
        ue_id: i32,
        information: UeCapabilityInformationData,
    ) {
        // Wave-6 C4-final: verify the tid the UE echoed against the outstanding
        // UECapabilityEnquiry transaction (TS 38.331 §5.6.1). Fail-closed on a
        // mismatch; tolerate NoOutstanding (no enquiry recorded for this UE).
        if let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) {
            match ctx
                .transactions
                .verify(RrcProcedure::UeCapability, information.rrc_transaction_id)
            {
                TidVerification::Mismatch { expected } => {
                    warn!(
                        "Discarding UECapabilityInformation from UE[{}]: echoed tid \
                         {} != outstanding {} (TS 38.331 §5.6.1)",
                        ue_id, information.rrc_transaction_id, expected
                    );
                    return;
                }
                TidVerification::Match | TidVerification::NoOutstanding => {}
            }
        }

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

    /// Builds the downlink NAS transport PDU delivered to the UE on SRB1.
    ///
    /// Wave-6 C5: when `C5_TYPED_DCCH_DISPATCH` is on, emits the real UPER
    /// DL-DCCH DLInformationTransfer (TS 38.331 §5.7.1 / §6.2.1). While off it
    /// keeps the legacy bespoke `[0x04, 0x00, 0x00, NAS…]` framing the
    /// matched-sim UE's nibble dispatcher expects (retired in C6).
    fn build_dl_information_transfer(&self, nas_pdu: &OctetString) -> OctetString {
        if C5_TYPED_DCCH_DISPATCH {
            if let Some(bytes) = Self::build_dl_information_transfer_typed(nas_pdu) {
                return OctetString::from_slice(&bytes);
            }
            // Typed encode failed — fall through to the legacy framing so the
            // NAS is still delivered (fully fail-closed handling is C6).
        }
        let mut pdu = Vec::with_capacity(nas_pdu.len() + 3);
        pdu.push(0x04);
        pdu.push(0x00);
        pdu.push(0x00);
        pdu.extend_from_slice(nas_pdu.data());
        OctetString::from_slice(&pdu)
    }

    /// Encodes a real UPER DL-DCCH DLInformationTransfer (tid 0) carrying
    /// `nas_pdu` (TS 38.331 §5.7.1). The typed encoder used on the wire when
    /// C5 is enabled; exercised directly by unit tests regardless of the
    /// compile-time gate.
    fn build_dl_information_transfer_typed(nas_pdu: &OctetString) -> Option<Vec<u8>> {
        match encode_dl_information_transfer(&DlInformationTransferParams {
            rrc_transaction_id: 0,
            dedicated_nas_message: Some(nas_pdu.data().to_vec()),
        }) {
            Ok(bytes) => Some(bytes),
            Err(e) => {
                error!("Failed to encode typed DLInformationTransfer: {}", e);
                None
            }
        }
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
        s_nssai_list: Vec<SNssai>,
    ) {
        let msg = NgapMessage::InitialNasDelivery {
            ue_id,
            pdu,
            rrc_establishment_cause,
            s_tmsi,
            s_nssai_list,
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
                            RrcMessage::RrcReconfiguration { ue_id, pdu } => {
                                // TS 38.331 §5.3.5.6: deliver the RRCReconfiguration
                                // (DRB setup) on SRB1 (DL-DCCH).
                                info!(
                                    "Sending RRC Reconfiguration to UE {} ({} bytes)",
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

    // ========================================================================
    // Wave-6 C3: UL-DCCH RRCSetupComplete dispatch — additive-accept.
    // The gNB must accept BOTH the UE's primary ASN.1 UPER encoding
    // (TS 38.331 §6.2.2, leading byte 0x10 for tid 0 — previously dropped by
    // the nibble dispatcher) and the UE's bespoke fallback framing
    // [0x04, tid, 0x01, NAS...]. One dedicated test per encoding.
    // ========================================================================

    fn run_async<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(fut)
    }

    /// Drives a real ASN.1 RRCSetupRequest through `handle_uplink_rrc` so the
    /// gNB creates the UE context and emits the RRCSetup (fresh task → tid 0).
    async fn establish_pending_setup(task: &mut RrcTask, ue_id: i32) {
        use nextgsim_rrc::procedures::rrc_setup::{
            encode_rrc_setup_request, RrcSetupRequestParams,
        };

        task.handle_radio_power_on();
        let req = encode_rrc_setup_request(&RrcSetupRequestParams {
            ue_identity: UeIdentity::RandomValue(0x1234567890),
            establishment_cause: AsnEstablishmentCause::MoSignalling,
        })
        .expect("encode RRCSetupRequest");
        task.handle_uplink_rrc(ue_id, RrcChannel::UlCcch, OctetString::from_slice(&req))
            .await;
    }

    /// Pops the next Initial UE Message from the NGAP channel, if any.
    fn try_take_initial_nas(
        ngap_rx: &mut mpsc::Receiver<TaskMessage<NgapMessage>>,
    ) -> Option<(i32, OctetString)> {
        while let Ok(msg) = ngap_rx.try_recv() {
            if let TaskMessage::Message(NgapMessage::InitialNasDelivery { ue_id, pdu, .. }) = msg {
                return Some((ue_id, pdu));
            }
        }
        None
    }

    #[test]
    fn test_ul_dcch_asn1_rrc_setup_complete_accepted() {
        use nextgsim_rrc::procedures::rrc_setup::{
            encode_rrc_setup_complete, RrcSetupCompleteParams,
        };

        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            establish_pending_setup(&mut task, 1).await;

            let nas: Vec<u8> = vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D];
            let complete = encode_rrc_setup_complete(&RrcSetupCompleteParams {
                registered_amf: None,
                rrc_transaction_id: 0,
                selected_plmn_identity: 1,
                guami_type: None,
                s_nssai_list: None,
                dedicated_nas_message: nas.clone(),
                ng_5g_s_tmsi_value: None,
                redcap_indication: false,
            })
            .expect("encode RRCSetupComplete");

            // UL-DCCH-Message c1 index 2 (rrcSetupComplete), tid 0: leading
            // byte 0x10, low nibble 0x0 — matches no legacy dispatch arm.
            assert_eq!(complete[0], 0x10, "ASN.1 RRCSetupComplete leading byte");

            task.handle_uplink_rrc(1, RrcChannel::UlDcch, OctetString::from_slice(&complete))
                .await;

            let (ue_id, pdu) = try_take_initial_nas(&mut ngap_rx)
                .expect("ASN.1 RRCSetupComplete must produce an Initial UE Message");
            assert_eq!(ue_id, 1);
            assert_eq!(pdu.data(), &nas[..], "NAS must be byte-for-byte identical");
        });
    }

    #[test]
    fn test_ul_dcch_bespoke_rrc_setup_complete_accepted() {
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            establish_pending_setup(&mut task, 1).await;

            let nas: Vec<u8> = vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D];
            // Bespoke fallback framing from the UE (ue/rrc/task.rs
            // send_rrc_setup_complete fallback): [0x04, tid, 0x01, NAS...].
            let mut bespoke = vec![0x04, 0x00, 0x01];
            bespoke.extend_from_slice(&nas);

            task.handle_uplink_rrc(1, RrcChannel::UlDcch, OctetString::from_slice(&bespoke))
                .await;

            let (ue_id, pdu) = try_take_initial_nas(&mut ngap_rx)
                .expect("bespoke RRCSetupComplete must still produce an Initial UE Message");
            assert_eq!(ue_id, 1);
            assert_eq!(pdu.data(), &nas[..], "NAS must be byte-for-byte identical");
        });
    }

    // ========================================================================
    // Wave-6 C5: fully-typed UL-DCCH dispatch (TS 38.331 §6.2.1). The gNB
    // routes every c1 message type from its typed fields — no `bytes[0] & 0x0F`
    // nibble matching, no `bytes[N..]` NAS slicing. `try_dispatch_typed_ul_dcch`
    // is exercised DIRECTLY (independent of the compile-time
    // C5_TYPED_DCCH_DISPATCH wire gate), mirroring the transaction-allocator
    // test pattern in transaction.rs.
    // ========================================================================

    /// Pops the next Uplink NAS Delivery from the NGAP channel, if any.
    fn try_take_uplink_nas(
        ngap_rx: &mut mpsc::Receiver<TaskMessage<NgapMessage>>,
    ) -> Option<(i32, OctetString)> {
        while let Ok(msg) = ngap_rx.try_recv() {
            if let TaskMessage::Message(NgapMessage::UplinkNasDelivery { ue_id, pdu }) = msg {
                return Some((ue_id, pdu));
            }
        }
        None
    }

    #[test]
    fn test_typed_ul_dcch_setup_complete_dispatched() {
        use nextgsim_rrc::procedures::rrc_setup::{
            encode_rrc_setup_complete, RrcSetupCompleteParams,
        };
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            establish_pending_setup(&mut task, 1).await;
            let nas = vec![0x7E, 0x00, 0x41, 0x02];
            let complete = encode_rrc_setup_complete(&RrcSetupCompleteParams {
                registered_amf: None,
                rrc_transaction_id: 0,
                selected_plmn_identity: 1,
                guami_type: None,
                s_nssai_list: None,
                dedicated_nas_message: nas.clone(),
                ng_5g_s_tmsi_value: None,
                redcap_indication: false,
            })
            .unwrap();
            assert!(
                task.try_dispatch_typed_ul_dcch(1, &complete).await,
                "typed dispatch must handle RRCSetupComplete"
            );
            let (ue_id, pdu) = try_take_initial_nas(&mut ngap_rx).expect("Initial UE Message");
            assert_eq!(ue_id, 1);
            assert_eq!(pdu.data(), &nas[..]);
        });
    }

    #[test]
    fn test_typed_ul_dcch_ul_information_transfer_forwards_nas() {
        use nextgsim_rrc::procedures::information_transfer::{
            encode_ul_information_transfer, UlInformationTransferParams,
        };
        let config = test_config();
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            let nas = vec![0x7E, 0x00, 0x55, 0xAB];
            let ulit = encode_ul_information_transfer(&UlInformationTransferParams {
                dedicated_nas_message: Some(nas.clone()),
            })
            .unwrap();
            assert!(task.try_dispatch_typed_ul_dcch(9, &ulit).await);
            let (ue_id, pdu) = try_take_uplink_nas(&mut ngap_rx).expect("Uplink NAS Delivery");
            assert_eq!(ue_id, 9);
            assert_eq!(
                pdu.data(),
                &nas[..],
                "uplink NAS must survive typed dispatch byte-for-byte"
            );
        });
    }

    #[test]
    fn test_typed_ul_dcch_security_mode_complete_tid_verified() {
        use nextgsim_rrc::procedures::security_mode::{
            encode_security_mode_complete, SecurityModeCompleteParams,
        };
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            // Seed a UE context with an outstanding SecurityMode tid (0).
            {
                let ctx = task.ue_manager.create_ue(1);
                assert_eq!(
                    ctx.transactions
                        .allocate_cycling(RrcProcedure::SecurityMode),
                    0
                );
            }
            // Mismatch (echo tid 1) — fail-closed: outstanding stays pending.
            let bad = encode_security_mode_complete(&SecurityModeCompleteParams {
                rrc_transaction_id: 1,
            })
            .unwrap();
            assert!(task.try_dispatch_typed_ul_dcch(1, &bad).await);
            assert_eq!(
                task.ue_manager
                    .try_find_ue(1)
                    .unwrap()
                    .transactions
                    .outstanding(RrcProcedure::SecurityMode),
                Some(0),
                "mismatched SecurityModeComplete must not clear the outstanding tid"
            );
            // Correct echo (tid 0) — verified & cleared.
            let good = encode_security_mode_complete(&SecurityModeCompleteParams {
                rrc_transaction_id: 0,
            })
            .unwrap();
            assert!(task.try_dispatch_typed_ul_dcch(1, &good).await);
            assert_eq!(
                task.ue_manager
                    .try_find_ue(1)
                    .unwrap()
                    .transactions
                    .outstanding(RrcProcedure::SecurityMode),
                None,
                "correct SecurityModeComplete tid clears the outstanding transaction"
            );
        });
    }

    #[test]
    fn test_typed_ul_dcch_reconfiguration_complete_dispatched() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            encode_rrc_reconfiguration_complete, RrcReconfigurationCompleteParams,
        };
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            let reconf_complete =
                encode_rrc_reconfiguration_complete(&RrcReconfigurationCompleteParams {
                    rrc_transaction_id: 0,
                })
                .unwrap();
            assert!(
                task.try_dispatch_typed_ul_dcch(3, &reconf_complete).await,
                "typed dispatch must handle RRCReconfigurationComplete"
            );
        });
    }

    #[test]
    fn test_typed_ul_dcch_unsupported_falls_through() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            // Leading bit 1 -> messageClassExtension: a well-formed but
            // non-dispatched UL-DCCH-Message. The typed dispatcher returns false
            // so the caller can fall through to the legacy path.
            assert!(!task.try_dispatch_typed_ul_dcch(1, &[0x80]).await);
        });
    }

    /// The C5-enabled typed DLInformationTransfer builder must produce the
    /// hand-derived golden UPER bytes (TS 38.331 §5.7.1; see nextgsim-rrc
    /// `golden_dl_information_transfer_bytes`). Exercised directly regardless
    /// of the compile-time wire gate.
    #[test]
    fn test_build_dl_information_transfer_typed_golden() {
        let nas = OctetString::from_slice(&[0x7E, 0x00, 0x42]);
        let bytes = RrcTask::build_dl_information_transfer_typed(&nas)
            .expect("typed DLInformationTransfer must encode");
        assert_eq!(
            bytes,
            vec![0x28, 0x80, 0x6F, 0xC0, 0x08, 0x40],
            "typed DLInformationTransfer(tid 0, NAS) must equal the golden UPER"
        );
    }

    /// While the wire gate is OFF (default, matched-sim safe), the legacy
    /// bespoke DL framing is preserved verbatim so the unchanged UE dispatcher
    /// keeps routing.
    #[test]
    fn test_build_dl_information_transfer_legacy_while_c5_off() {
        if C5_TYPED_DCCH_DISPATCH {
            return;
        }
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        let nas = OctetString::from_slice(&[0x7E, 0x00, 0x42]);
        let pdu = task.build_dl_information_transfer(&nas);
        assert_eq!(
            pdu.data(),
            &[0x04, 0x00, 0x00, 0x7E, 0x00, 0x42][..],
            "C5-off: legacy bespoke DL framing preserved for the matched-sim UE"
        );
    }
}
