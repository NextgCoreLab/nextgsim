//! RRC Task Implementation

use std::time::Duration;

use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    GnbTaskBase, GutiMobileIdentity, HandoverInitiation, IsacMessage, NgapMessage, NkefMessage,
    RlsMessage, RrcMessage, SheMessage, Task, TaskMessage,
};
use nextgsim_common::frame_clock;
use nextgsim_common::OctetString;
use nextgsim_common::SNssai;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::dcch_dispatch::{dispatch_ul_dcch, UlDcchMessage};
use nextgsim_rrc::procedures::information_transfer::{
    encode_dl_information_transfer, DlInformationTransferParams,
};
use nextgsim_rrc::procedures::paging::{encode_paging, PagingRecordParams, FIVE_G_S_TMSI_LEN};
use nextgsim_rrc::procedures::paging_occasion::{
    self, paging_occasion, ue_id_from_s_tmsi, PagingCycleConfig,
};
use nextgsim_rrc::procedures::rrc_reestablishment::{
    decode_rrc_reestablishment_complete, decode_rrc_reestablishment_request,
    RrcReestablishmentRequestData,
};
use nextgsim_rrc::procedures::rrc_resume::{decode_rrc_resume_request, decode_rrc_resume_request1};
use nextgsim_rrc::procedures::rrc_setup::{
    decode_rrc_setup_complete, decode_rrc_setup_request,
    RrcEstablishmentCause as AsnEstablishmentCause, RrcSetupCompleteData, UeIdentity,
};
use nextgsim_rrc::procedures::scell_config::{encode_scell_config, ScellConfig};
use nextgsim_rrc::procedures::ue_capability::{
    decode_ue_capability_information, encode_ue_capability_enquiry, parse_nr_capability_bands,
    RatType, UeCapabilityEnquiryParams, UeCapabilityInformationData,
};

/// Simplified DL/UL-DCCH envelope code: first byte 0x06 marks a UE capability
/// transfer message; the remaining bytes are the real ASN.1 UPER encoding of
/// UECapabilityEnquiry (DL) / UECapabilityInformation (UL).
const RRC_MSG_TYPE_UE_CAPABILITY: u8 = 0x06;

/// Simplified DL-DCCH envelope code for an RRCReconfiguration carrying a
/// secondary-cell configuration: `[0x0E][transaction id][UPER CellGroupConfig]`.
/// The UE's matching constant is `nextgsim-ue/src/rrc/task.rs`.
const RECONFIGURATION_WITH_SCELL: u8 = 0x0E;

/// `sCellIndex` used for the one secondary cell this gNB can be configured with.
/// `SCellIndex` is `INTEGER (1..31)` and 1 is its first value.
const DEFAULT_SCELL_INDEX: u8 = 1;

/// Period the system-information timer is parked at when the broadcast is
/// disabled (`si_broadcast_period_ms == 0`). An hour: long enough never to matter,
/// finite so the select! arm stays well-formed.
const SI_PARKED_PERIOD_MS: u64 = 3_600_000;

/// The cell's default paging cycle in radio frames when neither the AMF's
/// `(default)PagingDRX` nor the configuration supplies one.
///
/// 128 frames (1.28 s) is the middle of `PCCH-Config.defaultPagingCycle`'s range
/// and matches the `PagingDrx::V128` this gNB already advertises in its NG Setup.
pub const DEFAULT_PAGING_CYCLE_FRAMES: u16 = 128;

/// When a paged UE's paging frame falls (TS 38.304 §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingSchedule {
    /// `UE_ID` = 5G-S-TMSI mod 1024.
    pub ue_id: u16,
    /// The DRX cycle actually used, in radio frames.
    pub t: u16,
    /// The SFN the PCCH Paging will be transmitted in.
    pub paging_frame: u16,
    /// How long from now until that frame.
    pub delay: std::time::Duration,
}

/// Octets per serialised TAI in an `RrcMessage::Paging` TAI list: a 3-octet
/// PLMN identity followed by a 3-octet TAC (TS 38.413 §9.3.3.11).
const TAI_OCTETS: usize = 6;

use super::connection::{
    ReestablishmentRequest, ResumeRequestPresented, RrcConnectionManager, SuspendParams,
};
use super::handover::GnbHandoverManager;
use super::system_info::{
    encode_cell_mib, encode_cell_sib1, encode_cell_system_information,
    release_cell_reselection_priorities,
};
use super::transaction::{RrcProcedure, TidVerification, C5_TYPED_DCCH_DISPATCH};
use super::ue_context::{ReestablishmentSecurity, RrcUeContextManager, SuspendedIdentity};
use nextgsim_pdcp::srb_security::SrbSecurity;

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
    /// UEs that have already been sent the configured secondary cell
    /// (`GnbConfig::scell_phys_cell_id`). `sCellToAddModList` is an add/modify
    /// list, so resending is harmless, but a UE acknowledges each one and the
    /// acknowledgement is what triggers the next reconfiguration — sending it
    /// once per UE keeps that from becoming a loop.
    scell_configured_ues: std::collections::HashSet<i32>,
    /// Mobility state per UE (issue #39).
    ///
    /// Instantiated here because `GnbHandoverManager` had no owner at all: it was
    /// exported from `rrc/mod.rs` and exercised only by a unit test, so an
    /// NWDAF-recommended handover could log success and reach no handover machinery.
    handover_manager: GnbHandoverManager,
}

/// PDCP BEARER for SRB1 (TS 38.323 §5.9: the SRB identity minus one, so 0).
const SRB1_PDCP_BEARER: u8 = 0;

/// The confidence below which an NWDAF handover recommendation is ignored.
///
/// Named because it is a policy the operator would want to see, and because a literal
/// buried in a comparison reads as arbitrary.
const NWDAF_HANDOVER_CONFIDENCE_THRESHOLD: f32 = 0.7;

/// PDCP DIRECTION for downlink (TS 33.501 Annex D: 1).
const PDCP_DIRECTION_DOWNLINK: u8 = 1;

/// PDCP DIRECTION for uplink (0).
const PDCP_DIRECTION_UPLINK: u8 = 0;

/// The PDCP COUNT the SecurityModeCommand is integrity-protected with.
///
/// Zero, matching the UE's `SMC_PDCP_COUNT`. The downlink sequence therefore
/// continues at 1, which is why `RrcUeContext::dl_pdcp_count` starts there.
const GNB_SMC_PDCP_COUNT: u32 = 0;

/// Re-exported for the tests, which assert the appended MAC-I's length.
#[cfg(test)]
const MAC_I_LEN_GNB: usize = nextgsim_pdcp::srb_security::MAC_I_LEN;

impl RrcTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        // Keyed on this cell's own identity, which is what an intra-gNB handover
        // decision compares a recommended target against. Read before `task_base` moves.
        let own_cell = (task_base.config.nci & 0xF_FFFF_FFFF) as i32;
        Self {
            task_base,
            ue_manager: RrcUeContextManager::new(),
            connection_manager: RrcConnectionManager::new(),
            pdu_id_counter: 0,
            ntn_config: None,
            scell_configured_ues: std::collections::HashSet::new(),
            handover_manager: GnbHandoverManager::new(own_cell),
        }
    }

    /// Send the configured secondary cell to a UE, once (TS 38.331 §5.3.5.5.9).
    ///
    /// No-op unless `GnbConfig::scell_phys_cell_id` is set. The envelope is the
    /// simulator's hand-rolled DL-DCCH framing (issue #107); the container is a
    /// real UPER `CellGroupConfig`, because `sCellToAddModList` is a Rel-15 IE.
    async fn send_scell_configuration(&mut self, ue_id: i32) {
        let Some(phys_cell_id) = self.task_base.config.scell_phys_cell_id else {
            return;
        };
        if !self.scell_configured_ues.insert(ue_id) {
            return;
        }

        let config = ScellConfig::add_one(DEFAULT_SCELL_INDEX, phys_cell_id);
        let container = match encode_scell_config(&config) {
            Ok(bytes) => bytes,
            Err(e) => {
                warn!(
                    "Not sending SCell configuration to UE[{}]: {} (scell_phys_cell_id={})",
                    ue_id, e, phys_cell_id
                );
                self.scell_configured_ues.remove(&ue_id);
                return;
            }
        };

        let transaction_id = self.connection_manager.next_tid();
        let mut pdu = Vec::with_capacity(container.len() + 2);
        pdu.push(RECONFIGURATION_WITH_SCELL);
        pdu.push(transaction_id);
        pdu.extend_from_slice(&container);

        info!(
            "Sending SCell configuration to UE[{}]: sCellIndex={}, physCellId={}, tid={}",
            ue_id, DEFAULT_SCELL_INDEX, phys_cell_id, transaction_id
        );
        self.send_rrc_message(ue_id, RrcChannel::DlDcch, OctetString::from_slice(&pdu))
            .await;
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

        // RRCReestablishmentRequest is tried FIRST, and the order is load-bearing
        // for the legacy fallback below rather than for the ASN.1 decoders (which
        // discriminate on the UL-CCCH c1 index and so reject each other's PDUs).
        if let Ok(request) = decode_rrc_reestablishment_request(bytes) {
            self.handle_rrc_reestablishment_request(ue_id, &request)
                .await;
            return;
        }

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
            // No legacy arm for an RRCReestablishmentRequest. There was one, on
            // 0x24, and nothing ever reached it: the UE's bespoke framing put
            // 0x05 in the first byte, which falls in the 0x00..=0x1F RRCSetupRequest
            // range above — and worse, that bespoke PDU *decodes* as a UPER
            // RRCSetupRequest, so a re-establishment was answered with an
            // RRCSetup built on a fabricated context. Both ends now use the real
            // UPER encoding, handled before this fallback ladder.
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

        // PDCP-unprotect BEFORE any dispatch (TS 38.323 §5.8/§5.9, issue #31). It
        // has to be before: every dispatcher below reads the PDU's leading bytes,
        // and a ciphered PDU has no meaningful leading nibble. A PDU that fails
        // integrity is DISCARDED here and never reaches a handler (TS 33.501 §6.5).
        let Some(data) = self.unprotect_ul_dcch(ue_id, data) else {
            return;
        };
        let data = &data;

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
                // A raw NAS PDU on UL-DCCH, from a UE that never sent an
                // RRCSetupComplete. EPD 0x7E is 5GMM, 0x2E is 5GSM.
                //
                // GATED, AND STRICT BY DEFAULT (issue #30, criterion 6). TS 38.331
                // §5.3.3 carries the initial NAS inside RRCSetupComplete, so a UE
                // that skips the establishment handshake must fail to attach
                // rather than be helped along. This leniency existed because this
                // gNB's own UE used to smuggle the initial NAS as raw DCCH; #30
                // wired the conformant library RrcTask into the UE binary, so the
                // matched pair no longer needs it and all it can do now is hide a
                // non-conformant peer -- which is the opposite of what a
                // conformance simulator is for.
                let is_nas_pdu = bytes.len() >= 3 && (bytes[0] == 0x7E || bytes[0] == 0x2E);
                if is_nas_pdu && !self.task_base.config.accept_raw_nas_on_dcch {
                    warn!(
                        "Discarding raw NAS PDU on UL-DCCH from UE[{}] (epd=0x{:02x}): \
                         TS 38.331 §5.3.3 requires the initial NAS inside \
                         RRCSetupComplete. Set accept_raw_nas_on_dcch to interop \
                         with a UE that sends bare NAS on DCCH.",
                        ue_id, bytes[0]
                    );
                    return;
                }
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
        // TS 38.331 §5.3.4.3: the UE has confirmed it can verify the command, so
        // AS security is ACTIVE. From here the gNB protects DL-DCCH and verifies
        // UL-DCCH (issue #31). Before this the log claimed activation "at RRC
        // framing level" and nothing was protected in either direction.
        if self.task_base.config.as_security_enabled {
            if let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) {
                ctx.on_as_security_activated();
            }
        }
        info!(
            "SecurityModeComplete from UE[{}], tid={} — AS security {}",
            ue_id,
            echoed_tid,
            if self.task_base.config.as_security_enabled {
                "ACTIVE: SRB1 is now integrity protected and ciphered in both directions"
            } else {
                "confirmed at RRC framing level only (as_security_enabled is off, so \
                 nothing is protected)"
            }
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

    /// Handles a decoded `RRCReestablishmentRequest` (TS 38.331 §5.3.7.2).
    ///
    /// The identity and `shortMAC-I` come from the ASN.1 decode. On a failed or
    /// unresolvable verification the gNB answers with an `RRCSetup` per §5.3.3.1
    /// rather than reestablishing a UE it cannot authenticate.
    async fn handle_rrc_reestablishment_request(
        &mut self,
        ue_id: i32,
        request: &RrcReestablishmentRequestData,
    ) {
        info!(
            "RRC Reestablishment Request on UE[{}]: c_rnti={:#06x}, pci={}, \
             shortMAC-I={:#06x}, cause={:?}",
            ue_id,
            request.ue_identity.c_rnti,
            request.ue_identity.phys_cell_id,
            request.ue_identity.short_mac_i,
            request.reestablishment_cause
        );

        let presented = ReestablishmentRequest {
            c_rnti: request.ue_identity.c_rnti,
            phys_cell_id: request.ue_identity.phys_cell_id,
            short_mac_i: request.ue_identity.short_mac_i,
            cause: request.reestablishment_cause,
            target_cell_identity: self.cell_identity(),
        };

        match self
            .connection_manager
            .process_rrc_reestablishment_request(&mut self.ue_manager, &presented)
        {
            Ok(result) => {
                self.send_rrc_message(result.ue_id, result.channel, result.rrc_reestablishment_pdu)
                    .await;
            }
            Err(rejection) if rejection.falls_back_to_setup() => {
                // §5.3.3.1: the network falls back to RRCSetup. The transport-level
                // ue_id is the only identity available for a UE whose stored
                // context could not be resolved, so the setup runs on that.
                //
                // Any context already on that ue_id is discarded first: §5.3.7.5
                // has the UE leave RRC_IDLE through establishment, so the network
                // starts a fresh context rather than reusing state it has just
                // declined to restore. `process_rrc_setup_request` refuses
                // outright when a context exists, so without this the fallback
                // would send nothing at all.
                if self.ue_manager.delete_ue(ue_id).is_some() {
                    debug!(
                        "Discarded the unverified context on UE[{}] before the \
                         RRCSetup fallback",
                        ue_id
                    );
                }
                if let Some(result) = self.connection_manager.process_rrc_setup_request(
                    &mut self.ue_manager,
                    ue_id,
                    0,
                    false,
                    // TS 38.413 §9.3.1.111 rrcCause: the UE is re-establishing an
                    // existing connection, so mo-Signalling is the honest cause.
                    3,
                ) {
                    info!(
                        "RRCSetup fallback for UE[{}] after {:?} (TS 38.331 §5.3.3.1)",
                        ue_id, rejection
                    );
                    self.send_rrc_message(result.ue_id, result.channel, result.rrc_setup_pdu)
                        .await;
                }
            }
            Err(rejection) => {
                warn!(
                    "RRC Reestablishment Request from UE[{}] answered with nothing: {:?}",
                    ue_id, rejection
                );
            }
        }
    }

    /// Records the AS security material a re-establishment is verified against
    /// (TS 38.331 §5.3.7.2), handed over by the NGAP task when it derives the AS
    /// keys at Initial Context Setup.
    ///
    /// A UE with no context yet is skipped rather than given a fresh one: the
    /// material belongs to a connection the RRC task already knows about, and
    /// inventing a context around it would be a context nothing established.
    fn handle_as_security_for_reestablishment(
        &mut self,
        ue_id: i32,
        security: ReestablishmentSecurity,
    ) {
        match self.ue_manager.try_find_ue_mut(ue_id) {
            Some(ctx) => {
                debug!(
                    "Recorded re-establishment security for UE[{}]: c_rnti={:#06x}, \
                     pci={}, ncc={}",
                    ue_id, security.c_rnti, security.phys_cell_id, security.next_hop_chaining_count
                );
                ctx.set_reestablishment_security(security);
            }
            None => warn!(
                "Dropping re-establishment security for UE[{}]: no RRC context",
                ue_id
            ),
        }
    }

    /// The 36-bit NR Cell Identity of this gNB's cell, the third input of the
    /// `VarShortMAC-Input` (TS 38.331 §5.3.7.4). `nci` is the 36-bit NR Cell
    /// Identity from the configuration.
    fn cell_identity(&self) -> u64 {
        self.task_base.config.nci & 0xF_FFFF_FFFF
    }

    async fn handle_rrc_reestablishment_complete(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        // Prefer the real UPER encoding; fall back to the bespoke framing's
        // second byte for a UE that has not been updated.
        let transaction_id = match decode_rrc_reestablishment_complete(bytes) {
            Ok(complete) => complete.rrc_transaction_id,
            Err(_) if bytes.len() >= 2 => bytes[1],
            Err(_) => 0,
        };

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

    /// Handles an `RRCResumeRequest` (UL-CCCH) or `RRCResumeRequest1` (UL-CCCH1).
    ///
    /// The request is **decoded**, not sampled: before issue #38 this read
    /// `bytes.get(1)` as a resume cause and ignored the `I-RNTI` and `resumeMAC-I`
    /// entirely, so the gNB fabricated a context for whatever asked. Now the I-RNTI
    /// selects a stored context and the `resumeMAC-I` authenticates it
    /// (TS 38.331 §5.3.13.3).
    ///
    /// Both message forms are accepted because which one the UE sends depends on
    /// SIB1's `useFullResumeID`, not on the network: `RRCResumeRequest1` carries the
    /// full 40-bit identity and is tried first, because a UL-CCCH1 message decoded as
    /// UL-CCCH would yield a plausible-looking but wrong short identity.
    async fn handle_rrc_resume_request(&mut self, ue_id: i32, data: &OctetString) {
        let bytes = data.data();
        let presented = if let Ok(full) = decode_rrc_resume_request1(bytes) {
            Some(ResumeRequestPresented {
                identity: SuspendedIdentity::Full(full.resume_identity),
                resume_mac_i: full.resume_mac_i,
                cause: full.resume_cause as u8,
                resuming_cell_identity: self.cell_identity(),
            })
        } else if let Ok(short) = decode_rrc_resume_request(bytes) {
            Some(ResumeRequestPresented {
                identity: SuspendedIdentity::Short(short.resume_identity),
                resume_mac_i: short.resume_mac_i,
                cause: short.resume_cause as u8,
                resuming_cell_identity: self.cell_identity(),
            })
        } else {
            None
        };

        let Some(presented) = presented else {
            // Not decodable at all. Refused rather than guessed at: the old
            // byte-sampling path is exactly what let an unauthenticated resume
            // through, and a UE whose request we cannot read has to fall back to
            // `RRCSetup` (§5.3.13.3).
            warn!(
                "Discarding an undecodable RRCResumeRequest from UE[{ue_id}] ({} bytes); \
                 the UE must fall back to RRCSetup",
                bytes.len()
            );
            return;
        };

        info!(
            "RRC Resume Request from UE[{ue_id}]: identity={:?}, cause={}",
            presented.identity, presented.cause
        );

        match self.connection_manager.process_rrc_resume_request(
            &mut self.ue_manager,
            ue_id,
            presented,
        ) {
            Ok(result) => {
                self.send_rrc_message(result.ue_id, result.channel, result.rrc_resume_pdu)
                    .await;
            }
            Err(rejection) => {
                // Nothing is sent. §5.3.13.3 leaves the UE to fall back to `RRCSetup`
                // on T319 expiry, which is the honest outcome: answering an
                // unauthenticated resume with anything at all is what this issue is
                // about, and no `RRCReject` can be sent on a connection that was never
                // resumed.
                warn!("RRC Resume for UE[{ue_id}] refused: {rejection}");
            }
        }
    }

    /// Suspends a UE to RRC_INACTIVE (TS 38.331 §5.3.8.3, issue #38).
    ///
    /// Falls back to a plain release when the UE cannot be suspended safely — see
    /// [`RrcConnectionManager::initiate_rrc_suspend`] for what "safely" excludes.
    /// The fallback matters: a UE told nothing would stay in RRC_CONNECTED talking to
    /// a gNB that had already moved on.
    async fn handle_suspend_ue(&mut self, ue_id: i32, params: SuspendParams) {
        let cell_identity = self.cell_identity();
        if let Some(result) = self.connection_manager.initiate_rrc_suspend(
            &mut self.ue_manager,
            ue_id,
            cell_identity,
            params,
        ) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_release_pdu)
                .await;
            return;
        }
        warn!("UE[{ue_id}] could not be suspended; releasing instead");
        self.handle_an_release(ue_id).await;
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
        if let Some(result) = self.connection_manager.initiate_rrc_release(
            &mut self.ue_manager,
            ue_id,
            release_cell_reselection_priorities(&self.task_base.config),
        ) {
            self.send_rrc_message(result.ue_id, result.channel, result.rrc_release_pdu)
                .await;
        }
        self.ue_manager.delete_ue(ue_id);
    }

    /// Handles `RrcMessage::Paging`: builds the TS 38.331 §5.3.2.2 `Paging`
    /// message from the NGAP-supplied 5G-S-TMSI and broadcasts it on PCCH.
    ///
    /// `ue_paging_tmsi` is the 48-bit 5G-S-TMSI serialised by the NGAP task
    /// (TS 23.003 §2.10.1: AMF Set ID + AMF Pointer packed into two octets,
    /// then the 32-bit 5G-TMSI). `tai_list_for_paging` is the set of served
    /// TAIs the NGAP layer already matched against this gNB's own TAI — it is
    /// what authorises the transmission, and it is not carried on the air
    /// interface: the RRC `Paging` message has no TAI member, because a UE
    /// reading PCCH is by definition in the cell.
    ///
    /// PAGING OCCASION (issue #99): the PDU is transmitted at the paged UE's
    /// paging frame, derived per TS 38.304 §7.1 from its 5G-S-TMSI and the DRX
    /// cycle. `drx_cycle_frames` is the NGAP `(default)PagingDRX` when the AMF
    /// signalled one, otherwise the cell's configured default paging cycle.
    ///
    /// The wait is up to `T` radio frames (2.56 s at rf256), which is realistic
    /// MT latency and is why this changes the timing of anything that pages. It is
    /// spawned rather than awaited inline, because holding the RRC task for up to
    /// 2.5 s would stall every other UE's signalling behind one paging.
    ///
    /// Public because it is a real message-handler entry point also driven
    /// directly by the in-process strict-peer harness
    /// (`tests/src/paging_mt_service_request.rs`).
    pub async fn handle_paging(
        &mut self,
        ue_paging_tmsi: Vec<u8>,
        tai_list_for_paging: Vec<u8>,
        drx_cycle_frames: Option<u16>,
    ) {
        let s_tmsi: [u8; FIVE_G_S_TMSI_LEN] = match ue_paging_tmsi.as_slice().try_into() {
            Ok(tmsi) => tmsi,
            Err(_) => {
                warn!(
                    "Paging dropped: 5G-S-TMSI is {} octets, expected {}",
                    ue_paging_tmsi.len(),
                    FIVE_G_S_TMSI_LEN
                );
                return;
            }
        };

        let pdu = match encode_paging(&[PagingRecordParams::five_g_s_tmsi(s_tmsi)]) {
            Ok(pdu) => pdu,
            Err(e) => {
                error!("Failed to encode PCCH Paging: {e}");
                return;
            }
        };

        let schedule = self.schedule_paging(&s_tmsi, drx_cycle_frames, frame_clock::current_sfn());

        info!(
            "PCCH Paging for 5G-S-TMSI {:02x?} ({} served TAI(s) matched): UE_ID {}, \
             paging frame SFN {} (T={}), waiting {} ms",
            s_tmsi,
            tai_list_for_paging.len() / TAI_OCTETS,
            schedule.ue_id,
            schedule.paging_frame,
            schedule.t,
            schedule.delay.as_millis()
        );

        if schedule.delay.is_zero() {
            // Already in the occasion: transmit now rather than spawning a task
            // to sleep for nothing.
            self.broadcast_rrc_message(RrcChannel::Pcch, OctetString::from_slice(&pdu))
                .await;
            return;
        }

        // Deferred on its own task so the RRC task keeps serving other UEs. The
        // handle is a channel clone, so a paging in flight at shutdown simply
        // fails to send rather than blocking the shutdown.
        let rls_tx = self.task_base.rls_tx.clone();
        let delay = schedule.delay;
        let target_frame = schedule.paging_frame;
        tokio::spawn(async move {
            tokio::time::sleep(delay).await;
            let msg = RlsMessage::BroadcastRrc {
                rrc_channel: RrcChannel::Pcch,
                pdu_id: 0,
                data: OctetString::from_slice(&pdu),
            };
            if let Err(e) = rls_tx.send(msg).await {
                error!("Failed to broadcast deferred PCCH Paging: {e}");
            } else {
                debug!(
                    "PCCH Paging transmitted at paging frame SFN {target_frame} \
                     (now SFN {})",
                    frame_clock::current_sfn()
                );
            }
        });
    }

    /// Work out when a paged UE's next paging frame falls (TS 38.304 §7.1).
    ///
    /// Separated from the transmission so the decision is testable without a
    /// clock-dependent send: everything here is a pure function of the 5G-S-TMSI,
    /// the DRX cycle and the current SFN.
    /// `current_sfn` is a parameter rather than read here so the whole mapping --
    /// identity and DRX cycle to paging frame and delay -- is deterministically
    /// testable. Reading the clock inside would make every assertion about it
    /// depend on when the test ran.
    pub fn schedule_paging(
        &self,
        s_tmsi: &[u8; FIVE_G_S_TMSI_LEN],
        drx_cycle_frames: Option<u16>,
        current_sfn: u16,
    ) -> PagingSchedule {
        let t = drx_cycle_frames.unwrap_or(self.task_base.config.paging_default_cycle_frames);
        // An unusable T falls back to the cell default rather than dropping the
        // paging: a UE that is not paged at all is worse off than one paged on a
        // cycle the AMF did not ask for, and the mismatch is logged.
        let config = PagingCycleConfig::with_default_spreading(t).unwrap_or_else(|e| {
            warn!(
                "Paging DRX cycle {t} unusable ({e}); falling back to {} frames",
                DEFAULT_PAGING_CYCLE_FRAMES
            );
            PagingCycleConfig::with_default_spreading(DEFAULT_PAGING_CYCLE_FRAMES)
                .expect("the built-in default paging cycle must be valid")
        });

        let ue_id = ue_id_from_s_tmsi(s_tmsi);
        let occasion = paging_occasion(ue_id, &config);
        let paging_frame = occasion.next_paging_frame_at_or_after(current_sfn);
        // Frames ahead, not a clock difference: the delay has to follow from the
        // SFN the caller passed, or the returned schedule would describe a
        // different instant from the frame it names.
        let frames_ahead = u64::from(
            (u32::from(paging_frame) + u32::from(paging_occasion::SFN_CYCLE)
                - u32::from(current_sfn))
                % u32::from(paging_occasion::SFN_CYCLE),
        );

        PagingSchedule {
            ue_id,
            t: config.t(),
            paging_frame,
            delay: std::time::Duration::from_millis(frames_ahead * frame_clock::RADIO_FRAME_MS),
        }
    }

    // ====================================================================
    // SRB PDCP security inputs (TS 38.323 §5.8/§5.9, issue #31)
    //
    // These four values must match the UE's `SRB1_BEARER`, `DIRECTION_*` and
    // `SMC_PDCP_COUNT` exactly. A mismatch fails every MAC with no other symptom,
    // which is why they are named on both sides rather than inlined.
    // ====================================================================

    /// Integrity-protects a SecurityModeCommand (TS 38.331 §5.3.4.2, issue #31).
    ///
    /// Integrity only and COUNT 0: the command is not ciphered, because the UE has
    /// not yet confirmed it can decipher anything, and its COUNT is the first of
    /// the downlink sequence (`SMC_PDCP_COUNT` on the UE side — both ends must use
    /// the same value or the MAC fails with no other symptom).
    ///
    /// With AS security disabled, or before the keys arrive, the command goes out
    /// unprotected — which is the pre-#31 behaviour and what a UE with the switch
    /// off expects.
    fn protect_security_mode_command(&self, ue_id: i32, pdu: OctetString) -> OctetString {
        if !self.task_base.config.as_security_enabled {
            return pdu;
        }
        let Some(keys) = self
            .ue_manager
            .try_find_ue(ue_id)
            .and_then(|ctx| ctx.reestablishment_security.as_ref())
        else {
            warn!(
                "Sending SecurityModeCommand to UE[{ue_id}] UNPROTECTED: no AS \
                 security keys yet, so the UE cannot verify it"
            );
            return pdu;
        };
        // Integrity-only, so the ciphering key is irrelevant and the ciphering
        // identity is NEA0. Built explicitly rather than reusing the UE's full
        // state, because the command must NOT be ciphered (TS 38.331 §5.3.4.2).
        let Ok(integrity_only) =
            SrbSecurity::new([0u8; 16], keys.k_rrc_int, 0, keys.integrity_alg_id)
        else {
            warn!(
                "Sending SecurityModeCommand to UE[{ue_id}] UNPROTECTED: unusable \
                 integrity algorithm identity {}",
                keys.integrity_alg_id
            );
            return pdu;
        };
        let mac = integrity_only.compute_mac_i(
            GNB_SMC_PDCP_COUNT,
            SRB1_PDCP_BEARER,
            PDCP_DIRECTION_DOWNLINK,
            pdu.data(),
        );
        let mut out = pdu.data().to_vec();
        out.extend_from_slice(&mac);
        OctetString::from_slice(&out)
    }

    /// PDCP-protects a DL-DCCH PDU when AS security is active for the UE.
    ///
    /// Returns `None` only when the PDU must be DROPPED — an unusable security
    /// state for a UE whose security is active. Any other case returns the PDU,
    /// protected or not, so the pre-#31 path is byte-for-byte unchanged with the
    /// switch off.
    fn protect_dl_dcch(
        &mut self,
        ue_id: i32,
        channel: RrcChannel,
        data: OctetString,
    ) -> Option<OctetString> {
        if !self.task_base.config.as_security_enabled || channel != RrcChannel::DlDcch {
            return Some(data);
        }
        let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) else {
            return Some(data);
        };
        if !ctx.as_security_active() {
            // Not yet activated: the SecurityModeCommand itself is handled by
            // `protect_security_mode_command`, and everything before activation
            // legitimately goes in the clear.
            return Some(data);
        }
        let Some(sec) = ctx.srb_security() else {
            error!(
                "Dropping DL-DCCH PDU for UE[{ue_id}]: AS security is active but the \
                 security state is unusable"
            );
            return None;
        };
        let count = ctx.next_dl_pdcp_count();
        Some(OctetString::from_slice(&sec.protect(
            count,
            SRB1_PDCP_BEARER,
            PDCP_DIRECTION_DOWNLINK,
            data.data(),
        )))
    }

    /// PDCP-unprotects an UL-DCCH PDU when AS security is active for the UE.
    ///
    /// Returns `None` when the PDU must be **discarded**: a failed integrity check
    /// (TS 33.501 §6.5) or an unusable security state. The plaintext is never
    /// surfaced for a PDU that failed, so a caller cannot dispatch on it.
    ///
    /// This runs BEFORE dispatch, which it has to: the uplink dispatcher routes on
    /// `bytes[0] & 0x0F` of the PDU, and a ciphered PDU has no meaningful leading
    /// nibble. That ordering is also why `as_security_enabled` defaults to `false`
    /// — see the config field's own note.
    fn unprotect_ul_dcch(&mut self, ue_id: i32, data: &OctetString) -> Option<OctetString> {
        if !self.task_base.config.as_security_enabled {
            return Some(data.clone());
        }
        let Some(ctx) = self.ue_manager.try_find_ue_mut(ue_id) else {
            return Some(data.clone());
        };
        if !ctx.as_security_active() {
            return Some(data.clone());
        }
        let Some(sec) = ctx.srb_security() else {
            error!(
                "Discarding UL-DCCH PDU from UE[{ue_id}]: AS security is active but \
                 the security state is unusable"
            );
            return None;
        };
        let count = ctx.next_ul_pdcp_count();
        match sec.unprotect(count, SRB1_PDCP_BEARER, PDCP_DIRECTION_UPLINK, data.data()) {
            Ok(plain) => Some(OctetString::from_slice(&plain)),
            Err(e) => {
                warn!(
                    "Discarding UL-DCCH PDU from UE[{ue_id}]: integrity check failed \
                     ({e}) -- TS 33.501 §6.5 discards it rather than acting on it"
                );
                None
            }
        }
    }

    async fn send_rrc_message(&mut self, ue_id: i32, channel: RrcChannel, data: OctetString) {
        // SRB PDCP protection on DL-DCCH once AS security is active for this UE
        // (TS 38.323 §5.8/§5.9, issue #31). Applied here because this is the single
        // downlink send point, so nothing can bypass it by calling a sibling.
        let data = match self.protect_dl_dcch(ue_id, channel, data) {
            Some(protected) => protected,
            // The context vanished or the security state is unusable. Dropping is
            // the only safe answer: sending the PDU in the clear to a UE that is
            // deciphering would have it read plaintext as ciphertext, and it is a
            // security regression besides.
            None => return,
        };
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

    /// Broadcasts the cell's system information on BCCH (TS 38.331 §5.2.1): the
    /// MIB on BCCH-BCH every call, SIB1 on BCCH-DL-SCH when `with_sib1`.
    ///
    /// Before this the MIB and SIB1 encoders existed with no caller anywhere, so
    /// a UE never saw the cell's real PLMN, TAC or identity — it assumed them
    /// (`provide_simulated_system_info` on the UE side).
    async fn broadcast_system_information(&mut self, with_sib1: bool) {
        match encode_cell_mib() {
            Ok(mib) => {
                self.broadcast_rrc_message(RrcChannel::BcchBch, OctetString::from_slice(&mib))
                    .await;
            }
            Err(e) => error!("Failed to encode MIB: {e}"),
        }
        if !with_sib1 {
            return;
        }
        match encode_cell_sib1(&self.task_base.config) {
            Ok(sib1) => {
                self.broadcast_rrc_message(RrcChannel::BcchDlSch, OctetString::from_slice(&sib1))
                    .await;
            }
            Err(e) => error!("Failed to encode SIB1: {e}"),
        }
        // SIB2/SIB3/SIB4 (issue #50), on the SAME channel as SIB1 but the other
        // arm of the BCCH-DL-SCH CHOICE. Scheduled with SIB1 rather than on their
        // own cadence: TS 38.331 §5.2.1 lets a cell choose, and a UE that has
        // just read SIB1 is exactly the UE that needs the reselection parameters.
        //
        // `None` means the operator turned the broadcast off, which is not an
        // error -- the UE falls back to its constants and logs that it did.
        if let Some(result) = encode_cell_system_information(&self.task_base.config) {
            match result {
                Ok(si) => {
                    self.broadcast_rrc_message(RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                        .await;
                }
                Err(e) => error!("Failed to encode SIB2/3/4 SystemInformation: {e}"),
            }
        }
    }

    /// Broadcasts an RRC PDU on a downlink common channel (PCCH paging).
    ///
    /// `pdu_id` is 0: a broadcast has no addressee to acknowledge it, and a
    /// non-zero PDU ID would make every receiving UE queue an ack for a PDU the
    /// gNB is not tracking.
    async fn broadcast_rrc_message(&mut self, channel: RrcChannel, data: OctetString) {
        let msg = RlsMessage::BroadcastRrc {
            rrc_channel: channel,
            pdu_id: 0,
            data,
        };
        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to broadcast RRC message to RLS: {}", e);
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

    /// Act on an NWDAF handover recommendation (issue #39).
    ///
    /// This used to validate the UE, check the confidence, log
    /// *"Initiating NWDAF-recommended handover"* and **return** — never touching the
    /// `GnbHandoverManager` and sending nothing. The log claimed a handover the code did
    /// not perform, which is worse than not having the feature.
    ///
    /// Now it does one of two real things, decided from this cell's own identity:
    ///
    /// - the recommended cell **is** this cell's neighbour on this gNB → an intra-gNB
    ///   handover: `GnbHandoverManager::initiate_handover` and an `RRCReconfiguration`
    ///   with `reconfigurationWithSync` (TS 38.331 §5.3.5.5.2);
    /// - the recommended cell is **not** one of ours → an inter-gNB handover:
    ///   `NgapMessage::InitiateHandover`, which sends HANDOVER REQUIRED
    ///   (TS 38.413 §8.4.1.1).
    ///
    /// The two are alternatives. Doing both would hand the UE over twice.
    async fn handle_nwdaf_handover(&mut self, ue_id: i32, target_cell: i32, confidence: f32) {
        if self.ue_manager.try_find_ue(ue_id).is_none() {
            debug!(
                "RRC: Ignoring NWDAF handover recommendation for unknown UE {}",
                ue_id
            );
            return;
        }
        // A confidence threshold of 0.7 is used to avoid spurious handovers.
        if confidence < NWDAF_HANDOVER_CONFIDENCE_THRESHOLD {
            debug!(
                "RRC: Ignoring low-confidence ({:.2}) handover recommendation for UE {} to cell {}",
                confidence, ue_id, target_cell
            );
            return;
        }
        let own_cell = (self.task_base.config.nci & 0xF_FFFF_FFFF) as i32;
        if target_cell == own_cell {
            debug!(
                "RRC: Ignoring NWDAF recommendation for UE {ue_id} to cell {target_cell}: \
                 it is already the serving cell"
            );
            return;
        }
        info!(
            "RRC: NWDAF-recommended handover for UE {ue_id} to cell {target_cell} \
             (confidence {confidence:.2})"
        );
        self.execute_handover(ue_id, target_cell).await;
    }

    /// Hand a UE over to `target_cell`, intra-gNB or inter-gNB (issue #39).
    ///
    /// Shared by the NWDAF path and anything else that decides a UE should move, so the
    /// intra/inter choice is made in one place.
    async fn execute_handover(&mut self, ue_id: i32, target_cell: i32) {
        let own_cell = (self.task_base.config.nci & 0xF_FFFF_FFFF) as i32;

        // Intra-gNB when the target is a cell this gNB configured as its secondary;
        // otherwise the target belongs to another node and the AMF has to be involved.
        // A single-cell gNB therefore always takes the inter-gNB path, which is the truth
        // rather than a fallback.
        let is_own_cell = self
            .task_base
            .config
            .scell_phys_cell_id
            .is_some_and(|pci| i32::from(pci) == target_cell)
            || target_cell == own_cell;

        if is_own_cell {
            let Some(command) =
                self.handover_manager
                    .initiate_handover(ue_id, own_cell, target_cell)
            else {
                warn!("RRC: the handover manager declined to prepare UE {ue_id}");
                return;
            };
            let Some(pdu) = command.build_rrc_pdu() else {
                warn!("RRC: could not encode the handover command for UE {ue_id}");
                return;
            };
            // The RRCReconfiguration with reconfigurationWithSync -- what actually moves
            // the UE (TS 38.331 §5.3.5.5.2). The old code emitted nothing at all.
            self.send_rrc_message(ue_id, RrcChannel::DlDcch, OctetString::from_slice(&pdu))
                .await;
            self.handover_manager.mark_executing(ue_id);
            info!("RRC: sent an intra-gNB handover command to UE {ue_id} for cell {target_cell}");
            return;
        }

        // Inter-gNB: HANDOVER REQUIRED through NGAP. The UE's capability container comes
        // from this context because the NGAP task does not hold it, and the target needs
        // it to configure the UE at all.
        let ue_nr_capability = self
            .ue_manager
            .try_find_ue(ue_id)
            .and_then(|ctx| ctx.nr_capability.clone());
        let msg = NgapMessage::InitiateHandover(Box::new(HandoverInitiation {
            ue_id,
            // The simulator has no inter-gNB topology, so the recommended cell's own
            // identity stands in for the target node's. Stated rather than hidden: an
            // AMF routes on the gNB ID, and with no neighbour table this is the only
            // value this gNB has.
            target_gnb_id: target_cell as u32,
            target_tac: self.task_base.config.tac,
            target_cell_identity: target_cell as u64,
            ue_nr_capability,
            // Time on the source cell is not tracked per UE, so 0 is reported: "no
            // measured dwell time" rather than an invented one.
            time_in_source_cell_s: 0,
        }));
        if let Err(e) = self.task_base.ngap_tx.send(msg).await {
            error!("RRC: failed to ask NGAP to start the handover: {e}");
        }
    }
}

#[async_trait::async_trait]
impl Task for RrcTask {
    type Message = RrcMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RRC task started");

        // System information broadcast (TS 38.331 §5.2.1): the MIB every period,
        // SIB1 every second period — the spec's default 80 ms / 160 ms cadence.
        // A configured period of 0 disables it, and the interval below is then a
        // parked timer whose arm returns immediately.
        let si_period_ms = self.task_base.config.si_broadcast_period_ms;
        let mut si_timer = interval(Duration::from_millis(if si_period_ms == 0 {
            SI_PARKED_PERIOD_MS
        } else {
            si_period_ms
        }));
        // `interval` fires its first tick immediately; broadcasting before any UE
        // has been discovered has nothing to reach, and consuming it keeps the
        // cadence at multiples of the period.
        si_timer.tick().await;
        let mut si_ticks: u64 = 0;

        loop {
            tokio::select! {
                _ = si_timer.tick() => {
                    if si_period_ms > 0 {
                        si_ticks = si_ticks.wrapping_add(1);
                        self.broadcast_system_information(si_ticks % 2 == 1).await;
                    }
                }
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
                                // on SRB1 (DL-DCCH), integrity protected and NOT
                                // ciphered (§5.3.4.2) so the UE can verify it with
                                // the keys it is about to derive (issue #31).
                                info!(
                                    "Sending RRC SecurityModeCommand to UE {} ({} bytes)",
                                    ue_id,
                                    pdu.len()
                                );
                                let pdu = self.protect_security_mode_command(ue_id, pdu);
                                self.send_rrc_message(ue_id, RrcChannel::DlDcch, pdu).await;
                            }
                            RrcMessage::AsSecurityForReestablishment {
                                ue_id, k_rrc_int, k_rrc_enc, integrity_alg_id,
                                ciphering_alg_id, c_rnti, phys_cell_id,
                                next_hop_chaining_count,
                            } => {
                                self.handle_as_security_for_reestablishment(
                                    ue_id,
                                    ReestablishmentSecurity {
                                        k_rrc_int,
                                        k_rrc_enc,
                                        integrity_alg_id,
                                        ciphering_alg_id,
                                        c_rnti,
                                        phys_cell_id,
                                        next_hop_chaining_count,
                                    },
                                );
                            }
                            RrcMessage::RrcReconfiguration { ue_id, pdu } => {
                                // TS 33.501 §6.5 / §6.6.1: a DRB carries user data,
                                // and user data must not flow before AS security is
                                // activated. Gated here rather than at the NGAP
                                // plane, because this is the point where the
                                // configuration reaches the air (issue #31,
                                // criterion 5).
                                //
                                // Only gated when the switch is on: with it off no
                                // UE ever activates AS security, so gating would
                                // block every DRB and take the data plane down.
                                if self.task_base.config.as_security_enabled
                                    && !self
                                        .ue_manager
                                        .try_find_ue(ue_id)
                                        .is_some_and(|c| c.as_security_active())
                                {
                                    warn!(
                                        "Refusing to establish a DRB for UE {ue_id}: AS \
                                         security is not activated, so user data would \
                                         cross the radio unprotected (TS 33.501 §6.6.1)"
                                    );
                                    continue;
                                }
                                // TS 38.331 §5.3.5.6: deliver the RRCReconfiguration
                                // (DRB setup) on SRB1 (DL-DCCH).
                                info!(
                                    "Sending RRC Reconfiguration to UE {} ({} bytes)",
                                    ue_id,
                                    pdu.len()
                                );
                                self.send_rrc_message(ue_id, RrcChannel::DlDcch, pdu).await;
                                // §5.3.5.5.9: a secondary cell is added by a
                                // reconfiguration too, and only makes sense once
                                // the UE has a DRB-bearing configuration to add
                                // it to. No-op unless one is configured.
                                self.send_scell_configuration(ue_id).await;
                            }
                            RrcMessage::AnRelease { ue_id } => {
                                self.handle_an_release(ue_id).await;
                            }
                            RrcMessage::SuspendUe { ue_id, t380_minutes } => {
                                self.handle_suspend_ue(
                                    ue_id,
                                    SuspendParams {
                                        t380_minutes,
                                        ..SuspendParams::default()
                                    },
                                )
                                .await;
                            }
                            RrcMessage::Paging { ue_paging_tmsi, tai_list_for_paging, drx_cycle_frames } => {
                                self.handle_paging(ue_paging_tmsi, tai_list_for_paging, drx_cycle_frames)
                                    .await;
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

    // ========================================================================
    // NWDAF-driven handover (issue #39)
    // ========================================================================

    /// A gNB RRC task with a connected UE and every receiver alive.
    #[allow(clippy::type_complexity)]
    fn rrc_task_with_connected_ue(
        config: GnbConfig,
        ue_id: i32,
    ) -> (
        RrcTask,
        tokio::sync::mpsc::Receiver<TaskMessage<NgapMessage>>,
        tokio::sync::mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) {
        let (task_base, _app_rx, ngap_rx, _rrc_rx, _gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 32);
        let mut task = RrcTask::new(task_base);
        task.handle_radio_power_on();
        let ctx = task.ue_manager.create_ue(ue_id);
        ctx.on_setup_request();
        ctx.on_setup_sent();
        ctx.on_setup_complete();
        (task, ngap_rx, rls_rx)
    }

    /// #39, criterion 6: an NWDAF recommendation above the confidence threshold results
    /// in a real `RRCReconfiguration` with `reconfigurationWithSync` — not a log line.
    ///
    /// The old handler validated the UE, checked the confidence, logged
    /// *"Initiating NWDAF-recommended handover"* and returned.
    #[test]
    fn a_confident_nwdaf_recommendation_emits_a_reconfiguration_with_sync() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::decode_handover_command;

        let mut config = test_config();
        // A configured secondary cell makes the recommended target one of ours, so this is
        // the intra-gNB arm. The inter-gNB arm is covered separately.
        config.scell_phys_cell_id = Some(42);
        let (mut task, _ngap_rx, mut rls_rx) = rrc_task_with_connected_ue(config, 1);

        run_async(async {
            task.handle_nwdaf_handover(1, 42, 0.95).await;
        });

        let pdu = loop {
            match rls_rx.try_recv() {
                Ok(TaskMessage::Message(RlsMessage::DownlinkRrc { data, .. })) => break data,
                Ok(_) => continue,
                Err(e) => panic!("no RRCReconfiguration was emitted: {e}"),
            }
        };
        let command = decode_handover_command(pdu.data())
            .expect("the emitted PDU must be a real handover command");
        assert_eq!(
            command.target_phys_cell_id, 42,
            "and it must name the recommended cell"
        );
        assert!(
            command.master_key_update.is_some(),
            "with a masterKeyUpdate, or the UE keeps the source cell's keys \
             (TS 33.501 §6.9.2.3.1)"
        );
    }

    /// The negative controls: below the threshold, for an unknown UE, and for the serving
    /// cell itself, nothing is emitted.
    ///
    /// Without these, a handler that fired unconditionally would pass the test above.
    #[test]
    fn a_recommendation_that_should_be_ignored_emits_nothing() {
        let mut config = test_config();
        config.scell_phys_cell_id = Some(42);
        let own_cell = (config.nci & 0xF_FFFF_FFFF) as i32;
        let (mut task, mut ngap_rx, mut rls_rx) = rrc_task_with_connected_ue(config, 1);

        run_async(async {
            // Below the 0.7 threshold.
            task.handle_nwdaf_handover(1, 42, 0.69).await;
            // Unknown UE.
            task.handle_nwdaf_handover(99, 42, 0.99).await;
            // Already the serving cell.
            task.handle_nwdaf_handover(1, own_cell, 0.99).await;
        });

        assert!(
            !matches!(
                rls_rx.try_recv(),
                Ok(TaskMessage::Message(RlsMessage::DownlinkRrc { .. }))
            ),
            "no handover command must be emitted for a recommendation that should be ignored"
        );
        assert!(
            ngap_rx.try_recv().is_err(),
            "and nothing must be asked of NGAP either"
        );
    }

    /// #39, criterion 3: a recommendation naming a cell this gNB does **not** serve takes
    /// the inter-gNB arm and asks NGAP to start the handover.
    #[test]
    fn a_recommendation_for_another_nodes_cell_asks_ngap_to_hand_over() {
        let config = test_config();
        // No secondary cell configured, so cell 777 belongs to another node.
        let (mut task, mut ngap_rx, _rls_rx) = rrc_task_with_connected_ue(config, 1);

        run_async(async {
            task.handle_nwdaf_handover(1, 777, 0.9).await;
        });

        let request = loop {
            match ngap_rx.try_recv() {
                Ok(TaskMessage::Message(NgapMessage::InitiateHandover(r))) => break r,
                Ok(_) => continue,
                Err(e) => panic!("NGAP was not asked to start a handover: {e}"),
            }
        };
        assert_eq!(request.ue_id, 1);
        assert_eq!(
            request.target_cell_identity, 777,
            "the target the NWDAF named must reach NGAP unchanged"
        );
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

    // ========================================================================
    // System information broadcast (#21, TS 38.331 §5.2.1)
    // ========================================================================

    /// The MIB goes out on BCCH-BCH on every period and decodes back to this
    /// cell's values.
    #[test]
    fn the_mib_is_broadcast_on_bcch_bch() {
        use nextgsim_rrc::procedures::system_information::decode_mib;

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.broadcast_system_information(false).await;
        });

        let (channel, _, pdu) = try_take_broadcast(&mut rls_rx).expect("a MIB broadcast");
        assert_eq!(channel, RrcChannel::BcchBch);
        decode_mib(pdu.data()).expect("the broadcast MIB must decode");
        assert!(
            try_take_broadcast(&mut rls_rx).is_none(),
            "SIB1 is only sent on every second period"
        );
    }

    /// SIB1 goes out on BCCH-DL-SCH and carries the configured PLMN, TAC and NCI —
    /// the values a UE would otherwise have to assume.
    #[test]
    fn sib1_is_broadcast_on_bcch_dl_sch_with_this_cells_identity() {
        use nextgsim_rrc::procedures::system_information::decode_sib1;

        let config = test_config();
        let (nci, tac) = (config.nci, config.tac);
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.broadcast_system_information(true).await;
        });

        let (mib_channel, _, _) = try_take_broadcast(&mut rls_rx).expect("the MIB comes first");
        assert_eq!(mib_channel, RrcChannel::BcchBch);

        let (channel, pdu_id, pdu) = try_take_broadcast(&mut rls_rx).expect("a SIB1 broadcast");
        assert_eq!(channel, RrcChannel::BcchDlSch);
        assert_eq!(pdu_id, 0, "broadcast PDUs are not acknowledged per UE");

        let sib1 = decode_sib1(pdu.data()).expect("the broadcast SIB1 must decode");
        let info = &sib1.plmn_identity_info_list[0];
        assert_eq!(info.cell_identity, nci);
        assert_eq!(info.tracking_area_code, Some(tac));
    }

    // ========================================================================
    // Paging (#35): NGAP Paging -> PCCH broadcast
    // ========================================================================

    /// 5G-S-TMSI as the NGAP task serialises it: AMF Set ID 0x155 and AMF
    /// Pointer 0x2A packed into 0x556A, then the 32-bit 5G-TMSI.
    const PAGED_S_TMSI: [u8; 6] = [0x55, 0x6A, 0xDE, 0xAD, 0xBE, 0xEF];

    /// One served TAI serialised as `RrcMessage::Paging` carries it: 3-octet
    /// PLMN identity followed by a 3-octet TAC.
    fn served_tai_list() -> Vec<u8> {
        vec![0x00, 0xF1, 0x10, 0x00, 0x00, 0x01]
    }

    /// Pops the next broadcast RRC PDU the gNB handed to its RLS.
    fn try_take_broadcast(
        rls_rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) -> Option<(RrcChannel, u32, OctetString)> {
        while let Ok(msg) = rls_rx.try_recv() {
            if let TaskMessage::Message(RlsMessage::BroadcastRrc {
                rrc_channel,
                pdu_id,
                data,
            }) = msg
            {
                return Some((rrc_channel, pdu_id, data));
            }
        }
        None
    }

    /// Waits up to `budget` for a broadcast to reach RLS.
    ///
    /// Needed because issue #99 made the PCCH Paging transmission DEFERRED to the
    /// paged UE's paging frame, on its own task -- so a `try_recv` immediately
    /// after `handle_paging` is racing the occasion by design.
    async fn await_broadcast(
        rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
        budget: Duration,
    ) -> Option<(RrcChannel, u32, OctetString)> {
        let deadline = tokio::time::Instant::now() + budget;
        loop {
            if let Some(found) = try_take_broadcast(rx) {
                return Some(found);
            }
            if tokio::time::Instant::now() >= deadline {
                return None;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    /// CRITERION 4 (issue #99): the transmission is scheduled at the paged UE's
    /// paging frame, not at whatever frame the paging arrived in.
    ///
    /// Asserted on the pure `schedule_paging` with a SYNTHETIC SFN, so it does not
    /// depend on when the test ran. The awaiting broadcast test below cannot make
    /// this claim: it passes just as well against a gNB that transmits
    /// immediately.
    #[test]
    fn the_paging_transmission_is_scheduled_at_the_paged_ues_own_frame() {
        use nextgsim_rrc::procedures::paging_occasion::{
            paging_occasion, ue_id_from_s_tmsi, PagingCycleConfig,
        };

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let task = RrcTask::new(task_base);

        let cycle = PagingCycleConfig::with_default_spreading(32).expect("valid cycle");
        let occasion = paging_occasion(ue_id_from_s_tmsi(&PAGED_S_TMSI), &cycle);
        let pf = occasion.pf_residue();

        // From the UE's own frame: transmit now.
        let at_occasion = task.schedule_paging(&PAGED_S_TMSI, Some(32), pf);
        assert_eq!(at_occasion.paging_frame, pf);
        assert!(at_occasion.delay.is_zero(), "already at the occasion");

        // From one frame past it: the next occasion is a full cycle away, and the
        // delay must be that many frames of 10 ms -- not zero.
        let one_past = (pf + 1) % 1024;
        let deferred = task.schedule_paging(&PAGED_S_TMSI, Some(32), one_past);
        assert_ne!(
            deferred.paging_frame, one_past,
            "the paging frame must be the UE's, not the frame the paging arrived in"
        );
        assert!(occasion.is_paging_frame(deferred.paging_frame));
        assert_eq!(
            deferred.delay,
            Duration::from_millis(31 * 10),
            "31 frames at 10 ms each"
        );
        assert_eq!(deferred.t, 32, "the DRX cycle the AMF asked for");
        assert_eq!(deferred.ue_id, ue_id_from_s_tmsi(&PAGED_S_TMSI));
    }

    /// An unusable DRX cycle falls back to the cell default rather than dropping
    /// the paging: a UE not paged at all is worse off than one paged on a cycle
    /// the AMF did not ask for.
    #[test]
    fn an_invalid_drx_cycle_falls_back_to_the_cell_default() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let task = RrcTask::new(task_base);

        // 100 frames is not a `defaultPagingCycle` value.
        let schedule = task.schedule_paging(&PAGED_S_TMSI, Some(100), 0);
        assert_eq!(schedule.t, DEFAULT_PAGING_CYCLE_FRAMES);
    }

    #[test]
    fn a_paged_5g_s_tmsi_is_broadcast_as_a_pcch_paging_record() {
        use nextgsim_rrc::procedures::paging::{decode_paging, PagedUeIdentity};

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        // ISSUE #99 CHANGED THE TIMING: the PCCH Paging now goes out at the UE's
        // paging frame, so the broadcast can be up to T radio frames away. rf32
        // bounds that at 320 ms.
        let (channel, pdu_id, pdu) = run_async(async {
            let schedule =
                task.schedule_paging(&PAGED_S_TMSI, Some(32), frame_clock::current_sfn());
            task.handle_paging(PAGED_S_TMSI.to_vec(), served_tai_list(), Some(32))
                .await;
            await_broadcast(&mut rls_rx, schedule.delay + Duration::from_millis(200)).await
        })
        .expect("paging must be broadcast to RLS at the UE's paging frame");
        assert_eq!(channel, RrcChannel::Pcch, "paging goes out on PCCH");
        assert_eq!(pdu_id, 0, "a broadcast is not per-UE acknowledged");

        let records = decode_paging(pdu.data()).expect("the broadcast PDU must be a PCCH Paging");
        assert_eq!(records.len(), 1);
        assert_eq!(
            records[0].ue_identity,
            PagedUeIdentity::FiveGSTmsi(PAGED_S_TMSI),
            "the record must carry the 5G-S-TMSI the AMF paged"
        );
        assert!(
            !records[0].non_3gpp_access,
            "3GPP-access paging must not set accessType"
        );
    }

    /// A 5G-S-TMSI of the wrong length cannot be encoded into the 48-bit
    /// `NG-5G-S-TMSI` BIT STRING, and zero-padding one would page a different
    /// UE. Nothing is transmitted.
    #[test]
    fn a_malformed_5g_s_tmsi_broadcasts_nothing() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_paging(vec![0xDE, 0xAD, 0xBE], served_tai_list(), None)
                .await;
        });

        assert!(
            try_take_broadcast(&mut rls_rx).is_none(),
            "a short 5G-S-TMSI must not be padded into a paging record"
        );
    }

    // ========================================================================
    // Secondary cell configuration (issue #112, TS 38.331 §5.3.5.5.9)
    // ========================================================================

    /// Pops the next dedicated downlink RRC PDU for `ue_id`, if any.
    fn try_take_downlink_rrc(
        rls_rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
        ue_id: i32,
    ) -> Option<(RrcChannel, OctetString)> {
        while let Ok(msg) = rls_rx.try_recv() {
            if let TaskMessage::Message(RlsMessage::DownlinkRrc {
                ue_id: id,
                rrc_channel,
                data,
                ..
            }) = msg
            {
                if id == ue_id {
                    return Some((rrc_channel, data));
                }
            }
        }
        None
    }

    /// With no `scell_phys_cell_id` configured, nothing is transmitted: the
    /// pre-#112 behaviour, byte for byte.
    #[test]
    fn no_scell_configuration_is_sent_by_default() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async { task.send_scell_configuration(1).await });

        assert!(
            try_take_downlink_rrc(&mut rls_rx, 1).is_none(),
            "scell_phys_cell_id defaults to None, so no SCell is added"
        );
    }

    /// A configured SCell is sent as `[0x0E][tid][UPER CellGroupConfig]`, and the
    /// container decodes back to the configured `physCellId` — the two ends have
    /// to agree on the bytes, which is what makes a real UPER container worth
    /// having.
    #[test]
    fn a_configured_scell_is_sent_once_as_uper() {
        use nextgsim_rrc::procedures::scell_config::decode_scell_config;

        let mut config = test_config();
        config.scell_phys_cell_id = Some(42);
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async { task.send_scell_configuration(7).await });

        let (channel, pdu) =
            try_take_downlink_rrc(&mut rls_rx, 7).expect("an SCell configuration is sent");
        assert_eq!(channel, RrcChannel::DlDcch, "SRB1, not CCCH");
        let bytes = pdu.data();
        assert_eq!(bytes[0], RECONFIGURATION_WITH_SCELL);
        let decoded = decode_scell_config(&bytes[2..]).expect("the container is real UPER");
        assert_eq!(decoded.to_add.len(), 1);
        assert_eq!(decoded.to_add[0].phys_cell_id, 42);
        assert_eq!(decoded.to_add[0].scell_index, DEFAULT_SCELL_INDEX);
        assert!(decoded.to_release.is_empty());

        // Once per UE: the UE acknowledges each reconfiguration, and the
        // acknowledgement is what would trigger the next one.
        run_async(async { task.send_scell_configuration(7).await });
        assert!(
            try_take_downlink_rrc(&mut rls_rx, 7).is_none(),
            "the same UE is not reconfigured a second time"
        );

        // A different UE still gets its own.
        run_async(async { task.send_scell_configuration(8).await });
        assert!(
            try_take_downlink_rrc(&mut rls_rx, 8).is_some(),
            "the once-per-UE guard is per UE, not global"
        );
    }

    // ========================================================================
    // RRC re-establishment dispatch (issue #37, TS 38.331 §5.3.7.2 / §5.3.3.1)
    // ========================================================================

    /// Drives a UE through setup and records its AS security context, as Initial
    /// Context Setup does.
    async fn establish_with_security(task: &mut RrcTask, ue_id: i32, k_rrc_int: [u8; 16], ncc: u8) {
        use nextgsim_rrc::procedures::rrc_reestablishment::{
            phys_cell_id_from_nci, SIMULATED_C_RNTI,
        };

        establish_pending_setup(task, ue_id).await;
        task.handle_as_security_for_reestablishment(
            ue_id,
            ReestablishmentSecurity {
                k_rrc_int,
                // These two fixtures do not exercise ciphering; NEA0 with a zero key
                // is the honest 'no ciphering configured' state (issue #31).
                k_rrc_enc: [0u8; 16],
                ciphering_alg_id: 0,
                integrity_alg_id: 2, // NIA2
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: phys_cell_id_from_nci(task.task_base.config.nci),
                next_hop_chaining_count: ncc,
            },
        );
    }

    /// A UPER `RRCReestablishmentRequest` as the UE builds it.
    fn reestablishment_request_pdu(nci: u64, k_rrc_int: &[u8; 16], corrupt: bool) -> OctetString {
        use nextgsim_rrc::procedures::rrc_reestablishment::{
            compute_short_mac_i, encode_rrc_reestablishment_request, phys_cell_id_from_nci,
            ReestablishmentCauseValue, ReestablishmentUeIdentity, RrcReestablishmentRequestParams,
            SIMULATED_C_RNTI,
        };

        let pci = phys_cell_id_from_nci(nci);
        let mut short_mac_i =
            compute_short_mac_i(k_rrc_int, 2, SIMULATED_C_RNTI, pci, nci & 0xF_FFFF_FFFF)
                .expect("compute");
        if corrupt {
            short_mac_i ^= 0xFFFF;
        }
        let bytes = encode_rrc_reestablishment_request(&RrcReestablishmentRequestParams {
            ue_identity: ReestablishmentUeIdentity {
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: pci,
                short_mac_i,
            },
            reestablishment_cause: ReestablishmentCauseValue::OtherFailure,
        })
        .expect("encode");
        OctetString::from_slice(&bytes)
    }

    /// The dispatch half of #37: a real UPER `RRCReestablishmentRequest` on
    /// UL-CCCH reaches the re-establishment handler and is answered with an
    /// `RRCReestablishment` on DL-DCCH.
    ///
    /// Before this the same logical message reached `process_rrc_setup_request`,
    /// because the UE's bespoke framing decoded as an `RRCSetupRequest`.
    #[test]
    fn a_verified_reestablishment_request_is_answered_on_dl_dcch() {
        use nextgsim_rrc::procedures::rrc_reestablishment::decode_rrc_reestablishment;

        let config = test_config();
        let nci = config.nci;
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            establish_with_security(&mut task, 1, [0xA1; 16], 6).await;
            // Drain the RRCSetup emitted by the setup above.
            let _setup = try_take_downlink_rrc(&mut rls_rx, 1);

            task.handle_uplink_rrc(
                1,
                RrcChannel::UlCcch,
                reestablishment_request_pdu(nci, &[0xA1; 16], false),
            )
            .await;

            let (channel, pdu) =
                try_take_downlink_rrc(&mut rls_rx, 1).expect("a reply is transmitted");
            assert_eq!(channel, RrcChannel::DlDcch, "SRB1, not CCCH");
            let decoded = decode_rrc_reestablishment(pdu.data())
                .expect("the reply is an RRCReestablishment, not an RRCSetup");
            assert_eq!(
                decoded.next_hop_chaining_count, 6,
                "carrying the security context's NCC"
            );
        });
    }

    /// A `shortMAC-I` the network cannot reproduce falls back to `RRCSetup`
    /// (§5.3.3.1) — on DL-CCCH, because that is where an `RRCSetup` belongs.
    #[test]
    fn an_unverifiable_reestablishment_request_is_answered_with_an_rrc_setup() {
        use nextgsim_rrc::procedures::rrc_reestablishment::decode_rrc_reestablishment;
        use nextgsim_rrc::procedures::rrc_setup::decode_rrc_setup;

        let config = test_config();
        let nci = config.nci;
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            establish_with_security(&mut task, 1, [0xA1; 16], 0).await;
            let _setup = try_take_downlink_rrc(&mut rls_rx, 1);

            task.handle_uplink_rrc(
                1,
                RrcChannel::UlCcch,
                reestablishment_request_pdu(nci, &[0xA1; 16], true),
            )
            .await;

            let (channel, pdu) =
                try_take_downlink_rrc(&mut rls_rx, 1).expect("the fallback is transmitted");
            assert_eq!(channel, RrcChannel::DlCcch, "an RRCSetup rides SRB0");
            assert!(
                decode_rrc_reestablishment(pdu.data()).is_err(),
                "not a re-establishment"
            );
            decode_rrc_setup(pdu.data()).expect("an RRCSetup fallback");
        });
    }

    /// A UE the network has no context for at all also gets the `RRCSetup`
    /// fallback rather than a fabricated re-establishment.
    #[test]
    fn a_reestablishment_from_an_unknown_ue_is_answered_with_an_rrc_setup() {
        use nextgsim_rrc::procedures::rrc_setup::decode_rrc_setup;

        let config = test_config();
        let nci = config.nci;
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_radio_power_on();

            task.handle_uplink_rrc(
                7,
                RrcChannel::UlCcch,
                reestablishment_request_pdu(nci, &[0xA1; 16], false),
            )
            .await;

            let (channel, pdu) =
                try_take_downlink_rrc(&mut rls_rx, 7).expect("the fallback is transmitted");
            assert_eq!(channel, RrcChannel::DlCcch);
            decode_rrc_setup(pdu.data()).expect("an RRCSetup fallback");
            assert_eq!(
                task.ue_manager.count(),
                1,
                "one fresh context, from the setup"
            );
        });
    }

    // ========================================================================
    // Strict RRC: no raw NAS on UL-DCCH (issue #30 criterion 6, TS 38.331 §5.3.3)
    // ========================================================================

    /// A bare 5GMM NAS PDU, as a UE that skips the establishment handshake sends
    /// it: EPD 0x7E, security header 0x00, then a Registration Request.
    fn raw_nas_on_dcch() -> OctetString {
        OctetString::from_slice(&[0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D])
    }

    /// **Strict by default.** TS 38.331 §5.3.3 carries the initial NAS inside
    /// `RRCSetupComplete`, so a UE that sends bare NAS on DCCH fails to attach: no
    /// Initial UE Message reaches NGAP and no context is fabricated for it.
    #[test]
    fn a_raw_nas_pdu_on_dcch_is_discarded_by_default() {
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        assert!(
            !task.task_base.config.accept_raw_nas_on_dcch,
            "the shipped default is strict"
        );

        run_async(async {
            task.handle_radio_power_on();
            task.handle_uplink_rrc(1, RrcChannel::UlDcch, raw_nas_on_dcch())
                .await;
        });

        assert!(
            try_take_initial_nas(&mut ngap_rx).is_none(),
            "no Initial UE Message may be sent for a UE that skipped RRCSetup"
        );
        assert_eq!(
            task.ue_manager.count(),
            0,
            "and no context is auto-created for it"
        );
    }

    /// With the transitional switch on, the leniency behaves exactly as it did
    /// before the gate: the context is auto-created and the NAS goes out as an
    /// Initial UE Message. Asserted so the interop path is not silently lost.
    #[test]
    fn a_raw_nas_pdu_on_dcch_is_accepted_when_configured() {
        let mut config = test_config();
        config.accept_raw_nas_on_dcch = true;
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_radio_power_on();
            task.handle_uplink_rrc(1, RrcChannel::UlDcch, raw_nas_on_dcch())
                .await;
        });

        let (ue_id, pdu) = try_take_initial_nas(&mut ngap_rx)
            .expect("the leniency forwards the NAS as an Initial UE Message");
        assert_eq!(ue_id, 1);
        assert_eq!(pdu.data(), raw_nas_on_dcch().data());
        assert_eq!(task.ue_manager.count(), 1, "the context is auto-created");
    }

    /// The strict discard is **specific to bare NAS**: an encapsulated uplink NAS
    /// still delivers, so the gate does not touch the path a UE that did the
    /// handshake uses.
    ///
    /// The fixture is the bespoke `[0x08, tid, NAS…]` UL-DCCH framing the gNB
    /// actually dispatches, not a real UPER `ULInformationTransfer` — a real one is
    /// **not routed today either**, because UL-DCCH c1 index 8 puts its leading
    /// byte in `0x40..=0x47` and the nibble matcher reads `0x00..=0x07`. That is a
    /// pre-existing gap in the hand-rolled dispatcher (issue #107), unrelated to
    /// this gate, and asserting against it would have made this test fail for the
    /// wrong reason.
    #[test]
    fn the_strict_default_does_not_affect_an_encapsulated_uplink_nas() {
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            establish_pending_setup(&mut task, 1).await;
            if let Some(ctx) = task.ue_manager.try_find_ue_mut(1) {
                ctx.on_setup_complete();
            }
            // [0x08, transaction id, NAS…] — the encapsulated form.
            let mut pdu = vec![0x08, 0x00];
            pdu.extend_from_slice(raw_nas_on_dcch().data());
            task.handle_uplink_rrc(1, RrcChannel::UlDcch, OctetString::from_slice(&pdu))
                .await;
        });

        let mut delivered = None;
        while let Ok(msg) = ngap_rx.try_recv() {
            if let TaskMessage::Message(NgapMessage::UplinkNasDelivery { ue_id, pdu, .. }) = msg {
                delivered = Some((ue_id, pdu));
            }
        }
        let (ue_id, pdu) = delivered.expect("an encapsulated uplink NAS must still reach NGAP");
        assert_eq!(ue_id, 1);
        assert_eq!(
            pdu.data(),
            raw_nas_on_dcch().data(),
            "and with the envelope stripped"
        );
    }

    /// Security material for a UE the RRC task has no context for is dropped,
    /// not used to invent one.
    #[test]
    fn security_for_an_unknown_ue_creates_no_context() {
        use nextgsim_rrc::procedures::rrc_reestablishment::SIMULATED_C_RNTI;

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        task.handle_as_security_for_reestablishment(
            42,
            ReestablishmentSecurity {
                k_rrc_int: [0; 16],
                k_rrc_enc: [0u8; 16],
                ciphering_alg_id: 0,
                integrity_alg_id: 2,
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: 1,
                next_hop_chaining_count: 0,
            },
        );

        assert_eq!(task.ue_manager.count(), 0);
    }

    /// A `physCellId` outside `INTEGER (0..1007)` has no encoding. Nothing is
    /// transmitted, and the UE is not marked as configured — so correcting the
    /// configuration and reconnecting works rather than silently staying quiet.
    #[test]
    fn an_unencodable_scell_phys_cell_id_transmits_nothing() {
        let mut config = test_config();
        config.scell_phys_cell_id = Some(2000); // > 1007
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async { task.send_scell_configuration(1).await });

        assert!(
            try_take_downlink_rrc(&mut rls_rx, 1).is_none(),
            "an out-of-range physCellId must not be truncated onto the wire"
        );
        assert!(
            !task.scell_configured_ues.contains(&1),
            "a UE that was never sent a configuration is not marked as configured"
        );
    }

    // ========================================================================
    // AS security end to end (issue #31, criteria 4/5/7)
    // ========================================================================

    fn as_security_config() -> nextgsim_common::config::GnbConfig {
        nextgsim_common::config::GnbConfig {
            as_security_enabled: true,
            ..test_config()
        }
    }

    const AS_K_INT: [u8; 16] = [0x2Au8; 16];
    const AS_K_ENC: [u8; 16] = [0x3Bu8; 16];

    /// Drives a UE to ACTIVE AS security with NEA2/NIA2, the way Initial Context
    /// Setup plus a SecurityModeComplete would.
    async fn activate_as_security(task: &mut RrcTask, ue_id: i32) {
        use nextgsim_rrc::procedures::rrc_reestablishment::{
            phys_cell_id_from_nci, SIMULATED_C_RNTI,
        };
        establish_pending_setup(task, ue_id).await;
        task.handle_as_security_for_reestablishment(
            ue_id,
            ReestablishmentSecurity {
                k_rrc_int: AS_K_INT,
                k_rrc_enc: AS_K_ENC,
                integrity_alg_id: 2,
                ciphering_alg_id: 2,
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: phys_cell_id_from_nci(task.task_base.config.nci),
                next_hop_chaining_count: 0,
            },
        );
        task.handle_security_mode_complete(ue_id, 0);
    }

    /// The UE's view of the same security state, built through the SHARED layer.
    fn ue_side_security() -> nextgsim_pdcp::srb_security::SrbSecurity {
        nextgsim_pdcp::srb_security::SrbSecurity::new(AS_K_ENC, AS_K_INT, 2, 2).expect("legal ids")
    }

    /// #31, criterion 4: the **SecurityModeCommand** itself is integrity protected
    /// and NOT ciphered (TS 38.331 §5.3.4.2), and the UE's verifier accepts it.
    ///
    /// Added because a revert round found this untested: making the command go out
    /// unprotected left every UE-side SMC test green, since those tests compute the
    /// MAC themselves rather than taking it from the gNB's send path.
    #[test]
    fn the_security_mode_command_is_integrity_protected_and_not_ciphered() {
        use nextgsim_rrc::procedures::rrc_reestablishment::{
            phys_cell_id_from_nci, SIMULATED_C_RNTI,
        };
        use nextgsim_rrc::procedures::security_mode::{
            decode_security_mode_command, encode_security_mode_command, CipheringAlgorithmType,
            IntegrityAlgorithmType, SecurityAlgorithms, SecurityModeCommandParams,
        };

        let (task_base, _app_rx, _ngap_rx, rrc_drop, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        drop(rrc_drop);
        let mut task = RrcTask::new(task_base);

        let smc = encode_security_mode_command(&SecurityModeCommandParams {
            rrc_transaction_id: 0,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea2,
                integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
            },
        })
        .expect("encodes");

        run_async(async {
            establish_pending_setup(&mut task, 7).await;
            task.handle_as_security_for_reestablishment(
                7,
                ReestablishmentSecurity {
                    k_rrc_int: AS_K_INT,
                    k_rrc_enc: AS_K_ENC,
                    integrity_alg_id: 2,
                    ciphering_alg_id: 2,
                    c_rnti: SIMULATED_C_RNTI,
                    phys_cell_id: phys_cell_id_from_nci(task.task_base.config.nci),
                    next_hop_chaining_count: 0,
                },
            );
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}

            let protected = task.protect_security_mode_command(7, OctetString::from_slice(&smc));
            task.send_rrc_message(7, RrcChannel::DlDcch, protected)
                .await;
        });

        let (_ch, out) = try_take_downlink_rrc(&mut rls_rx, 7).expect("the command");
        let bytes = out.data();
        assert_eq!(
            bytes.len(),
            smc.len() + MAC_I_LEN_GNB,
            "the command must carry a MAC-I"
        );
        assert_eq!(
            &bytes[..smc.len()],
            &smc[..],
            "and must NOT be ciphered: the UE has not confirmed it can decipher \
             anything yet (TS 38.331 §5.3.4.2)"
        );
        assert!(
            decode_security_mode_command(&bytes[..smc.len()]).is_ok(),
            "so the command is still decodable as it stands on the wire"
        );

        // The MAC-I the UE would compute: integrity only, COUNT 0.
        let expected = nextgsim_pdcp::srb_security::SrbSecurity::new([0u8; 16], AS_K_INT, 0, 2)
            .expect("ids")
            .compute_mac_i(
                GNB_SMC_PDCP_COUNT,
                SRB1_PDCP_BEARER,
                PDCP_DIRECTION_DOWNLINK,
                &smc,
            );
        assert_eq!(
            &bytes[smc.len()..],
            &expected[..],
            "the MAC must be the one the UE computes, or activation fails with no \
             other symptom"
        );
    }

    /// #31, criterion 7: a downlink SRB1 PDU carries a MAC-I and is ciphered, and
    /// the **UE's** verifier accepts it.
    #[test]
    fn a_downlink_srb1_pdu_is_protected_and_the_ue_verifies_it() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        let mut task = RrcTask::new(task_base);

        let plaintext = vec![0x00u8, 0x01, 0x02, 0x03];
        run_async(async {
            activate_as_security(&mut task, 7).await;
            // Drain everything the setup emitted, so the next PDU is ours.
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}
            task.send_rrc_message(7, RrcChannel::DlDcch, OctetString::from_slice(&plaintext))
                .await;
        });

        let (channel, protected) = try_take_downlink_rrc(&mut rls_rx, 7).expect("a downlink PDU");
        assert_eq!(channel, RrcChannel::DlDcch);
        let bytes = protected.data();
        assert_eq!(
            bytes.len(),
            plaintext.len() + MAC_I_LEN_GNB,
            "a MAC-I must be appended"
        );
        assert_ne!(
            &bytes[..plaintext.len()],
            &plaintext[..],
            "and the PDU must be ciphered, not just tagged"
        );

        // The DL sequence continues at 1: COUNT 0 was the SecurityModeCommand.
        let recovered = ue_side_security()
            .unprotect(1, 0, 1, bytes)
            .expect("the UE must verify what the gNB protected");
        assert_eq!(recovered, plaintext);
    }

    /// #31, criterion 7: an uplink PDU with a forged MAC-I is **discarded** and
    /// never reaches a handler (TS 33.501 §6.5).
    #[test]
    fn an_uplink_srb1_pdu_with_a_forged_mac_is_discarded_before_dispatch() {
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            activate_as_security(&mut task, 7).await;
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}
            while ngap_rx.try_recv().is_ok() {}

            // A well-formed uplink NAS PDU, protected then forged.
            let nas = vec![0x08u8, 0x00, 0x7E, 0x00, 0x41];
            let mut protected = ue_side_security().protect(0, 0, 0, &nas);
            let last = protected.len() - 1;
            protected[last] ^= 0xFF;

            task.handle_ul_dcch_message(7, &OctetString::from_slice(&protected))
                .await;
        });

        assert!(
            ngap_rx.try_recv().is_err(),
            "a PDU that failed integrity must reach NO handler: nothing may be \
             forwarded to NGAP from it"
        );
    }

    /// The guard's own verdict, asserted directly (issue #31, criterion 4).
    ///
    /// Added because the dispatch-level test above is **not sufficient on its own**:
    /// a revert round that passed the forged PDU through unchanged also left it
    /// green, because ciphertext is undispatchable anyway and nothing reached NGAP
    /// either way. This asserts what actually distinguishes discard from
    /// pass-through — `unprotect_ul_dcch` returning `None` — with the valid PDU as
    /// the positive control.
    #[test]
    fn unprotect_ul_dcch_returns_none_for_a_forged_mac_and_the_plaintext_for_a_valid_one() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        let mut task = RrcTask::new(task_base);

        let nas = vec![0x08u8, 0x00, 0x7E, 0x00, 0x41];
        run_async(async {
            activate_as_security(&mut task, 7).await;
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}

            // Valid first, consuming UL COUNT 0.
            let good = ue_side_security().protect(0, 0, 0, &nas);
            let recovered = task
                .unprotect_ul_dcch(7, &OctetString::from_slice(&good))
                .expect("a valid PDU must be accepted");
            assert_eq!(
                recovered.data(),
                &nas[..],
                "and must be DECIPHERED: returning the ciphertext unchanged would \
                 make every dispatcher fail for the wrong reason"
            );

            // Then a forged one at UL COUNT 1.
            let mut bad = ue_side_security().protect(1, 0, 0, &nas);
            let last = bad.len() - 1;
            bad[last] ^= 0xFF;
            assert!(
                task.unprotect_ul_dcch(7, &OctetString::from_slice(&bad))
                    .is_none(),
                "a forged MAC-I must yield None -- the PDU is DISCARDED, not passed \
                 on for a dispatcher to fail on (TS 33.501 §6.5)"
            );
        });
    }

    /// The positive control: the SAME uplink PDU with its real MAC-I is accepted
    /// and dispatched. Without it, "discarded" could mean the gNB discards
    /// everything once security is on.
    #[test]
    fn an_uplink_srb1_pdu_with_a_valid_mac_is_accepted_and_dispatched() {
        let (task_base, _app_rx, mut ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            activate_as_security(&mut task, 7).await;
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}
            while ngap_rx.try_recv().is_ok() {}

            let nas = vec![0x08u8, 0x00, 0x7E, 0x00, 0x41];
            let protected = ue_side_security().protect(0, 0, 0, &nas);
            task.handle_ul_dcch_message(7, &OctetString::from_slice(&protected))
                .await;
        });

        assert!(
            ngap_rx.try_recv().is_ok(),
            "the valid PDU must be deciphered and dispatched; otherwise the \
             forged-MAC test above says nothing"
        );
    }

    /// #31, criterion 5: a DRB-establishing RRCReconfiguration is refused while AS
    /// security is not activated, because a DRB carries user data
    /// (TS 33.501 §6.6.1).
    #[test]
    fn a_drb_reconfiguration_is_refused_before_as_security_is_activated() {
        let (task_base, _app_rx, _ngap_rx, rrc_tx_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
            GnbTaskBase::new(as_security_config(), 16);
        let mut task = RrcTask::new(task_base);
        let (tx, rx) = tokio::sync::mpsc::channel::<TaskMessage<RrcMessage>>(4);
        drop(rrc_tx_rx);

        run_async(async {
            // Set up the UE but do NOT activate security.
            establish_pending_setup(&mut task, 7).await;
            while try_take_downlink_rrc(&mut rls_rx, 7).is_some() {}

            tx.send(TaskMessage::Message(RrcMessage::RrcReconfiguration {
                ue_id: 7,
                pdu: OctetString::from_slice(&[0x00, 0x00, 0x00]),
            }))
            .await
            .expect("queued");
            tx.send(TaskMessage::Shutdown).await.expect("queued");
            drop(tx);
            task.run(rx).await;
        });

        assert!(
            try_take_downlink_rrc(&mut rls_rx, 7).is_none(),
            "no DRB configuration may cross the radio before AS security is active"
        );
    }
}
