//! RRC Connection Management
//!
//! This module implements RRC connection procedures for the gNB:
//! - RRC Setup procedure (`RRCSetupRequest` → `RRCSetup` → `RRCSetupComplete`)
//! - RRC Release procedure
//! - RRC Reconfiguration procedure
//! - Security Mode Command procedure

use tracing::{debug, info, warn};

use nextgsim_common::OctetString;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::{
    rrc_reestablishment::{
        compute_short_mac_i, encode_rrc_reestablishment, ReestablishmentCauseValue,
        RrcReestablishmentError, RrcReestablishmentParams,
    },
    rrc_release::{encode_rrc_release, CellReselectionPrioritiesParams, RrcReleaseParams},
    rrc_setup::{encode_rrc_setup, srb1_rrc_setup_params},
};

use super::transaction::{RrcProcedure, TidVerification};
use super::ue_context::RrcUeContextManager;
use crate::tasks::GutiMobileIdentity;

/// Result of processing an RRC Setup Request
#[derive(Debug)]
pub struct RrcSetupResult {
    /// UE ID
    pub ue_id: i32,
    /// Transaction ID used
    pub transaction_id: u8,
    /// RRC Setup message to send (encoded)
    pub rrc_setup_pdu: OctetString,
    /// RRC channel to use
    pub channel: RrcChannel,
}

/// Result of processing an RRC Setup Complete
#[derive(Debug)]
pub struct RrcSetupCompleteResult {
    /// UE ID
    pub ue_id: i32,
    /// Dedicated NAS message to forward to NGAP
    pub nas_pdu: OctetString,
    /// RRC establishment cause
    pub establishment_cause: i64,
    /// 5G-S-TMSI if available
    pub s_tmsi: Option<GutiMobileIdentity>,
}

/// Result of an RRC Release
#[derive(Debug)]
pub struct RrcReleaseResult {
    /// UE ID
    pub ue_id: i32,
    /// Transaction ID used
    pub transaction_id: u8,
    /// RRC Release message to send (encoded)
    pub rrc_release_pdu: OctetString,
    /// RRC channel to use
    pub channel: RrcChannel,
}

/// An `RRCReestablishmentRequest` as presented by the UE, plus the identity of
/// the cell it arrived on.
///
/// The first three fields are the `ReestabUE-Identity` of TS 38.331 §6.2.2 and
/// come from the ASN.1 decode, not from byte offsets. `target_cell_identity` is
/// the 36-bit NR Cell Identity of the cell handling the request, which the
/// `VarShortMAC-Input` needs and which the message itself does not carry.
#[derive(Debug, Clone, Copy)]
pub struct ReestablishmentRequest {
    /// `ueIdentity.c-RNTI` — the C-RNTI in the source PCell
    pub c_rnti: u16,
    /// `ueIdentity.physCellId` — the PCI of the source PCell
    pub phys_cell_id: u16,
    /// `ueIdentity.shortMAC-I` — the 16 LSBs of the MAC-I the UE computed
    pub short_mac_i: u16,
    /// `reestablishmentCause`
    pub cause: ReestablishmentCauseValue,
    /// 36-bit NR Cell Identity of the cell the UE is re-establishing on
    pub target_cell_identity: u64,
}

/// Why a re-establishment did not complete, i.e. why the caller must fall back
/// to `RRCSetup` (TS 38.331 §5.3.3.1) or send nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReestablishmentRejection {
    /// The cell is barred: nothing is sent at all.
    CellBarred,
    /// No stored context matches the presented `(C-RNTI, PCI)`.
    UnknownIdentity,
    /// A context matched the identity but no candidate's `K_RRCint` reproduced
    /// the presented `shortMAC-I`.
    MacVerificationFailed,
    /// The context verified but the reply could not be encoded.
    EncodingFailed,
}

impl ReestablishmentRejection {
    /// Whether the network should answer with an `RRCSetup` (§5.3.3.1). A barred
    /// cell answers with nothing at all, so it is the one case that does not.
    pub fn falls_back_to_setup(self) -> bool {
        !matches!(self, ReestablishmentRejection::CellBarred)
    }
}

/// Result of processing an RRC Reestablishment Request
#[derive(Debug)]
pub struct RrcReestablishmentResult {
    /// UE ID
    pub ue_id: i32,
    /// Transaction ID used
    pub transaction_id: u8,
    /// RRC Reestablishment message to send (encoded)
    pub rrc_reestablishment_pdu: OctetString,
    /// RRC channel to use
    pub channel: RrcChannel,
    /// Reestablishment cause, as decoded (no longer a raw byte offset)
    pub cause: ReestablishmentCauseValue,
}

/// Result of processing an RRC Reestablishment Complete
#[derive(Debug)]
pub struct RrcReestablishmentCompleteResult {
    /// UE ID
    pub ue_id: i32,
    /// NAS PDU to forward to NGAP (if any)
    pub nas_pdu: Option<OctetString>,
}

/// Result of processing an RRC Resume Request
#[derive(Debug)]
pub struct RrcResumeResult {
    /// UE ID
    pub ue_id: i32,
    /// Transaction ID used
    pub transaction_id: u8,
    /// RRC Resume message to send (encoded)
    pub rrc_resume_pdu: OctetString,
    /// RRC channel to use
    pub channel: RrcChannel,
    /// Resume cause
    pub cause: u8,
}

/// Result of processing an RRC Resume Complete
#[derive(Debug)]
pub struct RrcResumeCompleteResult {
    /// UE ID
    pub ue_id: i32,
    /// NAS PDU to forward to NGAP (if any)
    pub nas_pdu: Option<OctetString>,
}

/// RRC connection manager
#[derive(Debug)]
pub struct RrcConnectionManager {
    /// Transaction ID counter for the DL-CCCH/DL-DCCH procedures that are NOT
    /// yet per-UE (RRC Release / Reestablishment / Resume). RRCSetup and the
    /// four C4-final sender procedures allocate from the per-UE
    /// [`RrcTransactionAllocator`](super::transaction::RrcTransactionAllocator)
    /// on the UE context instead.
    tid_counter: u8,
    /// Whether the cell is barred
    is_barred: bool,
}

impl Default for RrcConnectionManager {
    fn default() -> Self {
        Self::new()
    }
}

impl RrcConnectionManager {
    /// Creates a new RRC connection manager
    pub fn new() -> Self {
        Self {
            tid_counter: 0,
            is_barred: true, // Initially barred until radio power on
        }
    }

    /// Gets the next transaction ID (cycles 0-3) for the not-yet-per-UE
    /// DL procedures (RRC Release / Reestablishment / Resume).
    pub fn next_tid(&mut self) -> u8 {
        let tid = self.tid_counter;
        self.tid_counter = (self.tid_counter + 1) % 4;
        tid
    }

    /// Sets the cell barred status
    pub fn set_barred(&mut self, barred: bool) {
        self.is_barred = barred;
        if barred {
            info!("Cell is now barred");
        } else {
            info!("Cell is now unbarred");
        }
    }

    /// Returns true if the cell is barred
    pub fn is_barred(&self) -> bool {
        self.is_barred
    }

    /// Processes an RRC Setup Request
    ///
    /// Returns the RRC Setup message to send, or None if the request should be rejected.
    pub fn process_rrc_setup_request(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        initial_id: i64,
        is_stmsi: bool,
        establishment_cause: i64,
    ) -> Option<RrcSetupResult> {
        // Check if cell is barred
        if self.is_barred {
            warn!("Rejecting RRC Setup Request: cell is barred");
            return None;
        }

        // Check if UE context already exists
        if ue_mgr.try_find_ue(ue_id).is_some() {
            warn!(
                "Discarding RRC Setup Request: UE context already exists for ue_id={}",
                ue_id
            );
            return None;
        }

        // Create UE context
        let ctx = ue_mgr.create_ue(ue_id);
        ctx.set_initial_id(initial_id, is_stmsi);
        ctx.set_establishment_cause(establishment_cause);
        ctx.on_setup_request();

        // Wave-6 C4-final: allocate the RRCSetup transaction id from THIS UE's
        // per-context allocator (TS 38.331 §6.3.2), NOT a gNB-global counter.
        // Because the context is fresh, the allocation is deterministically tid
        // 0 (nibble-safe on DL-CCCH) while C5_TYPED_DCCH_DISPATCH is off — this
        // retires the former global cycler's latent 2nd/4th-connection RRCSetup
        // drop without the {0,2} parity hack. The tid is recorded as
        // outstanding so process_rrc_setup_complete can verify the UE's echo.
        let transaction_id = ctx.transactions.allocate(RrcProcedure::Setup);

        // Build RRC Setup message
        let rrc_setup_pdu = self.build_rrc_setup(transaction_id);

        // Mark setup sent
        if let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) {
            ctx.on_setup_sent();
        }

        info!(
            "RRC Setup for UE[{}], tid={}, initial_id={:x}, is_stmsi={}",
            ue_id, transaction_id, initial_id, is_stmsi
        );

        Some(RrcSetupResult {
            ue_id,
            transaction_id,
            rrc_setup_pdu,
            channel: RrcChannel::DlCcch,
        })
    }

    /// Processes an RRC Setup Complete
    ///
    /// Returns the NAS PDU to forward to NGAP.
    pub fn process_rrc_setup_complete(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        transaction_id: u8,
        nas_pdu: OctetString,
        s_tmsi_value: Option<GutiMobileIdentity>,
    ) -> Option<RrcSetupCompleteResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;

        // Wave-6 C4-final: verify the tid the UE echoed against the outstanding
        // RRCSetup transaction (TS 38.331 §5.3.3). Fail-closed — discard a
        // Complete carrying the wrong tid instead of pairing it to the wrong
        // procedure. NoOutstanding (a context auto-created on a raw-NAS uplink
        // with no gNB-sent RRCSetup) is tolerated: there is nothing to match.
        match ctx.transactions.verify(RrcProcedure::Setup, transaction_id) {
            TidVerification::Mismatch { expected } => {
                warn!(
                    "Discarding RRCSetupComplete for UE[{}]: echoed tid {} != \
                     outstanding {} (TS 38.331 §5.3.3)",
                    ue_id, transaction_id, expected
                );
                return None;
            }
            TidVerification::Match | TidVerification::NoOutstanding => {}
        }

        // Handle 5G-S-TMSI if provided
        if let Some(stmsi) = s_tmsi_value.clone() {
            ctx.set_s_tmsi(stmsi);
        }

        // Transition to Connected state
        ctx.on_setup_complete();

        let establishment_cause = ctx.establishment_cause;
        let s_tmsi = ctx.s_tmsi.clone();

        debug!(
            "RRC Setup Complete for UE[{}], nas_pdu_len={}, s_tmsi={:?}",
            ue_id,
            nas_pdu.len(),
            s_tmsi
        );

        Some(RrcSetupCompleteResult {
            ue_id,
            nas_pdu,
            establishment_cause,
            s_tmsi,
        })
    }

    /// Initiates an RRC Release for a UE.
    ///
    /// `cell_reselection_priorities` is the DEDICATED priority list the released
    /// UE should use in idle (TS 38.331 §6.3.2 `cellReselectionPriorities`,
    /// TS 38.304 §5.2.4.1). It is a parameter rather than read from a config here
    /// because this manager holds no configuration; the caller builds it from the
    /// cell's own `reselection` block. `None` omits the IE, which leaves the UE on
    /// the broadcast priorities -- the pre-#50 behaviour, and the correct one when
    /// the cell has nothing dedicated to say.
    pub fn initiate_rrc_release(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        cell_reselection_priorities: Option<CellReselectionPrioritiesParams>,
    ) -> Option<RrcReleaseResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;

        if !ctx.is_connected() {
            debug!("UE[{}] not connected, skipping RRC Release", ue_id);
            return None;
        }

        let transaction_id = self.next_tid();

        // Build RRC Release message
        let rrc_release_pdu = self.build_rrc_release(transaction_id, cell_reselection_priorities);

        // Transition to Releasing state
        ctx.on_release();

        info!("RRC Release for UE[{}], tid={}", ue_id, transaction_id);

        Some(RrcReleaseResult {
            ue_id,
            transaction_id,
            rrc_release_pdu,
            channel: RrcChannel::DlDcch,
        })
    }

    /// Processes an RRC Reestablishment Request (TS 38.331 §5.3.7.2, §5.3.3.1).
    ///
    /// The network-side procedure proper:
    ///
    /// 1. Look the stored context up by the **presented** `(C-RNTI, PCI)`, not by
    ///    the transport-level `ue_id`. The lookup returns candidates; see
    ///    [`super::ue_context::ReestablishmentSecurity::c_rnti`] for why more than
    ///    one is possible here.
    /// 2. Recompute the `shortMAC-I` over the `VarShortMAC-Input` with each
    ///    candidate's stored `K_RRCint` and compare the 16 least significant bits
    ///    against the presented value. The candidate that matches is the context.
    /// 3. **Fall back to `RRCSetup`** when no candidate matches, when none can be
    ///    verified, or when the identity resolves to nothing — §5.3.3.1. No fresh
    ///    `RRCReestablishment` context is fabricated: reestablishing a UE the
    ///    network cannot authenticate is the defect this replaces.
    ///
    /// `target_cell_identity` is the 36-bit NR Cell Identity of the cell the UE is
    /// re-establishing on, which is the third input of the `VarShortMAC-Input`.
    ///
    /// Returns `Ok` with the reply on success and `Err(ReestablishmentRejection)`
    /// carrying the reason when the caller should send an `RRCSetup` instead.
    pub fn process_rrc_reestablishment_request(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        request: &ReestablishmentRequest,
    ) -> Result<RrcReestablishmentResult, ReestablishmentRejection> {
        if self.is_barred {
            warn!("Rejecting RRC Reestablishment: cell is barred");
            return Err(ReestablishmentRejection::CellBarred);
        }

        let candidates =
            ue_mgr.candidates_for_reestablishment(request.c_rnti, request.phys_cell_id);
        if candidates.is_empty() {
            warn!(
                "RRC Reestablishment falls back to RRCSetup: no stored context for \
                 (c_rnti={:#06x}, pci={}) (TS 38.331 §5.3.3.1)",
                request.c_rnti, request.phys_cell_id
            );
            return Err(ReestablishmentRejection::UnknownIdentity);
        }

        let verified = candidates.into_iter().find(|&candidate| {
            let Some(security) = ue_mgr
                .try_find_ue(candidate)
                .and_then(|ctx| ctx.reestablishment_security.as_ref())
            else {
                return false;
            };
            match compute_short_mac_i(
                &security.k_rrc_int,
                security.integrity_alg_id,
                security.c_rnti,
                security.phys_cell_id,
                request.target_cell_identity,
            ) {
                Ok(expected) => expected == request.short_mac_i,
                Err(e) => {
                    warn!(
                        "ShortMAC-I derivation failed for candidate UE[{}]: {}",
                        candidate, e
                    );
                    false
                }
            }
        });

        let Some(ue_id) = verified else {
            warn!(
                "RRC Reestablishment falls back to RRCSetup: shortMAC-I {:#06x} \
                 verified against no stored context for (c_rnti={:#06x}, pci={}) \
                 (TS 38.331 §5.3.7.2)",
                request.short_mac_i, request.c_rnti, request.phys_cell_id
            );
            return Err(ReestablishmentRejection::MacVerificationFailed);
        };

        // The verified context supplies the nextHopChainingCount the reply carries
        // (TS 33.501 §6.9.4.1) -- not a hardcoded zero.
        let next_hop_chaining_count = ue_mgr
            .try_find_ue(ue_id)
            .and_then(|ctx| ctx.reestablishment_security.as_ref())
            .map(|s| s.next_hop_chaining_count)
            .unwrap_or(0);

        let transaction_id = {
            let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) else {
                return Err(ReestablishmentRejection::UnknownIdentity);
            };
            ctx.transactions.allocate(RrcProcedure::Reestablishment)
        };

        let rrc_reestablishment_pdu =
            match Self::build_rrc_reestablishment(transaction_id, next_hop_chaining_count) {
                Ok(pdu) => pdu,
                Err(e) => {
                    warn!(
                        "RRCReestablishment encoding failed: {} — falling back to RRCSetup",
                        e
                    );
                    return Err(ReestablishmentRejection::EncodingFailed);
                }
            };

        if let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) {
            ctx.on_setup_sent();
        }

        info!(
            "RRC Reestablishment for UE[{}], tid={}, c_rnti={:#06x}, pci={}, \
             cause={:?}, ncc={} (shortMAC-I verified)",
            ue_id,
            transaction_id,
            request.c_rnti,
            request.phys_cell_id,
            request.cause,
            next_hop_chaining_count
        );

        Ok(RrcReestablishmentResult {
            ue_id,
            transaction_id,
            rrc_reestablishment_pdu,
            // TS 38.331 §6.2.1: RRCReestablishment is a DL-DCCH / SRB1 message.
            // Only the preceding RRCReestablishmentRequest rides SRB0 / UL-CCCH.
            channel: RrcChannel::DlDcch,
            cause: request.cause,
        })
    }

    /// Processes an RRC Reestablishment Complete.
    ///
    /// The echoed `rrc-TransactionIdentifier` is verified against the outstanding
    /// Reestablishment transaction: a mismatch means the UE is completing a
    /// different procedure, so it is discarded rather than transitioning the
    /// context to Connected.
    pub fn process_rrc_reestablishment_complete(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        transaction_id: u8,
    ) -> Option<RrcReestablishmentCompleteResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;
        match ctx
            .transactions
            .verify(RrcProcedure::Reestablishment, transaction_id)
        {
            TidVerification::Mismatch { expected } => {
                warn!(
                    "Discarding RRCReestablishmentComplete from UE[{}]: echoed tid {} \
                     != outstanding {} (TS 38.331 §5.3.7)",
                    ue_id, transaction_id, expected
                );
                return None;
            }
            TidVerification::Match | TidVerification::NoOutstanding => {}
        }
        ctx.on_setup_complete();

        info!(
            "RRC Reestablishment Complete for UE[{}], tid={}",
            ue_id, transaction_id
        );

        Some(RrcReestablishmentCompleteResult {
            ue_id,
            nas_pdu: None,
        })
    }

    /// Processes an RRC Resume Request
    ///
    /// Called when a UE in `RRC_INACTIVE` state resumes its connection.
    pub fn process_rrc_resume_request(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        resume_cause: u8,
    ) -> Option<RrcResumeResult> {
        if self.is_barred {
            warn!("Rejecting RRC Resume: cell is barred");
            return None;
        }

        // For resume, the UE may or may not have an existing context
        if ue_mgr.try_find_ue(ue_id).is_none() {
            let ctx = ue_mgr.create_ue(ue_id);
            ctx.on_setup_request();
        }

        let transaction_id = self.next_tid();
        let rrc_resume_pdu = self.build_rrc_resume(transaction_id);

        if let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) {
            ctx.on_setup_sent();
        }

        info!(
            "RRC Resume for UE[{}], tid={}, cause={}",
            ue_id, transaction_id, resume_cause
        );

        Some(RrcResumeResult {
            ue_id,
            transaction_id,
            rrc_resume_pdu,
            channel: RrcChannel::DlCcch,
            cause: resume_cause,
        })
    }

    /// Processes an RRC Resume Complete
    pub fn process_rrc_resume_complete(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        _transaction_id: u8,
        nas_pdu: Option<OctetString>,
    ) -> Option<RrcResumeCompleteResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;
        ctx.on_setup_complete();

        info!("RRC Resume Complete for UE[{}]", ue_id);

        Some(RrcResumeCompleteResult { ue_id, nas_pdu })
    }

    /// Builds an `RRCReestablishment` (TS 38.331 §6.2.2) as real UPER on
    /// DL-DCCH, carrying the `nextHopChainingCount` of the verified UE's AS
    /// security context.
    ///
    /// This replaces a hand-rolled four-byte DL-CCCH PDU with a hardcoded NCC of
    /// 0; the comment there said the ASN.1 builder was unusable "since the ASN.1
    /// gNB-side builder requires nextHopChainingCount from security context which
    /// is not yet wired", which is now wired.
    fn build_rrc_reestablishment(
        transaction_id: u8,
        next_hop_chaining_count: u8,
    ) -> Result<OctetString, RrcReestablishmentError> {
        let bytes = encode_rrc_reestablishment(&RrcReestablishmentParams {
            rrc_transaction_id: transaction_id,
            next_hop_chaining_count,
        })?;
        Ok(OctetString::from_slice(&bytes))
    }

    /// Builds an RRC Resume message
    ///
    /// Note: The RRC procedures module provides UE-side RRCResumeRequest
    /// and RRCResumeComplete. The gNB-side RRCResume (DL-CCCH) uses a simplified
    /// encoding since the ASN.1 gNB-side builder requires masterCellGroup and
    /// radio bearer configuration from the suspended UE context.
    fn build_rrc_resume(&self, transaction_id: u8) -> OctetString {
        let mut pdu = Vec::with_capacity(16);
        // DL-CCCH-Message with RRCResume
        pdu.push(0x28); // c1 choice = rrcResume
        pdu.push(transaction_id);
        pdu.push(0x00); // criticalExtensions = rrcResume
                        // Minimal masterCellGroup
        pdu.extend_from_slice(&[0x00, 0x00]);
        OctetString::from_slice(&pdu)
    }

    /// Handles radio link failure for a UE
    pub fn handle_radio_link_failure(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
    ) -> bool {
        if let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) {
            ctx.on_release();
            info!("Radio link failure for UE[{}]", ue_id);
            true
        } else {
            warn!("Radio link failure for unknown UE[{}]", ue_id);
            false
        }
    }

    /// Builds an RRC Setup message using proper ASN.1 UPER encoding.
    ///
    /// Wave-6 C1: carries the REAL SRB1 configuration — a RadioBearerConfig
    /// with srb-ToAddModList = { SRB-Identity 1 } (TS 38.331 §5.3.5.6.3:
    /// RRCSetup shall establish SRB1) and a masterCellGroup CONTAINING a
    /// CellGroupConfig with one RLC-BearerConfig (LCID 1) serving SRB1
    /// (TS 38.331 §5.3.3.4 / §6.3.2).
    fn build_rrc_setup(&self, transaction_id: u8) -> OctetString {
        // Wave-6 C4-final: RRC-TransactionIdentifier is INTEGER(0..3)
        // (TS 38.331 §6.3.2). Until C5 types the UE's DL-CCCH dispatcher, the
        // per-UE allocator pins gNB→UE tids to the nibble-0-safe value 0
        // (leading byte 0x20); flipping C5_TYPED_DCCH_DISPATCH enables the full
        // per-UE 0..3 cycle. Guard both invariants.
        debug_assert!(
            transaction_id <= 3,
            "RRC-TransactionIdentifier is INTEGER(0..3), got {transaction_id}"
        );
        debug_assert!(
            super::transaction::C5_TYPED_DCCH_DISPATCH || transaction_id == 0,
            "RRCSetup tid must be 0 (nibble-safe on the legacy UE DL-CCCH \
             dispatcher) until C5, got {transaction_id}"
        );

        match srb1_rrc_setup_params(transaction_id).and_then(|params| encode_rrc_setup(&params)) {
            Ok(bytes) => OctetString::from_slice(&bytes),
            Err(e) => {
                warn!("ASN.1 RRC Setup encoding failed ({}), using fallback", e);
                // Fallback: simplified encoding for interop with simplified UE
                // parser (retired in C6)
                let mut pdu = Vec::with_capacity(16);
                pdu.push(0x20);
                pdu.push(transaction_id);
                pdu.push(0x00);
                pdu.extend_from_slice(&[0x00, 0x00]);
                pdu.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
                OctetString::from_slice(&pdu)
            }
        }
    }

    /// Builds an RRC Release message using proper ASN.1 UPER encoding
    fn build_rrc_release(
        &self,
        transaction_id: u8,
        cell_reselection_priorities: Option<CellReselectionPrioritiesParams>,
    ) -> OctetString {
        let params = RrcReleaseParams {
            rrc_transaction_id: transaction_id,
            cell_reselection_priorities,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: None,
            wait_time: None,
        };

        match encode_rrc_release(&params) {
            Ok(bytes) => OctetString::from_slice(&bytes),
            Err(e) => {
                warn!("ASN.1 RRC Release encoding failed ({}), using fallback", e);
                let mut pdu = Vec::with_capacity(8);
                pdu.push(0x0D);
                pdu.push(transaction_id);
                pdu.push(0x00);
                OctetString::from_slice(&pdu)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_manager_new() {
        let mgr = RrcConnectionManager::new();
        assert!(mgr.is_barred());
    }

    #[test]
    fn test_tid_cycling() {
        let mut mgr = RrcConnectionManager::new();
        assert_eq!(mgr.next_tid(), 0);
        assert_eq!(mgr.next_tid(), 1);
        assert_eq!(mgr.next_tid(), 2);
        assert_eq!(mgr.next_tid(), 3);
        assert_eq!(mgr.next_tid(), 0);
    }

    #[test]
    fn test_set_barred() {
        let mut mgr = RrcConnectionManager::new();
        assert!(mgr.is_barred());

        mgr.set_barred(false);
        assert!(!mgr.is_barred());

        mgr.set_barred(true);
        assert!(mgr.is_barred());
    }

    #[test]
    fn test_rrc_setup_request_barred() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        // Cell is barred by default
        let result = conn_mgr.process_rrc_setup_request(
            &mut ue_mgr,
            1,
            0x1234567890,
            false,
            3, // MO_SIGNALLING
        );

        assert!(result.is_none());
    }

    #[test]
    fn test_rrc_setup_request_success() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        conn_mgr.set_barred(false);

        let result = conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.ue_id, 1);
        assert_eq!(result.channel, RrcChannel::DlCcch);

        // Verify UE context was created
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert_eq!(ctx.initial_id, Some(0x1234567890));
        assert!(!ctx.is_initial_id_s_tmsi);
    }

    /// Wave-6 C1: the emitted RRCSetup must be EXACTLY the hand-derived
    /// golden UPER PDU carrying the SRB1 configuration (TS 38.331
    /// §5.3.5.6.3). The literal is derived by hand from
    /// tools/rrc-15.6.0.asn1 — see nextgsim-rrc rrc_setup.rs
    /// `golden_rrc_setup_srb1_bytes` for the bit-by-bit derivation.
    #[test]
    fn test_rrc_setup_request_emits_golden_srb1_pdu_tid0() {
        const GOLDEN_RRC_SETUP_SRB1_TID0: [u8; 8] =
            [0x20, 0x40, 0x00, 0x22, 0x00, 0x04, 0x00, 0x00];

        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);

        let result = conn_mgr
            .process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3)
            .expect("setup must succeed");
        assert_eq!(result.transaction_id, 0, "fresh manager -> tid 0");
        assert_eq!(
            result.rrc_setup_pdu.data(),
            &GOLDEN_RRC_SETUP_SRB1_TID0[..],
            "RRCSetup(tid 0) must be the hand-derived SRB1 golden PDU"
        );
    }

    /// Wave-6 C4-final: RRCSetup tids come from the PER-UE allocator, not the
    /// former gNB-global cycler. Each UE context is fresh, so every UE's
    /// RRCSetup is deterministically tid 0 — nibble-0-safe on the UE's legacy
    /// DL-CCCH dispatcher (byte0 0x20, low nibble 0x0) — with no {0,2} parity
    /// hack and no 2nd/4th-connection drop (TS 38.331 §6.3.2). One UE's
    /// allocations no longer shift another UE's tids.
    #[test]
    fn test_rrc_setup_tid_per_ue_is_nibble_safe_zero() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);

        for ue_id in 1..=4 {
            let result = conn_mgr
                .process_rrc_setup_request(&mut ue_mgr, ue_id, 0x1000 + i64::from(ue_id), false, 3)
                .expect("setup must succeed");
            assert_eq!(
                result.transaction_id, 0,
                "each fresh per-UE allocator's first RRCSetup tid is 0"
            );
            let byte0 = result.rrc_setup_pdu.data()[0];
            assert_eq!(
                byte0, 0x20,
                "RRCSetup(tid 0) leading byte must be 0x20 (low nibble 0x0), got 0x{byte0:02x}"
            );
            // The tid is recorded as outstanding on THIS UE's context.
            assert_eq!(
                ue_mgr
                    .try_find_ue(ue_id)
                    .unwrap()
                    .transactions
                    .outstanding(RrcProcedure::Setup),
                Some(0)
            );
        }
    }

    /// Wave-6 C4-final: the gNB verifies the tid the UE echoes in
    /// RRCSetupComplete against the outstanding RRCSetup transaction and
    /// discards a mismatched Complete fail-closed, rather than pairing it to
    /// the wrong procedure (TS 38.331 §5.3.3).
    #[test]
    fn test_rrc_setup_complete_tid_mismatch_is_discarded() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);

        let setup = conn_mgr
            .process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3)
            .expect("setup must succeed");
        assert_eq!(setup.transaction_id, 0, "outstanding setup tid is 0");

        // A Complete echoing the WRONG tid is discarded (fail-closed) and must
        // not advance the UE to Connected.
        let bad = conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            1, // != outstanding 0
            OctetString::from_slice(&[0x7E, 0x00, 0x41]),
            None,
        );
        assert!(
            bad.is_none(),
            "mismatched RRCSetupComplete must be discarded"
        );
        assert!(
            !ue_mgr.try_find_ue(1).unwrap().is_connected(),
            "a discarded Complete must not move the UE to Connected"
        );

        // The correctly-echoed tid is still accepted afterwards.
        let good = conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            0,
            OctetString::from_slice(&[0x7E, 0x00, 0x41]),
            None,
        );
        assert!(good.is_some(), "the correct echoed tid must be accepted");
        assert!(ue_mgr.try_find_ue(1).unwrap().is_connected());
    }

    /// A raw-NAS-auto-created context (no gNB-sent RRCSetup, so no outstanding
    /// transaction) must NOT be rejected by the tid check — there is nothing to
    /// verify against (NoOutstanding is tolerated).
    #[test]
    fn test_rrc_setup_complete_no_outstanding_is_tolerated() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        // Context created directly (mirrors the raw-NAS UL-DCCH auto-create).
        ue_mgr.create_ue(7);
        let result = conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            7,
            0,
            OctetString::from_slice(&[0x7E]),
            None,
        );
        assert!(
            result.is_some(),
            "no outstanding transaction -> tolerate, do not reject"
        );
    }

    #[test]
    fn test_rrc_setup_request_duplicate() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        conn_mgr.set_barred(false);

        // First request succeeds
        let result1 = conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);
        assert!(result1.is_some());

        // Second request for same UE fails
        let result2 = conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);
        assert!(result2.is_none());
    }

    #[test]
    fn test_rrc_setup_complete() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        conn_mgr.set_barred(false);

        // Setup request
        conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);

        // Setup complete
        let nas_pdu = OctetString::from_slice(&[0x7E, 0x00, 0x41]);
        let result = conn_mgr.process_rrc_setup_complete(&mut ue_mgr, 1, 0, nas_pdu.clone(), None);

        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.ue_id, 1);
        assert_eq!(result.nas_pdu, nas_pdu);

        // Verify UE is now connected
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert!(ctx.is_connected());
    }

    #[test]
    fn test_rrc_release() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        conn_mgr.set_barred(false);

        // Setup
        conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);
        conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            0,
            OctetString::from_slice(&[0x7E]),
            None,
        );

        // Release with no dedicated priorities: the IE is absent, which leaves the
        // UE on the broadcast ones (issue #50).
        let result = conn_mgr.initiate_rrc_release(&mut ue_mgr, 1, None);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.ue_id, 1);
        assert_eq!(result.channel, RrcChannel::DlDcch);

        // Verify UE is now releasing
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert!(!ctx.is_connected());
    }

    /// #50, criterion 3: `build_rrc_release` no longer hardcodes
    /// `cell_reselection_priorities: None`, and the dedicated list reaches the
    /// encoded PDU rather than being accepted and dropped.
    ///
    /// Asserted by DECODING the emitted PDU, not by inspecting the params struct:
    /// the defect being fixed was a value that never reached the wire, so reading
    /// back the input would have passed against it.
    #[test]
    fn a_dedicated_reselection_priority_list_reaches_the_encoded_rrc_release() {
        use nextgsim_rrc::procedures::rrc_release::{decode_rrc_release, FreqPriorityNrParams};

        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        // The cell is barred by default, and a barred cell rejects the setup
        // request, so without this the release below has no connected UE to
        // release and returns None.
        conn_mgr.set_barred(false);
        conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);
        conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            0,
            OctetString::from_slice(&[0x7E]),
            None,
        );

        let priorities = CellReselectionPrioritiesParams {
            freq_priority_list_nr: vec![
                FreqPriorityNrParams {
                    carrier_freq: 632628,
                    priority: 5,
                },
                FreqPriorityNrParams {
                    carrier_freq: 500000,
                    priority: 2,
                },
            ],
            t320: None,
        };

        let result = conn_mgr
            .initiate_rrc_release(&mut ue_mgr, 1, Some(priorities))
            .expect("release emitted");

        let decoded = decode_rrc_release(result.rrc_release_pdu.data())
            .expect("the emitted RRCRelease must be decodable, not the byte fallback");
        let carried = decoded
            .cell_reselection_priorities
            .expect("cellReselectionPriorities must be on the wire");
        assert_eq!(carried.freq_priority_list_nr.len(), 2);
        assert_eq!(carried.freq_priority_list_nr[0].carrier_freq, 632628);
        assert_eq!(carried.freq_priority_list_nr[0].priority, 5);
        assert_eq!(carried.freq_priority_list_nr[1].carrier_freq, 500000);
        assert_eq!(carried.freq_priority_list_nr[1].priority, 2);
    }

    #[test]
    fn test_radio_link_failure() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();

        conn_mgr.set_barred(false);

        // Setup
        conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);
        conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            0,
            OctetString::from_slice(&[0x7E]),
            None,
        );

        // Radio link failure
        let result = conn_mgr.handle_radio_link_failure(&mut ue_mgr, 1);
        assert!(result);

        // Verify UE is now releasing
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert!(!ctx.is_connected());
    }

    // ========================================================================
    // RRC re-establishment verification and fallback (issue #37,
    // TS 38.331 §5.3.7.2 / §5.3.3.1)
    // ========================================================================

    use super::super::ue_context::ReestablishmentSecurity;
    use nextgsim_rrc::procedures::rrc_reestablishment::{
        decode_rrc_reestablishment, SIMULATED_C_RNTI,
    };

    const TEST_PCI: u16 = 16;
    const TEST_CELL_IDENTITY: u64 = 0x10;
    const TEST_INTEGRITY_ALG: u8 = 2; // NIA2

    fn key(seed: u8) -> [u8; 16] {
        [seed; 16]
    }

    /// A connected UE whose AS security context has been recorded, as it is after
    /// Initial Context Setup.
    fn connected_ue_with_security(
        conn_mgr: &mut RrcConnectionManager,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        k_rrc_int: [u8; 16],
        ncc: u8,
    ) {
        conn_mgr.process_rrc_setup_request(
            ue_mgr,
            ue_id,
            0x1234567890 + i64::from(ue_id),
            false,
            3,
        );
        conn_mgr.process_rrc_setup_complete(
            ue_mgr,
            ue_id,
            0,
            OctetString::from_slice(&[0x7E]),
            None,
        );
        ue_mgr
            .try_find_ue_mut(ue_id)
            .expect("context")
            .set_reestablishment_security(ReestablishmentSecurity {
                k_rrc_int,
                integrity_alg_id: TEST_INTEGRITY_ALG,
                c_rnti: SIMULATED_C_RNTI,
                phys_cell_id: TEST_PCI,
                next_hop_chaining_count: ncc,
            });
    }

    /// The request a UE holding `k_rrc_int` would present.
    fn request_for(k_rrc_int: &[u8; 16]) -> ReestablishmentRequest {
        ReestablishmentRequest {
            c_rnti: SIMULATED_C_RNTI,
            phys_cell_id: TEST_PCI,
            short_mac_i: compute_short_mac_i(
                k_rrc_int,
                TEST_INTEGRITY_ALG,
                SIMULATED_C_RNTI,
                TEST_PCI,
                TEST_CELL_IDENTITY,
            )
            .expect("compute"),
            cause: ReestablishmentCauseValue::OtherFailure,
            target_cell_identity: TEST_CELL_IDENTITY,
        }
    }

    /// A `shortMAC-I` computed with the matching `K_RRCint` for a known
    /// `(C-RNTI, PCI)` yields an `RRCReestablishment` **on DL-DCCH** carrying the
    /// context's own `nextHopChainingCount`.
    #[test]
    fn a_verified_reestablishment_replies_on_dl_dcch_with_the_contexts_ncc() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 5);

        let result = conn_mgr
            .process_rrc_reestablishment_request(&mut ue_mgr, &request_for(&key(0xA1)))
            .expect("the shortMAC-I verifies");

        assert_eq!(result.ue_id, 1, "the stored context, not a fresh one");
        assert_eq!(
            result.channel,
            RrcChannel::DlDcch,
            "TS 38.331 §6.2.1: RRCReestablishment is an SRB1 message"
        );
        let decoded =
            decode_rrc_reestablishment(result.rrc_reestablishment_pdu.data()).expect("real UPER");
        assert_eq!(
            decoded.next_hop_chaining_count, 5,
            "the NCC comes from the security context, not a hardcoded 0"
        );
        assert_eq!(decoded.rrc_transaction_id, result.transaction_id);
    }

    /// A `shortMAC-I` computed with the wrong key does not verify, so §5.3.3.1's
    /// `RRCSetup` fallback applies and **no** fresh re-establishment context is
    /// fabricated: the stored one is untouched.
    #[test]
    fn an_invalid_short_mac_i_falls_back_to_setup() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 0);

        let mut request = request_for(&key(0xA1));
        request.short_mac_i ^= 0xFFFF;

        let rejection = conn_mgr
            .process_rrc_reestablishment_request(&mut ue_mgr, &request)
            .expect_err("a wrong MAC must not reestablish");

        assert_eq!(rejection, ReestablishmentRejection::MacVerificationFailed);
        assert!(rejection.falls_back_to_setup());
        assert!(
            ue_mgr.try_find_ue(1).expect("context").is_connected(),
            "the stored context is left alone, not transitioned"
        );
        assert_eq!(ue_mgr.count(), 1, "no fresh context was fabricated");
    }

    /// An unresolvable `(C-RNTI, PCI)` also falls back — the identity names
    /// nothing the network holds.
    #[test]
    fn an_unknown_identity_falls_back_to_setup() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 0);

        for (label, mut request) in [
            ("a different PCI", request_for(&key(0xA1))),
            ("a different C-RNTI", request_for(&key(0xA1))),
        ] {
            if label == "a different PCI" {
                request.phys_cell_id = TEST_PCI + 1;
            } else {
                request.c_rnti = SIMULATED_C_RNTI + 1;
            }
            let rejection = conn_mgr
                .process_rrc_reestablishment_request(&mut ue_mgr, &request)
                .unwrap_err();
            assert_eq!(
                rejection,
                ReestablishmentRejection::UnknownIdentity,
                "{label} must not resolve"
            );
        }
    }

    /// A context with no AS security recorded cannot be verified, so it is not
    /// even a candidate: an unverifiable context means `RRCSetup`.
    #[test]
    fn a_context_without_as_security_is_not_a_candidate() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        conn_mgr.process_rrc_setup_request(&mut ue_mgr, 1, 0x1234567890, false, 3);

        assert!(ue_mgr
            .candidates_for_reestablishment(SIMULATED_C_RNTI, TEST_PCI)
            .is_empty());
        assert_eq!(
            conn_mgr
                .process_rrc_reestablishment_request(&mut ue_mgr, &request_for(&key(0xA1)))
                .unwrap_err(),
            ReestablishmentRejection::UnknownIdentity
        );
    }

    /// The `(C-RNTI, PCI)` lookup returns candidates, because this simulator has
    /// no C-RNTI allocation and every UE presents the same constant. The
    /// `shortMAC-I` is what resolves them, and it does: the second UE's request
    /// picks the second UE's context even though the first matched the identity.
    #[test]
    fn the_short_mac_i_resolves_two_candidates_sharing_one_identity() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 1);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 2, key(0xB2), 2);

        assert_eq!(
            ue_mgr.candidates_for_reestablishment(SIMULATED_C_RNTI, TEST_PCI),
            vec![1, 2],
            "both UEs match the presented identity"
        );

        let result = conn_mgr
            .process_rrc_reestablishment_request(&mut ue_mgr, &request_for(&key(0xB2)))
            .expect("UE 2's MAC verifies");
        assert_eq!(
            result.ue_id, 2,
            "the shortMAC-I selected the context whose K_RRCint produced it"
        );
        let decoded =
            decode_rrc_reestablishment(result.rrc_reestablishment_pdu.data()).expect("UPER");
        assert_eq!(decoded.next_hop_chaining_count, 2, "UE 2's NCC");
    }

    /// A barred cell answers nothing at all — not even an `RRCSetup`.
    #[test]
    fn a_barred_cell_answers_a_reestablishment_with_nothing() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 0);
        conn_mgr.set_barred(true);

        let rejection = conn_mgr
            .process_rrc_reestablishment_request(&mut ue_mgr, &request_for(&key(0xA1)))
            .unwrap_err();

        assert_eq!(rejection, ReestablishmentRejection::CellBarred);
        assert!(!rejection.falls_back_to_setup());
    }

    /// The echoed transaction id is verified: a mismatch is discarded rather than
    /// completing the procedure.
    #[test]
    fn a_mismatched_completion_transaction_id_is_discarded() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        conn_mgr.set_barred(false);
        connected_ue_with_security(&mut conn_mgr, &mut ue_mgr, 1, key(0xA1), 0);

        let result = conn_mgr
            .process_rrc_reestablishment_request(&mut ue_mgr, &request_for(&key(0xA1)))
            .expect("verifies");
        let tid = result.transaction_id;

        assert!(
            conn_mgr
                .process_rrc_reestablishment_complete(&mut ue_mgr, 1, tid.wrapping_add(1) % 4)
                .is_none(),
            "a completion echoing the wrong tid is discarded"
        );
        assert!(
            !ue_mgr.try_find_ue(1).expect("context").is_connected(),
            "and the context is not moved to Connected"
        );

        assert!(
            conn_mgr
                .process_rrc_reestablishment_complete(&mut ue_mgr, 1, tid)
                .is_some(),
            "the right tid completes it"
        );
        assert!(ue_mgr.try_find_ue(1).expect("context").is_connected());
    }
}
