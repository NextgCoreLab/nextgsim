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
    rrc_release::{encode_rrc_release, RrcReleaseParams},
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
    /// Reestablishment cause
    pub cause: u8,
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

    /// Initiates an RRC Release for a UE
    pub fn initiate_rrc_release(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
    ) -> Option<RrcReleaseResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;

        if !ctx.is_connected() {
            debug!("UE[{}] not connected, skipping RRC Release", ue_id);
            return None;
        }

        let transaction_id = self.next_tid();

        // Build RRC Release message
        let rrc_release_pdu = self.build_rrc_release(transaction_id);

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

    /// Processes an RRC Reestablishment Request
    ///
    /// Called when a UE attempts to reestablish its RRC connection after
    /// radio link failure, handover failure, or integrity check failure.
    pub fn process_rrc_reestablishment_request(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        c_rnti: u16,
        phys_cell_id: u16,
        cause: u8,
    ) -> Option<RrcReestablishmentResult> {
        if self.is_barred {
            warn!("Rejecting RRC Reestablishment: cell is barred");
            return None;
        }

        // Try to find existing UE context by C-RNTI
        // If found, the UE is reestablishing on the same cell
        let existing = ue_mgr.try_find_ue(ue_id);
        if existing.is_none() {
            // Create a new context for this UE (may be reestablishing from another cell)
            let ctx = ue_mgr.create_ue(ue_id);
            ctx.on_setup_request();
        }

        let transaction_id = self.next_tid();
        let rrc_reestablishment_pdu = self.build_rrc_reestablishment(transaction_id);

        if let Some(ctx) = ue_mgr.try_find_ue_mut(ue_id) {
            ctx.on_setup_sent();
        }

        info!(
            "RRC Reestablishment for UE[{}], tid={}, c_rnti={}, phys_cell_id={}, cause={}",
            ue_id, transaction_id, c_rnti, phys_cell_id, cause
        );

        Some(RrcReestablishmentResult {
            ue_id,
            transaction_id,
            rrc_reestablishment_pdu,
            channel: RrcChannel::DlCcch,
            cause,
        })
    }

    /// Processes an RRC Reestablishment Complete
    pub fn process_rrc_reestablishment_complete(
        &mut self,
        ue_mgr: &mut RrcUeContextManager,
        ue_id: i32,
        _transaction_id: u8,
    ) -> Option<RrcReestablishmentCompleteResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;
        ctx.on_setup_complete();

        info!("RRC Reestablishment Complete for UE[{}]", ue_id);

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

    /// Builds an RRC Reestablishment message
    ///
    /// Note: The RRC procedures module provides UE-side RRCReestablishmentRequest
    /// and RRCReestablishmentComplete. The gNB-side RRCReestablishment (DL-CCCH)
    /// uses a simplified encoding since the ASN.1 gNB-side builder requires
    /// nextHopChainingCount from security context which is not yet wired.
    fn build_rrc_reestablishment(&self, transaction_id: u8) -> OctetString {
        let mut pdu = Vec::with_capacity(16);
        // DL-CCCH-Message with RRCReestablishment
        pdu.push(0x24); // c1 choice = rrcReestablishment
        pdu.push(transaction_id);
        pdu.push(0x00); // criticalExtensions = rrcReestablishment
                        // nextHopChainingCount (3 bits, set to 0)
        pdu.push(0x00);
        OctetString::from_slice(&pdu)
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
    fn build_rrc_release(&self, transaction_id: u8) -> OctetString {
        let params = RrcReleaseParams {
            rrc_transaction_id: transaction_id,
            cell_reselection_priorities: None,
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

        // Release
        let result = conn_mgr.initiate_rrc_release(&mut ue_mgr, 1);
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.ue_id, 1);
        assert_eq!(result.channel, RrcChannel::DlDcch);

        // Verify UE is now releasing
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert!(!ctx.is_connected());
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
}
