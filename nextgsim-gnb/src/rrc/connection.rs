//! RRC Connection Management
//!
//! This module implements RRC connection procedures for the gNB:
//! - RRC Setup procedure (RRCSetupRequest → RRCSetup → RRCSetupComplete)
//! - RRC Release procedure
//! - RRC Reconfiguration procedure
//! - Security Mode Command procedure

use tracing::{debug, info, warn};

use nextgsim_common::OctetString;
use nextgsim_rls::RrcChannel;

use crate::tasks::GutiMobileIdentity;
use super::ue_context::RrcUeContextManager;

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

/// RRC connection manager
#[derive(Debug)]
pub struct RrcConnectionManager {
    /// Transaction ID counter
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

    /// Gets the next transaction ID (cycles 0-3)
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

        let transaction_id = self.next_tid();

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
        _transaction_id: u8,
        nas_pdu: OctetString,
        s_tmsi_value: Option<GutiMobileIdentity>,
    ) -> Option<RrcSetupCompleteResult> {
        let ctx = ue_mgr.try_find_ue_mut(ue_id)?;

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

    /// Builds an RRC Setup message (simplified)
    fn build_rrc_setup(&self, transaction_id: u8) -> OctetString {
        // Simplified RRC Setup message
        // In a full implementation, this would use nextgsim_rrc::procedures::rrc_setup
        // For now, create a minimal placeholder
        let mut pdu = Vec::with_capacity(16);
        
        // DL-CCCH-Message with RRCSetup
        // This is a simplified encoding - real implementation would use ASN.1
        pdu.push(0x20); // c1 choice = rrcSetup
        pdu.push(transaction_id); // rrc-TransactionIdentifier
        pdu.push(0x00); // criticalExtensions = rrcSetup
        
        // Minimal RadioBearerConfig
        pdu.extend_from_slice(&[0x00, 0x00]);
        
        // Minimal MasterCellGroup
        pdu.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]);
        
        OctetString::from_slice(&pdu)
    }

    /// Builds an RRC Release message (simplified)
    fn build_rrc_release(&self, transaction_id: u8) -> OctetString {
        // Simplified RRC Release message
        let mut pdu = Vec::with_capacity(8);
        
        // DL-DCCH-Message with RRCRelease
        pdu.push(0x0D); // c1 choice = rrcRelease
        pdu.push(transaction_id); // rrc-TransactionIdentifier
        pdu.push(0x00); // criticalExtensions = rrcRelease
        
        OctetString::from_slice(&pdu)
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
        
        let result = conn_mgr.process_rrc_setup_request(
            &mut ue_mgr,
            1,
            0x1234567890,
            false,
            3,
        );
        
        assert!(result.is_some());
        let result = result.unwrap();
        assert_eq!(result.ue_id, 1);
        assert_eq!(result.channel, RrcChannel::DlCcch);
        
        // Verify UE context was created
        let ctx = ue_mgr.try_find_ue(1).unwrap();
        assert_eq!(ctx.initial_id, Some(0x1234567890));
        assert!(!ctx.is_initial_id_s_tmsi);
    }

    #[test]
    fn test_rrc_setup_request_duplicate() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        
        conn_mgr.set_barred(false);
        
        // First request succeeds
        let result1 = conn_mgr.process_rrc_setup_request(
            &mut ue_mgr,
            1,
            0x1234567890,
            false,
            3,
        );
        assert!(result1.is_some());
        
        // Second request for same UE fails
        let result2 = conn_mgr.process_rrc_setup_request(
            &mut ue_mgr,
            1,
            0x1234567890,
            false,
            3,
        );
        assert!(result2.is_none());
    }

    #[test]
    fn test_rrc_setup_complete() {
        let mut conn_mgr = RrcConnectionManager::new();
        let mut ue_mgr = RrcUeContextManager::new();
        
        conn_mgr.set_barred(false);
        
        // Setup request
        conn_mgr.process_rrc_setup_request(
            &mut ue_mgr,
            1,
            0x1234567890,
            false,
            3,
        );
        
        // Setup complete
        let nas_pdu = OctetString::from_slice(&[0x7E, 0x00, 0x41]);
        let result = conn_mgr.process_rrc_setup_complete(
            &mut ue_mgr,
            1,
            0,
            nas_pdu.clone(),
            None,
        );
        
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
