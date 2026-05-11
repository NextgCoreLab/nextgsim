//! DAPS (Dual Active Protocol Stack) Handover for UE (Rel-17, TS 38.331 §5.4.2.3)
//!
//! Implements UE-side DAPS procedure:
//! - Dual RLC operation during handover (source UL + target DL)
//! - RRC Reconfiguration with DAPS indicator processing
//! - Source cell release after target synchronization

/// DAPS handover state from UE perspective
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UeDapsState {
    #[default]
    /// DAPS not active
    Inactive,
    /// RRC Reconfiguration with DAPS indicator received, applying config
    Configuring,
    /// Connected to both source (UL) and target (DL)
    DualConnected,
    /// Target cell confirmed; releasing source RLC
    ReleasingSource,
    /// DAPS complete — single connection at target
    Complete,
    /// DAPS failed — reverted to source
    Failed,
}

/// DAPS RLC configuration for a DRB
#[derive(Debug, Clone)]
pub struct DapsRlcConfig {
    /// DRB identity
    pub drb_id: u8,
    /// Source cell C-RNTI
    pub source_c_rnti: u16,
    /// Target cell C-RNTI (from RRC Reconfiguration)
    pub target_c_rnti: u16,
    /// Whether source RLC is still transmitting UL
    pub source_ul_active: bool,
    /// Whether target RLC is receiving DL
    pub target_dl_active: bool,
}

impl DapsRlcConfig {
    pub fn new(drb_id: u8, source_c_rnti: u16, target_c_rnti: u16) -> Self {
        Self {
            drb_id,
            source_c_rnti,
            target_c_rnti,
            source_ul_active: true,
            target_dl_active: false,
        }
    }

    /// Activates target DL (after RRC Reconfiguration Complete at target)
    pub fn activate_target(&mut self) {
        self.target_dl_active = true;
    }

    /// Releases source UL (after data forwarding complete)
    pub fn release_source(&mut self) {
        self.source_ul_active = false;
    }
}

/// UE DAPS context
#[derive(Debug, Default)]
pub struct UeDapsContext {
    /// Current state
    pub state: UeDapsState,
    /// Source cell ID
    pub source_cell_id: i32,
    /// Target cell ID
    pub target_cell_id: i32,
    /// DRB configurations in DAPS mode
    pub drb_configs: Vec<DapsRlcConfig>,
    /// Whether DAPS is indicated in the RRC Reconfiguration
    pub daps_indicated: bool,
}

impl UeDapsContext {
    pub fn new() -> Self {
        Self {
            state: UeDapsState::Inactive,
            ..Default::default()
        }
    }

    /// Processes an RRC Reconfiguration with DAPS indicator
    pub fn process_rrc_reconfiguration(
        &mut self,
        source_cell_id: i32,
        target_cell_id: i32,
        drbs: Vec<DapsRlcConfig>,
    ) {
        self.source_cell_id = source_cell_id;
        self.target_cell_id = target_cell_id;
        self.drb_configs = drbs;
        self.daps_indicated = true;
        self.state = UeDapsState::Configuring;
    }

    /// Called after successful synchronization with target cell
    pub fn on_target_sync_success(&mut self) {
        self.state = UeDapsState::DualConnected;
        for drb in &mut self.drb_configs {
            drb.activate_target();
        }
    }

    /// Called when source cell UL is no longer needed
    pub fn release_source(&mut self) {
        self.state = UeDapsState::ReleasingSource;
        for drb in &mut self.drb_configs {
            drb.release_source();
        }
    }

    /// Called when DAPS is fully complete
    pub fn complete(&mut self) {
        self.state = UeDapsState::Complete;
    }

    /// Called when DAPS fails (reverts to source)
    pub fn fail_and_revert(&mut self) {
        self.state = UeDapsState::Failed;
        for drb in &mut self.drb_configs {
            drb.source_ul_active = true;
            drb.target_dl_active = false;
        }
    }

    /// Returns true if UL data should go to source cell
    pub fn use_source_for_ul(&self) -> bool {
        matches!(
            self.state,
            UeDapsState::DualConnected | UeDapsState::ReleasingSource
        )
    }

    /// Returns true if DL data should come from target cell
    pub fn use_target_for_dl(&self) -> bool {
        self.state == UeDapsState::DualConnected
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_drbs() -> Vec<DapsRlcConfig> {
        vec![
            DapsRlcConfig::new(1, 0x0100, 0x0200),
            DapsRlcConfig::new(2, 0x0100, 0x0200),
        ]
    }

    #[test]
    fn test_daps_lifecycle() {
        let mut ctx = UeDapsContext::new();
        assert_eq!(ctx.state, UeDapsState::Inactive);

        ctx.process_rrc_reconfiguration(1, 2, test_drbs());
        assert_eq!(ctx.state, UeDapsState::Configuring);

        ctx.on_target_sync_success();
        assert_eq!(ctx.state, UeDapsState::DualConnected);
        assert!(ctx.use_source_for_ul());
        assert!(ctx.use_target_for_dl());
        assert!(ctx.drb_configs[0].target_dl_active);

        ctx.release_source();
        assert_eq!(ctx.state, UeDapsState::ReleasingSource);
        assert!(!ctx.drb_configs[0].source_ul_active);

        ctx.complete();
        assert_eq!(ctx.state, UeDapsState::Complete);
    }

    #[test]
    fn test_daps_fail_and_revert() {
        let mut ctx = UeDapsContext::new();
        ctx.process_rrc_reconfiguration(1, 2, test_drbs());
        ctx.on_target_sync_success();
        ctx.fail_and_revert();
        assert_eq!(ctx.state, UeDapsState::Failed);
        assert!(ctx.drb_configs[0].source_ul_active);
        assert!(!ctx.drb_configs[0].target_dl_active);
    }

    #[test]
    fn test_dual_connected_routing() {
        let mut ctx = UeDapsContext::new();
        ctx.process_rrc_reconfiguration(1, 2, test_drbs());
        ctx.on_target_sync_success();
        assert!(ctx.use_source_for_ul());
        assert!(ctx.use_target_for_dl());
    }
}
