//! Handover Handling for UE
//!
//! Implements handover procedure per 3GPP TS 38.331.
//!
//! # Handover Procedure
//!
//! The handover procedure is initiated by the network through an RRC Reconfiguration
//! message containing a handover command (reconfigurationWithSync).
//!
//! ## Intra-frequency Handover Steps:
//! 1. UE receives RRCReconfiguration with reconfigurationWithSync
//! 2. UE synchronizes with target cell
//! 3. UE sends RRCReconfigurationComplete to target cell
//! 4. Handover complete
//!
//! ## Handover Failure Handling:
//! - If T304 expires: Handover failure
//! - If sync with target cell fails: Handover failure
//! - On failure: UE initiates RRC re-establishment
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification, Section 5.3.5

use std::time::{Duration, Instant};

/// Handover states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoverState {
    /// No handover in progress
    Idle,
    /// Handover preparation - received RRC Reconfiguration
    Preparing,
    /// Synchronizing with target cell
    Synchronizing,
    /// Handover complete - waiting for confirmation
    Completing,
    /// Handover failed
    Failed,
}

impl Default for HandoverState {
    fn default() -> Self {
        Self::Idle
    }
}

/// Handover failure cause
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandoverFailureCause {
    /// T304 timer expired
    T304Expired,
    /// Failed to sync with target cell
    SyncFailure,
    /// Target cell not reachable
    TargetCellUnreachable,
    /// Invalid reconfiguration
    InvalidReconfiguration,
}

/// Target cell information for handover
#[derive(Debug, Clone, Default)]
pub struct TargetCellInfo {
    /// Physical cell ID
    pub pci: u32,
    /// Cell ID (NCI)
    pub cell_id: i32,
    /// New C-RNTI assigned by target cell
    pub new_ue_id: Option<i32>,
    /// Target cell ARFCN (frequency)
    pub arfcn: Option<u32>,
    /// SSB subcarrier offset
    pub ssb_offset: Option<u8>,
}

/// Handover command extracted from RRC Reconfiguration
#[derive(Debug, Clone, Default)]
pub struct HandoverCommand {
    /// Target cell information
    pub target_cell: TargetCellInfo,
    /// New security configuration
    pub new_security_config: bool,
    /// Full reconfiguration required
    pub full_config: bool,
    /// Transaction ID from RRC Reconfiguration
    pub transaction_id: u8,
}

/// Handover manager for UE
pub struct HandoverManager {
    /// Current handover state
    state: HandoverState,
    /// Current handover command (if any)
    command: Option<HandoverCommand>,
    /// Source cell ID
    source_cell_id: Option<i32>,
    /// T304 timer start time
    t304_start: Option<Instant>,
    /// T304 timer duration (default 100ms per 3GPP)
    t304_duration: Duration,
    /// Handover start time
    ho_start_time: Option<Instant>,
    /// Handover complete time
    ho_complete_time: Option<Instant>,
}

impl HandoverManager {
    pub fn new() -> Self {
        Self {
            state: HandoverState::Idle,
            command: None,
            source_cell_id: None,
            t304_start: None,
            t304_duration: Duration::from_millis(100),
            ho_start_time: None,
            ho_complete_time: None,
        }
    }

    /// Get current handover state
    pub fn state(&self) -> HandoverState {
        self.state
    }

    /// Check if handover is in progress
    pub fn is_in_progress(&self) -> bool {
        !matches!(self.state, HandoverState::Idle | HandoverState::Failed)
    }

    /// Get the target cell info if handover is in progress
    pub fn target_cell(&self) -> Option<&TargetCellInfo> {
        self.command.as_ref().map(|c| &c.target_cell)
    }

    /// Get the source cell ID
    pub fn source_cell_id(&self) -> Option<i32> {
        self.source_cell_id
    }

    /// Start handover procedure
    pub fn start_handover(&mut self, source_cell: i32, command: HandoverCommand) {
        tracing::info!(
            "Starting handover: source_cell={}, target_pci={}, target_cell_id={}",
            source_cell, command.target_cell.pci, command.target_cell.cell_id
        );

        self.source_cell_id = Some(source_cell);
        self.command = Some(command);
        self.state = HandoverState::Preparing;
        self.ho_start_time = Some(Instant::now());
    }

    /// Transition to synchronizing state
    pub fn start_synchronization(&mut self) {
        if self.state == HandoverState::Preparing {
            tracing::debug!("Starting synchronization with target cell");
            self.state = HandoverState::Synchronizing;
            self.t304_start = Some(Instant::now());
        }
    }

    /// Called when synchronization with target cell is complete
    pub fn sync_complete(&mut self) {
        if self.state == HandoverState::Synchronizing {
            tracing::debug!("Synchronization complete, transitioning to completing");
            self.state = HandoverState::Completing;
        }
    }

    /// Complete the handover
    pub fn complete(&mut self) -> Option<i32> {
        if matches!(self.state, HandoverState::Synchronizing | HandoverState::Completing) {
            self.state = HandoverState::Idle;
            self.t304_start = None;
            self.ho_complete_time = Some(Instant::now());

            let target_cell_id = self.command.as_ref().map(|c| c.target_cell.cell_id);

            if let Some(start) = self.ho_start_time {
                tracing::info!(
                    "Handover complete: target_cell_id={:?}, duration={:?}",
                    target_cell_id, start.elapsed()
                );
            }

            self.command = None;
            self.source_cell_id = None;
            self.ho_start_time = None;

            return target_cell_id;
        }
        None
    }

    /// Fail the handover
    pub fn fail(&mut self, cause: HandoverFailureCause) -> Option<i32> {
        tracing::warn!("Handover failed: {:?}", cause);
        self.state = HandoverState::Failed;
        self.t304_start = None;
        self.command = None;

        let source = self.source_cell_id;
        self.source_cell_id = None;
        self.ho_start_time = None;

        source
    }

    /// Reset handover state
    pub fn reset(&mut self) {
        self.state = HandoverState::Idle;
        self.command = None;
        self.source_cell_id = None;
        self.t304_start = None;
        self.ho_start_time = None;
    }

    /// Check if T304 timer has expired
    pub fn check_t304_expired(&mut self) -> bool {
        if let Some(start) = self.t304_start {
            if start.elapsed() >= self.t304_duration {
                tracing::warn!("T304 timer expired during handover");
                self.fail(HandoverFailureCause::T304Expired);
                return true;
            }
        }
        false
    }

    /// Set T304 timer duration (from RRC Reconfiguration)
    pub fn set_t304_duration(&mut self, duration: Duration) {
        self.t304_duration = duration;
    }

    /// Get last handover duration (if completed)
    pub fn last_handover_duration(&self) -> Option<Duration> {
        match (self.ho_complete_time, self.ho_start_time) {
            (Some(end), Some(start)) => {
                if end > start {
                    Some(end - start)
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

impl Default for HandoverManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse handover command from RRC Reconfiguration PDU (simplified)
///
/// In a real implementation, this would use ASN.1 decoding.
/// This simplified version extracts basic handover info from a simplified format.
pub fn parse_handover_command(pdu: &[u8]) -> Option<HandoverCommand> {
    // Simplified format (not real ASN.1):
    // [0] = message type (0x00 = RRC Reconfiguration with HO)
    // [1] = transaction_id
    // [2-3] = target PCI (big endian)
    // [4-7] = target cell ID (big endian)
    // [8] = flags (0x01 = has new_ue_id, 0x02 = full_config)
    // [9-12] = new_ue_id (if flag set)

    if pdu.len() < 9 {
        return None;
    }

    // Check if this is a handover reconfiguration
    if pdu[0] != 0x00 {
        return None;
    }

    let transaction_id = pdu[1];
    let target_pci = u16::from_be_bytes([pdu[2], pdu[3]]) as u32;
    let target_cell_id = i32::from_be_bytes([pdu[4], pdu[5], pdu[6], pdu[7]]);
    let flags = pdu[8];

    let new_ue_id = if flags & 0x01 != 0 && pdu.len() >= 13 {
        Some(i32::from_be_bytes([pdu[9], pdu[10], pdu[11], pdu[12]]))
    } else {
        None
    };

    Some(HandoverCommand {
        target_cell: TargetCellInfo {
            pci: target_pci,
            cell_id: target_cell_id,
            new_ue_id,
            arfcn: None,
            ssb_offset: None,
        },
        new_security_config: false,
        full_config: flags & 0x02 != 0,
        transaction_id,
    })
}

/// Build RRC Reconfiguration Complete PDU (simplified)
pub fn build_reconfiguration_complete(transaction_id: u8) -> Vec<u8> {
    // Simplified format:
    // [0] = message type (0x08 = RRC Reconfiguration Complete)
    // [1] = transaction_id
    vec![0x08, transaction_id]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handover_manager_creation() {
        let manager = HandoverManager::new();
        assert_eq!(manager.state(), HandoverState::Idle);
        assert!(!manager.is_in_progress());
    }

    #[test]
    fn test_handover_lifecycle() {
        let mut manager = HandoverManager::new();

        // Start handover
        let command = HandoverCommand {
            target_cell: TargetCellInfo {
                pci: 1,
                cell_id: 100,
                new_ue_id: Some(1),
                arfcn: None,
                ssb_offset: None,
            },
            new_security_config: false,
            full_config: false,
            transaction_id: 1,
        };

        manager.start_handover(1, command);
        assert_eq!(manager.state(), HandoverState::Preparing);
        assert!(manager.is_in_progress());

        // Start sync
        manager.start_synchronization();
        assert_eq!(manager.state(), HandoverState::Synchronizing);

        // Complete sync
        manager.sync_complete();
        assert_eq!(manager.state(), HandoverState::Completing);

        // Complete handover
        let target = manager.complete();
        assert_eq!(target, Some(100));
        assert_eq!(manager.state(), HandoverState::Idle);
        assert!(!manager.is_in_progress());
    }

    #[test]
    fn test_handover_failure() {
        let mut manager = HandoverManager::new();

        let command = HandoverCommand {
            target_cell: TargetCellInfo {
                pci: 1,
                cell_id: 100,
                ..Default::default()
            },
            ..Default::default()
        };

        manager.start_handover(50, command);
        manager.start_synchronization();

        let source = manager.fail(HandoverFailureCause::SyncFailure);
        assert_eq!(source, Some(50));
        assert_eq!(manager.state(), HandoverState::Failed);
    }

    #[test]
    fn test_parse_handover_command() {
        // Build a test handover command
        let pdu = vec![
            0x00, // message type
            0x05, // transaction_id
            0x00, 0x10, // target PCI = 16
            0x00, 0x00, 0x00, 0x64, // target cell ID = 100
            0x01, // flags (has new_ue_id)
            0x00, 0x00, 0x00, 0x01, // new_ue_id = 1
        ];

        let cmd = parse_handover_command(&pdu).unwrap();
        assert_eq!(cmd.transaction_id, 5);
        assert_eq!(cmd.target_cell.pci, 16);
        assert_eq!(cmd.target_cell.cell_id, 100);
        assert_eq!(cmd.target_cell.new_ue_id, Some(1));
    }

    #[test]
    fn test_build_reconfiguration_complete() {
        let pdu = build_reconfiguration_complete(5);
        assert_eq!(pdu, vec![0x08, 0x05]);
    }
}
