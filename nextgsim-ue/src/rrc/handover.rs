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
//! 1. UE receives `RRCReconfiguration` with reconfigurationWithSync
//! 2. UE synchronizes with target cell
//! 3. UE sends `RRCReconfigurationComplete` to target cell
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

// ============================================================================
// DAPS (Dual Active Protocol Stack) Handover Support (Rel-16)
// ============================================================================

/// UE-side DAPS handover context.
///
/// Maintains state for dual connection to both source and target cells during
/// make-before-break handover (zero interruption time).
///
/// Reference: 3GPP TS 38.331 Section 5.3.5.9
#[derive(Debug, Clone)]
pub struct UeDapsContext {
    /// Source cell connection info
    pub source_cell: DapsCellConnection,
    /// Target cell connection info
    pub target_cell: DapsCellConnection,
    /// DAPS handover state
    pub state: DapsState,
    /// T304daps timer start time
    pub t304_daps_start: Option<Instant>,
    /// T304daps duration
    pub t304_daps_duration: Duration,
    /// Data path currently active (Source or Target)
    pub active_data_path: DataPath,
}

/// DAPS cell connection information.
#[derive(Debug, Clone)]
pub struct DapsCellConnection {
    /// Physical cell ID
    pub pci: u32,
    /// Cell ID
    pub cell_id: i32,
    /// C-RNTI assigned by this cell
    pub crnti: u16,
    /// Whether this cell is synchronized
    pub synchronized: bool,
    /// Uplink data buffer size (bytes)
    pub ul_buffer_bytes: usize,
    /// Downlink data buffer size (bytes)
    pub dl_buffer_bytes: usize,
}

/// DAPS handover state for UE.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DapsState {
    /// No DAPS handover in progress
    Inactive,
    /// Received DAPS RRC Reconfiguration, preparing target cell
    Preparing,
    /// Both cells active, synchronizing with target
    DualActive,
    /// Synchronized with target, switching data path
    Switching,
    /// Complete, releasing source cell
    Complete,
    /// DAPS handover failed
    Failed,
}

/// Data path selection during DAPS handover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataPath {
    /// Data routed through source cell
    Source,
    /// Data routed through target cell
    Target,
    /// Both paths active (transition state)
    Both,
}

impl UeDapsContext {
    /// Creates a new UE DAPS context from RRC Reconfiguration.
    pub fn new(
        source_pci: u32,
        source_cell_id: i32,
        source_crnti: u16,
        target_pci: u32,
        target_cell_id: i32,
        target_crnti: u16,
        t304_daps_ms: u64,
    ) -> Self {
        Self {
            source_cell: DapsCellConnection {
                pci: source_pci,
                cell_id: source_cell_id,
                crnti: source_crnti,
                synchronized: true, // Source already synchronized
                ul_buffer_bytes: 0,
                dl_buffer_bytes: 0,
            },
            target_cell: DapsCellConnection {
                pci: target_pci,
                cell_id: target_cell_id,
                crnti: target_crnti,
                synchronized: false, // Need to sync with target
                ul_buffer_bytes: 0,
                dl_buffer_bytes: 0,
            },
            state: DapsState::Preparing,
            t304_daps_start: Some(Instant::now()),
            t304_daps_duration: Duration::from_millis(t304_daps_ms),
            active_data_path: DataPath::Source,
        }
    }

    /// Starts synchronization with target cell.
    pub fn start_target_sync(&mut self) {
        if self.state == DapsState::Preparing {
            tracing::debug!("DAPS: Starting synchronization with target cell PCI={}",
                          self.target_cell.pci);
            self.state = DapsState::DualActive;
        }
    }

    /// Marks target cell as synchronized.
    pub fn target_sync_complete(&mut self) {
        if self.state == DapsState::DualActive {
            tracing::info!("DAPS: Target cell synchronized, preparing for data path switch");
            self.target_cell.synchronized = true;
        }
    }

    /// Switches data path from source to target cell.
    ///
    /// Called after target sync is complete and RRC Reconfiguration Complete is sent.
    pub fn switch_to_target(&mut self) {
        if self.state == DapsState::DualActive && self.target_cell.synchronized {
            tracing::info!("DAPS: Switching data path from source to target cell");
            self.state = DapsState::Switching;
            self.active_data_path = DataPath::Both; // Temporarily use both
        }
    }

    /// Completes data path switch to target.
    pub fn complete_switch(&mut self) {
        if self.state == DapsState::Switching {
            tracing::debug!("DAPS: Data path switch complete, all data via target cell");
            self.active_data_path = DataPath::Target;
        }
    }

    /// Releases source cell resources.
    ///
    /// Called after all buffered data from source is delivered.
    pub fn release_source(&mut self) {
        if self.state == DapsState::Switching {
            tracing::info!("DAPS: Releasing source cell PCI={}", self.source_cell.pci);
            self.state = DapsState::Complete;
            self.source_cell.ul_buffer_bytes = 0;
            self.source_cell.dl_buffer_bytes = 0;
        }
    }

    /// Sends uplink data on source cell.
    ///
    /// Used while in DualActive state to maintain UL on source.
    pub fn send_ul_source(&mut self, data_bytes: usize) -> bool {
        if self.active_data_path == DataPath::Source || self.active_data_path == DataPath::Both {
            self.source_cell.ul_buffer_bytes += data_bytes;
            return true;
        }
        false
    }

    /// Sends uplink data on target cell.
    ///
    /// Used after switch to send UL via target.
    pub fn send_ul_target(&mut self, data_bytes: usize) -> bool {
        if self.active_data_path == DataPath::Target || self.active_data_path == DataPath::Both {
            self.target_cell.ul_buffer_bytes += data_bytes;
            return true;
        }
        false
    }

    /// Receives downlink data on source cell.
    pub fn receive_dl_source(&mut self, data_bytes: usize) -> bool {
        if self.active_data_path == DataPath::Source || self.active_data_path == DataPath::Both {
            self.source_cell.dl_buffer_bytes += data_bytes;
            return true;
        }
        false
    }

    /// Receives downlink data on target cell.
    pub fn receive_dl_target(&mut self, data_bytes: usize) -> bool {
        if self.active_data_path == DataPath::Target || self.active_data_path == DataPath::Both {
            self.target_cell.dl_buffer_bytes += data_bytes;
            return true;
        }
        false
    }

    /// Checks if T304daps timer has expired.
    pub fn check_t304_daps_expired(&self) -> bool {
        if let Some(start) = self.t304_daps_start {
            if self.state != DapsState::Inactive && self.state != DapsState::Complete {
                return start.elapsed() >= self.t304_daps_duration;
            }
        }
        false
    }

    /// Fails the DAPS handover.
    pub fn fail(&mut self) {
        tracing::warn!("DAPS handover failed");
        self.state = DapsState::Failed;
    }

    /// Completes and resets the DAPS context.
    pub fn complete(&mut self) {
        if let Some(start) = self.t304_daps_start {
            tracing::info!("DAPS handover complete: duration={:?}", start.elapsed());
        }
        self.state = DapsState::Complete;
    }
}

impl HandoverManager {
    /// Starts DAPS handover procedure.
    ///
    /// Called when UE receives RRC Reconfiguration with DAPS configuration.
    pub fn start_daps_handover(&mut self, daps_ctx: UeDapsContext) {
        tracing::info!(
            "Starting DAPS handover: source_cell={}, target_cell={}",
            daps_ctx.source_cell.cell_id,
            daps_ctx.target_cell.cell_id
        );

        // Store DAPS context separately (in real implementation)
        // For now, use regular handover state
        self.state = HandoverState::Preparing;
        self.ho_start_time = Some(Instant::now());
    }

    /// Completes DAPS handover after target cell is synchronized.
    pub fn complete_daps(&mut self) -> Option<i32> {
        if matches!(self.state, HandoverState::Preparing | HandoverState::Synchronizing | HandoverState::Completing) {
            self.state = HandoverState::Idle;
            self.ho_complete_time = Some(Instant::now());

            if let Some(start) = self.ho_start_time {
                tracing::info!(
                    "DAPS handover complete: duration={:?}",
                    start.elapsed()
                );
            }

            self.command = None;
            self.source_cell_id = None;
            self.ho_start_time = None;
            return Some(0); // Placeholder target cell
        }
        None
    }
}

/// Parses DAPS RRC Reconfiguration message (simplified format).
///
/// Simplified format:
/// [0] = message type (0x10 = RRC Reconfiguration with DAPS)
/// [1] = transaction_id
/// [2-3] = source PCI (big endian)
/// [4-5] = source C-RNTI (big endian)
/// [6-7] = target PCI (big endian)
/// [8-9] = target C-RNTI (big endian)
/// [10-11] = T304daps (big endian, ms)
/// [12] = flags (0x01 = data_forwarding_enabled)
pub fn parse_daps_reconfiguration(pdu: &[u8]) -> Option<UeDapsContext> {
    if pdu.len() < 13 || pdu[0] != 0x10 {
        return None;
    }

    let _transaction_id = pdu[1];
    let source_pci = u16::from_be_bytes([pdu[2], pdu[3]]) as u32;
    let source_crnti = u16::from_be_bytes([pdu[4], pdu[5]]);
    let target_pci = u16::from_be_bytes([pdu[6], pdu[7]]) as u32;
    let target_crnti = u16::from_be_bytes([pdu[8], pdu[9]]);
    let t304_daps_ms = u16::from_be_bytes([pdu[10], pdu[11]]) as u64;
    let _flags = pdu[12];

    Some(UeDapsContext::new(
        source_pci,
        source_pci as i32, // Simplified: use PCI as cell_id
        source_crnti,
        target_pci,
        target_pci as i32,
        target_crnti,
        t304_daps_ms,
    ))
}
