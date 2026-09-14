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

use nextgsim_rrc::procedures::rrc_reconfiguration::decode_handover_command;
use std::time::{Duration, Instant};

/// Handover states
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HandoverState {
    /// No handover in progress
    #[default]
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
    /// The UE's own local index for the target cell, resolved from the
    /// `physCellId` the command carried.
    ///
    /// `None` when the UE cannot match the PCI to a cell it can hear. Not a
    /// sentinel like 0, because cell index 0 is a real cell here — the same
    /// reasoning that made `CellSelector::current_cell` an `Option` (issue #30).
    pub cell_id: Option<i32>,
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
    /// The `masterKeyUpdate` the command carried, if any (issue #39).
    ///
    /// `None` means the UE keeps its current keys. This replaced a
    /// `new_security_config: bool` that was hardcoded `false` and read by nothing —
    /// a flag saying "re-key" with no derivation behind it is worse than no flag.
    pub key_update: Option<KeyUpdate>,
    /// Full reconfiguration required
    pub full_config: bool,
    /// Transaction ID from RRC Reconfiguration
    pub transaction_id: u8,
}

/// What a `masterKeyUpdate` tells the UE to do (TS 38.331 §5.3.5.7,
/// TS 33.501 §6.9.2.3.1; issue #39).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyUpdate {
    /// `keySetChangeIndicator`: `true` for a **vertical** derivation from a fresh NH,
    /// `false` for a **horizontal** one from the UE's current `KgNB`.
    ///
    /// A named field rather than a bare bool at the call site, because only the vertical
    /// case gives forward security against a compromised source gNB — and the two derive
    /// *different* keys, so guessing makes every PDCP MAC on the target fail.
    pub vertical: bool,
    /// `nextHopChainingCount` the derivation chains on.
    pub next_hop_chaining_count: u8,
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
    /// How long the last completed handover took.
    ///
    /// Recorded at completion rather than derived from the two timestamps above:
    /// completing a handover clears `ho_start_time`, so a derived duration was
    /// always `None` and the accessor could never report one.
    last_duration: Option<Duration>,
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
            last_duration: None,
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
            "Starting handover: source_cell={}, target_pci={}, target_cell_id={:?}",
            source_cell,
            command.target_cell.pci,
            command.target_cell.cell_id
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
        if matches!(
            self.state,
            HandoverState::Synchronizing | HandoverState::Completing
        ) {
            self.state = HandoverState::Idle;
            self.t304_start = None;
            self.ho_complete_time = Some(Instant::now());

            // Flattened: an unresolved PCI (`cell_id: None`) and no command at all
            // both mean "no target cell to move to", and the caller treats them the
            // same way. Keeping them nested as Option<Option<i32>> would only invite
            // a `.flatten()` at every call site.
            let target_cell_id = self.command.as_ref().and_then(|c| c.target_cell.cell_id);

            if let Some(start) = self.ho_start_time {
                self.last_duration = Some(start.elapsed());
                tracing::info!(
                    "Handover complete: target_cell_id={:?}, duration={:?}",
                    target_cell_id,
                    start.elapsed()
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

    /// How long the last completed handover took, or `None` if none has
    /// completed since the manager was created.
    pub fn last_handover_duration(&self) -> Option<Duration> {
        self.last_duration
    }
}

impl Default for HandoverManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Parses a handover command: an `RRCReconfiguration` whose `masterCellGroup`
/// carries a `reconfigurationWithSync` (TS 38.331 §5.3.5.5.2, issue #107).
///
/// Replaces a hand-rolled byte parser whose comment said "In a real
/// implementation, this would use ASN.1 decoding". The gNB's encoder flipped in
/// the **same change** — a one-sided flip means the UE reads a UPER message as the
/// old byte layout, and `pdu[0] != 0x00` would make every handover command look
/// like "not a handover".
///
/// Returns `None` for an `RRCReconfiguration` that is not a handover command
/// (no `reconfigurationWithSync`), which is how the caller distinguishes it from a
/// DRB-establishing reconfiguration. Both are DL-DCCH `RRCReconfiguration`s.
///
/// # `cell_id` is resolved by the caller, not read off the wire
///
/// `TargetCellInfo::cell_id` used to come from four bytes the gNB put in its
/// bespoke format. `reconfigurationWithSync` has no field for it, because the
/// simulator's cell index is not a 3GPP identity — the target is named by
/// `physCellId`. So this returns `cell_id: None` and the caller resolves the PCI
/// against the cells the UE can actually hear. A UE that cannot find the PCI has
/// learned something true: it cannot reach the target.
pub fn parse_handover_command(pdu: &[u8]) -> Option<HandoverCommand> {
    let decoded = decode_handover_command(pdu)?;
    Some(HandoverCommand {
        target_cell: TargetCellInfo {
            pci: u32::from(decoded.target_phys_cell_id),
            // Unresolved: see the note above.
            cell_id: None,
            new_ue_id: Some(i32::from(decoded.new_ue_identity)),
            // The target's frequency is absent from the command by design (this
            // simulator is single-carrier), so the UE stays on its own.
            arfcn: None,
            ssb_offset: None,
        },
        // `masterKeyUpdate` is the IE that says the UE re-keys, and it IS in the command
        // now (issue #39). Its `keySetChangeIndicator` decides vertical vs horizontal,
        // which is a security property and not a flag -- see `KeyUpdate`.
        key_update: decoded.master_key_update.map(|u| KeyUpdate {
            vertical: u.key_set_change_indicator,
            next_hop_chaining_count: u.next_hop_chaining_count,
        }),
        full_config: decoded.full_config,
        transaction_id: decoded.rrc_transaction_id,
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
                cell_id: Some(100),
                new_ue_id: Some(1),
                arfcn: None,
                ssb_offset: None,
            },
            key_update: None,
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

    /// The duration accessor reports a completed handover. It used to derive the
    /// duration from `ho_start_time`, which `complete` clears, so it always
    /// answered `None`.
    #[test]
    fn a_completed_handover_reports_its_duration() {
        let mut manager = HandoverManager::new();
        assert_eq!(manager.last_handover_duration(), None);

        manager.start_handover(
            1,
            HandoverCommand {
                target_cell: TargetCellInfo {
                    pci: 2,
                    cell_id: Some(2),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        manager.start_synchronization();
        manager.sync_complete();
        assert_eq!(manager.complete(), Some(2));

        assert!(
            manager.last_handover_duration().is_some(),
            "a completed handover has a duration"
        );
    }

    #[test]
    fn test_handover_failure() {
        let mut manager = HandoverManager::new();

        let command = HandoverCommand {
            target_cell: TargetCellInfo {
                pci: 1,
                cell_id: Some(100),
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

    /// #107, criterion 3: the UE parses a real `RRCReconfiguration` with
    /// `reconfigurationWithSync`, built by the **shared encoder the gNB uses**.
    ///
    /// This test used to hand-build the byte layout the gNB used to emit. Rewritten
    /// rather than deleted: the same facts, off the conformant wire. The bytes come
    /// from `nextgsim-rrc`'s `encode_handover_command`, which is exactly what the
    /// gNB calls — so a one-sided change to either end fails here.
    #[test]
    fn test_parse_handover_command() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            encode_handover_command, HandoverCommandParams,
        };

        let pdu = encode_handover_command(&HandoverCommandParams {
            rrc_transaction_id: 3,
            target_phys_cell_id: 16,
            new_ue_identity: 1,
            t304_ms: 1000,
            full_config: true,
            master_key_update: None,
        })
        .expect("the shared encoder must produce a handover command");

        let cmd = parse_handover_command(&pdu).expect("and the UE must parse it");
        assert_eq!(cmd.transaction_id, 3);
        assert_eq!(cmd.target_cell.pci, 16, "the target is named by physCellId");
        assert_eq!(
            cmd.target_cell.cell_id, None,
            "the simulator's cell index is NOT on the wire; the caller resolves the \
             PCI against the cells the UE can hear"
        );
        assert_eq!(cmd.target_cell.new_ue_id, Some(1));
        assert!(cmd.full_config);
        assert_eq!(
            cmd.key_update, None,
            "no masterKeyUpdate was asked for, so the UE keeps its keys"
        );
    }

    /// #39: a `masterKeyUpdate` on the wire reaches the parsed command, with the
    /// `keySetChangeIndicator` that decides vertical vs horizontal.
    ///
    /// The IE used to be absent from every command this gNB built, so
    /// `new_security_config` was hardcoded `false` and read by nothing.
    #[test]
    fn a_master_key_update_reaches_the_parsed_handover_command() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            encode_handover_command, HandoverCommandParams, MasterKeyUpdateParams,
        };
        for (vertical, ncc) in [(false, 0u8), (true, 5u8)] {
            let pdu = encode_handover_command(&HandoverCommandParams {
                rrc_transaction_id: 1,
                target_phys_cell_id: 407,
                new_ue_identity: 1,
                t304_ms: 1000,
                full_config: true,
                master_key_update: Some(MasterKeyUpdateParams {
                    key_set_change_indicator: vertical,
                    next_hop_chaining_count: ncc,
                }),
            })
            .expect("encode");
            let cmd = parse_handover_command(&pdu).expect("parse");
            assert_eq!(
                cmd.key_update,
                Some(KeyUpdate {
                    vertical,
                    next_hop_chaining_count: ncc
                }),
                "vertical={vertical} NCC={ncc}: both fields must survive, because the \
                 two chainings derive DIFFERENT keys"
            );
        }
    }

    /// The old byte format must not parse. A UE that still accepted it would let a
    /// half-flipped pair keep working, which is what criterion 4's "flipped in the
    /// same change" exists to prevent.
    #[test]
    fn the_legacy_byte_format_handover_command_no_longer_parses() {
        let legacy = vec![
            0x00, 0x05, 0x00, 0x10, 0x00, 0x00, 0x00, 0x64, 0x01, 0x00, 0x00, 0x00, 0x01,
        ];
        assert!(parse_handover_command(&legacy).is_none());
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
            tracing::debug!(
                "DAPS: Starting synchronization with target cell PCI={}",
                self.target_cell.pci
            );
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
        if matches!(
            self.state,
            HandoverState::Preparing | HandoverState::Synchronizing | HandoverState::Completing
        ) {
            self.state = HandoverState::Idle;
            self.ho_complete_time = Some(Instant::now());

            if let Some(start) = self.ho_start_time {
                self.last_duration = Some(start.elapsed());
                tracing::info!("DAPS handover complete: duration={:?}", start.elapsed());
            }

            self.command = None;
            self.source_cell_id = None;
            self.ho_start_time = None;
            return Some(0); // Placeholder target cell
        }
        None
    }
}

/// Parses a DAPS reconfiguration: the real `RRCReconfiguration` with
/// `reconfigurationWithSync` the gNB now emits (issue #107).
///
/// Replaces a 13-byte hand-rolled parser keyed on a `0x10` leading byte.
///
/// # The SOURCE half is a parameter, because it is not on the wire
///
/// DAPS keeps the source link up, so a conformant DAPS reconfiguration carries
/// `daps-SourceRelease` and a per-DRB `daps-Config` — **neither of which is in the
/// Rel-15 schema this tree compiles** (issue #105). The old byte format invented
/// fields for the source PCI, source C-RNTI and a data-forwarding flag, and no
/// conformant peer would ever have sent them.
///
/// So the source half comes from where the UE already knows it: the cell it is
/// camped on. That is not a workaround, it is what "source cell" means — the UE
/// does not need to be told which cell it is currently on.
///
/// Returns `None` for a reconfiguration that is not a handover command.
pub fn parse_daps_reconfiguration(
    pdu: &[u8],
    source_pci: u32,
    source_cell_id: i32,
    source_crnti: u16,
) -> Option<UeDapsContext> {
    let command = decode_handover_command(pdu)?;
    Some(UeDapsContext::new(
        source_pci,
        source_cell_id,
        source_crnti,
        u32::from(command.target_phys_cell_id),
        // The target's local cell index is unknown here for the same reason it is
        // in `parse_handover_command`: the wire names the target by PCI. The PCI is
        // used as the index, which is what the old parser did too -- but now it is
        // the ONLY identity the message carried, rather than one of two that could
        // disagree.
        command.target_phys_cell_id as i32,
        command.new_ue_identity,
        u64::from(command.t304_ms),
    ))
}
