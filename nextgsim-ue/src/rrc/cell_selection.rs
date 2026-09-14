//! Cell Selection and Reselection
//!
//! Implements cell selection per 3GPP TS 38.304.
//!
//! # Cell Selection Criteria
//!
//! A cell is considered suitable if:
//! - The cell is not barred
//! - The cell is not reserved
//! - The cell belongs to the selected PLMN
//! - The TAI is not in the forbidden list
//! - Signal strength meets minimum requirements (Srxlev > 0)
//!
//! A cell is considered acceptable if:
//! - The cell is not barred
//! - The cell is not reserved
//! - The TAI is not in the forbidden list
//!   (PLMN matching is not required)
//!
//! # Reference
//! - 3GPP TS 38.304: NR; User Equipment (UE) procedures in Idle mode and RRC Inactive state
//! - UERANSIM: src/ue/rrc/idle.cpp

use nextgsim_rrc::procedures::rrc_reestablishment::phys_cell_id_from_nci;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Signal strength threshold below which a cell is considered lost (dBm)
pub const CELL_LOST_THRESHOLD_DBM: i32 = -120;

/// Minimum time after startup before cell selection starts (ms)
pub const CELL_SELECTION_STARTUP_DELAY_MS: u64 = 1000;

/// Time between cell selection failure logs (ms)
pub const CELL_SELECTION_LOG_INTERVAL_MS: u64 = 30000;

/// Default hysteresis value for cell reselection (dB)
/// Per 3GPP TS 38.304 Section 5.2.4.5
pub const DEFAULT_Q_HYST_DB: i32 = 4;

/// Time-to-trigger for cell reselection (ms)
/// Minimum time a cell must be better before reselection
pub const CELL_RESELECTION_TIME_TO_TRIGGER_MS: u64 = 1000;

/// PLMN (Public Land Mobile Network) identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Plmn {
    pub mcc: u16,
    pub mnc: u16,
    pub long_mnc: bool,
}

impl Plmn {
    pub fn new(mcc: u16, mnc: u16, long_mnc: bool) -> Self {
        Self { mcc, mnc, long_mnc }
    }

    pub fn has_value(&self) -> bool {
        self.mcc != 0 || self.mnc != 0
    }
}

impl std::fmt::Display for Plmn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.long_mnc {
            write!(f, "{:03}-{:03}", self.mcc, self.mnc)
        } else {
            write!(f, "{:03}-{:02}", self.mcc, self.mnc)
        }
    }
}

/// Tracking Area Identity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Tai {
    pub plmn: Plmn,
    pub tac: u32,
}

impl Tai {
    pub fn new(plmn: Plmn, tac: u32) -> Self {
        Self { plmn, tac }
    }
}

/// Cell category after selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CellCategory {
    #[default]
    None,
    /// Cell meets all criteria including selected PLMN
    SuitableCell,
    /// Cell meets criteria except PLMN (can be used in limited service)
    AcceptableCell,
}

/// MIB (Master Information Block) information from cell
#[derive(Debug, Clone, Default)]
pub struct MibInfo {
    pub has_mib: bool,
    pub is_barred: bool,
    pub is_intra_freq_reselect_allowed: bool,
}

/// SIB1 (System Information Block 1) information from cell
#[derive(Debug, Clone, Default)]
pub struct Sib1Info {
    pub has_sib1: bool,
    pub is_reserved: bool,
    pub nci: i64, // NR Cell Identity
    pub tac: u32, // Tracking Area Code
    pub plmn: Plmn,
    // Cell selection parameters (from CellSelectionInfo)
    pub q_rx_lev_min: i8, // Minimum required RX level
    pub q_rx_lev_min_offset: Option<u8>,
    pub q_qual_min: Option<i8>, // Minimum quality level
    /// Broadcast SNPN NID (Rel-17, TS 23.501 §5.30). When the cell is an SNPN
    /// cell, SIB1 advertises the Network Identifier (`npn-IdentityInfoList`).
    /// `None` for a public (PLMN) cell.
    pub nid: Option<String>,
}

/// Description of a detected cell
#[derive(Debug, Clone, Default)]
pub struct CellDescription {
    /// Signal strength in dBm
    pub dbm: i32,
    /// Last time signal was received
    pub last_seen: Option<Instant>,
    /// MIB information
    pub mib: MibInfo,
    /// SIB1 information
    pub sib1: Sib1Info,
}

impl CellDescription {
    pub fn new(dbm: i32) -> Self {
        Self {
            dbm,
            last_seen: Some(Instant::now()),
            mib: MibInfo::default(),
            sib1: Sib1Info::default(),
        }
    }

    /// Check if the cell has valid system information
    pub fn has_system_info(&self) -> bool {
        self.mib.has_mib && self.sib1.has_sib1
    }

    /// Check if the cell is barred
    pub fn is_barred(&self) -> bool {
        self.mib.is_barred
    }

    /// Check if the cell is reserved
    pub fn is_reserved(&self) -> bool {
        self.sib1.is_reserved
    }

    /// Get the TAI of this cell
    pub fn tai(&self) -> Tai {
        Tai::new(self.sib1.plmn, self.sib1.tac)
    }

    /// Calculate Srxlev (cell selection RX level value)
    /// Srxlev = `Q_rxlevmeas` - (`Q_rxlevmin` + `Q_rxlevminoffset`)
    /// Per 3GPP TS 38.304 Section 5.2.3.2
    pub fn srxlev(&self) -> i32 {
        let q_rxlev_min = self.sib1.q_rx_lev_min as i32 * 2; // Convert to dBm
        let q_rxlev_min_offset = self.sib1.q_rx_lev_min_offset.unwrap_or(0) as i32 * 2;
        self.dbm - (q_rxlev_min + q_rxlev_min_offset)
    }

    /// Calculate Squal (cell selection quality value)
    /// Squal = Q_qualmeas - (Q_qualmin + Q_qualminoffset)
    /// Per 3GPP TS 38.304 Section 5.2.3.2
    /// Returns None if Q_qualmin is not configured (quality criteria not applicable)
    pub fn squal(&self) -> Option<i32> {
        let q_qual_min = self.sib1.q_qual_min? as i32;
        // RSRQ is approximated from RSRP for simulation:
        // RSRQ ~ RSRP + 10*log10(N_RB) - noise, simplified as RSRP + 10 for typical load
        let q_qual_meas = self.dbm + 10;
        Some(q_qual_meas - q_qual_min)
    }

    /// Combined cell selection criterion S per TS 38.304 Section 5.2.3.2
    /// Cell is selected if Srxlev > 0 AND (Squal > 0 if configured)
    pub fn meets_s_criteria(&self) -> bool {
        if self.srxlev() <= 0 {
            return false;
        }
        // If Squal is configured, it must also be positive
        if let Some(squal) = self.squal() {
            if squal <= 0 {
                return false;
            }
        }
        true
    }

    /// Cell ranking value R per TS 38.304 Section 5.2.4.6
    /// Rs = Q_meas,s + Q_hyst (serving cell)
    /// Rn = Q_meas,n - Q_offset (neighbor cell)
    pub fn ranking_value(&self, q_offset: i32) -> i32 {
        self.dbm - q_offset
    }
}

/// Active cell information
#[derive(Debug, Clone, Default)]
pub struct ActiveCellInfo {
    pub cell_id: i32,
    pub category: CellCategory,
    pub plmn: Plmn,
    pub tac: u32,
}

impl ActiveCellInfo {
    pub fn has_value(&self) -> bool {
        self.cell_id != 0
    }
}

/// Report of cell selection results
#[derive(Debug, Clone, Default)]
pub struct CellSelectionReport {
    pub out_of_plmn_cells: u32,
    pub si_missing_cells: u32,
    pub reserved_cells: u32,
    pub barred_cells: u32,
    pub forbidden_tai_cells: u32,
    pub low_signal_cells: u32,
}

/// Cell reselection parameters per 3GPP TS 38.304
#[derive(Debug, Clone)]
pub struct CellReselectionParams {
    /// Hysteresis value (dB) - Q-Hyst
    pub q_hyst: i32,
    /// Time-to-trigger for reselection (ms)
    pub t_reselection: u64,
    /// Candidate cell ID that may be reselected to
    pub reselection_candidate: Option<i32>,
    /// Time when candidate first became better
    pub candidate_better_since: Option<Instant>,
    /// Whether `q_hyst` and `t_reselection` came from broadcast SIB2 rather than
    /// from [`DEFAULT_Q_HYST_DB`] / [`CELL_RESELECTION_TIME_TO_TRIGGER_MS`]
    /// (issue #50).
    ///
    /// Tracked rather than inferred, because a cell may legitimately broadcast
    /// exactly the default values and "the broadcast agrees with the constant"
    /// must not be indistinguishable from "no broadcast was read". The UE logs
    /// which it is, so an operator can tell a configured cell from an assuming UE.
    pub from_broadcast: bool,
    /// The serving frequency's `cellReselectionPriority` from SIB2 (0..7).
    ///
    /// `None` until a SIB2 is read. TS 38.304 §5.2.4.1 orders frequencies by
    /// priority BEFORE ranking cells, so a UE with no priority information can
    /// only rank.
    pub serving_priority: Option<u8>,
}

impl Default for CellReselectionParams {
    fn default() -> Self {
        Self {
            q_hyst: DEFAULT_Q_HYST_DB,
            t_reselection: CELL_RESELECTION_TIME_TO_TRIGGER_MS,
            reselection_candidate: None,
            candidate_better_since: None,
            from_broadcast: false,
            serving_priority: None,
        }
    }
}

/// Cell selection and reselection manager
pub struct CellSelector {
    /// Detected cells indexed by `cell_id`
    cells: HashMap<i32, CellDescription>,
    /// Currently selected cell. `None` until the first successful selection.
    ///
    /// Deliberately an `Option` rather than an `ActiveCellInfo` whose
    /// `cell_id == 0` means "none": cell ID 0 is a perfectly legal NR cell
    /// identity (a gNB with `nci: 0x10` and `gnb_id_length: 32` has cell-ID
    /// bits 0), so the old sentinel made a real cell indistinguishable from no
    /// cell and camping on it silently failed.
    current_cell: Option<ActiveCellInfo>,
    /// Selected PLMN (from NAS)
    selected_plmn: Option<Plmn>,
    /// Required SNPN NID (Rel-17, TS 23.501 §5.30). When set (SNPN mode), only
    /// a cell whose broadcast NID matches is treated as suitable; cells with a
    /// different or absent NID are rejected.
    required_nid: Option<String>,
    /// Forbidden TAIs for roaming
    forbidden_tai_roaming: Vec<Tai>,
    /// Forbidden TAIs for regional provision of service
    forbidden_tai_rps: Vec<Tai>,
    /// Time when cell selector was started
    started_time: Instant,
    /// Last time cell selection failure was logged
    last_failure_logged: Option<Instant>,
    /// Cell reselection parameters
    reselection_params: CellReselectionParams,
    /// `q-OffsetCell` per neighbour, keyed by `physCellId`, from broadcast SIB3
    /// (issue #50).
    ///
    /// Keyed by PCI and not by the simulator's `cell_id` because that is how
    /// TS 38.331 keys `intraFreqNeighCellList`, and the UE derives a cell's PCI
    /// from the NR Cell Identity its SIB1 broadcast — the convention recorded for
    /// #37, since a real PCI comes from the SSB and there is no PHY here. A cell
    /// whose SIB1 has not been read therefore has no PCI and gets `Qoffset = 0`,
    /// which is the same treatment TS 38.304 gives a neighbour absent from the
    /// list.
    intra_freq_q_offset_by_pci: HashMap<u16, i32>,
    /// `cellReselectionPriority` per carrier ARFCN, from broadcast SIB4 and from
    /// a dedicated RRCRelease `cellReselectionPriorities` list.
    ///
    /// Stored but not yet used to order carriers: this simulator's RLS presents
    /// one carrier, so every cell is intra-frequency and the priority ordering of
    /// TS 38.304 §5.2.4.1 has nothing to order. Kept because it is what a
    /// released UE was handed, and reporting it is how an operator can see the
    /// dedicated list arrived.
    carrier_priorities: HashMap<u32, u8>,
    /// Whether [`Self::carrier_priorities`] came from a DEDICATED RRCRelease list
    /// rather than from broadcast SIB4. TS 38.304 §5.2.4.1 has dedicated
    /// priorities override broadcast ones while they are valid.
    carrier_priorities_dedicated: bool,
}

impl CellSelector {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            current_cell: None,
            selected_plmn: None,
            required_nid: None,
            forbidden_tai_roaming: Vec::new(),
            forbidden_tai_rps: Vec::new(),
            started_time: Instant::now(),
            last_failure_logged: None,
            reselection_params: CellReselectionParams::default(),
            intra_freq_q_offset_by_pci: HashMap::new(),
            carrier_priorities: HashMap::new(),
            carrier_priorities_dedicated: false,
        }
    }

    /// Set cell reselection parameters
    pub fn set_reselection_params(&mut self, params: CellReselectionParams) {
        self.reselection_params = params;
    }

    /// Get current reselection parameters
    pub fn reselection_params(&self) -> &CellReselectionParams {
        &self.reselection_params
    }

    /// Set the selected PLMN (called by NAS after PLMN selection)
    pub fn set_selected_plmn(&mut self, plmn: Option<Plmn>) {
        self.selected_plmn = plmn;
    }

    /// Get the selected PLMN
    pub fn selected_plmn(&self) -> Option<Plmn> {
        self.selected_plmn
    }

    /// Set the required SNPN NID for NID-qualified cell selection (Rel-17,
    /// TS 23.501 §5.30). When `Some`, only cells broadcasting the matching NID
    /// are selectable; clears SNPN gating when `None`.
    pub fn set_required_nid(&mut self, nid: Option<String>) {
        self.required_nid = nid;
    }

    /// Get the required SNPN NID, if SNPN-qualified selection is active.
    pub fn required_nid(&self) -> Option<&str> {
        self.required_nid.as_deref()
    }

    /// Add a TAI to the forbidden roaming list
    pub fn add_forbidden_tai_roaming(&mut self, tai: Tai) {
        if !self.forbidden_tai_roaming.contains(&tai) {
            self.forbidden_tai_roaming.push(tai);
        }
    }

    /// Add a TAI to the forbidden RPS list
    pub fn add_forbidden_tai_rps(&mut self, tai: Tai) {
        if !self.forbidden_tai_rps.contains(&tai) {
            self.forbidden_tai_rps.push(tai);
        }
    }

    /// Clear forbidden TAI lists
    pub fn clear_forbidden_tais(&mut self) {
        self.forbidden_tai_roaming.clear();
        self.forbidden_tai_rps.clear();
    }

    /// Handle a signal strength change for a cell
    /// Returns true if the cell was added or removed
    pub fn handle_signal_change(&mut self, cell_id: i32, dbm: i32) -> CellChangeEvent {
        let consider_lost = dbm < CELL_LOST_THRESHOLD_DBM;

        if let std::collections::hash_map::Entry::Vacant(e) = self.cells.entry(cell_id) {
            if !consider_lost {
                // New cell detected
                e.insert(CellDescription::new(dbm));
                tracing::debug!(
                    "New cell detected: cell_id={}, dbm={}, total_cells={}",
                    cell_id,
                    dbm,
                    self.cells.len()
                );
                return CellChangeEvent::CellDetected(cell_id);
            }
            CellChangeEvent::None
        } else {
            if consider_lost {
                // Cell lost
                let was_active = self
                    .current_cell
                    .as_ref()
                    .is_some_and(|c| c.cell_id == cell_id);
                self.cells.remove(&cell_id);
                tracing::debug!(
                    "Cell lost: cell_id={}, was_active={}, total_cells={}",
                    cell_id,
                    was_active,
                    self.cells.len()
                );
                if was_active {
                    // `take` leaves None, i.e. "not camped", which is exactly the
                    // post-condition of losing the active cell.
                    if let Some(old_cell) = self.current_cell.take() {
                        return CellChangeEvent::ActiveCellLost(old_cell);
                    }
                }
                return CellChangeEvent::CellLost(cell_id);
            }
            // Update signal strength
            if let Some(cell) = self.cells.get_mut(&cell_id) {
                cell.dbm = dbm;
                cell.last_seen = Some(Instant::now());
            }
            CellChangeEvent::SignalUpdated(cell_id, dbm)
        }
    }

    /// Update MIB information for a cell
    pub fn update_mib(&mut self, cell_id: i32, mib: MibInfo) {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            tracing::debug!(
                "MIB updated for cell {}: barred={}, intra_freq_reselect={}",
                cell_id,
                mib.is_barred,
                mib.is_intra_freq_reselect_allowed
            );
            cell.mib = mib;
        }
    }

    /// Update SIB1 information for a cell
    pub fn update_sib1(&mut self, cell_id: i32, sib1: Sib1Info) {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            tracing::debug!(
                "SIB1 updated for cell {}: plmn={}, tac={}, reserved={}",
                cell_id,
                sib1.plmn,
                sib1.tac,
                sib1.is_reserved
            );
            cell.sib1 = sib1;
        }
    }

    /// Get the current active cell, or `None` while not camped on any cell.
    pub fn current_cell(&self) -> Option<&ActiveCellInfo> {
        self.current_cell.as_ref()
    }

    /// Get a cell description by ID
    pub fn get_cell(&self, cell_id: i32) -> Option<&CellDescription> {
        self.cells.get(&cell_id)
    }

    /// Get all detected cells
    pub fn cells(&self) -> &HashMap<i32, CellDescription> {
        &self.cells
    }

    /// Get available PLMNs from detected cells
    pub fn available_plmns(&self) -> Vec<Plmn> {
        let mut plmns = Vec::new();
        for cell in self.cells.values() {
            if cell.sib1.has_sib1 && !plmns.contains(&cell.sib1.plmn) {
                plmns.push(cell.sib1.plmn);
            }
        }
        plmns
    }

    /// Check if we have signal to a specific cell
    pub fn has_signal_to_cell(&self, cell_id: i32) -> bool {
        self.cells.contains_key(&cell_id)
    }

    /// Perform cell selection
    /// Returns `Some(cell_info)` if a new cell was selected, None if no change
    pub fn perform_cell_selection(&mut self) -> Option<ActiveCellInfo> {
        let elapsed = self.started_time.elapsed();

        // Wait for initial discovery period
        if elapsed < Duration::from_millis(CELL_SELECTION_STARTUP_DELAY_MS) && self.cells.is_empty()
        {
            return None;
        }

        // Wait longer if no PLMN selected yet
        if elapsed < Duration::from_millis(4000) && self.selected_plmn.is_none() {
            return None;
        }

        let last_cell = self.current_cell.clone();
        let should_log_errors = last_cell.is_some()
            || self.last_failure_logged.is_none_or(|t| {
                t.elapsed() >= Duration::from_millis(CELL_SELECTION_LOG_INTERVAL_MS)
            });

        let mut cell_info = ActiveCellInfo::default();
        let mut report = CellSelectionReport::default();

        // First, try to find a suitable cell (matches selected PLMN)
        let mut cell_found = false;
        if self.selected_plmn.is_some() {
            cell_found = self.look_for_suitable_cell(&mut cell_info, &mut report);
            if !cell_found && should_log_errors && !self.cells.is_empty() {
                tracing::warn!(
                    "Suitable cell selection failed in {} cells: out_of_plmn={}, si_missing={}, reserved={}, barred={}, forbidden_tai={}",
                    self.cells.len(), report.out_of_plmn_cells, report.si_missing_cells,
                    report.reserved_cells, report.barred_cells, report.forbidden_tai_cells
                );
            }
        }

        // If no suitable cell, look for acceptable cell
        if !cell_found {
            report = CellSelectionReport::default();
            cell_found = self.look_for_acceptable_cell(&mut cell_info, &mut report);

            if !cell_found && should_log_errors {
                if !self.cells.is_empty() {
                    tracing::warn!(
                        "Acceptable cell selection failed in {} cells: si_missing={}, reserved={}, barred={}, forbidden_tai={}",
                        self.cells.len(), report.si_missing_cells, report.reserved_cells,
                        report.barred_cells, report.forbidden_tai_cells
                    );
                } else {
                    tracing::warn!("Cell selection failed, no cells in coverage");
                }
                self.last_failure_logged = Some(Instant::now());
            }
        }

        // Nothing selectable this round. Leave any existing camp untouched:
        // losing the serving cell is signalled by ActiveCellLost, not by a
        // failed selection round.
        if !cell_found {
            return None;
        }

        // Apply cell reselection with hysteresis if we already have a serving cell
        if let Some(ref last) = last_cell {
            let should_reselect = self.evaluate_cell_reselection(last.cell_id, cell_info.cell_id);
            if !should_reselect {
                // Keep current cell, reset candidate
                cell_info = last.clone();
            }
        }

        let changed = last_cell.as_ref().map(|c| c.cell_id) != Some(cell_info.cell_id);

        // Update current cell
        self.current_cell = Some(cell_info.clone());

        // Log if selection changed
        if changed {
            tracing::info!(
                "Cell reselection: id={}, plmn={}, tac={}, category={:?}",
                cell_info.cell_id,
                cell_info.plmn,
                cell_info.tac,
                cell_info.category
            );
            // Clear reselection candidate on successful reselection
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
        }

        // Return the cell only when the camp actually changed, so callers can
        // treat Some(..) as "newly camped / reselected" and drive registration
        // off that edge. Comparing Options rather than raw ids is what fixes the
        // cell-ID-0 case: previously `0 != 0` made a successful first selection
        // on cell 0 indistinguishable from "no cell found", so the UE camped
        // internally but never told NAS, and registration never started.
        if changed {
            Some(cell_info)
        } else {
            None
        }
    }

    /// The `physCellId` this UE ascribes to a cell, or `None` when its SIB1 has
    /// not been read.
    ///
    /// Derived from the broadcast NR Cell Identity via
    /// `phys_cell_id_from_nci` — the same function the gNB uses, because both
    /// ends have to compute the same number (the convention recorded for #37; a
    /// real PCI comes from the SSB and there is no PHY here).
    ///
    /// `None` rather than a fabricated value for a cell with no SIB1: a made-up
    /// PCI could collide with a real neighbour's entry in SIB3 and apply that
    /// neighbour's `Qoffset` to the wrong cell.
    pub fn phys_cell_id_of(&self, cell_id: i32) -> Option<u16> {
        let cell = self.cells.get(&cell_id)?;
        if !cell.sib1.has_sib1 {
            return None;
        }
        Some(phys_cell_id_from_nci(cell.sib1.nci as u64))
    }

    /// `Qoffset` for a cell, in dB, from broadcast SIB3.
    ///
    /// Zero when SIB3 named no offset for it, when no SIB3 has been read, or when
    /// the cell has no PCI to key on — all three being cases TS 38.304 treats the
    /// same way, as a neighbour with no cell-specific offset.
    pub fn q_offset_of(&self, cell_id: i32) -> i32 {
        self.phys_cell_id_of(cell_id)
            .and_then(|pci| self.intra_freq_q_offset_by_pci.get(&pci).copied())
            .unwrap_or(0)
    }

    /// Applies a broadcast SIB2 (issue #50).
    ///
    /// This is what makes `Q_hyst` and `Treselection` come from the network
    /// instead of from [`DEFAULT_Q_HYST_DB`] and
    /// [`CELL_RESELECTION_TIME_TO_TRIGGER_MS`]. The in-flight candidate is
    /// deliberately NOT reset: a cell re-broadcasting the same SIB2 every SI
    /// period would otherwise restart the time-to-trigger on every broadcast and
    /// reselection could never fire.
    pub fn apply_sib2(&mut self, q_hyst_db: i32, t_reselection_s: u8, serving_priority: u8) {
        self.reselection_params.q_hyst = q_hyst_db;
        self.reselection_params.t_reselection = u64::from(t_reselection_s) * 1000;
        self.reselection_params.serving_priority = Some(serving_priority);
        self.reselection_params.from_broadcast = true;
    }

    /// Applies a broadcast SIB3's `intraFreqNeighCellList` (issue #50).
    ///
    /// REPLACES the stored map rather than merging into it: SIB3 carries the
    /// cell's complete neighbour list, so an entry that disappeared from the
    /// broadcast has had its offset withdrawn, and merging would keep applying a
    /// `Qoffset` the network no longer advertises.
    pub fn apply_sib3(&mut self, neighbours: &[(u16, i32)]) {
        self.intra_freq_q_offset_by_pci = neighbours.iter().copied().collect();
    }

    /// Applies a broadcast SIB4's `interFreqCarrierFreqList` (issue #50).
    ///
    /// Ignored while a DEDICATED list from RRCRelease is in force, per
    /// TS 38.304 §5.2.4.1: dedicated priorities override broadcast ones.
    pub fn apply_sib4(&mut self, carriers: &[(u32, u8)]) {
        if self.carrier_priorities_dedicated {
            return;
        }
        self.carrier_priorities = carriers.iter().copied().collect();
    }

    /// Applies a DEDICATED `cellReselectionPriorities` list from an RRCRelease
    /// (TS 38.331 §6.3.2, TS 38.304 §5.2.4.1).
    ///
    /// An EMPTY list is not the same as no list: TS 38.304 has the UE delete its
    /// stored dedicated priorities when it receives one, falling back to the
    /// broadcast values. So an empty slice clears the dedicated state rather than
    /// storing an empty override.
    pub fn apply_dedicated_carrier_priorities(&mut self, carriers: &[(u32, u8)]) {
        if carriers.is_empty() {
            self.carrier_priorities.clear();
            self.carrier_priorities_dedicated = false;
            return;
        }
        self.carrier_priorities = carriers.iter().copied().collect();
        self.carrier_priorities_dedicated = true;
    }

    /// The carrier priorities currently in force, and whether they are dedicated.
    pub fn carrier_priorities(&self) -> (&HashMap<u32, u8>, bool) {
        (&self.carrier_priorities, self.carrier_priorities_dedicated)
    }

    /// Evaluate whether cell reselection should occur, per the TS 38.304 §5.2.4.6
    /// R-criterion.
    ///
    /// ```text
    /// R_s = Q_meas,s + Q_hyst          (serving cell)
    /// R_n = Q_meas,n - Qoffset         (neighbour cell)
    /// ```
    ///
    /// and the neighbour is reselected only once `R_n > R_s` has held for
    /// `Treselection`.
    ///
    /// This used to compare `best_dbm > current_dbm + q_hyst` directly, which is
    /// the same inequality **only when every `Qoffset` is zero** — and there was
    /// no `Qoffset` at all, so the per-cell term of `R_n` was silently absent
    /// rather than zero by configuration. `Q_hyst` and `Treselection` also came
    /// from compile-time constants; both now come from SIB2 when a cell
    /// broadcasts one (issue #50).
    fn evaluate_cell_reselection(&mut self, current_cell_id: i32, best_cell_id: i32) -> bool {
        // If best cell is same as current, no reselection needed
        if current_cell_id == best_cell_id {
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        }

        // R_s = Q_meas,s + Q_hyst. Expressed as `ranking_value(-q_hyst)` so the
        // serving and neighbour sides go through ONE ranking function: two
        // formulas is how a sign error hides.
        let Some(r_s) = self
            .cells
            .get(&current_cell_id)
            .map(|c| c.ranking_value(-self.reselection_params.q_hyst))
        else {
            // The serving cell is gone from the measurement set. That is
            // ActiveCellLost's business, not reselection's; treating an absent
            // serving cell as infinitely bad would reselect on a single missed
            // measurement.
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        };

        // R_n = Q_meas,n - Qoffset, with Qoffset from broadcast SIB3.
        let q_offset_n = self.q_offset_of(best_cell_id);
        let Some(r_n) = self
            .cells
            .get(&best_cell_id)
            .map(|c| c.ranking_value(q_offset_n))
        else {
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        };

        // Strictly better: TS 38.304 §5.2.4.6 says "better ranked", and equal
        // ranking must not ping-pong between two cells.
        if r_n <= r_s {
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        }

        // Arm the Treselection timer for a newly better-ranked candidate.
        if self.reselection_params.reselection_candidate != Some(best_cell_id) {
            self.reselection_params.reselection_candidate = Some(best_cell_id);
            self.reselection_params.candidate_better_since = Some(Instant::now());
            tracing::debug!(
                "Cell reselection candidate: cell_id={best_cell_id} R_n={r_n} \
                 (serving cell_id={current_cell_id} R_s={r_s}, q_hyst={} dB, \
                 q_offset={q_offset_n} dB, t_reselection={} ms, from_broadcast={})",
                self.reselection_params.q_hyst,
                self.reselection_params.t_reselection,
                self.reselection_params.from_broadcast,
            );
            // Deliberately NOT an early return. Arming used to end the call, so a
            // candidate could never win on the evaluation that first saw it --
            // which made `t-ReselectionNR = 0` (a legal broadcast value meaning no
            // time-to-trigger) still cost a whole CELL_SELECTION_INTERVAL_MS.
            // Falling through lets the elapsed check below decide, and for a zero
            // timer it is already satisfied.
        }

        if let Some(since) = self.reselection_params.candidate_better_since {
            if since.elapsed() >= Duration::from_millis(self.reselection_params.t_reselection) {
                tracing::debug!(
                    "Cell reselection triggered: cell_id={current_cell_id} -> \
                     cell_id={best_cell_id} (R_n={r_n} > R_s={r_s}, elapsed {:?})",
                    since.elapsed()
                );
                return true;
            }
        }

        false
    }

    /// Look for a suitable cell (matches selected PLMN)
    fn look_for_suitable_cell(
        &self,
        cell_info: &mut ActiveCellInfo,
        report: &mut CellSelectionReport,
    ) -> bool {
        let selected_plmn = match self.selected_plmn {
            Some(plmn) => plmn,
            None => return false,
        };

        let mut candidates: Vec<(i32, i32)> = Vec::new(); // (cell_id, dbm)

        for (&cell_id, cell) in &self.cells {
            // Check system info
            if !cell.sib1.has_sib1 {
                report.si_missing_cells += 1;
                continue;
            }
            if !cell.mib.has_mib {
                report.si_missing_cells += 1;
                continue;
            }

            // Check PLMN
            if cell.sib1.plmn != selected_plmn {
                report.out_of_plmn_cells += 1;
                continue;
            }

            // SNPN NID-qualified selection (Rel-17, TS 23.501 §5.30): in SNPN
            // mode only a cell broadcasting the configured NID is suitable.
            if let Some(ref required_nid) = self.required_nid {
                if cell.sib1.nid.as_deref() != Some(required_nid.as_str()) {
                    report.out_of_plmn_cells += 1;
                    tracing::debug!(
                        "Cell {} rejected: SNPN NID mismatch (required={}, broadcast={:?})",
                        cell_id,
                        required_nid,
                        cell.sib1.nid
                    );
                    continue;
                }
            }

            // Check barred
            if cell.mib.is_barred {
                report.barred_cells += 1;
                continue;
            }

            // Check reserved
            if cell.sib1.is_reserved {
                report.reserved_cells += 1;
                continue;
            }

            // Check forbidden TAIs
            let tai = cell.tai();
            if self.forbidden_tai_roaming.contains(&tai) || self.forbidden_tai_rps.contains(&tai) {
                report.forbidden_tai_cells += 1;
                continue;
            }

            // Check combined S criteria: Srxlev > 0 AND Squal > 0 (if configured)
            if !cell.meets_s_criteria() {
                report.low_signal_cells += 1;
                continue;
            }

            // Cell is suitable - use ranking value for comparison
            candidates.push((cell_id, cell.ranking_value(0)));
        }

        if candidates.is_empty() {
            return false;
        }

        // Sort by signal strength (highest first)
        candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

        let selected_id = candidates[0].0;
        let selected_cell = &self.cells[&selected_id];

        *cell_info = ActiveCellInfo {
            cell_id: selected_id,
            plmn: selected_cell.sib1.plmn,
            tac: selected_cell.sib1.tac,
            category: CellCategory::SuitableCell,
        };

        true
    }

    /// Look for an acceptable cell (any PLMN)
    fn look_for_acceptable_cell(
        &self,
        cell_info: &mut ActiveCellInfo,
        report: &mut CellSelectionReport,
    ) -> bool {
        let mut candidates: Vec<(i32, i32, bool)> = Vec::new(); // (cell_id, dbm, matches_selected_plmn)

        for (&cell_id, cell) in &self.cells {
            // Check system info
            if !cell.sib1.has_sib1 {
                report.si_missing_cells += 1;
                continue;
            }
            if !cell.mib.has_mib {
                report.si_missing_cells += 1;
                continue;
            }

            // Check barred
            if cell.mib.is_barred {
                report.barred_cells += 1;
                continue;
            }

            // Check reserved
            if cell.sib1.is_reserved {
                report.reserved_cells += 1;
                continue;
            }

            // Check forbidden TAIs
            let tai = cell.tai();
            if self.forbidden_tai_roaming.contains(&tai) || self.forbidden_tai_rps.contains(&tai) {
                report.forbidden_tai_cells += 1;
                continue;
            }

            // Cell is acceptable
            let matches_plmn = self.selected_plmn == Some(cell.sib1.plmn);
            candidates.push((cell_id, cell.dbm, matches_plmn));
        }

        if candidates.is_empty() {
            return false;
        }

        // Sort by signal strength first
        candidates.sort_by_key(|c| std::cmp::Reverse(c.1));

        // Then prioritize cells matching selected PLMN (stable sort)
        candidates.sort_by_key(|c| std::cmp::Reverse(c.2));

        let selected_id = candidates[0].0;
        let selected_cell = &self.cells[&selected_id];

        *cell_info = ActiveCellInfo {
            cell_id: selected_id,
            plmn: selected_cell.sib1.plmn,
            tac: selected_cell.sib1.tac,
            category: CellCategory::AcceptableCell,
        };

        true
    }
}

impl Default for CellSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Events generated by cell management
#[derive(Debug, Clone)]
pub enum CellChangeEvent {
    None,
    CellDetected(i32),
    CellLost(i32),
    ActiveCellLost(ActiveCellInfo),
    SignalUpdated(i32, i32),
}

// ============================================================================
// PLMN Selection (3GPP TS 23.122 Section 4.4)
// ============================================================================

/// Default higher-priority PLMN periodic search interval in seconds
/// (TS 23.122 Section 4.4.3.3: timer T, default 60 minutes when the SIM
/// does not configure a value; allowed range 6 minutes to 8 hours)
pub const DEFAULT_HP_PLMN_SEARCH_INTERVAL_SECS: u32 = 60 * 60;

/// PLMN selection mode (TS 23.122 Section 3.1)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlmnSelectionMode {
    /// Automatic network selection mode
    #[default]
    Automatic,
    /// Manual network selection mode
    Manual,
}

/// Decode a 3-octet BCD PLMN (TS 24.501 Section 9.11.3.4 layout, as kept
/// in the MM orchestrator's forbidden-PLMN list) into a [`Plmn`].
/// Parse a configured PLMN written as `"<mcc>-<mnc>"`, e.g. `"001-01"` or
/// `"262-030"` (issue #49).
///
/// The MNC's DIGIT COUNT is significant and is taken from the text: a 3-digit MNC
/// is a different PLMN from the 2-digit one with the same value (TS 23.003 §2.2),
/// and they encode differently on the wire. So `"262-03"` and `"262-030"` are not
/// the same network, and writing one when the other was meant is a mistake this
/// preserves rather than normalises away.
///
/// `None` for anything that is not two numeric fields with a 3-digit MCC and a
/// 2- or 3-digit MNC.
pub fn parse_configured_plmn(text: &str) -> Option<Plmn> {
    let (mcc, mnc) = text.trim().split_once('-')?;
    let mcc = mcc.trim();
    let mnc = mnc.trim();
    if mcc.len() != 3 || !matches!(mnc.len(), 2 | 3) {
        return None;
    }
    if !mcc.chars().all(|c| c.is_ascii_digit()) || !mnc.chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(Plmn::new(
        mcc.parse().ok()?,
        mnc.parse().ok()?,
        mnc.len() == 3,
    ))
}

pub fn plmn_from_bcd(bcd: &[u8; 3]) -> Plmn {
    let mcc1 = u16::from(bcd[0] & 0x0F);
    let mcc2 = u16::from(bcd[0] >> 4);
    let mcc3 = u16::from(bcd[1] & 0x0F);
    let mnc3 = bcd[1] >> 4;
    let mnc1 = u16::from(bcd[2] & 0x0F);
    let mnc2 = u16::from(bcd[2] >> 4);
    let mcc = mcc1 * 100 + mcc2 * 10 + mcc3;
    if mnc3 == 0x0F {
        Plmn::new(mcc, mnc1 * 10 + mnc2, false)
    } else {
        Plmn::new(mcc, mnc1 * 100 + mnc2 * 10 + u16::from(mnc3), true)
    }
}

/// PLMN selector implementing the TS 23.122 Section 4.4.3 network
/// selection procedures.
///
/// In automatic mode candidates are evaluated in the priority order of
/// Section 4.4.3.1.1:
///
/// 1. the registered PLMN (RPLMN) or a PLMN equivalent to it,
/// 2. HPLMN or the highest-priority EHPLMN (when an EHPLMN list exists),
/// 3. user-controlled preferred PLMNs (in priority order),
/// 4. operator-controlled preferred PLMNs (in priority order),
/// 5. any other available PLMN.
///
/// PLMNs in the "forbidden PLMNs" list are excluded in automatic mode;
/// in manual mode the user-chosen PLMN may be attempted even when
/// forbidden (Section 4.4.3.1.2).
///
/// The selector also runs the higher-priority PLMN periodic search timer
/// of Section 4.4.3.3: while roaming on a lower-priority PLMN, `tick()`
/// returns `true` whenever a background scan for higher-priority PLMNs
/// is due.
#[derive(Debug)]
pub struct PlmnSelector {
    mode: PlmnSelectionMode,
    hplmn: Plmn,
    /// Equivalent HPLMN list (when present, replaces the HPLMN for
    /// selection priority, TS 23.122 Section 1.2)
    ehplmn: Vec<Plmn>,
    /// Registered PLMN, if any
    registered_plmn: Option<Plmn>,
    /// Equivalent PLMN list signalled by the network for the RPLMN
    equivalent_plmns: Vec<Plmn>,
    /// User-controlled PLMN selector list (priority order)
    user_preferred: Vec<Plmn>,
    /// Operator-controlled PLMN selector list (priority order)
    operator_preferred: Vec<Plmn>,
    /// Forbidden PLMN list (TS 23.122 Section 3.1)
    forbidden: Vec<Plmn>,
    /// PLMNs currently available on the radio, reported by the RRC
    /// CellSelector; the candidate set for automatic selection (TS 23.122
    /// §4.4.3).
    available: Vec<Plmn>,
    /// PLMN broadcast by the cell the UE is camped on (issue #49), which is what a
    /// successful registration records as the RPLMN. `None` until a cell has been
    /// selected -- and a cell is only selected once its SIB1 has been read, so this
    /// is a broadcast value rather than an assumed one.
    serving_plmn: Option<Plmn>,
    /// Manually chosen PLMN (manual mode)
    manual_selection: Option<Plmn>,
    /// Higher-priority PLMN search interval (timer T) in seconds
    hp_search_interval_secs: u32,
    /// Seconds elapsed since the last higher-priority search
    hp_search_elapsed_secs: u32,
}

impl PlmnSelector {
    /// Create a selector for the given home PLMN with the default
    /// higher-priority search interval.
    pub fn new(hplmn: Plmn) -> Self {
        Self {
            mode: PlmnSelectionMode::Automatic,
            hplmn,
            ehplmn: Vec::new(),
            registered_plmn: None,
            equivalent_plmns: Vec::new(),
            user_preferred: Vec::new(),
            operator_preferred: Vec::new(),
            forbidden: Vec::new(),
            available: Vec::new(),
            serving_plmn: None,
            manual_selection: None,
            hp_search_interval_secs: DEFAULT_HP_PLMN_SEARCH_INTERVAL_SECS,
            hp_search_elapsed_secs: 0,
        }
    }

    /// Current selection mode
    pub fn mode(&self) -> PlmnSelectionMode {
        self.mode
    }

    /// Switch to automatic network selection mode
    pub fn set_automatic(&mut self) {
        self.mode = PlmnSelectionMode::Automatic;
        self.manual_selection = None;
    }

    /// Manual mode hook: select the given PLMN manually
    /// (TS 23.122 Section 4.4.3.1.2)
    pub fn select_manual(&mut self, plmn: Plmn) {
        self.mode = PlmnSelectionMode::Manual;
        self.manual_selection = Some(plmn);
    }

    /// Set the EHPLMN list (priority order)
    pub fn set_ehplmn_list(&mut self, ehplmn: Vec<Plmn>) {
        self.ehplmn = ehplmn;
    }

    /// Set the user-controlled preferred PLMN list (priority order)
    pub fn set_user_preferred(&mut self, plmns: Vec<Plmn>) {
        self.user_preferred = plmns;
    }

    /// Set the operator-controlled preferred PLMN list (priority order)
    pub fn set_operator_preferred(&mut self, plmns: Vec<Plmn>) {
        self.operator_preferred = plmns;
    }

    /// Record the PLMN broadcast by the cell the UE is camped on (issue #49).
    ///
    /// Held here rather than threaded through the MM output handler because the rest
    /// of the selection state -- available, registered, selected, forbidden -- already
    /// lives here, and a second home for one of them is how they drift apart.
    pub fn set_serving_plmn(&mut self, plmn: Option<Plmn>) {
        self.serving_plmn = plmn;
    }

    /// The PLMN of the camped cell, when one has been read from SIB1.
    pub fn serving_plmn(&self) -> Option<Plmn> {
        self.serving_plmn
    }

    /// Record a successful registration on the given PLMN
    pub fn set_registered_plmn(&mut self, plmn: Option<Plmn>) {
        self.registered_plmn = plmn;
        if plmn.is_none() || self.is_home_plmn(plmn.unwrap_or_default()) {
            self.hp_search_elapsed_secs = 0;
        }
    }

    /// Registered PLMN, if any
    pub fn registered_plmn(&self) -> Option<Plmn> {
        self.registered_plmn
    }

    /// Set the equivalent-PLMN list received from the network
    pub fn set_equivalent_plmns(&mut self, plmns: Vec<Plmn>) {
        self.equivalent_plmns = plmns;
    }

    /// Replace the forbidden PLMN list (e.g. with the MM orchestrator's
    /// BCD-encoded list converted via [`plmn_from_bcd`])
    pub fn set_forbidden(&mut self, plmns: Vec<Plmn>) {
        self.forbidden = plmns;
    }

    /// Sets the PLMNs currently available on the radio (TS 23.122 §4.4.3),
    /// reported by the RRC CellSelector; the candidate set for [`Self::select`].
    pub fn set_available_plmns(&mut self, plmns: Vec<Plmn>) {
        self.available = plmns;
    }

    /// The PLMNs currently reported as available on the radio.
    pub fn available_plmns(&self) -> &[Plmn] {
        &self.available
    }

    /// Add a PLMN to the forbidden list
    pub fn add_forbidden(&mut self, plmn: Plmn) {
        if !self.forbidden.contains(&plmn) {
            self.forbidden.push(plmn);
        }
    }

    /// Remove a PLMN from the forbidden list (e.g. after a successful
    /// manual registration, TS 23.122 Section 3.1)
    pub fn remove_forbidden(&mut self, plmn: Plmn) {
        self.forbidden.retain(|p| *p != plmn);
    }

    /// Whether the given PLMN is in the forbidden list
    pub fn is_forbidden(&self, plmn: Plmn) -> bool {
        self.forbidden.contains(&plmn)
    }

    /// Whether the given PLMN is the HPLMN or an EHPLMN
    pub fn is_home_plmn(&self, plmn: Plmn) -> bool {
        if self.ehplmn.is_empty() {
            plmn == self.hplmn
        } else {
            self.ehplmn.contains(&plmn)
        }
    }

    /// Run PLMN selection over the available PLMNs and return the chosen
    /// PLMN, if any (TS 23.122 Section 4.4.3.1).
    pub fn select(&self, available: &[Plmn]) -> Option<Plmn> {
        match self.mode {
            PlmnSelectionMode::Manual => {
                // Manual mode: only the user-chosen PLMN is attempted; the
                // forbidden list does not prevent the attempt
                // (TS 23.122 Section 4.4.3.1.2)
                let manual = self.manual_selection?;
                available.iter().copied().find(|p| *p == manual)
            }
            PlmnSelectionMode::Automatic => self.select_automatic(available),
        }
    }

    fn select_automatic(&self, available: &[Plmn]) -> Option<Plmn> {
        let allowed: Vec<Plmn> = available
            .iter()
            .copied()
            .filter(|p| !self.is_forbidden(*p))
            .collect();

        // 1) RPLMN or an equivalent PLMN
        if let Some(rplmn) = self.registered_plmn {
            if allowed.contains(&rplmn) {
                return Some(rplmn);
            }
            if let Some(eq) = self
                .equivalent_plmns
                .iter()
                .copied()
                .find(|p| allowed.contains(p))
            {
                return Some(eq);
            }
        }

        // 2) HPLMN / highest-priority EHPLMN
        if self.ehplmn.is_empty() {
            if allowed.contains(&self.hplmn) {
                return Some(self.hplmn);
            }
        } else if let Some(e) = self.ehplmn.iter().copied().find(|p| allowed.contains(p)) {
            return Some(e);
        }

        // 3) User-controlled preferred list (priority order)
        if let Some(p) = self
            .user_preferred
            .iter()
            .copied()
            .find(|p| allowed.contains(p))
        {
            return Some(p);
        }

        // 4) Operator-controlled preferred list (priority order)
        if let Some(p) = self
            .operator_preferred
            .iter()
            .copied()
            .find(|p| allowed.contains(p))
        {
            return Some(p);
        }

        // 5) Any other allowed PLMN
        allowed.first().copied()
    }

    /// Drive the higher-priority PLMN periodic search timer; call roughly
    /// once a second. Returns `true` when a periodic higher-priority
    /// search is due (TS 23.122 Section 4.4.3.3): only while registered
    /// on a PLMN that is neither the HPLMN/EHPLMN nor a higher-priority
    /// preferred PLMN, and only in automatic mode.
    pub fn tick(&mut self) -> bool {
        let Some(rplmn) = self.registered_plmn else {
            self.hp_search_elapsed_secs = 0;
            return false;
        };
        if self.mode != PlmnSelectionMode::Automatic || self.is_home_plmn(rplmn) {
            self.hp_search_elapsed_secs = 0;
            return false;
        }
        self.hp_search_elapsed_secs += 1;
        if self.hp_search_elapsed_secs >= self.hp_search_interval_secs {
            self.hp_search_elapsed_secs = 0;
            return true;
        }
        false
    }

    /// Override the higher-priority search interval (timer T). Values are
    /// clamped to the TS 23.122 Section 4.4.3.3 range (6 minutes to
    /// 8 hours).
    pub fn set_hp_search_interval(&mut self, secs: u32) {
        self.hp_search_interval_secs = secs.clamp(6 * 60, 8 * 60 * 60);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell_with_sib1(
        dbm: i32,
        plmn: Plmn,
        tac: u32,
        barred: bool,
        reserved: bool,
    ) -> CellDescription {
        CellDescription {
            dbm,
            last_seen: Some(Instant::now()),
            mib: MibInfo {
                has_mib: true,
                is_barred: barred,
                is_intra_freq_reselect_allowed: true,
            },
            sib1: Sib1Info {
                has_sib1: true,
                is_reserved: reserved,
                nci: 1,
                tac,
                plmn,
                q_rx_lev_min: -70,
                q_rx_lev_min_offset: None,
                q_qual_min: None,
                nid: None,
            },
        }
    }

    #[test]
    fn test_cell_selection_suitable_cell() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add a suitable cell
        selector
            .cells
            .insert(1, make_cell_with_sib1(-80, plmn, 1, false, false));
        // Override started_time to bypass startup delay
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_some());
        let cell = result.unwrap();
        assert_eq!(cell.cell_id, 1);
        assert_eq!(cell.category, CellCategory::SuitableCell);
    }

    #[test]
    fn test_cell_selection_snpn_nid_match_suitable() {
        // SNPN (Rel-17, TS 23.501 §5.30): with a required NID, a cell whose
        // broadcast NID matches is suitable.
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));
        selector.set_required_nid(Some("7AB01234567".to_string()));

        let mut cell = make_cell_with_sib1(-80, plmn, 1, false, false);
        cell.sib1.nid = Some("7AB01234567".to_string());
        selector.cells.insert(1, cell);
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_some());
        assert_eq!(result.unwrap().category, CellCategory::SuitableCell);
    }

    #[test]
    fn test_cell_selection_snpn_nid_mismatch_rejected() {
        // SNPN (Rel-17, TS 23.501 §5.30): a cell broadcasting a different NID
        // is not suitable and must be rejected (falls back to acceptable-cell
        // search, which on the same PLMN yields an acceptable, not suitable,
        // selection).
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));
        selector.set_required_nid(Some("7AB01234567".to_string()));

        let mut cell = make_cell_with_sib1(-80, plmn, 1, false, false);
        cell.sib1.nid = Some("FFFFFFFFFFF".to_string());
        selector.cells.insert(1, cell);
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        // Not selected as a suitable (NID-matching) cell.
        if let Some(info) = result {
            assert_ne!(info.category, CellCategory::SuitableCell);
        }
    }

    #[test]
    fn test_cell_selection_acceptable_cell() {
        let mut selector = CellSelector::new();
        let plmn1 = Plmn::new(999, 70, false);
        let plmn2 = Plmn::new(999, 71, false);
        selector.set_selected_plmn(Some(plmn1));

        // Add a cell with different PLMN (acceptable but not suitable)
        selector
            .cells
            .insert(1, make_cell_with_sib1(-80, plmn2, 1, false, false));
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_some());
        let cell = result.unwrap();
        assert_eq!(cell.cell_id, 1);
        assert_eq!(cell.category, CellCategory::AcceptableCell);
    }

    #[test]
    fn test_cell_selection_barred_cell_rejected() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add a barred cell
        selector
            .cells
            .insert(1, make_cell_with_sib1(-80, plmn, 1, true, false));
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_none() || result.unwrap().cell_id == 0);
    }

    #[test]
    fn test_cell_selection_best_signal() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add multiple suitable cells with different signal strengths
        selector
            .cells
            .insert(1, make_cell_with_sib1(-90, plmn, 1, false, false));
        selector
            .cells
            .insert(2, make_cell_with_sib1(-70, plmn, 1, false, false)); // Best
        selector
            .cells
            .insert(3, make_cell_with_sib1(-85, plmn, 1, false, false));
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_some());
        let cell = result.unwrap();
        assert_eq!(cell.cell_id, 2); // Should select cell with best signal
    }

    #[test]
    fn test_cell_lost_signal() {
        let mut selector = CellSelector::new();

        // Detect a cell
        let event = selector.handle_signal_change(1, -80);
        assert!(matches!(event, CellChangeEvent::CellDetected(1)));
        assert!(selector.has_signal_to_cell(1));

        // Cell signal drops below threshold
        let event = selector.handle_signal_change(1, -125);
        assert!(matches!(event, CellChangeEvent::CellLost(1)));
        assert!(!selector.has_signal_to_cell(1));
    }

    // ========================================================================
    // PLMN selection tests (TS 23.122 Section 4.4.3)
    // ========================================================================

    const HOME: Plmn = Plmn {
        mcc: 999,
        mnc: 70,
        long_mnc: false,
    };
    const VISITED: Plmn = Plmn {
        mcc: 999,
        mnc: 71,
        long_mnc: false,
    };
    const OTHER: Plmn = Plmn {
        mcc: 310,
        mnc: 410,
        long_mnc: true,
    };

    #[test]
    fn test_plmn_selection_prefers_hplmn() {
        let selector = PlmnSelector::new(HOME);
        assert_eq!(selector.select(&[VISITED, HOME, OTHER]), Some(HOME));
    }

    #[test]
    fn test_plmn_selection_ehplmn_replaces_hplmn() {
        let mut selector = PlmnSelector::new(HOME);
        // EHPLMN list present: HPLMN itself is no longer used directly
        selector.set_ehplmn_list(vec![OTHER, VISITED]);
        assert_eq!(selector.select(&[VISITED, HOME, OTHER]), Some(OTHER));
        // Highest-priority available EHPLMN wins
        assert_eq!(selector.select(&[VISITED, HOME]), Some(VISITED));
    }

    #[test]
    fn test_plmn_selection_registered_plmn_first() {
        let mut selector = PlmnSelector::new(HOME);
        selector.set_registered_plmn(Some(VISITED));
        // RPLMN outranks even the HPLMN per Section 4.4.3.1.1 step 0
        assert_eq!(selector.select(&[HOME, VISITED]), Some(VISITED));
    }

    #[test]
    fn test_plmn_selection_equivalent_plmn_of_rplmn() {
        let mut selector = PlmnSelector::new(HOME);
        selector.set_registered_plmn(Some(VISITED));
        selector.set_equivalent_plmns(vec![OTHER]);
        // RPLMN not available, but its equivalent is
        assert_eq!(selector.select(&[HOME, OTHER]), Some(OTHER));
    }

    #[test]
    fn test_plmn_selection_consults_forbidden_list() {
        let mut selector = PlmnSelector::new(HOME);
        selector.add_forbidden(HOME);
        assert!(selector.is_forbidden(HOME));
        assert_eq!(selector.select(&[HOME, VISITED]), Some(VISITED));
        // Everything forbidden: no selection
        selector.add_forbidden(VISITED);
        assert_eq!(selector.select(&[HOME, VISITED]), None);
        // Removing restores selectability
        selector.remove_forbidden(HOME);
        assert_eq!(selector.select(&[HOME, VISITED]), Some(HOME));
    }

    #[test]
    fn test_plmn_selection_preferred_lists_in_order() {
        let mut selector = PlmnSelector::new(HOME);
        selector.set_user_preferred(vec![OTHER]);
        selector.set_operator_preferred(vec![VISITED]);
        // HPLMN absent: user-preferred outranks operator-preferred
        assert_eq!(selector.select(&[VISITED, OTHER]), Some(OTHER));
        // Only the operator-preferred is available
        assert_eq!(selector.select(&[VISITED]), Some(VISITED));
    }

    #[test]
    fn test_plmn_selection_manual_mode_overrides_forbidden() {
        let mut selector = PlmnSelector::new(HOME);
        selector.add_forbidden(VISITED);
        selector.select_manual(VISITED);
        assert_eq!(selector.mode(), PlmnSelectionMode::Manual);
        // Manual selection may attempt a forbidden PLMN
        assert_eq!(selector.select(&[HOME, VISITED]), Some(VISITED));
        // Manual PLMN not available: nothing is selected (no fallback)
        assert_eq!(selector.select(&[HOME]), None);
        // Back to automatic
        selector.set_automatic();
        assert_eq!(selector.select(&[HOME, VISITED]), Some(HOME));
    }

    // --- Issue #49: the wiring that was missing ---

    /// CRITERION 6: with two PLMNs available on the radio and the HPLMN forbidden
    /// (reject cause #11/#74), the UE selects the OTHER available PLMN rather than
    /// staying in limited service.
    ///
    /// This is the case the hardcoded `[home_plmn]` candidate list made impossible:
    /// with one forbidden candidate and nothing else to choose from, selection had
    /// no answer however the radio looked.
    #[test]
    fn a_forbidden_hplmn_is_replaced_by_another_available_plmn() {
        let mut selector = PlmnSelector::new(HOME);
        // The radio reports two PLMNs, as `CellSelector::available_plmns` now does.
        selector.set_available_plmns(vec![HOME, VISITED]);
        assert_eq!(selector.available_plmns(), &[HOME, VISITED]);

        // Before the rejection the HPLMN wins, which is the happy path.
        assert_eq!(selector.select(selector.available_plmns()), Some(HOME));

        // Reject cause #11 for the HPLMN.
        selector.add_forbidden(HOME);
        assert_eq!(
            selector.select(selector.available_plmns()),
            Some(VISITED),
            "a forbidden HPLMN must not leave the UE in limited service when \
             another PLMN is available"
        );

        // And with NOTHING else available it correctly has no answer -- so the
        // assertion above is about the second candidate, not about ignoring the
        // forbidden list.
        assert_eq!(selector.select(&[HOME]), None);
    }

    /// CRITERION 7: registered on a VPLMN, the periodic higher-priority search
    /// eventually fires; registered on the HPLMN it never does.
    ///
    /// Both halves matter. The guard was permanently taken because the RPLMN was
    /// recorded as the HPLMN unconditionally, so `tick()` always returned false and
    /// the call site that drives the search was dead code.
    #[test]
    fn the_periodic_higher_priority_search_fires_on_a_vplmn_and_not_on_the_hplmn() {
        let interval = 6 * 60; // the spec's lower bound, so the loop stays short

        // On a VPLMN: fires exactly on the interval, and not before.
        let mut selector = PlmnSelector::new(HOME);
        selector.set_hp_search_interval(interval);
        selector.set_registered_plmn(Some(VISITED));
        for elapsed in 1..interval {
            assert!(
                !selector.tick(),
                "the search must not fire after only {elapsed}s"
            );
        }
        assert!(selector.tick(), "it must fire once timer T elapses");

        // On the HPLMN: never, however long it runs.
        let mut selector = PlmnSelector::new(HOME);
        selector.set_hp_search_interval(interval);
        selector.set_registered_plmn(Some(HOME));
        for _ in 0..(interval * 2) {
            assert!(
                !selector.tick(),
                "a UE already on its HPLMN has no higher-priority PLMN to seek"
            );
        }
    }

    /// The serving PLMN is what a successful registration records as the RPLMN, so
    /// the selector has to hold it (issue #49).
    #[test]
    fn the_serving_plmn_is_recorded_and_readable() {
        let mut selector = PlmnSelector::new(HOME);
        assert_eq!(
            selector.serving_plmn(),
            None,
            "unknown until a cell is selected"
        );
        selector.set_serving_plmn(Some(VISITED));
        assert_eq!(selector.serving_plmn(), Some(VISITED));

        // And it is independent of the REGISTERED PLMN: the UE can be camped on a
        // cell it has not registered through.
        assert_eq!(selector.registered_plmn(), None);
    }

    #[test]
    fn a_configured_plmn_parses_and_keeps_its_mnc_digit_count() {
        assert_eq!(
            parse_configured_plmn("001-01"),
            Some(Plmn::new(1, 1, false))
        );
        assert_eq!(
            parse_configured_plmn("262-030"),
            Some(Plmn::new(262, 30, true)),
            "a 3-digit MNC must stay a 3-digit MNC"
        );
        // The digit count is significant: these are DIFFERENT networks and they
        // encode differently on the wire (TS 23.003 §2.2).
        assert_ne!(
            parse_configured_plmn("262-03"),
            parse_configured_plmn("262-030")
        );
        // Whitespace is tolerated, because a YAML list invites it.
        assert_eq!(
            parse_configured_plmn(" 001 - 01 "),
            Some(Plmn::new(1, 1, false))
        );
    }

    #[test]
    fn a_malformed_configured_plmn_is_rejected_rather_than_guessed() {
        // A wrong entry in a PREFERENCE list sends the UE to the wrong network, so
        // none of these may resolve to something plausible.
        for text in [
            "", "001", "001-", "-01", "01-01", "0001-01", "001-1", "001-0001", "abc-01", "001-0a",
            "001/01",
            // A SIGNED field is the case a length check alone lets through:
            // `"+01".parse::<u16>()` is Ok(1), so without the explicit digit check
            // this would silently resolve to MNC 01 -- a different network, from
            // text nobody meant as one.
            "001-+01", "+01-01", "001--1",
        ] {
            assert!(
                parse_configured_plmn(text).is_none(),
                "'{text}' must not parse"
            );
        }
    }

    #[test]
    fn test_plmn_higher_priority_periodic_search_timer() {
        let mut selector = PlmnSelector::new(HOME);
        selector.set_hp_search_interval(10 * 60);
        assert_eq!(selector.hp_search_interval_secs, 10 * 60);
        // Interval clamped to the 6-minute lower bound
        selector.set_hp_search_interval(1);
        assert_eq!(selector.hp_search_interval_secs, 6 * 60);

        // Not registered: no periodic search
        assert!(!selector.tick());

        // Registered on the HPLMN: no periodic search
        selector.set_registered_plmn(Some(HOME));
        assert!(!selector.tick());

        // Roaming on a visited PLMN: the search fires after the interval
        selector.set_registered_plmn(Some(VISITED));
        for _ in 0..(6 * 60 - 1) {
            assert!(!selector.tick());
        }
        assert!(selector.tick(), "search must be due after the interval");
        // ... and the timer restarts
        assert!(!selector.tick());
    }

    #[test]
    fn test_plmn_from_bcd_roundtrip() {
        // 999/70 (2-digit MNC): MCC digits 9,9,9; MNC digits 7,0
        // byte0 = 0x99, byte1 = 0xF9, byte2 = 0x07
        assert_eq!(plmn_from_bcd(&[0x99, 0xF9, 0x07]), HOME);
        // 310/410 (3-digit MNC): byte0 = 0x13, byte1 = 0x00, byte2 = 0x14
        assert_eq!(plmn_from_bcd(&[0x13, 0x00, 0x14]), OTHER);
    }

    #[test]
    fn test_forbidden_tai() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add a cell
        selector
            .cells
            .insert(1, make_cell_with_sib1(-80, plmn, 1, false, false));

        // Add TAI to forbidden list
        selector.add_forbidden_tai_roaming(Tai::new(plmn, 1));
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_none() || result.unwrap().cell_id == 0);
    }
}
