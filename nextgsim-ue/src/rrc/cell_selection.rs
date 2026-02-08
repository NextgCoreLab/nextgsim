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
//! (PLMN matching is not required)
//!
//! # Reference
//! - 3GPP TS 38.304: NR; User Equipment (UE) procedures in Idle mode and RRC Inactive state
//! - UERANSIM: src/ue/rrc/idle.cpp

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
    pub nci: i64,       // NR Cell Identity
    pub tac: u32,       // Tracking Area Code
    pub plmn: Plmn,
    // Cell selection parameters (from CellSelectionInfo)
    pub q_rx_lev_min: i8,        // Minimum required RX level
    pub q_rx_lev_min_offset: Option<u8>,
    pub q_qual_min: Option<i8>,  // Minimum quality level
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
    /// Srxlev = Q_rxlevmeas - (Q_rxlevmin + Q_rxlevminoffset)
    /// Per 3GPP TS 38.304 Section 5.2.3.2
    pub fn srxlev(&self) -> i32 {
        let q_rxlev_min = self.sib1.q_rx_lev_min as i32 * 2; // Convert to dBm
        let q_rxlev_min_offset = self.sib1.q_rx_lev_min_offset.unwrap_or(0) as i32 * 2;
        self.dbm - (q_rxlev_min + q_rxlev_min_offset)
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
}

impl Default for CellReselectionParams {
    fn default() -> Self {
        Self {
            q_hyst: DEFAULT_Q_HYST_DB,
            t_reselection: CELL_RESELECTION_TIME_TO_TRIGGER_MS,
            reselection_candidate: None,
            candidate_better_since: None,
        }
    }
}

/// Cell selection and reselection manager
pub struct CellSelector {
    /// Detected cells indexed by cell_id
    cells: HashMap<i32, CellDescription>,
    /// Currently selected cell
    current_cell: ActiveCellInfo,
    /// Selected PLMN (from NAS)
    selected_plmn: Option<Plmn>,
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
}

impl CellSelector {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
            current_cell: ActiveCellInfo::default(),
            selected_plmn: None,
            forbidden_tai_roaming: Vec::new(),
            forbidden_tai_rps: Vec::new(),
            started_time: Instant::now(),
            last_failure_logged: None,
            reselection_params: CellReselectionParams::default(),
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
                    cell_id, dbm, self.cells.len()
                );
                return CellChangeEvent::CellDetected(cell_id);
            }
            CellChangeEvent::None
        } else {
            if consider_lost {
                // Cell lost
                let was_active = self.current_cell.cell_id == cell_id;
                self.cells.remove(&cell_id);
                tracing::debug!(
                    "Cell lost: cell_id={}, was_active={}, total_cells={}",
                    cell_id, was_active, self.cells.len()
                );
                if was_active {
                    let old_cell = std::mem::take(&mut self.current_cell);
                    return CellChangeEvent::ActiveCellLost(old_cell);
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
                cell_id, mib.is_barred, mib.is_intra_freq_reselect_allowed
            );
            cell.mib = mib;
        }
    }

    /// Update SIB1 information for a cell
    pub fn update_sib1(&mut self, cell_id: i32, sib1: Sib1Info) {
        if let Some(cell) = self.cells.get_mut(&cell_id) {
            tracing::debug!(
                "SIB1 updated for cell {}: plmn={}, tac={}, reserved={}",
                cell_id, sib1.plmn, sib1.tac, sib1.is_reserved
            );
            cell.sib1 = sib1;
        }
    }

    /// Get the current active cell
    pub fn current_cell(&self) -> &ActiveCellInfo {
        &self.current_cell
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
    /// Returns Some(cell_info) if a new cell was selected, None if no change
    pub fn perform_cell_selection(&mut self) -> Option<ActiveCellInfo> {
        let elapsed = self.started_time.elapsed();

        // Wait for initial discovery period
        if elapsed < Duration::from_millis(CELL_SELECTION_STARTUP_DELAY_MS) && self.cells.is_empty() {
            return None;
        }

        // Wait longer if no PLMN selected yet
        if elapsed < Duration::from_millis(4000) && self.selected_plmn.is_none() {
            return None;
        }

        let last_cell = self.current_cell.clone();
        let should_log_errors = last_cell.cell_id != 0 ||
            self.last_failure_logged.is_none_or(|t| {
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

        // Apply cell reselection with hysteresis if we already have a serving cell
        if last_cell.cell_id != 0 && cell_found {
            let should_reselect = self.evaluate_cell_reselection(
                last_cell.cell_id,
                cell_info.cell_id,
            );
            if !should_reselect {
                // Keep current cell, reset candidate
                cell_info = last_cell.clone();
            }
        }

        // Update current cell
        self.current_cell = cell_info.clone();

        // Log if selection changed
        if cell_info.cell_id != 0 && cell_info.cell_id != last_cell.cell_id {
            tracing::info!(
                "Cell reselection: id={}, plmn={}, tac={}, category={:?}",
                cell_info.cell_id, cell_info.plmn, cell_info.tac, cell_info.category
            );
            // Clear reselection candidate on successful reselection
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
        }

        // Return new cell info if changed
        if cell_info.cell_id != last_cell.cell_id {
            Some(cell_info)
        } else {
            None
        }
    }

    /// Evaluate if cell reselection should occur based on hysteresis and time-to-trigger
    /// Per 3GPP TS 38.304 Section 5.2.4
    fn evaluate_cell_reselection(&mut self, current_cell_id: i32, best_cell_id: i32) -> bool {
        // If best cell is same as current, no reselection needed
        if current_cell_id == best_cell_id {
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        }

        // Get signal strengths
        let current_dbm = self.cells.get(&current_cell_id).map(|c| c.dbm).unwrap_or(i32::MIN);
        let best_dbm = self.cells.get(&best_cell_id).map(|c| c.dbm).unwrap_or(i32::MIN);

        // Apply hysteresis: new cell must be better by q_hyst dB
        let required_margin = self.reselection_params.q_hyst;
        if best_dbm <= current_dbm + required_margin {
            // Candidate is not sufficiently better
            self.reselection_params.reselection_candidate = None;
            self.reselection_params.candidate_better_since = None;
            return false;
        }

        // Track how long this candidate has been better (time-to-trigger)
        if self.reselection_params.reselection_candidate != Some(best_cell_id) {
            // New candidate
            self.reselection_params.reselection_candidate = Some(best_cell_id);
            self.reselection_params.candidate_better_since = Some(Instant::now());
            tracing::debug!(
                "Cell reselection candidate: cell_id={}, dbm={} (current: cell_id={}, dbm={})",
                best_cell_id, best_dbm, current_cell_id, current_dbm
            );
            return false;
        }

        // Check if time-to-trigger has elapsed
        if let Some(since) = self.reselection_params.candidate_better_since {
            if since.elapsed() >= Duration::from_millis(self.reselection_params.t_reselection) {
                tracing::debug!(
                    "Cell reselection triggered: cell_id={} -> cell_id={} (margin: {}dB, elapsed: {:?})",
                    current_cell_id, best_cell_id, best_dbm - current_dbm, since.elapsed()
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

            // Check signal level (Srxlev > 0)
            if cell.srxlev() <= 0 {
                report.low_signal_cells += 1;
                continue;
            }

            // Cell is suitable
            candidates.push((cell_id, cell.dbm));
        }

        if candidates.is_empty() {
            return false;
        }

        // Sort by signal strength (highest first)
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

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
        candidates.sort_by(|a, b| b.1.cmp(&a.1));

        // Then prioritize cells matching selected PLMN (stable sort)
        candidates.sort_by(|a, b| b.2.cmp(&a.2));

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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cell_with_sib1(dbm: i32, plmn: Plmn, tac: u32, barred: bool, reserved: bool) -> CellDescription {
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
            },
        }
    }

    #[test]
    fn test_cell_selection_suitable_cell() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add a suitable cell
        selector.cells.insert(1, make_cell_with_sib1(-80, plmn, 1, false, false));
        // Override started_time to bypass startup delay
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_some());
        let cell = result.unwrap();
        assert_eq!(cell.cell_id, 1);
        assert_eq!(cell.category, CellCategory::SuitableCell);
    }

    #[test]
    fn test_cell_selection_acceptable_cell() {
        let mut selector = CellSelector::new();
        let plmn1 = Plmn::new(999, 70, false);
        let plmn2 = Plmn::new(999, 71, false);
        selector.set_selected_plmn(Some(plmn1));

        // Add a cell with different PLMN (acceptable but not suitable)
        selector.cells.insert(1, make_cell_with_sib1(-80, plmn2, 1, false, false));
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
        selector.cells.insert(1, make_cell_with_sib1(-80, plmn, 1, true, false));
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
        selector.cells.insert(1, make_cell_with_sib1(-90, plmn, 1, false, false));
        selector.cells.insert(2, make_cell_with_sib1(-70, plmn, 1, false, false)); // Best
        selector.cells.insert(3, make_cell_with_sib1(-85, plmn, 1, false, false));
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

    #[test]
    fn test_forbidden_tai() {
        let mut selector = CellSelector::new();
        let plmn = Plmn::new(999, 70, false);
        selector.set_selected_plmn(Some(plmn));

        // Add a cell
        selector.cells.insert(1, make_cell_with_sib1(-80, plmn, 1, false, false));

        // Add TAI to forbidden list
        selector.add_forbidden_tai_roaming(Tai::new(plmn, 1));
        selector.started_time = Instant::now() - Duration::from_secs(10);

        let result = selector.perform_cell_selection();
        assert!(result.is_none() || result.unwrap().cell_id == 0);
    }
}
