//! Measurement Framework for Handover
//!
//! Implements measurement configuration and reporting per 3GPP TS 38.331.
//!
//! # Overview
//!
//! In `RRC_CONNECTED` state, the UE performs measurements and reports them
//! to the network based on measurement configuration provided by the gNB.
//! The network uses these reports to make handover decisions.
//!
//! # Measurement Types
//!
//! - **SS-RSRP**: Synchronization Signal Reference Signal Received Power
//! - **SS-RSRQ**: Synchronization Signal Reference Signal Received Quality
//! - **SS-SINR**: Synchronization Signal Signal-to-Interference-plus-Noise Ratio
//!
//! # Event Types (for event-triggered reporting)
//!
//! - **A1**: Serving becomes better than threshold
//! - **A2**: Serving becomes worse than threshold
//! - **A3**: Neighbor becomes amount better than serving
//! - **A4**: Neighbor becomes better than threshold
//! - **A5**: Serving becomes worse than threshold1 AND neighbor becomes better than threshold2
//! - **A6**: Neighbor becomes offset better than the `SCell` (§5.5.4.7)
//! - **B1/B2**: inter-RAT: an E-UTRA neighbour above a threshold, B2 also
//!   requiring a weak PCell (§5.5.4.8, §5.5.4.9)
//!
//! # Triggering state machine
//!
//! Each event has an **entering** and a **leaving** inequality (§5.5.4), which
//! differ by the sign of the hysteresis and so create a dead band: a cell that
//! has entered stays triggered until its level has moved a full 2·Hys back the
//! other way. Membership is tracked per cell in `cellsTriggeredList` (§5.5.4.1),
//! so several neighbours can hold one `measId` triggered and one of them
//! dropping out does not release the event. `timeToTrigger` gates both
//! directions, per cell.
//!
//! # What this manager does not model
//!
//! - **Layer-3 filtering** (§5.5.3.2): a measurement is the level the radio last
//!   reported, unfiltered.
//! - **`cellIndividualOffset` for NR cells** (Ocn/Ocs): zero for every NR cell,
//!   as §5.5.4 permits when it is not configured. E-UTRA neighbours do carry
//!   both offsets, because B1/B2's inequalities are written around them.
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification, §5.5.4
//! - 3GPP TS 38.215: NR; Physical layer measurements

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::time::{Duration, Instant};

/// Measurement quantity types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MeasQuantity {
    /// SS-RSRP (Reference Signal Received Power)
    #[default]
    SsRsrp,
    /// SS-RSRQ (Reference Signal Received Quality)
    SsRsrq,
    /// SS-SINR (Signal-to-Interference-plus-Noise Ratio)
    SsSinr,
}

/// Measurement event types for event-triggered reporting
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasEventType {
    /// A1: Serving becomes better than threshold
    A1,
    /// A2: Serving becomes worse than threshold
    A2,
    /// A3: Neighbor becomes offset better than serving
    A3,
    /// A4: Neighbor becomes better than threshold
    A4,
    /// A5: Serving < threshold1 AND neighbor > threshold2
    A5,
    /// A6: Neighbour becomes offset better than the `SCell` (TS 38.331 §5.5.4.7)
    ///
    /// The event's reference is the **SCell**, not the PCell, so it can only
    /// trigger once [`MeasurementManager::set_scell`] has been given one.
    A6,
    /// B1: an inter-RAT (E-UTRA) neighbour becomes better than a threshold
    /// (TS 38.331 §5.5.4.8)
    B1,
    /// B2: the PCell becomes worse than threshold1 AND an inter-RAT (E-UTRA)
    /// neighbour becomes better than threshold2 (TS 38.331 §5.5.4.9)
    B2,
}

/// Report trigger type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportTriggerType {
    /// Event-triggered reporting
    Event(MeasEventType),
    /// Periodic reporting
    Periodic,
}

/// Measurement report trigger configuration
#[derive(Debug, Clone)]
pub struct ReportTriggerConfig {
    /// Type of trigger
    pub trigger_type: ReportTriggerType,
    /// Threshold for A1/A2/A4 events (dBm)
    pub threshold: Option<i32>,
    /// Threshold1 for A5 event (serving cell, dBm)
    pub threshold1: Option<i32>,
    /// Threshold2 for A5 event (neighbor cell, dBm)
    pub threshold2: Option<i32>,
    /// Offset for A3 event (dB)
    pub a3_offset: Option<i32>,
    /// Offset for A6 event (dB), i.e. `a6-Offset` of TS 38.331 §5.5.4.7
    pub a6_offset: Option<i32>,
    /// Hysteresis (dB * 2, e.g., 4 = 2 dB)
    pub hysteresis: i32,
    /// Time-to-trigger (ms)
    pub time_to_trigger: u64,
}

impl Default for ReportTriggerConfig {
    fn default() -> Self {
        Self {
            trigger_type: ReportTriggerType::Event(MeasEventType::A3),
            threshold: None,
            threshold1: None,
            threshold2: None,
            a3_offset: Some(3), // 3 dB offset
            a6_offset: None,
            hysteresis: 2,        // 1 dB
            time_to_trigger: 640, // 640 ms
        }
    }
}

/// Measurement configuration for a single measurement ID
#[derive(Debug, Clone)]
pub struct MeasConfig {
    /// Measurement ID
    pub meas_id: u8,
    /// Measurement object ID (identifies frequency/cell to measure)
    pub meas_object_id: u8,
    /// Report configuration ID
    pub report_config_id: u8,
    /// Measurement quantity
    pub quantity: MeasQuantity,
    /// Report trigger configuration
    pub trigger_config: ReportTriggerConfig,
    /// Report amount (0 = infinite)
    pub report_amount: u32,
    /// Report interval (ms)
    pub report_interval: u64,
    /// Max report cells
    pub max_report_cells: u8,
}

impl Default for MeasConfig {
    fn default() -> Self {
        Self {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            quantity: MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig::default(),
            report_amount: 8,
            report_interval: 480,
            max_report_cells: 4,
        }
    }
}

/// Measurement result for a single cell
#[derive(Debug, Clone, Default)]
pub struct CellMeasResult {
    /// Physical cell ID
    pub pci: u32,
    /// NR Cell Global Identity (if available)
    pub nci: Option<i64>,
    /// SS-RSRP measurement (dBm)
    pub rsrp: Option<i32>,
    /// SS-RSRQ measurement (dB)
    pub rsrq: Option<i32>,
    /// SS-SINR measurement (dB)
    pub sinr: Option<i32>,
}

/// Identifies an inter-RAT (E-UTRA) cell: `measObjectEUTRA` is per carrier
/// frequency, and the physical cell identity is unique within it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EutraCellKey {
    /// E-UTRA carrier frequency (EARFCN)
    pub earfcn: u32,
    /// E-UTRA physical cell identity (0-503)
    pub pci: u16,
}

impl EutraCellKey {
    /// An E-UTRA cell on `earfcn` with physical cell identity `pci`.
    pub fn new(earfcn: u32, pci: u16) -> Self {
        Self { earfcn, pci }
    }
}

/// Measurement result for a single inter-RAT (E-UTRA) cell.
///
/// `cell_individual_offset` is Ocn of TS 38.331 §5.5.4.8/§5.5.4.9; the
/// per-frequency offset Ofn (`eutra-Q-OffsetRange`) is held by the manager
/// against the EARFCN, since it applies to every cell on that carrier.
#[derive(Debug, Clone)]
pub struct EutraMeasResult {
    /// Which cell this result is for
    pub cell: EutraCellKey,
    /// RSRP measurement (dBm)
    pub rsrp: Option<i32>,
    /// Cell specific offset Ocn (dB)
    pub cell_individual_offset: i32,
}

/// A cell in an event's `cellsTriggeredList` (TS 38.331 §5.5.4.1).
///
/// Neighbour events name a cell, and an inter-RAT event names one that has no
/// simulator cell id — hence the two arms rather than a bare `i32`. `Ord` is
/// derived so the list can be a set: the ordering is an arbitrary but stable
/// key order, not a signal-strength order, which the report path applies
/// separately.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TriggeringCell {
    /// An NR neighbour, by simulator cell id
    Nr(i32),
    /// An inter-RAT (E-UTRA) neighbour
    Eutra(EutraCellKey),
}

impl TriggeringCell {
    /// The NR cell id, when the trigger was an NR neighbour.
    pub fn nr_cell_id(&self) -> Option<i32> {
        match self {
            TriggeringCell::Nr(cell_id) => Some(*cell_id),
            TriggeringCell::Eutra(_) => None,
        }
    }
}

/// Measurement report to be sent to network
#[derive(Debug, Clone)]
pub struct MeasurementReport {
    /// Measurement ID
    pub meas_id: u8,
    /// Serving cell results
    pub serving_cell: CellMeasResult,
    /// Neighbor cell results
    pub neighbor_cells: Vec<CellMeasResult>,
    /// Inter-RAT (E-UTRA) neighbour results, for a B1/B2 report
    pub eutra_neighbor_cells: Vec<EutraMeasResult>,
    /// The event's `cellsTriggeredList` at the moment the report was generated,
    /// strongest first (§5.5.4.1). Empty for a periodic report, which is not
    /// triggered by any cell. The neighbour lists above lead with these cells.
    pub triggered_cells: Vec<TriggeringCell>,
    /// Timestamp of the report
    pub timestamp: Instant,
}

/// Which of the two inequalities TS 38.331 §5.5.4 defines for an event.
///
/// They differ by the sign of the hysteresis, so a cell that has entered needs
/// its level to move a full 2·Hys before it leaves again.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Condition {
    /// The entering inequality (`…-1`, and `…-2` for a two-part event), all
    /// parts of which must hold.
    Entering,
    /// The leaving inequality. A two-part event leaves when *either* part holds
    /// (§5.5.4.6 A5-3/A5-4, §5.5.4.9 B2-3/B2-4), not both.
    Leaving,
}

/// Where one applicable cell stands against an event's two inequalities.
struct CellVerdict {
    /// The cell, as it appears in `cellsTriggeredList`
    cell: TriggeringCell,
    /// The level the event's inequality is written in terms of: Mn for an NR
    /// cell, Mn + Ofn + Ocn for an inter-RAT one. Orders the report.
    level: i32,
    /// The entering inequality holds now
    entering: bool,
    /// The leaving inequality holds now
    leaving: bool,
}

/// One of the two inequalities of TS 38.331 §5.5.4 for one NR neighbour cell.
///
/// The single place the NR neighbour events' arithmetic lives: the per-event
/// applicable-cell sweep and the conditional-reconfiguration runtime
/// ([`MeasurementManager::candidate_condition_holds`]) both go through it, so a
/// CHO candidate is judged by exactly the inequality its measurement event uses.
///
/// `reference_rsrp` is the level the event compares against — Mp of the PCell for
/// A3 and A5, Ms of the SCell for A6 (§5.5.4.7) — and is `None` for the
/// threshold-only A4, which needs no reference. An event that *does* need one and
/// has none cannot be entered, and cannot be sustained either, so it reports
/// `false` for entering and `true` for leaving.
///
/// Hysteresis is halved because `ReportTriggerConfig::hysteresis` is in the
/// signalled 0.5 dB units. Ocn and Ocs are zero: no NR
/// `cellIndividualOffset` is modelled, which §5.5.4 allows when it is not
/// configured. Events that no *neighbour* cell can satisfy (A1 and A2 measure
/// the serving cell; B1/B2 are inter-RAT) never enter here and always leave,
/// rather than being silently accepted.
///
/// A threshold event whose threshold is absent does not trigger, and does not
/// stay triggered: the code this replaced substituted `i32::MIN`/`i32::MAX` and
/// then subtracted the hysteresis from it, overflowing.
fn nr_condition(
    which: Condition,
    event_type: MeasEventType,
    trigger: &ReportTriggerConfig,
    reference_rsrp: Option<i32>,
    neighbour_rsrp: i32,
) -> bool {
    let hyst = trigger.hysteresis / 2;
    let entering = which == Condition::Entering;

    match event_type {
        // A3-1: Mn + Ocn - Hys > Mp + Ofp + Ocp + Off
        // A3-2: Mn + Ocn + Hys < Mp + Ofp + Ocp + Off
        MeasEventType::A3 => {
            let off = trigger.a3_offset.unwrap_or(0);
            match reference_rsrp {
                Some(mp) if entering => neighbour_rsrp > mp + off + hyst,
                Some(mp) => neighbour_rsrp < mp + off - hyst,
                None => !entering,
            }
        }
        // A4-1: Mn + Ocn - Hys > Thresh
        // A4-2: Mn + Ocn + Hys < Thresh
        MeasEventType::A4 => match trigger.threshold {
            Some(thresh) if entering => neighbour_rsrp > thresh + hyst,
            Some(thresh) => neighbour_rsrp < thresh - hyst,
            None => !entering,
        },
        // A5-1 and A5-2: Mp + Hys < Thresh1 AND Mn + Ocn - Hys > Thresh2
        // A5-3  or A5-4: Mp - Hys > Thresh1  OR Mn + Ocn + Hys < Thresh2
        MeasEventType::A5 => match (reference_rsrp, trigger.threshold1, trigger.threshold2) {
            (Some(mp), Some(thresh1), Some(thresh2)) if entering => {
                mp < thresh1 - hyst && neighbour_rsrp > thresh2 + hyst
            }
            (Some(mp), Some(thresh1), Some(thresh2)) => {
                mp > thresh1 + hyst || neighbour_rsrp < thresh2 - hyst
            }
            _ => !entering,
        },
        // A6-1: Mn + Ocn - Hys > Ms + Ocs + Off, with Ms the SCell
        // A6-2: Mn + Ocn + Hys < Ms + Ocs + Off
        MeasEventType::A6 => {
            let off = trigger.a6_offset.unwrap_or(0);
            match reference_rsrp {
                Some(ms) if entering => neighbour_rsrp > ms + off + hyst,
                Some(ms) => neighbour_rsrp < ms + off - hyst,
                None => !entering,
            }
        }
        MeasEventType::A1 | MeasEventType::A2 | MeasEventType::B1 | MeasEventType::B2 => !entering,
    }
}

/// Event state tracking for a single event
#[derive(Debug, Clone, Default)]
struct EventState {
    /// `cellsTriggeredList` (TS 38.331 §5.5.4.1): the cells currently holding
    /// this `measId` triggered. The event is triggered while this is non-empty.
    cells_triggered: BTreeSet<TriggeringCell>,
    /// Per cell, when its entering condition first held while it was *not* in
    /// `cells_triggered` — the start of `timeToTrigger`.
    entering_since: BTreeMap<TriggeringCell, Instant>,
    /// Per cell, when its leaving condition first held while it *was* in
    /// `cells_triggered`. §5.5.4.1 applies `timeToTrigger` to both directions.
    leaving_since: BTreeMap<TriggeringCell, Instant>,
    /// Number of reports sent for this event
    reports_sent: u32,
    /// Last report time
    last_report: Option<Instant>,
}

impl EventState {
    /// Forget every cell, keeping the report counters: used when the reference
    /// the events are measured against changes under them.
    fn clear_triggers(&mut self) {
        self.cells_triggered.clear();
        self.entering_since.clear();
        self.leaving_since.clear();
    }
}

/// Measurement manager for handling RRC measurements
pub struct MeasurementManager {
    /// Active measurement configurations
    configs: HashMap<u8, MeasConfig>,
    /// Current measurements per cell (`cell_id` -> measurement)
    measurements: HashMap<i32, CellMeasResult>,
    /// Event states per measurement ID
    event_states: HashMap<u8, EventState>,
    /// Serving cell ID (the PCell)
    serving_cell_id: Option<i32>,
    /// Secondary cell ID, the reference for event A6 (TS 38.331 §5.5.4.7).
    /// `None` until an SCell is configured, and no A6 event can trigger then.
    scell_id: Option<i32>,
    /// Inter-RAT (E-UTRA) measurements, for events B1/B2
    eutra_measurements: HashMap<EutraCellKey, EutraMeasResult>,
    /// Per-carrier E-UTRA offset Ofn (`eutra-Q-OffsetRange`), in dB
    eutra_freq_offsets: HashMap<u32, i32>,
    /// Pending reports to be sent
    pending_reports: Vec<MeasurementReport>,
}

impl MeasurementManager {
    pub fn new() -> Self {
        Self {
            configs: HashMap::new(),
            measurements: HashMap::new(),
            event_states: HashMap::new(),
            serving_cell_id: None,
            scell_id: None,
            eutra_measurements: HashMap::new(),
            eutra_freq_offsets: HashMap::new(),
            pending_reports: Vec::new(),
        }
    }

    /// Set the serving cell
    pub fn set_serving_cell(&mut self, cell_id: Option<i32>) {
        if self.serving_cell_id == cell_id {
            return;
        }
        self.serving_cell_id = cell_id;
        // Every event's inequality is written against the serving cell, so a
        // change invalidates each cellsTriggeredList wholesale.
        for state in self.event_states.values_mut() {
            state.clear_triggers();
        }
    }

    /// The serving cell (PCell), if one is set.
    pub fn serving_cell_id(&self) -> Option<i32> {
        self.serving_cell_id
    }

    /// Set the secondary cell that event A6 measures against (TS 38.331
    /// §5.5.4.7: "consider the (secondary) cell corresponding to the measObjectNR
    /// associated to this event to be the serving cell").
    ///
    /// Carrier aggregation is not modelled in the simulator, so nothing in the
    /// UE sets an SCell today and A6 therefore never fires in a live run — see
    /// issue #112. The event is implemented and evaluated so that an SCell, once
    /// there is one, needs no measurement work.
    pub fn set_scell(&mut self, cell_id: Option<i32>) {
        self.scell_id = cell_id;
    }

    /// The secondary cell A6 measures against, if one is configured.
    pub fn scell_id(&self) -> Option<i32> {
        self.scell_id
    }

    /// The last reported level for a cell, if it has been measured.
    pub fn rsrp(&self, cell_id: i32) -> Option<i32> {
        self.measurements.get(&cell_id).and_then(|m| m.rsrp)
    }

    /// Record a measurement for an inter-RAT (E-UTRA) cell (events B1/B2).
    ///
    /// The simulator has no E-UTRA radio — RLS carries NR cells only — so the
    /// source is `UeConfig::eutra_neighbours`, a configured stand-in fed in by
    /// `RrcTask::update_eutra_measurements` on every measurement cycle. The level
    /// is therefore whatever the configuration says and never changes with
    /// distance, fading or the channel model.
    pub fn update_eutra_measurement(&mut self, cell: EutraCellKey, rsrp: i32) {
        let result = self
            .eutra_measurements
            .entry(cell)
            .or_insert_with(|| EutraMeasResult {
                cell,
                rsrp: None,
                cell_individual_offset: 0,
            });
        result.rsrp = Some(rsrp);
    }

    /// Set Ocn, the cell specific offset of one inter-RAT cell (dB).
    pub fn set_eutra_cell_offset(&mut self, cell: EutraCellKey, offset_db: i32) {
        let result = self
            .eutra_measurements
            .entry(cell)
            .or_insert_with(|| EutraMeasResult {
                cell,
                rsrp: None,
                cell_individual_offset: 0,
            });
        result.cell_individual_offset = offset_db;
    }

    /// Set Ofn, the offset of an E-UTRA carrier frequency (`eutra-Q-OffsetRange`,
    /// dB), which applies to every cell measured on that EARFCN.
    pub fn set_eutra_frequency_offset(&mut self, earfcn: u32, offset_db: i32) {
        self.eutra_freq_offsets.insert(earfcn, offset_db);
    }

    /// Drop an inter-RAT measurement.
    pub fn remove_eutra_measurement(&mut self, cell: &EutraCellKey) {
        self.eutra_measurements.remove(cell);
    }

    /// All inter-RAT measurements.
    pub fn eutra_measurements(&self) -> &HashMap<EutraCellKey, EutraMeasResult> {
        &self.eutra_measurements
    }

    /// Add a measurement configuration
    pub fn add_config(&mut self, config: MeasConfig) {
        let meas_id = config.meas_id;
        self.configs.insert(meas_id, config);
        self.event_states.insert(meas_id, EventState::default());
    }

    /// Remove a measurement configuration
    pub fn remove_config(&mut self, meas_id: u8) {
        self.configs.remove(&meas_id);
        self.event_states.remove(&meas_id);
    }

    /// Clear all measurement configurations
    pub fn clear_configs(&mut self) {
        self.configs.clear();
        self.event_states.clear();
    }

    /// Update measurement for a cell
    pub fn update_measurement(&mut self, cell_id: i32, rsrp: i32) {
        let result = self
            .measurements
            .entry(cell_id)
            .or_insert_with(|| CellMeasResult {
                pci: cell_id as u32,
                ..Default::default()
            });
        result.rsrp = Some(rsrp);
    }

    /// Remove measurement for a cell
    pub fn remove_measurement(&mut self, cell_id: i32) {
        self.measurements.remove(&cell_id);
    }

    /// Get pending reports and clear them
    pub fn take_pending_reports(&mut self) -> Vec<MeasurementReport> {
        std::mem::take(&mut self.pending_reports)
    }

    /// Evaluate measurement events and generate reports if needed
    pub fn evaluate_events(&mut self) {
        let serving_cell_id = match self.serving_cell_id {
            Some(id) => id,
            None => return,
        };

        // `None` when the PCell has not been measured yet. Events written against
        // Mp (A1, A2, A3, A5, B2) then have no applicable cell at all, rather
        // than being evaluated against a substituted `i32::MIN` that overflows
        // the moment a leaving inequality subtracts the hysteresis from it.
        let serving_rsrp = self.measurements.get(&serving_cell_id).and_then(|m| m.rsrp);

        // Evaluate each measurement configuration
        let configs: Vec<_> = self.configs.values().cloned().collect();
        for config in configs {
            self.evaluate_event(&config, serving_cell_id, serving_rsrp);
        }
    }

    fn evaluate_event(
        &mut self,
        config: &MeasConfig,
        serving_cell_id: i32,
        serving_rsrp: Option<i32>,
    ) {
        let meas_id = config.meas_id;
        let trigger = &config.trigger_config;
        let ttt = Duration::from_millis(trigger.time_to_trigger);

        // A periodic report is not triggered by any cell, so it has no
        // cellsTriggeredList and reports on its interval regardless.
        let periodic = matches!(trigger.trigger_type, ReportTriggerType::Periodic);
        let verdicts = match trigger.trigger_type {
            ReportTriggerType::Periodic => Vec::new(),
            ReportTriggerType::Event(event_type) => {
                self.cell_verdicts(event_type, trigger, serving_cell_id, serving_rsrp)
            }
        };

        let now = Instant::now();
        let mut generate_report = false;
        let mut triggered_cells: Vec<TriggeringCell> = Vec::new();

        {
            let state = self.event_states.entry(meas_id).or_default();

            // A cell with no measurement this round is no longer applicable, and
            // an inapplicable cell cannot hold the event triggered.
            let applicable: BTreeSet<TriggeringCell> = verdicts.iter().map(|v| v.cell).collect();
            state.cells_triggered.retain(|c| applicable.contains(c));
            state.entering_since.retain(|c, _| applicable.contains(c));
            state.leaving_since.retain(|c, _| applicable.contains(c));

            // §5.5.4.1: include a cell once its entering condition has held for
            // timeToTrigger, exclude it once its leaving condition has.
            for verdict in &verdicts {
                let cell = verdict.cell;
                if state.cells_triggered.contains(&cell) {
                    if verdict.leaving {
                        let since = *state.leaving_since.entry(cell).or_insert(now);
                        if now.duration_since(since) >= ttt {
                            state.cells_triggered.remove(&cell);
                            state.leaving_since.remove(&cell);
                        }
                    } else {
                        // Back inside the dead band: the leaving run is broken.
                        state.leaving_since.remove(&cell);
                    }
                } else if verdict.entering {
                    let since = *state.entering_since.entry(cell).or_insert(now);
                    if now.duration_since(since) >= ttt {
                        state.cells_triggered.insert(cell);
                        state.entering_since.remove(&cell);
                    }
                } else {
                    state.entering_since.remove(&cell);
                }
            }

            if periodic || !state.cells_triggered.is_empty() {
                let should_report = state.last_report.is_none_or(|last| {
                    last.elapsed() >= Duration::from_millis(config.report_interval)
                });
                let reports_remaining =
                    config.report_amount == 0 || state.reports_sent < config.report_amount;

                if should_report && reports_remaining {
                    generate_report = true;
                    // §5.5.5 orders the reported cells by the quantity the event
                    // is written in, strongest first; ties keep the set's order
                    // so the list is deterministic.
                    let mut ranked: Vec<(TriggeringCell, i32)> = verdicts
                        .iter()
                        .filter(|v| state.cells_triggered.contains(&v.cell))
                        .map(|v| (v.cell, v.level))
                        .collect();
                    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                    triggered_cells = ranked.into_iter().map(|(cell, _)| cell).collect();

                    state.reports_sent += 1;
                    state.last_report = Some(now);

                    tracing::info!(
                        "Measurement report generated: meas_id={}, event={:?}, cells_triggered={}, reports_sent={}",
                        meas_id,
                        trigger.trigger_type,
                        state.cells_triggered.len(),
                        state.reports_sent
                    );
                }
            }
        }

        // Generate report outside the borrow scope
        if generate_report {
            let report = self.generate_report(
                meas_id,
                serving_cell_id,
                &triggered_cells,
                config.max_report_cells,
            );
            self.pending_reports.push(report);
        }
    }

    /// Where every cell applicable to `event_type` stands against its two
    /// inequalities.
    ///
    /// "Applicable" is per event: A1/A2 measure the serving cell and nothing
    /// else, A3-A5 the NR neighbours of the PCell, A6 the NR neighbours of the
    /// SCell (and neither serving cell is a neighbour of itself), B1/B2 the
    /// inter-RAT cells. An event whose reference level is missing has no
    /// applicable cell rather than a substituted one.
    fn cell_verdicts(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        serving_cell_id: i32,
        serving_rsrp: Option<i32>,
    ) -> Vec<CellVerdict> {
        let hyst = trigger.hysteresis / 2; // Convert to dB

        match event_type {
            // A1-1: Ms - Hys > Thresh   A1-2: Ms + Hys < Thresh
            // A2-1: Ms + Hys < Thresh   A2-2: Ms - Hys > Thresh
            MeasEventType::A1 | MeasEventType::A2 => {
                let (Some(ms), Some(thresh)) = (serving_rsrp, trigger.threshold) else {
                    return Vec::new();
                };
                let (entering, leaving) = if event_type == MeasEventType::A1 {
                    (ms > thresh + hyst, ms < thresh - hyst)
                } else {
                    (ms < thresh - hyst, ms > thresh + hyst)
                };
                vec![CellVerdict {
                    cell: TriggeringCell::Nr(serving_cell_id),
                    level: ms,
                    entering,
                    leaving,
                }]
            }
            // A3/A5 measure against Mp; A4 needs no reference at all.
            MeasEventType::A3 | MeasEventType::A5 => {
                self.nr_cell_verdicts(event_type, trigger, serving_rsrp, |cell_id| {
                    cell_id != serving_cell_id
                })
            }
            MeasEventType::A4 => self.nr_cell_verdicts(event_type, trigger, None, |cell_id| {
                cell_id != serving_cell_id
            }),
            MeasEventType::A6 => {
                // §5.5.4.7: the reference is the SCell, not the PCell. Without an
                // SCell there is nothing to compare against, so no A6 event.
                let Some(scell_id) = self.scell_id else {
                    return Vec::new();
                };
                let scell_rsrp = self.rsrp(scell_id);
                if scell_rsrp.is_none() {
                    return Vec::new();
                }
                // Neither serving cell is a neighbour of itself.
                self.nr_cell_verdicts(event_type, trigger, scell_rsrp, |cell_id| {
                    cell_id != serving_cell_id && cell_id != scell_id
                })
            }
            MeasEventType::B1 => self.eutra_cell_verdicts(event_type, trigger, None),
            MeasEventType::B2 => self.eutra_cell_verdicts(event_type, trigger, serving_rsrp),
        }
    }

    /// Verdicts for every measured NR cell `is_neighbour` accepts.
    ///
    /// `reference_rsrp` is what the event compares against: the PCell for
    /// A3/A5, the SCell for A6, and nothing at all for A4.
    fn nr_cell_verdicts(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        reference_rsrp: Option<i32>,
        is_neighbour: impl Fn(i32) -> bool,
    ) -> Vec<CellVerdict> {
        let mut verdicts = Vec::new();

        for (&cell_id, meas) in &self.measurements {
            if !is_neighbour(cell_id) {
                continue;
            }
            let Some(rsrp) = meas.rsrp else { continue };
            verdicts.push(CellVerdict {
                cell: TriggeringCell::Nr(cell_id),
                level: rsrp,
                entering: nr_condition(
                    Condition::Entering,
                    event_type,
                    trigger,
                    reference_rsrp,
                    rsrp,
                ),
                leaving: nr_condition(
                    Condition::Leaving,
                    event_type,
                    trigger,
                    reference_rsrp,
                    rsrp,
                ),
            });
        }

        verdicts
    }

    /// Verdicts for every measured inter-RAT cell against B1/B2.
    ///
    /// The level compared is Mn + Ofn + Ocn, the same sum the inequalities use,
    /// so the cell the report leads with is the one the event is about.
    ///
    /// B2's PCell half is not per cell: B2-1 (Mp + Hys < Thresh1) gates every
    /// neighbour's entering condition, and B2-3 (Mp - Hys > Thresh1) releases
    /// every triggered one on its own, since §5.5.4.9's leaving condition is
    /// B2-3 *or* B2-4.
    fn eutra_cell_verdicts(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        pcell_rsrp: Option<i32>,
    ) -> Vec<CellVerdict> {
        let hyst = trigger.hysteresis / 2;

        let (pcell_entering, pcell_leaving) = if event_type == MeasEventType::B2 {
            match (pcell_rsrp, trigger.threshold1) {
                (Some(mp), Some(thresh1)) => (mp < thresh1 - hyst, mp > thresh1 + hyst),
                // No Mp or no threshold: B2 can neither enter nor be sustained.
                _ => (false, true),
            }
        } else {
            (true, false)
        };

        // B1 reads b1-ThresholdEUTRA from `threshold`, B2 b2-Threshold2EUTRA
        // from `threshold2`.
        let threshold = match event_type {
            MeasEventType::B1 => trigger.threshold,
            _ => trigger.threshold2,
        };

        let mut verdicts = Vec::new();
        for (&cell, meas) in &self.eutra_measurements {
            let Some(rsrp) = meas.rsrp else { continue };
            let ofn = self
                .eutra_freq_offsets
                .get(&cell.earfcn)
                .copied()
                .unwrap_or(0);
            let level = rsrp + ofn + meas.cell_individual_offset;
            // B1-1/B2-2: Mn + Ofn + Ocn - Hys > Thresh
            // B1-2/B2-4: Mn + Ofn + Ocn + Hys < Thresh
            let (neighbour_entering, neighbour_leaving) = match threshold {
                Some(thresh) => (level - hyst > thresh, level + hyst < thresh),
                None => (false, true),
            };
            verdicts.push(CellVerdict {
                cell: TriggeringCell::Eutra(cell),
                level,
                entering: pcell_entering && neighbour_entering,
                leaving: pcell_leaving || neighbour_leaving,
            });
        }

        verdicts
    }

    /// The cells currently holding `meas_id` triggered (`cellsTriggeredList`).
    ///
    /// Key order, not signal order — the report path ranks by level.
    pub fn triggered_cells(&self, meas_id: u8) -> Vec<TriggeringCell> {
        self.event_states
            .get(&meas_id)
            .map(|state| state.cells_triggered.iter().copied().collect())
            .unwrap_or_default()
    }

    /// Whether an execution condition holds for one specific candidate cell.
    ///
    /// Conditional reconfiguration evaluates a candidate's condition "for the
    /// applicable cell" (TS 38.331 §5.3.5.13.4) — the cell the candidate targets,
    /// not the best neighbour — so the CHO runtime asks per candidate rather than
    /// reusing the best-neighbour search. The inequality itself is the same one
    /// the measurement events use.
    ///
    /// `false` when the candidate is the serving cell, when either measurement is
    /// missing, or when the event is not one a candidate cell can satisfy
    /// (A1/A2 measure only the serving cell; B1/B2 are inter-RAT).
    pub fn candidate_condition_holds(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        candidate_cell_id: i32,
    ) -> bool {
        let Some(serving_cell_id) = self.serving_cell_id else {
            return false;
        };
        if candidate_cell_id == serving_cell_id {
            return false;
        }
        if matches!(
            event_type,
            MeasEventType::A1 | MeasEventType::A2 | MeasEventType::B1 | MeasEventType::B2
        ) {
            return false;
        }

        let reference_rsrp = match event_type {
            MeasEventType::A6 => match self.scell_id.and_then(|id| self.rsrp(id)) {
                Some(rsrp) => rsrp,
                None => return false,
            },
            _ => match self.rsrp(serving_cell_id) {
                Some(rsrp) => rsrp,
                None => return false,
            },
        };
        let Some(candidate_rsrp) = self.rsrp(candidate_cell_id) else {
            return false;
        };

        nr_condition(
            Condition::Entering,
            event_type,
            trigger,
            Some(reference_rsrp),
            candidate_rsrp,
        )
    }

    /// Build the report for `meas_id`, leading both neighbour lists with the
    /// event's triggered cells in the order given (strongest first).
    fn generate_report(
        &self,
        meas_id: u8,
        serving_cell_id: i32,
        triggered_cells: &[TriggeringCell],
        max_cells: u8,
    ) -> MeasurementReport {
        // The serving cell need not have been measured: A4, A6 and B1 are all
        // written without Mp, so a report can be due before the PCell has a
        // level. Report it as unmeasured rather than panicking.
        let serving_cell = self
            .measurements
            .get(&serving_cell_id)
            .cloned()
            .unwrap_or_else(|| CellMeasResult {
                pci: serving_cell_id as u32,
                ..Default::default()
            });

        // Get neighbor cell measurements, sorted by RSRP
        let mut neighbors: Vec<_> = self
            .measurements
            .iter()
            .filter(|(&id, _)| id != serving_cell_id)
            .map(|(_, m)| m.clone())
            .collect();

        neighbors.sort_by(|a, b| b.rsrp.unwrap_or(i32::MIN).cmp(&a.rsrp.unwrap_or(i32::MIN)));

        // Every triggered NR cell leads the list, in the given order — not just
        // one of them: cellsTriggeredList is a set (§5.5.4.1).
        let mut hoisted = Vec::with_capacity(neighbors.len());
        for pci in triggered_cells
            .iter()
            .filter_map(TriggeringCell::nr_cell_id)
            .map(|id| id as u32)
        {
            if let Some(pos) = neighbors.iter().position(|m| m.pci == pci) {
                hoisted.push(neighbors.remove(pos));
            }
        }
        hoisted.append(&mut neighbors);
        let mut neighbors = hoisted;

        // Limit to max cells
        neighbors.truncate(max_cells as usize);

        // Inter-RAT results, triggered cells first and then the rest by level.
        //
        // These do not reach the network. Two things stop them, and the second is
        // the hard one: the UE's uplink report is a hand-rolled byte format with
        // no inter-RAT section (issue #107), AND `measResultListEUTRA` is an
        // extension arm of the `measResultNeighCells` CHOICE, which asn1-codecs
        // 0.7 refuses to encode at all ("Encode of extended choice not yet
        // implemented") -- see
        // `nextgsim-rrc` `the_eutra_choice_arm_cannot_be_uper_encoded_by_this_codec`.
        // So a B1/B2 report is UE-observable only, and #107 alone will not change
        // that.
        let mut eutra_neighbors: Vec<_> = self.eutra_measurements.values().cloned().collect();
        eutra_neighbors.sort_by(|a, b| b.rsrp.unwrap_or(i32::MIN).cmp(&a.rsrp.unwrap_or(i32::MIN)));
        let mut eutra_hoisted = Vec::with_capacity(eutra_neighbors.len());
        for trig in triggered_cells.iter().filter_map(|cell| match cell {
            TriggeringCell::Eutra(key) => Some(*key),
            TriggeringCell::Nr(_) => None,
        }) {
            if let Some(pos) = eutra_neighbors.iter().position(|m| m.cell == trig) {
                eutra_hoisted.push(eutra_neighbors.remove(pos));
            }
        }
        eutra_hoisted.append(&mut eutra_neighbors);
        let mut eutra_neighbors = eutra_hoisted;
        eutra_neighbors.truncate(max_cells as usize);

        MeasurementReport {
            meas_id,
            serving_cell,
            neighbor_cells: neighbors,
            eutra_neighbor_cells: eutra_neighbors,
            triggered_cells: triggered_cells.to_vec(),
            timestamp: Instant::now(),
        }
    }

    /// Get the number of active measurement configurations
    pub fn config_count(&self) -> usize {
        self.configs.len()
    }

    /// Get all cell measurements
    pub fn measurements(&self) -> &HashMap<i32, CellMeasResult> {
        &self.measurements
    }
}

impl Default for MeasurementManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether an event is triggered: TS 38.331 §5.5.4.1 defines that as its
    /// `cellsTriggeredList` being non-empty.
    fn is_triggered(manager: &MeasurementManager, meas_id: u8) -> bool {
        !manager.triggered_cells(meas_id).is_empty()
    }

    #[test]
    fn test_measurement_manager_creation() {
        let manager = MeasurementManager::new();
        assert!(manager.serving_cell_id.is_none());
        assert_eq!(manager.config_count(), 0);
    }

    #[test]
    fn test_add_measurement_config() {
        let mut manager = MeasurementManager::new();
        let config = MeasConfig::default();
        manager.add_config(config);
        assert_eq!(manager.config_count(), 1);
    }

    #[test]
    fn test_update_measurement() {
        let mut manager = MeasurementManager::new();
        manager.update_measurement(1, -80);
        assert_eq!(manager.measurements().len(), 1);
        assert_eq!(manager.measurements()[&1].rsrp, Some(-80));
    }

    #[test]
    fn test_a3_event_detection() {
        let mut manager = MeasurementManager::new();

        // Set up serving cell
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -90);

        // Set up neighbor cell that is better
        manager.update_measurement(2, -80);

        // Add A3 configuration
        let mut config = MeasConfig::default();
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A3);
        config.trigger_config.a3_offset = Some(3);
        config.trigger_config.time_to_trigger = 0; // Immediate for test
        manager.add_config(config);

        // Evaluate - should detect A3 event
        manager.evaluate_events();

        // Check event state
        assert_eq!(manager.triggered_cells(1), vec![TriggeringCell::Nr(2)]);
    }

    /// Builds a manager camped on cell 1 with an SCell on cell 2, so A6 has a
    /// reference to measure against.
    fn manager_with_scell() -> MeasurementManager {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.set_scell(Some(2));
        manager.update_measurement(1, -70); // PCell, deliberately the strongest
        manager.update_measurement(2, -95); // SCell: the A6 reference
        manager
    }

    fn a6_config(offset_db: i32) -> MeasConfig {
        let mut config = MeasConfig {
            meas_id: 6,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A6);
        config.trigger_config.a3_offset = None;
        config.trigger_config.a6_offset = Some(offset_db);
        config.trigger_config.hysteresis = 2; // 1 dB
        config.trigger_config.time_to_trigger = 0;
        config
    }

    /// A6-1: Mn - Hys > Ms + Off, with Ms the SCell (TS 38.331 §5.5.4.7).
    /// SCell -95, offset 3, hysteresis 1 dB: a neighbour must beat -91 dBm.
    #[test]
    fn an_a6_event_enters_when_a_neighbour_beats_the_scell_by_the_offset() {
        let mut manager = manager_with_scell();
        manager.update_measurement(3, -90); // > -95 + 3 + 1
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        assert_eq!(
            manager.triggered_cells(6),
            vec![TriggeringCell::Nr(3)],
            "-90 dBm beats the -91 dBm A6 bar"
        );
    }

    #[test]
    fn an_a6_event_does_not_enter_below_the_offset() {
        let mut manager = manager_with_scell();
        manager.update_measurement(3, -92); // one dB short of the bar
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        assert!(
            !is_triggered(&manager, 6),
            "-92 dBm is below the -91 dBm A6 bar"
        );
    }

    /// A6 measures against the SCell, so the PCell — the strongest cell here —
    /// must not be picked up as its own neighbour, and neither must the SCell.
    #[test]
    fn an_a6_event_treats_neither_serving_cell_as_a_neighbour() {
        let mut manager = manager_with_scell();
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        assert!(
            !is_triggered(&manager, 6),
            "the PCell at -70 dBm would satisfy the inequality, but it is not a neighbour"
        );
    }

    /// Without an SCell the event has no reference: §5.5.4.7 defines A6 against
    /// the secondary cell, so it cannot trigger off the PCell instead.
    #[test]
    fn an_a6_event_cannot_trigger_without_an_scell() {
        let mut manager = manager_with_scell();
        manager.set_scell(None);
        manager.update_measurement(3, -50); // stronger than everything
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        assert!(!is_triggered(&manager, 6));
    }

    fn inter_rat_config(event: MeasEventType) -> MeasConfig {
        let mut config = MeasConfig {
            meas_id: 7,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(event);
        config.trigger_config.a3_offset = None;
        config.trigger_config.hysteresis = 2; // 1 dB
        config.trigger_config.time_to_trigger = 0;
        config
    }

    /// B1-1: Mn + Ofn + Ocn - Hys > Thresh (TS 38.331 §5.5.4.8).
    #[test]
    fn a_b1_event_enters_on_an_inter_rat_neighbour_above_the_threshold() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);

        let eutra = EutraCellKey::new(1850, 42);
        manager.update_eutra_measurement(eutra, -95);

        let mut config = inter_rat_config(MeasEventType::B1);
        config.trigger_config.threshold = Some(-100); // -95 - 1 > -100
        manager.add_config(config);

        manager.evaluate_events();

        assert_eq!(
            manager.triggered_cells(7),
            vec![TriggeringCell::Eutra(eutra)]
        );
    }

    /// The offsets are part of the inequality: a neighbour one dB short of the
    /// threshold enters once Ofn or Ocn lifts it over.
    #[test]
    fn a_b1_event_applies_the_frequency_and_cell_offsets() {
        let eutra = EutraCellKey::new(1850, 42);

        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_eutra_measurement(eutra, -101);
        let mut config = inter_rat_config(MeasEventType::B1);
        config.trigger_config.threshold = Some(-100);
        manager.add_config(config.clone());

        manager.evaluate_events();
        assert!(
            !is_triggered(&manager, 7),
            "-101 dBm with no offsets is below the -99 dBm bar"
        );

        manager.set_eutra_frequency_offset(1850, 2); // Ofn
        manager.set_eutra_cell_offset(eutra, 1); // Ocn
        manager.evaluate_events();
        assert!(
            is_triggered(&manager, 7),
            "Ofn + Ocn = 3 dB lifts it to -98 dBm, over the bar"
        );
    }

    /// B2 needs both halves: a weak PCell (B2-1) and a strong inter-RAT
    /// neighbour (B2-2). Either alone must not trigger it (§5.5.4.9).
    #[test]
    fn a_b2_event_needs_both_a_weak_pcell_and_a_strong_inter_rat_neighbour() {
        let eutra = EutraCellKey::new(1850, 7);
        let mut config = inter_rat_config(MeasEventType::B2);
        config.trigger_config.threshold1 = Some(-100); // PCell must be < -101
        config.trigger_config.threshold2 = Some(-110); // neighbour must be > -109

        // Strong neighbour, but the PCell is healthy: B2-1 fails.
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_eutra_measurement(eutra, -105);
        manager.add_config(config.clone());
        manager.evaluate_events();
        assert!(!is_triggered(&manager, 7), "PCell is not weak");

        // Weak PCell, but no neighbour worth going to: B2-2 fails.
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -115);
        manager.update_eutra_measurement(eutra, -120);
        manager.add_config(config.clone());
        manager.evaluate_events();
        assert!(
            !is_triggered(&manager, 7),
            "the inter-RAT neighbour is below threshold2"
        );

        // Both halves hold.
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -115);
        manager.update_eutra_measurement(eutra, -105);
        manager.add_config(config);
        manager.evaluate_events();
        assert!(is_triggered(&manager, 7));
        assert_eq!(
            manager.triggered_cells(7).first().copied(),
            Some(TriggeringCell::Eutra(eutra))
        );
    }

    /// §5.5.4.9's leaving condition is B2-3 *or* B2-4, so the PCell recovering
    /// releases the event even while the inter-RAT neighbour is still strong.
    #[test]
    fn a_b2_event_leaves_when_the_pcell_recovers_alone() {
        let eutra = EutraCellKey::new(1850, 7);
        let mut config = inter_rat_config(MeasEventType::B2);
        config.trigger_config.threshold1 = Some(-100); // PCell enters below -101
        config.trigger_config.threshold2 = Some(-110); // neighbour enters above -109

        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -115);
        manager.update_eutra_measurement(eutra, -105);
        manager.add_config(config);

        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(7),
            vec![TriggeringCell::Eutra(eutra)]
        );

        // B2-3: Mp - Hys > Thresh1. The neighbour has not moved.
        manager.update_measurement(1, -98);
        manager.evaluate_events();
        assert!(
            manager.triggered_cells(7).is_empty(),
            "the PCell recovering satisfies B2-3 on its own"
        );
    }

    /// The dead band applies to B2's PCell half too: a PCell between the
    /// entering and leaving bars keeps the event triggered.
    #[test]
    fn a_b2_event_holds_while_the_pcell_is_inside_its_dead_band() {
        let eutra = EutraCellKey::new(1850, 7);
        let mut config = inter_rat_config(MeasEventType::B2);
        config.trigger_config.hysteresis = 4; // Hys = 2 dB
        config.trigger_config.threshold1 = Some(-100); // enters below -102, leaves above -98
        config.trigger_config.threshold2 = Some(-110);

        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -105);
        manager.update_eutra_measurement(eutra, -105);
        manager.add_config(config);

        manager.evaluate_events();
        assert!(is_triggered(&manager, 7));

        manager.update_measurement(1, -100); // inside the band: neither B2-1 nor B2-3
        manager.evaluate_events();
        assert!(
            is_triggered(&manager, 7),
            "-100 dBm satisfies neither the entering nor the leaving half"
        );
    }

    /// A B1 report carries the inter-RAT neighbour it triggered on. The uplink
    /// PDU cannot carry it (the CHOICE extension arm has no encoder); the report
    /// the manager produces can.
    #[test]
    fn a_b1_report_carries_the_inter_rat_neighbour() {
        let eutra = EutraCellKey::new(1850, 42);
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_eutra_measurement(eutra, -95);
        manager.update_eutra_measurement(EutraCellKey::new(1850, 43), -97);

        let mut config = inter_rat_config(MeasEventType::B1);
        config.trigger_config.threshold = Some(-100);
        manager.add_config(config);

        manager.evaluate_events();

        let reports = manager.take_pending_reports();
        assert_eq!(reports.len(), 1, "one B1 report");
        let cells: Vec<_> = reports[0]
            .eutra_neighbor_cells
            .iter()
            .map(|m| m.cell)
            .collect();
        assert_eq!(
            cells[0], eutra,
            "the triggering cell comes first, then the rest by level"
        );
        assert_eq!(cells.len(), 2);
    }

    /// An A4 measId, whose bar is a plain threshold, so a neighbour's level is
    /// the only thing that moves it in or out.
    fn a4_config(threshold_dbm: i32, hysteresis: i32) -> MeasConfig {
        let mut config = MeasConfig {
            meas_id: 4,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A4);
        config.trigger_config.a3_offset = None;
        config.trigger_config.threshold = Some(threshold_dbm);
        config.trigger_config.hysteresis = hysteresis;
        config.trigger_config.time_to_trigger = 0;
        config.report_amount = 0; // report on every interval, not a fixed count
        config.report_interval = 0;
        config
    }

    /// The dead band, which is the whole point of a separate leaving inequality
    /// (TS 38.331 §5.5.4.5): A4-1 is Mn - Hys > Thresh and A4-2 is
    /// Mn + Hys < Thresh, so with Thresh -100 and Hys 2 dB a cell enters above
    /// -98 dBm and does not leave until it falls below -102 dBm. Between those
    /// it stays triggered, where an entering-only manager would toggle.
    #[test]
    fn a_triggered_cell_stays_triggered_inside_the_dead_band() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.add_config(a4_config(-100, 4)); // Hys = 2 dB

        manager.update_measurement(2, -97); // -97 - 2 > -100: enters
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "-97 dBm is over the -98 dBm entering bar"
        );

        manager.update_measurement(2, -101); // below entering, above leaving
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "-101 dBm is inside the dead band, so the cell stays in \
             cellsTriggeredList"
        );

        manager.update_measurement(2, -103); // -103 + 2 < -100: leaves
        manager.evaluate_events();
        assert!(
            manager.triggered_cells(4).is_empty(),
            "-103 dBm satisfies A4-2, so the cell leaves"
        );
    }

    /// A cell that never crossed the entering bar must not be triggered just
    /// because it is inside the dead band: the band only holds cells that got in.
    #[test]
    fn a_cell_inside_the_dead_band_that_never_entered_stays_untriggered() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.add_config(a4_config(-100, 4));

        manager.update_measurement(2, -101);
        manager.evaluate_events();

        assert!(
            manager.triggered_cells(4).is_empty(),
            "-101 dBm satisfies neither A4-1 nor A4-2"
        );
    }

    /// §5.5.4.1 keeps a *set* per measId: two neighbours over the bar both
    /// belong to it, and one leaving does not release the event.
    #[test]
    fn two_neighbours_over_the_bar_are_both_triggered_and_one_can_drop_out() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.add_config(a4_config(-100, 4));

        manager.update_measurement(2, -95);
        manager.update_measurement(3, -90);
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2), TriggeringCell::Nr(3)],
            "both neighbours are over the -98 dBm bar"
        );

        // The stronger one collapses past the leaving bar; the weaker one holds.
        manager.update_measurement(3, -110);
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "cell 3 left; cell 2 keeps the event triggered"
        );
    }

    /// The report leads with every triggered cell, strongest first, rather than
    /// with one of them (§5.5.5).
    ///
    /// The dead band is what makes triggered order differ from level order here:
    /// cells 2 and 3 both entered and then sank into the band, while cell 4 —
    /// stronger than either of them — never crossed the entering bar. A report
    /// sorted purely by level would lead with cell 4, which the event did not
    /// fire on.
    #[test]
    fn a_report_leads_with_all_triggered_cells_strongest_first() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_measurement(2, -95);
        manager.update_measurement(3, -95);
        manager.add_config(a4_config(-100, 4)); // enters above -98, leaves below -102

        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2), TriggeringCell::Nr(3)]
        );
        manager.take_pending_reports();

        // Both sink into the dead band, cell 3 the stronger of the two; cell 4
        // appears between them without ever entering.
        manager.update_measurement(2, -101);
        manager.update_measurement(3, -100);
        manager.update_measurement(4, -99);
        manager.evaluate_events();

        let reports = manager.take_pending_reports();
        assert_eq!(reports.len(), 1, "one A4 report");
        assert_eq!(
            reports[0].triggered_cells,
            vec![TriggeringCell::Nr(3), TriggeringCell::Nr(2)],
            "strongest triggered cell first"
        );
        let pcis: Vec<u32> = reports[0].neighbor_cells.iter().map(|m| m.pci).collect();
        assert_eq!(
            pcis,
            vec![3, 2, 4],
            "both triggered cells lead the neighbour list, ahead of the stronger \
             cell 4 that the event never fired on"
        );
    }

    /// A cell that stops being measured cannot keep an event triggered: it is no
    /// longer an applicable cell (§5.5.4.1).
    #[test]
    fn an_unmeasured_cell_leaves_the_triggered_list() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_measurement(2, -90);
        manager.add_config(a4_config(-100, 4));

        manager.evaluate_events();
        assert_eq!(manager.triggered_cells(4), vec![TriggeringCell::Nr(2)]);

        manager.remove_measurement(2);
        manager.evaluate_events();
        assert!(manager.triggered_cells(4).is_empty());
    }

    /// `timeToTrigger` gates both directions (§5.5.4.1): a cell over the
    /// entering bar for less than it is not yet in the list, and a triggered
    /// cell past the leaving bar for less than it is not yet out.
    #[test]
    fn time_to_trigger_gates_entering_and_leaving() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        let mut config = a4_config(-100, 4);
        config.trigger_config.time_to_trigger = 50;
        manager.add_config(config);

        manager.update_measurement(2, -90);
        manager.evaluate_events();
        assert!(
            manager.triggered_cells(4).is_empty(),
            "the entering condition has not held for timeToTrigger yet"
        );

        std::thread::sleep(Duration::from_millis(60));
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "50 ms of an unbroken entering run puts the cell in"
        );

        manager.update_measurement(2, -110);
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "the leaving condition has not held for timeToTrigger yet"
        );

        std::thread::sleep(Duration::from_millis(60));
        manager.evaluate_events();
        assert!(
            manager.triggered_cells(4).is_empty(),
            "50 ms of an unbroken leaving run takes it out"
        );
    }

    /// The UE re-declares its serving cell on every measurement cycle, so only a
    /// *change* may reset the triggered state. Resetting unconditionally, as this
    /// did, restarted `timeToTrigger` on every tick and no event could ever
    /// accumulate one in a live run.
    #[test]
    fn redeclaring_the_same_serving_cell_does_not_restart_time_to_trigger() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        let mut config = a4_config(-100, 4);
        config.trigger_config.time_to_trigger = 50;
        manager.add_config(config);

        manager.update_measurement(2, -90);
        manager.evaluate_events(); // starts the entering run

        std::thread::sleep(Duration::from_millis(60));
        manager.set_serving_cell(Some(1)); // same cell, as the task does each cycle
        manager.evaluate_events();

        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "the entering run survives a no-op serving-cell declaration"
        );
    }

    /// A real serving-cell change does invalidate every list: the inequalities
    /// are written against the cell that just went away.
    #[test]
    fn a_serving_cell_change_clears_the_triggered_lists() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        manager.update_measurement(2, -90);
        manager.add_config(a4_config(-100, 4));

        manager.evaluate_events();
        assert_eq!(manager.triggered_cells(4), vec![TriggeringCell::Nr(2)]);

        manager.set_serving_cell(Some(2));
        assert!(manager.triggered_cells(4).is_empty());
    }

    /// §5.5.4.1 requires the leaving condition to hold for *all* measurements
    /// taken during `timeToTrigger`, so a level that recrosses the entering bar
    /// discards the run rather than pausing it. The second run below must be
    /// timed from scratch: if the first run's start survived, the cell would drop
    /// on the very next evaluation, 60 ms of wall clock after it began.
    #[test]
    fn recrossing_the_entering_bar_restarts_the_leaving_run() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -80);
        let mut config = a4_config(-100, 4);
        config.trigger_config.time_to_trigger = 50;
        manager.add_config(config);

        manager.update_measurement(2, -90);
        manager.evaluate_events();
        std::thread::sleep(Duration::from_millis(60));
        manager.evaluate_events();
        assert_eq!(manager.triggered_cells(4), vec![TriggeringCell::Nr(2)]);

        manager.update_measurement(2, -110); // starts a leaving run
        manager.evaluate_events();
        std::thread::sleep(Duration::from_millis(60));
        manager.update_measurement(2, -90); // breaks it, before it can expire
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "the cell is over the entering bar again, so it did not leave"
        );

        manager.update_measurement(2, -110); // a fresh leaving run
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(4),
            vec![TriggeringCell::Nr(2)],
            "the new run is timed from now, not from the abandoned one"
        );

        std::thread::sleep(Duration::from_millis(60));
        manager.evaluate_events();
        assert!(
            manager.triggered_cells(4).is_empty(),
            "and it does expire on its own 50 ms"
        );
    }

    /// The PCell's own two events use the same dead band. A2-1 is
    /// Ms + Hys < Thresh and A2-2 is Ms - Hys > Thresh, so with Thresh -90 and
    /// Hys 1 dB the serving cell enters below -91 and leaves above -89.
    #[test]
    fn an_a2_event_has_a_dead_band_on_the_serving_cell() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        let mut config = MeasConfig {
            meas_id: 2,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A2);
        config.trigger_config.threshold = Some(-90);
        config.trigger_config.time_to_trigger = 0;
        manager.add_config(config);

        manager.update_measurement(1, -95);
        manager.evaluate_events();
        assert_eq!(
            manager.triggered_cells(2),
            vec![TriggeringCell::Nr(1)],
            "A2's applicable cell is the serving cell itself"
        );

        manager.update_measurement(1, -90); // inside the band
        manager.evaluate_events();
        assert!(is_triggered(&manager, 2), "-90 dBm is inside the dead band");

        manager.update_measurement(1, -88); // -88 - 1 > -90: A2-2
        manager.evaluate_events();
        assert!(!is_triggered(&manager, 2));
    }

    /// An event written against Mp has no applicable cell before the PCell has
    /// been measured. It used to be evaluated against `i32::MIN`, which A2
    /// treated as "worse than any threshold" and which a leaving inequality
    /// subtracting the hysteresis from would overflow.
    #[test]
    fn an_unmeasured_pcell_triggers_no_serving_cell_event() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        let mut config = MeasConfig {
            meas_id: 2,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A2);
        config.trigger_config.threshold = Some(-90);
        config.trigger_config.time_to_trigger = 0;
        manager.add_config(config);

        manager.evaluate_events();

        assert!(!is_triggered(&manager, 2));
        assert!(manager.take_pending_reports().is_empty());
    }

    /// An A5 config with no thresholds must simply not trigger. It used to
    /// substitute `i32::MIN` for the missing threshold1 and then subtract the
    /// hysteresis from it, which overflows.
    #[test]
    fn an_a5_event_without_thresholds_does_not_trigger() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -90);
        manager.update_measurement(2, -60);

        let mut config = MeasConfig {
            meas_id: 5,
            ..Default::default()
        };
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A5);
        config.trigger_config.threshold1 = None;
        config.trigger_config.threshold2 = None;
        config.trigger_config.time_to_trigger = 0;
        manager.add_config(config);

        manager.evaluate_events();

        assert!(!is_triggered(&manager, 5));
    }

    /// Conditional reconfiguration judges a candidate by its own cell
    /// (§5.3.5.13.4), so a strong *other* neighbour must not make a weak
    /// candidate's condition hold.
    #[test]
    fn a_candidate_condition_is_evaluated_for_that_candidate_cell_only() {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -90);
        manager.update_measurement(2, -95); // candidate: weaker than serving
        manager.update_measurement(3, -60); // a different, very strong neighbour

        let trigger = ReportTriggerConfig {
            trigger_type: ReportTriggerType::Event(MeasEventType::A3),
            a3_offset: Some(3),
            hysteresis: 2,
            time_to_trigger: 0,
            ..Default::default()
        };

        assert!(
            !manager.candidate_condition_holds(MeasEventType::A3, &trigger, 2),
            "cell 2 does not beat the serving cell, whatever cell 3 does"
        );
        assert!(
            manager.candidate_condition_holds(MeasEventType::A3, &trigger, 3),
            "cell 3 does"
        );
        assert!(
            !manager.candidate_condition_holds(MeasEventType::A3, &trigger, 1),
            "the serving cell is not a candidate for its own handover"
        );
        assert!(
            !manager.candidate_condition_holds(MeasEventType::A3, &trigger, 9),
            "an unmeasured cell cannot satisfy a condition"
        );
    }

    #[test]
    fn test_a2_event_detection() {
        let mut manager = MeasurementManager::new();

        // Set up serving cell with weak signal
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -100);

        // Add A2 configuration (serving worse than threshold)
        let mut config = MeasConfig::default();
        config.meas_id = 2;
        config.trigger_config.trigger_type = ReportTriggerType::Event(MeasEventType::A2);
        config.trigger_config.threshold = Some(-90);
        config.trigger_config.time_to_trigger = 0;
        manager.add_config(config);

        // Evaluate - should detect A2 event
        manager.evaluate_events();

        assert!(is_triggered(&manager, 2));
    }
}
