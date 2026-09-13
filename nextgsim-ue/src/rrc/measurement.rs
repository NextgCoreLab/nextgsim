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
//! # What this manager does not model
//!
//! - **Leaving conditions.** TS 38.331 §5.5.4 gives each event an entering *and*
//!   a leaving inequality, which differ by the sign of the hysteresis and so
//!   create a dead band. Here an event is triggered exactly while its entering
//!   condition holds, and `cellsTriggeredList` is one cell rather than a set.
//!   Issue #111 covers the dead band and the per-cell triggered list.
//! - **Layer-3 filtering** (§5.5.3.2): a measurement is the level the radio last
//!   reported, unfiltered.
//! - **`cellIndividualOffset` for NR cells** (Ocn/Ocs): zero for every NR cell,
//!   as §5.5.4 permits when it is not configured. E-UTRA neighbours do carry
//!   both offsets, because B1/B2's inequalities are written around them.
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification, §5.5.4
//! - 3GPP TS 38.215: NR; Physical layer measurements

use std::collections::HashMap;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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

/// The cell that satisfied an event's entering condition.
///
/// Neighbour events name a cell, and an inter-RAT event names one that has no
/// simulator cell id — hence the two arms rather than a bare `i32`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Timestamp of the report
    pub timestamp: Instant,
}

/// The entering inequality of TS 38.331 §5.5.4 for one NR neighbour cell.
///
/// The single place the NR neighbour events' arithmetic lives: the per-event
/// best-neighbour search and the conditional-reconfiguration runtime
/// ([`MeasurementManager::candidate_condition_holds`]) both go through it, so a
/// CHO candidate is judged by exactly the inequality its measurement event uses.
///
/// `reference_rsrp` is the level the event compares against — Ms of the PCell for
/// A3, Ms of the SCell for A6 (§5.5.4.7), Mp of the PCell for A5 — and is ignored
/// by the threshold-only A4.
///
/// Hysteresis is halved because `ReportTriggerConfig::hysteresis` is in the
/// signalled 0.5 dB units. Ocn and Ocs are zero: no NR
/// `cellIndividualOffset` is modelled, which §5.5.4 allows when it is not
/// configured. Events that no neighbour cell can satisfy (A1, A2, and the
/// inter-RAT B1/B2) return `false` rather than being silently accepted.
///
/// A threshold event whose threshold is absent does not trigger, which is also
/// what an absent A5 threshold now does: the previous code substituted
/// `i32::MIN`/`i32::MAX` and then subtracted the hysteresis from it, overflowing.
fn nr_entering_condition(
    event_type: MeasEventType,
    trigger: &ReportTriggerConfig,
    reference_rsrp: i32,
    neighbour_rsrp: i32,
) -> bool {
    let hyst = trigger.hysteresis / 2;

    match event_type {
        // A3-1: Mn + Ocn - Hys > Mp + Ofp + Ocp + Off
        MeasEventType::A3 => {
            neighbour_rsrp > reference_rsrp + trigger.a3_offset.unwrap_or(0) + hyst
        }
        // A4-1: Mn + Ocn - Hys > Thresh
        MeasEventType::A4 => trigger
            .threshold
            .is_some_and(|thresh| neighbour_rsrp > thresh + hyst),
        // A5-1 and A5-2: Mp + Hys < Thresh1 AND Mn + Ocn - Hys > Thresh2
        MeasEventType::A5 => match (trigger.threshold1, trigger.threshold2) {
            (Some(thresh1), Some(thresh2)) => {
                reference_rsrp < thresh1 - hyst && neighbour_rsrp > thresh2 + hyst
            }
            _ => false,
        },
        // A6-1: Mn + Ocn - Hys > Ms + Ocs + Off, with Ms the SCell
        MeasEventType::A6 => {
            neighbour_rsrp > reference_rsrp + trigger.a6_offset.unwrap_or(0) + hyst
        }
        MeasEventType::A1 | MeasEventType::A2 | MeasEventType::B1 | MeasEventType::B2 => false,
    }
}

/// Event state tracking for a single event
#[derive(Debug, Clone, Default)]
struct EventState {
    /// Whether the event condition is currently met
    condition_met: bool,
    /// Cell that triggered the condition (for neighbor events)
    triggering_cell: Option<TriggeringCell>,
    /// Time when condition was first met
    condition_met_since: Option<Instant>,
    /// Number of reports sent for this event
    reports_sent: u32,
    /// Last report time
    last_report: Option<Instant>,
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
        self.serving_cell_id = cell_id;
        // Reset event states on serving cell change
        for state in self.event_states.values_mut() {
            state.condition_met = false;
            state.condition_met_since = None;
            state.triggering_cell = None;
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
    /// The simulator has no E-UTRA radio: RLS carries NR cells only, so nothing
    /// feeds this in a live run (issue #113). The trigger logic is here so that
    /// an inter-RAT measurement source, once there is one, needs none.
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

        let serving_rsrp = self
            .measurements
            .get(&serving_cell_id)
            .and_then(|m| m.rsrp)
            .unwrap_or(i32::MIN);

        // Evaluate each measurement configuration
        let configs: Vec<_> = self.configs.values().cloned().collect();
        for config in configs {
            self.evaluate_event(&config, serving_cell_id, serving_rsrp);
        }
    }

    fn evaluate_event(&mut self, config: &MeasConfig, serving_cell_id: i32, serving_rsrp: i32) {
        let meas_id = config.meas_id;
        let trigger = &config.trigger_config;

        let (condition_met, triggering_cell) = match trigger.trigger_type {
            ReportTriggerType::Event(event_type) => {
                self.check_event_condition(event_type, trigger, serving_cell_id, serving_rsrp)
            }
            ReportTriggerType::Periodic => (true, None),
        };

        // Check if we need to generate a report
        let mut generate_report = false;
        let mut report_params: Option<(u8, i32, Option<TriggeringCell>, u8)> = None;

        {
            let state = self.event_states.entry(meas_id).or_default();

            if condition_met {
                if !state.condition_met {
                    // Condition just became true
                    state.condition_met = true;
                    state.triggering_cell = triggering_cell;
                    state.condition_met_since = Some(Instant::now());
                }

                // Check time-to-trigger
                if let Some(since) = state.condition_met_since {
                    if since.elapsed() >= Duration::from_millis(trigger.time_to_trigger) {
                        // Check if we should send a report
                        let should_report = state.last_report.is_none_or(|last| {
                            last.elapsed() >= Duration::from_millis(config.report_interval)
                        });

                        let reports_remaining =
                            config.report_amount == 0 || state.reports_sent < config.report_amount;

                        if should_report && reports_remaining {
                            generate_report = true;
                            report_params = Some((
                                meas_id,
                                serving_cell_id,
                                triggering_cell,
                                config.max_report_cells,
                            ));
                            state.reports_sent += 1;
                            state.last_report = Some(Instant::now());

                            tracing::info!(
                                "Measurement report generated: meas_id={}, event={:?}, reports_sent={}",
                                meas_id, trigger.trigger_type, state.reports_sent
                            );
                        }
                    }
                }
            } else {
                // Condition no longer met - reset
                state.condition_met = false;
                state.condition_met_since = None;
                state.triggering_cell = None;
            }
        }

        // Generate report outside the borrow scope
        if generate_report {
            if let Some((meas_id, serving_cell_id, triggering_cell, max_cells)) = report_params {
                let report =
                    self.generate_report(meas_id, serving_cell_id, triggering_cell, max_cells);
                self.pending_reports.push(report);
            }
        }
    }

    fn check_event_condition(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        serving_cell_id: i32,
        serving_rsrp: i32,
    ) -> (bool, Option<TriggeringCell>) {
        let hyst = trigger.hysteresis / 2; // Convert to dB

        match event_type {
            MeasEventType::A1 => {
                // Serving > threshold
                if let Some(thresh) = trigger.threshold {
                    (serving_rsrp > thresh + hyst, None)
                } else {
                    (false, None)
                }
            }
            MeasEventType::A2 => {
                // Serving < threshold
                if let Some(thresh) = trigger.threshold {
                    (serving_rsrp < thresh - hyst, None)
                } else {
                    (false, None)
                }
            }
            MeasEventType::A3 | MeasEventType::A4 | MeasEventType::A5 => {
                // The best NR neighbour satisfying the event, measured against
                // the PCell (Ms/Mp of §5.5.4.4-.6).
                self.best_nr_neighbour(event_type, trigger, serving_rsrp, |cell_id| {
                    cell_id != serving_cell_id
                })
            }
            MeasEventType::A6 => {
                // §5.5.4.7: the reference is the SCell, not the PCell. Without an
                // SCell there is nothing to compare against, so no A6 event.
                let scell_id = match self.scell_id {
                    Some(id) => id,
                    None => return (false, None),
                };
                let scell_rsrp = match self.rsrp(scell_id) {
                    Some(rsrp) => rsrp,
                    None => return (false, None),
                };
                // Neither serving cell is a neighbour of itself.
                self.best_nr_neighbour(event_type, trigger, scell_rsrp, |cell_id| {
                    cell_id != serving_cell_id && cell_id != scell_id
                })
            }
            MeasEventType::B1 | MeasEventType::B2 => {
                self.best_eutra_neighbour(event_type, trigger, serving_rsrp)
            }
        }
    }

    /// The strongest NR neighbour whose entering condition holds, among the cells
    /// `is_neighbour` accepts.
    ///
    /// `reference_rsrp` is what the event compares against: the PCell for
    /// A3/A5, the SCell for A6, and nothing at all for A4.
    fn best_nr_neighbour(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        reference_rsrp: i32,
        is_neighbour: impl Fn(i32) -> bool,
    ) -> (bool, Option<TriggeringCell>) {
        let mut best: Option<(i32, i32)> = None;

        for (&cell_id, meas) in &self.measurements {
            if !is_neighbour(cell_id) {
                continue;
            }
            if let Some(rsrp) = meas.rsrp {
                if nr_entering_condition(event_type, trigger, reference_rsrp, rsrp)
                    && best.is_none_or(|(_, best_rsrp)| rsrp > best_rsrp)
                {
                    best = Some((cell_id, rsrp));
                }
            }
        }

        best.map_or((false, None), |(cell_id, _)| {
            (true, Some(TriggeringCell::Nr(cell_id)))
        })
    }

    /// The strongest inter-RAT neighbour whose B1/B2 entering condition holds.
    ///
    /// "Strongest" compares Mn + Ofn + Ocn, the same sum the inequality uses, so
    /// the cell reported is the one the event is actually about.
    fn best_eutra_neighbour(
        &self,
        event_type: MeasEventType,
        trigger: &ReportTriggerConfig,
        pcell_rsrp: i32,
    ) -> (bool, Option<TriggeringCell>) {
        let hyst = trigger.hysteresis / 2;

        // B2-1 (entering condition 1): Mp + Hys < Thresh1. Checked once, before
        // any neighbour: if the PCell is not weak enough, B2 cannot enter at all.
        if event_type == MeasEventType::B2 {
            match trigger.threshold1 {
                Some(thresh1) if pcell_rsrp < thresh1 - hyst => {}
                _ => return (false, None),
            }
        }

        // B1 reads b1-ThresholdEUTRA from `threshold`, B2 b2-Threshold2EUTRA
        // from `threshold2`.
        let threshold = match event_type {
            MeasEventType::B1 => trigger.threshold,
            _ => trigger.threshold2,
        };
        let threshold = match threshold {
            Some(t) => t,
            None => return (false, None),
        };

        let mut best: Option<(EutraCellKey, i32)> = None;
        for (&cell, meas) in &self.eutra_measurements {
            let Some(rsrp) = meas.rsrp else { continue };
            // Mn + Ofn + Ocn - Hys > Thresh  (B1-1, B2-2)
            let ofn = self
                .eutra_freq_offsets
                .get(&cell.earfcn)
                .copied()
                .unwrap_or(0);
            let level = rsrp + ofn + meas.cell_individual_offset;
            if level - hyst > threshold && best.is_none_or(|(_, best_level)| level > best_level) {
                best = Some((cell, level));
            }
        }

        best.map_or((false, None), |(cell, _)| {
            (true, Some(TriggeringCell::Eutra(cell)))
        })
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

        nr_entering_condition(event_type, trigger, reference_rsrp, candidate_rsrp)
    }

    fn generate_report(
        &self,
        meas_id: u8,
        serving_cell_id: i32,
        triggering_cell: Option<TriggeringCell>,
        max_cells: u8,
    ) -> MeasurementReport {
        // Get serving cell measurement
        let serving_cell = self
            .measurements
            .get(&serving_cell_id)
            .cloned()
            .expect("value expected");

        // Get neighbor cell measurements, sorted by RSRP
        let mut neighbors: Vec<_> = self
            .measurements
            .iter()
            .filter(|(&id, _)| id != serving_cell_id)
            .map(|(_, m)| m.clone())
            .collect();

        neighbors.sort_by(|a, b| b.rsrp.unwrap_or(i32::MIN).cmp(&a.rsrp.unwrap_or(i32::MIN)));

        // If there's a triggering NR cell, put it first
        if let Some(trig_id) = triggering_cell.and_then(|cell| cell.nr_cell_id()) {
            if let Some(pos) = neighbors.iter().position(|m| m.pci == trig_id as u32) {
                let trig = neighbors.remove(pos);
                neighbors.insert(0, trig);
            }
        }

        // Limit to max cells
        neighbors.truncate(max_cells as usize);

        // Inter-RAT results, strongest first, with the triggering cell hoisted.
        // The uplink MeasurementReport does not carry these yet: the UE's report
        // is a hand-rolled byte format (issue #107) and measResultListEUTRA needs
        // the real UPER encoder, so a B1/B2 report currently reaches the network
        // as its NR part only (issue #113).
        let mut eutra_neighbors: Vec<_> = self.eutra_measurements.values().cloned().collect();
        eutra_neighbors.sort_by(|a, b| b.rsrp.unwrap_or(i32::MIN).cmp(&a.rsrp.unwrap_or(i32::MIN)));
        if let Some(TriggeringCell::Eutra(trig)) = triggering_cell {
            if let Some(pos) = eutra_neighbors.iter().position(|m| m.cell == trig) {
                let trig = eutra_neighbors.remove(pos);
                eutra_neighbors.insert(0, trig);
            }
        }
        eutra_neighbors.truncate(max_cells as usize);

        MeasurementReport {
            meas_id,
            serving_cell,
            neighbor_cells: neighbors,
            eutra_neighbor_cells: eutra_neighbors,
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
        let state = manager.event_states.get(&1).unwrap();
        assert!(state.condition_met);
        assert_eq!(state.triggering_cell, Some(TriggeringCell::Nr(2)));
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

        let state = manager.event_states.get(&6).expect("A6 state");
        assert!(state.condition_met, "-90 dBm beats the -91 dBm A6 bar");
        assert_eq!(state.triggering_cell, Some(TriggeringCell::Nr(3)));
    }

    #[test]
    fn an_a6_event_does_not_enter_below_the_offset() {
        let mut manager = manager_with_scell();
        manager.update_measurement(3, -92); // one dB short of the bar
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        let state = manager.event_states.get(&6).expect("A6 state");
        assert!(!state.condition_met, "-92 dBm is below the -91 dBm A6 bar");
    }

    /// A6 measures against the SCell, so the PCell — the strongest cell here —
    /// must not be picked up as its own neighbour, and neither must the SCell.
    #[test]
    fn an_a6_event_treats_neither_serving_cell_as_a_neighbour() {
        let mut manager = manager_with_scell();
        manager.add_config(a6_config(3));

        manager.evaluate_events();

        let state = manager.event_states.get(&6).expect("A6 state");
        assert!(
            !state.condition_met,
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

        let state = manager.event_states.get(&6).expect("A6 state");
        assert!(!state.condition_met);
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

        let state = manager.event_states.get(&7).expect("B1 state");
        assert!(state.condition_met);
        assert_eq!(state.triggering_cell, Some(TriggeringCell::Eutra(eutra)));
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
            !manager.event_states[&7].condition_met,
            "-101 dBm with no offsets is below the -99 dBm bar"
        );

        manager.set_eutra_frequency_offset(1850, 2); // Ofn
        manager.set_eutra_cell_offset(eutra, 1); // Ocn
        manager.evaluate_events();
        assert!(
            manager.event_states[&7].condition_met,
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
        assert!(!manager.event_states[&7].condition_met, "PCell is not weak");

        // Weak PCell, but no neighbour worth going to: B2-2 fails.
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -115);
        manager.update_eutra_measurement(eutra, -120);
        manager.add_config(config.clone());
        manager.evaluate_events();
        assert!(
            !manager.event_states[&7].condition_met,
            "the inter-RAT neighbour is below threshold2"
        );

        // Both halves hold.
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -115);
        manager.update_eutra_measurement(eutra, -105);
        manager.add_config(config);
        manager.evaluate_events();
        assert!(manager.event_states[&7].condition_met);
        assert_eq!(
            manager.event_states[&7].triggering_cell,
            Some(TriggeringCell::Eutra(eutra))
        );
    }

    /// A B1 report carries the inter-RAT neighbour it triggered on. The uplink
    /// PDU cannot yet carry it (issue #113); the report the manager produces can.
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

        assert!(!manager.event_states[&5].condition_met);
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

        let state = manager.event_states.get(&2).unwrap();
        assert!(state.condition_met);
    }
}
