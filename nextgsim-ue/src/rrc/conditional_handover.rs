//! Conditional reconfiguration (CHO) runtime for the UE
//!
//! The UE side of TS 38.331 §5.3.5.13: candidate target configurations are
//! stored, their execution conditions are evaluated on every measurement tick,
//! and the first candidate whose condition has held for its time-to-trigger is
//! handed to the handover manager.
//!
//! # What lives where
//!
//! - The container codec is `nextgsim_rrc::procedures::conditional_handover`
//!   ([`ChoConfig`], [`ChoCandidateCell`], `decode_cho_config`). This module
//!   consumes those types; it does not define a format of its own.
//! - The inequalities are [`MeasurementManager::candidate_condition_holds`],
//!   the same ones events A3/A5 use. §5.3.5.13.4 requires the entry condition to
//!   be fulfilled "for the applicable cell" — the candidate's own target, not the
//!   best neighbour — which is why this asks per candidate.
//! - Execution is the existing [`HandoverManager`](crate::rrc::HandoverManager)
//!   path (T304, sync, RRCReconfigurationComplete): a conditional handover is a
//!   normal handover whose command the UE already had.
//!
//! # Conditions this runtime acts on
//!
//! Only the normative Rel-16 execution conditions `condEventA3` and
//! `condEventA5`. The container can also carry a timer-based, a predictive and
//! an AI-assisted condition; those are this simulator's research extensions with
//! no defined execution semantics in TS 38.331, so a candidate carrying one is
//! **stored and never triggered** rather than acted on by invented rules.
//! [`CondReconfigStore::unevaluated_count`] reports how many such candidates are
//! held, so they cannot be mistaken for armed ones.
//!
//! An A3 condition asking for RSRQ (`use_rsrp: false`) is unevaluated for the
//! same reason: the measurement manager holds RSRP only, and judging an RSRQ
//! condition against RSRP would be a different condition.
//!
//! # Reference
//! - 3GPP TS 38.331 §5.3.5.13.2 (removal), §5.3.5.13.3 (addition/modification),
//!   §5.3.5.13.4 (evaluation), §5.3.5.13.5 (execution); §5.3.5.3 for the release
//!   of stored candidates once a `reconfigurationWithSync` is applied.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use nextgsim_rrc::procedures::conditional_handover::{ChoCandidateCell, ChoCondition, ChoConfig};

use crate::rrc::handover::{HandoverCommand, TargetCellInfo};
use crate::rrc::measurement::{
    MeasEventType, MeasurementManager, ReportTriggerConfig, ReportTriggerType,
};

/// Identifies one stored conditional reconfiguration.
///
/// `condReconfigId` in `VarConditionalReconfig` terms. The container numbers
/// candidates within a configuration, so the pair is what is unique.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CondReconfigId {
    /// The configuration the candidate arrived in
    pub config_id: u8,
    /// The candidate's index within that configuration
    pub candidate_index: u8,
}

impl CondReconfigId {
    /// The identifier of candidate `candidate_index` of configuration `config_id`.
    pub fn new(config_id: u8, candidate_index: u8) -> Self {
        Self {
            config_id,
            candidate_index,
        }
    }
}

/// A candidate whose execution condition has held for its time-to-trigger.
#[derive(Debug, Clone)]
pub struct TriggeredCandidate {
    /// Which stored candidate triggered
    pub id: CondReconfigId,
    /// The candidate's stored target configuration
    pub candidate: ChoCandidateCell,
}

/// A stored candidate and the trigger state of its execution condition.
#[derive(Debug, Clone)]
struct StoredCandidate {
    /// The candidate as it arrived in the container
    candidate: ChoCandidateCell,
    /// The execution condition mapped onto a measurement event, or `None` for a
    /// condition this runtime does not act on
    trigger: Option<(MeasEventType, ReportTriggerConfig)>,
    /// When the condition first held; cleared whenever it stops holding, so the
    /// time-to-trigger measures an unbroken run (§5.3.5.13.4)
    condition_met_since: Option<Instant>,
}

/// The UE's store of conditional reconfigurations (`VarConditionalReconfig`).
#[derive(Debug, Default)]
pub struct CondReconfigStore {
    /// Ordered so that evaluation and candidate selection are deterministic
    candidates: BTreeMap<CondReconfigId, StoredCandidate>,
}

impl CondReconfigStore {
    /// An empty store: no conditional reconfiguration is configured.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a decoded configuration (§5.3.5.13.3, addition/modification).
    ///
    /// A candidate whose `condReconfigId` is already stored is replaced, which
    /// also resets its trigger state: modification is not a continuation of the
    /// old condition's time-to-trigger.
    ///
    /// Returns the number of candidates now held for this `config_id`.
    pub fn add_config(&mut self, config: &ChoConfig) -> usize {
        for candidate in &config.candidate_cells {
            let id = CondReconfigId::new(config.config_id, candidate.candidate_index);
            let trigger = map_condition(&candidate.condition);
            if trigger.is_none() {
                tracing::info!(
                    "CHO candidate {}/{} stored but not armed: {:?} has no execution semantics this UE acts on",
                    id.config_id,
                    id.candidate_index,
                    candidate.condition.condition_type()
                );
            }
            self.candidates.insert(
                id,
                StoredCandidate {
                    candidate: candidate.clone(),
                    trigger,
                    condition_met_since: None,
                },
            );
        }
        self.candidates
            .keys()
            .filter(|id| id.config_id == config.config_id)
            .count()
    }

    /// Drop every candidate of one configuration (§5.3.5.13.2, removal).
    ///
    /// Removing a `condReconfigId` that is not stored is not an error, per
    /// NOTE 1 of that clause.
    pub fn remove_config(&mut self, config_id: u8) {
        self.candidates.retain(|id, _| id.config_id != config_id);
    }

    /// Release every stored candidate.
    ///
    /// What §5.3.5.3 requires once a `reconfigurationWithSync` is applied, which
    /// includes the conditional handover the UE just executed: the candidates
    /// were prepared by the source cell and mean nothing at the target.
    pub fn clear(&mut self) {
        self.candidates.clear();
    }

    /// How many candidates are stored.
    pub fn candidate_count(&self) -> usize {
        self.candidates.len()
    }

    /// How many stored candidates carry a condition this runtime does not
    /// evaluate, and so can never trigger.
    pub fn unevaluated_count(&self) -> usize {
        self.candidates
            .values()
            .filter(|stored| stored.trigger.is_none())
            .count()
    }

    /// The identifiers of the stored candidates, in a stable order.
    pub fn ids(&self) -> Vec<CondReconfigId> {
        self.candidates.keys().copied().collect()
    }

    /// Evaluate every stored candidate and select one to execute.
    ///
    /// Per §5.3.5.13.4 a candidate is triggered when its entry condition has
    /// been fulfilled for its target cell throughout the condition's
    /// time-to-trigger; a condition that stops holding resets that run. When
    /// several candidates are triggered, §5.3.5.13.5 leaves the choice to the UE:
    /// this picks the lowest `priority` value, tie-broken by `condReconfigId`, so
    /// the same measurements always select the same candidate.
    pub fn evaluate(&mut self, measurements: &MeasurementManager) -> Option<TriggeredCandidate> {
        let mut triggered: Option<(u8, CondReconfigId)> = None;

        for (&id, stored) in &mut self.candidates {
            let Some((event_type, ref trigger)) = stored.trigger else {
                continue;
            };
            let cell_id = candidate_cell_id(&stored.candidate);

            if !measurements.candidate_condition_holds(event_type, trigger, cell_id) {
                stored.condition_met_since = None;
                continue;
            }

            let since = *stored.condition_met_since.get_or_insert_with(Instant::now);
            if since.elapsed() < Duration::from_millis(trigger.time_to_trigger) {
                continue;
            }

            let priority = stored.candidate.priority;
            if triggered.is_none_or(|(best, _)| priority < best) {
                triggered = Some((priority, id));
            }
        }

        let (_, id) = triggered?;
        let stored = self.candidates.get(&id)?;
        tracing::info!(
            "CHO condition met: candidate {}/{} targeting PCI {} (priority {})",
            id.config_id,
            id.candidate_index,
            stored.candidate.target_cell.phys_cell_id,
            stored.candidate.priority
        );
        Some(TriggeredCandidate {
            id,
            candidate: stored.candidate.clone(),
        })
    }
}

/// The simulator cell id of a candidate's target cell.
///
/// The UE measurement path keys cells by the RLS cell id and reports that id as
/// the physical cell identity (`MeasurementManager::update_measurement` sets
/// `pci: cell_id as u32`), so the two are the same number throughout this
/// simulator and a candidate's `phys_cell_id` identifies the measured cell.
pub fn candidate_cell_id(candidate: &ChoCandidateCell) -> i32 {
    i32::from(candidate.target_cell.phys_cell_id)
}

/// Build the handover command that executes a triggered candidate.
///
/// §5.3.5.13.5: the UE applies the candidate's stored `condRRCReconfig` and then
/// performs the ordinary §5.3.5.3 actions — so the conditional handover joins the
/// same path a network-ordered handover takes.
pub fn handover_command_for(candidate: &ChoCandidateCell, transaction_id: u8) -> HandoverCommand {
    HandoverCommand {
        target_cell: TargetCellInfo {
            pci: u32::from(candidate.target_cell.phys_cell_id),
            cell_id: Some(candidate_cell_id(candidate)),
            // The container carries no C-RNTI: the target assigns one, and this
            // simulator's RLS does not model a random-access exchange.
            new_ue_id: None,
            arfcn: Some(candidate.target_cell.ssb_frequency_arfcn),
            ssb_offset: None,
        },
        new_security_config: false,
        full_config: false,
        transaction_id,
    }
}

/// Map an execution condition onto the measurement event that decides it.
///
/// `None` for the conditions this runtime does not act on: the three research
/// extensions, and an A3 condition asking for RSRQ (see the module docs).
///
/// The container's dB values are `f64` on the RRC 0.5 dB grid while the
/// measurement manager works in whole dB with hysteresis in 0.5 dB units, so the
/// offsets round to the nearest dB and the hysteresis converts to its signalled
/// units.
fn map_condition(condition: &ChoCondition) -> Option<(MeasEventType, ReportTriggerConfig)> {
    match condition {
        ChoCondition::EventA3(a3) if a3.use_rsrp => Some((
            MeasEventType::A3,
            ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::A3),
                threshold: None,
                threshold1: None,
                threshold2: None,
                a3_offset: Some(a3.a3_offset.0.round() as i32),
                a6_offset: None,
                hysteresis: (a3.hysteresis.0 * 2.0).round() as i32,
                time_to_trigger: u64::from(a3.time_to_trigger.to_ms()),
            },
        )),
        ChoCondition::EventA5(a5) => Some((
            MeasEventType::A5,
            ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::A5),
                threshold: None,
                threshold1: Some(i32::from(a5.threshold1.0)),
                threshold2: Some(i32::from(a5.threshold2.0)),
                a3_offset: None,
                a6_offset: None,
                hysteresis: (a5.hysteresis.0 * 2.0).round() as i32,
                time_to_trigger: u64::from(a5.time_to_trigger.to_ms()),
            },
        )),
        ChoCondition::EventA3(_)
        | ChoCondition::Timer(_)
        | ChoCondition::Predictive(_)
        | ChoCondition::AiAssisted(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_rrc::procedures::conditional_handover::{
        encode_cho_config, A3Offset, AiAssistedCondition, ChoTargetCellConfig, EventA3Condition,
        EventA5Condition, Hysteresis, RsrpThreshold, TimeToTrigger, TimerBasedCondition,
    };

    fn target(pci: u16) -> ChoTargetCellConfig {
        ChoTargetCellConfig {
            phys_cell_id: pci,
            ssb_frequency_arfcn: 620000,
            ssb_subcarrier_spacing_khz: 30,
            nr_cell_identity: Some(0x1_2345_6789),
            plmn_identity: Some([0x00, 0xF1, 0x10]),
            rrc_reconfiguration: None,
        }
    }

    fn a3_condition(ttt: TimeToTrigger) -> ChoCondition {
        ChoCondition::EventA3(EventA3Condition {
            a3_offset: A3Offset::new(3.0).unwrap(),
            hysteresis: Hysteresis::new(1.0).unwrap(),
            time_to_trigger: ttt,
            use_rsrp: true,
        })
    }

    fn candidate(index: u8, pci: u16, priority: u8, condition: ChoCondition) -> ChoCandidateCell {
        ChoCandidateCell {
            candidate_index: index,
            condition,
            target_cell: target(pci),
            priority,
        }
    }

    fn config(candidates: Vec<ChoCandidateCell>) -> ChoConfig {
        ChoConfig {
            config_id: 1,
            candidate_cells: candidates,
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        }
    }

    /// Serving cell 1 at -90 dBm; cells 2 and 3 are measurable neighbours.
    fn measurements(cell2: i32, cell3: i32) -> MeasurementManager {
        let mut manager = MeasurementManager::new();
        manager.set_serving_cell(Some(1));
        manager.update_measurement(1, -90);
        manager.update_measurement(2, cell2);
        manager.update_measurement(3, cell3);
        manager
    }

    #[test]
    fn an_empty_store_triggers_nothing() {
        let mut store = CondReconfigStore::new();
        assert_eq!(store.candidate_count(), 0);
        assert!(store.evaluate(&measurements(-60, -60)).is_none());
    }

    /// A3 with offset 3 and hysteresis 1 dB against a -90 dBm serving cell: the
    /// candidate must beat -86 dBm. Time-to-trigger 0, so it fires at once.
    #[test]
    fn an_a3_candidate_triggers_once_its_own_cell_beats_the_serving_cell() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms0),
        )]));

        assert!(
            store.evaluate(&measurements(-87, -60)).is_none(),
            "-87 dBm is below the -86 dBm bar, and cell 3 is not a candidate"
        );

        let triggered = store
            .evaluate(&measurements(-85, -60))
            .expect("-85 dBm beats the bar");
        assert_eq!(triggered.id, CondReconfigId::new(1, 0));
        assert_eq!(triggered.candidate.target_cell.phys_cell_id, 2);
    }

    /// §5.3.5.13.4: the condition must hold *throughout* the time-to-trigger.
    #[test]
    fn a_candidate_waits_out_its_time_to_trigger() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms40),
        )]));

        let strong = measurements(-70, -120);
        assert!(
            store.evaluate(&strong).is_none(),
            "the condition has only just started holding"
        );

        std::thread::sleep(Duration::from_millis(45));
        assert!(
            store.evaluate(&strong).is_some(),
            "40 ms of an unbroken condition is enough"
        );
    }

    /// A condition that lapses restarts its time-to-trigger rather than
    /// resuming it.
    #[test]
    fn a_lapsed_condition_restarts_the_time_to_trigger() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms40),
        )]));

        store.evaluate(&measurements(-70, -120));
        std::thread::sleep(Duration::from_millis(30));
        // The candidate cell fades: the run is broken.
        assert!(store.evaluate(&measurements(-100, -120)).is_none());
        std::thread::sleep(Duration::from_millis(20));
        assert!(
            store.evaluate(&measurements(-70, -120)).is_none(),
            "the 50 ms since the first evaluation do not count: the run restarted"
        );
    }

    /// The condition is evaluated for the candidate's own cell (§5.3.5.13.4), so
    /// a strong cell nobody nominated triggers nothing.
    #[test]
    fn a_strong_cell_that_is_not_a_candidate_triggers_nothing() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms0),
        )]));

        assert!(store.evaluate(&measurements(-120, -50)).is_none());
    }

    /// §5.3.5.13.5 leaves the selection to the UE; this one goes by priority.
    #[test]
    fn the_highest_priority_triggered_candidate_is_selected() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![
            candidate(0, 2, 5, a3_condition(TimeToTrigger::Ms0)),
            candidate(1, 3, 1, a3_condition(TimeToTrigger::Ms0)),
        ]));

        let triggered = store
            .evaluate(&measurements(-60, -70))
            .expect("both candidates satisfy their condition");
        assert_eq!(
            triggered.candidate.target_cell.phys_cell_id, 3,
            "priority 1 outranks priority 5 even though cell 2 is stronger"
        );
    }

    /// An A5 candidate needs the serving cell below threshold1 and its own cell
    /// above threshold2.
    #[test]
    fn an_a5_candidate_needs_a_weak_serving_cell() {
        let a5 = ChoCondition::EventA5(EventA5Condition {
            threshold1: RsrpThreshold::new(-100).unwrap(),
            threshold2: RsrpThreshold::new(-110).unwrap(),
            hysteresis: Hysteresis::new(1.0).unwrap(),
            time_to_trigger: TimeToTrigger::Ms0,
        });
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(0, 2, 0, a5)]));

        // Serving cell at -90 dBm is healthier than threshold1 - hysteresis.
        assert!(store.evaluate(&measurements(-105, -120)).is_none());

        let mut weak = measurements(-105, -120);
        weak.update_measurement(1, -115);
        assert!(store.evaluate(&weak).is_some());
    }

    /// The research conditions are stored so a configuration is not silently
    /// dropped, and never trigger, because no execution semantics are defined
    /// for them.
    #[test]
    fn a_research_condition_is_stored_but_never_triggers() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![
            candidate(
                0,
                2,
                0,
                ChoCondition::Timer(TimerBasedCondition {
                    timer_ms: 0,
                    restart_on_improvement: false,
                }),
            ),
            candidate(
                1,
                3,
                0,
                ChoCondition::AiAssisted(AiAssistedCondition {
                    model_id: "beam-predict-v3".to_string(),
                    model_version: "1.0.0".to_string(),
                    feature_vector: vec![0.5],
                    decision_threshold: 0.75,
                    fallback_condition: None,
                }),
            ),
        ]));

        assert_eq!(store.candidate_count(), 2);
        assert_eq!(store.unevaluated_count(), 2);
        assert!(
            store.evaluate(&measurements(-40, -40)).is_none(),
            "no measurement can trigger a condition with no semantics"
        );
    }

    /// An RSRQ-based A3 condition is not evaluated against RSRP.
    #[test]
    fn an_rsrq_based_condition_is_not_evaluated() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            ChoCondition::EventA3(EventA3Condition {
                a3_offset: A3Offset::new(3.0).unwrap(),
                hysteresis: Hysteresis::new(1.0).unwrap(),
                time_to_trigger: TimeToTrigger::Ms0,
                use_rsrp: false,
            }),
        )]));

        assert_eq!(store.unevaluated_count(), 1);
        assert!(store.evaluate(&measurements(-40, -40)).is_none());
    }

    /// §5.3.5.13.3: re-adding a `condReconfigId` replaces it, and the new
    /// condition's time-to-trigger starts over.
    #[test]
    fn re_adding_a_candidate_replaces_it_and_resets_its_run() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms40),
        )]));
        store.evaluate(&measurements(-60, -120));
        std::thread::sleep(Duration::from_millis(45));

        // Same id, a longer time-to-trigger: the elapsed run does not carry over.
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms640),
        )]));
        assert_eq!(store.candidate_count(), 1);
        assert!(store.evaluate(&measurements(-60, -120)).is_none());
    }

    /// §5.3.5.13.2: removal drops the configuration's candidates, and removing
    /// one that was never stored is not an error.
    #[test]
    fn removal_drops_a_configuration_and_tolerates_an_unknown_id() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![candidate(
            0,
            2,
            0,
            a3_condition(TimeToTrigger::Ms0),
        )]));

        store.remove_config(9); // never configured
        assert_eq!(store.candidate_count(), 1);

        store.remove_config(1);
        assert_eq!(store.candidate_count(), 0);
        assert!(store.evaluate(&measurements(-40, -40)).is_none());
    }

    /// The store's input is a container decoded by the nextgsim-rrc CHO codec,
    /// not a hand-built struct: a candidate that has been through the wire
    /// triggers a handover command for its target.
    #[test]
    fn a_candidate_decoded_from_a_container_triggers_a_handover_command() {
        let encoded = encode_cho_config(&config(vec![candidate(
            4,
            3,
            0,
            a3_condition(TimeToTrigger::Ms0),
        )]))
        .expect("encode");
        let decoded = nextgsim_rrc::procedures::conditional_handover::decode_cho_config(&encoded)
            .expect("decode");

        let mut store = CondReconfigStore::new();
        assert_eq!(store.add_config(&decoded), 1);

        let triggered = store
            .evaluate(&measurements(-120, -60))
            .expect("cell 3 beats the serving cell by more than the offset");
        assert_eq!(triggered.id, CondReconfigId::new(1, 4));

        let command = handover_command_for(&triggered.candidate, 7);
        assert_eq!(command.target_cell.cell_id, Some(3));
        assert_eq!(command.target_cell.pci, 3);
        assert_eq!(command.target_cell.arfcn, Some(620000));
        assert_eq!(command.transaction_id, 7);
    }

    #[test]
    fn clearing_the_store_releases_every_candidate() {
        let mut store = CondReconfigStore::new();
        store.add_config(&config(vec![
            candidate(0, 2, 0, a3_condition(TimeToTrigger::Ms0)),
            candidate(1, 3, 0, a3_condition(TimeToTrigger::Ms0)),
        ]));
        assert_eq!(store.candidate_count(), 2);

        store.clear();
        assert_eq!(store.candidate_count(), 0);
    }
}
