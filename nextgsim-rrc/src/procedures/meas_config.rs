//! `measConfig`: the measurement configuration the network gives a UE
//! (TS 38.331 §5.5.2, §6.3.2 `MeasConfig`).
//!
//! # Why this module exists (issue #170)
//!
//! `RRCReconfiguration.measConfig` and `RRCResume.measConfig` were hardcoded
//! `None`, so a gNB could not configure UE measurements at all. The UE ran its A3
//! *reporting* trigger off a hard-coded local default while the gNB decided
//! handovers on its own configured margin, and nothing tied the two together —
//! two independent A3 margins with no wire path between them.
//!
//! This builds a real, generated-codec `MeasConfig` from the gNB's configured
//! offset and hysteresis, and reads one back on the UE side. One source of truth,
//! no new wire format.
//!
//! # The two blockers #170 named do not apply, and both were verified
//!
//! **#117 (an extended CHOICE cannot be encoded) never touched this.** The refusal
//! was for an extension *arm*, not for an extensible *type*. Every arm an intra-NR
//! A3 configuration selects is a `extended = false` ROOT arm that always encoded:
//! `MeasObjectToAddModMeasObject::MeasObjectNR`,
//! `ReportConfigToAddModReportConfig::ReportConfigNR`,
//! `ReportConfigNRReportType::EventTriggered` and
//! `EventTriggerConfigEventId::EventA3`. (The extension arms of those CHOICEs are
//! the inter-RAT and SFTD/CGI ones, which this module does not build.) #117 has
//! landed anyway.
//!
//! **#105 (the vendored schema is Rel-15) did not gate it either.** A3 reporting
//! is a Rel-15 feature: `MeasConfig` is fully modelled with all ten optional
//! fields, `EventTriggerConfigEventId_eventA3` carries `a3-Offset`,
//! `reportOnLeave`, `hysteresis`, `timeToTrigger` and `useAllowedCellList`, and
//! `RRCReconfiguration_IEs.meas_config` sits at `optional_idx = 2`. #105 has since
//! landed: the schema is now Rel-19 (`tools/rrc-19.3.0.asn1`), which renamed
//! `useWhiteCellList` to `useAllowedCellList` in place without moving it.
//!
//! # Units
//!
//! The two quantities are signalled in DIFFERENT units, which is the whole reason
//! [`a3_offset_signalled`] and [`hysteresis_signalled`] are named functions rather
//! than arithmetic at each site:
//!
//! * `a3-Offset` is a `MeasTriggerQuantityOffset`, `INTEGER (-30..30)` in units of
//!   **1 dB** ("Values in the unit of 'dB'", TS 38.331 §6.3.2).
//! * `hysteresis` is `INTEGER (0..30)` in units of **0.5 dB** ("The actual value
//!   is field value * 0.5 dB", §6.3.2).
//!
//! Those are exactly the units the UE's own measurement runtime works in
//! (`ReportTriggerConfig.a3_offset` is whole dB; `.hysteresis` is 0.5 dB units),
//! which is why a received configuration installs with **no** further conversion —
//! and why the conditional-handover path, whose container carries half-dB for
//! both, converts through these same two functions rather than inlining a second
//! copy of the arithmetic.

use crate::codec::generated::*;
use thiserror::Error;

/// Errors building or reading a `measConfig`.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MeasConfigError {
    /// A field value TS 38.331 does not allow.
    #[error("invalid measConfig field: {0}")]
    InvalidFieldValue(String),
}

/// `MeasId`, `MeasObjectId` and `ReportConfigId` are all `INTEGER (1..64)`.
const ID_RANGE: std::ops::RangeInclusive<u8> = 1..=64;

/// `maxReportCells` is `INTEGER (1..8)` in an `EventTriggerConfig`.
const MAX_REPORT_CELLS_RANGE: std::ops::RangeInclusive<u8> = 1..=8;

/// `a3-Offset` as the wire integer, from a value in dB.
///
/// `MeasTriggerQuantityOffset ::= CHOICE { rsrp INTEGER (-30..30), … }` in units of
/// **1 dB**, so a fractional dB cannot be signalled and rounds to the grid.
///
/// Not fused with [`hysteresis_signalled`]: the two IEs are in different units, and
/// a single "to signalled units" helper would make one of the two call sites wrong
/// by a factor of two with nothing at the site to show it.
pub fn a3_offset_signalled(db: f64) -> Result<i8, MeasConfigError> {
    let rounded = db.round();
    if !(-30.0..=30.0).contains(&rounded) {
        return Err(MeasConfigError::InvalidFieldValue(format!(
            "a3-Offset {db} dB is outside MeasTriggerQuantityOffset's INTEGER (-30..30)"
        )));
    }
    Ok(rounded as i8)
}

/// `hysteresis` as the wire integer, from a value in dB.
///
/// `Hysteresis ::= INTEGER (0..30)` in units of **0.5 dB**, so the dB value
/// doubles and a value off the half-dB grid rounds to it.
pub fn hysteresis_signalled(db: f64) -> Result<u8, MeasConfigError> {
    let half_db = (db * 2.0).round();
    if !(0.0..=30.0).contains(&half_db) {
        return Err(MeasConfigError::InvalidFieldValue(format!(
            "hysteresis {db} dB is {half_db} in 0.5 dB units, outside INTEGER (0..30)"
        )));
    }
    Ok(half_db as u8)
}

/// The parameters of the intra-NR A3 reporting configuration a gNB signals.
///
/// One `measObject` (the NR frequency), one `reportConfig` (`eventA3`) and one
/// `measId` binding them, which is the minimum TS 38.331 §5.5.2 needs for the UE
/// to report at all.
#[derive(Debug, Clone, PartialEq)]
pub struct A3MeasConfigParams {
    /// `measId` the binding is installed under (1..=64).
    pub meas_id: u8,
    /// `measObjectId` of the NR measurement object (1..=64).
    pub meas_object_id: u8,
    /// `reportConfigId` of the A3 report configuration (1..=64).
    pub report_config_id: u8,
    /// `ssbFrequency` of the measured NR carrier, in ARFCN.
    pub ssb_frequency_arfcn: u32,
    /// `ssbSubcarrierSpacing` in kHz (15, 30, 60, 120 or 240).
    pub ssb_subcarrier_spacing_khz: u16,
    /// `a3-Offset` in dB: the neighbour must beat the serving cell by this much.
    pub a3_offset_db: f64,
    /// `hysteresis` in dB.
    pub hysteresis_db: f64,
    /// `timeToTrigger` in ms; must be a value of the ENUMERATED.
    pub time_to_trigger_ms: u16,
    /// `reportInterval` in ms; must be a value of the ENUMERATED.
    pub report_interval_ms: u32,
    /// `reportAmount`: how many reports per trigger. `None` is `infinity`.
    pub report_amount: Option<u32>,
    /// `maxReportCells` (1..=8).
    pub max_report_cells: u8,
}

/// What a received `measConfig`'s A3 `measId` says, in the signalled units.
///
/// Deliberately the WIRE units rather than dB: the UE's measurement runtime works
/// in whole-dB offsets and 0.5 dB hysteresis, which is what the two IEs already
/// carry, so a `measConfig` installs with no conversion and no rounding. A dB
/// round trip here would introduce one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct A3MeasConfigRead {
    /// `measId` this binding is under.
    pub meas_id: u8,
    /// `measObjectId` the binding names.
    pub meas_object_id: u8,
    /// `reportConfigId` the binding names.
    pub report_config_id: u8,
    /// `a3-Offset` in whole dB, as signalled.
    pub a3_offset_db: i8,
    /// `hysteresis` in 0.5 dB units, as signalled.
    pub hysteresis_half_db: u8,
    /// `timeToTrigger` in ms.
    pub time_to_trigger_ms: u16,
    /// `reportInterval` in ms.
    pub report_interval_ms: u32,
    /// `reportAmount`; `None` for `infinity`.
    pub report_amount: Option<u32>,
    /// `maxReportCells`.
    pub max_report_cells: u8,
}

/// `timeToTrigger` in ms → its ENUMERATED index (TS 38.331 §6.3.2).
fn time_to_trigger_index(ms: u16) -> Result<u8, MeasConfigError> {
    Ok(match ms {
        0 => TimeToTrigger::MS0,
        40 => TimeToTrigger::MS40,
        64 => TimeToTrigger::MS64,
        80 => TimeToTrigger::MS80,
        100 => TimeToTrigger::MS100,
        128 => TimeToTrigger::MS128,
        160 => TimeToTrigger::MS160,
        256 => TimeToTrigger::MS256,
        320 => TimeToTrigger::MS320,
        480 => TimeToTrigger::MS480,
        512 => TimeToTrigger::MS512,
        640 => TimeToTrigger::MS640,
        1024 => TimeToTrigger::MS1024,
        1280 => TimeToTrigger::MS1280,
        2560 => TimeToTrigger::MS2560,
        5120 => TimeToTrigger::MS5120,
        other => {
            return Err(MeasConfigError::InvalidFieldValue(format!(
                "timeToTrigger {other} ms is not a value of the ENUMERATED"
            )))
        }
    })
}

/// The ms a `timeToTrigger` index denotes. `None` for an index the ENUMERATED has
/// no value for, which a successful decode cannot produce.
fn time_to_trigger_ms(index: u8) -> Option<u16> {
    Some(match index {
        TimeToTrigger::MS0 => 0,
        TimeToTrigger::MS40 => 40,
        TimeToTrigger::MS64 => 64,
        TimeToTrigger::MS80 => 80,
        TimeToTrigger::MS100 => 100,
        TimeToTrigger::MS128 => 128,
        TimeToTrigger::MS160 => 160,
        TimeToTrigger::MS256 => 256,
        TimeToTrigger::MS320 => 320,
        TimeToTrigger::MS480 => 480,
        TimeToTrigger::MS512 => 512,
        TimeToTrigger::MS640 => 640,
        TimeToTrigger::MS1024 => 1024,
        TimeToTrigger::MS1280 => 1280,
        TimeToTrigger::MS2560 => 2560,
        TimeToTrigger::MS5120 => 5120,
        _ => return None,
    })
}

/// `reportInterval` in ms → its ENUMERATED index. The `min1`..`min30` values are
/// expressed in ms too, because that is the unit the UE's report timer uses.
fn report_interval_index(ms: u32) -> Result<u8, MeasConfigError> {
    Ok(match ms {
        120 => ReportInterval::MS120,
        240 => ReportInterval::MS240,
        480 => ReportInterval::MS480,
        640 => ReportInterval::MS640,
        1024 => ReportInterval::MS1024,
        2048 => ReportInterval::MS2048,
        5120 => ReportInterval::MS5120,
        10240 => ReportInterval::MS10240,
        20480 => ReportInterval::MS20480,
        40960 => ReportInterval::MS40960,
        60_000 => ReportInterval::MIN1,
        360_000 => ReportInterval::MIN6,
        720_000 => ReportInterval::MIN12,
        1_800_000 => ReportInterval::MIN30,
        other => {
            return Err(MeasConfigError::InvalidFieldValue(format!(
                "reportInterval {other} ms is not a value of the ENUMERATED"
            )))
        }
    })
}

/// The ms a `reportInterval` index denotes.
fn report_interval_ms(index: u8) -> Option<u32> {
    Some(match index {
        ReportInterval::MS120 => 120,
        ReportInterval::MS240 => 240,
        ReportInterval::MS480 => 480,
        ReportInterval::MS640 => 640,
        ReportInterval::MS1024 => 1024,
        ReportInterval::MS2048 => 2048,
        ReportInterval::MS5120 => 5120,
        ReportInterval::MS10240 => 10240,
        ReportInterval::MS20480 => 20480,
        ReportInterval::MS40960 => 40960,
        ReportInterval::MIN1 => 60_000,
        ReportInterval::MIN6 => 360_000,
        ReportInterval::MIN12 => 720_000,
        ReportInterval::MIN30 => 1_800_000,
        _ => return None,
    })
}

/// `reportAmount` → its ENUMERATED index. `None` is `infinity`.
fn report_amount_index(amount: Option<u32>) -> Result<u8, MeasConfigError> {
    Ok(match amount {
        None => EventTriggerConfigReportAmount::INFINITY,
        Some(1) => EventTriggerConfigReportAmount::R1,
        Some(2) => EventTriggerConfigReportAmount::R2,
        Some(4) => EventTriggerConfigReportAmount::R4,
        Some(8) => EventTriggerConfigReportAmount::R8,
        Some(16) => EventTriggerConfigReportAmount::R16,
        Some(32) => EventTriggerConfigReportAmount::R32,
        Some(64) => EventTriggerConfigReportAmount::R64,
        Some(other) => {
            return Err(MeasConfigError::InvalidFieldValue(format!(
            "reportAmount {other} is not a value of the ENUMERATED (1,2,4,8,16,32,64 or infinity)"
        )))
        }
    })
}

/// The count a `reportAmount` index denotes; `None` for `infinity`.
fn report_amount_value(index: u8) -> Option<u32> {
    match index {
        EventTriggerConfigReportAmount::R1 => Some(1),
        EventTriggerConfigReportAmount::R2 => Some(2),
        EventTriggerConfigReportAmount::R4 => Some(4),
        EventTriggerConfigReportAmount::R8 => Some(8),
        EventTriggerConfigReportAmount::R16 => Some(16),
        EventTriggerConfigReportAmount::R32 => Some(32),
        EventTriggerConfigReportAmount::R64 => Some(64),
        // `infinity`, and any index the ENUMERATED has no value for — which a
        // successful decode cannot produce. Both mean "do not stop".
        _ => None,
    }
}

/// Subcarrier spacing in kHz → its `SubcarrierSpacing` ENUMERATED index.
fn subcarrier_spacing_index(khz: u16) -> Result<u8, MeasConfigError> {
    Ok(match khz {
        15 => SubcarrierSpacing::K_HZ15,
        30 => SubcarrierSpacing::K_HZ30,
        60 => SubcarrierSpacing::K_HZ60,
        120 => SubcarrierSpacing::K_HZ120,
        240 => SubcarrierSpacing::K_HZ240,
        other => {
            return Err(MeasConfigError::InvalidFieldValue(format!(
                "ssbSubcarrierSpacing {other} kHz is not a value of SubcarrierSpacing"
            )))
        }
    })
}

/// Build the `measConfig` that configures one intra-NR A3 reporting measurement.
///
/// # What is deliberately absent
///
/// * **`quantityConfig`.** The layer-3 filter coefficients (§5.5.3.2). This
///   simulator applies no filtering — `MeasurementManager` uses the level it was
///   handed — so a `quantityConfig` would describe a transform neither end
///   performs.
/// * **`sMeasureConfig`.** The serving-cell level above which the UE may stop
///   measuring neighbours. Omitting it means "always measure", which is what the
///   UE does.
/// * **`measGapConfig` / `measGapSharingConfig`.** Gaps are a scheduler concept
///   and there is no scheduler.
/// * **the `*ToRemoveList`s.** This is a fresh configuration, not a delta: the UE
///   has nothing signalled to remove. (Its own pre-signalling default is replaced
///   by `measId`, which is how a `measIdToAddModList` entry for an existing
///   `measId` behaves per §5.5.2.5.)
/// * **`cellsToAddModList`.** Per-cell `cellIndividualOffset`s (Ocn of §5.5.4).
///   The UE's A3 arithmetic takes Ocn as zero, which §5.5.4 allows when it is not
///   configured, so signalling one would be read as zero anyway.
pub fn build_a3_meas_config(params: &A3MeasConfigParams) -> Result<MeasConfig, MeasConfigError> {
    for (name, id) in [
        ("measId", params.meas_id),
        ("measObjectId", params.meas_object_id),
        ("reportConfigId", params.report_config_id),
    ] {
        if !ID_RANGE.contains(&id) {
            return Err(MeasConfigError::InvalidFieldValue(format!(
                "{name} {id} is outside INTEGER (1..64)"
            )));
        }
    }
    if !MAX_REPORT_CELLS_RANGE.contains(&params.max_report_cells) {
        return Err(MeasConfigError::InvalidFieldValue(format!(
            "maxReportCells {} is outside INTEGER (1..8)",
            params.max_report_cells
        )));
    }
    if params.ssb_frequency_arfcn > ARFCN_VALUE_NR_MAX {
        return Err(MeasConfigError::InvalidFieldValue(format!(
            "ssbFrequency {} is outside ARFCN-ValueNR's INTEGER (0..{ARFCN_VALUE_NR_MAX})",
            params.ssb_frequency_arfcn
        )));
    }

    let a3_offset = a3_offset_signalled(params.a3_offset_db)?;
    let hysteresis = hysteresis_signalled(params.hysteresis_db)?;

    let meas_object = MeasObjectNR {
        ssb_frequency: Some(ARFCN_ValueNR(params.ssb_frequency_arfcn)),
        ssb_subcarrier_spacing: Some(SubcarrierSpacing(subcarrier_spacing_index(
            params.ssb_subcarrier_spacing_khz,
        )?)),
        // No SSB measurement timing configuration: there is no PHY to time a
        // measurement window against.
        smtc1: None,
        smtc2: None,
        ref_freq_csi_rs: None,
        // Both arms absent: the UE measures SSB RSRP directly off RLS, so neither
        // an `ssb-ConfigMobility` nor a CSI-RS resource config describes anything
        // it does.
        reference_signal_config: ReferenceSignalConfig {
            ssb_config_mobility: None,
            csi_rs_resource_config_mobility: None,
        },
        abs_thresh_ss_blocks_consolidation: None,
        abs_thresh_csi_rs_consolidation: None,
        nrof_ss_blocks_to_average: None,
        nrof_csi_rs_resources_to_average: None,
        // Index 1: the first `quantityConfigNR` of a `quantityConfigNR-List`.
        // Mandatory in the ASN.1, and the only legal value when no
        // `quantityConfig` is signalled.
        quantity_config_index: MeasObjectNRQuantityConfigIndex(1),
        // Mandatory, and every offset within it is OPTIONAL: absent means zero,
        // which is the Ofn the UE's arithmetic uses.
        offset_mo: Q_OffsetRangeList {
            rsrp_offset_ssb: None,
            rsrq_offset_ssb: None,
            sinr_offset_ssb: None,
            rsrp_offset_csi_rs: None,
            rsrq_offset_csi_rs: None,
            sinr_offset_csi_rs: None,
        },
        cells_to_remove_list: None,
        cells_to_add_mod_list: None,
        // Rel-16 renamed `blackCellsTo*List` to `excludedCellsTo*List` and
        // `whiteCellsTo*List` to `allowedCellsTo*List` in the same SEQUENCE
        // positions, so the UPER layout is unchanged by the rename.
        excluded_cells_to_remove_list: None,
        excluded_cells_to_add_mod_list: None,
        allowed_cells_to_remove_list: None,
        allowed_cells_to_add_mod_list: None,
    };

    let event_a3 = EventTriggerConfigEventId_eventA3 {
        a3_offset: MeasTriggerQuantityOffset::Rsrp(MeasTriggerQuantityOffset_rsrp(a3_offset)),
        // `false`: a leaving cell does not itself cause a report. The UE's
        // measurement runtime removes it from `cellsTriggeredList` and reports on
        // its interval while any cell remains, which is §5.5.4.1 without the
        // `reportOnLeave` extra. Signalling `true` would ask for a report the UE
        // does not generate.
        report_on_leave: EventTriggerConfigEventId_eventA3ReportOnLeave(false),
        hysteresis: Hysteresis(hysteresis),
        time_to_trigger: TimeToTrigger(time_to_trigger_index(params.time_to_trigger_ms)?),
        // `false`: no `allowedCellsToAddModList` is signalled, so restricting the
        // event to an allowed list would restrict it to the empty set and nothing
        // could ever trigger. (Rel-16 renamed the field from `useWhiteCellList`
        // in place; the bit position is unchanged.)
        use_allowed_cell_list: EventTriggerConfigEventId_eventA3UseAllowedCellList(false),
    };

    let report_config = ReportConfigNR {
        report_type: ReportConfigNRReportType::EventTriggered(EventTriggerConfig {
            event_id: EventTriggerConfigEventId::EventA3(event_a3),
            rs_type: NR_RS_Type(NR_RS_Type::SSB),
            report_interval: ReportInterval(report_interval_index(params.report_interval_ms)?),
            report_amount: EventTriggerConfigReportAmount(report_amount_index(
                params.report_amount,
            )?),
            // RSRP only: it is the one quantity this simulator measures, and
            // asking for RSRQ or SINR would ask for numbers it has to invent.
            report_quantity_cell: MeasReportQuantity {
                rsrp: MeasReportQuantityRsrp(true),
                rsrq: MeasReportQuantityRsrq(false),
                sinr: MeasReportQuantitySinr(false),
            },
            max_report_cells: EventTriggerConfigMaxReportCells(params.max_report_cells),
            // Per-beam reporting: there are no beams.
            report_quantity_rs_indexes: None,
            max_nrof_rs_indexes_to_report: None,
            include_beam_measurements: EventTriggerConfigIncludeBeamMeasurements(false),
            report_add_neigh_meas: None,
        }),
    };

    Ok(MeasConfig {
        meas_object_to_remove_list: None,
        meas_object_to_add_mod_list: Some(MeasObjectToAddModList(vec![MeasObjectToAddMod {
            meas_object_id: MeasObjectId(params.meas_object_id),
            meas_object: MeasObjectToAddModMeasObject::MeasObjectNR(meas_object),
        }])),
        report_config_to_remove_list: None,
        report_config_to_add_mod_list: Some(ReportConfigToAddModList(vec![ReportConfigToAddMod {
            report_config_id: ReportConfigId(params.report_config_id),
            report_config: ReportConfigToAddModReportConfig::ReportConfigNR(report_config),
        }])),
        meas_id_to_remove_list: None,
        meas_id_to_add_mod_list: Some(MeasIdToAddModList(vec![MeasIdToAddMod {
            meas_id: MeasId(params.meas_id),
            meas_object_id: MeasObjectId(params.meas_object_id),
            report_config_id: ReportConfigId(params.report_config_id),
        }])),
        s_measure_config: None,
        quantity_config: None,
        meas_gap_config: None,
        meas_gap_sharing_config: None,
    })
}

/// `ARFCN-ValueNR ::= INTEGER (0..3279165)` — TS 38.331 §6.3.2.
pub const ARFCN_VALUE_NR_MAX: u32 = 3_279_165;

/// Read every intra-NR A3 reporting binding out of a received `measConfig`.
///
/// One entry per `measIdToAddModList` binding whose `reportConfig` is an
/// `eventA3` on RSRP. A binding whose `measObjectId` or `reportConfigId` names
/// something the same message does not add is **skipped**, not defaulted: a
/// `measId` pointing at an object the UE was never given is a configuration it
/// cannot evaluate, and installing it with invented parameters would have the UE
/// trigger on a margin the network did not choose.
///
/// Returns an empty vector for a `measConfig` that configures no A3 reporting —
/// which is not an error. A `measConfig` may carry only `quantityConfig`, or only
/// removals, or a periodical report.
pub fn read_a3_meas_configs(config: &MeasConfig) -> Vec<A3MeasConfigRead> {
    let Some(bindings) = config.meas_id_to_add_mod_list.as_ref() else {
        return Vec::new();
    };
    bindings
        .0
        .iter()
        .filter_map(|binding| {
            let object_id = binding.meas_object_id.0;
            let report_id = binding.report_config_id.0;

            // The object must be one this message adds, and must be an NR one.
            // Checked even though nothing downstream reads the object's contents:
            // the binding is what says "measure THIS frequency with THIS report
            // configuration", and a dangling half is not a configuration.
            let object_is_nr = config
                .meas_object_to_add_mod_list
                .as_ref()?
                .0
                .iter()
                .any(|o| {
                    o.meas_object_id.0 == object_id
                        && matches!(o.meas_object, MeasObjectToAddModMeasObject::MeasObjectNR(_))
                });
            if !object_is_nr {
                return None;
            }

            let report = config
                .report_config_to_add_mod_list
                .as_ref()?
                .0
                .iter()
                .find(|r| r.report_config_id.0 == report_id)?;
            let ReportConfigToAddModReportConfig::ReportConfigNR(nr) = &report.report_config else {
                return None;
            };
            let ReportConfigNRReportType::EventTriggered(event) = &nr.report_type else {
                return None;
            };
            let EventTriggerConfigEventId::EventA3(a3) = &event.event_id else {
                return None;
            };
            // Only the RSRP arm: the UE measures RSRP. An A3 configured on RSRQ or
            // SINR asks for a quantity this simulator does not produce, so it is
            // reported as unreadable rather than evaluated against RSRP — which
            // would silently compare the wrong quantity.
            let MeasTriggerQuantityOffset::Rsrp(offset) = &a3.a3_offset else {
                return None;
            };

            Some(A3MeasConfigRead {
                meas_id: binding.meas_id.0,
                meas_object_id: object_id,
                report_config_id: report_id,
                a3_offset_db: offset.0,
                hysteresis_half_db: a3.hysteresis.0,
                time_to_trigger_ms: time_to_trigger_ms(a3.time_to_trigger.0)?,
                report_interval_ms: report_interval_ms(event.report_interval.0)?,
                report_amount: report_amount_value(event.report_amount.0),
                max_report_cells: event.max_report_cells.0,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode_rrc, encode_rrc};

    /// The A3 configuration a default gNB signals: 3 dB offset, 1 dB hysteresis,
    /// the UE's own former defaults for the rest.
    fn default_params() -> A3MeasConfigParams {
        A3MeasConfigParams {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            ssb_frequency_arfcn: 632_628,
            ssb_subcarrier_spacing_khz: 30,
            a3_offset_db: 3.0,
            hysteresis_db: 1.0,
            time_to_trigger_ms: 640,
            report_interval_ms: 480,
            report_amount: Some(8),
            max_report_cells: 4,
        }
    }

    // ====================================================================
    // Units. These are the two numbers the whole issue is about, and they are
    // in DIFFERENT units — so they are asserted against the spec's own words
    // rather than against each other.
    // ====================================================================

    /// `a3-Offset` is whole dB: 3 dB is the integer 3, NOT 6.
    #[test]
    fn the_a3_offset_is_signalled_in_whole_db() {
        assert_eq!(a3_offset_signalled(3.0), Ok(3));
        assert_eq!(a3_offset_signalled(-30.0), Ok(-30), "lower bound");
        assert_eq!(a3_offset_signalled(30.0), Ok(30), "upper bound");
        // Off the 1 dB grid: rounds, because there is no fractional dB to signal.
        assert_eq!(a3_offset_signalled(3.4), Ok(3));
        assert_eq!(a3_offset_signalled(3.6), Ok(4));
    }

    /// `hysteresis` is 0.5 dB units: 1 dB is the integer 2, NOT 1.
    #[test]
    fn the_hysteresis_is_signalled_in_half_db_units() {
        assert_eq!(hysteresis_signalled(1.0), Ok(2));
        assert_eq!(hysteresis_signalled(0.0), Ok(0), "lower bound");
        assert_eq!(
            hysteresis_signalled(15.0),
            Ok(30),
            "upper bound, 15 dB = 30"
        );
        assert_eq!(hysteresis_signalled(0.5), Ok(1), "the grid step itself");
        assert_eq!(hysteresis_signalled(1.2), Ok(2), "rounds to the grid");
    }

    /// Past the ends of either INTEGER is refused rather than clamped or wrapped.
    #[test]
    fn an_out_of_range_offset_or_hysteresis_is_refused() {
        assert!(a3_offset_signalled(31.0).is_err());
        assert!(a3_offset_signalled(-31.0).is_err());
        // 15.5 dB is 31 in half-dB units, one past `INTEGER (0..30)`.
        assert!(hysteresis_signalled(15.5).is_err());
        assert!(hysteresis_signalled(-0.5).is_err());
    }

    // ====================================================================
    // BYTE round trip. Load-bearing: #117 found an open-type framing bug that
    // round-tripped to PLAUSIBLE garbage — valid-looking cell IDs and RSRP with
    // the wrong cell count — which both `is_ok()` and a structure-level
    // comparison accepted. Only `encode(decode(x)) == x` on the octets catches
    // a decoder that consumed the wrong number of bits.
    // ====================================================================

    /// `encode(decode(bytes)) == bytes` for the default A3 `measConfig`.
    #[test]
    fn the_a3_meas_config_byte_round_trips() {
        let config = build_a3_meas_config(&default_params()).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let decoded: MeasConfig = decode_rrc(&bytes).expect("decode");
        let reencoded = encode_rrc(&decoded).expect("re-encode");
        assert_eq!(
            reencoded, bytes,
            "the decoder must consume exactly the bits the encoder wrote"
        );
    }

    /// The same, over a spread that hits both ends of every bounded field —
    /// where an off-by-one in a range or a length determinant shows up.
    #[test]
    fn the_a3_meas_config_byte_round_trips_at_the_field_boundaries() {
        let spread = [
            // Every identifier and count at its LOWER bound.
            A3MeasConfigParams {
                meas_id: 1,
                meas_object_id: 1,
                report_config_id: 1,
                ssb_frequency_arfcn: 0,
                ssb_subcarrier_spacing_khz: 15,
                a3_offset_db: -30.0,
                hysteresis_db: 0.0,
                time_to_trigger_ms: 0,
                report_interval_ms: 120,
                report_amount: Some(1),
                max_report_cells: 1,
            },
            // Every identifier and count at its UPPER bound.
            A3MeasConfigParams {
                meas_id: 64,
                meas_object_id: 64,
                report_config_id: 64,
                ssb_frequency_arfcn: ARFCN_VALUE_NR_MAX,
                ssb_subcarrier_spacing_khz: 240,
                a3_offset_db: 30.0,
                hysteresis_db: 15.0,
                time_to_trigger_ms: 5120,
                report_interval_ms: 1_800_000,
                report_amount: Some(64),
                max_report_cells: 8,
            },
            // `reportAmount: infinity` is the last ENUMERATED index, and is the
            // one value that is NOT a count — so it gets its own vector.
            A3MeasConfigParams {
                report_amount: None,
                a3_offset_db: 0.0,
                ..default_params()
            },
            // A negative offset: the sign is where a semi-constrained INTEGER
            // encoded as unsigned would show up.
            A3MeasConfigParams {
                a3_offset_db: -1.0,
                ..default_params()
            },
        ];
        for (index, params) in spread.iter().enumerate() {
            let config = build_a3_meas_config(params).expect("build");
            let bytes = encode_rrc(&config).expect("encode");
            let decoded: MeasConfig = decode_rrc(&bytes).expect("decode");
            assert_eq!(
                encode_rrc(&decoded).expect("re-encode"),
                bytes,
                "vector {index} must byte round trip"
            );
        }
    }

    /// The STRUCTURE round trip, as a separate fault domain: byte identity
    /// cannot see a symmetric encoder/decoder bug that writes and reads the same
    /// wrong field, and structure identity cannot see a framing bug. Both.
    #[test]
    fn the_a3_meas_config_structure_round_trips() {
        let config = build_a3_meas_config(&default_params()).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let decoded: MeasConfig = decode_rrc(&bytes).expect("decode");
        assert_eq!(
            decoded, config,
            "the decoded structure must equal the built one"
        );
    }

    // ====================================================================
    // What the UE reads back. POSITIVE assertions on the two numbers, in the
    // units the UE installs them in.
    // ====================================================================

    /// A gNB's 3 dB / 1 dB reaches the reader as 3 whole dB and 2 half-dB units.
    #[test]
    fn the_configured_margin_survives_the_wire() {
        let config = build_a3_meas_config(&default_params()).expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let decoded: MeasConfig = decode_rrc(&bytes).expect("decode");

        let read = read_a3_meas_configs(&decoded);
        assert_eq!(read.len(), 1, "one A3 binding was signalled");
        assert_eq!(
            read[0],
            A3MeasConfigRead {
                meas_id: 1,
                meas_object_id: 1,
                report_config_id: 1,
                a3_offset_db: 3,
                hysteresis_half_db: 2,
                time_to_trigger_ms: 640,
                report_interval_ms: 480,
                report_amount: Some(8),
                max_report_cells: 4,
            }
        );
    }

    /// A NON-default margin is what actually proves the reader is reading the
    /// wire rather than reproducing a default: 7 dB / 2.5 dB is a value no
    /// default in this tree holds.
    #[test]
    fn a_non_default_margin_reaches_the_reader_unchanged() {
        let config = build_a3_meas_config(&A3MeasConfigParams {
            a3_offset_db: 7.0,
            hysteresis_db: 2.5,
            time_to_trigger_ms: 256,
            report_interval_ms: 1024,
            report_amount: Some(2),
            max_report_cells: 6,
            meas_id: 5,
            meas_object_id: 3,
            report_config_id: 4,
            ..default_params()
        })
        .expect("build");
        let bytes = encode_rrc(&config).expect("encode");
        let decoded: MeasConfig = decode_rrc(&bytes).expect("decode");

        assert_eq!(
            read_a3_meas_configs(&decoded),
            vec![A3MeasConfigRead {
                meas_id: 5,
                meas_object_id: 3,
                report_config_id: 4,
                a3_offset_db: 7,
                hysteresis_half_db: 5,
                time_to_trigger_ms: 256,
                report_interval_ms: 1024,
                report_amount: Some(2),
                max_report_cells: 6,
            }]
        );
    }

    /// A `measId` naming an object the message does not add is skipped, not
    /// installed with invented parameters.
    #[test]
    fn a_binding_whose_object_is_absent_is_not_read() {
        let mut config = build_a3_meas_config(&default_params()).expect("build");
        // Point the binding at object 9 while object 1 is what was added.
        config.meas_id_to_add_mod_list.as_mut().expect("present").0[0].meas_object_id =
            MeasObjectId(9);
        assert!(
            read_a3_meas_configs(&config).is_empty(),
            "a dangling binding must not be installed"
        );
    }

    /// A `measConfig` with no `measIdToAddModList` reads as no A3 configuration,
    /// not as an error: that is a legal message carrying something else.
    #[test]
    fn a_meas_config_with_no_binding_reads_as_no_a3_configuration() {
        let config = MeasConfig {
            meas_object_to_remove_list: None,
            meas_object_to_add_mod_list: None,
            report_config_to_remove_list: None,
            report_config_to_add_mod_list: None,
            meas_id_to_remove_list: None,
            meas_id_to_add_mod_list: None,
            s_measure_config: None,
            quantity_config: None,
            meas_gap_config: None,
            meas_gap_sharing_config: None,
        };
        assert!(read_a3_meas_configs(&config).is_empty());
    }

    // ====================================================================
    // Field-value validation.
    // ====================================================================

    /// Every identifier is `INTEGER (1..64)`, so 0 and 65 are refused. A codec
    /// handed an out-of-range value would write bits outside the constraint.
    #[test]
    fn an_out_of_range_identifier_is_refused() {
        for params in [
            A3MeasConfigParams {
                meas_id: 0,
                ..default_params()
            },
            A3MeasConfigParams {
                meas_id: 65,
                ..default_params()
            },
            A3MeasConfigParams {
                meas_object_id: 0,
                ..default_params()
            },
            A3MeasConfigParams {
                report_config_id: 65,
                ..default_params()
            },
            A3MeasConfigParams {
                max_report_cells: 0,
                ..default_params()
            },
            A3MeasConfigParams {
                max_report_cells: 9,
                ..default_params()
            },
            A3MeasConfigParams {
                ssb_frequency_arfcn: ARFCN_VALUE_NR_MAX + 1,
                ..default_params()
            },
        ] {
            assert!(
                build_a3_meas_config(&params).is_err(),
                "out-of-range field must be refused: {params:?}"
            );
        }
    }

    /// A `timeToTrigger`, `reportInterval`, `reportAmount` or subcarrier spacing
    /// that is not a value of its ENUMERATED is refused rather than rounded to a
    /// neighbouring one — a UE told 500 ms would run on whatever the nearest
    /// value happened to be.
    #[test]
    fn a_value_outside_an_enumerated_is_refused() {
        for params in [
            A3MeasConfigParams {
                time_to_trigger_ms: 500,
                ..default_params()
            },
            A3MeasConfigParams {
                report_interval_ms: 500,
                ..default_params()
            },
            A3MeasConfigParams {
                report_amount: Some(3),
                ..default_params()
            },
            A3MeasConfigParams {
                ssb_subcarrier_spacing_khz: 480,
                ..default_params()
            },
        ] {
            assert!(
                build_a3_meas_config(&params).is_err(),
                "a non-ENUMERATED value must be refused: {params:?}"
            );
        }
    }

    /// Every `timeToTrigger` and `reportInterval` value maps out and back.
    /// Without this, one wrong arm in a sixteen-way match would only be caught
    /// by a test that happened to use that value.
    #[test]
    fn every_enumerated_value_maps_out_and_back() {
        for ms in [
            0u16, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120,
        ] {
            let index = time_to_trigger_index(ms).expect("a legal timeToTrigger");
            assert_eq!(time_to_trigger_ms(index), Some(ms), "timeToTrigger {ms} ms");
        }
        for ms in [
            120u32, 240, 480, 640, 1024, 2048, 5120, 10240, 20480, 40960, 60_000, 360_000, 720_000,
            1_800_000,
        ] {
            let index = report_interval_index(ms).expect("a legal reportInterval");
            assert_eq!(
                report_interval_ms(index),
                Some(ms),
                "reportInterval {ms} ms"
            );
        }
        for amount in [
            None,
            Some(1),
            Some(2),
            Some(4),
            Some(8),
            Some(16),
            Some(32),
            Some(64),
        ] {
            let index = report_amount_index(amount).expect("a legal reportAmount");
            assert_eq!(
                report_amount_value(index),
                amount,
                "reportAmount {amount:?}"
            );
        }
    }

    /// The `measObject` arm is the ROOT `MeasObjectNR`, and the `reportConfig`
    /// arm the ROOT `ReportConfigNR` — the fact that made #117 a non-blocker.
    /// Asserted so a future change that reached for an extension arm would have
    /// to say so here.
    #[test]
    fn the_built_config_selects_only_root_choice_arms() {
        let config = build_a3_meas_config(&default_params()).expect("build");
        assert!(matches!(
            config
                .meas_object_to_add_mod_list
                .as_ref()
                .expect("present")
                .0[0]
                .meas_object,
            MeasObjectToAddModMeasObject::MeasObjectNR(_)
        ));
        let report = &config
            .report_config_to_add_mod_list
            .as_ref()
            .expect("present")
            .0[0]
            .report_config;
        let ReportConfigToAddModReportConfig::ReportConfigNR(nr) = report else {
            panic!("the root ReportConfigNR arm");
        };
        assert!(matches!(
            nr.report_type,
            ReportConfigNRReportType::EventTriggered(_)
        ));
    }
}
