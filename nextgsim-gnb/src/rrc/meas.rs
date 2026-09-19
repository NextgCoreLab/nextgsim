//! The gNB's A3 measurement configuration, derived from its own configuration
//! (TS 38.331 §5.5.2; issue #170).
//!
//! # What this closes
//!
//! `GnbConfig::cho_a3_offset_db` / `cho_hysteresis_db` used to reach the UE only
//! through the *conditional-reconfiguration* container, which governs handover
//! **execution**. A3 *reporting* — the trigger that makes the UE send a
//! `MeasurementReport` at all, which the gNB then acts on — ran off a hard-coded
//! UE-local default, because `RRCReconfiguration.measConfig` was hardcoded `None`.
//! So the gNB decided handovers on one margin and the UE decided whether to report
//! on another, with no wire path between them.
//!
//! This is that wire path. The two fields are now the gNB's **single** A3
//! configuration, which is what #170 asked a decision about: they name one margin
//! that governs both reporting and CHO execution, rather than two that happen to
//! share a default.

use nextgsim_common::config::GnbConfig;
use nextgsim_rrc::procedures::meas_config::{
    a3_offset_signalled, hysteresis_signalled, A3MeasConfigParams,
};
use tracing::warn;

/// `measId` of the A3 reporting measurement the gNB configures.
///
/// 1, deliberately: it is the `measId` the UE's own pre-signalling default used, so
/// a signalled configuration **replaces** that default rather than sitting beside
/// it (TS 38.331 §5.5.2.5: a `measIdToAddMod` for an existing `measId` modifies
/// it). Installing under a different `measId` would leave the UE evaluating both,
/// and the looser of the two would decide when it reported.
pub const A3_MEAS_ID: u8 = 1;

/// `measObjectId` of the NR measurement object.
pub const A3_MEAS_OBJECT_ID: u8 = 1;

/// `reportConfigId` of the A3 report configuration.
pub const A3_REPORT_CONFIG_ID: u8 = 1;

/// The `ssbSubcarrierSpacing` this gNB reports for its own carrier, in kHz.
///
/// 30 kHz FR1, the same value the MIB's physical-layer fields and the CHO
/// container's `ssb_subcarrier_spacing_khz` assume. Not derived from anything: this
/// simulator has no PHY, so nothing computes a numerology.
pub const A3_SSB_SCS_KHZ: u16 = 30;

/// `timeToTrigger` for A3 **reporting**, in ms.
///
/// 640 ms — the value the UE's former hard-coded default used, kept so that
/// signalling the configuration changes *which end chose it* without changing how
/// long a condition must hold.
///
/// Deliberately NOT the `Ms0` the CHO container carries. TS 38.331 configures
/// reporting (`measConfig`) and execution (`condReconfigToAddModList`) separately
/// and allows them to differ, and they differ here for a reason: a CHO candidate
/// executes on the UE's own timer, which #114's runtime does not have, so `Ms0` is
/// the only value it can honour. A report has no such constraint.
pub const A3_TIME_TO_TRIGGER_MS: u16 = 640;

/// `reportInterval` in ms, and `reportAmount` — the UE's former defaults.
pub const A3_REPORT_INTERVAL_MS: u32 = 480;
/// How many reports one trigger produces.
const A3_REPORT_AMOUNT: u32 = 8;
/// `maxReportCells`.
const A3_MAX_REPORT_CELLS: u8 = 4;

/// The A3 reporting configuration this gNB signals, or `None` when its configured
/// margin is not a value TS 38.331 can carry.
///
/// `None` rather than an error, and checked HERE rather than at the encoder, so a
/// mis-configured margin costs the UE its measurement configuration and nothing
/// else. Failing the whole `RRCReconfiguration` would take down whatever else it
/// carried — and the reconfiguration this rides is the one that establishes the
/// user plane.
pub fn a3_meas_config_params(config: &GnbConfig) -> Option<A3MeasConfigParams> {
    // Validated through the same two functions the builder uses, so "would this
    // encode?" is answered by the code that encodes it rather than by a second
    // copy of the bounds.
    if let Err(e) = a3_offset_signalled(config.cho_a3_offset_db) {
        warn!(
            "Not configuring UE measurements: a3Offset {} dB is not signallable ({e})",
            config.cho_a3_offset_db
        );
        return None;
    }
    if let Err(e) = hysteresis_signalled(config.cho_hysteresis_db) {
        warn!(
            "Not configuring UE measurements: hysteresis {} dB is not signallable ({e})",
            config.cho_hysteresis_db
        );
        return None;
    }

    Some(A3MeasConfigParams {
        meas_id: A3_MEAS_ID,
        meas_object_id: A3_MEAS_OBJECT_ID,
        report_config_id: A3_REPORT_CONFIG_ID,
        ssb_frequency_arfcn: config.dl_arfcn,
        ssb_subcarrier_spacing_khz: A3_SSB_SCS_KHZ,
        a3_offset_db: config.cho_a3_offset_db,
        hysteresis_db: config.cho_hysteresis_db,
        time_to_trigger_ms: A3_TIME_TO_TRIGGER_MS,
        report_interval_ms: A3_REPORT_INTERVAL_MS,
        report_amount: Some(A3_REPORT_AMOUNT),
        max_report_cells: A3_MAX_REPORT_CELLS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_rrc::procedures::meas_config::build_a3_meas_config;

    /// A default gNB's margin becomes a signallable configuration, and carries the
    /// configured numbers rather than a constant of this module's own.
    #[test]
    fn the_configured_margin_becomes_the_signalled_one() {
        let config = GnbConfig {
            cho_a3_offset_db: 7.0,
            cho_hysteresis_db: 2.5,
            dl_arfcn: 620_000,
            ..GnbConfig::default()
        };

        let params = a3_meas_config_params(&config).expect("7 dB / 2.5 dB is signallable");
        assert_eq!(params.a3_offset_db, 7.0);
        assert_eq!(params.hysteresis_db, 2.5);
        assert_eq!(
            params.ssb_frequency_arfcn, 620_000,
            "the measured carrier is the cell's own"
        );
        assert_eq!(
            params.meas_id, 1,
            "measId 1 REPLACES the UE's pre-signalling default rather than \
             sitting beside it"
        );
        // And it is a configuration the codec accepts, which is what makes the
        // `None` arm below the only way it can fail.
        assert!(build_a3_meas_config(&params).is_ok());
    }

    /// A margin TS 38.331 cannot carry yields `None` — so the reconfiguration it
    /// would have ridden is still sent, without a measConfig.
    #[test]
    fn an_unsignallable_margin_yields_no_configuration() {
        assert!(
            a3_meas_config_params(&GnbConfig {
                cho_a3_offset_db: 99.0,
                ..GnbConfig::default()
            })
            .is_none(),
            "a3Offset 99 dB is outside INTEGER (-30..30)"
        );

        assert!(a3_meas_config_params(&GnbConfig {
            // 15.5 dB is 31 in 0.5 dB units, one past
            // `Hysteresis ::= INTEGER (0..30)`.
            cho_hysteresis_db: 15.5,
            ..GnbConfig::default()
        })
        .is_none());
    }

    /// The default configuration IS signallable. Without this, every UE would
    /// silently get no measurement configuration out of the box and the tests
    /// above would still pass.
    #[test]
    fn the_default_configuration_is_signallable() {
        let params =
            a3_meas_config_params(&GnbConfig::default()).expect("the default must be signallable");
        assert_eq!(params.a3_offset_db, 3.0, "the documented default");
        assert_eq!(params.hysteresis_db, 1.0);
        assert!(build_a3_meas_config(&params).is_ok());
    }
}
