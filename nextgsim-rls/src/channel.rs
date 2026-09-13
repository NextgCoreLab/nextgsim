//! Pluggable channel models for the simulated radio link.
//!
//! The gNB cell tracker turns a UE's position into a received level in dBm. That
//! computation used to be a hardcoded negated distance inside the tracker, with no
//! seam to put a physically grounded model behind. This module is that seam.
//!
//! # What ships here
//!
//! - [`DistanceModel`] — the default, and byte-for-byte what the tracker did
//!   before: received dBm is the negated Euclidean distance. It is not physics; it
//!   is a monotone stand-in that keeps the existing end-to-end behaviour
//!   unchanged.
//! - [`FreeSpaceModel`] — Friis free-space path loss, i.e. actual physics for an
//!   unobstructed line of sight, opt-in via
//!   [`GnbCellTracker::with_channel_model`](crate::GnbCellTracker::with_channel_model).
//!
//! `FreeSpaceModel` is **compiled always and selected never** by default. It is not
//! behind a cargo feature deliberately: CI runs `cargo test --workspace` with
//! default features, so a feature-gated model would ship without ever being
//! compiled by the gate meant to cover it — the same reasoning recorded for the
//! runtime switches in this tree (GTP-U path supervision, URSP evaluation, the RLC
//! AM bearer list).
//!
//! # Intended future implementations
//!
//! TR 38.901 (0.5-100 GHz stochastic channel), sub-THz, RIS-assisted and NTN
//! models all belong behind this trait. The last three are **non-normative**: no
//! frozen Rel-20 stage-3 specification exists, so they would be research models
//! rather than conformance targets. `nextgsim-rrc`'s `SubThzChannelModel` (an RRC
//! configuration payload carrying a path-loss exponent) is a plausible future
//! source of parameters for such an implementation; it does no channel maths today.

use crate::protocol::SimCoord;

/// A model that turns gNB and UE positions into a received level.
///
/// `Send + Sync` because the tracker holding it lives in an async task; `Debug`
/// because the tracker derives it.
pub trait ChannelModel: core::fmt::Debug + Send + Sync {
    /// Received signal strength in dBm at `ue_pos`, for a transmitter at
    /// `gnb_pos`.
    fn estimate_dbm(&self, gnb_pos: &SimCoord, ue_pos: &SimCoord) -> i32;
}

/// Euclidean distance between two simulated positions, in metres.
fn distance_m(a: &SimCoord, b: &SimCoord) -> f64 {
    let dx = i64::from(a.x - b.x);
    let dy = i64::from(a.y - b.y);
    let dz = i64::from(a.z - b.z);
    ((dx * dx + dy * dy + dz * dz) as f64).sqrt()
}

/// The default model: received dBm is the negated distance in metres.
///
/// Not a path-loss model — 10 m away reads as -10 dBm, which no radio does. It is
/// kept as the default because it is what every existing scenario, test and
/// end-to-end run was tuned against, and because it is monotone in distance, which
/// is the only property cell selection actually uses.
///
/// Zero distance reports -1 rather than 0, as it always has: a 0 dBm reading is
/// easy to mistake for "no measurement".
#[derive(Debug, Clone, Copy, Default)]
pub struct DistanceModel;

impl ChannelModel for DistanceModel {
    fn estimate_dbm(&self, gnb_pos: &SimCoord, ue_pos: &SimCoord) -> i32 {
        let distance = distance_m(gnb_pos, ue_pos) as i32;
        if distance == 0 {
            -1 // 0 may be confusing
        } else {
            -distance
        }
    }
}

/// Friis free-space path loss (classical, not a 3GPP deliverable):
///
/// ```text
/// FSPL(dB) = 20·log10(d_m) + 20·log10(f_Hz) - 147.55
/// received_dBm = tx_power_dBm - FSPL(dB)
/// ```
///
/// The -147.55 dB constant is `20·log10(4π/c)` with `c = 299 792 458 m/s`.
///
/// Valid only in the far field with an unobstructed line of sight: it models no
/// shadowing, no fading, no ground reflection and no atmospheric absorption, so it
/// is optimistic everywhere else. That is why it is opt-in rather than the default
/// — swapping it in changes every level in a scenario, and the thresholds those
/// scenarios use (`MIN_ALLOWED_DBM`, the UE's `q-RxLevMin`) were chosen against the
/// distance model.
#[derive(Debug, Clone, Copy)]
pub struct FreeSpaceModel {
    /// Carrier frequency in Hz
    frequency_hz: f64,
    /// Transmit power (EIRP) in dBm
    tx_power_dbm: f64,
}

/// `20·log10(4π/c)` in dB, the Friis constant for metres and hertz.
const FSPL_CONSTANT_DB: f64 = -147.55;

impl FreeSpaceModel {
    /// A model for a carrier at `frequency_hz` transmitting `tx_power_dbm`.
    ///
    /// A typical macro cell in n78 is `FreeSpaceModel::new(3.5e9, 46.0)`.
    pub fn new(frequency_hz: f64, tx_power_dbm: f64) -> Self {
        Self {
            frequency_hz,
            tx_power_dbm,
        }
    }

    /// Free-space path loss in dB at `distance_m`.
    ///
    /// Distances below one metre are treated as one metre: Friis diverges at zero,
    /// and a co-located UE and gNB is a modelling artefact rather than a link whose
    /// loss is meaningful.
    fn path_loss_db(&self, distance_m: f64) -> f64 {
        let distance = distance_m.max(1.0);
        20.0 * distance.log10() + 20.0 * self.frequency_hz.log10() + FSPL_CONSTANT_DB
    }
}

impl ChannelModel for FreeSpaceModel {
    fn estimate_dbm(&self, gnb_pos: &SimCoord, ue_pos: &SimCoord) -> i32 {
        let received = self.tx_power_dbm - self.path_loss_db(distance_m(gnb_pos, ue_pos));
        // Round rather than truncate: truncation biases every level upward by up
        // to 1 dB, and toward zero on both sides of it.
        received.round() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(x: i32, y: i32, z: i32) -> SimCoord {
        SimCoord::new(x, y, z)
    }

    #[test]
    fn the_distance_model_reports_the_negated_distance() {
        let model = DistanceModel;
        assert_eq!(model.estimate_dbm(&at(0, 0, 0), &at(10, 0, 0)), -10);
        assert_eq!(model.estimate_dbm(&at(0, 0, 0), &at(0, 30, 40)), -50);
        assert_eq!(
            model.estimate_dbm(&at(5, 5, 5), &at(5, 5, 5)),
            -1,
            "a co-located UE reports -1, not 0"
        );
    }

    /// Friis at 3.5 GHz: 100 m of free space is about 83 dB of loss, so a 46 dBm
    /// transmitter is heard at roughly -37 dBm. Checked against the closed form
    /// rather than against the implementation's own output.
    #[test]
    fn the_free_space_model_matches_the_friis_closed_form() {
        let model = FreeSpaceModel::new(3.5e9, 46.0);
        let expected_loss = 20.0 * 100.0_f64.log10() + 20.0 * 3.5e9_f64.log10() - 147.55;
        let expected = (46.0 - expected_loss).round() as i32;

        assert_eq!(model.estimate_dbm(&at(0, 0, 0), &at(100, 0, 0)), expected);
        assert_eq!(expected, -37, "sanity: 3.5 GHz, 100 m, 46 dBm EIRP");
    }

    /// The level is rounded, not truncated. 150 m at 3.5 GHz lands on -40.85 dBm,
    /// which rounds to -41 but truncates toward zero to -40 — one of the geometries
    /// where the two differ, unlike the 100 m case above.
    #[test]
    fn the_free_space_level_is_rounded_not_truncated() {
        let model = FreeSpaceModel::new(3.5e9, 46.0);
        let exact = 46.0 - (20.0 * 150.0_f64.log10() + 20.0 * 3.5e9_f64.log10() - 147.55);

        assert!(
            (exact - -40.853).abs() < 0.01,
            "sanity: the exact level is -40.85 dBm, got {exact}"
        );
        assert_eq!(
            model.estimate_dbm(&at(0, 0, 0), &at(150, 0, 0)),
            -41,
            "truncation would report -40, biasing the level upward"
        );
    }

    #[test]
    fn free_space_loss_grows_monotonically_with_distance() {
        let model = FreeSpaceModel::new(3.5e9, 46.0);
        let mut previous = i32::MAX;
        for distance in [1, 2, 5, 10, 50, 100, 500, 1000, 5000] {
            let level = model.estimate_dbm(&at(0, 0, 0), &at(distance, 0, 0));
            assert!(
                level < previous,
                "level at {distance} m ({level} dBm) must be below the previous one ({previous} dBm)"
            );
            previous = level;
        }
    }

    /// Doubling the frequency costs 6 dB (20·log10(2)), which is the property that
    /// distinguishes a real path-loss model from a distance stand-in.
    #[test]
    fn doubling_the_carrier_frequency_costs_about_six_db() {
        let low = FreeSpaceModel::new(3.5e9, 46.0).estimate_dbm(&at(0, 0, 0), &at(200, 0, 0));
        let high = FreeSpaceModel::new(7.0e9, 46.0).estimate_dbm(&at(0, 0, 0), &at(200, 0, 0));
        assert_eq!(low - high, 6);
    }

    #[test]
    fn the_two_models_disagree_for_the_same_geometry() {
        let geometry = (at(0, 0, 0), at(100, 0, 0));
        let distance = DistanceModel.estimate_dbm(&geometry.0, &geometry.1);
        let free_space = FreeSpaceModel::new(3.5e9, 46.0).estimate_dbm(&geometry.0, &geometry.1);
        assert_eq!(distance, -100);
        assert_ne!(
            distance, free_space,
            "the point of the seam is that a different model gives a different level"
        );
    }

    /// A co-located UE must not produce an infinite or NaN level: Friis diverges at
    /// zero distance, so the model floors the distance at one metre.
    #[test]
    fn free_space_at_zero_distance_is_finite() {
        let model = FreeSpaceModel::new(3.5e9, 46.0);
        let level = model.estimate_dbm(&at(7, 7, 7), &at(7, 7, 7));
        assert!(
            level > -200 && level < 200,
            "a co-located UE must give a finite level, got {level}"
        );
        assert_eq!(
            level,
            model.estimate_dbm(&at(0, 0, 0), &at(1, 0, 0)),
            "and it equals the one-metre level, which is what the floor means"
        );
    }
}
