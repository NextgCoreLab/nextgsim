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
pub(crate) fn distance_m(a: &SimCoord, b: &SimCoord) -> f64 {
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

/// Seeded, spatially consistent log-normal shadowing on top of a channel model
/// (nextgsim#467). **Non-normative**: a simulator model, not a 3GPP deliverable.
///
/// Shadowing is the slow, location-dependent fading that buildings and terrain
/// put on a link: a UE standing at the same spot hears the same transmitter at
/// the same level every time, and two nearby spots are correlated. So the offset
/// is a deterministic function of `(seed, link_key, position)`, never a fresh
/// draw per heartbeat:
///
/// - a lattice with spacing `decorrelation_m` holds one seeded N(0, 1) draw per
///   lattice point, keyed by the seed and the transmitter (`link_key`, the gNB's
///   NCI, which is stable across starts);
/// - a position's value is the trilinear interpolation of its eight surrounding
///   lattice points, divided by `sqrt(Σw²)` so that every position, on or between
///   lattice points, is exactly N(0, 1);
/// - the offset is that value times `sigma_db`.
///
/// The UE is deliberately not part of the key: shadowing belongs to the location
/// and the transmitter, and a UE's RLS identity is random per start, so keying on
/// it would make a seeded run irreproducible.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shadowing {
    sigma_db: f64,
    seed: u64,
    decorrelation_m: f64,
    link_key: u64,
}

impl Shadowing {
    /// Shadowing of standard deviation `sigma_db` with lattice spacing
    /// `decorrelation_m`, for the transmitter identified by `link_key`.
    ///
    /// The caller validates the parameters (finite, `sigma_db >= 0`,
    /// `decorrelation_m > 0`); a zero `sigma_db` gives a zero offset everywhere.
    pub fn new(sigma_db: f64, seed: u64, decorrelation_m: f64, link_key: u64) -> Self {
        Self {
            sigma_db,
            seed,
            decorrelation_m,
            link_key,
        }
    }

    /// The shadowing offset in dB at `pos`.
    pub fn offset_db(&self, pos: &SimCoord) -> f64 {
        if self.sigma_db == 0.0 {
            return 0.0;
        }
        let d = self.decorrelation_m;
        let axis = |v: i32| {
            let f = f64::from(v) / d;
            let base = f.floor();
            (base as i64, f - base)
        };
        let (ix, ux) = axis(pos.x);
        let (iy, uy) = axis(pos.y);
        let (iz, uz) = axis(pos.z);
        let (mut acc, mut weight_sq) = (0.0, 0.0);
        for corner in 0..8u8 {
            let pick = |bit: u8, u: f64| {
                if corner & bit == 0 {
                    (0, 1.0 - u)
                } else {
                    (1, u)
                }
            };
            let (dx, wx) = pick(1, ux);
            let (dy, wy) = pick(2, uy);
            let (dz, wz) = pick(4, uz);
            let w = wx * wy * wz;
            if w == 0.0 {
                continue;
            }
            acc += w * self.lattice_normal(ix + dx, iy + dy, iz + dz);
            weight_sq += w * w;
        }
        // The eight weights sum to 1, so Σw² >= 1/8 and never divides by zero.
        self.sigma_db * acc / weight_sq.sqrt()
    }

    /// The seeded N(0, 1) draw at one lattice point (Box-Muller on two uniforms
    /// hashed from the key).
    fn lattice_normal(&self, i: i64, j: i64, k: i64) -> f64 {
        let mut h = splitmix64(self.seed);
        for part in [self.link_key, i as u64, j as u64, k as u64] {
            h = splitmix64(h ^ part);
        }
        // 53-bit mantissas: u1 in (0, 1] so ln(u1) is finite, u2 in [0, 1).
        const SCALE: f64 = 1.0 / (1u64 << 53) as f64;
        let u1 = ((h >> 11) as f64 + 1.0) * SCALE;
        let u2 = (splitmix64(h) >> 11) as f64 * SCALE;
        (-2.0 * u1.ln()).sqrt() * (core::f64::consts::TAU * u2).cos()
    }
}

/// SplitMix64's finaliser: a fast, well-mixed, dependency-free 64-bit hash.
fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A channel model with [`Shadowing`] added to its level (nextgsim#467).
///
/// The offset is added once, here, so every consumer of the gNB's received
/// level (the UE's cell search and A3 evaluation via the heartbeat ack, and the
/// gNB's own NWDAF feed) sees the same shadowed value.
#[derive(Debug)]
pub struct ShadowedModel {
    inner: Box<dyn ChannelModel>,
    shadowing: Shadowing,
}

impl ShadowedModel {
    /// `inner`'s level plus `shadowing`'s offset at the UE's position.
    pub fn new(inner: Box<dyn ChannelModel>, shadowing: Shadowing) -> Self {
        Self { inner, shadowing }
    }
}

impl ChannelModel for ShadowedModel {
    fn estimate_dbm(&self, gnb_pos: &SimCoord, ue_pos: &SimCoord) -> i32 {
        let level = f64::from(self.inner.estimate_dbm(gnb_pos, ue_pos));
        (level + self.shadowing.offset_db(ue_pos)).round() as i32
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

    // ---- Shadowing (nextgsim#467) ----

    fn shadow(sigma: f64, seed: u64, nci: u64) -> Shadowing {
        Shadowing::new(sigma, seed, 20.0, nci)
    }

    /// The ticket's unit criterion: the same (gNB, position, seed) gives the same
    /// level on every call and from every instance, so the A3 and NWDAF arms of
    /// one seeded run, and two runs of the same seed, see the same levels.
    #[test]
    fn shadowing_is_identical_across_calls_and_instances() {
        let a = shadow(6.0, 42, 0x10);
        let b = shadow(6.0, 42, 0x10);
        for x in (-50..200).step_by(7) {
            let p = at(x, 13, 0);
            let first = a.offset_db(&p);
            assert_eq!(
                first.to_bits(),
                a.offset_db(&p).to_bits(),
                "same instance, x={x}"
            );
            assert_eq!(
                first.to_bits(),
                b.offset_db(&p).to_bits(),
                "new instance, x={x}"
            );
        }
        let m1 = ShadowedModel::new(Box::new(DistanceModel), a);
        let m2 = ShadowedModel::new(Box::new(DistanceModel), b);
        for x in 0..120 {
            let p = at(x, 0, 0);
            assert_eq!(
                m1.estimate_dbm(&at(0, 0, 0), &p),
                m2.estimate_dbm(&at(0, 0, 0), &p)
            );
        }
    }

    /// A different seed, or a different gNB, is a different field.
    #[test]
    fn the_seed_and_the_gnb_select_the_field() {
        let points: Vec<SimCoord> = (0..40).map(|i| at(i * 10, 5, 0)).collect();
        let field = |s: &Shadowing| points.iter().map(|p| s.offset_db(p)).collect::<Vec<_>>();
        let base = field(&shadow(6.0, 1, 0x10));
        assert_ne!(base, field(&shadow(6.0, 2, 0x10)), "another seed");
        assert_ne!(base, field(&shadow(6.0, 1, 0x20)), "another gNB");
    }

    /// σ 0 is off: the shadowed model reports exactly what the inner one does.
    #[test]
    fn zero_sigma_is_byte_identical_to_the_inner_model() {
        let model = ShadowedModel::new(Box::new(DistanceModel), shadow(0.0, 7, 0x10));
        for x in [0, 1, 31, 62, 85, 120, 500] {
            let (g, u) = (at(0, 0, 0), at(x, 3, 0));
            assert_eq!(
                model.estimate_dbm(&g, &u),
                DistanceModel.estimate_dbm(&g, &u)
            );
        }
    }

    /// On and off the lattice, the offset is N(0, σ²): sample mean and standard
    /// deviation over many independent lattice cells, at an off-lattice point of
    /// each, are within a few standard errors of 0 and σ.
    #[test]
    fn the_marginal_is_zero_mean_with_the_configured_sigma() {
        let s = shadow(5.0, 9, 0x10);
        // Every 3rd lattice cell (60 m apart, independent), at the cell's middle
        // (where interpolation shrinks variance most before renormalisation).
        let samples: Vec<f64> = (0..4000)
            .map(|i| s.offset_db(&at((i % 80) * 60 + 10, (i / 80) * 60 + 10, 0)))
            .collect();
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let sd = (samples.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        assert!(mean.abs() < 0.3, "mean {mean} dB, expected ~0");
        assert!((sd - 5.0).abs() < 0.3, "sd {sd} dB, expected ~5");
    }

    /// Spatially consistent: a metre apart the offsets nearly agree; points many
    /// decorrelation distances apart are uncorrelated.
    #[test]
    fn nearby_points_correlate_and_far_points_do_not() {
        let s = shadow(6.0, 3, 0x10);
        let corr = |gap: i32| {
            let pairs: Vec<(f64, f64)> = (0..3000)
                .map(|i| {
                    let p = at((i % 60) * 97, (i / 60) * 89, 0);
                    let q = at(p.x + gap, p.y, 0);
                    (s.offset_db(&p), s.offset_db(&q))
                })
                .collect();
            let n = pairs.len() as f64;
            let (ma, mb) = (
                pairs.iter().map(|p| p.0).sum::<f64>() / n,
                pairs.iter().map(|p| p.1).sum::<f64>() / n,
            );
            let cov = pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>();
            let va = pairs.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>();
            let vb = pairs.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>();
            cov / (va * vb).sqrt()
        };
        assert!(corr(1) > 0.9, "1 m apart: {}", corr(1));
        assert!(corr(200).abs() < 0.1, "200 m apart: {}", corr(200));
    }

    /// The offset rides on top of the inner level and is rounded once.
    #[test]
    fn the_shadowed_level_is_the_inner_level_plus_the_rounded_offset() {
        let s = shadow(8.0, 11, 0x10);
        let model = ShadowedModel::new(Box::new(DistanceModel), s);
        let mut moved = 0;
        for x in 31..120 {
            let u = at(x, 0, 0);
            let expected = (f64::from(-x) + s.offset_db(&u)).round() as i32;
            let got = model.estimate_dbm(&at(0, 0, 0), &u);
            assert_eq!(got, expected, "x={x}");
            moved += i32::from(got != -x);
        }
        assert!(moved > 40, "σ 8 dB must move most levels, moved {moved}");
    }
}
