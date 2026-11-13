//! Constant-turn trajectory extrapolation over a Kalman-filtered track
//! (nextgsim#462, the D440A `ct` baseline): a per-axis constant-velocity
//! Kalman filter, a least-squares turn-rate fit on the filtered heading, and a
//! constant-turn arc extrapolation. Not a CT-state filter (the turn rate is
//! not in the filter's state).
//!
//! The physics baseline a learned trajectory model is measured against: a UE
//! is assumed to keep its current speed and its current rate of turn.
//!
//! 1. A constant-velocity Kalman filter, one per axis, runs over the most recent
//!    [`CT_WINDOW`] samples. The measurement noise is [`CT_MEASUREMENT_SIGMA_M`]
//!    (the RLS heartbeat carries whole metres) and the process noise is a white
//!    acceleration of spectral density [`CT_ACCEL_PSD`].
//! 2. The turn rate is the least-squares slope of the filtered heading over
//!    time, using only the steps where the filtered speed is at least
//!    [`CT_MIN_SPEED_MPS`] (a heading is noise below that), clamped to
//!    [`CT_MAX_TURN_RATE_RAD_S`].
//! 3. The track is extrapolated along the circular arc that speed, heading and
//!    turn rate define (a straight line when the turn rate is zero). The height
//!    is extrapolated linearly from its own filter.
//!
//! The confidence is the R² of that same arc hindcast over the window's
//! samples, scaled for short histories as the least-squares fit's is
//! ([`crate::predictor::least_squares_trajectory`]): a UE on a steady arc
//! scores near 1, one that changes its turn or speed scores lower.
//!
//! Every constant here was fixed before the first D440A replay ran, from the
//! simulator's sampling (1 s heartbeats, whole metres), not tuned on results.

use crate::Vector3;

/// The most recent samples the filter runs over: 10 s of 1 s heartbeats, the
/// same window the least-squares fit uses.
pub const CT_WINDOW: usize = 10;

/// The position measurement's standard deviation in metres. The heartbeat
/// rounds to whole metres (uniform error, sigma about 0.29 m); 0.5 m leaves
/// room for the UE's own motion between its position and its heartbeat.
pub const CT_MEASUREMENT_SIGMA_M: f64 = 0.5;

/// The white-acceleration process noise's spectral density, (m/s²)²·s. About
/// a 0.7 m/s² standard deviation per second: a pedestrian or a vehicle
/// changing speed gently.
pub const CT_ACCEL_PSD: f64 = 0.5;

/// Below this filtered speed a heading is not used to estimate the turn rate.
pub const CT_MIN_SPEED_MPS: f64 = 0.5;

/// The largest turn rate the extrapolation uses, in rad/s (about 29 °/s).
pub const CT_MAX_TURN_RATE_RAD_S: f64 = 0.5;

/// Samples for full confidence, as for the least-squares fit.
const CT_FULL_CONFIDENCE_SAMPLES: usize = 5;

/// A constant-turn prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantTurnPrediction {
    /// Ten waypoints evenly spaced over the horizon, `(position, timestamp_ms)`;
    /// the last is the position at `current_time_ms + horizon_ms`.
    pub waypoints: Vec<(Vector3, u64)>,
    /// The filtered horizontal speed at the newest sample, m/s.
    pub speed_mps: f64,
    /// The filtered heading at the newest sample, radians from +x towards +y.
    pub heading_rad: f64,
    /// The estimated turn rate, rad/s (positive: counter-clockwise).
    pub turn_rate_rad_s: f64,
    /// The arc's hindcast R², scaled for short histories, in `0..=1`.
    pub confidence: f32,
}

/// One axis of a constant-velocity Kalman filter: state `(position, velocity)`
/// and its 2×2 covariance.
#[derive(Debug, Clone, Copy)]
struct AxisFilter {
    x: [f64; 2],
    p: [[f64; 2]; 2],
}

impl AxisFilter {
    fn new(position: f64) -> Self {
        let r = CT_MEASUREMENT_SIGMA_M * CT_MEASUREMENT_SIGMA_M;
        // The velocity starts unknown: a 10 m/s standard deviation.
        Self {
            x: [position, 0.0],
            p: [[r, 0.0], [0.0, 100.0]],
        }
    }

    /// Predict `dt` seconds ahead, then update with the measured position.
    fn step(&mut self, dt: f64, z: f64) {
        let [pos, vel] = self.x;
        let x = [pos + vel * dt, vel];
        let [[p00, p01], [p10, p11]] = self.p;
        // F P F^T + Q, with F = [[1, dt], [0, 1]] and the white-acceleration Q.
        let q = CT_ACCEL_PSD;
        let a00 = p00 + dt * (p01 + p10) + dt * dt * p11 + q * dt.powi(3) / 3.0;
        let a01 = p01 + dt * p11 + q * dt * dt / 2.0;
        let a10 = p10 + dt * p11 + q * dt * dt / 2.0;
        let a11 = p11 + q * dt;
        // Update with H = [1, 0].
        let s = a00 + CT_MEASUREMENT_SIGMA_M * CT_MEASUREMENT_SIGMA_M;
        let (k0, k1) = (a00 / s, a10 / s);
        let innovation = z - x[0];
        self.x = [x[0] + k0 * innovation, x[1] + k1 * innovation];
        self.p = [
            [(1.0 - k0) * a00, (1.0 - k0) * a01],
            [a10 - k1 * a00, a11 - k1 * a01],
        ];
    }
}

/// The offset after `dt` seconds along an arc of speed `v`, heading `theta`
/// and turn rate `omega` (a straight line for a zero turn rate).
fn arc_offset(v: f64, theta: f64, omega: f64, dt: f64) -> (f64, f64) {
    if omega.abs() < 1e-9 {
        (v * dt * theta.cos(), v * dt * theta.sin())
    } else {
        let r = v / omega;
        (
            r * ((theta + omega * dt).sin() - theta.sin()),
            -r * ((theta + omega * dt).cos() - theta.cos()),
        )
    }
}

/// Constant-turn extrapolation of `positions` (timestamps in ms, one per
/// position) `horizon_ms` past `current_time_ms`. See the module docs.
///
/// `None` with fewer than two samples, a timestamp list that does not match
/// the positions, or timestamps that do not increase.
pub fn constant_turn_trajectory(
    positions: &[Vector3],
    timestamps: &[u64],
    horizon_ms: u32,
    current_time_ms: u64,
) -> Option<ConstantTurnPrediction> {
    if positions.len() < 2 || timestamps.len() != positions.len() {
        return None;
    }
    let n = positions.len().min(CT_WINDOW);
    let pos = &positions[positions.len() - n..];
    let ts = &timestamps[timestamps.len() - n..];
    if ts.windows(2).any(|w| w[1] <= w[0]) {
        return None;
    }
    let t_last = ts[n - 1];
    let t: Vec<f64> = ts
        .iter()
        .map(|&x| (x as f64 - t_last as f64) / 1000.0)
        .collect();

    let mut fx = AxisFilter::new(pos[0].x);
    let mut fy = AxisFilter::new(pos[0].y);
    let mut fz = AxisFilter::new(pos[0].z);
    // (time, heading) of every step moving fast enough to have one.
    let mut headings: Vec<(f64, f64)> = Vec::with_capacity(n);
    for k in 1..n {
        let dt = t[k] - t[k - 1];
        fx.step(dt, pos[k].x);
        fy.step(dt, pos[k].y);
        fz.step(dt, pos[k].z);
        let (vx, vy) = (fx.x[1], fy.x[1]);
        if vx.hypot(vy) >= CT_MIN_SPEED_MPS {
            let mut h = vy.atan2(vx);
            // Unwrap against the previous heading.
            if let Some(&(_, prev)) = headings.last() {
                let two_pi = std::f64::consts::TAU;
                h -= two_pi * ((h - prev) / two_pi).round();
            }
            headings.push((t[k], h));
        }
    }

    let speed = fx.x[1].hypot(fy.x[1]);
    let (heading, omega) = if headings.len() >= 3 {
        let m = headings.len() as f64;
        let tm = headings.iter().map(|h| h.0).sum::<f64>() / m;
        let hm = headings.iter().map(|h| h.1).sum::<f64>() / m;
        let sxx: f64 = headings.iter().map(|h| (h.0 - tm).powi(2)).sum();
        let sxy: f64 = headings.iter().map(|h| (h.0 - tm) * (h.1 - hm)).sum();
        let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
        // The fitted heading at the newest sample (t = 0).
        (
            hm - slope * tm,
            slope.clamp(-CT_MAX_TURN_RATE_RAD_S, CT_MAX_TURN_RATE_RAD_S),
        )
    } else {
        (fy.x[1].atan2(fx.x[1]), 0.0)
    };
    let (x0, y0, z0, vz) = (fx.x[0], fy.x[0], fz.x[0], fz.x[1]);

    // Hindcast the arc over the window for its R².
    let mean = |get: fn(&Vector3) -> f64| pos.iter().map(get).sum::<f64>() / n as f64;
    let (mx, my, mz) = (mean(|p| p.x), mean(|p| p.y), mean(|p| p.z));
    let mut ss_tot = 0.0;
    let mut ss_res = 0.0;
    for (p, &tk) in pos.iter().zip(&t) {
        let (dx, dy) = arc_offset(speed, heading, omega, tk);
        let (hx, hy, hz) = (x0 + dx, y0 + dy, z0 + vz * tk);
        ss_tot += (p.x - mx).powi(2) + (p.y - my).powi(2) + (p.z - mz).powi(2);
        ss_res += (p.x - hx).powi(2) + (p.y - hy).powi(2) + (p.z - hz).powi(2);
    }
    let r2 = if ss_tot <= 1e-9 {
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };
    let history = (n.saturating_sub(2) as f64 / (CT_FULL_CONFIDENCE_SAMPLES - 2) as f64).min(1.0);
    let confidence = (r2 * history) as f32;
    let confidence = if confidence.is_finite() {
        confidence
    } else {
        0.0
    };

    let steps: u32 = 10;
    let step_ms = horizon_ms / steps;
    let offset_s = (current_time_ms as f64 - t_last as f64) / 1000.0;
    let waypoints = (1..=steps)
        .map(|i| {
            let dt = offset_s + f64::from(i * step_ms) / 1000.0;
            let (dx, dy) = arc_offset(speed, heading, omega, dt);
            (
                Vector3::new(x0 + dx, y0 + dy, z0 + vz * dt),
                current_time_ms + u64::from(i * step_ms),
            )
        })
        .collect();
    let all_finite = [x0, y0, z0, speed, heading, omega]
        .iter()
        .all(|v| v.is_finite());
    if !all_finite {
        return None;
    }
    Some(ConstantTurnPrediction {
        waypoints,
        speed_mps: speed,
        heading_rad: heading,
        turn_rate_rad_s: omega,
        confidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples at 1 s along a circle of radius `r` (centre at the origin),
    /// starting at angle `a0`, angular speed `w` rad/s, rounded to whole metres
    /// as the RLS heartbeat does.
    fn circle(r: f64, a0: f64, w: f64, n: usize) -> (Vec<Vector3>, Vec<u64>) {
        (0..n)
            .map(|i| {
                let a = a0 + w * i as f64;
                (
                    Vector3::new((r * a.cos()).round(), (r * a.sin()).round(), 0.0),
                    1_000_000 + 1000 * i as u64,
                )
            })
            .unzip()
    }

    /// #462: on a steady arc (radius 100 m, 5 m/s, 0.05 rad/s) the constant-turn
    /// extrapolation lands within 8 m of where the UE is 20 s (100 m) later,
    /// while a straight line through the same samples is off by tens of metres:
    /// what the `ct` baseline exists for.
    #[test]
    fn a_steady_arc_is_followed_where_a_straight_line_is_not() {
        let (r, w) = (100.0, 0.05);
        let (pos, ts) = circle(r, 0.0, w, 20);
        let now = *ts.last().unwrap();
        let ct = constant_turn_trajectory(&pos, &ts, 20_000, now).expect("a prediction");
        let truth_a = w * 19.0 + w * 20.0;
        let truth = Vector3::new(r * truth_a.cos(), r * truth_a.sin(), 0.0);
        let (p, at) = ct.waypoints.last().copied().unwrap();
        assert_eq!(at, now + 20_000);
        let err = (p.x - truth.x).hypot(p.y - truth.y);
        assert!(err < 8.0, "ct error {err:.1} m: {ct:?}");
        assert!(
            (ct.turn_rate_rad_s - w).abs() < 0.01,
            "turn rate {}",
            ct.turn_rate_rad_s
        );
        assert!((ct.speed_mps - r * w).abs() < 0.5, "speed {}", ct.speed_mps);
        assert!(ct.confidence > 0.9, "confidence {}", ct.confidence);

        let lsq = crate::predictor::least_squares_trajectory(&pos, &ts, 20_000, now);
        let q = lsq.waypoints.last().unwrap().0;
        let lsq_err = (q.x - truth.x).hypot(q.y - truth.y);
        assert!(lsq_err > 4.0 * err, "lsq {lsq_err:.1} m vs ct {err:.1} m");
    }

    /// A straight track gives a turn rate of about zero, and the extrapolation
    /// is the straight line; clockwise arcs give a negative rate.
    #[test]
    fn a_straight_track_has_no_turn_and_a_clockwise_arc_turns_negative() {
        let pos: Vec<Vector3> = (0..10)
            .map(|i| Vector3::new(5.0 * i as f64, 0.0, 0.0))
            .collect();
        let ts: Vec<u64> = (0..10).map(|i| 1000 * i as u64).collect();
        let ct = constant_turn_trajectory(&pos, &ts, 10_000, 9000).unwrap();
        assert!(ct.turn_rate_rad_s.abs() < 1e-6, "{}", ct.turn_rate_rad_s);
        let end = ct.waypoints.last().unwrap().0;
        assert!((end.x - 95.0).abs() < 1.0 && end.y.abs() < 0.5, "{end:?}");

        let (pos, ts) = circle(100.0, 1.0, -0.05, 12);
        let ct = constant_turn_trajectory(&pos, &ts, 10_000, *ts.last().unwrap()).unwrap();
        assert!(ct.turn_rate_rad_s < -0.03, "{}", ct.turn_rate_rad_s);
    }

    /// A UE that does not move is predicted to stay, with full confidence; a
    /// heading wrap past ±π does not break the turn-rate estimate.
    #[test]
    fn a_stationary_ue_stays_and_a_heading_wrap_is_unwrapped() {
        let pos = vec![Vector3::new(7.0, 3.0, 0.0); 10];
        let ts: Vec<u64> = (0..10).map(|i| 1000 * i as u64).collect();
        let ct = constant_turn_trajectory(&pos, &ts, 20_000, 9000).unwrap();
        let end = ct.waypoints.last().unwrap().0;
        assert!((end.x - 7.0).abs() < 1e-6 && (end.y - 3.0).abs() < 1e-6);
        assert_eq!(ct.turn_rate_rad_s, 0.0);
        assert!((ct.confidence - 1.0).abs() < 1e-6);

        // Heading passes through ±π: moving in -x at the top of a circle.
        let (pos, ts) = circle(100.0, 2.8, 0.05, 15);
        let ct = constant_turn_trajectory(&pos, &ts, 10_000, *ts.last().unwrap()).unwrap();
        assert!(
            (ct.turn_rate_rad_s - 0.05).abs() < 0.01,
            "{}",
            ct.turn_rate_rad_s
        );
    }

    /// Too little or malformed history gives no prediction, never a NaN.
    #[test]
    fn short_or_malformed_history_gives_no_prediction() {
        let p = Vector3::new(0.0, 0.0, 0.0);
        assert!(constant_turn_trajectory(&[p], &[0], 1000, 0).is_none());
        assert!(constant_turn_trajectory(&[p, p], &[0], 1000, 0).is_none());
        assert!(constant_turn_trajectory(&[p, p], &[5, 5], 1000, 5).is_none());
        let nan = Vector3::new(f64::NAN, 0.0, 0.0);
        assert!(constant_turn_trajectory(&[p, nan, p], &[0, 1000, 2000], 1000, 2000).is_none());
        // Two samples: a straight line, no turn, confidence 0.
        let q = Vector3::new(5.0, 0.0, 0.0);
        let ct = constant_turn_trajectory(&[p, q], &[0, 1000], 1000, 1000).unwrap();
        assert_eq!(ct.turn_rate_rad_s, 0.0);
        assert_eq!(ct.confidence, 0.0);
    }
}
