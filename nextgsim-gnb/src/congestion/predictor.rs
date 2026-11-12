//! The 8-sample least-squares occupancy predictor of mode `prediction`
//! (nextgsim#465).
//!
//! **Not a learned model**: an ordinary least-squares line through the last
//! [`SAMPLES`] occupancy samples, extrapolated `horizon` ahead of the newest.
//! Its confidence is the fit's R² in percent, so a clean ramp predicts with
//! confidence 100 and a noisy series with less. The demo's point is that a
//! prediction moves the policy change earlier, not the predictor's accuracy.

/// How many samples a prediction needs, and uses.
pub const SAMPLES: usize = 8;

/// One prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// Predicted occupancy at the horizon, percent, clamped to 0..100.
    pub occupancy_pct: f64,
    /// round(R² x 100), 0..100.
    pub confidence: u8,
}

/// Predict from `samples` (`(t seconds, occupancy %)`, oldest first) at
/// `horizon_s` after the newest. `None` with fewer than [`SAMPLES`] samples.
/// Only the newest [`SAMPLES`] are used.
pub fn predict(samples: &[(f64, f64)], horizon_s: f64) -> Option<Prediction> {
    if samples.len() < SAMPLES {
        return None;
    }
    let s = &samples[samples.len() - SAMPLES..];
    let n = s.len() as f64;
    let mean_t = s.iter().map(|p| p.0).sum::<f64>() / n;
    let mean_y = s.iter().map(|p| p.1).sum::<f64>() / n;
    let sxx: f64 = s.iter().map(|p| (p.0 - mean_t).powi(2)).sum();
    let sxy: f64 = s.iter().map(|p| (p.0 - mean_t) * (p.1 - mean_y)).sum();
    let syy: f64 = s.iter().map(|p| (p.1 - mean_y).powi(2)).sum();
    // All samples at one instant: no slope can be fitted.
    if sxx <= f64::EPSILON {
        return None;
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_t;
    let t_last = s[s.len() - 1].0;
    let y = intercept + slope * (t_last + horizon_s);
    // A flat series is fitted exactly by a flat line.
    let r2 = if syy <= f64::EPSILON {
        1.0
    } else {
        let ss_res: f64 = s
            .iter()
            .map(|p| (p.1 - (intercept + slope * p.0)).powi(2))
            .sum();
        (1.0 - ss_res / syy).clamp(0.0, 1.0)
    };
    Some(Prediction {
        occupancy_pct: y.clamp(0.0, 100.0),
        confidence: (r2 * 100.0).round().clamp(0.0, 100.0) as u8,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clean ramp of 5 %/s from 20 % predicts the extrapolated value with
    /// confidence 100: at t=7 it is 55 %, so 5 s later 80 %.
    #[test]
    fn a_linear_ramp_extrapolates_with_full_confidence() {
        let ramp: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 20.0 + 5.0 * i as f64)).collect();
        let p = predict(&ramp, 5.0).unwrap();
        assert!((p.occupancy_pct - 80.0).abs() < 1e-9, "{p:?}");
        assert_eq!(p.confidence, 100);
    }

    /// Noise around the same ramp lowers the confidence.
    #[test]
    fn noise_lowers_the_confidence() {
        let noise = [9.0, -9.0, 7.0, -8.0, 9.0, -7.0, 8.0, -9.0];
        let noisy: Vec<(f64, f64)> = (0..8)
            .map(|i| (i as f64, 20.0 + 5.0 * i as f64 + noise[i]))
            .collect();
        let p = predict(&noisy, 5.0).unwrap();
        assert!(p.confidence < 90, "{p:?}");
        assert!(p.confidence > 0, "{p:?}");
    }

    /// Fewer than eight samples predict nothing; only the newest eight count.
    #[test]
    fn eight_samples_are_needed_and_used() {
        let ramp: Vec<(f64, f64)> = (0..7).map(|i| (i as f64, 10.0 * i as f64)).collect();
        assert_eq!(predict(&ramp, 1.0), None);
        // Twelve samples: an old flat stretch, then a ramp; the fit sees only
        // the ramp, so the prediction is the ramp's.
        let mut s: Vec<(f64, f64)> = (0..4).map(|i| (i as f64, 90.0)).collect();
        s.extend((4..12).map(|i| (i as f64, 2.0 * i as f64)));
        let p = predict(&s, 1.0).unwrap();
        assert!((p.occupancy_pct - 24.0).abs() < 1e-9, "{p:?}");
        assert_eq!(p.confidence, 100);
    }

    /// The prediction is clamped to 0..100.
    #[test]
    fn the_prediction_is_clamped() {
        let up: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 60.0 + 10.0 * i as f64)).collect();
        assert_eq!(predict(&up, 10.0).unwrap().occupancy_pct, 100.0);
        let down: Vec<(f64, f64)> = (0..8).map(|i| (i as f64, 70.0 - 10.0 * i as f64)).collect();
        assert_eq!(predict(&down, 10.0).unwrap().occupancy_pct, 0.0);
    }
}
