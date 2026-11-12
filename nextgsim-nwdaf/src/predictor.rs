//! Trajectory and load prediction (linear extrapolation; optional ONNX)
//!
//! OPERATIONAL PATH: the simulator ships no ONNX model, so `OnnxPredictor` runs
//! least-squares linear extrapolation (`least_squares_trajectory`) — it fits a
//! line through the recent position/time samples, projects forward, and reports
//! the fit's R² as its confidence. It wraps the `nextgsim-ai`
//! `InferenceEngine` and will use a learned model if one is loaded, but since
//! none is shipped, linear extrapolation is what actually runs.

use std::path::Path;

use nextgsim_ai::{ExecutionProvider, InferenceEngine, OnnxEngine, TensorData};
use tracing::{debug, info, warn};

use crate::error::PredictionError;
use crate::Vector3;

/// Prediction method used for the most recent inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionMethod {
    /// Used an ONNX ML model for prediction
    OnnxModel,
    /// Fell back to linear extrapolation (no model loaded)
    LinearExtrapolation,
}

/// Raw prediction output from the predictor
#[derive(Debug, Clone)]
pub struct PredictionOutput {
    /// Predicted waypoints as (position, `timestamp_ms`) pairs
    pub waypoints: Vec<(Vector3, u64)>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Which method produced this prediction
    pub method: PredictionMethod,
}

/// Predictor with optional ONNX support; linear extrapolation is operational.
///
/// OPERATIONAL PATH: no ONNX model ships, so this runs `predict_linear` (a
/// least-squares constant-velocity fit over the recent samples, R² confidence). It wraps
/// the `nextgsim-ai` `InferenceEngine` and will run a learned model if one is
/// loaded, but since none is shipped, linear extrapolation is what executes.
///
/// # Example
///
/// ```ignore
/// use nextgsim_nwdaf::predictor::OnnxPredictor;
///
/// let mut predictor = OnnxPredictor::new()?;
/// // Optionally load an ONNX model:
/// // predictor.load_model(Path::new("trajectory.onnx"))?;
///
/// let positions = vec![/* position history */];
/// let timestamps = vec![/* corresponding timestamps */];
/// let output = predictor.predict_trajectory(&positions, &timestamps, 1000, base_time)?;
/// ```
pub struct OnnxPredictor {
    /// Underlying ONNX inference engine, owned by this predictor.
    engine: OnnxEngine,
    /// An engine resolved from the shared registry (issue #17), which takes
    /// precedence over `engine` when present.
    ///
    /// Additive: `load_model` still works exactly as before and a predictor that
    /// never touches the registry is unchanged. What this adds is the option of
    /// sharing one warmed session with another consumer of the same model id
    /// rather than loading the file a second time.
    #[cfg(feature = "model-registry")]
    shared_engine: Option<std::sync::Arc<OnnxEngine>>,
    /// Whether a model has been successfully loaded
    model_loaded: bool,
}

impl std::fmt::Debug for OnnxPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxPredictor")
            .field("model_loaded", &self.model_loaded)
            .field("is_ready", &self.active_engine().is_ready())
            .finish()
    }
}

impl OnnxPredictor {
    /// Creates a new `OnnxPredictor` with CPU execution provider
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX runtime cannot be initialized.
    pub fn new() -> Result<Self, PredictionError> {
        let engine = OnnxEngine::new(ExecutionProvider::Cpu).map_err(|e| {
            PredictionError::InferenceFailed {
                reason: format!("Failed to create ONNX engine: {e}"),
            }
        })?;
        Ok(Self {
            engine,
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            model_loaded: false,
        })
    }

    /// Creates a new `OnnxPredictor` with a specific execution provider
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX runtime cannot be initialized.
    pub fn with_execution_provider(provider: ExecutionProvider) -> Result<Self, PredictionError> {
        let engine = OnnxEngine::new(provider).map_err(|e| PredictionError::InferenceFailed {
            reason: format!("Failed to create ONNX engine: {e}"),
        })?;
        Ok(Self {
            engine,
            #[cfg(feature = "model-registry")]
            shared_engine: None,
            model_loaded: false,
        })
    }

    /// Loads an ONNX model from a file path
    ///
    /// Once loaded, all subsequent predictions will use the ML model
    /// instead of linear extrapolation.
    ///
    /// # Errors
    ///
    /// Returns `PredictionError::ModelNotFound` if the file does not exist,
    /// or `PredictionError::InferenceFailed` if loading fails.
    pub fn load_model(&mut self, path: &Path) -> Result<(), PredictionError> {
        if !path.exists() {
            return Err(PredictionError::ModelNotFound {
                path: path.to_path_buf(),
            });
        }

        info!("Loading trajectory prediction model from {:?}", path);

        self.engine
            .load_model(path)
            .map_err(|e| PredictionError::InferenceFailed {
                reason: format!("Failed to load model: {e}"),
            })?;

        self.model_loaded = true;
        info!("Trajectory prediction model loaded successfully");
        Ok(())
    }

    /// Resolve this predictor's model from the shared registry (issue #17)
    /// instead of loading a file, so a second consumer of the same model id shares
    /// one warmed session.
    ///
    /// # Errors
    ///
    /// `RegistryError::NotProvisioned` when the deployment has no model under
    /// `id` — the expected case in a tree that ships no `.onnx`. A caller should
    /// carry on with linear extrapolation, which is what this predictor does
    /// anyway when no model is loaded.
    #[cfg(feature = "model-registry")]
    pub fn resolve_from_registry(
        &mut self,
        registry: &nextgsim_ai::SharedModelRegistry,
        id: &str,
    ) -> Result<(), nextgsim_ai::RegistryError> {
        let engine = registry.load_by_id(id)?;
        info!("Trajectory prediction resolved model {id} from the shared registry");
        self.model_loaded = true;
        self.shared_engine = Some(engine);
        Ok(())
    }

    /// The engine predictions run against: the shared one when the registry
    /// supplied it, else the one this predictor owns.
    fn active_engine(&self) -> &dyn InferenceEngine {
        #[cfg(feature = "model-registry")]
        if let Some(ref shared) = self.shared_engine {
            return shared.as_ref();
        }
        &self.engine
    }

    /// The identity of the engine [`Self::active_engine`] routes to.
    ///
    /// Exists because that routing is otherwise unobservable in this repo: with no
    /// `.onnx` to load, an owned engine and a registry engine are both
    /// `is_ready() == false`, so nothing in an inference result distinguishes
    /// them. Comparing the pointer against `Arc::as_ptr` of the registry's engine
    /// is what actually pins "the predictor uses the shared session".
    #[cfg(feature = "model-registry")]
    pub fn active_engine_ptr(&self) -> *const OnnxEngine {
        #[cfg(feature = "model-registry")]
        if let Some(ref shared) = self.shared_engine {
            return std::sync::Arc::as_ptr(shared);
        }
        &self.engine as *const OnnxEngine
    }

    /// Whether a model has been recorded as loaded, irrespective of whether its
    /// session is usable.
    ///
    /// Distinct from [`Self::has_model`], which also requires the engine to be
    /// ready — so with no `.onnx` in the tree `has_model` is always false and
    /// cannot show whether a resolution succeeded.
    #[cfg(feature = "model-registry")]
    pub fn model_is_recorded(&self) -> bool {
        self.model_loaded
    }

    /// Returns whether an ML model is loaded
    pub fn has_model(&self) -> bool {
        self.model_loaded && self.active_engine().is_ready()
    }

    /// Predicts a trajectory given a position history
    ///
    /// If an ONNX model is loaded, flattens the position history into a
    /// tensor `[1, num_points, 3]` and runs inference. The model is expected
    /// to output `[1, num_waypoints, 3]` predicted positions.
    ///
    /// If no model is loaded, falls back to linear extrapolation using the
    /// last two positions in the history.
    ///
    /// # Arguments
    ///
    /// * `positions` - Historical position samples (at least 2 required)
    /// * `timestamps` - Corresponding timestamps in milliseconds (same length as positions)
    /// * `horizon_ms` - How far into the future to predict (milliseconds)
    /// * `current_time_ms` - Current timestamp for anchoring predictions
    ///
    /// # Errors
    ///
    /// Returns `PredictionError::InsufficientHistory` if fewer than 2 samples
    /// are provided. Returns `PredictionError::InferenceFailed` if the model
    /// produces unexpected output.
    pub fn predict_trajectory(
        &self,
        positions: &[Vector3],
        timestamps: &[u64],
        horizon_ms: u32,
        current_time_ms: u64,
    ) -> Result<PredictionOutput, PredictionError> {
        if positions.len() < 2 {
            return Err(PredictionError::InsufficientHistory {
                required: 2,
                available: positions.len(),
            });
        }

        if horizon_ms == 0 {
            return Err(PredictionError::InvalidHorizon { horizon_ms });
        }

        if self.has_model() {
            match self.predict_with_model(positions, horizon_ms, current_time_ms) {
                Ok(output) => return Ok(output),
                Err(e) => {
                    warn!(
                        "ONNX model inference failed, falling back to linear extrapolation: {}",
                        e
                    );
                }
            }
        } else {
            debug!("No ONNX model loaded, using linear extrapolation fallback");
        }

        Ok(self.predict_linear(positions, timestamps, horizon_ms, current_time_ms))
    }

    /// Runs prediction through the loaded ONNX model
    fn predict_with_model(
        &self,
        positions: &[Vector3],
        horizon_ms: u32,
        current_time_ms: u64,
    ) -> Result<PredictionOutput, PredictionError> {
        // Flatten positions into [1, num_points, 3] tensor
        let num_points = positions.len();
        let mut flat_data = Vec::with_capacity(num_points * 3);
        for pos in positions {
            flat_data.push(pos.x as f32);
            flat_data.push(pos.y as f32);
            flat_data.push(pos.z as f32);
        }

        let input = TensorData::float32(flat_data, vec![1i64, num_points as i64, 3]);

        // The active engine, not `self.engine`: with a registry-resolved model
        // (nextgsim#463) `self.engine` is the never-loaded owned one, and running
        // on it fails every inference into the linear fallback while
        // `has_model()` says a model is serving.
        let output =
            self.active_engine()
                .infer(&input)
                .map_err(|e| PredictionError::InferenceFailed {
                    reason: format!("Inference execution failed: {e}"),
                })?;

        // Parse output tensor into waypoints
        let output_data =
            output
                .as_f32_slice()
                .ok_or_else(|| PredictionError::InferenceFailed {
                    reason: "Model output is not Float32".to_string(),
                })?;

        if output_data.len() % 3 != 0 {
            return Err(PredictionError::InferenceFailed {
                reason: format!(
                    "Model output length {} is not divisible by 3",
                    output_data.len()
                ),
            });
        }

        let num_waypoints = output_data.len() / 3;
        let step_ms = if num_waypoints > 0 {
            horizon_ms / num_waypoints as u32
        } else {
            horizon_ms
        };

        let waypoints: Vec<(Vector3, u64)> = output_data
            .chunks(3)
            .enumerate()
            .map(|(i, chunk)| {
                let pos = Vector3::new(
                    f64::from(chunk[0]),
                    f64::from(chunk[1]),
                    f64::from(chunk[2]),
                );
                let ts = current_time_ms + ((i as u64 + 1) * u64::from(step_ms));
                (pos, ts)
            })
            .collect();

        Ok(PredictionOutput {
            waypoints,
            confidence: 0.9, // ML model predictions get high base confidence
            method: PredictionMethod::OnnxModel,
        })
    }

    /// OPERATIONAL ALGORITHM: least-squares linear extrapolation (the path that
    /// actually runs, since no ONNX model ships). See [`least_squares_trajectory`].
    fn predict_linear(
        &self,
        positions: &[Vector3],
        timestamps: &[u64],
        horizon_ms: u32,
        current_time_ms: u64,
    ) -> PredictionOutput {
        least_squares_trajectory(positions, timestamps, horizon_ms, current_time_ms)
    }

    /// Predicts future load given a history of load values
    ///
    /// If an ONNX model is loaded, uses it for prediction. Otherwise falls
    /// back to linear trend extrapolation on the load values.
    ///
    /// # Arguments
    ///
    /// * `load_values` - Historical load measurements (0.0 to 1.0, e.g. PRB usage)
    /// * `timestamps` - Corresponding timestamps in ms
    /// * `horizon_steps` - Number of future steps to predict
    ///
    /// # Errors
    ///
    /// Returns `PredictionError::InsufficientHistory` if fewer than 2 samples
    /// are provided.
    pub fn predict_load(
        &self,
        load_values: &[f32],
        timestamps: &[u64],
        horizon_steps: usize,
    ) -> Result<(Vec<f32>, PredictionMethod), PredictionError> {
        if load_values.len() < 2 {
            return Err(PredictionError::InsufficientHistory {
                required: 2,
                available: load_values.len(),
            });
        }

        if self.has_model() {
            // Try model-based prediction
            let input = TensorData::float32(
                load_values.to_vec(),
                vec![1i64, load_values.len() as i64, 1],
            );
            match self.active_engine().infer(&input) {
                Ok(output) => {
                    if let Some(data) = output.as_f32_slice() {
                        let predictions: Vec<f32> = data
                            .iter()
                            .take(horizon_steps)
                            .map(|&v| v.clamp(0.0, 1.0))
                            .collect();
                        if !predictions.is_empty() {
                            return Ok((predictions, PredictionMethod::OnnxModel));
                        }
                    }
                    warn!("Model output could not be parsed, falling back to linear trend");
                }
                Err(e) => {
                    warn!("Model inference failed for load prediction: {}", e);
                }
            }
        }

        // Linear trend fallback
        Ok((
            self.predict_load_linear(load_values, timestamps, horizon_steps),
            PredictionMethod::LinearExtrapolation,
        ))
    }

    /// Simple linear trend extrapolation for load prediction
    fn predict_load_linear(
        &self,
        load_values: &[f32],
        _timestamps: &[u64],
        horizon_steps: usize,
    ) -> Vec<f32> {
        let n = load_values.len();
        let last = load_values[n - 1];
        let prev = load_values[n - 2];
        let slope = last - prev;

        (1..=horizon_steps)
            .map(|i| (last + slope * i as f32).clamp(0.0, 1.0))
            .collect()
    }
}

/// The most recent samples a least-squares fit uses (nextgsim#467). At the
/// 1 s RLS heartbeat that is the last 10 s of movement: long enough to average
/// out integer-metre position rounding, short enough that a turning UE is fitted
/// by its current heading rather than where it was going a minute ago.
pub const LSQ_WINDOW: usize = 10;

/// Samples a fit needs for full confidence (nextgsim#467). A line through two
/// points always fits perfectly (R² = 1) and says nothing, so the confidence is
/// R² scaled by `(n - 2) / (LSQ_FULL_CONFIDENCE_SAMPLES - 2)`, capped at 1: two
/// samples give 0, three a third, five or more the full R².
pub const LSQ_FULL_CONFIDENCE_SAMPLES: usize = 5;

/// The speed in m/s a list of waypoints implies: the straight-line distance
/// from the first to the last waypoint over the time between them. 0 for fewer
/// than two waypoints, a zero time span, or a non-finite result.
pub fn waypoint_speed_mps(waypoints: &[(Vector3, u64)]) -> f64 {
    let (Some((a, ta)), Some((b, tb))) = (waypoints.first(), waypoints.last()) else {
        return 0.0;
    };
    let dt_s = tb.saturating_sub(*ta) as f64 / 1000.0;
    if dt_s <= 0.0 {
        return 0.0;
    }
    let speed = a.distance_to(b) / dt_s;
    if speed.is_finite() {
        speed
    } else {
        0.0
    }
}

/// Least-squares constant-velocity trajectory prediction (nextgsim#467).
///
/// Fits `p(t) = a + b·t` per axis by ordinary least squares over the last
/// [`LSQ_WINDOW`] samples (time in seconds, anchored at the newest sample) and
/// projects ten waypoints over `horizon_ms` from `current_time_ms`.
///
/// The confidence is the fit's R² over all three axes together
/// (`1 - SS_res / SS_tot`), scaled down for short histories as
/// [`LSQ_FULL_CONFIDENCE_SAMPLES`] describes, and clamped to `0..=1`:
/// - a UE on a straight track at a steady speed scores close to 1;
/// - a UE whose positions scatter around the line (turning, stop-and-go,
///   noisy positions) scores low, because most of the movement is residual;
/// - a UE that did not move at all (`SS_tot` = 0) is predicted to stay, with the
///   fit's full confidence, since there is no variance left unexplained;
/// - samples that all carry one timestamp give no velocity and confidence 0;
/// - a non-finite input (a NaN or infinite coordinate) gives confidence 0, so no
///   consumer's `confidence < threshold` test can be slipped by a NaN.
///
/// Because the three axes are pooled, the axis with the most movement
/// dominates: a fast walk along x with a few metres of lateral scatter in y
/// still scores high. For a handover the lateral error is the one that can pick
/// the wrong neighbour, so a consumer that cares about it must look at the
/// waypoints, not only at this number.
///
/// A confident "it stays here" is a valid trajectory prediction, but it is not
/// mobility: [`crate::is_predicted_mobility`] also requires a minimum speed.
///
/// This replaces a two-sample velocity whose confidence was the sample count
/// capped at 0.7, which the `> 0.7` mobility test could never pass.
///
/// `timestamps` normally has one entry per position; when it does not, a
/// 100 ms spacing ending at the newest sample is assumed (the old default).
/// `positions` must not be empty.
pub fn least_squares_trajectory(
    positions: &[Vector3],
    timestamps: &[u64],
    horizon_ms: u32,
    current_time_ms: u64,
) -> PredictionOutput {
    let n_all = positions.len();
    let n = n_all.min(LSQ_WINDOW);
    let pos = &positions[n_all - n..];
    // Seconds relative to the newest sample (<= 0).
    let t: Vec<f64> = if timestamps.len() == n_all {
        let ts = &timestamps[n_all - n..];
        let last = ts[n - 1] as f64;
        ts.iter().map(|&x| (x as f64 - last) / 1000.0).collect()
    } else {
        (0..n).map(|i| (i as f64 - (n - 1) as f64) * 0.1).collect()
    };
    let t_last_ms = if timestamps.len() == n_all {
        timestamps[n_all - 1]
    } else {
        current_time_ms
    };

    let nf = n as f64;
    let t_mean = t.iter().sum::<f64>() / nf;
    let sxx: f64 = t.iter().map(|ti| (ti - t_mean).powi(2)).sum();
    let axis = |get: fn(&Vector3) -> f64| {
        let mean = pos.iter().map(get).sum::<f64>() / nf;
        let slope = if sxx > 0.0 {
            t.iter()
                .zip(pos)
                .map(|(ti, p)| (ti - t_mean) * (get(p) - mean))
                .sum::<f64>()
                / sxx
        } else {
            0.0
        };
        let intercept = mean - slope * t_mean;
        let ss_tot: f64 = pos.iter().map(|p| (get(p) - mean).powi(2)).sum();
        let ss_res: f64 = t
            .iter()
            .zip(pos)
            .map(|(ti, p)| (get(p) - (intercept + slope * ti)).powi(2))
            .sum();
        (intercept, slope, ss_tot, ss_res)
    };
    let (ax, bx, tot_x, res_x) = axis(|p| p.x);
    let (ay, by, tot_y, res_y) = axis(|p| p.y);
    let (az, bz, tot_z, res_z) = axis(|p| p.z);

    let ss_tot = tot_x + tot_y + tot_z;
    let ss_res = res_x + res_y + res_z;
    let r2 = if sxx <= 0.0 {
        0.0
    } else if ss_tot <= 1e-9 {
        1.0
    } else {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    };
    let history_factor =
        (n.saturating_sub(2) as f64 / (LSQ_FULL_CONFIDENCE_SAMPLES - 2) as f64).min(1.0);
    let confidence = (r2 * history_factor) as f32;
    let confidence = if confidence.is_finite() {
        confidence
    } else {
        0.0
    };

    let num_waypoints: u32 = 10;
    let step_ms = horizon_ms / num_waypoints;
    let offset_s = (current_time_ms as f64 - t_last_ms as f64) / 1000.0;
    let waypoints = (1..=num_waypoints)
        .map(|i| {
            let dt = offset_s + f64::from(i * step_ms) / 1000.0;
            let p = Vector3::new(ax + bx * dt, ay + by * dt, az + bz * dt);
            (p, current_time_ms + u64::from(i * step_ms))
        })
        .collect();

    PredictionOutput {
        waypoints,
        confidence,
        method: PredictionMethod::LinearExtrapolation,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_predictor_creation() {
        let predictor = OnnxPredictor::new();
        assert!(predictor.is_ok());
        let predictor = predictor.expect("should create");
        assert!(!predictor.has_model());
    }

    #[test]
    fn test_predict_trajectory_linear_fallback() {
        let predictor = OnnxPredictor::new().expect("should create");
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 5.0, 0.0),
            Vector3::new(20.0, 10.0, 0.0),
        ];
        let timestamps = vec![0, 100, 200];

        let result = predictor.predict_trajectory(&positions, &timestamps, 1000, 200);
        assert!(result.is_ok());

        let output = result.expect("should predict");
        assert_eq!(output.method, PredictionMethod::LinearExtrapolation);
        assert!(!output.waypoints.is_empty());
        // Three clean samples: a perfect fit, but only a third of the credit a
        // five-sample history earns.
        assert!(output.confidence > 0.0 && output.confidence < 0.7);

        // First waypoint should continue the linear trend
        let (first_pos, _first_ts) = &output.waypoints[0];
        assert!(first_pos.x > 20.0, "x should increase");
        assert!(first_pos.y > 10.0, "y should increase");
    }

    #[test]
    fn test_predict_trajectory_insufficient_history() {
        let predictor = OnnxPredictor::new().expect("should create");
        let positions = vec![Vector3::new(0.0, 0.0, 0.0)];
        let timestamps = vec![0];

        let result = predictor.predict_trajectory(&positions, &timestamps, 1000, 0);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            PredictionError::InsufficientHistory {
                required: 2,
                available: 1
            }
        ));
    }

    #[test]
    fn test_predict_trajectory_zero_horizon() {
        let predictor = OnnxPredictor::new().expect("should create");
        let positions = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 1.0, 0.0)];
        let timestamps = vec![0, 100];

        let result = predictor.predict_trajectory(&positions, &timestamps, 0, 100);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            PredictionError::InvalidHorizon { horizon_ms: 0 }
        ));
    }

    #[test]
    fn test_load_nonexistent_model() {
        let mut predictor = OnnxPredictor::new().expect("should create");
        let result = predictor.load_model(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            PredictionError::ModelNotFound { .. }
        ));
    }

    #[test]
    fn test_predict_load_linear_fallback() {
        let predictor = OnnxPredictor::new().expect("should create");
        let loads = vec![0.3, 0.35, 0.4, 0.45, 0.5];
        let timestamps = vec![0, 100, 200, 300, 400];

        let result = predictor.predict_load(&loads, &timestamps, 5);
        assert!(result.is_ok());

        let (predictions, method) = result.expect("should predict");
        assert_eq!(method, PredictionMethod::LinearExtrapolation);
        assert_eq!(predictions.len(), 5);
        // Trend is +0.05 per step
        assert!((predictions[0] - 0.55).abs() < 0.01);
        assert!((predictions[1] - 0.60).abs() < 0.01);
    }

    #[test]
    fn test_predict_load_clamping() {
        let predictor = OnnxPredictor::new().expect("should create");
        // Load near 1.0 with strong upward trend
        let loads = vec![0.9, 0.95];
        let timestamps = vec![0, 100];

        let result = predictor.predict_load(&loads, &timestamps, 5);
        assert!(result.is_ok());

        let (predictions, _method) = result.expect("should predict");
        // All predictions should be clamped to [0.0, 1.0]
        for p in &predictions {
            assert!(*p >= 0.0 && *p <= 1.0, "load prediction {p} out of range");
        }
    }

    #[test]
    fn test_predictor_debug() {
        let predictor = OnnxPredictor::new().expect("should create");
        let debug_str = format!("{predictor:?}");
        assert!(debug_str.contains("OnnxPredictor"));
        assert!(debug_str.contains("model_loaded: false"));
    }

    // ---- least-squares predictor (nextgsim#467) ----

    /// A track sampled at the 1 s heartbeat, with integer-metre positions as
    /// the RLS `sim_pos` carries them.
    fn track(f: impl Fn(f64) -> (f64, f64), n: usize) -> (Vec<Vector3>, Vec<u64>) {
        let positions = (0..n)
            .map(|i| {
                let (x, y) = f(i as f64);
                Vector3::new(x.round(), y.round(), 0.0)
            })
            .collect();
        let timestamps = (0..n).map(|i| 1_000 + i as u64 * 1_000).collect();
        (positions, timestamps)
    }

    /// The ticket's unit criterion, first half: a clean track (2.5 m/s along
    /// x, as `config/ue-n2ho.yaml` walks) scores above the 0.7 mobility
    /// threshold, which the old 0.7 cap made unreachable.
    #[test]
    fn a_clean_track_is_confident_above_the_mobility_threshold() {
        let (p, t) = track(|i| (35.0 + 2.5 * i, 0.0), 10);
        let out = least_squares_trajectory(&p, &t, 1000, *t.last().unwrap());
        assert!(out.confidence > 0.7, "confidence {}", out.confidence);
        assert!(out.confidence > 0.99, "R² of a rounded straight line is ~1");
        // 100 ms after the last fix, the fit is ~0.25 m further along x (within
        // half a metre: the positions are rounded to whole metres).
        let (first, ts) = out.waypoints[0];
        assert_eq!(ts, t[9] + 100);
        assert!(
            (first.x - (35.0 + 2.5 * 9.0 + 0.25)).abs() < 0.5,
            "x {}",
            first.x
        );
        let (last, _) = out.waypoints[9];
        assert!((last.x - (35.0 + 2.5 * 10.0)).abs() < 0.5, "x {}", last.x);
    }

    /// Second half: positions scattered around the line (a zig-zag of ±15 m
    /// on a 2.5 m/s walk) leave most of the movement unexplained, so the fit
    /// stays below 0.7 however many samples there are.
    #[test]
    fn high_residuals_keep_the_confidence_below_the_threshold() {
        let (p, t) = track(
            |i| {
                (
                    35.0 + 2.5 * i,
                    if (i as i64) % 2 == 0 { 15.0 } else { -15.0 },
                )
            },
            10,
        );
        let out = least_squares_trajectory(&p, &t, 1000, *t.last().unwrap());
        assert!(out.confidence < 0.7, "confidence {}", out.confidence);
    }

    /// It is a least-squares fit, not the last two samples: one jittered final
    /// fix barely moves the velocity, where a two-point velocity would read it
    /// as a 14 m/s jump.
    #[test]
    fn the_fit_uses_the_window_not_the_last_two_samples() {
        let (mut p, t) = track(|i| (10.0 * i, 0.0), 6);
        p[5].x = 54.0; // 50 expected; two-point velocity (54-40)/1 s = 14 m/s
        let out = least_squares_trajectory(&p, &t, 1000, t[5]);
        let (last, _) = out.waypoints[9];
        // The OLS line through the six points: slope 185/17.5 = 10.571 m/s,
        // value at the newest sample 52.095 m, so one second later 62.667 m
        // (the two-point velocity would give 68).
        assert!((last.x - 62.667).abs() < 0.01, "x {}", last.x);
    }

    /// Only the last LSQ_WINDOW samples count: an old leg in another direction
    /// does not bend the current heading.
    #[test]
    fn samples_older_than_the_window_are_ignored() {
        let (mut p, t) = track(|i| (5.0 * i, 0.0), LSQ_WINDOW + 5);
        for q in p.iter_mut().take(5) {
            q.y = 500.0; // a different leg, outside the window
        }
        let out = least_squares_trajectory(&p, &t, 1000, *t.last().unwrap());
        assert!(out.confidence > 0.99, "confidence {}", out.confidence);
        assert!(out.waypoints.iter().all(|(w, _)| w.y.abs() < 1e-9));
    }

    /// Two samples always fit perfectly and earn nothing; credit grows to the
    /// full R² at LSQ_FULL_CONFIDENCE_SAMPLES.
    #[test]
    fn short_histories_earn_proportionally_less_confidence() {
        let conf = |n| {
            let (p, t) = track(|i| (2.0 * i, 0.0), n);
            least_squares_trajectory(&p, &t, 1000, *t.last().unwrap()).confidence
        };
        assert_eq!(conf(2), 0.0);
        assert!((conf(3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((conf(4) - 2.0 / 3.0).abs() < 1e-6);
        assert!((conf(LSQ_FULL_CONFIDENCE_SAMPLES) - 1.0).abs() < 1e-6);
    }

    /// A UE that has not moved is predicted to stay, with full confidence; one
    /// whose samples share a timestamp has no velocity and no confidence.
    #[test]
    fn a_stationary_ue_stays_and_a_zero_time_span_has_no_confidence() {
        let (p, t) = track(|_| (40.0, 7.0), 6);
        let out = least_squares_trajectory(&p, &t, 1000, *t.last().unwrap());
        assert!((out.confidence - 1.0).abs() < 1e-6);
        assert!(out
            .waypoints
            .iter()
            .all(|(w, _)| (w.x - 40.0).abs() < 1e-9 && (w.y - 7.0).abs() < 1e-9));

        let (p, _) = track(|i| (3.0 * i, 0.0), 6);
        let same = vec![5_000u64; 6];
        let out = least_squares_trajectory(&p, &same, 1000, 5_000);
        assert_eq!(out.confidence, 0.0);
        assert!(out.waypoints.iter().all(|(w, _)| w.x.is_finite()));
    }

    /// #527 review: a NaN coordinate must not give a NaN confidence, which a
    /// `confidence < min` skip test would let through.
    #[test]
    fn a_non_finite_sample_gives_zero_confidence() {
        let (mut p, t) = track(|i| (2.5 * i, 0.0), 8);
        p[3].y = f64::NAN;
        let out = least_squares_trajectory(&p, &t, 1000, t[7]);
        assert_eq!(out.confidence, 0.0);
        let (mut p, t) = track(|i| (2.5 * i, 0.0), 8);
        p[5].x = f64::INFINITY;
        assert_eq!(least_squares_trajectory(&p, &t, 1000, t[7]).confidence, 0.0);
    }

    /// The waypoint speed is distance over time between the first and last.
    #[test]
    fn waypoint_speed_is_first_to_last_distance_over_time() {
        let (p, t) = track(|i| (35.0 + 2.5 * i, 0.0), 10);
        let out = least_squares_trajectory(&p, &t, 1000, t[9]);
        assert!((waypoint_speed_mps(&out.waypoints) - 2.5).abs() < 0.1);
        let (p, t) = track(|_| (40.0, 7.0), 10);
        let out = least_squares_trajectory(&p, &t, 1000, t[9]);
        assert_eq!(waypoint_speed_mps(&out.waypoints), 0.0);
        assert_eq!(waypoint_speed_mps(&[]), 0.0);
        let same_time = [
            (Vector3::new(0.0, 0.0, 0.0), 5),
            (Vector3::new(9.0, 0.0, 0.0), 5),
        ];
        assert_eq!(waypoint_speed_mps(&same_time), 0.0);
    }

    /// The predictor's fallback is exactly this fit.
    #[test]
    fn the_predictor_fallback_is_the_least_squares_fit() {
        let predictor = OnnxPredictor::new().expect("should create");
        let (p, t) = track(|i| (35.0 + 2.5 * i, 1.0 * i), 8);
        let a = predictor.predict_trajectory(&p, &t, 1000, t[7]).unwrap();
        let b = least_squares_trajectory(&p, &t, 1000, t[7]);
        assert_eq!(a.confidence, b.confidence);
        assert_eq!(a.waypoints, b.waypoints);
        assert!(a.confidence > 0.7);
    }
}

// ============================================================================
// Shared model registry (issue #17)
// ============================================================================

#[cfg(all(test, feature = "model-registry"))]
mod registry_tests {
    use super::*;
    use nextgsim_ai::SharedModelRegistry;
    use std::sync::Arc;

    /// The NWDAF predictor resolves its trajectory model by id, and a second
    /// consumer of the same id shares the engine rather than loading it again.
    #[test]
    fn the_predictor_shares_a_registry_engine() {
        let registry = SharedModelRegistry::new(ExecutionProvider::Cpu);
        let engine = Arc::new(OnnxEngine::new(ExecutionProvider::Cpu).expect("engine"));
        registry
            .register_preloaded(
                "nwdaf-trajectory",
                "1.0.0",
                "/models/trajectory.onnx",
                Arc::clone(&engine),
            )
            .expect("register");

        let mut first = OnnxPredictor::new().expect("predictor");
        let mut second = OnnxPredictor::new().expect("predictor");
        first
            .resolve_from_registry(&registry, "nwdaf-trajectory")
            .expect("first resolves");
        second
            .resolve_from_registry(&registry, "nwdaf-trajectory")
            .expect("second resolves");

        assert_eq!(
            registry.loaded_count(),
            1,
            "one engine for one model id, whatever the consumer count"
        );
        // Exactly four holders: this test, the registry, and each predictor. An
        // inequality would be satisfied by only one predictor having resolved.
        assert_eq!(
            Arc::strong_count(&engine),
            4,
            "this test, the registry and both predictors"
        );
        // And each predictor actually *routes* to that engine, which the counts
        // above do not show: a predictor could hold the Arc and still run on its
        // own engine.
        assert_eq!(first.active_engine_ptr(), Arc::as_ptr(&engine));
        assert_eq!(second.active_engine_ptr(), Arc::as_ptr(&engine));
    }

    /// An unprovisioned id leaves the predictor on linear extrapolation rather
    /// than failing, which is what it does with no model anyway.
    #[test]
    fn an_unprovisioned_id_leaves_the_predictor_without_a_model() {
        let registry = SharedModelRegistry::new(ExecutionProvider::Cpu);
        let mut predictor = OnnxPredictor::new().expect("predictor");

        let err = predictor
            .resolve_from_registry(&registry, "absent")
            .expect_err("nothing is provisioned");
        assert!(err.is_not_provisioned(), "{err}");
        assert!(
            !predictor.model_is_recorded(),
            "a failed resolution must not mark a model as loaded"
        );
        assert_eq!(
            predictor.active_engine_ptr(),
            &predictor.engine as *const OnnxEngine,
            "and it still routes to its own engine"
        );
    }
}
