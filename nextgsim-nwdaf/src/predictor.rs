//! Trajectory and load prediction using ONNX models
//!
//! Provides an `OnnxPredictor` that wraps the `nextgsim-ai` `InferenceEngine`
//! for ML-based predictions, with automatic fallback to linear extrapolation
//! when no model is loaded.

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
    /// Predicted waypoints as (position, timestamp_ms) pairs
    pub waypoints: Vec<(Vector3, u64)>,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Which method produced this prediction
    pub method: PredictionMethod,
}

/// ONNX-based predictor with linear extrapolation fallback
///
/// Wraps the `nextgsim-ai` `InferenceEngine` to run trajectory or load
/// prediction models. When no model file has been loaded, it transparently
/// falls back to a simple linear extrapolation so that callers always get
/// a result.
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
    /// Underlying ONNX inference engine
    engine: OnnxEngine,
    /// Whether a model has been successfully loaded
    model_loaded: bool,
}

impl std::fmt::Debug for OnnxPredictor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OnnxPredictor")
            .field("model_loaded", &self.model_loaded)
            .field("is_ready", &self.engine.is_ready())
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
            model_loaded: false,
        })
    }

    /// Creates a new `OnnxPredictor` with a specific execution provider
    ///
    /// # Errors
    ///
    /// Returns an error if the ONNX runtime cannot be initialized.
    pub fn with_execution_provider(
        provider: ExecutionProvider,
    ) -> Result<Self, PredictionError> {
        let engine = OnnxEngine::new(provider).map_err(|e| {
            PredictionError::InferenceFailed {
                reason: format!("Failed to create ONNX engine: {e}"),
            }
        })?;
        Ok(Self {
            engine,
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

        self.engine.load_model(path).map_err(|e| {
            PredictionError::InferenceFailed {
                reason: format!("Failed to load model: {e}"),
            }
        })?;

        self.model_loaded = true;
        info!("Trajectory prediction model loaded successfully");
        Ok(())
    }

    /// Returns whether an ML model is loaded
    pub fn has_model(&self) -> bool {
        self.model_loaded && self.engine.is_ready()
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

        let input = TensorData::float32(
            flat_data,
            vec![1i64, num_points as i64, 3],
        );

        let output = self.engine.infer(&input).map_err(|e| {
            PredictionError::InferenceFailed {
                reason: format!("Inference execution failed: {e}"),
            }
        })?;

        // Parse output tensor into waypoints
        let output_data = output.as_f32_slice().ok_or_else(|| {
            PredictionError::InferenceFailed {
                reason: "Model output is not Float32".to_string(),
            }
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

    /// Linear extrapolation fallback
    ///
    /// Uses the last two position/timestamp pairs to compute velocity,
    /// then projects forward for the given horizon.
    fn predict_linear(
        &self,
        positions: &[Vector3],
        timestamps: &[u64],
        horizon_ms: u32,
        current_time_ms: u64,
    ) -> PredictionOutput {
        let n = positions.len();
        let p1 = &positions[n - 2];
        let p2 = &positions[n - 1];

        // Compute time delta between last two samples
        let dt_ms = if timestamps.len() >= 2 {
            let t1 = timestamps[timestamps.len() - 2];
            let t2 = timestamps[timestamps.len() - 1];
            if t2 > t1 { t2 - t1 } else { 100 } // default 100ms if timestamps are not ordered
        } else {
            100 // assume 100ms interval
        };

        let dt_s = dt_ms as f64 / 1000.0;
        let vx = (p2.x - p1.x) / dt_s;
        let vy = (p2.y - p1.y) / dt_s;
        let vz = (p2.z - p1.z) / dt_s;

        let num_waypoints: u32 = 10;
        let step_ms = horizon_ms / num_waypoints;

        let waypoints: Vec<(Vector3, u64)> = (1..=num_waypoints)
            .map(|i| {
                let t = (f64::from(i) * f64::from(step_ms)) / 1000.0;
                let pos = Vector3::new(p2.x + vx * t, p2.y + vy * t, p2.z + vz * t);
                let timestamp = current_time_ms + (u64::from(i) * u64::from(step_ms));
                (pos, timestamp)
            })
            .collect();

        // Confidence based on how many samples we have (more history = more confident)
        // Cap at 0.7 for linear extrapolation (lower than ML model)
        let confidence = (n as f32 / 20.0).min(0.7);

        PredictionOutput {
            waypoints,
            confidence,
            method: PredictionMethod::LinearExtrapolation,
        }
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
            match self.engine.infer(&input) {
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
        assert!(output.confidence > 0.0 && output.confidence <= 0.7);

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
            PredictionError::InsufficientHistory { required: 2, available: 1 }
        ));
    }

    #[test]
    fn test_predict_trajectory_zero_horizon() {
        let predictor = OnnxPredictor::new().expect("should create");
        let positions = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 1.0, 0.0),
        ];
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
}
