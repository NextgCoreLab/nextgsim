//! Model Training Logical Function (MTLF)
//!
//! Implements the MTLF component of NWDAF as defined in 3GPP TS 23.288.
//! MTLF is responsible for:
//! - Training ML models using collected data
//! - Managing the ML model lifecycle (versioning, storage, distribution)
//! - Providing trained models to AnLF or external consumers
//!
//! In the simulator context, MTLF manages loading and distributing
//! pre-trained ONNX models rather than performing real training.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::analytics_id::AnalyticsId;
use crate::error::NwdafError;
use crate::predictor::OnnxPredictor;

/// Information about a managed ML model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlModelInfo {
    /// Unique model identifier
    pub model_id: String,
    /// Analytics ID this model serves
    pub analytics_id: AnalyticsId,
    /// Model version string
    pub version: String,
    /// File path to the model
    pub path: PathBuf,
    /// Model file size in bytes
    pub file_size: u64,
    /// Accuracy metric from training/evaluation (0.0 to 1.0)
    pub accuracy: Option<f32>,
    /// Whether the model is currently loaded and available
    pub is_loaded: bool,
    /// Description of the model
    pub description: String,
}

/// Model provision request from a consumer (e.g. AnLF)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvisionRequest {
    /// Requested analytics ID
    pub analytics_id: AnalyticsId,
    /// Optional specific model version
    pub version: Option<String>,
    /// Requestor identifier
    pub requestor_id: String,
}

/// Model provision response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProvisionResponse {
    /// Model information
    pub model_info: MlModelInfo,
    /// Whether the model was successfully provisioned
    pub success: bool,
    /// Error message if provisioning failed
    pub error: Option<String>,
}

/// Model Training Logical Function
///
/// Manages the ML model lifecycle within NWDAF. In a production 3GPP
/// deployment, MTLF would perform actual model training on collected data.
/// In the simulator, it manages pre-trained ONNX models, handles versioning,
/// and distributes models to the AnLF and external consumers.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 6.2A: NWDAF containing MTLF
/// - TS 23.288 Section 7.5: Nnwdaf_MLModelProvision service
#[derive(Debug)]
pub struct Mtlf {
    /// Registered ML models indexed by model ID
    models: HashMap<String, MlModelInfo>,
    /// Best model per analytics ID (model_id)
    best_model_per_analytics: HashMap<AnalyticsId, String>,
    /// Active trajectory predictor (loaded ONNX model)
    trajectory_predictor: Option<OnnxPredictor>,
    /// Model storage directory
    model_dir: Option<PathBuf>,
}

impl Mtlf {
    /// Creates a new MTLF instance
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            best_model_per_analytics: HashMap::new(),
            trajectory_predictor: None,
            model_dir: None,
        }
    }

    /// Creates a new MTLF with a model storage directory
    pub fn with_model_dir(model_dir: PathBuf) -> Self {
        Self {
            models: HashMap::new(),
            best_model_per_analytics: HashMap::new(),
            trajectory_predictor: None,
            model_dir: Some(model_dir),
        }
    }

    /// Registers an ML model
    ///
    /// Adds the model to the registry. If this is the first model for its
    /// analytics ID, it becomes the default (best) model for that type.
    pub fn register_model(&mut self, model_info: MlModelInfo) {
        let model_id = model_info.model_id.clone();
        let analytics_id = model_info.analytics_id;

        // If no best model for this analytics ID, set this one
        if !self.best_model_per_analytics.contains_key(&analytics_id) {
            self.best_model_per_analytics
                .insert(analytics_id, model_id.clone());
        } else {
            // If new model has better accuracy, update best
            if let Some(current_best_id) = self.best_model_per_analytics.get(&analytics_id) {
                if let Some(current_best) = self.models.get(current_best_id) {
                    if model_info.accuracy > current_best.accuracy {
                        self.best_model_per_analytics
                            .insert(analytics_id, model_id.clone());
                    }
                }
            }
        }

        info!(
            "MTLF: Registered model {} for {:?} (version={})",
            model_id, analytics_id, model_info.version
        );
        self.models.insert(model_id, model_info);
    }

    /// Loads a trajectory prediction model from file
    ///
    /// Creates an `OnnxPredictor` and loads the model, making it available
    /// for the AnLF to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file cannot be loaded.
    pub fn load_trajectory_model(&mut self, path: &Path) -> Result<(), NwdafError> {
        let mut predictor =
            OnnxPredictor::new().map_err(NwdafError::Prediction)?;
        predictor
            .load_model(path)
            .map_err(NwdafError::Prediction)?;

        // Register model info
        let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let model_info = MlModelInfo {
            model_id: format!(
                "trajectory_{}",
                path.file_stem()
                    .map(|s| s.to_string_lossy().to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            ),
            analytics_id: AnalyticsId::UeMobility,
            version: "1.0.0".to_string(),
            path: path.to_path_buf(),
            file_size,
            accuracy: None,
            is_loaded: true,
            description: "Trajectory prediction ONNX model".to_string(),
        };
        self.register_model(model_info);
        self.trajectory_predictor = Some(predictor);

        info!("MTLF: Trajectory model loaded from {:?}", path);
        Ok(())
    }

    /// Returns a reference to the trajectory predictor if loaded
    pub fn trajectory_predictor(&self) -> Option<&OnnxPredictor> {
        self.trajectory_predictor.as_ref()
    }

    /// Returns model info for a specific model ID
    pub fn get_model(&self, model_id: &str) -> Option<&MlModelInfo> {
        self.models.get(model_id)
    }

    /// Returns the best model for a given analytics ID
    pub fn get_best_model(&self, analytics_id: AnalyticsId) -> Option<&MlModelInfo> {
        self.best_model_per_analytics
            .get(&analytics_id)
            .and_then(|id| self.models.get(id))
    }

    /// Lists all registered models
    pub fn list_models(&self) -> Vec<&MlModelInfo> {
        self.models.values().collect()
    }

    /// Lists models for a specific analytics ID
    pub fn list_models_for_analytics(&self, analytics_id: AnalyticsId) -> Vec<&MlModelInfo> {
        self.models
            .values()
            .filter(|m| m.analytics_id == analytics_id)
            .collect()
    }

    /// Handles a model provision request (Nnwdaf_MLModelProvision)
    ///
    /// Returns the best available model for the requested analytics ID,
    /// or an error if no model is available.
    pub fn handle_provision_request(
        &self,
        request: &ModelProvisionRequest,
    ) -> ModelProvisionResponse {
        // If specific version requested, find that exact model
        if let Some(ref version) = request.version {
            for model in self.models.values() {
                if model.analytics_id == request.analytics_id && model.version == *version {
                    return ModelProvisionResponse {
                        model_info: model.clone(),
                        success: true,
                        error: None,
                    };
                }
            }
            return ModelProvisionResponse {
                model_info: MlModelInfo {
                    model_id: String::new(),
                    analytics_id: request.analytics_id,
                    version: version.clone(),
                    path: PathBuf::new(),
                    file_size: 0,
                    accuracy: None,
                    is_loaded: false,
                    description: String::new(),
                },
                success: false,
                error: Some(format!(
                    "No model found for {:?} version {}",
                    request.analytics_id, version
                )),
            };
        }

        // Otherwise return the best model
        match self.get_best_model(request.analytics_id) {
            Some(model) => {
                info!(
                    "MTLF: Provisioning model {} to {}",
                    model.model_id, request.requestor_id
                );
                ModelProvisionResponse {
                    model_info: model.clone(),
                    success: true,
                    error: None,
                }
            }
            None => {
                warn!(
                    "MTLF: No model available for {:?} (requested by {})",
                    request.analytics_id, request.requestor_id
                );
                ModelProvisionResponse {
                    model_info: MlModelInfo {
                        model_id: String::new(),
                        analytics_id: request.analytics_id,
                        version: String::new(),
                        path: PathBuf::new(),
                        file_size: 0,
                        accuracy: None,
                        is_loaded: false,
                        description: String::new(),
                    },
                    success: false,
                    error: Some(format!(
                        "No model available for {:?}",
                        request.analytics_id
                    )),
                }
            }
        }
    }

    /// Removes a model from the registry
    pub fn unregister_model(&mut self, model_id: &str) -> Option<MlModelInfo> {
        let removed = self.models.remove(model_id);
        if let Some(ref info) = removed {
            // Clean up best_model_per_analytics if this was the best
            if let Some(best_id) = self.best_model_per_analytics.get(&info.analytics_id) {
                if best_id == model_id {
                    // Find next best model for this analytics ID
                    let next_best = self
                        .models
                        .values()
                        .filter(|m| m.analytics_id == info.analytics_id)
                        .max_by(|a, b| {
                            a.accuracy
                                .partial_cmp(&b.accuracy)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|m| m.model_id.clone());

                    match next_best {
                        Some(id) => {
                            self.best_model_per_analytics
                                .insert(info.analytics_id, id);
                        }
                        None => {
                            self.best_model_per_analytics.remove(&info.analytics_id);
                        }
                    }
                }
            }
        }
        removed
    }

    /// Returns the number of registered models
    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Returns the configured model storage directory
    pub fn model_dir(&self) -> Option<&PathBuf> {
        self.model_dir.as_ref()
    }

    /// Triggers model retraining for analytics IDs with degraded accuracy.
    ///
    /// In production this would schedule an async retraining job.
    /// Here we invalidate the current model's accuracy to mark it as stale,
    /// forcing the next model provision to prefer a freshly-registered model.
    pub fn trigger_retraining(&mut self, analytics_ids: &[AnalyticsId]) -> Vec<AnalyticsId> {
        let mut triggered = Vec::new();
        for &aid in analytics_ids {
            if let Some(best_id) = self.best_model_per_analytics.get(&aid).cloned() {
                if let Some(model) = self.models.get_mut(&best_id) {
                    info!(
                        "MTLF: Triggering retraining for {:?} (model={})",
                        aid, best_id
                    );
                    // Mark model as needing retrain by resetting accuracy
                    model.accuracy = None;
                    triggered.push(aid);
                }
            }
        }
        triggered
    }
}

impl Default for Mtlf {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model_info(id: &str, analytics_id: AnalyticsId, accuracy: Option<f32>) -> MlModelInfo {
        MlModelInfo {
            model_id: id.to_string(),
            analytics_id,
            version: "1.0.0".to_string(),
            path: PathBuf::from(format!("/models/{id}.onnx")),
            file_size: 1024,
            accuracy,
            is_loaded: false,
            description: format!("Test model {id}"),
        }
    }

    #[test]
    fn test_mtlf_creation() {
        let mtlf = Mtlf::new();
        assert_eq!(mtlf.model_count(), 0);
        assert!(mtlf.trajectory_predictor().is_none());
    }

    #[test]
    fn test_register_model() {
        let mut mtlf = Mtlf::new();
        let model = make_model_info("m1", AnalyticsId::UeMobility, Some(0.9));
        mtlf.register_model(model);
        assert_eq!(mtlf.model_count(), 1);
        assert!(mtlf.get_model("m1").is_some());
    }

    #[test]
    fn test_best_model_selection() {
        let mut mtlf = Mtlf::new();

        let m1 = make_model_info("m1", AnalyticsId::UeMobility, Some(0.8));
        let m2 = make_model_info("m2", AnalyticsId::UeMobility, Some(0.95));

        mtlf.register_model(m1);
        mtlf.register_model(m2);

        let best = mtlf.get_best_model(AnalyticsId::UeMobility);
        assert!(best.is_some());
        assert_eq!(best.expect("should exist").model_id, "m2");
    }

    #[test]
    fn test_list_models_for_analytics() {
        let mut mtlf = Mtlf::new();
        mtlf.register_model(make_model_info("m1", AnalyticsId::UeMobility, None));
        mtlf.register_model(make_model_info("m2", AnalyticsId::NfLoad, None));
        mtlf.register_model(make_model_info("m3", AnalyticsId::UeMobility, None));

        let mobility_models = mtlf.list_models_for_analytics(AnalyticsId::UeMobility);
        assert_eq!(mobility_models.len(), 2);

        let load_models = mtlf.list_models_for_analytics(AnalyticsId::NfLoad);
        assert_eq!(load_models.len(), 1);
    }

    #[test]
    fn test_model_provision_success() {
        let mut mtlf = Mtlf::new();
        mtlf.register_model(make_model_info("m1", AnalyticsId::UeMobility, Some(0.9)));

        let request = ModelProvisionRequest {
            analytics_id: AnalyticsId::UeMobility,
            version: None,
            requestor_id: "anlf-1".to_string(),
        };

        let response = mtlf.handle_provision_request(&request);
        assert!(response.success);
        assert!(response.error.is_none());
        assert_eq!(response.model_info.model_id, "m1");
    }

    #[test]
    fn test_model_provision_not_found() {
        let mtlf = Mtlf::new();

        let request = ModelProvisionRequest {
            analytics_id: AnalyticsId::NfLoad,
            version: None,
            requestor_id: "anlf-1".to_string(),
        };

        let response = mtlf.handle_provision_request(&request);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_model_provision_specific_version() {
        let mut mtlf = Mtlf::new();
        let mut model = make_model_info("m1", AnalyticsId::UeMobility, Some(0.9));
        model.version = "2.0.0".to_string();
        mtlf.register_model(model);

        // Request non-existent version
        let request = ModelProvisionRequest {
            analytics_id: AnalyticsId::UeMobility,
            version: Some("3.0.0".to_string()),
            requestor_id: "anlf-1".to_string(),
        };
        let response = mtlf.handle_provision_request(&request);
        assert!(!response.success);

        // Request existing version
        let request = ModelProvisionRequest {
            analytics_id: AnalyticsId::UeMobility,
            version: Some("2.0.0".to_string()),
            requestor_id: "anlf-1".to_string(),
        };
        let response = mtlf.handle_provision_request(&request);
        assert!(response.success);
    }

    #[test]
    fn test_unregister_model() {
        let mut mtlf = Mtlf::new();
        mtlf.register_model(make_model_info("m1", AnalyticsId::UeMobility, Some(0.9)));
        mtlf.register_model(make_model_info("m2", AnalyticsId::UeMobility, Some(0.8)));

        let removed = mtlf.unregister_model("m1");
        assert!(removed.is_some());
        assert_eq!(mtlf.model_count(), 1);

        // Best model should now be m2
        let best = mtlf.get_best_model(AnalyticsId::UeMobility);
        assert!(best.is_some());
        assert_eq!(best.expect("should exist").model_id, "m2");
    }

    #[test]
    fn test_unregister_last_model() {
        let mut mtlf = Mtlf::new();
        mtlf.register_model(make_model_info("m1", AnalyticsId::UeMobility, None));

        mtlf.unregister_model("m1");
        assert_eq!(mtlf.model_count(), 0);
        assert!(mtlf.get_best_model(AnalyticsId::UeMobility).is_none());
    }
}
