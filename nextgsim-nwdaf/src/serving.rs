//! Verified model loading and serving counters for the trajectory model
//! (nextgsim#463). `model-registry` builds only.
//!
//! - The model at `nwdaf_model_path` is activated only after
//!   [`nextgsim_ai::SharedModelRegistry::register_verified`] passes it against
//!   the manifest beside it (`<stem>.manifest.json`).
//! - Every trajectory prediction is counted by the method that produced it, with
//!   its latency, so "the model served" is a counter and not an assumption:
//!   `onnx_inferences` only moves when the ONNX engine produced the output.
//! - [`NwdafStatus`] is the `nwdaf-status` CLI answer: the counters, the model
//!   state, and an inventory shaped like TS 28.105 `MLModel-Single`.
//!
//! Non-normative: TS 28.105 §6.4.1.1 leaves model loading out of scope, and the
//! inventory is NRM-shaped, not an MnS.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Duration;

use nextgsim_ai::{InferenceMetrics, ModelManifest};
use serde::Serialize;

/// The `schema` field of every [`NwdafStatus`].
pub const STATUS_SCHEMA: &str = "nextgsim.nwdaf-status/1";

/// The manifest that pins `model`: `<stem>.manifest.json` beside it.
pub fn manifest_path(model: &Path) -> PathBuf {
    model.with_extension("manifest.json")
}

/// Where the configured trajectory model stands.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "state", rename_all = "kebab-case")]
pub enum ModelState {
    /// No `nwdaf_model_path` is configured.
    NotConfigured,
    /// A model is configured but was neither verified nor activated (the
    /// d440b `fallback` mode at startup).
    Deferred {
        /// The configured model path.
        path: String,
    },
    /// Verified and activated: predictions may use it.
    Verified {
        /// `mLModelId`.
        id: String,
        /// `mLModelVersion`.
        version: String,
        /// The verified sha256.
        sha256: String,
        /// The verified opset.
        opset: i64,
        /// Where it was read from.
        path: String,
    },
    /// Refused: predictions use the fallback.
    Refused {
        /// The manifest's id, if one was read.
        id: String,
        /// The manifest's version, if one was read.
        version: String,
        /// The short reason (`sha256 mismatch`, ...).
        reason: String,
        /// What failed.
        detail: String,
        /// The configured model path.
        path: String,
    },
}

impl ModelState {
    /// Whether a verified model is active.
    pub fn is_verified(&self) -> bool {
        matches!(self, Self::Verified { .. })
    }
}

/// Per-method serving counters and latencies, shared by `&self` predictions.
#[derive(Debug)]
pub struct ServingStats {
    measurements_recorded: AtomicU64,
    onnx_inferences: AtomicU64,
    fallback_inferences: AtomicU64,
    onnx_latency: Mutex<InferenceMetrics>,
    fallback_latency: Mutex<InferenceMetrics>,
}

impl Default for ServingStats {
    fn default() -> Self {
        Self {
            measurements_recorded: AtomicU64::new(0),
            onnx_inferences: AtomicU64::new(0),
            fallback_inferences: AtomicU64::new(0),
            onnx_latency: Mutex::new(InferenceMetrics::new("trajectory-onnx")),
            fallback_latency: Mutex::new(InferenceMetrics::new("trajectory-fallback")),
        }
    }
}

impl ServingStats {
    /// Count one recorded UE measurement.
    pub fn record_measurement(&self) {
        self.measurements_recorded.fetch_add(1, Ordering::Relaxed);
    }

    /// Count one prediction: by the ONNX model when `onnx`, else by a fallback.
    pub fn record_inference(&self, onnx: bool, latency: Duration) {
        let (count, latencies) = if onnx {
            (&self.onnx_inferences, &self.onnx_latency)
        } else {
            (&self.fallback_inferences, &self.fallback_latency)
        };
        count.fetch_add(1, Ordering::Relaxed);
        latencies
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .record_inference(latency, 0, 0);
    }

    /// Recorded measurements so far.
    pub fn measurements_recorded(&self) -> u64 {
        self.measurements_recorded.load(Ordering::Relaxed)
    }

    /// Predictions the ONNX model produced.
    pub fn onnx_inferences(&self) -> u64 {
        self.onnx_inferences.load(Ordering::Relaxed)
    }

    /// Predictions a fallback produced.
    pub fn fallback_inferences(&self) -> u64 {
        self.fallback_inferences.load(Ordering::Relaxed)
    }

    fn latency(m: &Mutex<InferenceMetrics>) -> LatencySummary {
        let m = m.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let ms = |d: Option<Duration>| d.map(|d| d.as_secs_f64() * 1000.0);
        LatencySummary {
            count: m.inference_count(),
            p50_ms: ms(m.p50_latency()),
            p99_ms: ms(m.p99_latency()),
        }
    }
}

/// Latency of one method's predictions (the most recent 1000).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LatencySummary {
    /// Predictions measured.
    pub count: u64,
    /// Median latency, ms.
    pub p50_ms: Option<f64>,
    /// 99th percentile latency, ms.
    pub p99_ms: Option<f64>,
}

/// Per-method latencies.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Latencies {
    /// ONNX-model predictions.
    pub onnx: LatencySummary,
    /// Fallback predictions.
    pub fallback: LatencySummary,
}

/// The local attributes of TS 28.105 `MLModel-Single` this inventory fills
/// (`TS28105_AiMlNrm.yaml`, `MLModel-Single.allOf[1].properties.attributes`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MlModelAttributes {
    /// `mLModelId`.
    #[serde(rename = "mLModelId")]
    pub ml_model_id: String,
    /// `mLModelVersion`.
    #[serde(rename = "mLModelVersion")]
    pub ml_model_version: String,
    /// `aIMLInferenceName` (here an `NgRanInferenceType`).
    #[serde(rename = "aIMLInferenceName", skip_serializing_if = "Option::is_none")]
    pub aiml_inference_name: Option<String>,
    /// `mLTrainingType`.
    #[serde(rename = "mLTrainingType", skip_serializing_if = "Option::is_none")]
    pub ml_training_type: Option<String>,
}

/// One inventory entry shaped like `MLModel-Single`: `Top`'s `id` plus the
/// local `attributes`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MlModelInventory {
    /// `Top.id`: `<mLModelId>:<mLModelVersion>`.
    pub id: String,
    /// The local attributes.
    pub attributes: MlModelAttributes,
}

impl MlModelInventory {
    /// The entry for a verified manifest.
    pub fn from_manifest(m: &ModelManifest) -> Self {
        Self {
            id: format!("{}:{}", m.id, m.version),
            attributes: MlModelAttributes {
                ml_model_id: m.id.clone(),
                ml_model_version: m.version.clone(),
                aiml_inference_name: m.inference_name.clone(),
                ml_training_type: m.training_type.clone(),
            },
        }
    }
}

/// The `nwdaf-status` answer.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NwdafStatus {
    /// [`STATUS_SCHEMA`].
    pub schema: &'static str,
    /// A verified model is active.
    pub verified: bool,
    /// Predictions are forced to the fallback (d440b `fallback` mode).
    pub force_fallback: bool,
    /// The model's state.
    pub model: ModelState,
    /// UE measurements recorded.
    pub measurements_recorded: u64,
    /// Trajectory predictions the ONNX model produced.
    pub onnx_inferences: u64,
    /// Trajectory predictions a fallback produced.
    pub fallback_inferences: u64,
    /// Per-method latency.
    pub latency: Latencies,
    /// The active model, `MLModel-Single`-shaped; empty unless verified.
    pub models: Vec<MlModelInventory>,
}

impl NwdafStatus {
    /// Assemble the status.
    pub fn new(
        stats: &ServingStats,
        model: &ModelState,
        manifest: Option<&ModelManifest>,
        force_fallback: bool,
    ) -> Self {
        let verified = model.is_verified();
        Self {
            schema: STATUS_SCHEMA,
            verified,
            force_fallback,
            model: model.clone(),
            measurements_recorded: stats.measurements_recorded(),
            onnx_inferences: stats.onnx_inferences(),
            fallback_inferences: stats.fallback_inferences(),
            latency: Latencies {
                onnx: ServingStats::latency(&stats.onnx_latency),
                fallback: ServingStats::latency(&stats.fallback_latency),
            },
            models: manifest
                .filter(|_| verified)
                .map(MlModelInventory::from_manifest)
                .into_iter()
                .collect(),
        }
    }

    /// The status as one line of JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

#[cfg(test)]
pub(crate) mod tests;
