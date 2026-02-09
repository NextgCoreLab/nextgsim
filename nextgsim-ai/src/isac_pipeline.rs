//! ISAC (Integrated Sensing and Communication) processing pipeline
//!
//! This module implements sensing data processing for radar/positioning
//! in 6G networks, fusing communication signals with sensing capabilities.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::config::IsacConfig;
use crate::error::ModelError;
use crate::inference::{InferenceEngine, OnnxEngine};
use crate::tensor::TensorData;

/// Errors that can occur during ISAC processing
#[derive(Error, Debug)]
pub enum IsacError {
    /// Positioning error
    #[error("Positioning error: {reason}")]
    PositioningError { reason: String },

    /// Data fusion error
    #[error("Data fusion error: {reason}")]
    FusionError { reason: String },

    /// Uncertainty threshold exceeded
    #[error("Position uncertainty {actual}m exceeds threshold {threshold}m")]
    UncertaintyExceeded { actual: f32, threshold: f32 },

    /// Model not loaded
    #[error("Positioning model not loaded")]
    ModelNotLoaded,

    /// Invalid sensor data
    #[error("Invalid sensor data: {reason}")]
    InvalidSensorData { reason: String },

    /// Too many data sources
    #[error("Too many data sources: {actual}, maximum {maximum}")]
    TooManyDataSources { actual: usize, maximum: usize },
}

/// Position estimate with uncertainty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionEstimate {
    /// X coordinate in meters
    pub x: f32,
    /// Y coordinate in meters
    pub y: f32,
    /// Z coordinate in meters (altitude)
    pub z: f32,
    /// Uncertainty radius in meters
    pub uncertainty: f32,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp of estimate
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
}

/// Sensing data from a single source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingData {
    /// Source identifier (e.g., "gNB-1", "radar-2")
    pub source_id: String,
    /// Raw sensor measurements
    pub measurements: TensorData,
    /// Signal strength (dBm)
    pub signal_strength: f32,
    /// Timestamp of measurement
    #[serde(skip, default = "Instant::now")]
    pub timestamp: Instant,
    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Fused sensing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedSensingResult {
    /// Position estimate
    pub position: PositionEstimate,
    /// Number of sources used in fusion
    pub num_sources: usize,
    /// Data sources used
    pub sources: Vec<String>,
    /// Processing duration
    #[serde(skip)]
    pub processing_time: Duration,
}

/// ISAC processing pipeline
///
/// Implements sensing data fusion and positioning for integrated sensing
/// and communication in 6G networks.
pub struct IsacPipeline {
    /// Configuration
    config: IsacConfig,
    /// Positioning model
    positioning_model: Option<Box<dyn InferenceEngine>>,
    /// Pending sensing data for fusion
    pending_data: Vec<SensingData>,
    /// Last fusion timestamp
    last_fusion: Option<Instant>,
    /// Processing statistics
    total_fusions: usize,
    total_sources_used: usize,
    avg_uncertainty: f32,
}

impl IsacPipeline {
    /// Creates a new ISAC pipeline with the given configuration
    pub fn new(config: IsacConfig) -> Self {
        Self {
            config,
            positioning_model: None,
            pending_data: Vec::new(),
            last_fusion: None,
            total_fusions: 0,
            total_sources_used: 0,
            avg_uncertainty: 0.0,
        }
    }

    /// Loads the positioning model
    pub fn load_positioning_model(&mut self, path: &Path) -> Result<(), ModelError> {
        info!("Loading ISAC positioning model from {:?}", path);

        let mut engine = OnnxEngine::new(crate::config::ExecutionProvider::Cpu)?;
        engine.load_model(path)?;

        // Warmup the model
        if let Err(e) = engine.warmup() {
            debug!("Positioning model warmup warning: {:?}", e);
        }

        self.positioning_model = Some(Box::new(engine));
        info!("ISAC positioning model loaded successfully");
        Ok(())
    }

    /// Adds sensing data from a source
    pub fn add_sensing_data(&mut self, data: SensingData) -> Result<(), IsacError> {
        // Check if we're exceeding the maximum number of data sources
        if self.pending_data.len() >= self.config.max_data_sources {
            return Err(IsacError::TooManyDataSources {
                actual: self.pending_data.len() + 1,
                maximum: self.config.max_data_sources,
            });
        }

        debug!(
            "Adding sensing data from source: {} (strength: {} dBm)",
            data.source_id, data.signal_strength
        );

        self.pending_data.push(data);
        Ok(())
    }

    /// Checks if fusion should be triggered based on timing
    pub fn should_fuse(&self) -> bool {
        if self.pending_data.is_empty() {
            return false;
        }

        match self.last_fusion {
            None => true, // First fusion
            Some(last) => {
                let elapsed = last.elapsed();
                elapsed >= Duration::from_millis(self.config.fusion_interval_ms as u64)
            }
        }
    }

    /// Performs data fusion and positioning
    pub fn fuse_and_position(&mut self) -> Result<FusedSensingResult, IsacError> {
        let start_time = Instant::now();

        if self.pending_data.is_empty() {
            return Err(IsacError::FusionError {
                reason: "No sensing data available for fusion".to_string(),
            });
        }

        let model = self
            .positioning_model
            .as_ref()
            .ok_or(IsacError::ModelNotLoaded)?;

        debug!(
            "Fusing {} sensing data sources",
            self.pending_data.len()
        );

        // Prepare input for positioning model
        let fused_input = self.prepare_fusion_input()?;

        // Run positioning inference
        let position_output = model
            .infer(&fused_input)
            .map_err(|e| IsacError::PositioningError {
                reason: format!("Model inference failed: {e}"),
            })?;

        // Extract position estimate from model output
        let position = self.extract_position_estimate(&position_output)?;

        // Validate uncertainty
        if position.uncertainty > self.config.position_uncertainty_threshold_m {
            warn!(
                "Position uncertainty {}m exceeds threshold {}m",
                position.uncertainty, self.config.position_uncertainty_threshold_m
            );
        }

        // Create result
        let sources: Vec<String> = self.pending_data.iter().map(|d| d.source_id.clone()).collect();
        let num_sources = sources.len();

        let result = FusedSensingResult {
            position,
            num_sources,
            sources,
            processing_time: start_time.elapsed(),
        };

        // Update statistics
        self.total_fusions += 1;
        self.total_sources_used += num_sources;
        self.avg_uncertainty = (self.avg_uncertainty * (self.total_fusions - 1) as f32
            + result.position.uncertainty)
            / self.total_fusions as f32;

        // Clear pending data and update last fusion time
        self.pending_data.clear();
        self.last_fusion = Some(Instant::now());

        info!(
            "ISAC fusion complete: position ({:.2}, {:.2}, {:.2}) ± {:.2}m, {} sources, {:?}",
            result.position.x,
            result.position.y,
            result.position.z,
            result.position.uncertainty,
            num_sources,
            result.processing_time
        );

        Ok(result)
    }

    /// Prepares input tensor for fusion from multiple sensing data sources
    fn prepare_fusion_input(&self) -> Result<TensorData, IsacError> {
        // Placeholder implementation: concatenate all measurements
        // In production, this would implement sophisticated fusion algorithms like:
        // - Weighted averaging based on signal strength
        // - Kalman filtering
        // - Particle filtering
        // - Maximum likelihood estimation

        if self.pending_data.is_empty() {
            return Err(IsacError::FusionError {
                reason: "No data to fuse".to_string(),
            });
        }

        // For now, use the first data source's measurements
        // In production, this would combine all sources intelligently
        let first_data = &self.pending_data[0];
        Ok(first_data.measurements.clone())
    }

    /// Extracts position estimate from model output
    fn extract_position_estimate(&self, output: &TensorData) -> Result<PositionEstimate, IsacError> {
        // Placeholder implementation: extract from model output tensor
        // Expected output format: [x, y, z, uncertainty]

        let data = output
            .as_f32_slice()
            .ok_or_else(|| IsacError::PositioningError {
                reason: "Output is not float32".to_string(),
            })?;

        if data.len() < 4 {
            return Err(IsacError::PositioningError {
                reason: format!("Expected at least 4 output values, got {}", data.len()),
            });
        }

        let position = PositionEstimate {
            x: data[0],
            y: data[1],
            z: data.get(2).copied().unwrap_or(0.0),
            uncertainty: data.get(3).copied().unwrap_or(1.0),
            confidence: 0.95, // Placeholder confidence
            timestamp: Instant::now(),
        };

        Ok(position)
    }

    /// Returns the number of pending sensing data sources
    pub fn pending_sources(&self) -> usize {
        self.pending_data.len()
    }

    /// Returns true if the positioning model is loaded
    pub fn is_ready(&self) -> bool {
        self.positioning_model.is_some()
    }

    /// Returns average uncertainty across all fusions
    pub fn avg_position_uncertainty(&self) -> f32 {
        self.avg_uncertainty
    }

    /// Returns total number of fusions performed
    pub fn total_fusions(&self) -> usize {
        self.total_fusions
    }

    /// Returns average number of sources per fusion
    pub fn avg_sources_per_fusion(&self) -> f32 {
        if self.total_fusions > 0 {
            self.total_sources_used as f32 / self.total_fusions as f32
        } else {
            0.0
        }
    }

    /// Clears pending sensing data
    pub fn clear_pending(&mut self) {
        self.pending_data.clear();
    }

    /// Returns the configuration
    pub fn config(&self) -> &IsacConfig {
        &self.config
    }

    /// Resets statistics
    pub fn reset_statistics(&mut self) {
        self.total_fusions = 0;
        self.total_sources_used = 0;
        self.avg_uncertainty = 0.0;
    }
}

/// Builder for creating an ISAC pipeline with custom configuration
pub struct IsacPipelineBuilder {
    config: IsacConfig,
    model_path: Option<std::path::PathBuf>,
}

impl IsacPipelineBuilder {
    /// Creates a new builder
    pub fn new() -> Self {
        Self {
            config: IsacConfig::default(),
            model_path: None,
        }
    }

    /// Sets the configuration
    pub fn with_config(mut self, config: IsacConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets the positioning model path
    pub fn with_model(mut self, path: impl Into<std::path::PathBuf>) -> Self {
        self.model_path = Some(path.into());
        self
    }

    /// Builds the ISAC pipeline
    pub fn build(self) -> Result<IsacPipeline, ModelError> {
        let mut pipeline = IsacPipeline::new(self.config);

        if let Some(model_path) = self.model_path {
            pipeline.load_positioning_model(&model_path)?;
        }

        Ok(pipeline)
    }
}

impl Default for IsacPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sensing_data(source_id: &str) -> SensingData {
        SensingData {
            source_id: source_id.to_string(),
            measurements: TensorData::float32(vec![1.0, 2.0, 3.0, 0.5], vec![4]),
            signal_strength: -70.0,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = IsacConfig::default();
        let pipeline = IsacPipeline::new(config);
        assert!(!pipeline.is_ready());
        assert_eq!(pipeline.pending_sources(), 0);
    }

    #[test]
    fn test_add_sensing_data() {
        let mut pipeline = IsacPipeline::new(IsacConfig::default());

        let data = create_test_sensing_data("gNB-1");
        assert!(pipeline.add_sensing_data(data).is_ok());
        assert_eq!(pipeline.pending_sources(), 1);

        let data2 = create_test_sensing_data("gNB-2");
        assert!(pipeline.add_sensing_data(data2).is_ok());
        assert_eq!(pipeline.pending_sources(), 2);
    }

    #[test]
    fn test_too_many_sources() {
        let mut config = IsacConfig::default();
        config.max_data_sources = 2;
        let mut pipeline = IsacPipeline::new(config);

        pipeline.add_sensing_data(create_test_sensing_data("gNB-1")).unwrap();
        pipeline.add_sensing_data(create_test_sensing_data("gNB-2")).unwrap();

        let result = pipeline.add_sensing_data(create_test_sensing_data("gNB-3"));
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            IsacError::TooManyDataSources { .. }
        ));
    }

    #[test]
    fn test_should_fuse() {
        let config = IsacConfig::default();
        let mut pipeline = IsacPipeline::new(config);

        // No data, should not fuse
        assert!(!pipeline.should_fuse());

        // Add data, should fuse (first time)
        pipeline.add_sensing_data(create_test_sensing_data("gNB-1")).unwrap();
        assert!(pipeline.should_fuse());
    }

    #[test]
    fn test_fuse_without_model() {
        let mut pipeline = IsacPipeline::new(IsacConfig::default());
        pipeline.add_sensing_data(create_test_sensing_data("gNB-1")).unwrap();

        let result = pipeline.fuse_and_position();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IsacError::ModelNotLoaded));
    }

    #[test]
    fn test_fuse_without_data() {
        let mut pipeline = IsacPipeline::new(IsacConfig::default());

        let result = pipeline.fuse_and_position();
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), IsacError::FusionError { .. }));
    }

    #[test]
    fn test_position_estimate() {
        let position = PositionEstimate {
            x: 10.5,
            y: 20.3,
            z: 5.0,
            uncertainty: 0.5,
            confidence: 0.95,
            timestamp: Instant::now(),
        };

        assert!((position.x - 10.5).abs() < 0.01);
        assert!((position.y - 20.3).abs() < 0.01);
        assert!((position.z - 5.0).abs() < 0.01);
        assert!((position.uncertainty - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_clear_pending() {
        let mut pipeline = IsacPipeline::new(IsacConfig::default());
        pipeline.add_sensing_data(create_test_sensing_data("gNB-1")).unwrap();
        pipeline.add_sensing_data(create_test_sensing_data("gNB-2")).unwrap();
        assert_eq!(pipeline.pending_sources(), 2);

        pipeline.clear_pending();
        assert_eq!(pipeline.pending_sources(), 0);
    }

    #[test]
    fn test_statistics() {
        let mut pipeline = IsacPipeline::new(IsacConfig::default());

        // Simulate some fusions
        pipeline.total_fusions = 5;
        pipeline.total_sources_used = 15; // average 3 sources per fusion
        pipeline.avg_uncertainty = 0.75;

        assert_eq!(pipeline.total_fusions(), 5);
        assert!((pipeline.avg_sources_per_fusion() - 3.0).abs() < 0.01);
        assert!((pipeline.avg_position_uncertainty() - 0.75).abs() < 0.01);

        pipeline.reset_statistics();
        assert_eq!(pipeline.total_fusions(), 0);
        assert_eq!(pipeline.avg_sources_per_fusion(), 0.0);
    }

    #[test]
    fn test_builder_pattern() {
        let builder = IsacPipelineBuilder::new().with_config(IsacConfig {
            enabled: true,
            positioning_model_path: None,
            fusion_interval_ms: 100,
            max_data_sources: 10,
            position_uncertainty_threshold_m: 2.0,
        });

        assert!(builder.config.enabled);
        assert_eq!(builder.config.fusion_interval_ms, 100);
    }

    #[test]
    fn test_sensing_data_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("beam_id".to_string(), "beam-1".to_string());
        metadata.insert("frequency".to_string(), "28GHz".to_string());

        let data = SensingData {
            source_id: "gNB-1".to_string(),
            measurements: TensorData::float32(vec![1.0, 2.0], vec![2]),
            signal_strength: -65.5,
            timestamp: Instant::now(),
            metadata: metadata.clone(),
        };

        assert_eq!(data.source_id, "gNB-1");
        assert_eq!(data.signal_strength, -65.5);
        assert_eq!(data.metadata.get("beam_id"), Some(&"beam-1".to_string()));
    }
}
