//! Model metadata and information
//!
//! This module provides structures for describing ML model metadata,
//! including input/output specifications and model versioning.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

use crate::tensor::TensorShape;

/// Information about a tensor (input or output)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorInfo {
    /// Name of the tensor
    pub name: String,
    /// Shape of the tensor (-1 indicates dynamic dimension)
    pub shape: TensorShape,
    /// Data type of the tensor (e.g., "float32", "int64")
    pub dtype: String,
}

impl TensorInfo {
    /// Creates a new TensorInfo
    pub fn new(name: impl Into<String>, shape: impl Into<TensorShape>, dtype: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
            dtype: dtype.into(),
        }
    }
}

/// Model metadata describing inputs, outputs, and properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Model name
    pub name: String,
    /// Model version
    pub version: String,
    /// Input tensor specifications
    pub inputs: Vec<TensorInfo>,
    /// Output tensor specifications
    pub outputs: Vec<TensorInfo>,
    /// Custom properties/metadata
    pub properties: HashMap<String, String>,
}

impl ModelMetadata {
    /// Creates a new ModelMetadata with required fields
    pub fn new(name: impl Into<String>, version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            properties: HashMap::new(),
        }
    }

    /// Adds an input tensor specification
    pub fn with_input(mut self, info: TensorInfo) -> Self {
        self.inputs.push(info);
        self
    }

    /// Adds an output tensor specification
    pub fn with_output(mut self, info: TensorInfo) -> Self {
        self.outputs.push(info);
        self
    }

    /// Adds a custom property
    pub fn with_property(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.properties.insert(key.into(), value.into());
        self
    }

    /// Returns the number of inputs
    pub fn num_inputs(&self) -> usize {
        self.inputs.len()
    }

    /// Returns the number of outputs
    pub fn num_outputs(&self) -> usize {
        self.outputs.len()
    }

    /// Gets input info by name
    pub fn get_input(&self, name: &str) -> Option<&TensorInfo> {
        self.inputs.iter().find(|i| i.name == name)
    }

    /// Gets output info by name
    pub fn get_output(&self, name: &str) -> Option<&TensorInfo> {
        self.outputs.iter().find(|o| o.name == name)
    }

    /// Gets input info by index
    pub fn get_input_by_index(&self, index: usize) -> Option<&TensorInfo> {
        self.inputs.get(index)
    }

    /// Gets output info by index
    pub fn get_output_by_index(&self, index: usize) -> Option<&TensorInfo> {
        self.outputs.get(index)
    }
}

impl Default for ModelMetadata {
    fn default() -> Self {
        Self::new("unknown", "0.0.0")
    }
}

/// Information about a loaded model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Path to the model file
    pub path: PathBuf,
    /// Model metadata
    pub metadata: ModelMetadata,
    /// File size in bytes
    pub file_size: u64,
    /// Whether the model is loaded and ready
    pub is_ready: bool,
    /// Model type (e.g., "onnx", "tflite")
    pub model_type: ModelType,
}

impl ModelInfo {
    /// Creates a new ModelInfo
    pub fn new(path: impl Into<PathBuf>, model_type: ModelType) -> Self {
        Self {
            path: path.into(),
            metadata: ModelMetadata::default(),
            file_size: 0,
            is_ready: false,
            model_type,
        }
    }

    /// Sets the metadata
    pub fn with_metadata(mut self, metadata: ModelMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Sets the file size
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = size;
        self
    }

    /// Marks the model as ready
    pub fn mark_ready(mut self) -> Self {
        self.is_ready = true;
        self
    }
}

/// Supported model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// ONNX Runtime model
    Onnx,
    /// TensorFlow Lite model (for future support)
    TfLite,
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelType::Onnx => write!(f, "ONNX"),
            ModelType::TfLite => write!(f, "TFLite"),
        }
    }
}

/// Model registry for managing multiple loaded models
#[derive(Debug, Default)]
pub struct ModelRegistry {
    /// Registered models by name
    models: HashMap<String, ModelInfo>,
}

impl ModelRegistry {
    /// Creates a new empty registry
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Registers a model
    pub fn register(&mut self, name: impl Into<String>, info: ModelInfo) {
        self.models.insert(name.into(), info);
    }

    /// Gets a model by name
    pub fn get(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name)
    }

    /// Removes a model by name
    pub fn unregister(&mut self, name: &str) -> Option<ModelInfo> {
        self.models.remove(name)
    }

    /// Lists all registered model names
    pub fn list(&self) -> Vec<&str> {
        self.models.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of registered models
    pub fn len(&self) -> usize {
        self.models.len()
    }

    /// Returns true if the registry is empty
    pub fn is_empty(&self) -> bool {
        self.models.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_info_creation() {
        let info = TensorInfo::new("input", vec![1i64, 10, 3], "float32");
        assert_eq!(info.name, "input");
        assert_eq!(info.dtype, "float32");
        assert_eq!(info.shape.dims(), &[1, 10, 3]);
    }

    #[test]
    fn test_model_metadata_builder() {
        let metadata = ModelMetadata::new("trajectory_model", "1.0.0")
            .with_input(TensorInfo::new("positions", vec![1i64, -1, 3], "float32"))
            .with_output(TensorInfo::new("predictions", vec![1i64, 10, 3], "float32"))
            .with_property("description", "UE trajectory prediction model");

        assert_eq!(metadata.name, "trajectory_model");
        assert_eq!(metadata.version, "1.0.0");
        assert_eq!(metadata.num_inputs(), 1);
        assert_eq!(metadata.num_outputs(), 1);
        assert!(metadata.get_input("positions").is_some());
        assert!(metadata.get_output("predictions").is_some());
        assert_eq!(
            metadata.properties.get("description"),
            Some(&"UE trajectory prediction model".to_string())
        );
    }

    #[test]
    fn test_model_metadata_getters() {
        let metadata = ModelMetadata::new("test_model", "1.0")
            .with_input(TensorInfo::new("input1", vec![1i64], "float32"))
            .with_input(TensorInfo::new("input2", vec![2i64], "int64"));

        assert!(metadata.get_input("input1").is_some());
        assert!(metadata.get_input("input2").is_some());
        assert!(metadata.get_input("nonexistent").is_none());

        assert!(metadata.get_input_by_index(0).is_some());
        assert!(metadata.get_input_by_index(1).is_some());
        assert!(metadata.get_input_by_index(2).is_none());
    }

    #[test]
    fn test_model_info() {
        let info = ModelInfo::new("/path/to/model.onnx", ModelType::Onnx)
            .with_metadata(ModelMetadata::new("test", "1.0"))
            .with_file_size(1024 * 1024)
            .mark_ready();

        assert!(info.is_ready);
        assert_eq!(info.file_size, 1024 * 1024);
        assert_eq!(info.model_type, ModelType::Onnx);
    }

    #[test]
    fn test_model_type_display() {
        assert_eq!(format!("{}", ModelType::Onnx), "ONNX");
        assert_eq!(format!("{}", ModelType::TfLite), "TFLite");
    }

    #[test]
    fn test_model_registry() {
        let mut registry = ModelRegistry::new();
        assert!(registry.is_empty());

        registry.register(
            "trajectory",
            ModelInfo::new("/models/trajectory.onnx", ModelType::Onnx),
        );
        registry.register(
            "handover",
            ModelInfo::new("/models/handover.onnx", ModelType::Onnx),
        );

        assert_eq!(registry.len(), 2);
        assert!(registry.get("trajectory").is_some());
        assert!(registry.get("handover").is_some());
        assert!(registry.get("nonexistent").is_none());

        let names = registry.list();
        assert!(names.contains(&"trajectory"));
        assert!(names.contains(&"handover"));

        let removed = registry.unregister("trajectory");
        assert!(removed.is_some());
        assert_eq!(registry.len(), 1);
    }
}
