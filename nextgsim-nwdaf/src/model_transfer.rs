//! ML Model Transfer Protocol for Inter-NWDAF Communication (A13.12)
//!
//! Implements protocol for transferring ML models between NWDAF instances.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model transfer message types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelTransferMessage {
    /// Request a model from another NWDAF
    ModelRequest {
        /// Requesting NWDAF ID
        requester_id: String,
        /// Model identifier
        model_id: String,
        /// Model type requested
        model_type: ModelType,
    },
    /// Transfer a model to another NWDAF
    ModelTransfer {
        /// Source NWDAF ID
        source_id: String,
        /// Model metadata
        model: ModelPackage,
    },
    /// Acknowledge model receipt
    ModelAck {
        /// Receiving NWDAF ID
        receiver_id: String,
        /// Model ID acknowledged
        model_id: String,
        /// Whether transfer was successful
        success: bool,
    },
    /// Query available models
    ModelQuery {
        /// Querying NWDAF ID
        requester_id: String,
        /// Filter criteria
        filter: ModelFilter,
    },
    /// Response to model query
    ModelQueryResponse {
        /// Available models
        models: Vec<ModelMetadata>,
    },
}

/// ML model type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelType {
    /// Mobility prediction model
    MobilityPrediction,
    /// Load prediction model
    LoadPrediction,
    /// Anomaly detection model
    AnomalyDetection,
    /// QoS prediction model
    QosPrediction,
    /// Energy efficiency model
    EnergyEfficiency,
    /// Network slicing optimization model
    SliceOptimization,
}

impl ModelType {
    /// Returns all model types
    pub fn all() -> &'static [ModelType] {
        &[
            ModelType::MobilityPrediction,
            ModelType::LoadPrediction,
            ModelType::AnomalyDetection,
            ModelType::QosPrediction,
            ModelType::EnergyEfficiency,
            ModelType::SliceOptimization,
        ]
    }
}

/// Model filter criteria
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelFilter {
    /// Filter by model type
    pub model_type: Option<ModelType>,
    /// Minimum accuracy threshold
    pub min_accuracy: Option<f32>,
    /// Maximum model age (seconds)
    pub max_age_secs: Option<u64>,
    /// Filter by training location
    pub training_location: Option<String>,
}

/// ML model package for transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackage {
    /// Model metadata
    pub metadata: ModelMetadata,
    /// Serialized model weights (simplified as Vec<f32>)
    pub weights: Vec<f32>,
    /// Model architecture description
    pub architecture: String,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
    /// Compressed size (bytes)
    pub size_bytes: usize,
}

/// ML model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Unique model identifier
    pub model_id: String,
    /// Model type
    pub model_type: ModelType,
    /// Model version
    pub version: String,
    /// Training timestamp (ms since epoch)
    pub trained_at_ms: u64,
    /// Source NWDAF that trained this model
    pub source_nwdaf_id: String,
    /// Training accuracy
    pub accuracy: f32,
    /// Training loss
    pub loss: f32,
    /// Number of training samples
    pub training_samples: usize,
    /// Model description
    pub description: String,
}

/// Model transfer protocol handler
#[derive(Debug)]
pub struct ModelTransferProtocol {
    /// This NWDAF's identifier
    nwdaf_id: String,
    /// Local model registry
    models: HashMap<String, ModelPackage>,
    /// Peer NWDAF connections
    peers: Vec<String>,
    /// Transfer statistics
    stats: TransferStatistics,
}

impl ModelTransferProtocol {
    /// Creates a new model transfer protocol handler
    pub fn new(nwdaf_id: String) -> Self {
        Self {
            nwdaf_id,
            models: HashMap::new(),
            peers: Vec::new(),
            stats: TransferStatistics::default(),
        }
    }

    /// Registers a peer NWDAF
    pub fn add_peer(&mut self, peer_id: String) {
        if !self.peers.contains(&peer_id) {
            self.peers.push(peer_id);
        }
    }

    /// Registers a local model
    pub fn register_model(&mut self, model: ModelPackage) {
        let model_id = model.metadata.model_id.clone();
        self.models.insert(model_id, model);
    }

    /// Handles incoming model transfer message
    pub fn handle_message(&mut self, message: ModelTransferMessage) -> Option<ModelTransferMessage> {
        match message {
            ModelTransferMessage::ModelRequest {
                requester_id,
                model_id,
                model_type: _,
            } => self.handle_model_request(&requester_id, &model_id),

            ModelTransferMessage::ModelTransfer { source_id: _, model } => {
                self.handle_model_transfer(model)
            }

            ModelTransferMessage::ModelQuery {
                requester_id: _,
                filter,
            } => Some(self.handle_model_query(filter)),

            _ => None,
        }
    }

    /// Handles model request
    fn handle_model_request(&mut self, requester_id: &str, model_id: &str) -> Option<ModelTransferMessage> {
        if let Some(model) = self.models.get(model_id) {
            self.stats.models_sent += 1;
            self.stats.total_bytes_sent += model.size_bytes;

            Some(ModelTransferMessage::ModelTransfer {
                source_id: self.nwdaf_id.clone(),
                model: model.clone(),
            })
        } else {
            // Model not found - send negative acknowledgment
            Some(ModelTransferMessage::ModelAck {
                receiver_id: requester_id.to_string(),
                model_id: model_id.to_string(),
                success: false,
            })
        }
    }

    /// Handles incoming model transfer
    fn handle_model_transfer(&mut self, model: ModelPackage) -> Option<ModelTransferMessage> {
        let model_id = model.metadata.model_id.clone();

        // Check if we should accept this model
        if self.should_accept_model(&model) {
            self.stats.models_received += 1;
            self.stats.total_bytes_received += model.size_bytes;
            self.models.insert(model_id.clone(), model);

            Some(ModelTransferMessage::ModelAck {
                receiver_id: self.nwdaf_id.clone(),
                model_id,
                success: true,
            })
        } else {
            Some(ModelTransferMessage::ModelAck {
                receiver_id: self.nwdaf_id.clone(),
                model_id,
                success: false,
            })
        }
    }

    /// Handles model query
    fn handle_model_query(&self, filter: ModelFilter) -> ModelTransferMessage {
        let mut models: Vec<ModelMetadata> = self
            .models
            .values()
            .filter(|m| self.matches_filter(&m.metadata, &filter))
            .map(|m| m.metadata.clone())
            .collect();

        // Sort by accuracy (descending)
        models.sort_by(|a, b| b.accuracy.partial_cmp(&a.accuracy).unwrap());

        ModelTransferMessage::ModelQueryResponse { models }
    }

    /// Checks if model matches filter criteria
    fn matches_filter(&self, metadata: &ModelMetadata, filter: &ModelFilter) -> bool {
        if let Some(model_type) = filter.model_type {
            if metadata.model_type != model_type {
                return false;
            }
        }

        if let Some(min_accuracy) = filter.min_accuracy {
            if metadata.accuracy < min_accuracy {
                return false;
            }
        }

        if let Some(max_age_secs) = filter.max_age_secs {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64;
            let age_secs = (now - metadata.trained_at_ms) / 1000;
            if age_secs > max_age_secs {
                return false;
            }
        }

        true
    }

    /// Determines if a model should be accepted
    fn should_accept_model(&self, model: &ModelPackage) -> bool {
        // Accept if we don't have this model
        if !self.models.contains_key(&model.metadata.model_id) {
            return true;
        }

        // Accept if the new model is better (higher accuracy)
        if let Some(existing) = self.models.get(&model.metadata.model_id) {
            if model.metadata.accuracy > existing.metadata.accuracy {
                return true;
            }
        }

        false
    }

    /// Creates a model request message
    pub fn create_model_request(&self, model_id: String, model_type: ModelType) -> ModelTransferMessage {
        ModelTransferMessage::ModelRequest {
            requester_id: self.nwdaf_id.clone(),
            model_id,
            model_type,
        }
    }

    /// Creates a model query message
    pub fn create_model_query(&self, filter: ModelFilter) -> ModelTransferMessage {
        ModelTransferMessage::ModelQuery {
            requester_id: self.nwdaf_id.clone(),
            filter,
        }
    }

    /// Gets a model by ID
    pub fn get_model(&self, model_id: &str) -> Option<&ModelPackage> {
        self.models.get(model_id)
    }

    /// Lists all available models
    pub fn list_models(&self) -> Vec<&ModelMetadata> {
        self.models.values().map(|m| &m.metadata).collect()
    }

    /// Returns transfer statistics
    pub fn statistics(&self) -> &TransferStatistics {
        &self.stats
    }

    /// Returns the NWDAF ID
    pub fn nwdaf_id(&self) -> &str {
        &self.nwdaf_id
    }

    /// Returns peer count
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}

/// Transfer statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TransferStatistics {
    /// Number of models sent to peers
    pub models_sent: u64,
    /// Number of models received from peers
    pub models_received: u64,
    /// Total bytes sent
    pub total_bytes_sent: usize,
    /// Total bytes received
    pub total_bytes_received: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_model(model_id: &str, model_type: ModelType, accuracy: f32) -> ModelPackage {
        ModelPackage {
            metadata: ModelMetadata {
                model_id: model_id.to_string(),
                model_type,
                version: "1.0".to_string(),
                trained_at_ms: 1000000,
                source_nwdaf_id: "nwdaf1".to_string(),
                accuracy,
                loss: 0.1,
                training_samples: 10000,
                description: "Test model".to_string(),
            },
            weights: vec![0.1, 0.2, 0.3],
            architecture: "MLP".to_string(),
            hyperparameters: HashMap::new(),
            size_bytes: 1024,
        }
    }

    #[test]
    fn test_model_transfer_protocol_creation() {
        let protocol = ModelTransferProtocol::new("nwdaf1".to_string());
        assert_eq!(protocol.nwdaf_id(), "nwdaf1");
        assert_eq!(protocol.peer_count(), 0);
    }

    #[test]
    fn test_add_peer() {
        let mut protocol = ModelTransferProtocol::new("nwdaf1".to_string());
        protocol.add_peer("nwdaf2".to_string());
        protocol.add_peer("nwdaf3".to_string());

        assert_eq!(protocol.peer_count(), 2);
    }

    #[test]
    fn test_register_model() {
        let mut protocol = ModelTransferProtocol::new("nwdaf1".to_string());
        let model = create_test_model("model1", ModelType::MobilityPrediction, 0.95);

        protocol.register_model(model);

        assert!(protocol.get_model("model1").is_some());
    }

    #[test]
    fn test_model_request_handling() {
        let mut protocol = ModelTransferProtocol::new("nwdaf1".to_string());
        let model = create_test_model("model1", ModelType::MobilityPrediction, 0.95);
        protocol.register_model(model);

        let request = ModelTransferMessage::ModelRequest {
            requester_id: "nwdaf2".to_string(),
            model_id: "model1".to_string(),
            model_type: ModelType::MobilityPrediction,
        };

        let response = protocol.handle_message(request);
        assert!(response.is_some());

        match response.unwrap() {
            ModelTransferMessage::ModelTransfer { source_id, model } => {
                assert_eq!(source_id, "nwdaf1");
                assert_eq!(model.metadata.model_id, "model1");
            }
            _ => panic!("Expected ModelTransfer response"),
        }
    }

    #[test]
    fn test_model_transfer_acceptance() {
        let mut protocol = ModelTransferProtocol::new("nwdaf2".to_string());
        let model = create_test_model("model1", ModelType::LoadPrediction, 0.90);

        let transfer = ModelTransferMessage::ModelTransfer {
            source_id: "nwdaf1".to_string(),
            model: model.clone(),
        };

        let response = protocol.handle_message(transfer);
        assert!(response.is_some());

        match response.unwrap() {
            ModelTransferMessage::ModelAck {
                receiver_id,
                model_id,
                success,
            } => {
                assert_eq!(receiver_id, "nwdaf2");
                assert_eq!(model_id, "model1");
                assert!(success);
            }
            _ => panic!("Expected ModelAck response"),
        }

        // Verify model was added
        assert!(protocol.get_model("model1").is_some());
    }

    #[test]
    fn test_model_query_with_filter() {
        let mut protocol = ModelTransferProtocol::new("nwdaf1".to_string());

        protocol.register_model(create_test_model("model1", ModelType::MobilityPrediction, 0.95));
        protocol.register_model(create_test_model("model2", ModelType::LoadPrediction, 0.85));
        protocol.register_model(create_test_model("model3", ModelType::MobilityPrediction, 0.80));

        let filter = ModelFilter {
            model_type: Some(ModelType::MobilityPrediction),
            min_accuracy: Some(0.90),
            max_age_secs: None,
            training_location: None,
        };

        let query = ModelTransferMessage::ModelQuery {
            requester_id: "nwdaf2".to_string(),
            filter,
        };

        let response = protocol.handle_message(query);
        assert!(response.is_some());

        match response.unwrap() {
            ModelTransferMessage::ModelQueryResponse { models } => {
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].model_id, "model1");
            }
            _ => panic!("Expected ModelQueryResponse"),
        }
    }

    #[test]
    fn test_transfer_statistics() {
        let mut protocol = ModelTransferProtocol::new("nwdaf1".to_string());
        let model = create_test_model("model1", ModelType::AnomalyDetection, 0.92);

        // Register and then transfer
        protocol.register_model(model.clone());

        let request = ModelTransferMessage::ModelRequest {
            requester_id: "nwdaf2".to_string(),
            model_id: "model1".to_string(),
            model_type: ModelType::AnomalyDetection,
        };

        protocol.handle_message(request);

        let stats = protocol.statistics();
        assert_eq!(stats.models_sent, 1);
        assert_eq!(stats.total_bytes_sent, 1024);
    }
}
