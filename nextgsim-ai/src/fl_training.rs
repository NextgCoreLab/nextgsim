//! Federated Learning training loop implementation
//!
//! This module implements the execution engine for Federated Learning,
//! supporting FedAvg and FedProx aggregation algorithms with optional
//! differential privacy.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;
use tracing::{debug, info, warn};

use crate::config::{AggregationAlgorithm, FlConfig};
use crate::tensor::TensorData;

/// Errors that can occur during FL training
#[derive(Error, Debug)]
pub enum FlError {
    /// Not enough participants for aggregation
    #[error("Insufficient participants: got {actual}, need at least {minimum}")]
    InsufficientParticipants { actual: usize, minimum: usize },

    /// Model update failed
    #[error("Model update failed: {reason}")]
    UpdateFailed { reason: String },

    /// Aggregation failed
    #[error("Aggregation failed: {reason}")]
    AggregationFailed { reason: String },

    /// Training round exceeded maximum iterations
    #[error("Training exceeded maximum rounds: {max_rounds}")]
    MaxRoundsExceeded { max_rounds: usize },

    /// Participant dropped out
    #[error("Participant {participant_id} dropped out: {reason}")]
    ParticipantDropout {
        participant_id: String,
        reason: String,
    },

    /// Invalid model weights
    #[error("Invalid model weights: {reason}")]
    InvalidWeights { reason: String },
}

impl From<String> for FlError {
    fn from(reason: String) -> Self {
        FlError::AggregationFailed { reason }
    }
}

/// Participant in federated learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlParticipant {
    /// Unique participant ID
    pub id: String,
    /// Number of local training samples
    pub num_samples: usize,
    /// Current model weights
    pub weights: HashMap<String, TensorData>,
    /// Participant status
    pub status: ParticipantStatus,
    /// Last update timestamp
    pub last_update: u64,
}

/// Status of an FL participant
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticipantStatus {
    /// Participant is active and training
    Active,
    /// Participant is waiting for aggregation
    Waiting,
    /// Participant has completed training
    Completed,
    /// Participant has dropped out
    Dropped,
}

/// Result of a training round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundResult {
    /// Round number
    pub round: usize,
    /// Aggregated model weights
    pub global_weights: HashMap<String, TensorData>,
    /// Number of participants
    pub num_participants: usize,
    /// Total samples across all participants
    pub total_samples: usize,
    /// Round duration
    pub duration: Duration,
    /// Average loss (if available)
    pub avg_loss: Option<f32>,
}

/// Federated Learning trainer
pub struct FlTrainer {
    /// Configuration
    config: FlConfig,
    /// Registered participants
    participants: HashMap<String, FlParticipant>,
    /// Global model weights
    global_weights: HashMap<String, TensorData>,
    /// Current round number
    current_round: usize,
    /// Training history
    history: Vec<RoundResult>,
    /// Training start time
    start_time: Option<Instant>,
}

impl FlTrainer {
    /// Creates a new FL trainer with the given configuration
    pub fn new(config: FlConfig) -> Self {
        Self {
            config,
            participants: HashMap::new(),
            global_weights: HashMap::new(),
            current_round: 0,
            history: Vec::new(),
            start_time: None,
        }
    }

    /// Registers a new participant
    pub fn register_participant(&mut self, participant: FlParticipant) {
        info!("Registering FL participant: {}", participant.id);
        self.participants.insert(participant.id.clone(), participant);
    }

    /// Removes a participant
    pub fn remove_participant(&mut self, participant_id: &str) -> Option<FlParticipant> {
        info!("Removing FL participant: {}", participant_id);
        self.participants.remove(participant_id)
    }

    /// Returns the number of active participants
    pub fn active_participant_count(&self) -> usize {
        self.participants
            .values()
            .filter(|p| p.status == ParticipantStatus::Active || p.status == ParticipantStatus::Waiting)
            .count()
    }

    /// Initializes global model weights
    pub fn initialize_global_weights(&mut self, weights: HashMap<String, TensorData>) {
        info!("Initializing global model with {} layers", weights.len());
        self.global_weights = weights;
    }

    /// Returns a reference to the global model weights
    pub fn global_weights(&self) -> &HashMap<String, TensorData> {
        &self.global_weights
    }

    /// Submits local model update from a participant
    pub fn submit_update(
        &mut self,
        participant_id: &str,
        weights: HashMap<String, TensorData>,
    ) -> Result<(), FlError> {
        let participant = self
            .participants
            .get_mut(participant_id)
            .ok_or_else(|| FlError::UpdateFailed {
                reason: format!("Participant {participant_id} not found"),
            })?;

        // Validate weights match global model structure
        if weights.keys().len() != self.global_weights.keys().len() {
            return Err(FlError::InvalidWeights {
                reason: format!(
                    "Weight count mismatch: expected {}, got {}",
                    self.global_weights.keys().len(),
                    weights.keys().len()
                ),
            });
        }

        participant.weights = weights;
        participant.status = ParticipantStatus::Waiting;
        participant.last_update = current_timestamp_ms();

        debug!(
            "Received update from participant {} with {} samples",
            participant_id, participant.num_samples
        );

        Ok(())
    }

    /// Runs one round of federated learning
    ///
    /// Returns the aggregated global model and round statistics
    pub fn run_round(&mut self) -> Result<RoundResult, FlError> {
        let round_start = Instant::now();
        self.current_round += 1;

        if self.start_time.is_none() {
            self.start_time = Some(Instant::now());
        }

        info!("Starting FL round {}", self.current_round);

        // Check if we have enough participants
        let active_count = self.active_participant_count();
        if active_count < self.config.min_participants {
            return Err(FlError::InsufficientParticipants {
                actual: active_count,
                minimum: self.config.min_participants,
            });
        }

        // Check round limit
        if self.current_round > self.config.max_rounds {
            return Err(FlError::MaxRoundsExceeded {
                max_rounds: self.config.max_rounds,
            });
        }

        // Collect updates from waiting participants
        let waiting_participants: Vec<_> = self
            .participants
            .values()
            .filter(|p| p.status == ParticipantStatus::Waiting)
            .cloned()
            .collect();

        if waiting_participants.is_empty() {
            warn!("No participants ready for aggregation in round {}", self.current_round);
            return Err(FlError::InsufficientParticipants {
                actual: 0,
                minimum: self.config.min_participants,
            });
        }

        // Perform aggregation
        let aggregated_weights = match self.config.aggregation_algorithm {
            AggregationAlgorithm::FedAvg => {
                self.fedavg_aggregate(&waiting_participants)?
            }
            AggregationAlgorithm::FedProx => {
                self.fedprox_aggregate(&waiting_participants)?
            }
            AggregationAlgorithm::SecAgg => {
                self.secure_aggregate(&waiting_participants)?
            }
        };

        // Apply differential privacy if enabled
        let final_weights = if self.config.enable_differential_privacy {
            self.apply_differential_privacy(aggregated_weights)?
        } else {
            aggregated_weights
        };

        // Update global model
        self.global_weights = final_weights.clone();

        // Calculate statistics
        let total_samples: usize = waiting_participants.iter().map(|p| p.num_samples).sum();
        let num_participants = waiting_participants.len();

        let round_result = RoundResult {
            round: self.current_round,
            global_weights: final_weights,
            num_participants,
            total_samples,
            duration: round_start.elapsed(),
            avg_loss: None, // Would be computed from participant metrics in production
        };

        // Reset participant status
        for participant in self.participants.values_mut() {
            if participant.status == ParticipantStatus::Waiting {
                participant.status = ParticipantStatus::Active;
            }
        }

        self.history.push(round_result.clone());

        info!(
            "FL round {} completed: {} participants, {} total samples, {:?}",
            self.current_round, num_participants, total_samples, round_result.duration
        );

        Ok(round_result)
    }

    /// FedAvg aggregation: weighted average based on number of samples
    fn fedavg_aggregate(
        &self,
        participants: &[FlParticipant],
    ) -> Result<HashMap<String, TensorData>, FlError> {
        debug!("Performing FedAvg aggregation");

        let total_samples: usize = participants.iter().map(|p| p.num_samples).sum();
        if total_samples == 0 {
            return Err(FlError::AggregationFailed {
                reason: "Total samples is zero".to_string(),
            });
        }

        let mut aggregated = HashMap::new();

        // For each layer in the model
        for layer_name in self.global_weights.keys() {
            // Weighted sum of participant weights
            let mut weighted_sum: Option<TensorData> = None;

            for participant in participants {
                let weight = participant
                    .weights
                    .get(layer_name)
                    .ok_or_else(|| FlError::InvalidWeights {
                        reason: format!(
                            "Participant {} missing weights for layer {}",
                            participant.id, layer_name
                        ),
                    })?;

                let weight_factor = participant.num_samples as f32 / total_samples as f32;
                let scaled_weight = weight.scale(weight_factor);

                weighted_sum = match weighted_sum {
                    None => Some(scaled_weight),
                    Some(sum) => Some(sum.add(&scaled_weight)?),
                };
            }

            aggregated.insert(
                layer_name.clone(),
                weighted_sum.ok_or_else(|| FlError::AggregationFailed {
                    reason: format!("Failed to aggregate layer {layer_name}"),
                })?,
            );
        }

        Ok(aggregated)
    }

    /// FedProx aggregation: FedAvg with proximal correction toward global model
    ///
    /// FedProx adds a proximal term `(mu/2) * ||w - w_global||^2` to each
    /// participant's local objective. At aggregation time we first compute the
    /// FedAvg-style weighted average and then shift the result back toward the
    /// current global model:
    ///
    ///   w_new = (1 / (1 + mu)) * w_fedavg + (mu / (1 + mu)) * w_global
    ///
    /// This stabilises convergence under heterogeneous data distributions.
    fn fedprox_aggregate(
        &self,
        participants: &[FlParticipant],
    ) -> Result<HashMap<String, TensorData>, FlError> {
        debug!("Performing FedProx aggregation with mu={}", self.config.fedprox_mu);

        // Start with FedAvg-style weighted average
        let fedavg_result = self.fedavg_aggregate(participants)?;

        let mu = self.config.fedprox_mu;
        if mu <= 0.0 || self.global_weights.is_empty() {
            // No proximal correction needed
            return Ok(fedavg_result);
        }

        let correction_factor = 1.0 / (1.0 + mu);
        let global_factor = mu / (1.0 + mu);

        let mut result = HashMap::new();
        for (layer_name, fedavg_tensor) in &fedavg_result {
            let global_tensor = match self.global_weights.get(layer_name) {
                Some(t) => t,
                None => {
                    // No global weight for this layer; keep FedAvg result
                    result.insert(layer_name.clone(), fedavg_tensor.clone());
                    continue;
                }
            };

            // w_new = correction_factor * w_fedavg + global_factor * w_global
            let scaled_fedavg = fedavg_tensor.scale(correction_factor);
            let scaled_global = global_tensor.scale(global_factor);
            let combined = scaled_fedavg.add(&scaled_global).map_err(|e| {
                FlError::AggregationFailed {
                    reason: format!("FedProx proximal correction failed for layer {layer_name}: {e}"),
                }
            })?;
            result.insert(layer_name.clone(), combined);
        }

        Ok(result)
    }

    /// Secure aggregation: uses cryptographic techniques to protect individual updates
    fn secure_aggregate(
        &self,
        participants: &[FlParticipant],
    ) -> Result<HashMap<String, TensorData>, FlError> {
        debug!("Performing secure aggregation");

        // Secure aggregation would implement:
        // 1. Participant secret sharing
        // 2. Masked model updates
        // 3. Secure unmasking after all updates received
        //
        // For now, we use FedAvg as a placeholder
        // In production, this would implement a protocol like SecAgg or SecAgg+
        warn!("Secure aggregation not fully implemented, using FedAvg");
        self.fedavg_aggregate(participants)
    }

    /// Applies differential privacy to aggregated weights
    fn apply_differential_privacy(
        &self,
        mut weights: HashMap<String, TensorData>,
    ) -> Result<HashMap<String, TensorData>, FlError> {
        debug!(
            "Applying differential privacy with noise multiplier {}",
            self.config.dp_noise_multiplier
        );

        // Apply Gaussian noise to each layer for differential privacy
        for (_layer_name, tensor) in weights.iter_mut() {
            *tensor = tensor.add_gaussian_noise(
                self.config.dp_noise_multiplier,
                self.config.dp_clipping_threshold,
            );
        }

        Ok(weights)
    }

    /// Returns the training history
    pub fn history(&self) -> &[RoundResult] {
        &self.history
    }

    /// Returns the current round number
    pub fn current_round(&self) -> usize {
        self.current_round
    }

    /// Returns true if training has reached max rounds
    pub fn is_training_complete(&self) -> bool {
        self.current_round >= self.config.max_rounds
    }

    /// Resets the trainer state
    pub fn reset(&mut self) {
        self.current_round = 0;
        self.history.clear();
        self.start_time = None;
        for participant in self.participants.values_mut() {
            participant.status = ParticipantStatus::Active;
        }
    }
}

/// Returns current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::TensorData;

    fn create_test_config() -> FlConfig {
        FlConfig {
            enabled: true,
            aggregation_algorithm: AggregationAlgorithm::FedAvg,
            min_participants: 2,
            max_rounds: 10,
            enable_differential_privacy: false,
            dp_noise_multiplier: 1.0,
            dp_clipping_threshold: 1.0,
            fedprox_mu: 0.01,
        }
    }

    fn create_test_participant(id: &str, num_samples: usize) -> FlParticipant {
        let mut weights = HashMap::new();
        weights.insert(
            "layer1".to_string(),
            TensorData::float32(vec![1.0, 2.0, 3.0], vec![3]),
        );
        weights.insert(
            "layer2".to_string(),
            TensorData::float32(vec![4.0, 5.0], vec![2]),
        );

        FlParticipant {
            id: id.to_string(),
            num_samples,
            weights,
            status: ParticipantStatus::Active,
            last_update: current_timestamp_ms(),
        }
    }

    #[test]
    fn test_trainer_creation() {
        let config = create_test_config();
        let trainer = FlTrainer::new(config);
        assert_eq!(trainer.current_round(), 0);
        assert_eq!(trainer.active_participant_count(), 0);
    }

    #[test]
    fn test_participant_registration() {
        let config = create_test_config();
        let mut trainer = FlTrainer::new(config);

        let participant = create_test_participant("participant-1", 100);
        trainer.register_participant(participant);

        assert_eq!(trainer.active_participant_count(), 1);
    }

    #[test]
    fn test_insufficient_participants() {
        let config = create_test_config();
        let mut trainer = FlTrainer::new(config);

        // Initialize global weights
        let mut global_weights = HashMap::new();
        global_weights.insert(
            "layer1".to_string(),
            TensorData::float32(vec![0.0, 0.0, 0.0], vec![3]),
        );
        global_weights.insert(
            "layer2".to_string(),
            TensorData::float32(vec![0.0, 0.0], vec![2]),
        );
        trainer.initialize_global_weights(global_weights);

        // Only 1 participant, need at least 2
        let participant = create_test_participant("participant-1", 100);
        trainer.register_participant(participant);

        let result = trainer.run_round();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FlError::InsufficientParticipants { .. }
        ));
    }

    #[test]
    fn test_submit_update() {
        let config = create_test_config();
        let mut trainer = FlTrainer::new(config);

        // Initialize global weights
        let mut global_weights = HashMap::new();
        global_weights.insert(
            "layer1".to_string(),
            TensorData::float32(vec![0.0, 0.0, 0.0], vec![3]),
        );
        global_weights.insert(
            "layer2".to_string(),
            TensorData::float32(vec![0.0, 0.0], vec![2]),
        );
        trainer.initialize_global_weights(global_weights);

        let participant = create_test_participant("participant-1", 100);
        trainer.register_participant(participant.clone());

        let result = trainer.submit_update("participant-1", participant.weights);
        assert!(result.is_ok());

        let p = trainer.participants.get("participant-1").unwrap();
        assert_eq!(p.status, ParticipantStatus::Waiting);
    }

    #[test]
    fn test_fedavg_round() {
        let config = create_test_config();
        let mut trainer = FlTrainer::new(config);

        // Initialize global weights
        let mut global_weights = HashMap::new();
        global_weights.insert(
            "layer1".to_string(),
            TensorData::float32(vec![0.0, 0.0, 0.0], vec![3]),
        );
        global_weights.insert(
            "layer2".to_string(),
            TensorData::float32(vec![0.0, 0.0], vec![2]),
        );
        trainer.initialize_global_weights(global_weights);

        // Register participants
        let p1 = create_test_participant("participant-1", 100);
        let p2 = create_test_participant("participant-2", 200);

        trainer.register_participant(p1.clone());
        trainer.register_participant(p2.clone());

        // Submit updates
        trainer.submit_update("participant-1", p1.weights).unwrap();
        trainer.submit_update("participant-2", p2.weights).unwrap();

        // Run aggregation
        let result = trainer.run_round();
        assert!(result.is_ok());

        let round_result = result.unwrap();
        assert_eq!(round_result.round, 1);
        assert_eq!(round_result.num_participants, 2);
        assert_eq!(round_result.total_samples, 300);
    }

    #[test]
    fn test_max_rounds() {
        let mut config = create_test_config();
        config.max_rounds = 2;
        let mut trainer = FlTrainer::new(config);

        // Initialize global weights
        let mut global_weights = HashMap::new();
        global_weights.insert(
            "layer1".to_string(),
            TensorData::float32(vec![0.0, 0.0, 0.0], vec![3]),
        );
        global_weights.insert(
            "layer2".to_string(),
            TensorData::float32(vec![0.0, 0.0], vec![2]),
        );
        trainer.initialize_global_weights(global_weights);

        let p1 = create_test_participant("participant-1", 100);
        let p2 = create_test_participant("participant-2", 100);

        trainer.register_participant(p1.clone());
        trainer.register_participant(p2.clone());

        // Round 1
        trainer.submit_update("participant-1", p1.weights.clone()).unwrap();
        trainer.submit_update("participant-2", p2.weights.clone()).unwrap();
        assert!(trainer.run_round().is_ok());

        // Round 2
        trainer.submit_update("participant-1", p1.weights.clone()).unwrap();
        trainer.submit_update("participant-2", p2.weights).unwrap();
        assert!(trainer.run_round().is_ok());

        assert!(trainer.is_training_complete());

        // Round 3 should fail
        trainer.submit_update("participant-1", p1.weights).unwrap();
        let result = trainer.run_round();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            FlError::MaxRoundsExceeded { .. }
        ));
    }

    #[test]
    fn test_reset() {
        let config = create_test_config();
        let mut trainer = FlTrainer::new(config);

        let participant = create_test_participant("participant-1", 100);
        trainer.register_participant(participant);

        // Simulate some rounds
        trainer.current_round = 5;
        trainer.history.push(RoundResult {
            round: 1,
            global_weights: HashMap::new(),
            num_participants: 1,
            total_samples: 100,
            duration: Duration::from_secs(1),
            avg_loss: None,
        });

        trainer.reset();

        assert_eq!(trainer.current_round(), 0);
        assert_eq!(trainer.history().len(), 0);
        assert!(!trainer.is_training_complete());
    }
}
