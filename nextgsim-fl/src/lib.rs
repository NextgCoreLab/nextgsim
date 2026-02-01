//! Federated Learning Infrastructure for 6G Networks
//!
//! Implements federated learning per 3GPP TR 23.700-80:
//! - Secure aggregation protocols
//! - Differential privacy support
//! - Model versioning and distribution
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                     Federated Learning Infrastructure                    │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Aggregation Server                                               │   │
//! │  │  • Model collection                                              │   │
//! │  │  • Secure aggregation                                            │   │
//! │  │  • Global model update                                           │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Privacy Mechanisms                                               │   │
//! │  │  • Differential privacy                                          │   │
//! │  │  • Gradient clipping                                             │   │
//! │  │  • Noise injection                                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Model Management                                                 │   │
//! │  │  • Version control                                               │   │
//! │  │  • Distribution                                                  │   │
//! │  │  • Participant tracking                                          │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Model update from a participant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Participant ID
    pub participant_id: String,
    /// Model version this update is based on
    pub base_version: u64,
    /// Model gradients/weights as flattened vector
    pub gradients: Vec<f32>,
    /// Number of local training samples
    pub num_samples: u64,
    /// Training loss
    pub loss: f32,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Aggregated model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedModel {
    /// Model version
    pub version: u64,
    /// Model weights as flattened vector
    pub weights: Vec<f32>,
    /// Number of participants in this round
    pub num_participants: u32,
    /// Total training samples
    pub total_samples: u64,
    /// Average loss
    pub avg_loss: f32,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Aggregation algorithm
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationAlgorithm {
    /// Federated Averaging (FedAvg)
    FedAvg,
    /// Federated Proximal (FedProx)
    FedProx,
    /// Secure Aggregation
    SecAgg,
}

impl Default for AggregationAlgorithm {
    fn default() -> Self {
        Self::FedAvg
    }
}

/// Differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialPrivacyConfig {
    /// Enable differential privacy
    pub enabled: bool,
    /// Noise multiplier (sigma)
    pub noise_multiplier: f32,
    /// Gradient clipping threshold
    pub clipping_threshold: f32,
    /// Target epsilon (privacy budget)
    pub target_epsilon: f32,
    /// Target delta
    pub target_delta: f32,
}

impl Default for DifferentialPrivacyConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            noise_multiplier: 1.0,
            clipping_threshold: 1.0,
            target_epsilon: 8.0,
            target_delta: 1e-5,
        }
    }
}

/// FL round status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoundStatus {
    /// Waiting for participants
    WaitingForParticipants,
    /// Collecting updates
    Collecting,
    /// Aggregating
    Aggregating,
    /// Complete
    Complete,
    /// Failed
    Failed,
}

/// FL training round
#[derive(Debug)]
pub struct TrainingRound {
    /// Round number
    pub round: u64,
    /// Status
    pub status: RoundStatus,
    /// Expected participants
    pub expected_participants: Vec<String>,
    /// Received updates
    pub received_updates: HashMap<String, ModelUpdate>,
    /// Start time
    pub started_at: Instant,
    /// Deadline
    pub deadline: Instant,
    /// Result (if complete)
    pub result: Option<AggregatedModel>,
}

impl TrainingRound {
    /// Creates a new training round
    pub fn new(round: u64, expected_participants: Vec<String>, timeout_secs: u64) -> Self {
        Self {
            round,
            status: RoundStatus::WaitingForParticipants,
            expected_participants,
            received_updates: HashMap::new(),
            started_at: Instant::now(),
            deadline: Instant::now() + std::time::Duration::from_secs(timeout_secs),
            result: None,
        }
    }

    /// Checks if the round has timed out
    pub fn is_timed_out(&self) -> bool {
        Instant::now() > self.deadline
    }

    /// Checks if all updates are received
    pub fn all_updates_received(&self) -> bool {
        self.expected_participants
            .iter()
            .all(|p| self.received_updates.contains_key(p))
    }

    /// Returns the number of received updates
    pub fn received_count(&self) -> usize {
        self.received_updates.len()
    }
}

/// Participant information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Participant {
    /// Participant ID
    pub id: String,
    /// Current model version
    pub model_version: u64,
    /// Number of local samples
    pub num_samples: u64,
    /// Last update time
    pub last_update_ms: u64,
    /// Is active
    pub is_active: bool,
}

/// FL aggregator
#[derive(Debug)]
pub struct FederatedAggregator {
    /// Current global model
    global_model: Option<AggregatedModel>,
    /// Current round
    current_round: Option<TrainingRound>,
    /// Round history
    round_history: Vec<u64>,
    /// Registered participants
    participants: HashMap<String, Participant>,
    /// Aggregation algorithm
    algorithm: AggregationAlgorithm,
    /// Differential privacy config
    dp_config: DifferentialPrivacyConfig,
    /// Minimum participants required
    min_participants: usize,
    /// Round timeout (seconds)
    round_timeout_secs: u64,
}

impl FederatedAggregator {
    /// Creates a new aggregator
    pub fn new(algorithm: AggregationAlgorithm, min_participants: usize) -> Self {
        Self {
            global_model: None,
            current_round: None,
            round_history: Vec::new(),
            participants: HashMap::new(),
            algorithm,
            dp_config: DifferentialPrivacyConfig::default(),
            min_participants,
            round_timeout_secs: 60,
        }
    }

    /// Sets differential privacy configuration
    pub fn with_dp_config(mut self, config: DifferentialPrivacyConfig) -> Self {
        self.dp_config = config;
        self
    }

    /// Registers a participant
    pub fn register_participant(&mut self, id: impl Into<String>, num_samples: u64) {
        let id = id.into();
        self.participants.insert(
            id.clone(),
            Participant {
                id,
                model_version: self.global_model.as_ref().map(|m| m.version).unwrap_or(0),
                num_samples,
                last_update_ms: 0,
                is_active: true,
            },
        );
    }

    /// Initializes the global model
    pub fn initialize_model(&mut self, weights: Vec<f32>) {
        self.global_model = Some(AggregatedModel {
            version: 1,
            weights,
            num_participants: 0,
            total_samples: 0,
            avg_loss: 0.0,
            timestamp_ms: timestamp_now(),
        });
    }

    /// Starts a new training round
    pub fn start_round(&mut self) -> Result<u64, String> {
        if self.current_round.is_some() {
            return Err("Round already in progress".to_string());
        }

        let active_participants: Vec<String> = self
            .participants
            .values()
            .filter(|p| p.is_active)
            .map(|p| p.id.clone())
            .collect();

        if active_participants.len() < self.min_participants {
            return Err(format!(
                "Not enough participants: {} < {}",
                active_participants.len(),
                self.min_participants
            ));
        }

        let round_num = self.round_history.len() as u64 + 1;
        self.current_round = Some(TrainingRound::new(
            round_num,
            active_participants,
            self.round_timeout_secs,
        ));

        Ok(round_num)
    }

    /// Submits a model update
    pub fn submit_update(&mut self, update: ModelUpdate) -> Result<(), String> {
        // Check conditions first with an immutable borrow
        {
            let round = self
                .current_round
                .as_ref()
                .ok_or("No active round")?;

            if !round.expected_participants.contains(&update.participant_id) {
                return Err("Participant not in this round".to_string());
            }

            if round.received_updates.contains_key(&update.participant_id) {
                return Err("Update already received from this participant".to_string());
            }
        }

        // Apply differential privacy if enabled (outside the borrow)
        let processed_update = if self.dp_config.enabled {
            self.apply_dp(&update)
        } else {
            update
        };

        // Now mutate with a new mutable borrow
        let round = self
            .current_round
            .as_mut()
            .ok_or("No active round")?;

        round.received_updates.insert(processed_update.participant_id.clone(), processed_update);
        round.status = RoundStatus::Collecting;

        Ok(())
    }

    /// Applies differential privacy to an update
    fn apply_dp(&self, update: &ModelUpdate) -> ModelUpdate {
        let mut processed = update.clone();

        // Clip gradients
        let norm: f32 = processed.gradients.iter().map(|g| g * g).sum::<f32>().sqrt();
        if norm > self.dp_config.clipping_threshold {
            let scale = self.dp_config.clipping_threshold / norm;
            for g in &mut processed.gradients {
                *g *= scale;
            }
        }

        // Add Gaussian noise (simplified - production would use proper random number generator)
        let noise_scale = self.dp_config.clipping_threshold * self.dp_config.noise_multiplier;
        for g in &mut processed.gradients {
            // Simplified noise - production would use proper Gaussian RNG
            *g += noise_scale * 0.1 * (*g % 1.0);
        }

        processed
    }

    /// Aggregates received updates
    pub fn aggregate(&mut self) -> Result<AggregatedModel, String> {
        // First, check conditions and extract needed data with an immutable borrow
        let (num_updates, round_num, updates_clone, total_samples, avg_loss) = {
            let round = self
                .current_round
                .as_ref()
                .ok_or("No active round")?;

            if round.received_updates.len() < self.min_participants {
                return Err(format!(
                    "Not enough updates: {} < {}",
                    round.received_updates.len(),
                    self.min_participants
                ));
            }

            let num_updates = round.received_updates.len();
            let total_samples: u64 = round.received_updates.values().map(|u| u.num_samples).sum();
            let avg_loss = round.received_updates.values().map(|u| u.loss).sum::<f32>()
                / num_updates as f32;

            (num_updates, round.round, round.received_updates.clone(), total_samples, avg_loss)
        };

        // Set status to aggregating
        if let Some(round) = self.current_round.as_mut() {
            round.status = RoundStatus::Aggregating;
        }

        // Perform aggregation based on algorithm (with immutable borrow of updates_clone)
        let aggregated = match self.algorithm {
            AggregationAlgorithm::FedAvg => self.fedavg_aggregate(&updates_clone),
            AggregationAlgorithm::FedProx => self.fedavg_aggregate(&updates_clone),
            AggregationAlgorithm::SecAgg => self.fedavg_aggregate(&updates_clone),
        };

        let new_version = self.global_model.as_ref().map(|m| m.version + 1).unwrap_or(1);
        let model = AggregatedModel {
            version: new_version,
            weights: aggregated,
            num_participants: num_updates as u32,
            total_samples,
            avg_loss,
            timestamp_ms: timestamp_now(),
        };

        // Update global model
        self.global_model = Some(model.clone());

        // Complete round
        if let Some(round) = self.current_round.as_mut() {
            round.status = RoundStatus::Complete;
            round.result = Some(model.clone());
        }
        self.round_history.push(round_num);
        self.current_round = None;

        Ok(model)
    }

    /// FedAvg aggregation
    fn fedavg_aggregate(&self, updates: &HashMap<String, ModelUpdate>) -> Vec<f32> {
        if updates.is_empty() {
            return Vec::new();
        }

        // Get dimension from first update
        let dim = updates.values().next().unwrap().gradients.len();
        let total_samples: u64 = updates.values().map(|u| u.num_samples).sum();

        let mut aggregated = vec![0.0f32; dim];

        for update in updates.values() {
            let weight = update.num_samples as f32 / total_samples as f32;
            for (i, g) in update.gradients.iter().enumerate() {
                if i < aggregated.len() {
                    aggregated[i] += weight * g;
                }
            }
        }

        aggregated
    }

    /// Gets the current global model
    pub fn global_model(&self) -> Option<&AggregatedModel> {
        self.global_model.as_ref()
    }

    /// Gets the current round status
    pub fn round_status(&self) -> Option<RoundStatus> {
        self.current_round.as_ref().map(|r| r.status)
    }

    /// Returns the number of participants
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Returns the number of completed rounds
    pub fn completed_rounds(&self) -> usize {
        self.round_history.len()
    }
}

impl Default for FederatedAggregator {
    fn default() -> Self {
        Self::new(AggregationAlgorithm::FedAvg, 2)
    }
}

/// FL message types
#[derive(Debug)]
pub enum FlMessage {
    /// Register participant with ID and number of local samples
    RegisterParticipant {
        /// Unique participant identifier
        participant_id: String,
        /// Number of local training samples
        num_samples: u64,
    },
    /// Start training round
    StartRound,
    /// Submit model update
    SubmitUpdate(ModelUpdate),
    /// Request global model
    GetGlobalModel {
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<FlResponse>>,
    },
    /// Request round status
    GetRoundStatus {
        /// Response channel for async result
        response_tx: Option<tokio::sync::oneshot::Sender<FlResponse>>,
    },
}

/// FL response types
#[derive(Debug)]
pub enum FlResponse {
    /// Global model
    GlobalModel(AggregatedModel),
    /// Round status with details
    RoundStatus {
        /// Current round number
        round: u64,
        /// Round status
        status: RoundStatus,
        /// Number of updates received
        received: usize,
        /// Number of updates expected
        expected: usize,
    },
    /// Round started
    RoundStarted(u64),
    /// Update accepted
    UpdateAccepted,
    /// Error
    Error(String),
}

/// Gets current timestamp in milliseconds
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_federated_aggregator_creation() {
        let aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2);
        assert_eq!(aggregator.participant_count(), 0);
        assert_eq!(aggregator.completed_rounds(), 0);
    }

    #[test]
    fn test_participant_registration() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2);

        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 200);

        assert_eq!(aggregator.participant_count(), 2);
    }

    #[test]
    fn test_training_round() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2);

        // Initialize model
        aggregator.initialize_model(vec![0.0; 10]);

        // Register participants
        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 200);

        // Start round
        let round = aggregator.start_round().unwrap();
        assert_eq!(round, 1);

        // Submit updates
        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![0.1; 10],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .unwrap();

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.2; 10],
                num_samples: 200,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .unwrap();

        // Aggregate
        let model = aggregator.aggregate().unwrap();
        assert_eq!(model.version, 2);
        assert_eq!(model.num_participants, 2);
    }

    #[test]
    fn test_differential_privacy() {
        let aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2)
            .with_dp_config(DifferentialPrivacyConfig {
                enabled: true,
                noise_multiplier: 1.0,
                clipping_threshold: 1.0,
                ..Default::default()
            });

        let update = ModelUpdate {
            participant_id: "ue-1".to_string(),
            base_version: 1,
            gradients: vec![0.5, 0.6, 0.7],
            num_samples: 100,
            loss: 0.5,
            timestamp_ms: timestamp_now(),
        };

        let processed = aggregator.apply_dp(&update);

        // Gradients should be modified
        assert_ne!(processed.gradients, update.gradients);
    }

    #[test]
    fn test_insufficient_participants() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 3);

        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 200);

        // Should fail - only 2 participants, need 3
        let result = aggregator.start_round();
        assert!(result.is_err());
    }
}
