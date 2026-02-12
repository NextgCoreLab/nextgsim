//! `Nnwdaf_MLModelTraining` service (TS 23.288 7.6, Rel-18)
//!
//! Provides ML model training coordination service between NWDAF instances.
//! In Rel-18, NWDAF can request other NWDAFs to train models or share training data.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use crate::analytics_id::AnalyticsId;
use crate::error::NwdafError;

/// Training data sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    /// Feature vector
    pub features: Vec<f64>,
    /// Label (for supervised learning)
    pub label: Option<f64>,
    /// Timestamp when sample was collected
    pub timestamp_ms: u64,
    /// Source identifier
    pub source_id: String,
}

/// Training dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingDataset {
    /// Dataset identifier
    pub dataset_id: String,
    /// Analytics ID this dataset is for
    pub analytics_id: AnalyticsId,
    /// Training samples
    pub samples: Vec<TrainingSample>,
    /// Feature names/descriptions
    pub feature_names: Vec<String>,
    /// Dataset metadata
    pub metadata: HashMap<String, String>,
}

impl TrainingDataset {
    /// Creates a new training dataset
    pub fn new(dataset_id: String, analytics_id: AnalyticsId) -> Self {
        Self {
            dataset_id,
            analytics_id,
            samples: Vec::new(),
            feature_names: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Adds a training sample
    pub fn add_sample(&mut self, sample: TrainingSample) {
        self.samples.push(sample);
    }

    /// Returns the number of samples
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Returns true if the dataset is empty
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Splits dataset into training and validation sets
    pub fn train_test_split(&self, train_ratio: f64) -> (TrainingDataset, TrainingDataset) {
        let split_idx = (self.samples.len() as f64 * train_ratio) as usize;
        let mut train_dataset = self.clone();
        let mut test_dataset = self.clone();

        train_dataset.dataset_id = format!("{}_train", self.dataset_id);
        test_dataset.dataset_id = format!("{}_test", self.dataset_id);

        train_dataset.samples = self.samples[..split_idx].to_vec();
        test_dataset.samples = self.samples[split_idx..].to_vec();

        (train_dataset, test_dataset)
    }
}

/// Model training request (`Nnwdaf_MLModelTraining`)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingRequest {
    /// Analytics ID to train model for
    pub analytics_id: AnalyticsId,
    /// Training dataset (or reference to dataset)
    pub dataset_id: String,
    /// Training configuration
    pub config: TrainingConfig,
    /// Requestor NWDAF instance ID
    pub requestor_id: String,
}

/// Training configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Model architecture type
    pub model_type: ModelArchitecture,
    /// Number of training epochs
    pub epochs: u32,
    /// Learning rate
    pub learning_rate: f64,
    /// Batch size
    pub batch_size: usize,
    /// Early stopping patience
    pub early_stopping_patience: Option<u32>,
    /// Additional hyperparameters
    pub hyperparameters: HashMap<String, String>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model_type: ModelArchitecture::NeuralNetwork,
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            early_stopping_patience: Some(10),
            hyperparameters: HashMap::new(),
        }
    }
}

/// Model architecture types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Fully-connected neural network
    NeuralNetwork,
    /// Recurrent neural network (LSTM/GRU)
    Rnn,
    /// Convolutional neural network
    Cnn,
    /// Transformer-based
    Transformer,
    /// Gradient boosting (`XGBoost`, `LightGBM`)
    GradientBoosting,
}

/// Model training response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTrainingResponse {
    /// Training job identifier
    pub job_id: String,
    /// Whether training was accepted
    pub accepted: bool,
    /// Estimated completion time (ms)
    pub estimated_completion_ms: Option<u64>,
    /// Error message if rejected
    pub error: Option<String>,
}

/// Training job status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingStatus {
    /// Training queued
    Queued,
    /// Training in progress
    InProgress,
    /// Training completed successfully
    Completed,
    /// Training failed
    Failed,
    /// Training cancelled
    Cancelled,
}

/// Training job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingJob {
    /// Job identifier
    pub job_id: String,
    /// Training request
    pub request: ModelTrainingRequest,
    /// Current status
    pub status: TrainingStatus,
    /// Training progress (0.0 to 1.0)
    pub progress: f32,
    /// Current epoch
    pub current_epoch: u32,
    /// Training metrics
    pub metrics: TrainingMetrics,
    /// Start time
    pub start_time_ms: u64,
    /// End time (if completed or failed)
    pub end_time_ms: Option<u64>,
}

/// Training metrics (updated during training)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss
    pub train_loss: f64,
    /// Validation loss
    pub val_loss: Option<f64>,
    /// Training accuracy
    pub train_accuracy: Option<f64>,
    /// Validation accuracy
    pub val_accuracy: Option<f64>,
    /// Best validation loss achieved
    pub best_val_loss: Option<f64>,
}

/// ML Model Training service
///
/// Implements the `Nnwdaf_MLModelTraining` service for distributed model training
/// coordination. This is a Rel-18 feature enabling NWDAF instances to:
/// - Request other NWDAFs to train models
/// - Share training data
/// - Coordinate federated learning scenarios
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 7.6: `Nnwdaf_MLModelTraining` service (Rel-18)
#[derive(Debug)]
pub struct MlModelTrainingService {
    /// Active training jobs
    jobs: HashMap<String, TrainingJob>,
    /// Available datasets
    datasets: HashMap<String, TrainingDataset>,
    /// Next job ID counter
    next_job_id: u64,
}

impl MlModelTrainingService {
    /// Creates a new ML model training service
    pub fn new() -> Self {
        Self {
            jobs: HashMap::new(),
            datasets: HashMap::new(),
            next_job_id: 1,
        }
    }

    /// Handles a model training request
    ///
    /// Validates the request, creates a training job, and returns the job ID.
    /// In a real implementation, this would spawn an async training task.
    pub fn handle_training_request(
        &mut self,
        request: ModelTrainingRequest,
    ) -> ModelTrainingResponse {
        // Check if dataset exists
        if !self.datasets.contains_key(&request.dataset_id) {
            return ModelTrainingResponse {
                job_id: String::new(),
                accepted: false,
                estimated_completion_ms: None,
                error: Some(format!("Dataset {} not found", request.dataset_id)),
            };
        }

        let job_id = format!("train-job-{}", self.next_job_id);
        self.next_job_id += 1;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let job = TrainingJob {
            job_id: job_id.clone(),
            request: request.clone(),
            status: TrainingStatus::Queued,
            progress: 0.0,
            current_epoch: 0,
            metrics: TrainingMetrics::default(),
            start_time_ms: now_ms,
            end_time_ms: None,
        };

        // Estimate completion time based on epochs and dataset size
        let dataset = self.datasets.get(&request.dataset_id).unwrap();
        let samples_per_sec = 1000; // Simplified estimate
        let estimated_time_sec = (dataset.len() * request.config.epochs as usize) / samples_per_sec;
        let estimated_completion_ms = Some(now_ms + (estimated_time_sec as u64 * 1000));

        info!(
            "MLModelTraining: Accepted training job {} for {:?} from {}",
            job_id, request.analytics_id, request.requestor_id
        );

        self.jobs.insert(job_id.clone(), job);

        ModelTrainingResponse {
            job_id,
            accepted: true,
            estimated_completion_ms,
            error: None,
        }
    }

    /// Gets the status of a training job
    pub fn get_job_status(&self, job_id: &str) -> Option<&TrainingJob> {
        self.jobs.get(job_id)
    }

    /// Updates job progress (simulated)
    ///
    /// In a real implementation, this would be called by the training loop.
    pub fn update_job_progress(
        &mut self,
        job_id: &str,
        epoch: u32,
        metrics: TrainingMetrics,
    ) -> Result<(), NwdafError> {
        let job = self.jobs.get_mut(job_id).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: job_id.to_string(),
            }
        })?;

        job.current_epoch = epoch;
        job.metrics = metrics;
        job.progress = epoch as f32 / job.request.config.epochs as f32;
        job.status = TrainingStatus::InProgress;

        debug!("Updated training job {} progress: {:.2}%", job_id, job.progress * 100.0);

        Ok(())
    }

    /// Marks a training job as completed
    pub fn complete_job(&mut self, job_id: &str) -> Result<(), NwdafError> {
        let job = self.jobs.get_mut(job_id).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: job_id.to_string(),
            }
        })?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        job.status = TrainingStatus::Completed;
        job.progress = 1.0;
        job.end_time_ms = Some(now_ms);

        info!("Training job {} completed", job_id);

        Ok(())
    }

    /// Registers a training dataset
    pub fn register_dataset(&mut self, dataset: TrainingDataset) {
        info!(
            "Registered training dataset {} for {:?} ({} samples)",
            dataset.dataset_id,
            dataset.analytics_id,
            dataset.len()
        );
        self.datasets.insert(dataset.dataset_id.clone(), dataset);
    }

    /// Gets a dataset by ID
    pub fn get_dataset(&self, dataset_id: &str) -> Option<&TrainingDataset> {
        self.datasets.get(dataset_id)
    }

    /// Lists all training jobs
    pub fn list_jobs(&self) -> Vec<&TrainingJob> {
        self.jobs.values().collect()
    }

    /// Lists all datasets
    pub fn list_datasets(&self) -> Vec<&TrainingDataset> {
        self.datasets.values().collect()
    }

    /// Returns the number of active jobs
    pub fn active_job_count(&self) -> usize {
        self.jobs
            .values()
            .filter(|j| matches!(j.status, TrainingStatus::Queued | TrainingStatus::InProgress))
            .count()
    }
}

impl Default for MlModelTrainingService {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Rel-18 Distributed ML Training Coordination (TS 23.288 7.6)
// ============================================================================

/// Federated learning aggregation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedStrategy {
    /// `FedAvg`: Weighted average of model parameters
    FedAvg,
    /// `FedProx`: Proximal term for heterogeneous data
    FedProx,
    /// `FedSGD`: Stochastic gradient aggregation
    FedSgd,
}

impl Default for FederatedStrategy {
    fn default() -> Self {
        Self::FedAvg
    }
}

/// Model update from a participant NWDAF instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelUpdate {
    /// Participant NWDAF instance ID
    pub nwdaf_id: String,
    /// Model parameter deltas (flattened weight updates)
    pub weight_deltas: Vec<f64>,
    /// Number of training samples used
    pub num_samples: u64,
    /// Training loss achieved
    pub loss: f64,
    /// Round number this update is for
    pub round: u32,
}

/// Federated learning round state.
#[derive(Debug, Clone)]
pub struct FederatedRound {
    /// Round number
    pub round: u32,
    /// Model updates received from participants
    pub updates: Vec<ModelUpdate>,
    /// Expected number of participants
    pub expected_participants: usize,
    /// Whether aggregation has been performed
    pub aggregated: bool,
    /// Aggregated global model weights (after aggregation)
    pub global_weights: Vec<f64>,
}

/// Distributed ML training coordinator (Rel-18).
///
/// Coordinates federated learning across multiple NWDAF instances.
/// Each NWDAF trains a local model on its data and sends weight updates
/// to the coordinator, which aggregates them into a global model.
#[derive(Debug)]
pub struct DistributedTrainingCoordinator {
    /// Aggregation strategy
    strategy: FederatedStrategy,
    /// Current training round
    current_round: u32,
    /// Active federated rounds
    rounds: HashMap<u32, FederatedRound>,
    /// Registered participant NWDAF instances
    participants: Vec<String>,
    /// Global model weights
    global_weights: Vec<f64>,
    /// Model dimension (number of parameters)
    model_dimension: usize,
}

impl DistributedTrainingCoordinator {
    /// Creates a new distributed training coordinator.
    pub fn new(strategy: FederatedStrategy, model_dimension: usize) -> Self {
        Self {
            strategy,
            current_round: 0,
            rounds: HashMap::new(),
            participants: Vec::new(),
            global_weights: vec![0.0; model_dimension],
            model_dimension,
        }
    }

    /// Registers a participant NWDAF instance.
    pub fn register_participant(&mut self, nwdaf_id: String) {
        if !self.participants.contains(&nwdaf_id) {
            info!("DistributedTraining: Registered participant {}", nwdaf_id);
            self.participants.push(nwdaf_id);
        }
    }

    /// Starts a new federated learning round.
    pub fn start_round(&mut self) -> u32 {
        self.current_round += 1;
        let round = FederatedRound {
            round: self.current_round,
            updates: Vec::new(),
            expected_participants: self.participants.len(),
            aggregated: false,
            global_weights: Vec::new(),
        };
        self.rounds.insert(self.current_round, round);
        info!("DistributedTraining: Started round {} with {} participants",
            self.current_round, self.participants.len());
        self.current_round
    }

    /// Submits a model update from a participant.
    pub fn submit_update(&mut self, update: ModelUpdate) -> Result<(), NwdafError> {
        let round_num = update.round;

        let round = self.rounds.get_mut(&round_num).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: format!("round-{round_num}"),
            }
        })?;

        if round.aggregated {
            return Err(NwdafError::Analytics(
                crate::error::AnalyticsError::TargetNotFound {
                    target: format!("round {round_num} already aggregated"),
                },
            ));
        }

        debug!("DistributedTraining: Received update from {} for round {} (loss={:.4})",
            update.nwdaf_id, update.round, update.loss);
        round.updates.push(update);

        let should_aggregate = round.updates.len() >= round.expected_participants;
        // Release the mutable borrow before calling aggregate_round
        let _ = round;

        // Auto-aggregate when all participants have submitted
        if should_aggregate {
            self.aggregate_round(round_num)?;
        }

        Ok(())
    }

    /// Aggregates model updates for a round using the configured strategy.
    fn aggregate_round(&mut self, round_num: u32) -> Result<(), NwdafError> {
        let round = self.rounds.get_mut(&round_num).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: format!("round-{round_num}"),
            }
        })?;

        if round.updates.is_empty() {
            return Ok(());
        }

        let aggregated = match self.strategy {
            FederatedStrategy::FedAvg => {
                // Weighted average by number of samples
                let total_samples: u64 = round.updates.iter().map(|u| u.num_samples).sum();
                let mut avg = vec![0.0; self.model_dimension];
                for update in &round.updates {
                    let weight = update.num_samples as f64 / total_samples as f64;
                    for (i, delta) in update.weight_deltas.iter().enumerate() {
                        if i < avg.len() {
                            avg[i] += delta * weight;
                        }
                    }
                }
                avg
            }
            FederatedStrategy::FedProx | FederatedStrategy::FedSgd => {
                // Simple average for FedProx/FedSGD (proximal term applied during local training)
                let n = round.updates.len() as f64;
                let mut avg = vec![0.0; self.model_dimension];
                for update in &round.updates {
                    for (i, delta) in update.weight_deltas.iter().enumerate() {
                        if i < avg.len() {
                            avg[i] += delta / n;
                        }
                    }
                }
                avg
            }
        };

        // Apply aggregated update to global model
        for (i, delta) in aggregated.iter().enumerate() {
            if i < self.global_weights.len() {
                self.global_weights[i] += delta;
            }
        }

        round.aggregated = true;
        round.global_weights = self.global_weights.clone();

        let avg_loss: f64 = round.updates.iter().map(|u| u.loss).sum::<f64>()
            / round.updates.len() as f64;
        info!("DistributedTraining: Round {} aggregated ({} updates, avg_loss={:.4})",
            round_num, round.updates.len(), avg_loss);

        Ok(())
    }

    /// Returns the current global model weights.
    pub fn global_weights(&self) -> &[f64] {
        &self.global_weights
    }

    /// Returns the current round number.
    pub fn current_round(&self) -> u32 {
        self.current_round
    }

    /// Returns the number of registered participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }

    /// Returns whether a round has been completed (aggregated).
    pub fn is_round_complete(&self, round: u32) -> bool {
        self.rounds.get(&round).is_some_and(|r| r.aggregated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_dataset_creation() {
        let mut dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::UeMobility);
        assert_eq!(dataset.len(), 0);
        assert!(dataset.is_empty());

        dataset.add_sample(TrainingSample {
            features: vec![1.0, 2.0, 3.0],
            label: Some(0.5),
            timestamp_ms: 1000,
            source_id: "ue-1".to_string(),
        });

        assert_eq!(dataset.len(), 1);
        assert!(!dataset.is_empty());
    }

    #[test]
    fn test_train_test_split() {
        let mut dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::UeMobility);
        for i in 0..100 {
            dataset.add_sample(TrainingSample {
                features: vec![i as f64],
                label: Some(i as f64 * 2.0),
                timestamp_ms: i * 100,
                source_id: "source".to_string(),
            });
        }

        let (train, test) = dataset.train_test_split(0.8);
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_ml_training_service_creation() {
        let service = MlModelTrainingService::new();
        assert_eq!(service.list_jobs().len(), 0);
        assert_eq!(service.list_datasets().len(), 0);
    }

    #[test]
    fn test_register_dataset() {
        let mut service = MlModelTrainingService::new();
        let dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::NfLoad);

        service.register_dataset(dataset);
        assert_eq!(service.list_datasets().len(), 1);
        assert!(service.get_dataset("ds-1").is_some());
    }

    #[test]
    fn test_training_request_no_dataset() {
        let mut service = MlModelTrainingService::new();
        let request = ModelTrainingRequest {
            analytics_id: AnalyticsId::UeMobility,
            dataset_id: "nonexistent".to_string(),
            config: TrainingConfig::default(),
            requestor_id: "nwdaf-1".to_string(),
        };

        let response = service.handle_training_request(request);
        assert!(!response.accepted);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_training_request_success() {
        let mut service = MlModelTrainingService::new();
        let dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::UeMobility);
        service.register_dataset(dataset);

        let request = ModelTrainingRequest {
            analytics_id: AnalyticsId::UeMobility,
            dataset_id: "ds-1".to_string(),
            config: TrainingConfig::default(),
            requestor_id: "nwdaf-1".to_string(),
        };

        let response = service.handle_training_request(request);
        assert!(response.accepted);
        assert!(response.error.is_none());
        assert!(!response.job_id.is_empty());

        let job = service.get_job_status(&response.job_id);
        assert!(job.is_some());
        assert_eq!(job.unwrap().status, TrainingStatus::Queued);
    }

    #[test]
    fn test_job_progress_update() {
        let mut service = MlModelTrainingService::new();
        let dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::NfLoad);
        service.register_dataset(dataset);

        let request = ModelTrainingRequest {
            analytics_id: AnalyticsId::NfLoad,
            dataset_id: "ds-1".to_string(),
            config: TrainingConfig {
                epochs: 10,
                ..Default::default()
            },
            requestor_id: "nwdaf-1".to_string(),
        };

        let response = service.handle_training_request(request);
        let job_id = response.job_id;

        let metrics = TrainingMetrics {
            train_loss: 0.5,
            val_loss: Some(0.6),
            train_accuracy: Some(0.85),
            val_accuracy: Some(0.82),
            best_val_loss: Some(0.6),
        };

        service.update_job_progress(&job_id, 5, metrics.clone()).unwrap();

        let job = service.get_job_status(&job_id).unwrap();
        assert_eq!(job.status, TrainingStatus::InProgress);
        assert_eq!(job.current_epoch, 5);
        assert!((job.progress - 0.5).abs() < 0.01);
        assert_eq!(job.metrics.train_loss, 0.5);
    }

    #[test]
    fn test_complete_job() {
        let mut service = MlModelTrainingService::new();
        let dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::UeMobility);
        service.register_dataset(dataset);

        let request = ModelTrainingRequest {
            analytics_id: AnalyticsId::UeMobility,
            dataset_id: "ds-1".to_string(),
            config: TrainingConfig::default(),
            requestor_id: "nwdaf-1".to_string(),
        };

        let response = service.handle_training_request(request);
        let job_id = response.job_id;

        service.complete_job(&job_id).unwrap();

        let job = service.get_job_status(&job_id).unwrap();
        assert_eq!(job.status, TrainingStatus::Completed);
        assert_eq!(job.progress, 1.0);
        assert!(job.end_time_ms.is_some());
    }

    #[test]
    fn test_active_job_count() {
        let mut service = MlModelTrainingService::new();
        let dataset = TrainingDataset::new("ds-1".to_string(), AnalyticsId::NfLoad);
        service.register_dataset(dataset);

        assert_eq!(service.active_job_count(), 0);

        let request = ModelTrainingRequest {
            analytics_id: AnalyticsId::NfLoad,
            dataset_id: "ds-1".to_string(),
            config: TrainingConfig::default(),
            requestor_id: "nwdaf-1".to_string(),
        };

        service.handle_training_request(request);
        assert_eq!(service.active_job_count(), 1);
    }
}
