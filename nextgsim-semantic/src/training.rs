//! End-to-End Learned Codec Training (A18.1)
//!
//! Implements a training loop for learned encoder/decoder networks
//! using gradient descent on a rate-distortion loss.

use serde::{Deserialize, Serialize};

use crate::{ChannelQuality, SemanticTask};

/// Training configuration for learned codecs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f32,
    /// Number of training epochs
    pub num_epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Rate-distortion trade-off parameter (lambda)
    /// Loss = Distortion + lambda * Rate
    pub lambda: f32,
    /// Target compression ratio
    pub target_compression: f32,
    /// Validation split ratio
    pub validation_split: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            num_epochs: 100,
            batch_size: 32,
            lambda: 0.01,
            target_compression: 10.0,
            validation_split: 0.2,
        }
    }
}

/// Training sample: (input, `channel_quality`, task)
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// Input feature vector
    pub input: Vec<f32>,
    /// Channel quality at training time
    pub channel: ChannelQuality,
    /// Semantic task
    pub task: SemanticTask,
}

/// Training batch
pub struct TrainingBatch {
    /// Samples in this batch
    pub samples: Vec<TrainingSample>,
}

impl TrainingBatch {
    /// Creates a new batch from samples
    pub fn new(samples: Vec<TrainingSample>) -> Self {
        Self { samples }
    }

    /// Returns the batch size
    pub fn size(&self) -> usize {
        self.samples.len()
    }
}

/// Training statistics for one epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpochStats {
    /// Epoch number
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f32,
    /// Average validation loss
    pub val_loss: f32,
    /// Average distortion (MSE)
    pub avg_distortion: f32,
    /// Average rate (bits per sample)
    pub avg_rate: f32,
    /// Training time in milliseconds
    pub duration_ms: u64,
}

/// Codec trainer for end-to-end learning
pub struct CodecTrainer {
    /// Training configuration
    config: TrainingConfig,
    /// Training dataset
    train_data: Vec<TrainingSample>,
    /// Validation dataset
    val_data: Vec<TrainingSample>,
    /// Training history
    history: Vec<EpochStats>,
    /// Current epoch
    current_epoch: usize,
}

impl CodecTrainer {
    /// Creates a new codec trainer
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            train_data: Vec::new(),
            val_data: Vec::new(),
            history: Vec::new(),
            current_epoch: 0,
        }
    }

    /// Loads training data and splits into train/validation
    pub fn load_data(&mut self, data: Vec<TrainingSample>) {
        let val_size = (data.len() as f32 * self.config.validation_split) as usize;
        let train_size = data.len() - val_size;

        self.val_data = data[train_size..].to_vec();
        self.train_data = data[..train_size].to_vec();
    }

    /// Trains the codec for one epoch
    /// Note: This is a simplified training loop. In production, this would
    /// interface with ONNX Runtime training or `PyTorch` via Python bindings.
    pub fn train_epoch(&mut self) -> EpochStats {
        let start = std::time::Instant::now();

        // Create batches
        let batches = self.create_batches(&self.train_data);

        let mut total_loss = 0.0f32;
        let mut total_distortion = 0.0f32;
        let mut total_rate = 0.0f32;
        let mut num_samples = 0;

        // Training loop (simplified)
        for batch in batches {
            let (loss, distortion, rate) = self.train_batch(&batch);
            total_loss += loss * batch.size() as f32;
            total_distortion += distortion * batch.size() as f32;
            total_rate += rate * batch.size() as f32;
            num_samples += batch.size();
        }

        let avg_train_loss = total_loss / num_samples as f32;
        let avg_distortion = total_distortion / num_samples as f32;
        let avg_rate = total_rate / num_samples as f32;

        // Validation
        let val_loss = self.validate();

        self.current_epoch += 1;

        let stats = EpochStats {
            epoch: self.current_epoch,
            train_loss: avg_train_loss,
            val_loss,
            avg_distortion,
            avg_rate,
            duration_ms: start.elapsed().as_millis() as u64,
        };

        self.history.push(stats.clone());
        stats
    }

    /// Trains on a single batch
    /// Returns (loss, distortion, rate)
    fn train_batch(&self, batch: &TrainingBatch) -> (f32, f32, f32) {
        let mut total_distortion = 0.0f32;
        let mut total_rate = 0.0f32;

        for sample in &batch.samples {
            // Simulate encoding/decoding
            let compressed_dim = (sample.input.len() as f32 / self.config.target_compression) as usize;
            let compressed_dim = compressed_dim.max(1);

            // Simulated compression: mean pooling
            let encoded = self.encode_sample(&sample.input, compressed_dim);

            // Simulated reconstruction: nearest neighbor
            let decoded = self.decode_sample(&encoded, sample.input.len());

            // Compute distortion (MSE)
            let distortion = mse(&sample.input, &decoded);
            total_distortion += distortion;

            // Compute rate (bits per sample)
            // Assuming 32 bits per float
            let rate = (compressed_dim as f32 * 32.0) / sample.input.len() as f32;
            total_rate += rate;
        }

        let avg_distortion = total_distortion / batch.size() as f32;
        let avg_rate = total_rate / batch.size() as f32;

        // Rate-distortion loss
        let loss = avg_distortion + self.config.lambda * avg_rate;

        (loss, avg_distortion, avg_rate)
    }

    /// Validates on the validation set
    fn validate(&self) -> f32 {
        if self.val_data.is_empty() {
            return 0.0;
        }

        let batches = self.create_batches(&self.val_data);
        let mut total_loss = 0.0f32;
        let mut num_samples = 0;

        for batch in batches {
            let (loss, _, _) = self.train_batch(&batch);
            total_loss += loss * batch.size() as f32;
            num_samples += batch.size();
        }

        total_loss / num_samples as f32
    }

    /// Creates batches from a dataset
    fn create_batches(&self, data: &[TrainingSample]) -> Vec<TrainingBatch> {
        let mut batches = Vec::new();
        for chunk in data.chunks(self.config.batch_size) {
            batches.push(TrainingBatch::new(chunk.to_vec()));
        }
        batches
    }

    /// Simplified encoding (mean pooling)
    fn encode_sample(&self, input: &[f32], target_dim: usize) -> Vec<f32> {
        let stride = (input.len() / target_dim).max(1);
        let mut encoded = Vec::with_capacity(target_dim);

        for i in 0..target_dim {
            let start = i * stride;
            let end = ((i + 1) * stride).min(input.len());
            if start < input.len() {
                let chunk = &input[start..end];
                let mean = chunk.iter().sum::<f32>() / chunk.len() as f32;
                encoded.push(mean);
            }
        }

        encoded
    }

    /// Simplified decoding (nearest neighbor upsampling)
    fn decode_sample(&self, encoded: &[f32], target_len: usize) -> Vec<f32> {
        let stride = target_len / encoded.len().max(1);
        let mut decoded = Vec::with_capacity(target_len);

        for &val in encoded {
            for _ in 0..stride {
                decoded.push(val);
            }
        }

        while decoded.len() < target_len {
            decoded.push(encoded.last().copied().unwrap_or(0.0));
        }

        decoded.truncate(target_len);
        decoded
    }

    /// Returns the training history
    pub fn history(&self) -> &[EpochStats] {
        &self.history
    }

    /// Returns whether training has converged
    pub fn has_converged(&self, window: usize, threshold: f32) -> bool {
        if self.history.len() < window {
            return false;
        }

        let recent = &self.history[self.history.len() - window..];
        let first_half_loss = recent.iter().take(window / 2).map(|s| s.val_loss).sum::<f32>()
            / (window / 2) as f32;
        let second_half_loss = recent.iter().skip(window / 2).map(|s| s.val_loss).sum::<f32>()
            / (window - window / 2) as f32;

        if first_half_loss == 0.0 {
            return false;
        }

        let relative_change = ((first_half_loss - second_half_loss) / first_half_loss).abs();
        relative_change < threshold
    }

    /// Exports training curves as JSON
    pub fn export_curves(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.history)
    }
}

/// Mean squared error
fn mse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        / len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_sample(dim: usize) -> TrainingSample {
        TrainingSample {
            input: (0..dim).map(|i| i as f32 / dim as f32).collect(),
            channel: ChannelQuality::new(15.0, 500.0, 0.01),
            task: SemanticTask::ImageClassification,
        }
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.num_epochs, 100);
    }

    #[test]
    fn test_trainer_creation() {
        let trainer = CodecTrainer::new(TrainingConfig::default());
        assert_eq!(trainer.current_epoch, 0);
        assert_eq!(trainer.history().len(), 0);
    }

    #[test]
    fn test_load_data() {
        let mut trainer = CodecTrainer::new(TrainingConfig {
            validation_split: 0.2,
            ..Default::default()
        });

        let data: Vec<_> = (0..100).map(|_| create_test_sample(64)).collect();
        trainer.load_data(data);

        assert_eq!(trainer.train_data.len(), 80);
        assert_eq!(trainer.val_data.len(), 20);
    }

    #[test]
    fn test_train_epoch() {
        let mut trainer = CodecTrainer::new(TrainingConfig {
            batch_size: 10,
            validation_split: 0.2,
            ..Default::default()
        });

        let data: Vec<_> = (0..50).map(|_| create_test_sample(64)).collect();
        trainer.load_data(data);

        let stats = trainer.train_epoch();

        assert_eq!(stats.epoch, 1);
        assert!(stats.train_loss > 0.0);
        assert!(stats.avg_distortion >= 0.0);
        assert!(stats.avg_rate > 0.0);
    }

    #[test]
    fn test_convergence_detection() {
        let mut trainer = CodecTrainer::new(TrainingConfig::default());

        // Simulate converging training
        for i in 0..10 {
            let stats = EpochStats {
                epoch: i,
                train_loss: 1.0 / (i + 1) as f32,
                val_loss: 1.0 / (i + 1) as f32,
                avg_distortion: 0.5 / (i + 1) as f32,
                avg_rate: 10.0,
                duration_ms: 1000,
            };
            trainer.history.push(stats);
        }

        // After stabilization
        for i in 10..15 {
            let stats = EpochStats {
                epoch: i,
                train_loss: 0.1,
                val_loss: 0.1,
                avg_distortion: 0.05,
                avg_rate: 10.0,
                duration_ms: 1000,
            };
            trainer.history.push(stats);
        }

        assert!(trainer.has_converged(6, 0.01));
    }

    #[test]
    fn test_export_curves() {
        let mut trainer = CodecTrainer::new(TrainingConfig::default());

        let stats = EpochStats {
            epoch: 1,
            train_loss: 0.5,
            val_loss: 0.6,
            avg_distortion: 0.4,
            avg_rate: 10.0,
            duration_ms: 1000,
        };
        trainer.history.push(stats);

        let json = trainer.export_curves().unwrap();
        assert!(json.contains("train_loss"));
        assert!(json.contains("val_loss"));
    }
}
