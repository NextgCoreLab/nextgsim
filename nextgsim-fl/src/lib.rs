//! Federated Learning Infrastructure for 6G Networks
//!
//! Implements federated learning per 3GPP TR 23.700-80:
//! - Secure aggregation protocols (Bonawitz et al.)
//! - Differential privacy support (Gaussian mechanism, Renyi DP, zCDP)
//! - Model versioning and distribution
//! - Asynchronous federated learning
//! - Gradient compression (top-k, 1-bit, ternary quantization)
//! - Hierarchical FL (edge->regional->cloud)
//! - Byzantine-robust aggregation (Krum, trimmed mean)

#![allow(missing_docs)]
//! - Personalization (local fine-tuning)
//!
//! # Architecture
//!
//! ```text
//! +-----------------------------------------------------------------------+
//! |                     Federated Learning Infrastructure                  |
//! |  +---------------------------------------------------------------+   |
//! |  | Hierarchical Aggregation                                       |   |
//! |  |  - Edge aggregation                                           |   |
//! |  |  - Regional aggregation                                       |   |
//! |  |  - Cloud aggregation                                          |   |
//! |  +---------------------------------------------------------------+   |
//! |  +---------------------------------------------------------------+   |
//! |  | Aggregation Server                                             |   |
//! |  |  - FedAvg, FedProx, SecAgg                                    |   |
//! |  |  - Async FL with staleness weighting                          |   |
//! |  |  - Byzantine-tolerant (Krum, trimmed mean)                    |   |
//! |  +---------------------------------------------------------------+   |
//! |  +---------------------------------------------------------------+   |
//! |  | Privacy Mechanisms                                             |   |
//! |  |  - Gaussian differential privacy                               |   |
//! |  |  - Renyi DP composition                                        |   |
//! |  |  - Zero-Concentrated DP (zCDP)                                 |   |
//! |  |  - Gradient clipping                                           |   |
//! |  |  - Privacy budget tracking                                     |   |
//! |  +---------------------------------------------------------------+   |
//! |  +---------------------------------------------------------------+   |
//! |  | Communication                                                  |   |
//! |  |  - Top-k gradient compression                                  |   |
//! |  |  - 1-bit and ternary quantization                              |   |
//! |  |  - Secure aggregation (x25519 key exchange)                    |   |
//! |  +---------------------------------------------------------------+   |
//! +-----------------------------------------------------------------------+
//! ```

use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// ── New modules ──────────────────────────────────────────────────────────────

/// Model versioning and distribution system
pub mod model_store;

/// Training convergence metrics and dashboard
pub mod metrics;

/// Integration with semantic communication for distributed codec training
pub mod semantic_integration;

/// Integration with Service Hosting Environment for FL workload placement
pub mod she_integration;

// ---------------------------------------------------------------------------
// Core data structures
// ---------------------------------------------------------------------------

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
    /// Federated Averaging (FedAvg) - McMahan et al. 2017
    FedAvg,
    /// Federated Proximal (FedProx) - Li et al. 2020
    FedProx,
    /// Secure Aggregation - Bonawitz et al. 2017
    SecAgg,
}

impl Default for AggregationAlgorithm {
    fn default() -> Self {
        Self::FedAvg
    }
}

// ---------------------------------------------------------------------------
// Differential Privacy
// ---------------------------------------------------------------------------

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

/// Privacy budget tracker using basic (linear) composition.
///
/// Each round consumes `epsilon_per_round` privacy budget. The tracker
/// accumulates the total spent epsilon and can report whether the budget
/// is exhausted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudgetTracker {
    /// Total epsilon budget
    pub target_epsilon: f32,
    /// Delta parameter
    pub target_delta: f32,
    /// Per-round epsilon computed from the Gaussian mechanism parameters
    pub epsilon_per_round: f32,
    /// Accumulated epsilon spent so far
    pub spent_epsilon: f32,
    /// Number of rounds that consumed privacy budget
    pub rounds_tracked: u64,
}

impl PrivacyBudgetTracker {
    /// Creates a new tracker from a DP config.
    ///
    /// The per-round epsilon is derived from the Gaussian mechanism:
    ///   sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    /// Solving for epsilon:
    ///   epsilon = sensitivity * sqrt(2 * ln(1.25 / delta)) / sigma
    pub fn new(config: &DifferentialPrivacyConfig) -> Self {
        let sensitivity = config.clipping_threshold;
        let sigma = config.clipping_threshold * config.noise_multiplier;
        let epsilon_per_round = if sigma > 0.0 {
            sensitivity * (2.0_f32 * (1.25_f32 / config.target_delta).ln()).sqrt() / sigma
        } else {
            f32::INFINITY
        };
        Self {
            target_epsilon: config.target_epsilon,
            target_delta: config.target_delta,
            epsilon_per_round,
            spent_epsilon: 0.0,
            rounds_tracked: 0,
        }
    }

    /// Records one round of privacy expenditure (basic composition).
    pub fn record_round(&mut self) {
        self.spent_epsilon += self.epsilon_per_round;
        self.rounds_tracked += 1;
    }

    /// Returns `true` if the budget is exhausted.
    pub fn is_exhausted(&self) -> bool {
        self.spent_epsilon >= self.target_epsilon
    }

    /// Returns the remaining epsilon budget.
    pub fn remaining_epsilon(&self) -> f32 {
        (self.target_epsilon - self.spent_epsilon).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// Round management
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// FedProx configuration
// ---------------------------------------------------------------------------

/// FedProx configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FedProxConfig {
    /// Proximal term coefficient (mu). Higher values keep local models
    /// closer to the global model. Typical values: 0.001 to 1.0.
    pub mu: f32,
}

impl Default for FedProxConfig {
    fn default() -> Self {
        Self { mu: 0.01 }
    }
}

// ---------------------------------------------------------------------------
// Gradient Compression
// ---------------------------------------------------------------------------

/// Compressed gradient using top-k sparsification.
///
/// Only the `k` largest (by absolute value) gradient entries are stored,
/// together with their original indices. This reduces communication cost
/// from O(d) to O(k) where d is the model dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedGradient {
    /// Indices of the top-k entries in the original gradient vector
    pub indices: Vec<usize>,
    /// Corresponding values
    pub values: Vec<f32>,
    /// Original dimension (needed for decompression)
    pub original_dim: usize,
}

/// Compresses a gradient vector using top-k sparsification.
///
/// Selects the `k` entries with the largest absolute value.
/// If `k >= gradients.len()`, returns all entries.
pub fn topk_compress(gradients: &[f32], k: usize) -> CompressedGradient {
    let dim = gradients.len();
    let k = k.min(dim);

    // Build (index, abs_value) pairs and partial-sort to find top-k
    let mut indexed: Vec<(usize, f32)> = gradients
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();

    // We only need the top-k, so partial sort is sufficient.
    // select_nth_unstable_by partitions around the k-th element.
    if k < dim {
        indexed.select_nth_unstable_by(dim - k, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    let top_k = &indexed[dim.saturating_sub(k)..];
    let indices: Vec<usize> = top_k.iter().map(|&(i, _)| i).collect();
    let values: Vec<f32> = indices.iter().map(|&i| gradients[i]).collect();

    CompressedGradient {
        indices,
        values,
        original_dim: dim,
    }
}

/// Decompresses a [`CompressedGradient`] back into a dense vector.
///
/// Positions not in the compressed set are filled with zero.
pub fn topk_decompress(compressed: &CompressedGradient) -> Vec<f32> {
    let mut dense = vec![0.0f32; compressed.original_dim];
    for (&idx, &val) in compressed.indices.iter().zip(compressed.values.iter()) {
        if idx < dense.len() {
            dense[idx] = val;
        }
    }
    dense
}

// ---------------------------------------------------------------------------
// Secure Aggregation (Bonawitz et al.)
// ---------------------------------------------------------------------------

/// A participant's keypair and public key for the SecAgg protocol.
///
/// Uses x25519 Diffie-Hellman for pairwise key agreement.
pub struct SecAggParticipant {
    /// Participant identifier
    pub id: String,
    /// Ephemeral secret key (kept private)
    secret: x25519_dalek::StaticSecret,
    /// Corresponding public key (shared with all other participants)
    pub public_key: x25519_dalek::PublicKey,
}

impl std::fmt::Debug for SecAggParticipant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SecAggParticipant")
            .field("id", &self.id)
            .field("public_key", &self.public_key.as_bytes())
            .finish()
    }
}

impl SecAggParticipant {
    /// Creates a new SecAgg participant with a fresh x25519 keypair.
    pub fn new(id: impl Into<String>) -> Self {
        let mut rng = rand::thread_rng();
        let secret = x25519_dalek::StaticSecret::random_from_rng(&mut rng);
        let public_key = x25519_dalek::PublicKey::from(&secret);
        Self {
            id: id.into(),
            secret,
            public_key,
        }
    }

    /// Derives a pairwise shared secret with another participant and uses it
    /// to produce a deterministic pseudo-random mask of length `dim`.
    ///
    /// The mask is derived by seeding a simple PRNG with the first 8 bytes of
    /// the shared secret. Because the DH shared secret is symmetric (A->B ==
    /// B->A), we use the lexicographic ordering of participant IDs to decide
    /// the sign: the participant with the "smaller" ID *adds* the mask while
    /// the one with the "larger" ID *subtracts* it. This ensures the masks
    /// cancel out upon summation.
    fn pairwise_mask(
        &self,
        other_id: &str,
        other_public_key: &x25519_dalek::PublicKey,
        dim: usize,
    ) -> Vec<f32> {
        let shared_secret = self.secret.diffie_hellman(other_public_key);
        let secret_bytes = shared_secret.as_bytes();

        // Derive a u64 seed from the shared secret
        let mut seed_bytes = [0u8; 8];
        seed_bytes.copy_from_slice(&secret_bytes[..8]);
        let seed = u64::from_le_bytes(seed_bytes);

        // Use the seed to generate deterministic mask values via a simple
        // splitmix64-style PRNG (fast, deterministic, sufficient for masking).
        let mut state = seed;
        let mask: Vec<f32> = (0..dim)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                // Map to small float in [-0.01, 0.01]
                let bits = ((state >> 33) as i32) % 10000;
                bits as f32 / 1_000_000.0
            })
            .collect();

        // Determine sign based on lexicographic ordering of IDs
        if self.id.as_str() < other_id {
            mask
        } else {
            mask.iter().map(|v| -v).collect()
        }
    }

    /// Masks a gradient vector given the public keys of all *other*
    /// participants.
    ///
    /// The masked gradient is: `gradient + sum_j mask(self, j)`.
    /// Because `mask(i,j) = -mask(j,i)`, when the server sums all masked
    /// gradients the masks cancel out and the true aggregate remains.
    pub fn mask_gradient(
        &self,
        gradient: &[f32],
        others: &[(String, x25519_dalek::PublicKey)],
    ) -> Vec<f32> {
        let dim = gradient.len();
        let mut masked = gradient.to_vec();

        for (other_id, other_pk) in others {
            if *other_id == self.id {
                continue;
            }
            let m = self.pairwise_mask(other_id, other_pk, dim);
            for (i, val) in m.iter().enumerate() {
                masked[i] += val;
            }
        }
        masked
    }
}

/// Server-side secure aggregation: simply sums masked updates. Because the
/// pairwise masks cancel (`mask(i,j) + mask(j,i) = 0`), the result is the
/// true (weighted) sum of the original gradients.
pub fn secagg_aggregate(masked_updates: &[Vec<f32>], weights: &[f32]) -> Vec<f32> {
    if masked_updates.is_empty() {
        return Vec::new();
    }
    let dim = masked_updates[0].len();
    let mut result = vec![0.0f32; dim];
    for (update, &w) in masked_updates.iter().zip(weights.iter()) {
        for (i, &v) in update.iter().enumerate() {
            if i < dim {
                result[i] += w * v;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Gaussian noise sampling (Box-Muller)
// ---------------------------------------------------------------------------

/// Samples a value from N(0, 1) using the Box-Muller transform.
fn sample_standard_normal<R: Rng>(rng: &mut R) -> f32 {
    // Box-Muller: given u1, u2 ~ Uniform(0,1), then
    //   z = sqrt(-2 ln u1) * cos(2 pi u2) ~ N(0,1)
    let u1: f32 = rng.gen();
    let u2: f32 = rng.gen();
    // Clamp u1 away from zero to avoid ln(0)
    let u1 = u1.max(f32::EPSILON);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
}

/// Samples a value from N(0, sigma^2).
fn sample_gaussian<R: Rng>(rng: &mut R, sigma: f32) -> f32 {
    sigma * sample_standard_normal(rng)
}

// ---------------------------------------------------------------------------
// FederatedAggregator (synchronous)
// ---------------------------------------------------------------------------

/// FL aggregator supporting FedAvg, FedProx, and SecAgg.
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
    /// Privacy budget tracker
    privacy_tracker: Option<PrivacyBudgetTracker>,
    /// Minimum participants required
    min_participants: usize,
    /// Round timeout (seconds)
    round_timeout_secs: u64,
    /// FedProx configuration (used when algorithm == FedProx)
    fedprox_config: FedProxConfig,
    /// SecAgg participants (participant_id -> SecAggParticipant)
    secagg_participants: HashMap<String, SecAggParticipant>,
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
            privacy_tracker: None,
            min_participants,
            round_timeout_secs: 60,
            fedprox_config: FedProxConfig::default(),
            secagg_participants: HashMap::new(),
        }
    }

    /// Sets differential privacy configuration and initialises the budget
    /// tracker.
    pub fn with_dp_config(mut self, config: DifferentialPrivacyConfig) -> Self {
        if config.enabled {
            self.privacy_tracker = Some(PrivacyBudgetTracker::new(&config));
        }
        self.dp_config = config;
        self
    }

    /// Sets the FedProx proximal term coefficient.
    pub fn with_fedprox_config(mut self, config: FedProxConfig) -> Self {
        self.fedprox_config = config;
        self
    }

    /// Returns a reference to the privacy budget tracker, if DP is enabled.
    pub fn privacy_tracker(&self) -> Option<&PrivacyBudgetTracker> {
        self.privacy_tracker.as_ref()
    }

    /// Returns a reference to the FedProx config.
    pub fn fedprox_config(&self) -> &FedProxConfig {
        &self.fedprox_config
    }

    /// Registers a participant.
    ///
    /// When using SecAgg, this also generates an x25519 keypair for the
    /// participant.
    pub fn register_participant(&mut self, id: impl Into<String>, num_samples: u64) {
        let id = id.into();
        self.participants.insert(
            id.clone(),
            Participant {
                id: id.clone(),
                model_version: self.global_model.as_ref().map(|m| m.version).unwrap_or(0),
                num_samples,
                last_update_ms: 0,
                is_active: true,
            },
        );
        if self.algorithm == AggregationAlgorithm::SecAgg {
            let sa_participant = SecAggParticipant::new(&id);
            self.secagg_participants.insert(id, sa_participant);
        }
    }

    /// Returns the public keys of all registered SecAgg participants.
    pub fn secagg_public_keys(&self) -> Vec<(String, x25519_dalek::PublicKey)> {
        self.secagg_participants
            .values()
            .map(|p| (p.id.clone(), p.public_key))
            .collect()
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

        round
            .received_updates
            .insert(processed_update.participant_id.clone(), processed_update);
        round.status = RoundStatus::Collecting;

        Ok(())
    }

    /// Applies differential privacy to an update using the Gaussian mechanism.
    ///
    /// 1. Clip the gradient to `clipping_threshold` (L2 norm).
    /// 2. Add per-parameter Gaussian noise N(0, sigma^2) where
    ///    sigma = clipping_threshold * noise_multiplier.
    fn apply_dp(&self, update: &ModelUpdate) -> ModelUpdate {
        let mut processed = update.clone();

        // Clip gradients (L2 norm clipping)
        let norm: f32 = processed
            .gradients
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        if norm > self.dp_config.clipping_threshold {
            let scale = self.dp_config.clipping_threshold / norm;
            for g in &mut processed.gradients {
                *g *= scale;
            }
        }

        // Add proper Gaussian noise: N(0, sigma^2)
        // sigma = clipping_threshold * noise_multiplier
        let sigma = self.dp_config.clipping_threshold * self.dp_config.noise_multiplier;
        let mut rng = rand::thread_rng();
        for g in &mut processed.gradients {
            *g += sample_gaussian(&mut rng, sigma);
        }

        processed
    }

    /// Aggregates received updates.
    ///
    /// The aggregation strategy depends on the selected algorithm:
    /// - **FedAvg**: sample-weighted average of gradients.
    /// - **FedProx**: FedAvg-style weighted average with an additional
    ///   proximal correction term that penalises deviation from the global
    ///   model.
    /// - **SecAgg**: secure aggregation with pairwise masking (the masks
    ///   cancel in the sum, yielding the true aggregate).
    pub fn aggregate(&mut self) -> Result<AggregatedModel, String> {
        // First, check conditions and extract needed data
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
            let total_samples: u64 =
                round.received_updates.values().map(|u| u.num_samples).sum();
            let avg_loss = round
                .received_updates
                .values()
                .map(|u| u.loss)
                .sum::<f32>()
                / num_updates as f32;

            (
                num_updates,
                round.round,
                round.received_updates.clone(),
                total_samples,
                avg_loss,
            )
        };

        // Set status to aggregating
        if let Some(round) = self.current_round.as_mut() {
            round.status = RoundStatus::Aggregating;
        }

        // Perform aggregation based on algorithm
        let aggregated = match self.algorithm {
            AggregationAlgorithm::FedAvg => self.fedavg_aggregate(&updates_clone),
            AggregationAlgorithm::FedProx => self.fedprox_aggregate(&updates_clone),
            AggregationAlgorithm::SecAgg => self.secagg_aggregate_internal(&updates_clone),
        };

        // Track privacy budget if DP is enabled
        if self.dp_config.enabled {
            if let Some(ref mut tracker) = self.privacy_tracker {
                tracker.record_round();
            }
        }

        let new_version = self
            .global_model
            .as_ref()
            .map(|m| m.version + 1)
            .unwrap_or(1);
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

    /// FedAvg aggregation: sample-weighted average of participant gradients.
    fn fedavg_aggregate(&self, updates: &HashMap<String, ModelUpdate>) -> Vec<f32> {
        if updates.is_empty() {
            return Vec::new();
        }

        let dim = updates.values().next().map(|u| u.gradients.len()).unwrap_or(0);
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

    /// FedProx aggregation.
    ///
    /// FedProx adds a proximal term `(mu/2) * ||w - w_global||^2` to each
    /// participant's local objective, which means participant gradients already
    /// incorporate the proximal penalty during local training.
    ///
    /// At aggregation time, FedProx performs a weighted average similar to
    /// FedAvg but applies a proximal correction step: the aggregated result
    /// is shifted toward the current global model proportionally to `mu`.
    ///
    /// Specifically:
    ///   w_new = (1 / (1 + mu)) * w_fedavg + (mu / (1 + mu)) * w_global
    ///
    /// This ensures that even after aggregation, the new global model does
    /// not drift too far from the previous global model, which stabilizes
    /// convergence in heterogeneous settings.
    fn fedprox_aggregate(&self, updates: &HashMap<String, ModelUpdate>) -> Vec<f32> {
        // Start with FedAvg-style weighted average
        let fedavg_result = self.fedavg_aggregate(updates);

        // Apply proximal correction toward global model
        let global_weights = match &self.global_model {
            Some(m) => &m.weights,
            None => return fedavg_result, // No global model yet; fall back
        };

        let mu = self.fedprox_config.mu;
        let correction_factor = 1.0 / (1.0 + mu);
        let global_factor = mu / (1.0 + mu);

        fedavg_result
            .iter()
            .enumerate()
            .map(|(i, &w_avg)| {
                let w_global = global_weights.get(i).copied().unwrap_or(0.0);
                correction_factor * w_avg + global_factor * w_global
            })
            .collect()
    }

    /// SecAgg aggregation using pairwise x25519 masking.
    ///
    /// Each participant masks their gradient with pairwise random masks
    /// derived from DH shared secrets. The masks are antisymmetric
    /// (`mask(i,j) = -mask(j,i)`), so they cancel when summed. The server
    /// performs a simple weighted sum of the masked gradients to recover the
    /// true aggregate.
    fn secagg_aggregate_internal(&self, updates: &HashMap<String, ModelUpdate>) -> Vec<f32> {
        if updates.is_empty() {
            return Vec::new();
        }

        let all_keys = self.secagg_public_keys();
        let total_samples: u64 = updates.values().map(|u| u.num_samples).sum();

        let mut masked_updates = Vec::new();
        let mut weights = Vec::new();

        for update in updates.values() {
            let weight = update.num_samples as f32 / total_samples as f32;
            weights.push(weight);

            // If this participant has a SecAgg identity, mask the gradient
            if let Some(sa_participant) = self.secagg_participants.get(&update.participant_id) {
                let masked = sa_participant.mask_gradient(&update.gradients, &all_keys);
                masked_updates.push(masked);
            } else {
                // Fallback: unmasked (should not happen in a proper setup)
                masked_updates.push(update.gradients.clone());
            }
        }

        secagg_aggregate(&masked_updates, &weights)
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

// ---------------------------------------------------------------------------
// Async Federated Aggregator
// ---------------------------------------------------------------------------

/// An update record for async FL, stamped with version info for staleness
/// computation.
#[derive(Debug, Clone)]
struct AsyncUpdate {
    /// The model update
    update: ModelUpdate,
    /// The global model version when this update was received (for diagnostics)
    #[allow(dead_code)]
    received_at_version: u64,
}

/// Asynchronous Federated Aggregator.
///
/// Unlike the synchronous [`FederatedAggregator`], this aggregator does not
/// wait for all participants before aggregating. Instead, it incorporates
/// updates as they arrive, weighting each contribution by a staleness factor:
///
///   weight_i = (num_samples_i / total_samples) * alpha^(current_version - base_version_i)
///
/// where `alpha` is a staleness decay factor in (0, 1]. An `alpha` of 1.0
/// gives equal weight regardless of staleness; smaller values down-weight
/// stale updates.
#[derive(Debug)]
pub struct AsyncFederatedAggregator {
    /// Current global model
    global_model: Option<AggregatedModel>,
    /// Pending updates not yet aggregated
    pending_updates: Vec<AsyncUpdate>,
    /// Registered participants
    participants: HashMap<String, Participant>,
    /// Staleness decay factor
    staleness_alpha: f32,
    /// Minimum number of pending updates before triggering aggregation
    min_updates_to_aggregate: usize,
    /// Differential privacy config
    dp_config: DifferentialPrivacyConfig,
    /// Privacy budget tracker
    privacy_tracker: Option<PrivacyBudgetTracker>,
    /// Number of aggregations performed
    aggregation_count: u64,
}

impl AsyncFederatedAggregator {
    /// Creates a new async aggregator.
    ///
    /// # Arguments
    /// * `staleness_alpha` - staleness decay in (0, 1]. 1.0 = no decay.
    /// * `min_updates_to_aggregate` - minimum pending updates before
    ///   aggregation is triggered.
    pub fn new(staleness_alpha: f32, min_updates_to_aggregate: usize) -> Self {
        Self {
            global_model: None,
            pending_updates: Vec::new(),
            participants: HashMap::new(),
            staleness_alpha: staleness_alpha.clamp(0.01, 1.0),
            min_updates_to_aggregate: min_updates_to_aggregate.max(1),
            dp_config: DifferentialPrivacyConfig::default(),
            privacy_tracker: None,
            aggregation_count: 0,
        }
    }

    /// Sets differential privacy configuration.
    pub fn with_dp_config(mut self, config: DifferentialPrivacyConfig) -> Self {
        if config.enabled {
            self.privacy_tracker = Some(PrivacyBudgetTracker::new(&config));
        }
        self.dp_config = config;
        self
    }

    /// Returns a reference to the privacy budget tracker.
    pub fn privacy_tracker(&self) -> Option<&PrivacyBudgetTracker> {
        self.privacy_tracker.as_ref()
    }

    /// Initializes the global model.
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

    /// Registers a participant.
    pub fn register_participant(&mut self, id: impl Into<String>, num_samples: u64) {
        let id = id.into();
        self.participants.insert(
            id.clone(),
            Participant {
                id,
                model_version: self
                    .global_model
                    .as_ref()
                    .map(|m| m.version)
                    .unwrap_or(0),
                num_samples,
                last_update_ms: 0,
                is_active: true,
            },
        );
    }

    /// Submits an update. The update is queued and aggregated when enough
    /// updates have accumulated.
    ///
    /// Returns `Ok(Some(model))` if aggregation was triggered, `Ok(None)`
    /// otherwise.
    pub fn submit_update(
        &mut self,
        update: ModelUpdate,
    ) -> Result<Option<AggregatedModel>, String> {
        // Optionally apply DP
        let processed = if self.dp_config.enabled {
            self.apply_dp(&update)
        } else {
            update
        };

        let current_version = self
            .global_model
            .as_ref()
            .map(|m| m.version)
            .unwrap_or(0);

        self.pending_updates.push(AsyncUpdate {
            update: processed,
            received_at_version: current_version,
        });

        // Check if we have enough to aggregate
        if self.pending_updates.len() >= self.min_updates_to_aggregate {
            let model = self.aggregate()?;
            return Ok(Some(model));
        }

        Ok(None)
    }

    /// Applies differential privacy using the Gaussian mechanism.
    fn apply_dp(&self, update: &ModelUpdate) -> ModelUpdate {
        let mut processed = update.clone();

        // Clip gradients
        let norm: f32 = processed
            .gradients
            .iter()
            .map(|g| g * g)
            .sum::<f32>()
            .sqrt();
        if norm > self.dp_config.clipping_threshold {
            let scale = self.dp_config.clipping_threshold / norm;
            for g in &mut processed.gradients {
                *g *= scale;
            }
        }

        // Gaussian noise
        let sigma = self.dp_config.clipping_threshold * self.dp_config.noise_multiplier;
        let mut rng = rand::thread_rng();
        for g in &mut processed.gradients {
            *g += sample_gaussian(&mut rng, sigma);
        }

        processed
    }

    /// Aggregates all pending updates using staleness-weighted averaging.
    ///
    /// Each update i receives weight:
    ///   `w_i = (samples_i / total_samples) * alpha^staleness_i`
    /// where `staleness_i = current_version - base_version_i`.
    pub fn aggregate(&mut self) -> Result<AggregatedModel, String> {
        if self.pending_updates.is_empty() {
            return Err("No pending updates".to_string());
        }

        let current_version = self
            .global_model
            .as_ref()
            .map(|m| m.version)
            .unwrap_or(0);

        // Compute staleness-weighted contributions
        let dim = self.pending_updates[0].update.gradients.len();
        let mut aggregated = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;
        let mut total_samples = 0u64;
        let mut total_loss = 0.0f32;

        for au in &self.pending_updates {
            let staleness = current_version.saturating_sub(au.update.base_version);
            let staleness_weight = self.staleness_alpha.powi(staleness as i32);
            let sample_weight = au.update.num_samples as f32;
            let weight = sample_weight * staleness_weight;
            total_weight += weight;
            total_samples += au.update.num_samples;
            total_loss += au.update.loss;
        }

        if total_weight == 0.0 {
            return Err("Total weight is zero".to_string());
        }

        for au in &self.pending_updates {
            let staleness = current_version.saturating_sub(au.update.base_version);
            let staleness_weight = self.staleness_alpha.powi(staleness as i32);
            let sample_weight = au.update.num_samples as f32;
            let weight = (sample_weight * staleness_weight) / total_weight;

            for (i, &g) in au.update.gradients.iter().enumerate() {
                if i < dim {
                    aggregated[i] += weight * g;
                }
            }
        }

        let num_participants = self.pending_updates.len() as u32;
        let avg_loss = total_loss / num_participants as f32;

        // Track privacy
        if self.dp_config.enabled {
            if let Some(ref mut tracker) = self.privacy_tracker {
                tracker.record_round();
            }
        }

        let new_version = current_version + 1;
        let model = AggregatedModel {
            version: new_version,
            weights: aggregated,
            num_participants,
            total_samples,
            avg_loss,
            timestamp_ms: timestamp_now(),
        };

        self.global_model = Some(model.clone());
        self.pending_updates.clear();
        self.aggregation_count += 1;

        Ok(model)
    }

    /// Returns the current global model.
    pub fn global_model(&self) -> Option<&AggregatedModel> {
        self.global_model.as_ref()
    }

    /// Returns the number of pending updates.
    pub fn pending_count(&self) -> usize {
        self.pending_updates.len()
    }

    /// Returns the number of aggregations performed.
    pub fn aggregation_count(&self) -> u64 {
        self.aggregation_count
    }

    /// Returns the number of registered participants.
    pub fn participant_count(&self) -> usize {
        self.participants.len()
    }
}

// ---------------------------------------------------------------------------
// FL message types (unchanged public API)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Hierarchical Federated Learning (A17.1)
// ---------------------------------------------------------------------------

/// Tier in the hierarchical FL architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FLTier {
    /// Edge aggregation (base stations, local clusters)
    Edge,
    /// Regional aggregation (regional data centers)
    Regional,
    /// Cloud aggregation (global model)
    Cloud,
}

/// Hierarchical aggregator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalConfig {
    /// Number of edge aggregators
    pub num_edge: usize,
    /// Number of regional aggregators
    pub num_regional: usize,
    /// Aggregation weights by tier (edge, regional, cloud)
    pub tier_weights: (f32, f32, f32),
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            num_edge: 4,
            num_regional: 2,
            tier_weights: (0.4, 0.3, 0.3), // Balanced weighting
        }
    }
}

/// Hierarchical FL aggregator supporting multi-tier federation
#[derive(Debug)]
pub struct HierarchicalAggregator {
    /// Edge-tier aggregators
    edge_aggregators: Vec<FederatedAggregator>,
    /// Regional-tier aggregators
    regional_aggregators: Vec<FederatedAggregator>,
    /// Cloud-tier aggregator
    cloud_aggregator: FederatedAggregator,
    /// Configuration
    config: HierarchicalConfig,
}

impl HierarchicalAggregator {
    /// Creates a new hierarchical aggregator
    pub fn new(config: HierarchicalConfig, algorithm: AggregationAlgorithm) -> Self {
        let edge_aggregators = (0..config.num_edge)
            .map(|_| FederatedAggregator::new(algorithm, 2))
            .collect();

        let regional_aggregators = (0..config.num_regional)
            .map(|_| FederatedAggregator::new(algorithm, 2))
            .collect();

        let cloud_aggregator = FederatedAggregator::new(algorithm, 2);

        Self {
            edge_aggregators,
            regional_aggregators,
            cloud_aggregator,
            config,
        }
    }

    /// Registers a participant to a specific edge aggregator
    pub fn register_participant(&mut self, edge_id: usize, participant_id: String, num_samples: u64) {
        if edge_id < self.edge_aggregators.len() {
            self.edge_aggregators[edge_id].register_participant(participant_id, num_samples);
        }
    }

    /// Performs hierarchical aggregation: edge -> regional -> cloud
    pub fn hierarchical_aggregate(&mut self) -> Result<AggregatedModel, String> {
        // Step 1: Aggregate at edge tier
        let mut edge_models = Vec::new();
        for aggregator in &mut self.edge_aggregators {
            if let Ok(model) = aggregator.aggregate() {
                edge_models.push(model);
            }
        }

        if edge_models.is_empty() {
            return Err("No edge models available".to_string());
        }

        // Step 2: Distribute edge models to regional aggregators
        let models_per_regional = edge_models.len() / self.config.num_regional.max(1);
        let mut regional_models = Vec::new();

        for (i, _regional) in self.regional_aggregators.iter_mut().enumerate() {
            let start = i * models_per_regional;
            let end = if i == self.config.num_regional - 1 {
                edge_models.len()
            } else {
                (i + 1) * models_per_regional
            };

            // Aggregate edge models in this regional partition
            if start < edge_models.len() {
                let chunk = &edge_models[start..end];
                let aggregated = Self::aggregate_models(chunk);
                regional_models.push(aggregated);
            }
        }

        // Step 3: Aggregate at cloud tier
        let global_model = Self::aggregate_models(&regional_models);

        Ok(global_model)
    }

    /// Aggregates a set of models by weighted averaging
    fn aggregate_models(models: &[AggregatedModel]) -> AggregatedModel {
        if models.is_empty() {
            return AggregatedModel {
                version: 0,
                weights: vec![],
                num_participants: 0,
                total_samples: 0,
                avg_loss: 0.0,
                timestamp_ms: timestamp_now(),
            };
        }

        let total_samples: u64 = models.iter().map(|m| m.total_samples).sum();
        let dim = models[0].weights.len();
        let mut aggregated_weights = vec![0.0f32; dim];

        for model in models {
            let weight = model.total_samples as f32 / total_samples as f32;
            for (i, &w) in model.weights.iter().enumerate() {
                if i < aggregated_weights.len() {
                    aggregated_weights[i] += weight * w;
                }
            }
        }

        let num_participants: u32 = models.iter().map(|m| m.num_participants).sum();
        let avg_loss = models.iter().map(|m| m.avg_loss).sum::<f32>() / models.len() as f32;
        let version = models.iter().map(|m| m.version).max().unwrap_or(0) + 1;

        AggregatedModel {
            version,
            weights: aggregated_weights,
            num_participants,
            total_samples,
            avg_loss,
            timestamp_ms: timestamp_now(),
        }
    }

    /// Returns the cloud aggregator (global model)
    pub fn cloud_aggregator(&self) -> &FederatedAggregator {
        &self.cloud_aggregator
    }
}

// ---------------------------------------------------------------------------
// Heterogeneity-Aware Client Selection (A17.2)
// ---------------------------------------------------------------------------

/// Client resource profile for heterogeneity-aware scheduling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClientProfile {
    /// Client identifier
    pub client_id: String,
    /// Computational capacity (FLOPS or relative score)
    pub compute_capacity: f32,
    /// Network bandwidth (Mbps)
    pub bandwidth_mbps: f32,
    /// Battery level (0.0 to 1.0)
    pub battery_level: f32,
    /// Data availability (number of samples)
    pub num_samples: u64,
    /// Past reliability (completion rate 0.0 to 1.0)
    pub reliability: f32,
}

impl ClientProfile {
    /// Creates a new client profile
    pub fn new(
        client_id: String,
        compute_capacity: f32,
        bandwidth_mbps: f32,
        num_samples: u64,
    ) -> Self {
        Self {
            client_id,
            compute_capacity,
            bandwidth_mbps,
            battery_level: 1.0,
            num_samples,
            reliability: 1.0,
        }
    }

    /// Computes a utility score for client selection
    /// Higher is better
    pub fn utility_score(&self) -> f32 {
        let compute_score = self.compute_capacity.ln_1p();
        let bandwidth_score = self.bandwidth_mbps.ln_1p();
        let data_score = (self.num_samples as f32).ln_1p();
        let battery_score = self.battery_level;
        let reliability_score = self.reliability;

        // Weighted combination
        0.2 * compute_score
            + 0.2 * bandwidth_score
            + 0.3 * data_score
            + 0.15 * battery_score
            + 0.15 * reliability_score
    }
}

/// Heterogeneity-aware client selector
pub struct ClientSelector {
    /// All registered client profiles
    profiles: HashMap<String, ClientProfile>,
    /// Selection strategy
    strategy: SelectionStrategy,
}

/// Client selection strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionStrategy {
    /// Select clients with highest utility scores
    TopK,
    /// Proportional sampling based on utility
    Proportional,
    /// Round-robin (fair but ignores heterogeneity)
    RoundRobin,
}

impl ClientSelector {
    /// Creates a new client selector
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            profiles: HashMap::new(),
            strategy,
        }
    }

    /// Registers a client profile
    pub fn register(&mut self, profile: ClientProfile) {
        self.profiles.insert(profile.client_id.clone(), profile);
    }

    /// Selects k clients for the next training round
    pub fn select_clients(&self, k: usize) -> Vec<String> {
        match self.strategy {
            SelectionStrategy::TopK => self.select_top_k(k),
            SelectionStrategy::Proportional => self.select_proportional(k),
            SelectionStrategy::RoundRobin => self.select_round_robin(k),
        }
    }

    /// Selects top-k clients by utility score
    fn select_top_k(&self, k: usize) -> Vec<String> {
        let mut scored: Vec<_> = self
            .profiles
            .values()
            .map(|p| (p.client_id.clone(), p.utility_score()))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().map(|(id, _)| id).collect()
    }

    /// Selects clients proportional to utility scores
    fn select_proportional(&self, k: usize) -> Vec<String> {
        let scores: Vec<_> = self
            .profiles
            .values()
            .map(|p| (p.client_id.clone(), p.utility_score()))
            .collect();

        let total_score: f32 = scores.iter().map(|(_, s)| s).sum();
        if total_score == 0.0 {
            return scores.into_iter().take(k).map(|(id, _)| id).collect();
        }

        let mut selected = Vec::new();
        let mut rng = rand::thread_rng();

        for _ in 0..k.min(scores.len()) {
            let mut r: f32 = rng.gen::<f32>() * total_score;
            for (id, score) in &scores {
                r -= score;
                if r <= 0.0 {
                    selected.push(id.clone());
                    break;
                }
            }
        }

        selected
    }

    /// Selects clients in round-robin order
    fn select_round_robin(&self, k: usize) -> Vec<String> {
        self.profiles
            .keys()
            .take(k)
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Advanced Privacy: Renyi DP and zCDP (A17.4)
// ---------------------------------------------------------------------------

/// Advanced privacy accounting using Renyi Differential Privacy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenyiDPTracker {
    /// RDP orders to track (typically 2, 4, 8, 16, 32, 64)
    pub orders: Vec<f32>,
    /// Accumulated RDP guarantees at each order
    pub rdp_epsilon: Vec<f32>,
    /// Target (epsilon, delta) for conversion
    pub target_epsilon: f32,
    pub target_delta: f32,
    /// Number of compositions
    pub num_compositions: u64,
}

impl RenyiDPTracker {
    /// Creates a new Renyi DP tracker
    pub fn new(target_epsilon: f32, target_delta: f32) -> Self {
        let orders = vec![2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
        let rdp_epsilon = vec![0.0; orders.len()];

        Self {
            orders,
            rdp_epsilon,
            target_epsilon,
            target_delta,
            num_compositions: 0,
        }
    }

    /// Records a Gaussian mechanism with noise multiplier sigma
    /// RDP guarantee at order alpha: epsilon_alpha = alpha / (2 * sigma^2)
    pub fn record_gaussian(&mut self, sigma: f32) {
        for (i, &alpha) in self.orders.iter().enumerate() {
            let epsilon_alpha = alpha / (2.0 * sigma * sigma);
            self.rdp_epsilon[i] += epsilon_alpha;
        }
        self.num_compositions += 1;
    }

    /// Converts RDP to (epsilon, delta)-DP using the optimal order
    /// epsilon = min_alpha (RDP_alpha + log(1/delta) / (alpha - 1))
    pub fn get_epsilon(&self) -> f32 {
        if self.num_compositions == 0 {
            return 0.0;
        }

        let log_delta = (1.0 / self.target_delta).ln();

        let mut min_epsilon = f32::INFINITY;
        for (i, &alpha) in self.orders.iter().enumerate() {
            if alpha <= 1.0 {
                continue;
            }
            let epsilon = self.rdp_epsilon[i] + log_delta / (alpha - 1.0);
            min_epsilon = min_epsilon.min(epsilon);
        }

        min_epsilon
    }

    /// Returns true if privacy budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.get_epsilon() >= self.target_epsilon
    }
}

/// Zero-Concentrated Differential Privacy (zCDP) tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZCDPTracker {
    /// Accumulated rho (zCDP parameter)
    pub rho: f32,
    /// Target (epsilon, delta) for conversion
    pub target_epsilon: f32,
    pub target_delta: f32,
}

impl ZCDPTracker {
    /// Creates a new zCDP tracker
    pub fn new(target_epsilon: f32, target_delta: f32) -> Self {
        Self {
            rho: 0.0,
            target_epsilon,
            target_delta,
        }
    }

    /// Records a Gaussian mechanism with noise multiplier sigma
    /// zCDP guarantee: rho = 1 / (2 * sigma^2)
    pub fn record_gaussian(&mut self, sigma: f32) {
        self.rho += 1.0 / (2.0 * sigma * sigma);
    }

    /// Converts zCDP to (epsilon, delta)-DP
    /// epsilon = rho + 2 * sqrt(rho * log(1/delta))
    pub fn get_epsilon(&self) -> f32 {
        let log_inv_delta = (1.0 / self.target_delta).ln();
        self.rho + 2.0 * (self.rho * log_inv_delta).sqrt()
    }

    /// Returns true if privacy budget is exhausted
    pub fn is_exhausted(&self) -> bool {
        self.get_epsilon() >= self.target_epsilon
    }
}

// ---------------------------------------------------------------------------
// Gradient Quantization: 1-bit and Ternary (A17.5)
// ---------------------------------------------------------------------------

/// Quantization scheme for gradient compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// No quantization (full precision)
    None,
    /// 1-bit quantization (sign only)
    OneBit,
    /// Ternary quantization {-1, 0, +1}
    Ternary,
    /// Top-k sparsification (existing)
    TopK { k: usize },
}

/// Quantized gradient representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedGradient {
    /// Quantization scheme used
    pub scheme: QuantizationScheme,
    /// Quantized values (encoding depends on scheme)
    pub values: Vec<i8>,
    /// Scale factor for reconstruction
    pub scale: f32,
    /// Original dimension
    pub dim: usize,
}

/// Quantizes a gradient vector using 1-bit quantization (sign only)
/// Stores only the sign of each element, reducing bandwidth by ~32x
pub fn onebit_quantize(gradient: &[f32]) -> QuantizedGradient {
    let scale = gradient.iter().map(|g| g.abs()).sum::<f32>() / gradient.len() as f32;

    let values: Vec<i8> = gradient
        .iter()
        .map(|&g| if g >= 0.0 { 1 } else { -1 })
        .collect();

    QuantizedGradient {
        scheme: QuantizationScheme::OneBit,
        values,
        scale,
        dim: gradient.len(),
    }
}

/// Dequantizes a 1-bit gradient
pub fn onebit_dequantize(quantized: &QuantizedGradient) -> Vec<f32> {
    quantized
        .values
        .iter()
        .map(|&v| v as f32 * quantized.scale)
        .collect()
}

/// Quantizes a gradient vector using ternary quantization {-1, 0, +1}
/// Values below threshold are set to 0 for additional sparsity
pub fn ternary_quantize(gradient: &[f32], threshold: f32) -> QuantizedGradient {
    let scale = gradient.iter().map(|g| g.abs()).max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(1.0);

    let values: Vec<i8> = gradient
        .iter()
        .map(|&g| {
            let normalized = g / scale;
            if normalized.abs() < threshold {
                0
            } else if normalized >= 0.0 {
                1
            } else {
                -1
            }
        })
        .collect();

    QuantizedGradient {
        scheme: QuantizationScheme::Ternary,
        values,
        scale,
        dim: gradient.len(),
    }
}

/// Dequantizes a ternary gradient
pub fn ternary_dequantize(quantized: &QuantizedGradient) -> Vec<f32> {
    quantized
        .values
        .iter()
        .map(|&v| v as f32 * quantized.scale)
        .collect()
}

// ---------------------------------------------------------------------------
// Personalization: Local Fine-Tuning (A17.6)
// ---------------------------------------------------------------------------

/// Personalization layer for local model adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonalizationLayer {
    /// Client identifier
    pub client_id: String,
    /// Local adaptation weights
    pub local_weights: Vec<f32>,
    /// Learning rate for local fine-tuning
    pub learning_rate: f32,
    /// Number of local fine-tuning steps
    pub num_local_steps: u32,
}

impl PersonalizationLayer {
    /// Creates a new personalization layer
    pub fn new(client_id: String, dim: usize, learning_rate: f32) -> Self {
        Self {
            client_id,
            local_weights: vec![0.0; dim],
            learning_rate,
            num_local_steps: 5,
        }
    }

    /// Adapts the global model with local fine-tuning
    pub fn personalize(&mut self, global_weights: &[f32], local_gradients: &[f32]) -> Vec<f32> {
        let dim = global_weights.len().min(local_gradients.len());
        let mut personalized = global_weights.to_vec();

        // Update local adaptation weights
        for i in 0..dim {
            self.local_weights[i] -= self.learning_rate * local_gradients[i];
        }

        // Combine global model with local adaptation
        for i in 0..dim {
            personalized[i] += self.local_weights[i];
        }

        personalized
    }

    /// Resets local adaptation (e.g., when global model changes significantly)
    pub fn reset(&mut self) {
        for w in &mut self.local_weights {
            *w = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// Byzantine-Tolerant Aggregation: Krum and Trimmed Mean (A17.7)
// ---------------------------------------------------------------------------

/// Byzantine-robust aggregation algorithms
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ByzantineRobustAlgorithm {
    /// Krum: selects the update closest to its neighbors
    Krum { num_byzantine: usize },
    /// Multi-Krum: averages top-m Krum selections
    MultiKrum { num_byzantine: usize, m: usize },
    /// Trimmed Mean: removes outliers and averages
    TrimmedMean { trim_ratio: f32 },
    /// Median: coordinate-wise median (robust but expensive)
    Median,
}

/// Krum aggregation: selects the gradient closest to its k-1 neighbors
pub fn krum_aggregate(updates: &[Vec<f32>], num_byzantine: usize) -> Vec<f32> {
    if updates.is_empty() {
        return vec![];
    }

    let n = updates.len();
    let k = n.saturating_sub(num_byzantine).saturating_sub(1).max(1);

    // Compute pairwise distances
    let mut scores = vec![0.0f32; n];
    for i in 0..n {
        let mut distances: Vec<(usize, f32)> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let dist = euclidean_distance(&updates[i], &updates[j]);
                (j, dist)
            })
            .collect();

        // Sort by distance and sum k nearest
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scores[i] = distances.iter().take(k).map(|(_, d)| d).sum();
    }

    // Select update with minimum score
    let best_idx = scores
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    updates[best_idx].clone()
}

/// Multi-Krum: averages the top-m Krum selections
pub fn multi_krum_aggregate(updates: &[Vec<f32>], num_byzantine: usize, m: usize) -> Vec<f32> {
    if updates.is_empty() {
        return vec![];
    }

    let n = updates.len();
    let k = n.saturating_sub(num_byzantine).saturating_sub(1).max(1);
    let m = m.min(n);

    // Compute Krum scores for all updates
    let mut scores: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let mut distances: Vec<f32> = (0..n)
                .filter(|&j| j != i)
                .map(|j| euclidean_distance(&updates[i], &updates[j]))
                .collect();

            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let score: f32 = distances.iter().take(k).sum();
            (i, score)
        })
        .collect();

    // Sort by score and select top m
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let selected_indices: Vec<usize> = scores.iter().take(m).map(|(idx, _)| *idx).collect();

    // Average selected updates
    let dim = updates[0].len();
    let mut result = vec![0.0f32; dim];
    for &idx in &selected_indices {
        for (i, &val) in updates[idx].iter().enumerate() {
            result[i] += val / m as f32;
        }
    }

    result
}

/// Trimmed mean: removes top and bottom trim_ratio of values per coordinate
pub fn trimmed_mean_aggregate(updates: &[Vec<f32>], trim_ratio: f32) -> Vec<f32> {
    if updates.is_empty() {
        return vec![];
    }

    let dim = updates[0].len();
    let n = updates.len();
    let trim_count = ((n as f32 * trim_ratio) / 2.0).floor() as usize;

    let mut result = vec![0.0f32; dim];

    for i in 0..dim {
        let mut values: Vec<f32> = updates.iter().map(|u| u.get(i).copied().unwrap_or(0.0)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Trim extremes
        let trimmed = &values[trim_count..n.saturating_sub(trim_count)];
        result[i] = if trimmed.is_empty() {
            0.0
        } else {
            trimmed.iter().sum::<f32>() / trimmed.len() as f32
        };
    }

    result
}

/// Coordinate-wise median aggregation
pub fn median_aggregate(updates: &[Vec<f32>]) -> Vec<f32> {
    if updates.is_empty() {
        return vec![];
    }

    let dim = updates[0].len();
    let mut result = vec![0.0f32; dim];

    for i in 0..dim {
        let mut values: Vec<f32> = updates.iter().map(|u| u.get(i).copied().unwrap_or(0.0)).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = values.len() / 2;
        result[i] = if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        };
    }

    result
}

/// Computes Euclidean distance between two vectors
fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    a[..len]
        .iter()
        .zip(b[..len].iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Integration with SHE (A17.9)
// ---------------------------------------------------------------------------

/// SHE (Semantic-Hierarchical Edge) integration for FL
/// This allows training on edge/regional/cloud tiers with semantic communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SHEFLConfig {
    /// Use semantic compression for model updates
    pub use_semantic_compression: bool,
    /// Target compression ratio
    pub compression_ratio: f32,
    /// Tier assignment for participants
    pub tier_assignment: HashMap<String, FLTier>,
}

impl Default for SHEFLConfig {
    fn default() -> Self {
        Self {
            use_semantic_compression: true,
            compression_ratio: 10.0,
            tier_assignment: HashMap::new(),
        }
    }
}

/// SHE-integrated FL aggregator
/// Cross-reference: nextgsim-she crate for semantic-hierarchical edge computing
pub struct SHEFLAggregator {
    /// Base hierarchical aggregator
    _base: HierarchicalAggregator,
    /// SHE configuration
    config: SHEFLConfig,
}

impl SHEFLAggregator {
    /// Creates a new SHE-FL aggregator
    pub fn new(
        hierarchical_config: HierarchicalConfig,
        she_config: SHEFLConfig,
        algorithm: AggregationAlgorithm,
    ) -> Self {
        Self {
            _base: HierarchicalAggregator::new(hierarchical_config, algorithm),
            config: she_config,
        }
    }

    /// Returns the tier assignment for a participant
    pub fn get_participant_tier(&self, participant_id: &str) -> Option<FLTier> {
        self.config.tier_assignment.get(participant_id).copied()
    }

    /// Assigns a participant to a specific tier
    pub fn assign_tier(&mut self, participant_id: String, tier: FLTier) {
        self.config.tier_assignment.insert(participant_id, tier);
    }
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Gets current timestamp in milliseconds
fn timestamp_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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
        let round = aggregator.start_round().expect("should start round");
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
            .expect("should accept update");

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.2; 10],
                num_samples: 200,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("should accept update");

        // Aggregate
        let model = aggregator.aggregate().expect("should aggregate");
        assert_eq!(model.version, 2);
        assert_eq!(model.num_participants, 2);

        // FedAvg: weighted average of [0.1]*10 (weight 1/3) and [0.2]*10 (weight 2/3)
        // Expected: 0.1*(1/3) + 0.2*(2/3) = 0.0333 + 0.1333 = 0.1667
        for &w in &model.weights {
            assert!((w - 0.1667).abs() < 0.01, "FedAvg weight mismatch: {w}");
        }
    }

    #[test]
    fn test_fedprox_aggregation() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::FedProx, 2)
            .with_fedprox_config(FedProxConfig { mu: 1.0 });

        // Initialize global model with weights = 0.5
        aggregator.initialize_model(vec![0.5; 10]);

        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 100);

        aggregator.start_round().expect("should start round");

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![0.1; 10],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .expect("should accept");

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.3; 10],
                num_samples: 100,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("should accept");

        let model = aggregator.aggregate().expect("should aggregate");

        // FedAvg would give 0.2 (equal weights, (0.1+0.3)/2)
        // FedProx with mu=1.0: (1/(1+1)) * 0.2 + (1/(1+1)) * 0.5 = 0.1 + 0.25 = 0.35
        for &w in &model.weights {
            assert!(
                (w - 0.35).abs() < 0.01,
                "FedProx weight mismatch: {w}, expected ~0.35"
            );
        }
    }

    #[test]
    fn test_differential_privacy_gaussian() {
        let aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2)
            .with_dp_config(DifferentialPrivacyConfig {
                enabled: true,
                noise_multiplier: 1.0,
                clipping_threshold: 1.0,
                target_epsilon: 8.0,
                target_delta: 1e-5,
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

        // Gradients should be modified by noise
        assert_ne!(processed.gradients, update.gradients);

        // Run it multiple times - results should differ (stochastic noise)
        let processed2 = aggregator.apply_dp(&update);
        // With overwhelming probability, two independent Gaussian samples differ
        assert_ne!(
            processed.gradients, processed2.gradients,
            "Two DP applications should produce different results"
        );
    }

    #[test]
    fn test_dp_gradient_clipping() {
        let aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2)
            .with_dp_config(DifferentialPrivacyConfig {
                enabled: true,
                noise_multiplier: 0.0001, // Very small noise to test clipping
                clipping_threshold: 1.0,
                target_epsilon: 8.0,
                target_delta: 1e-5,
            });

        // Create a gradient with L2 norm = 10 (well above threshold of 1.0)
        let update = ModelUpdate {
            participant_id: "ue-1".to_string(),
            base_version: 1,
            gradients: vec![10.0, 0.0, 0.0],
            num_samples: 100,
            loss: 0.5,
            timestamp_ms: timestamp_now(),
        };

        let processed = aggregator.apply_dp(&update);

        // After clipping to norm 1.0, first element should be ~1.0 (plus tiny noise)
        assert!(
            processed.gradients[0].abs() < 1.5,
            "Clipped gradient should be near 1.0, got {}",
            processed.gradients[0]
        );
    }

    #[test]
    fn test_privacy_budget_tracking() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::FedAvg, 2)
            .with_dp_config(DifferentialPrivacyConfig {
                enabled: true,
                noise_multiplier: 1.0,
                clipping_threshold: 1.0,
                target_epsilon: 8.0,
                target_delta: 1e-5,
            });

        aggregator.initialize_model(vec![0.0; 5]);
        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 100);

        let tracker = aggregator.privacy_tracker().expect("tracker should exist");
        assert_eq!(tracker.rounds_tracked, 0);
        assert_eq!(tracker.spent_epsilon, 0.0);
        assert!(!tracker.is_exhausted());

        // Run a round
        aggregator.start_round().expect("start round");
        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![0.1; 5],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");
        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.2; 5],
                num_samples: 100,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");
        aggregator.aggregate().expect("aggregate");

        let tracker = aggregator.privacy_tracker().expect("tracker");
        assert_eq!(tracker.rounds_tracked, 1);
        assert!(tracker.spent_epsilon > 0.0);
        assert!(
            tracker.remaining_epsilon() < tracker.target_epsilon,
            "Budget should decrease"
        );
    }

    #[test]
    fn test_secagg_masks_cancel() {
        let mut aggregator = FederatedAggregator::new(AggregationAlgorithm::SecAgg, 2);

        aggregator.initialize_model(vec![0.0; 10]);
        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 100);

        aggregator.start_round().expect("start round");

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![0.1; 10],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");

        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.3; 10],
                num_samples: 100,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");

        let model = aggregator.aggregate().expect("aggregate");

        // With equal samples, the result should be the average = 0.2
        // The masks cancel so the result should be close to unmasked FedAvg
        for &w in &model.weights {
            assert!(
                (w - 0.2).abs() < 0.01,
                "SecAgg result should match FedAvg: got {w}, expected ~0.2"
            );
        }
    }

    #[test]
    fn test_secagg_participant_masking() {
        let p1 = SecAggParticipant::new("alice");
        let p2 = SecAggParticipant::new("bob");

        let keys = vec![
            ("alice".to_string(), p1.public_key),
            ("bob".to_string(), p2.public_key),
        ];

        let gradient1 = vec![1.0, 2.0, 3.0];
        let gradient2 = vec![4.0, 5.0, 6.0];

        let masked1 = p1.mask_gradient(&gradient1, &keys);
        let masked2 = p2.mask_gradient(&gradient2, &keys);

        // Masked values should differ from originals
        assert_ne!(masked1, gradient1);
        assert_ne!(masked2, gradient2);

        // But their sum should equal the sum of originals (masks cancel)
        for i in 0..3 {
            let masked_sum = masked1[i] + masked2[i];
            let original_sum = gradient1[i] + gradient2[i];
            assert!(
                (masked_sum - original_sum).abs() < 1e-4,
                "Mask cancellation failed at index {i}: masked_sum={masked_sum}, original_sum={original_sum}"
            );
        }
    }

    #[test]
    fn test_topk_compression() {
        let gradients = vec![0.1, -5.0, 0.3, 4.0, -0.2, 3.5, 0.01, -0.5];

        // Select top 3 by absolute value: indices 1 (-5.0), 3 (4.0), 5 (3.5)
        let compressed = topk_compress(&gradients, 3);

        assert_eq!(compressed.indices.len(), 3);
        assert_eq!(compressed.values.len(), 3);
        assert_eq!(compressed.original_dim, 8);

        // The selected values should be the largest by absolute value
        let mut abs_selected: Vec<f32> = compressed.values.iter().map(|v| v.abs()).collect();
        abs_selected.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        assert!(
            abs_selected[0] >= 4.0,
            "Top value should be >= 4.0, got {}",
            abs_selected[0]
        );

        // Decompression should restore selected values and zero others
        let decompressed = topk_decompress(&compressed);
        assert_eq!(decompressed.len(), 8);

        // Non-selected indices should be zero
        let selected_set: std::collections::HashSet<usize> =
            compressed.indices.iter().copied().collect();
        for (i, &v) in decompressed.iter().enumerate() {
            if !selected_set.contains(&i) {
                assert_eq!(v, 0.0, "Non-selected index {i} should be 0.0");
            }
        }
    }

    #[test]
    fn test_topk_roundtrip() {
        let gradients = vec![1.0, 2.0, 3.0];

        // k >= dim should preserve everything
        let compressed = topk_compress(&gradients, 10);
        let decompressed = topk_decompress(&compressed);

        assert_eq!(decompressed.len(), 3);
        for (i, (&orig, &dec)) in gradients.iter().zip(decompressed.iter()).enumerate() {
            assert!(
                (orig - dec).abs() < 1e-6,
                "Mismatch at index {i}: {orig} vs {dec}"
            );
        }
    }

    #[test]
    fn test_async_aggregator_basic() {
        let mut aggregator = AsyncFederatedAggregator::new(0.9, 2);

        aggregator.initialize_model(vec![0.0; 5]);
        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 100);

        // First update - should not trigger aggregation
        let result = aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![0.1; 5],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");
        assert!(result.is_none());
        assert_eq!(aggregator.pending_count(), 1);

        // Second update - should trigger aggregation (min_updates=2)
        let result = aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 1,
                gradients: vec![0.3; 5],
                num_samples: 100,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");
        assert!(result.is_some());
        assert_eq!(aggregator.pending_count(), 0);
        assert_eq!(aggregator.aggregation_count(), 1);
    }

    #[test]
    fn test_async_staleness_weighting() {
        let mut aggregator = AsyncFederatedAggregator::new(0.5, 2);

        aggregator.initialize_model(vec![0.0; 3]);
        aggregator.register_participant("ue-1", 100);
        aggregator.register_participant("ue-2", 100);

        // Simulate: ue-1 has a stale update (base_version=1, current=3)
        // ue-2 has a fresh update (base_version=3, current=3)
        // Manually set global model version to 3
        aggregator.global_model.as_mut().map(|m| m.version = 3);

        // Stale update (base_version=1, staleness=2, weight factor = 0.5^2 = 0.25)
        aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-1".to_string(),
                base_version: 1,
                gradients: vec![1.0, 1.0, 1.0],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");

        // Fresh update (base_version=3, staleness=0, weight factor = 0.5^0 = 1.0)
        let result = aggregator
            .submit_update(ModelUpdate {
                participant_id: "ue-2".to_string(),
                base_version: 3,
                gradients: vec![2.0, 2.0, 2.0],
                num_samples: 100,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            })
            .expect("submit");

        let model = result.expect("should aggregate");

        // Weights:
        //   ue-1: 100 * 0.5^2 = 25
        //   ue-2: 100 * 0.5^0 = 100
        //   total_weight = 125
        //   w1 = 25/125 = 0.2, w2 = 100/125 = 0.8
        //   result = 0.2*1.0 + 0.8*2.0 = 0.2 + 1.6 = 1.8
        for &w in &model.weights {
            assert!(
                (w - 1.8).abs() < 0.01,
                "Staleness-weighted result should be ~1.8, got {w}"
            );
        }
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

    #[test]
    fn test_gaussian_sampling_distribution() {
        // Verify that our Box-Muller implementation produces values with
        // approximately correct mean and std dev.
        let mut rng = rand::thread_rng();
        let n = 10_000;
        let sigma = 2.0f32;
        let samples: Vec<f32> = (0..n).map(|_| sample_gaussian(&mut rng, sigma)).collect();

        let mean = samples.iter().sum::<f32>() / n as f32;
        let variance =
            samples.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        let stddev = variance.sqrt();

        assert!(
            mean.abs() < 0.1,
            "Mean should be near 0, got {mean}"
        );
        assert!(
            (stddev - sigma).abs() < 0.2,
            "Stddev should be near {sigma}, got {stddev}"
        );
    }

    // ---------------------------------------------------------------------------
    // Tests for new features (A17)
    // ---------------------------------------------------------------------------

    #[test]
    fn test_hierarchical_aggregator() {
        let config = HierarchicalConfig {
            num_edge: 2,
            num_regional: 1,
            tier_weights: (0.4, 0.3, 0.3),
        };
        let aggregator = HierarchicalAggregator::new(config, AggregationAlgorithm::FedAvg);
        assert_eq!(aggregator.edge_aggregators.len(), 2);
        assert_eq!(aggregator.regional_aggregators.len(), 1);
    }

    #[test]
    fn test_client_selector_top_k() {
        let mut selector = ClientSelector::new(SelectionStrategy::TopK);

        selector.register(ClientProfile::new("client1".to_string(), 100.0, 50.0, 1000));
        selector.register(ClientProfile::new("client2".to_string(), 200.0, 100.0, 2000));
        selector.register(ClientProfile::new("client3".to_string(), 50.0, 25.0, 500));

        let selected = selector.select_clients(2);
        assert_eq!(selected.len(), 2);
        // Should prefer client2 and client1 based on higher utility
    }

    #[test]
    fn test_client_profile_utility() {
        let profile = ClientProfile::new("test".to_string(), 100.0, 50.0, 1000);
        let utility = profile.utility_score();
        assert!(utility > 0.0);
    }

    #[test]
    fn test_renyi_dp_tracker() {
        let mut tracker = RenyiDPTracker::new(8.0, 1e-5);
        assert_eq!(tracker.num_compositions, 0);

        // Record a Gaussian mechanism
        tracker.record_gaussian(1.0);
        assert_eq!(tracker.num_compositions, 1);

        let epsilon = tracker.get_epsilon();
        assert!(epsilon > 0.0);
        assert!(epsilon < 10.0);
    }

    #[test]
    fn test_zcdp_tracker() {
        let mut tracker = ZCDPTracker::new(8.0, 1e-5);

        tracker.record_gaussian(1.0);
        let epsilon = tracker.get_epsilon();
        assert!(epsilon > 0.0);
    }

    #[test]
    fn test_onebit_quantization() {
        let gradient = vec![0.5, -0.3, 0.8, -0.1];
        let quantized = onebit_quantize(&gradient);

        assert_eq!(quantized.scheme, QuantizationScheme::OneBit);
        assert_eq!(quantized.values.len(), 4);
        assert_eq!(quantized.values[0], 1); // 0.5 > 0
        assert_eq!(quantized.values[1], -1); // -0.3 < 0

        let dequantized = onebit_dequantize(&quantized);
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_ternary_quantization() {
        let gradient = vec![0.5, -0.3, 0.05, -0.8];
        let quantized = ternary_quantize(&gradient, 0.1);

        assert_eq!(quantized.scheme, QuantizationScheme::Ternary);
        // Small values (< threshold) should be 0
        assert_eq!(quantized.values.len(), 4);

        let dequantized = ternary_dequantize(&quantized);
        assert_eq!(dequantized.len(), 4);
    }

    #[test]
    fn test_personalization_layer() {
        let mut layer = PersonalizationLayer::new("client1".to_string(), 5, 0.01);
        let global = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let local_grad = vec![0.1, 0.2, 0.3, 0.4, 0.5];

        let personalized = layer.personalize(&global, &local_grad);
        assert_eq!(personalized.len(), 5);
        // Should be different from global due to local adaptation
    }

    #[test]
    fn test_krum_aggregation() {
        let updates = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 1.9, 3.2],
            vec![10.0, 10.0, 10.0], // Byzantine outlier
        ];

        let result = krum_aggregate(&updates, 1);
        assert_eq!(result.len(), 3);
        // Should select one of the first three (not the outlier)
        assert!(result[0] < 5.0);
    }

    #[test]
    fn test_multi_krum_aggregation() {
        let updates = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 1.9, 3.2],
            vec![10.0, 10.0, 10.0], // Byzantine outlier
        ];

        let result = multi_krum_aggregate(&updates, 1, 2);
        assert_eq!(result.len(), 3);
        // Should average top 2 selections
    }

    #[test]
    fn test_trimmed_mean_aggregation() {
        let updates = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.5, 2.5, 3.5],
            vec![2.0, 3.0, 4.0],
            vec![100.0, 100.0, 100.0], // Outlier
        ];

        let result = trimmed_mean_aggregate(&updates, 0.5);
        assert_eq!(result.len(), 3);
        // Should be close to middle values, excluding outlier
        assert!(result[0] < 10.0);
    }

    #[test]
    fn test_median_aggregation() {
        let updates = vec![
            vec![1.0, 2.0, 3.0],
            vec![2.0, 3.0, 4.0],
            vec![3.0, 4.0, 5.0],
            vec![100.0, 100.0, 100.0], // Outlier
        ];

        let result = median_aggregate(&updates);
        assert_eq!(result.len(), 3);
        // Median should be robust to outlier
        assert_eq!(result[0], 2.5); // Median of [1, 2, 3, 100]
    }

    #[test]
    fn test_she_fl_aggregator() {
        let hierarchical_config = HierarchicalConfig::default();
        let she_config = SHEFLConfig::default();
        let mut aggregator = SHEFLAggregator::new(
            hierarchical_config,
            she_config,
            AggregationAlgorithm::FedAvg,
        );

        aggregator.assign_tier("client1".to_string(), FLTier::Edge);
        assert_eq!(
            aggregator.get_participant_tier("client1"),
            Some(FLTier::Edge)
        );
    }
}
