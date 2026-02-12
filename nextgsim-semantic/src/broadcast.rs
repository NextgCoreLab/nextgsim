//! Multi-User Semantic Broadcast (A18.9)
//!
//! Implements efficient broadcasting where a base semantic representation
//! is shared among all users, with user-specific refinements sent separately.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{ChannelQuality, SemanticFeatures, SemanticTask};

/// User profile for personalized broadcasting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    /// User identifier
    pub user_id: String,
    /// User's channel quality
    pub channel: ChannelQuality,
    /// User-specific interests/preferences (sparse feature vector)
    pub interests: Vec<f32>,
    /// Required quality level (0.0 to 1.0)
    pub quality_requirement: f32,
}

impl UserProfile {
    /// Creates a new user profile
    pub fn new(user_id: String, channel: ChannelQuality, interests: Vec<f32>) -> Self {
        Self {
            user_id,
            channel,
            interests,
            quality_requirement: 0.8,
        }
    }
}

/// Base semantic layer shared by all users
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseLayer {
    /// Task ID
    pub task_id: u32,
    /// Common features (essential for all users)
    pub features: Vec<f32>,
    /// Importance weights
    pub importance: Vec<f32>,
    /// Original dimensions
    pub original_dims: Vec<usize>,
}

impl BaseLayer {
    /// Creates a base layer from semantic features
    pub fn from_features(features: &SemanticFeatures, keep_ratio: f32) -> Self {
        // Keep only the most important features for the base layer
        let keep_count = ((features.num_features() as f32 * keep_ratio) as usize).max(1);

        let mut indexed: Vec<(usize, f32, f32)> = (0..features.num_features())
            .map(|i| {
                (
                    i,
                    features.features[i],
                    features.importance.get(i).copied().unwrap_or(1.0),
                )
            })
            .collect();

        // Sort by importance
        indexed.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(keep_count);

        let base_features: Vec<f32> = indexed.iter().map(|(_, f, _)| *f).collect();
        let base_importance: Vec<f32> = indexed.iter().map(|(_, _, i)| *i).collect();

        Self {
            task_id: features.task_id,
            features: base_features,
            importance: base_importance,
            original_dims: features.original_dims.clone(),
        }
    }

    /// Returns the number of features in the base layer
    pub fn num_features(&self) -> usize {
        self.features.len()
    }
}

/// User-specific refinement layer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefinementLayer {
    /// User ID this refinement is for
    pub user_id: String,
    /// Refinement features (what's missing from base layer)
    pub features: Vec<f32>,
    /// Indices in the original feature vector
    pub indices: Vec<usize>,
}

impl RefinementLayer {
    /// Creates an empty refinement
    pub fn empty(user_id: String) -> Self {
        Self {
            user_id,
            features: Vec::new(),
            indices: Vec::new(),
        }
    }

    /// Returns the number of refinement features
    pub fn num_features(&self) -> usize {
        self.features.len()
    }
}

/// Broadcast message containing base + refinements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastMessage {
    /// Shared base layer
    pub base: BaseLayer,
    /// User-specific refinements
    pub refinements: HashMap<String, RefinementLayer>,
}

impl BroadcastMessage {
    /// Creates a new broadcast message
    pub fn new(base: BaseLayer) -> Self {
        Self {
            base,
            refinements: HashMap::new(),
        }
    }

    /// Adds a refinement for a specific user
    pub fn add_refinement(&mut self, user_id: String, refinement: RefinementLayer) {
        self.refinements.insert(user_id, refinement);
    }

    /// Reconstructs features for a specific user
    pub fn reconstruct_for_user(&self, user_id: &str) -> SemanticFeatures {
        let base_features = self.base.features.clone();
        let base_importance = self.base.importance.clone();

        // Apply user-specific refinement if available
        let (features, importance) = if let Some(refinement) = self.refinements.get(user_id) {
            // Merge base and refinement
            let mut merged_features = base_features.clone();
            let mut merged_importance = base_importance.clone();

            for (i, &idx) in refinement.indices.iter().enumerate() {
                if let Some(&feat) = refinement.features.get(i) {
                    // Append or update feature
                    if idx < merged_features.len() {
                        merged_features[idx] = feat;
                    } else {
                        merged_features.push(feat);
                        merged_importance.push(1.0);
                    }
                }
            }

            (merged_features, merged_importance)
        } else {
            (base_features, base_importance)
        };

        SemanticFeatures::new(self.base.task_id, features, self.base.original_dims.clone())
            .with_importance(importance)
    }

    /// Returns the total size (base + all refinements) in number of floats
    pub fn total_size(&self) -> usize {
        let base_size = self.base.num_features();
        let refinement_size: usize = self.refinements.values().map(RefinementLayer::num_features).sum();
        base_size + refinement_size
    }
}

/// Semantic broadcast encoder
pub struct BroadcastEncoder {
    /// Base layer keep ratio (fraction of features in base layer)
    base_ratio: f32,
    /// Task
    task: SemanticTask,
}

impl BroadcastEncoder {
    /// Creates a new broadcast encoder
    pub fn new(base_ratio: f32, task: SemanticTask) -> Self {
        Self { base_ratio, task }
    }

    /// Encodes data for multi-user broadcast
    pub fn encode(
        &self,
        data: &[f32],
        users: &[UserProfile],
    ) -> BroadcastMessage {
        // First, create semantic features
        let task_id = task_to_id(self.task);
        let features = SemanticFeatures::new(task_id, data.to_vec(), vec![data.len()]);

        // Create base layer (shared by all users)
        let base = BaseLayer::from_features(&features, self.base_ratio);

        let mut message = BroadcastMessage::new(base);

        // Create user-specific refinements
        for user in users {
            let refinement = self.create_refinement(&features, user);
            message.add_refinement(user.user_id.clone(), refinement);
        }

        message
    }

    /// Creates a user-specific refinement layer
    fn create_refinement(&self, features: &SemanticFeatures, user: &UserProfile) -> RefinementLayer {
        // Determine how many features this user needs based on channel quality
        let user_ratio = 1.0 / user.channel.recommended_compression();
        let user_ratio = user_ratio.clamp(self.base_ratio, 1.0);

        let additional_ratio = user_ratio - self.base_ratio;
        if additional_ratio <= 0.0 {
            return RefinementLayer::empty(user.user_id.clone());
        }

        let additional_count = (features.num_features() as f32 * additional_ratio) as usize;

        // Select features based on user interests
        let mut scored: Vec<(usize, f32, f32)> = (0..features.num_features())
            .map(|i| {
                let feature = features.features[i];
                let importance = features.importance.get(i).copied().unwrap_or(1.0);

                // Compute relevance to user interests
                let user_relevance = if i < user.interests.len() {
                    user.interests[i].abs()
                } else {
                    0.0
                };

                let score = importance * (1.0 + user_relevance);
                (i, feature, score)
            })
            .collect();

        // Sort by score
        scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Skip base layer features and take additional
        let start = (features.num_features() as f32 * self.base_ratio) as usize;
        let refinement_features: Vec<_> = scored
            .iter()
            .skip(start)
            .take(additional_count)
            .collect();

        let features: Vec<f32> = refinement_features.iter().map(|(_, f, _)| *f).collect();
        let indices: Vec<usize> = refinement_features.iter().map(|(i, _, _)| *i).collect();

        RefinementLayer {
            user_id: user.user_id.clone(),
            features,
            indices,
        }
    }
}

/// Converts `SemanticTask` to task ID
fn task_to_id(task: SemanticTask) -> u32 {
    match task {
        SemanticTask::ImageClassification => 0,
        SemanticTask::ObjectDetection => 1,
        SemanticTask::SpeechRecognition => 2,
        SemanticTask::TextUnderstanding => 3,
        SemanticTask::SensorFusion => 4,
        SemanticTask::VideoAnalytics => 5,
        SemanticTask::Custom(id) => id,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_user(id: &str, snr: f32) -> UserProfile {
        UserProfile::new(
            id.to_string(),
            ChannelQuality::new(snr, 100.0, 0.01),
            vec![0.5; 10],
        )
    }

    #[test]
    fn test_user_profile() {
        let user = create_test_user("user1", 15.0);
        assert_eq!(user.user_id, "user1");
        assert_eq!(user.quality_requirement, 0.8);
    }

    #[test]
    fn test_base_layer_creation() {
        let features = SemanticFeatures::new(
            0,
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5],
        ).with_importance(vec![0.9, 0.5, 0.8, 0.3, 0.7]);

        let base = BaseLayer::from_features(&features, 0.6);

        // Should keep 60% of features = 3 features
        assert_eq!(base.num_features(), 3);
    }

    #[test]
    fn test_refinement_layer() {
        let refinement = RefinementLayer::empty("user1".to_string());
        assert_eq!(refinement.num_features(), 0);
        assert_eq!(refinement.user_id, "user1");
    }

    #[test]
    fn test_broadcast_message() {
        let features = SemanticFeatures::new(0, vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let base = BaseLayer::from_features(&features, 0.5);

        let mut message = BroadcastMessage::new(base);
        assert_eq!(message.refinements.len(), 0);

        let refinement = RefinementLayer {
            user_id: "user1".to_string(),
            features: vec![6.0, 7.0],
            indices: vec![3, 4],
        };
        message.add_refinement("user1".to_string(), refinement);

        assert_eq!(message.refinements.len(), 1);
    }

    #[test]
    fn test_reconstruct_for_user() {
        let features = SemanticFeatures::new(0, vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let base = BaseLayer::from_features(&features, 0.4);

        let mut message = BroadcastMessage::new(base);

        // Add refinement for user1
        let refinement = RefinementLayer {
            user_id: "user1".to_string(),
            features: vec![6.0, 7.0],
            indices: vec![2, 3],
        };
        message.add_refinement("user1".to_string(), refinement);

        // Reconstruct for user1
        let reconstructed = message.reconstruct_for_user("user1");
        assert!(reconstructed.num_features() >= 2);

        // Reconstruct for user2 (no refinement)
        let reconstructed2 = message.reconstruct_for_user("user2");
        assert!(reconstructed2.num_features() > 0);
    }

    #[test]
    fn test_broadcast_encoder() {
        let encoder = BroadcastEncoder::new(0.5, SemanticTask::ImageClassification);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let users = vec![
            create_test_user("user1", 20.0), // Good channel
            create_test_user("user2", 5.0),  // Poor channel
        ];

        let message = encoder.encode(&data, &users);

        assert!(message.base.num_features() > 0);
        assert_eq!(message.refinements.len(), 2);
    }

    #[test]
    fn test_total_size() {
        let features = SemanticFeatures::new(0, vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let base = BaseLayer::from_features(&features, 0.4);

        let mut message = BroadcastMessage::new(base.clone());

        let refinement1 = RefinementLayer {
            user_id: "user1".to_string(),
            features: vec![6.0, 7.0],
            indices: vec![3, 4],
        };
        message.add_refinement("user1".to_string(), refinement1);

        let total = message.total_size();
        assert_eq!(total, base.num_features() + 2);
    }

    #[test]
    fn test_user_specific_refinement() {
        let encoder = BroadcastEncoder::new(0.3, SemanticTask::SensorFusion);

        let data = vec![1.0; 10];
        let mut user = create_test_user("user1", 10.0);
        user.interests = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let users = vec![user];
        let message = encoder.encode(&data, &users);

        let refinement = message.refinements.get("user1").unwrap();
        // Should have some refinement features
        assert!(refinement.num_features() > 0);
    }
}
