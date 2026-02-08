//! Reinforcement Learning (RL) based agent learning and adaptation
//!
//! This module implements online model updates with reinforcement learning,
//! allowing agents to learn from experience and improve their decision-making
//! over time.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use thiserror::Error;
use tracing::{debug, info};

use crate::{AgentId, IntentType};

/// Errors that can occur during RL learning
#[derive(Error, Debug)]
pub enum LearningError {
    /// Not enough experience for learning
    #[error("Insufficient experience: need at least {minimum}, have {actual}")]
    InsufficientExperience { actual: usize, minimum: usize },

    /// Invalid reward value
    #[error("Invalid reward: {reason}")]
    InvalidReward { reason: String },

    /// Model update failed
    #[error("Model update failed: {reason}")]
    UpdateFailed { reason: String },

    /// Invalid state
    #[error("Invalid state: {reason}")]
    InvalidState { reason: String },
}

/// State representation for RL
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct State {
    /// Network load (0.0 to 1.0)
    pub network_load: f32,
    /// Number of active UEs
    pub active_ues: usize,
    /// Average latency in milliseconds
    pub avg_latency_ms: f32,
    /// Success rate of recent intents (0.0 to 1.0)
    pub success_rate: f32,
    /// Custom state features
    pub features: HashMap<String, f32>,
}

impl State {
    /// Creates a new state with default values
    pub fn new() -> Self {
        Self {
            network_load: 0.0,
            active_ues: 0,
            avg_latency_ms: 0.0,
            success_rate: 1.0,
            features: HashMap::new(),
        }
    }

    /// Adds a custom feature
    pub fn with_feature(mut self, key: impl Into<String>, value: f32) -> Self {
        self.features.insert(key.into(), value);
        self
    }
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

/// Action taken by an agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    /// Intent type
    pub intent_type: IntentType,
    /// Action parameters
    pub parameters: HashMap<String, String>,
}

/// Experience tuple for RL: (state, action, reward, next_state)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Experience {
    /// State before action
    pub state: State,
    /// Action taken
    pub action: Action,
    /// Reward received
    pub reward: f32,
    /// State after action
    pub next_state: State,
    /// Whether this was a terminal state
    pub terminal: bool,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Q-value for state-action pair
#[derive(Debug, Clone, Copy)]
struct QValue {
    /// Q-value estimate
    value: f32,
    /// Number of times this state-action has been visited
    visits: usize,
}

/// RL learning algorithm type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Q-Learning
    QLearning,
    /// SARSA (State-Action-Reward-State-Action)
    Sarsa,
    /// Deep Q-Network (placeholder)
    Dqn,
}

/// Configuration for RL learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningConfig {
    /// Learning algorithm
    pub algorithm: Algorithm,
    /// Learning rate (alpha)
    pub learning_rate: f32,
    /// Discount factor (gamma)
    pub discount_factor: f32,
    /// Exploration rate (epsilon)
    pub exploration_rate: f32,
    /// Minimum exploration rate
    pub min_exploration_rate: f32,
    /// Exploration decay rate
    pub exploration_decay: f32,
    /// Replay buffer size
    pub replay_buffer_size: usize,
    /// Minimum experiences before learning
    pub min_experiences: usize,
    /// Batch size for learning
    pub batch_size: usize,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::QLearning,
            learning_rate: 0.1,
            discount_factor: 0.95,
            exploration_rate: 0.3,
            min_exploration_rate: 0.01,
            exploration_decay: 0.995,
            replay_buffer_size: 10000,
            min_experiences: 100,
            batch_size: 32,
        }
    }
}

/// RL-based learning agent
pub struct RLAgent {
    /// Agent ID
    agent_id: AgentId,
    /// Configuration
    config: LearningConfig,
    /// Q-table (state hash -> action hash -> Q-value)
    q_table: HashMap<u64, HashMap<u64, QValue>>,
    /// Experience replay buffer
    replay_buffer: VecDeque<Experience>,
    /// Current exploration rate
    current_exploration_rate: f32,
    /// Total learning steps
    total_steps: usize,
    /// Total rewards accumulated
    total_reward: f32,
}

impl RLAgent {
    /// Creates a new RL agent
    pub fn new(agent_id: AgentId, config: LearningConfig) -> Self {
        let current_exploration_rate = config.exploration_rate;

        Self {
            agent_id,
            config,
            q_table: HashMap::new(),
            replay_buffer: VecDeque::new(),
            current_exploration_rate,
            total_steps: 0,
            total_reward: 0.0,
        }
    }

    /// Adds an experience to the replay buffer
    pub fn add_experience(&mut self, experience: Experience) {
        self.replay_buffer.push_back(experience);

        // Limit buffer size
        while self.replay_buffer.len() > self.config.replay_buffer_size {
            self.replay_buffer.pop_front();
        }

        debug!(
            "Agent {}: Added experience (buffer size: {})",
            self.agent_id,
            self.replay_buffer.len()
        );
    }

    /// Selects an action using epsilon-greedy policy
    pub fn select_action(&self, state: &State, available_actions: &[Action]) -> Action {
        if available_actions.is_empty() {
            panic!("No available actions");
        }

        // Epsilon-greedy exploration
        let random_val: f32 = rand::random();
        if random_val < self.current_exploration_rate {
            // Explore: random action
            let idx = (rand::random::<f32>() * available_actions.len() as f32) as usize;
            available_actions[idx.min(available_actions.len() - 1)].clone()
        } else {
            // Exploit: best known action
            self.best_action(state, available_actions)
        }
    }

    /// Returns the best action for a state according to Q-values
    fn best_action(&self, state: &State, available_actions: &[Action]) -> Action {
        let state_hash = self.hash_state(state);

        let mut best_action = available_actions[0].clone();
        let mut best_q = f32::NEG_INFINITY;

        if let Some(action_map) = self.q_table.get(&state_hash) {
            for action in available_actions {
                let action_hash = self.hash_action(action);
                if let Some(q_value) = action_map.get(&action_hash) {
                    if q_value.value > best_q {
                        best_q = q_value.value;
                        best_action = action.clone();
                    }
                }
            }
        }

        // If no Q-values found, return first action
        best_action
    }

    /// Performs a learning update from experiences
    pub fn learn(&mut self) -> Result<f32, LearningError> {
        if self.replay_buffer.len() < self.config.min_experiences {
            return Err(LearningError::InsufficientExperience {
                actual: self.replay_buffer.len(),
                minimum: self.config.min_experiences,
            });
        }

        let batch_size = self.config.batch_size.min(self.replay_buffer.len());
        let mut total_loss = 0.0;

        // Sample random batch from replay buffer
        for _ in 0..batch_size {
            let idx = (rand::random::<f32>() * self.replay_buffer.len() as f32) as usize;
            let idx = idx.min(self.replay_buffer.len() - 1);

            if let Some(exp) = self.replay_buffer.get(idx).cloned() {
                let loss = self.update_q_value(&exp)?;
                total_loss += loss;
            }
        }

        // Decay exploration rate
        self.current_exploration_rate =
            (self.current_exploration_rate * self.config.exploration_decay)
                .max(self.config.min_exploration_rate);

        self.total_steps += 1;

        let avg_loss = total_loss / batch_size as f32;
        info!(
            "Agent {}: Learning step {} complete, avg loss: {:.4}, exploration: {:.3}",
            self.agent_id, self.total_steps, avg_loss, self.current_exploration_rate
        );

        Ok(avg_loss)
    }

    /// Updates Q-value for a single experience
    fn update_q_value(&mut self, exp: &Experience) -> Result<f32, LearningError> {
        let state_hash = self.hash_state(&exp.state);
        let action_hash = self.hash_action(&exp.action);
        let next_state_hash = self.hash_state(&exp.next_state);

        // Get current Q-value (read first)
        let old_value = self
            .q_table
            .get(&state_hash)
            .and_then(|action_map| action_map.get(&action_hash))
            .map(|q| q.value)
            .unwrap_or(0.0);

        // Calculate target Q-value
        let target = if exp.terminal {
            exp.reward
        } else {
            // Find max Q-value for next state
            let max_next_q = self
                .q_table
                .get(&next_state_hash)
                .and_then(|action_map| {
                    action_map.values().map(|q| q.value).max_by(|a, b| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                    })
                })
                .unwrap_or(0.0);

            exp.reward + self.config.discount_factor * max_next_q
        };

        // Q-learning update
        let new_value = old_value + self.config.learning_rate * (target - old_value);

        // Update Q-value (write after all reads are complete)
        let current_q = self
            .q_table
            .entry(state_hash)
            .or_default()
            .entry(action_hash)
            .or_insert(QValue {
                value: 0.0,
                visits: 0,
            });

        current_q.value = new_value;
        current_q.visits += 1;

        self.total_reward += exp.reward;

        // Return loss (TD error)
        Ok((target - old_value).abs())
    }

    /// Computes a hash for a state (simplified)
    fn hash_state(&self, state: &State) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();

        // Discretize continuous values for hashing
        ((state.network_load * 10.0) as i32).hash(&mut hasher);
        state.active_ues.hash(&mut hasher);
        ((state.avg_latency_ms / 10.0) as i32).hash(&mut hasher);
        ((state.success_rate * 10.0) as i32).hash(&mut hasher);

        hasher.finish()
    }

    /// Computes a hash for an action (simplified)
    fn hash_action(&self, action: &Action) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        format!("{:?}", action.intent_type).hash(&mut hasher);
        hasher.finish()
    }

    /// Returns the agent ID
    pub fn agent_id(&self) -> &AgentId {
        &self.agent_id
    }

    /// Returns the total number of learning steps
    pub fn total_steps(&self) -> usize {
        self.total_steps
    }

    /// Returns the total accumulated reward
    pub fn total_reward(&self) -> f32 {
        self.total_reward
    }

    /// Returns the average reward per step
    pub fn avg_reward(&self) -> f32 {
        if self.total_steps > 0 {
            self.total_reward / self.total_steps as f32
        } else {
            0.0
        }
    }

    /// Returns the current exploration rate
    pub fn exploration_rate(&self) -> f32 {
        self.current_exploration_rate
    }

    /// Returns the size of the Q-table
    pub fn q_table_size(&self) -> usize {
        self.q_table.values().map(std::collections::HashMap::len).sum()
    }

    /// Returns the number of experiences in the replay buffer
    pub fn buffer_size(&self) -> usize {
        self.replay_buffer.len()
    }

    /// Resets the agent's learning state
    pub fn reset(&mut self) {
        self.q_table.clear();
        self.replay_buffer.clear();
        self.current_exploration_rate = self.config.exploration_rate;
        self.total_steps = 0;
        self.total_reward = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_state() -> State {
        State {
            network_load: 0.5,
            active_ues: 10,
            avg_latency_ms: 50.0,
            success_rate: 0.9,
            features: HashMap::new(),
        }
    }

    fn create_test_action() -> Action {
        Action {
            intent_type: IntentType::OptimizeResources,
            parameters: HashMap::new(),
        }
    }

    #[test]
    fn test_rl_agent_creation() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig::default();
        let agent = RLAgent::new(agent_id.clone(), config);

        assert_eq!(agent.agent_id(), &agent_id);
        assert_eq!(agent.total_steps(), 0);
        assert_eq!(agent.buffer_size(), 0);
    }

    #[test]
    fn test_add_experience() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig::default();
        let mut agent = RLAgent::new(agent_id, config);

        let exp = Experience {
            state: create_test_state(),
            action: create_test_action(),
            reward: 1.0,
            next_state: create_test_state(),
            terminal: false,
            timestamp_ms: 0,
        };

        agent.add_experience(exp);
        assert_eq!(agent.buffer_size(), 1);
    }

    #[test]
    fn test_select_action() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig::default();
        let agent = RLAgent::new(agent_id, config);

        let state = create_test_state();
        let actions = vec![
            create_test_action(),
            Action {
                intent_type: IntentType::AdjustQos,
                parameters: HashMap::new(),
            },
        ];

        let selected = agent.select_action(&state, &actions);
        // Should select one of the available actions
        assert!(actions.iter().any(|a| format!("{:?}", a.intent_type) == format!("{:?}", selected.intent_type)));
    }

    #[test]
    fn test_learning_insufficient_experience() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig {
            min_experiences: 100,
            ..Default::default()
        };
        let mut agent = RLAgent::new(agent_id, config);

        // Add only 10 experiences
        for _ in 0..10 {
            let exp = Experience {
                state: create_test_state(),
                action: create_test_action(),
                reward: 1.0,
                next_state: create_test_state(),
                terminal: false,
                timestamp_ms: 0,
            };
            agent.add_experience(exp);
        }

        let result = agent.learn();
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            LearningError::InsufficientExperience { .. }
        ));
    }

    #[test]
    fn test_learning_update() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig {
            min_experiences: 10,
            batch_size: 5,
            ..Default::default()
        };
        let mut agent = RLAgent::new(agent_id, config);

        // Add sufficient experiences
        for i in 0..20 {
            let exp = Experience {
                state: create_test_state(),
                action: create_test_action(),
                reward: 1.0 + (i as f32 * 0.1),
                next_state: create_test_state(),
                terminal: i == 19,
                timestamp_ms: i as u64,
            };
            agent.add_experience(exp);
        }

        let result = agent.learn();
        assert!(result.is_ok());
        assert_eq!(agent.total_steps(), 1);
        assert!(agent.q_table_size() > 0);
    }

    #[test]
    fn test_exploration_decay() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig {
            exploration_rate: 1.0,
            min_exploration_rate: 0.1,
            exploration_decay: 0.9,
            min_experiences: 10,
            ..Default::default()
        };
        let mut agent = RLAgent::new(agent_id, config);

        let initial_rate = agent.exploration_rate();

        // Add experiences and learn
        for _ in 0..20 {
            let exp = Experience {
                state: create_test_state(),
                action: create_test_action(),
                reward: 1.0,
                next_state: create_test_state(),
                terminal: false,
                timestamp_ms: 0,
            };
            agent.add_experience(exp);
        }

        agent.learn().unwrap();
        let after_learning = agent.exploration_rate();

        assert!(after_learning < initial_rate);
        assert!(after_learning >= 0.1); // Min exploration rate
    }

    #[test]
    fn test_reset() {
        let agent_id = AgentId::new("rl-agent-1");
        let config = LearningConfig::default();
        let mut agent = RLAgent::new(agent_id, config);

        // Add some state
        agent.total_steps = 10;
        agent.total_reward = 50.0;
        agent.add_experience(Experience {
            state: create_test_state(),
            action: create_test_action(),
            reward: 1.0,
            next_state: create_test_state(),
            terminal: false,
            timestamp_ms: 0,
        });

        agent.reset();

        assert_eq!(agent.total_steps(), 0);
        assert_eq!(agent.total_reward(), 0.0);
        assert_eq!(agent.buffer_size(), 0);
        assert_eq!(agent.q_table_size(), 0);
    }

    #[test]
    fn test_state_with_features() {
        let state = State::new()
            .with_feature("custom_metric", 0.75)
            .with_feature("another_metric", 1.5);

        assert_eq!(state.features.get("custom_metric"), Some(&0.75));
        assert_eq!(state.features.get("another_metric"), Some(&1.5));
    }
}
