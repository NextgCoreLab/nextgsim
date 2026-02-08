//! Agent <-> NWDAF Cross-Crate Integration (A19.7)
//!
//! Enables AI agents to consume NWDAF analytics for informed decision-making.
//! Agents can subscribe to analytics, query on-demand insights, and use
//! analytics data to drive intent-based network operations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};

use nextgsim_nwdaf::analytics_id::{AnalyticsId, AnalyticsTarget};
use nextgsim_nwdaf::anlf::AnalyticsResult;

/// Configuration for the Agent-NWDAF bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentNwdafBridgeConfig {
    /// Analytics IDs this agent is interested in
    pub subscribed_analytics: Vec<AnalyticsId>,
    /// Maximum analytics cache size
    pub max_cache_size: usize,
    /// Minimum confidence to cache an analytics result
    pub min_confidence: f32,
}

impl Default for AgentNwdafBridgeConfig {
    fn default() -> Self {
        Self {
            subscribed_analytics: vec![
                AnalyticsId::UeMobility,
                AnalyticsId::NfLoad,
                AnalyticsId::QosSustainability,
            ],
            max_cache_size: 1000,
            min_confidence: 0.5,
        }
    }
}

/// Cached analytics entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedAnalytics {
    /// The analytics result
    pub result: AnalyticsResult,
    /// When it was cached (ms)
    pub cached_at_ms: u64,
    /// Whether it has been consumed by an intent
    pub consumed: bool,
}

/// Bridge between Agent framework and NWDAF analytics
///
/// Maintains a local cache of analytics results received from the NWDAF
/// and provides query methods for agents to consume analytics when making
/// decisions.
pub struct AgentNwdafBridge {
    /// Configuration
    config: AgentNwdafBridgeConfig,
    /// Cached analytics indexed by analytics ID
    cache: HashMap<AnalyticsId, Vec<CachedAnalytics>>,
    /// Total analytics received
    received_count: u64,
}

impl AgentNwdafBridge {
    /// Creates a new bridge with the given configuration
    pub fn new(config: AgentNwdafBridgeConfig) -> Self {
        Self {
            config,
            cache: HashMap::new(),
            received_count: 0,
        }
    }

    /// Ingests an analytics result from the NWDAF
    ///
    /// Caches the result if it matches the agent's subscribed analytics
    /// and meets the minimum confidence threshold.
    pub fn ingest_analytics(&mut self, result: AnalyticsResult) {
        if !self.config.subscribed_analytics.contains(&result.analytics_id) {
            debug!(
                "AgentNwdafBridge: ignoring {:?} (not subscribed)",
                result.analytics_id
            );
            return;
        }

        if result.confidence < self.config.min_confidence {
            debug!(
                "AgentNwdafBridge: ignoring {:?} (confidence {:.2} below {:.2})",
                result.analytics_id, result.confidence, self.config.min_confidence
            );
            return;
        }

        let entry = CachedAnalytics {
            cached_at_ms: timestamp_now(),
            result: result.clone(),
            consumed: false,
        };

        let entries = self.cache.entry(result.analytics_id).or_default();
        entries.push(entry);

        // Prune old entries per analytics ID
        let max_per_id = self.config.max_cache_size / self.config.subscribed_analytics.len().max(1);
        if entries.len() > max_per_id {
            let to_remove = entries.len() - max_per_id;
            entries.drain(0..to_remove);
        }

        self.received_count += 1;
        info!(
            "AgentNwdafBridge: cached {:?} analytics (total={})",
            result.analytics_id, self.received_count
        );
    }

    /// Queries the latest analytics result for a given analytics ID and target
    pub fn latest_analytics(
        &self,
        analytics_id: AnalyticsId,
        target: &AnalyticsTarget,
    ) -> Option<&AnalyticsResult> {
        self.cache.get(&analytics_id).and_then(|entries| {
            entries
                .iter()
                .rev()
                .find(|e| target_matches(target, &e.result.target))
                .map(|e| &e.result)
        })
    }

    /// Returns all cached analytics for a given analytics ID
    pub fn get_analytics(&self, analytics_id: AnalyticsId) -> Vec<&AnalyticsResult> {
        self.cache
            .get(&analytics_id)
            .map(|entries| entries.iter().map(|e| &e.result).collect())
            .unwrap_or_default()
    }

    /// Builds a decision context by collecting latest analytics for all subscribed IDs
    ///
    /// Returns a map from analytics ID to the latest result (if any).
    pub fn decision_context(&self) -> HashMap<AnalyticsId, &AnalyticsResult> {
        let mut context = HashMap::new();
        for &aid in &self.config.subscribed_analytics {
            if let Some(entries) = self.cache.get(&aid) {
                if let Some(latest) = entries.last() {
                    context.insert(aid, &latest.result);
                }
            }
        }
        context
    }

    /// Returns the total number of analytics received
    pub fn received_count(&self) -> u64 {
        self.received_count
    }

    /// Returns the total number of cached entries
    pub fn cache_size(&self) -> usize {
        self.cache.values().map(std::vec::Vec::len).sum()
    }

    /// Returns the bridge configuration
    pub fn config(&self) -> &AgentNwdafBridgeConfig {
        &self.config
    }
}

/// Checks if a subscription target matches a result target
fn target_matches(subscription: &AnalyticsTarget, result: &AnalyticsTarget) -> bool {
    match (subscription, result) {
        (AnalyticsTarget::Any, _) => true,
        (AnalyticsTarget::Ue { ue_id: a }, AnalyticsTarget::Ue { ue_id: b }) => a == b,
        (AnalyticsTarget::Cell { cell_id: a }, AnalyticsTarget::Cell { cell_id: b }) => a == b,
        (AnalyticsTarget::Slice { snssai: a }, AnalyticsTarget::Slice { snssai: b }) => a == b,
        _ => false,
    }
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
    use nextgsim_nwdaf::analytics_id::AnalyticsOutputType;
    use nextgsim_nwdaf::anlf::AnalyticsPayload;

    fn make_mobility_result(ue_id: i32, confidence: f32) -> AnalyticsResult {
        AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 1000,
            confidence,
            payload: AnalyticsPayload::UeMobility {
                trajectory: None,
                handover_recommendation: None,
            },
        }
    }

    #[test]
    fn test_bridge_creation() {
        let bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig::default());
        assert_eq!(bridge.received_count(), 0);
        assert_eq!(bridge.cache_size(), 0);
    }

    #[test]
    fn test_ingest_analytics() {
        let mut bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig::default());

        bridge.ingest_analytics(make_mobility_result(1, 0.9));
        assert_eq!(bridge.received_count(), 1);
        assert_eq!(bridge.cache_size(), 1);
    }

    #[test]
    fn test_ingest_ignored_unsubscribed() {
        let mut bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig {
            subscribed_analytics: vec![AnalyticsId::NfLoad],
            ..Default::default()
        });

        // UeMobility is not subscribed
        bridge.ingest_analytics(make_mobility_result(1, 0.9));
        assert_eq!(bridge.received_count(), 0);
    }

    #[test]
    fn test_ingest_ignored_low_confidence() {
        let mut bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig {
            min_confidence: 0.95,
            ..Default::default()
        });

        bridge.ingest_analytics(make_mobility_result(1, 0.5));
        assert_eq!(bridge.received_count(), 0);
    }

    #[test]
    fn test_latest_analytics() {
        let mut bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig::default());

        bridge.ingest_analytics(make_mobility_result(1, 0.8));
        bridge.ingest_analytics(make_mobility_result(1, 0.9));

        let latest = bridge
            .latest_analytics(AnalyticsId::UeMobility, &AnalyticsTarget::Ue { ue_id: 1 })
            .unwrap();
        assert!((latest.confidence - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_decision_context() {
        let mut bridge = AgentNwdafBridge::new(AgentNwdafBridgeConfig::default());
        bridge.ingest_analytics(make_mobility_result(1, 0.9));

        let context = bridge.decision_context();
        assert!(context.contains_key(&AnalyticsId::UeMobility));
        assert!(!context.contains_key(&AnalyticsId::NfLoad));
    }
}
