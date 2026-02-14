//! NWDAF-NWDAF Coordination
//!
//! Implements inter-NWDAF coordination for distributed analytics, model sharing,
//! and load distribution across multiple NWDAF instances.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::analytics_id::AnalyticsId;
use crate::error::NwdafError;

/// NWDAF instance descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwdafInstance {
    /// Instance identifier
    pub instance_id: String,
    /// Instance address (hostname:port or URL)
    pub address: String,
    /// Supported analytics IDs
    pub supported_analytics: Vec<AnalyticsId>,
    /// Current load level (0.0 to 1.0)
    pub load_level: f32,
    /// Available capacity (0.0 to 1.0)
    pub available_capacity: f32,
    /// Whether instance is currently reachable
    pub is_active: bool,
    /// Last heartbeat timestamp
    pub last_heartbeat_ms: u64,
    /// Instance capabilities
    pub capabilities: NwdafCapabilities,
}

/// NWDAF instance capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NwdafCapabilities {
    /// Whether instance has MTLF (model training capability)
    pub has_mtlf: bool,
    /// Whether instance has `AnLF` (analytics capability)
    pub has_anlf: bool,
    /// Maximum concurrent analytics requests
    pub max_concurrent_requests: usize,
    /// Supported model types
    pub supported_model_types: Vec<String>,
}

impl Default for NwdafCapabilities {
    fn default() -> Self {
        Self {
            has_mtlf: true,
            has_anlf: true,
            max_concurrent_requests: 100,
            supported_model_types: vec!["onnx".to_string()],
        }
    }
}

/// Analytics delegation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRequest {
    /// Request identifier
    pub request_id: String,
    /// Analytics ID to delegate
    pub analytics_id: AnalyticsId,
    /// Target instance ID (if specific instance requested)
    pub target_instance_id: Option<String>,
    /// Request payload
    pub payload: String,
    /// Priority (higher = more urgent)
    pub priority: u8,
    /// Requestor instance ID
    pub requestor_id: String,
}

/// Analytics delegation response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationResponse {
    /// Request identifier
    pub request_id: String,
    /// Whether delegation was accepted
    pub accepted: bool,
    /// Assigned instance ID
    pub assigned_instance_id: Option<String>,
    /// Estimated completion time (ms)
    pub estimated_completion_ms: Option<u64>,
    /// Error message if rejected
    pub error: Option<String>,
}

/// Load balancing strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Least-loaded instance first
    LeastLoaded,
    /// Based on analytics type specialization
    AnalyticsAware,
    /// Nearest instance (latency-based)
    Nearest,
}

/// NWDAF coordination manager
///
/// Manages a pool of NWDAF instances and coordinates analytics requests
/// across them for load distribution and high availability.
#[derive(Debug)]
pub struct NwdafCoordinator {
    /// Registered NWDAF instances
    instances: HashMap<String, NwdafInstance>,
    /// Active delegation requests
    delegations: HashMap<String, DelegationRequest>,
    /// Load balancing strategy
    strategy: LoadBalancingStrategy,
    /// Next request ID counter
    next_request_id: u64,
    /// Heartbeat timeout (ms)
    heartbeat_timeout_ms: u64,
    /// Shared analytics results from peer instances
    shared_results: HashMap<(AnalyticsId, String), Vec<SharedAnalyticsResult>>,
}

impl NwdafCoordinator {
    /// Creates a new NWDAF coordinator
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            instances: HashMap::new(),
            delegations: HashMap::new(),
            strategy,
            next_request_id: 1,
            heartbeat_timeout_ms: 30_000, // 30 seconds
            shared_results: HashMap::new(),
        }
    }

    /// Registers a new NWDAF instance
    pub fn register_instance(&mut self, instance: NwdafInstance) {
        info!(
            "Registering NWDAF instance {} at {} (analytics: {:?})",
            instance.instance_id, instance.address, instance.supported_analytics
        );
        self.instances.insert(instance.instance_id.clone(), instance);
    }

    /// Unregisters an NWDAF instance
    pub fn unregister_instance(&mut self, instance_id: &str) -> Option<NwdafInstance> {
        info!("Unregistering NWDAF instance {}", instance_id);
        self.instances.remove(instance_id)
    }

    /// Updates instance heartbeat
    pub fn update_heartbeat(
        &mut self,
        instance_id: &str,
        load_level: f32,
    ) -> Result<(), NwdafError> {
        let instance = self.instances.get_mut(instance_id).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: instance_id.to_string(),
            }
        })?;

        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        instance.last_heartbeat_ms = now_ms;
        instance.load_level = load_level;
        instance.available_capacity = (1.0 - load_level).max(0.0);
        instance.is_active = true;

        debug!("Updated heartbeat for {} (load: {:.2})", instance_id, load_level);

        Ok(())
    }

    /// Checks for inactive instances (haven't sent heartbeat recently)
    pub fn check_inactive_instances(&mut self) -> Vec<String> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        let mut inactive = Vec::new();

        for (id, instance) in &mut self.instances {
            if instance.is_active
                && now_ms.saturating_sub(instance.last_heartbeat_ms) > self.heartbeat_timeout_ms
            {
                warn!("Instance {} is inactive (last heartbeat: {}ms ago)",
                    id, now_ms.saturating_sub(instance.last_heartbeat_ms));
                instance.is_active = false;
                inactive.push(id.clone());
            }
        }

        inactive
    }

    /// Delegates an analytics request to another NWDAF instance
    ///
    /// Selects the best instance based on the load balancing strategy.
    pub fn delegate_analytics(
        &mut self,
        analytics_id: AnalyticsId,
        payload: String,
        priority: u8,
        requestor_id: String,
    ) -> DelegationResponse {
        let request_id = format!("req-{}", self.next_request_id);
        self.next_request_id += 1;

        // Select target instance based on strategy
        let target = self.select_instance(analytics_id, None);

        match target {
            Some(instance_id) => {
                let instance = self.instances.get(&instance_id).unwrap();

                let request = DelegationRequest {
                    request_id: request_id.clone(),
                    analytics_id,
                    target_instance_id: Some(instance_id.clone()),
                    payload,
                    priority,
                    requestor_id,
                };

                self.delegations.insert(request_id.clone(), request);

                info!(
                    "Delegated {:?} analytics to instance {} (load: {:.2})",
                    analytics_id, instance_id, instance.load_level
                );

                DelegationResponse {
                    request_id,
                    accepted: true,
                    assigned_instance_id: Some(instance_id),
                    estimated_completion_ms: Some(1000), // Simplified estimate
                    error: None,
                }
            }
            None => {
                warn!("No available instance found for {:?} analytics", analytics_id);

                DelegationResponse {
                    request_id,
                    accepted: false,
                    assigned_instance_id: None,
                    estimated_completion_ms: None,
                    error: Some(format!(
                        "No available NWDAF instance for {analytics_id:?}"
                    )),
                }
            }
        }
    }

    /// Selects the best NWDAF instance for an analytics request
    fn select_instance(
        &self,
        analytics_id: AnalyticsId,
        preferred_instance: Option<&str>,
    ) -> Option<String> {
        // If specific instance requested, use it if available
        if let Some(instance_id) = preferred_instance {
            if let Some(instance) = self.instances.get(instance_id) {
                if instance.is_active && instance.supported_analytics.contains(&analytics_id) {
                    return Some(instance_id.to_string());
                }
            }
        }

        // Filter instances that support the analytics type and are active
        let mut candidates: Vec<_> = self
            .instances
            .values()
            .filter(|i| i.is_active && i.supported_analytics.contains(&analytics_id))
            .collect();

        if candidates.is_empty() {
            return None;
        }

        // Apply selection strategy
        let selected = match self.strategy {
            LoadBalancingStrategy::LeastLoaded => {
                candidates.sort_by(|a, b| {
                    a.load_level
                        .partial_cmp(&b.load_level)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first()
            }
            LoadBalancingStrategy::RoundRobin => {
                // Simple: pick first available (in real impl, would maintain counter)
                candidates.first()
            }
            LoadBalancingStrategy::AnalyticsAware => {
                // Prefer instances with AnLF for analytics execution
                candidates.sort_by(|a, b| {
                    let a_score = if a.capabilities.has_anlf { 1.0 } else { 0.5 }
                        * (1.0 - a.load_level);
                    let b_score = if b.capabilities.has_anlf { 1.0 } else { 0.5 }
                        * (1.0 - b.load_level);
                    b_score
                        .partial_cmp(&a_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first()
            }
            LoadBalancingStrategy::Nearest => {
                // In real impl, would use network latency measurements
                candidates.sort_by(|a, b| {
                    a.load_level
                        .partial_cmp(&b.load_level)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.first()
            }
        };

        selected.map(|i| i.instance_id.clone())
    }

    /// Returns all registered instances
    pub fn list_instances(&self) -> Vec<&NwdafInstance> {
        self.instances.values().collect()
    }

    /// Returns active instances
    pub fn active_instances(&self) -> Vec<&NwdafInstance> {
        self.instances.values().filter(|i| i.is_active).collect()
    }

    /// Returns the number of active instances
    pub fn active_count(&self) -> usize {
        self.instances.values().filter(|i| i.is_active).count()
    }

    /// Returns total instance count
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Gets an instance by ID
    pub fn get_instance(&self, instance_id: &str) -> Option<&NwdafInstance> {
        self.instances.get(instance_id)
    }

    /// Gets the current load balancing strategy
    pub fn strategy(&self) -> LoadBalancingStrategy {
        self.strategy
    }

    /// Sets the load balancing strategy
    pub fn set_strategy(&mut self, strategy: LoadBalancingStrategy) {
        info!("Changing load balancing strategy to {:?}", strategy);
        self.strategy = strategy;
    }

    /// Shares an analytics result with peer NWDAF instances (TS 23.288 6.2B.3).
    ///
    /// Posts the result to the shared result store, where other instances can
    /// query it. Results are stored indexed by analytics ID and target.
    pub fn share_result(&mut self, result: SharedAnalyticsResult) {
        let key = (result.analytics_id, result.target_key.clone());
        debug!(
            "Sharing analytics result from {} for {:?} target={}",
            result.source_instance_id, result.analytics_id, result.target_key
        );
        self.shared_results
            .entry(key)
            .or_default()
            .push(result);
    }

    /// Queries shared analytics results from peer instances.
    ///
    /// Returns all results matching the given analytics ID, optionally filtered
    /// by target key and maximum age.
    pub fn query_shared_results(
        &self,
        analytics_id: AnalyticsId,
        target_key: Option<&str>,
        max_age_ms: Option<u64>,
    ) -> Vec<&SharedAnalyticsResult> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        self.shared_results
            .iter()
            .filter(|((aid, tkey), _)| {
                *aid == analytics_id
                    && target_key.is_none_or(|tk| tkey == tk)
            })
            .flat_map(|(_, results)| results.iter())
            .filter(|r| {
                max_age_ms.is_none_or(|max| now_ms.saturating_sub(r.timestamp_ms) <= max)
            })
            .collect()
    }

    /// Returns the count of shared results
    pub fn shared_result_count(&self) -> usize {
        self.shared_results.values().map(std::vec::Vec::len).sum()
    }
}

/// Analytics result shared across NWDAF instances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedAnalyticsResult {
    /// Source NWDAF instance that produced the result
    pub source_instance_id: String,
    /// Analytics ID
    pub analytics_id: AnalyticsId,
    /// Target key (e.g. "ue-42", "cell-7", "any")
    pub target_key: String,
    /// Confidence score
    pub confidence: f32,
    /// Timestamp of the result
    pub timestamp_ms: u64,
    /// Serialized result payload (JSON)
    pub payload_json: String,
}

impl Default for NwdafCoordinator {
    fn default() -> Self {
        Self::new(LoadBalancingStrategy::LeastLoaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_instance(id: &str, analytics: Vec<AnalyticsId>, load: f32) -> NwdafInstance {
        NwdafInstance {
            instance_id: id.to_string(),
            address: format!("nwdaf-{id}.example.com:8080"),
            supported_analytics: analytics,
            load_level: load,
            available_capacity: 1.0 - load,
            is_active: true,
            last_heartbeat_ms: 0,
            capabilities: NwdafCapabilities::default(),
        }
    }

    #[test]
    fn test_coordinator_creation() {
        let coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        assert_eq!(coordinator.instance_count(), 0);
        assert_eq!(coordinator.active_count(), 0);
    }

    #[test]
    fn test_register_instance() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        let instance = make_instance("nwdaf-1", vec![AnalyticsId::UeMobility], 0.3);

        coordinator.register_instance(instance);
        assert_eq!(coordinator.instance_count(), 1);
        assert_eq!(coordinator.active_count(), 1);
    }

    #[test]
    fn test_unregister_instance() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        let instance = make_instance("nwdaf-1", vec![AnalyticsId::UeMobility], 0.3);

        coordinator.register_instance(instance);
        assert_eq!(coordinator.instance_count(), 1);

        let removed = coordinator.unregister_instance("nwdaf-1");
        assert!(removed.is_some());
        assert_eq!(coordinator.instance_count(), 0);
    }

    #[test]
    fn test_update_heartbeat() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        let mut instance = make_instance("nwdaf-1", vec![AnalyticsId::UeMobility], 0.3);
        instance.last_heartbeat_ms = 0;

        coordinator.register_instance(instance);

        coordinator.update_heartbeat("nwdaf-1", 0.5).unwrap();

        let updated = coordinator.get_instance("nwdaf-1").unwrap();
        assert_eq!(updated.load_level, 0.5);
        assert!(updated.last_heartbeat_ms > 0);
    }

    #[test]
    fn test_delegation_least_loaded() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);

        coordinator.register_instance(make_instance(
            "nwdaf-1",
            vec![AnalyticsId::UeMobility],
            0.7,
        ));
        coordinator.register_instance(make_instance(
            "nwdaf-2",
            vec![AnalyticsId::UeMobility],
            0.3,
        ));

        let response = coordinator.delegate_analytics(
            AnalyticsId::UeMobility,
            "payload".to_string(),
            5,
            "requestor".to_string(),
        );

        assert!(response.accepted);
        assert_eq!(response.assigned_instance_id.as_deref(), Some("nwdaf-2"));
    }

    #[test]
    fn test_delegation_no_capable_instance() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);

        coordinator.register_instance(make_instance(
            "nwdaf-1",
            vec![AnalyticsId::NfLoad],
            0.3,
        ));

        let response = coordinator.delegate_analytics(
            AnalyticsId::UeMobility,
            "payload".to_string(),
            5,
            "requestor".to_string(),
        );

        assert!(!response.accepted);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_check_inactive_instances() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        coordinator.heartbeat_timeout_ms = 100; // 100ms for testing

        let mut instance = make_instance("nwdaf-1", vec![AnalyticsId::UeMobility], 0.3);
        instance.last_heartbeat_ms = 0; // Very old heartbeat
        coordinator.register_instance(instance);

        std::thread::sleep(std::time::Duration::from_millis(150));

        let inactive = coordinator.check_inactive_instances();
        assert_eq!(inactive.len(), 1);
        assert_eq!(inactive[0], "nwdaf-1");

        let instance = coordinator.get_instance("nwdaf-1").unwrap();
        assert!(!instance.is_active);
    }

    #[test]
    fn test_active_instances() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);

        let instance1 = make_instance("nwdaf-1", vec![AnalyticsId::UeMobility], 0.3);
        let mut instance2 = make_instance("nwdaf-2", vec![AnalyticsId::NfLoad], 0.5);
        instance2.is_active = false;

        coordinator.register_instance(instance1);
        coordinator.register_instance(instance2);

        assert_eq!(coordinator.instance_count(), 2);
        assert_eq!(coordinator.active_count(), 1);

        let active = coordinator.active_instances();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].instance_id, "nwdaf-1");
    }

    #[test]
    fn test_share_and_query_results() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);

        coordinator.share_result(SharedAnalyticsResult {
            source_instance_id: "nwdaf-1".to_string(),
            analytics_id: AnalyticsId::UeMobility,
            target_key: "ue-42".to_string(),
            confidence: 0.9,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            payload_json: "{}".to_string(),
        });

        coordinator.share_result(SharedAnalyticsResult {
            source_instance_id: "nwdaf-2".to_string(),
            analytics_id: AnalyticsId::NfLoad,
            target_key: "cell-7".to_string(),
            confidence: 0.85,
            timestamp_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            payload_json: "{}".to_string(),
        });

        assert_eq!(coordinator.shared_result_count(), 2);

        let mobility_results = coordinator.query_shared_results(
            AnalyticsId::UeMobility, Some("ue-42"), None,
        );
        assert_eq!(mobility_results.len(), 1);
        assert_eq!(mobility_results[0].source_instance_id, "nwdaf-1");

        let load_results = coordinator.query_shared_results(
            AnalyticsId::NfLoad, None, None,
        );
        assert_eq!(load_results.len(), 1);

        // Query for nonexistent
        let empty = coordinator.query_shared_results(
            AnalyticsId::AbnormalBehavior, None, None,
        );
        assert!(empty.is_empty());
    }

    #[test]
    fn test_strategy_change() {
        let mut coordinator = NwdafCoordinator::new(LoadBalancingStrategy::LeastLoaded);
        assert_eq!(coordinator.strategy(), LoadBalancingStrategy::LeastLoaded);

        coordinator.set_strategy(LoadBalancingStrategy::RoundRobin);
        assert_eq!(coordinator.strategy(), LoadBalancingStrategy::RoundRobin);
    }
}
