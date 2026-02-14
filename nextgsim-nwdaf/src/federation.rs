//! Cross-Operator Data Federation (TS 23.288 / TS 29.520)
//!
//! Implements privacy-preserving analytics sharing across PLMN boundaries.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────────────────┐
//! │                    Data Federation Manager                            │
//! │                                                                       │
//! │  ┌─────────────────────┐      ┌─────────────────────────────────┐   │
//! │  │ Federation Registry │      │ Privacy Controller               │   │
//! │  │  • Peer operators    │      │  • K-anonymity enforcement       │   │
//! │  │  • Trust levels      │      │  • Differential privacy noise    │   │
//! │  │  • Roaming partners  │      │  • Data minimization             │   │
//! │  └─────────────────────┘      │  • Consent tracking              │   │
//! │                                └─────────────────────────────────┘   │
//! │  ┌─────────────────────┐      ┌─────────────────────────────────┐   │
//! │  │ Cross-PLMN Exchange │      │ Aggregation Engine               │   │
//! │  │  • Secure transport  │      │  • Federated averaging           │   │
//! │  │  • Request/response  │      │  • Cross-operator merge          │   │
//! │  │  • Rate limiting     │      │  • Conflict resolution           │   │
//! │  └─────────────────────┘      └─────────────────────────────────┘   │
//! │                                                                       │
//! │  Nnwdaf_DataManagement cross-operator extensions                      │
//! └───────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info, warn};

use crate::analytics_id::AnalyticsId;
use crate::error::NwdafError;

// ---------------------------------------------------------------------------
// PLMN and operator identity
// ---------------------------------------------------------------------------

/// PLMN identity (MCC + MNC)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PlmnId {
    /// Mobile Country Code (3 digits)
    pub mcc: String,
    /// Mobile Network Code (2-3 digits)
    pub mnc: String,
}

impl PlmnId {
    /// Creates a new PLMN ID.
    pub fn new(mcc: &str, mnc: &str) -> Self {
        Self {
            mcc: mcc.to_string(),
            mnc: mnc.to_string(),
        }
    }

    /// Returns canonical string form "MCC-MNC".
    pub fn canonical(&self) -> String {
        format!("{}-{}", self.mcc, self.mnc)
    }
}

impl std::fmt::Display for PlmnId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}-{}", self.mcc, self.mnc)
    }
}

/// Trust level assigned to a federation peer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TrustLevel {
    /// Untrusted - no data sharing
    Untrusted = 0,
    /// Basic - only aggregate statistics, strong anonymization
    Basic = 1,
    /// Roaming - data for roaming subscribers, k-anonymity enforced
    Roaming = 2,
    /// Full - trusted partner, individual analytics with consent
    Full = 3,
}

/// Federation peer descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationPeer {
    /// Peer PLMN identity
    pub plmn_id: PlmnId,
    /// Display name of the operator
    pub operator_name: String,
    /// Endpoint URL for cross-PLMN analytics exchange
    pub endpoint: String,
    /// Trust level
    pub trust_level: TrustLevel,
    /// Analytics IDs this peer is willing to share
    pub shared_analytics: Vec<AnalyticsId>,
    /// Whether peer is currently reachable
    pub is_active: bool,
    /// Last successful exchange timestamp (ms since epoch)
    pub last_exchange_ms: u64,
    /// Roaming agreement active
    pub roaming_agreement: bool,
    /// Maximum requests per minute from/to this peer
    pub rate_limit_rpm: u32,
}

// ---------------------------------------------------------------------------
// Privacy controls
// ---------------------------------------------------------------------------

/// Privacy policy for cross-operator data exchange.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyPolicy {
    /// Minimum group size for k-anonymity (default: 5)
    pub k_anonymity_threshold: usize,
    /// Epsilon parameter for differential privacy noise (default: 1.0)
    pub dp_epsilon: f64,
    /// Whether to strip subscriber identities from shared data
    pub strip_subscriber_ids: bool,
    /// Whether to generalize location data (round to cell level)
    pub generalize_location: bool,
    /// Maximum data retention period for shared data (seconds)
    pub retention_seconds: u64,
    /// Required consent types before sharing
    pub required_consent: Vec<ConsentType>,
}

impl Default for PrivacyPolicy {
    fn default() -> Self {
        Self {
            k_anonymity_threshold: 5,
            dp_epsilon: 1.0,
            strip_subscriber_ids: true,
            generalize_location: true,
            retention_seconds: 86_400, // 24 hours
            required_consent: vec![ConsentType::Analytics],
        }
    }
}

/// Types of subscriber consent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsentType {
    /// Consent for analytics processing
    Analytics,
    /// Consent for cross-operator data sharing
    CrossOperatorSharing,
    /// Consent for location data sharing
    LocationSharing,
}

// ---------------------------------------------------------------------------
// Cross-operator exchange messages
// ---------------------------------------------------------------------------

/// Request for analytics data from a peer operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationRequest {
    /// Unique request identifier
    pub request_id: String,
    /// Requesting operator PLMN
    pub source_plmn: PlmnId,
    /// Target operator PLMN
    pub target_plmn: PlmnId,
    /// Requested analytics type
    pub analytics_id: AnalyticsId,
    /// Optional geographic area filter
    pub area_filter: Option<AreaFilter>,
    /// Optional time range (start_ms, end_ms)
    pub time_range: Option<(u64, u64)>,
    /// Maximum number of results requested
    pub max_results: usize,
    /// Whether aggregated data is acceptable
    pub accept_aggregated: bool,
}

/// Response to a federation analytics request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederationResponse {
    /// Matches the request_id
    pub request_id: String,
    /// Responding operator PLMN
    pub source_plmn: PlmnId,
    /// Whether the request was accepted
    pub accepted: bool,
    /// Analytics results (privacy-filtered)
    pub results: Vec<FederatedAnalyticsResult>,
    /// Error message if rejected
    pub error: Option<String>,
    /// Applied privacy level
    pub privacy_level: Option<TrustLevel>,
}

/// A single analytics result shared across operators.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedAnalyticsResult {
    /// Analytics ID
    pub analytics_id: AnalyticsId,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp of the result
    pub timestamp_ms: u64,
    /// Area of validity (if location-based)
    pub area: Option<AreaFilter>,
    /// Number of data points aggregated into this result
    pub sample_count: usize,
    /// Result payload (JSON, privacy-filtered)
    pub payload_json: String,
}

/// Geographic area filter for federation requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaFilter {
    /// Tracking Area Codes
    pub tac_list: Vec<u32>,
    /// Cell ID list (optional)
    pub cell_ids: Option<Vec<u32>>,
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

/// Method for aggregating cross-operator results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FederatedAggregationMethod {
    /// Weighted average by sample count
    WeightedAverage,
    /// Take the most recent result
    MostRecent,
    /// Union of all results (deduped)
    Union,
    /// Majority vote (for categorical results)
    MajorityVote,
}

/// Aggregated cross-operator analytics result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedFederationResult {
    /// Analytics ID
    pub analytics_id: AnalyticsId,
    /// Contributing operators
    pub contributing_plmns: Vec<PlmnId>,
    /// Total sample count across all operators
    pub total_samples: usize,
    /// Aggregated confidence score
    pub confidence: f32,
    /// Aggregation method used
    pub method: FederatedAggregationMethod,
    /// Aggregated payload (JSON)
    pub payload_json: String,
    /// Timestamp
    pub timestamp_ms: u64,
}

// ---------------------------------------------------------------------------
// Federation Manager
// ---------------------------------------------------------------------------

/// Cross-operator data federation manager.
///
/// Handles peer registration, privacy enforcement, request routing,
/// and result aggregation for cross-PLMN analytics exchange.
#[derive(Debug)]
pub struct DataFederationManager {
    /// Local operator identity
    local_plmn: PlmnId,
    /// Registered federation peers
    peers: HashMap<PlmnId, FederationPeer>,
    /// Privacy policy applied to outgoing data
    privacy_policy: PrivacyPolicy,
    /// Pending outgoing requests
    pending_requests: HashMap<String, FederationRequest>,
    /// Received results cache indexed by (analytics_id, peer_plmn)
    result_cache: HashMap<(AnalyticsId, PlmnId), Vec<FederatedAnalyticsResult>>,
    /// Rate limit counters: PLMN -> (count, window_start_ms)
    rate_counters: HashMap<PlmnId, (u32, u64)>,
    /// Next request ID counter
    next_request_id: u64,
    /// Subscriber consent records: subscriber_id -> consents
    consent_records: HashMap<String, Vec<ConsentType>>,
}

impl DataFederationManager {
    /// Creates a new data federation manager for the given local operator.
    pub fn new(local_plmn: PlmnId) -> Self {
        info!("Creating DataFederationManager for PLMN {}", local_plmn);
        Self {
            local_plmn,
            peers: HashMap::new(),
            privacy_policy: PrivacyPolicy::default(),
            pending_requests: HashMap::new(),
            result_cache: HashMap::new(),
            rate_counters: HashMap::new(),
            next_request_id: 1,
            consent_records: HashMap::new(),
        }
    }

    /// Creates a new manager with a custom privacy policy.
    pub fn with_privacy_policy(local_plmn: PlmnId, policy: PrivacyPolicy) -> Self {
        let mut mgr = Self::new(local_plmn);
        mgr.privacy_policy = policy;
        mgr
    }

    /// Returns the local PLMN identity.
    pub fn local_plmn(&self) -> &PlmnId {
        &self.local_plmn
    }

    /// Returns the current privacy policy.
    pub fn privacy_policy(&self) -> &PrivacyPolicy {
        &self.privacy_policy
    }

    /// Updates the privacy policy.
    pub fn set_privacy_policy(&mut self, policy: PrivacyPolicy) {
        info!("Updating privacy policy (k={}, epsilon={:.2})",
              policy.k_anonymity_threshold, policy.dp_epsilon);
        self.privacy_policy = policy;
    }

    // -----------------------------------------------------------------------
    // Peer management
    // -----------------------------------------------------------------------

    /// Registers a federation peer operator.
    pub fn register_peer(&mut self, peer: FederationPeer) -> Result<(), NwdafError> {
        if peer.plmn_id == self.local_plmn {
            return Err(crate::error::AnalyticsError::ComputationFailed {
                reason: "Cannot register self as federation peer".to_string(),
            }
            .into());
        }

        info!(
            "Registering federation peer {} ({}) trust={:?} analytics={:?}",
            peer.plmn_id, peer.operator_name, peer.trust_level, peer.shared_analytics
        );
        self.peers.insert(peer.plmn_id.clone(), peer);
        Ok(())
    }

    /// Removes a federation peer.
    pub fn remove_peer(&mut self, plmn_id: &PlmnId) -> Option<FederationPeer> {
        info!("Removing federation peer {}", plmn_id);
        self.peers.remove(plmn_id)
    }

    /// Returns registered peers.
    pub fn list_peers(&self) -> Vec<&FederationPeer> {
        self.peers.values().collect()
    }

    /// Returns the number of registered peers.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    /// Returns active peers with the given minimum trust level.
    pub fn active_peers_with_trust(&self, min_trust: TrustLevel) -> Vec<&FederationPeer> {
        self.peers
            .values()
            .filter(|p| p.is_active && p.trust_level >= min_trust)
            .collect()
    }

    /// Returns peers that share a given analytics type.
    pub fn peers_for_analytics(&self, analytics_id: AnalyticsId) -> Vec<&FederationPeer> {
        self.peers
            .values()
            .filter(|p| p.is_active && p.shared_analytics.contains(&analytics_id))
            .collect()
    }

    // -----------------------------------------------------------------------
    // Consent management
    // -----------------------------------------------------------------------

    /// Records subscriber consent for data sharing.
    pub fn record_consent(&mut self, subscriber_id: &str, consent: ConsentType) {
        debug!("Recording consent {:?} for subscriber {}", consent, subscriber_id);
        self.consent_records
            .entry(subscriber_id.to_string())
            .or_default()
            .push(consent);
    }

    /// Revokes subscriber consent.
    pub fn revoke_consent(&mut self, subscriber_id: &str, consent: ConsentType) {
        if let Some(consents) = self.consent_records.get_mut(subscriber_id) {
            consents.retain(|c| *c != consent);
        }
    }

    /// Checks whether a subscriber has all required consents for sharing.
    pub fn has_required_consent(&self, subscriber_id: &str) -> bool {
        let consents = self.consent_records.get(subscriber_id);
        match consents {
            None => self.privacy_policy.required_consent.is_empty(),
            Some(granted) => self
                .privacy_policy
                .required_consent
                .iter()
                .all(|required| granted.contains(required)),
        }
    }

    // -----------------------------------------------------------------------
    // Request creation and handling
    // -----------------------------------------------------------------------

    /// Creates a federation request to a peer operator.
    ///
    /// Returns the request object that should be sent to the peer.
    /// The request is also stored in pending_requests.
    pub fn create_request(
        &mut self,
        target_plmn: &PlmnId,
        analytics_id: AnalyticsId,
        area_filter: Option<AreaFilter>,
        max_results: usize,
    ) -> Result<FederationRequest, NwdafError> {
        // Verify peer exists and is active
        let peer = self.peers.get(target_plmn).ok_or_else(|| {
            crate::error::AnalyticsError::TargetNotFound {
                target: format!("Federation peer {target_plmn}"),
            }
        })?;

        if !peer.is_active {
            return Err(crate::error::AnalyticsError::ComputationFailed {
                reason: format!("Peer {target_plmn} is not active"),
            }
            .into());
        }

        if !peer.shared_analytics.contains(&analytics_id) {
            return Err(crate::error::AnalyticsError::UnsupportedAnalyticsId { id: analytics_id }
                .into());
        }

        // Check rate limit
        if !self.check_rate_limit(target_plmn) {
            return Err(crate::error::AnalyticsError::ComputationFailed {
                reason: format!("Rate limit exceeded for peer {target_plmn}"),
            }
            .into());
        }

        let request_id = format!("fed-{}-{}", self.local_plmn.canonical(), self.next_request_id);
        self.next_request_id += 1;

        let request = FederationRequest {
            request_id: request_id.clone(),
            source_plmn: self.local_plmn.clone(),
            target_plmn: target_plmn.clone(),
            analytics_id,
            area_filter,
            time_range: None,
            max_results,
            accept_aggregated: true,
        };

        self.pending_requests.insert(request_id, request.clone());
        self.increment_rate_counter(target_plmn);

        info!(
            "Created federation request {} to {} for {:?}",
            request.request_id, target_plmn, analytics_id
        );

        Ok(request)
    }

    /// Handles an incoming federation request from a peer.
    ///
    /// Validates the request against privacy policy and trust level,
    /// then prepares a response with privacy-filtered results.
    pub fn handle_request(
        &self,
        request: &FederationRequest,
        local_results: &[FederatedAnalyticsResult],
    ) -> FederationResponse {
        let peer = match self.peers.get(&request.source_plmn) {
            Some(p) => p,
            None => {
                warn!(
                    "Rejecting request {} from unknown peer {}",
                    request.request_id, request.source_plmn
                );
                return FederationResponse {
                    request_id: request.request_id.clone(),
                    source_plmn: self.local_plmn.clone(),
                    accepted: false,
                    results: vec![],
                    error: Some("Unknown peer operator".to_string()),
                    privacy_level: None,
                };
            }
        };

        if peer.trust_level == TrustLevel::Untrusted {
            return FederationResponse {
                request_id: request.request_id.clone(),
                source_plmn: self.local_plmn.clone(),
                accepted: false,
                results: vec![],
                error: Some("Insufficient trust level".to_string()),
                privacy_level: None,
            };
        }

        // Apply privacy filtering based on trust level
        let filtered = self.apply_privacy_filter(local_results, peer.trust_level);

        // Apply k-anonymity check
        let k_filtered: Vec<FederatedAnalyticsResult> = filtered
            .into_iter()
            .filter(|r| r.sample_count >= self.privacy_policy.k_anonymity_threshold)
            .collect();

        // Limit results
        let limited: Vec<FederatedAnalyticsResult> =
            k_filtered.into_iter().take(request.max_results).collect();

        info!(
            "Responding to request {} from {} with {} results (trust={:?})",
            request.request_id,
            request.source_plmn,
            limited.len(),
            peer.trust_level,
        );

        FederationResponse {
            request_id: request.request_id.clone(),
            source_plmn: self.local_plmn.clone(),
            accepted: true,
            results: limited,
            error: None,
            privacy_level: Some(peer.trust_level),
        }
    }

    /// Processes a federation response from a peer.
    ///
    /// Stores results in the cache for later aggregation.
    pub fn process_response(
        &mut self,
        response: FederationResponse,
    ) -> Result<usize, NwdafError> {
        // Remove from pending
        let request = self.pending_requests.remove(&response.request_id);

        if request.is_none() {
            warn!(
                "Received response for unknown request {}",
                response.request_id
            );
        }

        if !response.accepted {
            return Err(crate::error::AnalyticsError::ComputationFailed {
                reason: format!(
                    "Peer {} rejected request: {}",
                    response.source_plmn,
                    response.error.unwrap_or_default()
                ),
            }
            .into());
        }

        let count = response.results.len();

        // Store in cache
        for result in response.results {
            let key = (result.analytics_id, response.source_plmn.clone());
            self.result_cache.entry(key).or_default().push(result);
        }

        // Update peer last_exchange timestamp
        let now_ms = current_time_ms();
        if let Some(peer) = self.peers.get_mut(&response.source_plmn) {
            peer.last_exchange_ms = now_ms;
        }

        debug!(
            "Cached {} results from peer {}",
            count, response.source_plmn
        );

        Ok(count)
    }

    // -----------------------------------------------------------------------
    // Aggregation
    // -----------------------------------------------------------------------

    /// Aggregates cached results from multiple operators for a given analytics type.
    pub fn aggregate_results(
        &self,
        analytics_id: AnalyticsId,
        method: FederatedAggregationMethod,
    ) -> Option<AggregatedFederationResult> {
        let mut all_results: Vec<(&PlmnId, &FederatedAnalyticsResult)> = Vec::new();

        for ((aid, plmn), results) in &self.result_cache {
            if *aid == analytics_id {
                for r in results {
                    all_results.push((plmn, r));
                }
            }
        }

        if all_results.is_empty() {
            return None;
        }

        let contributing_plmns: Vec<PlmnId> = all_results
            .iter()
            .map(|(p, _)| (*p).clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let total_samples: usize = all_results.iter().map(|(_, r)| r.sample_count).sum();

        let (confidence, payload_json) = match method {
            FederatedAggregationMethod::WeightedAverage => {
                let total_weight: f64 = all_results.iter().map(|(_, r)| r.sample_count as f64).sum();
                let weighted_confidence: f64 = all_results
                    .iter()
                    .map(|(_, r)| r.confidence as f64 * r.sample_count as f64)
                    .sum::<f64>()
                    / total_weight.max(1.0);
                (
                    weighted_confidence as f32,
                    format!(
                        r#"{{"method":"weighted_average","sources":{},"total_samples":{}}}"#,
                        contributing_plmns.len(),
                        total_samples
                    ),
                )
            }
            FederatedAggregationMethod::MostRecent => {
                let latest = all_results
                    .iter()
                    .max_by_key(|(_, r)| r.timestamp_ms)
                    .unwrap();
                (latest.1.confidence, latest.1.payload_json.clone())
            }
            FederatedAggregationMethod::Union => {
                let avg_confidence =
                    all_results.iter().map(|(_, r)| r.confidence).sum::<f32>()
                        / all_results.len() as f32;
                (
                    avg_confidence,
                    format!(
                        r#"{{"method":"union","result_count":{},"total_samples":{}}}"#,
                        all_results.len(),
                        total_samples
                    ),
                )
            }
            FederatedAggregationMethod::MajorityVote => {
                let avg_confidence =
                    all_results.iter().map(|(_, r)| r.confidence).sum::<f32>()
                        / all_results.len() as f32;
                (
                    avg_confidence,
                    format!(
                        r#"{{"method":"majority_vote","voters":{}}}"#,
                        contributing_plmns.len()
                    ),
                )
            }
        };

        let timestamp_ms = all_results
            .iter()
            .map(|(_, r)| r.timestamp_ms)
            .max()
            .unwrap_or(0);

        Some(AggregatedFederationResult {
            analytics_id,
            contributing_plmns,
            total_samples,
            confidence,
            method,
            payload_json,
            timestamp_ms,
        })
    }

    /// Returns the number of cached results for a given analytics type.
    pub fn cached_result_count(&self, analytics_id: AnalyticsId) -> usize {
        self.result_cache
            .iter()
            .filter(|((aid, _), _)| *aid == analytics_id)
            .map(|(_, results)| results.len())
            .sum()
    }

    /// Clears cached results older than the retention period.
    pub fn evict_expired_cache(&mut self) -> usize {
        let now_ms = current_time_ms();
        let retention_ms = self.privacy_policy.retention_seconds * 1000;
        let mut evicted = 0;

        for results in self.result_cache.values_mut() {
            let before = results.len();
            results.retain(|r| now_ms.saturating_sub(r.timestamp_ms) <= retention_ms);
            evicted += before - results.len();
        }

        // Remove empty keys
        self.result_cache.retain(|_, v| !v.is_empty());

        if evicted > 0 {
            debug!("Evicted {} expired federation results", evicted);
        }

        evicted
    }

    /// Returns the total number of cached results across all peers.
    pub fn total_cached_results(&self) -> usize {
        self.result_cache.values().map(Vec::len).sum()
    }

    // -----------------------------------------------------------------------
    // Privacy filtering (internal)
    // -----------------------------------------------------------------------

    /// Applies privacy filtering to results based on trust level.
    fn apply_privacy_filter(
        &self,
        results: &[FederatedAnalyticsResult],
        trust_level: TrustLevel,
    ) -> Vec<FederatedAnalyticsResult> {
        results
            .iter()
            .map(|r| {
                let mut filtered = r.clone();

                // Apply differential privacy noise to confidence scores
                if trust_level <= TrustLevel::Basic {
                    let noise = laplace_noise(self.privacy_policy.dp_epsilon);
                    filtered.confidence = (filtered.confidence + noise as f32).clamp(0.0, 1.0);
                }

                // Generalize location data for lower trust levels
                if self.privacy_policy.generalize_location && trust_level < TrustLevel::Full {
                    filtered.area = filtered.area.map(|mut a| {
                        a.cell_ids = None; // Strip cell-level detail
                        a
                    });
                }

                filtered
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Rate limiting (internal)
    // -----------------------------------------------------------------------

    /// Checks whether a request to the given peer is within rate limits.
    fn check_rate_limit(&self, plmn: &PlmnId) -> bool {
        let peer = match self.peers.get(plmn) {
            Some(p) => p,
            None => return false,
        };

        let now_ms = current_time_ms();
        match self.rate_counters.get(plmn) {
            Some((count, window_start)) => {
                if now_ms.saturating_sub(*window_start) >= 60_000 {
                    // Window expired, reset
                    true
                } else {
                    *count < peer.rate_limit_rpm
                }
            }
            None => true,
        }
    }

    /// Increments the rate limit counter for a peer.
    fn increment_rate_counter(&mut self, plmn: &PlmnId) {
        let now_ms = current_time_ms();
        let entry = self.rate_counters.entry(plmn.clone()).or_insert((0, now_ms));
        if now_ms.saturating_sub(entry.1) >= 60_000 {
            *entry = (1, now_ms);
        } else {
            entry.0 += 1;
        }
    }
}

/// Simple deterministic Laplace-like noise for differential privacy.
///
/// In production this would use a cryptographic PRNG; here we use a
/// simplified version for simulation purposes.
fn laplace_noise(epsilon: f64) -> f64 {
    // Deterministic small perturbation scaled by 1/epsilon
    // Real implementation would sample from Laplace(0, 1/epsilon)
    let scale = 1.0 / epsilon;
    // Use a simple hash of the current time for reproducible noise
    let t = current_time_ms();
    let pseudo_uniform = ((t % 1000) as f64 / 1000.0) - 0.5; // [-0.5, 0.5)
    -scale * pseudo_uniform.signum() * (1.0 - 2.0 * pseudo_uniform.abs()).ln()
}

/// Returns the current time in milliseconds since epoch.
fn current_time_ms() -> u64 {
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

    fn make_peer(mcc: &str, mnc: &str, name: &str, trust: TrustLevel) -> FederationPeer {
        FederationPeer {
            plmn_id: PlmnId::new(mcc, mnc),
            operator_name: name.to_string(),
            endpoint: format!("https://nwdaf.{name}.example.com"),
            trust_level: trust,
            shared_analytics: vec![AnalyticsId::UeMobility, AnalyticsId::NfLoad],
            is_active: true,
            last_exchange_ms: 0,
            roaming_agreement: true,
            rate_limit_rpm: 60,
        }
    }

    fn make_result(analytics_id: AnalyticsId, samples: usize, confidence: f32) -> FederatedAnalyticsResult {
        FederatedAnalyticsResult {
            analytics_id,
            confidence,
            timestamp_ms: current_time_ms(),
            area: None,
            sample_count: samples,
            payload_json: r#"{"test": true}"#.to_string(),
        }
    }

    #[test]
    fn test_plmn_id() {
        let plmn = PlmnId::new("310", "410");
        assert_eq!(plmn.canonical(), "310-410");
        assert_eq!(plmn.to_string(), "310-410");
    }

    #[test]
    fn test_federation_manager_creation() {
        let mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        assert_eq!(mgr.local_plmn().canonical(), "001-01");
        assert_eq!(mgr.peer_count(), 0);
        assert_eq!(mgr.privacy_policy().k_anonymity_threshold, 5);
    }

    #[test]
    fn test_register_peer() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let peer = make_peer("310", "410", "AT&T", TrustLevel::Roaming);
        mgr.register_peer(peer).unwrap();
        assert_eq!(mgr.peer_count(), 1);
    }

    #[test]
    fn test_cannot_register_self() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let peer = make_peer("001", "01", "Self", TrustLevel::Full);
        assert!(mgr.register_peer(peer).is_err());
    }

    #[test]
    fn test_remove_peer() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let plmn = PlmnId::new("310", "410");
        mgr.register_peer(make_peer("310", "410", "AT&T", TrustLevel::Roaming))
            .unwrap();
        assert_eq!(mgr.peer_count(), 1);

        let removed = mgr.remove_peer(&plmn);
        assert!(removed.is_some());
        assert_eq!(mgr.peer_count(), 0);
    }

    #[test]
    fn test_active_peers_with_trust() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Basic))
            .unwrap();
        mgr.register_peer(make_peer("310", "260", "OpB", TrustLevel::Roaming))
            .unwrap();
        mgr.register_peer(make_peer("262", "01", "OpC", TrustLevel::Full))
            .unwrap();

        let basic_plus = mgr.active_peers_with_trust(TrustLevel::Basic);
        assert_eq!(basic_plus.len(), 3);

        let roaming_plus = mgr.active_peers_with_trust(TrustLevel::Roaming);
        assert_eq!(roaming_plus.len(), 2);

        let full_only = mgr.active_peers_with_trust(TrustLevel::Full);
        assert_eq!(full_only.len(), 1);
    }

    #[test]
    fn test_peers_for_analytics() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();

        let mobility_peers = mgr.peers_for_analytics(AnalyticsId::UeMobility);
        assert_eq!(mobility_peers.len(), 1);

        let behavior_peers = mgr.peers_for_analytics(AnalyticsId::AbnormalBehavior);
        assert_eq!(behavior_peers.len(), 0);
    }

    #[test]
    fn test_consent_management() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));

        // No consent required by default except Analytics
        assert!(!mgr.has_required_consent("sub-1"));

        mgr.record_consent("sub-1", ConsentType::Analytics);
        assert!(mgr.has_required_consent("sub-1"));

        mgr.revoke_consent("sub-1", ConsentType::Analytics);
        assert!(!mgr.has_required_consent("sub-1"));
    }

    #[test]
    fn test_create_request() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let target = PlmnId::new("310", "410");
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();

        let req = mgr
            .create_request(&target, AnalyticsId::UeMobility, None, 10)
            .unwrap();
        assert_eq!(req.source_plmn, PlmnId::new("001", "01"));
        assert_eq!(req.target_plmn, target);
        assert_eq!(req.analytics_id, AnalyticsId::UeMobility);
        assert_eq!(req.max_results, 10);
    }

    #[test]
    fn test_create_request_unknown_peer() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let result = mgr.create_request(
            &PlmnId::new("999", "99"),
            AnalyticsId::UeMobility,
            None,
            10,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_create_request_unsupported_analytics() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();

        let result = mgr.create_request(
            &PlmnId::new("310", "410"),
            AnalyticsId::AbnormalBehavior,
            None,
            10,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_handle_request_from_known_peer() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();

        let request = FederationRequest {
            request_id: "req-1".to_string(),
            source_plmn: PlmnId::new("310", "410"),
            target_plmn: PlmnId::new("001", "01"),
            analytics_id: AnalyticsId::UeMobility,
            area_filter: None,
            time_range: None,
            max_results: 10,
            accept_aggregated: true,
        };

        let results = vec![
            make_result(AnalyticsId::UeMobility, 10, 0.9),
            make_result(AnalyticsId::UeMobility, 3, 0.7), // Below k-anonymity threshold
        ];

        let response = mgr.handle_request(&request, &results);
        assert!(response.accepted);
        // Second result has sample_count=3 < k_anonymity_threshold=5, should be filtered
        assert_eq!(response.results.len(), 1);
    }

    #[test]
    fn test_handle_request_from_unknown_peer() {
        let mgr = DataFederationManager::new(PlmnId::new("001", "01"));

        let request = FederationRequest {
            request_id: "req-1".to_string(),
            source_plmn: PlmnId::new("999", "99"),
            target_plmn: PlmnId::new("001", "01"),
            analytics_id: AnalyticsId::UeMobility,
            area_filter: None,
            time_range: None,
            max_results: 10,
            accept_aggregated: true,
        };

        let response = mgr.handle_request(&request, &[]);
        assert!(!response.accepted);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_handle_request_untrusted_peer() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let mut peer = make_peer("310", "410", "OpA", TrustLevel::Untrusted);
        peer.trust_level = TrustLevel::Untrusted;
        mgr.register_peer(peer).unwrap();

        let request = FederationRequest {
            request_id: "req-1".to_string(),
            source_plmn: PlmnId::new("310", "410"),
            target_plmn: PlmnId::new("001", "01"),
            analytics_id: AnalyticsId::UeMobility,
            area_filter: None,
            time_range: None,
            max_results: 10,
            accept_aggregated: true,
        };

        let response = mgr.handle_request(&request, &[]);
        assert!(!response.accepted);
    }

    #[test]
    fn test_process_response_and_cache() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();

        // Create a pending request first
        let req = mgr
            .create_request(
                &PlmnId::new("310", "410"),
                AnalyticsId::UeMobility,
                None,
                10,
            )
            .unwrap();

        let response = FederationResponse {
            request_id: req.request_id,
            source_plmn: PlmnId::new("310", "410"),
            accepted: true,
            results: vec![make_result(AnalyticsId::UeMobility, 20, 0.85)],
            error: None,
            privacy_level: Some(TrustLevel::Roaming),
        };

        let count = mgr.process_response(response).unwrap();
        assert_eq!(count, 1);
        assert_eq!(mgr.cached_result_count(AnalyticsId::UeMobility), 1);
        assert_eq!(mgr.total_cached_results(), 1);
    }

    #[test]
    fn test_process_rejected_response() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));

        let response = FederationResponse {
            request_id: "req-unknown".to_string(),
            source_plmn: PlmnId::new("310", "410"),
            accepted: false,
            results: vec![],
            error: Some("Forbidden".to_string()),
            privacy_level: None,
        };

        let result = mgr.process_response(response);
        assert!(result.is_err());
    }

    #[test]
    fn test_aggregate_weighted_average() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        mgr.register_peer(make_peer("310", "410", "OpA", TrustLevel::Roaming))
            .unwrap();
        mgr.register_peer(make_peer("310", "260", "OpB", TrustLevel::Roaming))
            .unwrap();

        // Insert results into cache directly for testing
        let key_a = (AnalyticsId::UeMobility, PlmnId::new("310", "410"));
        let key_b = (AnalyticsId::UeMobility, PlmnId::new("310", "260"));
        mgr.result_cache
            .entry(key_a)
            .or_default()
            .push(make_result(AnalyticsId::UeMobility, 100, 0.9));
        mgr.result_cache
            .entry(key_b)
            .or_default()
            .push(make_result(AnalyticsId::UeMobility, 50, 0.6));

        let aggregated = mgr
            .aggregate_results(AnalyticsId::UeMobility, FederatedAggregationMethod::WeightedAverage)
            .unwrap();

        assert_eq!(aggregated.analytics_id, AnalyticsId::UeMobility);
        assert_eq!(aggregated.contributing_plmns.len(), 2);
        assert_eq!(aggregated.total_samples, 150);
        // Weighted average: (0.9*100 + 0.6*50) / 150 = 120/150 = 0.8
        assert!((aggregated.confidence - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_aggregate_most_recent() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));

        let key_a = (AnalyticsId::NfLoad, PlmnId::new("310", "410"));
        let mut old_result = make_result(AnalyticsId::NfLoad, 50, 0.7);
        old_result.timestamp_ms = 1000;
        let new_result = make_result(AnalyticsId::NfLoad, 30, 0.95);

        mgr.result_cache.entry(key_a).or_default().push(old_result);
        mgr.result_cache
            .entry((AnalyticsId::NfLoad, PlmnId::new("310", "260")))
            .or_default()
            .push(new_result);

        let aggregated = mgr
            .aggregate_results(AnalyticsId::NfLoad, FederatedAggregationMethod::MostRecent)
            .unwrap();

        // Should pick the more recent result
        assert_eq!(aggregated.confidence, 0.95);
    }

    #[test]
    fn test_aggregate_no_results() {
        let mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let result = mgr.aggregate_results(
            AnalyticsId::UeMobility,
            FederatedAggregationMethod::WeightedAverage,
        );
        assert!(result.is_none());
    }

    #[test]
    fn test_privacy_policy_custom() {
        let policy = PrivacyPolicy {
            k_anonymity_threshold: 10,
            dp_epsilon: 0.5,
            strip_subscriber_ids: true,
            generalize_location: true,
            retention_seconds: 3600,
            required_consent: vec![ConsentType::Analytics, ConsentType::CrossOperatorSharing],
        };

        let mgr =
            DataFederationManager::with_privacy_policy(PlmnId::new("001", "01"), policy);

        assert_eq!(mgr.privacy_policy().k_anonymity_threshold, 10);
        assert_eq!(mgr.privacy_policy().dp_epsilon, 0.5);
    }

    #[test]
    fn test_trust_level_ordering() {
        assert!(TrustLevel::Untrusted < TrustLevel::Basic);
        assert!(TrustLevel::Basic < TrustLevel::Roaming);
        assert!(TrustLevel::Roaming < TrustLevel::Full);
    }

    #[test]
    fn test_rate_limiting() {
        let mut mgr = DataFederationManager::new(PlmnId::new("001", "01"));
        let target = PlmnId::new("310", "410");
        let mut peer = make_peer("310", "410", "OpA", TrustLevel::Roaming);
        peer.rate_limit_rpm = 3;
        mgr.register_peer(peer).unwrap();

        // Should succeed for first 3 requests
        for _ in 0..3 {
            let result = mgr.create_request(&target, AnalyticsId::UeMobility, None, 10);
            assert!(result.is_ok());
        }

        // 4th request should fail (rate limit)
        let result = mgr.create_request(&target, AnalyticsId::UeMobility, None, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_full_federation_workflow() {
        // Simulate two operators exchanging analytics

        // Operator A
        let mut op_a = DataFederationManager::new(PlmnId::new("001", "01"));
        // Operator B
        let mut op_b = DataFederationManager::new(PlmnId::new("310", "410"));

        // Register each other as peers
        op_a.register_peer(make_peer("310", "410", "OpB", TrustLevel::Roaming))
            .unwrap();
        op_b.register_peer(make_peer("001", "01", "OpA", TrustLevel::Roaming))
            .unwrap();

        // Op A creates request to Op B
        let request = op_a
            .create_request(
                &PlmnId::new("310", "410"),
                AnalyticsId::UeMobility,
                None,
                10,
            )
            .unwrap();

        // Op B has local results
        let local_results = vec![
            make_result(AnalyticsId::UeMobility, 50, 0.9),
            make_result(AnalyticsId::UeMobility, 30, 0.8),
        ];

        // Op B handles the request
        let response = op_b.handle_request(&request, &local_results);
        assert!(response.accepted);
        assert_eq!(response.results.len(), 2); // Both pass k-anonymity (50>=5, 30>=5)

        // Op A processes the response
        let count = op_a.process_response(response).unwrap();
        assert_eq!(count, 2);

        // Op A aggregates
        let aggregated = op_a
            .aggregate_results(
                AnalyticsId::UeMobility,
                FederatedAggregationMethod::WeightedAverage,
            )
            .unwrap();

        assert_eq!(aggregated.contributing_plmns.len(), 1); // Only Op B contributed
        assert_eq!(aggregated.total_samples, 80);
        assert!(aggregated.confidence > 0.0);
    }
}
