//! Sensing-as-a-Service exposure API — models concepts from 3GPP TR 22.837
//! (a Stage-1 feasibility study — no normative procedures/encodings exist).
//!
//! Illustrative in-process exposure interface for 6G sensing capabilities;
//! NOT a conformance implementation.

#![allow(missing_docs)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::{FusedPosition, Vector3};

/// Sensing service type per TR 22.837
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SensingServiceType {
    /// Object detection and classification
    ObjectDetection,
    /// Positioning and localization
    Positioning,
    /// Velocity estimation
    VelocityEstimation,
    /// Environment mapping
    EnvironmentMapping,
    /// Gesture recognition
    GestureRecognition,
    /// Intrusion detection
    IntrusionDetection,
}

impl std::fmt::Display for SensingServiceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensingServiceType::ObjectDetection => write!(f, "ObjectDetection"),
            SensingServiceType::Positioning => write!(f, "Positioning"),
            SensingServiceType::VelocityEstimation => write!(f, "VelocityEstimation"),
            SensingServiceType::EnvironmentMapping => write!(f, "EnvironmentMapping"),
            SensingServiceType::GestureRecognition => write!(f, "GestureRecognition"),
            SensingServiceType::IntrusionDetection => write!(f, "IntrusionDetection"),
        }
    }
}

/// Identity and consent state of a sensing-service consumer.
///
/// Carries the caller identity, an explicit user/operator consent flag and the
/// set of sensing service types the consumer is authorized for. Modeled after
/// TR 22.837 §7.1.2 "Configuration and authorization" — the TR pervasively
/// frames sensing as "subject to user consent, regulation and operator's
/// policy … authorized". This is a research-grade model, not a wire format.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SensingConsumer {
    /// Consumer identity (e.g. application / third-party id).
    pub consumer_id: String,
    /// Whether user/operator consent has been granted to this consumer.
    pub consent_granted: bool,
    /// Sensing service types this consumer is authorized for (its scopes).
    pub scopes: Vec<SensingServiceType>,
}

impl SensingConsumer {
    /// Creates a new consumer with the given identity, consent flag and scopes.
    pub fn new(
        consumer_id: impl Into<String>,
        consent_granted: bool,
        scopes: Vec<SensingServiceType>,
    ) -> Self {
        Self {
            consumer_id: consumer_id.into(),
            consent_granted,
            scopes,
        }
    }

    /// An anonymous consumer with no consent and no scopes.
    ///
    /// Used as a backwards-compatible default for callers that do not supply an
    /// identity. Under a [`ConsentRequiredPolicy`] such a consumer is denied.
    pub fn anonymous() -> Self {
        Self::default()
    }
}

/// Outcome of an authorization decision by a [`SensingPolicy`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The request is authorized.
    Allow,
    /// The request is denied, with a human-readable reason.
    Deny {
        /// Reason the request was denied (surfaced as the 403 error message).
        reason: String,
    },
}

/// Authorization decision point for sensing requests (TR 22.837 §7.1.2/§7.1.4-2).
///
/// A pluggable policy that decides whether a [`SensingConsumer`] may perform a
/// given [`SensingApiRequest`]. The TR repeats authorization/consent/operator-
/// policy gating on nearly every requirement; this trait makes that concept
/// present and testable. It is NOT a 3GPP-defined interface.
pub trait SensingPolicy: std::fmt::Debug + Send + Sync {
    /// Returns the authorization decision for `consumer` performing `req`.
    fn authorize(&self, consumer: &SensingConsumer, req: &SensingApiRequest) -> PolicyDecision;
}

/// Research stub policy that authorizes every request.
///
/// This is the default wired by [`SensingAsAService::new`]. It exists so the
/// authorization concept is present without changing existing behavior; it is
/// explicitly NOT a real access-control policy.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultAllowPolicy;

impl SensingPolicy for DefaultAllowPolicy {
    fn authorize(&self, _consumer: &SensingConsumer, _req: &SensingApiRequest) -> PolicyDecision {
        PolicyDecision::Allow
    }
}

/// Policy that requires explicit consent and that the requested service type is
/// within the consumer's authorized scopes (TR 22.837 §7.1.2 / §7.1.4-2).
#[derive(Debug, Default, Clone, Copy)]
pub struct ConsentRequiredPolicy;

impl SensingPolicy for ConsentRequiredPolicy {
    fn authorize(&self, consumer: &SensingConsumer, req: &SensingApiRequest) -> PolicyDecision {
        if !consumer.consent_granted {
            return PolicyDecision::Deny {
                reason: "consent not granted".to_string(),
            };
        }
        if let Some(service_type) = req.service_type() {
            if !consumer.scopes.contains(&service_type) {
                return PolicyDecision::Deny {
                    reason: format!("service type {service_type} not in authorized scopes"),
                };
            }
        }
        PolicyDecision::Allow
    }
}

/// Lifecycle status of a sensing subscription.
///
/// Models TR 22.837 §7.1.2 CPR 7.1.2-2's authorize/**revoke** concept: an
/// authorized subscription may be suspended or revoked subject to operator
/// policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum SubscriptionStatus {
    /// Active and delivering results.
    #[default]
    Active,
    /// Temporarily suspended (authorization retained).
    Suspended,
    /// Revoked (authorization withdrawn); queries are rejected.
    Revoked,
}

/// Sensing service subscription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingSubscription {
    /// Subscription ID
    pub subscription_id: u64,
    /// Service type
    pub service_type: SensingServiceType,
    /// Target area (optional)
    pub target_area: Option<GeographicArea>,
    /// Update interval (milliseconds)
    pub update_interval_ms: u32,
    /// Quality of Service requirements
    pub qos: SensingQos,
    /// Callback URL for notifications
    pub callback_url: Option<String>,
    /// Creation timestamp (milliseconds since epoch)
    pub created_at_ms: u64,
    /// Identity of the consumer that created this subscription.
    #[serde(default)]
    pub consumer_id: Option<String>,
    /// Whether consent was granted at subscription time.
    #[serde(default)]
    pub consent_granted: bool,
    /// Lifecycle status (TR 22.837 §7.1.2-2 authorize/revoke).
    #[serde(default)]
    pub status: SubscriptionStatus,
}

/// Geographic area for sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeographicArea {
    /// Center point
    pub center: Vector3,
    /// Radius (meters)
    pub radius_m: f64,
}

impl GeographicArea {
    /// Creates a new geographic area
    pub fn new(center: Vector3, radius_m: f64) -> Self {
        Self { center, radius_m }
    }

    /// Checks if a position is within the area
    pub fn contains(&self, position: &Vector3) -> bool {
        self.center.distance_to(position) <= self.radius_m
    }
}

/// Quality of Service parameters for sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingQos {
    /// Required accuracy (meters)
    pub accuracy_m: f64,
    /// Maximum latency (milliseconds)
    pub max_latency_ms: u32,
    /// Minimum confidence (0.0 to 1.0)
    pub min_confidence: f64,
    /// Update rate (Hz)
    pub update_rate_hz: f64,
}

impl Default for SensingQos {
    fn default() -> Self {
        Self {
            accuracy_m: 1.0,
            max_latency_ms: 100,
            min_confidence: 0.8,
            update_rate_hz: 10.0,
        }
    }
}

/// Sensing API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensingApiRequest {
    /// Subscribe to a sensing service
    Subscribe {
        /// Identity/consent of the requesting consumer (authorization input).
        #[serde(default)]
        consumer: SensingConsumer,
        service_type: SensingServiceType,
        target_area: Option<GeographicArea>,
        qos: SensingQos,
        update_interval_ms: u32,
    },
    /// Unsubscribe from a service
    Unsubscribe { subscription_id: u64 },
    /// Query current sensing data
    Query {
        /// Identity/consent of the requesting consumer (authorization input).
        #[serde(default)]
        consumer: SensingConsumer,
        service_type: SensingServiceType,
        target_area: Option<GeographicArea>,
    },
    /// Update `QoS` parameters
    UpdateQos {
        subscription_id: u64,
        qos: SensingQos,
    },
    /// Revoke a subscription's authorization (TR 22.837 §7.1.2-2).
    Revoke { subscription_id: u64 },
    /// Re-activate a suspended subscription.
    Activate { subscription_id: u64 },
    /// Suspend an active subscription without revoking it.
    Deactivate { subscription_id: u64 },
}

impl SensingApiRequest {
    /// Returns the consumer identity for requests that carry one
    /// (`Subscribe`/`Query`), used at the authorization decision point.
    pub fn consumer(&self) -> Option<&SensingConsumer> {
        match self {
            SensingApiRequest::Subscribe { consumer, .. }
            | SensingApiRequest::Query { consumer, .. } => Some(consumer),
            _ => None,
        }
    }

    /// Returns the requested service type for requests that target one.
    pub fn service_type(&self) -> Option<SensingServiceType> {
        match self {
            SensingApiRequest::Subscribe { service_type, .. }
            | SensingApiRequest::Query { service_type, .. } => Some(*service_type),
            _ => None,
        }
    }
}

/// Sensing API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensingApiResponse {
    /// Subscription created
    Subscribed {
        subscription_id: u64,
        service_type: SensingServiceType,
    },
    /// Unsubscribed successfully
    Unsubscribed { subscription_id: u64 },
    /// Query result
    QueryResult { results: Vec<SensingResult> },
    /// `QoS` updated
    QosUpdated { subscription_id: u64 },
    /// Subscription lifecycle status changed (revoke/activate/deactivate).
    StatusUpdated {
        subscription_id: u64,
        status: SubscriptionStatus,
    },
    /// Error
    Error { code: u32, message: String },
}

/// Sensing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingResult {
    /// Service type
    pub service_type: SensingServiceType,
    /// Target ID (if applicable)
    pub target_id: Option<i32>,
    /// Position (if applicable)
    pub position: Option<Vector3>,
    /// Velocity (if applicable)
    pub velocity: Option<Vector3>,
    /// Confidence score
    pub confidence: f64,
    /// Timestamp
    pub timestamp_ms: u64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Sensing-as-a-Service manager
#[derive(Debug)]
pub struct SensingAsAService {
    /// Active subscriptions
    subscriptions: HashMap<u64, SensingSubscription>,
    /// Next subscription ID
    next_subscription_id: u64,
    /// Cached sensing results
    cached_results: HashMap<SensingServiceType, Vec<SensingResult>>,
    /// Maximum cache size per service type
    max_cache_size: usize,
    /// Authorization decision point for Subscribe/Query (TR 22.837 §7.1.2).
    policy: Box<dyn SensingPolicy>,
}

impl SensingAsAService {
    /// Creates a new `SaaS` manager wired with the [`DefaultAllowPolicy`].
    pub fn new() -> Self {
        Self::with_policy(Box::new(DefaultAllowPolicy))
    }

    /// Creates a new `SaaS` manager with a custom authorization policy.
    pub fn with_policy(policy: Box<dyn SensingPolicy>) -> Self {
        Self {
            subscriptions: HashMap::new(),
            next_subscription_id: 1,
            cached_results: HashMap::new(),
            max_cache_size: 1000,
            policy,
        }
    }

    /// Handles an API request
    pub fn handle_request(&mut self, request: SensingApiRequest) -> SensingApiResponse {
        // Authorization decision point (TR 22.837 §7.1.2 / §7.1.4-2): gate the
        // consumer-bearing requests (Subscribe/Query) through the policy.
        if let Some(consumer) = request.consumer() {
            if let PolicyDecision::Deny { reason } = self.policy.authorize(consumer, &request) {
                return SensingApiResponse::Error {
                    code: 403,
                    message: reason,
                };
            }
        }

        match request {
            SensingApiRequest::Subscribe {
                consumer,
                service_type,
                target_area,
                qos,
                update_interval_ms,
            } => {
                let subscription_id = self.next_subscription_id;
                self.next_subscription_id += 1;

                let subscription = SensingSubscription {
                    subscription_id,
                    service_type,
                    target_area,
                    update_interval_ms,
                    qos,
                    callback_url: None,
                    created_at_ms: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .expect("value expected")
                        .as_millis() as u64,
                    consumer_id: (!consumer.consumer_id.is_empty())
                        .then_some(consumer.consumer_id),
                    consent_granted: consumer.consent_granted,
                    status: SubscriptionStatus::Active,
                };

                self.subscriptions.insert(subscription_id, subscription);

                SensingApiResponse::Subscribed {
                    subscription_id,
                    service_type,
                }
            }
            SensingApiRequest::Unsubscribe { subscription_id } => {
                self.subscriptions.remove(&subscription_id);
                SensingApiResponse::Unsubscribed { subscription_id }
            }
            SensingApiRequest::Query {
                consumer,
                service_type,
                target_area,
            } => {
                // Reject if this consumer holds a revoked subscription for the
                // requested service type (TR 22.837 §7.1.2-2 revoke).
                let revoked = self.subscriptions.values().any(|s| {
                    s.service_type == service_type
                        && s.status == SubscriptionStatus::Revoked
                        && s.consumer_id.as_deref() == Some(consumer.consumer_id.as_str())
                });
                if revoked {
                    return SensingApiResponse::Error {
                        code: 403,
                        message: "subscription revoked".to_string(),
                    };
                }
                let results = self.query_sensing_data(service_type, target_area.as_ref());
                SensingApiResponse::QueryResult { results }
            }
            SensingApiRequest::UpdateQos {
                subscription_id,
                qos,
            } => {
                if let Some(sub) = self.subscriptions.get_mut(&subscription_id) {
                    sub.qos = qos;
                    SensingApiResponse::QosUpdated { subscription_id }
                } else {
                    SensingApiResponse::Error {
                        code: 404,
                        message: "Subscription not found".to_string(),
                    }
                }
            }
            SensingApiRequest::Revoke { subscription_id } => {
                self.set_subscription_status(subscription_id, SubscriptionStatus::Revoked)
            }
            SensingApiRequest::Activate { subscription_id } => {
                self.set_subscription_status(subscription_id, SubscriptionStatus::Active)
            }
            SensingApiRequest::Deactivate { subscription_id } => {
                self.set_subscription_status(subscription_id, SubscriptionStatus::Suspended)
            }
        }
    }

    /// Sets a subscription's lifecycle status (TR 22.837 §7.1.2-2).
    fn set_subscription_status(
        &mut self,
        subscription_id: u64,
        status: SubscriptionStatus,
    ) -> SensingApiResponse {
        if let Some(sub) = self.subscriptions.get_mut(&subscription_id) {
            sub.status = status;
            SensingApiResponse::StatusUpdated {
                subscription_id,
                status,
            }
        } else {
            SensingApiResponse::Error {
                code: 404,
                message: "Subscription not found".to_string(),
            }
        }
    }

    /// Queries sensing data
    fn query_sensing_data(
        &self,
        service_type: SensingServiceType,
        target_area: Option<&GeographicArea>,
    ) -> Vec<SensingResult> {
        if let Some(cached) = self.cached_results.get(&service_type) {
            // Filter by target area if specified
            if let Some(area) = target_area {
                cached
                    .iter()
                    .filter(|r| {
                        if let Some(pos) = &r.position {
                            area.contains(pos)
                        } else {
                            true
                        }
                    })
                    .cloned()
                    .collect()
            } else {
                cached.clone()
            }
        } else {
            Vec::new()
        }
    }

    /// Publishes sensing results (called by sensing system)
    pub fn publish_result(&mut self, result: SensingResult) {
        let service_type = result.service_type;
        let cache = self.cached_results.entry(service_type).or_default();

        cache.push(result);

        // Trim cache
        if cache.len() > self.max_cache_size {
            cache.drain(0..cache.len() - self.max_cache_size);
        }

        // Notify subscribers (in a real implementation)
        // This would trigger callbacks for active subscriptions
    }

    /// Returns active subscriptions
    pub fn active_subscriptions(&self) -> Vec<&SensingSubscription> {
        self.subscriptions.values().collect()
    }

    /// Returns the number of active subscriptions
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Converts a `FusedPosition` to a `SensingResult`
    pub fn fused_position_to_result(
        fused: &FusedPosition,
        service_type: SensingServiceType,
    ) -> SensingResult {
        SensingResult {
            service_type,
            target_id: Some(fused.target_id),
            position: Some(fused.position),
            velocity: Some(fused.velocity),
            confidence: fused.confidence as f64,
            timestamp_ms: fused.timestamp_ms,
            metadata: HashMap::new(),
        }
    }
}

impl Default for SensingAsAService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_geographic_area() {
        let area = GeographicArea::new(Vector3::new(0.0, 0.0, 0.0), 100.0);

        assert!(area.contains(&Vector3::new(50.0, 50.0, 0.0)));
        assert!(!area.contains(&Vector3::new(150.0, 0.0, 0.0)));
    }

    #[test]
    fn test_saas_subscribe() {
        let mut saas = SensingAsAService::new();

        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };

        let response = saas.handle_request(request);

        match response {
            SensingApiResponse::Subscribed {
                subscription_id,
                service_type,
            } => {
                assert_eq!(subscription_id, 1);
                assert_eq!(service_type, SensingServiceType::Positioning);
            }
            _ => panic!("Expected Subscribed response"),
        }

        assert_eq!(saas.subscription_count(), 1);
    }

    #[test]
    fn test_saas_unsubscribe() {
        let mut saas = SensingAsAService::new();

        // Subscribe first
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        saas.handle_request(request);

        // Unsubscribe
        let request = SensingApiRequest::Unsubscribe { subscription_id: 1 };
        let response = saas.handle_request(request);

        match response {
            SensingApiResponse::Unsubscribed { subscription_id } => {
                assert_eq!(subscription_id, 1);
            }
            _ => panic!("Expected Unsubscribed response"),
        }

        assert_eq!(saas.subscription_count(), 0);
    }

    #[test]
    fn test_saas_publish_and_query() {
        let mut saas = SensingAsAService::new();

        // Publish a result
        let result = SensingResult {
            service_type: SensingServiceType::Positioning,
            target_id: Some(1),
            position: Some(Vector3::new(10.0, 20.0, 0.0)),
            velocity: None,
            confidence: 0.9,
            timestamp_ms: 1000,
            metadata: HashMap::new(),
        };

        saas.publish_result(result);

        // Query
        let request = SensingApiRequest::Query {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
        };

        let response = saas.handle_request(request);

        match response {
            SensingApiResponse::QueryResult { results } => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].target_id, Some(1));
            }
            _ => panic!("Expected QueryResult response"),
        }
    }

    #[test]
    fn test_saas_query_with_area() {
        let mut saas = SensingAsAService::new();

        // Publish results at different locations
        saas.publish_result(SensingResult {
            service_type: SensingServiceType::Positioning,
            target_id: Some(1),
            position: Some(Vector3::new(10.0, 10.0, 0.0)),
            velocity: None,
            confidence: 0.9,
            timestamp_ms: 1000,
            metadata: HashMap::new(),
        });

        saas.publish_result(SensingResult {
            service_type: SensingServiceType::Positioning,
            target_id: Some(2),
            position: Some(Vector3::new(200.0, 200.0, 0.0)),
            velocity: None,
            confidence: 0.9,
            timestamp_ms: 1000,
            metadata: HashMap::new(),
        });

        // Query with area filter
        let area = GeographicArea::new(Vector3::new(0.0, 0.0, 0.0), 50.0);
        let request = SensingApiRequest::Query {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: Some(area),
        };

        let response = saas.handle_request(request);

        match response {
            SensingApiResponse::QueryResult { results } => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].target_id, Some(1));
            }
            _ => panic!("Expected QueryResult response"),
        }
    }

    #[test]
    fn test_saas_update_qos() {
        let mut saas = SensingAsAService::new();

        // Subscribe
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        saas.handle_request(request);

        // Update QoS
        let new_qos = SensingQos {
            accuracy_m: 0.5,
            max_latency_ms: 50,
            min_confidence: 0.95,
            update_rate_hz: 20.0,
        };

        let request = SensingApiRequest::UpdateQos {
            subscription_id: 1,
            qos: new_qos.clone(),
        };

        let response = saas.handle_request(request);

        match response {
            SensingApiResponse::QosUpdated { subscription_id } => {
                assert_eq!(subscription_id, 1);
            }
            _ => panic!("Expected QosUpdated response"),
        }

        let sub = saas.subscriptions.get(&1).unwrap();
        assert!((sub.qos.accuracy_m - 0.5).abs() < 0.01);
    }

    // ----- isac-02: authorization / consent / operator-policy gating -----

    #[test]
    fn test_default_allow_policy_allows() {
        let mut saas = SensingAsAService::new();
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::anonymous(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        assert!(matches!(
            saas.handle_request(request),
            SensingApiResponse::Subscribed { .. }
        ));
    }

    #[test]
    fn test_consent_required_policy_denies_without_consent() {
        let mut saas = SensingAsAService::with_policy(Box::new(ConsentRequiredPolicy));
        // consent_granted = false
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::new("app-1", false, vec![SensingServiceType::Positioning]),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        match saas.handle_request(request) {
            SensingApiResponse::Error { code, .. } => assert_eq!(code, 403),
            other => panic!("expected 403 Error, got {other:?}"),
        }
        assert_eq!(saas.subscription_count(), 0);
    }

    #[test]
    fn test_consent_required_policy_allows_with_consent_and_scope() {
        let mut saas = SensingAsAService::with_policy(Box::new(ConsentRequiredPolicy));
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::new("app-1", true, vec![SensingServiceType::Positioning]),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        assert!(matches!(
            saas.handle_request(request),
            SensingApiResponse::Subscribed { .. }
        ));
        assert_eq!(saas.subscription_count(), 1);
    }

    #[test]
    fn test_consent_required_policy_denies_out_of_scope() {
        let mut saas = SensingAsAService::with_policy(Box::new(ConsentRequiredPolicy));
        // Consent granted but ObjectDetection not in scopes.
        let request = SensingApiRequest::Subscribe {
            consumer: SensingConsumer::new("app-1", true, vec![SensingServiceType::Positioning]),
            service_type: SensingServiceType::ObjectDetection,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        };
        match saas.handle_request(request) {
            SensingApiResponse::Error { code, .. } => assert_eq!(code, 403),
            other => panic!("expected 403 Error, got {other:?}"),
        }
    }

    #[test]
    fn test_revoked_subscription_rejected_on_query() {
        let mut saas = SensingAsAService::new();
        let consumer = SensingConsumer::new("app-1", true, vec![SensingServiceType::Positioning]);

        // Subscribe.
        let sub_id = match saas.handle_request(SensingApiRequest::Subscribe {
            consumer: consumer.clone(),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        }) {
            SensingApiResponse::Subscribed {
                subscription_id, ..
            } => subscription_id,
            other => panic!("expected Subscribed, got {other:?}"),
        };

        // Query before revoke succeeds.
        assert!(matches!(
            saas.handle_request(SensingApiRequest::Query {
                consumer: consumer.clone(),
                service_type: SensingServiceType::Positioning,
                target_area: None,
            }),
            SensingApiResponse::QueryResult { .. }
        ));

        // Revoke.
        match saas.handle_request(SensingApiRequest::Revoke {
            subscription_id: sub_id,
        }) {
            SensingApiResponse::StatusUpdated { status, .. } => {
                assert_eq!(status, SubscriptionStatus::Revoked)
            }
            other => panic!("expected StatusUpdated, got {other:?}"),
        }

        // Query after revoke is rejected.
        match saas.handle_request(SensingApiRequest::Query {
            consumer,
            service_type: SensingServiceType::Positioning,
            target_area: None,
        }) {
            SensingApiResponse::Error { code, .. } => assert_eq!(code, 403),
            other => panic!("expected 403 Error, got {other:?}"),
        }
    }

    #[test]
    fn test_subscription_persists_consumer_and_status() {
        let mut saas = SensingAsAService::new();
        saas.handle_request(SensingApiRequest::Subscribe {
            consumer: SensingConsumer::new("app-7", true, vec![SensingServiceType::Positioning]),
            service_type: SensingServiceType::Positioning,
            target_area: None,
            qos: SensingQos::default(),
            update_interval_ms: 100,
        });
        let sub = saas.subscriptions.get(&1).unwrap();
        assert_eq!(sub.consumer_id.as_deref(), Some("app-7"));
        assert!(sub.consent_granted);
        assert_eq!(sub.status, SubscriptionStatus::Active);
    }
}
