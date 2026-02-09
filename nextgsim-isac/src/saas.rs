//! Sensing-as-a-Service API per 3GPP TR 22.837
//!
//! Exposure interface for 6G sensing capabilities.

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
        service_type: SensingServiceType,
        target_area: Option<GeographicArea>,
        qos: SensingQos,
        update_interval_ms: u32,
    },
    /// Unsubscribe from a service
    Unsubscribe { subscription_id: u64 },
    /// Query current sensing data
    Query {
        service_type: SensingServiceType,
        target_area: Option<GeographicArea>,
    },
    /// Update QoS parameters
    UpdateQos {
        subscription_id: u64,
        qos: SensingQos,
    },
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
    /// QoS updated
    QosUpdated { subscription_id: u64 },
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
}

impl SensingAsAService {
    /// Creates a new SaaS manager
    pub fn new() -> Self {
        Self {
            subscriptions: HashMap::new(),
            next_subscription_id: 1,
            cached_results: HashMap::new(),
            max_cache_size: 1000,
        }
    }

    /// Handles an API request
    pub fn handle_request(&mut self, request: SensingApiRequest) -> SensingApiResponse {
        match request {
            SensingApiRequest::Subscribe {
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
                        .unwrap()
                        .as_millis() as u64,
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
                service_type,
                target_area,
            } => {
                let results = self.query_sensing_data(service_type, target_area.as_ref());
                SensingApiResponse::QueryResult { results }
            }
            SensingApiRequest::UpdateQos { subscription_id, qos } => {
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

    /// Converts a FusedPosition to a SensingResult
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
}
