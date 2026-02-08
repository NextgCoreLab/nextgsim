//! Integration with Service Hosting Environment (SHE) for edge sensing inference
//!
//! This module provides interfaces for ISAC to offload sensing processing workloads
//! to the Service Hosting Environment for edge compute processing.

#![allow(missing_docs)]

use crate::{FusedPosition, SensingData, Vector3};
use serde::{Deserialize, Serialize};

/// Sensing workload request for SHE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingWorkloadRequest {
    /// Workload ID
    pub workload_id: u64,
    /// Sensing data to process
    pub sensing_data: SensingData,
    /// Processing type
    pub processing_type: SensingProcessingType,
    /// Latency requirement (milliseconds)
    pub latency_requirement_ms: u32,
    /// Priority (higher = more urgent)
    pub priority: u8,
}

/// Types of sensing processing that can be offloaded to SHE
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensingProcessingType {
    /// Position fusion and estimation
    PositionFusion,
    /// Object detection and tracking
    ObjectTracking,
    /// Velocity estimation
    VelocityEstimation,
    /// Environment mapping
    EnvironmentMapping,
    /// ML-based positioning inference
    MlPositioning,
}

impl std::fmt::Display for SensingProcessingType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SensingProcessingType::PositionFusion => write!(f, "PositionFusion"),
            SensingProcessingType::ObjectTracking => write!(f, "ObjectTracking"),
            SensingProcessingType::VelocityEstimation => write!(f, "VelocityEstimation"),
            SensingProcessingType::EnvironmentMapping => write!(f, "EnvironmentMapping"),
            SensingProcessingType::MlPositioning => write!(f, "MlPositioning"),
        }
    }
}

/// Result from sensing workload processing in SHE
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingWorkloadResult {
    /// Original workload ID
    pub workload_id: u64,
    /// Processing type that was executed
    pub processing_type: SensingProcessingType,
    /// Result data
    pub result: SensingProcessingResult,
    /// Processing latency (milliseconds)
    pub processing_latency_ms: u32,
    /// Tier where processing occurred
    pub processing_tier: String,
}

/// Result data from sensing processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensingProcessingResult {
    /// Fused position result
    Position(FusedPosition),
    /// Object detection result
    Objects(Vec<DetectedObject>),
    /// Velocity estimation result
    Velocity {
        target_id: i32,
        velocity: Vector3,
        uncertainty: f64,
    },
    /// Environment map result
    EnvironmentMap {
        map_id: u64,
        features: Vec<MapFeature>,
    },
}

/// Detected object from sensing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Object ID
    pub object_id: u64,
    /// Position
    pub position: Vector3,
    /// Velocity
    pub velocity: Vector3,
    /// Classification (if available)
    pub classification: Option<String>,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

/// Map feature from environment mapping
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapFeature {
    /// Feature ID
    pub feature_id: u64,
    /// Position
    pub position: Vector3,
    /// Feature type
    pub feature_type: String,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

/// Client for submitting sensing workloads to SHE
#[derive(Debug)]
pub struct SheIsacClient {
    next_workload_id: u64,
}

impl SheIsacClient {
    /// Creates a new SHE-ISAC client
    pub fn new() -> Self {
        Self {
            next_workload_id: 1,
        }
    }

    /// Submits a sensing workload to SHE for processing
    ///
    /// # Arguments
    ///
    /// * `sensing_data` - The sensing data to process
    /// * `processing_type` - Type of processing to perform
    /// * `latency_requirement_ms` - Maximum acceptable latency
    /// * `priority` - Priority level (0-10, higher is more urgent)
    ///
    /// # Returns
    ///
    /// The workload ID assigned to this request
    pub fn submit_sensing_workload(
        &mut self,
        _sensing_data: SensingData,
        _processing_type: SensingProcessingType,
        _latency_requirement_ms: u32,
        _priority: u8,
    ) -> u64 {
        let workload_id = self.next_workload_id;
        self.next_workload_id += 1;

        workload_id
    }

    /// Creates a sensing workload request
    pub fn create_request(
        &mut self,
        sensing_data: SensingData,
        processing_type: SensingProcessingType,
        latency_requirement_ms: u32,
        priority: u8,
    ) -> SensingWorkloadRequest {
        let workload_id = self.submit_sensing_workload(
            sensing_data.clone(),
            processing_type,
            latency_requirement_ms,
            priority,
        );

        SensingWorkloadRequest {
            workload_id,
            sensing_data,
            processing_type,
            latency_requirement_ms,
            priority,
        }
    }

    /// Submits position fusion workload to SHE
    pub fn submit_position_fusion(
        &mut self,
        sensing_data: SensingData,
    ) -> SensingWorkloadRequest {
        self.create_request(
            sensing_data,
            SensingProcessingType::PositionFusion,
            10, // 10ms latency requirement for edge
            7,  // High priority
        )
    }

    /// Submits object tracking workload to SHE
    pub fn submit_object_tracking(
        &mut self,
        sensing_data: SensingData,
    ) -> SensingWorkloadRequest {
        self.create_request(
            sensing_data,
            SensingProcessingType::ObjectTracking,
            15, // 15ms latency requirement
            6,
        )
    }

    /// Submits ML-based positioning workload to SHE
    pub fn submit_ml_positioning(&mut self, sensing_data: SensingData) -> SensingWorkloadRequest {
        self.create_request(
            sensing_data,
            SensingProcessingType::MlPositioning,
            20, // 20ms latency for regional edge
            5,
        )
    }
}

impl Default for SheIsacClient {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SensingMeasurement, SensingType};

    #[test]
    fn test_she_isac_client_creation() {
        let client = SheIsacClient::new();
        assert_eq!(client.next_workload_id, 1);
    }

    #[test]
    fn test_submit_sensing_workload() {
        let mut client = SheIsacClient::new();

        let sensing_data = SensingData {
            target_id: 1,
            cell_id: 100,
            measurements: vec![SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 50.0,
                uncertainty: 5.0,
                timestamp_ms: 1000,
            }],
            timestamp_ms: 1000,
        };

        let workload_id = client.submit_sensing_workload(
            sensing_data.clone(),
            SensingProcessingType::PositionFusion,
            10,
            7,
        );

        assert_eq!(workload_id, 1);

        // Submit another to verify ID increment
        let workload_id2 = client.submit_sensing_workload(
            sensing_data,
            SensingProcessingType::ObjectTracking,
            15,
            6,
        );

        assert_eq!(workload_id2, 2);
    }

    #[test]
    fn test_create_position_fusion_request() {
        let mut client = SheIsacClient::new();

        let sensing_data = SensingData {
            target_id: 1,
            cell_id: 100,
            measurements: vec![],
            timestamp_ms: 1000,
        };

        let request = client.submit_position_fusion(sensing_data);

        assert_eq!(request.processing_type, SensingProcessingType::PositionFusion);
        assert_eq!(request.latency_requirement_ms, 10);
        assert_eq!(request.priority, 7);
        assert_eq!(request.workload_id, 1);
    }

    #[test]
    fn test_create_ml_positioning_request() {
        let mut client = SheIsacClient::new();

        let sensing_data = SensingData {
            target_id: 1,
            cell_id: 100,
            measurements: vec![],
            timestamp_ms: 1000,
        };

        let request = client.submit_ml_positioning(sensing_data);

        assert_eq!(request.processing_type, SensingProcessingType::MlPositioning);
        assert_eq!(request.latency_requirement_ms, 20);
        assert_eq!(request.priority, 5);
    }

    #[test]
    fn test_sensing_processing_type_display() {
        assert_eq!(
            format!("{}", SensingProcessingType::PositionFusion),
            "PositionFusion"
        );
        assert_eq!(
            format!("{}", SensingProcessingType::MlPositioning),
            "MlPositioning"
        );
    }
}
