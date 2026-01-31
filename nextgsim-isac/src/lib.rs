//! Integrated Sensing and Communication (ISAC) for 6G Networks
//!
//! Implements ISAC per 3GPP TR 22.837:
//! - RAN-level sensing data collection
//! - Core aggregation and fusion
//! - Edge inference for positioning/tracking
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────┐
//! │                              ISAC                                        │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Sensing Data Sources                                             │   │
//! │  │  • Time of Arrival (ToA)                                         │   │
//! │  │  • Angle of Arrival (AoA)                                        │   │
//! │  │  • Received Signal Strength (RSS)                                │   │
//! │  │  • Doppler measurements                                          │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Data Fusion                                                      │   │
//! │  │  • Kalman filtering                                              │   │
//! │  │  • Particle filtering                                            │   │
//! │  │  • Multi-sensor fusion                                           │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Positioning/Tracking                                             │   │
//! │  │  • Position estimation                                           │   │
//! │  │  • Velocity estimation                                           │   │
//! │  │  • Object tracking                                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// 3D position vector
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct Vector3 {
    /// X coordinate (meters)
    pub x: f64,
    /// Y coordinate (meters)
    pub y: f64,
    /// Z coordinate (meters)
    pub z: f64,
}

impl Vector3 {
    /// Creates a new Vector3
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Calculates Euclidean distance to another point
    pub fn distance_to(&self, other: &Vector3) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculates magnitude
    pub fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

/// Sensing measurement type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensingType {
    /// Time of Arrival
    ToA,
    /// Time Difference of Arrival
    TDoA,
    /// Angle of Arrival (azimuth)
    AoA,
    /// Angle of Arrival (elevation)
    ZoA,
    /// Received Signal Strength
    Rss,
    /// Doppler shift
    Doppler,
    /// Round-Trip Time
    Rtt,
}

/// Single sensing measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingMeasurement {
    /// Measurement type
    pub measurement_type: SensingType,
    /// Cell/anchor ID
    pub anchor_id: i32,
    /// Measured value
    pub value: f64,
    /// Measurement uncertainty (standard deviation)
    pub uncertainty: f64,
    /// Timestamp (ms since epoch)
    pub timestamp_ms: u64,
}

/// Aggregated sensing data from multiple sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingData {
    /// Target entity ID (UE ID or object ID)
    pub target_id: i32,
    /// Cell ID where sensing originated
    pub cell_id: i32,
    /// Individual measurements
    pub measurements: Vec<SensingMeasurement>,
    /// Collection timestamp
    pub timestamp_ms: u64,
}

/// Data source for fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    /// Source type
    pub source_type: SensingType,
    /// Source cell/anchor ID
    pub anchor_id: i32,
    /// Source position (known)
    pub position: Vector3,
}

/// Fused position result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedPosition {
    /// Target ID
    pub target_id: i32,
    /// Estimated position
    pub position: Vector3,
    /// Estimated velocity
    pub velocity: Vector3,
    /// Position uncertainty (meters)
    pub position_uncertainty: f64,
    /// Velocity uncertainty (m/s)
    pub velocity_uncertainty: f64,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// Tracking state for an object
#[derive(Debug, Clone)]
pub struct TrackingState {
    /// Object ID
    pub object_id: u64,
    /// Current position estimate
    pub position: Vector3,
    /// Current velocity estimate
    pub velocity: Vector3,
    /// Position covariance (simplified as uncertainty)
    pub position_uncertainty: f64,
    /// Last update time
    pub last_update: Instant,
    /// Track quality (0.0 to 1.0)
    pub quality: f32,
}

impl TrackingState {
    /// Creates a new tracking state
    pub fn new(object_id: u64, initial_position: Vector3) -> Self {
        Self {
            object_id,
            position: initial_position,
            velocity: Vector3::default(),
            position_uncertainty: 10.0, // Initial uncertainty: 10 meters
            last_update: Instant::now(),
            quality: 0.5,
        }
    }

    /// Updates state with new position measurement
    pub fn update(&mut self, measured_position: Vector3, measurement_uncertainty: f64) {
        let dt = self.last_update.elapsed().as_secs_f64();
        self.last_update = Instant::now();

        if dt > 0.0 {
            // Simple Kalman-like update
            let kalman_gain = self.position_uncertainty
                / (self.position_uncertainty + measurement_uncertainty);

            // Update position
            self.position.x += kalman_gain * (measured_position.x - self.position.x);
            self.position.y += kalman_gain * (measured_position.y - self.position.y);
            self.position.z += kalman_gain * (measured_position.z - self.position.z);

            // Update velocity estimate
            if dt < 1.0 {
                // Only update velocity for reasonable time intervals
                let old_pos = self.position;
                self.velocity.x = (measured_position.x - old_pos.x) / dt;
                self.velocity.y = (measured_position.y - old_pos.y) / dt;
                self.velocity.z = (measured_position.z - old_pos.z) / dt;
            }

            // Update uncertainty
            self.position_uncertainty =
                ((1.0 - kalman_gain) * self.position_uncertainty).max(0.1);

            // Update quality based on update frequency and uncertainty
            self.quality = (1.0 / (1.0 + self.position_uncertainty / 10.0)).min(1.0) as f32;
        }
    }

    /// Predicts position after elapsed time
    pub fn predict(&self, dt_seconds: f64) -> Vector3 {
        Vector3::new(
            self.position.x + self.velocity.x * dt_seconds,
            self.position.y + self.velocity.y * dt_seconds,
            self.position.z + self.velocity.z * dt_seconds,
        )
    }
}

/// ISAC messages
#[derive(Debug)]
pub enum IsacMessage {
    /// Sensing data from cell
    SensingData(SensingData),
    /// Request position fusion
    FusionRequest {
        /// Target ID
        target_id: i32,
        /// Data sources to use
        data_sources: Vec<DataSource>,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<IsacResponse>>,
    },
    /// Update tracking
    TrackingUpdate {
        /// Object ID
        object_id: u64,
        /// New position measurement
        position: Vector3,
        /// Measurement uncertainty
        uncertainty: f64,
    },
    /// Query tracking state
    QueryTracking {
        /// Object ID
        object_id: u64,
        /// Response channel
        response_tx: Option<tokio::sync::oneshot::Sender<IsacResponse>>,
    },
}

/// ISAC responses
#[derive(Debug)]
pub enum IsacResponse {
    /// Fused position result
    FusedPosition(FusedPosition),
    /// Tracking state
    TrackingState {
        /// Object ID
        object_id: u64,
        /// Position
        position: Vector3,
        /// Velocity
        velocity: Vector3,
        /// Quality
        quality: f32,
    },
    /// Error
    Error(String),
}

/// Simple position fusion using weighted least squares
pub fn fuse_positions(
    measurements: &[SensingMeasurement],
    anchors: &HashMap<i32, Vector3>,
) -> Option<(Vector3, f64)> {
    // Collect ToA/RSS measurements with known anchor positions
    let mut positions: Vec<(Vector3, f64)> = Vec::new();

    for measurement in measurements {
        if let Some(anchor_pos) = anchors.get(&measurement.anchor_id) {
            match measurement.measurement_type {
                SensingType::ToA | SensingType::Rtt => {
                    // ToA gives distance - use trilateration
                    // For simplicity, assume measurement.value is distance in meters
                    let distance = measurement.value;
                    let weight = 1.0 / measurement.uncertainty.max(0.1);
                    positions.push((*anchor_pos, distance * weight));
                }
                SensingType::Rss => {
                    // RSS can be converted to distance using path loss model
                    // Simplified: RSS_dBm = -40 - 20*log10(distance)
                    let rss = measurement.value;
                    let distance = 10.0_f64.powf((-40.0 - rss) / 20.0);
                    let weight = 1.0 / measurement.uncertainty.max(0.1);
                    positions.push((*anchor_pos, distance * weight));
                }
                _ => continue,
            }
        }
    }

    if positions.len() < 3 {
        return None; // Need at least 3 anchors for 2D positioning
    }

    // Simple centroid-based estimation (production would use proper trilateration)
    let total_weight: f64 = positions.iter().map(|(_, w)| *w).sum();
    let x = positions.iter().map(|(p, w)| p.x * w).sum::<f64>() / total_weight;
    let y = positions.iter().map(|(p, w)| p.y * w).sum::<f64>() / total_weight;
    let z = positions.iter().map(|(p, w)| p.z * w).sum::<f64>() / total_weight;

    // Estimate uncertainty from measurement uncertainties
    let avg_uncertainty: f64 = measurements.iter().map(|m| m.uncertainty).sum::<f64>()
        / measurements.len() as f64;

    Some((Vector3::new(x, y, z), avg_uncertainty))
}

/// ISAC manager
#[derive(Debug, Default)]
pub struct IsacManager {
    /// Anchor positions (cell_id -> position)
    anchors: HashMap<i32, Vector3>,
    /// Tracking states (object_id -> state)
    tracking: HashMap<u64, TrackingState>,
    /// Recent sensing data
    recent_data: HashMap<i32, SensingData>,
    /// Fusion interval (ms)
    fusion_interval_ms: u32,
}

impl IsacManager {
    /// Creates a new ISAC manager
    pub fn new(fusion_interval_ms: u32) -> Self {
        Self {
            anchors: HashMap::new(),
            tracking: HashMap::new(),
            recent_data: HashMap::new(),
            fusion_interval_ms,
        }
    }

    /// Registers an anchor (cell/base station) position
    pub fn register_anchor(&mut self, anchor_id: i32, position: Vector3) {
        self.anchors.insert(anchor_id, position);
    }

    /// Records sensing data
    pub fn record_sensing_data(&mut self, data: SensingData) {
        self.recent_data.insert(data.target_id, data);
    }

    /// Performs position fusion for a target
    pub fn fuse_position(&self, target_id: i32) -> Option<FusedPosition> {
        let data = self.recent_data.get(&target_id)?;

        let (position, uncertainty) = fuse_positions(&data.measurements, &self.anchors)?;

        Some(FusedPosition {
            target_id,
            position,
            velocity: Vector3::default(),
            position_uncertainty: uncertainty,
            velocity_uncertainty: 1.0,
            confidence: (1.0 / (1.0 + uncertainty / 10.0)).min(1.0) as f32,
            timestamp_ms: data.timestamp_ms,
        })
    }

    /// Updates or creates a tracking state
    pub fn update_tracking(&mut self, object_id: u64, position: Vector3, uncertainty: f64) {
        let state = self
            .tracking
            .entry(object_id)
            .or_insert_with(|| TrackingState::new(object_id, position));

        state.update(position, uncertainty);
    }

    /// Gets tracking state for an object
    pub fn get_tracking(&self, object_id: u64) -> Option<&TrackingState> {
        self.tracking.get(&object_id)
    }

    /// Removes stale tracks
    pub fn cleanup_stale_tracks(&mut self, max_age_seconds: f64) {
        self.tracking.retain(|_, state| {
            state.last_update.elapsed().as_secs_f64() < max_age_seconds
        });
    }

    /// Returns the number of active tracks
    pub fn active_track_count(&self) -> usize {
        self.tracking.len()
    }

    /// Returns the configured fusion interval in milliseconds
    pub fn fusion_interval_ms(&self) -> u32 {
        self.fusion_interval_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector3_distance() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_tracking_state() {
        let mut state = TrackingState::new(1, Vector3::new(0.0, 0.0, 0.0));

        // Simulate movement
        std::thread::sleep(std::time::Duration::from_millis(100));
        state.update(Vector3::new(10.0, 5.0, 0.0), 1.0);

        assert!(state.position.x > 0.0);
        assert!(state.quality > 0.0);
    }

    #[test]
    fn test_tracking_prediction() {
        let mut state = TrackingState::new(1, Vector3::new(0.0, 0.0, 0.0));
        state.velocity = Vector3::new(10.0, 5.0, 0.0); // 10 m/s in x, 5 m/s in y

        let predicted = state.predict(1.0); // Predict 1 second ahead
        assert!((predicted.x - 10.0).abs() < 0.001);
        assert!((predicted.y - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_isac_manager() {
        let mut manager = IsacManager::new(50);

        // Register anchors
        manager.register_anchor(1, Vector3::new(0.0, 0.0, 0.0));
        manager.register_anchor(2, Vector3::new(100.0, 0.0, 0.0));
        manager.register_anchor(3, Vector3::new(50.0, 86.6, 0.0));

        // Update tracking
        manager.update_tracking(1, Vector3::new(50.0, 40.0, 0.0), 1.0);

        let state = manager.get_tracking(1);
        assert!(state.is_some());
    }

    #[test]
    fn test_position_fusion() {
        let mut anchors = HashMap::new();
        anchors.insert(1, Vector3::new(0.0, 0.0, 0.0));
        anchors.insert(2, Vector3::new(100.0, 0.0, 0.0));
        anchors.insert(3, Vector3::new(50.0, 86.6, 0.0));

        let measurements = vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: 50.0, // 50m from anchor 1
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: 50.0, // 50m from anchor 2
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 3,
                value: 50.0, // 50m from anchor 3
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
        ];

        let result = fuse_positions(&measurements, &anchors);
        assert!(result.is_some());
    }
}
