//! Sidelink Positioning Enhancement (Rel-18, TS 23.586)
//!
//! Implements advanced sidelink positioning features:
//! - SL-PRS (Sidelink Positioning Reference Signal) resource configuration
//! - RTT-based ranging measurement between UEs
//! - AoA/AoD (Angle of Arrival/Departure) estimation from sidelink
//! - Position calculation using multilateration
//! - Accuracy reporting (horizontal/vertical confidence)

use std::collections::HashMap;

/// SL-PRS (Sidelink Positioning Reference Signal) resource configuration.
///
/// Defines the time-frequency resources used for sidelink positioning signals.
#[derive(Debug, Clone)]
pub struct SlPrsResourceConfig {
    /// Resource ID
    pub resource_id: u16,
    /// Periodicity in slots (e.g., 10, 20, 40, 80, 160)
    pub periodicity_slots: u16,
    /// Slot offset for this resource
    pub slot_offset: u16,
    /// Number of symbols allocated
    pub num_symbols: u8,
    /// Starting symbol index
    pub start_symbol: u8,
    /// Comb size (2, 4, or 6)
    pub comb_size: u8,
    /// Transmission power in dBm
    pub tx_power_dbm: i8,
}

impl SlPrsResourceConfig {
    /// Creates a new SL-PRS resource configuration with default parameters.
    pub fn new(resource_id: u16) -> Self {
        Self {
            resource_id,
            periodicity_slots: 40,     // 40 slots = 40ms for 1kHz SCS
            slot_offset: 0,
            num_symbols: 12,           // One slot
            start_symbol: 0,
            comb_size: 4,
            tx_power_dbm: 23,
        }
    }

    /// Sets the periodicity in slots.
    pub fn with_periodicity(mut self, periodicity_slots: u16) -> Self {
        self.periodicity_slots = periodicity_slots;
        self
    }

    /// Sets the slot offset.
    pub fn with_offset(mut self, slot_offset: u16) -> Self {
        self.slot_offset = slot_offset;
        self
    }

    /// Sets the transmission power.
    pub fn with_tx_power(mut self, tx_power_dbm: i8) -> Self {
        self.tx_power_dbm = tx_power_dbm;
        self
    }
}

/// RTT (Round-Trip Time) measurement result between UEs.
#[derive(Debug, Clone, Copy)]
pub struct RttMeasurement {
    /// Peer UE identifier
    pub peer_ue_id: u64,
    /// Measured RTT in nanoseconds
    pub rtt_ns: u64,
    /// Estimated distance in meters (RTT * c / 2)
    pub distance_m: f64,
    /// Measurement quality (0.0 - 1.0)
    pub quality: f64,
    /// Timestamp of measurement
    pub timestamp_ms: u64,
}

impl RttMeasurement {
    /// Creates a new RTT measurement.
    ///
    /// Automatically calculates distance from RTT using speed of light.
    pub fn new(peer_ue_id: u64, rtt_ns: u64, quality: f64, timestamp_ms: u64) -> Self {
        // distance = (RTT * c) / 2, where c = 3e8 m/s = 0.3 m/ns
        let distance_m = (rtt_ns as f64 * 0.3) / 2.0;

        Self {
            peer_ue_id,
            rtt_ns,
            distance_m,
            quality,
            timestamp_ms,
        }
    }
}

/// AoA (Angle of Arrival) measurement.
///
/// Estimates the angle from which a signal arrived, used for triangulation.
#[derive(Debug, Clone, Copy)]
pub struct AoaMeasurement {
    /// Peer UE identifier
    pub peer_ue_id: u64,
    /// Azimuth angle in degrees (0-360, 0 = North, clockwise)
    pub azimuth_deg: f64,
    /// Elevation angle in degrees (-90 to +90, 0 = horizontal)
    pub elevation_deg: f64,
    /// Measurement confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Timestamp of measurement
    pub timestamp_ms: u64,
}

impl AoaMeasurement {
    /// Creates a new AoA measurement.
    pub fn new(
        peer_ue_id: u64,
        azimuth_deg: f64,
        elevation_deg: f64,
        confidence: f64,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            peer_ue_id,
            azimuth_deg: azimuth_deg % 360.0,
            elevation_deg: elevation_deg.clamp(-90.0, 90.0),
            confidence,
            timestamp_ms,
        }
    }
}

/// AoD (Angle of Departure) measurement.
///
/// Estimates the angle at which a signal was transmitted from peer UE.
#[derive(Debug, Clone, Copy)]
pub struct AodMeasurement {
    /// Peer UE identifier
    pub peer_ue_id: u64,
    /// Azimuth angle in degrees (0-360)
    pub azimuth_deg: f64,
    /// Elevation angle in degrees (-90 to +90)
    pub elevation_deg: f64,
    /// Measurement confidence (0.0 - 1.0)
    pub confidence: f64,
    /// Timestamp of measurement
    pub timestamp_ms: u64,
}

impl AodMeasurement {
    /// Creates a new AoD measurement.
    pub fn new(
        peer_ue_id: u64,
        azimuth_deg: f64,
        elevation_deg: f64,
        confidence: f64,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            peer_ue_id,
            azimuth_deg: azimuth_deg % 360.0,
            elevation_deg: elevation_deg.clamp(-90.0, 90.0),
            confidence,
            timestamp_ms,
        }
    }
}

/// 3D position estimate (ENU coordinates: East, North, Up).
#[derive(Debug, Clone, Copy)]
pub struct Position3D {
    /// East coordinate in meters
    pub east_m: f64,
    /// North coordinate in meters
    pub north_m: f64,
    /// Up coordinate in meters (altitude)
    pub up_m: f64,
}

impl Position3D {
    /// Creates a new 3D position.
    pub fn new(east_m: f64, north_m: f64, up_m: f64) -> Self {
        Self {
            east_m,
            north_m,
            up_m,
        }
    }

    /// Calculates Euclidean distance to another position.
    pub fn distance_to(&self, other: &Position3D) -> f64 {
        let dx = self.east_m - other.east_m;
        let dy = self.north_m - other.north_m;
        let dz = self.up_m - other.up_m;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Calculates horizontal distance to another position.
    pub fn horizontal_distance_to(&self, other: &Position3D) -> f64 {
        let dx = self.east_m - other.east_m;
        let dy = self.north_m - other.north_m;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Anchor UE with known position for multilateration.
#[derive(Debug, Clone)]
pub struct AnchorUe {
    /// Anchor UE identifier
    pub ue_id: u64,
    /// Known position of anchor UE
    pub position: Position3D,
    /// Distance measurement to this anchor
    pub distance_m: f64,
    /// Measurement quality (0.0 - 1.0)
    pub quality: f64,
}

impl AnchorUe {
    /// Creates a new anchor UE.
    pub fn new(ue_id: u64, position: Position3D, distance_m: f64, quality: f64) -> Self {
        Self {
            ue_id,
            position,
            distance_m,
            quality,
        }
    }
}

/// Position estimate with accuracy metrics.
#[derive(Debug, Clone, Copy)]
pub struct PositionEstimate {
    /// Estimated position
    pub position: Position3D,
    /// Horizontal accuracy (95% confidence) in meters
    pub horizontal_accuracy_m: f64,
    /// Vertical accuracy (95% confidence) in meters
    pub vertical_accuracy_m: f64,
    /// Number of measurements used
    pub num_measurements: usize,
    /// Timestamp of estimate
    pub timestamp_ms: u64,
}

impl PositionEstimate {
    /// Creates a new position estimate.
    pub fn new(
        position: Position3D,
        horizontal_accuracy_m: f64,
        vertical_accuracy_m: f64,
        num_measurements: usize,
        timestamp_ms: u64,
    ) -> Self {
        Self {
            position,
            horizontal_accuracy_m,
            vertical_accuracy_m,
            num_measurements,
            timestamp_ms,
        }
    }

    /// Returns true if position meets target accuracy.
    pub fn meets_accuracy(&self, target_horizontal_m: f64, target_vertical_m: f64) -> bool {
        self.horizontal_accuracy_m <= target_horizontal_m
            && self.vertical_accuracy_m <= target_vertical_m
    }
}

/// Sidelink positioning engine using multilateration.
#[derive(Debug, Clone)]
pub struct SidelinkPositioningEngine {
    /// SL-PRS resource configurations
    pub sl_prs_resources: Vec<SlPrsResourceConfig>,
    /// RTT measurements by peer UE ID
    pub rtt_measurements: HashMap<u64, RttMeasurement>,
    /// AoA measurements by peer UE ID
    pub aoa_measurements: HashMap<u64, AoaMeasurement>,
    /// AoD measurements by peer UE ID
    pub aod_measurements: HashMap<u64, AodMeasurement>,
    /// Known anchor UEs for multilateration
    pub anchors: Vec<AnchorUe>,
}

impl SidelinkPositioningEngine {
    /// Creates a new sidelink positioning engine.
    pub fn new() -> Self {
        Self {
            sl_prs_resources: Vec::new(),
            rtt_measurements: HashMap::new(),
            aoa_measurements: HashMap::new(),
            aod_measurements: HashMap::new(),
            anchors: Vec::new(),
        }
    }

    /// Adds an SL-PRS resource configuration.
    pub fn add_sl_prs_resource(&mut self, resource: SlPrsResourceConfig) {
        tracing::debug!(
            "SL-PRS: Adding resource {} with periodicity {} slots",
            resource.resource_id,
            resource.periodicity_slots
        );
        self.sl_prs_resources.push(resource);
    }

    /// Adds an RTT measurement.
    pub fn add_rtt_measurement(&mut self, measurement: RttMeasurement) {
        tracing::debug!(
            "SL-PRS: RTT measurement from peer {} = {:.2}m (RTT={}ns, quality={:.2})",
            measurement.peer_ue_id,
            measurement.distance_m,
            measurement.rtt_ns,
            measurement.quality
        );
        self.rtt_measurements
            .insert(measurement.peer_ue_id, measurement);
    }

    /// Adds an AoA measurement.
    pub fn add_aoa_measurement(&mut self, measurement: AoaMeasurement) {
        tracing::debug!(
            "SL-PRS: AoA measurement from peer {} = az={:.1}° el={:.1}° conf={:.2}",
            measurement.peer_ue_id,
            measurement.azimuth_deg,
            measurement.elevation_deg,
            measurement.confidence
        );
        self.aoa_measurements
            .insert(measurement.peer_ue_id, measurement);
    }

    /// Adds an AoD measurement.
    pub fn add_aod_measurement(&mut self, measurement: AodMeasurement) {
        tracing::debug!(
            "SL-PRS: AoD measurement from peer {} = az={:.1}° el={:.1}° conf={:.2}",
            measurement.peer_ue_id,
            measurement.azimuth_deg,
            measurement.elevation_deg,
            measurement.confidence
        );
        self.aod_measurements
            .insert(measurement.peer_ue_id, measurement);
    }

    /// Adds an anchor UE with known position.
    pub fn add_anchor(&mut self, anchor: AnchorUe) {
        tracing::info!(
            "SL-PRS: Added anchor UE {} at ({:.2}, {:.2}, {:.2}), dist={:.2}m",
            anchor.ue_id,
            anchor.position.east_m,
            anchor.position.north_m,
            anchor.position.up_m,
            anchor.distance_m
        );
        self.anchors.push(anchor);
    }

    /// Calculates position using multilateration from anchor UEs.
    ///
    /// Requires at least 3 anchors for 2D position, 4 for 3D position.
    pub fn calculate_position(&self, timestamp_ms: u64) -> Option<PositionEstimate> {
        if self.anchors.len() < 3 {
            tracing::warn!(
                "SL-PRS: Cannot calculate position, need at least 3 anchors (have {})",
                self.anchors.len()
            );
            return None;
        }

        // Use weighted least squares multilateration
        let position = self.multilateration_weighted_least_squares()?;

        // Estimate accuracy based on anchor geometry and measurement quality
        let (horizontal_accuracy, vertical_accuracy) = self.estimate_accuracy();

        let estimate = PositionEstimate::new(
            position,
            horizontal_accuracy,
            vertical_accuracy,
            self.anchors.len(),
            timestamp_ms,
        );

        tracing::info!(
            "SL-PRS: Position calculated at ({:.2}, {:.2}, {:.2}) ±{:.2}m(h) ±{:.2}m(v)",
            estimate.position.east_m,
            estimate.position.north_m,
            estimate.position.up_m,
            estimate.horizontal_accuracy_m,
            estimate.vertical_accuracy_m
        );

        Some(estimate)
    }

    /// Performs weighted least squares multilateration.
    ///
    /// Solves for position that minimizes weighted sum of squared range errors.
    fn multilateration_weighted_least_squares(&self) -> Option<Position3D> {
        if self.anchors.is_empty() {
            return None;
        }

        // Simple iterative solution using Gauss-Newton method
        // Initial guess: centroid of anchors
        let mut pos = Position3D::new(
            self.anchors.iter().map(|a| a.position.east_m).sum::<f64>() / self.anchors.len() as f64,
            self.anchors.iter().map(|a| a.position.north_m).sum::<f64>() / self.anchors.len() as f64,
            self.anchors.iter().map(|a| a.position.up_m).sum::<f64>() / self.anchors.len() as f64,
        );

        // Gauss-Newton iterations
        for _iter in 0..10 {
            let mut h_sum_e = 0.0;
            let mut h_sum_n = 0.0;
            let mut h_sum_u = 0.0;
            let mut weight_sum = 0.0;

            for anchor in &self.anchors {
                let dx = pos.east_m - anchor.position.east_m;
                let dy = pos.north_m - anchor.position.north_m;
                let dz = pos.up_m - anchor.position.up_m;
                let dist = (dx * dx + dy * dy + dz * dz).sqrt();

                if dist < 0.1 {
                    continue; // Avoid division by zero
                }

                let weight = anchor.quality;
                let error = dist - anchor.distance_m;

                h_sum_e += weight * error * dx / dist;
                h_sum_n += weight * error * dy / dist;
                h_sum_u += weight * error * dz / dist;
                weight_sum += weight;
            }

            if weight_sum < 0.001 {
                break;
            }

            // Update position estimate
            let step_size = 0.5; // Damping factor for stability
            pos.east_m -= step_size * h_sum_e / weight_sum;
            pos.north_m -= step_size * h_sum_n / weight_sum;
            pos.up_m -= step_size * h_sum_u / weight_sum;
        }

        Some(pos)
    }

    /// Estimates position accuracy based on anchor geometry and measurement quality.
    fn estimate_accuracy(&self) -> (f64, f64) {
        if self.anchors.is_empty() {
            return (100.0, 100.0); // Default poor accuracy
        }

        // Average measurement quality
        let avg_quality = self.anchors.iter().map(|a| a.quality).sum::<f64>() / self.anchors.len() as f64;

        // Base accuracy from measurement quality
        // High quality (0.9-1.0) -> 1-2m, medium (0.5-0.9) -> 2-5m, low (<0.5) -> 5-20m
        let base_accuracy = if avg_quality > 0.9 {
            1.5
        } else if avg_quality > 0.7 {
            3.0
        } else if avg_quality > 0.5 {
            5.0
        } else {
            10.0
        };

        // Geometric dilution of precision (GDOP) factor
        // More anchors and better distribution -> better accuracy
        let gdop_factor = if self.anchors.len() >= 5 {
            1.0 // Good geometry
        } else if self.anchors.len() == 4 {
            1.5 // Adequate geometry
        } else {
            2.0 // Marginal geometry
        };

        let horizontal_accuracy = base_accuracy * gdop_factor;
        let vertical_accuracy = horizontal_accuracy * 1.5; // Vertical typically worse

        (horizontal_accuracy, vertical_accuracy)
    }

    /// Clears all measurements and anchors.
    pub fn clear(&mut self) {
        self.rtt_measurements.clear();
        self.aoa_measurements.clear();
        self.aod_measurements.clear();
        self.anchors.clear();
        tracing::debug!("SL-PRS: Cleared all measurements and anchors");
    }
}

impl Default for SidelinkPositioningEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sl_prs_resource_config() {
        let resource = SlPrsResourceConfig::new(1)
            .with_periodicity(80)
            .with_offset(10)
            .with_tx_power(20);

        assert_eq!(resource.resource_id, 1);
        assert_eq!(resource.periodicity_slots, 80);
        assert_eq!(resource.slot_offset, 10);
        assert_eq!(resource.tx_power_dbm, 20);
    }

    #[test]
    fn test_rtt_measurement_distance() {
        let rtt_ns = 1000; // 1 microsecond
        let measurement = RttMeasurement::new(123, rtt_ns, 0.9, 0);

        // RTT=1000ns -> distance = (1000 * 0.3) / 2 = 150m
        assert!((measurement.distance_m - 150.0).abs() < 0.01);
    }

    #[test]
    fn test_aoa_measurement_angle_clamping() {
        let aoa = AoaMeasurement::new(123, 370.0, 100.0, 0.8, 0);

        // Azimuth should wrap to 0-360
        assert!((aoa.azimuth_deg - 10.0).abs() < 0.01);
        // Elevation should clamp to -90 to +90
        assert!((aoa.elevation_deg - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_position_3d_distance() {
        let p1 = Position3D::new(0.0, 0.0, 0.0);
        let p2 = Position3D::new(3.0, 4.0, 0.0);

        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.01); // 3-4-5 triangle
        assert!((p1.horizontal_distance_to(&p2) - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_multilateration_simple() {
        let mut engine = SidelinkPositioningEngine::new();

        // Create 4 anchors in a square around origin
        engine.add_anchor(AnchorUe::new(1, Position3D::new(10.0, 10.0, 0.0), 14.14, 1.0));
        engine.add_anchor(AnchorUe::new(2, Position3D::new(-10.0, 10.0, 0.0), 14.14, 1.0));
        engine.add_anchor(AnchorUe::new(3, Position3D::new(-10.0, -10.0, 0.0), 14.14, 1.0));
        engine.add_anchor(AnchorUe::new(4, Position3D::new(10.0, -10.0, 0.0), 14.14, 1.0));

        let estimate = engine.calculate_position(0);
        assert!(estimate.is_some());

        let pos = estimate.unwrap();
        // Should converge close to origin
        assert!(pos.position.east_m.abs() < 2.0);
        assert!(pos.position.north_m.abs() < 2.0);
    }

    #[test]
    fn test_position_estimate_accuracy_check() {
        let estimate = PositionEstimate::new(
            Position3D::new(0.0, 0.0, 0.0),
            2.0,
            3.0,
            4,
            0,
        );

        assert!(estimate.meets_accuracy(5.0, 5.0));
        assert!(!estimate.meets_accuracy(1.0, 5.0));
        assert!(!estimate.meets_accuracy(5.0, 2.0));
    }

    #[test]
    fn test_sidelink_positioning_engine_measurements() {
        let mut engine = SidelinkPositioningEngine::new();

        engine.add_rtt_measurement(RttMeasurement::new(123, 1000, 0.9, 0));
        engine.add_aoa_measurement(AoaMeasurement::new(123, 45.0, 10.0, 0.8, 0));
        engine.add_aod_measurement(AodMeasurement::new(123, 225.0, -5.0, 0.85, 0));

        assert_eq!(engine.rtt_measurements.len(), 1);
        assert_eq!(engine.aoa_measurements.len(), 1);
        assert_eq!(engine.aod_measurements.len(), 1);

        engine.clear();
        assert_eq!(engine.rtt_measurements.len(), 0);
        assert_eq!(engine.aoa_measurements.len(), 0);
        assert_eq!(engine.aod_measurements.len(), 0);
    }
}
