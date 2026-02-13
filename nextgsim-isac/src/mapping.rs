//! Environment mapping (SLAM-like) for spatial awareness
//!
//! Implements radio-based environment mapping for 6G sensing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::Vector3;

/// Environment map cell (grid-based representation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapCell {
    /// Cell position (center)
    pub position: Vector3,
    /// Occupancy probability (0.0 = free, 1.0 = occupied)
    pub occupancy: f64,
    /// Confidence in occupancy estimate
    pub confidence: f64,
    /// Number of observations
    pub observation_count: u32,
    /// Last update timestamp
    pub last_update_ms: u64,
}

impl MapCell {
    /// Creates a new map cell
    pub fn new(position: Vector3) -> Self {
        Self {
            position,
            occupancy: 0.5, // Unknown initially
            confidence: 0.0,
            observation_count: 0,
            last_update_ms: 0,
        }
    }

    /// Updates occupancy based on a new observation
    pub fn update_occupancy(&mut self, observed_occupied: bool, timestamp_ms: u64) {
        self.observation_count += 1;
        self.last_update_ms = timestamp_ms;

        // Bayesian update (simplified)
        let observation = if observed_occupied { 0.9 } else { 0.1 };
        let alpha = 0.3; // Learning rate
        self.occupancy = (1.0 - alpha) * self.occupancy + alpha * observation;

        // Update confidence based on observation count
        self.confidence = (self.observation_count as f64 / 100.0).min(1.0);
    }

    /// Returns if the cell is likely occupied
    pub fn is_occupied(&self, threshold: f64) -> bool {
        self.occupancy > threshold && self.confidence > 0.5
    }
}

/// 3D occupancy grid map
#[derive(Debug)]
pub struct OccupancyGrid {
    /// Grid cells indexed by (x, y, z) coordinates
    cells: HashMap<(i32, i32, i32), MapCell>,
    /// Cell size (meters)
    cell_size_m: f64,
    /// Origin of the grid
    origin: Vector3,
}

impl OccupancyGrid {
    /// Creates a new occupancy grid
    pub fn new(origin: Vector3, cell_size_m: f64) -> Self {
        Self {
            cells: HashMap::new(),
            cell_size_m,
            origin,
        }
    }

    /// Converts a world position to grid coordinates
    fn world_to_grid(&self, position: &Vector3) -> (i32, i32, i32) {
        (
            ((position.x - self.origin.x) / self.cell_size_m).floor() as i32,
            ((position.y - self.origin.y) / self.cell_size_m).floor() as i32,
            ((position.z - self.origin.z) / self.cell_size_m).floor() as i32,
        )
    }

    /// Converts grid coordinates to world position (cell center)
    fn grid_to_world(&self, grid: (i32, i32, i32)) -> Vector3 {
        Vector3::new(
            self.origin.x + (grid.0 as f64 + 0.5) * self.cell_size_m,
            self.origin.y + (grid.1 as f64 + 0.5) * self.cell_size_m,
            self.origin.z + (grid.2 as f64 + 0.5) * self.cell_size_m,
        )
    }

    /// Updates a cell based on observation
    pub fn update_cell(&mut self, position: &Vector3, occupied: bool, timestamp_ms: u64) {
        let grid_coords = self.world_to_grid(position);
        let world_pos = self.grid_to_world(grid_coords);
        let cell = self.cells.entry(grid_coords).or_insert_with(|| {
            MapCell::new(world_pos)
        });

        cell.update_occupancy(occupied, timestamp_ms);
    }

    /// Gets a cell at a position
    pub fn get_cell(&self, position: &Vector3) -> Option<&MapCell> {
        let grid_coords = self.world_to_grid(position);
        self.cells.get(&grid_coords)
    }

    /// Returns all occupied cells above a threshold
    pub fn get_occupied_cells(&self, threshold: f64) -> Vec<&MapCell> {
        self.cells
            .values()
            .filter(|cell| cell.is_occupied(threshold))
            .collect()
    }

    /// Returns the number of cells in the map
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Clears old cells (older than `max_age_ms`)
    pub fn cleanup_old_cells(&mut self, current_time_ms: u64, max_age_ms: u64) {
        self.cells.retain(|_, cell| {
            current_time_ms - cell.last_update_ms < max_age_ms
        });
    }

    /// Returns the cell size in meters
    pub fn cell_size_m(&self) -> f64 {
        self.cell_size_m
    }
}

/// Environment feature (detected landmark or object)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentFeature {
    /// Feature ID
    pub feature_id: u64,
    /// Position
    pub position: Vector3,
    /// Feature type
    pub feature_type: FeatureType,
    /// Confidence
    pub confidence: f64,
    /// First detected timestamp
    pub first_detected_ms: u64,
    /// Last observed timestamp
    pub last_observed_ms: u64,
}

/// Feature type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// Reflector (strong scatter point)
    Reflector,
    /// Wall or large surface
    Wall,
    /// Corner
    Corner,
    /// Object
    Object,
}

/// Environment mapper (SLAM-like)
#[derive(Debug)]
pub struct EnvironmentMapper {
    /// Occupancy grid
    grid: OccupancyGrid,
    /// Detected features
    features: HashMap<u64, EnvironmentFeature>,
    /// Next feature ID
    next_feature_id: u64,
    /// Feature detection threshold
    feature_threshold: f64,
}

impl EnvironmentMapper {
    /// Creates a new environment mapper
    pub fn new(origin: Vector3, cell_size_m: f64) -> Self {
        Self {
            grid: OccupancyGrid::new(origin, cell_size_m),
            features: HashMap::new(),
            next_feature_id: 1,
            feature_threshold: 0.8,
        }
    }

    /// Processes a range measurement to update the map
    pub fn process_range_measurement(
        &mut self,
        sensor_pos: &Vector3,
        measured_range_m: f64,
        timestamp_ms: u64,
    ) {
        // Ray-tracing: mark cells along the ray as free, endpoint as occupied

        let num_steps = (measured_range_m / self.grid.cell_size_m).ceil() as usize;

        // Direction from sensor to target (simplified: assume azimuth=0)
        let dx = measured_range_m;
        let dy = 0.0;
        let dz = 0.0;

        // Mark free cells
        for i in 0..num_steps {
            let t = i as f64 / num_steps as f64;
            let pos = Vector3::new(
                sensor_pos.x + dx * t,
                sensor_pos.y + dy * t,
                sensor_pos.z + dz * t,
            );
            self.grid.update_cell(&pos, false, timestamp_ms);
        }

        // Mark endpoint as occupied
        let endpoint = Vector3::new(
            sensor_pos.x + dx,
            sensor_pos.y + dy,
            sensor_pos.z + dz,
        );
        self.grid.update_cell(&endpoint, true, timestamp_ms);

        // Check for feature detection
        if let Some(cell) = self.grid.get_cell(&endpoint) {
            if cell.occupancy > self.feature_threshold && cell.confidence > 0.7 {
                self.add_feature(endpoint, FeatureType::Reflector, cell.occupancy, timestamp_ms);
            }
        }
    }

    /// Adds or updates a feature
    fn add_feature(
        &mut self,
        position: Vector3,
        feature_type: FeatureType,
        confidence: f64,
        timestamp_ms: u64,
    ) {
        // Check if there's already a feature nearby (within 2m)
        let nearby_threshold = 2.0;
        for feature in self.features.values_mut() {
            if position.distance_to(&feature.position) < nearby_threshold {
                // Update existing feature
                feature.last_observed_ms = timestamp_ms;
                feature.confidence = (feature.confidence + confidence) / 2.0;
                return;
            }
        }

        // Add new feature
        let feature_id = self.next_feature_id;
        self.next_feature_id += 1;

        self.features.insert(
            feature_id,
            EnvironmentFeature {
                feature_id,
                position,
                feature_type,
                confidence,
                first_detected_ms: timestamp_ms,
                last_observed_ms: timestamp_ms,
            },
        );
    }

    /// Returns all detected features
    pub fn get_features(&self) -> Vec<&EnvironmentFeature> {
        self.features.values().collect()
    }

    /// Returns the occupancy grid
    pub fn grid(&self) -> &OccupancyGrid {
        &self.grid
    }

    /// Returns a mutable reference to the occupancy grid
    pub fn grid_mut(&mut self) -> &mut OccupancyGrid {
        &mut self.grid
    }

    /// Returns the number of detected features
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    /// Cleans up old data
    pub fn cleanup(&mut self, current_time_ms: u64, max_age_ms: u64) {
        self.grid.cleanup_old_cells(current_time_ms, max_age_ms);

        // Remove stale features
        self.features.retain(|_, feature| {
            current_time_ms - feature.last_observed_ms < max_age_ms
        });
    }
}

impl Default for EnvironmentMapper {
    fn default() -> Self {
        Self::new(Vector3::new(0.0, 0.0, 0.0), 1.0)
    }
}

// ─── SLAM (Simultaneous Localization And Mapping) ─────────────────────────────

/// 2D pose (x, y, heading) for SLAM
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Pose2D {
    /// X position (meters)
    pub x: f64,
    /// Y position (meters)
    pub y: f64,
    /// Heading angle (radians, counter-clockwise from X axis)
    pub heading: f64,
}

impl Pose2D {
    /// Creates a new 2D pose
    pub fn new(x: f64, y: f64, heading: f64) -> Self {
        Self { x, y, heading }
    }

    /// Applies a motion model: move forward then rotate
    pub fn apply_motion(&self, forward_m: f64, rotation_rad: f64) -> Self {
        Self {
            x: self.x + forward_m * self.heading.cos(),
            y: self.y + forward_m * self.heading.sin(),
            heading: self.heading + rotation_rad,
        }
    }

    /// Distance to another pose
    pub fn distance_to(&self, other: &Pose2D) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl Default for Pose2D {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

/// Scan observation: set of range measurements at given angles
#[derive(Debug, Clone)]
pub struct ScanObservation {
    /// Angles of each ray (radians)
    pub angles: Vec<f64>,
    /// Range measurement for each ray (meters, 0.0 = no return)
    pub ranges: Vec<f64>,
    /// Timestamp
    pub timestamp_ms: u64,
}

/// SLAM system combining localization with mapping
///
/// Implements a simple scan-matching SLAM approach where:
/// 1. Odometry provides initial pose estimate
/// 2. Scan matching refines the pose against the current map
/// 3. Map is updated with the corrected pose and scan data
#[derive(Debug)]
pub struct SlamSystem {
    /// Current estimated pose
    pub current_pose: Pose2D,
    /// Pose history
    pub pose_history: Vec<(u64, Pose2D)>,
    /// Environment mapper (occupancy grid)
    pub mapper: EnvironmentMapper,
    /// Pose uncertainty (meters)
    pub pose_uncertainty: f64,
    /// Maximum scan range (meters)
    max_range_m: f64,
}

impl SlamSystem {
    /// Creates a new SLAM system
    pub fn new(initial_pose: Pose2D, cell_size_m: f64, max_range_m: f64) -> Self {
        let origin = Vector3::new(
            initial_pose.x - max_range_m * 2.0,
            initial_pose.y - max_range_m * 2.0,
            0.0,
        );
        Self {
            current_pose: initial_pose,
            pose_history: Vec::new(),
            mapper: EnvironmentMapper::new(origin, cell_size_m),
            pose_uncertainty: 1.0,
            max_range_m,
        }
    }

    /// Processes odometry + scan observation to update pose and map
    pub fn update(
        &mut self,
        forward_m: f64,
        rotation_rad: f64,
        scan: &ScanObservation,
    ) {
        // 1. Predict pose from odometry
        let predicted_pose = self.current_pose.apply_motion(forward_m, rotation_rad);

        // 2. Scan matching: refine pose against current map
        let corrected_pose = self.scan_match(&predicted_pose, scan);

        // 3. Update pose
        self.current_pose = corrected_pose;
        self.pose_history.push((scan.timestamp_ms, corrected_pose));

        // 4. Update map with corrected pose and scan
        self.update_map_from_scan(&corrected_pose, scan);

        // 5. Update uncertainty (decreases with good scan matches)
        let correction_dist = predicted_pose.distance_to(&corrected_pose);
        self.pose_uncertainty = (self.pose_uncertainty * 0.95 + correction_dist * 0.1).max(0.1);
    }

    /// Simple scan matching by evaluating map consistency at nearby poses
    fn scan_match(&self, predicted: &Pose2D, scan: &ScanObservation) -> Pose2D {
        let mut best_pose = *predicted;
        let mut best_score = self.evaluate_scan_match(predicted, scan);

        // Search in a small neighborhood around the predicted pose
        let dx_steps = [-0.2, -0.1, 0.0, 0.1, 0.2];
        let dy_steps = [-0.2, -0.1, 0.0, 0.1, 0.2];
        let dh_steps = [-0.05, -0.02, 0.0, 0.02, 0.05];

        for &dx in &dx_steps {
            for &dy in &dy_steps {
                for &dh in &dh_steps {
                    let candidate = Pose2D::new(
                        predicted.x + dx,
                        predicted.y + dy,
                        predicted.heading + dh,
                    );
                    let score = self.evaluate_scan_match(&candidate, scan);
                    if score > best_score {
                        best_score = score;
                        best_pose = candidate;
                    }
                }
            }
        }

        best_pose
    }

    /// Evaluates how well a scan matches the current map at a given pose
    fn evaluate_scan_match(&self, pose: &Pose2D, scan: &ScanObservation) -> f64 {
        let mut score = 0.0;
        let mut count = 0;

        for (angle, range) in scan.angles.iter().zip(scan.ranges.iter()) {
            if *range <= 0.0 || *range >= self.max_range_m {
                continue;
            }

            let world_angle = pose.heading + angle;
            let endpoint = Vector3::new(
                pose.x + range * world_angle.cos(),
                pose.y + range * world_angle.sin(),
                0.0,
            );

            // Check if endpoint matches an occupied cell
            if let Some(cell) = self.mapper.grid().get_cell(&endpoint) {
                score += cell.occupancy * cell.confidence;
            }
            count += 1;
        }

        if count > 0 {
            score / count as f64
        } else {
            0.0
        }
    }

    /// Updates the map using a scan from a given pose
    fn update_map_from_scan(&mut self, pose: &Pose2D, scan: &ScanObservation) {
        for (angle, range) in scan.angles.iter().zip(scan.ranges.iter()) {
            if *range <= 0.0 || *range >= self.max_range_m {
                continue;
            }

            let world_angle = pose.heading + angle;

            // Mark cells along the ray as free
            let num_steps = (*range / self.mapper.grid().cell_size_m).ceil() as usize;
            for step in 0..num_steps {
                let t = step as f64 / num_steps as f64;
                let pos = Vector3::new(
                    pose.x + range * t * world_angle.cos(),
                    pose.y + range * t * world_angle.sin(),
                    0.0,
                );
                self.mapper.grid_mut().update_cell(&pos, false, scan.timestamp_ms);
            }

            // Mark endpoint as occupied
            let endpoint = Vector3::new(
                pose.x + range * world_angle.cos(),
                pose.y + range * world_angle.sin(),
                0.0,
            );
            self.mapper.grid_mut().update_cell(&endpoint, true, scan.timestamp_ms);
        }
    }

    /// Returns the current estimated pose
    pub fn pose(&self) -> &Pose2D {
        &self.current_pose
    }

    /// Returns pose uncertainty
    pub fn uncertainty(&self) -> f64 {
        self.pose_uncertainty
    }

    /// Returns pose history length
    pub fn pose_history_len(&self) -> usize {
        self.pose_history.len()
    }

    /// Returns the environment map
    pub fn map(&self) -> &EnvironmentMapper {
        &self.mapper
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_cell_update() {
        let mut cell = MapCell::new(Vector3::new(0.0, 0.0, 0.0));

        assert_eq!(cell.observation_count, 0);
        assert_eq!(cell.occupancy, 0.5);

        cell.update_occupancy(true, 1000);
        assert!(cell.occupancy > 0.5);
        assert_eq!(cell.observation_count, 1);

        cell.update_occupancy(false, 2000);
        assert_eq!(cell.observation_count, 2);
    }

    #[test]
    fn test_occupancy_grid() {
        let mut grid = OccupancyGrid::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

        let pos = Vector3::new(5.0, 5.0, 0.0);
        grid.update_cell(&pos, true, 1000);

        let cell = grid.get_cell(&pos);
        assert!(cell.is_some());
        assert_eq!(cell.unwrap().observation_count, 1);
        assert_eq!(grid.cell_count(), 1);
    }

    #[test]
    fn test_occupancy_grid_cleanup() {
        let mut grid = OccupancyGrid::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

        grid.update_cell(&Vector3::new(0.0, 0.0, 0.0), true, 1000);
        grid.update_cell(&Vector3::new(5.0, 5.0, 0.0), true, 10000);

        assert_eq!(grid.cell_count(), 2);

        grid.cleanup_old_cells(15000, 6000);
        assert_eq!(grid.cell_count(), 1);
    }

    #[test]
    fn test_environment_mapper() {
        let mut mapper = EnvironmentMapper::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

        let sensor_pos = Vector3::new(0.0, 0.0, 0.0);
        mapper.process_range_measurement(&sensor_pos, 10.0, 1000);

        assert!(mapper.grid().cell_count() > 0);
    }

    #[test]
    fn test_feature_detection() {
        let mut mapper = EnvironmentMapper::new(Vector3::new(0.0, 0.0, 0.0), 1.0);
        mapper.feature_threshold = 0.6; // Lower threshold for testing

        let sensor_pos = Vector3::new(0.0, 0.0, 0.0);

        // Multiple measurements at the same point to trigger feature detection
        // Need at least 71 observations to get confidence > 0.7
        for _ in 0..80 {
            mapper.process_range_measurement(&sensor_pos, 10.0, 1000);
        }

        assert!(mapper.feature_count() > 0);
    }

    #[test]
    fn test_feature_update() {
        let mut mapper = EnvironmentMapper::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

        let pos = Vector3::new(10.0, 0.0, 0.0);
        mapper.add_feature(pos, FeatureType::Reflector, 0.9, 1000);
        assert_eq!(mapper.feature_count(), 1);

        // Add feature nearby (should update existing)
        let nearby_pos = Vector3::new(10.5, 0.0, 0.0);
        mapper.add_feature(nearby_pos, FeatureType::Reflector, 0.95, 2000);
        assert_eq!(mapper.feature_count(), 1); // Should still be 1

        // Add feature far away (should create new)
        let far_pos = Vector3::new(20.0, 0.0, 0.0);
        mapper.add_feature(far_pos, FeatureType::Wall, 0.8, 3000);
        assert_eq!(mapper.feature_count(), 2);
    }

    #[test]
    fn test_cleanup() {
        let mut mapper = EnvironmentMapper::new(Vector3::new(0.0, 0.0, 0.0), 1.0);

        mapper.add_feature(Vector3::new(0.0, 0.0, 0.0), FeatureType::Reflector, 0.9, 1000);
        mapper.add_feature(Vector3::new(10.0, 0.0, 0.0), FeatureType::Wall, 0.8, 10000);

        assert_eq!(mapper.feature_count(), 2);

        mapper.cleanup(15000, 6000);
        assert_eq!(mapper.feature_count(), 1); // Only recent feature remains
    }

    // ── SLAM tests ────────────────────────────────────────────────────────

    #[test]
    fn test_pose2d() {
        let pose = Pose2D::new(0.0, 0.0, 0.0);
        let moved = pose.apply_motion(1.0, 0.0);
        assert!((moved.x - 1.0).abs() < 0.01);
        assert!(moved.y.abs() < 0.01);
    }

    #[test]
    fn test_pose2d_rotation() {
        let pose = Pose2D::new(0.0, 0.0, std::f64::consts::FRAC_PI_2);
        let moved = pose.apply_motion(1.0, 0.0);
        assert!(moved.x.abs() < 0.01);
        assert!((moved.y - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_slam_creation() {
        let slam = SlamSystem::new(Pose2D::default(), 0.5, 50.0);
        assert!((slam.pose().x).abs() < 0.01);
        assert_eq!(slam.pose_history_len(), 0);
    }

    #[test]
    fn test_slam_update() {
        let mut slam = SlamSystem::new(Pose2D::default(), 0.5, 50.0);

        let scan = ScanObservation {
            angles: vec![0.0, std::f64::consts::FRAC_PI_2],
            ranges: vec![5.0, 10.0],
            timestamp_ms: 1000,
        };

        slam.update(1.0, 0.0, &scan);
        assert_eq!(slam.pose_history_len(), 1);
        assert!(slam.pose().x > 0.0); // Should have moved forward
    }

    #[test]
    fn test_slam_map_update() {
        let mut slam = SlamSystem::new(Pose2D::default(), 1.0, 50.0);

        let scan = ScanObservation {
            angles: vec![0.0],
            ranges: vec![5.0],
            timestamp_ms: 1000,
        };

        slam.update(0.0, 0.0, &scan);
        assert!(slam.map().grid().cell_count() > 0);
    }
}
