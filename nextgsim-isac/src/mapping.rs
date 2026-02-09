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

    /// Clears old cells (older than max_age_ms)
    pub fn cleanup_old_cells(&mut self, current_time_ms: u64, max_age_ms: u64) {
        self.cells.retain(|_, cell| {
            current_time_ms - cell.last_update_ms < max_age_ms
        });
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
}
