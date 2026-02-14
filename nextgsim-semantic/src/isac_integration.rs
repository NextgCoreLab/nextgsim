//! Integration with ISAC for sensing data compression
//!
//! This module provides semantic compression specifically optimized for
//! ISAC sensing data (measurements, position data, environment maps).

#![allow(missing_docs)]

use crate::SemanticFeatures;
use serde::{Deserialize, Serialize};

/// Sensing data compression request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingCompressionRequest {
    /// Request ID
    pub request_id: u64,
    /// Sensing data to compress
    pub sensing_data: SensingDataPayload,
    /// Compression parameters
    pub compression_params: SensingCompressionParams,
}

/// Sensing data payload types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SensingDataPayload {
    /// Raw measurements (`ToA`, RSS, Doppler, etc.)
    Measurements {
        measurements: Vec<f32>,
        metadata: MeasurementMetadata,
    },
    /// Position/tracking data
    PositionData {
        positions: Vec<Position3D>,
        velocities: Vec<Velocity3D>,
        timestamps_ms: Vec<u64>,
    },
    /// Environment map data
    EnvironmentMap {
        occupancy_grid: Vec<f32>,
        grid_dimensions: Vec<usize>,
        features: Vec<MapFeatureData>,
    },
    /// Radar waveform data
    RadarWaveform {
        iq_samples: Vec<f32>,
        sample_rate_hz: f64,
        carrier_freq_hz: f64,
    },
}

/// Metadata for sensing measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementMetadata {
    /// Measurement types (e.g., `ToA`, RSS, Doppler)
    pub measurement_types: Vec<String>,
    /// Anchor/cell IDs
    pub anchor_ids: Vec<i32>,
    /// Uncertainties
    pub uncertainties: Vec<f32>,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
}

/// 3D position
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Position3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// 3D velocity
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Velocity3D {
    pub vx: f32,
    pub vy: f32,
    pub vz: f32,
}

/// Map feature data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapFeatureData {
    /// Feature type (e.g., "wall", "obstacle", "reflector")
    pub feature_type: String,
    /// Position
    pub position: Position3D,
    /// Confidence (0.0 to 1.0)
    pub confidence: f32,
}

/// Compression parameters for sensing data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingCompressionParams {
    /// Target compression ratio
    pub target_compression: f32,
    /// Preserve accuracy for specific measurement types
    pub preserve_types: Vec<String>,
    /// Minimum acceptable accuracy (meters for position, dB for RSS, etc.)
    pub min_accuracy: f32,
    /// Enable lossy compression
    pub allow_lossy: bool,
}

impl Default for SensingCompressionParams {
    fn default() -> Self {
        Self {
            target_compression: 5.0,
            preserve_types: Vec::new(),
            min_accuracy: 1.0,
            allow_lossy: true,
        }
    }
}

/// Compressed sensing data result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedSensingData {
    /// Request ID
    pub request_id: u64,
    /// Compressed features
    pub features: SemanticFeatures,
    /// Compression statistics
    pub compression_stats: CompressionStats,
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStats {
    /// Achieved compression ratio
    pub compression_ratio: f32,
    /// Original size (bytes)
    pub original_size_bytes: usize,
    /// Compressed size (bytes)
    pub compressed_size_bytes: usize,
    /// Estimated reconstruction error
    pub estimated_error: f32,
    /// Processing time (milliseconds)
    pub processing_time_ms: u32,
}

/// Semantic compressor for ISAC sensing data
#[derive(Debug)]
pub struct IsacSemanticCompressor {
    next_request_id: u64,
}

impl IsacSemanticCompressor {
    /// Creates a new ISAC semantic compressor
    pub fn new() -> Self {
        Self {
            next_request_id: 1,
        }
    }

    /// Compresses sensing measurements
    ///
    /// # Arguments
    ///
    /// * `measurements` - Raw measurement values
    /// * `metadata` - Measurement metadata
    /// * `params` - Compression parameters
    ///
    /// # Returns
    ///
    /// Compressed sensing data result
    pub fn compress_measurements(
        &mut self,
        measurements: Vec<f32>,
        _metadata: MeasurementMetadata,
        params: SensingCompressionParams,
    ) -> CompressedSensingData {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let original_size_bytes = measurements.len() * std::mem::size_of::<f32>();

        // Simulate compression: use mean pooling for simplicity
        let num_features = (measurements.len() as f32 / params.target_compression) as usize;
        let num_features = num_features.max(1);

        let mut compressed = Vec::with_capacity(num_features);
        let chunk_size = measurements.len() / num_features;

        for i in 0..num_features {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(measurements.len());
            let chunk_mean: f32 = measurements[start..end].iter().sum::<f32>() / (end - start) as f32;
            compressed.push(chunk_mean);
        }

        let compressed_size_bytes = compressed.len() * std::mem::size_of::<f32>();
        let actual_compression = original_size_bytes as f32 / compressed_size_bytes as f32;

        let features = SemanticFeatures::new(1, compressed, vec![measurements.len()]);

        CompressedSensingData {
            request_id,
            features,
            compression_stats: CompressionStats {
                compression_ratio: actual_compression,
                original_size_bytes,
                compressed_size_bytes,
                estimated_error: 0.1, // Simulated error
                processing_time_ms: 5,
            },
        }
    }

    /// Compresses position/tracking data
    pub fn compress_position_data(
        &mut self,
        positions: Vec<Position3D>,
        velocities: Vec<Velocity3D>,
        _timestamps_ms: Vec<u64>,
        params: SensingCompressionParams,
    ) -> CompressedSensingData {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        // Flatten position and velocity data
        let mut data = Vec::new();
        for pos in &positions {
            data.push(pos.x);
            data.push(pos.y);
            data.push(pos.z);
        }
        for vel in &velocities {
            data.push(vel.vx);
            data.push(vel.vy);
            data.push(vel.vz);
        }

        let original_size_bytes = data.len() * std::mem::size_of::<f32>();

        // Simple compression: keep every Nth point
        let stride = params.target_compression as usize;
        let compressed: Vec<f32> = data.iter().step_by(stride).copied().collect();

        let compressed_size_bytes = compressed.len() * std::mem::size_of::<f32>();
        let actual_compression = original_size_bytes as f32 / compressed_size_bytes as f32;

        let features = SemanticFeatures::new(2, compressed, vec![data.len()]);

        CompressedSensingData {
            request_id,
            features,
            compression_stats: CompressionStats {
                compression_ratio: actual_compression,
                original_size_bytes,
                compressed_size_bytes,
                estimated_error: 0.5, // Simulated error in meters
                processing_time_ms: 3,
            },
        }
    }

    /// Compresses environment map data
    pub fn compress_environment_map(
        &mut self,
        occupancy_grid: Vec<f32>,
        grid_dimensions: Vec<usize>,
        _params: SensingCompressionParams,
    ) -> CompressedSensingData {
        let request_id = self.next_request_id;
        self.next_request_id += 1;

        let original_size_bytes = occupancy_grid.len() * std::mem::size_of::<f32>();

        // Compress using threshold-based sparsification
        let threshold = 0.5; // Only keep cells with occupancy > 0.5
        let compressed: Vec<f32> = occupancy_grid
            .iter()
            .filter(|&&v| v > threshold)
            .copied()
            .collect();

        let compressed_size_bytes = compressed.len() * std::mem::size_of::<f32>();
        let actual_compression = original_size_bytes as f32 / compressed_size_bytes.max(1) as f32;

        let features = SemanticFeatures::new(3, compressed, grid_dimensions);

        CompressedSensingData {
            request_id,
            features,
            compression_stats: CompressionStats {
                compression_ratio: actual_compression,
                original_size_bytes,
                compressed_size_bytes,
                estimated_error: 0.05, // 5% occupancy error
                processing_time_ms: 8,
            },
        }
    }

    /// Creates a compression request
    pub fn create_request(
        &self,
        sensing_data: SensingDataPayload,
        params: SensingCompressionParams,
    ) -> SensingCompressionRequest {
        SensingCompressionRequest {
            request_id: self.next_request_id,
            sensing_data,
            compression_params: params,
        }
    }
}

impl Default for IsacSemanticCompressor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isac_semantic_compressor_creation() {
        let compressor = IsacSemanticCompressor::new();
        assert_eq!(compressor.next_request_id, 1);
    }

    #[test]
    fn test_compress_measurements() {
        let mut compressor = IsacSemanticCompressor::new();

        let measurements: Vec<f32> = (0..100).map(|i| (i as f32) * 0.1).collect();
        let metadata = MeasurementMetadata {
            measurement_types: vec!["ToA".to_string(), "RSS".to_string()],
            anchor_ids: vec![1, 2, 3],
            uncertainties: vec![5.0, 2.0, 3.0],
            timestamp_ms: 1000,
        };

        let params = SensingCompressionParams {
            target_compression: 5.0,
            preserve_types: vec!["ToA".to_string()],
            min_accuracy: 1.0,
            allow_lossy: true,
        };

        let result = compressor.compress_measurements(measurements, metadata, params);

        assert_eq!(result.request_id, 1);
        assert!(result.compression_stats.compression_ratio > 1.0);
        assert!(result.features.num_features() > 0);
        assert_eq!(
            result.compression_stats.original_size_bytes,
            100 * std::mem::size_of::<f32>()
        );
    }

    #[test]
    fn test_compress_position_data() {
        let mut compressor = IsacSemanticCompressor::new();

        let positions = vec![
            Position3D {
                x: 0.0,
                y: 0.0,
                z: 0.0,
            },
            Position3D {
                x: 10.0,
                y: 5.0,
                z: 2.0,
            },
            Position3D {
                x: 20.0,
                y: 10.0,
                z: 4.0,
            },
        ];

        let velocities = vec![
            Velocity3D {
                vx: 1.0,
                vy: 0.5,
                vz: 0.0,
            },
            Velocity3D {
                vx: 1.0,
                vy: 0.5,
                vz: 0.0,
            },
            Velocity3D {
                vx: 1.0,
                vy: 0.5,
                vz: 0.0,
            },
        ];

        let timestamps = vec![1000, 2000, 3000];

        let result = compressor.compress_position_data(
            positions,
            velocities,
            timestamps,
            SensingCompressionParams::default(),
        );

        assert_eq!(result.request_id, 1);
        assert!(result.compression_stats.compression_ratio > 1.0);
    }

    #[test]
    fn test_compress_environment_map() {
        let mut compressor = IsacSemanticCompressor::new();

        // Create a 10x10 occupancy grid
        let mut occupancy_grid: Vec<f32> = vec![0.0; 100];
        // Add some obstacles
        occupancy_grid[25] = 0.9;
        occupancy_grid[26] = 0.8;
        occupancy_grid[35] = 0.95;
        occupancy_grid[36] = 0.85;
        occupancy_grid[45] = 0.7;

        let grid_dimensions = vec![10, 10];

        let result =
            compressor.compress_environment_map(occupancy_grid, grid_dimensions, SensingCompressionParams::default());

        assert_eq!(result.request_id, 1);
        assert!(result.compression_stats.compression_ratio > 1.0);
        // Should have filtered out most zero-occupancy cells
        assert!(result.features.num_features() < 100);
    }

    #[test]
    fn test_compression_params_default() {
        let params = SensingCompressionParams::default();
        assert_eq!(params.target_compression, 5.0);
        assert_eq!(params.min_accuracy, 1.0);
        assert!(params.allow_lossy);
        assert!(params.preserve_types.is_empty());
    }

    #[test]
    fn test_request_id_increment() {
        let mut compressor = IsacSemanticCompressor::new();

        let measurements1 = vec![1.0, 2.0, 3.0];
        let metadata = MeasurementMetadata {
            measurement_types: vec!["ToA".to_string()],
            anchor_ids: vec![1],
            uncertainties: vec![1.0],
            timestamp_ms: 1000,
        };

        let result1 = compressor.compress_measurements(
            measurements1,
            metadata.clone(),
            SensingCompressionParams::default(),
        );

        let measurements2 = vec![4.0, 5.0, 6.0];
        let result2 = compressor.compress_measurements(
            measurements2,
            metadata,
            SensingCompressionParams::default(),
        );

        assert_eq!(result1.request_id, 1);
        assert_eq!(result2.request_id, 2);
    }
}
