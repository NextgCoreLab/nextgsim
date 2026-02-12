//! Integrated Sensing and Communication (ISAC) for 6G Networks
//!
//! Implements ISAC per 3GPP TR 22.837:
//! - RAN-level sensing data collection
//! - Core aggregation and fusion
//! - Edge inference for positioning/tracking
//! - OFDM-radar waveform modeling
//! - AI-native sensing with ML models
//! - Sensing-as-a-Service API
//! - Environment mapping (SLAM-like)
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
//! │  │  • Extended Kalman filtering                                     │   │
//! │  │  • Multi-sensor Bayesian fusion                                  │   │
//! │  │  • Trilateration via least-squares                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Positioning/Tracking                                             │   │
//! │  │  • Position estimation                                           │   │
//! │  │  • Velocity estimation                                           │   │
//! │  │  • Object tracking                                               │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ 3GPP TR 22.837 Sensing Use Cases                                 │   │
//! │  │  • Object detection                                              │   │
//! │  │  • Velocity estimation (Doppler)                                 │   │
//! │  │  • Range estimation (RTT)                                        │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! │  ┌─────────────────────────────────────────────────────────────────┐   │
//! │  │ Resource Management                                              │   │
//! │  │  • Sensing vs Communication resource split                       │   │
//! │  └─────────────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────────────┘
//! ```

pub mod mapping;
pub mod ml_sensing;
pub mod saas;
pub mod she_integration;
pub mod waveform;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

// Re-export modules
pub use mapping::{EnvironmentFeature, EnvironmentMapper, FeatureType, MapCell, OccupancyGrid};
pub use ml_sensing::{MlModelType, MlPositioningEngine, MlPositioningResult, TrajectoryPredictor};
pub use saas::{
    GeographicArea, SensingApiRequest, SensingApiResponse, SensingAsAService, SensingQos,
    SensingResult, SensingServiceType, SensingSubscription,
};
pub use she_integration::{
    DetectedObject, MapFeature, SensingProcessingResult, SensingProcessingType,
    SensingWorkloadRequest, SensingWorkloadResult, SheIsacClient,
};
pub use waveform::{
    BistaticGeometry, CfarDetection, CfarDetector, ClutterModel, OfdmRadarWaveform,
};

/// Speed of light in m/s.
const SPEED_OF_LIGHT: f64 = 299_792_458.0;

// ─── Core data types (backward-compatible) ────────────────────────────────────

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

// ─── 1. Trilateration via Iterative Least-Squares ─────────────────────────────

/// Configuration for the trilateration solver.
#[derive(Debug, Clone)]
pub struct TrilaterationConfig {
    /// Maximum number of Gauss-Newton iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on position update norm (meters).
    pub tolerance: f64,
}

impl Default for TrilaterationConfig {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
        }
    }
}

/// Result of a trilateration solve.
#[derive(Debug, Clone)]
pub struct TrilaterationResult {
    /// Estimated position.
    pub position: Vector3,
    /// Residual norm (sum of squared range residuals after convergence).
    pub residual_norm: f64,
    /// Number of iterations taken.
    pub iterations: usize,
    /// Whether the solver converged within tolerance.
    pub converged: bool,
}

/// Solves for a 3-D position given range measurements from known anchor
/// positions using iterative weighted least-squares (Gauss-Newton).
///
/// Each entry in `anchor_positions` is paired with the corresponding entry in
/// `ranges` (measured distance in meters) and `weights` (inverse variance,
/// i.e. `1 / sigma^2`).  At least 3 anchor/range pairs are required for a
/// 3-D fix (4 for full observability but the solver handles rank deficiency
/// gracefully).
///
/// The initial guess is the weighted centroid of the anchors which is then
/// refined iteratively.
///
/// # Errors
///
/// Returns `None` when fewer than 3 measurements are provided or when the
/// normal-equation matrix is singular (collinear anchors in the same plane
/// for 3-D, etc.).
pub fn trilaterate(
    anchor_positions: &[Vector3],
    ranges: &[f64],
    weights: &[f64],
    config: &TrilaterationConfig,
) -> Option<TrilaterationResult> {
    let n = anchor_positions.len();
    if n < 3 || ranges.len() != n || weights.len() != n {
        return None;
    }

    // Build weight vector (clamp to avoid zero / negative)
    let w: Vec<f64> = weights.iter().map(|&wi| wi.max(1e-12)).collect();

    // Initial guess: weighted centroid of anchors (weighted by measurement weight)
    let total_w: f64 = w.iter().sum();
    let mut x = w
        .iter()
        .zip(anchor_positions.iter())
        .fold([0.0; 3], |mut acc, (&wi, a)| {
            acc[0] += wi * a.x;
            acc[1] += wi * a.y;
            acc[2] += wi * a.z;
            acc
        });
    x[0] /= total_w;
    x[1] /= total_w;
    x[2] /= total_w;

    let mut iterations = 0;
    let mut converged = false;

    for _iter in 0..config.max_iterations {
        iterations = _iter + 1;

        // Jacobian J (n x 3) and residual vector r (n)
        let mut j = Array2::<f64>::zeros((n, 3));
        let mut r = Array1::<f64>::zeros(n);
        let mut w_diag = Array2::<f64>::zeros((n, n));

        for i in 0..n {
            let dx = x[0] - anchor_positions[i].x;
            let dy = x[1] - anchor_positions[i].y;
            let dz = x[2] - anchor_positions[i].z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);

            // Partial derivatives of range w.r.t. position
            j[[i, 0]] = dx / dist;
            j[[i, 1]] = dy / dist;
            j[[i, 2]] = dz / dist;

            // Residual: predicted range minus measured range
            r[i] = dist - ranges[i];

            w_diag[[i, i]] = w[i];
        }

        // Normal equations: (J^T W J + lambda*I) delta = -J^T W r
        // Levenberg-Marquardt damping handles rank-deficient geometry
        // (e.g. all anchors coplanar with target -> z column is zero).
        let jt = j.t();
        let jtw = jt.dot(&w_diag);
        let mut jtw_j = jtw.dot(&j); // 3x3
        let jtw_r = jtw.dot(&r); // 3

        // Add damping: lambda = small fraction of diagonal average
        let diag_avg = (jtw_j[[0, 0]] + jtw_j[[1, 1]] + jtw_j[[2, 2]]) / 3.0;
        let lambda = (diag_avg * 1e-6).max(1e-10);
        jtw_j[[0, 0]] += lambda;
        jtw_j[[1, 1]] += lambda;
        jtw_j[[2, 2]] += lambda;

        // Solve 3x3 system via explicit inverse (Cramer's rule)
        let delta = solve_3x3(&jtw_j, &(-&jtw_r))?;

        x[0] += delta[0];
        x[1] += delta[1];
        x[2] += delta[2];

        let step_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if step_norm < config.tolerance {
            converged = true;
            break;
        }
    }

    // Compute final residual norm
    let residual_norm: f64 = anchor_positions
        .iter()
        .zip(ranges.iter())
        .zip(w.iter())
        .map(|((a, &ri), &wi)| {
            let d = Vector3::new(x[0], x[1], x[2]).distance_to(a);
            wi * (d - ri).powi(2)
        })
        .sum();

    Some(TrilaterationResult {
        position: Vector3::new(x[0], x[1], x[2]),
        residual_norm,
        iterations,
        converged,
    })
}

/// Solves a 3x3 linear system `A * x = b` using Cramer's rule.
/// Returns `None` if the determinant is near-zero (singular matrix).
fn solve_3x3(a: &Array2<f64>, b: &Array1<f64>) -> Option<[f64; 3]> {
    let det = a[[0, 0]] * (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]])
        - a[[0, 1]] * (a[[1, 0]] * a[[2, 2]] - a[[1, 2]] * a[[2, 0]])
        + a[[0, 2]] * (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]);

    if det.abs() < 1e-30 {
        return None;
    }

    let inv_det = 1.0 / det;

    // Cofactor matrix transpose (adjugate) row by row
    let inv00 = (a[[1, 1]] * a[[2, 2]] - a[[1, 2]] * a[[2, 1]]) * inv_det;
    let inv01 = (a[[0, 2]] * a[[2, 1]] - a[[0, 1]] * a[[2, 2]]) * inv_det;
    let inv02 = (a[[0, 1]] * a[[1, 2]] - a[[0, 2]] * a[[1, 1]]) * inv_det;
    let inv10 = (a[[1, 2]] * a[[2, 0]] - a[[1, 0]] * a[[2, 2]]) * inv_det;
    let inv11 = (a[[0, 0]] * a[[2, 2]] - a[[0, 2]] * a[[2, 0]]) * inv_det;
    let inv12 = (a[[0, 2]] * a[[1, 0]] - a[[0, 0]] * a[[1, 2]]) * inv_det;
    let inv20 = (a[[1, 0]] * a[[2, 1]] - a[[1, 1]] * a[[2, 0]]) * inv_det;
    let inv21 = (a[[0, 1]] * a[[2, 0]] - a[[0, 0]] * a[[2, 1]]) * inv_det;
    let inv22 = (a[[0, 0]] * a[[1, 1]] - a[[0, 1]] * a[[1, 0]]) * inv_det;

    Some([
        inv00 * b[0] + inv01 * b[1] + inv02 * b[2],
        inv10 * b[0] + inv11 * b[1] + inv12 * b[2],
        inv20 * b[0] + inv21 * b[1] + inv22 * b[2],
    ])
}

// ─── TDoA Solver ───────────────────────────────────────────────────────────────

/// `TDoA` (Time Difference of Arrival) positioning result
#[derive(Debug, Clone)]
pub struct TdoaResult {
    /// Estimated position
    pub position: Vector3,
    /// Residual norm
    pub residual_norm: f64,
    /// Number of iterations
    pub iterations: usize,
    /// Converged flag
    pub converged: bool,
}

/// Solves for position using `TDoA` measurements
///
/// `TDoA` measures the difference in arrival times between pairs of anchors.
/// This is equivalent to hyperbolic positioning.
///
/// # Arguments
/// * `anchor_positions` - Positions of the anchors (first is reference)
/// * `tdoa_measurements` - Time differences relative to first anchor (seconds)
/// * `weights` - Measurement weights (inverse variance)
pub fn solve_tdoa(
    anchor_positions: &[Vector3],
    tdoa_measurements: &[f64],
    weights: &[f64],
) -> Option<TdoaResult> {
    if anchor_positions.len() < 4 || tdoa_measurements.len() != anchor_positions.len() - 1 {
        return None;
    }

    let c = SPEED_OF_LIGHT;

    // Convert TDoA to range differences
    let range_diffs: Vec<f64> = tdoa_measurements.iter().map(|&td| td * c).collect();

    // Use iterative solver similar to trilateration
    // Initial guess: centroid of anchors
    let n = anchor_positions.len();
    let mut x = [0.0; 3];
    for anchor in anchor_positions {
        x[0] += anchor.x;
        x[1] += anchor.y;
        x[2] += anchor.z;
    }
    x[0] /= n as f64;
    x[1] /= n as f64;
    x[2] /= n as f64;

    let max_iterations = 50;
    let tolerance = 1e-6;
    let mut iterations = 0;
    let mut converged = false;

    let ref_anchor = &anchor_positions[0];

    for _iter in 0..max_iterations {
        iterations = _iter + 1;

        // Build Jacobian and residuals
        let m = anchor_positions.len() - 1;
        let mut j = Array2::<f64>::zeros((m, 3));
        let mut r = Array1::<f64>::zeros(m);

        let dx0 = x[0] - ref_anchor.x;
        let dy0 = x[1] - ref_anchor.y;
        let dz0 = x[2] - ref_anchor.z;
        let r0 = (dx0 * dx0 + dy0 * dy0 + dz0 * dz0).sqrt().max(1e-12);

        for i in 0..m {
            let anchor = &anchor_positions[i + 1];
            let dx = x[0] - anchor.x;
            let dy = x[1] - anchor.y;
            let dz = x[2] - anchor.z;
            let ri = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);

            // Jacobian: d(ri - r0)/dx
            j[[i, 0]] = dx / ri - dx0 / r0;
            j[[i, 1]] = dy / ri - dy0 / r0;
            j[[i, 2]] = dz / ri - dz0 / r0;

            // Residual
            r[i] = (ri - r0) - range_diffs[i];
        }

        // Weighted least-squares update
        let mut w_diag = Array2::<f64>::zeros((m, m));
        for i in 0..m {
            w_diag[[i, i]] = weights[i.min(weights.len() - 1)];
        }

        let jt = j.t();
        let jtw = jt.dot(&w_diag);
        let mut jtw_j = jtw.dot(&j);
        let jtw_r = jtw.dot(&r);

        // Add damping
        let diag_avg = (jtw_j[[0, 0]] + jtw_j[[1, 1]] + jtw_j[[2, 2]]) / 3.0;
        let lambda = (diag_avg * 1e-6).max(1e-10);
        jtw_j[[0, 0]] += lambda;
        jtw_j[[1, 1]] += lambda;
        jtw_j[[2, 2]] += lambda;

        let delta = solve_3x3(&jtw_j, &(-&jtw_r))?;

        x[0] += delta[0];
        x[1] += delta[1];
        x[2] += delta[2];

        let step_norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
        if step_norm < tolerance {
            converged = true;
            break;
        }
    }

    // Compute residual norm
    let ref_range = Vector3::new(x[0], x[1], x[2]).distance_to(ref_anchor);
    let residual_norm: f64 = anchor_positions[1..]
        .iter()
        .zip(range_diffs.iter())
        .map(|(a, &rd)| {
            let ri = Vector3::new(x[0], x[1], x[2]).distance_to(a);
            (ri - ref_range - rd).powi(2)
        })
        .sum::<f64>()
        .sqrt();

    Some(TdoaResult {
        position: Vector3::new(x[0], x[1], x[2]),
        residual_norm,
        iterations,
        converged,
    })
}

/// Backward-compatible wrapper: simple position fusion using trilateration.
///
/// Accepts `ToA`/`Rtt`/`Rss` measurements and known anchor positions, converts
/// them to range estimates, then calls the iterative least-squares
/// trilateration solver.
pub fn fuse_positions(
    measurements: &[SensingMeasurement],
    anchors: &HashMap<i32, Vector3>,
) -> Option<(Vector3, f64)> {
    let mut anchor_pos = Vec::new();
    let mut ranges = Vec::new();
    let mut weights = Vec::new();

    for measurement in measurements {
        if let Some(anchor) = anchors.get(&measurement.anchor_id) {
            let range = match measurement.measurement_type {
                SensingType::ToA | SensingType::Rtt => measurement.value,
                SensingType::Rss => {
                    // Path loss model: RSS_dBm = -40 - 20*log10(distance)
                    let rss = measurement.value;
                    10.0_f64.powf((-40.0 - rss) / 20.0)
                }
                _ => continue,
            };

            anchor_pos.push(*anchor);
            ranges.push(range);
            // Weight = 1 / variance = 1 / sigma^2
            let sigma = measurement.uncertainty.max(0.1);
            weights.push(1.0 / (sigma * sigma));
        }
    }

    if anchor_pos.len() < 3 {
        return None;
    }

    let result = trilaterate(
        &anchor_pos,
        &ranges,
        &weights,
        &TrilaterationConfig::default(),
    )?;

    // Uncertainty estimate: geometric mean of measurement sigmas scaled by
    // residual.  Provides a reasonable single-number uncertainty.
    let avg_sigma: f64 = measurements.iter().map(|m| m.uncertainty).sum::<f64>()
        / measurements.len() as f64;
    let uncertainty = avg_sigma * (1.0 + result.residual_norm.sqrt());

    Some((result.position, uncertainty))
}

// ─── 2. Extended Kalman Filter (EKF) ──────────────────────────────────────────

/// State indices for the EKF 6-state vector `[x, y, z, vx, vy, vz]`.
const STATE_DIM: usize = 6;

/// Extended Kalman Filter for 3-D position and velocity tracking.
///
/// State vector: `[x, y, z, vx, vy, vz]`
/// Process model: constant-velocity with additive acceleration noise.
/// Measurement model: range from known anchor (non-linear).
#[derive(Debug, Clone)]
pub struct ExtendedKalmanFilter {
    /// State estimate `[x, y, z, vx, vy, vz]`.
    pub state: Array1<f64>,
    /// State covariance matrix (6x6).
    pub covariance: Array2<f64>,
    /// Process noise spectral density for acceleration (m/s^2)^2.
    /// Larger values allow faster manoeuvring.
    pub process_noise_accel: f64,
    /// Whether the filter has been initialised with a measurement.
    initialised: bool,
}

impl ExtendedKalmanFilter {
    /// Creates a new EKF with a given initial position and velocity.
    ///
    /// `initial_pos_uncertainty` is the 1-sigma position uncertainty (m).
    /// `initial_vel_uncertainty` is the 1-sigma velocity uncertainty (m/s).
    /// `process_noise_accel` is the power spectral density of the acceleration
    /// noise process (typical: 0.1 .. 5.0 for pedestrian .. vehicle).
    pub fn new(
        initial_position: Vector3,
        initial_velocity: Vector3,
        initial_pos_uncertainty: f64,
        initial_vel_uncertainty: f64,
        process_noise_accel: f64,
    ) -> Self {
        let state = Array1::from(vec![
            initial_position.x,
            initial_position.y,
            initial_position.z,
            initial_velocity.x,
            initial_velocity.y,
            initial_velocity.z,
        ]);

        let mut p = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
        let pos_var = initial_pos_uncertainty * initial_pos_uncertainty;
        let vel_var = initial_vel_uncertainty * initial_vel_uncertainty;
        p[[0, 0]] = pos_var;
        p[[1, 1]] = pos_var;
        p[[2, 2]] = pos_var;
        p[[3, 3]] = vel_var;
        p[[4, 4]] = vel_var;
        p[[5, 5]] = vel_var;

        Self {
            state,
            covariance: p,
            process_noise_accel,
            initialised: true,
        }
    }

    /// Creates an uninitialised EKF; it will self-initialise on the first
    /// position measurement.
    pub fn uninitialised(process_noise_accel: f64) -> Self {
        Self {
            state: Array1::zeros(STATE_DIM),
            covariance: Array2::eye(STATE_DIM) * 100.0,
            process_noise_accel,
            initialised: false,
        }
    }

    /// Returns whether the filter has received at least one measurement.
    pub fn is_initialised(&self) -> bool {
        self.initialised
    }

    /// Returns the current position estimate.
    pub fn position(&self) -> Vector3 {
        Vector3::new(self.state[0], self.state[1], self.state[2])
    }

    /// Returns the current velocity estimate.
    pub fn velocity(&self) -> Vector3 {
        Vector3::new(self.state[3], self.state[4], self.state[5])
    }

    /// Returns the position uncertainty (square root of the trace of the
    /// position block of the covariance matrix).
    pub fn position_uncertainty(&self) -> f64 {
        (self.covariance[[0, 0]] + self.covariance[[1, 1]] + self.covariance[[2, 2]]).sqrt()
    }

    /// Returns the velocity uncertainty.
    pub fn velocity_uncertainty(&self) -> f64 {
        (self.covariance[[3, 3]] + self.covariance[[4, 4]] + self.covariance[[5, 5]]).sqrt()
    }

    // ── Prediction step ──────────────────────────────────────────────────

    /// Constructs the state transition matrix F for a constant-velocity model:
    ///
    /// ```text
    /// F = | I   dt*I |
    ///     | 0    I   |
    /// ```
    fn state_transition(dt: f64) -> Array2<f64> {
        let mut f = Array2::<f64>::eye(STATE_DIM);
        f[[0, 3]] = dt;
        f[[1, 4]] = dt;
        f[[2, 5]] = dt;
        f
    }

    /// Constructs the process noise covariance matrix Q for a piece-wise
    /// constant white-noise jerk model:
    ///
    /// ```text
    /// Q = q * | (dt^3/3)*I  (dt^2/2)*I |
    ///         | (dt^2/2)*I   dt*I       |
    /// ```
    ///
    /// where `q = process_noise_accel`.
    fn process_noise(&self, dt: f64) -> Array2<f64> {
        let q = self.process_noise_accel;
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;

        let mut qmat = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
        for i in 0..3 {
            qmat[[i, i]] = q * dt3 / 3.0;
            qmat[[i, i + 3]] = q * dt2 / 2.0;
            qmat[[i + 3, i]] = q * dt2 / 2.0;
            qmat[[i + 3, i + 3]] = q * dt;
        }
        qmat
    }

    /// Predict step: propagate the state and covariance forward by `dt`
    /// seconds using the constant-velocity motion model.
    pub fn predict(&mut self, dt: f64) {
        if dt <= 0.0 {
            return;
        }
        let f = Self::state_transition(dt);
        let q = self.process_noise(dt);
        self.state = f.dot(&self.state);
        self.covariance = f.dot(&self.covariance).dot(&f.t()) + q;
    }

    // ── Update steps ─────────────────────────────────────────────────────

    /// Update with a range measurement from a known anchor.
    ///
    /// The measurement model is:
    ///   `h(x) = || pos - anchor ||`
    /// which is non-linear.  The Jacobian is computed analytically.
    ///
    /// `range` is the measured distance (m); `range_sigma` is the measurement
    /// standard deviation (m).
    pub fn update_range(&mut self, anchor: &Vector3, range: f64, range_sigma: f64) {
        if !self.initialised {
            // Initialise position from range intersection is hard with a single
            // measurement; just store a rough estimate at the anchor and mark
            // as initialised so subsequent measurements refine it.
            self.state[0] = anchor.x + range;
            self.state[1] = anchor.y;
            self.state[2] = anchor.z;
            self.initialised = true;
            return;
        }

        let dx = self.state[0] - anchor.x;
        let dy = self.state[1] - anchor.y;
        let dz = self.state[2] - anchor.z;
        let predicted_range = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);

        // Jacobian H (1 x 6): partial derivatives of range w.r.t. state
        let mut h = Array2::<f64>::zeros((1, STATE_DIM));
        h[[0, 0]] = dx / predicted_range;
        h[[0, 1]] = dy / predicted_range;
        h[[0, 2]] = dz / predicted_range;
        // Velocity components have zero partial derivative for range measurement

        let r_var = range_sigma * range_sigma;

        // Innovation
        let y = range - predicted_range;

        // Innovation covariance S = H P H^T + R
        let hp = h.dot(&self.covariance); // 1 x 6
        let s = hp.dot(&h.t())[[0, 0]] + r_var;

        if s.abs() < 1e-30 {
            return;
        }

        // Kalman gain K = P H^T / S   (6 x 1)
        let k = self.covariance.dot(&h.t()) / s;

        // State update
        for i in 0..STATE_DIM {
            self.state[i] += k[[i, 0]] * y;
        }

        // Covariance update (Joseph form for numerical stability):
        // P = (I - K H) P (I - K H)^T + K R K^T
        let eye = Array2::<f64>::eye(STATE_DIM);
        let kh = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * h[[0, j]];
                }
            }
            m
        };
        let i_kh = &eye - &kh;
        let kr_kt = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * r_var * k[[j, 0]];
                }
            }
            m
        };
        self.covariance = i_kh.dot(&self.covariance).dot(&i_kh.t()) + kr_kt;
    }

    /// Update with a direct position measurement (e.g. from trilateration or
    /// GNSS).  The measurement model is linear: `h(x) = [x, y, z]`.
    ///
    /// `pos_sigma` is the 1-sigma position uncertainty per axis.
    pub fn update_position(&mut self, measured: &Vector3, pos_sigma: f64) {
        if !self.initialised {
            self.state[0] = measured.x;
            self.state[1] = measured.y;
            self.state[2] = measured.z;
            self.initialised = true;
            return;
        }

        // H (3 x 6): position observation
        let mut h = Array2::<f64>::zeros((3, STATE_DIM));
        h[[0, 0]] = 1.0;
        h[[1, 1]] = 1.0;
        h[[2, 2]] = 1.0;

        let r_var = pos_sigma * pos_sigma;
        let mut r_mat = Array2::<f64>::zeros((3, 3));
        r_mat[[0, 0]] = r_var;
        r_mat[[1, 1]] = r_var;
        r_mat[[2, 2]] = r_var;

        // Innovation
        let z = Array1::from(vec![measured.x, measured.y, measured.z]);
        let z_pred = Array1::from(vec![self.state[0], self.state[1], self.state[2]]);
        let y = &z - &z_pred;

        // S = H P H^T + R  (3 x 3)
        let s = h.dot(&self.covariance).dot(&h.t()) + &r_mat;

        // Invert S (3x3)
        let s_inv = match invert_3x3(&s) {
            Some(inv) => inv,
            None => return, // Singular innovation covariance; skip update
        };

        // K = P H^T S^{-1}  (6 x 3)
        let k = self.covariance.dot(&h.t()).dot(&s_inv);

        // State update
        let dx = k.dot(&y);
        self.state = &self.state + &dx;

        // Covariance update (Joseph form)
        let eye = Array2::<f64>::eye(STATE_DIM);
        let kh = k.dot(&h);
        let i_kh = &eye - &kh;
        let kr_kt = k.dot(&r_mat).dot(&k.t());
        self.covariance = i_kh.dot(&self.covariance).dot(&i_kh.t()) + kr_kt;
    }

    /// Update with an angle-of-arrival measurement.
    ///
    /// `azimuth_rad` is the measured azimuth angle (radians) from the anchor
    /// to the target in the XY plane, measured counter-clockwise from the
    /// positive X axis.  `azimuth_sigma` is the measurement standard deviation
    /// (radians).
    pub fn update_aoa(&mut self, anchor: &Vector3, azimuth_rad: f64, azimuth_sigma: f64) {
        if !self.initialised {
            return; // Cannot initialise from a single AoA
        }

        let dx = self.state[0] - anchor.x;
        let dy = self.state[1] - anchor.y;
        let d_horiz_sq = dx * dx + dy * dy;
        let _d_horiz = d_horiz_sq.sqrt().max(1e-12);

        // Predicted azimuth
        let predicted_az = dy.atan2(dx);

        // Innovation (wrap to [-pi, pi])
        let mut innov = azimuth_rad - predicted_az;
        while innov > std::f64::consts::PI {
            innov -= 2.0 * std::f64::consts::PI;
        }
        while innov < -std::f64::consts::PI {
            innov += 2.0 * std::f64::consts::PI;
        }

        // Jacobian H (1 x 6): d(atan2(dy,dx))/d(state)
        let mut h = Array2::<f64>::zeros((1, STATE_DIM));
        h[[0, 0]] = -dy / d_horiz_sq;
        h[[0, 1]] = dx / d_horiz_sq;
        // z, vx, vy, vz components are zero

        let r_var = azimuth_sigma * azimuth_sigma;

        let hp = h.dot(&self.covariance);
        let s = hp.dot(&h.t())[[0, 0]] + r_var;
        if s.abs() < 1e-30 {
            return;
        }

        let k = self.covariance.dot(&h.t()) / s;

        for i in 0..STATE_DIM {
            self.state[i] += k[[i, 0]] * innov;
        }

        let eye = Array2::<f64>::eye(STATE_DIM);
        let kh = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * h[[0, j]];
                }
            }
            m
        };
        let i_kh = &eye - &kh;
        let kr_kt = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * r_var * k[[j, 0]];
                }
            }
            m
        };
        self.covariance = i_kh.dot(&self.covariance).dot(&i_kh.t()) + kr_kt;
    }

    /// Update with a `ZoA` (elevation angle) measurement.
    ///
    /// `elevation_rad` is the measured elevation angle (radians) from the anchor
    /// to the target, measured from the XY plane upwards.
    /// `elevation_sigma` is the measurement standard deviation (radians).
    pub fn update_zoa(&mut self, anchor: &Vector3, elevation_rad: f64, elevation_sigma: f64) {
        if !self.initialised {
            return; // Cannot initialise from a single angle
        }

        let dx = self.state[0] - anchor.x;
        let dy = self.state[1] - anchor.y;
        let dz = self.state[2] - anchor.z;
        let d_horiz = (dx * dx + dy * dy).sqrt().max(1e-12);

        // Predicted elevation
        let predicted_elev = (dz / d_horiz).atan();

        // Innovation
        let mut innov = elevation_rad - predicted_elev;
        // Wrap to [-pi/2, pi/2]
        while innov > std::f64::consts::FRAC_PI_2 {
            innov -= std::f64::consts::PI;
        }
        while innov < -std::f64::consts::FRAC_PI_2 {
            innov += std::f64::consts::PI;
        }

        // Jacobian H (1 x 6): d(atan(dz/d_horiz))/d(state)
        let mut h = Array2::<f64>::zeros((1, STATE_DIM));
        let d_horiz_sq = d_horiz * d_horiz;
        let denom = 1.0 + (dz / d_horiz).powi(2);

        h[[0, 0]] = (-dz * dx) / (d_horiz_sq * d_horiz * denom);
        h[[0, 1]] = (-dz * dy) / (d_horiz_sq * d_horiz * denom);
        h[[0, 2]] = 1.0 / (d_horiz * denom);

        let r_var = elevation_sigma * elevation_sigma;

        let hp = h.dot(&self.covariance);
        let s = hp.dot(&h.t())[[0, 0]] + r_var;
        if s.abs() < 1e-30 {
            return;
        }

        let k = self.covariance.dot(&h.t()) / s;

        for i in 0..STATE_DIM {
            self.state[i] += k[[i, 0]] * innov;
        }

        let eye = Array2::<f64>::eye(STATE_DIM);
        let kh = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * h[[0, j]];
                }
            }
            m
        };
        let i_kh = &eye - &kh;
        let kr_kt = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * r_var * k[[j, 0]];
                }
            }
            m
        };
        self.covariance = i_kh.dot(&self.covariance).dot(&i_kh.t()) + kr_kt;
    }

    /// Update with a Doppler shift measurement (direct velocity observation).
    ///
    /// `anchor` is the position of the measuring anchor.
    /// `doppler_shift_hz` is the measured Doppler frequency shift.
    /// `carrier_freq_hz` is the carrier frequency.
    /// `doppler_sigma_hz` is the measurement standard deviation.
    pub fn update_doppler(
        &mut self,
        anchor: &Vector3,
        doppler_shift_hz: f64,
        carrier_freq_hz: f64,
        doppler_sigma_hz: f64,
    ) {
        if !self.initialised {
            return; // Cannot initialise from Doppler alone
        }

        let dx = self.state[0] - anchor.x;
        let dy = self.state[1] - anchor.y;
        let dz = self.state[2] - anchor.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);

        // Unit vector from anchor to target
        let ux = dx / dist;
        let uy = dy / dist;
        let uz = dz / dist;

        // Radial velocity: v_r = v . u
        let predicted_vr = self.state[3] * ux + self.state[4] * uy + self.state[5] * uz;

        // Convert to Doppler
        let lambda = SPEED_OF_LIGHT / carrier_freq_hz;
        let predicted_doppler = predicted_vr / lambda;

        // Innovation
        let innov = doppler_shift_hz - predicted_doppler;

        // Jacobian H (1 x 6): d(doppler)/d(state)
        // Position derivatives (from change in unit vector)
        // Simplified: assume velocity changes dominate
        let mut h = Array2::<f64>::zeros((1, STATE_DIM));
        h[[0, 0]] = 0.0; // Position has indirect effect (ignored for simplicity)
        h[[0, 1]] = 0.0;
        h[[0, 2]] = 0.0;
        h[[0, 3]] = ux / lambda;
        h[[0, 4]] = uy / lambda;
        h[[0, 5]] = uz / lambda;

        let r_var = (doppler_sigma_hz * lambda).powi(2); // Convert to velocity variance

        let hp = h.dot(&self.covariance);
        let s = hp.dot(&h.t())[[0, 0]] + r_var;
        if s.abs() < 1e-30 {
            return;
        }

        let k = self.covariance.dot(&h.t()) / s;

        for i in 0..STATE_DIM {
            self.state[i] += k[[i, 0]] * innov * lambda; // Convert innovation to velocity
        }

        let eye = Array2::<f64>::eye(STATE_DIM);
        let kh = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * h[[0, j]];
                }
            }
            m
        };
        let i_kh = &eye - &kh;
        let kr_kt = {
            let mut m = Array2::<f64>::zeros((STATE_DIM, STATE_DIM));
            for i in 0..STATE_DIM {
                for j in 0..STATE_DIM {
                    m[[i, j]] = k[[i, 0]] * r_var * k[[j, 0]];
                }
            }
            m
        };
        self.covariance = i_kh.dot(&self.covariance).dot(&i_kh.t()) + kr_kt;
    }
}

/// Return type for fallible internal helper; uses `Option` rather than pulling
/// in a separate error type for the 3x3 inverse.
fn invert_3x3(m: &Array2<f64>) -> Option<Array2<f64>> {
    let det = m[[0, 0]] * (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]])
        - m[[0, 1]] * (m[[1, 0]] * m[[2, 2]] - m[[1, 2]] * m[[2, 0]])
        + m[[0, 2]] * (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv_det = 1.0 / det;
    let mut inv = Array2::<f64>::zeros((3, 3));
    inv[[0, 0]] = (m[[1, 1]] * m[[2, 2]] - m[[1, 2]] * m[[2, 1]]) * inv_det;
    inv[[0, 1]] = (m[[0, 2]] * m[[2, 1]] - m[[0, 1]] * m[[2, 2]]) * inv_det;
    inv[[0, 2]] = (m[[0, 1]] * m[[1, 2]] - m[[0, 2]] * m[[1, 1]]) * inv_det;
    inv[[1, 0]] = (m[[1, 2]] * m[[2, 0]] - m[[1, 0]] * m[[2, 2]]) * inv_det;
    inv[[1, 1]] = (m[[0, 0]] * m[[2, 2]] - m[[0, 2]] * m[[2, 0]]) * inv_det;
    inv[[1, 2]] = (m[[0, 2]] * m[[1, 0]] - m[[0, 0]] * m[[1, 2]]) * inv_det;
    inv[[2, 0]] = (m[[1, 0]] * m[[2, 1]] - m[[1, 1]] * m[[2, 0]]) * inv_det;
    inv[[2, 1]] = (m[[0, 1]] * m[[2, 0]] - m[[0, 0]] * m[[2, 1]]) * inv_det;
    inv[[2, 2]] = (m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]]) * inv_det;
    Some(inv)
}

// ─── 3. Multi-Sensor Bayesian Fusion ──────────────────────────────────────────

/// A single sensor estimate produced by any positioning method (trilateration,
/// `AoA` intersection, RSS fingerprinting, etc.).
///
/// The estimate is described by a Gaussian: position mean + 3x3 covariance.
#[derive(Debug, Clone)]
pub struct SensorEstimate {
    /// Source sensor type.
    pub sensor_type: SensingType,
    /// Estimated position (mean).
    pub position: Vector3,
    /// 3x3 position covariance matrix (row-major `[[xx,xy,xz],[yx,yy,yz],[zx,zy,zz]]`).
    pub covariance: Array2<f64>,
}

impl SensorEstimate {
    /// Convenience constructor for isotropic (diagonal) covariance with the
    /// given standard deviation per axis.
    pub fn isotropic(sensor_type: SensingType, position: Vector3, sigma: f64) -> Self {
        let var = sigma * sigma;
        let mut cov = Array2::<f64>::zeros((3, 3));
        cov[[0, 0]] = var;
        cov[[1, 1]] = var;
        cov[[2, 2]] = var;
        Self {
            sensor_type,
            position,
            covariance: cov,
        }
    }
}

/// Result of Bayesian multi-sensor fusion.
#[derive(Debug, Clone)]
pub struct BayesianFusionResult {
    /// Fused position (mean of the posterior Gaussian).
    pub position: Vector3,
    /// Fused 3x3 covariance matrix.
    pub covariance: Array2<f64>,
    /// Number of sensor estimates that were fused.
    pub num_sources: usize,
}

/// Fuses multiple Gaussian position estimates into a single posterior
/// estimate using the information-filter form of Bayesian fusion:
///
/// ```text
/// P_fused^{-1} = sum_i  C_i^{-1}
/// x_fused      = P_fused  * sum_i  C_i^{-1} * x_i
/// ```
///
/// Each sensor estimate is weighted by its inverse covariance (precision),
/// which is the correct Bayesian combination of independent Gaussian
/// observations.
///
/// Returns `None` when fewer than 1 estimate is provided or the fused
/// information matrix is singular.
pub fn bayesian_fuse(estimates: &[SensorEstimate]) -> Option<BayesianFusionResult> {
    if estimates.is_empty() {
        return None;
    }

    // Accumulate information matrix (sum of precisions) and information
    // vector (sum of precision * mean).
    let mut info_matrix = Array2::<f64>::zeros((3, 3));
    let mut info_vector = Array1::<f64>::zeros(3);

    for est in estimates {
        let precision = invert_3x3(&est.covariance)?;
        let mu = Array1::from(vec![est.position.x, est.position.y, est.position.z]);
        info_vector = info_vector + precision.dot(&mu);
        info_matrix = info_matrix + precision;
    }

    let fused_cov = invert_3x3(&info_matrix)?;
    let fused_mean = fused_cov.dot(&info_vector);

    Some(BayesianFusionResult {
        position: Vector3::new(fused_mean[0], fused_mean[1], fused_mean[2]),
        covariance: fused_cov,
        num_sources: estimates.len(),
    })
}

/// Convenience function that converts raw [`SensingMeasurement`]s from
/// different sensor types into [`SensorEstimate`]s and fuses them.
///
/// For ToA/RTT measurements, trilateration is run on the set of range
/// measurements.  For `AoA` measurements, simple geometric intersection is
/// attempted.  For RSS measurements, a path-loss-based range is computed and
/// fed into trilateration.
///
/// All resulting single-method estimates are then combined through
/// [`bayesian_fuse`].
pub fn fuse_multi_sensor(
    measurements: &[SensingMeasurement],
    anchors: &HashMap<i32, Vector3>,
) -> Option<BayesianFusionResult> {
    let mut estimates: Vec<SensorEstimate> = Vec::new();

    // ── Group measurements by type ───────────────────────────────────────
    let mut toa_rtt: Vec<&SensingMeasurement> = Vec::new();
    let mut rss: Vec<&SensingMeasurement> = Vec::new();
    let mut aoa: Vec<&SensingMeasurement> = Vec::new();

    for m in measurements {
        match m.measurement_type {
            SensingType::ToA | SensingType::Rtt => toa_rtt.push(m),
            SensingType::Rss => rss.push(m),
            SensingType::AoA => aoa.push(m),
            _ => {}
        }
    }

    // ── ToA/RTT trilateration ────────────────────────────────────────────
    if toa_rtt.len() >= 3 {
        let mut a_pos = Vec::new();
        let mut ranges = Vec::new();
        let mut w = Vec::new();
        for m in &toa_rtt {
            if let Some(a) = anchors.get(&m.anchor_id) {
                a_pos.push(*a);
                ranges.push(m.value);
                let sigma = m.uncertainty.max(0.1);
                w.push(1.0 / (sigma * sigma));
            }
        }
        if a_pos.len() >= 3 {
            if let Some(res) = trilaterate(&a_pos, &ranges, &w, &TrilaterationConfig::default()) {
                let avg_sigma: f64 = toa_rtt.iter().map(|m| m.uncertainty).sum::<f64>()
                    / toa_rtt.len() as f64;
                estimates.push(SensorEstimate::isotropic(
                    SensingType::ToA,
                    res.position,
                    avg_sigma * (1.0 + res.residual_norm.sqrt()),
                ));
            }
        }
    }

    // ── RSS path-loss trilateration ──────────────────────────────────────
    if rss.len() >= 3 {
        let mut a_pos = Vec::new();
        let mut ranges = Vec::new();
        let mut w = Vec::new();
        for m in &rss {
            if let Some(a) = anchors.get(&m.anchor_id) {
                a_pos.push(*a);
                let distance = 10.0_f64.powf((-40.0 - m.value) / 20.0);
                ranges.push(distance);
                let sigma = m.uncertainty.max(0.1);
                w.push(1.0 / (sigma * sigma));
            }
        }
        if a_pos.len() >= 3 {
            if let Some(res) = trilaterate(&a_pos, &ranges, &w, &TrilaterationConfig::default()) {
                // RSS-based positioning is generally less precise
                let avg_sigma: f64 =
                    rss.iter().map(|m| m.uncertainty).sum::<f64>() / rss.len() as f64;
                let rss_uncertainty = (avg_sigma * 3.0) * (1.0 + res.residual_norm.sqrt());
                estimates.push(SensorEstimate::isotropic(
                    SensingType::Rss,
                    res.position,
                    rss_uncertainty,
                ));
            }
        }
    }

    // ── AoA bearing intersection (2 or more bearings) ────────────────────
    if aoa.len() >= 2 {
        // Weighted least-squares intersection of bearing lines in XY plane.
        // Each bearing from anchor_i at angle theta_i defines a line.
        // We solve:  A * [x, y]^T = b  where each row is derived from the
        // line equation  sin(theta)*x - cos(theta)*y = sin(theta)*ax - cos(theta)*ay.
        let valid_aoa: Vec<_> = aoa
            .iter()
            .filter_map(|m| anchors.get(&m.anchor_id).map(|a| (*m, *a)))
            .collect();

        if valid_aoa.len() >= 2 {
            let nn = valid_aoa.len();
            let mut a_mat = Array2::<f64>::zeros((nn, 2));
            let mut b_vec = Array1::<f64>::zeros(nn);
            let mut w_diag = Array2::<f64>::zeros((nn, nn));

            for (i, (m, anchor)) in valid_aoa.iter().enumerate() {
                let theta = m.value; // azimuth in radians
                let s = theta.sin();
                let c = theta.cos();
                a_mat[[i, 0]] = s;
                a_mat[[i, 1]] = -c;
                b_vec[i] = s * anchor.x - c * anchor.y;
                let sigma = m.uncertainty.max(0.01);
                w_diag[[i, i]] = 1.0 / (sigma * sigma);
            }

            // Normal equations: (A^T W A) x = A^T W b
            let atw = a_mat.t().dot(&w_diag);
            let atwa = atw.dot(&a_mat);
            let atwb = atw.dot(&b_vec);

            // Solve 2x2
            let det = atwa[[0, 0]] * atwa[[1, 1]] - atwa[[0, 1]] * atwa[[1, 0]];
            if det.abs() > 1e-20 {
                let inv_det = 1.0 / det;
                let x_est = inv_det * (atwa[[1, 1]] * atwb[0] - atwa[[0, 1]] * atwb[1]);
                let y_est = inv_det * (atwa[[0, 0]] * atwb[1] - atwa[[1, 0]] * atwb[0]);
                // Average z from anchors
                let z_est: f64 =
                    valid_aoa.iter().map(|(_, a)| a.z).sum::<f64>() / valid_aoa.len() as f64;

                let avg_sigma: f64 = valid_aoa.iter().map(|(m, _)| m.uncertainty).sum::<f64>()
                    / valid_aoa.len() as f64;

                // AoA uncertainty depends on range to anchors
                let mean_range: f64 = valid_aoa
                    .iter()
                    .map(|(_, a)| {
                        let ddx = x_est - a.x;
                        let ddy = y_est - a.y;
                        (ddx * ddx + ddy * ddy).sqrt()
                    })
                    .sum::<f64>()
                    / valid_aoa.len() as f64;
                let pos_sigma = mean_range * avg_sigma; // arc-length approximation

                estimates.push(SensorEstimate::isotropic(
                    SensingType::AoA,
                    Vector3::new(x_est, y_est, z_est),
                    pos_sigma.max(0.1),
                ));
            }
        }
    }

    if estimates.is_empty() {
        return None;
    }

    bayesian_fuse(&estimates)
}

// ─── 4. 3GPP TR 22.837 Sensing Use Cases ─────────────────────────────────────

/// Object detection result per 3GPP TR 22.837 use case 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetectionResult {
    /// Whether an object is detected as present.
    pub detected: bool,
    /// Detection confidence (0.0 to 1.0).
    pub confidence: f64,
    /// RSS change that triggered detection (dB). Positive means signal
    /// strengthened (reflection from new object); negative means attenuation.
    pub rss_change_db: f64,
}

/// Detects object presence/absence based on RSS changes relative to a
/// baseline.  When the absolute RSS change exceeds `threshold_db`, an
/// object is declared as detected.
///
/// * `current_rss_dbm` - current measured RSS values per anchor.
/// * `baseline_rss_dbm` - reference (empty-room) RSS values per anchor.
/// * `threshold_db` - detection threshold on absolute RSS change (dB).
///
/// The confidence is computed from the ratio of the observed change to the
/// threshold, clamped to `[0, 1]`.
pub fn detect_object(
    current_rss_dbm: &[f64],
    baseline_rss_dbm: &[f64],
    threshold_db: f64,
) -> ObjectDetectionResult {
    let n = current_rss_dbm.len().min(baseline_rss_dbm.len());
    if n == 0 {
        return ObjectDetectionResult {
            detected: false,
            confidence: 0.0,
            rss_change_db: 0.0,
        };
    }

    // Compute per-anchor absolute RSS change and take the maximum.
    let mut max_change: f64 = 0.0;
    let mut sum_sq_change: f64 = 0.0;

    for i in 0..n {
        let delta = current_rss_dbm[i] - baseline_rss_dbm[i];
        sum_sq_change += delta * delta;
        if delta.abs() > max_change.abs() {
            max_change = delta;
        }
    }

    let rms_change = (sum_sq_change / n as f64).sqrt();
    let detected = rms_change > threshold_db;

    // Confidence based on how far above threshold
    let confidence = if threshold_db > 0.0 {
        (rms_change / threshold_db).clamp(0.0, 1.0)
    } else if detected {
        1.0
    } else {
        0.0
    };

    ObjectDetectionResult {
        detected,
        confidence,
        rss_change_db: max_change,
    }
}

/// Velocity estimation result from Doppler measurements per 3GPP TR 22.837.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VelocityEstimate {
    /// Radial velocity towards/away from anchor (m/s). Positive = moving away.
    pub radial_velocity: f64,
    /// Full 3-D velocity vector (if multiple Doppler measurements are
    /// available and the geometry is sufficient). `None` if only radial
    /// velocity could be determined.
    pub velocity_3d: Option<Vector3>,
    /// Velocity uncertainty (m/s, 1-sigma).
    pub uncertainty: f64,
}

/// Estimates velocity from Doppler shift measurements.
///
/// The Doppler frequency shift `f_d` relates to radial velocity `v_r` as:
///   `f_d = (v_r / c) * f_c`
/// so  `v_r = f_d * c / f_c`
///
/// When multiple Doppler measurements from different anchors are available
/// together with anchor positions and an estimated target position, a full
/// 3-D velocity can be resolved via least-squares.
///
/// * `doppler_measurements` - slice of `(anchor_id, doppler_shift_hz, sigma_hz)`.
/// * `carrier_freq_hz` - the carrier frequency used for sensing.
/// * `target_position` - current estimated target position (needed for 3-D).
/// * `anchors` - map of `anchor_id` to position.
pub fn estimate_velocity_doppler(
    doppler_measurements: &[(i32, f64, f64)],
    carrier_freq_hz: f64,
    target_position: Option<&Vector3>,
    anchors: &HashMap<i32, Vector3>,
) -> Option<VelocityEstimate> {
    if doppler_measurements.is_empty() || carrier_freq_hz <= 0.0 {
        return None;
    }

    let lambda = SPEED_OF_LIGHT / carrier_freq_hz;

    // Convert each Doppler shift to radial velocity: v_r = f_d * lambda
    let radial_vels: Vec<(i32, f64, f64)> = doppler_measurements
        .iter()
        .map(|&(id, fd, sigma_fd)| {
            let vr = fd * lambda;
            let sigma_vr = sigma_fd * lambda;
            (id, vr, sigma_vr)
        })
        .collect();

    // If only one measurement, return radial velocity only.
    if radial_vels.len() < 3 || target_position.is_none() {
        let (_, vr, sigma_vr) = radial_vels[0];
        return Some(VelocityEstimate {
            radial_velocity: vr,
            velocity_3d: None,
            uncertainty: sigma_vr,
        });
    }

    // Multiple measurements: solve for 3-D velocity.
    // The radial velocity from anchor i is v_r_i = u_i . v  where u_i is the
    // unit vector from target to anchor i.
    let target = target_position?;
    let valid: Vec<_> = radial_vels
        .iter()
        .filter_map(|&(id, vr, sigma)| anchors.get(&id).map(|a| (a, vr, sigma)))
        .collect();

    if valid.len() < 3 {
        let (_, vr, sigma_vr) = radial_vels[0];
        return Some(VelocityEstimate {
            radial_velocity: vr,
            velocity_3d: None,
            uncertainty: sigma_vr,
        });
    }

    let nn = valid.len();
    let mut a_mat = Array2::<f64>::zeros((nn, 3));
    let mut b_vec = Array1::<f64>::zeros(nn);
    let mut w_diag = Array2::<f64>::zeros((nn, nn));

    for (i, (anchor, vr, sigma)) in valid.iter().enumerate() {
        let dx = anchor.x - target.x;
        let dy = anchor.y - target.y;
        let dz = anchor.z - target.z;
        let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(1e-12);

        // Unit vector from target to anchor (direction of positive radial velocity)
        a_mat[[i, 0]] = dx / dist;
        a_mat[[i, 1]] = dy / dist;
        a_mat[[i, 2]] = dz / dist;
        b_vec[i] = *vr;
        w_diag[[i, i]] = 1.0 / (sigma * sigma).max(1e-12);
    }

    // Weighted least-squares: (A^T W A + lambda*I) v = A^T W b
    let atw = a_mat.t().dot(&w_diag);
    let mut atwa = atw.dot(&a_mat);
    let atwb = atw.dot(&b_vec);

    // Levenberg-Marquardt damping for near-singular geometries
    let diag_avg = (atwa[[0, 0]] + atwa[[1, 1]] + atwa[[2, 2]]) / 3.0;
    let lambda = (diag_avg * 1e-6).max(1e-10);
    atwa[[0, 0]] += lambda;
    atwa[[1, 1]] += lambda;
    atwa[[2, 2]] += lambda;

    let v_arr = {
        let b_arr = Array1::from(vec![atwb[0], atwb[1], atwb[2]]);
        solve_3x3(&atwa, &b_arr)?
    };

    let velocity = Vector3::new(v_arr[0], v_arr[1], v_arr[2]);
    let _speed = velocity.magnitude();

    // Average radial velocity (for the scalar field)
    let avg_radial: f64 = radial_vels.iter().map(|(_, vr, _)| *vr).sum::<f64>()
        / radial_vels.len() as f64;

    // Velocity uncertainty from the covariance of the LS solution: (A^T W A)^{-1}
    let vel_cov = invert_3x3(&atwa);
    let uncertainty = vel_cov
        .map(|c| (c[[0, 0]] + c[[1, 1]] + c[[2, 2]]).sqrt())
        .unwrap_or(1.0);

    Some(VelocityEstimate {
        radial_velocity: avg_radial,
        velocity_3d: Some(velocity),
        uncertainty,
    })
}

/// Range estimation result from RTT measurements per 3GPP TR 22.837.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeEstimate {
    /// Estimated range in meters.
    pub range_m: f64,
    /// Range uncertainty (1-sigma, meters).
    pub uncertainty_m: f64,
    /// Raw RTT in nanoseconds.
    pub rtt_ns: f64,
}

/// Estimates range from a Round-Trip Time measurement.
///
/// `rtt_ns` is the measured round-trip time in nanoseconds.
/// `rtt_sigma_ns` is the measurement uncertainty in nanoseconds.
/// `processing_delay_ns` is the known processing delay at the far end that
/// should be subtracted from the RTT.
pub fn estimate_range_rtt(
    rtt_ns: f64,
    rtt_sigma_ns: f64,
    processing_delay_ns: f64,
) -> RangeEstimate {
    // range = c * (RTT - processing_delay) / 2
    let effective_rtt_ns = (rtt_ns - processing_delay_ns).max(0.0);
    let effective_rtt_s = effective_rtt_ns * 1e-9;
    let range_m = SPEED_OF_LIGHT * effective_rtt_s / 2.0;

    // uncertainty propagation: sigma_range = c * sigma_rtt / 2
    let range_sigma = SPEED_OF_LIGHT * rtt_sigma_ns * 1e-9 / 2.0;

    RangeEstimate {
        range_m,
        uncertainty_m: range_sigma,
        rtt_ns,
    }
}

// ─── 5. Sensing-Communication Resource Manager ───────────────────────────────

/// Tracks the allocation of time/frequency resources between sensing and
/// communication in an ISAC system.
///
/// The manager maintains a `sensing_fraction` in `[0, 1]` representing the
/// share of OFDM symbols (or subcarriers, depending on the duplexing scheme)
/// dedicated to sensing.  The remaining `1 - sensing_fraction` is available
/// for communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensingCommResourceManager {
    /// Fraction of resources allocated to sensing (0.0 to 1.0).
    sensing_fraction: f64,
    /// Minimum allowed sensing fraction.
    min_sensing: f64,
    /// Maximum allowed sensing fraction.
    max_sensing: f64,
    /// Total available resource blocks.
    total_resource_blocks: u32,
}

impl SensingCommResourceManager {
    /// Creates a new resource manager.
    ///
    /// * `initial_sensing_fraction` - initial sensing share in `[0, 1]`.
    /// * `min_sensing` - floor for the sensing fraction.
    /// * `max_sensing` - ceiling for the sensing fraction.
    /// * `total_resource_blocks` - total number of OFDM resource blocks.
    pub fn new(
        initial_sensing_fraction: f64,
        min_sensing: f64,
        max_sensing: f64,
        total_resource_blocks: u32,
    ) -> Self {
        let min_s = min_sensing.clamp(0.0, 1.0);
        let max_s = max_sensing.clamp(min_s, 1.0);
        Self {
            sensing_fraction: initial_sensing_fraction.clamp(min_s, max_s),
            min_sensing: min_s,
            max_sensing: max_s,
            total_resource_blocks,
        }
    }

    /// Returns the current fraction of resources allocated to sensing.
    pub fn sensing_fraction(&self) -> f64 {
        self.sensing_fraction
    }

    /// Returns the current fraction of resources allocated to communication.
    pub fn communication_fraction(&self) -> f64 {
        1.0 - self.sensing_fraction
    }

    /// Returns the number of resource blocks allocated to sensing.
    pub fn sensing_resource_blocks(&self) -> u32 {
        (self.total_resource_blocks as f64 * self.sensing_fraction).round() as u32
    }

    /// Returns the number of resource blocks allocated to communication.
    pub fn communication_resource_blocks(&self) -> u32 {
        self.total_resource_blocks - self.sensing_resource_blocks()
    }

    /// Returns the total number of resource blocks.
    pub fn total_resource_blocks(&self) -> u32 {
        self.total_resource_blocks
    }

    /// Sets the sensing fraction, clamped to `[min_sensing, max_sensing]`.
    pub fn set_sensing_fraction(&mut self, fraction: f64) {
        self.sensing_fraction = fraction.clamp(self.min_sensing, self.max_sensing);
    }

    /// Increases sensing allocation by `delta` (clamped).
    pub fn increase_sensing(&mut self, delta: f64) {
        self.set_sensing_fraction(self.sensing_fraction + delta);
    }

    /// Decreases sensing allocation by `delta` (clamped).
    pub fn decrease_sensing(&mut self, delta: f64) {
        self.set_sensing_fraction(self.sensing_fraction - delta);
    }

    /// Adjusts the split based on a sensing quality metric.
    ///
    /// If `current_quality` is below `target_quality`, the sensing fraction
    /// is increased by a proportional step.  If above, it is decreased.
    /// `step_size` controls the maximum adjustment per call.
    pub fn adjust_for_quality(
        &mut self,
        current_quality: f64,
        target_quality: f64,
        step_size: f64,
    ) {
        let error = target_quality - current_quality;
        // Proportional controller
        let adjustment = (error * step_size).clamp(-step_size, step_size);
        self.set_sensing_fraction(self.sensing_fraction + adjustment);
    }

    /// Updates the total resource block count (e.g., when bandwidth changes).
    pub fn set_total_resource_blocks(&mut self, total: u32) {
        self.total_resource_blocks = total;
    }
}

impl Default for SensingCommResourceManager {
    fn default() -> Self {
        Self::new(0.2, 0.05, 0.5, 100)
    }
}

// ─── Clutter and NLOS Detection ───────────────────────────────────────────────

/// NLOS (Non-Line-of-Sight) detection result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NlosStatus {
    /// Line-of-sight path
    Los,
    /// Non-line-of-sight (obstructed)
    Nlos,
    /// Unknown
    Unknown,
}

/// Detects NLOS conditions using measurement consistency checks
///
/// Uses multiple range measurements to detect inconsistencies that indicate
/// NLOS propagation (reflections, diffraction).
pub fn detect_nlos(
    measurements: &[SensingMeasurement],
    anchors: &HashMap<i32, Vector3>,
    estimated_position: &Vector3,
) -> Vec<(i32, NlosStatus)> {
    let mut results = Vec::new();
    let nlos_threshold = 10.0; // 10 meters error threshold

    for measurement in measurements {
        if let Some(anchor_pos) = anchors.get(&measurement.anchor_id) {
            let expected_range = estimated_position.distance_to(anchor_pos);
            let measured_range = match measurement.measurement_type {
                SensingType::ToA | SensingType::Rtt => measurement.value,
                SensingType::Rss => {
                    // Path loss model
                    10.0_f64.powf((-40.0 - measurement.value) / 20.0)
                }
                _ => continue,
            };

            let error = (measured_range - expected_range).abs();

            let status = if error > nlos_threshold {
                NlosStatus::Nlos
            } else if error < measurement.uncertainty * 3.0 {
                NlosStatus::Los
            } else {
                NlosStatus::Unknown
            };

            results.push((measurement.anchor_id, status));
        }
    }

    results
}

/// Multipath component
#[derive(Debug, Clone)]
pub struct MultipathComponent {
    /// Path delay (seconds)
    pub delay: f64,
    /// Path attenuation (dB)
    pub attenuation_db: f64,
    /// Angle of arrival (radians)
    pub aoa_rad: f64,
    /// Is this the direct path?
    pub is_direct: bool,
}

/// Models multipath propagation for a measurement
///
/// Returns possible multipath components based on environment geometry
pub fn model_multipath(
    anchor: &Vector3,
    target: &Vector3,
    num_reflections: usize,
) -> Vec<MultipathComponent> {
    let mut components = Vec::new();

    // Direct path (LOS)
    let direct_distance = anchor.distance_to(target);
    let direct_delay = direct_distance / SPEED_OF_LIGHT;
    components.push(MultipathComponent {
        delay: direct_delay,
        attenuation_db: 0.0,
        aoa_rad: (target.y - anchor.y).atan2(target.x - anchor.x),
        is_direct: true,
    });

    // Simplified multipath model: assume reflections from ground and walls
    for i in 1..=num_reflections {
        // Ground reflection (simplified)
        let reflected_z = -target.z;
        let reflected_pos = Vector3::new(target.x, target.y, reflected_z);
        let reflected_distance = anchor.distance_to(&reflected_pos);
        let reflected_delay = reflected_distance / SPEED_OF_LIGHT;

        components.push(MultipathComponent {
            delay: reflected_delay,
            attenuation_db: -3.0 * i as f64, // 3 dB loss per reflection
            aoa_rad: (reflected_pos.y - anchor.y).atan2(reflected_pos.x - anchor.x),
            is_direct: false,
        });
    }

    components
}

// ─── Bistatic/Multistatic Sensing ─────────────────────────────────────────────

/// Bistatic sensing configuration
#[derive(Debug, Clone)]
pub struct BistaticConfig {
    /// Transmitter position
    pub transmitter: Vector3,
    /// Receiver position
    pub receiver: Vector3,
    /// Carrier frequency (Hz)
    pub carrier_freq_hz: f64,
}

impl BistaticConfig {
    /// Computes bistatic range (transmitter -> target -> receiver)
    pub fn bistatic_range(&self, target: &Vector3) -> f64 {
        self.transmitter.distance_to(target) + target.distance_to(&self.receiver)
    }

    /// Computes bistatic Doppler shift
    pub fn bistatic_doppler(&self, target: &Vector3, target_velocity: &Vector3) -> f64 {
        let lambda = SPEED_OF_LIGHT / self.carrier_freq_hz;

        // Unit vectors
        let tx_dir = Vector3::new(
            target.x - self.transmitter.x,
            target.y - self.transmitter.y,
            target.z - self.transmitter.z,
        );
        let tx_dist = tx_dir.magnitude();
        let tx_unit = Vector3::new(
            tx_dir.x / tx_dist,
            tx_dir.y / tx_dist,
            tx_dir.z / tx_dist,
        );

        let rx_dir = Vector3::new(
            self.receiver.x - target.x,
            self.receiver.y - target.y,
            self.receiver.z - target.z,
        );
        let rx_dist = rx_dir.magnitude();
        let rx_unit = Vector3::new(
            rx_dir.x / rx_dist,
            rx_dir.y / rx_dist,
            rx_dir.z / rx_dist,
        );

        // Radial velocities
        let v_tx = target_velocity.x * tx_unit.x
            + target_velocity.y * tx_unit.y
            + target_velocity.z * tx_unit.z;
        let v_rx = target_velocity.x * rx_unit.x
            + target_velocity.y * rx_unit.y
            + target_velocity.z * rx_unit.z;

        // Bistatic Doppler
        (v_tx + v_rx) / lambda
    }
}

// ─── ISAC Manager (backward-compatible, extended) ─────────────────────────────

/// ISAC manager
#[derive(Debug)]
pub struct IsacManager {
    /// Anchor positions (`cell_id` -> position)
    anchors: HashMap<i32, Vector3>,
    /// Tracking states (`object_id` -> state)
    tracking: HashMap<u64, TrackingState>,
    /// Extended Kalman Filters per target (`object_id` -> EKF)
    ekf_states: HashMap<u64, ExtendedKalmanFilter>,
    /// Recent sensing data
    recent_data: HashMap<i32, SensingData>,
    /// Fusion interval (ms)
    fusion_interval_ms: u32,
    /// Sensing-communication resource manager
    resource_manager: SensingCommResourceManager,
}

impl Default for IsacManager {
    fn default() -> Self {
        Self {
            anchors: HashMap::new(),
            tracking: HashMap::new(),
            ekf_states: HashMap::new(),
            recent_data: HashMap::new(),
            fusion_interval_ms: 50,
            resource_manager: SensingCommResourceManager::default(),
        }
    }
}

impl IsacManager {
    /// Creates a new ISAC manager
    pub fn new(fusion_interval_ms: u32) -> Self {
        Self {
            fusion_interval_ms,
            ..Self::default()
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

    /// Performs position fusion for a target using trilateration.
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

    /// Performs multi-sensor Bayesian fusion for a target, combining `ToA`, `AoA`,
    /// and RSS measurements.
    pub fn fuse_position_bayesian(&self, target_id: i32) -> Option<FusedPosition> {
        let data = self.recent_data.get(&target_id)?;
        let result = fuse_multi_sensor(&data.measurements, &self.anchors)?;
        let uncertainty =
            (result.covariance[[0, 0]] + result.covariance[[1, 1]] + result.covariance[[2, 2]])
                .sqrt();

        Some(FusedPosition {
            target_id,
            position: result.position,
            velocity: Vector3::default(),
            position_uncertainty: uncertainty,
            velocity_uncertainty: 1.0,
            confidence: (1.0 / (1.0 + uncertainty / 10.0)).min(1.0) as f32,
            timestamp_ms: data.timestamp_ms,
        })
    }

    /// Gets or creates the EKF for a target object.
    pub fn get_or_create_ekf(
        &mut self,
        object_id: u64,
        process_noise_accel: f64,
    ) -> &mut ExtendedKalmanFilter {
        self.ekf_states
            .entry(object_id)
            .or_insert_with(|| ExtendedKalmanFilter::uninitialised(process_noise_accel))
    }

    /// Runs an EKF predict + update cycle for the given object using the
    /// latest range measurements.  Returns the updated position estimate.
    pub fn ekf_update(
        &mut self,
        object_id: u64,
        dt: f64,
        range_measurements: &[(i32, f64, f64)], // (anchor_id, range_m, sigma_m)
        process_noise_accel: f64,
    ) -> Option<Vector3> {
        let ekf = self
            .ekf_states
            .entry(object_id)
            .or_insert_with(|| ExtendedKalmanFilter::uninitialised(process_noise_accel));

        ekf.predict(dt);

        for &(anchor_id, range, sigma) in range_measurements {
            if let Some(anchor) = self.anchors.get(&anchor_id) {
                ekf.update_range(anchor, range, sigma);
            }
        }

        if ekf.is_initialised() {
            Some(ekf.position())
        } else {
            None
        }
    }

    /// Returns a reference to the EKF state for the given object (if it
    /// exists).
    pub fn get_ekf(&self, object_id: u64) -> Option<&ExtendedKalmanFilter> {
        self.ekf_states.get(&object_id)
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

    /// Returns a reference to the resource manager.
    pub fn resource_manager(&self) -> &SensingCommResourceManager {
        &self.resource_manager
    }

    /// Returns a mutable reference to the resource manager.
    pub fn resource_manager_mut(&mut self) -> &mut SensingCommResourceManager {
        &mut self.resource_manager
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // --- Vector3 tests (unchanged) ---

    #[test]
    fn test_vector3_distance() {
        let p1 = Vector3::new(0.0, 0.0, 0.0);
        let p2 = Vector3::new(3.0, 4.0, 0.0);
        assert!((p1.distance_to(&p2) - 5.0).abs() < 0.001);
    }

    // --- Tracking tests (unchanged) ---

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
        state.velocity = Vector3::new(10.0, 5.0, 0.0);

        let predicted = state.predict(1.0);
        assert!((predicted.x - 10.0).abs() < 0.001);
        assert!((predicted.y - 5.0).abs() < 0.001);
    }

    // --- ISAC Manager test (unchanged) ---

    #[test]
    fn test_isac_manager() {
        let mut manager = IsacManager::new(50);

        manager.register_anchor(1, Vector3::new(0.0, 0.0, 0.0));
        manager.register_anchor(2, Vector3::new(100.0, 0.0, 0.0));
        manager.register_anchor(3, Vector3::new(50.0, 86.6, 0.0));

        manager.update_tracking(1, Vector3::new(50.0, 40.0, 0.0), 1.0);

        let state = manager.get_tracking(1);
        assert!(state.is_some());
    }

    // --- Position fusion test (updated to verify trilateration) ---

    #[test]
    fn test_position_fusion() {
        let mut anchors = HashMap::new();
        anchors.insert(1, Vector3::new(0.0, 0.0, 0.0));
        anchors.insert(2, Vector3::new(100.0, 0.0, 0.0));
        anchors.insert(3, Vector3::new(50.0, 86.6, 0.0));

        // True target at (50, 28.87, 0) - centroid of the triangle
        let true_pos = Vector3::new(50.0, 28.87, 0.0);
        let measurements = vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: true_pos.distance_to(&anchors[&1]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: true_pos.distance_to(&anchors[&2]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 3,
                value: true_pos.distance_to(&anchors[&3]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
        ];

        let result = fuse_positions(&measurements, &anchors);
        assert!(result.is_some());
        let (pos, _uncertainty) = result.unwrap();
        // Trilateration should converge close to the true position
        assert!(
            (pos.x - true_pos.x).abs() < 1.0,
            "x error too large: {} vs {}",
            pos.x,
            true_pos.x
        );
        assert!(
            (pos.y - true_pos.y).abs() < 1.0,
            "y error too large: {} vs {}",
            pos.y,
            true_pos.y
        );
    }

    // --- Trilateration tests ---

    #[test]
    fn test_trilaterate_exact() {
        // Equilateral triangle anchors, target at known location
        let anchors = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 0.0, 0.0),
            Vector3::new(50.0, 86.6, 0.0),
        ];
        let true_target = Vector3::new(40.0, 30.0, 0.0);
        let ranges: Vec<f64> = anchors.iter().map(|a| true_target.distance_to(a)).collect();
        let weights = vec![1.0, 1.0, 1.0];

        let result =
            trilaterate(&anchors, &ranges, &weights, &TrilaterationConfig::default()).unwrap();

        assert!(result.converged);
        assert!((result.position.x - 40.0).abs() < 0.01);
        assert!((result.position.y - 30.0).abs() < 0.01);
        assert!(result.residual_norm < 0.001);
    }

    #[test]
    fn test_trilaterate_noisy() {
        let anchors = vec![
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(100.0, 0.0, 0.0),
            Vector3::new(50.0, 86.6, 0.0),
            Vector3::new(50.0, 28.87, 50.0),
        ];
        let true_target = Vector3::new(40.0, 30.0, 5.0);
        // Add small noise to ranges
        let ranges: Vec<f64> = anchors
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let noise = [0.5, -0.3, 0.2, -0.1][i];
                true_target.distance_to(a) + noise
            })
            .collect();
        let weights = vec![1.0, 1.0, 1.0, 1.0];

        let result =
            trilaterate(&anchors, &ranges, &weights, &TrilaterationConfig::default()).unwrap();

        assert!(result.converged);
        assert!(
            (result.position.x - 40.0).abs() < 2.0,
            "x: {}",
            result.position.x
        );
        assert!(
            (result.position.y - 30.0).abs() < 2.0,
            "y: {}",
            result.position.y
        );
    }

    #[test]
    fn test_trilaterate_insufficient_anchors() {
        let anchors = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(100.0, 0.0, 0.0)];
        let ranges = vec![50.0, 50.0];
        let weights = vec![1.0, 1.0];

        let result = trilaterate(&anchors, &ranges, &weights, &TrilaterationConfig::default());
        assert!(result.is_none());
    }

    // --- EKF tests ---

    #[test]
    fn test_ekf_constant_velocity() {
        let mut ekf = ExtendedKalmanFilter::new(
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(10.0, 5.0, 0.0), // 10 m/s x, 5 m/s y
            1.0,
            1.0,
            0.1,
        );

        // Predict 1 second forward
        ekf.predict(1.0);
        let pos = ekf.position();
        assert!((pos.x - 10.0).abs() < 0.1);
        assert!((pos.y - 5.0).abs() < 0.1);

        // Predict another second
        ekf.predict(1.0);
        let pos = ekf.position();
        assert!((pos.x - 20.0).abs() < 0.2);
        assert!((pos.y - 10.0).abs() < 0.2);
    }

    #[test]
    fn test_ekf_range_update() {
        let mut ekf = ExtendedKalmanFilter::new(
            Vector3::new(50.0, 50.0, 0.0),
            Vector3::default(),
            20.0,
            5.0,
            1.0,
        );

        let anchor1 = Vector3::new(0.0, 0.0, 0.0);
        let anchor2 = Vector3::new(100.0, 0.0, 0.0);
        let anchor3 = Vector3::new(50.0, 86.6, 0.0);

        let true_pos = Vector3::new(40.0, 30.0, 0.0);

        // Feed range measurements multiple times to converge
        for _ in 0..20 {
            ekf.predict(0.1);
            ekf.update_range(&anchor1, true_pos.distance_to(&anchor1), 1.0);
            ekf.update_range(&anchor2, true_pos.distance_to(&anchor2), 1.0);
            ekf.update_range(&anchor3, true_pos.distance_to(&anchor3), 1.0);
        }

        let pos = ekf.position();
        assert!(
            (pos.x - 40.0).abs() < 2.0,
            "EKF x error: {} vs 40",
            pos.x
        );
        assert!(
            (pos.y - 30.0).abs() < 2.0,
            "EKF y error: {} vs 30",
            pos.y
        );
    }

    #[test]
    fn test_ekf_position_update() {
        let mut ekf = ExtendedKalmanFilter::new(
            Vector3::new(100.0, 100.0, 0.0),
            Vector3::default(),
            50.0,
            5.0,
            1.0,
        );

        let true_pos = Vector3::new(40.0, 30.0, 0.0);

        for _ in 0..10 {
            ekf.predict(0.1);
            ekf.update_position(&true_pos, 2.0);
        }

        let pos = ekf.position();
        assert!(
            (pos.x - 40.0).abs() < 1.0,
            "x: {} vs 40.0",
            pos.x
        );
        assert!(
            (pos.y - 30.0).abs() < 1.0,
            "y: {} vs 30.0",
            pos.y
        );
    }

    #[test]
    fn test_ekf_uncertainty_decreases() {
        let mut ekf = ExtendedKalmanFilter::new(
            Vector3::new(50.0, 50.0, 0.0),
            Vector3::default(),
            20.0,
            5.0,
            0.5,
        );

        let initial_unc = ekf.position_uncertainty();

        let anchor = Vector3::new(0.0, 0.0, 0.0);
        let true_pos = Vector3::new(50.0, 50.0, 0.0);
        let range = true_pos.distance_to(&anchor);

        for _ in 0..5 {
            ekf.predict(0.1);
            ekf.update_range(&anchor, range, 1.0);
        }

        assert!(
            ekf.position_uncertainty() < initial_unc,
            "Uncertainty should decrease: {} vs {}",
            ekf.position_uncertainty(),
            initial_unc
        );
    }

    // --- Bayesian fusion tests ---

    #[test]
    fn test_bayesian_fuse_two_sensors() {
        let est1 = SensorEstimate::isotropic(
            SensingType::ToA,
            Vector3::new(10.0, 0.0, 0.0),
            2.0, // sigma = 2m
        );
        let est2 = SensorEstimate::isotropic(
            SensingType::Rss,
            Vector3::new(0.0, 0.0, 0.0),
            4.0, // sigma = 4m (less precise)
        );

        let result = bayesian_fuse(&[est1, est2]).unwrap();

        // Fused position should be closer to est1 (more precise)
        assert!(result.position.x > 5.0, "Fused x: {}", result.position.x);
        assert!(result.position.x < 10.0, "Fused x: {}", result.position.x);
        assert_eq!(result.num_sources, 2);
    }

    #[test]
    fn test_bayesian_fuse_equal_weight() {
        let est1 = SensorEstimate::isotropic(
            SensingType::ToA,
            Vector3::new(10.0, 0.0, 0.0),
            1.0,
        );
        let est2 = SensorEstimate::isotropic(
            SensingType::Rss,
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
        );

        let result = bayesian_fuse(&[est1, est2]).unwrap();

        // Equal weights should give midpoint
        assert!(
            (result.position.x - 5.0).abs() < 0.01,
            "Fused x: {}",
            result.position.x
        );
    }

    #[test]
    fn test_bayesian_fuse_empty() {
        let result = bayesian_fuse(&[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_multi_sensor_fusion() {
        let mut anchors = HashMap::new();
        anchors.insert(1, Vector3::new(0.0, 0.0, 0.0));
        anchors.insert(2, Vector3::new(100.0, 0.0, 0.0));
        anchors.insert(3, Vector3::new(50.0, 86.6, 0.0));

        let true_pos = Vector3::new(40.0, 30.0, 0.0);

        let measurements = vec![
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 1,
                value: true_pos.distance_to(&anchors[&1]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 2,
                value: true_pos.distance_to(&anchors[&2]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
            SensingMeasurement {
                measurement_type: SensingType::ToA,
                anchor_id: 3,
                value: true_pos.distance_to(&anchors[&3]),
                uncertainty: 1.0,
                timestamp_ms: 0,
            },
        ];

        let result = fuse_multi_sensor(&measurements, &anchors);
        assert!(result.is_some());
        let r = result.unwrap();
        assert!(
            (r.position.x - 40.0).abs() < 2.0,
            "x: {}",
            r.position.x
        );
        assert!(
            (r.position.y - 30.0).abs() < 2.0,
            "y: {}",
            r.position.y
        );
    }

    // --- Object detection tests ---

    #[test]
    fn test_object_detection_present() {
        let baseline = vec![-50.0, -55.0, -60.0];
        let current = vec![-40.0, -45.0, -50.0]; // +10 dB change everywhere
        let result = detect_object(&current, &baseline, 5.0);
        assert!(result.detected);
        assert!(result.confidence > 0.5);
    }

    #[test]
    fn test_object_detection_absent() {
        let baseline = vec![-50.0, -55.0, -60.0];
        let current = vec![-50.5, -55.5, -60.5]; // tiny change
        let result = detect_object(&current, &baseline, 5.0);
        assert!(!result.detected);
        assert!(result.confidence < 0.5);
    }

    #[test]
    fn test_object_detection_empty() {
        let result = detect_object(&[], &[], 5.0);
        assert!(!result.detected);
    }

    // --- Velocity estimation tests ---

    #[test]
    fn test_velocity_single_doppler() {
        let carrier_freq = 28.0e9; // 28 GHz mmWave
        let true_radial_vel = 30.0; // 30 m/s
        let doppler_shift = true_radial_vel * carrier_freq / SPEED_OF_LIGHT;

        let measurements = vec![(1, doppler_shift, 10.0)];
        let anchors = HashMap::new();

        let result =
            estimate_velocity_doppler(&measurements, carrier_freq, None, &anchors).unwrap();

        assert!(
            (result.radial_velocity - true_radial_vel).abs() < 0.1,
            "radial vel: {}",
            result.radial_velocity
        );
        assert!(result.velocity_3d.is_none());
    }

    #[test]
    fn test_velocity_3d_doppler() {
        let carrier_freq = 3.5e9; // 3.5 GHz mid-band
        let lambda = SPEED_OF_LIGHT / carrier_freq;

        // Use 4 well-distributed anchors around the target for good geometry
        let mut anchors = HashMap::new();
        anchors.insert(1, Vector3::new(100.0, 0.0, 10.0));
        anchors.insert(2, Vector3::new(0.0, 100.0, 10.0));
        anchors.insert(3, Vector3::new(-100.0, 0.0, 10.0));
        anchors.insert(4, Vector3::new(0.0, -100.0, 10.0));

        let target = Vector3::new(0.0, 0.0, 0.0);
        let true_vel = Vector3::new(10.0, -5.0, 2.0);

        // Compute expected Doppler for each anchor
        let mut doppler_measurements = Vec::new();
        for (&id, anchor) in &anchors {
            let dx = anchor.x - target.x;
            let dy = anchor.y - target.y;
            let dz = anchor.z - target.z;
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            // Radial velocity = v . u_hat
            let vr = (true_vel.x * dx + true_vel.y * dy + true_vel.z * dz) / dist;
            let fd = vr / lambda;
            doppler_measurements.push((id, fd, 1.0));
        }

        let result = estimate_velocity_doppler(
            &doppler_measurements,
            carrier_freq,
            Some(&target),
            &anchors,
        )
        .unwrap();

        assert!(result.velocity_3d.is_some());
        let v = result.velocity_3d.unwrap();
        assert!(
            (v.x - true_vel.x).abs() < 1.0,
            "vx: {} vs {}",
            v.x,
            true_vel.x
        );
        assert!(
            (v.y - true_vel.y).abs() < 1.0,
            "vy: {} vs {}",
            v.y,
            true_vel.y
        );
    }

    // --- Range estimation tests ---

    #[test]
    fn test_range_estimation_rtt() {
        // 100m range -> RTT = 2 * 100 / c = ~666.7 ns
        let true_range = 100.0;
        let rtt_ns = 2.0 * true_range / SPEED_OF_LIGHT * 1e9;

        let result = estimate_range_rtt(rtt_ns, 1.0, 0.0);
        assert!(
            (result.range_m - true_range).abs() < 0.01,
            "range: {} vs {}",
            result.range_m,
            true_range
        );
    }

    #[test]
    fn test_range_estimation_with_processing_delay() {
        let true_range = 100.0;
        let processing_delay_ns = 50.0; // 50 ns processing delay
        let rtt_ns = 2.0 * true_range / SPEED_OF_LIGHT * 1e9 + processing_delay_ns;

        let result = estimate_range_rtt(rtt_ns, 1.0, processing_delay_ns);
        assert!(
            (result.range_m - true_range).abs() < 0.01,
            "range: {}",
            result.range_m
        );
    }

    // --- Resource manager tests ---

    #[test]
    fn test_resource_manager_basic() {
        let rm = SensingCommResourceManager::new(0.3, 0.1, 0.5, 200);
        assert!((rm.sensing_fraction() - 0.3).abs() < 1e-10);
        assert!((rm.communication_fraction() - 0.7).abs() < 1e-10);
        assert_eq!(rm.sensing_resource_blocks(), 60);
        assert_eq!(rm.communication_resource_blocks(), 140);
        assert_eq!(rm.total_resource_blocks(), 200);
    }

    #[test]
    fn test_resource_manager_clamping() {
        let mut rm = SensingCommResourceManager::new(0.3, 0.1, 0.5, 100);

        rm.set_sensing_fraction(0.8);
        assert!((rm.sensing_fraction() - 0.5).abs() < 1e-10);

        rm.set_sensing_fraction(0.01);
        assert!((rm.sensing_fraction() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_resource_manager_increase_decrease() {
        let mut rm = SensingCommResourceManager::new(0.3, 0.1, 0.5, 100);

        rm.increase_sensing(0.1);
        assert!((rm.sensing_fraction() - 0.4).abs() < 1e-10);

        rm.decrease_sensing(0.2);
        assert!((rm.sensing_fraction() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_resource_manager_quality_adjust() {
        let mut rm = SensingCommResourceManager::new(0.2, 0.05, 0.5, 100);

        // Quality is below target -> should increase sensing
        rm.adjust_for_quality(0.5, 0.8, 0.1);
        assert!(rm.sensing_fraction() > 0.2);

        // Quality is above target -> should decrease sensing
        rm.adjust_for_quality(0.9, 0.8, 0.1);
        let prev = rm.sensing_fraction();
        rm.adjust_for_quality(0.9, 0.8, 0.1);
        assert!(rm.sensing_fraction() < prev);
    }

    #[test]
    fn test_resource_manager_default() {
        let rm = SensingCommResourceManager::default();
        assert!((rm.sensing_fraction() - 0.2).abs() < 1e-10);
        assert_eq!(rm.total_resource_blocks(), 100);
    }

    // --- IsacManager extended tests ---

    #[test]
    fn test_isac_manager_ekf_integration() {
        let mut manager = IsacManager::new(50);
        manager.register_anchor(1, Vector3::new(0.0, 0.0, 0.0));
        manager.register_anchor(2, Vector3::new(100.0, 0.0, 0.0));
        manager.register_anchor(3, Vector3::new(50.0, 86.6, 0.0));

        let true_pos = Vector3::new(40.0, 30.0, 0.0);
        let ranges: Vec<(i32, f64, f64)> = vec![
            (1, true_pos.distance_to(&Vector3::new(0.0, 0.0, 0.0)), 1.0),
            (
                2,
                true_pos.distance_to(&Vector3::new(100.0, 0.0, 0.0)),
                1.0,
            ),
            (
                3,
                true_pos.distance_to(&Vector3::new(50.0, 86.6, 0.0)),
                1.0,
            ),
        ];

        for _ in 0..20 {
            let _ = manager.ekf_update(42, 0.1, &ranges, 1.0);
        }

        let ekf = manager.get_ekf(42);
        assert!(ekf.is_some());
        let pos = ekf.unwrap().position();
        assert!(
            (pos.x - 40.0).abs() < 3.0,
            "Manager EKF x: {}",
            pos.x
        );
        assert!(
            (pos.y - 30.0).abs() < 3.0,
            "Manager EKF y: {}",
            pos.y
        );
    }

    #[test]
    fn test_isac_manager_resource_manager() {
        let mut manager = IsacManager::new(50);
        assert!((manager.resource_manager().sensing_fraction() - 0.2).abs() < 1e-10);
        manager.resource_manager_mut().set_sensing_fraction(0.4);
        assert!((manager.resource_manager().sensing_fraction() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_isac_manager_bayesian_fusion() {
        let mut manager = IsacManager::new(50);
        manager.register_anchor(1, Vector3::new(0.0, 0.0, 0.0));
        manager.register_anchor(2, Vector3::new(100.0, 0.0, 0.0));
        manager.register_anchor(3, Vector3::new(50.0, 86.6, 0.0));

        let true_pos = Vector3::new(40.0, 30.0, 0.0);
        let data = SensingData {
            target_id: 10,
            cell_id: 1,
            measurements: vec![
                SensingMeasurement {
                    measurement_type: SensingType::ToA,
                    anchor_id: 1,
                    value: true_pos.distance_to(&Vector3::new(0.0, 0.0, 0.0)),
                    uncertainty: 1.0,
                    timestamp_ms: 1000,
                },
                SensingMeasurement {
                    measurement_type: SensingType::ToA,
                    anchor_id: 2,
                    value: true_pos.distance_to(&Vector3::new(100.0, 0.0, 0.0)),
                    uncertainty: 1.0,
                    timestamp_ms: 1000,
                },
                SensingMeasurement {
                    measurement_type: SensingType::ToA,
                    anchor_id: 3,
                    value: true_pos.distance_to(&Vector3::new(50.0, 86.6, 0.0)),
                    uncertainty: 1.0,
                    timestamp_ms: 1000,
                },
            ],
            timestamp_ms: 1000,
        };

        manager.record_sensing_data(data);
        let result = manager.fuse_position_bayesian(10);
        assert!(result.is_some());
        let fp = result.unwrap();
        assert!(
            (fp.position.x - 40.0).abs() < 2.0,
            "Bayesian x: {}",
            fp.position.x
        );
        assert!(
            (fp.position.y - 30.0).abs() < 2.0,
            "Bayesian y: {}",
            fp.position.y
        );
    }
}
