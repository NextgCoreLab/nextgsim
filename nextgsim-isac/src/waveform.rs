//! OFDM-radar waveform modeling for joint sensing-communication
//!
//! Implements joint ISAC waveform design per 6G requirements.

use serde::{Deserialize, Serialize};

/// OFDM-radar waveform parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OfdmRadarWaveform {
    /// Number of subcarriers
    pub num_subcarriers: usize,
    /// Subcarrier spacing (Hz)
    pub subcarrier_spacing_hz: f64,
    /// OFDM symbol duration (seconds)
    pub symbol_duration_s: f64,
    /// Cyclic prefix length (samples)
    pub cyclic_prefix_length: usize,
    /// Carrier frequency (Hz)
    pub carrier_freq_hz: f64,
    /// Bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Sensing subcarrier indices (dedicated to sensing)
    pub sensing_subcarriers: Vec<usize>,
    /// Communication subcarrier indices
    pub comm_subcarriers: Vec<usize>,
}

impl OfdmRadarWaveform {
    /// Creates a new OFDM-radar waveform with default 5G NR-like parameters
    pub fn new_5g_nr(carrier_freq_hz: f64, sensing_fraction: f64) -> Self {
        let num_subcarriers = 1200; // 100 MHz bandwidth with 15 kHz spacing
        let subcarrier_spacing_hz = 15_000.0;
        let bandwidth_hz = num_subcarriers as f64 * subcarrier_spacing_hz;
        let symbol_duration_s = 1.0 / subcarrier_spacing_hz;
        let cyclic_prefix_length = 144; // Normal CP

        // Allocate subcarriers
        let num_sensing = (num_subcarriers as f64 * sensing_fraction) as usize;
        let sensing_subcarriers: Vec<usize> = (0..num_sensing).collect();
        let comm_subcarriers: Vec<usize> = (num_sensing..num_subcarriers).collect();

        Self {
            num_subcarriers,
            subcarrier_spacing_hz,
            symbol_duration_s,
            cyclic_prefix_length,
            carrier_freq_hz,
            bandwidth_hz,
            sensing_subcarriers,
            comm_subcarriers,
        }
    }

    /// Returns the total OFDM symbol duration including CP (seconds)
    pub fn total_symbol_duration_s(&self) -> f64 {
        let cp_duration = self.cyclic_prefix_length as f64 / self.bandwidth_hz;
        self.symbol_duration_s + cp_duration
    }

    /// Computes the range resolution (meters)
    pub fn range_resolution_m(&self) -> f64 {
        let c = 299_792_458.0; // Speed of light
        c / (2.0 * self.bandwidth_hz)
    }

    /// Computes the maximum unambiguous range (meters)
    pub fn max_unambiguous_range_m(&self) -> f64 {
        let c = 299_792_458.0;
        c * self.total_symbol_duration_s() / 2.0
    }

    /// Computes the Doppler resolution (Hz)
    pub fn doppler_resolution_hz(&self, num_symbols: usize) -> f64 {
        1.0 / (num_symbols as f64 * self.total_symbol_duration_s())
    }

    /// Computes the maximum unambiguous Doppler (Hz)
    pub fn max_unambiguous_doppler_hz(&self) -> f64 {
        self.subcarrier_spacing_hz / 2.0
    }

    /// Converts Doppler shift to radial velocity (m/s)
    pub fn doppler_to_velocity(&self, doppler_hz: f64) -> f64 {
        let c = 299_792_458.0;
        let lambda = c / self.carrier_freq_hz;
        doppler_hz * lambda
    }

    /// Converts radial velocity to Doppler shift (Hz)
    pub fn velocity_to_doppler(&self, velocity_ms: f64) -> f64 {
        let c = 299_792_458.0;
        let lambda = c / self.carrier_freq_hz;
        velocity_ms / lambda
    }

    /// Simulates a range-Doppler map for the waveform
    ///
    /// Returns a simplified 2D representation where:
    /// - Rows represent range bins
    /// - Columns represent Doppler bins
    #[allow(clippy::needless_range_loop)]
    pub fn generate_range_doppler_map(
        &self,
        num_range_bins: usize,
        num_doppler_bins: usize,
        targets: &[(f64, f64, f64)], // (range_m, velocity_ms, rcs)
    ) -> Vec<Vec<f64>> {
        let mut map = vec![vec![0.0; num_doppler_bins]; num_range_bins];

        let range_bin_size = self.max_unambiguous_range_m() / num_range_bins as f64;
        let doppler_bin_size = 2.0 * self.max_unambiguous_doppler_hz() / num_doppler_bins as f64;

        for &(range, velocity, rcs) in targets {
            let doppler = self.velocity_to_doppler(velocity);

            // Map to bins
            let range_bin = (range / range_bin_size).floor() as usize;
            let doppler_bin = ((doppler + self.max_unambiguous_doppler_hz()) / doppler_bin_size)
                .floor() as usize;

            if range_bin < num_range_bins && doppler_bin < num_doppler_bins {
                // Simplified response (Gaussian-like spread)
                let sigma_range = 2.0;
                let sigma_doppler = 2.0;

                for r in range_bin.saturating_sub(3)..=(range_bin + 3).min(num_range_bins - 1) {
                    for d in doppler_bin.saturating_sub(3)..=(doppler_bin + 3).min(num_doppler_bins - 1) {
                        let dr = (r as f64 - range_bin as f64) / sigma_range;
                        let dd = (d as f64 - doppler_bin as f64) / sigma_doppler;
                        let response = rcs * (-0.5 * (dr * dr + dd * dd)).exp();
                        map[r][d] += response;
                    }
                }
            }
        }

        map
    }

    /// Computes the sensing overhead (fraction of resources for sensing)
    pub fn sensing_overhead(&self) -> f64 {
        self.sensing_subcarriers.len() as f64 / self.num_subcarriers as f64
    }

    /// Computes the communication efficiency (fraction for communication)
    pub fn communication_efficiency(&self) -> f64 {
        1.0 - self.sensing_overhead()
    }
}

impl Default for OfdmRadarWaveform {
    fn default() -> Self {
        Self::new_5g_nr(3.5e9, 0.2) // 3.5 GHz, 20% sensing
    }
}

// ─── Clutter Model ─────────────────────────────────────────────────────────────

/// Weibull-distributed clutter model for radar sensing (TR 22.837)
///
/// Models ground/environmental clutter with range-dependent power
/// and Weibull amplitude distribution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClutterModel {
    /// Noise floor power (dBm)
    pub noise_floor_dbm: f64,
    /// Clutter-to-noise ratio at reference range (dB)
    pub cnr_db: f64,
    /// Weibull shape parameter (k). k=2 → Rayleigh, k>2 → less spiky
    pub shape_parameter: f64,
    /// Reference range for CNR (meters)
    pub reference_range_m: f64,
    /// Range exponent for power decay (typically 3.0–4.0)
    pub range_exponent: f64,
}

impl Default for ClutterModel {
    fn default() -> Self {
        Self {
            noise_floor_dbm: -110.0,
            cnr_db: 20.0,
            shape_parameter: 2.5,
            reference_range_m: 100.0,
            range_exponent: 3.5,
        }
    }
}

impl ClutterModel {
    /// Computes clutter power at a given range (linear scale)
    ///
    /// Uses range-dependent power law: P(r) = `P_ref` * (`r_ref` / r)^n
    pub fn clutter_power(&self, range_m: f64) -> f64 {
        if range_m <= 0.0 {
            return 0.0;
        }
        let noise_linear = 10.0_f64.powf(self.noise_floor_dbm / 10.0);
        let cnr_linear = 10.0_f64.powf(self.cnr_db / 10.0);
        let ref_power = noise_linear * cnr_linear;
        ref_power * (self.reference_range_m / range_m).powf(self.range_exponent)
    }

    /// Computes clutter power in dBm at a given range
    pub fn clutter_power_dbm(&self, range_m: f64) -> f64 {
        let linear = self.clutter_power(range_m);
        if linear <= 0.0 {
            return f64::NEG_INFINITY;
        }
        10.0 * linear.log10()
    }

    /// Computes signal-to-clutter ratio (dB) for a target at given range and RCS
    pub fn signal_to_clutter_db(&self, range_m: f64, target_power: f64) -> f64 {
        let clutter = self.clutter_power(range_m);
        if clutter <= 0.0 {
            return f64::INFINITY;
        }
        10.0 * (target_power / clutter).log10()
    }
}

// ─── CFAR Detection ────────────────────────────────────────────────────────────

/// Detection result from CFAR processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfarDetection {
    /// Range bin index of detection
    pub range_bin: usize,
    /// Doppler bin index of detection
    pub doppler_bin: usize,
    /// Detected power (linear)
    pub power: f64,
    /// Adaptive threshold at detection point (linear)
    pub threshold: f64,
}

/// Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector
///
/// Implements 2D CA-CFAR on a range-Doppler map per standard radar
/// signal processing for ISAC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CfarDetector {
    /// Number of guard cells on each side (range dimension)
    pub guard_cells: usize,
    /// Number of training cells on each side (range dimension)
    pub training_cells: usize,
    /// Probability of false alarm (0..1)
    pub pfa: f64,
}

impl Default for CfarDetector {
    fn default() -> Self {
        Self {
            guard_cells: 2,
            training_cells: 8,
            pfa: 1e-4,
        }
    }
}

impl CfarDetector {
    /// Computes the CFAR threshold multiplier for the given PFA and
    /// number of training cells. For CA-CFAR: alpha = N * (Pfa^(-1/N) - 1)
    fn threshold_factor(&self) -> f64 {
        let n = (2 * self.training_cells) as f64;
        n * (self.pfa.powf(-1.0 / n) - 1.0)
    }

    /// Runs 2D CA-CFAR detection on a range-Doppler map
    ///
    /// Returns a list of detections where the cell under test exceeds the
    /// adaptive threshold computed from surrounding training cells.
    #[allow(clippy::needless_range_loop)]
    pub fn detect(&self, range_doppler_map: &[Vec<f64>]) -> Vec<CfarDetection> {
        let mut detections = Vec::new();
        let alpha = self.threshold_factor();
        let num_range = range_doppler_map.len();
        if num_range == 0 {
            return detections;
        }
        let num_doppler = range_doppler_map[0].len();
        let window = self.guard_cells + self.training_cells;

        for r in window..num_range.saturating_sub(window) {
            for d in window..num_doppler.saturating_sub(window) {
                let cut = range_doppler_map[r][d]; // cell under test

                // Average over training cells (excluding guard cells)
                let mut sum = 0.0;
                let mut count = 0u32;
                for ri in (r - window)..=(r + window) {
                    for di in (d - window)..=(d + window) {
                        let dr = ri.abs_diff(r);
                        let dd = di.abs_diff(d);
                        // Skip CUT and guard cells
                        if dr <= self.guard_cells && dd <= self.guard_cells {
                            continue;
                        }
                        if ri < num_range && di < num_doppler {
                            sum += range_doppler_map[ri][di];
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let noise_estimate = sum / count as f64;
                    let threshold = alpha * noise_estimate;
                    if cut > threshold {
                        detections.push(CfarDetection {
                            range_bin: r,
                            doppler_bin: d,
                            power: cut,
                            threshold,
                        });
                    }
                }
            }
        }

        detections
    }
}

// ─── Bistatic Geometry ─────────────────────────────────────────────────────────

/// Bistatic radar geometry for multi-site ISAC (TR 22.837)
///
/// Models transmitter/receiver separated by a baseline distance, enabling
/// bistatic range, Doppler, and angle computations for multistatic sensing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BistaticGeometry {
    /// Transmitter position (x, y, z) in meters
    pub tx_position: [f64; 3],
    /// Receiver position (x, y, z) in meters
    pub rx_position: [f64; 3],
    /// Carrier frequency (Hz) for Doppler calculations
    pub carrier_freq_hz: f64,
}

impl BistaticGeometry {
    /// Creates a new bistatic geometry
    pub fn new(tx_position: [f64; 3], rx_position: [f64; 3], carrier_freq_hz: f64) -> Self {
        Self {
            tx_position,
            rx_position,
            carrier_freq_hz,
        }
    }

    /// Returns the baseline distance between TX and RX (meters)
    pub fn baseline_m(&self) -> f64 {
        let dx = self.tx_position[0] - self.rx_position[0];
        let dy = self.tx_position[1] - self.rx_position[1];
        let dz = self.tx_position[2] - self.rx_position[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn distance(a: &[f64; 3], b: &[f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    /// Computes the bistatic range for a target at the given position.
    ///
    /// Bistatic range = d(TX, target) + d(target, RX) - baseline
    pub fn bistatic_range(&self, target: &[f64; 3]) -> f64 {
        let d_tx = Self::distance(&self.tx_position, target);
        let d_rx = Self::distance(&self.rx_position, target);
        d_tx + d_rx - self.baseline_m()
    }

    /// Computes the bistatic Doppler shift (Hz) for a target with given velocity.
    ///
    /// Uses the bistatic Doppler formula:
    /// `f_d` = (1/λ) * (v⃗ · (`t̂_TX` + `t̂_RX`))
    /// where `t̂_TX` and `t̂_RX` are unit vectors from target to TX and RX.
    pub fn bistatic_doppler(&self, target: &[f64; 3], velocity: &[f64; 3]) -> f64 {
        let c = 299_792_458.0_f64;
        let lambda = c / self.carrier_freq_hz;

        // Unit vector from target to TX
        let d_tx = Self::distance(&self.tx_position, target);
        let d_rx = Self::distance(&self.rx_position, target);

        if d_tx < 1e-10 || d_rx < 1e-10 {
            return 0.0;
        }

        let tx_hat = [
            (self.tx_position[0] - target[0]) / d_tx,
            (self.tx_position[1] - target[1]) / d_tx,
            (self.tx_position[2] - target[2]) / d_tx,
        ];
        let rx_hat = [
            (self.rx_position[0] - target[0]) / d_rx,
            (self.rx_position[1] - target[1]) / d_rx,
            (self.rx_position[2] - target[2]) / d_rx,
        ];

        // Dot product: v · (tx_hat + rx_hat)
        let dot: f64 = (0..3)
            .map(|i| velocity[i] * (tx_hat[i] + rx_hat[i]))
            .sum();

        dot / lambda
    }

    /// Computes the bistatic angle (radians) — the angle at the target
    /// between the TX and RX directions.
    pub fn bistatic_angle(&self, target: &[f64; 3]) -> f64 {
        let d_tx = Self::distance(&self.tx_position, target);
        let d_rx = Self::distance(&self.rx_position, target);

        if d_tx < 1e-10 || d_rx < 1e-10 {
            return 0.0;
        }

        // cos(β) = (d_tx² + d_rx² - baseline²) / (2 * d_tx * d_rx)
        let baseline = self.baseline_m();
        let cos_beta =
            (d_tx * d_tx + d_rx * d_rx - baseline * baseline) / (2.0 * d_tx * d_rx);
        cos_beta.clamp(-1.0, 1.0).acos()
    }
}

// ─── Joint Communication-Sensing Waveform Design ──────────────────────────────

/// Performance metrics for joint sensing-communication waveform
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointWaveformMetrics {
    /// Range resolution (meters)
    pub range_resolution_m: f64,
    /// Maximum unambiguous range (meters)
    pub max_range_m: f64,
    /// Velocity resolution (m/s)
    pub velocity_resolution_ms: f64,
    /// Maximum unambiguous velocity (m/s)
    pub max_velocity_ms: f64,
    /// Communication spectral efficiency (bps/Hz)
    pub comm_spectral_efficiency: f64,
    /// Sensing SNR (dB) for given communication SNR
    pub sensing_snr_db: f64,
}

/// Joint communication-sensing OFDM waveform design
///
/// Optimizes power allocation between sensing and communication subcarriers
/// to maximize both radar detection performance and data throughput.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointWaveformDesign {
    /// Carrier frequency (Hz)
    pub carrier_freq_hz: f64,
    /// Total bandwidth (Hz)
    pub bandwidth_hz: f64,
    /// Number of subcarriers
    pub num_subcarriers: usize,
    /// Number of OFDM symbols per frame
    pub num_symbols: usize,
    /// Subcarrier spacing (Hz)
    pub subcarrier_spacing_hz: f64,
    /// Fraction of total power allocated to sensing (0..1)
    pub sensing_power_fraction: f64,
    /// Sensing subcarrier allocation pattern (true = sensing, false = comm)
    pub subcarrier_allocation: Vec<bool>,
}

impl JointWaveformDesign {
    /// Creates a new joint waveform design with default even allocation
    pub fn new(
        carrier_freq_hz: f64,
        bandwidth_hz: f64,
        num_subcarriers: usize,
        num_symbols: usize,
    ) -> Self {
        let subcarrier_spacing_hz = bandwidth_hz / num_subcarriers as f64;

        // Default: interleaved allocation (every 4th subcarrier for sensing)
        let subcarrier_allocation: Vec<bool> = (0..num_subcarriers)
            .map(|i| i % 4 == 0)
            .collect();

        let sensing_count = subcarrier_allocation.iter().filter(|&&s| s).count();
        let sensing_power_fraction = sensing_count as f64 / num_subcarriers as f64;

        Self {
            carrier_freq_hz,
            bandwidth_hz,
            num_subcarriers,
            num_symbols,
            subcarrier_spacing_hz,
            sensing_power_fraction,
            subcarrier_allocation,
        }
    }

    /// Optimizes power allocation between sensing and communication
    ///
    /// `target_snr_db` - minimum required communication SNR
    /// `sensing_priority` - 0.0 (comm-only) to 1.0 (sensing-only)
    pub fn optimize_power_allocation(
        &mut self,
        target_snr_db: f64,
        sensing_priority: f64,
    ) {
        let priority = sensing_priority.clamp(0.0, 1.0);

        // Water-filling inspired allocation:
        // Higher comm SNR requirement -> less power for sensing
        let snr_factor = 1.0 / (1.0 + 10.0_f64.powf(target_snr_db / 10.0));
        self.sensing_power_fraction = (priority * 0.5 + snr_factor * 0.3).clamp(0.05, 0.5);

        // Update subcarrier allocation based on power fraction
        let sensing_count = (self.num_subcarriers as f64 * self.sensing_power_fraction) as usize;
        self.subcarrier_allocation = vec![false; self.num_subcarriers];

        // Distribute sensing subcarriers evenly across bandwidth
        if sensing_count > 0 {
            let step = self.num_subcarriers / sensing_count;
            for i in 0..sensing_count {
                let idx = (i * step).min(self.num_subcarriers - 1);
                self.subcarrier_allocation[idx] = true;
            }
        }
    }

    /// Computes the ambiguity function value at (delay, doppler)
    ///
    /// The ambiguity function characterizes the waveform's ability to resolve
    /// targets in range and velocity simultaneously.
    pub fn ambiguity_function(&self, delay_s: f64, doppler_hz: f64) -> f64 {
        let sensing_indices: Vec<usize> = self.subcarrier_allocation
            .iter()
            .enumerate()
            .filter(|(_, &s)| s)
            .map(|(i, _)| i)
            .collect();

        if sensing_indices.is_empty() {
            return 0.0;
        }

        let n = sensing_indices.len() as f64;
        let mut real_sum = 0.0;
        let mut imag_sum = 0.0;

        for &k in &sensing_indices {
            let freq = k as f64 * self.subcarrier_spacing_hz;
            let phase = 2.0 * std::f64::consts::PI
                * (freq * delay_s + doppler_hz * k as f64 / (self.num_subcarriers as f64 * self.subcarrier_spacing_hz));
            real_sum += phase.cos();
            imag_sum += phase.sin();
        }

        (real_sum * real_sum + imag_sum * imag_sum).sqrt() / n
    }

    /// Computes performance metrics for the current waveform configuration
    pub fn performance_metrics(&self, comm_snr_db: f64) -> JointWaveformMetrics {
        let c = 299_792_458.0;
        let lambda = c / self.carrier_freq_hz;

        let sensing_bw = self.bandwidth_hz * self.sensing_power_fraction;
        let comm_bw = self.bandwidth_hz * (1.0 - self.sensing_power_fraction);

        let symbol_duration = 1.0 / self.subcarrier_spacing_hz;
        let frame_duration = symbol_duration * self.num_symbols as f64;

        JointWaveformMetrics {
            range_resolution_m: c / (2.0 * sensing_bw),
            max_range_m: c * symbol_duration / 2.0,
            velocity_resolution_ms: lambda / (2.0 * frame_duration),
            max_velocity_ms: lambda * self.subcarrier_spacing_hz / 4.0,
            comm_spectral_efficiency: (1.0 + 10.0_f64.powf(comm_snr_db / 10.0) * comm_bw / self.bandwidth_hz).log2(),
            sensing_snr_db: comm_snr_db + 10.0 * self.sensing_power_fraction.log10(),
        }
    }
}

// ─── Multistatic Sensing Network ──────────────────────────────────────────────

/// A node in a multistatic sensing network (can be TX, RX, or both)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultstaticNode {
    /// Node position [x, y, z] in meters
    pub position: [f64; 3],
    /// Whether this node can transmit
    pub is_transmitter: bool,
    /// Whether this node can receive
    pub is_receiver: bool,
}

/// Multistatic sensing network with multiple TX/RX points
///
/// Enables multi-geometry sensing using combinations of transmitters
/// and receivers for improved target detection and localization.
#[derive(Debug, Clone)]
pub struct MultstaticNetwork {
    /// Nodes in the network
    pub nodes: Vec<MultstaticNode>,
    /// Carrier frequency (Hz)
    pub carrier_freq_hz: f64,
}

impl MultstaticNetwork {
    /// Creates a new empty multistatic network
    pub fn new(carrier_freq_hz: f64) -> Self {
        Self {
            nodes: Vec::new(),
            carrier_freq_hz,
        }
    }

    /// Adds a node to the network
    pub fn add_node(&mut self, position: [f64; 3], is_transmitter: bool, is_receiver: bool) {
        self.nodes.push(MultstaticNode {
            position,
            is_transmitter,
            is_receiver,
        });
    }

    /// Returns all valid bistatic TX-RX pairs
    pub fn bistatic_pairs(&self) -> Vec<BistaticGeometry> {
        let mut pairs = Vec::new();
        for (i, tx) in self.nodes.iter().enumerate() {
            if !tx.is_transmitter {
                continue;
            }
            for (j, rx) in self.nodes.iter().enumerate() {
                if i == j || !rx.is_receiver {
                    continue;
                }
                pairs.push(BistaticGeometry::new(
                    tx.position,
                    rx.position,
                    self.carrier_freq_hz,
                ));
            }
        }
        pairs
    }

    /// Locates a target using bistatic range measurements from all pairs
    ///
    /// Uses least-squares minimization of bistatic range residuals.
    /// Returns the estimated target position or None if insufficient geometry.
    pub fn locate_target(&self, true_target: &[f64; 3]) -> Option<[f64; 3]> {
        let pairs = self.bistatic_pairs();
        if pairs.len() < 2 {
            return None;
        }

        // Compute bistatic ranges as "measurements"
        let measured_ranges: Vec<f64> = pairs
            .iter()
            .map(|bg| bg.bistatic_range(true_target))
            .collect();

        // Initial guess: centroid of all nodes
        let n = self.nodes.len() as f64;
        let mut x = [0.0; 3];
        for node in &self.nodes {
            x[0] += node.position[0] / n;
            x[1] += node.position[1] / n;
            x[2] += node.position[2] / n;
        }

        // Gauss-Newton iteration on bistatic range residuals
        for _ in 0..50 {
            let mut jtj = [[0.0f64; 3]; 3];
            let mut jtr = [0.0f64; 3];

            for (k, bg) in pairs.iter().enumerate() {
                let pred = bg.bistatic_range(&x);
                let residual = pred - measured_ranges[k];

                // Jacobian of bistatic range w.r.t. target position
                let d_tx = BistaticGeometry::distance(&bg.tx_position, &x).max(1e-12);
                let d_rx = BistaticGeometry::distance(&bg.rx_position, &x).max(1e-12);

                let jac = [
                    (x[0] - bg.tx_position[0]) / d_tx + (x[0] - bg.rx_position[0]) / d_rx,
                    (x[1] - bg.tx_position[1]) / d_tx + (x[1] - bg.rx_position[1]) / d_rx,
                    (x[2] - bg.tx_position[2]) / d_tx + (x[2] - bg.rx_position[2]) / d_rx,
                ];

                for i in 0..3 {
                    for j in 0..3 {
                        jtj[i][j] += jac[i] * jac[j];
                    }
                    jtr[i] += jac[i] * residual;
                }
            }

            // Add damping
            for (i, row) in jtj.iter_mut().enumerate() {
                row[i] += 1e-6;
            }

            // Solve 3x3 system
            let det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
                - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
                + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

            if det.abs() < 1e-30 {
                break;
            }

            let inv_det = 1.0 / det;
            let delta = [
                inv_det * ((jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1]) * jtr[0]
                    + (jtj[0][2] * jtj[2][1] - jtj[0][1] * jtj[2][2]) * jtr[1]
                    + (jtj[0][1] * jtj[1][2] - jtj[0][2] * jtj[1][1]) * jtr[2]),
                inv_det * ((jtj[1][2] * jtj[2][0] - jtj[1][0] * jtj[2][2]) * jtr[0]
                    + (jtj[0][0] * jtj[2][2] - jtj[0][2] * jtj[2][0]) * jtr[1]
                    + (jtj[0][2] * jtj[1][0] - jtj[0][0] * jtj[1][2]) * jtr[2]),
                inv_det * ((jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]) * jtr[0]
                    + (jtj[0][1] * jtj[2][0] - jtj[0][0] * jtj[2][1]) * jtr[1]
                    + (jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0]) * jtr[2]),
            ];

            x[0] -= delta[0];
            x[1] -= delta[1];
            x[2] -= delta[2];

            let step = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            if step < 1e-6 {
                break;
            }
        }

        Some(x)
    }

    /// Computes the spatial diversity gain from multiple bistatic pairs
    ///
    /// More diverse geometries yield better target localization.
    pub fn diversity_gain(&self) -> f64 {
        let pairs = self.bistatic_pairs();
        if pairs.is_empty() {
            return 0.0;
        }

        // Diversity gain proportional to sqrt(number of independent pairs)
        (pairs.len() as f64).sqrt()
    }

    /// Returns the number of nodes
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_waveform_creation() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);

        assert_eq!(wf.num_subcarriers, 1200);
        assert_eq!(wf.subcarrier_spacing_hz, 15_000.0);
        assert_eq!(wf.sensing_subcarriers.len(), 240); // 20% of 1200
        assert_eq!(wf.comm_subcarriers.len(), 960);
    }

    #[test]
    fn test_range_resolution() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);
        let range_res = wf.range_resolution_m();

        // With ~18 MHz bandwidth, range resolution should be ~8.3 m
        assert!(range_res > 8.0 && range_res < 9.0);
    }

    #[test]
    fn test_doppler_conversion() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);

        let velocity = 30.0; // 30 m/s
        let doppler = wf.velocity_to_doppler(velocity);
        let velocity_back = wf.doppler_to_velocity(doppler);

        assert!((velocity - velocity_back).abs() < 0.01);
    }

    #[test]
    fn test_sensing_overhead() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);

        assert!((wf.sensing_overhead() - 0.2).abs() < 0.01);
        assert!((wf.communication_efficiency() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_range_doppler_map() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);

        // Simulate a target at 100m, 30 m/s, RCS=1.0
        let targets = vec![(100.0, 30.0, 1.0)];
        let map = wf.generate_range_doppler_map(64, 64, &targets);

        assert_eq!(map.len(), 64);
        assert_eq!(map[0].len(), 64);

        // Check that the map has non-zero values
        let max_val: f64 = map.iter().flat_map(|row| row.iter()).copied().fold(0.0, f64::max);
        assert!(max_val > 0.0);
    }

    #[test]
    fn test_max_unambiguous_range() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);
        let max_range = wf.max_unambiguous_range_m();

        // Should be reasonable for typical OFDM parameters
        assert!(max_range > 1000.0); // At least 1 km
    }

    #[test]
    fn test_doppler_resolution() {
        let wf = OfdmRadarWaveform::new_5g_nr(28e9, 0.2);
        let doppler_res = wf.doppler_resolution_hz(100); // 100 symbols

        assert!(doppler_res > 0.0);
        assert!(doppler_res < wf.max_unambiguous_doppler_hz());
    }

    // ── Clutter Model tests ────────────────────────────────────────────────

    #[test]
    fn test_clutter_power_range_decay() {
        let clutter = ClutterModel::default();
        let p_near = clutter.clutter_power(50.0);
        let p_far = clutter.clutter_power(500.0);
        // Power must decrease with range
        assert!(p_near > p_far);
        assert!(p_near > 0.0);
    }

    #[test]
    fn test_clutter_power_zero_range() {
        let clutter = ClutterModel::default();
        assert_eq!(clutter.clutter_power(0.0), 0.0);
        assert_eq!(clutter.clutter_power(-1.0), 0.0);
    }

    #[test]
    fn test_clutter_power_dbm() {
        let clutter = ClutterModel::default();
        let p_linear = clutter.clutter_power(100.0);
        let p_dbm = clutter.clutter_power_dbm(100.0);
        // dBm = 10*log10(linear)
        assert!((p_dbm - 10.0 * p_linear.log10()).abs() < 1e-6);
    }

    #[test]
    fn test_signal_to_clutter() {
        let clutter = ClutterModel::default();
        let scr = clutter.signal_to_clutter_db(100.0, 1e-6);
        // Should be finite
        assert!(scr.is_finite());
    }

    // ── CFAR Detector tests ────────────────────────────────────────────────

    #[test]
    fn test_cfar_threshold_factor() {
        let det = CfarDetector::default();
        let alpha = det.threshold_factor();
        assert!(alpha > 0.0);
    }

    #[test]
    fn test_cfar_detect_on_noise() {
        let det = CfarDetector {
            guard_cells: 1,
            training_cells: 4,
            pfa: 1e-4,
        };
        // Uniform noise floor → few or no detections
        let map = vec![vec![1.0; 32]; 32];
        let dets = det.detect(&map);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_cfar_detect_target() {
        let det = CfarDetector {
            guard_cells: 1,
            training_cells: 4,
            pfa: 1e-4,
        };
        let mut map = vec![vec![0.01; 32]; 32];
        // Insert a strong target
        map[16][16] = 100.0;
        let dets = det.detect(&map);
        assert!(!dets.is_empty());
        // Target should be near bin (16, 16)
        let found = dets.iter().any(|d| d.range_bin == 16 && d.doppler_bin == 16);
        assert!(found);
    }

    #[test]
    fn test_cfar_detect_empty() {
        let det = CfarDetector::default();
        let map: Vec<Vec<f64>> = vec![];
        let dets = det.detect(&map);
        assert!(dets.is_empty());
    }

    // ── Bistatic Geometry tests ────────────────────────────────────────────

    #[test]
    fn test_bistatic_baseline() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        assert!((bg.baseline_m() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_bistatic_range_on_baseline() {
        // Target on the baseline midpoint
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        let target = [50.0, 0.0, 0.0];
        // d_tx=50, d_rx=50, baseline=100 → bistatic_range = 0
        assert!(bg.bistatic_range(&target).abs() < 1e-6);
    }

    #[test]
    fn test_bistatic_range_off_baseline() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        let target = [50.0, 50.0, 0.0]; // Above midpoint
        let br = bg.bistatic_range(&target);
        // d_tx = d_rx = sqrt(50^2+50^2) ≈ 70.71, baseline=100
        // bistatic_range ≈ 2*70.71 - 100 ≈ 41.42
        assert!((br - 41.42).abs() < 0.1);
    }

    #[test]
    fn test_bistatic_doppler_stationary() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        let target = [50.0, 50.0, 0.0];
        let velocity = [0.0, 0.0, 0.0]; // Stationary
        let fd = bg.bistatic_doppler(&target, &velocity);
        assert!(fd.abs() < 1e-6);
    }

    #[test]
    fn test_bistatic_doppler_moving() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        let target = [50.0, 50.0, 0.0];
        let velocity = [0.0, -30.0, 0.0]; // Moving toward baseline
        let fd = bg.bistatic_doppler(&target, &velocity);
        // Should produce positive Doppler (target approaching both TX and RX)
        assert!(fd > 0.0);
    }

    #[test]
    fn test_bistatic_angle_perpendicular() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        let target = [50.0, 50.0, 0.0]; // Equidistant, above midpoint
        let angle = bg.bistatic_angle(&target);
        // d_tx = d_rx ≈ 70.71, baseline = 100
        // cos(β) = (70.71² + 70.71² - 100²) / (2*70.71*70.71) = (5000+5000-10000)/10000 = 0
        // β = π/2
        assert!((angle - std::f64::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_bistatic_angle_collinear() {
        let bg = BistaticGeometry::new([0.0, 0.0, 0.0], [100.0, 0.0, 0.0], 3.5e9);
        // Target far away on the x-axis → angle approaches 0
        let target = [1000.0, 0.0, 0.0];
        let angle = bg.bistatic_angle(&target);
        assert!(angle < 0.15); // Small angle
    }

    // ── Joint Waveform Design tests ──────────────────────────────────────

    #[test]
    fn test_joint_waveform_creation() {
        let jw = JointWaveformDesign::new(3.5e9, 100e6, 1200, 14);
        assert_eq!(jw.num_subcarriers, 1200);
        assert_eq!(jw.num_symbols, 14);
        assert!(jw.sensing_power_fraction > 0.0);
    }

    #[test]
    fn test_joint_waveform_optimize() {
        let mut jw = JointWaveformDesign::new(3.5e9, 100e6, 1200, 14);
        jw.optimize_power_allocation(20.0, 0.5);
        assert!(jw.sensing_power_fraction > 0.0);
        assert!(jw.sensing_power_fraction <= 1.0);
    }

    #[test]
    fn test_joint_waveform_ambiguity() {
        let jw = JointWaveformDesign::new(3.5e9, 100e6, 1200, 14);
        let amb = jw.ambiguity_function(0.0, 0.0);
        assert!((amb - 1.0).abs() < 0.01); // Peak at (0,0) should be ~1
    }

    #[test]
    fn test_joint_waveform_metrics() {
        let jw = JointWaveformDesign::new(3.5e9, 100e6, 1200, 14);
        let metrics = jw.performance_metrics(20.0);
        assert!(metrics.range_resolution_m > 0.0);
        assert!(metrics.max_range_m > 0.0);
        assert!(metrics.comm_spectral_efficiency > 0.0);
    }

    // ── Multistatic Network tests ─────────────────────────────────────────

    #[test]
    fn test_multistatic_network() {
        let mut net = MultstaticNetwork::new(3.5e9);
        net.add_node([0.0, 0.0, 0.0], true, true);
        net.add_node([100.0, 0.0, 0.0], true, true);
        net.add_node([50.0, 100.0, 0.0], false, true);

        let pairs = net.bistatic_pairs();
        assert!(!pairs.is_empty());
    }

    #[test]
    fn test_multistatic_locate() {
        let mut net = MultstaticNetwork::new(3.5e9);
        net.add_node([0.0, 0.0, 0.0], true, true);
        net.add_node([100.0, 0.0, 0.0], true, true);
        net.add_node([50.0, 100.0, 0.0], true, true);

        let target = [50.0, 50.0, 0.0];
        let result = net.locate_target(&target);
        assert!(result.is_some());
        let pos = result.unwrap();
        assert!((pos[0] - 50.0).abs() < 5.0);
        assert!((pos[1] - 50.0).abs() < 5.0);
    }

    #[test]
    fn test_multistatic_diversity_gain() {
        let mut net = MultstaticNetwork::new(3.5e9);
        net.add_node([0.0, 0.0, 0.0], true, true);
        net.add_node([100.0, 0.0, 0.0], true, true);
        let gain = net.diversity_gain();
        assert!(gain >= 1.0);
    }
}
