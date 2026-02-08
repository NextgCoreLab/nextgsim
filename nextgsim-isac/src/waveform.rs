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
}
