//! RedCap (Reduced Capability) UE Mode (Rel-17)
//!
//! Implements RedCap UE mode with reduced bandwidth, processing capabilities,
//! and antenna configurations.
//!
//! # 3GPP Reference
//!
//! - TS 38.306: UE Radio Access Capabilities
//! - TS 38.331: RRC Protocol Specification (RedCap extensions)

/// RedCap UE mode configuration
#[derive(Debug, Clone)]
pub struct RedCapMode {
    /// Whether RedCap mode is enabled
    pub enabled: bool,
    /// Maximum bandwidth in MHz (20 for Rel-17, 5-20 for Rel-18)
    pub bandwidth_mhz: u8,
    /// Half-duplex FDD mode
    pub half_duplex: bool,
    /// Reduced MIMO (max layers)
    pub reduced_mimo: ReducedMimoMode,
    /// RedCap release version
    pub release: RedCapRelease,
}

/// Reduced MIMO mode for RedCap
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducedMimoMode {
    /// Single layer only
    SingleLayer,
    /// Up to 2 layers
    TwoLayers,
}

/// RedCap release version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedCapRelease {
    /// Rel-17 RedCap
    Rel17,
    /// Rel-18 RedCap with further reductions
    Rel18,
}

impl Default for RedCapMode {
    fn default() -> Self {
        Self {
            enabled: false,
            bandwidth_mhz: 100, // Normal UE
            half_duplex: false,
            reduced_mimo: ReducedMimoMode::TwoLayers,
            release: RedCapRelease::Rel17,
        }
    }
}

impl RedCapMode {
    /// Creates a disabled RedCap mode (normal UE)
    pub fn disabled() -> Self {
        Self::default()
    }

    /// Creates Rel-17 RedCap mode (20 MHz, single layer, half-duplex)
    pub fn rel17() -> Self {
        Self {
            enabled: true,
            bandwidth_mhz: 20,
            half_duplex: true,
            reduced_mimo: ReducedMimoMode::SingleLayer,
            release: RedCapRelease::Rel17,
        }
    }

    /// Creates Rel-18 RedCap mode with ultra-reduced capabilities (5 MHz)
    pub fn rel18_ultra_reduced() -> Self {
        Self {
            enabled: true,
            bandwidth_mhz: 5,
            half_duplex: true,
            reduced_mimo: ReducedMimoMode::SingleLayer,
            release: RedCapRelease::Rel18,
        }
    }

    /// Creates Rel-18 RedCap mode with enhanced capabilities (20 MHz, 2 layers)
    pub fn rel18_enhanced() -> Self {
        Self {
            enabled: true,
            bandwidth_mhz: 20,
            half_duplex: false,
            reduced_mimo: ReducedMimoMode::TwoLayers,
            release: RedCapRelease::Rel18,
        }
    }

    /// Returns true if RedCap mode is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Returns the maximum bandwidth in MHz
    pub fn max_bandwidth_mhz(&self) -> u8 {
        if self.enabled {
            self.bandwidth_mhz
        } else {
            100 // Normal UE
        }
    }

    /// Returns true if this is a half-duplex UE
    pub fn is_half_duplex(&self) -> bool {
        self.enabled && self.half_duplex
    }

    /// Returns the maximum number of MIMO layers
    pub fn max_mimo_layers(&self) -> u8 {
        if !self.enabled {
            return 4; // Normal UE
        }
        match self.reduced_mimo {
            ReducedMimoMode::SingleLayer => 1,
            ReducedMimoMode::TwoLayers => 2,
        }
    }

    /// Applies bandwidth restriction to a measurement
    pub fn apply_bandwidth_restriction(&self, bw_mhz: u8) -> u8 {
        if self.enabled {
            bw_mhz.min(self.bandwidth_mhz)
        } else {
            bw_mhz
        }
    }

    /// Encodes RedCap capabilities for UECapabilityInformation
    pub fn encode_capabilities(&self) -> Vec<u8> {
        if !self.enabled {
            return vec![]; // No RedCap capabilities
        }

        let mut caps = Vec::with_capacity(16);

        // RedCap indicator (1 byte)
        caps.push(1); // RedCap enabled

        // Max bandwidth (1 byte)
        caps.push(self.bandwidth_mhz);

        // Flags (1 byte): bit 0 = half_duplex, bits 1-2 = mimo_mode
        let mut flags: u8 = 0;
        if self.half_duplex {
            flags |= 0x01;
        }
        match self.reduced_mimo {
            ReducedMimoMode::SingleLayer => flags |= 0x02,
            ReducedMimoMode::TwoLayers => flags |= 0x04,
        }
        caps.push(flags);

        // Release version (1 byte)
        caps.push(match self.release {
            RedCapRelease::Rel17 => 17,
            RedCapRelease::Rel18 => 18,
        });

        // Rel-18 specific capabilities
        if self.release == RedCapRelease::Rel18 {
            // Extended capabilities byte (bit flags)
            let mut ext_caps: u8 = 0;
            // Bit 0: supports eDRX
            if self.bandwidth_mhz <= 20 {
                ext_caps |= 0x01;
            }
            // Bit 1: supports relaxed monitoring (1 Rx antenna)
            if self.bandwidth_mhz <= 5 {
                ext_caps |= 0x02;
            }
            // Bit 2: supports extended T_proc
            ext_caps |= 0x04;
            caps.push(ext_caps);

            // Peak data rate cap (4 bytes, kbps)
            let peak_rate_kbps: u32 = if self.bandwidth_mhz <= 5 {
                50_000 // 50 Mbps max for 5 MHz variant
            } else {
                200_000 // 200 Mbps for 20 MHz variant
            };
            caps.extend_from_slice(&peak_rate_kbps.to_be_bytes());

            // Number of Rx antennas (1 byte)
            let rx_antennas = if self.bandwidth_mhz <= 5 { 1 } else { 2 };
            caps.push(rx_antennas);
        }

        caps
    }
}

/// RedCap measurement restrictions
pub struct RedCapMeasurementRestrictions {
    /// Maximum measurement bandwidth in MHz
    pub max_meas_bandwidth_mhz: u8,
    /// Relaxed measurement timing
    pub relaxed_meas_timing: bool,
}

impl RedCapMeasurementRestrictions {
    /// Creates measurement restrictions from RedCap mode
    pub fn from_mode(mode: &RedCapMode) -> Self {
        Self {
            max_meas_bandwidth_mhz: mode.max_bandwidth_mhz(),
            relaxed_meas_timing: mode.is_enabled(),
        }
    }

    /// Applies restrictions to a measurement bandwidth
    pub fn restrict_measurement_bandwidth(&self, bw_mhz: u8) -> u8 {
        bw_mhz.min(self.max_meas_bandwidth_mhz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redcap_mode_disabled() {
        let mode = RedCapMode::disabled();
        assert!(!mode.is_enabled());
        assert_eq!(mode.max_bandwidth_mhz(), 100);
        assert_eq!(mode.max_mimo_layers(), 4);
        assert!(!mode.is_half_duplex());
    }

    #[test]
    fn test_redcap_mode_rel17() {
        let mode = RedCapMode::rel17();
        assert!(mode.is_enabled());
        assert_eq!(mode.max_bandwidth_mhz(), 20);
        assert_eq!(mode.max_mimo_layers(), 1);
        assert!(mode.is_half_duplex());
        assert_eq!(mode.release, RedCapRelease::Rel17);
    }

    #[test]
    fn test_redcap_mode_rel18_ultra_reduced() {
        let mode = RedCapMode::rel18_ultra_reduced();
        assert!(mode.is_enabled());
        assert_eq!(mode.max_bandwidth_mhz(), 5);
        assert_eq!(mode.max_mimo_layers(), 1);
        assert!(mode.is_half_duplex());
        assert_eq!(mode.release, RedCapRelease::Rel18);
    }

    #[test]
    fn test_redcap_mode_rel18_enhanced() {
        let mode = RedCapMode::rel18_enhanced();
        assert!(mode.is_enabled());
        assert_eq!(mode.max_bandwidth_mhz(), 20);
        assert_eq!(mode.max_mimo_layers(), 2);
        assert!(!mode.is_half_duplex());
        assert_eq!(mode.release, RedCapRelease::Rel18);
    }

    #[test]
    fn test_bandwidth_restriction() {
        let mode = RedCapMode::rel17();
        assert_eq!(mode.apply_bandwidth_restriction(100), 20);
        assert_eq!(mode.apply_bandwidth_restriction(10), 10);

        let mode = RedCapMode::disabled();
        assert_eq!(mode.apply_bandwidth_restriction(100), 100);
    }

    #[test]
    fn test_encode_capabilities() {
        let mode = RedCapMode::rel17();
        let caps = mode.encode_capabilities();

        assert_eq!(caps.len(), 4);
        assert_eq!(caps[0], 1); // RedCap enabled
        assert_eq!(caps[1], 20); // Max bandwidth
        assert_eq!(caps[3], 17); // Release version
    }

    #[test]
    fn test_measurement_restrictions() {
        let mode = RedCapMode::rel17();
        let restrictions = RedCapMeasurementRestrictions::from_mode(&mode);

        assert_eq!(restrictions.max_meas_bandwidth_mhz, 20);
        assert!(restrictions.relaxed_meas_timing);

        assert_eq!(restrictions.restrict_measurement_bandwidth(100), 20);
        assert_eq!(restrictions.restrict_measurement_bandwidth(10), 10);
    }
}
