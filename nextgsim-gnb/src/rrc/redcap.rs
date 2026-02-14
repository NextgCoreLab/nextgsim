//! RedCap (Reduced Capability) UE Support (Rel-17)
//!
//! Implements support for RedCap UEs with reduced bandwidth, processing capabilities,
//! and antenna configurations as defined in 3GPP TS 38.306.
//!
//! # 3GPP Reference
//!
//! - TS 38.306: UE Radio Access Capabilities
//! - TS 38.331: RRC Protocol Specification (RedCap extensions)
//! - TS 38.101-1: UE Radio Transmission and Reception (RedCap requirements)

/// RedCap UE capabilities
///
/// Defines the reduced capabilities of a RedCap UE compared to a normal UE
#[derive(Debug, Clone)]
pub struct RedCapUeCapabilities {
    /// Maximum supported bandwidth in MHz (20 MHz for Rel-17, 5 MHz for Rel-18)
    pub max_bandwidth_mhz: u8,
    /// Half-duplex FDD support (cannot transmit and receive simultaneously)
    pub half_duplex_fdd: bool,
    /// Reduced UE processing time capability (relaxed HARQ timing)
    pub reduced_processing_time: bool,
    /// Maximum number of MIMO layers (typically 1-2 for RedCap)
    pub max_mimo_layers: u8,
    /// Reduced number of HARQ processes
    pub max_harq_processes: u8,
    /// Reduced peak data rate (Mbps)
    pub peak_data_rate_mbps: u16,
    /// Number of receive antennas (typically 1 or 2)
    pub num_rx_antennas: u8,
    /// Number of transmit antennas (typically 1)
    pub num_tx_antennas: u8,
    /// Support for carrier aggregation (typically false for RedCap)
    pub carrier_aggregation: bool,
    /// RedCap release version
    pub redcap_release: RedCapRelease,
}

/// RedCap release version
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RedCapRelease {
    /// Rel-17 RedCap (20 MHz max bandwidth)
    Rel17,
    /// Rel-18 RedCap (5 MHz option, further reduced capabilities)
    Rel18,
}

impl RedCapUeCapabilities {
    /// Creates Rel-17 RedCap capabilities (20 MHz max)
    pub fn rel17() -> Self {
        Self {
            max_bandwidth_mhz: 20,
            half_duplex_fdd: true,
            reduced_processing_time: true,
            max_mimo_layers: 1,
            max_harq_processes: 8,
            peak_data_rate_mbps: 150,
            num_rx_antennas: 1,
            num_tx_antennas: 1,
            carrier_aggregation: false,
            redcap_release: RedCapRelease::Rel17,
        }
    }

    /// Creates Rel-18 RedCap capabilities (5 MHz option)
    pub fn rel18_reduced() -> Self {
        Self {
            max_bandwidth_mhz: 5,
            half_duplex_fdd: true,
            reduced_processing_time: true,
            max_mimo_layers: 1,
            max_harq_processes: 4,
            peak_data_rate_mbps: 50,
            num_rx_antennas: 1,
            num_tx_antennas: 1,
            carrier_aggregation: false,
            redcap_release: RedCapRelease::Rel18,
        }
    }

    /// Creates Rel-18 RedCap capabilities (20 MHz option with improved features)
    pub fn rel18_enhanced() -> Self {
        Self {
            max_bandwidth_mhz: 20,
            half_duplex_fdd: false, // Full duplex support in some Rel-18 variants
            reduced_processing_time: true,
            max_mimo_layers: 2,
            max_harq_processes: 8,
            peak_data_rate_mbps: 200,
            num_rx_antennas: 2,
            num_tx_antennas: 1,
            carrier_aggregation: false,
            redcap_release: RedCapRelease::Rel18,
        }
    }

    /// Validates the RedCap capabilities
    pub fn validate(&self) -> bool {
        // Check bandwidth limits
        if self.max_bandwidth_mhz > 20 {
            return false;
        }

        // Check MIMO limits
        if self.max_mimo_layers > 2 {
            return false;
        }

        // Check antenna limits
        if self.num_rx_antennas > 2 || self.num_tx_antennas > 1 {
            return false;
        }

        // Carrier aggregation should be disabled for RedCap
        if self.carrier_aggregation {
            return false;
        }

        true
    }

    /// Returns true if this UE has very limited capabilities (Rel-18 5 MHz)
    pub fn is_ultra_reduced(&self) -> bool {
        self.max_bandwidth_mhz <= 5
    }

    /// Returns the bandwidth reduction factor compared to normal UE
    pub fn bandwidth_reduction_factor(&self) -> f32 {
        // Normal UE supports up to 100 MHz in FR1
        100.0 / self.max_bandwidth_mhz as f32
    }
}

/// RedCap RRC configuration
///
/// Contains RRC configuration parameters specific to RedCap UEs
#[derive(Debug, Clone)]
pub struct RedCapRrcConfig {
    /// UE capabilities
    pub capabilities: RedCapUeCapabilities,
    /// Bandwidth restriction in MHz for this UE
    pub restricted_bandwidth_mhz: u8,
    /// Half-duplex FDD gap configuration
    pub hd_fdd_gap_config: Option<HdFddGapConfig>,
    /// Relaxed processing time configuration
    pub relaxed_processing: Option<RelaxedProcessingConfig>,
    /// MIMO layer restriction
    pub mimo_restriction: MimoRestriction,
    /// eDRX configuration for Rel-18 RedCap (extended discontinuous reception)
    pub edrx_config: Option<EDrxConfig>,
}

/// Half-duplex FDD gap configuration
#[derive(Debug, Clone)]
pub struct HdFddGapConfig {
    /// Gap pattern type
    pub gap_pattern: u8,
    /// Gap length in slots
    pub gap_length_slots: u8,
    /// Gap repetition period in slots
    pub gap_period_slots: u16,
}

/// Relaxed processing time configuration
#[derive(Debug, Clone)]
pub struct RelaxedProcessingConfig {
    /// Processing time capability (1 or 2)
    pub processing_time_capability: u8,
    /// HARQ-ACK timing adjustment (slots)
    pub harq_timing_offset: u8,
    /// Extended T_proc for Rel-18 (additional processing time in ms)
    pub extended_t_proc_ms: Option<u8>,
}

/// MIMO layer restriction for RedCap UE
#[derive(Debug, Clone, Copy)]
pub enum MimoRestriction {
    /// Single layer transmission only
    SingleLayer,
    /// Up to 2 layers
    TwoLayers,
}

/// eDRX (extended Discontinuous Reception) configuration for Rel-18 RedCap
///
/// Enables ultra-long sleep cycles for IoT/wearable devices
#[derive(Debug, Clone)]
pub struct EDrxConfig {
    /// eDRX cycle length in seconds (up to 10485.76s per TS 24.008)
    pub edrx_cycle_s: f32,
    /// Paging Time Window (PTW) in seconds
    pub ptw_s: f32,
    /// Whether UE is allowed to skip paging occasions within PTW
    pub skip_paging_allowed: bool,
}

impl RedCapRrcConfig {
    /// Creates a RedCap RRC configuration from capabilities
    pub fn from_capabilities(capabilities: RedCapUeCapabilities) -> Self {
        let hd_fdd_gap_config = if capabilities.half_duplex_fdd {
            Some(HdFddGapConfig {
                gap_pattern: 0,
                gap_length_slots: 4,
                gap_period_slots: 20,
            })
        } else {
            None
        };

        let relaxed_processing = if capabilities.reduced_processing_time {
            // Rel-18 RedCap gets extended T_proc for ultra-low complexity
            let extended_t_proc = if capabilities.redcap_release == RedCapRelease::Rel18 && capabilities.max_bandwidth_mhz <= 5 {
                Some(8) // 8ms extended processing time for Rel-18 5MHz variant
            } else {
                None
            };
            Some(RelaxedProcessingConfig {
                processing_time_capability: 2,
                harq_timing_offset: 4,
                extended_t_proc_ms: extended_t_proc,
            })
        } else {
            None
        };

        let mimo_restriction = if capabilities.max_mimo_layers == 1 {
            MimoRestriction::SingleLayer
        } else {
            MimoRestriction::TwoLayers
        };

        // Rel-18 RedCap supports eDRX for power savings
        let edrx_config = if capabilities.redcap_release == RedCapRelease::Rel18 {
            Some(EDrxConfig {
                edrx_cycle_s: 20.48, // 20.48s eDRX cycle for wearables/sensors
                ptw_s: 2.56,          // 2.56s Paging Time Window
                skip_paging_allowed: true,
            })
        } else {
            None
        };

        Self {
            restricted_bandwidth_mhz: capabilities.max_bandwidth_mhz,
            capabilities,
            hd_fdd_gap_config,
            relaxed_processing,
            mimo_restriction,
            edrx_config,
        }
    }

    /// Applies RedCap restrictions to a UE configuration
    pub fn apply_restrictions(&self) -> RedCapRestrictions {
        RedCapRestrictions {
            max_bandwidth_mhz: self.restricted_bandwidth_mhz,
            max_mimo_layers: match self.mimo_restriction {
                MimoRestriction::SingleLayer => 1,
                MimoRestriction::TwoLayers => 2,
            },
            half_duplex_fdd: self.hd_fdd_gap_config.is_some(),
            harq_timing_offset: self
                .relaxed_processing
                .as_ref()
                .map(|rp| rp.harq_timing_offset)
                .unwrap_or(0),
        }
    }
}

/// Applied RedCap restrictions for a UE
#[derive(Debug, Clone)]
pub struct RedCapRestrictions {
    /// Maximum bandwidth in MHz
    pub max_bandwidth_mhz: u8,
    /// Maximum MIMO layers
    pub max_mimo_layers: u8,
    /// Half-duplex FDD enabled
    pub half_duplex_fdd: bool,
    /// HARQ timing offset in slots
    pub harq_timing_offset: u8,
}

/// RedCap UE processor
///
/// Handles RedCap-specific RRC processing at the gNB
#[derive(Debug, Clone)]
pub struct RedCapProcessor {
    /// RedCap configuration for this UE
    config: Option<RedCapRrcConfig>,
}

impl RedCapProcessor {
    /// Creates a new RedCap processor
    pub fn new() -> Self {
        Self { config: None }
    }

    /// Checks if this is a RedCap UE
    pub fn is_redcap_ue(&self) -> bool {
        self.config.is_some()
    }

    /// Configures RedCap capabilities for this UE
    pub fn configure(&mut self, capabilities: RedCapUeCapabilities) {
        let config = RedCapRrcConfig::from_capabilities(capabilities);
        self.config = Some(config);
    }

    /// Gets the current RedCap configuration
    pub fn get_redcap_config(&self) -> Option<&RedCapRrcConfig> {
        self.config.as_ref()
    }

    /// Gets applied restrictions
    pub fn get_restrictions(&self) -> Option<RedCapRestrictions> {
        self.config.as_ref().map(RedCapRrcConfig::apply_restrictions)
    }

    /// Applies bandwidth restriction to a measurement
    pub fn restrict_bandwidth(&self, requested_bw_mhz: u8) -> u8 {
        if let Some(config) = &self.config {
            requested_bw_mhz.min(config.restricted_bandwidth_mhz)
        } else {
            requested_bw_mhz
        }
    }

    /// Calculates HARQ timing adjustment for this RedCap UE
    pub fn get_harq_timing_adjustment(&self) -> u8 {
        self.config
            .as_ref()
            .and_then(|c| c.relaxed_processing.as_ref())
            .map(|rp| rp.harq_timing_offset)
            .unwrap_or(0)
    }

    /// Determines if half-duplex FDD gaps are needed
    pub fn needs_hd_fdd_gaps(&self) -> bool {
        self.config
            .as_ref()
            .and_then(|c| c.hd_fdd_gap_config.as_ref())
            .is_some()
    }

    /// Gets the maximum number of MIMO layers for this UE
    pub fn get_max_mimo_layers(&self) -> u8 {
        self.config
            .as_ref()
            .map(|c| match c.mimo_restriction {
                MimoRestriction::SingleLayer => 1,
                MimoRestriction::TwoLayers => 2,
            })
            .unwrap_or(4) // Default for normal UE
    }
}

impl Default for RedCapProcessor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_redcap_rel17_capabilities() {
        let cap = RedCapUeCapabilities::rel17();
        assert_eq!(cap.max_bandwidth_mhz, 20);
        assert_eq!(cap.max_mimo_layers, 1);
        assert!(cap.half_duplex_fdd);
        assert!(cap.validate());
    }

    #[test]
    fn test_redcap_rel18_reduced_capabilities() {
        let cap = RedCapUeCapabilities::rel18_reduced();
        assert_eq!(cap.max_bandwidth_mhz, 5);
        assert_eq!(cap.max_mimo_layers, 1);
        assert!(cap.is_ultra_reduced());
        assert!(cap.validate());
    }

    #[test]
    fn test_redcap_rel18_enhanced_capabilities() {
        let cap = RedCapUeCapabilities::rel18_enhanced();
        assert_eq!(cap.max_bandwidth_mhz, 20);
        assert_eq!(cap.max_mimo_layers, 2);
        assert!(!cap.is_ultra_reduced());
        assert!(cap.validate());
    }

    #[test]
    fn test_bandwidth_reduction_factor() {
        let cap = RedCapUeCapabilities::rel17();
        assert_eq!(cap.bandwidth_reduction_factor(), 5.0); // 100/20 = 5

        let cap = RedCapUeCapabilities::rel18_reduced();
        assert_eq!(cap.bandwidth_reduction_factor(), 20.0); // 100/5 = 20
    }

    #[test]
    fn test_redcap_config_from_capabilities() {
        let cap = RedCapUeCapabilities::rel17();
        let config = RedCapRrcConfig::from_capabilities(cap);

        assert_eq!(config.restricted_bandwidth_mhz, 20);
        assert!(config.hd_fdd_gap_config.is_some());
        assert!(config.relaxed_processing.is_some());
        assert!(matches!(
            config.mimo_restriction,
            MimoRestriction::SingleLayer
        ));
    }

    #[test]
    fn test_redcap_restrictions() {
        let cap = RedCapUeCapabilities::rel17();
        let config = RedCapRrcConfig::from_capabilities(cap);
        let restrictions = config.apply_restrictions();

        assert_eq!(restrictions.max_bandwidth_mhz, 20);
        assert_eq!(restrictions.max_mimo_layers, 1);
        assert!(restrictions.half_duplex_fdd);
        assert_eq!(restrictions.harq_timing_offset, 4);
    }

    #[test]
    fn test_redcap_processor() {
        let mut processor = RedCapProcessor::new();
        assert!(!processor.is_redcap_ue());

        let cap = RedCapUeCapabilities::rel17();
        processor.configure(cap);

        assert!(processor.is_redcap_ue());
        assert_eq!(processor.get_max_mimo_layers(), 1);
        assert_eq!(processor.get_harq_timing_adjustment(), 4);
        assert!(processor.needs_hd_fdd_gaps());
    }

    #[test]
    fn test_bandwidth_restriction() {
        let mut processor = RedCapProcessor::new();
        let cap = RedCapUeCapabilities::rel17();
        processor.configure(cap);

        assert_eq!(processor.restrict_bandwidth(100), 20);
        assert_eq!(processor.restrict_bandwidth(10), 10);
    }

    #[test]
    fn test_invalid_capabilities() {
        let mut cap = RedCapUeCapabilities::rel17();
        cap.max_bandwidth_mhz = 50; // Exceeds RedCap limit
        assert!(!cap.validate());

        let mut cap = RedCapUeCapabilities::rel17();
        cap.max_mimo_layers = 4; // Exceeds RedCap limit
        assert!(!cap.validate());

        let mut cap = RedCapUeCapabilities::rel17();
        cap.carrier_aggregation = true; // Not allowed for RedCap
        assert!(!cap.validate());
    }
}
