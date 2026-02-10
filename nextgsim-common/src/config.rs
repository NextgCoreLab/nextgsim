//! Configuration structures for gNB and UE
//!
//! This module provides configuration types for the nextgsim simulator,
//! including gNB (gNodeB) and UE (User Equipment) configurations.

use std::fmt;
use std::net::IpAddr;

use serde::{Deserialize, Serialize};

use crate::types::{NetworkSlice, Plmn, SNssai, Supi};

/// AMF (Access and Mobility Management Function) configuration.
///
/// Defines the connection parameters for an AMF that the gNB connects to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AmfConfig {
    /// IP address of the AMF
    pub address: IpAddr,
    /// SCTP port of the AMF (typically 38412)
    pub port: u16,
}

impl AmfConfig {
    /// Creates a new AMF configuration.
    ///
    /// # Arguments
    /// * `address` - IP address of the AMF
    /// * `port` - SCTP port of the AMF
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self { address, port }
    }
}

/// gNB (gNodeB) configuration.
///
/// Contains all configuration parameters for a 5G base station.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GnbConfig {
    /// NR Cell Identity (36-bit value)
    pub nci: u64,
    /// gNB ID length in bits (22-32)
    pub gnb_id_length: u8,
    /// Public Land Mobile Network identifier
    pub plmn: Plmn,
    /// Tracking Area Code (24-bit)
    pub tac: u32,
    /// Network Slice Selection Assistance Information
    pub nssai: Vec<SNssai>,
    /// List of AMF configurations
    pub amf_configs: Vec<AmfConfig>,
    /// IP address for RLS link layer
    pub link_ip: IpAddr,
    /// IP address for NGAP interface
    pub ngap_ip: IpAddr,
    /// IP address for GTP-U interface
    pub gtp_ip: IpAddr,
    /// Advertised GTP IP address (for NAT scenarios)
    pub gtp_advertise_ip: Option<IpAddr>,
    /// Whether to ignore SCTP stream IDs
    pub ignore_stream_ids: bool,
    /// UPF GTP-U address for data plane forwarding (if None, uses loopback mode)
    #[serde(default)]
    pub upf_addr: Option<IpAddr>,
    /// UPF GTP-U port (default: 2152)
    #[serde(default = "default_gtp_port")]
    pub upf_port: u16,
    /// Post-quantum cryptography configuration
    #[serde(default)]
    pub pqc_config: PqcConfig,
    /// NTN (Non-Terrestrial Network) configuration (optional)
    #[serde(default)]
    pub ntn_config: Option<NtnConfig>,
    /// MBS (Multicast/Broadcast) support enabled (Rel-17, TS 23.247)
    #[serde(default)]
    pub mbs_enabled: bool,
    /// ProSe/Sidelink support enabled (Rel-17)
    #[serde(default)]
    pub prose_enabled: bool,
    /// LCS/Positioning support enabled (Rel-17)
    #[serde(default)]
    pub lcs_enabled: bool,
    /// SNPN (Standalone Non-Public Network) configuration
    #[serde(default)]
    pub snpn_config: Option<SnpnConfig>,
}

fn default_gtp_port() -> u16 {
    2152
}

impl GnbConfig {
    /// Returns the gNB ID extracted from the NCI.
    ///
    /// The gNB ID is the upper bits of the NCI, with the number of bits
    /// determined by `gnb_id_length`.
    pub fn gnb_id(&self) -> u32 {
        let shift = 36 - self.gnb_id_length as i64;
        ((self.nci & 0xFFFFFFFFF) >> shift) as u32
    }

    /// Returns the Cell ID extracted from the NCI.
    ///
    /// The Cell ID is the lower bits of the NCI, with the number of bits
    /// being (36 - gnb_id_length).
    pub fn cell_id(&self) -> u32 {
        let cell_id_bits = 36 - self.gnb_id_length;
        let mask = (1u64 << cell_id_bits) - 1;
        (self.nci & mask) as u32
    }
}

/// Operator key type for authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum OpType {
    /// Operator key (OP) - needs to be converted to OPc
    Op,
    /// Operator key derived (OPc) - used directly
    #[default]
    Opc,
}


/// Supported NAS security algorithms.
///
/// Defines which integrity and ciphering algorithms the UE supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SupportedAlgs {
    /// NIA1 (SNOW3G-based integrity)
    pub nia1: bool,
    /// NIA2 (AES-based integrity)
    pub nia2: bool,
    /// NIA3 (ZUC-based integrity)
    pub nia3: bool,
    /// NEA1 (SNOW3G-based ciphering)
    pub nea1: bool,
    /// NEA2 (AES-based ciphering)
    pub nea2: bool,
    /// NEA3 (ZUC-based ciphering)
    pub nea3: bool,
}

impl Default for SupportedAlgs {
    fn default() -> Self {
        Self {
            nia1: true,
            nia2: true,
            nia3: true,
            nea1: true,
            nea2: true,
            nea3: true,
        }
    }
}

// ============================================================================
// 6G Post-Quantum Cryptography (PQC) Configuration
// ============================================================================

/// Post-quantum Key Encapsulation Mechanism (KEM) algorithm.
///
/// Defines the KEM algorithm to use for post-quantum secure key exchange.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum KemAlgorithm {
    /// No post-quantum KEM (classical only)
    #[default]
    None,
    /// CRYSTALS-Kyber (NIST standard)
    Kyber512,
    /// CRYSTALS-Kyber-768
    Kyber768,
    /// CRYSTALS-Kyber-1024
    Kyber1024,
    /// NTRU (alternative PQC algorithm)
    Ntru,
    /// SABER (alternative PQC algorithm)
    Saber,
}


impl fmt::Display for KemAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KemAlgorithm::None => write!(f, "none"),
            KemAlgorithm::Kyber512 => write!(f, "kyber512"),
            KemAlgorithm::Kyber768 => write!(f, "kyber768"),
            KemAlgorithm::Kyber1024 => write!(f, "kyber1024"),
            KemAlgorithm::Ntru => write!(f, "ntru"),
            KemAlgorithm::Saber => write!(f, "saber"),
        }
    }
}

/// Post-quantum signature algorithm.
///
/// Defines the signature algorithm to use for post-quantum secure authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum SignAlgorithm {
    /// No post-quantum signatures (classical only)
    #[default]
    None,
    /// CRYSTALS-Dilithium (NIST standard)
    Dilithium2,
    /// CRYSTALS-Dilithium-3
    Dilithium3,
    /// CRYSTALS-Dilithium-5
    Dilithium5,
    /// FALCON (alternative PQC algorithm)
    Falcon512,
    /// FALCON-1024
    Falcon1024,
    /// SPHINCS+ (hash-based signatures)
    SphincsSha256,
}


impl fmt::Display for SignAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignAlgorithm::None => write!(f, "none"),
            SignAlgorithm::Dilithium2 => write!(f, "dilithium2"),
            SignAlgorithm::Dilithium3 => write!(f, "dilithium3"),
            SignAlgorithm::Dilithium5 => write!(f, "dilithium5"),
            SignAlgorithm::Falcon512 => write!(f, "falcon512"),
            SignAlgorithm::Falcon1024 => write!(f, "falcon1024"),
            SignAlgorithm::SphincsSha256 => write!(f, "sphincs-sha256"),
        }
    }
}

/// Hybrid mode for combining classical and post-quantum cryptography.
///
/// Defines how classical and PQC algorithms are combined for transitional security.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum HybridMode {
    /// Classical cryptography only (no PQC)
    #[default]
    ClassicalOnly,
    /// Post-quantum cryptography only (no classical)
    PqcOnly,
    /// Hybrid: use both classical and PQC in parallel
    HybridParallel,
    /// Hybrid: concatenate outputs of classical and PQC
    HybridConcatenate,
    /// Hybrid: XOR outputs of classical and PQC
    HybridXor,
}


impl fmt::Display for HybridMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HybridMode::ClassicalOnly => write!(f, "classical-only"),
            HybridMode::PqcOnly => write!(f, "pqc-only"),
            HybridMode::HybridParallel => write!(f, "hybrid-parallel"),
            HybridMode::HybridConcatenate => write!(f, "hybrid-concatenate"),
            HybridMode::HybridXor => write!(f, "hybrid-xor"),
        }
    }
}

/// Post-quantum cryptography configuration.
///
/// Defines the PQC algorithms and modes to use for quantum-resistant security.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub struct PqcConfig {
    /// Whether PQC is enabled
    pub enabled: bool,
    /// KEM algorithm for key exchange
    pub kem_algorithm: KemAlgorithm,
    /// Signature algorithm for authentication
    pub sign_algorithm: SignAlgorithm,
    /// Hybrid mode for combining classical and PQC
    pub hybrid_mode: HybridMode,
}


impl PqcConfig {
    /// Creates a new PQC configuration with specified algorithms.
    pub fn new(kem: KemAlgorithm, sign: SignAlgorithm, mode: HybridMode) -> Self {
        Self {
            enabled: kem != KemAlgorithm::None || sign != SignAlgorithm::None,
            kem_algorithm: kem,
            sign_algorithm: sign,
            hybrid_mode: mode,
        }
    }
}

/// NTN (Non-Terrestrial Network) configuration.
///
/// Configures gNB for satellite-based 5G/6G operation (3GPP TS 38.300 Rel-17).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NtnConfig {
    /// Satellite type (LEO, MEO, GEO, HAPS)
    pub satellite_type: String,
    /// Satellite ID
    pub satellite_id: u32,
    /// One-way propagation delay in microseconds
    pub propagation_delay_us: u64,
    /// Common timing advance in microseconds
    pub common_ta_us: u64,
    /// K-offset for HARQ timing (slots)
    pub k_offset: u16,
    /// Cell center latitude in degrees
    pub cell_center_lat: f64,
    /// Cell center longitude in degrees
    pub cell_center_lon: f64,
    /// Cell radius in km
    pub cell_radius_km: f64,
    /// Whether the cell footprint is earth-fixed
    #[serde(default = "default_true")]
    pub earth_fixed: bool,
    /// Enable autonomous TA calculation by UE
    #[serde(default)]
    pub autonomous_ta: bool,
    /// Maximum Doppler shift in Hz
    #[serde(default)]
    pub max_doppler_hz: f64,
}

fn default_true() -> bool { true }

/// SNPN (Standalone Non-Public Network) configuration (Rel-17, TS 23.501).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnpnConfig {
    /// Network Identifier (NID) for the SNPN
    pub nid: String,
    /// Closed Access Group (CAG) ID list
    #[serde(default)]
    pub cag_ids: Vec<u32>,
    /// Whether onboarding is allowed for non-subscribed UEs
    #[serde(default)]
    pub onboarding_enabled: bool,
}

/// UE Route Selection Policy rule (Rel-17, TS 24.526).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrspRule {
    /// Rule precedence (lower = higher priority)
    pub precedence: u8,
    /// Traffic descriptor (app ID or IP descriptor)
    pub traffic_descriptor: String,
    /// Route selection descriptors
    pub route_descriptors: Vec<RouteDescriptor>,
}

/// Route selection descriptor for URSP.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDescriptor {
    /// Preferred S-NSSAI
    pub s_nssai: Option<SNssai>,
    /// Preferred DNN
    pub dnn: Option<String>,
    /// PDU session type preference
    pub session_type: Option<PduSessionType>,
    /// SSC mode preference (1, 2, or 3)
    pub ssc_mode: Option<u8>,
}

/// PIN (Personal IoT Network) role (Rel-18, TS 23.542).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinRole {
    /// PIN element (IoT device)
    PinElement,
    /// PIN gateway (relay to network)
    PinGateway,
    /// PIN management entity
    PinManagement,
}

/// PDU session type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum PduSessionType {
    /// IPv4 PDU session
    #[default]
    Ipv4,
    /// IPv6 PDU session
    Ipv6,
    /// IPv4v6 (dual-stack) PDU session
    Ipv4v6,
    /// Unstructured PDU session
    Unstructured,
    /// Ethernet PDU session
    Ethernet,
}


/// PDU session configuration.
///
/// Defines the parameters for establishing a PDU session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// PDU session type
    #[serde(rename = "type")]
    pub session_type: PduSessionType,
    /// S-NSSAI for the session (optional)
    pub s_nssai: Option<SNssai>,
    /// Access Point Name (optional)
    pub apn: Option<String>,
    /// Whether this is an emergency session
    pub is_emergency: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_type: PduSessionType::Ipv4,
            s_nssai: None,
            apn: None,
            is_emergency: false,
        }
    }
}

/// UE (User Equipment) configuration.
///
/// Contains all configuration parameters for a 5G UE.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UeConfig {
    /// Subscription Permanent Identifier (optional)
    pub supi: Option<Supi>,
    /// SUCI protection scheme (0: null, 1: Profile A, 2: Profile B)
    pub protection_scheme: u8,
    /// Home network public key identifier
    pub home_network_public_key_id: u8,
    /// Home network public key for SUCI calculation
    pub home_network_public_key: Vec<u8>,
    /// Routing indicator (optional)
    pub routing_indicator: Option<String>,
    /// Home PLMN
    pub hplmn: Plmn,
    /// Subscriber key K (128-bit)
    pub key: [u8; 16],
    /// Operator key OP or OPc (128-bit)
    pub op: [u8; 16],
    /// Type of operator key (OP or OPc)
    pub op_type: OpType,
    /// Authentication Management Field (16-bit)
    pub amf: [u8; 2],
    /// International Mobile Equipment Identity (optional)
    pub imei: Option<String>,
    /// IMEI Software Version (optional)
    pub imei_sv: Option<String>,
    /// Supported NAS security algorithms
    pub supported_algs: SupportedAlgs,
    /// List of gNB addresses to search for
    pub gnb_search_list: Vec<String>,
    /// Default PDU sessions to establish
    pub sessions: Vec<SessionConfig>,
    /// Configured NSSAI (network slices)
    pub configured_nssai: NetworkSlice,
    /// TUN interface name (optional)
    pub tun_name: Option<String>,
    /// Post-quantum cryptography configuration
    #[serde(default)]
    pub pqc_config: PqcConfig,
    /// RedCap (Reduced Capability) UE indication (Rel-17, TS 38.101)
    #[serde(default)]
    pub redcap: bool,
    /// SNPN access configuration
    #[serde(default)]
    pub snpn_config: Option<SnpnConfig>,
    /// ProSe/Sidelink capability
    #[serde(default)]
    pub prose_enabled: bool,
    /// UE Route Selection Policy rules
    #[serde(default)]
    pub ursp_rules: Vec<UrspRule>,
    /// PIN (Personal IoT Network) role
    #[serde(default)]
    pub pin_role: Option<PinRole>,
}

impl Default for UeConfig {
    fn default() -> Self {
        Self {
            supi: None,
            protection_scheme: 0,
            home_network_public_key_id: 0,
            home_network_public_key: Vec::new(),
            routing_indicator: None,
            hplmn: Plmn::default(),
            key: [0u8; 16],
            op: [0u8; 16],
            op_type: OpType::default(),
            amf: [0x80, 0x00], // Default AMF value per 3GPP
            imei: None,
            imei_sv: None,
            supported_algs: SupportedAlgs::default(),
            gnb_search_list: Vec::new(),
            sessions: Vec::new(),
            configured_nssai: NetworkSlice::new(),
            tun_name: None,
            pqc_config: PqcConfig::default(),
            redcap: false,
            snpn_config: None,
            prose_enabled: false,
            ursp_rules: Vec::new(),
            pin_role: None,
        }
    }
}

// ============================================================================
// YAML Configuration Parsing
// ============================================================================

use crate::error::Error;
use std::fs;
use std::path::Path;

impl GnbConfig {
    /// Parses a gNB configuration from a YAML string.
    ///
    /// # Arguments
    /// * `yaml` - YAML string containing the gNB configuration
    ///
    /// # Returns
    /// * `Ok(GnbConfig)` - Successfully parsed configuration
    /// * `Err(Error)` - YAML parsing error
    ///
    /// # Example
    /// ```
    /// use nextgsim_common::GnbConfig;
    ///
    /// let yaml = r#"
    /// nci: 16
    /// gnb_id_length: 24
    /// plmn:
    ///   mcc: 310
    ///   mnc: 410
    ///   long_mnc: false
    /// tac: 1
    /// nssai: []
    /// amf_configs: []
    /// link_ip: 127.0.0.1
    /// ngap_ip: 127.0.0.1
    /// gtp_ip: 127.0.0.1
    /// ignore_stream_ids: false
    /// "#;
    ///
    /// let config = GnbConfig::from_yaml(yaml).unwrap();
    /// assert_eq!(config.tac, 1);
    /// ```
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml::from_str(yaml)?)
    }

    /// Loads a gNB configuration from a YAML file.
    ///
    /// # Arguments
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Returns
    /// * `Ok(GnbConfig)` - Successfully loaded configuration
    /// * `Err(Error)` - File I/O or YAML parsing error
    ///
    /// # Example
    /// ```no_run
    /// use nextgsim_common::GnbConfig;
    ///
    /// let config = GnbConfig::from_yaml_file("config/gnb.yaml").unwrap();
    /// ```
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let contents = fs::read_to_string(path)?;
        Self::from_yaml(&contents)
    }

    /// Serializes the gNB configuration to a YAML string.
    ///
    /// # Returns
    /// * `Ok(String)` - YAML representation of the configuration
    /// * `Err(Error)` - Serialization error
    pub fn to_yaml(&self) -> Result<String, Error> {
        Ok(serde_yaml::to_string(self)?)
    }
}

impl UeConfig {
    /// Parses a UE configuration from a YAML string.
    ///
    /// # Arguments
    /// * `yaml` - YAML string containing the UE configuration
    ///
    /// # Returns
    /// * `Ok(UeConfig)` - Successfully parsed configuration
    /// * `Err(Error)` - YAML parsing error
    ///
    /// # Example
    /// ```
    /// use nextgsim_common::UeConfig;
    ///
    /// let yaml = r#"
    /// protection_scheme: 0
    /// home_network_public_key_id: 0
    /// home_network_public_key: []
    /// hplmn:
    ///   mcc: 310
    ///   mnc: 410
    ///   long_mnc: false
    /// key: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    /// op: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    /// op_type: Opc
    /// amf: [128, 0]
    /// supported_algs:
    ///   nia1: true
    ///   nia2: true
    ///   nia3: true
    ///   nea1: true
    ///   nea2: true
    ///   nea3: true
    /// gnb_search_list: []
    /// sessions: []
    /// configured_nssai:
    ///   slices: []
    /// "#;
    ///
    /// let config = UeConfig::from_yaml(yaml).unwrap();
    /// assert_eq!(config.protection_scheme, 0);
    /// ```
    pub fn from_yaml(yaml: &str) -> Result<Self, Error> {
        Ok(serde_yaml::from_str(yaml)?)
    }

    /// Loads a UE configuration from a YAML file.
    ///
    /// # Arguments
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Returns
    /// * `Ok(UeConfig)` - Successfully loaded configuration
    /// * `Err(Error)` - File I/O or YAML parsing error
    ///
    /// # Example
    /// ```no_run
    /// use nextgsim_common::UeConfig;
    ///
    /// let config = UeConfig::from_yaml_file("config/ue.yaml").unwrap();
    /// ```
    pub fn from_yaml_file<P: AsRef<Path>>(path: P) -> Result<Self, Error> {
        let contents = fs::read_to_string(path)?;
        Self::from_yaml(&contents)
    }

    /// Serializes the UE configuration to a YAML string.
    ///
    /// # Returns
    /// * `Ok(String)` - YAML representation of the configuration
    /// * `Err(Error)` - Serialization error
    pub fn to_yaml(&self) -> Result<String, Error> {
        Ok(serde_yaml::to_string(self)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    #[test]
    fn test_amf_config_new() {
        let config = AmfConfig::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 38412);
        assert_eq!(config.address, IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)));
        assert_eq!(config.port, 38412);
    }

    #[test]
    fn test_gnb_config_gnb_id() {
        let config = GnbConfig {
            nci: 0x000000001, // NCI with gnb_id in upper bits
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, true),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            ngap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        };
        // With gnb_id_length=24, cell_id is 12 bits
        // NCI = 0x000000001, gnb_id = upper 24 bits = 0, cell_id = lower 12 bits = 1
        assert_eq!(config.gnb_id(), 0);
        assert_eq!(config.cell_id(), 1);
    }

    #[test]
    fn test_gnb_config_gnb_id_with_value() {
        let config = GnbConfig {
            nci: 0x123456789, // 36-bit NCI
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, true),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            ngap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        };
        // gnb_id_length=24, so gnb_id is upper 24 bits, cell_id is lower 12 bits
        // NCI = 0x123456789
        // gnb_id = 0x123456789 >> 12 = 0x123456
        // cell_id = 0x123456789 & 0xFFF = 0x789
        assert_eq!(config.gnb_id(), 0x123456);
        assert_eq!(config.cell_id(), 0x789);
    }

    #[test]
    fn test_op_type_default() {
        assert_eq!(OpType::default(), OpType::Opc);
    }

    #[test]
    fn test_supported_algs_default() {
        let algs = SupportedAlgs::default();
        assert!(algs.nia1);
        assert!(algs.nia2);
        assert!(algs.nia3);
        assert!(algs.nea1);
        assert!(algs.nea2);
        assert!(algs.nea3);
    }

    #[test]
    fn test_pdu_session_type_default() {
        assert_eq!(PduSessionType::default(), PduSessionType::Ipv4);
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.session_type, PduSessionType::Ipv4);
        assert!(config.s_nssai.is_none());
        assert!(config.apn.is_none());
        assert!(!config.is_emergency);
    }

    #[test]
    fn test_ue_config_default() {
        let config = UeConfig::default();
        assert!(config.supi.is_none());
        assert_eq!(config.protection_scheme, 0);
        assert_eq!(config.home_network_public_key_id, 0);
        assert!(config.home_network_public_key.is_empty());
        assert!(config.routing_indicator.is_none());
        assert_eq!(config.hplmn, Plmn::default());
        assert_eq!(config.key, [0u8; 16]);
        assert_eq!(config.op, [0u8; 16]);
        assert_eq!(config.op_type, OpType::Opc);
        assert_eq!(config.amf, [0x80, 0x00]);
        assert!(config.imei.is_none());
        assert!(config.imei_sv.is_none());
        assert!(config.gnb_search_list.is_empty());
        assert!(config.sessions.is_empty());
        assert!(config.configured_nssai.is_empty());
        assert!(config.tun_name.is_none());
    }

    // YAML parsing tests

    #[test]
    fn test_gnb_config_from_yaml() {
        let yaml = r#"
nci: 16
gnb_id_length: 24
plmn:
  mcc: 310
  mnc: 410
  long_mnc: false
tac: 1
nssai:
  - sst: 1
    sd: [0, 0, 1]
amf_configs:
  - address: 127.0.0.1
    port: 38412
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
ignore_stream_ids: false
"#;
        let config = GnbConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.nci, 16);
        assert_eq!(config.gnb_id_length, 24);
        assert_eq!(config.plmn.mcc, 310);
        assert_eq!(config.plmn.mnc, 410);
        assert_eq!(config.tac, 1);
        assert_eq!(config.nssai.len(), 1);
        assert_eq!(config.amf_configs.len(), 1);
        assert_eq!(config.amf_configs[0].port, 38412);
        assert!(!config.ignore_stream_ids);
    }

    #[test]
    fn test_gnb_config_to_yaml() {
        let config = GnbConfig {
            nci: 16,
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![AmfConfig::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 38412)],
            link_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            ngap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        };
        let yaml = config.to_yaml().unwrap();
        assert!(yaml.contains("nci: 16"));
        assert!(yaml.contains("gnb_id_length: 24"));
        assert!(yaml.contains("tac: 1"));
    }

    #[test]
    fn test_gnb_config_roundtrip() {
        let original = GnbConfig {
            nci: 0x123456789,
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, true),
            tac: 100,
            nssai: vec![SNssai::with_sd_u32(1, 0x010203)],
            amf_configs: vec![AmfConfig::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)), 38412)],
            link_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
            ngap_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 2)),
            gtp_ip: IpAddr::V4(Ipv4Addr::new(10, 0, 0, 3)),
            gtp_advertise_ip: Some(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1))),
            ignore_stream_ids: true,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        };
        let yaml = original.to_yaml().unwrap();
        let parsed = GnbConfig::from_yaml(&yaml).unwrap();
        assert_eq!(original.nci, parsed.nci);
        assert_eq!(original.gnb_id_length, parsed.gnb_id_length);
        assert_eq!(original.tac, parsed.tac);
        assert_eq!(original.ignore_stream_ids, parsed.ignore_stream_ids);
    }

    #[test]
    fn test_ue_config_from_yaml() {
        let yaml = r#"
protection_scheme: 0
home_network_public_key_id: 1
home_network_public_key: [1, 2, 3, 4]
hplmn:
  mcc: 310
  mnc: 410
  long_mnc: false
key: [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
op: [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
op_type: Opc
amf: [128, 0]
supported_algs:
  nia1: true
  nia2: true
  nia3: false
  nea1: true
  nea2: true
  nea3: false
gnb_search_list:
  - 127.0.0.1
sessions: []
configured_nssai:
  slices:
    - sst: 1
      sd: [0, 0, 1]
"#;
        let config = UeConfig::from_yaml(yaml).unwrap();
        assert_eq!(config.protection_scheme, 0);
        assert_eq!(config.home_network_public_key_id, 1);
        assert_eq!(config.home_network_public_key, vec![1, 2, 3, 4]);
        assert_eq!(config.hplmn.mcc, 310);
        assert_eq!(config.op_type, OpType::Opc);
        assert!(config.supported_algs.nia1);
        assert!(!config.supported_algs.nia3);
        assert_eq!(config.gnb_search_list.len(), 1);
    }

    #[test]
    fn test_ue_config_to_yaml() {
        let config = UeConfig::default();
        let yaml = config.to_yaml().unwrap();
        assert!(yaml.contains("protection_scheme: 0"));
        assert!(yaml.contains("op_type: Opc"));
    }

    #[test]
    fn test_ue_config_roundtrip() {
        let original = UeConfig {
            supi: Some(Supi::imsi("310410123456789")),
            protection_scheme: 1,
            home_network_public_key_id: 2,
            home_network_public_key: vec![0xAB, 0xCD],
            routing_indicator: Some("1234".to_string()),
            hplmn: Plmn::new(310, 410, true),
            key: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            op: [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
            op_type: OpType::Op,
            amf: [0x90, 0x01],
            imei: Some("123456789012345".to_string()),
            imei_sv: Some("1234567890123456".to_string()),
            supported_algs: SupportedAlgs::default(),
            gnb_search_list: vec!["192.168.1.1".to_string()],
            sessions: vec![SessionConfig::default()],
            configured_nssai: NetworkSlice::new(),
            tun_name: Some("tun0".to_string()),
            pqc_config: PqcConfig::default(),
            redcap: false,
            snpn_config: None,
            prose_enabled: false,
            ursp_rules: vec![],
            pin_role: None,
        };
        let yaml = original.to_yaml().unwrap();
        let parsed = UeConfig::from_yaml(&yaml).unwrap();
        assert_eq!(original.protection_scheme, parsed.protection_scheme);
        assert_eq!(original.home_network_public_key_id, parsed.home_network_public_key_id);
        assert_eq!(original.key, parsed.key);
        assert_eq!(original.op, parsed.op);
        assert_eq!(original.op_type, parsed.op_type);
    }

    #[test]
    fn test_gnb_config_from_yaml_invalid() {
        let yaml = "invalid: yaml: content: [";
        let result = GnbConfig::from_yaml(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_ue_config_from_yaml_invalid() {
        let yaml = "not valid yaml at all {{{";
        let result = UeConfig::from_yaml(yaml);
        assert!(result.is_err());
    }

    #[test]
    fn test_gnb_config_from_yaml_file_not_found() {
        let result = GnbConfig::from_yaml_file("/nonexistent/path/config.yaml");
        assert!(result.is_err());
    }

    #[test]
    fn test_ue_config_from_yaml_file_not_found() {
        let result = UeConfig::from_yaml_file("/nonexistent/path/config.yaml");
        assert!(result.is_err());
    }

    // PQC configuration tests

    #[test]
    fn test_kem_algorithm_display() {
        assert_eq!(KemAlgorithm::Kyber512.to_string(), "kyber512");
        assert_eq!(KemAlgorithm::Kyber768.to_string(), "kyber768");
        assert_eq!(KemAlgorithm::Ntru.to_string(), "ntru");
    }

    #[test]
    fn test_sign_algorithm_display() {
        assert_eq!(SignAlgorithm::Dilithium2.to_string(), "dilithium2");
        assert_eq!(SignAlgorithm::Falcon512.to_string(), "falcon512");
        assert_eq!(SignAlgorithm::SphincsSha256.to_string(), "sphincs-sha256");
    }

    #[test]
    fn test_hybrid_mode_display() {
        assert_eq!(HybridMode::ClassicalOnly.to_string(), "classical-only");
        assert_eq!(HybridMode::PqcOnly.to_string(), "pqc-only");
        assert_eq!(HybridMode::HybridParallel.to_string(), "hybrid-parallel");
    }

    #[test]
    fn test_pqc_config_default() {
        let config = PqcConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.kem_algorithm, KemAlgorithm::None);
        assert_eq!(config.sign_algorithm, SignAlgorithm::None);
        assert_eq!(config.hybrid_mode, HybridMode::ClassicalOnly);
    }

    #[test]
    fn test_pqc_config_new() {
        let config = PqcConfig::new(
            KemAlgorithm::Kyber768,
            SignAlgorithm::Dilithium3,
            HybridMode::HybridParallel,
        );
        assert!(config.enabled);
        assert_eq!(config.kem_algorithm, KemAlgorithm::Kyber768);
        assert_eq!(config.sign_algorithm, SignAlgorithm::Dilithium3);
        assert_eq!(config.hybrid_mode, HybridMode::HybridParallel);
    }

    #[test]
    fn test_pqc_config_new_no_algorithms() {
        let config = PqcConfig::new(
            KemAlgorithm::None,
            SignAlgorithm::None,
            HybridMode::ClassicalOnly,
        );
        assert!(!config.enabled);
    }

    #[test]
    fn test_gnb_config_with_pqc() {
        let config = GnbConfig {
            nci: 16,
            gnb_id_length: 24,
            plmn: Plmn::new(310, 410, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            ngap_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: PqcConfig::new(
                KemAlgorithm::Kyber512,
                SignAlgorithm::Dilithium2,
                HybridMode::HybridParallel,
            ),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
        };
        assert!(config.pqc_config.enabled);
        assert_eq!(config.pqc_config.kem_algorithm, KemAlgorithm::Kyber512);
    }

    #[test]
    fn test_ue_config_with_pqc() {
        let config = UeConfig {
            supi: None,
            protection_scheme: 0,
            home_network_public_key_id: 0,
            home_network_public_key: Vec::new(),
            routing_indicator: None,
            hplmn: Plmn::default(),
            key: [0u8; 16],
            op: [0u8; 16],
            op_type: OpType::default(),
            amf: [0x80, 0x00],
            imei: None,
            imei_sv: None,
            supported_algs: SupportedAlgs::default(),
            gnb_search_list: Vec::new(),
            sessions: Vec::new(),
            configured_nssai: NetworkSlice::new(),
            tun_name: None,
            pqc_config: PqcConfig::new(
                KemAlgorithm::Kyber1024,
                SignAlgorithm::Dilithium5,
                HybridMode::HybridConcatenate,
            ),
            redcap: false,
            snpn_config: None,
            prose_enabled: false,
            ursp_rules: vec![],
            pin_role: None,
        };
        assert!(config.pqc_config.enabled);
        assert_eq!(config.pqc_config.kem_algorithm, KemAlgorithm::Kyber1024);
        assert_eq!(config.pqc_config.sign_algorithm, SignAlgorithm::Dilithium5);
    }
}
