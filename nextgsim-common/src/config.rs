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
    /// Secondary AMF addresses for SCTP multi-homing (e.g. `["192.168.1.2"]`).
    ///
    /// When non-empty the gNB uses a `MultihomeSctpAssociation` so that the
    /// transport can fail over to an alternate path without dropping the NGAP
    /// session.  Addresses are plain IP strings; the same port as the primary
    /// address is used.
    #[serde(default)]
    pub secondary_addresses: Vec<String>,
}

impl AmfConfig {
    /// Creates a new AMF configuration.
    ///
    /// # Arguments
    /// * `address` - IP address of the AMF
    /// * `port` - SCTP port of the AMF
    pub fn new(address: IpAddr, port: u16) -> Self {
        Self {
            address,
            port,
            secondary_addresses: Vec::new(),
        }
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
    /// Post-quantum cryptography configuration.
    ///
    /// Deserialised from the `pqc` key. Without the rename the field name
    /// `pqc_config` would be the expected key, and because nothing in this
    /// config sets `deny_unknown_fields`, a `pqc:` block would be silently
    /// dropped instead of rejected.
    #[serde(default, rename = "pqc", alias = "pqc_config")]
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
    /// XR `QoS` support enabled (Rel-18 5QI 82-85)
    #[serde(default)]
    pub xr_qos_enabled: bool,
    /// AI/ML for NR support enabled (Rel-18, TS 38.843)
    #[serde(default)]
    pub ai_ml_nr_enabled: bool,
    /// Ambient `IoT` reader support enabled (Rel-18, TS 22.369)
    #[serde(default)]
    pub ambient_iot_enabled: bool,
    /// UAV tracking and identification enabled (Rel-18, TS 23.256)
    #[serde(default)]
    pub uav_enabled: bool,
    /// Ranging/sidelink positioning enabled (Rel-18, TS 23.586)
    #[serde(default)]
    pub ranging_enabled: bool,
    /// Energy saving features enabled (Rel-18)
    #[serde(default)]
    pub energy_saving_enabled: bool,
    // ========================================================================
    // Rel-20 6G feature flags
    //
    // NOTE: "Rel-20" here is a research label, not a conformance claim — 3GPP
    // Rel-20 (6G) has no frozen stage-3 spec. The flags below gate non-normative
    // prototypes (design informed by TR 22.870 use cases); disabled by default.
    // ========================================================================
    /// Service Hosting Environment (SHE) task enabled (Rel-20)
    #[serde(default)]
    pub she_enabled: bool,
    /// Network Data Analytics Function (NWDAF) task enabled (Rel-20)
    #[serde(default)]
    pub nwdaf_enabled: bool,
    /// Network Knowledge Exposure Function (NKEF) task enabled (Rel-20)
    #[serde(default)]
    pub nkef_enabled: bool,
    /// Integrated Sensing and Communication (ISAC) task enabled (Rel-20)
    #[serde(default)]
    pub isac_enabled: bool,
    /// ISAC anchor positions in metres [x, y, z].
    ///
    /// Defaults to a 100 m equilateral triangle at 10 m height:
    /// `(0,0,10)`, `(100,0,10)`, `(50,87,10)`.
    #[serde(default = "default_isac_anchors")]
    pub isac_anchors: Vec<[f64; 3]>,
    /// AI Agent Framework task enabled (Rel-20)
    #[serde(default)]
    pub agent_enabled: bool,
    /// Federated Learning Aggregator task enabled (Rel-20)
    #[serde(default)]
    pub federated_learning_enabled: bool,
    /// Use QUIC transport instead of SCTP for NGAP (6G forward-looking, Rel-20+).
    ///
    /// When `true` the gNB will attempt to use `QuicTransport` for the AMF
    /// connection.  QUIC provides built-in TLS 1.3, connection migration, and
    /// multiplexed streams — capabilities that align with 6G requirements.
    ///
    /// **Note**: full QUIC transport selection is scaffolded here; the runtime
    /// wiring logs a warning and falls back to SCTP until the path is fully
    /// integrated.
    #[serde(default)]
    pub quic_enabled: bool,
    /// NGAP SCTP transport backend: `"userspace"` (default) or `"kernel"`.
    ///
    /// * `"userspace"` — the in-process `sctp-proto`-over-UDP transport. This
    ///   is the default and interoperates with the matching nextgcore
    ///   simulator, but not with a real AMF (the wire is UDP, not SCTP).
    /// * `"kernel"` — real kernel SCTP (IP protocol 132) via lksctp/libsctp,
    ///   which is what lets the gNB associate with a real / Open5GS AMF. Only
    ///   effective on a Linux build compiled with the gNB `kernel-sctp`
    ///   feature; otherwise the connection fails loud at runtime.
    #[serde(default)]
    pub sctp_backend: SctpBackendKind,
}

/// gNB NGAP SCTP transport backend selector (deserialized from
/// `sctp_backend: "userspace"|"kernel"` in the gNB YAML).
///
/// Defaults to [`SctpBackendKind::Userspace`] so existing configs and behavior
/// are unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SctpBackendKind {
    /// Userspace `sctp-proto` over UDP (default; sim-to-sim only).
    #[default]
    Userspace,
    /// Kernel SCTP (IP proto 132) via lksctp/libsctp (Linux + `kernel-sctp`).
    Kernel,
}

fn default_gtp_port() -> u16 {
    2152
}

fn default_isac_anchors() -> Vec<[f64; 3]> {
    // 100 m equilateral triangle at 10 m height
    vec![[0.0, 0.0, 10.0], [100.0, 0.0, 10.0], [50.0, 87.0, 10.0]]
}

impl Default for GnbConfig {
    fn default() -> Self {
        Self {
            nci: 0,
            gnb_id_length: 24,
            plmn: Plmn::default(),
            tac: 0,
            nssai: Vec::new(),
            amf_configs: Vec::new(),
            link_ip: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            ngap_ip: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
            gtp_ip: IpAddr::V4(std::net::Ipv4Addr::LOCALHOST),
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
            xr_qos_enabled: false,
            ai_ml_nr_enabled: false,
            ambient_iot_enabled: false,
            uav_enabled: false,
            ranging_enabled: false,
            energy_saving_enabled: false,
            she_enabled: false,
            nwdaf_enabled: false,
            nkef_enabled: false,
            isac_enabled: false,
            isac_anchors: default_isac_anchors(),
            agent_enabled: false,
            federated_learning_enabled: false,
            quic_enabled: false,
            sctp_backend: SctpBackendKind::Userspace,
        }
    }
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
    /// being (36 - `gnb_id_length`).
    pub fn cell_id(&self) -> u32 {
        let cell_id_bits = 36 - self.gnb_id_length;
        let mask = (1u64 << cell_id_bits) - 1;
        (self.nci & mask) as u32
    }
}

/// Operator key type for authentication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OpType {
    /// Operator key (OP) - needs to be converted to `OPc`
    Op,
    /// Operator key derived (`OPc`) - used directly
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
/// Only algorithms `nextgsim-crypto` actually implements are selectable. The
/// backing implementation is RustCrypto `ml-kem` (FIPS 203), whose parameter
/// sets are ML-KEM-512/768/1024; the `Kyber*` names here are the pre-
/// standardisation spelling of exactly those three.
///
/// `Ntru` and `Saber` were previously offered here with no implementation
/// behind them, so selecting one silently produced no PQC at all. They are
/// removed rather than left as traps: NIST did not standardise either, and
/// SABER was not selected in the PQC process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum KemAlgorithm {
    /// No post-quantum KEM (classical only)
    #[default]
    None,
    /// ML-KEM-512 (FIPS 203; formerly CRYSTALS-Kyber-512)
    Kyber512,
    /// ML-KEM-768 (FIPS 203; formerly CRYSTALS-Kyber-768)
    Kyber768,
    /// ML-KEM-1024 (FIPS 203; formerly CRYSTALS-Kyber-1024)
    Kyber1024,
}

impl fmt::Display for KemAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KemAlgorithm::None => write!(f, "none"),
            KemAlgorithm::Kyber512 => write!(f, "kyber512"),
            KemAlgorithm::Kyber768 => write!(f, "kyber768"),
            KemAlgorithm::Kyber1024 => write!(f, "kyber1024"),
        }
    }
}

/// Post-quantum signature algorithm.
///
/// Only algorithms `nextgsim-crypto` actually implements are selectable. The
/// backing implementation is RustCrypto `ml-dsa` (FIPS 204), whose parameter
/// sets are ML-DSA-44/65/87; the `Dilithium*` names here are the pre-
/// standardisation spelling of exactly those three.
///
/// `Falcon512`, `Falcon1024` and `SphincsSha256` were previously offered here
/// with no implementation behind them, so selecting one silently produced no
/// PQC signatures at all. FN-DSA (Falcon) and SLH-DSA (SPHINCS+) are real NIST
/// selections, so these can come back — but as variants with code behind them,
/// not as config values that quietly do nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SignAlgorithm {
    /// No post-quantum signatures (classical only)
    #[default]
    None,
    /// ML-DSA-44 (FIPS 204; formerly CRYSTALS-Dilithium-2)
    Dilithium2,
    /// ML-DSA-65 (FIPS 204; formerly CRYSTALS-Dilithium-3)
    Dilithium3,
    /// ML-DSA-87 (FIPS 204; formerly CRYSTALS-Dilithium-5)
    Dilithium5,
}

impl fmt::Display for SignAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SignAlgorithm::None => write!(f, "none"),
            SignAlgorithm::Dilithium2 => write!(f, "dilithium2"),
            SignAlgorithm::Dilithium3 => write!(f, "dilithium3"),
            SignAlgorithm::Dilithium5 => write!(f, "dilithium5"),
        }
    }
}

/// Hybrid mode for combining classical and post-quantum cryptography.
///
/// Defines how classical and PQC algorithms are combined for transitional security.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
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
///
/// Every field is `#[serde(default)]` so a partial `pqc:` block is accepted —
/// which the shipped `config/ue.yaml` relies on, since it sets only `enabled`.
/// Without these, correcting the block's YAML key turned a silently-ignored
/// section into a hard startup failure ("missing field `kem_algorithm`").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PqcConfig {
    /// Whether PQC is enabled
    #[serde(default)]
    pub enabled: bool,
    /// KEM algorithm for key exchange
    #[serde(default)]
    pub kem_algorithm: KemAlgorithm,
    /// Signature algorithm for authentication
    #[serde(default)]
    pub sign_algorithm: SignAlgorithm,
    /// Hybrid mode for combining classical and PQC
    #[serde(default)]
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

fn default_true() -> bool {
    true
}

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

/// PIN (Personal `IoT` Network) role (Rel-18, TS 23.542).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PinRole {
    /// PIN element (`IoT` device)
    PinElement,
    /// PIN gateway (relay to network)
    PinGateway,
    /// PIN management entity
    PinManagement,
}

// ============================================================================
// Rel-18 XR (Extended Reality) Configuration (TS 26.928)
// ============================================================================

/// XR traffic type for `QoS` differentiation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum XrTrafficType {
    /// Cloud/split rendering video (downlink heavy)
    CloudRendering,
    /// Pose/control data (uplink, low latency)
    PoseControl,
    /// Haptic feedback (bidirectional, ultra-low latency)
    Haptic,
    /// Audio stream
    Audio,
    /// Scene description updates
    SceneUpdate,
}

/// XR application profile configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XrConfig {
    /// Whether XR optimizations are enabled
    #[serde(default)]
    pub enabled: bool,
    /// XR traffic type
    pub traffic_type: XrTrafficType,
    /// Target frame rate (fps)
    #[serde(default = "default_xr_fps")]
    pub target_fps: u32,
    /// Motion-to-photon latency budget (ms)
    #[serde(default = "default_xr_mtp_latency")]
    pub mtp_latency_ms: u32,
    /// PDU Set awareness enabled (groups related PDUs)
    #[serde(default)]
    pub pdu_set_enabled: bool,
    /// C-DRX cycle for XR power saving (ms, 0 = disabled)
    #[serde(default)]
    pub cdrx_cycle_ms: u32,
    /// Jitter tolerance (ms)
    #[serde(default = "default_xr_jitter")]
    pub jitter_tolerance_ms: u32,
}

fn default_xr_fps() -> u32 {
    90
}
fn default_xr_mtp_latency() -> u32 {
    20
}
fn default_xr_jitter() -> u32 {
    5
}

impl Default for XrConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            traffic_type: XrTrafficType::CloudRendering,
            target_fps: 90,
            mtp_latency_ms: 20,
            pdu_set_enabled: false,
            cdrx_cycle_ms: 0,
            jitter_tolerance_ms: 5,
        }
    }
}

// ============================================================================
// Rel-18 Ambient IoT Configuration (TS 22.369)
// ============================================================================

/// Ambient `IoT` device type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AmbientIotDeviceType {
    /// Type A: Energy harvesting, no battery (backscatter only)
    TypeA,
    /// Type B: Assisted energy harvesting with small capacitor
    TypeB,
    /// Type C: Battery-assisted with active Tx
    TypeC,
}

/// Ambient `IoT` configuration for a UE acting as reader/writer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbientIotConfig {
    /// Device type
    pub device_type: AmbientIotDeviceType,
    /// Maximum read range (meters)
    #[serde(default = "default_aiot_range")]
    pub max_range_meters: f64,
    /// Inventory round interval (ms)
    #[serde(default = "default_aiot_interval")]
    pub inventory_interval_ms: u32,
    /// Command data size (bytes, max for backscatter response)
    #[serde(default = "default_aiot_payload")]
    pub max_payload_bytes: u32,
}

fn default_aiot_range() -> f64 {
    10.0
}
fn default_aiot_interval() -> u32 {
    1000
}
fn default_aiot_payload() -> u32 {
    96
}

impl Default for AmbientIotConfig {
    fn default() -> Self {
        Self {
            device_type: AmbientIotDeviceType::TypeA,
            max_range_meters: 10.0,
            inventory_interval_ms: 1000,
            max_payload_bytes: 96,
        }
    }
}

// ============================================================================
// Rel-18 UAV (Unmanned Aerial Vehicle) Configuration (TS 23.256)
// ============================================================================

/// UAV UE configuration for aerial vehicle identification and tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UavConfig {
    /// Whether this UE is a UAV (aerial UE)
    #[serde(default)]
    pub is_aerial_ue: bool,
    /// UAV ID (Civil Aviation Authority assigned)
    pub uav_id: Option<String>,
    /// USS (UAV Traffic Management) address
    pub uss_address: Option<String>,
    /// Maximum flight altitude (meters)
    #[serde(default = "default_uav_alt")]
    pub max_altitude_meters: f64,
    /// Remote identification enabled (broadcast ID)
    #[serde(default)]
    pub remote_id_enabled: bool,
    /// C2 (Command and Control) link `QoS` required
    #[serde(default)]
    pub c2_link_required: bool,
}

fn default_uav_alt() -> f64 {
    120.0
}

impl Default for UavConfig {
    fn default() -> Self {
        Self {
            is_aerial_ue: false,
            uav_id: None,
            uss_address: None,
            max_altitude_meters: 120.0,
            remote_id_enabled: false,
            c2_link_required: false,
        }
    }
}

// ============================================================================
// Rel-16 V2X (Vehicle-to-Everything) Configuration (TS 23.287)
// ============================================================================

/// V2X service type for differentiated QoS handling.
///
/// Reference: 3GPP TS 23.287 Section 5.2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum V2xServiceType {
    /// V2V (Vehicle-to-Vehicle) communication
    V2V,
    /// V2I (Vehicle-to-Infrastructure) communication
    V2I,
    /// V2P (Vehicle-to-Pedestrian) communication
    V2P,
    /// V2N (Vehicle-to-Network) communication
    V2N,
}

/// V2X QoS requirements per service type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2xQosRequirements {
    /// Maximum end-to-end latency (ms)
    pub max_latency_ms: u32,
    /// Required reliability (probability 0.0-1.0)
    pub reliability: f64,
    /// Communication range (meters)
    pub range_meters: f64,
    /// Message priority (0-7, 7 = highest)
    pub priority: u8,
}

/// V2X UE configuration.
///
/// Configures UE for V2X operation with network slicing (SST=3).
/// Reference: 3GPP TS 23.287
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2xConfig {
    /// Whether V2X is enabled
    #[serde(default)]
    pub enabled: bool,
    /// V2X service types this UE supports
    pub service_types: Vec<V2xServiceType>,
    /// S-NSSAI for V2X slice (typically SST=3)
    #[serde(default = "default_v2x_snssai")]
    pub s_nssai: SNssai,
    /// QoS requirements per service type
    pub qos_requirements: Vec<(V2xServiceType, V2xQosRequirements)>,
    /// Geographical area of operation (optional)
    pub geo_area: Option<V2xGeoArea>,
    /// Preferred communication mode (PC5 sidelink or Uu network)
    #[serde(default)]
    pub preferred_mode: V2xCommMode,
}

fn default_v2x_snssai() -> SNssai {
    SNssai {
        sst: 3, // V2X slice type
        sd: None,
    }
}

/// V2X geographical area of operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct V2xGeoArea {
    /// Center latitude (degrees)
    pub center_lat: f64,
    /// Center longitude (degrees)
    pub center_lon: f64,
    /// Radius (meters)
    pub radius_meters: f64,
}

/// V2X communication mode preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum V2xCommMode {
    /// Prefer Uu (network) communication
    #[default]
    Uu,
    /// Prefer PC5 (sidelink) communication
    Pc5,
    /// Use both Uu and PC5 (hybrid)
    Hybrid,
}

impl Default for V2xConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            service_types: vec![V2xServiceType::V2V, V2xServiceType::V2I],
            s_nssai: SNssai {
                sst: 3, // V2X slice
                sd: None,
            },
            qos_requirements: vec![
                // Default V2V requirements (critical safety messages)
                (
                    V2xServiceType::V2V,
                    V2xQosRequirements {
                        max_latency_ms: 20,
                        reliability: 0.99,
                        range_meters: 300.0,
                        priority: 7,
                    },
                ),
                // Default V2I requirements
                (
                    V2xServiceType::V2I,
                    V2xQosRequirements {
                        max_latency_ms: 50,
                        reliability: 0.95,
                        range_meters: 500.0,
                        priority: 5,
                    },
                ),
                // Default V2P requirements
                (
                    V2xServiceType::V2P,
                    V2xQosRequirements {
                        max_latency_ms: 100,
                        reliability: 0.95,
                        range_meters: 200.0,
                        priority: 6,
                    },
                ),
                // Default V2N requirements
                (
                    V2xServiceType::V2N,
                    V2xQosRequirements {
                        max_latency_ms: 100,
                        reliability: 0.90,
                        range_meters: 1000.0,
                        priority: 4,
                    },
                ),
            ],
            geo_area: None,
            preferred_mode: V2xCommMode::Uu,
        }
    }
}

// ============================================================================
// Rel-18 Ranging/Sidelink Positioning (TS 23.586)
// ============================================================================

/// Ranging method for UE-to-UE positioning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RangingMethod {
    /// RTT (Round-Trip Time) based ranging
    Rtt,
    /// RSRP (Reference Signal Received Power) based
    Rsrp,
    /// `AoA` (Angle of Arrival) based
    AoA,
    /// Carrier phase based (high precision)
    CarrierPhase,
    /// Multi-RTT (triangulation with multiple UEs)
    MultiRtt,
}

/// Ranging/sidelink positioning configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangingConfig {
    /// Whether ranging is enabled
    #[serde(default)]
    pub enabled: bool,
    /// Ranging method
    pub method: RangingMethod,
    /// Maximum ranging distance (meters)
    #[serde(default = "default_ranging_distance")]
    pub max_distance_meters: f64,
    /// Ranging interval (ms)
    #[serde(default = "default_ranging_interval")]
    pub interval_ms: u32,
    /// Target accuracy (meters)
    #[serde(default = "default_ranging_accuracy")]
    pub target_accuracy_meters: f64,
}

fn default_ranging_distance() -> f64 {
    200.0
}
fn default_ranging_interval() -> u32 {
    100
}
fn default_ranging_accuracy() -> f64 {
    0.3
}

impl Default for RangingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            method: RangingMethod::Rtt,
            max_distance_meters: 200.0,
            interval_ms: 100,
            target_accuracy_meters: 0.3,
        }
    }
}

// ============================================================================
// Rel-18 MINT (Multi-IMSI/Multi-USIM) Configuration (TS 23.761)
// ============================================================================

/// MINT (Multi-IMSI) configuration for UEs with multiple subscriptions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MintConfig {
    /// Whether MINT is enabled (dual-SIM/multi-USIM)
    #[serde(default)]
    pub enabled: bool,
    /// Secondary SUPIs for additional subscriptions
    #[serde(default)]
    pub secondary_supis: Vec<String>,
    /// Active subscription index (0 = primary)
    #[serde(default)]
    pub active_subscription: u8,
    /// Allow simultaneous registration on multiple PLMNs
    #[serde(default)]
    pub simultaneous_registration: bool,
    /// DNN → subscription-index routing. A PDU session whose DNN matches an
    /// entry's `dnn` is established on the named subscription's SUPI (e.g. an
    /// "enterprise" DNN on secondary SUPI index 1). DNNs absent from this map
    /// default to the primary subscription (index 0). (TS 23.761 §4.2: the UE
    /// selects which subscription serves a given service/DNN.)
    #[serde(default)]
    pub dnn_subscription_map: Vec<DnnSubscription>,
    /// Disaster-roaming indication (TS 23.761 §4.2 / TS 24.501): when true the
    /// UE sets the disaster-roaming registration indication in its secondary
    /// SUPI Registration Request so the AMF can apply MINT handling.
    #[serde(default)]
    pub disaster_roaming: bool,
}

/// Maps a DNN to the MINT subscription index that should serve it.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DnnSubscription {
    /// Data Network Name (e.g. "enterprise")
    pub dnn: String,
    /// Subscription index (0 = primary SUPI, 1 = first secondary SUPI, ...)
    pub subscription_index: u8,
}

impl MintConfig {
    /// Resolve which subscription index serves the given DNN. DNNs not present
    /// in `dnn_subscription_map` default to the primary subscription (0).
    pub fn subscription_for_dnn(&self, dnn: Option<&str>) -> u8 {
        let Some(dnn) = dnn else { return 0 };
        self.dnn_subscription_map
            .iter()
            .find(|m| m.dnn.eq_ignore_ascii_case(dnn))
            .map_or(0, |m| m.subscription_index)
    }

    /// The SUPI string for a subscription index (0 = primary). Index 1 maps to
    /// `secondary_supis[0]`, etc. Returns `None` for an out-of-range index.
    pub fn supi_for_index(&self, primary: &str, index: u8) -> Option<String> {
        if index == 0 {
            return Some(primary.to_string());
        }
        self.secondary_supis.get((index - 1) as usize).cloned()
    }
}

// ============================================================================
// Rel-18 Enhanced RedCap Configuration (TS 38.300 v18)
// ============================================================================

/// Enhanced `RedCap` configuration (Rel-18 extends Rel-17 `RedCap`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedCapR18Config {
    /// 5 MHz bandwidth support (Rel-18 extends Rel-17's 20 MHz minimum)
    #[serde(default)]
    pub bw_5mhz: bool,
    /// Enhanced Power Saving Mode (ePSM)
    #[serde(default)]
    pub enhanced_psm: bool,
    /// Extended eDRX cycle (seconds, 0 = default)
    #[serde(default)]
    pub edrx_cycle_seconds: u32,
    /// Relaxed measurement criteria enabled
    #[serde(default)]
    pub relaxed_measurements: bool,
    /// Reduced PDCCH monitoring
    #[serde(default)]
    pub reduced_pdcch_monitoring: bool,
}

/// PDU session type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PduSessionType {
    /// IPv4 PDU session
    #[default]
    Ipv4,
    /// IPv6 PDU session
    Ipv6,
    /// `IPv4v6` (dual-stack) PDU session
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
    /// Requested 5QI for this session (optional, TS 23.501 §5.7).
    ///
    /// When set to an XR delay-critical GBR value (82-85, Rel-18) the UE
    /// requests an XR PDU session: if no XR APN/DNN is explicitly configured,
    /// a default XR DNN is selected so the SMF maps the session to the XR 5QI
    /// and installs the XR QoS flow end to end.
    #[serde(default)]
    pub requested_5qi: Option<u8>,
}

impl SessionConfig {
    /// Whether this session requests an XR delay-critical GBR 5QI (82-85).
    pub fn is_xr(&self) -> bool {
        matches!(self.requested_5qi, Some(q) if (82..=85).contains(&q))
    }

    /// DNN the SMF uses to derive the XR 5QI when no explicit APN is set.
    /// Mirrors the SMF-side `xr_5qi_for_dnn` mapping.
    pub fn xr_dnn_for_5qi(five_qi: u8) -> &'static str {
        match five_qi {
            84 => "xr-split",
            85 => "xr-haptic",
            _ => "xr",
        }
    }
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            session_type: PduSessionType::Ipv4,
            s_nssai: None,
            apn: None,
            is_emergency: false,
            requested_5qi: None,
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
    /// Operator key OP or `OPc` (128-bit)
    pub op: [u8; 16],
    /// Type of operator key (OP or `OPc`)
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
    /// Post-quantum cryptography configuration.
    ///
    /// Deserialised from the `pqc` key, which is what `config/ue.yaml` has
    /// always used. Without this rename the expected key was `pqc_config`, so
    /// the shipped `pqc:` block was silently discarded and setting
    /// `enabled: true` had no effect whatsoever.
    #[serde(default, rename = "pqc", alias = "pqc_config")]
    pub pqc_config: PqcConfig,
    /// `RedCap` (Reduced Capability) UE indication (Rel-17, TS 38.101)
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
    /// PIN (Personal `IoT` Network) role
    #[serde(default)]
    pub pin_role: Option<PinRole>,
    /// XR (Extended Reality) configuration (Rel-18, TS 26.928)
    #[serde(default)]
    pub xr_config: Option<XrConfig>,
    /// Ambient `IoT` reader/writer configuration (Rel-18, TS 22.369)
    #[serde(default)]
    pub ambient_iot_config: Option<AmbientIotConfig>,
    /// UAV (aerial UE) configuration (Rel-18, TS 23.256)
    #[serde(default)]
    pub uav_config: Option<UavConfig>,
    /// Ranging/sidelink positioning configuration (Rel-18, TS 23.586)
    #[serde(default)]
    pub ranging_config: Option<RangingConfig>,
    /// MINT (Multi-IMSI) configuration (Rel-18, TS 23.761)
    #[serde(default)]
    pub mint_config: Option<MintConfig>,
    /// Enhanced `RedCap` configuration (Rel-18 extensions)
    #[serde(default)]
    pub redcap_r18: Option<RedCapR18Config>,
    /// V2X (Vehicle-to-Everything) configuration (Rel-16, TS 23.287)
    #[serde(default)]
    pub v2x_config: Option<V2xConfig>,
    // ========================================================================
    // Rel-20 6G feature flags
    //
    // NOTE: "Rel-20" here is a research label, not a conformance claim — 3GPP
    // Rel-20 (6G) has no frozen stage-3 spec. The flags below gate non-normative
    // prototypes (design informed by TR 22.870 use cases); disabled by default.
    // ========================================================================
    /// Service Hosting Environment (SHE) client task enabled (Rel-20)
    #[serde(default)]
    pub she_enabled: bool,
    /// AI/ML NWDAF reporter task enabled (Rel-20)
    #[serde(default)]
    pub ai_ml_enabled: bool,
    /// Integrated Sensing and Communication (ISAC) sensor task enabled (Rel-20)
    #[serde(default)]
    pub isac_enabled: bool,
    /// Federated Learning participant task enabled (Rel-20)
    #[serde(default)]
    pub federated_learning_enabled: bool,
    /// Semantic Communication codec task enabled (Rel-20)
    #[serde(default)]
    pub semantic_comm_enabled: bool,
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
            xr_config: None,
            ambient_iot_config: None,
            uav_config: None,
            ranging_config: None,
            mint_config: None,
            redcap_r18: None,
            v2x_config: None,
            she_enabled: false,
            ai_ml_enabled: false,
            isac_enabled: false,
            federated_learning_enabled: false,
            semantic_comm_enabled: false,
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
    /// let config = GnbConfig::from_yaml(yaml).expect("value expected");
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
    /// let config = GnbConfig::from_yaml_file("config/gnb.yaml").expect("value expected");
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
    /// let config = UeConfig::from_yaml(yaml).expect("value expected");
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
    /// let config = UeConfig::from_yaml_file("config/ue.yaml").expect("value expected");
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
            ..Default::default()
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
            ..Default::default()
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
            ..Default::default()
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
            amf_configs: vec![AmfConfig::new(
                IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
                38412,
            )],
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
            ..Default::default()
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
            ..Default::default()
        };
        let yaml = original.to_yaml().unwrap();
        let parsed = UeConfig::from_yaml(&yaml).unwrap();
        assert_eq!(original.protection_scheme, parsed.protection_scheme);
        assert_eq!(
            original.home_network_public_key_id,
            parsed.home_network_public_key_id
        );
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
        // Only the ML-KEM parameter sets nextgsim-crypto implements. The Ntru
        // assertion this replaced covered a variant with no implementation.
        assert_eq!(KemAlgorithm::None.to_string(), "none");
        assert_eq!(KemAlgorithm::Kyber512.to_string(), "kyber512");
        assert_eq!(KemAlgorithm::Kyber768.to_string(), "kyber768");
        assert_eq!(KemAlgorithm::Kyber1024.to_string(), "kyber1024");
    }

    #[test]
    fn test_sign_algorithm_display() {
        // Only the ML-DSA parameter sets nextgsim-crypto implements. The
        // Falcon512 / SphincsSha256 assertions this replaced covered variants
        // with no implementation.
        assert_eq!(SignAlgorithm::None.to_string(), "none");
        assert_eq!(SignAlgorithm::Dilithium2.to_string(), "dilithium2");
        assert_eq!(SignAlgorithm::Dilithium3.to_string(), "dilithium3");
        assert_eq!(SignAlgorithm::Dilithium5.to_string(), "dilithium5");
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

    /// A full `pqc:` block body for embedding in a YAML fixture.
    fn pqc_yaml_block(kem: &str, sign: &str, hybrid: &str) -> String {
        format!(
            "  enabled: true\n  kem_algorithm: {kem}\n  \
             sign_algorithm: {sign}\n  hybrid_mode: {hybrid}\n"
        )
    }

    #[test]
    fn test_shipped_ue_yaml_parses_and_its_pqc_block_is_honoured() {
        // Two regressions in one, both about the same block.
        //
        // 1. The field is `pqc_config` while every shipped YAML writes `pqc:`,
        //    and nothing here sets deny_unknown_fields -- so the block was
        //    silently discarded and `enabled: true` did nothing at all.
        // 2. PqcConfig's fields were not individually #[serde(default)], so once
        //    the key was corrected the shipped file (which sets only `enabled`)
        //    stopped parsing outright: "missing field `kem_algorithm`".
        //
        // Parsing the REAL config/ue.yaml is what catches (2). A hand-rolled
        // fixture with all four fields present passes either way, which is
        // precisely how the second defect could have shipped behind a fix for
        // the first. The sibling test_ue_config_with_pqc builds UeConfig
        // struct-literally and so never exercises serde at all.
        let shipped = include_str!("../../config/ue.yaml");
        let config: UeConfig =
            serde_yaml::from_str(shipped).expect("the shipped config/ue.yaml must parse");

        // The shipped file sets enabled: false and omits the algorithm fields,
        // which must fall back to their defaults rather than failing.
        assert!(!config.pqc_config.enabled);
        assert_eq!(config.pqc_config.kem_algorithm, KemAlgorithm::None);
        assert_eq!(config.pqc_config.sign_algorithm, SignAlgorithm::None);
        assert_eq!(config.pqc_config.hybrid_mode, HybridMode::ClassicalOnly);
    }

    #[test]
    fn test_ue_config_pqc_yaml_key_reaches_the_field() {
        // The `pqc:` key must actually populate pqc_config. Uses the shipped
        // file as the base with the PQC block overridden, so the assertion is
        // about the real document shape.
        // Strip the shipped `pqc:` block (its body is the indented lines that
        // follow it) and substitute a fully-populated one. Appending a second
        // `pqc:` key would be a duplicate-key parse error, not an override.
        let shipped = include_str!("../../config/ue.yaml");
        let without_pqc: String = {
            let mut out = String::new();
            let mut skipping = false;
            for line in shipped.lines() {
                if line.trim_start().starts_with("pqc:") {
                    skipping = true;
                    continue;
                }
                // The block ends at the next line that is neither indented nor
                // blank nor a comment.
                if skipping {
                    let is_body = line.starts_with(' ') || line.trim().is_empty();
                    if is_body {
                        continue;
                    }
                    skipping = false;
                }
                out.push_str(line);
                out.push('\n');
            }
            out
        };
        assert!(
            !without_pqc.contains("pqc:"),
            "the shipped pqc block must have been stripped"
        );
        let overridden = format!(
            "{without_pqc}\npqc:\n{}",
            pqc_yaml_block("Kyber768", "Dilithium3", "HybridParallel")
        );

        let config: UeConfig = serde_yaml::from_str(&overridden).expect("UE config must parse");
        assert!(
            config.pqc_config.enabled,
            "the `pqc:` YAML key must populate pqc_config, not be dropped"
        );
        assert_eq!(config.pqc_config.kem_algorithm, KemAlgorithm::Kyber768);
        assert_eq!(config.pqc_config.sign_algorithm, SignAlgorithm::Dilithium3);
        assert_eq!(config.pqc_config.hybrid_mode, HybridMode::HybridParallel);
    }

    #[test]
    fn test_pqc_algorithms_are_limited_to_implemented_ones() {
        // The enums used to offer Ntru, Saber, Falcon512, Falcon1024 and
        // SphincsSha256, none of which nextgsim-crypto implements: selecting one
        // silently produced no PQC. They must not deserialize.
        for bad in ["Ntru", "Saber"] {
            let yaml = pqc_yaml_block(bad, "Dilithium3", "HybridParallel");
            assert!(
                serde_yaml::from_str::<PqcConfig>(&yaml).is_err(),
                "KEM {bad} must not be selectable: it has no implementation"
            );
        }
        for bad in ["Falcon512", "Falcon1024", "SphincsSha256"] {
            let yaml = pqc_yaml_block("Kyber768", bad, "HybridParallel");
            assert!(
                serde_yaml::from_str::<PqcConfig>(&yaml).is_err(),
                "signature {bad} must not be selectable: it has no implementation"
            );
        }
        // The implemented ones must still parse.
        for good in ["Kyber512", "Kyber768", "Kyber1024"] {
            let yaml = pqc_yaml_block(good, "Dilithium3", "HybridParallel");
            serde_yaml::from_str::<PqcConfig>(&yaml)
                .unwrap_or_else(|e| panic!("KEM {good} must parse: {e}"));
        }
        for good in ["Dilithium2", "Dilithium3", "Dilithium5"] {
            let yaml = pqc_yaml_block("Kyber768", good, "HybridParallel");
            serde_yaml::from_str::<PqcConfig>(&yaml)
                .unwrap_or_else(|e| panic!("signature {good} must parse: {e}"));
        }
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
            ..Default::default()
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
            ..Default::default()
        };
        assert!(config.pqc_config.enabled);
        assert_eq!(config.pqc_config.kem_algorithm, KemAlgorithm::Kyber1024);
        assert_eq!(config.pqc_config.sign_algorithm, SignAlgorithm::Dilithium5);
    }

    #[test]
    fn test_mint_dnn_subscription_routing() {
        // MINT (Rel-18, TS 23.761): a DNN in the map routes to its subscription
        // index; unmapped DNNs and None default to the primary subscription.
        let mint = MintConfig {
            enabled: true,
            secondary_supis: vec!["999700000000002".to_string()],
            active_subscription: 0,
            simultaneous_registration: true,
            dnn_subscription_map: vec![DnnSubscription {
                dnn: "enterprise".to_string(),
                subscription_index: 1,
            }],
            disaster_roaming: true,
        };

        assert_eq!(mint.subscription_for_dnn(Some("enterprise")), 1);
        // Case-insensitive match
        assert_eq!(mint.subscription_for_dnn(Some("ENTERPRISE")), 1);
        // Unmapped / absent DNN → primary
        assert_eq!(mint.subscription_for_dnn(Some("internet")), 0);
        assert_eq!(mint.subscription_for_dnn(None), 0);

        // supi_for_index: 0 = primary, 1 = first secondary, out-of-range = None
        assert_eq!(
            mint.supi_for_index("999700000000001", 0).as_deref(),
            Some("999700000000001")
        );
        assert_eq!(
            mint.supi_for_index("999700000000001", 1).as_deref(),
            Some("999700000000002")
        );
        assert_eq!(mint.supi_for_index("999700000000001", 2), None);
    }
}
