//! NAS security context and message protection
//!
//! Implements security-protected NAS message structures and security context
//! management according to 3GPP TS 24.501.
//!
//! # Security Context Management
//!
//! The NAS security context contains all the security-related information needed
//! for protecting NAS messages between the UE and the AMF:
//!
//! - Security keys (KAUSF, KSEAF, KAMF, KNASint, KNASenc)
//! - Selected security algorithms (ciphering and integrity)
//! - NAS COUNT values (uplink and downlink)
//! - Key set identifier (ngKSI)
//!
//! # Example
//!
//! ```ignore
//! use nextgsim_nas::security::{NasSecurityContext, SecurityContextState};
//!
//! // Create a new security context
//! let mut ctx = NasSecurityContext::new();
//!
//! // Context starts in Null state
//! assert_eq!(ctx.state(), SecurityContextState::Null);
//!
//! // After authentication, set keys and activate
//! ctx.keys_mut().set_kamf(&kamf);
//! ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2);
//! ctx.activate();
//!
//! // Now context is active and can be used for message protection
//! assert_eq!(ctx.state(), SecurityContextState::Active);
//! ```

use crate::enums::{ExtendedProtocolDiscriminator, SecurityHeaderType};
use nextgsim_crypto::kdf::{derive_knas_enc, derive_knas_int, KEY_128_SIZE, KEY_256_SIZE};
use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac, KEY_SIZE, MAC_SIZE};

/// Security-protected NAS message header
///
/// This structure represents the header of a security-protected 5G NAS message
/// as defined in 3GPP TS 24.501 Section 9.1.
///
/// The format is:
/// - Extended Protocol Discriminator (1 octet)
/// - Security Header Type (4 bits) + Spare half octet (4 bits)
/// - Message Authentication Code (4 octets)
/// - Sequence Number (1 octet)
/// - Plain NAS message (variable)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecuredNasMessage {
    /// Extended Protocol Discriminator
    pub epd: ExtendedProtocolDiscriminator,
    /// Security header type
    pub security_header_type: SecurityHeaderType,
    /// Message Authentication Code (MAC) - 32 bits
    pub mac: [u8; 4],
    /// NAS message sequence number
    pub sequence_number: u8,
    /// The plain NAS message (encrypted if security header indicates ciphering)
    pub plain_nas_message: Vec<u8>,
}

impl SecuredNasMessage {
    /// Create a new security-protected NAS message
    pub fn new(
        epd: ExtendedProtocolDiscriminator,
        security_header_type: SecurityHeaderType,
        mac: [u8; 4],
        sequence_number: u8,
        plain_nas_message: Vec<u8>,
    ) -> Self {
        Self {
            epd,
            security_header_type,
            mac,
            sequence_number,
            plain_nas_message,
        }
    }

    /// Returns true if the message is integrity protected
    pub fn is_integrity_protected(&self) -> bool {
        self.security_header_type.is_protected()
    }

    /// Returns true if the message is ciphered
    pub fn is_ciphered(&self) -> bool {
        self.security_header_type.is_ciphered()
    }

    /// Returns true if this message indicates a new security context
    pub fn is_new_security_context(&self) -> bool {
        self.security_header_type.is_new_security_context()
    }

    /// Get the total encoded length of the security header (excluding plain message)
    pub const fn header_length() -> usize {
        // EPD (1) + Security Header Type (1) + MAC (4) + Sequence Number (1) = 7
        7
    }
}

/// NAS security algorithms selection
///
/// Represents the selected NAS security algorithms for ciphering and integrity
/// protection as defined in 3GPP TS 24.501.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NasSecurityAlgorithms {
    /// Type of ciphering algorithm (NEA0-NEA3)
    pub ciphering: CipheringAlgorithm,
    /// Type of integrity protection algorithm (NIA0-NIA3)
    pub integrity: IntegrityAlgorithm,
}

impl NasSecurityAlgorithms {
    /// Create a new NAS security algorithms selection
    pub fn new(ciphering: CipheringAlgorithm, integrity: IntegrityAlgorithm) -> Self {
        Self {
            ciphering,
            integrity,
        }
    }

    /// Encode to a single octet (ciphering in high nibble, integrity in low nibble)
    pub fn encode(&self) -> u8 {
        ((self.ciphering as u8) << 4) | (self.integrity as u8)
    }

    /// Decode from a single octet
    pub fn decode(value: u8) -> Result<Self, SecurityError> {
        let ciphering = CipheringAlgorithm::try_from((value >> 4) & 0x0F)?;
        let integrity = IntegrityAlgorithm::try_from(value & 0x0F)?;
        Ok(Self {
            ciphering,
            integrity,
        })
    }
}

/// 5G NAS ciphering algorithm type
///
/// 3GPP TS 24.501 Section 9.11.3.34
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum CipheringAlgorithm {
    /// 5G-EA0 (null ciphering)
    #[default]
    Nea0 = 0x00,
    /// 128-5G-EA1 (SNOW3G based)
    Nea1 = 0x01,
    /// 128-5G-EA2 (AES based)
    Nea2 = 0x02,
    /// 128-5G-EA3 (ZUC based)
    Nea3 = 0x03,
}

impl TryFrom<u8> for CipheringAlgorithm {
    type Error = SecurityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(CipheringAlgorithm::Nea0),
            0x01 => Ok(CipheringAlgorithm::Nea1),
            0x02 => Ok(CipheringAlgorithm::Nea2),
            0x03 => Ok(CipheringAlgorithm::Nea3),
            _ => Err(SecurityError::InvalidCipheringAlgorithm(value)),
        }
    }
}

/// 5G NAS integrity protection algorithm type
///
/// 3GPP TS 24.501 Section 9.11.3.34
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum IntegrityAlgorithm {
    /// 5G-IA0 (null integrity)
    #[default]
    Nia0 = 0x00,
    /// 128-5G-IA1 (SNOW3G based)
    Nia1 = 0x01,
    /// 128-5G-IA2 (AES based)
    Nia2 = 0x02,
    /// 128-5G-IA3 (ZUC based)
    Nia3 = 0x03,
}

impl TryFrom<u8> for IntegrityAlgorithm {
    type Error = SecurityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(IntegrityAlgorithm::Nia0),
            0x01 => Ok(IntegrityAlgorithm::Nia1),
            0x02 => Ok(IntegrityAlgorithm::Nia2),
            0x03 => Ok(IntegrityAlgorithm::Nia3),
            _ => Err(SecurityError::InvalidIntegrityAlgorithm(value)),
        }
    }
}

/// Type of security context
///
/// 3GPP TS 24.501 Section 9.11.3.32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum SecurityContextType {
    /// Native security context
    #[default]
    Native = 0,
    /// Mapped security context (from EPS)
    Mapped = 1,
}

impl TryFrom<u8> for SecurityContextType {
    type Error = SecurityError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(SecurityContextType::Native),
            1 => Ok(SecurityContextType::Mapped),
            _ => Err(SecurityError::InvalidSecurityContextType(value)),
        }
    }
}

/// NAS Key Set Identifier (ngKSI)
///
/// 3GPP TS 24.501 Section 9.11.3.32
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct NasKeySetIdentifier {
    /// Type of security context
    pub tsc: SecurityContextType,
    /// NAS key set identifier value (0-6, 7 = no key available)
    pub ksi: u8,
}

impl NasKeySetIdentifier {
    /// Value indicating no key is available
    pub const NO_KEY_AVAILABLE: u8 = 0x07;

    /// Create a new NAS key set identifier
    pub fn new(tsc: SecurityContextType, ksi: u8) -> Self {
        Self { tsc, ksi: ksi & 0x07 }
    }

    /// Create a "no key available" identifier
    pub fn no_key() -> Self {
        Self {
            tsc: SecurityContextType::Native,
            ksi: Self::NO_KEY_AVAILABLE,
        }
    }

    /// Returns true if no key is available
    pub fn is_no_key(&self) -> bool {
        self.ksi == Self::NO_KEY_AVAILABLE
    }

    /// Encode to a half octet (4 bits)
    pub fn encode(&self) -> u8 {
        ((self.tsc as u8) << 3) | (self.ksi & 0x07)
    }

    /// Decode from a half octet (4 bits)
    pub fn decode(value: u8) -> Result<Self, SecurityError> {
        let tsc = SecurityContextType::try_from((value >> 3) & 0x01)?;
        let ksi = value & 0x07;
        Ok(Self { tsc, ksi })
    }
}

/// Security-related errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SecurityError {
    /// Invalid ciphering algorithm value
    #[error("Invalid ciphering algorithm: 0x{0:02X}")]
    InvalidCipheringAlgorithm(u8),
    /// Invalid integrity algorithm value
    #[error("Invalid integrity algorithm: 0x{0:02X}")]
    InvalidIntegrityAlgorithm(u8),
    /// Invalid security context type value
    #[error("Invalid security context type: {0}")]
    InvalidSecurityContextType(u8),
    /// Invalid security header type value
    #[error("Invalid security header type: 0x{0:02X}")]
    InvalidSecurityHeaderType(u8),
    /// MAC verification failed
    #[error("MAC verification failed")]
    MacVerificationFailed,
    /// Security context not established
    #[error("Security context not established")]
    NoSecurityContext,
    /// Sequence number out of range
    #[error("Sequence number out of range")]
    SequenceNumberOutOfRange,
    /// NAS count overflow detected
    #[error("NAS count overflow detected")]
    NasCountOverflow,
    /// Security context not active
    #[error("Security context not active")]
    SecurityContextNotActive,
    /// Invalid key length
    #[error("Invalid key length: expected {expected}, got {actual}")]
    InvalidKeyLength { expected: usize, actual: usize },
}

/// State of the NAS security context
///
/// The security context transitions through these states during the
/// authentication and security mode control procedures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SecurityContextState {
    /// No security context exists (initial state)
    #[default]
    Null,
    /// Security context is being established (during authentication)
    Establishing,
    /// Security context is established but not yet activated
    Inactive,
    /// Security context is active and can be used for message protection
    Active,
}

/// UE security keys derived during authentication
///
/// Contains all the keys derived during the 5G-AKA authentication procedure
/// and subsequent key derivations. Keys are stored as fixed-size arrays.
///
/// Key hierarchy (3GPP TS 33.501):
/// ```text
/// CK, IK (from USIM)
///    └── KAUSF (256-bit)
///           └── KSEAF (256-bit)
///                  └── KAMF (256-bit)
///                         ├── KNASint (128-bit)
///                         ├── KNASenc (128-bit)
///                         └── KgNB (256-bit)
/// ```
#[derive(Debug, Clone, Default)]
pub struct UeKeys {
    /// ABBA (Anti-Bidding down Between Architectures) parameter
    pub abba: Vec<u8>,
    /// KAUSF - Key for AUSF (256-bit)
    kausf: Option<[u8; KEY_256_SIZE]>,
    /// KSEAF - Key for SEAF (256-bit)
    kseaf: Option<[u8; KEY_256_SIZE]>,
    /// KAMF - Key for AMF (256-bit)
    kamf: Option<[u8; KEY_256_SIZE]>,
    /// KNASint - NAS integrity key (128-bit)
    knas_int: Option<[u8; KEY_128_SIZE]>,
    /// KNASenc - NAS encryption key (128-bit)
    knas_enc: Option<[u8; KEY_128_SIZE]>,
}

impl UeKeys {
    /// Create a new empty UeKeys structure
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the KAUSF key
    pub fn set_kausf(&mut self, key: &[u8; KEY_256_SIZE]) {
        self.kausf = Some(*key);
    }

    /// Get the KAUSF key
    pub fn kausf(&self) -> Option<&[u8; KEY_256_SIZE]> {
        self.kausf.as_ref()
    }

    /// Set the KSEAF key
    pub fn set_kseaf(&mut self, key: &[u8; KEY_256_SIZE]) {
        self.kseaf = Some(*key);
    }

    /// Get the KSEAF key
    pub fn kseaf(&self) -> Option<&[u8; KEY_256_SIZE]> {
        self.kseaf.as_ref()
    }

    /// Set the KAMF key
    pub fn set_kamf(&mut self, key: &[u8; KEY_256_SIZE]) {
        self.kamf = Some(*key);
    }

    /// Get the KAMF key
    pub fn kamf(&self) -> Option<&[u8; KEY_256_SIZE]> {
        self.kamf.as_ref()
    }

    /// Set the KNASint key
    pub fn set_knas_int(&mut self, key: &[u8; KEY_128_SIZE]) {
        self.knas_int = Some(*key);
    }

    /// Get the KNASint key
    pub fn knas_int(&self) -> Option<&[u8; KEY_128_SIZE]> {
        self.knas_int.as_ref()
    }

    /// Set the KNASenc key
    pub fn set_knas_enc(&mut self, key: &[u8; KEY_128_SIZE]) {
        self.knas_enc = Some(*key);
    }

    /// Get the KNASenc key
    pub fn knas_enc(&self) -> Option<&[u8; KEY_128_SIZE]> {
        self.knas_enc.as_ref()
    }

    /// Check if KAMF is available (required for NAS key derivation)
    pub fn has_kamf(&self) -> bool {
        self.kamf.is_some()
    }

    /// Check if NAS keys are available
    pub fn has_nas_keys(&self) -> bool {
        self.knas_int.is_some() && self.knas_enc.is_some()
    }

    /// Clear all keys (for security context reset)
    pub fn clear(&mut self) {
        // Securely clear keys by overwriting with zeros before dropping
        if let Some(ref mut key) = self.kausf {
            key.fill(0);
        }
        if let Some(ref mut key) = self.kseaf {
            key.fill(0);
        }
        if let Some(ref mut key) = self.kamf {
            key.fill(0);
        }
        if let Some(ref mut key) = self.knas_int {
            key.fill(0);
        }
        if let Some(ref mut key) = self.knas_enc {
            key.fill(0);
        }
        self.kausf = None;
        self.kseaf = None;
        self.kamf = None;
        self.knas_int = None;
        self.knas_enc = None;
        self.abba.clear();
    }

    /// Create a deep copy of the keys
    pub fn deep_copy(&self) -> Self {
        Self {
            abba: self.abba.clone(),
            kausf: self.kausf,
            kseaf: self.kseaf,
            kamf: self.kamf,
            knas_int: self.knas_int,
            knas_enc: self.knas_enc,
        }
    }
}

/// NAS Security Context
///
/// Contains all security-related state for NAS message protection between
/// the UE and the AMF. This includes:
/// - Security keys (derived during authentication)
/// - Selected security algorithms
/// - NAS COUNT values for replay protection
/// - Key set identifier (ngKSI)
///
/// # State Machine
///
/// The security context transitions through the following states:
/// ```text
/// Null -> Establishing -> Inactive -> Active
///   ^                        |          |
///   |________________________|__________|
///              (reset)
/// ```
///
/// # Example
///
/// ```ignore
/// use nextgsim_nas::security::{NasSecurityContext, CipheringAlgorithm, IntegrityAlgorithm};
///
/// let mut ctx = NasSecurityContext::new();
///
/// // Set up keys after authentication
/// ctx.keys_mut().set_kamf(&kamf);
///
/// // Derive NAS keys with selected algorithms
/// ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2)?;
///
/// // Activate the context
/// ctx.activate();
///
/// // Now the context can be used for message protection
/// let mac = ctx.compute_uplink_mac(1, &message)?;
/// ```
#[derive(Debug, Clone, Default)]
pub struct NasSecurityContext {
    /// Current state of the security context
    state: SecurityContextState,
    /// Type of security context (native or mapped)
    tsc: SecurityContextType,
    /// NAS key set identifier (3-bit, 0-6, 7 = no key)
    ng_ksi: u8,
    /// Downlink NAS COUNT
    downlink_count: NasCount,
    /// Uplink NAS COUNT
    uplink_count: NasCount,
    /// Whether this is 3GPP access (vs non-3GPP)
    is_3gpp_access: bool,
    /// Security keys
    keys: UeKeys,
    /// Selected integrity algorithm
    integrity_algorithm: IntegrityAlgorithm,
    /// Selected ciphering algorithm
    ciphering_algorithm: CipheringAlgorithm,
    /// Recent NAS sequence numbers for replay detection (optional)
    recent_sequence_numbers: Vec<u8>,
    /// Maximum number of recent sequence numbers to track
    max_recent_sqn: usize,
}

impl NasSecurityContext {
    /// Default maximum recent sequence numbers to track
    const DEFAULT_MAX_RECENT_SQN: usize = 16;

    /// Create a new security context in Null state
    pub fn new() -> Self {
        Self {
            state: SecurityContextState::Null,
            tsc: SecurityContextType::Native,
            ng_ksi: NasKeySetIdentifier::NO_KEY_AVAILABLE,
            downlink_count: NasCount::default(),
            uplink_count: NasCount::default(),
            is_3gpp_access: true,
            keys: UeKeys::new(),
            integrity_algorithm: IntegrityAlgorithm::Nia0,
            ciphering_algorithm: CipheringAlgorithm::Nea0,
            recent_sequence_numbers: Vec::new(),
            max_recent_sqn: Self::DEFAULT_MAX_RECENT_SQN,
        }
    }

    /// Create a security context for 3GPP access
    pub fn new_3gpp() -> Self {
        let mut ctx = Self::new();
        ctx.is_3gpp_access = true;
        ctx
    }

    /// Create a security context for non-3GPP access
    pub fn new_non_3gpp() -> Self {
        let mut ctx = Self::new();
        ctx.is_3gpp_access = false;
        ctx
    }

    /// Get the current state of the security context
    pub fn state(&self) -> SecurityContextState {
        self.state
    }

    /// Check if the security context is active
    pub fn is_active(&self) -> bool {
        self.state == SecurityContextState::Active
    }

    /// Check if the security context is null (no context)
    pub fn is_null(&self) -> bool {
        self.state == SecurityContextState::Null
    }

    /// Get the security context type
    pub fn tsc(&self) -> SecurityContextType {
        self.tsc
    }

    /// Set the security context type
    pub fn set_tsc(&mut self, tsc: SecurityContextType) {
        self.tsc = tsc;
    }

    /// Get the NAS key set identifier
    pub fn ng_ksi(&self) -> u8 {
        self.ng_ksi
    }

    /// Set the NAS key set identifier
    pub fn set_ng_ksi(&mut self, ksi: u8) {
        self.ng_ksi = ksi & 0x07;
    }

    /// Get the NAS key set identifier as a NasKeySetIdentifier struct
    pub fn nas_ksi(&self) -> NasKeySetIdentifier {
        NasKeySetIdentifier::new(self.tsc, self.ng_ksi)
    }

    /// Set the NAS key set identifier from a NasKeySetIdentifier struct
    pub fn set_nas_ksi(&mut self, ksi: NasKeySetIdentifier) {
        self.tsc = ksi.tsc;
        self.ng_ksi = ksi.ksi;
    }

    /// Check if this is 3GPP access
    pub fn is_3gpp_access(&self) -> bool {
        self.is_3gpp_access
    }

    /// Set whether this is 3GPP access
    pub fn set_3gpp_access(&mut self, is_3gpp: bool) {
        self.is_3gpp_access = is_3gpp;
    }

    /// Get a reference to the security keys
    pub fn keys(&self) -> &UeKeys {
        &self.keys
    }

    /// Get a mutable reference to the security keys
    pub fn keys_mut(&mut self) -> &mut UeKeys {
        &mut self.keys
    }

    /// Get the selected integrity algorithm
    pub fn integrity_algorithm(&self) -> IntegrityAlgorithm {
        self.integrity_algorithm
    }

    /// Get the selected ciphering algorithm
    pub fn ciphering_algorithm(&self) -> CipheringAlgorithm {
        self.ciphering_algorithm
    }

    /// Get the selected algorithms as NasSecurityAlgorithms
    pub fn algorithms(&self) -> NasSecurityAlgorithms {
        NasSecurityAlgorithms::new(self.ciphering_algorithm, self.integrity_algorithm)
    }

    /// Set the selected security algorithms
    pub fn set_algorithms(&mut self, ciphering: CipheringAlgorithm, integrity: IntegrityAlgorithm) {
        self.ciphering_algorithm = ciphering;
        self.integrity_algorithm = integrity;
    }

    /// Get the downlink NAS COUNT
    pub fn downlink_count(&self) -> &NasCount {
        &self.downlink_count
    }

    /// Get the uplink NAS COUNT
    pub fn uplink_count(&self) -> &NasCount {
        &self.uplink_count
    }

    /// Update the downlink count after validating a received message
    pub fn update_downlink_count(&mut self, validated_count: &NasCount) {
        self.downlink_count = *validated_count;
    }

    /// Estimate the downlink count from a received sequence number
    pub fn estimate_downlink_count(&self, sequence_number: u8) -> NasCount {
        self.downlink_count.estimate_from_sqn(sequence_number)
    }

    /// Increment the uplink count (called before sending a message)
    ///
    /// Returns the current count value before incrementing.
    pub fn increment_uplink_count(&mut self) -> Result<NasCount, SecurityError> {
        let current = self.uplink_count;
        self.uplink_count.increment()?;
        Ok(current)
    }

    /// Get the current uplink sequence number
    pub fn uplink_sqn(&self) -> u8 {
        self.uplink_count.sqn
    }

    /// Rollback the uplink count (used after failed encryption)
    pub fn rollback_uplink_count(&mut self) {
        self.uplink_count.decrement();
    }

    /// Derive NAS keys from KAMF
    ///
    /// This derives KNASint and KNASenc from KAMF using the specified algorithms.
    /// KAMF must be set before calling this method.
    ///
    /// # Arguments
    /// * `ciphering` - The ciphering algorithm to use
    /// * `integrity` - The integrity algorithm to use
    ///
    /// # Returns
    /// * `Ok(())` if keys were derived successfully
    /// * `Err(SecurityError::NoSecurityContext)` if KAMF is not set
    pub fn derive_nas_keys(
        &mut self,
        ciphering: CipheringAlgorithm,
        integrity: IntegrityAlgorithm,
    ) -> Result<(), SecurityError> {
        // Copy KAMF to avoid borrow issues
        let kamf = *self.keys.kamf().ok_or(SecurityError::NoSecurityContext)?;

        // Derive KNASenc
        let knas_enc = derive_knas_enc(&kamf, ciphering as u8);
        self.keys.set_knas_enc(&knas_enc);

        // Derive KNASint
        let knas_int = derive_knas_int(&kamf, integrity as u8);
        self.keys.set_knas_int(&knas_int);

        // Store the selected algorithms
        self.ciphering_algorithm = ciphering;
        self.integrity_algorithm = integrity;

        Ok(())
    }

    /// Transition to Establishing state (during authentication)
    pub fn begin_establishing(&mut self) {
        self.state = SecurityContextState::Establishing;
    }

    /// Transition to Inactive state (after authentication, before security mode)
    pub fn set_inactive(&mut self) {
        self.state = SecurityContextState::Inactive;
    }

    /// Activate the security context
    ///
    /// This should be called after the Security Mode Complete message is sent/received.
    pub fn activate(&mut self) {
        self.state = SecurityContextState::Active;
    }

    /// Reset the security context to Null state
    ///
    /// This clears all keys and resets all counters.
    pub fn reset(&mut self) {
        self.state = SecurityContextState::Null;
        self.tsc = SecurityContextType::Native;
        self.ng_ksi = NasKeySetIdentifier::NO_KEY_AVAILABLE;
        self.downlink_count = NasCount::default();
        self.uplink_count = NasCount::default();
        self.keys.clear();
        self.integrity_algorithm = IntegrityAlgorithm::Nia0;
        self.ciphering_algorithm = CipheringAlgorithm::Nea0;
        self.recent_sequence_numbers.clear();
    }

    /// Reset the NAS counts (e.g., after re-authentication)
    pub fn reset_counts(&mut self) {
        self.downlink_count = NasCount::default();
        self.uplink_count = NasCount::default();
        self.recent_sequence_numbers.clear();
    }

    /// Compute MAC for an uplink message
    ///
    /// # Arguments
    /// * `sequence_number` - The NAS sequence number
    /// * `plain_message` - The plain NAS message
    ///
    /// # Returns
    /// * `Ok([u8; 4])` - The computed MAC
    /// * `Err(SecurityError)` - If the context is not active or keys are missing
    pub fn compute_uplink_mac(
        &self,
        sequence_number: u8,
        plain_message: &[u8],
    ) -> Result<[u8; MAC_SIZE], SecurityError> {
        if !self.is_active() {
            return Err(SecurityError::SecurityContextNotActive);
        }

        let key = self.keys.knas_int().ok_or(SecurityError::NoSecurityContext)?;

        Ok(compute_nas_mac(
            self.integrity_algorithm,
            key,
            &self.uplink_count,
            NasDirection::Uplink,
            sequence_number,
            plain_message,
        ))
    }

    /// Compute MAC for a downlink message
    ///
    /// # Arguments
    /// * `count` - The estimated NAS COUNT
    /// * `sequence_number` - The NAS sequence number
    /// * `plain_message` - The plain NAS message
    ///
    /// # Returns
    /// * `Ok([u8; 4])` - The computed MAC
    /// * `Err(SecurityError)` - If the context is not active or keys are missing
    pub fn compute_downlink_mac(
        &self,
        count: &NasCount,
        sequence_number: u8,
        plain_message: &[u8],
    ) -> Result<[u8; MAC_SIZE], SecurityError> {
        if !self.is_active() {
            return Err(SecurityError::SecurityContextNotActive);
        }

        let key = self.keys.knas_int().ok_or(SecurityError::NoSecurityContext)?;

        Ok(compute_nas_mac(
            self.integrity_algorithm,
            key,
            count,
            NasDirection::Downlink,
            sequence_number,
            plain_message,
        ))
    }

    /// Verify MAC of a received downlink message
    ///
    /// # Arguments
    /// * `count` - The estimated NAS COUNT
    /// * `sequence_number` - The NAS sequence number
    /// * `plain_message` - The plain NAS message
    /// * `received_mac` - The MAC from the received message
    ///
    /// # Returns
    /// * `Ok(())` - If the MAC is valid
    /// * `Err(SecurityError)` - If verification fails or context is not active
    pub fn verify_downlink_mac(
        &self,
        count: &NasCount,
        sequence_number: u8,
        plain_message: &[u8],
        received_mac: &[u8; MAC_SIZE],
    ) -> Result<(), SecurityError> {
        if !self.is_active() {
            return Err(SecurityError::SecurityContextNotActive);
        }

        let key = self.keys.knas_int().ok_or(SecurityError::NoSecurityContext)?;

        verify_nas_mac(
            self.integrity_algorithm,
            key,
            count,
            NasDirection::Downlink,
            sequence_number,
            plain_message,
            received_mac,
        )
    }

    /// Track a received sequence number for replay detection
    pub fn track_sequence_number(&mut self, sqn: u8) {
        if self.recent_sequence_numbers.len() >= self.max_recent_sqn {
            self.recent_sequence_numbers.remove(0);
        }
        self.recent_sequence_numbers.push(sqn);
    }

    /// Check if a sequence number was recently seen (potential replay)
    pub fn is_sequence_number_seen(&self, sqn: u8) -> bool {
        self.recent_sequence_numbers.contains(&sqn)
    }

    /// Create a deep copy of the security context
    pub fn deep_copy(&self) -> Self {
        Self {
            state: self.state,
            tsc: self.tsc,
            ng_ksi: self.ng_ksi,
            downlink_count: self.downlink_count,
            uplink_count: self.uplink_count,
            is_3gpp_access: self.is_3gpp_access,
            keys: self.keys.deep_copy(),
            integrity_algorithm: self.integrity_algorithm,
            ciphering_algorithm: self.ciphering_algorithm,
            recent_sequence_numbers: self.recent_sequence_numbers.clone(),
            max_recent_sqn: self.max_recent_sqn,
        }
    }
}

/// NAS COUNT structure for uplink and downlink message counting
///
/// The NAS COUNT is a 32-bit value composed of:
/// - Overflow counter (16 bits, bits 8-23)
/// - Sequence number (8 bits, bits 0-7)
///
/// The NAS COUNT is used for:
/// - Replay protection (detecting duplicate messages)
/// - Input to ciphering and integrity algorithms
///
/// 3GPP TS 24.501 Section 4.4.3.1
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NasCount {
    /// Overflow counter (16 bits) - increments when SQN wraps around
    pub overflow: u16,
    /// Sequence number (8 bits) - increments with each message
    pub sqn: u8,
}

impl NasCount {
    /// Create a new NAS count with specified overflow and sequence number
    pub fn new(overflow: u16, sqn: u8) -> Self {
        Self { overflow, sqn }
    }

    /// Convert NAS count to 32-bit value for use in crypto algorithms
    ///
    /// Format: [0x00][overflow_high][overflow_low][sqn]
    pub fn to_u32(&self) -> u32 {
        ((self.overflow as u32) << 8) | (self.sqn as u32)
    }

    /// Create NAS count from a 32-bit value
    pub fn from_u32(value: u32) -> Self {
        Self {
            overflow: ((value >> 8) & 0xFFFF) as u16,
            sqn: (value & 0xFF) as u8,
        }
    }

    /// Increment the count (used after sending a message)
    ///
    /// Returns `Err(SecurityError::NasCountOverflow)` if the count would overflow
    /// (i.e., both overflow and sqn are at maximum values)
    pub fn increment(&mut self) -> Result<(), SecurityError> {
        // Check for overflow before incrementing
        if self.sqn == 0xFF && self.overflow == 0xFFFF {
            return Err(SecurityError::NasCountOverflow);
        }

        self.sqn = self.sqn.wrapping_add(1);
        if self.sqn == 0 {
            self.overflow = self.overflow.wrapping_add(1);
        }
        Ok(())
    }

    /// Decrement the count (used to rollback after failed encryption)
    ///
    /// This is used when encryption fails and we need to restore the previous count
    pub fn decrement(&mut self) {
        if self.sqn == 0 {
            self.sqn = 0xFF;
            if self.overflow == 0 {
                self.overflow = 0xFFFF;
            } else {
                self.overflow = self.overflow.wrapping_sub(1);
            }
        } else {
            self.sqn = self.sqn.wrapping_sub(1);
        }
    }

    /// Estimate the full NAS count from a received sequence number
    ///
    /// When receiving a message, only the 8-bit sequence number is transmitted.
    /// This method estimates the full 32-bit count by comparing with the
    /// current expected count and handling wrap-around.
    ///
    /// # Arguments
    /// * `received_sqn` - The sequence number from the received message
    ///
    /// # Returns
    /// The estimated full NAS count
    pub fn estimate_from_sqn(&self, received_sqn: u8) -> NasCount {
        let mut estimated = *self;

        // If received SQN is less than current SQN, overflow has occurred
        if self.sqn > received_sqn {
            estimated.overflow = self.overflow.wrapping_add(1);
        }
        estimated.sqn = received_sqn;

        estimated
    }
}

/// NAS bearer identity for integrity/ciphering
///
/// For NAS messages, the bearer is always 0 as per 3GPP TS 33.501.
pub const NAS_BEARER: u8 = 0;

/// Direction values for NAS security
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NasDirection {
    /// Uplink (UE to network)
    Uplink = 0,
    /// Downlink (network to UE)
    Downlink = 1,
}

/// Compute the Message Authentication Code (MAC) for a NAS message
///
/// This function computes the 32-bit MAC for integrity protection of NAS messages
/// according to 3GPP TS 24.501 Section 4.4.3.
///
/// # Input Construction
/// The input to the integrity algorithm is:
/// - Sequence Number (1 octet)
/// - Plain NAS message (variable)
///
/// # Parameters
/// - `algorithm`: The integrity algorithm to use (NIA0-NIA3)
/// - `key`: 128-bit integrity key (KNASint)
/// - `count`: NAS COUNT value (32-bit)
/// - `direction`: Direction (uplink or downlink)
/// - `sequence_number`: NAS sequence number (8-bit)
/// - `plain_message`: The plain NAS message to protect
///
/// # Returns
/// 32-bit MAC value, or `[0, 0, 0, 0]` for NIA0 (null integrity)
///
/// # Example
/// ```ignore
/// use nextgsim_nas::security::{compute_nas_mac, IntegrityAlgorithm, NasDirection, NasCount};
///
/// let key = [0u8; 16];
/// let count = NasCount::new(0, 1);
/// let message = vec![0x7E, 0x00, 0x41]; // Example NAS message
///
/// let mac = compute_nas_mac(
///     IntegrityAlgorithm::Nia2,
///     &key,
///     &count,
///     NasDirection::Uplink,
///     1,
///     &message,
/// );
/// ```
pub fn compute_nas_mac(
    algorithm: IntegrityAlgorithm,
    key: &[u8; KEY_SIZE],
    count: &NasCount,
    direction: NasDirection,
    sequence_number: u8,
    plain_message: &[u8],
) -> [u8; MAC_SIZE] {
    match algorithm {
        IntegrityAlgorithm::Nia0 => {
            // NIA0 is null integrity - return zero MAC
            [0u8; MAC_SIZE]
        }
        IntegrityAlgorithm::Nia1 => {
            // NIA1 (SNOW3G-based)
            // Build the message: SQN || Plain NAS message
            let mut data = Vec::with_capacity(1 + plain_message.len());
            data.push(sequence_number);
            data.extend_from_slice(plain_message);

            nia1_compute_mac(count.to_u32(), NAS_BEARER, direction as u8, key, &data)
        }
        IntegrityAlgorithm::Nia2 => {
            // NIA2 (AES-CMAC based)
            // Build the message: SQN || Plain NAS message
            let mut data = Vec::with_capacity(1 + plain_message.len());
            data.push(sequence_number);
            data.extend_from_slice(plain_message);

            nia2_compute_mac(count.to_u32(), NAS_BEARER, direction as u8, key, &data)
        }
        IntegrityAlgorithm::Nia3 => {
            // NIA3 (ZUC-based)
            // Build the message: SQN || Plain NAS message
            let mut data = Vec::with_capacity(1 + plain_message.len());
            data.push(sequence_number);
            data.extend_from_slice(plain_message);

            nia3_compute_mac(count.to_u32(), NAS_BEARER, direction as u8, key, &data)
        }
    }
}

/// Verify the Message Authentication Code (MAC) of a received NAS message
///
/// This function verifies the integrity of a received NAS message by computing
/// the expected MAC and comparing it with the received MAC.
///
/// # Parameters
/// - `algorithm`: The integrity algorithm to use (NIA0-NIA3)
/// - `key`: 128-bit integrity key (KNASint)
/// - `count`: NAS COUNT value (32-bit)
/// - `direction`: Direction (uplink or downlink)
/// - `sequence_number`: NAS sequence number (8-bit)
/// - `plain_message`: The plain NAS message
/// - `received_mac`: The MAC received in the message
///
/// # Returns
/// - `Ok(())` if the MAC is valid
/// - `Err(SecurityError::MacVerificationFailed)` if the MAC is invalid
///
/// # Note
/// For NIA0 (null integrity), this function always returns `Ok(())` as no
/// integrity protection is applied.
///
/// # Example
/// ```ignore
/// use nextgsim_nas::security::{verify_nas_mac, IntegrityAlgorithm, NasDirection, NasCount};
///
/// let key = [0u8; 16];
/// let count = NasCount::new(0, 1);
/// let message = vec![0x7E, 0x00, 0x41];
/// let received_mac = [0x12, 0x34, 0x56, 0x78];
///
/// match verify_nas_mac(
///     IntegrityAlgorithm::Nia2,
///     &key,
///     &count,
///     NasDirection::Downlink,
///     1,
///     &message,
///     &received_mac,
/// ) {
///     Ok(()) => println!("MAC verified successfully"),
///     Err(e) => println!("MAC verification failed: {}", e),
/// }
/// ```
pub fn verify_nas_mac(
    algorithm: IntegrityAlgorithm,
    key: &[u8; KEY_SIZE],
    count: &NasCount,
    direction: NasDirection,
    sequence_number: u8,
    plain_message: &[u8],
    received_mac: &[u8; MAC_SIZE],
) -> Result<(), SecurityError> {
    // For NIA0, no integrity check is performed
    if algorithm == IntegrityAlgorithm::Nia0 {
        return Ok(());
    }

    let computed_mac = compute_nas_mac(
        algorithm,
        key,
        count,
        direction,
        sequence_number,
        plain_message,
    );

    // Constant-time comparison to prevent timing attacks
    if constant_time_compare(&computed_mac, received_mac) {
        Ok(())
    } else {
        Err(SecurityError::MacVerificationFailed)
    }
}

/// Constant-time comparison of two byte slices
///
/// This function compares two byte slices in constant time to prevent
/// timing side-channel attacks during MAC verification.
fn constant_time_compare(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut result = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        result |= x ^ y;
    }
    result == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secured_nas_message_creation() {
        let msg = SecuredNasMessage::new(
            ExtendedProtocolDiscriminator::MobilityManagement,
            SecurityHeaderType::IntegrityProtectedAndCiphered,
            [0x12, 0x34, 0x56, 0x78],
            0x01,
            vec![0x7E, 0x00, 0x41],
        );

        assert_eq!(msg.epd, ExtendedProtocolDiscriminator::MobilityManagement);
        assert_eq!(
            msg.security_header_type,
            SecurityHeaderType::IntegrityProtectedAndCiphered
        );
        assert_eq!(msg.mac, [0x12, 0x34, 0x56, 0x78]);
        assert_eq!(msg.sequence_number, 0x01);
        assert!(msg.is_integrity_protected());
        assert!(msg.is_ciphered());
        assert!(!msg.is_new_security_context());
    }

    #[test]
    fn test_secured_nas_message_new_security_context() {
        let msg = SecuredNasMessage::new(
            ExtendedProtocolDiscriminator::MobilityManagement,
            SecurityHeaderType::IntegrityProtectedAndCipheredWithNewSecurityContext,
            [0x00; 4],
            0x00,
            vec![],
        );

        assert!(msg.is_new_security_context());
        assert!(msg.is_ciphered());
    }

    #[test]
    fn test_header_length() {
        assert_eq!(SecuredNasMessage::header_length(), 7);
    }

    #[test]
    fn test_nas_security_algorithms_encode_decode() {
        let algs = NasSecurityAlgorithms::new(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2);
        let encoded = algs.encode();
        assert_eq!(encoded, 0x22);

        let decoded = NasSecurityAlgorithms::decode(encoded).unwrap();
        assert_eq!(decoded, algs);
    }

    #[test]
    fn test_ciphering_algorithm_try_from() {
        assert_eq!(CipheringAlgorithm::try_from(0x00).unwrap(), CipheringAlgorithm::Nea0);
        assert_eq!(CipheringAlgorithm::try_from(0x01).unwrap(), CipheringAlgorithm::Nea1);
        assert_eq!(CipheringAlgorithm::try_from(0x02).unwrap(), CipheringAlgorithm::Nea2);
        assert_eq!(CipheringAlgorithm::try_from(0x03).unwrap(), CipheringAlgorithm::Nea3);
        assert!(CipheringAlgorithm::try_from(0x04).is_err());
    }

    #[test]
    fn test_integrity_algorithm_try_from() {
        assert_eq!(IntegrityAlgorithm::try_from(0x00).unwrap(), IntegrityAlgorithm::Nia0);
        assert_eq!(IntegrityAlgorithm::try_from(0x01).unwrap(), IntegrityAlgorithm::Nia1);
        assert_eq!(IntegrityAlgorithm::try_from(0x02).unwrap(), IntegrityAlgorithm::Nia2);
        assert_eq!(IntegrityAlgorithm::try_from(0x03).unwrap(), IntegrityAlgorithm::Nia3);
        assert!(IntegrityAlgorithm::try_from(0x04).is_err());
    }

    #[test]
    fn test_nas_key_set_identifier() {
        let ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 3);
        assert_eq!(ksi.encode(), 0x03);
        assert!(!ksi.is_no_key());

        let ksi = NasKeySetIdentifier::no_key();
        assert!(ksi.is_no_key());
        assert_eq!(ksi.encode(), 0x07);
    }

    #[test]
    fn test_nas_key_set_identifier_decode() {
        let ksi = NasKeySetIdentifier::decode(0x0B).unwrap(); // TSC=1, KSI=3
        assert_eq!(ksi.tsc, SecurityContextType::Mapped);
        assert_eq!(ksi.ksi, 3);
    }

    #[test]
    fn test_security_context_type() {
        assert_eq!(SecurityContextType::try_from(0).unwrap(), SecurityContextType::Native);
        assert_eq!(SecurityContextType::try_from(1).unwrap(), SecurityContextType::Mapped);
        assert!(SecurityContextType::try_from(2).is_err());
    }

    // NAS Count tests
    #[test]
    fn test_nas_count_creation() {
        let count = NasCount::new(0x1234, 0x56);
        assert_eq!(count.overflow, 0x1234);
        assert_eq!(count.sqn, 0x56);
    }

    #[test]
    fn test_nas_count_default() {
        let count = NasCount::default();
        assert_eq!(count.overflow, 0);
        assert_eq!(count.sqn, 0);
    }

    #[test]
    fn test_nas_count_to_u32() {
        // Test basic conversion
        let count = NasCount::new(0x0000, 0x01);
        assert_eq!(count.to_u32(), 0x00000001);

        // Test with overflow
        let count = NasCount::new(0x0001, 0x00);
        assert_eq!(count.to_u32(), 0x00000100);

        // Test with both overflow and sqn
        let count = NasCount::new(0x1234, 0x56);
        assert_eq!(count.to_u32(), 0x00123456);

        // Test maximum values
        let count = NasCount::new(0xFFFF, 0xFF);
        assert_eq!(count.to_u32(), 0x00FFFFFF);
    }

    #[test]
    fn test_nas_count_from_u32() {
        let count = NasCount::from_u32(0x00123456);
        assert_eq!(count.overflow, 0x1234);
        assert_eq!(count.sqn, 0x56);

        let count = NasCount::from_u32(0x00000001);
        assert_eq!(count.overflow, 0x0000);
        assert_eq!(count.sqn, 0x01);

        let count = NasCount::from_u32(0x00FFFFFF);
        assert_eq!(count.overflow, 0xFFFF);
        assert_eq!(count.sqn, 0xFF);
    }

    #[test]
    fn test_nas_count_roundtrip() {
        let original = NasCount::new(0xABCD, 0xEF);
        let value = original.to_u32();
        let restored = NasCount::from_u32(value);
        assert_eq!(original, restored);
    }

    #[test]
    fn test_nas_count_increment_basic() {
        let mut count = NasCount::new(0, 0);
        assert!(count.increment().is_ok());
        assert_eq!(count.sqn, 1);
        assert_eq!(count.overflow, 0);
    }

    #[test]
    fn test_nas_count_increment_sqn_wrap() {
        let mut count = NasCount::new(0, 0xFF);
        assert!(count.increment().is_ok());
        assert_eq!(count.sqn, 0);
        assert_eq!(count.overflow, 1);
    }

    #[test]
    fn test_nas_count_increment_multiple_wraps() {
        let mut count = NasCount::new(0, 0xFE);
        
        // Increment to 0xFF
        assert!(count.increment().is_ok());
        assert_eq!(count.sqn, 0xFF);
        assert_eq!(count.overflow, 0);
        
        // Wrap to 0x00, overflow becomes 1
        assert!(count.increment().is_ok());
        assert_eq!(count.sqn, 0);
        assert_eq!(count.overflow, 1);
        
        // Continue incrementing
        assert!(count.increment().is_ok());
        assert_eq!(count.sqn, 1);
        assert_eq!(count.overflow, 1);
    }

    #[test]
    fn test_nas_count_increment_overflow_error() {
        let mut count = NasCount::new(0xFFFF, 0xFF);
        let result = count.increment();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SecurityError::NasCountOverflow);
        // Count should remain unchanged
        assert_eq!(count.overflow, 0xFFFF);
        assert_eq!(count.sqn, 0xFF);
    }

    #[test]
    fn test_nas_count_decrement_basic() {
        let mut count = NasCount::new(0, 5);
        count.decrement();
        assert_eq!(count.sqn, 4);
        assert_eq!(count.overflow, 0);
    }

    #[test]
    fn test_nas_count_decrement_sqn_wrap() {
        let mut count = NasCount::new(1, 0);
        count.decrement();
        assert_eq!(count.sqn, 0xFF);
        assert_eq!(count.overflow, 0);
    }

    #[test]
    fn test_nas_count_decrement_at_zero() {
        let mut count = NasCount::new(0, 0);
        count.decrement();
        assert_eq!(count.sqn, 0xFF);
        assert_eq!(count.overflow, 0xFFFF);
    }

    #[test]
    fn test_nas_count_increment_decrement_roundtrip() {
        let mut count = NasCount::new(0x1234, 0x56);
        let original = count;
        
        assert!(count.increment().is_ok());
        count.decrement();
        
        assert_eq!(count, original);
    }

    #[test]
    fn test_nas_count_estimate_from_sqn_no_wrap() {
        let current = NasCount::new(0, 10);
        
        // Received SQN is greater than current - no wrap
        let estimated = current.estimate_from_sqn(15);
        assert_eq!(estimated.overflow, 0);
        assert_eq!(estimated.sqn, 15);
    }

    #[test]
    fn test_nas_count_estimate_from_sqn_with_wrap() {
        let current = NasCount::new(0, 250);
        
        // Received SQN is less than current - wrap occurred
        let estimated = current.estimate_from_sqn(5);
        assert_eq!(estimated.overflow, 1);
        assert_eq!(estimated.sqn, 5);
    }

    #[test]
    fn test_nas_count_estimate_from_sqn_equal() {
        let current = NasCount::new(5, 100);
        
        // Received SQN equals current - no wrap
        let estimated = current.estimate_from_sqn(100);
        assert_eq!(estimated.overflow, 5);
        assert_eq!(estimated.sqn, 100);
    }

    #[test]
    fn test_nas_count_estimate_from_sqn_boundary() {
        // Test at boundary: current SQN is 0xFF, received is 0x00
        let current = NasCount::new(0, 0xFF);
        let estimated = current.estimate_from_sqn(0x00);
        assert_eq!(estimated.overflow, 1);
        assert_eq!(estimated.sqn, 0x00);
    }

    // NAS Integrity Protection tests
    #[test]
    fn test_compute_nas_mac_nia0_returns_zero() {
        let key = [0u8; 16];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41];

        let mac = compute_nas_mac(
            IntegrityAlgorithm::Nia0,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        assert_eq!(mac, [0u8; 4]);
    }

    #[test]
    fn test_compute_nas_mac_nia2_deterministic() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41, 0x01, 0x02, 0x03];

        let mac1 = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        let mac2 = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        assert_eq!(mac1, mac2);
        // MAC should not be all zeros for NIA2
        assert_ne!(mac1, [0u8; 4]);
    }

    #[test]
    fn test_compute_nas_mac_nia3_deterministic() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41, 0x01, 0x02, 0x03];

        let mac1 = compute_nas_mac(
            IntegrityAlgorithm::Nia3,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        let mac2 = compute_nas_mac(
            IntegrityAlgorithm::Nia3,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        assert_eq!(mac1, mac2);
        // MAC should not be all zeros for NIA3
        assert_ne!(mac1, [0u8; 4]);
    }

    #[test]
    fn test_compute_nas_mac_different_messages_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let count = NasCount::new(0, 1);

        let mac1 = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &[0x01, 0x02, 0x03],
        );

        let mac2 = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &[0x04, 0x05, 0x06],
        );

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_compute_nas_mac_different_directions_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41];

        let mac_uplink = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        let mac_downlink = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Downlink,
            1,
            &message,
        );

        assert_ne!(mac_uplink, mac_downlink);
    }

    #[test]
    fn test_verify_nas_mac_nia0_always_succeeds() {
        let key = [0u8; 16];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41];
        let any_mac = [0x12, 0x34, 0x56, 0x78];

        // NIA0 should always succeed regardless of MAC value
        let result = verify_nas_mac(
            IntegrityAlgorithm::Nia0,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
            &any_mac,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_nas_mac_nia2_valid_mac() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41, 0x01, 0x02, 0x03];

        // Compute the correct MAC
        let correct_mac = compute_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
        );

        // Verify should succeed with correct MAC
        let result = verify_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
            &correct_mac,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_nas_mac_nia2_invalid_mac() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41, 0x01, 0x02, 0x03];
        let wrong_mac = [0x00, 0x00, 0x00, 0x00];

        // Verify should fail with wrong MAC
        let result = verify_nas_mac(
            IntegrityAlgorithm::Nia2,
            &key,
            &count,
            NasDirection::Uplink,
            1,
            &message,
            &wrong_mac,
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SecurityError::MacVerificationFailed);
    }

    #[test]
    fn test_verify_nas_mac_nia3_valid_mac() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let count = NasCount::new(0, 1);
        let message = vec![0x7E, 0x00, 0x41, 0x01, 0x02, 0x03];

        // Compute the correct MAC
        let correct_mac = compute_nas_mac(
            IntegrityAlgorithm::Nia3,
            &key,
            &count,
            NasDirection::Downlink,
            1,
            &message,
        );

        // Verify should succeed with correct MAC
        let result = verify_nas_mac(
            IntegrityAlgorithm::Nia3,
            &key,
            &count,
            NasDirection::Downlink,
            1,
            &message,
            &correct_mac,
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_constant_time_compare() {
        assert!(constant_time_compare(&[1, 2, 3, 4], &[1, 2, 3, 4]));
        assert!(!constant_time_compare(&[1, 2, 3, 4], &[1, 2, 3, 5]));
        assert!(!constant_time_compare(&[1, 2, 3], &[1, 2, 3, 4]));
        assert!(constant_time_compare(&[], &[]));
    }

    #[test]
    fn test_nas_direction_values() {
        assert_eq!(NasDirection::Uplink as u8, 0);
        assert_eq!(NasDirection::Downlink as u8, 1);
    }

    #[test]
    fn test_nas_bearer_constant() {
        // NAS bearer should always be 0 per 3GPP spec
        assert_eq!(NAS_BEARER, 0);
    }

    // ============================================
    // Security Context Management Tests
    // ============================================

    #[test]
    fn test_ue_keys_new() {
        let keys = UeKeys::new();
        assert!(keys.kausf().is_none());
        assert!(keys.kseaf().is_none());
        assert!(keys.kamf().is_none());
        assert!(keys.knas_int().is_none());
        assert!(keys.knas_enc().is_none());
        assert!(!keys.has_kamf());
        assert!(!keys.has_nas_keys());
    }

    #[test]
    fn test_ue_keys_set_and_get() {
        let mut keys = UeKeys::new();
        
        let kausf = [0x11u8; 32];
        let kseaf = [0x22u8; 32];
        let kamf = [0x33u8; 32];
        let knas_int = [0x44u8; 16];
        let knas_enc = [0x55u8; 16];
        
        keys.set_kausf(&kausf);
        keys.set_kseaf(&kseaf);
        keys.set_kamf(&kamf);
        keys.set_knas_int(&knas_int);
        keys.set_knas_enc(&knas_enc);
        
        assert_eq!(keys.kausf(), Some(&kausf));
        assert_eq!(keys.kseaf(), Some(&kseaf));
        assert_eq!(keys.kamf(), Some(&kamf));
        assert_eq!(keys.knas_int(), Some(&knas_int));
        assert_eq!(keys.knas_enc(), Some(&knas_enc));
        assert!(keys.has_kamf());
        assert!(keys.has_nas_keys());
    }

    #[test]
    fn test_ue_keys_clear() {
        let mut keys = UeKeys::new();
        keys.set_kamf(&[0x33u8; 32]);
        keys.set_knas_int(&[0x44u8; 16]);
        keys.set_knas_enc(&[0x55u8; 16]);
        keys.abba = vec![0x00, 0x00];
        
        keys.clear();
        
        assert!(keys.kamf().is_none());
        assert!(keys.knas_int().is_none());
        assert!(keys.knas_enc().is_none());
        assert!(keys.abba.is_empty());
        assert!(!keys.has_kamf());
        assert!(!keys.has_nas_keys());
    }

    #[test]
    fn test_ue_keys_deep_copy() {
        let mut keys = UeKeys::new();
        keys.set_kamf(&[0x33u8; 32]);
        keys.set_knas_int(&[0x44u8; 16]);
        keys.abba = vec![0x00, 0x00];
        
        let copy = keys.deep_copy();
        
        assert_eq!(copy.kamf(), keys.kamf());
        assert_eq!(copy.knas_int(), keys.knas_int());
        assert_eq!(copy.abba, keys.abba);
    }

    #[test]
    fn test_security_context_state_default() {
        let state = SecurityContextState::default();
        assert_eq!(state, SecurityContextState::Null);
    }

    #[test]
    fn test_nas_security_context_new() {
        let ctx = NasSecurityContext::new();
        
        assert_eq!(ctx.state(), SecurityContextState::Null);
        assert!(ctx.is_null());
        assert!(!ctx.is_active());
        assert_eq!(ctx.tsc(), SecurityContextType::Native);
        assert_eq!(ctx.ng_ksi(), NasKeySetIdentifier::NO_KEY_AVAILABLE);
        assert!(ctx.is_3gpp_access());
        assert_eq!(ctx.integrity_algorithm(), IntegrityAlgorithm::Nia0);
        assert_eq!(ctx.ciphering_algorithm(), CipheringAlgorithm::Nea0);
    }

    #[test]
    fn test_nas_security_context_3gpp_access() {
        let ctx_3gpp = NasSecurityContext::new_3gpp();
        assert!(ctx_3gpp.is_3gpp_access());
        
        let ctx_non_3gpp = NasSecurityContext::new_non_3gpp();
        assert!(!ctx_non_3gpp.is_3gpp_access());
    }

    #[test]
    fn test_nas_security_context_state_transitions() {
        let mut ctx = NasSecurityContext::new();
        
        // Null -> Establishing
        ctx.begin_establishing();
        assert_eq!(ctx.state(), SecurityContextState::Establishing);
        
        // Establishing -> Inactive
        ctx.set_inactive();
        assert_eq!(ctx.state(), SecurityContextState::Inactive);
        
        // Inactive -> Active
        ctx.activate();
        assert_eq!(ctx.state(), SecurityContextState::Active);
        assert!(ctx.is_active());
        
        // Active -> Null (reset)
        ctx.reset();
        assert_eq!(ctx.state(), SecurityContextState::Null);
        assert!(ctx.is_null());
    }

    #[test]
    fn test_nas_security_context_ksi() {
        let mut ctx = NasSecurityContext::new();
        
        ctx.set_ng_ksi(3);
        assert_eq!(ctx.ng_ksi(), 3);
        
        // Test masking (only 3 bits)
        ctx.set_ng_ksi(0xFF);
        assert_eq!(ctx.ng_ksi(), 0x07);
        
        // Test NasKeySetIdentifier
        let ksi = NasKeySetIdentifier::new(SecurityContextType::Mapped, 5);
        ctx.set_nas_ksi(ksi);
        assert_eq!(ctx.tsc(), SecurityContextType::Mapped);
        assert_eq!(ctx.ng_ksi(), 5);
        
        let retrieved_ksi = ctx.nas_ksi();
        assert_eq!(retrieved_ksi.tsc, SecurityContextType::Mapped);
        assert_eq!(retrieved_ksi.ksi, 5);
    }

    #[test]
    fn test_nas_security_context_algorithms() {
        let mut ctx = NasSecurityContext::new();
        
        ctx.set_algorithms(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2);
        
        assert_eq!(ctx.ciphering_algorithm(), CipheringAlgorithm::Nea2);
        assert_eq!(ctx.integrity_algorithm(), IntegrityAlgorithm::Nia2);
        
        let algs = ctx.algorithms();
        assert_eq!(algs.ciphering, CipheringAlgorithm::Nea2);
        assert_eq!(algs.integrity, IntegrityAlgorithm::Nia2);
    }

    #[test]
    fn test_nas_security_context_derive_nas_keys() {
        let mut ctx = NasSecurityContext::new();
        
        // Should fail without KAMF
        let result = ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SecurityError::NoSecurityContext);
        
        // Set KAMF
        let kamf = [0x55u8; 32];
        ctx.keys_mut().set_kamf(&kamf);
        
        // Now derivation should succeed
        let result = ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2);
        assert!(result.is_ok());
        
        // Keys should be set
        assert!(ctx.keys().has_nas_keys());
        assert!(ctx.keys().knas_int().is_some());
        assert!(ctx.keys().knas_enc().is_some());
        
        // Algorithms should be updated
        assert_eq!(ctx.ciphering_algorithm(), CipheringAlgorithm::Nea2);
        assert_eq!(ctx.integrity_algorithm(), IntegrityAlgorithm::Nia2);
    }

    #[test]
    fn test_nas_security_context_count_management() {
        let mut ctx = NasSecurityContext::new();
        
        // Initial counts should be zero
        assert_eq!(ctx.uplink_count().sqn, 0);
        assert_eq!(ctx.uplink_count().overflow, 0);
        assert_eq!(ctx.downlink_count().sqn, 0);
        assert_eq!(ctx.downlink_count().overflow, 0);
        
        // Increment uplink count
        let count = ctx.increment_uplink_count().unwrap();
        assert_eq!(count.sqn, 0); // Returns count before increment
        assert_eq!(ctx.uplink_count().sqn, 1);
        
        // Rollback
        ctx.rollback_uplink_count();
        assert_eq!(ctx.uplink_count().sqn, 0);
        
        // Update downlink count
        let new_count = NasCount::new(1, 50);
        ctx.update_downlink_count(&new_count);
        assert_eq!(ctx.downlink_count().sqn, 50);
        assert_eq!(ctx.downlink_count().overflow, 1);
        
        // Estimate downlink count
        let estimated = ctx.estimate_downlink_count(60);
        assert_eq!(estimated.sqn, 60);
        assert_eq!(estimated.overflow, 1);
        
        // Reset counts
        ctx.reset_counts();
        assert_eq!(ctx.uplink_count().sqn, 0);
        assert_eq!(ctx.downlink_count().sqn, 0);
    }

    #[test]
    fn test_nas_security_context_sequence_tracking() {
        let mut ctx = NasSecurityContext::new();
        
        // Track some sequence numbers
        ctx.track_sequence_number(1);
        ctx.track_sequence_number(2);
        ctx.track_sequence_number(3);
        
        assert!(ctx.is_sequence_number_seen(1));
        assert!(ctx.is_sequence_number_seen(2));
        assert!(ctx.is_sequence_number_seen(3));
        assert!(!ctx.is_sequence_number_seen(4));
    }

    #[test]
    fn test_nas_security_context_mac_operations() {
        let mut ctx = NasSecurityContext::new();
        
        // Set up keys
        let kamf = [0x55u8; 32];
        ctx.keys_mut().set_kamf(&kamf);
        ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2).unwrap();
        
        // MAC operations should fail when not active
        let message = vec![0x7E, 0x00, 0x41];
        let result = ctx.compute_uplink_mac(1, &message);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), SecurityError::SecurityContextNotActive);
        
        // Activate context
        ctx.activate();
        
        // Now MAC operations should succeed
        let mac = ctx.compute_uplink_mac(1, &message).unwrap();
        assert_ne!(mac, [0u8; 4]); // Should not be zero for NIA2
        
        // Compute downlink MAC
        let count = NasCount::new(0, 1);
        let dl_mac = ctx.compute_downlink_mac(&count, 1, &message).unwrap();
        assert_ne!(dl_mac, [0u8; 4]);
        
        // Verify downlink MAC
        let result = ctx.verify_downlink_mac(&count, 1, &message, &dl_mac);
        assert!(result.is_ok());
        
        // Verify with wrong MAC should fail
        let wrong_mac = [0x00, 0x00, 0x00, 0x00];
        let result = ctx.verify_downlink_mac(&count, 1, &message, &wrong_mac);
        assert!(result.is_err());
    }

    #[test]
    fn test_nas_security_context_reset() {
        let mut ctx = NasSecurityContext::new();
        
        // Set up context
        ctx.set_ng_ksi(3);
        ctx.set_tsc(SecurityContextType::Mapped);
        ctx.keys_mut().set_kamf(&[0x55u8; 32]);
        ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2).unwrap();
        ctx.activate();
        ctx.increment_uplink_count().unwrap();
        ctx.track_sequence_number(1);
        
        // Reset
        ctx.reset();
        
        // Verify everything is reset
        assert_eq!(ctx.state(), SecurityContextState::Null);
        assert_eq!(ctx.tsc(), SecurityContextType::Native);
        assert_eq!(ctx.ng_ksi(), NasKeySetIdentifier::NO_KEY_AVAILABLE);
        assert!(!ctx.keys().has_kamf());
        assert!(!ctx.keys().has_nas_keys());
        assert_eq!(ctx.uplink_count().sqn, 0);
        assert_eq!(ctx.downlink_count().sqn, 0);
        assert_eq!(ctx.integrity_algorithm(), IntegrityAlgorithm::Nia0);
        assert_eq!(ctx.ciphering_algorithm(), CipheringAlgorithm::Nea0);
        assert!(!ctx.is_sequence_number_seen(1));
    }

    #[test]
    fn test_nas_security_context_deep_copy() {
        let mut ctx = NasSecurityContext::new();
        
        // Set up context
        ctx.set_ng_ksi(3);
        ctx.keys_mut().set_kamf(&[0x55u8; 32]);
        ctx.derive_nas_keys(CipheringAlgorithm::Nea2, IntegrityAlgorithm::Nia2).unwrap();
        ctx.activate();
        ctx.increment_uplink_count().unwrap();
        
        // Deep copy
        let copy = ctx.deep_copy();
        
        // Verify copy matches original
        assert_eq!(copy.state(), ctx.state());
        assert_eq!(copy.ng_ksi(), ctx.ng_ksi());
        assert_eq!(copy.uplink_count(), ctx.uplink_count());
        assert_eq!(copy.integrity_algorithm(), ctx.integrity_algorithm());
        assert_eq!(copy.ciphering_algorithm(), ctx.ciphering_algorithm());
        assert_eq!(copy.keys().kamf(), ctx.keys().kamf());
        assert_eq!(copy.keys().knas_int(), ctx.keys().knas_int());
    }
}
