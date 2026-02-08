//! Security context (attestation, TEE) for trustworthy AI
//!
//! Implements security features for SHE per IMT-2030 trustworthy AI requirements.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Trusted Execution Environment (TEE) type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TeeType {
    /// Intel SGX
    IntelSgx,
    /// AMD SEV
    AmdSev,
    /// ARM TrustZone
    ArmTrustZone,
    /// AWS Nitro Enclaves
    AwsNitro,
    /// No TEE (software only)
    None,
}

impl std::fmt::Display for TeeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TeeType::IntelSgx => write!(f, "Intel SGX"),
            TeeType::AmdSev => write!(f, "AMD SEV"),
            TeeType::ArmTrustZone => write!(f, "ARM TrustZone"),
            TeeType::AwsNitro => write!(f, "AWS Nitro"),
            TeeType::None => write!(f, "None"),
        }
    }
}

/// Attestation status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AttestationStatus {
    /// Attestation successful
    Valid,
    /// Attestation failed
    Invalid,
    /// Attestation pending
    Pending,
    /// Attestation expired
    Expired,
    /// Not attested
    NotAttested,
}

impl std::fmt::Display for AttestationStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttestationStatus::Valid => write!(f, "Valid"),
            AttestationStatus::Invalid => write!(f, "Invalid"),
            AttestationStatus::Pending => write!(f, "Pending"),
            AttestationStatus::Expired => write!(f, "Expired"),
            AttestationStatus::NotAttested => write!(f, "Not Attested"),
        }
    }
}

/// Remote attestation evidence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationEvidence {
    /// TEE type
    pub tee_type: TeeType,
    /// Attestation quote (platform-specific)
    pub quote: Vec<u8>,
    /// Measurement (hash of code/data)
    pub measurement: Vec<u8>,
    /// Nonce (replay protection)
    pub nonce: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

impl AttestationEvidence {
    /// Creates new attestation evidence
    pub fn new(tee_type: TeeType, quote: Vec<u8>, measurement: Vec<u8>) -> Self {
        Self {
            tee_type,
            quote,
            measurement,
            nonce: vec![0; 32], // Would be random in production
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }
}

/// Attestation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestationResult {
    /// Status
    pub status: AttestationStatus,
    /// Reason (if invalid)
    pub reason: Option<String>,
    /// Valid until (timestamp in milliseconds since creation)
    #[serde(skip)]
    pub valid_until: Option<Instant>,
    /// Attested properties
    pub properties: HashMap<String, String>,
}

impl AttestationResult {
    /// Creates a successful attestation result
    pub fn valid(validity_duration: Duration) -> Self {
        Self {
            status: AttestationStatus::Valid,
            reason: None,
            valid_until: Some(Instant::now() + validity_duration),
            properties: HashMap::new(),
        }
    }

    /// Creates a failed attestation result
    pub fn invalid(reason: impl Into<String>) -> Self {
        Self {
            status: AttestationStatus::Invalid,
            reason: Some(reason.into()),
            valid_until: None,
            properties: HashMap::new(),
        }
    }

    /// Checks if attestation is still valid
    pub fn is_valid(&self) -> bool {
        if self.status != AttestationStatus::Valid {
            return false;
        }

        if let Some(valid_until) = self.valid_until {
            Instant::now() < valid_until
        } else {
            false
        }
    }
}

/// Security context for a compute node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityContext {
    /// Node ID
    pub node_id: u32,
    /// TEE type
    pub tee_type: TeeType,
    /// Attestation result
    pub attestation: AttestationResult,
    /// Last attestation time (timestamp, not serializable)
    #[serde(skip)]
    pub last_attestation: Option<Instant>,
    /// Security level (0-5, higher is more secure)
    pub security_level: u8,
    /// Enabled security features
    pub features: SecurityFeatures,
}

impl SecurityContext {
    /// Creates a new security context
    pub fn new(node_id: u32, tee_type: TeeType) -> Self {
        let security_level = match tee_type {
            TeeType::IntelSgx | TeeType::AmdSev => 5,
            TeeType::ArmTrustZone | TeeType::AwsNitro => 4,
            TeeType::None => 1,
        };

        Self {
            node_id,
            tee_type,
            attestation: AttestationResult {
                status: AttestationStatus::NotAttested,
                reason: None,
                valid_until: None,
                properties: HashMap::new(),
            },
            last_attestation: None,
            security_level,
            features: SecurityFeatures::default(),
        }
    }

    /// Performs attestation
    pub fn attest(&mut self, evidence: AttestationEvidence) -> AttestationResult {
        // Simplified attestation verification
        // In production, this would verify signatures, check revocation lists, etc.

        let result = if evidence.tee_type == self.tee_type && !evidence.quote.is_empty() {
            AttestationResult::valid(Duration::from_secs(3600)) // 1 hour validity
        } else {
            AttestationResult::invalid("TEE type mismatch or invalid quote")
        };

        self.attestation = result.clone();
        self.last_attestation = Some(Instant::now());

        result
    }

    /// Checks if the security context meets minimum requirements
    pub fn meets_requirements(&self, min_security_level: u8, require_attestation: bool) -> bool {
        if self.security_level < min_security_level {
            return false;
        }

        if require_attestation && !self.attestation.is_valid() {
            return false;
        }

        true
    }

    /// Checks if attestation needs renewal
    pub fn needs_attestation(&self) -> bool {
        if self.attestation.status == AttestationStatus::NotAttested {
            return true;
        }

        !self.attestation.is_valid()
    }
}

/// Security features bitmask
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct SecurityFeatures {
    /// Memory encryption enabled
    pub memory_encryption: bool,
    /// Encrypted communication channels
    pub encrypted_channels: bool,
    /// Secure boot
    pub secure_boot: bool,
    /// Runtime integrity measurement
    pub runtime_integrity: bool,
    /// Side-channel resistance
    pub side_channel_resistance: bool,
}

impl SecurityFeatures {
    /// Creates security features for a TEE type
    pub fn for_tee(tee_type: TeeType) -> Self {
        match tee_type {
            TeeType::IntelSgx => Self {
                memory_encryption: true,
                encrypted_channels: true,
                secure_boot: false,
                runtime_integrity: true,
                side_channel_resistance: true,
            },
            TeeType::AmdSev => Self {
                memory_encryption: true,
                encrypted_channels: false,
                secure_boot: true,
                runtime_integrity: true,
                side_channel_resistance: false,
            },
            TeeType::ArmTrustZone => Self {
                memory_encryption: true,
                encrypted_channels: false,
                secure_boot: true,
                runtime_integrity: true,
                side_channel_resistance: false,
            },
            TeeType::AwsNitro => Self {
                memory_encryption: true,
                encrypted_channels: true,
                secure_boot: true,
                runtime_integrity: true,
                side_channel_resistance: true,
            },
            TeeType::None => Self::default(),
        }
    }
}

/// Security manager for tracking node security contexts
#[derive(Debug, Default)]
pub struct SecurityManager {
    /// Security contexts by node ID
    contexts: HashMap<u32, SecurityContext>,
    /// Minimum security level required
    min_security_level: u8,
    /// Whether attestation is required
    require_attestation: bool,
}

impl SecurityManager {
    /// Creates a new security manager
    pub fn new(min_security_level: u8, require_attestation: bool) -> Self {
        Self {
            contexts: HashMap::new(),
            min_security_level,
            require_attestation,
        }
    }

    /// Registers a node's security context
    pub fn register_node(&mut self, node_id: u32, tee_type: TeeType) {
        let mut context = SecurityContext::new(node_id, tee_type);
        context.features = SecurityFeatures::for_tee(tee_type);
        self.contexts.insert(node_id, context);
    }

    /// Gets a security context
    pub fn get_context(&self, node_id: u32) -> Option<&SecurityContext> {
        self.contexts.get(&node_id)
    }

    /// Gets a mutable security context
    pub fn get_context_mut(&mut self, node_id: u32) -> Option<&mut SecurityContext> {
        self.contexts.get_mut(&node_id)
    }

    /// Attests a node
    pub fn attest_node(&mut self, node_id: u32, evidence: AttestationEvidence) -> Option<AttestationResult> {
        self.contexts.get_mut(&node_id).map(|context| context.attest(evidence))
    }

    /// Checks if a node is trusted (meets security requirements)
    pub fn is_node_trusted(&self, node_id: u32) -> bool {
        if let Some(context) = self.contexts.get(&node_id) {
            context.meets_requirements(self.min_security_level, self.require_attestation)
        } else {
            false
        }
    }

    /// Returns nodes that need attestation renewal
    pub fn nodes_needing_attestation(&self) -> Vec<u32> {
        self.contexts
            .iter()
            .filter(|(_, ctx)| ctx.needs_attestation())
            .map(|(id, _)| *id)
            .collect()
    }

    /// Returns the number of trusted nodes
    pub fn trusted_node_count(&self) -> usize {
        self.contexts
            .values()
            .filter(|ctx| ctx.meets_requirements(self.min_security_level, self.require_attestation))
            .count()
    }

    /// Sets minimum security level
    pub fn set_min_security_level(&mut self, level: u8) {
        self.min_security_level = level;
    }

    /// Sets attestation requirement
    pub fn set_require_attestation(&mut self, required: bool) {
        self.require_attestation = required;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tee_type_display() {
        assert_eq!(format!("{}", TeeType::IntelSgx), "Intel SGX");
        assert_eq!(format!("{}", TeeType::None), "None");
    }

    #[test]
    fn test_security_context_creation() {
        let ctx = SecurityContext::new(1, TeeType::IntelSgx);
        assert_eq!(ctx.node_id, 1);
        assert_eq!(ctx.tee_type, TeeType::IntelSgx);
        assert_eq!(ctx.security_level, 5);
        assert_eq!(ctx.attestation.status, AttestationStatus::NotAttested);
    }

    #[test]
    fn test_attestation_valid() {
        let mut ctx = SecurityContext::new(1, TeeType::IntelSgx);

        let evidence = AttestationEvidence::new(
            TeeType::IntelSgx,
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        );

        let result = ctx.attest(evidence);
        assert_eq!(result.status, AttestationStatus::Valid);
        assert!(result.is_valid());
        assert!(ctx.last_attestation.is_some());
    }

    #[test]
    fn test_attestation_invalid() {
        let mut ctx = SecurityContext::new(1, TeeType::IntelSgx);

        let evidence = AttestationEvidence::new(
            TeeType::AmdSev, // Wrong TEE type
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        );

        let result = ctx.attest(evidence);
        assert_eq!(result.status, AttestationStatus::Invalid);
        assert!(!result.is_valid());
    }

    #[test]
    fn test_security_requirements() {
        let mut ctx = SecurityContext::new(1, TeeType::IntelSgx);

        // Without attestation
        assert!(!ctx.meets_requirements(5, true));

        // With valid attestation
        let evidence = AttestationEvidence::new(
            TeeType::IntelSgx,
            vec![1, 2, 3, 4],
            vec![5, 6, 7, 8],
        );
        ctx.attest(evidence);

        assert!(ctx.meets_requirements(5, true));
        assert!(ctx.meets_requirements(3, true));
        assert!(!ctx.meets_requirements(6, true));
    }

    #[test]
    fn test_security_features() {
        let sgx_features = SecurityFeatures::for_tee(TeeType::IntelSgx);
        assert!(sgx_features.memory_encryption);
        assert!(sgx_features.side_channel_resistance);

        let none_features = SecurityFeatures::for_tee(TeeType::None);
        assert!(!none_features.memory_encryption);
    }

    #[test]
    fn test_security_manager() {
        let mut manager = SecurityManager::new(3, true);

        manager.register_node(1, TeeType::IntelSgx);
        manager.register_node(2, TeeType::None);

        assert_eq!(manager.contexts.len(), 2);

        // Node 2 should not be trusted (security level too low)
        assert!(!manager.is_node_trusted(2));

        // Node 1 needs attestation
        assert!(manager.nodes_needing_attestation().contains(&1));

        // Attest node 1
        let evidence = AttestationEvidence::new(
            TeeType::IntelSgx,
            vec![1, 2, 3],
            vec![4, 5, 6],
        );
        manager.attest_node(1, evidence);

        // Now node 1 should be trusted
        assert!(manager.is_node_trusted(1));
        assert_eq!(manager.trusted_node_count(), 1);
    }

    #[test]
    fn test_attestation_expiry() {
        let result = AttestationResult::valid(Duration::from_millis(100));
        assert!(result.is_valid());

        std::thread::sleep(Duration::from_millis(150));
        assert!(!result.is_valid());
    }
}
