//! ML-DSA (CRYSTALS-Dilithium) post-quantum digital signature scheme
//!
//! Implements ML-DSA (Module-Lattice-Based Digital Signature Algorithm)
//! as standardized in FIPS 204, supporting three security levels:
//! - ML-DSA-44: NIST Level 2 security
//! - ML-DSA-65: NIST Level 3 security (recommended)
//! - ML-DSA-87: NIST Level 5 security
//!
//! ML-DSA provides quantum-resistant digital signatures for authentication
//! and integrity in 5G/6G security protocols.

use ml_dsa::{MlDsa44, MlDsa65, MlDsa87};
use ml_dsa::signature::Signer;
use ml_dsa::signature::Verifier;
use rand::rngs::OsRng;
use rand::RngCore;
use thiserror::Error;

/// ML-DSA security level parameter sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlDsaLevel {
    /// ML-DSA-44: NIST Level 2 security
    MlDsa44,
    /// ML-DSA-65: NIST Level 3 security (recommended)
    MlDsa65,
    /// ML-DSA-87: NIST Level 5 security (highest security)
    MlDsa87,
}

/// ML-DSA error types
#[derive(Debug, Error)]
pub enum MlDsaError {
    /// Signing failed
    #[error("Signing failed: {0}")]
    SigningFailed(String),
    /// Verification failed
    #[error("Verification failed: {0}")]
    VerificationFailed(String),
    /// Invalid key data
    #[error("Invalid key data: {0}")]
    InvalidKeyData(String),
    /// Invalid signature data
    #[error("Invalid signature data: {0}")]
    InvalidSignature(String),
}

/// Result type for ML-DSA operations
pub type MlDsaResult<T> = Result<T, MlDsaError>;

/// ML-DSA key pair wrapping signing and verifying keys
///
/// Stores a 32-byte seed for the signing key (from which the full signing key
/// can be deterministically derived via `from_seed`) and the encoded verifying key.
#[derive(Clone)]
pub struct MlDsaKeyPair {
    /// The security level of this key pair
    level: MlDsaLevel,
    /// 32-byte seed for signing key reconstruction
    signing_key_seed: Vec<u8>,
    /// Serialized verifying (public) key
    verifying_key: Vec<u8>,
}

impl MlDsaKeyPair {
    /// Get the security level of this key pair
    pub fn level(&self) -> MlDsaLevel {
        self.level
    }

    /// Get the signing key seed bytes
    pub fn signing_key(&self) -> &[u8] {
        &self.signing_key_seed
    }

    /// Get the verifying (public) key bytes
    pub fn verifying_key(&self) -> &[u8] {
        &self.verifying_key
    }
}

/// Generate an ML-DSA key pair at the specified security level
///
/// # Arguments
/// * `level` - The ML-DSA security level to use
///
/// # Returns
/// A new ML-DSA key pair
pub fn ml_dsa_generate_keypair(level: MlDsaLevel) -> MlDsaKeyPair {
    // Generate a random 32-byte seed
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);

    let vk_bytes = match level {
        MlDsaLevel::MlDsa44 => {
            let sk = ml_dsa::SigningKey::<MlDsa44>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
        MlDsaLevel::MlDsa65 => {
            let sk = ml_dsa::SigningKey::<MlDsa65>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
        MlDsaLevel::MlDsa87 => {
            let sk = ml_dsa::SigningKey::<MlDsa87>::from_seed((&seed).into());
            let vk = sk.verifying_key();
            vk.encode().to_vec()
        }
    };

    MlDsaKeyPair {
        level,
        signing_key_seed: seed.to_vec(),
        verifying_key: vk_bytes,
    }
}

/// Sign a message using an ML-DSA signing key
///
/// # Arguments
/// * `level` - The ML-DSA security level
/// * `signing_key` - The signing key seed bytes (32 bytes)
/// * `message` - The message to sign
///
/// # Returns
/// The digital signature bytes
///
/// # Errors
/// Returns an error if the signing key is invalid.
pub fn ml_dsa_sign(
    level: MlDsaLevel,
    signing_key: &[u8],
    message: &[u8],
) -> MlDsaResult<Vec<u8>> {
    let seed: &[u8; 32] = signing_key.try_into().map_err(|_| {
        MlDsaError::InvalidKeyData(format!(
            "Invalid signing key seed length: {} (expected 32)",
            signing_key.len()
        ))
    })?;

    match level {
        MlDsaLevel::MlDsa44 => {
            let sk = ml_dsa::SigningKey::<MlDsa44>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
        MlDsaLevel::MlDsa65 => {
            let sk = ml_dsa::SigningKey::<MlDsa65>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
        MlDsaLevel::MlDsa87 => {
            let sk = ml_dsa::SigningKey::<MlDsa87>::from_seed(seed.into());
            let sig = sk.sign(message);
            Ok(sig.encode().to_vec())
        }
    }
}

/// Verify a signature using an ML-DSA verifying key
///
/// # Arguments
/// * `level` - The ML-DSA security level
/// * `verifying_key` - The verifying (public) key bytes
/// * `message` - The original message
/// * `signature` - The signature to verify
///
/// # Returns
/// `true` if the signature is valid, `false` otherwise
///
/// # Errors
/// Returns an error if the verifying key or signature format is invalid.
pub fn ml_dsa_verify(
    level: MlDsaLevel,
    verifying_key: &[u8],
    message: &[u8],
    signature: &[u8],
) -> MlDsaResult<bool> {
    match level {
        MlDsaLevel::MlDsa44 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-44 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa44>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-44 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa44>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-44 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
        MlDsaLevel::MlDsa65 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-65 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa65>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-65 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa65>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-65 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
        MlDsaLevel::MlDsa87 => {
            let vk_encoded = verifying_key.try_into().map_err(|_| {
                MlDsaError::InvalidKeyData(format!(
                    "Invalid ML-DSA-87 verifying key length: {}",
                    verifying_key.len()
                ))
            })?;
            let vk = ml_dsa::VerifyingKey::<MlDsa87>::decode(vk_encoded);
            let sig_encoded = signature.try_into().map_err(|_| {
                MlDsaError::InvalidSignature(format!(
                    "Invalid ML-DSA-87 signature length: {}",
                    signature.len()
                ))
            })?;
            let sig = ml_dsa::Signature::<MlDsa87>::decode(sig_encoded).ok_or_else(|| {
                MlDsaError::InvalidSignature("Failed to decode ML-DSA-87 signature".to_string())
            })?;
            match vk.verify(message, &sig) {
                Ok(()) => Ok(true),
                Err(_) => Ok(false),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_dsa_44_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa44);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa44);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_65_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa65);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_87_keygen() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa87);
        assert_eq!(kp.level(), MlDsaLevel::MlDsa87);
        assert!(!kp.signing_key().is_empty());
        assert!(!kp.verifying_key().is_empty());
    }

    #[test]
    fn test_ml_dsa_44_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa44);
        let message = b"Test message for ML-DSA-44";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa44, kp.signing_key(), message)
            .expect("sign");
        let valid = ml_dsa_verify(MlDsaLevel::MlDsa44, kp.verifying_key(), message, &sig)
            .expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_65_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Test message for ML-DSA-65";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message)
            .expect("sign");
        let valid = ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), message, &sig)
            .expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_87_sign_verify() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa87);
        let message = b"Test message for ML-DSA-87";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa87, kp.signing_key(), message)
            .expect("sign");
        let valid = ml_dsa_verify(MlDsaLevel::MlDsa87, kp.verifying_key(), message, &sig)
            .expect("verify");

        assert!(valid);
    }

    #[test]
    fn test_ml_dsa_65_wrong_message_fails() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Original message";
        let wrong_message = b"Tampered message";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message)
            .expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), wrong_message, &sig)
                .expect("verify");

        assert!(!valid);
    }

    #[test]
    fn test_ml_dsa_65_wrong_key_fails() {
        let kp1 = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let kp2 = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"Test message";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp1.signing_key(), message)
            .expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp2.verifying_key(), message, &sig)
                .expect("verify");

        assert!(!valid);
    }

    #[test]
    fn test_ml_dsa_invalid_signing_key() {
        let result = ml_dsa_sign(MlDsaLevel::MlDsa65, &[0u8; 10], b"test");
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_dsa_invalid_verifying_key() {
        let result = ml_dsa_verify(MlDsaLevel::MlDsa65, &[0u8; 10], b"test", &[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_dsa_empty_message() {
        let kp = ml_dsa_generate_keypair(MlDsaLevel::MlDsa65);
        let message = b"";

        let sig = ml_dsa_sign(MlDsaLevel::MlDsa65, kp.signing_key(), message)
            .expect("sign");
        let valid =
            ml_dsa_verify(MlDsaLevel::MlDsa65, kp.verifying_key(), message, &sig)
                .expect("verify");

        assert!(valid);
    }
}
