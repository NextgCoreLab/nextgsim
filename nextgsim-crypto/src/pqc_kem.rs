//! ML-KEM (CRYSTALS-Kyber) post-quantum key encapsulation mechanism
//!
//! Implements ML-KEM (Module-Lattice-Based Key Encapsulation Mechanism)
//! as standardized in FIPS 203, supporting three security levels:
//! - ML-KEM-512: NIST Level 1 security
//! - ML-KEM-768: NIST Level 3 security (recommended)
//! - ML-KEM-1024: NIST Level 5 security
//!
//! ML-KEM provides quantum-resistant key encapsulation for establishing
//! shared secrets between parties.

use ml_kem::kem::{Decapsulate, Encapsulate};
use ml_kem::{EncodedSizeUser, KemCore, MlKem512, MlKem768, MlKem1024};
use rand::rngs::OsRng;
use thiserror::Error;

/// ML-KEM security level parameter sets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MlKemLevel {
    /// ML-KEM-512: NIST Level 1 security (smallest keys and ciphertexts)
    MlKem512,
    /// ML-KEM-768: NIST Level 3 security (recommended for most applications)
    MlKem768,
    /// ML-KEM-1024: NIST Level 5 security (highest security)
    MlKem1024,
}

/// ML-KEM error types
#[derive(Debug, Error)]
pub enum MlKemError {
    /// Encapsulation failed
    #[error("Encapsulation failed: {0}")]
    EncapsulationFailed(String),
    /// Decapsulation failed
    #[error("Decapsulation failed: {0}")]
    DecapsulationFailed(String),
    /// Invalid key data
    #[error("Invalid key data: {0}")]
    InvalidKeyData(String),
}

/// Result type for ML-KEM operations
pub type MlKemResult<T> = Result<T, MlKemError>;

/// ML-KEM key pair wrapping encapsulation and decapsulation keys
///
/// Stores the key pair as serialized bytes so it can work across
/// all three parameter sets (512, 768, 1024).
#[derive(Clone)]
pub struct MlKemKeyPair {
    /// The security level of this key pair
    level: MlKemLevel,
    /// Serialized encapsulation (public) key
    encapsulation_key: Vec<u8>,
    /// Serialized decapsulation (secret) key
    decapsulation_key: Vec<u8>,
}

impl MlKemKeyPair {
    /// Get the security level of this key pair
    pub fn level(&self) -> MlKemLevel {
        self.level
    }

    /// Get the encapsulation (public) key bytes
    pub fn encapsulation_key(&self) -> &[u8] {
        &self.encapsulation_key
    }

    /// Get the decapsulation (secret) key bytes
    pub fn decapsulation_key(&self) -> &[u8] {
        &self.decapsulation_key
    }
}

/// Concrete encapsulation/decapsulation using a macro to avoid generic trait bound issues
macro_rules! ml_kem_ops {
    ($kem_type:ty, $ek_bytes:expr, encapsulate) => {{
        let ek_encoded = $ek_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid encapsulation key length: {}",
                $ek_bytes.len()
            ))
        })?;
        let ek = <$kem_type as KemCore>::EncapsulationKey::from_bytes(ek_encoded);
        let (ct, ss) = ek.encapsulate(&mut OsRng).map_err(|e| {
            MlKemError::EncapsulationFailed(format!("{e:?}"))
        })?;
        let ct_slice: &[u8] = ct.as_ref();
        let ss_slice: &[u8] = ss.as_ref();
        Ok((ct_slice.to_vec(), ss_slice.to_vec()))
    }};
    ($kem_type:ty, $dk_bytes:expr, $ct_bytes:expr, decapsulate) => {{
        let dk_encoded = $dk_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid decapsulation key length: {}",
                $dk_bytes.len()
            ))
        })?;
        let dk = <$kem_type as KemCore>::DecapsulationKey::from_bytes(dk_encoded);
        let ct_encoded = $ct_bytes.try_into().map_err(|_| {
            MlKemError::InvalidKeyData(format!(
                "Invalid ciphertext length: {}",
                $ct_bytes.len()
            ))
        })?;
        let ss = dk.decapsulate(ct_encoded).map_err(|e| {
            MlKemError::DecapsulationFailed(format!("{e:?}"))
        })?;
        let ss_slice: &[u8] = ss.as_ref();
        Ok(ss_slice.to_vec())
    }};
}

/// Generate an ML-KEM key pair at the specified security level
///
/// # Arguments
/// * `level` - The ML-KEM security level to use
///
/// # Returns
/// A new ML-KEM key pair
pub fn ml_kem_generate_keypair(level: MlKemLevel) -> MlKemKeyPair {
    match level {
        MlKemLevel::MlKem512 => {
            let (dk, ek) = MlKem512::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem768 => {
            let (dk, ek) = MlKem768::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
        MlKemLevel::MlKem1024 => {
            let (dk, ek) = MlKem1024::generate(&mut OsRng);
            MlKemKeyPair {
                level,
                encapsulation_key: ek.as_bytes().to_vec(),
                decapsulation_key: dk.as_bytes().to_vec(),
            }
        }
    }
}

/// Encapsulate a shared secret using an ML-KEM public key
///
/// Generates a ciphertext and shared secret from the given encapsulation key.
///
/// # Arguments
/// * `level` - The ML-KEM security level
/// * `encapsulation_key` - The recipient's encapsulation (public) key bytes
///
/// # Returns
/// A tuple of (ciphertext_bytes, shared_secret_bytes)
///
/// # Errors
/// Returns an error if the encapsulation key is invalid.
pub fn ml_kem_encapsulate(
    level: MlKemLevel,
    encapsulation_key: &[u8],
) -> MlKemResult<(Vec<u8>, Vec<u8>)> {
    match level {
        MlKemLevel::MlKem512 => {
            ml_kem_ops!(MlKem512, encapsulation_key, encapsulate)
        }
        MlKemLevel::MlKem768 => {
            ml_kem_ops!(MlKem768, encapsulation_key, encapsulate)
        }
        MlKemLevel::MlKem1024 => {
            ml_kem_ops!(MlKem1024, encapsulation_key, encapsulate)
        }
    }
}

/// Decapsulate a shared secret using an ML-KEM secret key
///
/// # Arguments
/// * `level` - The ML-KEM security level
/// * `decapsulation_key` - The recipient's decapsulation (secret) key bytes
/// * `ciphertext` - The ciphertext from encapsulation
///
/// # Returns
/// The shared secret bytes (32 bytes)
///
/// # Errors
/// Returns an error if the decapsulation key or ciphertext is invalid.
pub fn ml_kem_decapsulate(
    level: MlKemLevel,
    decapsulation_key: &[u8],
    ciphertext: &[u8],
) -> MlKemResult<Vec<u8>> {
    match level {
        MlKemLevel::MlKem512 => {
            ml_kem_ops!(MlKem512, decapsulation_key, ciphertext, decapsulate)
        }
        MlKemLevel::MlKem768 => {
            ml_kem_ops!(MlKem768, decapsulation_key, ciphertext, decapsulate)
        }
        MlKemLevel::MlKem1024 => {
            ml_kem_ops!(MlKem1024, decapsulation_key, ciphertext, decapsulate)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_kem_512_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem512);
        assert_eq!(kp.level(), MlKemLevel::MlKem512);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_768_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        assert_eq!(kp.level(), MlKemLevel::MlKem768);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_1024_keygen() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem1024);
        assert_eq!(kp.level(), MlKemLevel::MlKem1024);
        assert!(!kp.encapsulation_key().is_empty());
        assert!(!kp.decapsulation_key().is_empty());
    }

    #[test]
    fn test_ml_kem_768_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp.encapsulation_key()).expect("encapsulate");
        let ss_dec =
            ml_kem_decapsulate(MlKemLevel::MlKem768, kp.decapsulation_key(), &ct)
                .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_512_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem512);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem512, kp.encapsulation_key()).expect("encapsulate");
        let ss_dec =
            ml_kem_decapsulate(MlKemLevel::MlKem512, kp.decapsulation_key(), &ct)
                .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_1024_roundtrip() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem1024);
        let (ct, ss_enc) =
            ml_kem_encapsulate(MlKemLevel::MlKem1024, kp.encapsulation_key())
                .expect("encapsulate");
        let ss_dec =
            ml_kem_decapsulate(MlKemLevel::MlKem1024, kp.decapsulation_key(), &ct)
                .expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_ml_kem_different_keypairs_different_secrets() {
        let kp1 = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let kp2 = ml_kem_generate_keypair(MlKemLevel::MlKem768);

        let (_, ss1) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp1.encapsulation_key())
                .expect("encapsulate");
        let (_, ss2) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp2.encapsulation_key())
                .expect("encapsulate");

        assert_ne!(ss1, ss2);
    }

    #[test]
    fn test_ml_kem_invalid_encapsulation_key() {
        let result = ml_kem_encapsulate(MlKemLevel::MlKem768, &[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ml_kem_invalid_decapsulation_key() {
        let kp = ml_kem_generate_keypair(MlKemLevel::MlKem768);
        let (ct, _) =
            ml_kem_encapsulate(MlKemLevel::MlKem768, kp.encapsulation_key())
                .expect("encapsulate");
        let result = ml_kem_decapsulate(MlKemLevel::MlKem768, &[0u8; 10], &ct);
        assert!(result.is_err());
    }
}
