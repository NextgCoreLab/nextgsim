//! Hybrid key exchange combining classical and post-quantum cryptography
//!
//! Implements a hybrid key exchange mechanism that combines:
//! - X25519 (classical Diffie-Hellman on Curve25519)
//! - ML-KEM-768 (post-quantum key encapsulation)
//!
//! The combined shared secret is derived as:
//! `shared_secret = SHA-256(x25519_shared_secret || ml_kem_shared_secret)`
//!
//! This provides defense-in-depth: the scheme is secure as long as at least
//! one of the underlying primitives remains unbroken.

use ml_kem::kem::{Decapsulate, Encapsulate};
use ml_kem::{EncodedSizeUser, KemCore, MlKem768};
use rand::rngs::OsRng;
use rand::RngCore;
use sha2::{Digest, Sha256};
use thiserror::Error;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret as X25519Secret};

/// X25519 key size in bytes
const X25519_KEY_SIZE: usize = 32;

/// Hybrid key exchange error types
#[derive(Debug, Error)]
pub enum HybridError {
    /// X25519 key exchange failed
    #[error("X25519 key exchange failed: {0}")]
    X25519Error(String),
    /// ML-KEM operation failed
    #[error("ML-KEM operation failed: {0}")]
    MlKemError(String),
    /// Invalid key data
    #[error("Invalid key data: {0}")]
    InvalidKeyData(String),
    /// Invalid ciphertext
    #[error("Invalid ciphertext: {0}")]
    InvalidCiphertext(String),
}

/// Result type for hybrid operations
pub type HybridResult<T> = Result<T, HybridError>;

/// Hybrid key pair containing both X25519 and ML-KEM-768 keys
pub struct HybridKeyPair {
    /// X25519 secret key
    x25519_secret: [u8; X25519_KEY_SIZE],
    /// X25519 public key
    x25519_public: [u8; X25519_KEY_SIZE],
    /// ML-KEM-768 encapsulation (public) key
    ml_kem_encapsulation_key: Vec<u8>,
    /// ML-KEM-768 decapsulation (secret) key
    ml_kem_decapsulation_key: Vec<u8>,
}

impl HybridKeyPair {
    /// Get the X25519 public key
    pub fn x25519_public_key(&self) -> &[u8; X25519_KEY_SIZE] {
        &self.x25519_public
    }

    /// Get the ML-KEM-768 encapsulation (public) key bytes
    pub fn ml_kem_encapsulation_key(&self) -> &[u8] {
        &self.ml_kem_encapsulation_key
    }

    /// Get the combined public key (X25519 public || ML-KEM encapsulation key)
    pub fn combined_public_key(&self) -> Vec<u8> {
        let mut combined = Vec::with_capacity(
            X25519_KEY_SIZE + self.ml_kem_encapsulation_key.len(),
        );
        combined.extend_from_slice(&self.x25519_public);
        combined.extend_from_slice(&self.ml_kem_encapsulation_key);
        combined
    }
}

/// Generate a hybrid key pair (X25519 + ML-KEM-768)
///
/// # Returns
/// A new hybrid key pair containing both classical and post-quantum keys
pub fn hybrid_generate_keypair() -> HybridKeyPair {
    // Generate X25519 key pair
    let mut rng = OsRng;
    let mut x25519_seed = [0u8; X25519_KEY_SIZE];
    rng.fill_bytes(&mut x25519_seed);
    let x25519_secret = X25519Secret::from(x25519_seed);
    let x25519_public = X25519PublicKey::from(&x25519_secret);

    // Generate ML-KEM-768 key pair
    let (dk, ek) = MlKem768::generate(&mut OsRng);

    HybridKeyPair {
        x25519_secret: x25519_seed,
        x25519_public: *x25519_public.as_bytes(),
        ml_kem_encapsulation_key: ek.as_bytes().to_vec(),
        ml_kem_decapsulation_key: dk.as_bytes().to_vec(),
    }
}

/// Perform hybrid encapsulation using the recipient's hybrid public key
///
/// This performs:
/// 1. X25519 Diffie-Hellman key agreement with an ephemeral key pair
/// 2. ML-KEM-768 encapsulation
/// 3. Combines shared secrets: `SHA-256(x25519_ss || ml_kem_ss)`
///
/// # Arguments
/// * `hybrid_pk` - The recipient's hybrid public key (combined X25519 + ML-KEM)
///
/// # Returns
/// A tuple of (combined_ciphertext, shared_secret) where:
/// - combined_ciphertext = X25519_ephemeral_public (32 bytes) || ML-KEM ciphertext
/// - shared_secret = SHA-256(x25519_ss || ml_kem_ss) (32 bytes)
///
/// # Errors
/// Returns an error if the public key is invalid or encapsulation fails.
pub fn hybrid_encapsulate(hybrid_pk: &HybridKeyPair) -> HybridResult<(Vec<u8>, [u8; 32])> {
    hybrid_encapsulate_from_keys(
        hybrid_pk.x25519_public_key(),
        hybrid_pk.ml_kem_encapsulation_key(),
    )
}

/// Perform hybrid encapsulation from raw public key components
///
/// # Arguments
/// * `x25519_public_key` - The recipient's X25519 public key (32 bytes)
/// * `ml_kem_ek` - The recipient's ML-KEM-768 encapsulation key bytes
///
/// # Returns
/// A tuple of (combined_ciphertext, shared_secret)
///
/// # Errors
/// Returns an error if the public keys are invalid or encapsulation fails.
pub fn hybrid_encapsulate_from_keys(
    x25519_public_key: &[u8; X25519_KEY_SIZE],
    ml_kem_ek: &[u8],
) -> HybridResult<(Vec<u8>, [u8; 32])> {
    // Step 1: X25519 ephemeral key exchange
    let mut rng = OsRng;
    let mut eph_seed = [0u8; X25519_KEY_SIZE];
    rng.fill_bytes(&mut eph_seed);
    let eph_secret = X25519Secret::from(eph_seed);
    let eph_public = X25519PublicKey::from(&eph_secret);

    let their_public = X25519PublicKey::from(*x25519_public_key);
    let x25519_ss = eph_secret.diffie_hellman(&their_public);

    // Step 2: ML-KEM-768 encapsulation
    let ek = <MlKem768 as KemCore>::EncapsulationKey::from_bytes(
        ml_kem_ek.try_into().map_err(|_| {
            HybridError::InvalidKeyData(format!(
                "Invalid ML-KEM-768 encapsulation key length: {}",
                ml_kem_ek.len()
            ))
        })?,
    );
    let (ml_kem_ct, ml_kem_ss) = ek.encapsulate(&mut OsRng).map_err(|e| {
        HybridError::MlKemError(format!("Encapsulation failed: {e:?}"))
    })?;

    // Step 3: Combine shared secrets
    let ml_kem_ss_bytes: &[u8] = ml_kem_ss.as_ref();
    let mut hasher = Sha256::new();
    hasher.update(x25519_ss.as_bytes());
    hasher.update(ml_kem_ss_bytes);
    let combined_ss: [u8; 32] = hasher.finalize().into();

    // Build combined ciphertext: ephemeral X25519 public key || ML-KEM ciphertext
    let ml_kem_ct_bytes: &[u8] = ml_kem_ct.as_ref();
    let mut combined_ct = Vec::with_capacity(X25519_KEY_SIZE + ml_kem_ct_bytes.len());
    combined_ct.extend_from_slice(eph_public.as_bytes());
    combined_ct.extend_from_slice(ml_kem_ct_bytes);

    Ok((combined_ct, combined_ss))
}

/// Perform hybrid decapsulation using the recipient's hybrid secret key
///
/// This performs:
/// 1. X25519 Diffie-Hellman with the ephemeral public key from the ciphertext
/// 2. ML-KEM-768 decapsulation
/// 3. Combines shared secrets: `SHA-256(x25519_ss || ml_kem_ss)`
///
/// # Arguments
/// * `hybrid_sk` - The recipient's hybrid key pair
/// * `combined_ct` - The combined ciphertext from encapsulation
///
/// # Returns
/// The 32-byte combined shared secret
///
/// # Errors
/// Returns an error if decapsulation fails or the ciphertext is invalid.
pub fn hybrid_decapsulate(
    hybrid_sk: &HybridKeyPair,
    combined_ct: &[u8],
) -> HybridResult<[u8; 32]> {
    hybrid_decapsulate_from_keys(
        &hybrid_sk.x25519_secret,
        &hybrid_sk.ml_kem_decapsulation_key,
        combined_ct,
    )
}

/// Perform hybrid decapsulation from raw secret key components
///
/// # Arguments
/// * `x25519_secret_key` - The recipient's X25519 secret key (32 bytes)
/// * `ml_kem_dk` - The recipient's ML-KEM-768 decapsulation key bytes
/// * `combined_ct` - The combined ciphertext
///
/// # Returns
/// The 32-byte combined shared secret
///
/// # Errors
/// Returns an error if decapsulation fails or the ciphertext is invalid.
pub fn hybrid_decapsulate_from_keys(
    x25519_secret_key: &[u8; X25519_KEY_SIZE],
    ml_kem_dk: &[u8],
    combined_ct: &[u8],
) -> HybridResult<[u8; 32]> {
    if combined_ct.len() <= X25519_KEY_SIZE {
        return Err(HybridError::InvalidCiphertext(format!(
            "Combined ciphertext too short: {} bytes",
            combined_ct.len()
        )));
    }

    // Split combined ciphertext
    let eph_public_bytes: [u8; X25519_KEY_SIZE] = combined_ct[..X25519_KEY_SIZE]
        .try_into()
        .expect("validated above");
    let ml_kem_ct_bytes = &combined_ct[X25519_KEY_SIZE..];

    // Step 1: X25519 key agreement
    let my_secret = X25519Secret::from(*x25519_secret_key);
    let eph_public = X25519PublicKey::from(eph_public_bytes);
    let x25519_ss = my_secret.diffie_hellman(&eph_public);

    // Step 2: ML-KEM-768 decapsulation
    let dk = <MlKem768 as KemCore>::DecapsulationKey::from_bytes(
        ml_kem_dk.try_into().map_err(|_| {
            HybridError::InvalidKeyData(format!(
                "Invalid ML-KEM-768 decapsulation key length: {}",
                ml_kem_dk.len()
            ))
        })?,
    );
    let ct_encoded = ml_kem_ct_bytes.try_into().map_err(|_| {
        HybridError::InvalidCiphertext(format!(
            "Invalid ML-KEM-768 ciphertext length: {}",
            ml_kem_ct_bytes.len()
        ))
    })?;
    let ml_kem_ss = dk.decapsulate(ct_encoded).map_err(|e| {
        HybridError::MlKemError(format!("Decapsulation failed: {e:?}"))
    })?;

    // Step 3: Combine shared secrets
    let ml_kem_ss_bytes: &[u8] = ml_kem_ss.as_ref();
    let mut hasher = Sha256::new();
    hasher.update(x25519_ss.as_bytes());
    hasher.update(ml_kem_ss_bytes);
    let combined_ss: [u8; 32] = hasher.finalize().into();

    Ok(combined_ss)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hybrid_keypair_generation() {
        let kp = hybrid_generate_keypair();
        assert_eq!(kp.x25519_public_key().len(), X25519_KEY_SIZE);
        assert!(!kp.ml_kem_encapsulation_key().is_empty());
    }

    #[test]
    fn test_hybrid_combined_public_key() {
        let kp = hybrid_generate_keypair();
        let combined = kp.combined_public_key();
        assert_eq!(
            combined.len(),
            X25519_KEY_SIZE + kp.ml_kem_encapsulation_key().len()
        );
        assert_eq!(&combined[..X25519_KEY_SIZE], kp.x25519_public_key());
        assert_eq!(&combined[X25519_KEY_SIZE..], kp.ml_kem_encapsulation_key());
    }

    #[test]
    fn test_hybrid_encapsulate_decapsulate_roundtrip() {
        let kp = hybrid_generate_keypair();

        let (combined_ct, ss_enc) = hybrid_encapsulate(&kp).expect("encapsulate");
        let ss_dec = hybrid_decapsulate(&kp, &combined_ct).expect("decapsulate");

        assert_eq!(ss_enc, ss_dec);
        assert_eq!(ss_enc.len(), 32);
    }

    #[test]
    fn test_hybrid_different_keypairs_different_secrets() {
        let kp1 = hybrid_generate_keypair();
        let kp2 = hybrid_generate_keypair();

        let (_, ss1) = hybrid_encapsulate(&kp1).expect("encapsulate");
        let (_, ss2) = hybrid_encapsulate(&kp2).expect("encapsulate");

        // Different keys should produce different shared secrets
        assert_ne!(ss1, ss2);
    }

    #[test]
    fn test_hybrid_wrong_key_produces_different_secret() {
        let kp1 = hybrid_generate_keypair();
        let kp2 = hybrid_generate_keypair();

        let (combined_ct, ss_enc) = hybrid_encapsulate(&kp1).expect("encapsulate");

        // Decapsulating with wrong key should produce a different shared secret
        // (ML-KEM has implicit rejection, so it won't error, but produces wrong ss)
        let ss_wrong = hybrid_decapsulate(&kp2, &combined_ct).expect("decapsulate");
        assert_ne!(ss_enc, ss_wrong);
    }

    #[test]
    fn test_hybrid_ciphertext_too_short() {
        let kp = hybrid_generate_keypair();
        let result = hybrid_decapsulate(&kp, &[0u8; 16]);
        assert!(result.is_err());
    }

    #[test]
    fn test_hybrid_encapsulate_from_raw_keys() {
        let kp = hybrid_generate_keypair();

        let (combined_ct, ss_enc) = hybrid_encapsulate_from_keys(
            kp.x25519_public_key(),
            kp.ml_kem_encapsulation_key(),
        )
        .expect("encapsulate");

        let ss_dec = hybrid_decapsulate(&kp, &combined_ct).expect("decapsulate");
        assert_eq!(ss_enc, ss_dec);
    }

    #[test]
    fn test_hybrid_shared_secret_is_32_bytes() {
        let kp = hybrid_generate_keypair();
        let (_, ss) = hybrid_encapsulate(&kp).expect("encapsulate");
        assert_eq!(ss.len(), 32);
    }

    #[test]
    fn test_hybrid_multiple_encapsulations_different_secrets() {
        let kp = hybrid_generate_keypair();

        let (ct1, ss1) = hybrid_encapsulate(&kp).expect("encapsulate 1");
        let (ct2, ss2) = hybrid_encapsulate(&kp).expect("encapsulate 2");

        // Different ephemeral keys should produce different ciphertexts and secrets
        assert_ne!(ct1, ct2);
        assert_ne!(ss1, ss2);

        // But both should decapsulate correctly
        let dec1 = hybrid_decapsulate(&kp, &ct1).expect("decapsulate 1");
        let dec2 = hybrid_decapsulate(&kp, &ct2).expect("decapsulate 2");
        assert_eq!(ss1, dec1);
        assert_eq!(ss2, dec2);
    }
}
