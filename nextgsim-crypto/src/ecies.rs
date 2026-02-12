//! ECIES implementation for SUPI concealment
//!
//! Implements ECIES (Elliptic Curve Integrated Encryption Scheme) Profile A and
//! Profile B as specified in 3GPP TS 33.501 for SUPI concealment in 5G networks.
//!
//! - Profile A: X25519 (Curve25519) for key exchange
//! - Profile B: P-256 (secp256r1) for key exchange
//!
//! Both profiles use:
//! - X9.63 KDF with SHA-256 for key derivation
//! - AES-128-CTR for encryption
//! - HMAC-SHA256 for MAC (truncated to 8 bytes)

use aes::cipher::{KeyIvInit, StreamCipher};
use elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use hmac::{Hmac, Mac};
use p256::elliptic_curve::rand_core::OsRng;
use p256::{EncodedPoint, PublicKey as P256PublicKey, SecretKey as P256SecretKey};
use rand::RngCore;
use sha2::{Digest, Sha256};
use thiserror::Error;
use x25519_dalek::{PublicKey, StaticSecret};

/// X25519 key size in bytes
pub const X25519_KEY_SIZE: usize = 32;
/// X25519 shared secret size in bytes
pub const X25519_SHARED_SIZE: usize = 32;
/// AES-128 key size in bytes
pub const AES_KEY_SIZE: usize = 16;
/// AES-128 IV size in bytes
pub const AES_IV_SIZE: usize = 16;
/// HMAC-SHA256 key size in bytes
pub const HMAC_KEY_SIZE: usize = 32;
/// MAC tag size in bytes (truncated HMAC-SHA256)
pub const MAC_TAG_SIZE: usize = 8;

/// P-256 compressed public key size in bytes (1 byte prefix + 32 bytes X coordinate)
pub const P256_COMPRESSED_POINT_SIZE: usize = 33;
/// P-256 secret key size in bytes
pub const P256_SECRET_KEY_SIZE: usize = 32;

/// ECIES error types
#[derive(Debug, Error)]
pub enum EciesError {
    /// Invalid public key
    #[error("Invalid public key: {0}")]
    InvalidPublicKey(String),
    /// Invalid ciphertext
    #[error("Invalid ciphertext: {0}")]
    InvalidCiphertext(String),
    /// MAC verification failed
    #[error("MAC verification failed")]
    MacVerificationFailed,
    /// Key derivation error
    #[error("Key derivation error: {0}")]
    KeyDerivationError(String),
    /// P-256 ECDH error
    #[error("P-256 ECDH error: {0}")]
    P256Error(String),
}

/// Result type for ECIES operations
pub type EciesResult<T> = Result<T, EciesError>;

/// X9.63 Key Derivation Function using SHA-256
pub fn x963_kdf(shared_secret: &[u8; X25519_SHARED_SIZE], shared_info: &[u8], key_size: usize) -> Vec<u8> {
    const SHA256_DIGEST_SIZE: usize = 32;
    let max_count = key_size.div_ceil(SHA256_DIGEST_SIZE);
    let mut result = Vec::with_capacity(max_count * SHA256_DIGEST_SIZE);
    for count in 1..=max_count {
        let mut hasher = Sha256::new();
        hasher.update(shared_secret);
        hasher.update((count as u32).to_be_bytes());
        hasher.update(shared_info);
        result.extend_from_slice(&hasher.finalize());
    }
    result.truncate(key_size);
    result
}

/// ECIES key pair for encryption/decryption
#[derive(Clone)]
pub struct EciesKeyPair {
    private_key: [u8; X25519_KEY_SIZE],
    public_key: [u8; X25519_KEY_SIZE],
}

impl EciesKeyPair {
    /// Generate a new random key pair
    pub fn generate() -> Self {
        let mut rng = rand::thread_rng();
        let mut seed = [0u8; X25519_KEY_SIZE];
        rng.fill_bytes(&mut seed);
        Self::from_seed(&seed)
    }

    /// Generate a key pair from a seed
    pub fn from_seed(seed: &[u8; X25519_KEY_SIZE]) -> Self {
        let secret = StaticSecret::from(*seed);
        let public = PublicKey::from(&secret);
        Self { private_key: *seed, public_key: *public.as_bytes() }
    }

    /// Get the public key
    pub fn public_key(&self) -> &[u8; X25519_KEY_SIZE] { &self.public_key }

    /// Get the private key
    pub fn private_key(&self) -> &[u8; X25519_KEY_SIZE] { &self.private_key }
}

/// Compute X25519 shared secret
pub fn x25519_shared_secret(
    my_private_key: &[u8; X25519_KEY_SIZE],
    their_public_key: &[u8; X25519_KEY_SIZE],
) -> [u8; X25519_SHARED_SIZE] {
    let secret = StaticSecret::from(*my_private_key);
    let public = PublicKey::from(*their_public_key);
    *secret.diffie_hellman(&public).as_bytes()
}

/// ECIES Profile A encryption for SUPI concealment
pub fn ecies_encrypt(
    plaintext: &[u8],
    home_network_public_key: &[u8; X25519_KEY_SIZE],
) -> EciesResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    ecies_encrypt_with_keypair(plaintext, home_network_public_key, &EciesKeyPair::generate())
}

/// ECIES Profile A encryption with a specific ephemeral key pair (for testing)
pub fn ecies_encrypt_with_keypair(
    plaintext: &[u8],
    home_network_public_key: &[u8; X25519_KEY_SIZE],
    ephemeral_keypair: &EciesKeyPair,
) -> EciesResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let shared_secret = x25519_shared_secret(ephemeral_keypair.private_key(), home_network_public_key);
    let derived_key = x963_kdf(&shared_secret, ephemeral_keypair.public_key(), 64);
    let encryption_key: [u8; AES_KEY_SIZE] = derived_key[0..16].try_into().expect("slice");
    let iv: [u8; AES_IV_SIZE] = derived_key[16..32].try_into().expect("slice");
    let mac_key: [u8; HMAC_KEY_SIZE] = derived_key[32..64].try_into().expect("slice");

    let mut ciphertext = plaintext.to_vec();
    let mut cipher = ctr::Ctr128BE::<aes::Aes128>::new(&encryption_key.into(), &iv.into());
    cipher.apply_keystream(&mut ciphertext);

    let mut mac = Hmac::<Sha256>::new_from_slice(&mac_key).expect("HMAC");
    mac.update(&ciphertext);
    let mac_tag = mac.finalize().into_bytes()[..MAC_TAG_SIZE].to_vec();

    Ok((ephemeral_keypair.public_key().to_vec(), ciphertext, mac_tag))
}

/// ECIES Profile A decryption for SUPI de-concealment
pub fn ecies_decrypt(
    ephemeral_public_key: &[u8],
    ciphertext: &[u8],
    mac_tag: &[u8],
    home_network_private_key: &[u8; X25519_KEY_SIZE],
) -> EciesResult<Vec<u8>> {
    if ephemeral_public_key.len() != X25519_KEY_SIZE {
        return Err(EciesError::InvalidPublicKey(format!(
            "Expected {} bytes, got {}", X25519_KEY_SIZE, ephemeral_public_key.len()
        )));
    }
    if mac_tag.len() != MAC_TAG_SIZE {
        return Err(EciesError::InvalidCiphertext(format!(
            "Invalid MAC tag length: expected {}, got {}", MAC_TAG_SIZE, mac_tag.len()
        )));
    }

    let ephemeral_pk: [u8; X25519_KEY_SIZE] = ephemeral_public_key.try_into().expect("validated");
    let shared_secret = x25519_shared_secret(home_network_private_key, &ephemeral_pk);
    let derived_key = x963_kdf(&shared_secret, &ephemeral_pk, 64);
    let encryption_key: [u8; AES_KEY_SIZE] = derived_key[0..16].try_into().expect("slice");
    let iv: [u8; AES_IV_SIZE] = derived_key[16..32].try_into().expect("slice");
    let mac_key: [u8; HMAC_KEY_SIZE] = derived_key[32..64].try_into().expect("slice");

    let mut mac = Hmac::<Sha256>::new_from_slice(&mac_key).expect("HMAC");
    mac.update(ciphertext);
    let computed_mac = mac.finalize().into_bytes();
    if &computed_mac[..MAC_TAG_SIZE] != mac_tag {
        return Err(EciesError::MacVerificationFailed);
    }

    let mut plaintext = ciphertext.to_vec();
    let mut cipher = ctr::Ctr128BE::<aes::Aes128>::new(&encryption_key.into(), &iv.into());
    cipher.apply_keystream(&mut plaintext);
    Ok(plaintext)
}

/// Generate SUCI scheme output for Profile A
pub fn generate_suci_profile_a(
    msin: &[u8],
    home_network_public_key: &[u8; X25519_KEY_SIZE],
) -> EciesResult<Vec<u8>> {
    let (ephemeral_pk, ciphertext, mac_tag) = ecies_encrypt(msin, home_network_public_key)?;
    let mut scheme_output = Vec::with_capacity(ephemeral_pk.len() + ciphertext.len() + mac_tag.len());
    scheme_output.extend_from_slice(&ephemeral_pk);
    scheme_output.extend_from_slice(&ciphertext);
    scheme_output.extend_from_slice(&mac_tag);
    Ok(scheme_output)
}

/// Decode SUCI scheme output for Profile A
pub fn decode_suci_profile_a(
    scheme_output: &[u8],
    home_network_private_key: &[u8; X25519_KEY_SIZE],
) -> EciesResult<Vec<u8>> {
    if scheme_output.len() < X25519_KEY_SIZE + 1 + MAC_TAG_SIZE {
        return Err(EciesError::InvalidCiphertext(format!(
            "Scheme output too short: {} bytes", scheme_output.len()
        )));
    }
    let ephemeral_pk = &scheme_output[..X25519_KEY_SIZE];
    let ciphertext_end = scheme_output.len() - MAC_TAG_SIZE;
    let ciphertext = &scheme_output[X25519_KEY_SIZE..ciphertext_end];
    let mac_tag = &scheme_output[ciphertext_end..];
    ecies_decrypt(ephemeral_pk, ciphertext, mac_tag, home_network_private_key)
}

/// X9.63 Key Derivation Function using SHA-256 (variable-length shared secret)
///
/// Same as [`x963_kdf`] but accepts a variable-length shared secret slice,
/// used by Profile B where the shared secret length differs from X25519.
fn x963_kdf_bytes(shared_secret: &[u8], shared_info: &[u8], key_size: usize) -> Vec<u8> {
    const SHA256_DIGEST_SIZE: usize = 32;
    let max_count = key_size.div_ceil(SHA256_DIGEST_SIZE);
    let mut result = Vec::with_capacity(max_count * SHA256_DIGEST_SIZE);
    for count in 1..=max_count {
        let mut hasher = Sha256::new();
        hasher.update(shared_secret);
        hasher.update((count as u32).to_be_bytes());
        hasher.update(shared_info);
        result.extend_from_slice(&hasher.finalize());
    }
    result.truncate(key_size);
    result
}

// ============================================================
// ECIES Profile B (P-256 / secp256r1)
// ============================================================

/// ECIES Profile B key pair for encryption/decryption (P-256)
#[derive(Clone)]
pub struct EciesProfileBKeyPair {
    secret_key: P256SecretKey,
    public_key: P256PublicKey,
}

impl EciesProfileBKeyPair {
    /// Generate a new random P-256 key pair
    pub fn generate() -> Self {
        let secret_key = P256SecretKey::random(&mut OsRng);
        let public_key = secret_key.public_key();
        Self { secret_key, public_key }
    }

    /// Create a key pair from raw secret key bytes
    ///
    /// # Errors
    /// Returns an error if the bytes do not represent a valid P-256 scalar.
    pub fn from_secret_bytes(bytes: &[u8; P256_SECRET_KEY_SIZE]) -> EciesResult<Self> {
        let secret_key = P256SecretKey::from_bytes(bytes.into())
            .map_err(|e| EciesError::P256Error(format!("Invalid secret key: {e}")))?;
        let public_key = secret_key.public_key();
        Ok(Self { secret_key, public_key })
    }

    /// Get the compressed public key (33 bytes)
    pub fn public_key_compressed(&self) -> [u8; P256_COMPRESSED_POINT_SIZE] {
        let encoded = self.public_key.to_encoded_point(true);
        let bytes = encoded.as_bytes();
        let mut out = [0u8; P256_COMPRESSED_POINT_SIZE];
        out.copy_from_slice(bytes);
        out
    }

    /// Get the secret key bytes
    pub fn secret_key_bytes(&self) -> [u8; P256_SECRET_KEY_SIZE] {
        let bytes = self.secret_key.to_bytes();
        let mut out = [0u8; P256_SECRET_KEY_SIZE];
        out.copy_from_slice(&bytes);
        out
    }

    /// Get a reference to the P-256 public key
    pub fn public_key(&self) -> &P256PublicKey {
        &self.public_key
    }

    /// Get a reference to the P-256 secret key
    pub fn secret_key(&self) -> &P256SecretKey {
        &self.secret_key
    }
}

/// Perform P-256 ECDH key agreement
fn p256_ecdh(
    my_secret: &P256SecretKey,
    their_public: &P256PublicKey,
) -> EciesResult<Vec<u8>> {
    use p256::ecdh::diffie_hellman;
    let shared_secret = diffie_hellman(
        my_secret.to_nonzero_scalar(),
        their_public.as_affine(),
    );
    Ok(shared_secret.raw_secret_bytes().to_vec())
}

/// ECIES Profile B encryption for SUPI concealment (P-256)
///
/// Uses P-256 ECDH for key agreement, X9.63 KDF with SHA-256,
/// AES-128-CTR for encryption, and HMAC-SHA256 (truncated to 8 bytes) for MAC.
///
/// # Returns
/// Tuple of (`ephemeral_public_key_compressed`, ciphertext, `mac_tag`)
///
/// # Errors
/// Returns an error if the public key is invalid or ECDH fails.
pub fn ecies_profile_b_encrypt(
    plaintext: &[u8],
    home_network_public_key: &P256PublicKey,
) -> EciesResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let ephemeral = EciesProfileBKeyPair::generate();
    ecies_profile_b_encrypt_with_keypair(plaintext, home_network_public_key, &ephemeral)
}

/// ECIES Profile B encryption with a specific ephemeral key pair (for testing)
///
/// # Errors
/// Returns an error if ECDH fails.
pub fn ecies_profile_b_encrypt_with_keypair(
    plaintext: &[u8],
    home_network_public_key: &P256PublicKey,
    ephemeral_keypair: &EciesProfileBKeyPair,
) -> EciesResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let shared_secret = p256_ecdh(ephemeral_keypair.secret_key(), home_network_public_key)?;
    let ephemeral_pk_compressed = ephemeral_keypair.public_key_compressed();
    let derived_key = x963_kdf_bytes(&shared_secret, &ephemeral_pk_compressed, 64);

    let encryption_key: [u8; AES_KEY_SIZE] = derived_key[0..16].try_into().expect("slice");
    let iv: [u8; AES_IV_SIZE] = derived_key[16..32].try_into().expect("slice");
    let mac_key: [u8; HMAC_KEY_SIZE] = derived_key[32..64].try_into().expect("slice");

    let mut ciphertext = plaintext.to_vec();
    let mut cipher = ctr::Ctr128BE::<aes::Aes128>::new(&encryption_key.into(), &iv.into());
    cipher.apply_keystream(&mut ciphertext);

    let mut mac = Hmac::<Sha256>::new_from_slice(&mac_key).expect("HMAC");
    mac.update(&ciphertext);
    let mac_tag = mac.finalize().into_bytes()[..MAC_TAG_SIZE].to_vec();

    Ok((ephemeral_pk_compressed.to_vec(), ciphertext, mac_tag))
}

/// ECIES Profile B decryption for SUPI de-concealment (P-256)
///
/// # Errors
/// Returns an error if the ephemeral public key is invalid, ECDH fails,
/// or MAC verification fails.
pub fn ecies_profile_b_decrypt(
    ephemeral_public_key: &[u8],
    ciphertext: &[u8],
    mac_tag: &[u8],
    home_network_secret_key: &P256SecretKey,
) -> EciesResult<Vec<u8>> {
    // Parse compressed ephemeral public key
    let encoded_point = EncodedPoint::from_bytes(ephemeral_public_key)
        .map_err(|e| EciesError::InvalidPublicKey(format!("Invalid P-256 point encoding: {e}")))?;
    let ephemeral_pk = P256PublicKey::from_encoded_point(&encoded_point);
    let ephemeral_pk = Option::from(ephemeral_pk)
        .ok_or_else(|| EciesError::InvalidPublicKey("Point not on P-256 curve".into()))?;

    if mac_tag.len() != MAC_TAG_SIZE {
        return Err(EciesError::InvalidCiphertext(format!(
            "Invalid MAC tag length: expected {}, got {}", MAC_TAG_SIZE, mac_tag.len()
        )));
    }

    let shared_secret = p256_ecdh(home_network_secret_key, &ephemeral_pk)?;
    let derived_key = x963_kdf_bytes(&shared_secret, ephemeral_public_key, 64);

    let encryption_key: [u8; AES_KEY_SIZE] = derived_key[0..16].try_into().expect("slice");
    let iv: [u8; AES_IV_SIZE] = derived_key[16..32].try_into().expect("slice");
    let mac_key: [u8; HMAC_KEY_SIZE] = derived_key[32..64].try_into().expect("slice");

    // Verify MAC
    let mut mac = Hmac::<Sha256>::new_from_slice(&mac_key).expect("HMAC");
    mac.update(ciphertext);
    let computed_mac = mac.finalize().into_bytes();
    if &computed_mac[..MAC_TAG_SIZE] != mac_tag {
        return Err(EciesError::MacVerificationFailed);
    }

    // Decrypt
    let mut plaintext = ciphertext.to_vec();
    let mut cipher = ctr::Ctr128BE::<aes::Aes128>::new(&encryption_key.into(), &iv.into());
    cipher.apply_keystream(&mut plaintext);
    Ok(plaintext)
}

/// Generate SUCI scheme output for Profile B (P-256)
///
/// The scheme output format is:
/// `ephemeral_public_key_compressed (33 bytes) || ciphertext || mac_tag (8 bytes)`
///
/// # Errors
/// Returns an error if encryption fails.
pub fn generate_suci_profile_b(
    msin: &[u8],
    home_network_public_key: &P256PublicKey,
) -> EciesResult<Vec<u8>> {
    let (ephemeral_pk, ciphertext, mac_tag) =
        ecies_profile_b_encrypt(msin, home_network_public_key)?;
    let mut scheme_output =
        Vec::with_capacity(ephemeral_pk.len() + ciphertext.len() + mac_tag.len());
    scheme_output.extend_from_slice(&ephemeral_pk);
    scheme_output.extend_from_slice(&ciphertext);
    scheme_output.extend_from_slice(&mac_tag);
    Ok(scheme_output)
}

/// Decode SUCI scheme output for Profile B (P-256)
///
/// Expects the format: `ephemeral_public_key_compressed (33 bytes) || ciphertext || mac_tag (8 bytes)`
///
/// # Errors
/// Returns an error if the scheme output is too short, the public key is invalid,
/// or MAC verification fails.
pub fn decode_suci_profile_b(
    scheme_output: &[u8],
    home_network_secret_key: &P256SecretKey,
) -> EciesResult<Vec<u8>> {
    if scheme_output.len() < P256_COMPRESSED_POINT_SIZE + 1 + MAC_TAG_SIZE {
        return Err(EciesError::InvalidCiphertext(format!(
            "Scheme output too short: {} bytes", scheme_output.len()
        )));
    }
    let ephemeral_pk = &scheme_output[..P256_COMPRESSED_POINT_SIZE];
    let ciphertext_end = scheme_output.len() - MAC_TAG_SIZE;
    let ciphertext = &scheme_output[P256_COMPRESSED_POINT_SIZE..ciphertext_end];
    let mac_tag = &scheme_output[ciphertext_end..];
    ecies_profile_b_decrypt(ephemeral_pk, ciphertext, mac_tag, home_network_secret_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x963_kdf_basic() {
        let shared_secret = [0x01u8; 32];
        let shared_info = [0x02u8; 32];
        let key = x963_kdf(&shared_secret, &shared_info, 64);
        assert_eq!(key.len(), 64);
        let key2 = x963_kdf(&shared_secret, &shared_info, 64);
        assert_eq!(key, key2);
    }

    #[test]
    fn test_x963_kdf_different_sizes() {
        let shared_secret = [0x03u8; 32];
        let shared_info = [0x04u8; 32];
        let key_32 = x963_kdf(&shared_secret, &shared_info, 32);
        let key_48 = x963_kdf(&shared_secret, &shared_info, 48);
        let key_64 = x963_kdf(&shared_secret, &shared_info, 64);
        assert_eq!(key_32.len(), 32);
        assert_eq!(key_48.len(), 48);
        assert_eq!(key_64.len(), 64);
        assert_eq!(&key_32[..], &key_48[..32]);
        assert_eq!(&key_32[..], &key_64[..32]);
    }

    #[test]
    fn test_ecies_keypair_generation() {
        let keypair1 = EciesKeyPair::generate();
        let keypair2 = EciesKeyPair::generate();
        assert_ne!(keypair1.public_key(), keypair2.public_key());
        assert_ne!(keypair1.private_key(), keypair2.private_key());
    }

    #[test]
    fn test_ecies_keypair_from_seed() {
        let seed = [0x42u8; 32];
        let keypair1 = EciesKeyPair::from_seed(&seed);
        let keypair2 = EciesKeyPair::from_seed(&seed);
        assert_eq!(keypair1.public_key(), keypair2.public_key());
        assert_eq!(keypair1.private_key(), keypair2.private_key());
    }

    #[test]
    fn test_x25519_shared_secret() {
        let alice = EciesKeyPair::from_seed(&[0x01u8; 32]);
        let bob = EciesKeyPair::from_seed(&[0x02u8; 32]);
        let alice_shared = x25519_shared_secret(alice.private_key(), bob.public_key());
        let bob_shared = x25519_shared_secret(bob.private_key(), alice.public_key());
        assert_eq!(alice_shared, bob_shared);
    }

    #[test]
    fn test_ecies_encrypt_decrypt_roundtrip() {
        let plaintext = b"Hello, ECIES!";
        let hn_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");
        assert_eq!(ciphertext.len(), plaintext.len());
        assert_ne!(&ciphertext[..], &plaintext[..]);
        let decrypted =
            ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key())
                .expect("decrypt");
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn test_ecies_encrypt_decrypt_with_known_keys() {
        let plaintext = [0x12, 0x34, 0x56, 0x78, 0x9A];
        let hn_keypair = EciesKeyPair::from_seed(&[0xAAu8; 32]);
        let ephemeral_keypair = EciesKeyPair::from_seed(&[0xBBu8; 32]);
        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_encrypt_with_keypair(&plaintext, hn_keypair.public_key(), &ephemeral_keypair)
                .expect("encrypt");
        let decrypted =
            ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key())
                .expect("decrypt");
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn test_ecies_mac_verification_failure() {
        let plaintext = b"Test data";
        let hn_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, ciphertext, mut mac_tag) =
            ecies_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");
        mac_tag[0] ^= 0xFF;
        let result = ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key());
        assert!(matches!(result, Err(EciesError::MacVerificationFailed)));
    }

    #[test]
    fn test_ecies_ciphertext_tampering() {
        let plaintext = b"Test data";
        let hn_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, mut ciphertext, mac_tag) =
            ecies_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");
        ciphertext[0] ^= 0xFF;
        let result = ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key());
        assert!(matches!(result, Err(EciesError::MacVerificationFailed)));
    }

    #[test]
    fn test_ecies_wrong_private_key() {
        let plaintext = b"Secret message";
        let hn_keypair = EciesKeyPair::generate();
        let wrong_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");
        let result =
            ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, wrong_keypair.private_key());
        assert!(matches!(result, Err(EciesError::MacVerificationFailed)));
    }

    #[test]
    fn test_suci_profile_a_roundtrip() {
        let msin = [0x00, 0x00, 0x00, 0x00, 0x01];
        let hn_keypair = EciesKeyPair::generate();
        let scheme_output = generate_suci_profile_a(&msin, hn_keypair.public_key()).expect("gen");
        assert_eq!(scheme_output.len(), 32 + msin.len() + 8);
        let decoded_msin = decode_suci_profile_a(&scheme_output, hn_keypair.private_key())
            .expect("decode");
        assert_eq!(&decoded_msin[..], &msin[..]);
    }

    #[test]
    fn test_suci_profile_a_with_real_msin() {
        let msin_bcd = [0x10, 0x32, 0x54, 0x76, 0x98];
        let hn_keypair = EciesKeyPair::generate();
        let scheme_output = generate_suci_profile_a(&msin_bcd, hn_keypair.public_key()).expect("gen");
        let decoded = decode_suci_profile_a(&scheme_output, hn_keypair.private_key()).expect("dec");
        assert_eq!(&decoded[..], &msin_bcd[..]);
    }

    #[test]
    fn test_decode_suci_invalid_length() {
        let hn_keypair = EciesKeyPair::generate();
        let short_output = [0u8; 30];
        let result = decode_suci_profile_a(&short_output, hn_keypair.private_key());
        assert!(matches!(result, Err(EciesError::InvalidCiphertext(_))));
    }

    #[test]
    fn test_ecies_empty_plaintext() {
        let plaintext: [u8; 0] = [];
        let hn_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_encrypt(&plaintext, hn_keypair.public_key()).expect("encrypt");
        assert_eq!(ciphertext.len(), 0);
        let decrypted =
            ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key())
                .expect("decrypt");
        assert_eq!(decrypted.len(), 0);
    }

    #[test]
    fn test_ecies_large_plaintext() {
        let plaintext = vec![0xABu8; 256];
        let hn_keypair = EciesKeyPair::generate();
        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_encrypt(&plaintext, hn_keypair.public_key()).expect("encrypt");
        assert_eq!(ciphertext.len(), plaintext.len());
        let decrypted =
            ecies_decrypt(&ephemeral_pk, &ciphertext, &mac_tag, hn_keypair.private_key())
                .expect("decrypt");
        assert_eq!(decrypted, plaintext);
    }

    // ========== Profile B (P-256) Tests ==========

    #[test]
    fn test_profile_b_keypair_generation() {
        let kp1 = EciesProfileBKeyPair::generate();
        let kp2 = EciesProfileBKeyPair::generate();
        assert_ne!(kp1.public_key_compressed(), kp2.public_key_compressed());
    }

    #[test]
    fn test_profile_b_compressed_key_size() {
        let kp = EciesProfileBKeyPair::generate();
        let pk = kp.public_key_compressed();
        assert_eq!(pk.len(), P256_COMPRESSED_POINT_SIZE);
        // Compressed point should start with 0x02 or 0x03
        assert!(pk[0] == 0x02 || pk[0] == 0x03);
    }

    #[test]
    fn test_profile_b_encrypt_decrypt_roundtrip() {
        let plaintext = b"Hello, ECIES Profile B!";
        let hn_keypair = EciesProfileBKeyPair::generate();

        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_profile_b_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");

        assert_eq!(ephemeral_pk.len(), P256_COMPRESSED_POINT_SIZE);
        assert_eq!(ciphertext.len(), plaintext.len());
        assert_ne!(&ciphertext[..], &plaintext[..]);

        let decrypted = ecies_profile_b_decrypt(
            &ephemeral_pk,
            &ciphertext,
            &mac_tag,
            hn_keypair.secret_key(),
        )
        .expect("decrypt");
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn test_profile_b_mac_verification_failure() {
        let plaintext = b"Test data Profile B";
        let hn_keypair = EciesProfileBKeyPair::generate();

        let (ephemeral_pk, ciphertext, mut mac_tag) =
            ecies_profile_b_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");
        mac_tag[0] ^= 0xFF;

        let result = ecies_profile_b_decrypt(
            &ephemeral_pk,
            &ciphertext,
            &mac_tag,
            hn_keypair.secret_key(),
        );
        assert!(matches!(result, Err(EciesError::MacVerificationFailed)));
    }

    #[test]
    fn test_profile_b_wrong_secret_key() {
        let plaintext = b"Secret Profile B message";
        let hn_keypair = EciesProfileBKeyPair::generate();
        let wrong_keypair = EciesProfileBKeyPair::generate();

        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_profile_b_encrypt(plaintext, hn_keypair.public_key()).expect("encrypt");

        let result = ecies_profile_b_decrypt(
            &ephemeral_pk,
            &ciphertext,
            &mac_tag,
            wrong_keypair.secret_key(),
        );
        assert!(matches!(result, Err(EciesError::MacVerificationFailed)));
    }

    #[test]
    fn test_suci_profile_b_roundtrip() {
        let msin = [0x00, 0x00, 0x00, 0x00, 0x01];
        let hn_keypair = EciesProfileBKeyPair::generate();

        let scheme_output =
            generate_suci_profile_b(&msin, hn_keypair.public_key()).expect("gen");
        // 33 (compressed P-256) + 5 (msin) + 8 (mac) = 46
        assert_eq!(scheme_output.len(), P256_COMPRESSED_POINT_SIZE + msin.len() + MAC_TAG_SIZE);

        let decoded_msin =
            decode_suci_profile_b(&scheme_output, hn_keypair.secret_key()).expect("decode");
        assert_eq!(&decoded_msin[..], &msin[..]);
    }

    #[test]
    fn test_suci_profile_b_with_real_msin() {
        let msin_bcd = [0x10, 0x32, 0x54, 0x76, 0x98];
        let hn_keypair = EciesProfileBKeyPair::generate();

        let scheme_output =
            generate_suci_profile_b(&msin_bcd, hn_keypair.public_key()).expect("gen");
        let decoded =
            decode_suci_profile_b(&scheme_output, hn_keypair.secret_key()).expect("dec");
        assert_eq!(&decoded[..], &msin_bcd[..]);
    }

    #[test]
    fn test_decode_suci_profile_b_invalid_length() {
        let hn_keypair = EciesProfileBKeyPair::generate();
        let short_output = [0u8; 30];
        let result = decode_suci_profile_b(&short_output, hn_keypair.secret_key());
        assert!(matches!(result, Err(EciesError::InvalidCiphertext(_))));
    }

    #[test]
    fn test_profile_b_empty_plaintext() {
        let plaintext: [u8; 0] = [];
        let hn_keypair = EciesProfileBKeyPair::generate();

        let (ephemeral_pk, ciphertext, mac_tag) =
            ecies_profile_b_encrypt(&plaintext, hn_keypair.public_key()).expect("encrypt");
        assert_eq!(ciphertext.len(), 0);

        let decrypted = ecies_profile_b_decrypt(
            &ephemeral_pk,
            &ciphertext,
            &mac_tag,
            hn_keypair.secret_key(),
        )
        .expect("decrypt");
        assert_eq!(decrypted.len(), 0);
    }

    #[test]
    fn test_profile_b_from_secret_bytes() {
        let kp = EciesProfileBKeyPair::generate();
        let secret_bytes = kp.secret_key_bytes();
        let kp2 = EciesProfileBKeyPair::from_secret_bytes(&secret_bytes).expect("from_bytes");
        assert_eq!(kp.public_key_compressed(), kp2.public_key_compressed());
    }
}
