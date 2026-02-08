//! SNOW5G next-generation stream cipher (placeholder)
//!
//! SNOW5G is a next-generation stream cipher being developed for 6G security
//! algorithms. It is the successor to SNOW3G (used in 4G/5G) with enhanced
//! security properties including:
//! - 256-bit key size
//! - 128-bit IV
//! - Higher throughput for future network speeds
//!
//! **Note:** SNOW5G is still being standardized. This module provides a
//! placeholder structure with a simple XOR-based keystream generator.
//! The actual SNOW5G algorithm will be implemented once the final
//! specification is published.
//!
//! Reference: 3GPP future specifications (pending)

use sha2::{Digest, Sha256};
use thiserror::Error;

/// SNOW5G key size in bytes (256 bits)
pub const KEY_SIZE: usize = 32;

/// SNOW5G IV size in bytes (128 bits)
pub const IV_SIZE: usize = 16;

/// SNOW5G error types
#[derive(Debug, Error)]
pub enum Snow5gError {
    /// Invalid key length
    #[error("Invalid key length: expected {}, got {0}", KEY_SIZE)]
    InvalidKeyLength(usize),
    /// Invalid IV length
    #[error("Invalid IV length: expected {}, got {0}", IV_SIZE)]
    InvalidIvLength(usize),
}

/// Result type for SNOW5G operations
pub type Snow5gResult<T> = Result<T, Snow5gError>;

/// SNOW5G cipher state
///
/// **TODO:** Replace this placeholder with the actual SNOW5G state machine
/// once the final specification is published. Currently uses a simplified
/// SHA-256-based keystream generator for structural testing.
pub struct Snow5g {
    /// Internal state derived from key and IV
    state: [u8; 64],
    /// Counter for keystream generation
    counter: u64,
}

impl Snow5g {
    /// Create a new SNOW5G instance initialized with key and IV
    ///
    /// # Arguments
    /// * `key` - 256-bit key (32 bytes)
    /// * `iv` - 128-bit initialization vector (16 bytes)
    pub fn new(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE]) -> Self {
        // TODO: Replace with actual SNOW5G initialization when spec is final
        // Current placeholder: derive initial state from key and IV using SHA-256
        let mut hasher = Sha256::new();
        hasher.update(key);
        hasher.update(iv);
        let hash1 = hasher.finalize();

        let mut hasher2 = Sha256::new();
        hasher2.update(hash1);
        hasher2.update(key);
        let hash2 = hasher2.finalize();

        let mut state = [0u8; 64];
        state[..32].copy_from_slice(&hash1);
        state[32..].copy_from_slice(&hash2);

        Snow5g { state, counter: 0 }
    }

    /// Generate the next block of keystream (32 bytes)
    ///
    /// **TODO:** Replace with actual SNOW5G keystream generation
    fn generate_keystream_block(&mut self) -> [u8; 32] {
        // Placeholder: SHA-256(state || counter)
        let mut hasher = Sha256::new();
        hasher.update(self.state);
        hasher.update(self.counter.to_le_bytes());
        self.counter += 1;

        let output = hasher.finalize();

        // Update state for next iteration
        let mut hasher2 = Sha256::new();
        hasher2.update(self.state);
        hasher2.update(output);
        let new_state_part = hasher2.finalize();
        self.state[..32].copy_from_slice(&new_state_part);

        let mut block = [0u8; 32];
        block.copy_from_slice(&output);
        block
    }

    /// Apply keystream to data (XOR)
    fn apply_keystream(&mut self, data: &mut [u8]) {
        let mut offset = 0;
        while offset < data.len() {
            let block = self.generate_keystream_block();
            let remaining = data.len() - offset;
            let to_process = remaining.min(32);
            for i in 0..to_process {
                data[offset + i] ^= block[i];
            }
            offset += 32;
        }
    }
}

/// SNOW5G encryption
///
/// Encrypts data using the SNOW5G stream cipher with a 256-bit key
/// and 128-bit IV.
///
/// **TODO:** This is a placeholder implementation. The actual SNOW5G
/// algorithm will be substituted when the final specification is available.
///
/// # Arguments
/// * `key` - 256-bit encryption key (32 bytes)
/// * `iv` - 128-bit initialization vector (16 bytes)
/// * `data` - Data to encrypt
///
/// # Returns
/// Encrypted data
pub fn snow5g_encrypt(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], data: &[u8]) -> Vec<u8> {
    let mut snow = Snow5g::new(key, iv);
    let mut output = data.to_vec();
    snow.apply_keystream(&mut output);
    output
}

/// SNOW5G decryption
///
/// Decrypts data using the SNOW5G stream cipher. Since SNOW5G is a
/// stream cipher (XOR-based), decryption is the same operation as encryption.
///
/// **TODO:** This is a placeholder implementation.
///
/// # Arguments
/// * `key` - 256-bit encryption key (32 bytes)
/// * `iv` - 128-bit initialization vector (16 bytes)
/// * `data` - Data to decrypt
///
/// # Returns
/// Decrypted data
pub fn snow5g_decrypt(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], data: &[u8]) -> Vec<u8> {
    // Stream cipher: encryption and decryption are the same operation
    snow5g_encrypt(key, iv, data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snow5g_encrypt_decrypt_roundtrip() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let plaintext = b"Hello, SNOW5G! This is a test message for the next-gen cipher.";

        let ciphertext = snow5g_encrypt(&key, &iv, plaintext);
        assert_ne!(&ciphertext[..], &plaintext[..]);
        assert_eq!(ciphertext.len(), plaintext.len());

        let decrypted = snow5g_decrypt(&key, &iv, &ciphertext);
        assert_eq!(&decrypted[..], &plaintext[..]);
    }

    #[test]
    fn test_snow5g_empty_data() {
        let key = [0u8; KEY_SIZE];
        let iv = [0u8; IV_SIZE];

        let ciphertext = snow5g_encrypt(&key, &iv, &[]);
        assert!(ciphertext.is_empty());
    }

    #[test]
    fn test_snow5g_deterministic() {
        let key = [0xABu8; KEY_SIZE];
        let iv = [0xCDu8; IV_SIZE];
        let plaintext = b"Determinism test";

        let ct1 = snow5g_encrypt(&key, &iv, plaintext);
        let ct2 = snow5g_encrypt(&key, &iv, plaintext);

        assert_eq!(ct1, ct2);
    }

    #[test]
    fn test_snow5g_different_keys_different_output() {
        let key1 = [0x01u8; KEY_SIZE];
        let key2 = [0x02u8; KEY_SIZE];
        let iv = [0x00u8; IV_SIZE];
        let plaintext = b"Test data";

        let ct1 = snow5g_encrypt(&key1, &iv, plaintext);
        let ct2 = snow5g_encrypt(&key2, &iv, plaintext);

        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_snow5g_different_ivs_different_output() {
        let key = [0x01u8; KEY_SIZE];
        let iv1 = [0x00u8; IV_SIZE];
        let iv2 = [0x01u8; IV_SIZE];
        let plaintext = b"Test data";

        let ct1 = snow5g_encrypt(&key, &iv1, plaintext);
        let ct2 = snow5g_encrypt(&key, &iv2, plaintext);

        assert_ne!(ct1, ct2);
    }

    #[test]
    fn test_snow5g_large_data() {
        let key = [0xABu8; KEY_SIZE];
        let iv = [0xCDu8; IV_SIZE];

        let original: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();

        let ciphertext = snow5g_encrypt(&key, &iv, &original);
        assert_ne!(ciphertext, original);
        assert_eq!(ciphertext.len(), original.len());

        let decrypted = snow5g_decrypt(&key, &iv, &ciphertext);
        assert_eq!(decrypted, original);
    }

    #[test]
    fn test_snow5g_single_byte() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let plaintext = [0xFFu8];

        let ciphertext = snow5g_encrypt(&key, &iv, &plaintext);
        assert_ne!(ciphertext[0], plaintext[0]);

        let decrypted = snow5g_decrypt(&key, &iv, &ciphertext);
        assert_eq!(decrypted[0], plaintext[0]);
    }

    #[test]
    fn test_snow5g_struct_state_independence() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];

        // Two independent instances should produce the same keystream
        let mut snow1 = Snow5g::new(&key, &iv);
        let mut snow2 = Snow5g::new(&key, &iv);

        let block1 = snow1.generate_keystream_block();
        let block2 = snow2.generate_keystream_block();
        assert_eq!(block1, block2);
    }
}
