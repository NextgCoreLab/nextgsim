//! NEA (5G encryption algorithms) implementations
//!
//! This module implements the 5G NAS encryption algorithms:
//! - NEA1: SNOW3G-based ciphering (128-EEA1)
//! - NEA2: AES-CTR based ciphering (128-EEA2)
//!
//! Reference: 3GPP TS 33.501 and TS 33.401

use aes::Aes128;
use ctr::cipher::{KeyIvInit, StreamCipher};

use crate::snow3g::uea2_f8;

/// AES-128 key size in bytes
pub const KEY_SIZE: usize = 16;

/// IV/Nonce size for AES-CTR
pub const IV_SIZE: usize = 16;

/// Type alias for AES-128 CTR mode
type Aes128Ctr = ctr::Ctr128BE<Aes128>;

/// NEA1 (128-EEA1) - SNOW3G-based ciphering algorithm
///
/// Implements the 5G encryption algorithm NEA1 as specified in 3GPP TS 33.501.
/// NEA1 is equivalent to UEA2 (F8) from the UMTS security algorithms,
/// reused in the 5G NR context.
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit encryption key (KNASenc or KUPenc)
/// - `data`: Data to encrypt/decrypt (modified in place)
///
/// # Note
/// Encryption and decryption are the same operation (XOR with keystream).
pub fn nea1_encrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    let length_bits = (data.len() * 8) as u32;
    uea2_f8(key, count, bearer as u32 & 0x1F, direction as u32 & 0x01, data, length_bits);
}

/// NEA1 decryption (same as encryption - XOR with keystream)
///
/// See [`nea1_encrypt`] for details.
#[inline]
pub fn nea1_decrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    nea1_encrypt(count, bearer, direction, key, data);
}

/// NEA2 (128-EEA2) - AES-CTR based ciphering algorithm
///
/// Implements the 5G encryption algorithm NEA2 as specified in 3GPP TS 33.501.
/// NEA2 uses AES-128 in CTR mode with a specific IV construction.
///
/// # IV Construction (128 bits)
/// ```text
/// | COUNT (32 bits) | BEARER (5 bits) | DIRECTION (1 bit) | 0...0 (90 bits) |
/// ```
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit encryption key (KNASenc or KUPenc)
/// - `data`: Data to encrypt/decrypt (modified in place)
///
/// # Note
/// Encryption and decryption are the same operation in CTR mode (XOR with keystream).
pub fn nea2_encrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    let iv = build_nea2_iv(count, bearer, direction);
    let mut cipher = Aes128Ctr::new(key.into(), &iv.into());
    cipher.apply_keystream(data);
}

/// NEA2 decryption (same as encryption in CTR mode)
///
/// See [`nea2_encrypt`] for details.
#[inline]
pub fn nea2_decrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    nea2_encrypt(count, bearer, direction, key, data);
}

/// Build the IV for NEA2 from COUNT, BEARER, and DIRECTION
///
/// IV format (128 bits):
/// ```text
/// | COUNT (32 bits) | BEARER (5 bits) | DIRECTION (1 bit) | 0...0 (90 bits) |
/// ```
fn build_nea2_iv(count: u32, bearer: u8, direction: u8) -> [u8; IV_SIZE] {
    let mut iv = [0u8; IV_SIZE];

    // COUNT: bits 0-31 (bytes 0-3)
    iv[0] = (count >> 24) as u8;
    iv[1] = (count >> 16) as u8;
    iv[2] = (count >> 8) as u8;
    iv[3] = count as u8;

    // BEARER (5 bits) + DIRECTION (1 bit) in byte 4
    // BEARER occupies bits 32-36, DIRECTION occupies bit 37
    // In byte 4: BEARER is in bits 7-3, DIRECTION is in bit 2
    iv[4] = ((bearer & 0x1F) << 3) | ((direction & 0x01) << 2);

    // Remaining bytes (5-15) are already zero

    iv
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nea1_encrypt_decrypt_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0xd6, 0x45, 0x9f, 0x82, 0xc5, 0xb3, 0x00,
            0x95, 0x2c, 0x49, 0x10, 0x48, 0x81, 0xff, 0x48,
        ];
        let count = 0x72A4F20F;
        let bearer = 0x0C;
        let direction = 1;

        let original = b"Hello, NEA1 SNOW3G! Test message.";
        let mut data = original.to_vec();

        // Encrypt
        nea1_encrypt(count, bearer, direction, &key, &mut data);
        assert_ne!(&data[..], &original[..]);

        // Decrypt
        nea1_decrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &original[..]);
    }

    #[test]
    fn test_nea1_empty_data() {
        let key: [u8; 16] = [0u8; 16];
        let mut data: Vec<u8> = vec![];

        // Should not panic on empty data
        nea1_encrypt(0, 0, 0, &key, &mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_nea1_deterministic() {
        let key: [u8; 16] = [
            0x2b, 0xd6, 0x45, 0x9f, 0x82, 0xc5, 0xb3, 0x00,
            0x95, 0x2c, 0x49, 0x10, 0x48, 0x81, 0xff, 0x48,
        ];
        let plaintext = b"Determinism test for NEA1";

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea1_encrypt(0x12345678, 5, 1, &key, &mut data1);
        nea1_encrypt(0x12345678, 5, 1, &key, &mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_nea1_different_counts_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea1_encrypt(0, 0, 0, &key, &mut data1);
        nea1_encrypt(1, 0, 0, &key, &mut data2);

        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea1_different_directions_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea1_encrypt(0, 0, 0, &key, &mut data1);
        nea1_encrypt(0, 0, 1, &key, &mut data2);

        assert_ne!(data1, data2);
    }

    /// Test NEA1 with 3GPP UEA2 test vector (NEA1 == UEA2)
    #[test]
    fn test_nea1_3gpp_test_vector_set1() {
        // Using 3GPP TS 35.222 UEA2 Test Set 1 (NEA1 is UEA2)
        let key: [u8; 16] = [
            0x2B, 0xD6, 0x45, 0x9F, 0x82, 0xC5, 0xB3, 0x00,
            0x95, 0x2C, 0x49, 0x10, 0x48, 0x81, 0xFF, 0x48,
        ];
        let count: u32 = 0x72A4F20F;
        let bearer: u8 = 0x0C;
        let direction: u8 = 1;

        let plaintext: [u8; 100] = [
            0x7E, 0xC6, 0x12, 0x72, 0x74, 0x3B, 0xF1, 0x61,
            0x47, 0x26, 0x44, 0x6A, 0x6C, 0x38, 0xCE, 0xD1,
            0x66, 0xF6, 0xCA, 0x76, 0xEB, 0x54, 0x30, 0x04,
            0x42, 0x86, 0x34, 0x6C, 0xEF, 0x13, 0x0F, 0x92,
            0x92, 0x2B, 0x03, 0x45, 0x0D, 0x3A, 0x99, 0x75,
            0xE5, 0xBD, 0x2E, 0xA0, 0xEB, 0x55, 0xAD, 0x8E,
            0x1B, 0x19, 0x9E, 0x3E, 0xC4, 0x31, 0x60, 0x20,
            0xE9, 0xA1, 0xB2, 0x85, 0xE7, 0x62, 0x79, 0x53,
            0x59, 0xB7, 0xBD, 0xFD, 0x39, 0xBE, 0xF4, 0xB2,
            0x48, 0x45, 0x83, 0xD5, 0xAF, 0xE0, 0x82, 0xAE,
            0xE6, 0x38, 0xBF, 0x5F, 0xD5, 0xA6, 0x06, 0x19,
            0x39, 0x01, 0xA0, 0x8F, 0x4A, 0xB4, 0x1A, 0xAB,
            0x9B, 0x13, 0x48, 0x80,
        ];

        let expected_ciphertext: [u8; 100] = [
            0x8C, 0xEB, 0xA6, 0x29, 0x43, 0xDC, 0xED, 0x3A,
            0x09, 0x90, 0xB0, 0x6E, 0xA1, 0xB0, 0xA2, 0xC4,
            0xFB, 0x3C, 0xED, 0xC7, 0x1B, 0x36, 0x9F, 0x42,
            0xBA, 0x64, 0xC1, 0xEB, 0x66, 0x65, 0xE7, 0x2A,
            0xA1, 0xC9, 0xBB, 0x0D, 0xEA, 0xA2, 0x0F, 0xE8,
            0x60, 0x58, 0xB8, 0xBA, 0xEE, 0x2C, 0x2E, 0x7F,
            0x0B, 0xEC, 0xCE, 0x48, 0xB5, 0x29, 0x32, 0xA5,
            0x3C, 0x9D, 0x5F, 0x93, 0x1A, 0x3A, 0x7C, 0x53,
            0x22, 0x59, 0xAF, 0x43, 0x25, 0xE2, 0xA6, 0x5E,
            0x30, 0x84, 0xAD, 0x5F, 0x6A, 0x51, 0x3B, 0x7B,
            0xDD, 0xC1, 0xB6, 0x5F, 0x0A, 0xA0, 0xD9, 0x7A,
            0x05, 0x3D, 0xB5, 0x5A, 0x88, 0xC4, 0xC4, 0xF9,
            0x60, 0x5E, 0x41, 0x43,
        ];

        let mut data = plaintext;
        nea1_encrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &expected_ciphertext[..]);

        // Verify decryption
        nea1_decrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &plaintext[..]);
    }

    #[test]
    fn test_build_nea2_iv() {
        // Test IV construction
        let iv = build_nea2_iv(0x12345678, 0x0A, 1);

        // COUNT should be in bytes 0-3
        assert_eq!(iv[0], 0x12);
        assert_eq!(iv[1], 0x34);
        assert_eq!(iv[2], 0x56);
        assert_eq!(iv[3], 0x78);

        // BEARER (0x0A = 01010) in bits 7-3, DIRECTION (1) in bit 2
        // 01010 << 3 = 01010000 = 0x50
        // 1 << 2 = 00000100 = 0x04
        // Combined: 0x54
        assert_eq!(iv[4], 0x54);

        // Rest should be zero
        for i in 5..16 {
            assert_eq!(iv[i], 0);
        }
    }

    #[test]
    fn test_build_nea2_iv_zero_values() {
        let iv = build_nea2_iv(0, 0, 0);
        assert_eq!(iv, [0u8; 16]);
    }

    #[test]
    fn test_build_nea2_iv_max_bearer() {
        // Max bearer is 31 (0x1F)
        let iv = build_nea2_iv(0, 0x1F, 0);
        // 0x1F << 3 = 0xF8
        assert_eq!(iv[4], 0xF8);
    }

    #[test]
    fn test_nea2_encrypt_decrypt_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let count = 0x12345678;
        let bearer = 0x0A;
        let direction = 1;

        let original = b"Hello, 5G World! This is a test message for NEA2.";
        let mut data = original.to_vec();

        // Encrypt
        nea2_encrypt(count, bearer, direction, &key, &mut data);

        // Data should be different after encryption
        assert_ne!(&data[..], &original[..]);

        // Decrypt
        nea2_decrypt(count, bearer, direction, &key, &mut data);

        // Should match original
        assert_eq!(&data[..], &original[..]);
    }

    #[test]
    fn test_nea2_empty_data() {
        let key: [u8; 16] = [0u8; 16];
        let mut data: Vec<u8> = vec![];

        // Should not panic on empty data
        nea2_encrypt(0, 0, 0, &key, &mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_nea2_single_byte() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let original = [0x42u8];
        let mut data = original.to_vec();

        nea2_encrypt(0, 0, 0, &key, &mut data);
        assert_ne!(data[0], original[0]);

        nea2_decrypt(0, 0, 0, &key, &mut data);
        assert_eq!(data[0], original[0]);
    }

    /// Test vector based on 3GPP TS 33.501 / TS 33.401 test data
    /// This uses the 128-EEA2 test set from 3GPP specifications
    #[test]
    fn test_nea2_3gpp_test_vector_set1() {
        // 3GPP 128-EEA2 Test Set 1
        let key: [u8; 16] = [
            0xd3, 0xc5, 0xd5, 0x92, 0x32, 0x7f, 0xb1, 0x1c, 0x40, 0x35, 0xc6, 0x68, 0x0a, 0xf8,
            0xc6, 0xd1,
        ];
        let count: u32 = 0x398a59b4;
        let bearer: u8 = 0x15;
        let direction: u8 = 1;

        let plaintext: [u8; 16] = [
            0x98, 0x1b, 0xa6, 0x82, 0x4c, 0x1b, 0xfb, 0x1a, 0xb4, 0x85, 0x47, 0x20, 0x29, 0xb7,
            0x1d, 0x80,
        ];

        let expected_ciphertext: [u8; 16] = [
            0xe9, 0xfe, 0xd8, 0xa6, 0x3d, 0x15, 0x53, 0x04, 0xd7, 0x1d, 0xf2, 0x0b, 0xf3, 0xe8,
            0x22, 0x14,
        ];

        let mut data = plaintext.to_vec();
        nea2_encrypt(count, bearer, direction, &key, &mut data);

        assert_eq!(&data[..], &expected_ciphertext[..]);

        // Verify decryption
        nea2_decrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &plaintext[..]);
    }

    #[test]
    fn test_nea2_different_counts_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea2_encrypt(0, 0, 0, &key, &mut data1);
        nea2_encrypt(1, 0, 0, &key, &mut data2);

        // Different counts should produce different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea2_different_bearers_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea2_encrypt(0, 0, 0, &key, &mut data1);
        nea2_encrypt(0, 1, 0, &key, &mut data2);

        // Different bearers should produce different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea2_different_directions_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea2_encrypt(0, 0, 0, &key, &mut data1);
        nea2_encrypt(0, 0, 1, &key, &mut data2);

        // Different directions should produce different ciphertext
        assert_ne!(data1, data2);
    }

    /// Additional test: verify encryption is deterministic
    #[test]
    fn test_nea2_deterministic() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let plaintext = b"Test message for determinism check";
        
        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();
        
        nea2_encrypt(0x12345678, 5, 1, &key, &mut data1);
        nea2_encrypt(0x12345678, 5, 1, &key, &mut data2);
        
        assert_eq!(data1, data2);
    }

    /// Test with different key produces different output
    #[test]
    fn test_nea2_different_keys_produce_different_output() {
        let key1: [u8; 16] = [0x2b; 16];
        let key2: [u8; 16] = [0x3c; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea2_encrypt(0, 0, 0, &key1, &mut data1);
        nea2_encrypt(0, 0, 0, &key2, &mut data2);

        assert_ne!(data1, data2);
    }

    /// Test large data encryption/decryption
    #[test]
    fn test_nea2_large_data() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        
        // 1KB of data
        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut data = original.clone();
        
        nea2_encrypt(0xABCDEF01, 15, 1, &key, &mut data);
        assert_ne!(data, original);
        
        nea2_decrypt(0xABCDEF01, 15, 1, &key, &mut data);
        assert_eq!(data, original);
    }
}
