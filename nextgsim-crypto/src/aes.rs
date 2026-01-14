//! AES-128 block cipher wrapper
//!
//! Provides AES-128 operations needed by 5G security algorithms:
//! - Block encryption/decryption (for Milenage)
//! - CTR mode (for NEA2 ciphering)
//! - CMAC (for NIA2 integrity)

use aes::cipher::{BlockEncrypt, KeyInit, generic_array::GenericArray};
use aes::Aes128;
use cmac::{Cmac, Mac};

/// AES-128 block size in bytes
pub const BLOCK_SIZE: usize = 16;

/// AES-128 key size in bytes
pub const KEY_SIZE: usize = 16;

/// AES-128 block cipher for single-block operations
///
/// Used primarily by Milenage algorithm which operates on individual 128-bit blocks.
#[derive(Clone)]
pub struct Aes128Block {
    cipher: Aes128,
}

impl Aes128Block {
    /// Create a new AES-128 block cipher with the given key
    pub fn new(key: &[u8; KEY_SIZE]) -> Self {
        let cipher = Aes128::new(GenericArray::from_slice(key));
        Self { cipher }
    }

    /// Encrypt a single 16-byte block in place
    pub fn encrypt_block(&self, block: &mut [u8; BLOCK_SIZE]) {
        let mut generic_block = GenericArray::clone_from_slice(block);
        self.cipher.encrypt_block(&mut generic_block);
        block.copy_from_slice(&generic_block);
    }

    /// Encrypt a single 16-byte block, returning the result
    pub fn encrypt_block_copy(&self, block: &[u8; BLOCK_SIZE]) -> [u8; BLOCK_SIZE] {
        let mut result = *block;
        self.encrypt_block(&mut result);
        result
    }
}

/// XOR two 16-byte blocks
#[inline]
pub fn xor_block(a: &mut [u8; BLOCK_SIZE], b: &[u8; BLOCK_SIZE]) {
    for i in 0..BLOCK_SIZE {
        a[i] ^= b[i];
    }
}

/// AES-128 CMAC size in bytes
pub const CMAC_SIZE: usize = 16;

/// AES-128 CMAC for message authentication
///
/// Used for NIA2 (5G integrity algorithm based on AES-CMAC).
/// Implements RFC 4493 AES-CMAC.
#[derive(Clone)]
pub struct Aes128Cmac {
    key: [u8; KEY_SIZE],
}

impl Aes128Cmac {
    /// Create a new AES-128 CMAC instance with the given key
    pub fn new(key: &[u8; KEY_SIZE]) -> Self {
        Self { key: *key }
    }

    /// Compute CMAC over the given message, returning the full 16-byte MAC
    pub fn compute(&self, message: &[u8]) -> [u8; CMAC_SIZE] {
        let mut mac = <Cmac<Aes128> as Mac>::new_from_slice(&self.key)
            .expect("CMAC key size is always valid");
        mac.update(message);
        let result = mac.finalize();
        let mut output = [0u8; CMAC_SIZE];
        output.copy_from_slice(&result.into_bytes());
        output
    }

    /// Compute CMAC and return only the first `len` bytes (truncated MAC)
    ///
    /// This is useful for NIA2 which uses a 32-bit (4-byte) MAC.
    pub fn compute_truncated(&self, message: &[u8], len: usize) -> Vec<u8> {
        let full_mac = self.compute(message);
        full_mac[..len.min(CMAC_SIZE)].to_vec()
    }

    /// Verify a CMAC tag against a message
    pub fn verify(&self, message: &[u8], tag: &[u8]) -> bool {
        let computed = self.compute(message);
        // Constant-time comparison for the length of the provided tag
        if tag.len() > CMAC_SIZE {
            return false;
        }
        let mut result = 0u8;
        for (a, b) in computed[..tag.len()].iter().zip(tag.iter()) {
            result |= a ^ b;
        }
        result == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aes128_encrypt_block() {
        // NIST FIPS 197 test vector
        let key: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        ];
        let plaintext: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let expected: [u8; 16] = [
            0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
            0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
        ];

        let cipher = Aes128Block::new(&key);
        let result = cipher.encrypt_block_copy(&plaintext);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_aes128_encrypt_block_in_place() {
        let key: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        ];
        let mut block: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
            0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let expected: [u8; 16] = [
            0x69, 0xc4, 0xe0, 0xd8, 0x6a, 0x7b, 0x04, 0x30,
            0xd8, 0xcd, 0xb7, 0x80, 0x70, 0xb4, 0xc5, 0x5a,
        ];

        let cipher = Aes128Block::new(&key);
        cipher.encrypt_block(&mut block);
        assert_eq!(block, expected);
    }

    #[test]
    fn test_xor_block() {
        let mut a: [u8; 16] = [0xff; 16];
        let b: [u8; 16] = [0xaa; 16];
        xor_block(&mut a, &b);
        assert_eq!(a, [0x55; 16]);
    }

    // RFC 4493 AES-CMAC test vectors
    // Key: 2b7e1516 28aed2a6 abf71588 09cf4f3c

    #[test]
    fn test_cmac_empty_message() {
        // RFC 4493 Example 1: Empty message
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 0] = [];
        let expected: [u8; 16] = [
            0xbb, 0x1d, 0x69, 0x29, 0xe9, 0x59, 0x37, 0x28,
            0x7f, 0xa3, 0x7d, 0x12, 0x9b, 0x75, 0x67, 0x46,
        ];

        let cmac = Aes128Cmac::new(&key);
        let result = cmac.compute(&message);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cmac_16_byte_message() {
        // RFC 4493 Example 2: 16-byte message (exactly one block)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 16] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
        ];
        let expected: [u8; 16] = [
            0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
            0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7c,
        ];

        let cmac = Aes128Cmac::new(&key);
        let result = cmac.compute(&message);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cmac_40_byte_message() {
        // RFC 4493 Example 3: 40-byte message (not block-aligned)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 40] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
            0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
            0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
            0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11,
        ];
        let expected: [u8; 16] = [
            0xdf, 0xa6, 0x67, 0x47, 0xde, 0x9a, 0xe6, 0x30,
            0x30, 0xca, 0x32, 0x61, 0x14, 0x97, 0xc8, 0x27,
        ];

        let cmac = Aes128Cmac::new(&key);
        let result = cmac.compute(&message);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cmac_64_byte_message() {
        // RFC 4493 Example 4: 64-byte message (exactly 4 blocks)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 64] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
            0xae, 0x2d, 0x8a, 0x57, 0x1e, 0x03, 0xac, 0x9c,
            0x9e, 0xb7, 0x6f, 0xac, 0x45, 0xaf, 0x8e, 0x51,
            0x30, 0xc8, 0x1c, 0x46, 0xa3, 0x5c, 0xe4, 0x11,
            0xe5, 0xfb, 0xc1, 0x19, 0x1a, 0x0a, 0x52, 0xef,
            0xf6, 0x9f, 0x24, 0x45, 0xdf, 0x4f, 0x9b, 0x17,
            0xad, 0x2b, 0x41, 0x7b, 0xe6, 0x6c, 0x37, 0x10,
        ];
        let expected: [u8; 16] = [
            0x51, 0xf0, 0xbe, 0xbf, 0x7e, 0x3b, 0x9d, 0x92,
            0xfc, 0x49, 0x74, 0x17, 0x79, 0x36, 0x3c, 0xfe,
        ];

        let cmac = Aes128Cmac::new(&key);
        let result = cmac.compute(&message);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cmac_truncated() {
        // Test truncated MAC (4 bytes, as used in NIA2)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 16] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
        ];
        // Full MAC: 070a16b4 6b4d4144 f79bdd9d d04a287c
        let expected_4_bytes: [u8; 4] = [0x07, 0x0a, 0x16, 0xb4];

        let cmac = Aes128Cmac::new(&key);
        let result = cmac.compute_truncated(&message, 4);
        assert_eq!(result, expected_4_bytes);
    }

    #[test]
    fn test_cmac_verify() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 16] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
        ];
        let valid_tag: [u8; 16] = [
            0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
            0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7c,
        ];
        let invalid_tag: [u8; 16] = [
            0x07, 0x0a, 0x16, 0xb4, 0x6b, 0x4d, 0x41, 0x44,
            0xf7, 0x9b, 0xdd, 0x9d, 0xd0, 0x4a, 0x28, 0x7d, // Last byte changed
        ];

        let cmac = Aes128Cmac::new(&key);
        assert!(cmac.verify(&message, &valid_tag));
        assert!(!cmac.verify(&message, &invalid_tag));
    }

    #[test]
    fn test_cmac_verify_truncated() {
        // Verify with truncated tag (4 bytes)
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
            0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
        ];
        let message: [u8; 16] = [
            0x6b, 0xc1, 0xbe, 0xe2, 0x2e, 0x40, 0x9f, 0x96,
            0xe9, 0x3d, 0x7e, 0x11, 0x73, 0x93, 0x17, 0x2a,
        ];
        let valid_truncated_tag: [u8; 4] = [0x07, 0x0a, 0x16, 0xb4];
        let invalid_truncated_tag: [u8; 4] = [0x07, 0x0a, 0x16, 0xb5];

        let cmac = Aes128Cmac::new(&key);
        assert!(cmac.verify(&message, &valid_truncated_tag));
        assert!(!cmac.verify(&message, &invalid_truncated_tag));
    }
}
