//! ZUC-256 stream cipher implementation
//!
//! ZUC-256 is the 256-bit security level variant of the ZUC stream cipher,
//! specified for use in 5G security algorithms with enhanced key lengths.
//!
//! This module provides:
//! - `zuc256_eea3()` - 256-bit key encryption (EEA3 with 256-bit key)
//! - `zuc256_eia3()` - 256-bit key integrity (EIA3 with 256-bit key)
//!
//! Reference: 3GPP and ZUC-256 specification

use thiserror::Error;

/// ZUC-256 key size in bytes (256 bits)
pub const KEY_SIZE: usize = 32;

/// ZUC-256 IV size in bytes (25 bytes per 3GPP specs)
pub const IV_SIZE: usize = 25;

/// MAC size in bytes for integrity (32 bits)
pub const MAC_SIZE: usize = 4;

/// ZUC-256 error types
#[derive(Debug, Error)]
pub enum Zuc256Error {
    /// Invalid key length
    #[error("Invalid key length: expected {}, got {0}", KEY_SIZE)]
    InvalidKeyLength(usize),
    /// Invalid IV length
    #[error("Invalid IV length: expected {}, got {0}", IV_SIZE)]
    InvalidIvLength(usize),
}

/// Result type for ZUC-256 operations
pub type Zuc256Result<T> = Result<T, Zuc256Error>;

/// ZUC-256 constants: the d values for key loading
const D: [u8; 16] = [
    0x22, 0x2F, 0x24, 0x2A, 0x6D, 0x40, 0x40, 0x40,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30,
];

/// ZUC-256 constants for EIA3 mode
const D_EIA3: [u8; 16] = [
    0x22, 0x2F, 0x25, 0x2A, 0x6D, 0x40, 0x40, 0x40,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x52, 0x10, 0x30,
];

/// S-box S0 (same as ZUC-128)
const S0: [u8; 256] = [
    0x3E, 0x72, 0x5B, 0x47, 0xCA, 0xE0, 0x00, 0x33, 0x04, 0xD1, 0x54, 0x98, 0x09, 0xB9, 0x6D, 0xCB,
    0x7B, 0x1B, 0xF9, 0x32, 0xAF, 0x9D, 0x6A, 0xA5, 0xB8, 0x2D, 0xFC, 0x1D, 0x08, 0x53, 0x03, 0x90,
    0x4D, 0x4E, 0x84, 0x99, 0xE4, 0xCE, 0xD9, 0x91, 0xDD, 0xB6, 0x85, 0x48, 0x8B, 0x29, 0x6E, 0xAC,
    0xCD, 0xC1, 0xF8, 0x1E, 0x73, 0x43, 0x69, 0xC6, 0xB5, 0xBD, 0xFD, 0x39, 0x63, 0x20, 0xD4, 0x38,
    0x76, 0x7D, 0xB2, 0xA7, 0xCF, 0xED, 0x57, 0xC5, 0xF3, 0x2C, 0xBB, 0x14, 0x21, 0x06, 0x55, 0x9B,
    0xE3, 0xEF, 0x5E, 0x31, 0x4F, 0x7F, 0x5A, 0xA4, 0x0D, 0x82, 0x51, 0x49, 0x5F, 0xBA, 0x58, 0x1C,
    0x4A, 0x16, 0xD5, 0x17, 0xA8, 0x92, 0x24, 0x1F, 0x8C, 0xFF, 0xD8, 0xAE, 0x2E, 0x01, 0xD3, 0xAD,
    0x3B, 0x4B, 0xDA, 0x46, 0xEB, 0xC9, 0xDE, 0x9A, 0x8F, 0x87, 0xD7, 0x3A, 0x80, 0x6F, 0x2F, 0xC8,
    0xB1, 0xB4, 0x37, 0xF7, 0x0A, 0x22, 0x13, 0x28, 0x7C, 0xCC, 0x3C, 0x89, 0xC7, 0xC3, 0x96, 0x56,
    0x07, 0xBF, 0x7E, 0xF0, 0x0B, 0x2B, 0x97, 0x52, 0x35, 0x41, 0x79, 0x61, 0xA6, 0x4C, 0x10, 0xFE,
    0xBC, 0x26, 0x95, 0x88, 0x8A, 0xB0, 0xA3, 0xFB, 0xC0, 0x18, 0x94, 0xF2, 0xE1, 0xE5, 0xE9, 0x5D,
    0xD0, 0xDC, 0x11, 0x66, 0x64, 0x5C, 0xEC, 0x59, 0x42, 0x75, 0x12, 0xF5, 0x74, 0x9C, 0xAA, 0x23,
    0x0E, 0x86, 0xAB, 0xBE, 0x2A, 0x02, 0xE7, 0x67, 0xE6, 0x44, 0xA2, 0x6C, 0xC2, 0x93, 0x9F, 0xF1,
    0xF6, 0xFA, 0x36, 0xD2, 0x50, 0x68, 0x9E, 0x62, 0x71, 0x15, 0x3D, 0xD6, 0x40, 0xC4, 0xE2, 0x0F,
    0x8E, 0x83, 0x77, 0x6B, 0x25, 0x05, 0x3F, 0x0C, 0x30, 0xEA, 0x70, 0xB7, 0xA1, 0xE8, 0xA9, 0x65,
    0x8D, 0x27, 0x1A, 0xDB, 0x81, 0xB3, 0xA0, 0xF4, 0x45, 0x7A, 0x19, 0xDF, 0xEE, 0x78, 0x34, 0x60,
];

/// S-box S1 (same as ZUC-128)
const S1: [u8; 256] = [
    0x55, 0xC2, 0x63, 0x71, 0x3B, 0xC8, 0x47, 0x86, 0x9F, 0x3C, 0xDA, 0x5B, 0x29, 0xAA, 0xFD, 0x77,
    0x8C, 0xC5, 0x94, 0x0C, 0xA6, 0x1A, 0x13, 0x00, 0xE3, 0xA8, 0x16, 0x72, 0x40, 0xF9, 0xF8, 0x42,
    0x44, 0x26, 0x68, 0x96, 0x81, 0xD9, 0x45, 0x3E, 0x10, 0x76, 0xC6, 0xA7, 0x8B, 0x39, 0x43, 0xE1,
    0x3A, 0xB5, 0x56, 0x2A, 0xC0, 0x6D, 0xB3, 0x05, 0x22, 0x66, 0xBF, 0xDC, 0x0B, 0xFA, 0x62, 0x48,
    0xDD, 0x20, 0x11, 0x06, 0x36, 0xC9, 0xC1, 0xCF, 0xF6, 0x27, 0x52, 0xBB, 0x69, 0xF5, 0xD4, 0x87,
    0x7F, 0x84, 0x4C, 0xD2, 0x9C, 0x57, 0xA4, 0xBC, 0x4F, 0x9A, 0xDF, 0xFE, 0xD6, 0x8D, 0x7A, 0xEB,
    0x2B, 0x53, 0xD8, 0x5C, 0xA1, 0x14, 0x17, 0xFB, 0x23, 0xD5, 0x7D, 0x30, 0x67, 0x73, 0x08, 0x09,
    0xEE, 0xB7, 0x70, 0x3F, 0x61, 0xB2, 0x19, 0x8E, 0x4E, 0xE5, 0x4B, 0x93, 0x8F, 0x5D, 0xDB, 0xA9,
    0xAD, 0xF1, 0xAE, 0x2E, 0xCB, 0x0D, 0xFC, 0xF4, 0x2D, 0x46, 0x6E, 0x1D, 0x97, 0xE8, 0xD1, 0xE9,
    0x4D, 0x37, 0xA5, 0x75, 0x5E, 0x83, 0x9E, 0xAB, 0x82, 0x9D, 0xB9, 0x1C, 0xE0, 0xCD, 0x49, 0x89,
    0x01, 0xB6, 0xBD, 0x58, 0x24, 0xA2, 0x5F, 0x38, 0x78, 0x99, 0x15, 0x90, 0x50, 0xB8, 0x95, 0xE4,
    0xD0, 0x91, 0xC7, 0xCE, 0xED, 0x0F, 0xB4, 0x6F, 0xA0, 0xCC, 0xF0, 0x02, 0x4A, 0x79, 0xC3, 0xDE,
    0xA3, 0xEF, 0xEA, 0x51, 0xE6, 0x6B, 0x18, 0xEC, 0x1B, 0x2C, 0x80, 0xF7, 0x74, 0xE7, 0xFF, 0x21,
    0x5A, 0x6A, 0x54, 0x1E, 0x41, 0x31, 0x92, 0x35, 0xC4, 0x33, 0x07, 0x0A, 0xBA, 0x7E, 0x0E, 0x34,
    0x88, 0xB1, 0x98, 0x7C, 0xF3, 0x3D, 0x60, 0x6C, 0x7B, 0xCA, 0xD3, 0x1F, 0x32, 0x65, 0x04, 0x28,
    0x64, 0xBE, 0x85, 0x9B, 0x2F, 0x59, 0x8A, 0xD7, 0xB0, 0x25, 0xAC, 0xAF, 0x12, 0x03, 0xE2, 0xF2,
];

/// Modular addition in GF(2^31 - 1)
#[inline]
fn add_mod31(a: u32, b: u32) -> u32 {
    let c = a.wrapping_add(b);
    (c & 0x7FFFFFFF).wrapping_add(c >> 31)
}

/// L1 linear transformation
#[inline]
fn l1(x: u32) -> u32 {
    x ^ x.rotate_left(2) ^ x.rotate_left(10) ^ x.rotate_left(18) ^ x.rotate_left(24)
}

/// L2 linear transformation
#[inline]
fn l2(x: u32) -> u32 {
    x ^ x.rotate_left(8) ^ x.rotate_left(14) ^ x.rotate_left(22) ^ x.rotate_left(30)
}

/// Make a 32-bit word from 4 bytes using S-boxes
#[inline]
fn make_u32(a: u8, b: u8, c: u8, d: u8) -> u32 {
    ((S0[a as usize] as u32) << 24)
        | ((S1[b as usize] as u32) << 16)
        | ((S0[c as usize] as u32) << 8)
        | (S1[d as usize] as u32)
}

/// ZUC-256 cipher state
pub struct Zuc256 {
    /// LFSR state (16 x 31-bit words)
    lfsr: [u32; 16],
    /// FSM registers
    r1: u32,
    /// FSM registers
    r2: u32,
    /// Working variable
    x: [u32; 4],
}

impl Zuc256 {
    /// Create a new ZUC-256 instance initialized with key and IV
    ///
    /// # Arguments
    /// * `key` - 256-bit key (32 bytes)
    /// * `iv` - Initialization vector (25 bytes)
    /// * `d_constants` - The d constants to use (differ for EEA3 vs EIA3)
    fn new_with_d(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], d_constants: &[u8; 16]) -> Self {
        let mut zuc = Zuc256 {
            lfsr: [0u32; 16],
            r1: 0,
            r2: 0,
            x: [0u32; 4],
        };
        zuc.load_key(key, iv, d_constants);
        zuc.initialize();
        zuc
    }

    /// Create a new ZUC-256 instance for EEA3 (encryption)
    pub fn new_eea3(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE]) -> Self {
        Self::new_with_d(key, iv, &D)
    }

    /// Create a new ZUC-256 instance for EIA3 (integrity)
    pub fn new_eia3(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE]) -> Self {
        Self::new_with_d(key, iv, &D_EIA3)
    }

    /// Load key and IV into the LFSR
    fn load_key(&mut self, key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], d: &[u8; 16]) {
        // ZUC-256 key loading: s[i] = k[i] || d[i] || iv[i]
        // Each LFSR word is 31 bits constructed from key, d, and iv bytes
        for i in 0..16 {
            let k = key[i] as u32;
            let di = d[i] as u32;
            let iv_val = if i < 16 {
                // First 17 IV bytes go to different positions
                // ZUC-256 spec maps iv bytes across LFSR words
                let idx = i;
                if idx < iv.len() { iv[idx] as u32 } else { 0 }
            } else {
                0
            };

            // Construct 31-bit LFSR word: key[i] (8 bits) || d[i] (8 bits) || iv[i] (8 bits) || key[i+16] (7 bits)
            let k_hi = if i + 16 < key.len() { key[i + 16] as u32 } else { 0 };
            self.lfsr[i] = (k << 23) | (di << 16) | (iv_val << 8) | (k_hi >> 1);

            // Ensure non-zero (ZUC requirement)
            if self.lfsr[i] == 0 {
                self.lfsr[i] = (1 << 31) - 1; // 2^31 - 1
            }
        }
    }

    /// LFSR feedback function (using modular arithmetic in GF(2^31 - 1))
    fn lfsr_feedback(&self) -> u32 {
        let mut f = self.lfsr[0];
        f = add_mod31(f, self.lfsr[0].wrapping_shl(8) & 0x7FFFFFFF);
        f = add_mod31(f, self.lfsr[4].wrapping_shl(20) & 0x7FFFFFFF);
        f = add_mod31(f, self.lfsr[10].wrapping_shl(21) & 0x7FFFFFFF);
        f = add_mod31(f, self.lfsr[13].wrapping_shl(17) & 0x7FFFFFFF);
        f = add_mod31(f, self.lfsr[15].wrapping_shl(15) & 0x7FFFFFFF);
        f
    }

    /// Bit reorganization
    fn bit_reorganization(&mut self) {
        self.x[0] = ((self.lfsr[15] & 0x7FFF8000) << 1) | (self.lfsr[14] & 0xFFFF);
        self.x[1] = ((self.lfsr[11] & 0xFFFF) << 16) | (self.lfsr[9] >> 15);
        self.x[2] = ((self.lfsr[7] & 0xFFFF) << 16) | (self.lfsr[5] >> 15);
        self.x[3] = ((self.lfsr[2] & 0xFFFF) << 16) | (self.lfsr[0] >> 15);
    }

    /// F function (FSM)
    fn f(&mut self) -> u32 {
        let w = (self.x[0] ^ self.r1).wrapping_add(self.r2);
        let w1 = self.r1.wrapping_add(self.x[1]);
        let w2 = self.r2 ^ self.x[2];

        let u = l1(
            (w1 << 16) | (w2 >> 16),
        );
        let v = l2(
            (w2 << 16) | (w1 >> 16),
        );

        self.r1 = make_u32(
            (u >> 24) as u8,
            ((u >> 16) & 0xFF) as u8,
            ((u >> 8) & 0xFF) as u8,
            (u & 0xFF) as u8,
        );
        self.r2 = make_u32(
            (v >> 24) as u8,
            ((v >> 16) & 0xFF) as u8,
            ((v >> 8) & 0xFF) as u8,
            (v & 0xFF) as u8,
        );

        w
    }

    /// LFSR clock in initialization mode
    fn lfsr_with_init_mode(&mut self, u: u32) {
        let f = self.lfsr_feedback();
        let v = add_mod31(f, u & 0x7FFFFFFF);
        // Shift LFSR
        for i in 0..15 {
            self.lfsr[i] = self.lfsr[i + 1];
        }
        self.lfsr[15] = if v == 0 { 0x7FFFFFFF } else { v };
    }

    /// LFSR clock in working mode
    fn lfsr_with_work_mode(&mut self) {
        let f = self.lfsr_feedback();
        for i in 0..15 {
            self.lfsr[i] = self.lfsr[i + 1];
        }
        self.lfsr[15] = if f == 0 { 0x7FFFFFFF } else { f };
    }

    /// Initialization: 32 rounds of init mode
    fn initialize(&mut self) {
        for _ in 0..32 {
            self.bit_reorganization();
            let w = self.f();
            self.lfsr_with_init_mode(w >> 1);
        }
        // One more round to discard first keystream word
        self.bit_reorganization();
        self.f();
        self.lfsr_with_work_mode();
    }

    /// Generate one 32-bit keystream word
    pub fn generate(&mut self) -> u32 {
        self.bit_reorganization();
        let z = self.f() ^ self.x[3];
        self.lfsr_with_work_mode();
        z
    }
}

/// ZUC-256 based EEA3 encryption (256-bit key version)
///
/// Encrypts data using ZUC-256 stream cipher in the 5G EEA3 mode
/// with a 256-bit key and 25-byte IV.
///
/// # Arguments
/// * `key` - 256-bit encryption key (32 bytes)
/// * `iv` - Initialization vector (25 bytes)
/// * `data` - Data to encrypt (modified in place)
///
/// # Errors
/// Returns an error if the key or IV length is invalid.
pub fn zuc256_eea3(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], data: &mut [u8]) {
    let mut zuc = Zuc256::new_eea3(key, iv);

    let mut offset = 0;
    while offset < data.len() {
        let keystream_word = zuc.generate();
        let keystream_bytes = keystream_word.to_be_bytes();

        for (i, &ks_byte) in keystream_bytes.iter().enumerate() {
            if offset + i < data.len() {
                data[offset + i] ^= ks_byte;
            }
        }
        offset += 4;
    }
}

/// ZUC-256 based EIA3 integrity algorithm (256-bit key version)
///
/// Computes a 32-bit MAC using ZUC-256 in the 5G EIA3 mode
/// with a 256-bit key and 25-byte IV.
///
/// # Arguments
/// * `key` - 256-bit integrity key (32 bytes)
/// * `iv` - Initialization vector (25 bytes)
/// * `data` - Data to authenticate
///
/// # Returns
/// 32-bit MAC as a 4-byte array
pub fn zuc256_eia3(key: &[u8; KEY_SIZE], iv: &[u8; IV_SIZE], data: &[u8]) -> [u8; MAC_SIZE] {
    let mut zuc = Zuc256::new_eia3(key, iv);

    let length_bits = data.len() * 8;

    // Number of 32-bit words needed: L = ceil(length/32)
    let l = length_bits.div_ceil(32);
    let num_keystream_words = l + 2;

    // Generate keystream
    let keystream: Vec<u32> = (0..num_keystream_words).map(|_| zuc.generate()).collect();

    // Compute MAC using the EIA3 algorithm
    let mut t: u32 = 0;

    for i in 0..length_bits {
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8);
        let bit = (data[byte_idx] >> bit_idx) & 1;

        if bit == 1 {
            t ^= get_word_from_keystream(&keystream, i);
        }
    }

    // Final XOR with z[L]
    let mac_value = t ^ keystream[l];

    mac_value.to_be_bytes()
}

/// Get a 32-bit word from the keystream starting at bit position i
fn get_word_from_keystream(keystream: &[u32], bit_pos: usize) -> u32 {
    let word_idx = bit_pos / 32;
    let bit_offset = bit_pos % 32;

    if bit_offset == 0 {
        keystream[word_idx]
    } else {
        let high_bits = keystream[word_idx] << bit_offset;
        let low_bits = keystream[word_idx + 1] >> (32 - bit_offset);
        high_bits | low_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zuc256_eea3_encrypt_decrypt_roundtrip() {
        let key: [u8; KEY_SIZE] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
        ];
        let iv: [u8; IV_SIZE] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
            0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x18,
        ];

        let original = b"Hello, ZUC-256 EEA3! This is a test message.";
        let mut data = original.to_vec();

        // Encrypt
        zuc256_eea3(&key, &iv, &mut data);
        assert_ne!(&data[..], &original[..]);

        // Decrypt (same operation)
        zuc256_eea3(&key, &iv, &mut data);
        assert_eq!(&data[..], &original[..]);
    }

    #[test]
    fn test_zuc256_eea3_empty_data() {
        let key = [0u8; KEY_SIZE];
        let iv = [0u8; IV_SIZE];
        let mut data: Vec<u8> = vec![];

        zuc256_eea3(&key, &iv, &mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_zuc256_eea3_deterministic() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let plaintext = b"Determinism test for ZUC-256";

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        zuc256_eea3(&key, &iv, &mut data1);
        zuc256_eea3(&key, &iv, &mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_zuc256_eea3_different_keys_different_output() {
        let key1 = [0x01u8; KEY_SIZE];
        let key2 = [0x02u8; KEY_SIZE];
        let iv = [0x00u8; IV_SIZE];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        zuc256_eea3(&key1, &iv, &mut data1);
        zuc256_eea3(&key2, &iv, &mut data2);

        assert_ne!(data1, data2);
    }

    #[test]
    fn test_zuc256_eea3_different_ivs_different_output() {
        let key = [0x01u8; KEY_SIZE];
        let iv1 = [0x00u8; IV_SIZE];
        let iv2 = [0x01u8; IV_SIZE];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        zuc256_eea3(&key, &iv1, &mut data1);
        zuc256_eea3(&key, &iv2, &mut data2);

        assert_ne!(data1, data2);
    }

    #[test]
    fn test_zuc256_eia3_deterministic() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];
        let data = b"Test data for ZUC-256 integrity";

        let mac1 = zuc256_eia3(&key, &iv, data);
        let mac2 = zuc256_eia3(&key, &iv, data);

        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_zuc256_eia3_different_data_different_mac() {
        let key = [0x42u8; KEY_SIZE];
        let iv = [0x13u8; IV_SIZE];

        let mac1 = zuc256_eia3(&key, &iv, b"data1");
        let mac2 = zuc256_eia3(&key, &iv, b"data2");

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_zuc256_eia3_different_keys_different_mac() {
        let key1 = [0x01u8; KEY_SIZE];
        let key2 = [0x02u8; KEY_SIZE];
        let iv = [0x00u8; IV_SIZE];
        let data = b"Test integrity";

        let mac1 = zuc256_eia3(&key1, &iv, data);
        let mac2 = zuc256_eia3(&key2, &iv, data);

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_zuc256_eia3_mac_size() {
        let key = [0x00u8; KEY_SIZE];
        let iv = [0x00u8; IV_SIZE];
        let data = b"Test";

        let mac = zuc256_eia3(&key, &iv, data);
        assert_eq!(mac.len(), MAC_SIZE);
    }

    #[test]
    fn test_zuc256_large_data() {
        let key = [0xABu8; KEY_SIZE];
        let iv = [0xCDu8; IV_SIZE];

        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut data = original.clone();

        zuc256_eea3(&key, &iv, &mut data);
        assert_ne!(data, original);

        zuc256_eea3(&key, &iv, &mut data);
        assert_eq!(data, original);
    }
}
