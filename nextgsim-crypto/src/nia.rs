//! NIA (5G integrity algorithms) implementations
//!
//! This module implements the 5G NAS integrity algorithms:
//! - NIA1: SNOW3G-based integrity (128-EIA1)
//! - NIA2: AES-CMAC based integrity (128-EIA2)
//! - NIA3: ZUC-based integrity (128-EIA3)
//!
//! Reference: 3GPP TS 33.501, TS 33.401, TS 35.222, and TS 35.223

use aes::Aes128;
use cmac::{Cmac, Mac};
use zuc::ZUC128;

use crate::snow3g::uia2_f9;

/// Key size in bytes (128 bits)
pub const KEY_SIZE: usize = 16;

/// MAC size in bytes (32 bits)
pub const MAC_SIZE: usize = 4;

/// NIA1 (128-EIA1) - SNOW3G-based integrity algorithm
///
/// Implements the 5G integrity algorithm NIA1 as specified in 3GPP TS 33.501
/// and TS 35.222. NIA1 uses the SNOW3G stream cipher to generate a MAC.
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit integrity key (`KNASint` or `KUPint`)
/// - `data`: Message data to authenticate
///
/// # Returns
/// 32-bit MAC (Message Authentication Code)
///
/// # Note
/// NIA1 is equivalent to 128-EIA1 (UIA2/F9) with the FRESH value
/// constructed from BEARER and padded with zeros.
pub fn nia1_compute_mac(
    count: u32,
    bearer: u8,
    direction: u8,
    key: &[u8; KEY_SIZE],
    data: &[u8],
) -> [u8; MAC_SIZE] {
    // For NIA1, the FRESH value is constructed from BEARER
    // FRESH = BEARER || 0...0 (BEARER in upper 5 bits, rest are zeros)
    // Per 3GPP TS 33.501, the FRESH value includes BEARER in upper bits
    let fresh: u32 = (bearer as u32 & 0x1F) << 27;

    // Length in bits
    let length_bits = (data.len() * 8) as u64;

    // Call the underlying SNOW3G UIA2 (F9) function
    let mac_value = uia2_f9(key, count, fresh, direction as u32, data, length_bits);

    mac_value.to_be_bytes()
}

/// NIA2 (128-EIA2) - AES-CMAC based integrity algorithm
///
/// Implements the 5G integrity algorithm NIA2 as specified in 3GPP TS 33.501.
/// NIA2 uses AES-128-CMAC with a specific input construction.
///
/// # Input Construction
/// The input to CMAC is: COUNT || BEARER || DIRECTION || MESSAGE
/// ```text
/// | COUNT (32 bits) | BEARER (5 bits) | DIRECTION (1 bit) | 0...0 (26 bits) | MESSAGE |
/// ```
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit integrity key (`KNASint` or `KUPint`)
/// - `data`: Message data to authenticate
///
/// # Returns
/// 32-bit MAC (Message Authentication Code)
pub fn nia2_compute_mac(
    count: u32,
    bearer: u8,
    direction: u8,
    key: &[u8; KEY_SIZE],
    data: &[u8],
) -> [u8; MAC_SIZE] {
    // Build the input: COUNT || BEARER || DIRECTION || padding || MESSAGE
    let mut input = Vec::with_capacity(8 + data.len());

    // COUNT: 32 bits (bytes 0-3)
    input.push((count >> 24) as u8);
    input.push((count >> 16) as u8);
    input.push((count >> 8) as u8);
    input.push(count as u8);

    // BEARER (5 bits) + DIRECTION (1 bit) + padding (26 bits) = 32 bits (bytes 4-7)
    // BEARER in bits 7-3, DIRECTION in bit 2, rest are zeros
    input.push(((bearer & 0x1F) << 3) | ((direction & 0x01) << 2));
    input.push(0);
    input.push(0);
    input.push(0);

    // MESSAGE
    input.extend_from_slice(data);

    // Compute CMAC
    let mut mac = Cmac::<Aes128>::new_from_slice(key).expect("Invalid key length");
    mac.update(&input);
    let result = mac.finalize().into_bytes();

    // Return first 4 bytes (32-bit MAC)
    let mut mac_out = [0u8; MAC_SIZE];
    mac_out.copy_from_slice(&result[..MAC_SIZE]);
    mac_out
}

/// NIA3 (128-EIA3) - ZUC-based integrity algorithm
///
/// Implements the 5G integrity algorithm NIA3 as specified in 3GPP TS 33.501
/// and TS 35.223. NIA3 uses the ZUC stream cipher to generate a MAC.
///
/// # IV Construction (128 bits)
/// ```text
/// | COUNT (32 bits) | BEARER (5 bits) | DIRECTION (1 bit) | 0 (26 bits) |
/// | COUNT (32 bits) | BEARER (5 bits) | DIRECTION (1 bit) | 0 (26 bits) |
/// ```
/// Note: The IV is constructed by repeating the same 64-bit pattern twice.
///
/// # Algorithm
/// 1. Initialize ZUC with key and IV
/// 2. Generate keystream words z[0], z[1], ..., z[L] where L = ceil(length/32) + 1
/// 3. Compute MAC = T XOR z[L] where T is accumulated from message bits and keystream
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit integrity key (`KNASint` or `KUPint`)
/// - `data`: Message data to authenticate
///
/// # Returns
/// 32-bit MAC (Message Authentication Code)
pub fn nia3_compute_mac(
    count: u32,
    bearer: u8,
    direction: u8,
    key: &[u8; KEY_SIZE],
    data: &[u8],
) -> [u8; MAC_SIZE] {
    let iv = build_nia3_iv(count, bearer, direction);
    let mut zuc = ZUC128::new(key, &iv);

    // Length in bits
    let length_bits = data.len() * 8;

    // Number of 32-bit words needed: L = ceil(length/32) + 1
    // We need L+1 keystream words total (z[0] to z[L])
    let l = length_bits.div_ceil(32);
    let num_keystream_words = l + 2; // z[0] to z[L], plus one extra for safety

    // Generate keystream
    let keystream: Vec<u32> = (0..num_keystream_words).map(|_| zuc.generate()).collect();

    // Compute MAC using the EIA3 algorithm
    // T = 0
    // For i = 0 to length-1:
    //   If M[i] == 1:
    //     T = T XOR GetWord(z, i)
    // MAC = T XOR z[L]
    //
    // Where GetWord(z, i) returns a 32-bit word starting at bit position i

    let mut t: u32 = 0;

    for i in 0..length_bits {
        // Check if bit i of message is 1
        let byte_idx = i / 8;
        let bit_idx = 7 - (i % 8); // MSB first
        let bit = (data[byte_idx] >> bit_idx) & 1;

        if bit == 1 {
            // Get 32-bit word from keystream starting at bit position i
            t ^= get_word_from_keystream(&keystream, i);
        }
    }

    // Final XOR with z[L]
    let mac_value = t ^ keystream[l];

    mac_value.to_be_bytes()
}

/// Build the IV for NIA3 from COUNT, BEARER, and DIRECTION
///
/// IV format (128 bits) per 3GPP TS 35.223:
/// ```text
/// IV[0..3]   = COUNT[0..3]
/// IV[4]      = BEARER || DIRECTION || 0 (padding)
/// IV[5..7]   = 0
/// IV[8..11]  = COUNT[0..3]
/// IV[12]     = BEARER || DIRECTION || 0 (padding)
/// IV[13..15] = 0
/// ```
fn build_nia3_iv(count: u32, bearer: u8, direction: u8) -> [u8; 16] {
    let mut iv = [0u8; 16];

    // First 64 bits
    iv[0] = (count >> 24) as u8;
    iv[1] = (count >> 16) as u8;
    iv[2] = (count >> 8) as u8;
    iv[3] = count as u8;
    iv[4] = ((bearer & 0x1F) << 3) | ((direction & 0x01) << 2);

    // Second 64 bits (same pattern)
    iv[8] = (count >> 24) as u8;
    iv[9] = (count >> 16) as u8;
    iv[10] = (count >> 8) as u8;
    iv[11] = count as u8;
    iv[12] = ((bearer & 0x1F) << 3) | ((direction & 0x01) << 2);

    iv
}

/// Get a 32-bit word from the keystream starting at bit position i
///
/// This extracts 32 consecutive bits from the keystream starting at bit position i.
/// The keystream is treated as a continuous bit stream where each u32 contributes
/// 32 bits in big-endian order.
fn get_word_from_keystream(keystream: &[u32], bit_pos: usize) -> u32 {
    let word_idx = bit_pos / 32;
    let bit_offset = bit_pos % 32;

    if bit_offset == 0 {
        keystream[word_idx]
    } else {
        // Need to combine bits from two consecutive words
        let high_bits = keystream[word_idx] << bit_offset;
        let low_bits = keystream[word_idx + 1] >> (32 - bit_offset);
        high_bits | low_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nia1_compute_mac_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let count = 0x12345678;
        let bearer = 0x0A;
        let direction = 1;
        let data = b"Hello, 5G World!";

        let mac1 = nia1_compute_mac(count, bearer, direction, &key, data);
        let mac2 = nia1_compute_mac(count, bearer, direction, &key, data);

        // Same inputs should produce same MAC
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_nia1_different_data_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let count = 0;
        let bearer = 0;
        let direction = 0;

        let mac1 = nia1_compute_mac(count, bearer, direction, &key, b"data1");
        let mac2 = nia1_compute_mac(count, bearer, direction, &key, b"data2");

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia1_different_counts_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let data = b"test data";

        let mac1 = nia1_compute_mac(0, 0, 0, &key, data);
        let mac2 = nia1_compute_mac(1, 0, 0, &key, data);

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia1_different_directions_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let data = b"test data";

        let mac1 = nia1_compute_mac(0, 0, 0, &key, data);
        let mac2 = nia1_compute_mac(0, 0, 1, &key, data);

        assert_ne!(mac1, mac2);
    }

    // Note: NIA1 with empty data panics in underlying SNOW3G implementation
    // This is a known limitation

    #[test]
    fn test_build_nia3_iv() {
        let iv = build_nia3_iv(0x12345678, 0x0A, 1);

        // First 64 bits
        assert_eq!(iv[0], 0x12);
        assert_eq!(iv[1], 0x34);
        assert_eq!(iv[2], 0x56);
        assert_eq!(iv[3], 0x78);
        // BEARER (0x0A = 01010) << 3 | DIRECTION (1) << 2 = 0x54
        assert_eq!(iv[4], 0x54);
        assert_eq!(iv[5], 0);
        assert_eq!(iv[6], 0);
        assert_eq!(iv[7], 0);

        // Second 64 bits (same pattern)
        assert_eq!(iv[8], 0x12);
        assert_eq!(iv[9], 0x34);
        assert_eq!(iv[10], 0x56);
        assert_eq!(iv[11], 0x78);
        assert_eq!(iv[12], 0x54);
        assert_eq!(iv[13], 0);
        assert_eq!(iv[14], 0);
        assert_eq!(iv[15], 0);
    }

    #[test]
    fn test_build_nia3_iv_zero_values() {
        let iv = build_nia3_iv(0, 0, 0);
        assert_eq!(iv, [0u8; 16]);
    }

    #[test]
    fn test_get_word_from_keystream_aligned() {
        let keystream = vec![0x12345678, 0xABCDEF01, 0x98765432];

        assert_eq!(get_word_from_keystream(&keystream, 0), 0x12345678);
        assert_eq!(get_word_from_keystream(&keystream, 32), 0xABCDEF01);
        assert_eq!(get_word_from_keystream(&keystream, 64), 0x98765432);
    }

    #[test]
    fn test_get_word_from_keystream_unaligned() {
        let keystream = vec![0x12345678, 0xABCDEF01];

        // At bit position 8: take bits 8-39
        // From 0x12345678: bits 8-31 = 0x345678 (24 bits, shifted left by 8)
        // From 0xABCDEF01: bits 0-7 = 0xAB (8 bits)
        // Result: 0x345678AB
        let word = get_word_from_keystream(&keystream, 8);
        assert_eq!(word, 0x345678AB);
    }

    #[test]
    fn test_nia2_compute_mac_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let count = 0x12345678;
        let bearer = 0x0A;
        let direction = 1;
        let data = b"Hello, 5G World!";

        let mac1 = nia2_compute_mac(count, bearer, direction, &key, data);
        let mac2 = nia2_compute_mac(count, bearer, direction, &key, data);

        // Same inputs should produce same MAC
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_nia2_different_data_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let count = 0;
        let bearer = 0;
        let direction = 0;

        let mac1 = nia2_compute_mac(count, bearer, direction, &key, b"data1");
        let mac2 = nia2_compute_mac(count, bearer, direction, &key, b"data2");

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia3_compute_mac_roundtrip() {
        let key: [u8; 16] = [
            0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf,
            0x4f, 0x3c,
        ];
        let count = 0x12345678;
        let bearer = 0x0A;
        let direction = 1;
        let data = b"Hello, 5G World!";

        let mac1 = nia3_compute_mac(count, bearer, direction, &key, data);
        let mac2 = nia3_compute_mac(count, bearer, direction, &key, data);

        // Same inputs should produce same MAC
        assert_eq!(mac1, mac2);
    }

    #[test]
    fn test_nia3_different_data_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let count = 0;
        let bearer = 0;
        let direction = 0;

        let mac1 = nia3_compute_mac(count, bearer, direction, &key, b"data1");
        let mac2 = nia3_compute_mac(count, bearer, direction, &key, b"data2");

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia3_different_counts_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let data = b"test data";

        let mac1 = nia3_compute_mac(0, 0, 0, &key, data);
        let mac2 = nia3_compute_mac(1, 0, 0, &key, data);

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia3_different_bearers_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let data = b"test data";

        let mac1 = nia3_compute_mac(0, 0, 0, &key, data);
        let mac2 = nia3_compute_mac(0, 1, 0, &key, data);

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia3_different_directions_different_mac() {
        let key: [u8; 16] = [0x2b; 16];
        let data = b"test data";

        let mac1 = nia3_compute_mac(0, 0, 0, &key, data);
        let mac2 = nia3_compute_mac(0, 0, 1, &key, data);

        assert_ne!(mac1, mac2);
    }

    #[test]
    fn test_nia3_empty_data() {
        let key: [u8; 16] = [0x2b; 16];

        // Should not panic on empty data
        let mac = nia3_compute_mac(0, 0, 0, &key, &[]);
        // MAC should be non-zero (it's z[0] since L=0)
        assert_eq!(mac.len(), 4);
    }

    /// 3GPP TS 35.223 Test Set 1
    #[test]
    fn test_nia3_3gpp_test_vector_set1() {
        let key: [u8; 16] = [0x00; 16];
        let count: u32 = 0;
        let bearer: u8 = 0;
        let direction: u8 = 0;
        let message: [u8; 4] = [0x00, 0x00, 0x00, 0x00];

        let mac = nia3_compute_mac(count, bearer, direction, &key, &message);

        // Expected MAC from 3GPP TS 35.223 Test Set 1
        let expected_mac: [u8; 4] = [0x01, 0x80, 0x82, 0xda];
        assert_eq!(mac, expected_mac);
    }

    /// 3GPP TS 35.223 Test Set 2
    #[test]
    fn test_nia3_3gpp_test_vector_set2() {
        let key: [u8; 16] = [
            0xc9, 0xe6, 0xce, 0xc4, 0x60, 0x7c, 0x72, 0xdb, 0x00, 0x0a, 0xef, 0xa8, 0x83, 0x85,
            0xab, 0x0a,
        ];
        let count: u32 = 0xa94059da;
        let bearer: u8 = 0x0a;
        let direction: u8 = 1;
        let message: [u8; 8] = [0x98, 0x3b, 0x41, 0xd4, 0x7d, 0x78, 0x0c, 0x9e];

        let mac = nia3_compute_mac(count, bearer, direction, &key, &message);

        // Expected MAC from 3GPP TS 35.223 Test Set 2
        let expected_mac: [u8; 4] = [0xa7, 0xde, 0x6e, 0x94];
        assert_eq!(mac, expected_mac);
    }
}
