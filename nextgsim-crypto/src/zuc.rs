//! ZUC stream cipher wrapper for 5G security algorithms
//!
//! This module provides wrappers around the `zuc` crate for 5G security:
//! - NEA3 (128-EEA3): ZUC-based confidentiality algorithm
//! - NIA3 (128-EIA3): ZUC-based integrity algorithm (implemented in nia.rs)
//!
//! Reference: 3GPP TS 35.221 (ZUC specification) and TS 35.222 (EEA3/EIA3)

use zuc::ZUC128;

/// Key size in bytes (128 bits)
pub const KEY_SIZE: usize = 16;

/// IV size in bytes (128 bits)
pub const IV_SIZE: usize = 16;

/// NEA3 (128-EEA3) - ZUC-based confidentiality algorithm
///
/// Implements the 5G encryption algorithm NEA3 as specified in 3GPP TS 33.501
/// and TS 35.222. NEA3 uses the ZUC stream cipher.
///
/// # IV Construction (128 bits)
/// The IV is constructed from COUNT, BEARER, and DIRECTION as follows:
/// ```text
/// IV[0..3]   = COUNT[0..3]
/// IV[4]      = BEARER || DIRECTION || 0 (padding)
/// IV[5..7]   = 0
/// IV[8..11]  = COUNT[0..3]
/// IV[12]     = BEARER || DIRECTION || 0 (padding)
/// IV[13..15] = 0
/// ```
///
/// # Parameters
/// - `count`: 32-bit counter value (NAS COUNT or PDCP COUNT)
/// - `bearer`: 5-bit bearer identity (0-31)
/// - `direction`: 1-bit direction (0 = uplink, 1 = downlink)
/// - `key`: 128-bit encryption key (`KNASenc` or `KUPenc`)
/// - `data`: Data to encrypt/decrypt (modified in place)
///
/// # Note
/// Encryption and decryption are the same operation (XOR with keystream).
pub fn nea3_encrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    let iv = build_nea3_iv(count, bearer, direction);
    let mut zuc = ZUC128::new(key, &iv);

    // Generate keystream and XOR with data
    // ZUC generates 32-bit words, so we process 4 bytes at a time
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

/// NEA3 decryption (same as encryption - XOR with keystream)
///
/// See [`nea3_encrypt`] for details.
#[inline]
pub fn nea3_decrypt(count: u32, bearer: u8, direction: u8, key: &[u8; KEY_SIZE], data: &mut [u8]) {
    nea3_encrypt(count, bearer, direction, key, data);
}

/// Build the IV for NEA3 from COUNT, BEARER, and DIRECTION
///
/// IV format (128 bits) per 3GPP TS 35.222:
/// ```text
/// IV[0..3]   = COUNT[0..3]
/// IV[4]      = BEARER || DIRECTION || 0 (padding)
/// IV[5..7]   = 0
/// IV[8..11]  = COUNT[0..3]
/// IV[12]     = BEARER || DIRECTION || 0 (padding)
/// IV[13..15] = 0
/// ```
fn build_nea3_iv(count: u32, bearer: u8, direction: u8) -> [u8; IV_SIZE] {
    let mut iv = [0u8; IV_SIZE];

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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_nea3_iv() {
        let iv = build_nea3_iv(0x12345678, 0x0A, 1);

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
    fn test_build_nea3_iv_zero_values() {
        let iv = build_nea3_iv(0, 0, 0);
        assert_eq!(iv, [0u8; 16]);
    }

    #[test]
    fn test_nea3_encrypt_decrypt_roundtrip() {
        let key: [u8; 16] = [
            0x17, 0x3d, 0x14, 0xba, 0x50, 0x03, 0x73, 0x1d,
            0x7a, 0x60, 0x04, 0x94, 0x70, 0xf0, 0x0a, 0x29,
        ];
        let count: u32 = 0x66035492;
        let bearer: u8 = 0x0f;
        let direction: u8 = 0;

        let original = b"Hello, ZUC NEA3! This is a test message.";
        let mut data = original.to_vec();

        // Encrypt
        nea3_encrypt(count, bearer, direction, &key, &mut data);

        // Data should be different after encryption
        assert_ne!(&data[..], &original[..]);

        // Decrypt
        nea3_decrypt(count, bearer, direction, &key, &mut data);

        // Should match original
        assert_eq!(&data[..], &original[..]);
    }

    #[test]
    fn test_nea3_empty_data() {
        let key: [u8; 16] = [0u8; 16];
        let mut data: Vec<u8> = vec![];

        // Should not panic on empty data
        nea3_encrypt(0, 0, 0, &key, &mut data);
        assert!(data.is_empty());
    }

    #[test]
    fn test_nea3_single_byte() {
        let key: [u8; 16] = [
            0x17, 0x3d, 0x14, 0xba, 0x50, 0x03, 0x73, 0x1d,
            0x7a, 0x60, 0x04, 0x94, 0x70, 0xf0, 0x0a, 0x29,
        ];
        let original = [0x42u8];
        let mut data = original.to_vec();

        nea3_encrypt(0, 0, 0, &key, &mut data);
        assert_ne!(data[0], original[0]);

        nea3_decrypt(0, 0, 0, &key, &mut data);
        assert_eq!(data[0], original[0]);
    }

    /// 3GPP TS 35.222 Test Set 1 for 128-EEA3
    #[test]
    fn test_nea3_3gpp_test_set_1() {
        let key: [u8; 16] = [
            0x17, 0x3d, 0x14, 0xba, 0x50, 0x03, 0x73, 0x1d,
            0x7a, 0x60, 0x04, 0x94, 0x70, 0xf0, 0x0a, 0x29,
        ];
        let count: u32 = 0x66035492;
        let bearer: u8 = 0x0f;
        let direction: u8 = 0;

        // Plaintext (25 bytes = 200 bits)
        let plaintext: [u8; 25] = [
            0x6c, 0xf6, 0x53, 0x40, 0x73, 0x55, 0x52, 0xab,
            0x0c, 0x97, 0x52, 0xfa, 0x6f, 0x90, 0x25, 0xfe,
            0x0b, 0xd6, 0x75, 0xd9, 0x00, 0x58, 0x75, 0xb2,
            0x00,
        ];

        // Expected ciphertext
        let expected_ciphertext: [u8; 25] = [
            0xa6, 0xc8, 0x5f, 0xc6, 0x6a, 0xfb, 0x85, 0x33,
            0xaa, 0xfc, 0x25, 0x18, 0xdf, 0xe7, 0x84, 0x94,
            0x0e, 0xe1, 0xe4, 0xb0, 0x30, 0x23, 0x8c, 0xc8,
            0x10,
        ];

        let mut data = plaintext.to_vec();
        nea3_encrypt(count, bearer, direction, &key, &mut data);

        assert_eq!(&data[..], &expected_ciphertext[..]);

        // Verify decryption
        nea3_decrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &plaintext[..]);
    }


    /// 3GPP TS 35.222 Test Set 2 for 128-EEA3
    #[test]
    fn test_nea3_3gpp_test_set_2() {
        let key: [u8; 16] = [
            0xe5, 0xbd, 0x3e, 0xa0, 0xeb, 0x55, 0xad, 0x8e,
            0x1b, 0x19, 0x9e, 0x3e, 0xc4, 0x31, 0x60, 0x20,
        ];
        let count: u32 = 0x56823;
        let bearer: u8 = 0x18;
        let direction: u8 = 1;

        // Plaintext (90 bytes = 720 bits)
        let plaintext: [u8; 90] = [
            0x14, 0xa8, 0xef, 0x69, 0x3d, 0x67, 0x85, 0x07,
            0xbb, 0xe7, 0x27, 0x0a, 0x7f, 0x67, 0xff, 0x50,
            0x06, 0xc3, 0x52, 0x5b, 0x98, 0x07, 0xe4, 0x67,
            0xc4, 0xe5, 0x60, 0x00, 0xba, 0x33, 0x8f, 0x5d,
            0x42, 0x95, 0x59, 0x03, 0x67, 0x51, 0x82, 0x22,
            0x46, 0xc8, 0x0d, 0x3b, 0x38, 0xf0, 0x7f, 0x4b,
            0xe2, 0xd8, 0xff, 0x58, 0x05, 0xf5, 0x13, 0x22,
            0x29, 0xbd, 0xe9, 0x3b, 0xbb, 0xdc, 0xaf, 0x38,
            0x2b, 0xf1, 0xee, 0x97, 0x2f, 0xbf, 0x99, 0x77,
            0xba, 0xda, 0x89, 0x45, 0x84, 0x7a, 0x2a, 0x6c,
            0x9a, 0xd3, 0x4a, 0x66, 0x75, 0x54, 0xe0, 0x4d,
            0x1f, 0x7f,
        ];

        // Expected ciphertext
        let expected_ciphertext: [u8; 90] = [
            0xf4, 0xbd, 0xcb, 0x5e, 0x8d, 0x02, 0x05, 0xda,
            0x77, 0x10, 0xcc, 0x63, 0x99, 0x5b, 0x6f, 0xa5,
            0xff, 0x8d, 0xd1, 0x18, 0x52, 0x39, 0x32, 0xd2,
            0x80, 0xc1, 0x1a, 0x18, 0xd5, 0xf0, 0x6e, 0x45,
            0x8f, 0x67, 0x49, 0x2c, 0xa2, 0x2a, 0xe5, 0x4e,
            0xe0, 0x25, 0x78, 0x94, 0x12, 0x3a, 0x0d, 0xf6,
            0x1d, 0x78, 0x12, 0xb2, 0x45, 0x0a, 0xc1, 0x85,
            0x48, 0x96, 0x67, 0x97, 0x99, 0x74, 0x86, 0x0d,
            0x6e, 0xdc, 0x13, 0xef, 0xe6, 0xd3, 0xc0, 0xcc,
            0x33, 0xe0, 0x2b, 0xc8, 0x8e, 0x78, 0x40, 0x1a,
            0x32, 0x94, 0x6e, 0x2e, 0x33, 0x30, 0xa7, 0xfd,
            0x3f, 0x94,
        ];

        let mut data = plaintext.to_vec();
        nea3_encrypt(count, bearer, direction, &key, &mut data);

        assert_eq!(&data[..], &expected_ciphertext[..]);

        // Verify decryption
        nea3_decrypt(count, bearer, direction, &key, &mut data);
        assert_eq!(&data[..], &plaintext[..]);
    }

    #[test]
    fn test_nea3_different_counts_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea3_encrypt(0, 0, 0, &key, &mut data1);
        nea3_encrypt(1, 0, 0, &key, &mut data2);

        // Different counts should produce different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea3_different_bearers_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea3_encrypt(0, 0, 0, &key, &mut data1);
        nea3_encrypt(0, 1, 0, &key, &mut data2);

        // Different bearers should produce different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea3_different_directions_produce_different_output() {
        let key: [u8; 16] = [0x2b; 16];
        let plaintext = [0x00u8; 16];

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea3_encrypt(0, 0, 0, &key, &mut data1);
        nea3_encrypt(0, 0, 1, &key, &mut data2);

        // Different directions should produce different ciphertext
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_nea3_deterministic() {
        let key: [u8; 16] = [
            0x17, 0x3d, 0x14, 0xba, 0x50, 0x03, 0x73, 0x1d,
            0x7a, 0x60, 0x04, 0x94, 0x70, 0xf0, 0x0a, 0x29,
        ];
        let plaintext = b"Test message for determinism check";

        let mut data1 = plaintext.to_vec();
        let mut data2 = plaintext.to_vec();

        nea3_encrypt(0x12345678, 5, 1, &key, &mut data1);
        nea3_encrypt(0x12345678, 5, 1, &key, &mut data2);

        assert_eq!(data1, data2);
    }

    #[test]
    fn test_nea3_large_data() {
        let key: [u8; 16] = [
            0x17, 0x3d, 0x14, 0xba, 0x50, 0x03, 0x73, 0x1d,
            0x7a, 0x60, 0x04, 0x94, 0x70, 0xf0, 0x0a, 0x29,
        ];

        // 1KB of data
        let original: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();
        let mut data = original.clone();

        nea3_encrypt(0xABCDEF01, 15, 1, &key, &mut data);
        assert_ne!(data, original);

        nea3_decrypt(0xABCDEF01, 15, 1, &key, &mut data);
        assert_eq!(data, original);
    }
}
