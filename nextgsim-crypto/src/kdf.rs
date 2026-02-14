//! Key derivation functions for 5G security
//!
//! Implements key derivation functions as specified in 3GPP TS 33.501.
//! These functions are used to derive various keys in the 5G security architecture:
//! - KAUSF: Key for AUSF (Authentication Server Function)
//! - KSEAF: Key for SEAF (Security Anchor Function)
//! - KAMF: Key for AMF (Access and Mobility Management Function)
//! - KNASint/KNASenc: Keys for NAS integrity and encryption
//! - `KgNB`: Key for gNB (Next Generation Node B)

use hmac::{Hmac, Mac};
use sha2::Sha256;
use unicode_normalization::UnicodeNormalization;

/// HMAC-SHA256 output size in bytes
pub const HMAC_SHA256_SIZE: usize = 32;

/// Key size for 256-bit keys
pub const KEY_256_SIZE: usize = 32;

/// Key size for 128-bit keys
pub const KEY_128_SIZE: usize = 16;

/// FC values for key derivation as defined in 3GPP TS 33.501 Annex A
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FcValue {
    /// FC = 0x6A: Derivation of KAUSF from CK and IK
    Kausf = 0x6A,
    /// FC = 0x6C: Derivation of KSEAF from KAUSF
    Kseaf = 0x6C,
    /// FC = 0x6D: Derivation of KAMF from KSEAF
    Kamf = 0x6D,
    /// FC = 0x69: Derivation of `KNASint` and `KNASenc` from KAMF
    KnasIntEnc = 0x69,
    /// FC = 0x6E: Derivation of `KgNB` from KAMF
    Kgnb = 0x6E,
    /// FC = 0x6B: Derivation of RES* from CK' and IK'
    ResStar = 0x6B,
    /// FC = 0x70: Derivation of NH (Next Hop) from KAMF and sync input
    Nh = 0x70,
}

/// Algorithm type distinguisher for NAS key derivation (TS 33.501 A.8)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AlgorithmTypeDistinguisher {
    /// NAS encryption algorithm
    NasEnc = 0x01,
    /// NAS integrity algorithm
    NasInt = 0x02,
    /// RRC encryption algorithm
    RrcEnc = 0x03,
    /// RRC integrity algorithm
    RrcInt = 0x04,
    /// UP encryption algorithm
    UpEnc = 0x05,
    /// UP integrity algorithm
    UpInt = 0x06,
}


/// Compute HMAC-SHA256
///
/// # Arguments
/// * `key` - The HMAC key
/// * `input` - The input data to authenticate
///
/// # Returns
/// 32-byte HMAC-SHA256 output
pub fn hmac_sha256(key: &[u8], input: &[u8]) -> [u8; HMAC_SHA256_SIZE] {
    // HMAC-SHA256 accepts keys of any size, so this should never fail
    let mut mac = Hmac::<Sha256>::new_from_slice(key)
        .unwrap_or_else(|_| unreachable!("HMAC-SHA256 accepts keys of any size"));
    mac.update(input);
    let result = mac.finalize();
    let mut output = [0u8; HMAC_SHA256_SIZE];
    output.copy_from_slice(&result.into_bytes());
    output
}

/// Calculate KDF key using HMAC-SHA256 as specified in 3GPP TS 33.220
///
/// The input string S is constructed as:
/// S = FC || P0 || L0 || P1 || L1 || ... || Pn || Ln
///
/// Where:
/// - FC is a single octet function code
/// - Pi are the input parameters
/// - Li are the lengths of Pi encoded as 2 octets (big-endian)
///
/// # Arguments
/// * `key` - The 256-bit key
/// * `fc` - Function code
/// * `parameters` - Slice of input parameters
///
/// # Returns
/// 32-byte derived key
pub fn calculate_kdf_key(key: &[u8; KEY_256_SIZE], fc: u8, parameters: &[&[u8]]) -> [u8; KEY_256_SIZE] {
    let mut input = Vec::new();
    input.push(fc);

    for param in parameters {
        input.extend_from_slice(param);
        // Length encoded as 2 octets (big-endian)
        let len = param.len() as u16;
        input.push((len >> 8) as u8);
        input.push((len & 0xff) as u8);
    }

    hmac_sha256(key, &input)
}


/// Calculate PRF' (Pseudo-Random Function Prime) as specified in 3GPP TS 33.501 Annex B
///
/// PRF' is used for key expansion when the output length exceeds 256 bits.
/// It uses HMAC-SHA256 in a counter mode construction.
///
/// # Arguments
/// * `key` - 256-bit key
/// * `input` - Input data
/// * `output_length` - Desired output length in bytes
///
/// # Returns
/// Derived key material of the requested length
///
/// # Panics
/// Panics if `output_length` would require more than 254 rounds
pub fn calculate_prf_prime(key: &[u8; KEY_256_SIZE], input: &[u8], output_length: usize) -> Vec<u8> {
    let round = output_length.div_ceil(32); // Ceiling division
    assert!(round > 0 && round <= 254, "Invalid output_length for PRF'");

    let mut t_values: Vec<[u8; HMAC_SHA256_SIZE]> = Vec::with_capacity(round);

    for i in 0..round {
        let mut s = Vec::new();
        if i > 0 {
            s.extend_from_slice(&t_values[i - 1]);
        }
        s.extend_from_slice(input);
        s.push((i + 1) as u8);

        t_values.push(hmac_sha256(key, &s));
    }

    let mut result = Vec::with_capacity(round * HMAC_SHA256_SIZE);
    for t in &t_values {
        result.extend_from_slice(t);
    }
    result.truncate(output_length);
    result
}

/// Derive KAUSF from CK and IK (3GPP TS 33.501 Annex A.2)
///
/// KAUSF = KDF(CK || IK, FC, SN name, SQN ⊕ AK)
///
/// # Arguments
/// * `ck` - 128-bit Cipher Key from AKA
/// * `ik` - 128-bit Integrity Key from AKA
/// * `sn_name` - Serving Network Name (e.g., "5G:mnc001.mcc001.3gppnetwork.org")
/// * `sqn_xor_ak` - SQN XOR AK (6 bytes)
///
/// # Returns
/// 256-bit KAUSF
pub fn derive_kausf(
    ck: &[u8; KEY_128_SIZE],
    ik: &[u8; KEY_128_SIZE],
    sn_name: &[u8],
    sqn_xor_ak: &[u8; 6],
) -> [u8; KEY_256_SIZE] {
    // Concatenate CK || IK to form the 256-bit key
    let mut key = [0u8; KEY_256_SIZE];
    key[..KEY_128_SIZE].copy_from_slice(ck);
    key[KEY_128_SIZE..].copy_from_slice(ik);

    calculate_kdf_key(&key, FcValue::Kausf as u8, &[sn_name, sqn_xor_ak])
}


/// Derive KSEAF from KAUSF (3GPP TS 33.501 Annex A.6)
///
/// KSEAF = KDF(KAUSF, FC, SN name)
///
/// # Arguments
/// * `kausf` - 256-bit KAUSF
/// * `sn_name` - Serving Network Name
///
/// # Returns
/// 256-bit KSEAF
pub fn derive_kseaf(kausf: &[u8; KEY_256_SIZE], sn_name: &[u8]) -> [u8; KEY_256_SIZE] {
    calculate_kdf_key(kausf, FcValue::Kseaf as u8, &[sn_name])
}

/// Derive KAMF from KSEAF (3GPP TS 33.501 Annex A.7)
///
/// KAMF = KDF(KSEAF, FC, SUPI, ABBA)
///
/// # Arguments
/// * `kseaf` - 256-bit KSEAF
/// * `supi` - Subscription Permanent Identifier (as bytes)
/// * `abba` - Anti-Bidding down Between Architectures parameter
///
/// # Returns
/// 256-bit KAMF
pub fn derive_kamf(kseaf: &[u8; KEY_256_SIZE], supi: &[u8], abba: &[u8]) -> [u8; KEY_256_SIZE] {
    calculate_kdf_key(kseaf, FcValue::Kamf as u8, &[supi, abba])
}

/// Derive NAS keys (`KNASint` and `KNASenc`) from KAMF (3GPP TS 33.501 Annex A.8)
///
/// KNASint/KNASenc = KDF(KAMF, FC, algorithm type distinguisher, algorithm identity)
///
/// # Arguments
/// * `kamf` - 256-bit KAMF
/// * `algorithm_type` - Algorithm type distinguisher (NAS enc or NAS int)
/// * `algorithm_id` - Algorithm identity (0-7)
///
/// # Returns
/// 128-bit NAS key (lower 128 bits of the KDF output)
pub fn derive_nas_key(
    kamf: &[u8; KEY_256_SIZE],
    algorithm_type: AlgorithmTypeDistinguisher,
    algorithm_id: u8,
) -> [u8; KEY_128_SIZE] {
    let type_byte = [algorithm_type as u8];
    let id_byte = [algorithm_id];

    let kdf_output = calculate_kdf_key(kamf, FcValue::KnasIntEnc as u8, &[&type_byte, &id_byte]);

    // Take the least significant 128 bits (last 16 bytes)
    let mut key = [0u8; KEY_128_SIZE];
    key.copy_from_slice(&kdf_output[KEY_128_SIZE..]);
    key
}

/// Derive `KNASenc` (NAS encryption key) from KAMF
///
/// # Arguments
/// * `kamf` - 256-bit KAMF
/// * `algorithm_id` - NAS encryption algorithm identity (0=NULL, 1=NEA1, 2=NEA2, 3=NEA3)
///
/// # Returns
/// 128-bit `KNASenc`
pub fn derive_knas_enc(kamf: &[u8; KEY_256_SIZE], algorithm_id: u8) -> [u8; KEY_128_SIZE] {
    derive_nas_key(kamf, AlgorithmTypeDistinguisher::NasEnc, algorithm_id)
}

/// Derive `KNASint` (NAS integrity key) from KAMF
///
/// # Arguments
/// * `kamf` - 256-bit KAMF
/// * `algorithm_id` - NAS integrity algorithm identity (0=NULL, 1=NIA1, 2=NIA2, 3=NIA3)
///
/// # Returns
/// 128-bit `KNASint`
pub fn derive_knas_int(kamf: &[u8; KEY_256_SIZE], algorithm_id: u8) -> [u8; KEY_128_SIZE] {
    derive_nas_key(kamf, AlgorithmTypeDistinguisher::NasInt, algorithm_id)
}


/// Derive `KgNB` from KAMF (3GPP TS 33.501 Annex A.9)
///
/// `KgNB` = KDF(KAMF, FC, uplink NAS COUNT, access type distinguisher)
///
/// # Arguments
/// * `kamf` - 256-bit KAMF
/// * `uplink_nas_count` - Uplink NAS COUNT (4 bytes)
/// * `access_type` - Access type distinguisher (0x01 for 3GPP access, 0x02 for non-3GPP access)
///
/// # Returns
/// 256-bit `KgNB`
pub fn derive_kgnb(
    kamf: &[u8; KEY_256_SIZE],
    uplink_nas_count: u32,
    access_type: u8,
) -> [u8; KEY_256_SIZE] {
    let nas_count_bytes = uplink_nas_count.to_be_bytes();
    let access_type_byte = [access_type];

    calculate_kdf_key(kamf, FcValue::Kgnb as u8, &[&nas_count_bytes, &access_type_byte])
}

/// Derive RRC/UP keys from `KgNB` (3GPP TS 33.501 Annex A.8)
///
/// # Arguments
/// * `kgnb` - 256-bit `KgNB`
/// * `algorithm_type` - Algorithm type distinguisher
/// * `algorithm_id` - Algorithm identity
///
/// # Returns
/// 128-bit derived key
pub fn derive_rrc_up_key(
    kgnb: &[u8; KEY_256_SIZE],
    algorithm_type: AlgorithmTypeDistinguisher,
    algorithm_id: u8,
) -> [u8; KEY_128_SIZE] {
    let type_byte = [algorithm_type as u8];
    let id_byte = [algorithm_id];

    let kdf_output = calculate_kdf_key(kgnb, FcValue::KnasIntEnc as u8, &[&type_byte, &id_byte]);

    // Take the least significant 128 bits (last 16 bytes)
    let mut key = [0u8; KEY_128_SIZE];
    key.copy_from_slice(&kdf_output[KEY_128_SIZE..]);
    key
}

/// Derive NH (Next Hop) from KAMF (3GPP TS 33.501 Annex A.10)
///
/// NH = KDF(KAMF, FC, Sync-Input)
///
/// # Arguments
/// * `kamf` - 256-bit KAMF
/// * `sync_input` - Synchronization input (`KgNB` or previous NH, 256 bits)
///
/// # Returns
/// 256-bit NH
pub fn derive_nh(kamf: &[u8; KEY_256_SIZE], sync_input: &[u8; KEY_256_SIZE]) -> [u8; KEY_256_SIZE] {
    calculate_kdf_key(kamf, FcValue::Nh as u8, &[sync_input])
}

/// Derive RES* from CK' and IK' (3GPP TS 33.501 Annex A.4)
///
/// RES* = KDF(CK' || IK', FC, SN name, RAND, RES)
///
/// # Arguments
/// * `ck_prime` - 128-bit CK'
/// * `ik_prime` - 128-bit IK'
/// * `sn_name` - Serving Network Name
/// * `rand` - RAND (16 bytes)
/// * `res` - RES from AKA (variable length, typically 8-16 bytes)
///
/// # Returns
/// 128-bit RES* (lower 128 bits of the KDF output)
pub fn derive_res_star(
    ck_prime: &[u8; KEY_128_SIZE],
    ik_prime: &[u8; KEY_128_SIZE],
    sn_name: &[u8],
    rand: &[u8; KEY_128_SIZE],
    res: &[u8],
) -> [u8; KEY_128_SIZE] {
    // Concatenate CK' || IK' to form the 256-bit key
    let mut key = [0u8; KEY_256_SIZE];
    key[..KEY_128_SIZE].copy_from_slice(ck_prime);
    key[KEY_128_SIZE..].copy_from_slice(ik_prime);

    let kdf_output = calculate_kdf_key(&key, FcValue::ResStar as u8, &[sn_name, rand, res]);

    // Take the least significant 128 bits (last 16 bytes)
    let mut result = [0u8; KEY_128_SIZE];
    result.copy_from_slice(&kdf_output[KEY_128_SIZE..]);
    result
}


/// Encode a string for KDF input as specified in 3GPP TS 33.501 Annex B.2.1.2
///
/// Character strings are first normalized using NFKC (Normalization Form
/// Compatibility Composition) and then encoded to octet strings according
/// to UTF-8 encoding rules.
pub fn encode_kdf_string(s: &str) -> Vec<u8> {
    let normalized: String = s.nfkc().collect();
    normalized.into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmac_sha256() {
        // RFC 4231 Test Case 1
        let key = [0x0b; 20];
        let data = b"Hi There";
        let expected: [u8; 32] = [
            0xb0, 0x34, 0x4c, 0x61, 0xd8, 0xdb, 0x38, 0x53,
            0x5c, 0xa8, 0xaf, 0xce, 0xaf, 0x0b, 0xf1, 0x2b,
            0x88, 0x1d, 0xc2, 0x00, 0xc9, 0x83, 0x3d, 0xa7,
            0x26, 0xe9, 0x37, 0x6c, 0x2e, 0x32, 0xcf, 0xf7,
        ];

        let result = hmac_sha256(&key, data);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_hmac_sha256_rfc4231_case2() {
        // RFC 4231 Test Case 2 - Key = "Jefe"
        let key = b"Jefe";
        let data = b"what do ya want for nothing?";
        let expected: [u8; 32] = [
            0x5b, 0xdc, 0xc1, 0x46, 0xbf, 0x60, 0x75, 0x4e,
            0x6a, 0x04, 0x24, 0x26, 0x08, 0x95, 0x75, 0xc7,
            0x5a, 0x00, 0x3f, 0x08, 0x9d, 0x27, 0x39, 0x83,
            0x9d, 0xec, 0x58, 0xb9, 0x64, 0xec, 0x38, 0x43,
        ];

        let result = hmac_sha256(key, data);
        assert_eq!(result, expected);
    }


    #[test]
    fn test_calculate_kdf_key_structure() {
        // Test that the KDF input is constructed correctly
        // S = FC || P0 || L0 || P1 || L1
        let key = [0u8; 32];
        let fc = 0x6C; // KSEAF
        let param1 = b"test";

        let result = calculate_kdf_key(&key, fc, &[param1]);

        // The result should be deterministic for the same inputs
        let result2 = calculate_kdf_key(&key, fc, &[param1]);
        assert_eq!(result, result2);

        // Different FC should produce different result
        let result3 = calculate_kdf_key(&key, 0x6D, &[param1]);
        assert_ne!(result, result3);
    }

    #[test]
    fn test_calculate_prf_prime_32_bytes() {
        // Test PRF' with 32-byte output (single round)
        let key = [0x01u8; 32];
        let input = b"test input";

        let result = calculate_prf_prime(&key, input, 32);
        assert_eq!(result.len(), 32);

        // Should be deterministic
        let result2 = calculate_prf_prime(&key, input, 32);
        assert_eq!(result, result2);
    }

    #[test]
    fn test_calculate_prf_prime_64_bytes() {
        // Test PRF' with 64-byte output (two rounds)
        let key = [0x02u8; 32];
        let input = b"test input for longer output";

        let result = calculate_prf_prime(&key, input, 64);
        assert_eq!(result.len(), 64);
    }

    #[test]
    fn test_calculate_prf_prime_48_bytes() {
        // Test PRF' with non-aligned output
        let key = [0x03u8; 32];
        let input = b"test";

        let result = calculate_prf_prime(&key, input, 48);
        assert_eq!(result.len(), 48);
    }

    #[test]
    fn test_derive_kausf() {
        // Test KAUSF derivation
        let ck = [0x11u8; 16];
        let ik = [0x22u8; 16];
        let sn_name = b"5G:mnc001.mcc001.3gppnetwork.org";
        let sqn_xor_ak = [0x00, 0x00, 0x00, 0x00, 0x00, 0x01];

        let kausf = derive_kausf(&ck, &ik, sn_name, &sqn_xor_ak);
        assert_eq!(kausf.len(), 32);

        // Should be deterministic
        let kausf2 = derive_kausf(&ck, &ik, sn_name, &sqn_xor_ak);
        assert_eq!(kausf, kausf2);
    }


    #[test]
    fn test_derive_kseaf() {
        let kausf = [0x33u8; 32];
        let sn_name = b"5G:mnc001.mcc001.3gppnetwork.org";

        let kseaf = derive_kseaf(&kausf, sn_name);
        assert_eq!(kseaf.len(), 32);

        // Different SN name should produce different KSEAF
        let kseaf2 = derive_kseaf(&kausf, b"5G:mnc002.mcc002.3gppnetwork.org");
        assert_ne!(kseaf, kseaf2);
    }

    #[test]
    fn test_derive_kamf() {
        let kseaf = [0x44u8; 32];
        let supi = b"imsi-001010000000001";
        let abba = [0x00, 0x00];

        let kamf = derive_kamf(&kseaf, supi, &abba);
        assert_eq!(kamf.len(), 32);

        // Different SUPI should produce different KAMF
        let kamf2 = derive_kamf(&kseaf, b"imsi-001010000000002", &abba);
        assert_ne!(kamf, kamf2);
    }

    #[test]
    fn test_derive_knas_enc() {
        let kamf = [0x55u8; 32];

        // Test with different algorithm IDs
        let knas_enc_null = derive_knas_enc(&kamf, 0);
        let knas_enc_nea1 = derive_knas_enc(&kamf, 1);
        let knas_enc_nea2 = derive_knas_enc(&kamf, 2);
        let knas_enc_nea3 = derive_knas_enc(&kamf, 3);

        assert_eq!(knas_enc_null.len(), 16);
        assert_eq!(knas_enc_nea1.len(), 16);
        assert_eq!(knas_enc_nea2.len(), 16);
        assert_eq!(knas_enc_nea3.len(), 16);

        // Different algorithm IDs should produce different keys
        assert_ne!(knas_enc_null, knas_enc_nea1);
        assert_ne!(knas_enc_nea1, knas_enc_nea2);
        assert_ne!(knas_enc_nea2, knas_enc_nea3);
    }

    #[test]
    fn test_derive_knas_int() {
        let kamf = [0x66u8; 32];

        let knas_int_null = derive_knas_int(&kamf, 0);
        let knas_int_nia1 = derive_knas_int(&kamf, 1);
        let knas_int_nia2 = derive_knas_int(&kamf, 2);

        assert_eq!(knas_int_null.len(), 16);
        assert_eq!(knas_int_nia1.len(), 16);
        assert_eq!(knas_int_nia2.len(), 16);

        // Different algorithm IDs should produce different keys
        assert_ne!(knas_int_null, knas_int_nia1);
        assert_ne!(knas_int_nia1, knas_int_nia2);
    }

    #[test]
    fn test_derive_knas_enc_vs_int() {
        // Encryption and integrity keys should be different even with same algorithm ID
        let kamf = [0x77u8; 32];

        let knas_enc = derive_knas_enc(&kamf, 2);
        let knas_int = derive_knas_int(&kamf, 2);

        assert_ne!(knas_enc, knas_int);
    }


    #[test]
    fn test_derive_kgnb() {
        let kamf = [0x88u8; 32];
        let uplink_nas_count = 0x00000001u32;
        let access_type_3gpp = 0x01u8;

        let kgnb = derive_kgnb(&kamf, uplink_nas_count, access_type_3gpp);
        assert_eq!(kgnb.len(), 32);

        // Different NAS count should produce different KgNB
        let kgnb2 = derive_kgnb(&kamf, 0x00000002, access_type_3gpp);
        assert_ne!(kgnb, kgnb2);

        // Different access type should produce different KgNB
        let kgnb3 = derive_kgnb(&kamf, uplink_nas_count, 0x02);
        assert_ne!(kgnb, kgnb3);
    }

    #[test]
    fn test_derive_rrc_up_key() {
        let kgnb = [0x99u8; 32];

        let krrc_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcEnc, 2);
        let krrc_int = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcInt, 2);
        let kup_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpEnc, 2);
        let kup_int = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpInt, 2);

        assert_eq!(krrc_enc.len(), 16);
        assert_eq!(krrc_int.len(), 16);
        assert_eq!(kup_enc.len(), 16);
        assert_eq!(kup_int.len(), 16);

        // All keys should be different
        assert_ne!(krrc_enc, krrc_int);
        assert_ne!(krrc_enc, kup_enc);
        assert_ne!(krrc_enc, kup_int);
        assert_ne!(krrc_int, kup_enc);
        assert_ne!(krrc_int, kup_int);
        assert_ne!(kup_enc, kup_int);
    }

    #[test]
    fn test_derive_nh() {
        let kamf = [0xAAu8; 32];
        let kgnb = [0xBBu8; 32];

        let nh = derive_nh(&kamf, &kgnb);
        assert_eq!(nh.len(), 32);

        // NH should be different from both inputs
        assert_ne!(nh, kamf);
        assert_ne!(nh, kgnb);

        // Chained NH derivation
        let nh2 = derive_nh(&kamf, &nh);
        assert_ne!(nh, nh2);
    }

    #[test]
    fn test_derive_res_star() {
        let ck_prime = [0xCCu8; 16];
        let ik_prime = [0xDDu8; 16];
        let sn_name = b"5G:mnc001.mcc001.3gppnetwork.org";
        let rand = [0xEEu8; 16];
        let res = [0xFFu8; 8];

        let res_star = derive_res_star(&ck_prime, &ik_prime, sn_name, &rand, &res);
        assert_eq!(res_star.len(), 16);

        // Different RES should produce different RES*
        let res2 = [0x00u8; 8];
        let res_star2 = derive_res_star(&ck_prime, &ik_prime, sn_name, &rand, &res2);
        assert_ne!(res_star, res_star2);
    }

    #[test]
    fn test_encode_kdf_string() {
        let s = "5G:mnc001.mcc001.3gppnetwork.org";
        let encoded = encode_kdf_string(s);
        assert_eq!(encoded, s.as_bytes());
    }

    #[test]
    fn test_encode_kdf_string_nfkc_normalization() {
        // NFKC should decompose compatibility characters
        // U+2126 (OHM SIGN) -> U+03A9 (GREEK CAPITAL LETTER OMEGA)
        let s_ohm = "\u{2126}";
        let encoded = encode_kdf_string(s_ohm);
        let expected: Vec<u8> = "\u{03A9}".as_bytes().to_vec();
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_encode_kdf_string_nfkc_fi_ligature() {
        // U+FB01 (LATIN SMALL LIGATURE FI) -> "fi"
        let s = "\u{FB01}";
        let encoded = encode_kdf_string(s);
        assert_eq!(encoded, b"fi");
    }

    #[test]
    fn test_encode_kdf_string_ascii_unchanged() {
        // Pure ASCII should pass through unchanged
        let s = "hello world 12345";
        let encoded = encode_kdf_string(s);
        assert_eq!(encoded, s.as_bytes());
    }

    #[test]
    fn test_full_key_derivation_chain() {
        // Test the full key derivation chain: CK/IK -> KAUSF -> KSEAF -> KAMF -> KNASenc/KNASint -> KgNB
        let ck = [0x01u8; 16];
        let ik = [0x02u8; 16];
        let sn_name = b"5G:mnc001.mcc001.3gppnetwork.org";
        let sqn_xor_ak = [0x00, 0x00, 0x00, 0x00, 0x00, 0x01];
        let supi = b"imsi-001010000000001";
        let abba = [0x00, 0x00];

        // Derive KAUSF
        let kausf = derive_kausf(&ck, &ik, sn_name, &sqn_xor_ak);

        // Derive KSEAF
        let kseaf = derive_kseaf(&kausf, sn_name);

        // Derive KAMF
        let kamf = derive_kamf(&kseaf, supi, &abba);

        // Derive NAS keys
        let knas_enc = derive_knas_enc(&kamf, 2); // NEA2
        let knas_int = derive_knas_int(&kamf, 2); // NIA2

        // Derive KgNB
        let kgnb = derive_kgnb(&kamf, 0, 0x01);

        // All keys should be valid (non-zero)
        assert!(kausf.iter().any(|&b| b != 0));
        assert!(kseaf.iter().any(|&b| b != 0));
        assert!(kamf.iter().any(|&b| b != 0));
        assert!(knas_enc.iter().any(|&b| b != 0));
        assert!(knas_int.iter().any(|&b| b != 0));
        assert!(kgnb.iter().any(|&b| b != 0));
    }
}
