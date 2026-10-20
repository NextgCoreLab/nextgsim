//! The EPS key hierarchy (TS 33.401 Annex A), shared by the nextgsim LTE UE
//! (its USIM/ME) and the eNB.
//!
//! Every derivation is the TS 33.220 KDF -- HMAC-SHA256 over
//! `S = FC || P0 || L0 || ...` -- with an FC from the EPS range 0x10-0x1F
//! (Annex A.1.2). This module only builds `S`; [`calculate_kdf_key`] is the
//! KDF itself, the same one the 5G hierarchy in [`crate::kdf`] uses.
//!
//! | Key | Clause | Input key | FC | Parameters |
//! |---|---|---|---|---|
//! | `K_ASME` | A.2 | `CK \|\| IK` | 0x10 | SN id, SQN ⊕ AK |
//! | `K_eNB` | A.3 | `K_ASME` | 0x11 | uplink NAS COUNT |
//! | NH | A.4 | `K_ASME` | 0x12 | SYNC-input (`K_eNB`, then the previous NH) |
//! | algorithm keys | A.7 | `K_ASME` (NAS) or `K_eNB` (RRC, UP) | 0x15 | type distinguisher, algorithm identity |
//!
//! The eNB derives only from `K_eNB` (A.7 RRC/UP keys); it never sees
//! `K_ASME`, which is why the A.7 function takes the input key as a parameter
//! rather than a NAS security context.

use crate::kdf::{calculate_kdf_key, KEY_128_SIZE, KEY_256_SIZE};

/// FC for `K_ASME` (A.2).
pub const FC_KASME: u8 = 0x10;
/// FC for `K_eNB` (A.3).
pub const FC_KENB: u8 = 0x11;
/// FC for NH (A.4).
pub const FC_NH: u8 = 0x12;
/// FC for the algorithm keys (A.7).
pub const FC_ALG_KEY: u8 = 0x15;

/// The algorithm type distinguishers of Table A.7-1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EpsAlgorithmType {
    /// NAS-enc-alg.
    NasEnc = 0x01,
    /// NAS-int-alg.
    NasInt = 0x02,
    /// RRC-enc-alg.
    RrcEnc = 0x03,
    /// RRC-int-alg.
    RrcInt = 0x04,
    /// UP-enc-alg.
    UpEnc = 0x05,
    /// UP-int-alg.
    UpInt = 0x06,
}

/// `K_ASME` from CK, IK, the serving network identity and SQN ⊕ AK (A.2).
///
/// `sn_id` is the MCC/MNC octet string of Figure A.2-1 -- the TS 24.301 PLMN
/// encoding, which is what `nextgsim_common::Plmn::encode` produces. `sqn_xor_ak`
/// is the first six octets of AUTN as received; A.2 uses it as sent, not the
/// recovered SQN.
pub fn derive_kasme(
    ck: &[u8; 16],
    ik: &[u8; 16],
    sn_id: &[u8; 3],
    sqn_xor_ak: &[u8; 6],
) -> [u8; KEY_256_SIZE] {
    let mut key = [0u8; KEY_256_SIZE];
    key[..16].copy_from_slice(ck);
    key[16..].copy_from_slice(ik);
    calculate_kdf_key(&key, FC_KASME, &[sn_id, sqn_xor_ak])
}

/// `K_eNB` from `K_ASME` and the uplink NAS COUNT (A.3).
///
/// Which COUNT is §7.2.6.2's rule, not this function's: after a NAS Security
/// Mode Command it is the uplink NAS COUNT of the most recent SECURITY MODE
/// COMPLETE; otherwise that of the NAS message that triggered the AS SMC.
pub fn derive_kenb(kasme: &[u8; KEY_256_SIZE], ul_nas_count: u32) -> [u8; KEY_256_SIZE] {
    calculate_kdf_key(kasme, FC_KENB, &[&ul_nas_count.to_be_bytes()])
}

/// NH from `K_ASME` and the SYNC-input (A.4): the fresh `K_eNB` for the first
/// NH, the previous NH after that.
pub fn derive_nh(
    kasme: &[u8; KEY_256_SIZE],
    sync_input: &[u8; KEY_256_SIZE],
) -> [u8; KEY_256_SIZE] {
    calculate_kdf_key(kasme, FC_NH, &[sync_input])
}

/// A 128-bit algorithm key (A.7): the 128 least significant bits of
/// `KDF(input_key, 0x15, type, algorithm identity)`.
///
/// `input_key` is `K_ASME` for the NAS keys and `K_eNB` for the RRC and UP
/// keys. `algorithm_id` is the 4-bit identity of TS 33.401 §5 (EEA0..EEA7,
/// EIA0..EIA7); A.7 puts it in the low nibble with the high nibble zero, so a
/// value above 15 cannot be encoded and is masked rather than silently widened.
pub fn derive_algorithm_key(
    input_key: &[u8; KEY_256_SIZE],
    algorithm_type: EpsAlgorithmType,
    algorithm_id: u8,
) -> [u8; KEY_128_SIZE] {
    let out = calculate_kdf_key(
        input_key,
        FC_ALG_KEY,
        &[&[algorithm_type as u8], &[algorithm_id & 0x0F]],
    );
    let mut key = [0u8; KEY_128_SIZE];
    key.copy_from_slice(&out[KEY_256_SIZE - KEY_128_SIZE..]);
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    // Golden values from an independent HMAC-SHA256 (Python's `hmac` /
    // `hashlib`) over the Annex A input strings, not from this code.
    fn hex32(s: &str) -> [u8; 32] {
        let v: Vec<u8> = (0..32)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        v.try_into().unwrap()
    }

    fn hex16(s: &str) -> [u8; 16] {
        let v: Vec<u8> = (0..16)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        v.try_into().unwrap()
    }

    fn kasme() -> [u8; 32] {
        let ck: [u8; 16] = core::array::from_fn(|i| i as u8);
        let ik: [u8; 16] = core::array::from_fn(|i| 0x10 + i as u8);
        // MCC 999 / MNC 70: 0x99, 0xF9, 0x07 (Figure A.2-1).
        derive_kasme(&ck, &ik, &[0x99, 0xF9, 0x07], &[1, 2, 3, 4, 5, 6])
    }

    #[test]
    fn kasme_matches_an_independent_hmac() {
        assert_eq!(
            kasme(),
            hex32("6ea6653ec5d254ef51ab02235924ea348f3fa229848fa2eb907e5ce02a563d8d")
        );
    }

    #[test]
    fn kenb_uses_the_count_as_four_big_endian_octets() {
        assert_eq!(
            derive_kenb(&kasme(), 0),
            hex32("bad173c5d1e11c7f63cb52faefe2b03f0a78f79b41bfbd692b9ec1dad8948d0f")
        );
        assert_eq!(
            derive_kenb(&kasme(), 0x0102_0304),
            hex32("6518ead8d9dea2c6c6e53a6504cfa435ce462ff9dec97f7bab892298e52cac45")
        );
    }

    #[test]
    fn nh_chains_from_kenb() {
        let kenb = derive_kenb(&kasme(), 0);
        assert_eq!(
            derive_nh(&kasme(), &kenb),
            hex32("e6e09ba546742ba45761fb32b666766fb4df2eb919cc7a8c9deadd00f501baf8")
        );
    }

    #[test]
    fn algorithm_keys_are_the_low_128_bits() {
        let k = kasme();
        let kenb = derive_kenb(&k, 0);
        assert_eq!(
            derive_algorithm_key(&k, EpsAlgorithmType::NasInt, 2),
            hex16("5fc59a2babb79d0e7c8930a662ee4bf2")
        );
        assert_eq!(
            derive_algorithm_key(&k, EpsAlgorithmType::NasEnc, 1),
            hex16("24f612e47508638e1d2fe7964807e4ee")
        );
        assert_eq!(
            derive_algorithm_key(&kenb, EpsAlgorithmType::RrcInt, 2),
            hex16("dc0fec7d8b1206e2dd023673533930f2")
        );
        assert_eq!(
            derive_algorithm_key(&kenb, EpsAlgorithmType::RrcEnc, 2),
            hex16("1a4501205554423c21aac5a857ec547a")
        );
        assert_eq!(
            derive_algorithm_key(&kenb, EpsAlgorithmType::UpEnc, 2),
            hex16("e8f36aa255b2c7e27b72383314e1b2f8")
        );
    }
}
