//! EAP-AKA' key derivation (ngsc-02)
//!
//! Implements the EAP-AKA' key hierarchy used for 5G primary authentication over
//! the EAP framework (TS 33.501 §6.1.3.1). The two derivations are:
//!
//! 1. **CK'/IK' transform** — RFC 9048 §3.3 / TS 33.402 Annex A.2: the AKA cipher
//!    and integrity keys are bound to the serving/access network name:
//!    `CK' || IK' = KDF(CK || IK, FC = 0x20, network-name, SQN ⊕ AK)` using the
//!    generic HMAC-SHA-256 KDF (TS 33.220 Annex B).
//! 2. **Master key schedule** — RFC 9048 §3.3: `MK = PRF'(IK' || CK',
//!    "EAP-AKA'" || Identity)` expanded into `K_encr (16) | K_aut (32) |
//!    K_re (32) | MSK (64) | EMSK (64)`. PRF' is the HMAC-SHA-256 iteration of
//!    RFC 9048 §3.4.1 (already provided by [`crate::kdf::calculate_prf_prime`]).
//!
//! For 5G, the AUSF takes `KAUSF = MSB256(EMSK)` (TS 33.501); KSEAF/KAMF then
//! reuse the existing [`crate::kdf::derive_kseaf`] / [`crate::kdf::derive_kamf`].
//!
//! All functions are additive primitives; UE/AUSF wiring lives in the consumers.

use crate::kdf::{calculate_kdf_key, calculate_prf_prime, KEY_128_SIZE, KEY_256_SIZE};

/// FC octet for the EAP-AKA' CK'/IK' derivation (TS 33.402 Annex A.2).
pub const FC_CK_IK_PRIME: u8 = 0x20;

/// Total master-key material length: K_encr(16) + K_aut(32) + K_re(32) +
/// MSK(64) + EMSK(64) = 208 octets (RFC 9048 §3.3).
const MK_TOTAL_LEN: usize = 16 + 32 + 32 + 64 + 64;

/// EAP-AKA' master-key outputs (RFC 9048 §3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EapAkaPrimeKeys {
    /// K_encr — 128-bit AT_ENCR_DATA key.
    pub k_encr: [u8; 16],
    /// K_aut — 256-bit AT_MAC key.
    pub k_aut: [u8; 32],
    /// K_re — 256-bit re-authentication key.
    pub k_re: [u8; 32],
    /// MSK — 512-bit Master Session Key.
    pub msk: [u8; 64],
    /// EMSK — 512-bit Extended Master Session Key (KAUSF source for 5G).
    pub emsk: [u8; 64],
}

/// Derive CK' and IK' from the AKA keys, bound to the network name
/// (RFC 9048 §3.3 / TS 33.402 Annex A.2).
///
/// `CK' || IK' = HMAC-SHA-256(CK || IK, 0x20 || network_name || L0 ||
/// (SQN ⊕ AK) || L1)`, where `SQN ⊕ AK` is the first 6 octets of AUTN.
///
/// # Arguments
/// * `ck` - 128-bit Cipher Key from AKA
/// * `ik` - 128-bit Integrity Key from AKA
/// * `network_name` - access/serving network name (AT_KDF_INPUT value)
/// * `sqn_xor_ak` - SQN ⊕ AK (6 octets, = AUTN\[0..6\])
///
/// # Returns
/// `(CK', IK')`, each 128 bits.
pub fn derive_ck_ik_prime(
    ck: &[u8; KEY_128_SIZE],
    ik: &[u8; KEY_128_SIZE],
    network_name: &[u8],
    sqn_xor_ak: &[u8],
) -> ([u8; KEY_128_SIZE], [u8; KEY_128_SIZE]) {
    let mut key = [0u8; KEY_256_SIZE];
    key[..KEY_128_SIZE].copy_from_slice(ck);
    key[KEY_128_SIZE..].copy_from_slice(ik);

    let out = calculate_kdf_key(&key, FC_CK_IK_PRIME, &[network_name, sqn_xor_ak]);

    let mut ck_prime = [0u8; KEY_128_SIZE];
    let mut ik_prime = [0u8; KEY_128_SIZE];
    ck_prime.copy_from_slice(&out[..KEY_128_SIZE]);
    ik_prime.copy_from_slice(&out[KEY_128_SIZE..]);
    (ck_prime, ik_prime)
}

/// Derive the EAP-AKA' master key schedule (RFC 9048 §3.3).
///
/// `MK = PRF'(IK' || CK', "EAP-AKA'" || Identity)`, sliced into
/// `K_encr | K_aut | K_re | MSK | EMSK`. Note the key is `IK' || CK'`
/// (integrity key first) and the salt prefix is the 8-octet ASCII string
/// `"EAP-AKA'"`.
///
/// # Arguments
/// * `identity` - the EAP peer identity used in this authentication (the octets
///   of the identity string, e.g. the permanent/pseudonym NAI without realm
///   transformation), per RFC 9048 §3.3
/// * `ck_prime` - 128-bit CK' from [`derive_ck_ik_prime`]
/// * `ik_prime` - 128-bit IK' from [`derive_ck_ik_prime`]
pub fn derive_eap_aka_prime_keys(
    identity: &[u8],
    ck_prime: &[u8; KEY_128_SIZE],
    ik_prime: &[u8; KEY_128_SIZE],
) -> EapAkaPrimeKeys {
    // Key = IK' || CK' (integrity key first, RFC 9048 §3.3).
    let mut key = [0u8; KEY_256_SIZE];
    key[..KEY_128_SIZE].copy_from_slice(ik_prime);
    key[KEY_128_SIZE..].copy_from_slice(ck_prime);

    // S = "EAP-AKA'" || Identity
    let mut salt = Vec::with_capacity(8 + identity.len());
    salt.extend_from_slice(b"EAP-AKA'");
    salt.extend_from_slice(identity);

    let mk = calculate_prf_prime(&key, &salt, MK_TOTAL_LEN);

    let mut k_encr = [0u8; 16];
    let mut k_aut = [0u8; 32];
    let mut k_re = [0u8; 32];
    let mut msk = [0u8; 64];
    let mut emsk = [0u8; 64];
    k_encr.copy_from_slice(&mk[0..16]);
    k_aut.copy_from_slice(&mk[16..48]);
    k_re.copy_from_slice(&mk[48..80]);
    msk.copy_from_slice(&mk[80..144]);
    emsk.copy_from_slice(&mk[144..208]);

    EapAkaPrimeKeys {
        k_encr,
        k_aut,
        k_re,
        msk,
        emsk,
    }
}

/// 5G `KAUSF` for EAP-AKA' = the most significant 256 bits of the EMSK
/// (TS 33.501 §6.1.3.1).
pub fn derive_kausf_eap(emsk: &[u8; 64]) -> [u8; KEY_256_SIZE] {
    let mut kausf = [0u8; KEY_256_SIZE];
    kausf.copy_from_slice(&emsk[..KEY_256_SIZE]);
    kausf
}

/// Convenience one-shot: run the full EAP-AKA' derivation from the AKA outputs.
///
/// Returns the master-key schedule and the 5G `KAUSF = MSB256(EMSK)`.
pub fn run_eap_aka_prime(
    ck: &[u8; KEY_128_SIZE],
    ik: &[u8; KEY_128_SIZE],
    network_name: &[u8],
    sqn_xor_ak: &[u8],
    identity: &[u8],
) -> (EapAkaPrimeKeys, [u8; KEY_256_SIZE]) {
    let (ck_prime, ik_prime) = derive_ck_ik_prime(ck, ik, network_name, sqn_xor_ak);
    let keys = derive_eap_aka_prime_keys(identity, &ck_prime, &ik_prime);
    let kausf = derive_kausf_eap(&keys.emsk);
    (keys, kausf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    fn arr16(s: &str) -> [u8; 16] {
        unhex(s).try_into().unwrap()
    }

    // RFC 9048 Appendix test vectors. Each value below is published by RFC 9048
    // AND independently reproduced byte-for-byte by a from-scratch HMAC-SHA-256
    // reference of the RFC 9048 §3.3 derivation (the agreement of two independent
    // computations rules out both an implementation bug and a transcription error).
    // Shared AKA inputs (both cases): CK/IK fixed, SQN⊕AK = AUTN[0..6].

    const CK: &str = "5349fbe098649f948f5d2e973a81c00f";
    const IK: &str = "9744871ad32bf9bbd1dd5ce54e3e2e5a";
    // AUTN = bb52e91c747ac3ab2a5c23d15ee351d5 => SQN⊕AK = first 6 octets.
    const SQN_XOR_AK: &str = "bb52e91c747a";
    // Identity "0555444333222111" used as ASCII octets.
    const IDENTITY: &[u8] = b"0555444333222111";

    /// RFC 9048 Case 1 — network name "WLAN". Pins all five master-key outputs.
    #[test]
    fn test_eap_aka_prime_rfc9048_case1_wlan() {
        let ck = arr16(CK);
        let ik = arr16(IK);
        let sqn_xor_ak = unhex(SQN_XOR_AK);

        let (ck_prime, ik_prime) = derive_ck_ik_prime(&ck, &ik, b"WLAN", &sqn_xor_ak);
        assert_eq!(ck_prime, arr16("0093962d0dd84aa5684b045c9edffa04"), "CK'");
        assert_eq!(ik_prime, arr16("ccfc230ca74fcc96c0a5d61164f5a76c"), "IK'");

        let keys = derive_eap_aka_prime_keys(IDENTITY, &ck_prime, &ik_prime);
        assert_eq!(
            &keys.k_encr[..],
            &unhex("766fa0a6c317174b812d52fbcd11a179")[..],
            "K_encr"
        );
        assert_eq!(
            &keys.k_aut[..],
            &unhex("0842ea722ff6835bfa2032499fc3ec23c2f0e388b4f07543ffc677f1696d71ea")[..],
            "K_aut"
        );
        assert_eq!(
            &keys.k_re[..],
            &unhex("cf83aa8bc7e0aced892acc98e76a9b2095b558c7795c7094715cb3393aa7d17a")[..],
            "K_re"
        );
        assert_eq!(
            &keys.msk[..],
            &unhex(concat!(
                "67c42d9aa56c1b79e295e3459fc3d187d42be0bf818d3070e362c5e967a4d544",
                "e8ecfe19358ab3039aff03b7c930588c055babee58a02650b067ec4e9347c75a"
            ))[..],
            "MSK"
        );
        assert_eq!(
            &keys.emsk[..],
            &unhex(concat!(
                "f861703cd775590e16c7679ea3874ada866311de290764d760cf76df647ea01c",
                "313f69924bdd7650ca9bac141ea075c4ef9e8029c0e290cdbad5638b63bc23fb"
            ))[..],
            "EMSK"
        );

        // 5G KAUSF = MSB256(EMSK).
        let kausf = derive_kausf_eap(&keys.emsk);
        assert_eq!(&kausf[..], &keys.emsk[..32]);
    }

    /// RFC 9048 Case 2 — network name "HRPD". Proves the network name feeds CK'/IK'
    /// (same CK/IK/SQN⊕AK, different name => different keys).
    #[test]
    fn test_eap_aka_prime_rfc9048_case2_hrpd() {
        let ck = arr16(CK);
        let ik = arr16(IK);
        let sqn_xor_ak = unhex(SQN_XOR_AK);

        let (ck_prime, ik_prime) = derive_ck_ik_prime(&ck, &ik, b"HRPD", &sqn_xor_ak);
        assert_eq!(ck_prime, arr16("3820f0277fa5f77732b1fb1d90c1a0da"), "CK'");
        assert_eq!(ik_prime, arr16("db94a0ab557ef6c9ab48619ca05b9a9f"), "IK'");

        // Different network name from Case 1 => different CK'.
        let (ck_prime_wlan, _) = derive_ck_ik_prime(&ck, &ik, b"WLAN", &sqn_xor_ak);
        assert_ne!(ck_prime, ck_prime_wlan);
    }

    /// The one-shot helper must agree with the staged calls.
    #[test]
    fn test_run_eap_aka_prime_matches_staged() {
        let ck = arr16(CK);
        let ik = arr16(IK);
        let sqn_xor_ak = unhex(SQN_XOR_AK);

        let (keys, kausf) = run_eap_aka_prime(&ck, &ik, b"WLAN", &sqn_xor_ak, IDENTITY);
        let (ckp, ikp) = derive_ck_ik_prime(&ck, &ik, b"WLAN", &sqn_xor_ak);
        let staged = derive_eap_aka_prime_keys(IDENTITY, &ckp, &ikp);
        assert_eq!(keys, staged);
        assert_eq!(&kausf[..], &staged.emsk[..32]);
    }
}
