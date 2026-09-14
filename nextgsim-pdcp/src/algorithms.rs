//! The NIA/NEA primitives PDCP protection is built from (TS 33.501 §5.11.1,
//! Annex D; TS 38.323 §5.8, §5.9).
//!
//! # Why these are free functions and not methods
//!
//! SRBs and DRBs protect *different byte layouts* — an SRB PDU's MAC-I covers the
//! RRC message alone, a DRB PDU's covers the PDCP header as well (§5.9), and only
//! the DRB's header stays in the clear (§5.8). But they run the **same** algorithm
//! selection over the same COUNT/BEARER/DIRECTION inputs.
//!
//! Splitting them this way is deliberate: [`SrbSecurity`](crate::SrbSecurity) and
//! [`UpSecurity`](crate::UpSecurity) each own one layout, and both call *these* for
//! the crypto, so a change to the algorithm table cannot reach one bearer type and
//! miss the other. It also keeps each layout's tests about the layout.

use nextgsim_crypto::nea::{nea1_encrypt, nea2_encrypt};
use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac};
use nextgsim_crypto::zuc::nea3_encrypt;

/// Length of the PDCP MAC-I in octets (TS 38.323 §6.3.4: 32 bits).
pub const MAC_I_LEN: usize = 4;

/// The 3GPP algorithm identity of null ciphering / null integrity
/// (TS 33.501 §5.11.1).
pub const ALG_ID_NULL: u8 = 0;

/// The highest 3GPP ciphering / integrity algorithm identity this layer accepts.
pub const ALG_ID_MAX: u8 = 3;

/// The 32-bit MAC-I over `message` (TS 38.323 §5.9, TS 33.501 Annex D).
///
/// NIA0 returns all zeros, which is what null integrity means — and is why a
/// caller must never treat "the MAC verified" as evidence of protection without
/// also checking the algorithm is not NIA0.
pub fn compute_mac_i(
    integrity_alg_id: u8,
    key: &[u8; 16],
    count: u32,
    bearer: u8,
    direction: u8,
    message: &[u8],
) -> [u8; MAC_I_LEN] {
    match integrity_alg_id {
        1 => nia1_compute_mac(count, bearer, direction, key, message),
        2 => nia2_compute_mac(count, bearer, direction, key, message),
        3 => nia3_compute_mac(count, bearer, direction, key, message),
        // NIA0: the null algorithm's MAC is all zeros (TS 33.501 §5.11.1).
        _ => [0u8; MAC_I_LEN],
    }
}

/// Applies the ciphering keystream in place (TS 38.323 §5.8).
///
/// All three NEAs are stream ciphers, so this same routine deciphers — which is
/// why there is one function rather than an encrypt/decrypt pair.
pub fn apply_ciphering(
    ciphering_alg_id: u8,
    key: &[u8; 16],
    count: u32,
    bearer: u8,
    direction: u8,
    data: &mut [u8],
) {
    match ciphering_alg_id {
        1 => nea1_encrypt(count, bearer, direction, key, data),
        2 => nea2_encrypt(count, bearer, direction, key, data),
        // NEA3 (ZUC). This used to be refused as "no keystream in
        // nextgsim-crypto", which was never true: `zuc::nea3_encrypt` is a
        // complete implementation and `nextgsim-nas` has been using it for NAS
        // ciphering all along (issue #31).
        3 => nea3_encrypt(count, bearer, direction, key, data),
        // NEA0: null ciphering leaves the PDU in the clear.
        _ => {}
    }
}

/// Constant-time 4-octet MAC comparison.
///
/// Constant time because a MAC check that short-circuits on the first differing
/// octet leaks how much of a forgery was correct, which is enough to find the
/// rest one octet at a time.
pub fn ct_eq_mac(computed: &[u8; MAC_I_LEN], received: &[u8]) -> bool {
    if received.len() != MAC_I_LEN {
        return false;
    }
    let mut diff = 0u8;
    for i in 0..MAC_I_LEN {
        diff |= computed[i] ^ received[i];
    }
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF,
    ];

    /// Every real NIA produces a MAC that depends on all four inputs. Pinned here
    /// rather than only in the two layout modules, because a table that ignored an
    /// input would still round trip in both of them.
    #[test]
    fn every_mac_input_changes_the_mac() {
        for alg in 1..=ALG_ID_MAX {
            let base = compute_mac_i(alg, &KEY, 5, 1, 0, b"payload");
            assert_ne!(base, compute_mac_i(alg, &KEY, 6, 1, 0, b"payload"), "COUNT");
            assert_ne!(
                base,
                compute_mac_i(alg, &KEY, 5, 2, 0, b"payload"),
                "BEARER"
            );
            assert_ne!(
                base,
                compute_mac_i(alg, &KEY, 5, 1, 1, b"payload"),
                "DIRECTION"
            );
            assert_ne!(
                base,
                compute_mac_i(alg, &KEY, 5, 1, 0, b"payloae"),
                "message"
            );
            let mut other = KEY;
            other[0] ^= 0xFF;
            assert_ne!(base, compute_mac_i(alg, &other, 5, 1, 0, b"payload"), "key");
        }
    }

    /// NIA0's MAC is all zeros for anything, which is the property that makes a
    /// successful NIA0 verification worthless.
    #[test]
    fn nia0_macs_everything_to_zero() {
        assert_eq!(
            compute_mac_i(ALG_ID_NULL, &KEY, 5, 1, 0, b"anything"),
            [0u8; MAC_I_LEN]
        );
        assert_eq!(
            compute_mac_i(ALG_ID_NULL, &KEY, 9, 3, 1, b"else"),
            [0u8; MAC_I_LEN]
        );
    }

    /// Applying the keystream twice is the identity, for every real NEA. This is
    /// what lets one function both cipher and decipher.
    #[test]
    fn every_nea_is_its_own_inverse() {
        for alg in 0..=ALG_ID_MAX {
            let plain = b"the-quick-brown-fox".to_vec();
            let mut buf = plain.clone();
            apply_ciphering(alg, &KEY, 3, 1, 0, &mut buf);
            if alg != ALG_ID_NULL {
                assert_ne!(buf, plain, "NEA{alg} must actually cipher");
            }
            apply_ciphering(alg, &KEY, 3, 1, 0, &mut buf);
            assert_eq!(buf, plain, "NEA{alg} must be its own inverse");
        }
    }

    /// A MAC comparison must reject a wrong length rather than compare a prefix.
    #[test]
    fn ct_eq_mac_rejects_a_wrong_length() {
        let mac = [1u8, 2, 3, 4];
        assert!(ct_eq_mac(&mac, &[1, 2, 3, 4]));
        assert!(!ct_eq_mac(&mac, &[1, 2, 3]));
        assert!(!ct_eq_mac(&mac, &[1, 2, 3, 4, 5]));
        assert!(!ct_eq_mac(&mac, &[]));
    }
}
