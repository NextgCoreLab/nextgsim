//! SRB PDCP integrity protection and ciphering (TS 38.323 §5.8, §5.9;
//! TS 33.501 §6.5).
//!
//! # Why this lives in `nextgsim-pdcp` and not in either endpoint
//!
//! **Both ends have to agree byte for byte.** The UE protects an uplink RRC PDU
//! and the gNB verifies it; the gNB protects a downlink one and the UE verifies
//! that. Two implementations of one MAC-and-cipher layout is a defect waiting to
//! happen — the same reasoning that put `compute_short_mac_i` in `nextgsim-rrc`.
//!
//! It belongs in *this* crate specifically because §5.8/§5.9 are PDCP's own
//! clauses, and because `nextgsim-pdcp` is the only crate both `nextgsim-gnb` and
//! `nextgsim-ue` already depend on that could host it without inverting the
//! layering.
//!
//! # The layout
//!
//! ```text
//! transmit:  MAC-I = f(K_RRCint, COUNT, BEARER, DIRECTION, plaintext)
//!            payload = cipher(K_RRCenc, COUNT, BEARER, DIRECTION,
//!                             plaintext || MAC-I)
//! receive:   plaintext || MAC-I = cipher(...)   // NEA is a stream cipher
//!            verify MAC-I in constant time, discard on mismatch
//! ```
//!
//! The MAC is computed over the **plaintext** and the concatenation is ciphered
//! as one, which is the order TS 38.323 §5.8 gives. Getting it the other way
//! round (cipher, then MAC the ciphertext) also "round trips", which is exactly
//! why both ends must share this code rather than each implement the clause.

use crate::algorithms::{apply_ciphering, compute_mac_i, ct_eq_mac};

// The algorithm table, the MAC length and the constant-time comparison live in
// `crate::algorithms` and are shared with `UpSecurity`, so the SRB and DRB layouts
// cannot drift onto different algorithm selections. Re-exported here because every
// caller reaches them through this module.
pub use crate::algorithms::{ALG_ID_MAX, ALG_ID_NULL, MAC_I_LEN};

/// Errors from SRB PDCP protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SrbSecurityError {
    /// A protected PDU shorter than the MAC-I it must contain.
    #[error("SRB PDCP payload is shorter than the MAC-I")]
    PduTooShort,

    /// The MAC-I did not match. The plaintext is NOT returned
    /// (TS 33.501 §6.5: a PDU failing integrity is discarded).
    #[error("SRB PDCP integrity check failed")]
    IntegrityCheckFailed,

    /// An algorithm identity outside 0..=3.
    #[error("unknown 3GPP algorithm identity {0}; refusing rather than falling back")]
    UnknownAlgorithm(u8),
}

/// The SRB PDCP security state for one bearer direction pair.
///
/// Constructed once per UE per activation. Holds keys, so it deliberately does
/// **not** derive `Debug` — see the manual impl.
#[derive(Clone)]
pub struct SrbSecurity {
    k_rrc_enc: [u8; 16],
    k_rrc_int: [u8; 16],
    ciphering_alg_id: u8,
    integrity_alg_id: u8,
}

impl std::fmt::Debug for SrbSecurity {
    /// Never prints the keys. A `Debug` that leaked `K_RRCint` would put it in
    /// every log line that formatted a UE context.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SrbSecurity")
            .field("ciphering_alg_id", &self.ciphering_alg_id)
            .field("integrity_alg_id", &self.integrity_alg_id)
            .field("k_rrc_enc", &"<redacted>")
            .field("k_rrc_int", &"<redacted>")
            .finish()
    }
}

impl SrbSecurity {
    /// Builds the SRB security state, refusing an algorithm identity it does not
    /// know.
    ///
    /// Fail-closed at construction rather than at first use: an unknown identity
    /// means the two ends did not agree on an algorithm, and discovering that on
    /// the first PDU would mean the activation already claimed success.
    pub fn new(
        k_rrc_enc: [u8; 16],
        k_rrc_int: [u8; 16],
        ciphering_alg_id: u8,
        integrity_alg_id: u8,
    ) -> Result<Self, SrbSecurityError> {
        if ciphering_alg_id > ALG_ID_MAX {
            return Err(SrbSecurityError::UnknownAlgorithm(ciphering_alg_id));
        }
        if integrity_alg_id > ALG_ID_MAX {
            return Err(SrbSecurityError::UnknownAlgorithm(integrity_alg_id));
        }
        Ok(Self {
            k_rrc_enc,
            k_rrc_int,
            ciphering_alg_id,
            integrity_alg_id,
        })
    }

    /// The negotiated ciphering algorithm identity.
    pub fn ciphering_alg_id(&self) -> u8 {
        self.ciphering_alg_id
    }

    /// The negotiated integrity algorithm identity.
    pub fn integrity_alg_id(&self) -> u8 {
        self.integrity_alg_id
    }

    /// The 32-bit MAC-I over `message` (TS 38.323 §5.9, TS 33.501 Annex D).
    ///
    /// NIA0 returns all zeros, which is what null integrity means — and is why a
    /// caller must never treat "the MAC verified" as evidence of protection
    /// without also checking the algorithm is not NIA0.
    pub fn compute_mac_i(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        message: &[u8],
    ) -> [u8; MAC_I_LEN] {
        compute_mac_i(
            self.integrity_alg_id,
            &self.k_rrc_int,
            count,
            bearer,
            direction,
            message,
        )
    }

    /// Applies the ciphering keystream in place (TS 38.323 §5.8).
    fn apply_ciphering(&self, count: u32, bearer: u8, direction: u8, data: &mut [u8]) {
        apply_ciphering(
            self.ciphering_alg_id,
            &self.k_rrc_enc,
            count,
            bearer,
            direction,
            data,
        );
    }

    /// Protects an SRB RRC message for transmission.
    ///
    /// MAC over the plaintext, appended, then the concatenation ciphered.
    pub fn protect(&self, count: u32, bearer: u8, direction: u8, plaintext: &[u8]) -> Vec<u8> {
        let mac = self.compute_mac_i(count, bearer, direction, plaintext);
        let mut out = Vec::with_capacity(plaintext.len() + MAC_I_LEN);
        out.extend_from_slice(plaintext);
        out.extend_from_slice(&mac);
        self.apply_ciphering(count, bearer, direction, &mut out);
        out
    }

    /// Unprotects a received SRB PDCP payload.
    ///
    /// Fail-closed: on a MAC mismatch or a too-short payload the plaintext is
    /// **never** returned, so a caller cannot accidentally act on an unverified
    /// PDU (TS 33.501 §6.5).
    pub fn unprotect(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        protected: &[u8],
    ) -> Result<Vec<u8>, SrbSecurityError> {
        if protected.len() < MAC_I_LEN {
            return Err(SrbSecurityError::PduTooShort);
        }
        let mut buf = protected.to_vec();
        self.apply_ciphering(count, bearer, direction, &mut buf);
        let split = buf.len() - MAC_I_LEN;
        let (data, mac_recv) = buf.split_at(split);
        let mac_calc = self.compute_mac_i(count, bearer, direction, data);
        if ct_eq_mac(&mac_calc, mac_recv) {
            Ok(data.to_vec())
        } else {
            Err(SrbSecurityError::IntegrityCheckFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const KEY_ENC: [u8; 16] = [
        0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE,
        0xFF,
    ];
    const KEY_INT: [u8; 16] = [
        0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11,
        0x00,
    ];
    const SRB1: u8 = 0;
    const UPLINK: u8 = 0;
    const DOWNLINK: u8 = 1;

    fn sec(ciph: u8, integ: u8) -> SrbSecurity {
        SrbSecurity::new(KEY_ENC, KEY_INT, ciph, integ).expect("legal algorithm ids")
    }

    /// #31, criterion 8: protect on one endpoint, unprotect on the other, for
    /// **NEA1, NEA2 and NEA3** — the third of which used to be refused outright.
    #[test]
    fn every_nea_round_trips_between_the_two_endpoints() {
        let msg = b"\x20\x40\x00\x22\x00\x04\x00\x00".to_vec();
        for ciph in 0..=ALG_ID_MAX {
            for integ in 1..=ALG_ID_MAX {
                // The gNB protects downlink; the UE unprotects the same octets.
                let gnb = sec(ciph, integ);
                let ue = sec(ciph, integ);
                let protected = gnb.protect(7, SRB1, DOWNLINK, &msg);
                assert_eq!(
                    protected.len(),
                    msg.len() + MAC_I_LEN,
                    "NEA{ciph}/NIA{integ}: the MAC-I must be appended"
                );
                let recovered = ue
                    .unprotect(7, SRB1, DOWNLINK, &protected)
                    .unwrap_or_else(|e| panic!("NEA{ciph}/NIA{integ} must verify: {e}"));
                assert_eq!(
                    recovered, msg,
                    "NEA{ciph}/NIA{integ}: plaintext must survive"
                );

                // And uplink, which uses a different DIRECTION bit and so a
                // different keystream and MAC.
                let protected_ul = ue.protect(7, SRB1, UPLINK, &msg);
                assert_ne!(
                    protected_ul, protected,
                    "NEA{ciph}/NIA{integ}: the DIRECTION bit must change the output, \
                     or uplink and downlink share a keystream"
                );
                assert_eq!(
                    gnb.unprotect(7, SRB1, UPLINK, &protected_ul)
                        .expect("uplink must verify"),
                    msg
                );
            }
        }
    }

    /// A tampered payload is rejected and the plaintext is never returned.
    #[test]
    fn a_tampered_payload_is_rejected_for_every_algorithm() {
        let msg = b"secret-rrc-pdu".to_vec();
        for ciph in 0..=ALG_ID_MAX {
            for integ in 1..=ALG_ID_MAX {
                let s = sec(ciph, integ);
                let mut protected = s.protect(1, SRB1, DOWNLINK, &msg);
                protected[0] ^= 0x01;
                assert_eq!(
                    s.unprotect(1, SRB1, DOWNLINK, &protected),
                    Err(SrbSecurityError::IntegrityCheckFailed),
                    "NEA{ciph}/NIA{integ}: a flipped payload bit must fail closed"
                );
            }
        }
    }

    /// A forged MAC-I is rejected. Distinct from the test above: this leaves the
    /// payload intact and attacks only the MAC, which is what an attacker who
    /// knows the plaintext would do.
    #[test]
    fn a_forged_mac_i_is_rejected_and_the_plaintext_is_not_surfaced() {
        let msg = b"secret-rrc-pdu".to_vec();
        let s = sec(2, 2);
        let mut protected = s.protect(1, SRB1, DOWNLINK, &msg);
        let last = protected.len() - 1;
        protected[last] ^= 0xFF;
        let result = s.unprotect(1, SRB1, DOWNLINK, &protected);
        assert_eq!(result, Err(SrbSecurityError::IntegrityCheckFailed));
        assert!(
            result.is_err(),
            "and the API must give the caller NO way to reach the plaintext of a \
             PDU that failed integrity"
        );
    }

    /// The COUNT is part of the input, so replaying a PDU under a different COUNT
    /// fails. Without this, a captured PDU would verify forever.
    #[test]
    fn a_pdu_replayed_under_a_different_count_is_rejected() {
        let s = sec(2, 2);
        let protected = s.protect(5, SRB1, DOWNLINK, b"rrc");
        assert!(s.unprotect(5, SRB1, DOWNLINK, &protected).is_ok());
        assert_eq!(
            s.unprotect(6, SRB1, DOWNLINK, &protected),
            Err(SrbSecurityError::IntegrityCheckFailed),
            "a different COUNT must not verify, or a captured PDU replays forever"
        );
    }

    /// A payload with no room for a MAC-I is refused rather than indexed into.
    #[test]
    fn a_payload_shorter_than_the_mac_is_refused() {
        let s = sec(2, 2);
        for len in 0..MAC_I_LEN {
            assert_eq!(
                s.unprotect(1, SRB1, DOWNLINK, &vec![0u8; len]),
                Err(SrbSecurityError::PduTooShort)
            );
        }
    }

    /// An unknown algorithm identity is refused at CONSTRUCTION, not silently
    /// treated as null.
    #[test]
    fn an_unknown_algorithm_identity_is_refused_rather_than_defaulted_to_null() {
        for bad in [ALG_ID_MAX + 1, 7, 255] {
            assert_eq!(
                SrbSecurity::new(KEY_ENC, KEY_INT, bad, 2).err(),
                Some(SrbSecurityError::UnknownAlgorithm(bad)),
                "an unknown CIPHERING id must be refused"
            );
            assert_eq!(
                SrbSecurity::new(KEY_ENC, KEY_INT, 2, bad).err(),
                Some(SrbSecurityError::UnknownAlgorithm(bad)),
                "an unknown INTEGRITY id must be refused"
            );
        }
    }

    /// NEA0 leaves the PDU readable, and NIA0's MAC is all zeros. Pinned because
    /// a caller must not read "unprotect succeeded" as "the PDU was protected".
    #[test]
    fn the_null_algorithms_are_null_and_visibly_so() {
        let s = sec(ALG_ID_NULL, ALG_ID_NULL);
        let protected = s.protect(1, SRB1, DOWNLINK, b"plain");
        assert_eq!(
            &protected[..5],
            b"plain",
            "NEA0 must leave the plaintext in the clear"
        );
        assert_eq!(
            &protected[5..],
            &[0u8; MAC_I_LEN],
            "NIA0's MAC is all zeros, so a successful verify proves nothing"
        );
        assert!(s.unprotect(1, SRB1, DOWNLINK, &protected).is_ok());
    }

    /// A ciphered PDU must not be readable. Obvious, and pinned anyway: an
    /// `apply_ciphering` that silently did nothing for a real NEA would pass every
    /// round-trip test above.
    #[test]
    fn a_ciphered_pdu_is_not_readable() {
        let msg = b"this-must-not-appear".to_vec();
        for ciph in 1..=ALG_ID_MAX {
            let s = sec(ciph, 2);
            let protected = s.protect(1, SRB1, DOWNLINK, &msg);
            assert!(
                !protected.windows(msg.len()).any(|w| w == msg.as_slice()),
                "NEA{ciph} left the plaintext visible in the protected PDU"
            );
        }
    }

    /// `Debug` must not print keys.
    #[test]
    fn debug_does_not_leak_the_keys() {
        let text = format!("{:?}", sec(2, 2));
        assert!(text.contains("redacted"));
        assert!(
            !text.contains("11, 22") && !text.contains("0x11"),
            "the key material must not appear: {text}"
        );
    }
}
