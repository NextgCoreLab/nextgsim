//! DRB (user-plane) PDCP integrity protection and ciphering (TS 38.323 §5.8,
//! §5.9; TS 33.501 §6.6).
//!
//! # The DRB layout is not the SRB layout
//!
//! [`SrbSecurity`](crate::SrbSecurity) protects an RRC message that has no PDCP
//! header on the wire at all in this simulator. A DRB PDU does have one — a 2- or
//! 3-octet data-PDU header carrying the SN — and the two clauses treat it
//! differently:
//!
//! ```text
//! §5.9  MESSAGE for integrity  = PDCP header || data part, before ciphering
//! §5.8  data ciphered          = data part || MAC-I        (NOT the header)
//!
//! transmit:  MAC-I = f(K_UPint, COUNT, BEARER, DIRECTION, header || plaintext)
//!            PDU   = header || cipher(K_UPenc, ..., plaintext || MAC-I)
//! receive:   plaintext || MAC-I = cipher(...)   // NEA is a stream cipher
//!            verify MAC-I over header || plaintext, discard on mismatch
//! ```
//!
//! The header stays in the clear because RLC and the receiving PDCP entity must
//! read the SN to recover the COUNT the ciphering is keyed on — a ciphered header
//! is a chicken-and-egg. It is still covered by the MAC, so an attacker cannot
//! renumber a PDU without invalidating it.
//!
//! # Integrity is optional per DRB, and "off" is not NIA0
//!
//! A DRB configured without integrity protection carries **no MAC-I field**
//! (§6.2.2.2). That is a different wire format from NIA0, which appends four zero
//! octets. So [`UpSecurity`] holds `Option<u8>` for the integrity algorithm rather
//! than treating 0 as "off": conflating them would make a receiver strip four
//! octets of user payload off every PDU.
//!
//! Confidentiality is controlled the same way through NEA0, which needs no
//! `Option` because null ciphering and no ciphering produce identical bytes.

use crate::algorithms::{apply_ciphering, compute_mac_i, ct_eq_mac, ALG_ID_MAX, MAC_I_LEN};

/// DIRECTION bit for uplink (TS 33.501 Annex D.3.1.2).
pub const DIRECTION_UPLINK: u8 = 0;

/// DIRECTION bit for downlink (TS 33.501 Annex D.3.1.2).
pub const DIRECTION_DOWNLINK: u8 = 1;

/// Errors from DRB PDCP protection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum UpSecurityError {
    /// A protected PDU with no room for the header and the MAC-I it must carry.
    #[error("DRB PDCP PDU is shorter than its header plus MAC-I")]
    PduTooShort,

    /// The MAC-I did not match. The plaintext is NOT returned (TS 33.501 §6.6.3:
    /// a PDU failing integrity is discarded).
    #[error("DRB PDCP integrity check failed")]
    IntegrityCheckFailed,

    /// An algorithm identity outside 0..=3.
    #[error("unknown 3GPP algorithm identity {0}; refusing rather than falling back")]
    UnknownAlgorithm(u8),
}

/// The user-plane PDCP security state for one DRB.
///
/// Holds keys, so it deliberately does **not** derive `Debug` — see the manual
/// impl.
#[derive(Clone)]
pub struct UpSecurity {
    k_up_enc: [u8; 16],
    k_up_int: [u8; 16],
    /// NEA identity. `0` (NEA0) leaves the data part in the clear, which is what
    /// a `not-needed` confidentiality policy asks for.
    ciphering_alg_id: u8,
    /// NIA identity, or `None` when the DRB is not integrity protected at all —
    /// which is a different wire format from NIA0. See the module docs.
    integrity_alg_id: Option<u8>,
}

impl std::fmt::Debug for UpSecurity {
    /// Never prints the keys. A `Debug` that leaked `K_UPint` would put it in every
    /// log line that formatted a PDCP entity.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpSecurity")
            .field("ciphering_alg_id", &self.ciphering_alg_id)
            .field("integrity_alg_id", &self.integrity_alg_id)
            .field("k_up_enc", &"<redacted>")
            .field("k_up_int", &"<redacted>")
            .finish()
    }
}

impl UpSecurity {
    /// Builds the DRB security state, refusing an algorithm identity it does not
    /// know.
    ///
    /// Fail-closed at construction rather than at first use: an unknown identity
    /// means the two ends did not agree on an algorithm, and discovering that on
    /// the first user-plane packet would mean the DRB had already been reported to
    /// the AMF as established.
    pub fn new(
        k_up_enc: [u8; 16],
        k_up_int: [u8; 16],
        ciphering_alg_id: u8,
        integrity_alg_id: Option<u8>,
    ) -> Result<Self, UpSecurityError> {
        if ciphering_alg_id > ALG_ID_MAX {
            return Err(UpSecurityError::UnknownAlgorithm(ciphering_alg_id));
        }
        if let Some(id) = integrity_alg_id {
            if id > ALG_ID_MAX {
                return Err(UpSecurityError::UnknownAlgorithm(id));
            }
        }
        Ok(Self {
            k_up_enc,
            k_up_int,
            ciphering_alg_id,
            integrity_alg_id,
        })
    }

    /// The negotiated ciphering algorithm identity.
    pub fn ciphering_alg_id(&self) -> u8 {
        self.ciphering_alg_id
    }

    /// The negotiated integrity algorithm identity, or `None` when this DRB is not
    /// integrity protected.
    pub fn integrity_alg_id(&self) -> Option<u8> {
        self.integrity_alg_id
    }

    /// Whether this DRB appends a MAC-I, and therefore whether its PDUs are four
    /// octets longer than their payload.
    pub fn integrity_protected(&self) -> bool {
        self.integrity_alg_id.is_some()
    }

    /// How many octets of MAC-I this DRB's PDUs carry: 4, or 0 when integrity is
    /// not configured.
    pub fn mac_i_len(&self) -> usize {
        if self.integrity_protected() {
            MAC_I_LEN
        } else {
            0
        }
    }

    /// Protects one DRB data PDU: returns `header || cipher(data || MAC-I)`.
    ///
    /// Takes the header rather than only the data because §5.9's MESSAGE spans
    /// both — a caller that could pass only the payload would silently produce a
    /// MAC that does not cover the SN.
    pub fn protect(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        header: &[u8],
        data: &[u8],
    ) -> Vec<u8> {
        let mut protected = Vec::with_capacity(data.len() + MAC_I_LEN);
        protected.extend_from_slice(data);
        if let Some(alg) = self.integrity_alg_id {
            // §5.9: the MESSAGE is the header and the data part, before ciphering.
            let mut message = Vec::with_capacity(header.len() + data.len());
            message.extend_from_slice(header);
            message.extend_from_slice(data);
            let mac = compute_mac_i(alg, &self.k_up_int, count, bearer, direction, &message);
            protected.extend_from_slice(&mac);
        }
        // §5.8: the ciphered unit is the data part and the MAC-I, never the header.
        apply_ciphering(
            self.ciphering_alg_id,
            &self.k_up_enc,
            count,
            bearer,
            direction,
            &mut protected,
        );
        let mut pdu = Vec::with_capacity(header.len() + protected.len());
        pdu.extend_from_slice(header);
        pdu.extend_from_slice(&protected);
        pdu
    }

    /// Unprotects a received DRB data PDU and returns its plaintext data part.
    ///
    /// `header_len` is the PDCP data-PDU header length the receiving entity is
    /// configured for; the header itself is read from `pdu`, so the caller cannot
    /// pass a header that differs from the one on the wire.
    ///
    /// Fail-closed: on a MAC mismatch the plaintext is **never** returned, so a
    /// caller cannot accidentally deliver an unverified packet upward
    /// (TS 33.501 §6.6.3).
    pub fn unprotect(
        &self,
        count: u32,
        bearer: u8,
        direction: u8,
        pdu: &[u8],
        header_len: usize,
    ) -> Result<Vec<u8>, UpSecurityError> {
        if pdu.len() < header_len + self.mac_i_len() {
            return Err(UpSecurityError::PduTooShort);
        }
        let (header, protected) = pdu.split_at(header_len);
        let mut buf = protected.to_vec();
        apply_ciphering(
            self.ciphering_alg_id,
            &self.k_up_enc,
            count,
            bearer,
            direction,
            &mut buf,
        );
        let Some(alg) = self.integrity_alg_id else {
            return Ok(buf);
        };
        let split = buf.len() - MAC_I_LEN;
        let (data, mac_recv) = buf.split_at(split);
        let mut message = Vec::with_capacity(header.len() + data.len());
        message.extend_from_slice(header);
        message.extend_from_slice(data);
        let mac_calc = compute_mac_i(alg, &self.k_up_int, count, bearer, direction, &message);
        if ct_eq_mac(&mac_calc, mac_recv) {
            Ok(data.to_vec())
        } else {
            Err(UpSecurityError::IntegrityCheckFailed)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::ALG_ID_NULL;

    const KEY_ENC: [u8; 16] = [
        0x0F, 0x1E, 0x2D, 0x3C, 0x4B, 0x5A, 0x69, 0x78, 0x87, 0x96, 0xA5, 0xB4, 0xC3, 0xD2, 0xE1,
        0xF0,
    ];
    const KEY_INT: [u8; 16] = [
        0xF0, 0xE1, 0xD2, 0xC3, 0xB4, 0xA5, 0x96, 0x87, 0x78, 0x69, 0x5A, 0x4B, 0x3C, 0x2D, 0x1E,
        0x0F,
    ];
    /// DRB 1: TS 33.501 Annex D.3.1.2 sets BEARER to the radio bearer identity
    /// minus one.
    const DRB1: u8 = 0;
    /// A 12-bit-SN data-PDU header for SN 5 (D/C = 1).
    const HEADER: [u8; 2] = [0x80, 0x05];

    fn sec(ciph: u8, integ: Option<u8>) -> UpSecurity {
        UpSecurity::new(KEY_ENC, KEY_INT, ciph, integ).expect("legal algorithm ids")
    }

    /// #32, criterion 3: the gNB protects downlink and the UE recovers it, and vice
    /// versa, for every NEA/NIA combination.
    #[test]
    fn every_algorithm_pair_round_trips_between_the_two_endpoints() {
        let payload = b"\x45\x00\x00\x1c\x00\x01\x00\x00\x40\x01".to_vec();
        for ciph in 0..=ALG_ID_MAX {
            for integ in [None, Some(1), Some(2), Some(3)] {
                let gnb = sec(ciph, integ);
                let ue = sec(ciph, integ);
                let dl = gnb.protect(5, DRB1, DIRECTION_DOWNLINK, &HEADER, &payload);
                assert_eq!(
                    dl.len(),
                    HEADER.len() + payload.len() + gnb.mac_i_len(),
                    "NEA{ciph}/NIA{integ:?}: the PDU is header + payload + MAC-I"
                );
                assert_eq!(
                    &dl[..HEADER.len()],
                    &HEADER,
                    "NEA{ciph}/NIA{integ:?}: the header must stay in the clear, \
                     or the receiver cannot read the SN it needs for the COUNT"
                );
                assert_eq!(
                    ue.unprotect(5, DRB1, DIRECTION_DOWNLINK, &dl, HEADER.len())
                        .unwrap_or_else(|e| panic!("NEA{ciph}/NIA{integ:?} must verify: {e}")),
                    payload,
                    "NEA{ciph}/NIA{integ:?}: the payload must survive"
                );

                let ul = ue.protect(5, DRB1, DIRECTION_UPLINK, &HEADER, &payload);
                // NEA0 with no integrity is a DRB with no protection at all, so
                // there is nothing for the DIRECTION bit to change. Every other
                // combination must differ, or uplink and downlink share a keystream
                // or a MAC.
                if ciph != ALG_ID_NULL || integ.is_some() {
                    assert_ne!(
                        ul, dl,
                        "NEA{ciph}/NIA{integ:?}: the DIRECTION bit must change the output"
                    );
                }
                assert_eq!(
                    gnb.unprotect(5, DRB1, DIRECTION_UPLINK, &ul, HEADER.len())
                        .expect("uplink must verify"),
                    payload
                );
            }
        }
    }

    /// The MAC covers the **header**, so renumbering a PDU invalidates it. This is
    /// the property that distinguishes the DRB layout from the SRB one, and the
    /// only test that can tell them apart.
    #[test]
    fn tampering_with_the_header_fails_integrity() {
        let s = sec(ALG_ID_NULL, Some(2));
        let pdu = s.protect(5, DRB1, DIRECTION_DOWNLINK, &HEADER, b"payload");
        // POSITIVE CONTROL first. Without it this test passes whenever `protect` and
        // `unprotect` disagree about the layout for ANY reason, since a mismatch also
        // yields `IntegrityCheckFailed` -- a revert round that MACed only the data on
        // the transmit side stayed green on exactly that.
        assert_eq!(
            s.unprotect(5, DRB1, DIRECTION_DOWNLINK, &pdu, HEADER.len()),
            Ok(b"payload".to_vec()),
            "the untampered PDU must verify, or the failure below proves nothing"
        );
        let mut renumbered = pdu.clone();
        // Change the SN in the header, leaving the ciphered data part untouched.
        renumbered[1] ^= 0x01;
        assert_eq!(
            s.unprotect(5, DRB1, DIRECTION_DOWNLINK, &renumbered, HEADER.len()),
            Err(UpSecurityError::IntegrityCheckFailed),
            "a MAC that did not cover the PDCP header would accept a renumbered PDU"
        );
    }

    /// The MAC is computed over the header **as it appears on the wire**, so the same
    /// payload under two different headers gets two different MACs.
    ///
    /// Distinct from the tampering test above: that one can be satisfied by any
    /// consistent rejection, and this one can only be satisfied by the header being
    /// an input. A `protect`/`unprotect` pair that both MACed only the data would pass
    /// the tampering test's shape and fail here.
    #[test]
    fn the_header_is_an_input_to_the_mac_not_merely_checked() {
        let s = sec(ALG_ID_NULL, Some(2));
        let other_header = [0x80u8, 0x06];
        let a = s.protect(5, DRB1, DIRECTION_DOWNLINK, &HEADER, b"payload");
        let b = s.protect(5, DRB1, DIRECTION_DOWNLINK, &other_header, b"payload");
        assert_eq!(
            &a[HEADER.len()..a.len() - MAC_I_LEN],
            &b[other_header.len()..b.len() - MAC_I_LEN],
            "NEA0, so the data parts are identical -- only the MAC may differ"
        );
        assert_ne!(
            &a[a.len() - MAC_I_LEN..],
            &b[b.len() - MAC_I_LEN..],
            "two different headers over the same payload must give different MACs, or \
             the header is not part of the MESSAGE (TS 38.323 §5.9)"
        );
    }

    /// A flipped payload bit is rejected for every algorithm pair that has
    /// integrity at all.
    #[test]
    fn a_tampered_payload_is_rejected_whenever_integrity_is_configured() {
        for ciph in 0..=ALG_ID_MAX {
            for integ in [Some(1), Some(2), Some(3)] {
                let s = sec(ciph, integ);
                let mut pdu = s.protect(1, DRB1, DIRECTION_DOWNLINK, &HEADER, b"user-data");
                let target = HEADER.len();
                pdu[target] ^= 0x01;
                assert_eq!(
                    s.unprotect(1, DRB1, DIRECTION_DOWNLINK, &pdu, HEADER.len()),
                    Err(UpSecurityError::IntegrityCheckFailed),
                    "NEA{ciph}/NIA{integ:?}: a flipped payload bit must fail closed"
                );
            }
        }
    }

    /// A forged MAC-I is rejected, leaving the payload intact — what an attacker
    /// who already knows the plaintext would try.
    #[test]
    fn a_forged_mac_i_is_rejected_and_the_payload_is_not_surfaced() {
        let s = sec(2, Some(2));
        let mut pdu = s.protect(1, DRB1, DIRECTION_DOWNLINK, &HEADER, b"user-data");
        let last = pdu.len() - 1;
        pdu[last] ^= 0xFF;
        assert_eq!(
            s.unprotect(1, DRB1, DIRECTION_DOWNLINK, &pdu, HEADER.len()),
            Err(UpSecurityError::IntegrityCheckFailed),
            "and the API must give the caller no way to reach the payload of a PDU \
             that failed integrity"
        );
    }

    /// An unprotected DRB carries **no** MAC-I. Pinned because treating "integrity
    /// off" as NIA0 would strip four octets of user payload off every PDU.
    #[test]
    fn an_unprotected_drb_carries_no_mac_i_at_all() {
        let none = sec(ALG_ID_NULL, None);
        let nia0 = sec(ALG_ID_NULL, Some(0));
        let pdu_none = none.protect(1, DRB1, DIRECTION_UPLINK, &HEADER, b"abc");
        let pdu_nia0 = nia0.protect(1, DRB1, DIRECTION_UPLINK, &HEADER, b"abc");
        assert_eq!(
            pdu_none.len(),
            HEADER.len() + 3,
            "no integrity means no MAC-I field"
        );
        assert_eq!(
            pdu_nia0.len(),
            HEADER.len() + 3 + MAC_I_LEN,
            "NIA0 still appends four zero octets, which is a different wire format"
        );
        assert_eq!(&pdu_nia0[pdu_nia0.len() - MAC_I_LEN..], &[0u8; MAC_I_LEN]);
        // And a receiver configured either way recovers exactly what was sent.
        assert_eq!(
            none.unprotect(1, DRB1, DIRECTION_UPLINK, &pdu_none, HEADER.len())
                .expect("must verify"),
            b"abc"
        );
        assert_eq!(
            nia0.unprotect(1, DRB1, DIRECTION_UPLINK, &pdu_nia0, HEADER.len())
                .expect("must verify"),
            b"abc"
        );
    }

    /// The COUNT is part of the input, so a captured PDU does not replay under a
    /// different SN.
    #[test]
    fn a_pdu_replayed_under_a_different_count_is_rejected() {
        let s = sec(2, Some(2));
        let pdu = s.protect(5, DRB1, DIRECTION_DOWNLINK, &HEADER, b"data");
        assert!(s
            .unprotect(5, DRB1, DIRECTION_DOWNLINK, &pdu, HEADER.len())
            .is_ok());
        assert_eq!(
            s.unprotect(6, DRB1, DIRECTION_DOWNLINK, &pdu, HEADER.len()),
            Err(UpSecurityError::IntegrityCheckFailed),
            "a different COUNT must not verify, or a captured PDU replays forever"
        );
    }

    /// A PDU with no room for its header and MAC-I is refused rather than indexed
    /// into.
    #[test]
    fn a_pdu_shorter_than_its_header_and_mac_is_refused() {
        let s = sec(2, Some(2));
        for len in 0..HEADER.len() + MAC_I_LEN {
            assert_eq!(
                s.unprotect(1, DRB1, DIRECTION_UPLINK, &vec![0u8; len], HEADER.len()),
                Err(UpSecurityError::PduTooShort),
                "a {len}-octet PDU cannot hold a 2-octet header and a 4-octet MAC-I"
            );
        }
        // Exactly header + MAC-I is a legal empty-payload PDU, not a short one.
        let empty = s.protect(1, DRB1, DIRECTION_UPLINK, &HEADER, b"");
        assert_eq!(empty.len(), HEADER.len() + MAC_I_LEN);
        assert_eq!(
            s.unprotect(1, DRB1, DIRECTION_UPLINK, &empty, HEADER.len()),
            Ok(Vec::new())
        );
    }

    /// An unknown algorithm identity is refused at CONSTRUCTION, not silently
    /// treated as null.
    #[test]
    fn an_unknown_algorithm_identity_is_refused_rather_than_defaulted_to_null() {
        for bad in [ALG_ID_MAX + 1, 7, 255] {
            assert_eq!(
                UpSecurity::new(KEY_ENC, KEY_INT, bad, Some(2)).err(),
                Some(UpSecurityError::UnknownAlgorithm(bad)),
                "an unknown CIPHERING id must be refused"
            );
            assert_eq!(
                UpSecurity::new(KEY_ENC, KEY_INT, 2, Some(bad)).err(),
                Some(UpSecurityError::UnknownAlgorithm(bad)),
                "an unknown INTEGRITY id must be refused"
            );
        }
    }

    /// A ciphered data part must not be readable, and the header must still be.
    #[test]
    fn ciphering_hides_the_payload_and_not_the_header() {
        let payload = b"this-must-not-appear".to_vec();
        for ciph in 1..=ALG_ID_MAX {
            let s = sec(ciph, Some(2));
            let pdu = s.protect(1, DRB1, DIRECTION_DOWNLINK, &HEADER, &payload);
            assert!(
                !pdu.windows(payload.len()).any(|w| w == payload.as_slice()),
                "NEA{ciph} left the payload visible in the protected PDU"
            );
            assert_eq!(
                &pdu[..HEADER.len()],
                &HEADER,
                "NEA{ciph} ciphered the header, which the receiver must read first"
            );
        }
        // NEA0 is the null algorithm and visibly so, which is what a `not-needed`
        // confidentiality policy asks for.
        let clear =
            sec(ALG_ID_NULL, Some(2)).protect(1, DRB1, DIRECTION_DOWNLINK, &HEADER, &payload);
        assert!(clear
            .windows(payload.len())
            .any(|w| w == payload.as_slice()));
    }

    /// Two DRBs on one UE share the keys but not the BEARER, so the same payload at
    /// the same COUNT produces different bytes. Without this a PDU could be
    /// replayed from one bearer onto another.
    #[test]
    fn the_bearer_identity_separates_two_drbs() {
        let s = sec(2, Some(2));
        let drb2 = 1u8;
        let a = s.protect(3, DRB1, DIRECTION_UPLINK, &HEADER, b"data");
        let b = s.protect(3, drb2, DIRECTION_UPLINK, &HEADER, b"data");
        assert_ne!(a, b, "the BEARER must be part of the crypto inputs");
        assert_eq!(
            s.unprotect(3, drb2, DIRECTION_UPLINK, &a, HEADER.len()),
            Err(UpSecurityError::IntegrityCheckFailed),
            "a PDU must not verify on a different bearer"
        );
    }

    /// `Debug` must not print keys.
    #[test]
    fn debug_does_not_leak_the_keys() {
        let text = format!("{:?}", sec(2, Some(2)));
        assert!(text.contains("redacted"));
        assert!(
            !text.contains("0x0F") && !text.contains("15, 30"),
            "the key material must not appear: {text}"
        );
    }
}
