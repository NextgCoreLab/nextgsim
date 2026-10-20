//! The USIM's side of EPS AKA (TS 33.401 §6.1.1, TS 33.102 §6.3.3) over
//! MILENAGE (TS 35.206).
//!
//! On AUTHENTICATION REQUEST the USIM recovers SQN from `AUTN = SQN ⊕ AK ‖ AMF
//! ‖ MAC`, checks MAC = f1(K, SQN, RAND, AMF) and the freshness of SQN, and
//! answers RES = f2 with CK = f3 and IK = f4. The ME then checks the AMF
//! "separation bit" (TS 33.401 §6.1.1, Annex H: bit 0 of AMF must be 1 for
//! E-UTRAN) and computes `K_ASME` (Annex A.2).

use nextgsim_crypto::eps_kdf::derive_kasme;
use nextgsim_crypto::milenage::Milenage;

/// The subscriber's long-term credentials.
#[derive(Clone)]
pub struct UsimCredentials {
    /// K.
    pub k: [u8; 16],
    /// OPc (already derived from OP when the operator gave OP).
    pub opc: [u8; 16],
}

impl std::fmt::Debug for UsimCredentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("UsimCredentials { <redacted> }")
    }
}

/// The outcome of processing an AUTHENTICATION REQUEST.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AkaOutcome {
    /// The network is authenticated: send RES, take `K_ASME` into use.
    Success {
        /// RES.
        res: [u8; 8],
        /// `K_ASME` (redacted in logs by the caller's discipline: never print).
        kasme: [u8; 32],
    },
    /// MAC ≠ XMAC: EMM cause #20 (TS 24.301 §5.4.2.6).
    MacFailure,
    /// SQN out of range: EMM cause #21 with AUTS (TS 33.102 §6.3.3).
    SynchFailure {
        /// `AUTS = SQN_MS ⊕ AK* ‖ MAC-S`.
        auts: [u8; 14],
    },
    /// The AMF separation bit is 0: EMM cause #26 (TS 33.401 §6.1.1).
    NonEpsAuthenticationUnacceptable,
}

/// The USIM's sequence number state (TS 33.102 Annex C).
///
/// This keeps the highest accepted SQN (`SQN_MS`) and accepts a SQN that is
/// strictly greater -- the freshness condition of Annex C.2, without the
/// IND-indexed array a single simulated USIM talking to one HSS does not need.
/// The optional wrap-around limit Δ is not applied: an HSS is free to start
/// its SQN anywhere, and refusing a first vector because it is far above the
/// USIM's initial 0 would force a re-synchronisation the standard does not
/// require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SqnState {
    /// `SQN_MS`.
    pub highest: u64,
}

fn sqn_bytes(v: u64) -> [u8; 6] {
    let b = v.to_be_bytes();
    [b[2], b[3], b[4], b[5], b[6], b[7]]
}

fn sqn_value(b: &[u8; 6]) -> u64 {
    b.iter().fold(0u64, |acc, x| (acc << 8) | u64::from(*x))
}

/// Process RAND and AUTN. `sn_id` is the serving network's MCC/MNC octets
/// (Figure A.2-1). On success the SQN state advances.
pub fn authenticate(
    creds: &UsimCredentials,
    sqn: &mut SqnState,
    rand: &[u8; 16],
    autn: &[u8; 16],
    sn_id: &[u8; 3],
) -> AkaOutcome {
    let m = Milenage::new(&creds.k, &creds.opc);
    let sqn_xor_ak: [u8; 6] = autn[..6].try_into().expect("6 octets");
    let amf: [u8; 2] = autn[6..8].try_into().expect("2 octets");
    let mac: [u8; 8] = autn[8..].try_into().expect("8 octets");
    let ak = m.f5(rand);
    let mut sqn_net = [0u8; 6];
    for i in 0..6 {
        sqn_net[i] = sqn_xor_ak[i] ^ ak[i];
    }
    // TS 33.102 §6.3.3: MAC first; a failed MAC says nothing about SQN.
    if m.f1(rand, &sqn_net, &amf) != mac {
        return AkaOutcome::MacFailure;
    }
    let received = sqn_value(&sqn_net);
    if received <= sqn.highest {
        // AUTS with the dummy AMF of all zeros (TS 33.102 §6.3.3).
        let sqn_ms = sqn_bytes(sqn.highest);
        let ak_star = m.f5_star(rand);
        let mac_s = m.f1_star(rand, &sqn_ms, &[0, 0]);
        let mut auts = [0u8; 14];
        for i in 0..6 {
            auts[i] = sqn_ms[i] ^ ak_star[i];
        }
        auts[6..].copy_from_slice(&mac_s);
        return AkaOutcome::SynchFailure { auts };
    }
    // TS 33.401 §6.1.1: "the ME shall check that the 'separation bit' in the
    // AMF field of AUTN is set to 1"; bit 0 is the most significant bit
    // (TS 33.102 Annex H numbering).
    if amf[0] & 0x80 == 0 {
        return AkaOutcome::NonEpsAuthenticationUnacceptable;
    }
    sqn.highest = received;
    let res = m.f2(rand);
    let ck = m.f3(rand);
    let ik = m.f4(rand);
    AkaOutcome::Success {
        res,
        kasme: derive_kasme(&ck, &ik, sn_id, &sqn_xor_ak),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex<const N: usize>(s: &str) -> [u8; N] {
        let v: Vec<u8> = (0..N)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect();
        v.try_into().unwrap()
    }

    // TS 35.208 §4.3 Test Set 1 (identical to TS 35.207 Test Set 1).
    fn set1() -> (UsimCredentials, [u8; 16], [u8; 6], [u8; 2]) {
        (
            UsimCredentials {
                k: hex("465b5ce8b199b49faa5f0a2ee238a6bc"),
                opc: hex("cd63cb71954a9f4e48a5994e37a02baf"),
            },
            hex("23553cbe9637a89d218ae64dae47bf35"),
            hex("ff9bb4d0b607"),
            hex("b9b9"),
        )
    }

    /// AUTN as the HSS builds it, from the Test Set 1 outputs (f1 = MAC-A
    /// 4a9ffac354dfafb3, f5 = AK aa689c648370), not from this code.
    fn autn_set1() -> [u8; 16] {
        let sqn: [u8; 6] = hex("ff9bb4d0b607");
        let ak: [u8; 6] = hex("aa689c648370");
        let mut a = [0u8; 16];
        for i in 0..6 {
            a[i] = sqn[i] ^ ak[i];
        }
        a[6..8].copy_from_slice(&hex::<2>("b9b9"));
        a[8..].copy_from_slice(&hex::<8>("4a9ffac354dfafb3"));
        a
    }

    #[test]
    fn test_set_1_authenticates_and_res_is_f2() {
        let (c, rand, _, _) = set1();
        let mut sqn = SqnState { highest: 0 };
        let out = authenticate(&c, &mut sqn, &rand, &autn_set1(), &[0x99, 0xF9, 0x07]);
        let AkaOutcome::Success { res, kasme } = out else {
            panic!("expected success, got {out:?}");
        };
        // f2 of Test Set 1.
        assert_eq!(res, hex::<8>("a54211d5e3ba50bf"));
        // K_ASME over the Test Set 1 CK (f3) and IK (f4).
        let ck: [u8; 16] = hex("b40ba9a3c58b2a05bbf0d987b21bf8cb");
        let ik: [u8; 16] = hex("f769bcd751044604127672711c6d3441");
        let mut sqn_xor_ak = [0u8; 6];
        sqn_xor_ak.copy_from_slice(&autn_set1()[..6]);
        assert_eq!(
            kasme,
            derive_kasme(&ck, &ik, &[0x99, 0xF9, 0x07], &sqn_xor_ak)
        );
        assert_eq!(sqn.highest, 0xff9b_b4d0_b607);
    }

    #[test]
    fn a_corrupted_mac_is_a_mac_failure() {
        let (c, rand, _, _) = set1();
        let mut a = autn_set1();
        a[15] ^= 1;
        let mut sqn = SqnState { highest: 0 };
        assert_eq!(
            authenticate(&c, &mut sqn, &rand, &a, &[0x99, 0xF9, 0x07]),
            AkaOutcome::MacFailure
        );
        assert_eq!(sqn.highest, 0, "a failed MAC does not move SQN_MS");
    }

    #[test]
    fn a_replayed_sqn_is_a_synch_failure_with_auts() {
        let (c, rand, _, _) = set1();
        let mut sqn = SqnState {
            highest: 0xff9b_b4d0_b607,
        };
        let out = authenticate(&c, &mut sqn, &rand, &autn_set1(), &[0x99, 0xF9, 0x07]);
        let AkaOutcome::SynchFailure { auts } = out else {
            panic!("expected synch failure, got {out:?}");
        };
        // SQN_MS ⊕ AK* with f5* of Test Set 1 (451e8beca43b).
        let ak_star: [u8; 6] = hex("451e8beca43b");
        let sqn_ms: [u8; 6] = hex("ff9bb4d0b607");
        for i in 0..6 {
            assert_eq!(auts[i], sqn_ms[i] ^ ak_star[i]);
        }
    }

    #[test]
    fn a_zero_separation_bit_is_refused_for_eps() {
        let (c, rand, _, _) = set1();
        // Rebuild AUTN with AMF 0x3939 (separation bit 0) and its own MAC.
        let m = Milenage::new(&c.k, &c.opc);
        let sqn: [u8; 6] = hex("ff9bb4d0b607");
        let ak = m.f5(&rand);
        let mut a = [0u8; 16];
        for i in 0..6 {
            a[i] = sqn[i] ^ ak[i];
        }
        a[6..8].copy_from_slice(&[0x39, 0x39]);
        a[8..].copy_from_slice(&m.f1(&rand, &sqn, &[0x39, 0x39]));
        let mut st = SqnState { highest: 0 };
        assert_eq!(
            authenticate(&c, &mut st, &rand, &a, &[0x99, 0xF9, 0x07]),
            AkaOutcome::NonEpsAuthenticationUnacceptable
        );
    }
}
