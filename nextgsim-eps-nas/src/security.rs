//! The UE's EPS NAS security context and security-protected NAS messages
//! (TS 24.301 §4.4, §9.3.1; TS 33.401 §7.2.3, Annex A.7, Annex B).
//!
//! A protected message is `security header type | PD` (one octet, §9.3.1),
//! the 4-octet MAC, the sequence number (the low 8 bits of NAS COUNT, §9.6),
//! then the NAS message -- ciphered for header types 2 and 4. The MAC is
//! computed over `sequence number ‖ (ciphered) message` with COUNT, BEARER = 0
//! and DIRECTION 0 uplink / 1 downlink (§4.4.3.3, TS 33.401 §B.2.1); ciphering
//! uses the same COUNT, BEARER and DIRECTION (§4.4.5, TS 33.401 §B.1.1).
//!
//! NAS COUNT is `overflow (16 bits) ‖ sequence number (8 bits)` (§4.4.3.1);
//! the 32-bit value the algorithms take is `0x00 ‖ NAS COUNT`.

use nextgsim_crypto::eps_kdf::{derive_algorithm_key, derive_kenb, EpsAlgorithmType};
use nextgsim_crypto::nea::nea1_encrypt;
use nextgsim_crypto::nea::nea2_encrypt;
use nextgsim_crypto::nia::{nia1_compute_mac, nia2_compute_mac, nia3_compute_mac};
use nextgsim_crypto::zuc::nea3_encrypt;

use crate::ie::NasKsi;

/// NAS BEARER: "set to the constant value 0" for EPS NAS (TS 33.401 §B.1.1,
/// §B.2.1 with TS 24.301 §4.4.3.3).
pub const NAS_BEARER: u8 = 0;
/// DIRECTION, uplink.
pub const UPLINK: u8 = 0;
/// DIRECTION, downlink.
pub const DOWNLINK: u8 = 1;

/// Security header types (§9.3.1).
pub mod header_type {
    /// Plain NAS message, not security protected.
    pub const PLAIN: u8 = 0;
    /// Integrity protected.
    pub const INTEGRITY: u8 = 1;
    /// Integrity protected and ciphered.
    pub const INTEGRITY_CIPHERED: u8 = 2;
    /// Integrity protected with new EPS security context.
    pub const INTEGRITY_NEW_CONTEXT: u8 = 3;
    /// Integrity protected and ciphered with new EPS security context.
    pub const INTEGRITY_CIPHERED_NEW_CONTEXT: u8 = 4;
    /// Security header for the SERVICE REQUEST message (§9.3.1, 1100).
    pub const SERVICE_REQUEST: u8 = 12;
}

/// Errors processing a security-protected NAS message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NasSecurityError {
    /// Shorter than the 6-octet security header.
    #[error("security protected NAS message of {0} octets")]
    TooShort(usize),
    /// The MAC did not verify (§4.4.4.3: discard).
    #[error("NAS MAC verification failed")]
    MacFailure,
}

/// A 128-bit NAS key. No `Debug` that could print it.
#[derive(Clone)]
struct Key128([u8; 16]);

impl std::fmt::Debug for Key128 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A 256-bit key.
#[derive(Clone)]
struct Key256([u8; 32]);

impl std::fmt::Debug for Key256 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("<redacted>")
    }
}

/// A full native EPS security context (§4.4.2.1).
#[derive(Debug, Clone)]
pub struct EpsSecurityContext {
    ksi: NasKsi,
    kasme: Key256,
    eea: u8,
    eia: u8,
    k_nas_enc: Key128,
    k_nas_int: Key128,
    /// Next uplink NAS COUNT.
    ul_count: u32,
    /// Highest downlink NAS COUNT accepted, if any.
    dl_count: Option<u32>,
    /// Uplink NAS COUNT of the most recent SECURITY MODE COMPLETE, the
    /// freshness parameter of `K_eNB` (TS 33.401 §7.2.6.2).
    kenb_ul_count: Option<u32>,
}

fn mac(eia: u8, key: &[u8; 16], count: u32, direction: u8, data: &[u8]) -> [u8; 4] {
    match eia {
        1 => nia1_compute_mac(count, NAS_BEARER, direction, key, data),
        2 => nia2_compute_mac(count, NAS_BEARER, direction, key, data),
        3 => nia3_compute_mac(count, NAS_BEARER, direction, key, data),
        // EIA0: "the MAC shall be set to all zeros" (TS 33.401 §5.1.4.1 via
        // the null algorithm's definition).
        _ => [0; 4],
    }
}

fn cipher(eea: u8, key: &[u8; 16], count: u32, direction: u8, data: &mut [u8]) {
    match eea {
        1 => nea1_encrypt(count, NAS_BEARER, direction, key, data),
        2 => nea2_encrypt(count, NAS_BEARER, direction, key, data),
        3 => nea3_encrypt(count, NAS_BEARER, direction, key, data),
        _ => {}
    }
}

/// The 24-bit NAS COUNT wraps (§4.4.3.1).
const NAS_COUNT_MASK: u32 = 0x00FF_FFFF;

impl EpsSecurityContext {
    /// A new context from `K_ASME` for the algorithms a SECURITY MODE COMMAND
    /// selected: NAS keys per Annex A.7, both NAS COUNTs at their start value 0
    /// (§4.4.3.1: "...set to zero when a new EPS security context is taken into
    /// use" after AKA).
    pub fn new(ksi: NasKsi, kasme: [u8; 32], eea: u8, eia: u8) -> Self {
        Self {
            ksi,
            k_nas_enc: Key128(derive_algorithm_key(&kasme, EpsAlgorithmType::NasEnc, eea)),
            k_nas_int: Key128(derive_algorithm_key(&kasme, EpsAlgorithmType::NasInt, eia)),
            kasme: Key256(kasme),
            eea,
            eia,
            ul_count: 0,
            dl_count: None,
            kenb_ul_count: None,
        }
    }

    /// The KSI this context is identified by.
    pub fn ksi(&self) -> NasKsi {
        self.ksi
    }

    /// The selected EEA.
    pub fn eea(&self) -> u8 {
        self.eea
    }

    /// The selected EIA.
    pub fn eia(&self) -> u8 {
        self.eia
    }

    /// The next uplink NAS COUNT.
    pub fn ul_count(&self) -> u32 {
        self.ul_count
    }

    /// Record that the message about to be protected is a SECURITY MODE
    /// COMPLETE, whose uplink NAS COUNT then keys `K_eNB`.
    pub fn mark_security_mode_complete(&mut self) {
        self.kenb_ul_count = Some(self.ul_count);
    }

    /// `K_eNB` per Annex A.3 with the freshness rule of TS 33.401 §7.2.6.2:
    /// the uplink NAS COUNT of the most recent SECURITY MODE COMPLETE, or --
    /// with no NAS SMC in this context -- that of the last uplink message.
    pub fn k_enb(&self) -> [u8; 32] {
        let count = self
            .kenb_ul_count
            .unwrap_or_else(|| self.ul_count.wrapping_sub(1) & NAS_COUNT_MASK);
        derive_kenb(&self.kasme.0, count)
    }

    /// SERVICE REQUEST (§8.2.25): the security header type 12 octet, the KSI
    /// and the 5 least significant bits of the uplink NAS COUNT (§9.9.3.19),
    /// and the short MAC -- the 2 least significant octets of the NAS-MAC
    /// computed over the first two octets with that COUNT (§9.9.3.28). The
    /// message consumes the COUNT, which also keys `K_eNB` for the connection
    /// it establishes (TS 33.401 §7.2.6.2 / A.3: the uplink NAS COUNT of the
    /// SERVICE REQUEST).
    pub fn service_request(&mut self) -> Vec<u8> {
        let count = self.ul_count;
        let first = (header_type::SERVICE_REQUEST << 4) | crate::msg::PD_EMM;
        let second = ((self.ksi.value & 0x07) << 5) | (count & 0x1F) as u8;
        let m = mac(self.eia, &self.k_nas_int.0, count, UPLINK, &[first, second]);
        self.kenb_ul_count = Some(count);
        self.ul_count = (count + 1) & NAS_COUNT_MASK;
        vec![first, second, m[2], m[3]]
    }

    /// Protect an uplink NAS message with security header type `ht` (1..4).
    pub fn protect(&mut self, plain: &[u8], ht: u8) -> Vec<u8> {
        let count = self.ul_count;
        let sn = (count & 0xFF) as u8;
        let mut body = plain.to_vec();
        let ciphered = matches!(
            ht,
            header_type::INTEGRITY_CIPHERED | header_type::INTEGRITY_CIPHERED_NEW_CONTEXT
        );
        if ciphered {
            cipher(self.eea, &self.k_nas_enc.0, count, UPLINK, &mut body);
        }
        let mut mac_input = Vec::with_capacity(1 + body.len());
        mac_input.push(sn);
        mac_input.extend_from_slice(&body);
        let m = mac(self.eia, &self.k_nas_int.0, count, UPLINK, &mac_input);
        let mut out = Vec::with_capacity(6 + body.len());
        out.push((ht << 4) | crate::msg::PD_EMM);
        out.extend_from_slice(&m);
        out.push(sn);
        out.extend_from_slice(&body);
        self.ul_count = (count + 1) & NAS_COUNT_MASK;
        out
    }

    /// The downlink NAS COUNT a received sequence number belongs to
    /// (§4.4.3.1): the stored overflow, plus one when the sequence number is
    /// not above the last one received.
    fn estimate_dl_count(&self, sn: u8) -> u32 {
        match self.dl_count {
            None => u32::from(sn),
            Some(last) => {
                let overflow = last >> 8;
                let candidate = (overflow << 8) | u32::from(sn);
                if candidate > last {
                    candidate
                } else {
                    ((overflow + 1) << 8 | u32::from(sn)) & NAS_COUNT_MASK
                }
            }
        }
    }

    /// Verify and, for header types 2 and 4, decipher a downlink protected
    /// message. Returns the plain NAS message. A message that fails the MAC is
    /// rejected without touching the context (§4.4.4.3).
    pub fn unprotect(&mut self, pdu: &[u8]) -> Result<Vec<u8>, NasSecurityError> {
        if pdu.len() < 6 {
            return Err(NasSecurityError::TooShort(pdu.len()));
        }
        let ht = pdu[0] >> 4;
        let received_mac = &pdu[1..5];
        let sn = pdu[5];
        // A replayed PDU is estimated into the next overflow (§4.4.3.1), so it
        // fails the MAC below -- the replay protection of §4.4.3.2.
        let count = self.estimate_dl_count(sn);
        let expected = mac(self.eia, &self.k_nas_int.0, count, DOWNLINK, &pdu[5..]);
        // EIA0 produces zeros on both sides; any other mismatch is a failure.
        if expected != received_mac {
            return Err(NasSecurityError::MacFailure);
        }
        let mut body = pdu[6..].to_vec();
        if matches!(
            ht,
            header_type::INTEGRITY_CIPHERED | header_type::INTEGRITY_CIPHERED_NEW_CONTEXT
        ) {
            cipher(self.eea, &self.k_nas_enc.0, count, DOWNLINK, &mut body);
        }
        self.dl_count = Some(count);
        Ok(body)
    }
}

/// The security header type of a NAS PDU (bits 8-5 of octet 1), for an EMM
/// PDU; ESM messages carry their EPS bearer identity there instead.
pub fn security_header_type(pdu: &[u8]) -> Option<u8> {
    let first = *pdu.first()?;
    (first & 0x0F == crate::msg::PD_EMM).then_some(first >> 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(eea: u8, eia: u8) -> EpsSecurityContext {
        EpsSecurityContext::new(
            NasKsi {
                mapped: false,
                value: 1,
            },
            [0x5A; 32],
            eea,
            eia,
        )
    }

    /// What an MME does to protect a downlink message: the mirror image of
    /// [`EpsSecurityContext::protect`], written out independently here.
    fn mme_protect(c: &EpsSecurityContext, plain: &[u8], ht: u8, count: u32) -> Vec<u8> {
        let mut body = plain.to_vec();
        if ht == 2 || ht == 4 {
            cipher(c.eea, &c.k_nas_enc.0, count, DOWNLINK, &mut body);
        }
        let mut input = vec![(count & 0xFF) as u8];
        input.extend_from_slice(&body);
        let m = mac(c.eia, &c.k_nas_int.0, count, DOWNLINK, &input);
        let mut out = vec![(ht << 4) | 7];
        out.extend_from_slice(&m);
        out.extend_from_slice(&input);
        out
    }

    #[test]
    fn uplink_layout_is_header_mac_sn_body() {
        let mut c = ctx(0, 2);
        let p = c.protect(&[0x07, 0x5E], header_type::INTEGRITY_CIPHERED_NEW_CONTEXT);
        assert_eq!(p[0], 0x47);
        assert_eq!(p[5], 0, "first uplink sequence number");
        assert_eq!(&p[6..], &[0x07, 0x5E], "EEA0 leaves the message in clear");
        let expected = nia2_compute_mac(0, 0, UPLINK, &c.k_nas_int.0, &[0x00, 0x07, 0x5E]);
        assert_eq!(&p[1..5], &expected);
        assert_eq!(c.ul_count(), 1);
    }

    #[test]
    fn ciphered_uplink_hides_the_message_and_the_mac_covers_the_ciphertext() {
        let mut c = ctx(2, 2);
        let p = c.protect(&[0x07, 0x43, 0x00, 0x00], header_type::INTEGRITY_CIPHERED);
        assert_ne!(&p[6..], &[0x07, 0x43, 0x00, 0x00]);
        let expected = nia2_compute_mac(0, 0, UPLINK, &c.k_nas_int.0, &p[5..]);
        assert_eq!(&p[1..5], &expected);
    }

    #[test]
    fn downlink_messages_verify_decipher_and_reject_replays_and_forgeries() {
        for (eea, eia) in [(0, 1), (1, 2), (2, 2), (3, 3), (2, 0)] {
            let mut c = ctx(eea, eia);
            let plain = [0x07, 0x42, 0x01, 0x02, 0x03];
            let pdu = mme_protect(&c, &plain, 2, 0);
            assert_eq!(c.unprotect(&pdu).unwrap(), plain, "EEA{eea}/EIA{eia}");
            if eia != 0 {
                assert_eq!(
                    c.unprotect(&pdu),
                    Err(NasSecurityError::MacFailure),
                    "replay"
                );
            }
            if eia != 0 {
                let mut forged = mme_protect(&c, &plain, 2, 1);
                forged[8] ^= 1;
                assert_eq!(c.unprotect(&forged), Err(NasSecurityError::MacFailure));
            }
        }
    }

    #[test]
    fn downlink_count_overflows_after_sequence_255() {
        let mut c = ctx(0, 2);
        for count in 0..300u32 {
            let pdu = mme_protect(&c, &[0x07, 0x61], 1, count);
            c.unprotect(&pdu)
                .unwrap_or_else(|e| panic!("COUNT {count}: {e}"));
        }
    }

    #[test]
    fn kenb_is_keyed_on_the_security_mode_complete_count() {
        let mut c = ctx(0, 2);
        c.mark_security_mode_complete();
        let _ = c.protect(&[0x07, 0x5E], header_type::INTEGRITY_CIPHERED_NEW_CONTEXT);
        let _ = c.protect(&[0x07, 0x43, 0, 0], header_type::INTEGRITY_CIPHERED);
        assert_eq!(c.k_enb(), derive_kenb(&[0x5A; 32], 0));
    }
}
