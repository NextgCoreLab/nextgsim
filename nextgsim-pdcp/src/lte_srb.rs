//! E-UTRA PDCP entity for the signalling radio bearers SRB1 and SRB2
//! (TS 36.323), shared by the nextgsim eNB and LTE UE.
//!
//! # Why it exists before the rest of LTE PDCP
//!
//! TS 36.331 §4.2.2 maps SRB1 and SRB2 onto PDCP; only SRB0 (CCCH) bypasses it.
//! So the RRCConnectionSetupComplete that carries the UE's first NAS message is,
//! on the air, a *PDCP Data PDU* — a header, the RRC message, and a MAC-I. RLS
//! replaces the physical layer and MAC only, so this framing is real protocol,
//! not simulation scaffolding: an eNB that read a bare RRC message off SRB1
//! would be accepting a PDU no conformant UE sends. DRB PDCP (12/7-bit SN, ROHC)
//! comes with the user plane in a later increment.
//!
//! # The PDU (§6.2.2, §6.3.2, §6.3.4)
//!
//! ```text
//! | R | R | R | PDCP SN (5) |   oct 1
//! | data (the RRC message) ...    |
//! | MAC-I (32 bits)               |   last 4 octets
//! ```
//!
//! Before AS security is activated "the MAC-I field is still present and should
//! be padded with padding bits set to 0" (§6.3.4).
//!
//! # Security (§5.6, §5.7; TS 33.401 §7.2.4.5)
//!
//! - **Integrity** (§5.7): "the data unit that is integrity protected is the PDU
//!   header and the data part of the PDU before ciphering". The MAC-I is
//!   EIA*n*(`K_RRCint`, COUNT, BEARER, DIRECTION, header ‖ data).
//! - **Ciphering** (§5.6.0): "for the control plane, the data unit that is
//!   ciphered is the data part of the PDCP PDU and the MAC-I". The header stays
//!   in the clear -- the receiver needs the SN to compute COUNT.
//! - BEARER is "RB identity − 1" (§5.6.0, §5.7), so 0 for SRB1 and 1 for SRB2;
//!   DIRECTION is 0 uplink, 1 downlink (TS 33.401 §B.1.1, §B.2.1).
//!
//! The four switches -- TX/RX integrity, TX/RX ciphering -- are separate because
//! TS 33.401 §7.2.4.5 starts them at four different moments of the AS Security
//! Mode Command: the eNB ciphers downlink after *sending* the command and
//! deciphers uplink after *verifying* the complete; the UE does the mirror
//! image, and its SecurityModeComplete goes out integrity protected but
//! unciphered (TS 36.331 §5.3.4.3). One "security on" flag could not express
//! that, and getting it wrong garbles exactly one message, which is the hardest
//! defect of this kind to see.
//!
//! The algorithms are the 5G ones -- 128-EEA1/2/3 and 128-EIA1/2/3 *are*
//! 128-NEA1/2/3 and 128-NIA1/2/3 on the same COUNT/BEARER/DIRECTION inputs
//! (TS 33.501 Annex D defines NEA/NIA by reference to the EEA/EIA of TS 33.401
//! Annex B) -- so this entity uses [`crate::algorithms`] rather than a copy.
//!
//! # Receive without reordering (§5.1.2.2.1)
//!
//! SRBs run on RLC AM, which delivers in order, so the reordering function is
//! not used. The receiver tracks `Next_PDCP_RX_SN`/`RX_HFN` exactly as the
//! clause lists them, and -- as the clause orders it -- advances them only for a
//! PDU that passed integrity verification.

use crate::algorithms::{apply_ciphering, compute_mac_i, ct_eq_mac, ALG_ID_MAX};

/// `Maximum_PDCP_SN` for an SRB: 2^5 - 1 (§7.1 b; §6.3.2: 5-bit SN).
pub const MAX_SRB_SN: u8 = 31;

/// MAC-I length (§6.3.4: 32 bits).
pub const LTE_MAC_I_LEN: usize = 4;

/// Header length of an SRB PDCP Data PDU.
pub const LTE_SRB_HEADER_LEN: usize = 1;

/// DIRECTION for uplink (TS 33.401 §B.1.1).
pub const DIRECTION_UPLINK: u8 = 0;
/// DIRECTION for downlink.
pub const DIRECTION_DOWNLINK: u8 = 1;

/// BEARER of SRB1: RB identity 1 − 1 (§5.6.0).
pub const SRB1_BEARER: u8 = 0;

/// Errors receiving an SRB PDCP Data PDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LteSrbPdcpError {
    /// Shorter than the header plus MAC-I.
    #[error("SRB PDCP PDU of {0} octets is shorter than header + MAC-I (5)")]
    TooShort(usize),
    /// Integrity verification failed (§5.7: indicate to upper layers and
    /// discard). The PDU's content is not returned.
    #[error("SRB PDCP integrity verification failed (COUNT {0})")]
    IntegrityCheckFailed(u32),
    /// An algorithm identity this entity does not implement.
    #[error("unknown EPS algorithm identity {0}")]
    UnknownAlgorithm(u8),
}

/// A received SRB PDCP Data PDU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LteSrbReceived {
    /// The PDCP SDU (the RRC message), deciphered when ciphering is active.
    pub sdu: Vec<u8>,
    /// The COUNT the PDU was received with.
    pub count: u32,
    /// The PDU header octet, needed to verify a MAC-I after the fact.
    pub header: u8,
    /// The received MAC-I, deciphered.
    pub mac_i: [u8; LTE_MAC_I_LEN],
}

/// One algorithm and its key. No `Debug` that could print the key.
#[derive(Clone)]
struct AlgKey {
    alg: u8,
    key: [u8; 16],
}

impl std::fmt::Debug for AlgKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "alg {} (key redacted)", self.alg)
    }
}

/// One E-UTRA SRB PDCP entity (one direction pair), §7.1 state variables.
#[derive(Debug, Clone)]
pub struct LteSrbPdcp {
    next_tx_sn: u8,
    tx_hfn: u32,
    next_rx_sn: u8,
    rx_hfn: u32,
    bearer: u8,
    tx_direction: u8,
    tx_integrity: Option<AlgKey>,
    rx_integrity: Option<AlgKey>,
    tx_ciphering: Option<AlgKey>,
    rx_ciphering: Option<AlgKey>,
}

impl Default for LteSrbPdcp {
    fn default() -> Self {
        Self::new()
    }
}

impl PartialEq for LteSrbPdcp {
    /// State-variable equality; security configuration is compared by algorithm
    /// only, never by key.
    fn eq(&self, other: &Self) -> bool {
        let alg = |a: &Option<AlgKey>| a.as_ref().map(|k| k.alg);
        self.next_tx_sn == other.next_tx_sn
            && self.tx_hfn == other.tx_hfn
            && self.next_rx_sn == other.next_rx_sn
            && self.rx_hfn == other.rx_hfn
            && self.bearer == other.bearer
            && self.tx_direction == other.tx_direction
            && alg(&self.tx_integrity) == alg(&other.tx_integrity)
            && alg(&self.rx_integrity) == alg(&other.rx_integrity)
            && alg(&self.tx_ciphering) == alg(&other.tx_ciphering)
            && alg(&self.rx_ciphering) == alg(&other.rx_ciphering)
    }
}

impl Eq for LteSrbPdcp {}

fn check_alg(alg: u8) -> Result<(), LteSrbPdcpError> {
    if alg > ALG_ID_MAX {
        Err(LteSrbPdcpError::UnknownAlgorithm(alg))
    } else {
        Ok(())
    }
}

impl LteSrbPdcp {
    /// A newly established SRB1 entity of a UE (it transmits uplink): all four
    /// variables 0 (§7.1), no security. The eNB's twin is
    /// [`LteSrbPdcp::with_bearer`]`(SRB1_BEARER, DIRECTION_DOWNLINK)`.
    pub fn new() -> Self {
        Self::with_bearer(SRB1_BEARER, DIRECTION_UPLINK)
    }

    /// A newly established entity for the bearer `bearer` (RB identity − 1)
    /// that transmits in `tx_direction`; it receives in the other direction.
    pub fn with_bearer(bearer: u8, tx_direction: u8) -> Self {
        Self {
            next_tx_sn: 0,
            tx_hfn: 0,
            next_rx_sn: 0,
            rx_hfn: 0,
            bearer: bearer & 0x1F,
            tx_direction: tx_direction & 1,
            tx_integrity: None,
            rx_integrity: None,
            tx_ciphering: None,
            rx_ciphering: None,
        }
    }

    fn rx_direction(&self) -> u8 {
        self.tx_direction ^ 1
    }

    /// The BEARER this entity protects with.
    pub fn bearer(&self) -> u8 {
        self.bearer
    }

    /// Whether transmitted PDUs are integrity protected.
    pub fn tx_integrity_active(&self) -> bool {
        self.tx_integrity.is_some()
    }

    /// Whether transmitted PDUs are ciphered.
    pub fn tx_ciphering_active(&self) -> bool {
        self.tx_ciphering.is_some()
    }

    /// Start integrity protection of transmitted PDUs with `alg` (EIA0..3) and
    /// `K_RRCint`.
    pub fn activate_tx_integrity(&mut self, alg: u8, key: [u8; 16]) -> Result<(), LteSrbPdcpError> {
        check_alg(alg)?;
        self.tx_integrity = Some(AlgKey { alg, key });
        Ok(())
    }

    /// Start integrity verification of received PDUs.
    pub fn activate_rx_integrity(&mut self, alg: u8, key: [u8; 16]) -> Result<(), LteSrbPdcpError> {
        check_alg(alg)?;
        self.rx_integrity = Some(AlgKey { alg, key });
        Ok(())
    }

    /// Start ciphering of transmitted PDUs with `alg` (EEA0..3) and `K_RRCenc`.
    pub fn activate_tx_ciphering(&mut self, alg: u8, key: [u8; 16]) -> Result<(), LteSrbPdcpError> {
        check_alg(alg)?;
        self.tx_ciphering = Some(AlgKey { alg, key });
        Ok(())
    }

    /// Start deciphering of received PDUs.
    pub fn activate_rx_ciphering(&mut self, alg: u8, key: [u8; 16]) -> Result<(), LteSrbPdcpError> {
        check_alg(alg)?;
        self.rx_ciphering = Some(AlgKey { alg, key });
        Ok(())
    }

    /// Stop all integrity protection and ciphering: TS 36.331 §5.3.4.3 "continue
    /// using the configuration used prior to the reception of the
    /// SecurityModeCommand", which for E-UTRAN means the eNB undoes what it
    /// switched on when it sent the command. The state variables are kept.
    pub fn deactivate_security(&mut self) {
        self.tx_integrity = None;
        self.rx_integrity = None;
        self.tx_ciphering = None;
        self.rx_ciphering = None;
    }

    /// Wrap an RRC message (a PDCP SDU) for transmission (§5.1.1): associate
    /// COUNT = TX_HFN | Next_PDCP_TX_SN, compute the MAC-I over header ‖ data
    /// when integrity is active (zero padding otherwise, §6.3.4), cipher
    /// data ‖ MAC-I when ciphering is active, then advance the SN, incrementing
    /// TX_HFN on wrap. Returns the PDU and the COUNT it used.
    pub fn transmit(&mut self, sdu: &[u8]) -> (Vec<u8>, u32) {
        let sn = self.next_tx_sn;
        let count = (self.tx_hfn << 5) | u32::from(sn);
        let mut pdu = Vec::with_capacity(LTE_SRB_HEADER_LEN + sdu.len() + LTE_MAC_I_LEN);
        pdu.push(sn & 0x1F); // R R R = 0
        pdu.extend_from_slice(sdu);
        let mac = match &self.tx_integrity {
            Some(k) => compute_mac_i(k.alg, &k.key, count, self.bearer, self.tx_direction, &pdu),
            None => [0; LTE_MAC_I_LEN],
        };
        pdu.extend_from_slice(&mac);
        if let Some(k) = &self.tx_ciphering {
            apply_ciphering(
                k.alg,
                &k.key,
                count,
                self.bearer,
                self.tx_direction,
                &mut pdu[LTE_SRB_HEADER_LEN..],
            );
        }
        if sn >= MAX_SRB_SN {
            self.next_tx_sn = 0;
            self.tx_hfn = self.tx_hfn.wrapping_add(1);
        } else {
            self.next_tx_sn = sn + 1;
        }
        (pdu, count)
    }

    /// Process a received SRB PDCP Data PDU (§5.1.2.2.1): compute COUNT from the
    /// SN, decipher when deciphering is active, verify the MAC-I when integrity
    /// verification is active, and only then advance `Next_PDCP_RX_SN`/`RX_HFN`.
    ///
    /// With integrity verification not (yet) active every well-formed PDU is
    /// delivered and the MAC-I is returned unexamined -- the UE's
    /// SecurityModeCommand is exactly such a PDU, and TS 36.323 §5.7 NOTE has
    /// RRC decode it before its integrity can be checked ([`verify_mac_i`]).
    pub fn receive(&mut self, pdu: &[u8]) -> Result<LteSrbReceived, LteSrbPdcpError> {
        self.receive_inner(pdu, true)
    }

    /// Receive a PDU the sender did not protect although verification is on:
    /// the one case is the AS SecurityModeFailure, which TS 33.401 §7.2.4.5 has
    /// the UE send "unprotected" in reply to an integrity-protected command.
    /// Deciphering still applies when on; the MAC-I is not checked. The caller
    /// must accept nothing but that message from what this returns.
    pub fn receive_unprotected(&mut self, pdu: &[u8]) -> Result<LteSrbReceived, LteSrbPdcpError> {
        self.receive_inner(pdu, false)
    }

    fn receive_inner(
        &mut self,
        pdu: &[u8],
        verify: bool,
    ) -> Result<LteSrbReceived, LteSrbPdcpError> {
        if pdu.len() < LTE_SRB_HEADER_LEN + LTE_MAC_I_LEN {
            return Err(LteSrbPdcpError::TooShort(pdu.len()));
        }
        let header = pdu[0];
        let sn = header & 0x1F;
        let next_hfn = sn < self.next_rx_sn;
        let hfn = if next_hfn {
            self.rx_hfn.wrapping_add(1)
        } else {
            self.rx_hfn
        };
        let count = (hfn << 5) | u32::from(sn);
        let rx_dir = self.rx_direction();
        let mut body = pdu[LTE_SRB_HEADER_LEN..].to_vec();
        if let Some(k) = &self.rx_ciphering {
            apply_ciphering(k.alg, &k.key, count, self.bearer, rx_dir, &mut body);
        }
        let split = body.len() - LTE_MAC_I_LEN;
        let mut mac_i = [0u8; LTE_MAC_I_LEN];
        mac_i.copy_from_slice(&body[split..]);
        body.truncate(split);
        if let Some(k) = self.rx_integrity.as_ref().filter(|_| verify) {
            let mut protected = Vec::with_capacity(1 + body.len());
            protected.push(header);
            protected.extend_from_slice(&body);
            let expected = compute_mac_i(k.alg, &k.key, count, self.bearer, rx_dir, &protected);
            if !ct_eq_mac(&expected, &mac_i) {
                // §5.7: discard; the state variables stay as they were, which is
                // what §5.1.2.2.1 gets by advancing them only on success.
                return Err(LteSrbPdcpError::IntegrityCheckFailed(count));
            }
        }
        if next_hfn {
            self.rx_hfn = self.rx_hfn.wrapping_add(1);
        }
        self.next_rx_sn = sn + 1;
        if self.next_rx_sn > MAX_SRB_SN {
            self.next_rx_sn = 0;
            self.rx_hfn = self.rx_hfn.wrapping_add(1);
        }
        Ok(LteSrbReceived {
            sdu: body,
            count,
            header,
            mac_i,
        })
    }
}

/// Verify the MAC-I of an already received PDU (TS 36.323 §5.7 NOTE: the RRC
/// message activating integrity protection is decoded first and checked with
/// the configuration it carries). `direction` is the PDU's direction.
pub fn verify_mac_i(
    received: &LteSrbReceived,
    alg: u8,
    key: &[u8; 16],
    bearer: u8,
    direction: u8,
) -> bool {
    let mut protected = Vec::with_capacity(1 + received.sdu.len());
    protected.push(received.header);
    protected.extend_from_slice(&received.sdu);
    let expected = compute_mac_i(alg, key, received.count, bearer, direction, &protected);
    ct_eq_mac(&expected, &received.mac_i)
}

#[cfg(test)]
mod tests {
    use super::*;

    const K_INT: [u8; 16] = [0x11; 16];
    const K_ENC: [u8; 16] = [0x22; 16];

    fn rx(e: &mut LteSrbPdcp, pdu: &[u8]) -> (Vec<u8>, u32) {
        let r = e.receive(pdu).unwrap();
        (r.sdu, r.count)
    }

    #[test]
    fn pdu_layout_is_header_sdu_zero_mac_i() {
        let mut e = LteSrbPdcp::new();
        let (pdu, count) = e.transmit(&[0xAA, 0xBB]);
        assert_eq!(pdu, [0x00, 0xAA, 0xBB, 0, 0, 0, 0]);
        assert_eq!(count, 0);
        let (pdu, count) = e.transmit(&[0xCC]);
        assert_eq!(pdu, [0x01, 0xCC, 0, 0, 0, 0]);
        assert_eq!(count, 1);
    }

    #[test]
    fn tx_sn_wraps_at_31_and_increments_hfn() {
        let mut e = LteSrbPdcp::new();
        for i in 0..=31u32 {
            let (pdu, count) = e.transmit(&[]);
            assert_eq!(u32::from(pdu[0]), i);
            assert_eq!(count, i);
        }
        let (pdu, count) = e.transmit(&[]);
        assert_eq!(pdu[0], 0);
        assert_eq!(count, 32, "COUNT = TX_HFN(1) << 5 | SN(0)");
    }

    #[test]
    fn rx_tracks_the_transmitter_across_two_wraps() {
        let mut tx = LteSrbPdcp::new();
        let mut rxe = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
        for i in 0..70u32 {
            let sdu = i.to_be_bytes();
            let (pdu, tx_count) = tx.transmit(&sdu);
            let (got, rx_count) = rx(&mut rxe, &pdu);
            assert_eq!(got, sdu);
            assert_eq!(rx_count, tx_count, "COUNT must agree at PDU {i}");
        }
    }

    #[test]
    fn rx_sn_below_next_uses_hfn_plus_one() {
        // §5.1.2.2.1: a received SN < Next_PDCP_RX_SN belongs to the next HFN.
        let mut e = LteSrbPdcp::new();
        rx(&mut e, &[5, 0, 0, 0, 0]); // Next_PDCP_RX_SN = 6
        let (_, count) = rx(&mut e, &[2, 0, 0, 0, 0]);
        assert_eq!(count, (1 << 5) | 2);
        // ...and the receiver now lives in that HFN: "increment RX_HFN by one",
        // so the following SN 3 is COUNT 35, not 3.
        let (_, count) = rx(&mut e, &[3, 0, 0, 0, 0]);
        assert_eq!(count, (1 << 5) | 3);
    }

    #[test]
    fn reserved_bits_are_ignored_and_short_pdus_refused() {
        let mut e = LteSrbPdcp::new();
        let (sdu, count) = rx(&mut e, &[0xE3, 0x42, 1, 2, 3, 4]);
        assert_eq!((sdu, count), (vec![0x42], 3));
        assert_eq!(e.receive(&[0, 0, 0, 0]), Err(LteSrbPdcpError::TooShort(4)));
    }

    /// The UE and eNB entities of one SRB1, as TS 33.401 §7.2.4.5 leaves them
    /// after a completed AS SMC: everything on, both ways.
    fn secured_pair(int: u8, enc: u8) -> (LteSrbPdcp, LteSrbPdcp) {
        let mut ue = LteSrbPdcp::new();
        let mut enb = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
        for e in [&mut ue, &mut enb] {
            e.activate_tx_integrity(int, K_INT).unwrap();
            e.activate_rx_integrity(int, K_INT).unwrap();
            e.activate_tx_ciphering(enc, K_ENC).unwrap();
            e.activate_rx_ciphering(enc, K_ENC).unwrap();
        }
        (ue, enb)
    }

    #[test]
    fn protected_pdus_round_trip_in_both_directions_for_every_algorithm() {
        for int in 0..=3u8 {
            for enc in 0..=3u8 {
                let (mut ue, mut enb) = secured_pair(int, enc);
                for i in 0..40u8 {
                    let msg = vec![i, 0x5A, 0xA5, i ^ 0xFF];
                    let (ul, _) = ue.transmit(&msg);
                    assert_eq!(enb.receive(&ul).unwrap().sdu, msg, "EIA{int}/EEA{enc} UL");
                    let (dl, _) = enb.transmit(&msg);
                    assert_eq!(ue.receive(&dl).unwrap().sdu, msg, "EIA{int}/EEA{enc} DL");
                }
            }
        }
    }

    #[test]
    fn the_mac_covers_the_header_and_ciphering_leaves_the_header_clear() {
        let (mut ue, _) = secured_pair(2, 2);
        let (pdu, count) = ue.transmit(&[1, 2, 3]);
        assert_eq!(pdu[0], 0, "the header is not ciphered (§5.6.0)");
        // The MAC-I is EIA2 over header || data, uplink, BEARER 0.
        let mut plain = pdu.clone();
        apply_ciphering(2, &K_ENC, count, 0, DIRECTION_UPLINK, &mut plain[1..]);
        let expected = compute_mac_i(2, &K_INT, count, 0, DIRECTION_UPLINK, &[0, 1, 2, 3]);
        assert_eq!(&plain[4..], &expected);
        assert_ne!(
            compute_mac_i(2, &K_INT, count, 0, DIRECTION_UPLINK, &[1, 2, 3]),
            expected,
            "a MAC over the data alone must differ, or this test proves nothing"
        );
    }

    #[test]
    fn a_tampered_pdu_is_discarded_without_advancing_state() {
        let (mut ue, mut enb) = secured_pair(2, 0);
        let (mut pdu, count) = ue.transmit(&[9, 9, 9]);
        pdu[1] ^= 0x01;
        assert_eq!(
            enb.receive(&pdu),
            Err(LteSrbPdcpError::IntegrityCheckFailed(count))
        );
        // The genuine PDU with the same SN is still accepted: Next_PDCP_RX_SN
        // did not move past it.
        pdu[1] ^= 0x01;
        assert_eq!(enb.receive(&pdu).unwrap().sdu, vec![9, 9, 9]);
    }

    #[test]
    fn verify_mac_i_checks_a_pdu_received_before_integrity_was_on() {
        // The UE's view of the AS SecurityModeCommand: the eNB protected it, the
        // UE's entity had no integrity yet, and RRC checks it afterwards.
        let mut enb = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
        enb.activate_tx_integrity(2, K_INT).unwrap();
        let (pdu, _) = enb.transmit(&[0x30, 0x00]);
        let mut ue = LteSrbPdcp::new();
        let r = ue.receive(&pdu).unwrap();
        assert!(verify_mac_i(&r, 2, &K_INT, SRB1_BEARER, DIRECTION_DOWNLINK));
        assert!(!verify_mac_i(
            &r,
            2,
            &[0x12; 16],
            SRB1_BEARER,
            DIRECTION_DOWNLINK
        ));
        assert!(!verify_mac_i(&r, 2, &K_INT, SRB1_BEARER, DIRECTION_UPLINK));
    }

    #[test]
    fn the_smc_complete_ordering_of_33401_7_2_4_5() {
        // eNB: integrity on, SMC out, then DL ciphering on.
        let mut enb = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
        enb.activate_tx_integrity(2, K_INT).unwrap();
        enb.activate_rx_integrity(2, K_INT).unwrap();
        let (smc, _) = enb.transmit(b"SMC");
        enb.activate_tx_ciphering(2, K_ENC).unwrap();
        // UE: verifies the SMC, turns on RX both and TX integrity, sends the
        // complete unciphered, then turns on TX ciphering.
        let mut ue = LteSrbPdcp::new();
        let r = ue.receive(&smc).unwrap();
        assert!(verify_mac_i(&r, 2, &K_INT, SRB1_BEARER, DIRECTION_DOWNLINK));
        ue.activate_rx_integrity(2, K_INT).unwrap();
        ue.activate_rx_ciphering(2, K_ENC).unwrap();
        ue.activate_tx_integrity(2, K_INT).unwrap();
        let (complete, _) = ue.transmit(b"SMC-COMPLETE");
        assert_eq!(&complete[1..13], b"SMC-COMPLETE", "sent unciphered");
        ue.activate_tx_ciphering(2, K_ENC).unwrap();
        assert_eq!(enb.receive(&complete).unwrap().sdu, b"SMC-COMPLETE");
        enb.activate_rx_ciphering(2, K_ENC).unwrap();
        // From here every message is ciphered and integrity protected.
        let (dl, _) = enb.transmit(b"RECONFIG");
        assert_ne!(&dl[1..9], b"RECONFIG");
        assert_eq!(ue.receive(&dl).unwrap().sdu, b"RECONFIG");
        let (ul, _) = ue.transmit(b"COMPLETE");
        assert_ne!(&ul[1..9], b"COMPLETE");
        assert_eq!(enb.receive(&ul).unwrap().sdu, b"COMPLETE");
    }

    #[test]
    fn an_unprotected_pdu_fails_verification_but_receive_unprotected_reads_it() {
        let mut enb = LteSrbPdcp::with_bearer(SRB1_BEARER, DIRECTION_DOWNLINK);
        enb.activate_rx_integrity(2, K_INT).unwrap();
        let mut ue = LteSrbPdcp::new();
        let (failure, count) = ue.transmit(b"SMF");
        assert_eq!(
            enb.clone().receive(&failure),
            Err(LteSrbPdcpError::IntegrityCheckFailed(count))
        );
        assert_eq!(enb.receive_unprotected(&failure).unwrap().sdu, b"SMF");
    }

    #[test]
    fn unknown_algorithms_are_refused() {
        let mut e = LteSrbPdcp::new();
        assert_eq!(
            e.activate_tx_integrity(4, K_INT),
            Err(LteSrbPdcpError::UnknownAlgorithm(4))
        );
        assert!(!e.tx_integrity_active());
    }
}
