//! E-UTRA PDCP entity for a data radio bearer on RLC UM with a 12-bit SN
//! (TS 36.323), shared by the nextgsim eNB and LTE UE.
//!
//! # The PDU (§6.2.3)
//!
//! ```text
//! | D/C | R | R | R | PDCP SN (4 MSBs) |   oct 1
//! |       PDCP SN (8 LSBs)             |   oct 2
//! |       data (the IP packet) ...     |
//! ```
//!
//! D/C is 1 for a Data PDU (§6.3.7). A DRB carries no MAC-I: TS 36.323 §5.7
//! has integrity protection "only for SRBs" (and relay-node DRBs), so the
//! only security function here is ciphering (§5.6), of the data part with
//! COUNT = RX/TX_HFN (20 bits) ‖ SN (12 bits).
//!
//! # Receive (§5.1.2.1.3, UE; §5.2.2.2, eNB)
//!
//! For RLC UM "when the reordering function is not used" -- the only
//! configuration of a non-split DRB -- the receiver takes each PDU as it
//! comes: an SN below `Next_PDCP_RX_SN` means the HFN wrapped, the PDU is
//! deciphered with that COUNT, and `Next_PDCP_RX_SN` follows the received SN.
//! The eNB's uplink clause (§5.2.2.2) is the same text.
//!
//! # BEARER
//!
//! TS 36.323 §5.6.0 has BEARER "RB identity − 1 as in TS 36.331", and TS
//! 36.331 numbers radio bearers with SRB1 and SRB2 first. Deployed E-UTRA
//! stacks therefore derive a DRB's BEARER from its logical channel identity
//! (LCID 3 → BEARER 2, as SRB1's LCID 1 gives BEARER 0), which is what
//! [`drb_bearer`] does; both ends of a nextgsim link use it.
//!
//! Header compression is not configured (`headerCompression notUsed`), and
//! PDCP Control PDUs (status reports, interspersed ROHC feedback) do not
//! occur on a UM bearer without ROHC: one received is reported, not
//! delivered.

use crate::algorithms::{apply_ciphering, ALG_ID_MAX};

/// `Maximum_PDCP_SN` with a 12-bit SN (§7.1 b).
pub const MAX_DRB_SN12: u16 = 4095;

/// Header length of a 12-bit-SN DRB PDCP Data PDU (§6.2.3).
pub const LTE_DRB_SN12_HEADER_LEN: usize = 2;

/// BEARER of the DRB on `logical_channel_identity` (3..=10): LCID − 1 (see
/// the module documentation).
pub fn drb_bearer(logical_channel_identity: u8) -> u8 {
    logical_channel_identity.saturating_sub(1) & 0x1F
}

/// Errors receiving a DRB PDCP PDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LteDrbPdcpError {
    /// Shorter than the 2-octet header.
    #[error("DRB PDCP PDU of {0} octets is shorter than the 12-bit-SN header")]
    TooShort(usize),
    /// D/C = 0: a PDCP Control PDU, which this bearer is not configured for.
    #[error("PDCP Control PDU (PDU type {0}) on a DRB without status reports or ROHC")]
    ControlPdu(u8),
    /// An algorithm identity above EEA3.
    #[error("unknown ciphering algorithm EEA{0}")]
    UnknownAlgorithm(u8),
}

/// One direction's ciphering key. Never printed.
#[derive(Clone)]
struct Cipher {
    alg: u8,
    key: [u8; 16],
}

/// A DRB PDCP entity (§7.1 state variables), one per bearer per side.
#[derive(Clone)]
pub struct LteDrbPdcp {
    next_tx_sn: u16,
    tx_hfn: u32,
    next_rx_sn: u16,
    rx_hfn: u32,
    bearer: u8,
    tx_direction: u8,
    cipher: Option<Cipher>,
}

impl std::fmt::Debug for LteDrbPdcp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LteDrbPdcp")
            .field("bearer", &self.bearer)
            .field("next_tx_sn", &self.next_tx_sn)
            .field("next_rx_sn", &self.next_rx_sn)
            .field("eea", &self.cipher.as_ref().map(|c| c.alg))
            .finish_non_exhaustive()
    }
}

impl LteDrbPdcp {
    /// A newly established DRB entity (§7.1: all variables 0) for BEARER
    /// `bearer` that transmits in `tx_direction` (0 uplink, 1 downlink) and
    /// ciphers both directions with `eea` and `K_UPenc` (TS 33.401 §7.2.4.5:
    /// a DRB is established after AS security, so it starts ciphered; EEA0
    /// is null ciphering).
    pub fn new(
        bearer: u8,
        tx_direction: u8,
        eea: u8,
        k_up_enc: [u8; 16],
    ) -> Result<Self, LteDrbPdcpError> {
        if eea > ALG_ID_MAX {
            return Err(LteDrbPdcpError::UnknownAlgorithm(eea));
        }
        Ok(Self {
            next_tx_sn: 0,
            tx_hfn: 0,
            next_rx_sn: 0,
            rx_hfn: 0,
            bearer: bearer & 0x1F,
            tx_direction: tx_direction & 1,
            cipher: Some(Cipher {
                alg: eea,
                key: k_up_enc,
            }),
        })
    }

    /// The BEARER this entity ciphers with.
    pub fn bearer(&self) -> u8 {
        self.bearer
    }

    /// Wrap an IP packet (a PDCP SDU, §5.1.1): COUNT = TX_HFN ‖
    /// Next_PDCP_TX_SN, cipher the data part, advance the SN (TX_HFN + 1 on
    /// wrap). Returns the PDU and the COUNT it used.
    pub fn transmit(&mut self, sdu: &[u8]) -> (Vec<u8>, u32) {
        let sn = self.next_tx_sn;
        let count = (self.tx_hfn << 12) | u32::from(sn);
        let mut pdu = Vec::with_capacity(LTE_DRB_SN12_HEADER_LEN + sdu.len());
        pdu.push(0x80 | ((sn >> 8) as u8 & 0x0F)); // D/C = 1, R R R = 0
        pdu.push(sn as u8);
        pdu.extend_from_slice(sdu);
        if let Some(c) = &self.cipher {
            apply_ciphering(
                c.alg,
                &c.key,
                count,
                self.bearer,
                self.tx_direction,
                &mut pdu[LTE_DRB_SN12_HEADER_LEN..],
            );
        }
        if sn >= MAX_DRB_SN12 {
            self.next_tx_sn = 0;
            self.tx_hfn = self.tx_hfn.wrapping_add(1) & 0x000F_FFFF;
        } else {
            self.next_tx_sn = sn + 1;
        }
        (pdu, count)
    }

    /// Process a received PDCP Data PDU (§5.1.2.1.3 / §5.2.2.2). Returns the
    /// SDU and the COUNT it was deciphered with.
    pub fn receive(&mut self, pdu: &[u8]) -> Result<(Vec<u8>, u32), LteDrbPdcpError> {
        if pdu.len() < LTE_DRB_SN12_HEADER_LEN {
            return Err(LteDrbPdcpError::TooShort(pdu.len()));
        }
        if pdu[0] & 0x80 == 0 {
            return Err(LteDrbPdcpError::ControlPdu((pdu[0] >> 4) & 0x07));
        }
        let sn = (u16::from(pdu[0] & 0x0F) << 8) | u16::from(pdu[1]);
        if sn < self.next_rx_sn {
            self.rx_hfn = self.rx_hfn.wrapping_add(1) & 0x000F_FFFF;
        }
        let count = (self.rx_hfn << 12) | u32::from(sn);
        let mut sdu = pdu[LTE_DRB_SN12_HEADER_LEN..].to_vec();
        if let Some(c) = &self.cipher {
            apply_ciphering(
                c.alg,
                &c.key,
                count,
                self.bearer,
                self.tx_direction ^ 1,
                &mut sdu,
            );
        }
        self.next_rx_sn = sn + 1;
        if self.next_rx_sn > MAX_DRB_SN12 {
            self.next_rx_sn = 0;
            self.rx_hfn = self.rx_hfn.wrapping_add(1) & 0x000F_FFFF;
        }
        Ok((sdu, count))
    }

    /// Set the TX state, for tests that need a particular COUNT.
    #[cfg(test)]
    fn set_tx(&mut self, hfn: u32, sn: u16) {
        self.tx_hfn = hfn;
        self.next_tx_sn = sn;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lte_srb::{DIRECTION_DOWNLINK, DIRECTION_UPLINK};

    const KEY: [u8; 16] = [0x11; 16];

    fn pair(eea: u8) -> (LteDrbPdcp, LteDrbPdcp) {
        let b = drb_bearer(3);
        (
            LteDrbPdcp::new(b, DIRECTION_UPLINK, eea, KEY).unwrap(),
            LteDrbPdcp::new(b, DIRECTION_DOWNLINK, eea, KEY).unwrap(),
        )
    }

    #[test]
    fn header_is_dc_one_and_the_12_bit_sn() {
        // §6.2.3, by hand: SN 0x5A3 → oct 1 = 1000 0101 = 0x85, oct 2 = 0xA3.
        let (mut ue, _) = pair(0);
        ue.set_tx(0, 0x5A3);
        let (pdu, count) = ue.transmit(&[0xDE, 0xAD]);
        assert_eq!(pdu, vec![0x85, 0xA3, 0xDE, 0xAD]);
        assert_eq!(count, 0x5A3);
        // SN 0 of the first PDU: 0x80 0x00.
        let (mut ue, _) = pair(0);
        assert_eq!(&ue.transmit(&[1]).0[..2], &[0x80, 0x00]);
    }

    #[test]
    fn bearer_is_lcid_minus_one() {
        assert_eq!(drb_bearer(3), 2);
        assert_eq!(drb_bearer(10), 9);
    }

    /// TS 33.401 Annex C.1, 128-EEA2 test set 1 (= TS 33.501 NEA2 set 1):
    /// COUNT 0x398a59b4, BEARER 0x15, DIRECTION 1, 253 bits. Driven through
    /// the PDCP entity, so the COUNT/BEARER/DIRECTION wiring is what is tested.
    #[test]
    fn eea2_ciphers_with_count_bearer_direction_of_the_pdu() {
        let key: [u8; 16] = [
            0xd3, 0xc5, 0xd5, 0x92, 0x32, 0x7f, 0xb1, 0x1c, 0x40, 0x35, 0xc6, 0x68, 0x0a, 0xf8,
            0xc6, 0xd1,
        ];
        let plain: [u8; 32] = [
            0x98, 0x1b, 0xa6, 0x82, 0x4c, 0x1b, 0xfb, 0x1a, 0xb4, 0x85, 0x47, 0x20, 0x29, 0xb7,
            0x1d, 0x80, 0x8c, 0xe3, 0x3e, 0x2c, 0xc3, 0xc0, 0xb5, 0xfc, 0x1f, 0x3d, 0xe8, 0xa6,
            0xdc, 0x66, 0xb1, 0xf0,
        ];
        let cipher: [u8; 32] = [
            0xe9, 0xfe, 0xd8, 0xa6, 0x3d, 0x15, 0x53, 0x04, 0xd7, 0x1d, 0xf2, 0x0b, 0xf3, 0xe8,
            0x22, 0x14, 0xb2, 0x0e, 0xd7, 0xda, 0xd2, 0xf2, 0x33, 0xdc, 0x3c, 0x22, 0xd7, 0xbd,
            0xee, 0xed, 0x8e, 0x78,
        ];
        let mut enb = LteDrbPdcp::new(0x15, DIRECTION_DOWNLINK, 2, key).unwrap();
        enb.set_tx(0x398a59b4 >> 12, (0x398a59b4 & 0xFFF) as u16);
        let (pdu, count) = enb.transmit(&plain);
        assert_eq!(count, 0x398a59b4);
        // 253 bits: the last 3 bits of the vector are not part of it.
        assert_eq!(&pdu[2..33], &cipher[..31]);
        assert_eq!(pdu[33] & 0xF8, cipher[31] & 0xF8);
    }

    #[test]
    fn round_trip_ciphered_in_both_directions() {
        for eea in 0..=3 {
            let (mut ue, mut enb) = pair(eea);
            for i in 0..5u8 {
                let ip = vec![0x45, i, 0, 20, 9, 9, 9, 9];
                let (ul, _) = ue.transmit(&ip);
                if eea != 0 {
                    assert_ne!(&ul[2..], &ip[..], "EEA{eea} left the data in the clear");
                }
                assert_eq!(enb.receive(&ul).unwrap().0, ip);
                let (dl, _) = enb.transmit(&ip);
                assert_eq!(ue.receive(&dl).unwrap().0, ip);
            }
        }
    }

    #[test]
    fn sn_wrap_advances_the_hfn_on_both_sides() {
        let (mut ue, mut enb) = pair(2);
        ue.set_tx(0, MAX_DRB_SN12 - 1);
        enb.next_rx_sn = MAX_DRB_SN12 - 1;
        for _ in 0..4 {
            let (pdu, count) = ue.transmit(b"x");
            assert_eq!(enb.receive(&pdu).unwrap(), (b"x".to_vec(), count));
        }
        assert_eq!(ue.tx_hfn, 1);
        assert_eq!(enb.rx_hfn, 1);
    }

    #[test]
    fn a_lost_wrap_is_still_followed() {
        // §5.1.2.1.3: an SN below Next_PDCP_RX_SN is the next HFN, even when
        // the PDUs around the wrap were lost.
        let (mut ue, mut enb) = pair(2);
        enb.next_rx_sn = 4000;
        ue.set_tx(1, 7);
        let (pdu, _) = ue.transmit(b"after");
        assert_eq!(enb.receive(&pdu).unwrap().0, b"after".to_vec());
        assert_eq!(enb.rx_hfn, 1);
    }

    #[test]
    fn control_and_short_pdus_are_refused() {
        let (_, mut enb) = pair(0);
        assert_eq!(enb.receive(&[0x80]), Err(LteDrbPdcpError::TooShort(1)));
        assert_eq!(
            enb.receive(&[0x00, 0x00]),
            Err(LteDrbPdcpError::ControlPdu(0))
        );
        assert!(LteDrbPdcp::new(2, 0, 4, KEY).is_err());
    }
}
