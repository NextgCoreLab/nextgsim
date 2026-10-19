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
//! be padded with padding bits set to 0" (§6.3.4). Ciphering and integrity
//! (§5.6, §5.7, TS 33.401) arrive with the Security Mode procedure; the COUNT
//! they need is already maintained here, which is why the HFN is tracked now.
//!
//! # Receive without reordering (§5.1.2.2.1)
//!
//! SRBs run on RLC AM, which delivers in order, so the reordering function is
//! not used. The receiver tracks `Next_PDCP_RX_SN`/`RX_HFN` exactly as the
//! clause lists them.

/// `Maximum_PDCP_SN` for an SRB: 2^5 - 1 (§7.1 b; §6.3.2: 5-bit SN).
pub const MAX_SRB_SN: u8 = 31;

/// MAC-I length (§6.3.4: 32 bits).
pub const LTE_MAC_I_LEN: usize = 4;

/// Header length of an SRB PDCP Data PDU.
pub const LTE_SRB_HEADER_LEN: usize = 1;

/// Errors receiving an SRB PDCP Data PDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LteSrbPdcpError {
    /// Shorter than the header plus MAC-I.
    #[error("SRB PDCP PDU of {0} octets is shorter than header + MAC-I (5)")]
    TooShort(usize),
}

/// One E-UTRA SRB PDCP entity (one direction pair), §7.1 state variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LteSrbPdcp {
    next_tx_sn: u8,
    tx_hfn: u32,
    next_rx_sn: u8,
    rx_hfn: u32,
}

impl Default for LteSrbPdcp {
    fn default() -> Self {
        Self::new()
    }
}

impl LteSrbPdcp {
    /// A newly established entity: all four variables 0 (§7.1).
    pub fn new() -> Self {
        Self {
            next_tx_sn: 0,
            tx_hfn: 0,
            next_rx_sn: 0,
            rx_hfn: 0,
        }
    }

    /// Wrap an RRC message (a PDCP SDU) for transmission (§5.1.1): associate
    /// COUNT = TX_HFN | Next_PDCP_TX_SN, form the PDU, then advance the SN,
    /// incrementing TX_HFN on wrap. Returns the PDU and the COUNT it used.
    pub fn transmit(&mut self, sdu: &[u8]) -> (Vec<u8>, u32) {
        let sn = self.next_tx_sn;
        let count = (self.tx_hfn << 5) | u32::from(sn);
        let mut pdu = Vec::with_capacity(LTE_SRB_HEADER_LEN + sdu.len() + LTE_MAC_I_LEN);
        pdu.push(sn & 0x1F); // R R R = 0
        pdu.extend_from_slice(sdu);
        // §6.3.4: no integrity protection yet, so MAC-I is zero padding.
        pdu.extend_from_slice(&[0; LTE_MAC_I_LEN]);
        if sn >= MAX_SRB_SN {
            self.next_tx_sn = 0;
            self.tx_hfn = self.tx_hfn.wrapping_add(1);
        } else {
            self.next_tx_sn = sn + 1;
        }
        (pdu, count)
    }

    /// Process a received SRB PDCP Data PDU (§5.1.2.2.1) and return the SDU and
    /// the COUNT it was received with. Integrity verification is not yet
    /// applicable, so every well-formed PDU is delivered; the MAC-I is not
    /// examined (§6.3.4 says it "should be" zero, which does not make a non-zero
    /// value a reason to discard).
    pub fn receive(&mut self, pdu: &[u8]) -> Result<(Vec<u8>, u32), LteSrbPdcpError> {
        if pdu.len() < LTE_SRB_HEADER_LEN + LTE_MAC_I_LEN {
            return Err(LteSrbPdcpError::TooShort(pdu.len()));
        }
        let sn = pdu[0] & 0x1F;
        let hfn = if sn < self.next_rx_sn {
            self.rx_hfn.wrapping_add(1)
        } else {
            self.rx_hfn
        };
        let count = (hfn << 5) | u32::from(sn);
        // "integrity verification is not applicable":
        if sn < self.next_rx_sn {
            self.rx_hfn = self.rx_hfn.wrapping_add(1);
        }
        self.next_rx_sn = sn + 1;
        if self.next_rx_sn > MAX_SRB_SN {
            self.next_rx_sn = 0;
            self.rx_hfn = self.rx_hfn.wrapping_add(1);
        }
        let sdu = pdu[LTE_SRB_HEADER_LEN..pdu.len() - LTE_MAC_I_LEN].to_vec();
        Ok((sdu, count))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let mut rx = LteSrbPdcp::new();
        for i in 0..70u32 {
            let sdu = i.to_be_bytes();
            let (pdu, tx_count) = tx.transmit(&sdu);
            let (got, rx_count) = rx.receive(&pdu).unwrap();
            assert_eq!(got, sdu);
            assert_eq!(rx_count, tx_count, "COUNT must agree at PDU {i}");
        }
    }

    #[test]
    fn rx_sn_below_next_uses_hfn_plus_one() {
        // §5.1.2.2.1: a received SN < Next_PDCP_RX_SN belongs to the next HFN.
        let mut rx = LteSrbPdcp::new();
        rx.receive(&[5, 0, 0, 0, 0]).unwrap(); // Next_PDCP_RX_SN = 6
        let (_, count) = rx.receive(&[2, 0, 0, 0, 0]).unwrap();
        assert_eq!(count, (1 << 5) | 2);
        // ...and the receiver now lives in that HFN: "increment RX_HFN by one",
        // so the following SN 3 is COUNT 35, not 3.
        let (_, count) = rx.receive(&[3, 0, 0, 0, 0]).unwrap();
        assert_eq!(count, (1 << 5) | 3);
    }

    #[test]
    fn reserved_bits_are_ignored_and_short_pdus_refused() {
        let mut rx = LteSrbPdcp::new();
        let (sdu, count) = rx.receive(&[0xE3, 0x42, 1, 2, 3, 4]).unwrap();
        assert_eq!((sdu, count), (vec![0x42], 3));
        assert_eq!(rx.receive(&[0, 0, 0, 0]), Err(LteSrbPdcpError::TooShort(4)));
    }
}
