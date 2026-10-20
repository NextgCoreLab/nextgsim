//! E-UTRA RLC unacknowledged mode with a 10-bit SN (TS 36.322), for the
//! DRBs of the nextgsim eNB and LTE UE.
//!
//! # UMD PDU (§6.2.1.3)
//!
//! ```text
//! | R1 | R1 | R1 | FI (2) | E | SN (2 MSBs) |   oct 1
//! |              SN (8 LSBs)                |   oct 2
//! | E | LI (11) | E | LI (11) | ... [pad 4] |   only when E = 1
//! | data field 1 | data field 2 | ...       |
//! ```
//!
//! FI (§6.2.2.6) says whether the first data field starts an SDU and whether
//! the last one ends it; each E bit says whether another E/LI pair follows,
//! and the LIs give the lengths of all data fields but the last (§6.2.2.4,
//! §6.2.2.5). An odd number of LIs is followed by 4 padding bits.
//!
//! # Transmit (§5.1.2.1)
//!
//! The transmitter "segments and/or concatenates the RLC SDUs ... so that
//! the UMD PDUs ... fit within the total size of RLC PDU(s) indicated by lower
//! layer". RLS, which replaces MAC and the physical layer, indicates no
//! transport block size: every SDU fits one PDU, so each SDU is sent as one
//! UMD PDU with FI = 00 and E = 0, and `VT(US)` advances by one.
//!
//! # Receive (§5.1.2.2)
//!
//! The receiver is the full clause, because it must accept what any
//! conformant transmitter sends: the reordering window of `UM_Window_Size`
//! 512 (§7.2) keyed on `VR(UH)`, duplicate and late-PDU discard, `VR(UR)`
//! and `VR(UX)`, t-Reordering (§5.1.2.2.3, §5.1.2.2.4), and reassembly
//! across segmented and concatenated PDUs, discarding the parts of SDUs that
//! lost a segment (§5.1.2.2.3 "reassemble RLC SDUs ... if not delivered
//! before"; §4.2.1.2.3). All SN comparisons are made modulo 1024 relative to
//! `VR(UH) − UM_Window_Size` (§7.1).

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

/// SN modulus with a 10-bit SN.
pub const UM_SN10_MODULUS: u16 = 1024;
/// `UM_Window_Size` with a 10-bit SN (§7.2).
pub const UM_SN10_WINDOW: u16 = 512;
/// Fixed header length of a 10-bit-SN UMD PDU.
pub const UM_SN10_HEADER_LEN: usize = 2;

/// Errors parsing a UMD PDU.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LteUmError {
    /// Shorter than its header (including the E/LI fields it announces).
    #[error("UMD PDU of {0} octets is shorter than its header")]
    TooShort(usize),
    /// LIs that add up to more than the PDU carries, or an LI of 0.
    #[error("UMD PDU length indicators do not fit the PDU")]
    BadLengthIndicator,
}

/// A parsed UMD PDU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UmdPdu {
    /// SN.
    pub sn: u16,
    /// FI bit 1: the first data field is *not* the start of an SDU.
    pub first_continues: bool,
    /// FI bit 2: the last data field is *not* the end of an SDU.
    pub last_continues: bool,
    /// The data fields, in order.
    pub fields: Vec<Vec<u8>>,
}

/// Parse a 10-bit-SN UMD PDU (§6.2.1.3).
pub fn parse_umd_pdu(pdu: &[u8]) -> Result<UmdPdu, LteUmError> {
    if pdu.len() < UM_SN10_HEADER_LEN {
        return Err(LteUmError::TooShort(pdu.len()));
    }
    let fi = (pdu[0] >> 3) & 0x03;
    let mut e = pdu[0] & 0x04 != 0;
    let sn = (u16::from(pdu[0] & 0x03) << 8) | u16::from(pdu[1]);
    // E/LI pairs are 12 bits each, packed from octet 3.
    let mut lis = Vec::new();
    let mut bit = UM_SN10_HEADER_LEN * 8;
    let get = |bit: usize, n: usize| -> Option<u16> {
        let mut v = 0u16;
        for i in 0..n {
            let b = bit + i;
            let byte = *pdu.get(b / 8)?;
            v = (v << 1) | u16::from((byte >> (7 - (b % 8))) & 1);
        }
        Some(v)
    };
    while e {
        let pair = get(bit, 12).ok_or(LteUmError::TooShort(pdu.len()))?;
        e = pair & 0x800 != 0;
        let li = pair & 0x7FF;
        if li == 0 {
            return Err(LteUmError::BadLengthIndicator);
        }
        lis.push(usize::from(li));
        bit += 12;
    }
    // An odd number of LIs leaves 4 padding bits; either way the data starts
    // on the next octet boundary.
    let mut off = bit.div_ceil(8);
    let mut fields = Vec::with_capacity(lis.len() + 1);
    for li in lis {
        let end = off + li;
        // Every LI is followed by at least one more data field (§6.2.2.4),
        // and a data field is never empty.
        if end >= pdu.len() {
            return Err(LteUmError::BadLengthIndicator);
        }
        fields.push(pdu[off..end].to_vec());
        off = end;
    }
    if off > pdu.len() {
        return Err(LteUmError::BadLengthIndicator);
    }
    fields.push(pdu[off..].to_vec());
    Ok(UmdPdu {
        sn,
        first_continues: fi & 0x02 != 0,
        last_continues: fi & 0x01 != 0,
        fields,
    })
}

/// The transmitting UM RLC entity (§5.1.2.1).
#[derive(Debug, Clone, Default)]
pub struct LteUmTx {
    vt_us: u16,
}

impl LteUmTx {
    /// A new entity, `VT(US)` = 0 (§7.1).
    pub fn new() -> Self {
        Self::default()
    }

    /// One SDU as one UMD PDU: FI = 00 (the field is a whole SDU), E = 0.
    pub fn transmit(&mut self, sdu: &[u8]) -> Vec<u8> {
        let sn = self.vt_us;
        self.vt_us = (sn + 1) % UM_SN10_MODULUS;
        let mut pdu = Vec::with_capacity(UM_SN10_HEADER_LEN + sdu.len());
        pdu.push((sn >> 8) as u8 & 0x03);
        pdu.push(sn as u8);
        pdu.extend_from_slice(sdu);
        pdu
    }
}

/// The receiving UM RLC entity (§5.1.2.2).
#[derive(Debug, Clone)]
pub struct LteUmRx {
    vr_ur: u16,
    vr_ux: u16,
    vr_uh: u16,
    buffer: BTreeMap<u16, UmdPdu>,
    t_reordering: Duration,
    reordering_expiry: Option<Instant>,
    /// An SDU being reassembled and the SN whose first field continues it.
    partial: Option<(u16, Vec<u8>)>,
}

impl LteUmRx {
    /// A new entity (§7.1: all variables 0) with `t-Reordering`.
    pub fn new(t_reordering: Duration) -> Self {
        Self {
            vr_ur: 0,
            vr_ux: 0,
            vr_uh: 0,
            buffer: BTreeMap::new(),
            t_reordering,
            reordering_expiry: None,
            partial: None,
        }
    }

    /// SN relative to the lower edge of the window, for modulus comparisons.
    fn rel(&self, sn: u16) -> u16 {
        let base = self.vr_uh.wrapping_add(UM_SN10_MODULUS - UM_SN10_WINDOW) % UM_SN10_MODULUS;
        sn.wrapping_add(UM_SN10_MODULUS - base) % UM_SN10_MODULUS
    }

    fn in_window(&self, sn: u16) -> bool {
        self.rel(sn) < UM_SN10_WINDOW
    }

    /// When t-Reordering, if running, expires.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.reordering_expiry
    }

    /// A UMD PDU from lower layers. Returns the SDUs it completes, in order.
    pub fn receive(&mut self, pdu: &[u8], now: Instant) -> Result<Vec<Vec<u8>>, LteUmError> {
        let p = parse_umd_pdu(pdu)?;
        let x = p.sn;
        // §5.1.2.2.2.
        let (rx, rur, ruh) = (self.rel(x), self.rel(self.vr_ur), self.rel(self.vr_uh));
        let duplicate = rur < rx && rx < ruh && self.buffer.contains_key(&x);
        let late = rx < rur;
        if duplicate || late {
            return Ok(Vec::new());
        }
        self.buffer.insert(x, p);
        // §5.1.2.2.3.
        let mut out = Vec::new();
        if !self.in_window(x) {
            self.vr_uh = (x + 1) % UM_SN10_MODULUS;
            self.deliver_outside_window(&mut out);
            if !self.in_window(self.vr_ur) {
                self.vr_ur =
                    self.vr_uh.wrapping_add(UM_SN10_MODULUS - UM_SN10_WINDOW) % UM_SN10_MODULUS;
            }
        }
        if self.buffer.contains_key(&self.vr_ur) {
            self.advance_vr_ur_from(self.vr_ur);
            self.deliver_below(self.vr_ur, &mut out);
        }
        if self.reordering_expiry.is_some() {
            let ux_le_ur = self.rel(self.vr_ux) <= self.rel(self.vr_ur);
            let ux_out = !self.in_window(self.vr_ux) && self.vr_ux != self.vr_uh;
            if ux_le_ur || ux_out {
                self.reordering_expiry = None;
            }
        }
        self.maybe_start_reordering(now);
        Ok(out)
    }

    /// t-Reordering, if it is due (§5.1.2.2.4).
    pub fn on_timer(&mut self, now: Instant) -> Vec<Vec<u8>> {
        match self.reordering_expiry {
            Some(t) if t <= now => {}
            _ => return Vec::new(),
        }
        self.reordering_expiry = None;
        self.advance_vr_ur_from(self.vr_ux);
        let mut out = Vec::new();
        self.deliver_below(self.vr_ur, &mut out);
        self.maybe_start_reordering(now);
        out
    }

    fn maybe_start_reordering(&mut self, now: Instant) {
        if self.reordering_expiry.is_none() && self.rel(self.vr_uh) > self.rel(self.vr_ur) {
            self.reordering_expiry = Some(now + self.t_reordering);
            self.vr_ux = self.vr_uh;
        }
    }

    /// VR(UR) := the first SN ≥ `from` (in window order) not received.
    fn advance_vr_ur_from(&mut self, from: u16) {
        let mut sn = from;
        while self.buffer.contains_key(&sn) && self.rel(sn) < self.rel(self.vr_uh) {
            sn = (sn + 1) % UM_SN10_MODULUS;
        }
        self.vr_ur = sn;
    }

    /// Reassemble and deliver every buffered PDU whose SN is outside the
    /// window, in ascending SN order.
    fn deliver_outside_window(&mut self, out: &mut Vec<Vec<u8>>) {
        let mut sns: Vec<u16> = self
            .buffer
            .keys()
            .copied()
            .filter(|s| !self.in_window(*s))
            .collect();
        // Outside the window means "older than the lower edge": order them by
        // distance below it.
        sns.sort_by_key(|s| self.rel(*s));
        for sn in sns {
            if let Some(p) = self.buffer.remove(&sn) {
                self.reassemble(p, out);
            }
        }
    }

    /// Reassemble and deliver every buffered PDU with SN < `limit`.
    fn deliver_below(&mut self, limit: u16, out: &mut Vec<Vec<u8>>) {
        let lim = self.rel(limit);
        let mut sns: Vec<u16> = self
            .buffer
            .keys()
            .copied()
            .filter(|s| self.rel(*s) < lim)
            .collect();
        sns.sort_by_key(|s| self.rel(*s));
        for sn in sns {
            if let Some(p) = self.buffer.remove(&sn) {
                self.reassemble(p, out);
            }
        }
    }

    fn reassemble(&mut self, p: UmdPdu, out: &mut Vec<Vec<u8>>) {
        let n = p.fields.len();
        let carried = self.partial.take();
        for (i, field) in p.fields.into_iter().enumerate() {
            let first = i == 0;
            let last = i + 1 == n;
            let mut sdu = if first && p.first_continues {
                match &carried {
                    // The segment continues the SDU the previous PDU left
                    // open, and that PDU was the one just before.
                    Some((next, buf)) if *next == p.sn => {
                        let mut b = buf.clone();
                        b.extend_from_slice(&field);
                        b
                    }
                    // Its start was lost: the rest of that SDU cannot be
                    // delivered (and if it also runs on, nothing is kept).
                    _ => continue,
                }
            } else {
                field
            };
            if last && p.last_continues {
                self.partial = Some(((p.sn + 1) % UM_SN10_MODULUS, std::mem::take(&mut sdu)));
            } else {
                out.push(sdu);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const T: Duration = Duration::from_millis(35);

    fn pdu(sn: u16, fi: u8, lis: &[u16], data: &[u8]) -> Vec<u8> {
        let e = u8::from(!lis.is_empty());
        let mut v = vec![(fi << 3) | (e << 2) | ((sn >> 8) as u8 & 3), sn as u8];
        // Pack E/LI pairs by hand, 12 bits each.
        let mut bits: Vec<bool> = Vec::new();
        for (i, li) in lis.iter().enumerate() {
            bits.push(i + 1 < lis.len());
            for b in (0..11).rev() {
                bits.push((li >> b) & 1 == 1);
            }
        }
        while !bits.len().is_multiple_of(8) {
            bits.push(false);
        }
        for ch in bits.chunks(8) {
            v.push(ch.iter().fold(0u8, |a, b| (a << 1) | u8::from(*b)));
        }
        v.extend_from_slice(data);
        v
    }

    #[test]
    fn header_by_hand() {
        // SN 0x2C5, FI 00, E 0: oct 1 = 000 00 0 10 = 0x02, oct 2 = 0xC5.
        let mut tx = LteUmTx { vt_us: 0x2C5 };
        assert_eq!(tx.transmit(&[0xAB]), vec![0x02, 0xC5, 0xAB]);
        // VT(US) wraps at 1024.
        let mut tx = LteUmTx { vt_us: 1023 };
        tx.transmit(&[]);
        assert_eq!(tx.vt_us, 0);
    }

    #[test]
    fn two_lis_are_packed_with_their_e_bits() {
        // FI 01, E 1, SN 3: oct 1 = 000 01 1 00 = 0x0C, oct 2 = 0x03.
        // E/LI pairs "1 00000000010" and "0 00000000011" are the 24 bits
        // 1000_0000 0010_0000 0000_0011 = 0x80 0x20 0x03.
        let p = pdu(3, 0b01, &[2, 3], &[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&p[..5], &[0x0C, 0x03, 0x80, 0x20, 0x03]);
        let u = parse_umd_pdu(&p).unwrap();
        assert_eq!(u.fields, vec![vec![1, 2], vec![3, 4, 5], vec![6, 7]]);
        assert!(!u.first_continues && u.last_continues);
        // One LI: 12 bits + 4 padding bits.
        let p = pdu(0, 0, &[1], &[9, 8]);
        assert_eq!(&p[2..4], &[0x00, 0x10]);
        assert_eq!(parse_umd_pdu(&p).unwrap().fields, vec![vec![9], vec![8]]);
    }

    #[test]
    fn malformed_pdus_are_refused() {
        assert_eq!(parse_umd_pdu(&[0x00]), Err(LteUmError::TooShort(1)));
        // E = 1 with no LI octets.
        assert!(parse_umd_pdu(&[0x04, 0x00]).is_err());
        // LI larger than the PDU.
        assert_eq!(
            parse_umd_pdu(&pdu(0, 0, &[50], &[1, 2])),
            Err(LteUmError::BadLengthIndicator)
        );
    }

    #[test]
    fn in_order_pdus_are_delivered_at_once() {
        let now = Instant::now();
        let mut tx = LteUmTx::new();
        let mut rx = LteUmRx::new(T);
        for i in 0..1500u16 {
            let sdu = i.to_be_bytes().to_vec();
            assert_eq!(rx.receive(&tx.transmit(&sdu), now).unwrap(), vec![sdu]);
        }
        assert!(rx.next_deadline().is_none());
    }

    #[test]
    fn reordering_holds_until_the_gap_fills() {
        let now = Instant::now();
        let mut rx = LteUmRx::new(T);
        assert_eq!(
            rx.receive(&pdu(0, 0, &[], b"a"), now).unwrap(),
            vec![b"a".to_vec()]
        );
        assert!(rx.receive(&pdu(2, 0, &[], b"c"), now).unwrap().is_empty());
        assert!(rx.next_deadline().is_some());
        assert_eq!(
            rx.receive(&pdu(1, 0, &[], b"b"), now).unwrap(),
            vec![b"b".to_vec(), b"c".to_vec()]
        );
        assert!(rx.next_deadline().is_none());
        // A duplicate of a delivered PDU is late and discarded -- not just
        // held: the next in-order PDU delivers only itself.
        assert!(rx.receive(&pdu(1, 0, &[], b"b"), now).unwrap().is_empty());
        assert_eq!(
            rx.receive(&pdu(3, 0, &[], b"d"), now).unwrap(),
            vec![b"d".to_vec()]
        );
    }

    #[test]
    fn t_reordering_gives_up_on_a_lost_pdu() {
        let now = Instant::now();
        let mut rx = LteUmRx::new(T);
        rx.receive(&pdu(0, 0, &[], b"a"), now).unwrap();
        assert!(rx.receive(&pdu(2, 0, &[], b"c"), now).unwrap().is_empty());
        assert!(rx.on_timer(now).is_empty(), "not yet due");
        assert_eq!(rx.on_timer(now + T), vec![b"c".to_vec()]);
        assert!(rx.next_deadline().is_none());
    }

    #[test]
    fn segments_reassemble_and_a_lost_segment_drops_only_its_sdu() {
        let now = Instant::now();
        let mut rx = LteUmRx::new(T);
        // SDU "hello" split over SN 0 (FI 01) and SN 1 (FI 10), concatenated
        // with a whole SDU "!" in SN 1.
        assert!(rx
            .receive(&pdu(0, 0b01, &[], b"hel"), now)
            .unwrap()
            .is_empty());
        assert_eq!(
            rx.receive(&pdu(1, 0b10, &[2], b"lo!"), now).unwrap(),
            vec![b"hello".to_vec(), b"!".to_vec()]
        );
        // SN 2 starts an SDU (FI 01), SN 3 (its end) is lost, SN 4 is whole.
        rx.receive(&pdu(2, 0b01, &[], b"lost-"), now).unwrap();
        assert!(rx
            .receive(&pdu(4, 0b00, &[], b"next"), now)
            .unwrap()
            .is_empty());
        assert_eq!(rx.on_timer(now + T), vec![b"next".to_vec()]);
        // SN 5 starts an SDU, SN 6 (its middle) is lost, SN 7 ends one: the
        // start and the end belong to different SDUs and must not be joined.
        rx.receive(&pdu(5, 0b01, &[], b"ab"), now).unwrap();
        assert!(rx
            .receive(&pdu(7, 0b10, &[], b"yz"), now)
            .unwrap()
            .is_empty());
        assert!(rx.on_timer(now + T * 2).is_empty(), "no SDU was complete");
    }

    #[test]
    fn the_window_follows_vr_uh_across_the_sn_wrap() {
        let now = Instant::now();
        let mut rx = LteUmRx::new(T);
        // §7.1 starts every variable at 0, so SNs 512..1023 are "late" for a
        // fresh receiver; run the SN up to 1001 in order first.
        for sn in 0..=1001u16 {
            assert_eq!(rx.receive(&pdu(sn, 0, &[], b"x"), now).unwrap().len(), 1);
        }
        // 1002..=1022 are lost; 1023, 0 and 1 wait for t-Reordering.
        for sn in [1023u16, 0, 1] {
            assert!(rx
                .receive(&pdu(sn, 0, &[], &sn.to_be_bytes()), now)
                .unwrap()
                .is_empty());
        }
        let sdu = |s: u16| s.to_be_bytes().to_vec();
        assert_eq!(rx.on_timer(now + T), vec![sdu(1023), sdu(0), sdu(1)]);
        // In order after the wrap.
        assert_eq!(
            rx.receive(&pdu(2, 0, &[], b"z"), now).unwrap(),
            vec![b"z".to_vec()]
        );
    }
}
