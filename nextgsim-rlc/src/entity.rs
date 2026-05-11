//! RLC entity — the stateful per-bearer RLC instance
//!
//! One `RlcEntity` is created per RLC bearer (per logical channel).
//! The same entity handles both the transmit and receive sides.

use std::collections::{BTreeMap, VecDeque};
use std::time::Duration;

use tracing::{debug, trace, warn};

use crate::error::RlcError;
use crate::pdu::{RlcAmPdu, RlcStatusPdu, RlcUmPdu, SegmentationInfo};
use crate::{RlcMode, SnSize};

// ── Reassembly bookkeeping ───────────────────────────────────────────────────

/// One received segment waiting for reassembly
#[derive(Debug, Clone)]
pub struct RlcSegment {
    /// Byte offset of this segment within the original SDU
    pub offset: u16,
    /// Whether this is the last segment (so we know total length eventually)
    pub is_last: bool,
    /// Payload of this segment
    pub data: Vec<u8>,
}

/// State of one SDU being reassembled
#[derive(Debug)]
struct Reassembly {
    /// All received segments, keyed by byte offset
    segments: BTreeMap<u16, RlcSegment>,
    /// Known total length once the last segment arrives
    total_len: Option<usize>,
}

impl Reassembly {
    fn new() -> Self {
        Self {
            segments: BTreeMap::new(),
            total_len: None,
        }
    }

    /// Insert a segment. Returns `Err` on duplicate.
    fn insert(&mut self, seg: RlcSegment) -> Result<(), RlcError> {
        use std::collections::btree_map::Entry;
        match self.segments.entry(seg.offset) {
            Entry::Vacant(e) => {
                if seg.is_last {
                    self.total_len = Some(seg.offset as usize + seg.data.len());
                }
                e.insert(seg);
                Ok(())
            }
            Entry::Occupied(_) => Err(RlcError::DuplicateSegment {
                sn: 0, // caller fills in SN for the error message
                so: seg.offset,
            }),
        }
    }

    /// Try to reassemble if we have a contiguous run from offset 0.
    fn try_reassemble(&self) -> Option<Vec<u8>> {
        let total = self.total_len?;
        let mut sdu = Vec::with_capacity(total);
        let mut expected = 0u16;
        for (offset, seg) in &self.segments {
            if *offset != expected {
                return None; // gap
            }
            sdu.extend_from_slice(&seg.data);
            expected = offset + seg.data.len() as u16;
        }
        if sdu.len() == total {
            Some(sdu)
        } else {
            None
        }
    }
}

// ── RlcEntity ────────────────────────────────────────────────────────────────

/// A single RLC bearer entity handling both TX and RX for one logical channel.
///
/// Create one per bearer with [`RlcEntity::new`], then drive it by:
/// 1. Calling [`submit_sdu`] when PDCP passes down an SDU.
/// 2. Calling [`build_pdu`] each TTI when MAC asks for data.
/// 3. Calling [`receive_pdu`] when MAC passes up a received PDU.
/// 4. Calling [`poll_reassembled`] to collect complete SDUs for PDCP.
///
/// [`submit_sdu`]: RlcEntity::submit_sdu
/// [`build_pdu`]: RlcEntity::build_pdu
/// [`receive_pdu`]: RlcEntity::receive_pdu
/// [`poll_reassembled`]: RlcEntity::poll_reassembled
#[derive(Debug)]
pub struct RlcEntity {
    /// Operating mode
    pub mode: RlcMode,
    /// Sequence number field size
    sn_size: SnSize,

    // ── TX side ──────────────────────────────────────────────────────────────
    /// SDUs waiting to be segmented and sent
    pub tx_buffer: VecDeque<Vec<u8>>,
    /// Next TX sequence number to assign
    pub tx_next: u32,
    /// The SDU currently being segmented (may have bytes already sent)
    tx_current_sdu: Option<Vec<u8>>,
    /// How many bytes of `tx_current_sdu` have already been placed in PDUs
    tx_current_offset: usize,

    // ── AM retransmit queue ───────────────────────────────────────────────────
    /// PDUs that have been sent but not yet acknowledged, keyed by SN
    am_unacked: BTreeMap<u32, Vec<u8>>,
    /// SNs pending retransmission (added on NACK / timer expiry)
    am_retx_queue: VecDeque<u32>,
    /// Next SN the receiver expects (updated from STATUS PDUs)
    pub rx_next: u32,
    /// Poll-retransmit timer duration (AM only)
    pub poll_retransmit_timer: Option<Duration>,

    // ── RX side ───────────────────────────────────────────────────────────────
    /// Reassembly state per SN: `BTreeMap<sn, Reassembly>`
    rx_reassembly: BTreeMap<u32, Reassembly>,
    /// Fully reassembled SDUs ready for PDCP
    rx_ready: VecDeque<Vec<u8>>,
    /// Highest in-sequence SN delivered to PDCP (for window management)
    pub rx_delivered_next: u32,
}

impl RlcEntity {
    /// Create a new RLC entity.
    ///
    /// # Panics
    /// Panics if `sn_size` is `Sn18` for `UnacknowledgedMode` or
    /// `Sn6` for `AcknowledgedMode` (illegal per TS 38.322).
    pub fn new(mode: RlcMode, sn_size: SnSize) -> Self {
        // Validate SN / mode combinations
        match (mode, sn_size) {
            (RlcMode::UnacknowledgedMode, SnSize::Sn18) => {
                panic!("18-bit SN is not valid for UM mode (TS 38.322 §6.2.3.3)");
            }
            (RlcMode::AcknowledgedMode, SnSize::Sn6) => {
                panic!("6-bit SN is not valid for AM mode (TS 38.322 §6.2.3.5)");
            }
            _ => {}
        }
        Self {
            mode,
            sn_size,
            tx_buffer: VecDeque::new(),
            tx_next: 0,
            tx_current_sdu: None,
            tx_current_offset: 0,
            am_unacked: BTreeMap::new(),
            am_retx_queue: VecDeque::new(),
            rx_next: 0,
            poll_retransmit_timer: None,
            rx_reassembly: BTreeMap::new(),
            rx_ready: VecDeque::new(),
            rx_delivered_next: 0,
        }
    }

    // ── Public SDU/PDU interface ──────────────────────────────────────────────

    /// Accept an SDU from PDCP / upper layer.
    ///
    /// In TM mode the SDU is passed through unchanged.
    /// In UM/AM mode it is queued for segmented transmission.
    pub fn submit_sdu(&mut self, data: Vec<u8>) {
        if data.is_empty() {
            warn!("RLC: empty SDU submitted, dropping");
            return;
        }
        trace!(mode = ?self.mode, len = data.len(), "RLC submit_sdu");
        self.tx_buffer.push_back(data);
    }

    /// Build one PDU of at most `max_size` bytes for MAC.
    ///
    /// Returns `None` when there is nothing to send.
    ///
    /// In TM mode the SDU is returned verbatim (no header added).
    /// In UM/AM mode segmentation headers are prepended.
    pub fn build_pdu(&mut self, max_size: usize) -> Option<Vec<u8>> {
        match self.mode {
            RlcMode::TransparentMode => self.build_tm_pdu(max_size),
            RlcMode::UnacknowledgedMode => self.build_um_pdu(max_size),
            RlcMode::AcknowledgedMode => self.build_am_pdu(max_size),
        }
    }

    /// Process a received PDU from MAC.
    ///
    /// Decoded segments are placed in the reassembly buffer.
    /// Complete SDUs are moved to the ready queue.
    pub fn receive_pdu(&mut self, data: &[u8]) {
        let result = match self.mode {
            RlcMode::TransparentMode => self.receive_tm_pdu(data),
            RlcMode::UnacknowledgedMode => self.receive_um_pdu(data),
            RlcMode::AcknowledgedMode => self.receive_am_pdu(data),
        };
        if let Err(e) = result {
            warn!("RLC receive_pdu error: {e}");
        }
    }

    /// Return the next fully reassembled SDU to PDCP, if any.
    pub fn poll_reassembled(&mut self) -> Option<Vec<u8>> {
        self.rx_ready.pop_front()
    }

    /// Return `true` if there are pending SDUs to transmit.
    pub fn has_data(&self) -> bool {
        self.tx_current_sdu.is_some() || !self.tx_buffer.is_empty()
    }

    // ── TM ────────────────────────────────────────────────────────────────────

    fn build_tm_pdu(&mut self, max_size: usize) -> Option<Vec<u8>> {
        let sdu = self.tx_buffer.pop_front()?;
        if sdu.len() > max_size {
            // TM does not support segmentation — drop and warn
            warn!(
                len = sdu.len(),
                max_size, "RLC TM: SDU too large for MAC grant, dropping"
            );
            return None;
        }
        Some(sdu)
    }

    fn receive_tm_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        self.rx_ready.push_back(data.to_vec());
        Ok(())
    }

    // ── UM ────────────────────────────────────────────────────────────────────

    /// Header overhead for a UM PDU given the SN size and whether SO is needed.
    fn um_header_overhead(&self, has_so: bool) -> usize {
        let base = self.sn_size.um_header_bytes();
        base + if has_so { 2 } else { 0 }
    }

    fn build_um_pdu(&mut self, max_size: usize) -> Option<Vec<u8>> {
        // Load the next SDU to work on if none is in progress
        if self.tx_current_sdu.is_none() {
            let sdu = self.tx_buffer.pop_front()?;
            self.tx_current_sdu = Some(sdu);
            self.tx_current_offset = 0;
        }

        let sdu = self.tx_current_sdu.as_ref().unwrap();
        let remaining = &sdu[self.tx_current_offset..];
        let is_first = self.tx_current_offset == 0;

        // Determine SI and header size
        let has_so = !is_first; // SO only on non-first segments
        let hdr = self.um_header_overhead(has_so);
        if max_size <= hdr {
            // Not enough room even for the header
            return None;
        }
        let payload_capacity = max_size - hdr;
        let payload_len = remaining.len().min(payload_capacity);
        let is_last = payload_len == remaining.len();

        let si = match (is_first, is_last) {
            (true, true) => SegmentationInfo::FullSdu,
            (true, false) => SegmentationInfo::FirstSegment,
            (false, true) => SegmentationInfo::LastSegment,
            (false, false) => SegmentationInfo::MiddleSegment,
        };

        let so = if has_so {
            Some(self.tx_current_offset as u16)
        } else {
            None
        };

        let sn = (self.tx_next % self.sn_size.modulus()) as u16;
        let pdu_data = remaining[..payload_len].to_vec();

        let pdu = RlcUmPdu {
            si,
            sn,
            so,
            data: pdu_data,
        };
        let encoded = match self.sn_size {
            SnSize::Sn6 => pdu.encode_sn6(),
            SnSize::Sn12 => pdu.encode_sn12(),
            SnSize::Sn18 => unreachable!(),
        };

        // Advance state
        self.tx_current_offset += payload_len;
        if is_last {
            self.tx_next += 1;
            self.tx_current_sdu = None;
            self.tx_current_offset = 0;
        }

        debug!(si = ?si, sn, payload_len, "RLC UM build_pdu");
        Some(encoded)
    }

    fn receive_um_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        let pdu = match self.sn_size {
            SnSize::Sn6 => RlcUmPdu::decode_sn6(data)?,
            SnSize::Sn12 => RlcUmPdu::decode_sn12(data)?,
            SnSize::Sn18 => unreachable!(),
        };

        let sn = pdu.sn as u32;
        trace!(si = ?pdu.si, sn, "RLC UM receive_pdu");

        if pdu.si == SegmentationInfo::FullSdu {
            // No reassembly needed
            self.rx_ready.push_back(pdu.data);
            return Ok(());
        }

        let so = pdu.so.unwrap_or(0);
        let is_last = pdu.si.is_last();
        let seg = RlcSegment {
            offset: so,
            is_last,
            data: pdu.data,
        };

        let entry = self.rx_reassembly.entry(sn).or_insert_with(Reassembly::new);
        if let Err(mut e) = entry.insert(seg) {
            // Patch in the real SN for the error message
            if let RlcError::DuplicateSegment { sn: ref mut s, .. } = e {
                *s = sn;
            }
            return Err(e);
        }

        if let Some(sdu) = entry.try_reassemble() {
            self.rx_reassembly.remove(&sn);
            self.rx_ready.push_back(sdu);
        }
        Ok(())
    }

    // ── AM ────────────────────────────────────────────────────────────────────

    fn am_header_overhead(&self, has_so: bool) -> usize {
        let base = match self.sn_size {
            SnSize::Sn12 => 2,
            SnSize::Sn18 => 3,
            SnSize::Sn6 => unreachable!(),
        };
        base + if has_so { 2 } else { 0 }
    }

    fn build_am_pdu(&mut self, max_size: usize) -> Option<Vec<u8>> {
        // Prioritise retransmissions
        if let Some(retx_sn) = self.am_retx_queue.front().copied() {
            if let Some(orig) = self.am_unacked.get(&retx_sn) {
                if orig.len() <= max_size {
                    let pdu = orig.clone();
                    self.am_retx_queue.pop_front();
                    debug!(sn = retx_sn, "RLC AM retransmitting PDU");
                    return Some(pdu);
                }
            } else {
                // SN no longer unacked (got ACKed between enqueue and now)
                self.am_retx_queue.pop_front();
            }
        }

        // New data
        if self.tx_current_sdu.is_none() {
            let sdu = self.tx_buffer.pop_front()?;
            self.tx_current_sdu = Some(sdu);
            self.tx_current_offset = 0;
        }

        let sdu = self.tx_current_sdu.as_ref().unwrap();
        let remaining = &sdu[self.tx_current_offset..];
        let is_first = self.tx_current_offset == 0;
        let has_so = !is_first;
        let hdr = self.am_header_overhead(has_so);

        if max_size <= hdr {
            return None;
        }
        let payload_capacity = max_size - hdr;
        let payload_len = remaining.len().min(payload_capacity);
        let is_last = payload_len == remaining.len();

        let si = match (is_first, is_last) {
            (true, true) => SegmentationInfo::FullSdu,
            (true, false) => SegmentationInfo::FirstSegment,
            (false, true) => SegmentationInfo::LastSegment,
            (false, false) => SegmentationInfo::MiddleSegment,
        };

        let so = if has_so {
            Some(self.tx_current_offset as u16)
        } else {
            None
        };
        let sn = self.tx_next;
        let pdu_data = remaining[..payload_len].to_vec();

        // Set poll bit on last segment of each SDU (simple poll strategy)
        let p = is_last;

        let am_pdu = RlcAmPdu {
            dc: true,
            p,
            si,
            sn,
            so,
            data: pdu_data,
        };
        let encoded = match self.sn_size {
            SnSize::Sn12 => am_pdu.encode_sn12(),
            SnSize::Sn18 => am_pdu.encode_sn18(),
            SnSize::Sn6 => unreachable!(),
        };

        // Store for possible retransmission
        self.am_unacked.insert(sn, encoded.clone());

        self.tx_current_offset += payload_len;
        if is_last {
            self.tx_next += 1;
            self.tx_current_sdu = None;
            self.tx_current_offset = 0;
        }

        debug!(si = ?si, sn, payload_len, poll = p, "RLC AM build_pdu");
        Some(encoded)
    }

    fn receive_am_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        // Distinguish Data vs Control by D/C bit
        if data.is_empty() {
            return Err(RlcError::PduTooShort { need: 1, got: 0 });
        }

        let dc_bit = (data[0] & 0x80) != 0;

        if !dc_bit {
            // Control (STATUS) PDU
            return self.handle_status_pdu(data);
        }

        // Data PDU
        let pdu = match self.sn_size {
            SnSize::Sn12 => RlcAmPdu::decode_sn12(data)?,
            SnSize::Sn18 => RlcAmPdu::decode_sn18(data)?,
            SnSize::Sn6 => unreachable!(),
        };

        let sn = pdu.sn;
        trace!(si = ?pdu.si, sn, poll = pdu.p, "RLC AM receive data PDU");

        if pdu.si == SegmentationInfo::FullSdu {
            self.rx_ready.push_back(pdu.data);
            self.advance_rx_next(sn);
            return Ok(());
        }

        let so = pdu.so.unwrap_or(0);
        let is_last = pdu.si.is_last();
        let seg = RlcSegment {
            offset: so,
            is_last,
            data: pdu.data,
        };

        let entry = self.rx_reassembly.entry(sn).or_insert_with(Reassembly::new);
        if let Err(mut e) = entry.insert(seg) {
            if let RlcError::DuplicateSegment { sn: ref mut s, .. } = e {
                *s = sn;
            }
            return Err(e);
        }

        if let Some(sdu) = entry.try_reassemble() {
            self.rx_reassembly.remove(&sn);
            self.rx_ready.push_back(sdu);
            self.advance_rx_next(sn);
        }
        Ok(())
    }

    fn handle_status_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        let status = RlcStatusPdu::decode_sn12(data)?;
        let ack_sn = status.ack_sn;
        debug!(ack_sn, "RLC AM received STATUS PDU");

        // Remove all unacked PDUs with SN < ack_sn (they have been received)
        let sn_mod = self.sn_size.modulus();
        self.am_unacked
            .retain(|&sn, _| (sn % sn_mod) >= (ack_sn % sn_mod));
        self.rx_next = ack_sn;
        Ok(())
    }

    /// Advance `rx_delivered_next` past `sn` if appropriate (simplified).
    fn advance_rx_next(&mut self, sn: u32) {
        let modulus = self.sn_size.modulus();
        if sn % modulus == self.rx_delivered_next % modulus {
            self.rx_delivered_next = (self.rx_delivered_next + 1) % modulus;
        }
    }

    /// Request retransmission of a specific SN (e.g. on NACK from STATUS PDU).
    pub fn request_retransmit(&mut self, sn: u32) {
        if self.am_unacked.contains_key(&sn) {
            debug!(sn, "RLC AM: scheduling retransmission");
            self.am_retx_queue.push_back(sn);
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── TM mode ───────────────────────────────────────────────────────────────

    #[test]
    fn tm_passthrough_small_sdu() {
        let mut e = RlcEntity::new(RlcMode::TransparentMode, SnSize::Sn12);
        let sdu = vec![1u8, 2, 3, 4, 5];
        e.submit_sdu(sdu.clone());
        let pdu = e.build_pdu(64).expect("should produce a PDU");
        assert_eq!(pdu, sdu, "TM must return SDU verbatim");
        assert!(e.build_pdu(64).is_none(), "no more PDUs");
    }

    #[test]
    fn tm_oversized_sdu_dropped() {
        let mut e = RlcEntity::new(RlcMode::TransparentMode, SnSize::Sn12);
        e.submit_sdu(vec![0u8; 200]);
        // MAC only grants 100 bytes — TM cannot segment, so PDU is dropped
        let pdu = e.build_pdu(100);
        assert!(pdu.is_none(), "oversized TM SDU must be dropped");
    }

    #[test]
    fn tm_receive_passthrough() {
        let mut e = RlcEntity::new(RlcMode::TransparentMode, SnSize::Sn12);
        let raw = vec![0xAA, 0xBB, 0xCC];
        e.receive_pdu(&raw);
        let out = e.poll_reassembled().expect("must have data");
        assert_eq!(out, raw);
        assert!(e.poll_reassembled().is_none());
    }

    #[test]
    fn tm_multiple_sdus_fifo_order() {
        let mut tx = RlcEntity::new(RlcMode::TransparentMode, SnSize::Sn12);
        tx.submit_sdu(vec![1, 2]);
        tx.submit_sdu(vec![3, 4]);
        assert_eq!(tx.build_pdu(64).unwrap(), vec![1, 2]);
        assert_eq!(tx.build_pdu(64).unwrap(), vec![3, 4]);
        assert!(tx.build_pdu(64).is_none());
    }

    // ── UM mode (SN12) ────────────────────────────────────────────────────────

    #[test]
    fn um_small_sdu_fits_in_one_pdu() {
        let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let sdu = b"hello world".to_vec();
        tx.submit_sdu(sdu.clone());

        let pdu = tx.build_pdu(256).expect("PDU");
        assert!(tx.build_pdu(256).is_none(), "only one PDU for small SDU");

        rx.receive_pdu(&pdu);
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    #[test]
    fn um_segmentation_and_reassembly() {
        let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);

        let sdu = vec![0xABu8; 100];
        tx.submit_sdu(sdu.clone());

        // Pull PDUs with 40-byte MAC grants (2-byte UM header)
        let mut pdus = Vec::new();
        while let Some(pdu) = tx.build_pdu(40) {
            pdus.push(pdu);
        }
        // 100 bytes / 38 bytes payload = 3 PDUs (38 + 38 + 24)
        assert!(pdus.len() >= 2, "must produce multiple segments");
        assert!(tx.build_pdu(40).is_none(), "all segments sent");

        for pdu in &pdus {
            rx.receive_pdu(pdu);
        }
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu, "reassembled SDU must match original");
    }

    #[test]
    fn um_out_of_order_reassembly() {
        let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);

        let sdu = vec![0x55u8; 60];
        tx.submit_sdu(sdu.clone());

        let pdu1 = tx.build_pdu(30).unwrap();
        let pdu2 = tx.build_pdu(30).unwrap();
        let pdu3 = tx.build_pdu(30).unwrap();
        assert!(tx.build_pdu(30).is_none());

        // Deliver out-of-order: last first, then middle, then first
        rx.receive_pdu(&pdu3);
        assert!(rx.poll_reassembled().is_none(), "not yet complete");
        rx.receive_pdu(&pdu2);
        assert!(rx.poll_reassembled().is_none(), "still not complete");
        rx.receive_pdu(&pdu1);
        let out = rx.poll_reassembled().expect("complete after all segments");
        assert_eq!(out, sdu);
    }

    #[test]
    fn um_multiple_sdus_sequential() {
        let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);

        for i in 0u8..4 {
            tx.submit_sdu(vec![i; 10]);
        }
        while let Some(pdu) = tx.build_pdu(64) {
            rx.receive_pdu(&pdu);
        }
        for i in 0u8..4 {
            let sdu = rx.poll_reassembled().unwrap_or_else(|| panic!("sdu {i}"));
            assert_eq!(sdu, vec![i; 10]);
        }
        assert!(rx.poll_reassembled().is_none());
    }

    #[test]
    fn um_sn6_segmentation_reassembly() {
        let mut tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn6);
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn6);

        let sdu = vec![0xCCu8; 50];
        tx.submit_sdu(sdu.clone());

        let mut pdus = Vec::new();
        while let Some(p) = tx.build_pdu(20) {
            pdus.push(p);
        }
        assert!(!pdus.is_empty());
        for p in &pdus {
            rx.receive_pdu(p);
        }
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    // ── AM mode (SN12) ────────────────────────────────────────────────────────

    #[test]
    fn am_small_sdu_fits_in_one_pdu() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);

        let sdu = vec![0x42u8; 20];
        tx.submit_sdu(sdu.clone());

        let pdu = tx.build_pdu(64).expect("PDU");
        assert!(tx.build_pdu(64).is_none());

        rx.receive_pdu(&pdu);
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    #[test]
    fn am_segmentation_and_reassembly() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);

        let sdu = vec![0xBBu8; 80];
        tx.submit_sdu(sdu.clone());

        let mut pdus = Vec::new();
        while let Some(p) = tx.build_pdu(30) {
            pdus.push(p);
        }
        assert!(pdus.len() >= 3);

        for p in &pdus {
            rx.receive_pdu(p);
        }
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    #[test]
    fn am_retransmission_on_request() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);

        let sdu = vec![0x77u8; 10];
        tx.submit_sdu(sdu.clone());

        // Transmit — store the PDU
        let pdu = tx.build_pdu(64).expect("original PDU");
        assert!(tx.build_pdu(64).is_none());

        // Simulate loss: don't deliver the PDU.
        // Sender schedules retransmission for SN 0
        tx.request_retransmit(0);

        // Second grant: should get the retransmitted PDU
        let retx_pdu = tx.build_pdu(64).expect("retransmitted PDU");
        assert_eq!(pdu, retx_pdu, "retransmit must equal original");

        // Now deliver to receiver
        rx.receive_pdu(&retx_pdu);
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    #[test]
    fn am_status_pdu_clears_unacked() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);

        tx.submit_sdu(vec![0x01u8; 10]);
        tx.submit_sdu(vec![0x02u8; 10]);

        // Build PDUs for both SDUs
        let _p0 = tx.build_pdu(64).unwrap();
        let _p1 = tx.build_pdu(64).unwrap();

        assert_eq!(tx.am_unacked.len(), 2);

        // Receiver sends STATUS PDU acking SN 2 (both SNs 0 and 1 received)
        let status = RlcStatusPdu::new(2);
        let status_bytes = status.encode_sn12();
        tx.receive_pdu(&status_bytes);

        assert_eq!(tx.am_unacked.len(), 0, "all SNs should be cleared on ACK");
    }

    #[test]
    fn am_sn18_roundtrip() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn18);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn18);

        let sdu = vec![0xEEu8; 50];
        tx.submit_sdu(sdu.clone());

        let mut pdus = Vec::new();
        while let Some(p) = tx.build_pdu(20) {
            pdus.push(p);
        }
        for p in &pdus {
            rx.receive_pdu(p);
        }
        let out = rx.poll_reassembled().expect("reassembled");
        assert_eq!(out, sdu);
    }

    // ── General ───────────────────────────────────────────────────────────────

    #[test]
    fn has_data_reflects_buffer_state() {
        let mut e = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        assert!(!e.has_data());
        e.submit_sdu(vec![1, 2, 3]);
        assert!(e.has_data());
        e.build_pdu(64);
        assert!(!e.has_data());
    }

    #[test]
    fn empty_sdu_is_dropped() {
        let mut e = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        e.submit_sdu(vec![]);
        assert!(!e.has_data(), "empty SDU should be dropped silently");
    }
}
