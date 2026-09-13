//! RLC entity — the stateful per-bearer RLC instance
//!
//! One `RlcEntity` is created per RLC bearer (per logical channel).
//! The same entity handles both the transmit and receive sides.

use std::collections::{BTreeMap, VecDeque};
use std::time::{Duration, Instant};

use tracing::{debug, trace, warn};

use crate::error::RlcError;
use crate::pdu::{RlcAmPdu, RlcStatusPdu, RlcUmPdu, SegmentationInfo};
use crate::{RlcMode, SnSize};

/// Default `t-Reassembly` (TS 38.331 `t-Reassembly`, `ms50`).
///
/// A UM receiver with no `t-Reassembly` keeps a partially received SDU forever,
/// so the reassembly buffer grows without bound on a lossy channel; 50 ms is
/// well inside the enumerated range and far above the in-process RLS latency, so
/// a lossless path never sees it fire.
pub const DEFAULT_T_REASSEMBLY: Duration = Duration::from_millis(50);

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

    // ── UM receive state variables (TS 38.322 §7.1) ──────────────────────────
    /// `RX_Next_Reassembly`: the SN of the earliest SDU still awaiting
    /// reassembly. Everything below it has been delivered or discarded.
    rx_next_reassembly: u32,
    /// `RX_Next_Highest`: one past the highest SN received so far; the upper
    /// edge of the reassembly window.
    rx_next_highest: u32,
    /// `RX_Timer_Trigger`: the SN that was `RX_Next_Highest` when `t-Reassembly`
    /// was started, i.e. the SN reassembly is waiting past.
    rx_timer_trigger: u32,
    /// `t-Reassembly` duration (RRC-configured, TS 38.331 `t-Reassembly`).
    /// `None` disables the timer, in which case a partially received SDU waits
    /// forever — the pre-#34 behaviour, kept configurable rather than removed.
    t_reassembly: Option<Duration>,
    /// Deadline of a running `t-Reassembly`, `None` when it is not running
    t_reassembly_deadline: Option<Instant>,
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
            rx_next_reassembly: 0,
            rx_next_highest: 0,
            rx_timer_trigger: 0,
            t_reassembly: Some(DEFAULT_T_REASSEMBLY),
            t_reassembly_deadline: None,
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
        if let Err(e) = self.receive_pdu_checked(data) {
            warn!("RLC receive_pdu error: {e}");
        }
    }

    /// As [`RlcEntity::receive_pdu`], but reports why a PDU was not accepted.
    ///
    /// A discard is a normal event on the receive path — a duplicate, an SN below
    /// `RX_Next_Reassembly`, a repeated byte segment — so [`receive_pdu`] logs and
    /// moves on. A caller that wants to count or assert on those reasons (a test,
    /// or a statistics counter) needs to see them.
    ///
    /// [`receive_pdu`]: RlcEntity::receive_pdu
    pub fn receive_pdu_checked(&mut self, data: &[u8]) -> Result<(), RlcError> {
        match self.mode {
            RlcMode::TransparentMode => self.receive_tm_pdu(data),
            RlcMode::UnacknowledgedMode => self.receive_um_pdu(data),
            RlcMode::AcknowledgedMode => self.receive_am_pdu(data),
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

    /// Whether `sn` falls in the reassembly window (TS 38.322 §5.2.2.2.1):
    /// `(RX_Next_Highest - UM_Window_Size) <= SN < RX_Next_Highest`.
    ///
    /// Compared modulo the SN space, so a window that straddles the wrap point
    /// still admits the SNs inside it.
    fn um_sn_in_reassembly_window(&self, sn: u32) -> bool {
        self.um_sn_distance(self.um_window_lower_edge(), sn) < self.sn_size.um_window_size()
    }

    /// `RX_Next_Highest - UM_Window_Size`, the inclusive lower edge of the
    /// reassembly window.
    fn um_window_lower_edge(&self) -> u32 {
        let modulus = self.sn_size.modulus();
        (self.rx_next_highest + modulus - self.sn_size.um_window_size()) % modulus
    }

    /// Distance from `base` forward to `sn`, modulo the SN space. Used to order
    /// SNs relative to a state variable without assuming they never wrap.
    fn um_sn_distance(&self, base: u32, sn: u32) -> u32 {
        let modulus = self.sn_size.modulus();
        (sn + modulus - base % modulus) % modulus
    }

    fn receive_um_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        let pdu = match self.sn_size {
            SnSize::Sn6 => RlcUmPdu::decode_sn6(data)?,
            SnSize::Sn12 => RlcUmPdu::decode_sn12(data)?,
            SnSize::Sn18 => unreachable!(),
        };

        let sn = pdu.sn as u32;
        trace!(si = ?pdu.si, sn, "RLC UM receive_pdu");

        // TS 38.322 §5.2.2.2.2: discard the PDU when
        // `(RX_Next_Highest - UM_Window_Size) <= SN < RX_Next_Reassembly`, i.e.
        // when its SDU has already been delivered or discarded. That is also
        // what suppresses a replay: UM has no ARQ, so an SN that comes back is a
        // duplicate from below.
        //
        // Note this tree's UMD PDU carries an SN even when it holds a complete
        // SDU, where TS 38.322 §6.2.2.3 has no SN field at all — so the "header
        // does not contain an SN" branch of §5.2.2.2.2 (deliver immediately, no
        // window check) is unreachable here, and complete SDUs go through the
        // same window and state variables as segments.
        let lower_edge = self.um_window_lower_edge();
        if self.um_sn_distance(lower_edge, sn)
            < self.um_sn_distance(lower_edge, self.rx_next_reassembly)
        {
            debug!(
                sn,
                rx_next_reassembly = self.rx_next_reassembly,
                "RLC UM: discarding PDU below RX_Next_Reassembly (already delivered)"
            );
            return Err(RlcError::SnOutsideWindow { sn });
        }

        let delivered = if pdu.si == SegmentationInfo::FullSdu {
            self.rx_ready.push_back(pdu.data);
            self.rx_reassembly.remove(&sn);
            true
        } else {
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

            match entry.try_reassemble() {
                Some(sdu) => {
                    self.rx_reassembly.remove(&sn);
                    self.rx_ready.push_back(sdu);
                    true
                }
                None => false,
            }
        };

        self.um_pdu_placed_in_reception_buffer(sn, delivered);
        Ok(())
    }

    /// TS 38.322 §5.2.2.2.3: the state-variable, window and timer updates that
    /// follow placing a UMD PDU with `SN = x` in the reception buffer.
    ///
    /// `delivered` says whether all byte segments of `x` are now in, i.e.
    /// whether the SDU was reassembled and handed up.
    fn um_pdu_placed_in_reception_buffer(&mut self, sn: u32, delivered: bool) {
        let modulus = self.sn_size.modulus();

        // §5.2.2.2.3, second branch: an SN ahead of the window MOVES the window
        // up rather than being dropped — dropping it would stall the receiver
        // permanently after a burst loss. Whatever then falls out of the bottom
        // is gone for good.
        //
        // The spec runs this only when the PDU did NOT complete an SDU, because
        // there a completed SDU always arrived as a later segment of an SN that
        // had already advanced the window. Here it is unconditional, because a
        // complete-SDU PDU in this tree carries its own SN (§6.2.2.3 deviation)
        // and would otherwise never advance anything.
        if !self.um_sn_in_reassembly_window(sn) {
            self.rx_next_highest = (sn + 1) % modulus;
            let lower_edge = self.um_window_lower_edge();
            let dropped: Vec<u32> = self
                .rx_reassembly
                .keys()
                .copied()
                .filter(|buffered| !self.um_sn_in_reassembly_window(*buffered))
                .collect();
            for stale in dropped {
                debug!(
                    sn = stale,
                    "RLC UM: discarding PDU outside the advanced window"
                );
                self.rx_reassembly.remove(&stale);
            }
            if !self.um_sn_in_reassembly_window(self.rx_next_reassembly)
                && self.rx_next_reassembly != self.rx_next_highest
            {
                self.rx_next_reassembly = self.first_sn_not_reassembled_after(lower_edge);
            }
        }

        // §5.2.2.2.3, first branch: the SDU at RX_Next_Reassembly completed, so
        // the pointer moves to the next SN that has not been delivered.
        if delivered && sn == self.rx_next_reassembly % modulus {
            self.rx_next_reassembly = self.first_sn_not_reassembled_after(self.rx_next_reassembly);
        }

        // §5.2.2.2.3: stop t-Reassembly when RX_Timer_Trigger has been reached,
        // when it has fallen out of the window, or when nothing is missing any
        // more. SNs are ordered by their distance from the window's lower edge
        // so a wrapped comparison still holds.
        if self.t_reassembly_deadline.is_some() {
            let edge = self.um_window_lower_edge();
            let trigger_reached = self.um_sn_distance(edge, self.rx_timer_trigger)
                <= self.um_sn_distance(edge, self.rx_next_reassembly);
            let trigger_out_of_window = !self.um_sn_in_reassembly_window(self.rx_timer_trigger)
                && self.rx_timer_trigger != self.rx_next_highest;
            if trigger_reached || trigger_out_of_window || !self.um_reassembly_outstanding() {
                trace!("RLC UM: stopping t-Reassembly");
                self.t_reassembly_deadline = None;
            }
        }

        if self.t_reassembly_deadline.is_none() && self.um_reassembly_outstanding() {
            if let Some(duration) = self.t_reassembly {
                self.t_reassembly_deadline = Some(Instant::now() + duration);
                self.rx_timer_trigger = self.rx_next_highest;
                trace!(
                    rx_timer_trigger = self.rx_timer_trigger,
                    "RLC UM: starting t-Reassembly"
                );
            }
        }
    }

    /// Whether reassembly is still outstanding, i.e. whether `t-Reassembly`
    /// should be running (TS 38.322 §5.2.2.2.3, last two conditions):
    /// either more than one SN separates `RX_Next_Reassembly` from
    /// `RX_Next_Highest`, or the SDU at `RX_Next_Reassembly` is still missing a
    /// byte segment.
    fn um_reassembly_outstanding(&self) -> bool {
        let gap = self.um_sn_distance(self.rx_next_reassembly, self.rx_next_highest);
        if gap > 1 {
            return true;
        }
        gap == 1 && self.rx_reassembly.contains_key(&self.rx_next_reassembly)
    }

    /// The first SN at or after `from` whose SDU has not been reassembled and
    /// delivered — i.e. either still in the reassembly buffer, or never seen.
    fn first_sn_not_reassembled_after(&self, from: u32) -> u32 {
        let modulus = self.sn_size.modulus();
        let mut sn = from % modulus;
        // Bounded by the window: past RX_Next_Highest nothing has been received,
        // so that SN is by definition not reassembled.
        while sn != self.rx_next_highest % modulus {
            if self.rx_reassembly.contains_key(&sn) {
                return sn;
            }
            sn = (sn + 1) % modulus;
        }
        sn
    }

    /// Configure `t-Reassembly` (TS 38.331 `t-Reassembly`); `None` disables it,
    /// which leaves a partially received SDU buffered indefinitely.
    pub fn set_t_reassembly(&mut self, duration: Option<Duration>) {
        self.t_reassembly = duration;
        if duration.is_none() {
            self.t_reassembly_deadline = None;
        }
    }

    /// Whether `t-Reassembly` is currently running.
    pub fn t_reassembly_running(&self) -> bool {
        self.t_reassembly_deadline.is_some()
    }

    /// `RX_Next_Reassembly` (TS 38.322 §7.1): the SN of the earliest SDU still
    /// awaiting reassembly.
    pub fn rx_next_reassembly(&self) -> u32 {
        self.rx_next_reassembly
    }

    /// `RX_Next_Highest` (TS 38.322 §7.1): one past the highest SN received.
    pub fn rx_next_highest(&self) -> u32 {
        self.rx_next_highest
    }

    /// Number of SNs currently held in the reassembly buffer (observability: an
    /// unbounded buffer was the leak `t-Reassembly` exists to bound).
    pub fn reassembly_buffer_len(&self) -> usize {
        self.rx_reassembly.len()
    }

    /// Drive `t-Reassembly` (TS 38.322 §5.2.2.2.4). Call periodically with the
    /// current time; returns `true` when the timer expired and its actions ran.
    ///
    /// On expiry `RX_Next_Reassembly` advances to the first SN at or after
    /// `RX_Timer_Trigger` that has not been reassembled, every buffered segment
    /// below it is discarded, and the timer restarts if reassembly is still
    /// outstanding.
    pub fn poll_t_reassembly(&mut self, now: Instant) -> bool {
        let Some(deadline) = self.t_reassembly_deadline else {
            return false;
        };
        if now < deadline {
            return false;
        }
        self.t_reassembly_deadline = None;

        let modulus = self.sn_size.modulus();
        let advanced = self.first_sn_not_reassembled_after(self.rx_timer_trigger);
        let previous = self.rx_next_reassembly;
        self.rx_next_reassembly = advanced;

        let discarded: Vec<u32> = self
            .rx_reassembly
            .keys()
            .copied()
            .filter(|sn| {
                let distance = self.um_sn_distance(previous, *sn);
                distance < self.um_sn_distance(previous, advanced % modulus)
            })
            .collect();
        for sn in &discarded {
            self.rx_reassembly.remove(sn);
        }
        debug!(
            rx_next_reassembly = self.rx_next_reassembly,
            discarded = discarded.len(),
            "RLC UM: t-Reassembly expired"
        );

        if self.um_reassembly_outstanding() {
            if let Some(duration) = self.t_reassembly {
                self.t_reassembly_deadline = Some(now + duration);
                self.rx_timer_trigger = self.rx_next_highest;
            }
        }
        true
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

    // ── UM receive window, state variables and t-Reassembly (#34) ─────────────

    /// A UM entity whose `t-Reassembly` fires the moment it is polled, so expiry
    /// is deterministic without sleeping.
    fn um_rx_with_instant_timer() -> RlcEntity {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.set_t_reassembly(Some(Duration::ZERO));
        rx
    }

    /// Encodes a UMD PDU directly, so a test can choose the SN and SI rather
    /// than only what a transmitter happens to produce.
    fn umd(si: SegmentationInfo, sn: u16, so: Option<u16>, data: &[u8]) -> Vec<u8> {
        RlcUmPdu {
            si,
            sn,
            so,
            data: data.to_vec(),
        }
        .encode_sn12()
    }

    #[test]
    fn a_complete_sdu_advances_the_um_state_variables() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.receive_pdu(&umd(SegmentationInfo::FullSdu, 0, None, &[1, 2, 3]));

        assert_eq!(rx.poll_reassembled(), Some(vec![1, 2, 3]));
        assert_eq!(rx.rx_next_highest(), 1, "RX_Next_Highest is one past SN 0");
        assert_eq!(
            rx.rx_next_reassembly(),
            1,
            "RX_Next_Reassembly moves past a delivered SDU"
        );
        assert!(!rx.t_reassembly_running(), "nothing is outstanding");
    }

    /// TS 38.322 §5.2.2.2.2: an SN below `RX_Next_Reassembly` is discarded. UM
    /// has no ARQ, so a repeated SN is a duplicate from below — and delivering it
    /// again would hand PDCP the same packet twice.
    #[test]
    fn a_replayed_complete_sdu_is_delivered_only_once() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let pdu = umd(SegmentationInfo::FullSdu, 0, None, &[0xAA; 8]);

        rx.receive_pdu(&pdu);
        assert_eq!(rx.poll_reassembled(), Some(vec![0xAA; 8]));

        rx.receive_pdu(&pdu);
        assert_eq!(
            rx.poll_reassembled(),
            None,
            "the duplicate must not reach PDCP a second time"
        );
    }

    #[test]
    fn a_stale_sn_is_reported_as_outside_the_reassembly_window() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.receive_pdu(&umd(SegmentationInfo::FullSdu, 0, None, &[1]));
        rx.receive_pdu(&umd(SegmentationInfo::FullSdu, 1, None, &[2]));
        while rx.poll_reassembled().is_some() {}

        // SN 0 has been delivered; a segment for it must not be buffered.
        let err = rx
            .receive_pdu_checked(&umd(SegmentationInfo::FirstSegment, 0, None, &[9, 9]))
            .expect_err("a stale SN must be refused");
        assert!(
            matches!(err, RlcError::SnOutsideWindow { sn: 0 }),
            "expected SnOutsideWindow, got {err:?}"
        );
        assert_eq!(
            rx.reassembly_buffer_len(),
            0,
            "a refused PDU must not occupy the reassembly buffer"
        );
    }

    /// The other direction is NOT a drop: TS 38.322 §5.2.2.2.3 advances
    /// `RX_Next_Highest` to admit an SN ahead of the window. Dropping it would
    /// stall the receiver permanently after a burst loss.
    #[test]
    fn an_sn_ahead_of_the_window_advances_it_instead_of_being_dropped() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.receive_pdu(&umd(SegmentationInfo::FirstSegment, 100, None, &[1, 2]));

        assert_eq!(rx.rx_next_highest(), 101);
        assert_eq!(
            rx.reassembly_buffer_len(),
            1,
            "the segment is buffered, not discarded"
        );
    }

    /// TS 38.322 §5.2.2.2.4: on expiry `RX_Next_Reassembly` advances past the
    /// SDU under reassembly and every segment below it is discarded — otherwise
    /// one lost middle segment strands the SDU and leaks the buffer forever.
    #[test]
    fn t_reassembly_expiry_discards_the_stranded_sdu_and_advances_the_pointer() {
        let mut rx = um_rx_with_instant_timer();

        // SDU 0 arrives as first + last with the middle segment lost.
        rx.receive_pdu(&umd(SegmentationInfo::FirstSegment, 0, None, &[1, 2, 3, 4]));
        rx.receive_pdu(&umd(SegmentationInfo::LastSegment, 0, Some(8), &[9, 9]));
        assert_eq!(rx.reassembly_buffer_len(), 1, "partial SDU is buffered");
        assert!(rx.t_reassembly_running(), "t-Reassembly must be running");
        assert!(rx.poll_reassembled().is_none(), "nothing to deliver yet");

        assert!(
            rx.poll_t_reassembly(Instant::now()),
            "the timer must report the expiry it handled"
        );

        assert_eq!(
            rx.reassembly_buffer_len(),
            0,
            "the stranded segments must be discarded"
        );
        assert_eq!(
            rx.rx_next_reassembly(),
            1,
            "RX_Next_Reassembly must advance past the abandoned SN"
        );
        assert!(
            rx.poll_reassembled().is_none(),
            "a partial SDU must never be delivered"
        );
        assert!(
            !rx.t_reassembly_running(),
            "nothing is outstanding, so the timer must not restart"
        );
    }

    /// After the abandoned SN, the next SDU still gets through: expiry must move
    /// the receiver on rather than wedge it.
    #[test]
    fn reception_continues_after_a_t_reassembly_expiry() {
        let mut rx = um_rx_with_instant_timer();
        rx.receive_pdu(&umd(SegmentationInfo::FirstSegment, 0, None, &[1, 2, 3, 4]));
        rx.poll_t_reassembly(Instant::now());

        rx.receive_pdu(&umd(SegmentationInfo::FullSdu, 1, None, &[7, 7, 7]));
        assert_eq!(rx.poll_reassembled(), Some(vec![7, 7, 7]));
    }

    /// A completed SDU stops the timer: nothing is missing any more.
    #[test]
    fn completing_the_awaited_sdu_stops_t_reassembly() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.receive_pdu(&umd(SegmentationInfo::FirstSegment, 0, None, &[1, 2]));
        assert!(rx.t_reassembly_running());

        rx.receive_pdu(&umd(SegmentationInfo::LastSegment, 0, Some(2), &[3, 4]));
        assert_eq!(rx.poll_reassembled(), Some(vec![1, 2, 3, 4]));
        assert!(
            !rx.t_reassembly_running(),
            "the SDU is complete, so t-Reassembly must stop"
        );
    }

    /// With `t-Reassembly` disabled the entity keeps the pre-#34 behaviour: the
    /// partial SDU waits indefinitely. Configurable rather than removed, because
    /// a bearer configured without the timer is a legal RRC configuration.
    #[test]
    fn a_disabled_t_reassembly_never_discards() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rx.set_t_reassembly(None);
        rx.receive_pdu(&umd(SegmentationInfo::FirstSegment, 0, None, &[1, 2]));

        assert!(!rx.t_reassembly_running());
        assert!(!rx.poll_t_reassembly(Instant::now()));
        assert_eq!(rx.reassembly_buffer_len(), 1);
    }

    /// A repeated byte segment must be refused rather than inserted twice: the
    /// reassembly buffer is keyed by offset, and accepting a second copy of one
    /// offset would either overwrite the first or corrupt the length bookkeeping.
    /// Pinned here because the revert pass showed nothing covered it.
    #[test]
    fn a_duplicate_byte_segment_is_refused() {
        let mut rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        let first = umd(SegmentationInfo::FirstSegment, 0, None, &[1, 2, 3, 4]);

        rx.receive_pdu_checked(&first).expect("first copy accepted");
        let err = rx
            .receive_pdu_checked(&first)
            .expect_err("the duplicate must be refused");
        assert!(
            matches!(err, RlcError::DuplicateSegment { sn: 0, so: 0 }),
            "expected DuplicateSegment{{sn: 0, so: 0}}, got {err:?}"
        );

        // And the SDU still completes from the remaining segment, so the refusal
        // did not damage what was already buffered.
        rx.receive_pdu(&umd(SegmentationInfo::LastSegment, 0, Some(4), &[5, 6]));
        assert_eq!(rx.poll_reassembled(), Some(vec![1, 2, 3, 4, 5, 6]));
    }

    #[test]
    fn um_window_size_is_half_the_sn_space() {
        assert_eq!(SnSize::Sn6.um_window_size(), 32);
        assert_eq!(SnSize::Sn12.um_window_size(), 2048);
    }
}
