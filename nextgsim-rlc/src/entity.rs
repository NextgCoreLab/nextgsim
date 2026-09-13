//! RLC entity — the stateful per-bearer RLC instance
//!
//! One `RlcEntity` is created per RLC bearer (per logical channel).
//! The same entity handles both the transmit and receive sides.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::time::{Duration, Instant};

use tracing::{debug, trace, warn};

use crate::error::RlcError;
use crate::pdu::{RlcAmPdu, RlcStatusNack, RlcStatusPdu, RlcUmPdu, SegmentationInfo};
use crate::{RlcMode, SnSize};

/// Default `t-Reassembly` (TS 38.331 `t-Reassembly`, `ms50`).
///
/// A UM receiver with no `t-Reassembly` keeps a partially received SDU forever,
/// so the reassembly buffer grows without bound on a lossy channel; 50 ms is
/// well inside the enumerated range and far above the in-process RLS latency, so
/// a lossless path never sees it fire.
pub const DEFAULT_T_REASSEMBLY: Duration = Duration::from_millis(50);

/// Default `t-PollRetransmit` (TS 38.331 `t-PollRetransmit`, `ms45`).
///
/// The AM sender restarts it whenever it sends a poll; expiry means the STATUS
/// report never came back, so the poll is repeated. Without it a lost STATUS
/// report stalls the bearer permanently, because nothing else asks again.
pub const DEFAULT_T_POLL_RETRANSMIT: Duration = Duration::from_millis(45);

/// Default `t-StatusProhibit` (TS 38.331 `t-StatusProhibit`, `ms10`).
///
/// Bounds how often the receiver may answer with a STATUS report, so a bearer
/// losing many PDUs does not answer every poll with its own control PDU.
pub const DEFAULT_T_STATUS_PROHIBIT: Duration = Duration::from_millis(10);

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

    // ── AM ARQ state (TS 38.322 §7.1) ────────────────────────────────────────
    /// PDUs that have been sent but not yet acknowledged, keyed by SN
    am_unacked: BTreeMap<u32, Vec<u8>>,
    /// SNs pending retransmission (added on NACK or `t-PollRetransmit` expiry)
    am_retx_queue: VecDeque<u32>,
    /// `TX_Next_Ack`: the lowest SN still awaiting acknowledgement
    tx_next_ack: u32,
    /// `RX_Next`: the next SN awaited in sequence on the AM receive side
    am_rx_next: u32,
    /// `RX_Next_Highest`: one past the highest SN received
    am_rx_next_highest: u32,
    /// `RX_Highest_Status`: the SN up to which a STATUS report reports
    am_rx_highest_status: u32,
    /// SNs whose SDU is fully received, above `RX_Next` (everything below it is
    /// received by definition, so those are pruned)
    am_received: BTreeSet<u32>,
    /// `RX_Next_Status_Trigger`: the SN `t-Reassembly` is waiting past
    am_rx_next_status_trigger: u32,
    /// Whether a STATUS report has been triggered and not yet built
    am_status_triggered: bool,
    /// `t-PollRetransmit` duration (TS 38.331); `None` disables it
    t_poll_retransmit: Option<Duration>,
    /// Deadline of a running `t-PollRetransmit`
    t_poll_retransmit_deadline: Option<Instant>,
    /// `t-StatusProhibit` duration (TS 38.331); `None` reports on every trigger
    t_status_prohibit: Option<Duration>,
    /// Deadline of a running `t-StatusProhibit`
    t_status_prohibit_deadline: Option<Instant>,

    // ── RX side ───────────────────────────────────────────────────────────────
    /// Reassembly state per SN: `BTreeMap<sn, Reassembly>`
    rx_reassembly: BTreeMap<u32, Reassembly>,
    /// Fully reassembled SDUs ready for PDCP
    rx_ready: VecDeque<Vec<u8>>,

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
            tx_next_ack: 0,
            am_rx_next: 0,
            am_rx_next_highest: 0,
            am_rx_highest_status: 0,
            am_received: BTreeSet::new(),
            am_rx_next_status_trigger: 0,
            am_status_triggered: false,
            t_poll_retransmit: Some(DEFAULT_T_POLL_RETRANSMIT),
            t_poll_retransmit_deadline: None,
            t_status_prohibit: Some(DEFAULT_T_STATUS_PROHIBIT),
            t_status_prohibit_deadline: None,
            rx_reassembly: BTreeMap::new(),
            rx_ready: VecDeque::new(),
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

    /// Drives whichever timers this entity's mode has, returning `true` when one
    /// expired: `t-Reassembly` for UM, `t-PollRetransmit` and `t-Reassembly` for
    /// AM, nothing for TM.
    ///
    /// The single entry point exists so a caller ticking a map of entities does
    /// not have to know each one's mode — the reason the AM timers went unwired
    /// before was that no caller had a reason to look.
    pub fn poll_timers(&mut self, now: Instant) -> bool {
        match self.mode {
            RlcMode::UnacknowledgedMode => self.poll_t_reassembly(now),
            RlcMode::AcknowledgedMode => self.poll_am_timers(now),
            RlcMode::TransparentMode => false,
        }
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
        if self.mode != RlcMode::UnacknowledgedMode {
            // AM shares the timer field but not the state variables it advances;
            // its expiry actions are in `poll_am_timers`.
            return false;
        }
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
        // Prioritise retransmissions (TS 38.322 §5.2.3.1.1)
        while let Some(retx_sn) = self.am_retx_queue.front().copied() {
            let Some(orig) = self.am_unacked.get(&retx_sn) else {
                // SN no longer unacked (got ACKed between enqueue and now)
                self.am_retx_queue.pop_front();
                continue;
            };
            if orig.len() > max_size {
                // Re-segmentation of an oversized retransmission (§5.2.3.1.1) is
                // not implemented; the PDU stays queued rather than being
                // dropped, so a later, larger grant still carries it.
                warn!(
                    sn = retx_sn,
                    len = orig.len(),
                    max_size,
                    "RLC AM: retransmission does not fit the grant, keeping it queued"
                );
                break;
            }
            let pdu = orig.clone();
            self.am_retx_queue.pop_front();
            debug!(sn = retx_sn, "RLC AM retransmitting PDU");
            // A retransmission carries a poll (§5.3.3.2), so the sender learns
            // whether this copy arrived.
            self.am_start_poll_retransmit();
            return Some(pdu);
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

        // TS 38.322 §5.3.3.2: sending a poll starts (restarts) t-PollRetransmit.
        // Before this the timer field existed and was never read, so a lost
        // STATUS report stalled the bearer with no recovery.
        if p {
            self.am_start_poll_retransmit();
        }

        debug!(si = ?si, sn, payload_len, poll = p, "RLC AM build_pdu");
        Some(encoded)
    }

    /// Starts or restarts `t-PollRetransmit` (TS 38.322 §5.3.3.2).
    fn am_start_poll_retransmit(&mut self) {
        if let Some(duration) = self.t_poll_retransmit {
            self.t_poll_retransmit_deadline = Some(Instant::now() + duration);
        }
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
        let polled = pdu.p;
        trace!(si = ?pdu.si, sn, poll = polled, "RLC AM receive data PDU");

        let complete = if pdu.si == SegmentationInfo::FullSdu {
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
                if let RlcError::DuplicateSegment { sn: ref mut s, .. } = e {
                    *s = sn;
                }
                // A duplicate is not a protocol failure on the AM path: the peer
                // retransmitted something already held. The STATUS trigger still
                // has to be honoured, or a poll on a duplicate goes unanswered
                // and the sender re-polls forever.
                if polled {
                    self.am_status_triggered = true;
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

        let modulus = self.sn_size.modulus();
        // TS 38.322 §5.2.3.2.3: RX_Next_Highest is one past the highest SN
        // received.
        if self.am_sn_distance(self.am_rx_next_highest, sn) < self.sn_size.am_window_size() {
            self.am_rx_next_highest = (sn + 1) % modulus;
        }

        if complete {
            // §5.2.3.2.3 delivers each SDU AS IT COMPLETES — AM does not reorder
            // in RLC (that is PDCP's t-Reordering, TS 38.323 §5.2.2). The state
            // variables below are what makes the STATUS report right, not a
            // delivery order.
            self.am_received.insert(sn);
            if sn == self.am_rx_highest_status % modulus {
                self.am_rx_highest_status = self.am_first_not_received_after(sn + 1);
            }
            if sn == self.am_rx_next % modulus {
                self.am_rx_next = self.am_first_not_received_after(sn + 1);
                // Everything below RX_Next is received by definition, so the
                // set only has to remember what arrived out of order above it.
                let next = self.am_rx_next;
                let window = self.sn_size.am_window_size();
                let modulus = self.sn_size.modulus();
                self.am_received
                    .retain(|held| (held + modulus - next % modulus) % modulus < window);
            }
        }

        // §5.3.4: a poll triggers a STATUS report. The spec delays it until the
        // polled SN is outside the receiving window (to let HARQ reordering
        // finish); there is no HARQ here, so it is honoured immediately.
        if polled {
            self.am_status_triggered = true;
        }
        self.am_update_reassembly_timer();
        Ok(())
    }

    /// The first SN at or after `from` whose SDU is not fully received
    /// (TS 38.322 §5.2.3.2.3 / §5.2.3.2.4), bounded by `RX_Next_Highest` because
    /// nothing beyond it has arrived at all.
    fn am_first_not_received_after(&self, from: u32) -> u32 {
        let modulus = self.sn_size.modulus();
        let mut sn = from % modulus;
        while sn != self.am_rx_next_highest % modulus {
            if !self.am_received.contains(&sn) {
                return sn;
            }
            sn = (sn + 1) % modulus;
        }
        sn
    }

    /// Distance from `base` forward to `sn`, modulo the AM SN space.
    fn am_sn_distance(&self, base: u32, sn: u32) -> u32 {
        let modulus = self.sn_size.modulus();
        (sn + modulus - base % modulus) % modulus
    }

    /// Starts or stops the AM `t-Reassembly` (TS 38.322 §5.2.3.2.3).
    ///
    /// It runs while an SDU at or above `RX_Next` is still missing: either more
    /// than one SN separates `RX_Next` from `RX_Next_Highest`, or the SDU at
    /// `RX_Next` is itself partly received.
    fn am_update_reassembly_timer(&mut self) {
        let modulus = self.sn_size.modulus();
        if self.t_reassembly_deadline.is_some() {
            // Stop when what the timer was waiting past has been reached.
            let trigger_reached = self.am_rx_next_status_trigger % modulus
                == self.am_rx_next % modulus
                || (self.am_sn_distance(self.am_rx_next, self.am_rx_next_status_trigger) == 1
                    && !self
                        .rx_reassembly
                        .contains_key(&(self.am_rx_next % modulus)));
            if trigger_reached {
                self.t_reassembly_deadline = None;
            }
        }
        if self.t_reassembly_deadline.is_none() && self.am_reassembly_outstanding() {
            if let Some(duration) = self.t_reassembly {
                self.t_reassembly_deadline = Some(Instant::now() + duration);
                self.am_rx_next_status_trigger = self.am_rx_next_highest;
            }
        }
    }

    /// Whether an SDU at or above `RX_Next` is still missing bytes
    /// (TS 38.322 §5.2.3.2.3, the two `start t-Reassembly` conditions).
    fn am_reassembly_outstanding(&self) -> bool {
        let modulus = self.sn_size.modulus();
        let gap = self.am_sn_distance(self.am_rx_next, self.am_rx_next_highest);
        if gap > 1 {
            return true;
        }
        gap == 1
            && self
                .rx_reassembly
                .contains_key(&(self.am_rx_next % modulus))
    }

    /// Builds a STATUS PDU when one is due (TS 38.322 §5.3.4), or `None`.
    ///
    /// Returns `None` when no report is triggered or `t-StatusProhibit` is still
    /// running — the prohibit timer is what stops a lossy bearer from answering
    /// every poll with its own PDU. Submitting a report starts the timer.
    pub fn build_status_pdu(&mut self) -> Option<Vec<u8>> {
        if self.mode != RlcMode::AcknowledgedMode || !self.am_status_triggered {
            return None;
        }
        let now = Instant::now();
        if let Some(deadline) = self.t_status_prohibit_deadline {
            if now < deadline {
                return None;
            }
            self.t_status_prohibit_deadline = None;
        }

        let modulus = self.sn_size.modulus();
        let mut nacks = Vec::new();
        let mut sn = self.am_rx_next % modulus;
        while sn != self.am_rx_highest_status % modulus {
            if !self.am_received.contains(&sn) {
                nacks.push(RlcStatusNack::sdu(sn));
            }
            sn = (sn + 1) % modulus;
        }

        // §6.2.3.10: ACK_SN is the next SN not received and not reported as
        // missing, i.e. the top of the reported range.
        let status = RlcStatusPdu::with_nacks(self.am_rx_highest_status % modulus, nacks);
        let encoded = match self.sn_size {
            SnSize::Sn12 => status.encode_sn12(),
            SnSize::Sn18 => status.encode_sn18(),
            SnSize::Sn6 => unreachable!("6-bit SN is UM only"),
        };
        debug!(
            ack_sn = status.ack_sn,
            nacks = status.nacks.len(),
            "RLC AM: sending STATUS report"
        );

        self.am_status_triggered = false;
        if let Some(duration) = self.t_status_prohibit {
            self.t_status_prohibit_deadline = Some(now + duration);
        }
        Some(encoded)
    }

    /// Whether a STATUS report is pending (triggered and not yet built).
    pub fn status_report_pending(&self) -> bool {
        self.am_status_triggered
    }

    fn handle_status_pdu(&mut self, data: &[u8]) -> Result<(), RlcError> {
        let status = match self.sn_size {
            SnSize::Sn12 => RlcStatusPdu::decode_sn12(data)?,
            SnSize::Sn18 => RlcStatusPdu::decode_sn18(data)?,
            // A 6-bit SN entity is UM, which has no STATUS PDU; reaching here
            // means an entity was built with an illegal mode/SN pair, which
            // `RlcEntity::new` rejects.
            SnSize::Sn6 => unreachable!("6-bit SN is UM only"),
        };
        let ack_sn = status.ack_sn;
        debug!(
            ack_sn,
            nacks = status.nacks.len(),
            "RLC AM received STATUS PDU"
        );

        // §6.2.3.10: everything below ACK_SN is acknowledged EXCEPT the NACKed
        // SNs. Clearing them all would drop exactly the PDUs that need resending.
        let nacked: Vec<u32> = status
            .nacks
            .iter()
            .flat_map(|nack| {
                let run = u32::from(nack.nack_range.unwrap_or(0)).max(1);
                (0..run).map(move |offset| nack.nack_sn + offset)
            })
            .collect();
        let modulus = self.sn_size.modulus();
        let acked: Vec<u32> = self
            .am_unacked
            .keys()
            .copied()
            .filter(|sn| {
                self.am_sn_distance(self.tx_next_ack, *sn)
                    < self.am_sn_distance(self.tx_next_ack, ack_sn)
                    && !nacked.contains(&(sn % modulus))
            })
            .collect();
        for sn in acked {
            self.am_unacked.remove(&sn);
            self.am_retx_queue.retain(|queued| *queued != sn);
        }

        // §5.3.2: a negative acknowledgement schedules retransmission. Nothing
        // else does: before this, `request_retransmit` had to be called by hand,
        // so a running AM bearer never retransmitted at all.
        for sn in nacked {
            let sn = sn % modulus;
            if self.am_unacked.contains_key(&sn) && !self.am_retx_queue.contains(&sn) {
                debug!(sn, "RLC AM: NACK received, scheduling retransmission");
                self.am_retx_queue.push_back(sn);
            }
        }

        // TX_Next_Ack is the lowest SN still awaiting acknowledgement.
        self.tx_next_ack = self
            .am_unacked
            .keys()
            .copied()
            .min_by_key(|sn| self.am_sn_distance(self.tx_next_ack, *sn))
            .unwrap_or(ack_sn % modulus);

        // A STATUS report answered the poll, so the poll retransmit timer stops.
        self.t_poll_retransmit_deadline = None;
        Ok(())
    }

    /// Drives the AM timers (`t-PollRetransmit`, `t-Reassembly`). Call
    /// periodically with the current time; returns `true` when a timer expired.
    ///
    /// `t-PollRetransmit` expiry re-offers an unacknowledged PDU with a poll
    /// (TS 38.322 §5.3.3.4) — the recovery path when the STATUS report itself is
    /// lost. `t-Reassembly` expiry advances `RX_Highest_Status` over the missing
    /// SDU and triggers a STATUS report (§5.2.3.2.4, §5.3.4).
    pub fn poll_am_timers(&mut self, now: Instant) -> bool {
        if self.mode != RlcMode::AcknowledgedMode {
            return false;
        }
        let mut expired = false;

        if let Some(deadline) = self.t_poll_retransmit_deadline {
            if now >= deadline {
                self.t_poll_retransmit_deadline = None;
                expired = true;
                // §5.3.3.4: consider the highest-SN unacknowledged SDU for
                // retransmission so the poll is repeated.
                if let Some(&sn) = self.am_unacked.keys().next_back() {
                    if !self.am_retx_queue.contains(&sn) {
                        debug!(sn, "RLC AM: t-PollRetransmit expired, re-offering PDU");
                        self.am_retx_queue.push_back(sn);
                    }
                    if let Some(duration) = self.t_poll_retransmit {
                        self.t_poll_retransmit_deadline = Some(now + duration);
                    }
                }
            }
        }

        if let Some(deadline) = self.t_reassembly_deadline {
            if now >= deadline {
                self.t_reassembly_deadline = None;
                expired = true;
                // §5.2.3.2.4: RX_Highest_Status moves to the first SDU at or
                // after RX_Next_Status_Trigger that is not fully received, so the
                // report that follows names every gap below it. §5.3.4 then
                // triggers that report -- in this order, per its NOTE 2.
                self.am_rx_highest_status =
                    self.am_first_not_received_after(self.am_rx_next_status_trigger);
                self.am_status_triggered = true;
                debug!(
                    rx_highest_status = self.am_rx_highest_status,
                    "RLC AM: t-Reassembly expired, STATUS report triggered"
                );
                self.am_update_reassembly_timer();
            }
        }

        expired
    }

    /// `RX_Next` (TS 38.322 §7.1): the SN of the next AM SDU awaited in
    /// sequence.
    pub fn am_rx_next(&self) -> u32 {
        self.am_rx_next
    }

    /// `TX_Next_Ack` (TS 38.322 §7.1): the lowest SN still awaiting
    /// acknowledgement.
    pub fn tx_next_ack(&self) -> u32 {
        self.tx_next_ack
    }

    /// Number of AM PDUs sent and not yet acknowledged.
    pub fn unacked_len(&self) -> usize {
        self.am_unacked.len()
    }

    /// Configure `t-PollRetransmit` (TS 38.331 `t-PollRetransmit`); `None`
    /// disables it, leaving a lost STATUS report unrecovered.
    pub fn set_t_poll_retransmit(&mut self, duration: Option<Duration>) {
        self.t_poll_retransmit = duration;
        if duration.is_none() {
            self.t_poll_retransmit_deadline = None;
        }
    }

    /// Configure `t-StatusProhibit` (TS 38.331 `t-StatusProhibit`); `None`
    /// sends a STATUS report for every trigger.
    pub fn set_t_status_prohibit(&mut self, duration: Option<Duration>) {
        self.t_status_prohibit = duration;
        if duration.is_none() {
            self.t_status_prohibit_deadline = None;
        }
    }

    /// Request retransmission of a specific SN.
    ///
    /// Retransmission is normally driven by the NACK list of a received STATUS
    /// PDU; this is the manual entry point for a caller that has its own reason
    /// to resend (and the pre-#15 behaviour, when nothing else did).
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

    // ── AM ARQ: STATUS, auto-retransmission and timers (#15) ─────────────────

    /// A pair of AM entities with instant timers, so expiry is deterministic
    /// without sleeping, and no STATUS prohibit in the way of a report.
    fn am_pair() -> (RlcEntity, RlcEntity) {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        tx.set_t_poll_retransmit(Some(Duration::ZERO));
        rx.set_t_status_prohibit(None);
        rx.set_t_reassembly(Some(Duration::ZERO));
        (tx, rx)
    }

    /// The whole point of AM: a dropped PDU is recovered **without any manual
    /// `request_retransmit` call**. The receiver NACKs it, the sender resends it,
    /// and the SDUs are delivered in order.
    #[test]
    fn a_dropped_am_pdu_is_recovered_by_status_and_retransmission() {
        let (mut tx, mut rx) = am_pair();
        let sdus: Vec<Vec<u8>> = (0..3u8).map(|i| vec![0xA0 + i; 12]).collect();
        for sdu in &sdus {
            tx.submit_sdu(sdu.clone());
        }

        let pdus: Vec<Vec<u8>> = std::iter::from_fn(|| tx.build_pdu(64)).collect();
        assert_eq!(pdus.len(), 3, "one PDU per SDU at this grant size");

        // SN 1 is lost on the air.
        rx.receive_pdu(&pdus[0]);
        rx.receive_pdu(&pdus[2]);

        assert_eq!(
            rx.poll_reassembled(),
            Some(sdus[0].clone()),
            "SN 0 is delivered"
        );
        assert_eq!(
            rx.poll_reassembled(),
            Some(sdus[2].clone()),
            "SN 2 is delivered as it completes: AM does not reorder in RLC \
             (TS 38.322 §5.2.3.2.3), PDCP does"
        );

        // The poll on SN 2 triggers a report, but RX_Highest_Status has not moved
        // past the gap yet, so this first report only ACKs SN 0
        // (TS 38.322 §5.2.3.2.3: RX_Highest_Status advances only over SDUs that
        // ARE received). Reporting SN 1 missing this early would be a guess.
        let acked_only = RlcStatusPdu::decode_sn12(
            &rx.build_status_pdu()
                .expect("the poll must be answered with a report"),
        )
        .unwrap();
        assert_eq!(acked_only.ack_sn, 1);
        assert!(
            acked_only.nacks.is_empty(),
            "the gap is not reported until t-Reassembly says it is lost"
        );

        // t-Reassembly is what declares SN 1 lost (§5.2.3.2.4) and triggers the
        // report that NACKs it (§5.3.4).
        assert!(
            rx.poll_am_timers(Instant::now()),
            "t-Reassembly must expire over the gap"
        );
        let status = rx
            .build_status_pdu()
            .expect("the expiry must trigger a second report");
        let decoded = RlcStatusPdu::decode_sn12(&status).unwrap();
        assert_eq!(
            decoded.nacks.iter().map(|n| n.nack_sn).collect::<Vec<_>>(),
            vec![1],
            "exactly the missing SN is NACKed"
        );
        assert_eq!(decoded.ack_sn, 3, "everything reported on, up to SN 3");

        // The sender acts on the NACK by itself.
        tx.receive_pdu(&status);
        assert_eq!(
            tx.unacked_len(),
            1,
            "SN 0 and SN 2 are acknowledged; only the NACKed SN remains"
        );
        let retx = tx
            .build_pdu(64)
            .expect("the NACK alone must schedule the retransmission");
        assert_eq!(retx, pdus[1], "the retransmission is the lost PDU");

        rx.receive_pdu(&retx);
        assert_eq!(rx.poll_reassembled(), Some(sdus[1].clone()));
        assert_eq!(rx.am_rx_next(), 3, "RX_Next is past all three SDUs");
    }

    /// An ACK_SN above a NACKed SN must not clear that SN: the NACK is the
    /// exception ACK_SN carries (TS 38.322 §6.2.3.10). Clearing it would drop
    /// exactly the PDU that has to be resent.
    #[test]
    fn a_nacked_sn_survives_an_ack_sn_above_it() {
        let (mut tx, _rx) = am_pair();
        for i in 0..3u8 {
            tx.submit_sdu(vec![i; 8]);
        }
        let pdus: Vec<Vec<u8>> = std::iter::from_fn(|| tx.build_pdu(64)).collect();
        assert_eq!(tx.unacked_len(), 3);

        // ACK up to 3 (i.e. 0, 1 and 2 reported on) but NACK SN 1.
        let status = RlcStatusPdu::with_nacks(3, vec![RlcStatusNack::sdu(1)]).encode_sn12();
        tx.receive_pdu(&status);

        assert_eq!(tx.unacked_len(), 1, "only the NACKed SN stays unacked");
        let retx = tx.build_pdu(64).expect("the NACKed SN is retransmitted");
        assert_eq!(retx, pdus[1]);
    }

    /// TS 38.322 §5.3.3.4: if no STATUS comes back, `t-PollRetransmit` expiry
    /// re-offers an unacknowledged PDU. Without it a lost STATUS report stalls
    /// the bearer, because the receiver has nothing left to answer.
    #[test]
    fn t_poll_retransmit_expiry_re_offers_the_unacknowledged_pdu() {
        let (mut tx, _rx) = am_pair();
        tx.submit_sdu(vec![0x5Au8; 16]);
        let original = tx.build_pdu(64).expect("original PDU");
        assert!(
            tx.build_pdu(64).is_none(),
            "nothing more to send until a timer fires"
        );

        assert!(
            tx.poll_am_timers(Instant::now()),
            "t-PollRetransmit must have expired"
        );

        let retx = tx
            .build_pdu(64)
            .expect("the expiry must re-offer the unacknowledged PDU");
        assert_eq!(retx, original);
    }

    /// `t-StatusProhibit` bounds the report rate: two triggers while it runs
    /// produce one report, not two.
    #[test]
    fn t_status_prohibit_collapses_repeated_triggers_into_one_report() {
        let mut tx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        // Long enough that it is still running for the second poll.
        rx.set_t_status_prohibit(Some(Duration::from_secs(60)));

        tx.submit_sdu(vec![1u8; 8]);
        tx.submit_sdu(vec![2u8; 8]);
        let first = tx.build_pdu(64).unwrap();
        let second = tx.build_pdu(64).unwrap();

        rx.receive_pdu(&first);
        assert!(
            rx.build_status_pdu().is_some(),
            "the first poll is answered"
        );

        rx.receive_pdu(&second);
        assert!(
            rx.status_report_pending(),
            "the second poll still triggers a report"
        );
        assert!(
            rx.build_status_pdu().is_none(),
            "but t-StatusProhibit holds it back"
        );
    }

    /// An AM `t-Reassembly` expiry reports on the missing SDU instead of waiting
    /// for a poll that may never come (TS 38.322 §5.2.3.2.4, §5.3.4).
    #[test]
    fn am_t_reassembly_expiry_triggers_a_status_report() {
        let mut rx = RlcEntity::new(RlcMode::AcknowledgedMode, SnSize::Sn12);
        rx.set_t_status_prohibit(None);
        rx.set_t_reassembly(Some(Duration::ZERO));

        // SN 1 arrives with no poll; SN 0 is missing.
        let pdu = RlcAmPdu {
            dc: true,
            p: false,
            si: SegmentationInfo::FullSdu,
            sn: 1,
            so: None,
            data: vec![7, 7, 7],
        }
        .encode_sn12();
        rx.receive_pdu(&pdu);
        assert!(
            !rx.status_report_pending(),
            "no poll, so nothing is triggered yet"
        );

        assert!(
            rx.poll_am_timers(Instant::now()),
            "t-Reassembly must expire"
        );
        assert!(rx.status_report_pending());

        let status = RlcStatusPdu::decode_sn12(&rx.build_status_pdu().unwrap()).unwrap();
        assert_eq!(
            status.nacks.iter().map(|n| n.nack_sn).collect::<Vec<_>>(),
            vec![0],
            "the report names the missing SDU"
        );
    }

    /// A UM entity has no STATUS PDU at all, and an AM entity's timers must not
    /// be driven by the UM helper — they advance different state variables.
    #[test]
    fn a_um_entity_never_produces_a_status_report() {
        let mut um = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        um.receive_pdu(&umd(SegmentationInfo::FullSdu, 0, None, &[1, 2]));
        assert!(um.build_status_pdu().is_none());
        assert!(!um.poll_am_timers(Instant::now()));
    }

    #[test]
    fn am_window_size_is_half_the_sn_space() {
        assert_eq!(SnSize::Sn12.am_window_size(), 2048);
        assert_eq!(SnSize::Sn18.am_window_size(), 131072);
    }

    #[test]
    fn um_window_size_is_half_the_sn_space() {
        assert_eq!(SnSize::Sn6.um_window_size(), 32);
        assert_eq!(SnSize::Sn12.um_window_size(), 2048);
    }
}
