//! NR PDCP entity for DRB user-plane bearers (3GPP TS 38.323)
//!
//! One entity per DRB, sitting between the upper layer (NAS/TUN on the UE, GTP-U
//! on the gNB) and RLC. Before this crate the two data paths wired RLC directly to
//! those upper layers (issue #33), so a DRB had no PDCP header, no sequence number,
//! no COUNT, no in-order delivery and no discard timer.
//!
//! ## What is implemented
//!
//! - **Transmit** (§5.2.1): `TX_NEXT`, the PDCP SN, the 32-bit COUNT, the data-PDU
//!   header, and a `discardTimer` armed per SDU whose expiry discards the SDU
//!   before it reaches RLC.
//! - **Receive** (§5.2.2.1): `RX_NEXT`, `RX_DELIV`, `RX_REORD`, the reordering
//!   window, duplicate and out-of-window discard, and strictly in-order delivery.
//! - **`t-Reordering`** (§5.2.2.2): on expiry, everything below `RX_REORD` is
//!   delivered and the state advances.
//!
//! ## What is deliberately not
//!
//! - **ROHC / EHC header compression.** All ROHC profiles are advertised
//!   unsupported by `nextgsim-rrc`, so compressing here would contradict the
//!   capability on the wire.
//! - **User-plane ciphering and integrity.** They belong to the separate UP
//!   security work, for which this entity's COUNT is the prerequisite. The COUNT
//!   is therefore exposed (`Pdcp::tx_count_for_next_sdu`, `PdcpPdu::count`) so that
//!   work needs no change here.
//! - **`outOfOrderDelivery`.** §5.2.2.1 allows immediate delivery when configured;
//!   this entity always reorders, which is the behaviour a DRB gets by default.
//!
//! ## Time is a parameter, never read here
//!
//! Every method that could care about time takes a `now_ms`. A PDCP entity that
//! read a clock would be untestable at the millisecond boundaries that decide
//! whether a discard timer expired, and this crate has no business choosing the
//! simulator's time source.

pub mod srb_security;
pub use srb_security::{SrbSecurity, SrbSecurityError, MAC_I_LEN};

use std::collections::BTreeMap;

use tracing::{debug, trace, warn};

/// PDCP SN length for a DRB (TS 38.331 `PDCP-Config.drb.pdcp-SN-SizeUL`/`DL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PdcpSnSize {
    /// 12-bit SN: a 2-octet data-PDU header.
    #[default]
    Sn12,
    /// 18-bit SN: a 3-octet data-PDU header.
    Sn18,
}

impl PdcpSnSize {
    /// Number of SN bits.
    pub fn bits(self) -> u32 {
        match self {
            Self::Sn12 => 12,
            Self::Sn18 => 18,
        }
    }

    /// Size of the data-PDU header in octets.
    pub fn header_len(self) -> usize {
        match self {
            Self::Sn12 => 2,
            Self::Sn18 => 3,
        }
    }

    /// `2^bits` — the SN modulus.
    pub fn sn_modulus(self) -> u32 {
        1u32 << self.bits()
    }

    /// `Window_Size = 2^(SN length - 1)` (TS 38.323 §7.2).
    pub fn window_size(self) -> u32 {
        1u32 << (self.bits() - 1)
    }
}

/// Why a received PDU was not delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdcpReceiveError {
    /// The PDU is shorter than its own header, so it carries no SN.
    Truncated,
    /// The D/C bit says this is a control PDU; this entity handles data PDUs.
    ///
    /// Not an error the peer made: PDCP control PDUs (status report, EHC feedback)
    /// are legitimate and simply have no place on this path yet.
    ControlPdu,
    /// `RCVD_COUNT < RX_DELIV`: the PDU is older than what has already been
    /// delivered, so delivering it would break in-order delivery (§5.2.2.1).
    BelowDeliveryWindow,
    /// A PDU with this COUNT has already been received.
    Duplicate,
}

/// A PDCP data PDU decoded far enough to place it in the reordering buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PdcpPdu {
    /// The COUNT this PDU was received at (`RCVD_COUNT`).
    pub count: u32,
    /// The SDU payload, after the header.
    pub sdu: Vec<u8>,
}

/// Configuration of one DRB's PDCP entity (TS 38.331 `PDCP-Config`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdcpConfig {
    /// SN length.
    pub sn_size: PdcpSnSize,
    /// `discardTimer` in milliseconds. `None` means "not configured", in which
    /// case §5.2.1 arms no timer and an SDU is never discarded for age.
    pub discard_timer_ms: Option<u64>,
    /// `t-Reordering` in milliseconds. `None` disables reordering *timeouts*, not
    /// reordering: a gap then waits indefinitely, which is what an unconfigured
    /// `t-Reordering` means.
    pub t_reordering_ms: Option<u64>,
}

impl Default for PdcpConfig {
    fn default() -> Self {
        Self {
            sn_size: PdcpSnSize::Sn12,
            // TS 38.331 offers ms10..ms1500 plus infinity. 500 ms discards a
            // genuinely stalled SDU while leaving room for the 500 ms task tick
            // that drains the transmit queue -- a shorter value would discard SDUs
            // the scheduler simply had not reached yet.
            discard_timer_ms: Some(500),
            // ms0..ms3000. 1000 ms, and the reason is a real interaction rather
            // than a preference: t-Reordering MUST exceed the RLC AM recovery
            // cycle, or PDCP gives up on a COUNT that ARQ is about to recover and
            // then discards the recovered PDU as too old (§5.2.2.1). In this
            // simulator that cycle is bounded by the 500 ms task tick which drives
            // both the STATUS report and the retransmission, so anything under
            // ~1 s defeats AM entirely. Found by an AM bearer test failing exactly
            // that way.
            t_reordering_ms: Some(1_000),
        }
    }
}

/// An SDU held on the transmit side awaiting submission to RLC.
#[derive(Debug, Clone)]
struct PendingSdu {
    count: u32,
    pdu: Vec<u8>,
    armed_at_ms: u64,
}

/// A PDCP entity for one DRB (TS 38.323 §5.2).
#[derive(Debug)]
pub struct Pdcp {
    config: PdcpConfig,

    // -- transmit state (§5.2.1) --
    /// COUNT of the next PDCP SDU to be transmitted.
    tx_next: u32,
    /// PDUs formed but not yet handed to RLC, so `discardTimer` has something to
    /// discard. An entity that submitted straight to RLC would have no window in
    /// which a discard could happen at all.
    tx_pending: Vec<PendingSdu>,

    // -- receive state (§5.2.2) --
    /// COUNT of the next PDCP SDU expected to be received.
    rx_next: u32,
    /// COUNT of the first PDCP SDU not delivered to upper layers, but still
    /// waited for.
    rx_deliv: u32,
    /// COUNT following the PDU that triggered `t-Reordering`.
    rx_reord: u32,
    /// Reception buffer, keyed by COUNT so iteration is in COUNT order.
    rx_buffer: BTreeMap<u32, Vec<u8>>,
    /// When `t-Reordering` was started, if it is running.
    t_reordering_started_ms: Option<u64>,
    /// Counters, so a caller can see what was dropped rather than inferring it
    /// from missing payload.
    discarded_duplicates: u64,
    discarded_out_of_window: u64,
    discarded_on_expiry: u64,
}

impl Pdcp {
    /// Create an entity with the given configuration.
    pub fn new(config: PdcpConfig) -> Self {
        Self {
            config,
            tx_next: 0,
            tx_pending: Vec::new(),
            rx_next: 0,
            rx_deliv: 0,
            rx_reord: 0,
            rx_buffer: BTreeMap::new(),
            t_reordering_started_ms: None,
            discarded_duplicates: 0,
            discarded_out_of_window: 0,
            discarded_on_expiry: 0,
        }
    }

    /// The configuration in force.
    pub fn config(&self) -> &PdcpConfig {
        &self.config
    }

    /// The COUNT the next submitted SDU will be associated with (`TX_NEXT`).
    ///
    /// Exposed for the user-plane security work, which needs the COUNT that
    /// ciphering and integrity are keyed on (§5.8, §5.9).
    pub fn tx_count_for_next_sdu(&self) -> u32 {
        self.tx_next
    }

    /// `RX_DELIV`: the COUNT of the first SDU not yet delivered upward.
    pub fn rx_deliv(&self) -> u32 {
        self.rx_deliv
    }

    /// `RX_NEXT`: one past the highest COUNT received.
    pub fn rx_next(&self) -> u32 {
        self.rx_next
    }

    /// `RX_REORD`, meaningful while `t-Reordering` runs.
    pub fn rx_reord(&self) -> u32 {
        self.rx_reord
    }

    /// Whether `t-Reordering` is running.
    pub fn t_reordering_running(&self) -> bool {
        self.t_reordering_started_ms.is_some()
    }

    /// How many SDUs are held in the reordering buffer.
    pub fn buffered_count(&self) -> usize {
        self.rx_buffer.len()
    }

    /// How many PDUs were discarded as duplicates.
    pub fn discarded_duplicates(&self) -> u64 {
        self.discarded_duplicates
    }

    /// How many PDUs were discarded as older than `RX_DELIV`.
    pub fn discarded_out_of_window(&self) -> u64 {
        self.discarded_out_of_window
    }

    /// How many transmit SDUs were discarded by `discardTimer` expiry.
    pub fn discarded_on_expiry(&self) -> u64 {
        self.discarded_on_expiry
    }

    // ========================================================================
    // Transmit (TS 38.323 §5.2.1)
    // ========================================================================

    /// Accept an SDU from upper layers, form its PDCP data PDU and arm its
    /// `discardTimer`.
    ///
    /// The PDU is *held*, not submitted: `take_transmittable` hands it to RLC.
    /// A discard timer that could never fire because the PDU had already left
    /// would be a timer in name only.
    ///
    /// `TX_NEXT` advances even when the SDU is later discarded, per §5.2.1's
    /// NOTE 1 — the COUNT is associated at reception from upper layers, so
    /// reusing it for the next SDU would give two SDUs one COUNT.
    pub fn submit_sdu(&mut self, sdu: &[u8], now_ms: u64) -> u32 {
        let count = self.tx_next;
        // The SN is the COUNT's low bits, and `encode_header`'s masks are what
        // narrow it -- the modulo `sn_modulus` would express the same thing twice,
        // and a reader would then have to check the two agree.
        let mut pdu = Self::encode_header(self.config.sn_size, count);
        pdu.extend_from_slice(sdu);

        self.tx_pending.push(PendingSdu {
            count,
            pdu,
            armed_at_ms: now_ms,
        });
        self.tx_next = self.tx_next.wrapping_add(1);
        trace!(
            "PDCP TX: SDU at COUNT {count} (SN {}), {} pending",
            count % self.config.sn_size.sn_modulus(),
            self.tx_pending.len()
        );
        count
    }

    /// Take the PDUs ready for RLC, discarding any whose `discardTimer` expired.
    ///
    /// Returns them in COUNT order. Expiry is evaluated here rather than on a
    /// separate tick so a discard cannot be missed by a caller that only ever
    /// drains.
    pub fn take_transmittable(&mut self, now_ms: u64) -> Vec<Vec<u8>> {
        self.discard_expired(now_ms);
        let mut ready: Vec<PendingSdu> = std::mem::take(&mut self.tx_pending);
        ready.sort_by_key(|pending| pending.count);
        ready.into_iter().map(|pending| pending.pdu).collect()
    }

    /// Drop every pending SDU whose `discardTimer` has expired (§5.2.1).
    ///
    /// Returns the COUNTs discarded, so a caller can log or count them.
    pub fn discard_expired(&mut self, now_ms: u64) -> Vec<u32> {
        let Some(timer_ms) = self.config.discard_timer_ms else {
            return Vec::new();
        };
        let mut discarded = Vec::new();
        self.tx_pending.retain(|pending| {
            // `saturating_sub`, because a caller passing a `now_ms` earlier than
            // the arming instant must not wrap into a huge elapsed time and
            // discard everything.
            let elapsed = now_ms.saturating_sub(pending.armed_at_ms);
            if elapsed >= timer_ms {
                discarded.push(pending.count);
                false
            } else {
                true
            }
        });
        if !discarded.is_empty() {
            self.discarded_on_expiry += discarded.len() as u64;
            debug!("PDCP TX: discardTimer expired for COUNT(s) {discarded:?} after {timer_ms} ms");
        }
        discarded
    }

    /// Encode a data-PDU header (TS 38.323 §6.2.2.2 / §6.2.2.3).
    ///
    /// D/C is 1 for a data PDU; the reserved bits are zero. The SN is big-endian
    /// across the header octets, which is what puts its most significant bits in
    /// the first octet alongside D/C.
    fn encode_header(sn_size: PdcpSnSize, sn: u32) -> Vec<u8> {
        const DATA_PDU: u8 = 0x80;
        match sn_size {
            PdcpSnSize::Sn12 => {
                // Octet 1: D/C | R R R | SN[11:8];  octet 2: SN[7:0].
                vec![DATA_PDU | ((sn >> 8) & 0x0F) as u8, (sn & 0xFF) as u8]
            }
            PdcpSnSize::Sn18 => {
                // Octet 1: D/C | R R R R R | SN[17:16]; octets 2-3: SN[15:0].
                vec![
                    DATA_PDU | ((sn >> 16) & 0x03) as u8,
                    ((sn >> 8) & 0xFF) as u8,
                    (sn & 0xFF) as u8,
                ]
            }
        }
    }

    /// Read the SN out of a data-PDU header, or report why it cannot be read.
    fn decode_header(sn_size: PdcpSnSize, pdu: &[u8]) -> Result<u32, PdcpReceiveError> {
        if pdu.len() < sn_size.header_len() {
            return Err(PdcpReceiveError::Truncated);
        }
        if pdu[0] & 0x80 == 0 {
            return Err(PdcpReceiveError::ControlPdu);
        }
        Ok(match sn_size {
            PdcpSnSize::Sn12 => (u32::from(pdu[0] & 0x0F) << 8) | u32::from(pdu[1]),
            PdcpSnSize::Sn18 => {
                (u32::from(pdu[0] & 0x03) << 16) | (u32::from(pdu[1]) << 8) | u32::from(pdu[2])
            }
        })
    }

    // ========================================================================
    // Receive (TS 38.323 §5.2.2)
    // ========================================================================

    /// Handle a PDCP data PDU from lower layers (§5.2.2.1).
    ///
    /// Returns the SDUs now deliverable to upper layers, **in COUNT order** —
    /// possibly empty when the PDU filled no gap, and possibly several when it
    /// filled one.
    ///
    /// # Errors
    /// Returns [`PdcpReceiveError`] when the PDU is not deliverable at all: it is
    /// truncated, it is a control PDU, it is older than `RX_DELIV`, or it
    /// duplicates one already received. Each is a *discard* in the spec's terms,
    /// and each is reported rather than silently dropped so a caller can count it.
    pub fn receive_pdu(
        &mut self,
        pdu: &[u8],
        now_ms: u64,
    ) -> Result<Vec<Vec<u8>>, PdcpReceiveError> {
        let sn_size = self.config.sn_size;
        let rcvd_sn = Self::decode_header(sn_size, pdu)?;
        let rcvd_count = self.rcvd_count(rcvd_sn);

        if rcvd_count < self.rx_deliv {
            self.discarded_out_of_window += 1;
            return Err(PdcpReceiveError::BelowDeliveryWindow);
        }
        if self.rx_buffer.contains_key(&rcvd_count) {
            self.discarded_duplicates += 1;
            return Err(PdcpReceiveError::Duplicate);
        }

        let sdu = pdu[sn_size.header_len()..].to_vec();
        self.rx_buffer.insert(rcvd_count, sdu);

        if rcvd_count >= self.rx_next {
            self.rx_next = rcvd_count.wrapping_add(1);
        }

        let mut delivered = Vec::new();
        if rcvd_count == self.rx_deliv {
            // Deliver every consecutively-numbered SDU from RX_DELIV upward, then
            // advance RX_DELIV to the first COUNT still missing.
            delivered = self.drain_consecutive_from(self.rx_deliv);
        }

        // §5.2.2.1: stop t-Reordering when the gap it was started for has closed,
        // then start it again if a gap remains. Both in that order, because a
        // single PDU can close one gap and reveal another.
        if self.t_reordering_running() && self.rx_deliv >= self.rx_reord {
            self.stop_t_reordering();
        }
        if !self.t_reordering_running() && self.rx_deliv < self.rx_next {
            self.rx_reord = self.rx_next;
            self.start_t_reordering(now_ms);
        }

        Ok(delivered)
    }

    /// Drive `t-Reordering` (§5.2.2.2).
    ///
    /// Returns the SDUs flushed by an expiry, empty when the timer is not running
    /// or has not expired. Called from whatever the caller already ticks; the
    /// entity has no timer of its own to fire.
    pub fn poll_t_reordering(&mut self, now_ms: u64) -> Vec<Vec<u8>> {
        let Some(started) = self.t_reordering_started_ms else {
            return Vec::new();
        };
        let Some(timer_ms) = self.config.t_reordering_ms else {
            return Vec::new();
        };
        if now_ms.saturating_sub(started) < timer_ms {
            return Vec::new();
        }

        // Everything below RX_REORD is given up on and delivered, then everything
        // consecutively numbered from RX_REORD.
        let mut delivered: Vec<Vec<u8>> = Vec::new();
        let below: Vec<u32> = self
            .rx_buffer
            .range(..self.rx_reord)
            .map(|(count, _)| *count)
            .collect();
        for count in below {
            if let Some(sdu) = self.rx_buffer.remove(&count) {
                delivered.push(sdu);
            }
        }
        // RX_DELIV moves to the first COUNT at or after RX_REORD that is missing,
        // which `drain_consecutive_from` assigns -- it returns having set RX_DELIV
        // to the first COUNT it could not deliver, and that is exactly the spec's
        // "first PDCP SDU which has not been delivered ... with COUNT >= RX_REORD".
        delivered.extend(self.drain_consecutive_from(self.rx_reord));

        self.stop_t_reordering();
        if self.rx_deliv < self.rx_next {
            self.rx_reord = self.rx_next;
            self.start_t_reordering(now_ms);
        }

        if !delivered.is_empty() {
            debug!(
                "PDCP RX: t-Reordering expired, flushed {} SDU(s); RX_DELIV now {}",
                delivered.len(),
                self.rx_deliv
            );
        }
        delivered
    }

    /// Deliver the consecutive run starting at `from`, advancing `RX_DELIV` to the
    /// first COUNT that is missing.
    fn drain_consecutive_from(&mut self, from: u32) -> Vec<Vec<u8>> {
        let mut delivered = Vec::new();
        let mut count = from;
        while let Some(sdu) = self.rx_buffer.remove(&count) {
            delivered.push(sdu);
            count = count.wrapping_add(1);
        }
        self.rx_deliv = count;
        delivered
    }

    /// `RCVD_COUNT` from `RCVD_SN` (§5.2.2.1), which is where the HFN is inferred.
    ///
    /// The three-way comparison against `SN(RX_DELIV) +/- Window_Size` is the
    /// spec's own, and it is what makes an SN that has wrapped resolve to the
    /// *next* hyper-frame rather than to a COUNT far in the past.
    fn rcvd_count(&self, rcvd_sn: u32) -> u32 {
        let sn_bits = self.config.sn_size.bits();
        let window = self.config.sn_size.window_size();
        let sn_modulus = self.config.sn_size.sn_modulus();

        let sn_deliv = self.rx_deliv % sn_modulus;
        let hfn_deliv = self.rx_deliv >> sn_bits;

        let rcvd_hfn = if rcvd_sn + window < sn_deliv {
            // RCVD_SN < SN(RX_DELIV) - Window_Size, without underflowing.
            hfn_deliv.wrapping_add(1)
        } else if rcvd_sn >= sn_deliv.saturating_add(window) {
            hfn_deliv.wrapping_sub(1)
        } else {
            hfn_deliv
        };

        (rcvd_hfn << sn_bits) | rcvd_sn
    }

    fn start_t_reordering(&mut self, now_ms: u64) {
        if self.config.t_reordering_ms.is_none() {
            // Not configured: a gap waits indefinitely rather than being flushed
            // on a timer the operator did not ask for.
            trace!(
                "PDCP RX: gap at RX_DELIV {} with no t-Reordering configured",
                self.rx_deliv
            );
            return;
        }
        self.t_reordering_started_ms = Some(now_ms);
        trace!(
            "PDCP RX: t-Reordering started, RX_REORD {} (RX_DELIV {})",
            self.rx_reord,
            self.rx_deliv
        );
    }

    fn stop_t_reordering(&mut self) {
        self.t_reordering_started_ms = None;
    }

    /// Flush the reordering buffer and reset receive state, delivering everything
    /// held in COUNT order.
    ///
    /// For a bearer release: TS 38.323 §5.1.2 delivers stored SDUs upward rather
    /// than dropping them, because the data arrived and only the ordering
    /// guarantee is being abandoned.
    pub fn flush_receive(&mut self) -> Vec<Vec<u8>> {
        let delivered: Vec<Vec<u8>> = std::mem::take(&mut self.rx_buffer).into_values().collect();
        if !delivered.is_empty() {
            warn!(
                "PDCP RX: flushing {} buffered SDU(s) out of order on release",
                delivered.len()
            );
        }
        self.rx_deliv = self.rx_next;
        self.stop_t_reordering();
        delivered
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity() -> Pdcp {
        // Short timers, stated explicitly: these tests are about the MECHANISM at
        // its boundaries, and riding the production defaults would make every
        // boundary assertion re-break whenever a default is retuned.
        Pdcp::new(PdcpConfig {
            sn_size: PdcpSnSize::Sn12,
            discard_timer_ms: Some(100),
            t_reordering_ms: Some(50),
        })
    }

    /// A test that the PRODUCTION defaults are internally consistent, which the
    /// short-timer entity above deliberately cannot check.
    #[test]
    fn the_default_t_reordering_exceeds_the_rlc_am_recovery_cycle() {
        let config = PdcpConfig::default();
        let t_reordering = config.t_reordering_ms.expect("a default t-Reordering");
        // The 500 ms task tick drives both the RLC STATUS report and the
        // retransmission, so a recovery needs at least two of them. A t-Reordering
        // below that makes PDCP discard the PDU ARQ just recovered.
        assert!(
            t_reordering >= 1_000,
            "t-Reordering {t_reordering} ms is shorter than the RLC AM recovery cycle"
        );
        // And the discard timer must not bin an SDU the receiver is still waiting
        // for -- otherwise a retransmission has nothing left to retransmit.
        let discard = config.discard_timer_ms.expect("a default discardTimer");
        assert!(
            discard >= 500,
            "discardTimer {discard} ms is shorter than the tick that drains the queue"
        );
    }

    /// Build the PDU a transmitter would produce for `sn`, carrying `payload`.
    fn pdu(sn_size: PdcpSnSize, sn: u32, payload: &[u8]) -> Vec<u8> {
        let mut p = Pdcp::encode_header(sn_size, sn);
        p.extend_from_slice(payload);
        p
    }

    // ---- header and SN/COUNT (§5.2.1, §6.2.2.2/§6.2.2.3) ----

    #[test]
    fn a_twelve_bit_header_carries_the_data_bit_and_the_sn_big_endian() {
        // Octet 1: D/C=1, R R R = 0, SN[11:8];  octet 2: SN[7:0].
        assert_eq!(Pdcp::encode_header(PdcpSnSize::Sn12, 0), vec![0x80, 0x00]);
        assert_eq!(Pdcp::encode_header(PdcpSnSize::Sn12, 1), vec![0x80, 0x01]);
        assert_eq!(
            Pdcp::encode_header(PdcpSnSize::Sn12, 0x0FFF),
            vec![0x8F, 0xFF]
        );
        assert_eq!(
            Pdcp::encode_header(PdcpSnSize::Sn12, 0x0100),
            vec![0x81, 0x00]
        );
        // The reserved bits stay zero.
        for sn in [0u32, 5, 0x0ABC, 0x0FFF] {
            assert_eq!(
                Pdcp::encode_header(PdcpSnSize::Sn12, sn)[0] & 0x70,
                0,
                "reserved bits must be zero for SN {sn}"
            );
        }
    }

    #[test]
    fn an_eighteen_bit_header_is_three_octets_with_two_sn_bits_in_the_first() {
        assert_eq!(
            Pdcp::encode_header(PdcpSnSize::Sn18, 0x3FFFF),
            vec![0x83, 0xFF, 0xFF]
        );
        assert_eq!(
            Pdcp::encode_header(PdcpSnSize::Sn18, 0x10000),
            vec![0x81, 0x00, 0x00]
        );
        for sn in [0u32, 7, 0x1234, 0x3FFFF] {
            assert_eq!(
                Pdcp::encode_header(PdcpSnSize::Sn18, sn)[0] & 0x7C,
                0,
                "reserved bits must be zero for SN {sn}"
            );
        }
    }

    #[test]
    fn every_sn_round_trips_through_the_header() {
        for sn_size in [PdcpSnSize::Sn12, PdcpSnSize::Sn18] {
            // Every value for SN12; a spread for SN18, whose space is 262144.
            let candidates: Vec<u32> = match sn_size {
                PdcpSnSize::Sn12 => (0..sn_size.sn_modulus()).collect(),
                PdcpSnSize::Sn18 => (0..sn_size.sn_modulus()).step_by(97).collect(),
            };
            for sn in candidates {
                let encoded = Pdcp::encode_header(sn_size, sn);
                assert_eq!(encoded.len(), sn_size.header_len());
                assert_eq!(Pdcp::decode_header(sn_size, &encoded), Ok(sn), "SN {sn}");
            }
        }
    }

    #[test]
    fn a_control_pdu_and_a_truncated_pdu_are_reported_not_parsed() {
        // D/C = 0 is a control PDU.
        assert_eq!(
            Pdcp::decode_header(PdcpSnSize::Sn12, &[0x00, 0x01]),
            Err(PdcpReceiveError::ControlPdu)
        );
        // Shorter than its own header.
        assert_eq!(
            Pdcp::decode_header(PdcpSnSize::Sn12, &[0x80]),
            Err(PdcpReceiveError::Truncated)
        );
        assert_eq!(
            Pdcp::decode_header(PdcpSnSize::Sn18, &[0x80, 0x00]),
            Err(PdcpReceiveError::Truncated)
        );
    }

    #[test]
    fn transmit_assigns_a_monotonically_increasing_sn_and_count() {
        let mut pdcp = entity();
        for expected in 0..5u32 {
            assert_eq!(pdcp.tx_count_for_next_sdu(), expected);
            assert_eq!(pdcp.submit_sdu(b"payload", 0), expected);
        }
        let pdus = pdcp.take_transmittable(0);
        assert_eq!(pdus.len(), 5);
        for (expected_sn, pdu) in pdus.iter().enumerate() {
            assert_eq!(
                Pdcp::decode_header(PdcpSnSize::Sn12, pdu),
                Ok(expected_sn as u32)
            );
            assert_eq!(&pdu[2..], b"payload");
        }
    }

    #[test]
    fn the_count_keeps_counting_across_an_sn_wrap() {
        // The point of a 32-bit COUNT over a 12-bit SN: at SN wrap the COUNT must
        // continue rather than restart, or two different SDUs share one COUNT and
        // any COUNT-keyed security would reuse a keystream.
        let mut pdcp = entity();
        let modulus = PdcpSnSize::Sn12.sn_modulus();
        pdcp.tx_next = modulus - 1;

        let last_of_hfn0 = pdcp.submit_sdu(b"a", 0);
        let first_of_hfn1 = pdcp.submit_sdu(b"b", 0);
        assert_eq!(last_of_hfn0, modulus - 1);
        assert_eq!(first_of_hfn1, modulus, "COUNT continues past the SN wrap");

        let pdus = pdcp.take_transmittable(0);
        assert_eq!(
            Pdcp::decode_header(PdcpSnSize::Sn12, &pdus[0]),
            Ok(modulus - 1)
        );
        assert_eq!(
            Pdcp::decode_header(PdcpSnSize::Sn12, &pdus[1]),
            Ok(0),
            "the SN wraps even though the COUNT did not"
        );
        // And the wrap must happen in the ENCODER, not be papered over by this
        // decoder's own mask: a COUNT past the SN space must not leak its high bits
        // into the reserved field. Both ends here share the code, so a decode-side
        // mask hides the defect -- only the reserved bits expose it, and only they
        // are what a conformant peer would see.
        assert_eq!(
            pdus[1][0] & 0x70,
            0,
            "reserved bits must stay zero for a COUNT beyond the SN space, got {:#04x}",
            pdus[1][0]
        );
    }

    // ---- discardTimer (§5.2.1) ----

    #[test]
    fn a_stalled_sdu_is_discarded_when_its_discard_timer_expires() {
        let mut pdcp = entity(); // discardTimer 100 ms
        pdcp.submit_sdu(b"stalled", 1_000);
        pdcp.submit_sdu(b"fresh", 1_080);

        // At 1 099 ms neither has reached 100 ms of age.
        assert!(pdcp.discard_expired(1_099).is_empty());
        // At 1 100 ms the first has, exactly.
        assert_eq!(pdcp.discard_expired(1_100), vec![0]);
        assert_eq!(pdcp.discarded_on_expiry(), 1);

        // And the survivor is still transmittable, carrying its own payload.
        let pdus = pdcp.take_transmittable(1_100);
        assert_eq!(pdus.len(), 1);
        assert_eq!(&pdus[0][2..], b"fresh");
    }

    #[test]
    fn a_discarded_sdu_does_not_reach_rlc_at_all() {
        // The claim that matters: the discard happens BEFORE submission, so a
        // draining caller never sees the PDU.
        let mut pdcp = entity();
        pdcp.submit_sdu(b"stalled", 0);
        let pdus = pdcp.take_transmittable(500);
        assert!(
            pdus.is_empty(),
            "an SDU whose discardTimer expired must not be handed to RLC"
        );
        assert_eq!(pdcp.discarded_on_expiry(), 1);
    }

    #[test]
    fn a_discarded_count_is_not_reused_by_the_next_sdu() {
        // §5.2.1 NOTE 1: the COUNT is associated when the SDU arrives from upper
        // layers. Reusing a discarded COUNT would give two SDUs one COUNT.
        let mut pdcp = entity();
        assert_eq!(pdcp.submit_sdu(b"doomed", 0), 0);
        assert_eq!(pdcp.discard_expired(1_000), vec![0]);
        assert_eq!(pdcp.submit_sdu(b"next", 1_000), 1, "COUNT 0 is spent");
    }

    #[test]
    fn an_unconfigured_discard_timer_never_discards() {
        let mut pdcp = Pdcp::new(PdcpConfig {
            discard_timer_ms: None,
            ..PdcpConfig::default()
        });
        pdcp.submit_sdu(b"kept", 0);
        assert!(pdcp.discard_expired(u64::MAX / 2).is_empty());
        assert_eq!(pdcp.take_transmittable(u64::MAX / 2).len(), 1);
    }

    #[test]
    fn a_now_earlier_than_the_arming_instant_does_not_discard_everything() {
        // A saturating subtraction, not a wrapping one: an unsigned wrap here
        // would read as an enormous elapsed time and bin the whole queue.
        let mut pdcp = entity();
        pdcp.submit_sdu(b"kept", 10_000);
        assert!(pdcp.discard_expired(5_000).is_empty());
    }

    // ---- in-order delivery and reordering (§5.2.2.1) ----

    #[test]
    fn in_order_pdus_are_delivered_immediately_and_in_order() {
        let mut pdcp = entity();
        for sn in 0..4u32 {
            let delivered = pdcp
                .receive_pdu(&pdu(PdcpSnSize::Sn12, sn, &[sn as u8]), 0)
                .expect("deliverable");
            assert_eq!(delivered, vec![vec![sn as u8]]);
        }
        assert_eq!(pdcp.rx_deliv(), 4);
        assert!(!pdcp.t_reordering_running(), "no gap, no timer");
        assert_eq!(pdcp.buffered_count(), 0);
    }

    #[test]
    fn out_of_order_pdus_are_released_in_count_order_once_the_gap_is_filled() {
        let mut pdcp = entity();

        // SN 1 and 2 arrive first: held, nothing delivered, and t-Reordering runs.
        assert!(pdcp
            .receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"one"), 0)
            .expect("accepted")
            .is_empty());
        assert!(pdcp
            .receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0)
            .expect("accepted")
            .is_empty());
        assert!(pdcp.t_reordering_running(), "a gap must start t-Reordering");
        assert_eq!(pdcp.buffered_count(), 2);

        // SN 0 closes the gap: all three come out, in order.
        let delivered = pdcp
            .receive_pdu(&pdu(PdcpSnSize::Sn12, 0, b"zero"), 0)
            .expect("accepted");
        assert_eq!(
            delivered,
            vec![b"zero".to_vec(), b"one".to_vec(), b"two".to_vec()]
        );
        assert_eq!(pdcp.rx_deliv(), 3);
        assert!(!pdcp.t_reordering_running(), "the gap closed");
        assert_eq!(pdcp.buffered_count(), 0);
    }

    #[test]
    fn a_duplicate_and_an_already_delivered_pdu_are_discarded() {
        let mut pdcp = entity();
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 0, b"zero"), 0)
            .expect("accepted");

        // Already delivered, so below RX_DELIV.
        assert_eq!(
            pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 0, b"zero"), 0),
            Err(PdcpReceiveError::BelowDeliveryWindow)
        );
        assert_eq!(pdcp.discarded_out_of_window(), 1);

        // Buffered but not yet delivered, so a true duplicate.
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0)
            .expect("accepted");
        assert_eq!(
            pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0),
            Err(PdcpReceiveError::Duplicate)
        );
        assert_eq!(pdcp.discarded_duplicates(), 1);
    }

    #[test]
    fn a_replayed_payload_cannot_overwrite_a_buffered_one() {
        // A duplicate is DISCARDED, not stored: if the second copy replaced the
        // first, a peer could rewrite buffered data by resending an SN.
        let mut pdcp = entity();
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"original"), 0)
            .expect("accepted");
        let _ = pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"replaced"), 0);

        let delivered = pdcp
            .receive_pdu(&pdu(PdcpSnSize::Sn12, 0, b"zero"), 0)
            .expect("accepted");
        assert_eq!(delivered, vec![b"zero".to_vec(), b"original".to_vec()]);
    }

    // ---- t-Reordering (§5.2.2.2) ----

    #[test]
    fn t_reordering_expiry_flushes_stored_sdus_up_to_the_reordering_point() {
        let mut pdcp = entity(); // t-Reordering 50 ms

        // SN 1, 2 arrive at t=0 with SN 0 missing; SN 3 at t=10.
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"one"), 0)
            .expect("accepted");
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0)
            .expect("accepted");
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 3, b"three"), 10)
            .expect("accepted");
        // RX_REORD is set to RX_NEXT when t-Reordering STARTS and not touched
        // afterwards (§5.2.2.1 updates it only on a (re)start), so it is 2 -- the
        // RX_NEXT at the moment SN 1 revealed the gap -- and not 4.
        assert_eq!(pdcp.rx_reord(), 2);

        // Not yet expired.
        assert!(pdcp.poll_t_reordering(49).is_empty());

        // At 50 ms: SN 0 is given up on, and 1..3 come out in order.
        let flushed = pdcp.poll_t_reordering(50);
        assert_eq!(
            flushed,
            vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()]
        );
        assert_eq!(pdcp.rx_deliv(), 4);
        assert!(
            !pdcp.t_reordering_running(),
            "no gap remains, so no new timer"
        );
    }

    #[test]
    fn t_reordering_restarts_when_a_gap_remains_after_a_flush() {
        let mut pdcp = entity();
        // Missing 0; have 1. Then missing 2; have 3.
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"one"), 0)
            .expect("accepted");
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 3, b"three"), 0)
            .expect("accepted");

        // RX_REORD is 2: the RX_NEXT at the moment SN 1 revealed the first gap.
        assert_eq!(pdcp.rx_reord(), 2);

        // The expiry gives up on SN 0 and delivers everything below RX_REORD --
        // SN 1 only. SN 2 is still missing, so SN 3 stays buffered and a NEW timer
        // starts for the remaining gap. This is the behaviour that stops one
        // expiry from flushing the entire buffer regardless of later gaps.
        let flushed = pdcp.poll_t_reordering(50);
        assert_eq!(flushed, vec![b"one".to_vec()]);
        assert_eq!(pdcp.rx_deliv(), 2, "waiting for SN 2 now");
        assert_eq!(pdcp.buffered_count(), 1, "SN 3 is still held");
        assert!(
            pdcp.t_reordering_running(),
            "a gap remains, so the timer restarts"
        );
        assert_eq!(
            pdcp.rx_reord(),
            4,
            "RX_REORD advanced to RX_NEXT on the restart"
        );

        // The second expiry gives up on SN 2 and releases SN 3.
        let flushed = pdcp.poll_t_reordering(100);
        assert_eq!(flushed, vec![b"three".to_vec()]);
        assert_eq!(pdcp.rx_deliv(), 4);
        assert!(!pdcp.t_reordering_running(), "nothing left to wait for");
    }

    #[test]
    fn an_sdu_below_the_reordering_point_is_delivered_by_the_expiry() {
        // §5.2.2.2's first bullet: everything with COUNT < RX_REORD goes up, even
        // the SDUs that are not consecutive with RX_DELIV.
        let mut pdcp = entity();
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 5, b"five"), 0)
            .expect("accepted");
        assert_eq!(pdcp.rx_reord(), 6);
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0)
            .expect("accepted");

        let flushed = pdcp.poll_t_reordering(50);
        assert_eq!(flushed, vec![b"two".to_vec(), b"five".to_vec()]);
        assert_eq!(pdcp.buffered_count(), 0);
    }

    #[test]
    fn an_unconfigured_t_reordering_waits_indefinitely_rather_than_flushing() {
        let mut pdcp = Pdcp::new(PdcpConfig {
            t_reordering_ms: None,
            ..PdcpConfig::default()
        });
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"one"), 0)
            .expect("accepted");
        assert!(!pdcp.t_reordering_running());
        assert!(pdcp.poll_t_reordering(u64::MAX / 2).is_empty());
        assert_eq!(pdcp.buffered_count(), 1, "held, not flushed");
    }

    #[test]
    fn a_release_flush_delivers_what_is_held() {
        // §5.1.2: the data arrived; only the ordering guarantee is abandoned.
        let mut pdcp = entity();
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 2, b"two"), 0)
            .expect("accepted");
        pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 1, b"one"), 0)
            .expect("accepted");

        let flushed = pdcp.flush_receive();
        assert_eq!(flushed.len(), 2);
        assert!(!pdcp.t_reordering_running());
        assert_eq!(pdcp.buffered_count(), 0);
    }

    // ---- COUNT derivation across the SN wrap (§5.2.2.1) ----

    #[test]
    fn a_wrapped_sn_resolves_to_the_next_hyper_frame_not_a_past_count() {
        // The whole reason for the three-way Window_Size comparison. With
        // RX_DELIV just below the wrap, a small SN belongs to the NEXT hyper-frame.
        let modulus = PdcpSnSize::Sn12.sn_modulus();
        let mut pdcp = entity();
        pdcp.rx_deliv = modulus - 1;
        pdcp.rx_next = modulus - 1;

        assert_eq!(
            pdcp.rcvd_count(modulus - 1),
            modulus - 1,
            "the current frame"
        );
        assert_eq!(
            pdcp.rcvd_count(0),
            modulus,
            "SN 0 after the wrap is COUNT 4096, not COUNT 0"
        );
        assert_eq!(pdcp.rcvd_count(5), modulus + 5);
    }

    #[test]
    fn an_sn_far_ahead_resolves_to_the_previous_hyper_frame() {
        // The other branch: with RX_DELIV early in a hyper-frame, an SN more than
        // a window ahead is a straggler from the previous one.
        let modulus = PdcpSnSize::Sn12.sn_modulus();
        let mut pdcp = entity();
        pdcp.rx_deliv = modulus + 2; // HFN 1, SN 2
        pdcp.rx_next = modulus + 2;

        assert_eq!(pdcp.rcvd_count(3), modulus + 3, "same hyper-frame");
        // SN 4000 is far above SN(RX_DELIV) + Window_Size (2 + 2048).
        assert_eq!(
            pdcp.rcvd_count(4_000),
            4_000,
            "HFN 0, i.e. the previous frame"
        );
    }

    #[test]
    fn a_pdu_from_the_previous_hyper_frame_is_discarded_as_too_old() {
        // Following from the derivation above: such a COUNT is below RX_DELIV, and
        // delivering it would break in-order delivery.
        let modulus = PdcpSnSize::Sn12.sn_modulus();
        let mut pdcp = entity();
        pdcp.rx_deliv = modulus + 2;
        pdcp.rx_next = modulus + 2;

        assert_eq!(
            pdcp.receive_pdu(&pdu(PdcpSnSize::Sn12, 4_000, b"stale"), 0),
            Err(PdcpReceiveError::BelowDeliveryWindow)
        );
    }

    #[test]
    fn an_18_bit_entity_carries_its_own_header_length_end_to_end() {
        let mut tx = Pdcp::new(PdcpConfig {
            sn_size: PdcpSnSize::Sn18,
            ..PdcpConfig::default()
        });
        let mut rx = Pdcp::new(PdcpConfig {
            sn_size: PdcpSnSize::Sn18,
            ..PdcpConfig::default()
        });

        tx.submit_sdu(b"eighteen", 0);
        let pdus = tx.take_transmittable(0);
        assert_eq!(pdus[0].len(), 3 + b"eighteen".len());
        let delivered = rx.receive_pdu(&pdus[0], 0).expect("deliverable");
        assert_eq!(delivered, vec![b"eighteen".to_vec()]);
    }

    #[test]
    fn a_transmit_receive_pair_carries_payload_intact_and_in_order() {
        // The in-crate half of the end-to-end criterion: what one entity produces,
        // another accepts, in order, unchanged -- including when the network
        // reorders it.
        let mut tx = entity();
        let mut rx = entity();
        let payloads: Vec<Vec<u8>> = (0..6u8).map(|i| vec![i; 20]).collect();
        for payload in &payloads {
            tx.submit_sdu(payload, 0);
        }
        let mut pdus = tx.take_transmittable(0);
        // Deliver 2 and 3 swapped, which is what a reordering network does.
        pdus.swap(2, 3);

        let mut delivered = Vec::new();
        for pdu in &pdus {
            delivered.extend(rx.receive_pdu(pdu, 0).expect("deliverable"));
        }
        assert_eq!(delivered, payloads, "payload intact and back in order");
        assert_eq!(rx.buffered_count(), 0);
    }

    // --- The duplicate-suppression handover from RLC (issue #103) ---

    /// CRITERION 3 of issue #103: a replayed **complete** SDU is delivered twice by
    /// RLC — which has no SN on such a PDU to recognise a replay by — and PDCP
    /// discards the second copy on its own SN.
    ///
    /// The one test that spans both layers, and the reason #103 had to land after
    /// #33: removing the RLC SN before this entity existed would have reintroduced
    /// duplicate delivery with nothing to catch it.
    #[test]
    fn rlc_delivers_a_replayed_complete_sdu_twice_and_pdcp_passes_one() {
        use nextgsim_rlc::{RlcEntity, RlcMode, SnSize};

        let payload = vec![0x33u8; 24];

        // One PDCP PDU, carried whole by RLC.
        let mut pdcp_tx = Pdcp::new(PdcpConfig::default());
        pdcp_tx.submit_sdu(&payload, 0);
        let pdcp_pdu = pdcp_tx
            .take_transmittable(0)
            .pop()
            .expect("one PDU per SDU");

        let mut rlc_tx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rlc_tx.submit_sdu(pdcp_pdu.clone());
        let rlc_pdu = rlc_tx.build_pdu(1500).expect("it fits whole");

        // RLC receives the SAME PDU twice and hands BOTH copies up: a conformant
        // complete-SDU UMD PDU has no SN, so there is nothing to dedupe on.
        let mut rlc_rx = RlcEntity::new(RlcMode::UnacknowledgedMode, SnSize::Sn12);
        rlc_rx.receive_pdu(&rlc_pdu);
        rlc_rx.receive_pdu(&rlc_pdu);
        let mut from_rlc = Vec::new();
        while let Some(sdu) = rlc_rx.poll_reassembled() {
            from_rlc.push(sdu);
        }
        assert_eq!(
            from_rlc.len(),
            2,
            "RLC must deliver both copies -- it cannot tell them apart"
        );
        assert_eq!(from_rlc[0], pdcp_pdu);
        assert_eq!(from_rlc[1], pdcp_pdu);

        // PDCP passes the first and discards the second on its PDCP SN.
        let mut pdcp_rx = Pdcp::new(PdcpConfig::default());
        assert_eq!(
            pdcp_rx.receive_pdu(&from_rlc[0], 0),
            Ok(vec![payload.clone()]),
            "the first copy is delivered"
        );
        assert_eq!(
            pdcp_rx.receive_pdu(&from_rlc[1], 0),
            Err(PdcpReceiveError::BelowDeliveryWindow),
            "the second is discarded: its COUNT is below RX_DELIV"
        );
        assert_eq!(pdcp_rx.discarded_out_of_window(), 1);
    }
}
