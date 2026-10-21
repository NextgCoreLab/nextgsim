//! Uplink data held while the user plane is down (issue #267).
//!
//! From CM-IDLE, uplink data is what triggers the SERVICE REQUEST
//! (TS 24.501 §5.6.1.2), and that same data cannot be carried until the
//! procedure completes and the N3 tunnel is back (TS 23.502 §4.2.3.2). It used
//! to be handed to the radio straight away, where the gNB, which had no context
//! for the UE, dropped it. So the packet that triggered every UE-initiated
//! service request was lost. This buffer holds that packet, and whatever
//! follows it, until the NAS task reports the user plane ready.

use std::collections::VecDeque;

use nextgsim_common::OctetString;

/// The default number of packets held. Bounded so a UE whose service request
/// never completes cannot grow without limit. The oldest packet is dropped
/// first, because the newest is the one most worth delivering late.
pub const DEFAULT_UPLINK_BUFFER_PACKETS: usize = 64;

/// Uplink packets waiting for the user plane, in arrival order.
#[derive(Debug)]
pub struct UplinkBuffer {
    pending: VecDeque<(i32, OctetString)>,
    capacity: usize,
    dropped: u64,
}

impl Default for UplinkBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_UPLINK_BUFFER_PACKETS)
    }
}

impl UplinkBuffer {
    /// Creates a buffer holding at most `capacity` packets (at least one).
    pub fn new(capacity: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            capacity: capacity.max(1),
            dropped: 0,
        }
    }

    /// Offers one packet. When the user plane is `ready`, it returns what
    /// should be sent now, in order: anything still held, then this packet.
    /// Otherwise it holds the packet and returns nothing.
    pub fn offer(&mut self, ready: bool, psi: i32, data: OctetString) -> Vec<(i32, OctetString)> {
        self.hold(psi, data);
        if ready {
            self.drain()
        } else {
            Vec::new()
        }
    }

    /// Takes every held packet, oldest first, once the user plane is up.
    pub fn drain(&mut self) -> Vec<(i32, OctetString)> {
        self.pending.drain(..).collect()
    }

    /// Number of packets held.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// True when nothing is held.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Packets dropped because the buffer was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    fn hold(&mut self, psi: i32, data: OctetString) {
        if self.pending.len() == self.capacity {
            self.pending.pop_front();
            self.dropped += 1;
        }
        self.pending.push_back((psi, data));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pkt(b: u8) -> OctetString {
        OctetString::from(vec![b])
    }

    #[test]
    fn a_packet_offered_while_the_user_plane_is_down_is_held_then_released_first() {
        let mut buf = UplinkBuffer::default();
        assert!(buf.offer(false, 1, pkt(1)).is_empty(), "held, not sent");
        assert!(buf.offer(false, 1, pkt(2)).is_empty());
        assert_eq!(buf.len(), 2);

        let out = buf.drain();
        assert_eq!(out, vec![(1, pkt(1)), (1, pkt(2))]);
        assert!(buf.is_empty());
    }

    #[test]
    fn a_ready_offer_sends_what_is_held_before_the_new_packet() {
        let mut buf = UplinkBuffer::default();
        buf.offer(false, 1, pkt(1));
        assert_eq!(buf.offer(true, 1, pkt(2)), vec![(1, pkt(1)), (1, pkt(2))]);
        assert_eq!(buf.offer(true, 1, pkt(3)), vec![(1, pkt(3))]);
    }

    #[test]
    fn a_full_buffer_drops_the_oldest() {
        let mut buf = UplinkBuffer::new(2);
        buf.offer(false, 1, pkt(1));
        buf.offer(false, 1, pkt(2));
        buf.offer(false, 1, pkt(3));
        assert_eq!(buf.dropped(), 1);
        assert_eq!(buf.drain(), vec![(1, pkt(2)), (1, pkt(3))]);
    }
}
