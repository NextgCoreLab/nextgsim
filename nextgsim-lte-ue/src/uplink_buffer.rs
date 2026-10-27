//! Uplink IP packets held while the user plane is down (#435).
//!
//! From RRC_IDLE, uplink data is what starts the service request (TS 24.301
//! §5.6.1.1 b), and that same data has no DRB to go on until the procedure
//! completes and the user plane radio bearers are set up (§5.6.1.4). The UE
//! used to drop it, so the packet that triggered every return from idle was
//! lost. This buffer holds it, and what follows it, until then: the LTE
//! counterpart of the 5G UE's `nextgsim_ue::tun::UplinkBuffer` (#267).
//!
//! Bounded both in packets and in bytes, so a UE whose service request never
//! completes cannot grow without limit. When full, the oldest packet is
//! dropped first, because the newest is the one most worth delivering late.

use std::collections::VecDeque;

/// The default number of packets held (the 5G UE's bound).
pub const DEFAULT_UPLINK_HOLD_PACKETS: usize = 64;
/// The default number of bytes held.
pub const DEFAULT_UPLINK_HOLD_BYTES: usize = 64 * 1024;

/// Uplink packets waiting for the user plane, in arrival order.
#[derive(Debug)]
pub struct UplinkBuffer {
    pending: VecDeque<Vec<u8>>,
    bytes: usize,
    max_packets: usize,
    max_bytes: usize,
    dropped: u64,
}

impl Default for UplinkBuffer {
    fn default() -> Self {
        Self::new(DEFAULT_UPLINK_HOLD_PACKETS, DEFAULT_UPLINK_HOLD_BYTES)
    }
}

impl UplinkBuffer {
    /// A buffer of at most `max_packets` packets (at least one) and
    /// `max_bytes` bytes.
    pub fn new(max_packets: usize, max_bytes: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            bytes: 0,
            max_packets: max_packets.max(1),
            max_bytes,
            dropped: 0,
        }
    }

    /// Holds `packet`, dropping the oldest held packets until it fits. A
    /// packet larger than the whole byte bound is dropped itself.
    pub fn hold(&mut self, packet: &[u8]) {
        if packet.len() > self.max_bytes {
            self.dropped += 1;
            return;
        }
        while self.pending.len() >= self.max_packets || self.bytes + packet.len() > self.max_bytes {
            let Some(old) = self.pending.pop_front() else {
                break;
            };
            self.bytes -= old.len();
            self.dropped += 1;
        }
        self.bytes += packet.len();
        self.pending.push_back(packet.to_vec());
    }

    /// Takes every held packet, oldest first.
    pub fn drain(&mut self) -> Vec<Vec<u8>> {
        self.bytes = 0;
        self.pending.drain(..).collect()
    }

    /// Number of packets held.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Bytes held.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// True when nothing is held.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Packets dropped because the buffer was full.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packets_come_out_in_arrival_order() {
        let mut b = UplinkBuffer::default();
        b.hold(&[1]);
        b.hold(&[2, 2]);
        assert_eq!((b.len(), b.bytes()), (2, 3));
        assert_eq!(b.drain(), vec![vec![1], vec![2, 2]]);
        assert!(b.is_empty());
        assert_eq!(b.bytes(), 0);
    }

    #[test]
    fn the_packet_bound_drops_the_oldest() {
        let mut b = UplinkBuffer::new(2, 1000);
        for p in 1..=3 {
            b.hold(&[p]);
        }
        assert_eq!(b.dropped(), 1);
        assert_eq!(b.drain(), vec![vec![2], vec![3]]);
    }

    #[test]
    fn the_byte_bound_drops_the_oldest_until_the_new_packet_fits() {
        let mut b = UplinkBuffer::new(10, 10);
        b.hold(&[1; 4]);
        b.hold(&[2; 4]);
        b.hold(&[3; 4]);
        assert_eq!(b.dropped(), 1);
        assert_eq!(b.bytes(), 8);
        b.hold(&[4; 11]);
        assert_eq!(b.dropped(), 2, "larger than the bound: dropped itself");
        assert_eq!(b.drain(), vec![vec![2; 4], vec![3; 4]]);
    }
}
