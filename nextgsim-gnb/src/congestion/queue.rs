//! The capacity-C downlink queue (nextgsim#465): one FIFO for the whole cell,
//! drained by a token bucket at C bits per second, drop-tail at a byte limit.
//!
//! **A simulator construct.** There is no PHY or MAC scheduler in this gNB; the
//! queue is what makes C a bottleneck at all, so that a best-effort UE's
//! downlink ramp delays a protected UE's packets in the same cell, which is the
//! congestion the demo's loop reacts to.
//!
//! What is charged against the bucket is the RLC PDU's octets, the unit #455's
//! cell-load counter uses, so "served" means the same thing in both.

use std::collections::VecDeque;
use std::net::SocketAddr;
use std::time::Instant;

use bytes::Bytes;

/// The smallest byte limit accepted: two 1500-octet grants.
pub const MIN_LIMIT_BYTES: usize = 3000;

/// How long the bucket may idle and still send at once: 5 ms of C, and never
/// less than one maximal RLC PDU plus header, so any PDU can eventually go.
const BURST_SECS: f64 = 0.005;
const MIN_BURST_OCTETS: f64 = 4000.0;

/// One queued transmission: where it goes, its encoded RLS message, and the RLC
/// PDU octets it is charged.
#[derive(Debug, Clone)]
struct Entry {
    dest: SocketAddr,
    encoded: Bytes,
    octets: usize,
}

/// The cell's downlink FIFO.
#[derive(Debug)]
pub struct DlQueue {
    /// C in octets per second.
    rate_octets_per_s: f64,
    burst_octets: f64,
    tokens: f64,
    last_refill: Instant,
    limit_bytes: usize,
    fifo: VecDeque<Entry>,
    depth_octets: usize,
    drops: u64,
    served_octets: u64,
}

impl DlQueue {
    /// A queue served at `capacity_mbps`, holding at most `limit_bytes` RLC
    /// octets, with a full bucket at `now`.
    pub fn new(capacity_mbps: f64, limit_bytes: usize, now: Instant) -> Self {
        let rate = capacity_mbps * 1_000_000.0 / 8.0;
        let burst = (rate * BURST_SECS).max(MIN_BURST_OCTETS);
        Self {
            rate_octets_per_s: rate,
            burst_octets: burst,
            tokens: burst,
            last_refill: now,
            limit_bytes: limit_bytes.max(MIN_LIMIT_BYTES),
            fifo: VecDeque::new(),
            depth_octets: 0,
            drops: 0,
            served_octets: 0,
        }
    }

    /// Queue one transmission charged `octets`. `false` (dropped and counted)
    /// when it would take the queue past its limit.
    pub fn enqueue(&mut self, dest: SocketAddr, encoded: Bytes, octets: usize) -> bool {
        if self.depth_octets + octets > self.limit_bytes {
            self.drops += 1;
            return false;
        }
        self.depth_octets += octets;
        self.fifo.push_back(Entry {
            dest,
            encoded,
            octets,
        });
        true
    }

    /// Refill the bucket for the time since the last call and pop, in order,
    /// every transmission the tokens cover.
    pub fn drain(&mut self, now: Instant) -> Vec<(SocketAddr, Bytes)> {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        self.last_refill = now;
        self.tokens = (self.tokens + dt * self.rate_octets_per_s).min(self.burst_octets);
        let mut out = Vec::new();
        while let Some(front) = self.fifo.front() {
            let need = front.octets as f64;
            if need > self.tokens {
                break;
            }
            self.tokens -= need;
            let e = self.fifo.pop_front().expect("front exists");
            self.depth_octets -= e.octets;
            self.served_octets += e.octets as u64;
            out.push((e.dest, e.encoded));
        }
        out
    }

    /// Served octets since the last call (the window's measurement).
    pub fn take_served(&mut self) -> u64 {
        std::mem::take(&mut self.served_octets)
    }

    /// Octets queued now.
    pub fn depth_octets(&self) -> usize {
        self.depth_octets
    }

    /// Tail drops since the start.
    pub fn drops(&self) -> u64 {
        self.drops
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn addr(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    /// Offered at twice C for one second, the queue serves C: 5 Mbps is
    /// 625 000 octets a second (plus the initial burst).
    #[test]
    fn it_serves_at_capacity_when_offered_twice_capacity() {
        let t0 = Instant::now();
        let mut q = DlQueue::new(5.0, 10_000_000, t0);
        let mut served = 0usize;
        // 1250 PDUs of 1000 octets = 1.25 MB offered over 1 s, i.e. 10 Mbps.
        for ms in 0..1000u64 {
            let now = t0 + Duration::from_millis(ms);
            q.enqueue(addr(1), Bytes::from_static(b"x"), 1000);
            if ms % 4 == 0 {
                q.enqueue(addr(1), Bytes::from_static(b"x"), 1000);
            }
            served += q.drain(now).len() * 1000;
        }
        let expected = 625_000.0 + 4000.0;
        assert!(
            (served as f64 - expected).abs() <= 2000.0,
            "served {served} octets, expected about {expected}"
        );
        assert_eq!(q.take_served() as usize, served);
        assert!(q.depth_octets() > 500_000, "the excess is queued");
    }

    /// One FIFO for the cell: a protected UE's packet queued behind a flood
    /// waits for the flood ahead of it.
    #[test]
    fn a_packet_behind_a_flood_waits_for_it() {
        let t0 = Instant::now();
        let mut q = DlQueue::new(5.0, 10_000_000, t0);
        for _ in 0..100 {
            q.enqueue(addr(1), Bytes::from_static(b"flood"), 1250);
        }
        q.enqueue(addr(2), Bytes::from_static(b"ping"), 100);
        let mut ping_at = None;
        for ms in 0..1000u64 {
            let now = t0 + Duration::from_millis(ms);
            if q.drain(now).iter().any(|(d, _)| *d == addr(2)) {
                ping_at = Some(ms);
                break;
            }
        }
        // 125 000 octets ahead of it at 625 000 octets/s: about 200 ms.
        let at = ping_at.expect("the ping is served");
        assert!((190..=210).contains(&at), "ping served at {at} ms");
    }

    /// Past the limit the tail is dropped and counted; what was queued is kept.
    #[test]
    fn the_tail_drops_past_the_limit() {
        let t0 = Instant::now();
        let mut q = DlQueue::new(5.0, 5000, t0);
        assert!(q.enqueue(addr(1), Bytes::new(), 3000));
        assert!(q.enqueue(addr(1), Bytes::new(), 2000));
        assert!(!q.enqueue(addr(1), Bytes::new(), 1));
        assert_eq!((q.depth_octets(), q.drops()), (5000, 1));
    }
}
