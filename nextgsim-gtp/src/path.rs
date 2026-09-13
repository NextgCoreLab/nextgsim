//! GTP-U path supervision (TS 29.281 §7.2.1, TS 23.007 §20.3.1)
//!
//! A GTP-U node checks that a path to a peer is alive by sending Echo Requests and
//! counting the ones that go unanswered. After N3-REQUESTS consecutive misses the path
//! is declared down, and the node stops pretending user data is reaching the far end.
//!
//! The state machine lives here, in the library, rather than inside the gNB's async
//! task, so that "declared down after N misses" is a plain unit test rather than a
//! test that has to drive a timer. The task owns the interval; this owns the decision.

use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;

/// Per-peer Echo supervision state.
#[derive(Debug, Default, Clone)]
struct PeerPath {
    /// Sequence numbers of Echo Requests sent and not yet answered.
    outstanding: HashSet<u16>,
    /// Consecutive supervision periods that ended with an unanswered request.
    consecutive_misses: u32,
    /// Whether the path is currently considered usable.
    down: bool,
    /// Last Recovery restart counter this peer advertised, once it has advertised one.
    peer_restart_counter: Option<u8>,
}

/// What an Echo Response told us beyond "the path is alive".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EchoOutcome {
    /// Nothing new: the peer answered and its restart counter is unchanged.
    Alive,
    /// The path was down and this response brought it back.
    Recovered,
    /// The peer's Recovery counter CHANGED, so it restarted and its contexts are gone
    /// (TS 23.007). Also reported for a path that was down.
    PeerRestarted {
        /// The counter this node had recorded, if any.
        previous: Option<u8>,
        /// The counter the peer is advertising now.
        current: u8,
    },
}

/// Echo-based supervision of the GTP-U paths to a set of peers.
///
/// Constructed with the two TS 23.007 §20.3.1 parameters. `max_misses` is
/// N3-REQUESTS: the number of consecutive unanswered supervision periods after which
/// the path is down.
#[derive(Debug, Clone)]
pub struct PathSupervisor {
    peers: HashMap<SocketAddr, PeerPath>,
    max_misses: u32,
    next_sequence: u16,
}

impl PathSupervisor {
    /// Create a supervisor that declares a path down after `max_misses` consecutive
    /// unanswered supervision periods.
    ///
    /// `max_misses` of 0 is clamped to 1: a threshold of zero would declare every path
    /// down before the first response could possibly arrive.
    pub fn new(max_misses: u32) -> Self {
        Self {
            peers: HashMap::new(),
            max_misses: max_misses.max(1),
            next_sequence: 0,
        }
    }

    /// Allocate the sequence number for the next Echo Request.
    pub fn next_sequence(&mut self) -> u16 {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1);
        seq
    }

    /// Record that an Echo Request with `seq` was sent to `peer`.
    pub fn note_echo_sent(&mut self, peer: SocketAddr, seq: u16) {
        self.peers.entry(peer).or_default().outstanding.insert(seq);
    }

    /// Record an Echo Response from `peer`, carrying `restart_counter` if the Recovery
    /// IE was present.
    ///
    /// Clears the outstanding set rather than only the matching sequence number: a
    /// response proves the path is alive regardless of which request it answers, and
    /// leaving older requests outstanding would count a live path as missing.
    pub fn note_echo_response(
        &mut self,
        peer: SocketAddr,
        restart_counter: Option<u8>,
    ) -> EchoOutcome {
        let path = self.peers.entry(peer).or_default();
        path.outstanding.clear();
        path.consecutive_misses = 0;
        let was_down = path.down;
        path.down = false;

        if let Some(current) = restart_counter {
            let previous = path.peer_restart_counter;
            path.peer_restart_counter = Some(current);
            if previous.is_some_and(|p| p != current) {
                return EchoOutcome::PeerRestarted { previous, current };
            }
        }
        if was_down {
            EchoOutcome::Recovered
        } else {
            EchoOutcome::Alive
        }
    }

    /// Close a supervision period for `peer`: anything still outstanding was missed.
    ///
    /// Returns `true` if this call is the one that declared the path down, so a caller
    /// can log the transition once instead of on every period afterwards.
    pub fn note_period_elapsed(&mut self, peer: SocketAddr) -> bool {
        let max_misses = self.max_misses;
        let path = self.peers.entry(peer).or_default();
        if path.outstanding.is_empty() {
            return false;
        }
        path.outstanding.clear();
        path.consecutive_misses = path.consecutive_misses.saturating_add(1);
        if path.consecutive_misses >= max_misses && !path.down {
            path.down = true;
            return true;
        }
        false
    }

    /// Whether the path to `peer` is currently usable. An unknown peer is treated as
    /// alive: nothing has failed yet.
    pub fn is_alive(&self, peer: &SocketAddr) -> bool {
        self.peers.get(peer).is_none_or(|p| !p.down)
    }

    /// Consecutive missed supervision periods for `peer`.
    pub fn consecutive_misses(&self, peer: &SocketAddr) -> u32 {
        self.peers.get(peer).map_or(0, |p| p.consecutive_misses)
    }

    /// Forget a peer entirely (its last session was released).
    pub fn forget(&mut self, peer: &SocketAddr) {
        self.peers.remove(peer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn peer(last: u8) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, last)), 2152)
    }

    /// The N3-REQUESTS threshold: down on the Nth miss, not before.
    #[test]
    fn a_path_goes_down_only_after_max_misses_consecutive_periods() {
        let mut sup = PathSupervisor::new(3);
        let upf = peer(1);

        for expected_misses in 1..=2 {
            let seq = sup.next_sequence();
            sup.note_echo_sent(upf, seq);
            assert!(
                !sup.note_period_elapsed(upf),
                "must not declare the path down before the threshold"
            );
            assert_eq!(sup.consecutive_misses(&upf), expected_misses);
            assert!(sup.is_alive(&upf));
        }

        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert!(
            sup.note_period_elapsed(upf),
            "the third consecutive miss is the transition"
        );
        assert!(!sup.is_alive(&upf), "the path must now be down");
        // Reported once, not on every later period.
        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert!(!sup.note_period_elapsed(upf));
        assert!(!sup.is_alive(&upf));
    }

    /// A responding peer stays alive however long supervision runs, and one answer
    /// resets the miss counter.
    #[test]
    fn a_responding_peer_stays_alive_and_resets_the_miss_count() {
        let mut sup = PathSupervisor::new(2);
        let upf = peer(2);

        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert!(!sup.note_period_elapsed(upf));
        assert_eq!(sup.consecutive_misses(&upf), 1);

        // One response clears the miss, so the next miss starts from 1 again and the
        // path never reaches the threshold.
        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert_eq!(sup.note_echo_response(upf, Some(7)), EchoOutcome::Alive);
        assert_eq!(sup.consecutive_misses(&upf), 0);

        for _ in 0..10 {
            let seq = sup.next_sequence();
            sup.note_echo_sent(upf, seq);
            sup.note_period_elapsed(upf);
            let seq = sup.next_sequence();
            sup.note_echo_sent(upf, seq);
            sup.note_echo_response(upf, Some(7));
        }
        assert!(
            sup.is_alive(&upf),
            "an answering peer must never be declared down"
        );
    }

    /// Two peers are supervised independently: one dying must not take the other down.
    #[test]
    fn one_dead_peer_does_not_affect_another() {
        let mut sup = PathSupervisor::new(1);
        let dead = peer(3);
        let live = peer(4);

        let seq = sup.next_sequence();
        sup.note_echo_sent(dead, seq);
        let seq = sup.next_sequence();
        sup.note_echo_sent(live, seq);
        sup.note_echo_response(live, Some(1));

        assert!(sup.note_period_elapsed(dead));
        assert!(!sup.note_period_elapsed(live));
        assert!(!sup.is_alive(&dead));
        assert!(sup.is_alive(&live));
    }

    /// A changed Recovery counter is a peer restart (TS 23.007); an unchanged one is
    /// not, and a FIRST sighting is not either — there is nothing to compare it with.
    #[test]
    fn a_changed_recovery_counter_reports_a_peer_restart() {
        let mut sup = PathSupervisor::new(3);
        let upf = peer(5);

        assert_eq!(
            sup.note_echo_response(upf, Some(4)),
            EchoOutcome::Alive,
            "the first counter seen cannot be a restart: nothing to compare"
        );
        assert_eq!(sup.note_echo_response(upf, Some(4)), EchoOutcome::Alive);
        assert_eq!(
            sup.note_echo_response(upf, Some(5)),
            EchoOutcome::PeerRestarted {
                previous: Some(4),
                current: 5
            }
        );
        // And the new value becomes the baseline.
        assert_eq!(sup.note_echo_response(upf, Some(5)), EchoOutcome::Alive);
    }

    /// A response on a down path reports the recovery, so the transition back is
    /// visible rather than silent.
    #[test]
    fn a_response_on_a_down_path_reports_recovery() {
        let mut sup = PathSupervisor::new(1);
        let upf = peer(6);
        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert!(sup.note_period_elapsed(upf));
        assert!(!sup.is_alive(&upf));

        assert_eq!(sup.note_echo_response(upf, None), EchoOutcome::Recovered);
        assert!(sup.is_alive(&upf));
    }

    /// A period with nothing outstanding is not a miss: with supervision disabled, or
    /// before the first request, an idle period must not count against the path.
    #[test]
    fn a_period_with_nothing_outstanding_is_not_a_miss() {
        let mut sup = PathSupervisor::new(1);
        let upf = peer(7);
        assert!(!sup.note_period_elapsed(upf));
        assert_eq!(sup.consecutive_misses(&upf), 0);
        assert!(sup.is_alive(&upf));
    }

    /// A threshold of 0 would declare every path down before any response could
    /// arrive, so it is clamped.
    #[test]
    fn a_zero_threshold_is_clamped_to_one() {
        let mut sup = PathSupervisor::new(0);
        let upf = peer(8);
        assert!(sup.is_alive(&upf), "nothing has failed yet");
        let seq = sup.next_sequence();
        sup.note_echo_sent(upf, seq);
        assert!(sup.note_period_elapsed(upf));
    }
}
