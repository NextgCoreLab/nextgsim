//! Simulated radio frame clock (issue #99)
//!
//! TS 38.304 §7.1 derives a UE's paging occasion from its identity and the DRX
//! cycle **in radio frames**, so paging cannot be scheduled at all without an SFN
//! that both the gNB and the UE agree on. This simulator maintained none: the MIB
//! carried a constant 0 and its own doc comment said why — *"a counter here would
//! be a number that looks like frame timing while corresponding to nothing."*
//!
//! ## Synchronisation model: wall clock on both sides
//!
//! The SFN is derived from the **UNIX epoch**, so every process on a host computes
//! the same value with no synchronisation message at all:
//!
//! ```text
//! SFN = (milliseconds since the UNIX epoch / 10) mod 1024
//! ```
//!
//! Two alternatives were rejected, and the reasons matter more than the choice:
//!
//! - **Carry the full 10-bit SFN in the MIB.** Exact and simple, but a
//!   non-conformant use of the field: TS 38.331 gives the MIB only the 6 most
//!   significant bits, with the 4 LSBs on the PBCH payload. Putting 10 bits there
//!   would make a decoder that trusts the field wrong about a real network.
//! - **Model the PBCH's 4 LSBs.** Correct, and much larger than paging: it means
//!   modelling the broadcast channel rather than just the message on it.
//!
//! The MIB therefore keeps carrying the conformant 6 MSBs, and those become a
//! **cross-check**: a UE compares the broadcast MSBs against its own clock, so
//! the synchronisation assumption is verified at runtime rather than assumed. See
//! [`sfn_msb6`] and [`msb6_matches`].
//!
//! ## What this model does and does not give
//!
//! - **Does**: a monotonic, wrap-correct SFN that two processes agree on without
//!   messaging, which is all TS 38.304 §7.1 needs.
//! - **Does not**: any relationship to a real radio frame boundary, and no
//!   agreement between processes on *different* hosts, whose clocks differ by
//!   however much NTP leaves them. A distributed run needs a real timing
//!   distribution model; this one is honest about being single-host.
//! - **Does not**: sub-frame or slot timing. A paging occasion is identified here
//!   to frame granularity, which is the granularity the paging frame has.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Duration of one radio frame (TS 38.211 §4.3.1).
pub const RADIO_FRAME_MS: u64 = 10;

/// The SFN wraps at 1024 (TS 38.211 §4.3.1: a 10-bit counter).
pub const SFN_CYCLE: u32 = 1024;

/// Number of SFN values the MIB's 6 most significant bits cannot distinguish.
///
/// The MIB carries `SFN >> 4`, so a UE reading only the MIB knows the frame
/// number to within 16 frames — 160 ms. That is why the MIB is a cross-check here
/// and not the clock.
pub const SFN_MSB6_GRANULARITY: u16 = 16;

/// The current system frame number, derived from the wall clock.
///
/// Never panics: a clock before the UNIX epoch yields frame 0 rather than an
/// error, because a paging decision must not fail on a clock reading.
pub fn current_sfn() -> u16 {
    sfn_at(SystemTime::now())
}

/// The system frame number at a given instant.
pub fn sfn_at(instant: SystemTime) -> u16 {
    let millis = instant
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    ((millis / u128::from(RADIO_FRAME_MS)) % u128::from(SFN_CYCLE)) as u16
}

/// The 6 most significant bits of an SFN, as the MIB carries them
/// (TS 38.331 `MIB.systemFrameNumber`).
pub fn sfn_msb6(sfn: u16) -> u8 {
    ((sfn % SFN_CYCLE as u16) >> 4) as u8
}

/// Whether a broadcast MIB's 6-bit SFN is consistent with a locally derived one.
///
/// `tolerance_frames` absorbs the transport delay between the gNB encoding the
/// MIB and the UE decoding it: the two clocks are the same clock, but the PDU
/// takes a nonzero time to arrive, and a comparison that ignored that would flag
/// every MIB that crossed a 16-frame boundary in flight.
///
/// Compared as a **circular** distance, because SFN 1023 and SFN 0 are adjacent.
pub fn msb6_matches(broadcast_msb6: u8, local_sfn: u16, tolerance_frames: u16) -> bool {
    let local_msb6 = sfn_msb6(local_sfn);
    if broadcast_msb6 == local_msb6 {
        return true;
    }
    // Reconstruct the coarse frame each MSB6 value stands for and compare those,
    // so a tolerance smaller than the 16-frame granularity still admits a PDU
    // that crossed a boundary.
    let broadcast_frame = u16::from(broadcast_msb6) * SFN_MSB6_GRANULARITY;
    let local_frame = u16::from(local_msb6) * SFN_MSB6_GRANULARITY;
    sfn_distance(broadcast_frame, local_frame) <= tolerance_frames + SFN_MSB6_GRANULARITY
}

/// The shortest distance between two SFNs, accounting for the wrap at 1024.
///
/// `sfn_distance(1023, 1) == 2`, not 1022 — a paging occasion two frames after a
/// wrap is two frames away, and treating it as 1022 would reject it.
pub fn sfn_distance(a: u16, b: u16) -> u16 {
    let cycle = SFN_CYCLE as u16;
    let a = a % cycle;
    let b = b % cycle;
    let forward = (a + cycle - b) % cycle;
    forward.min(cycle - forward)
}

/// How long to wait from now until the given SFN comes round.
///
/// Returns [`Duration::ZERO`] when the target *is* the current frame, so a caller
/// that is already at the occasion transmits immediately rather than waiting a
/// full cycle.
pub fn time_until_sfn(target_sfn: u16) -> Duration {
    duration_until_sfn(SystemTime::now(), target_sfn)
}

/// How long from `from` until `target_sfn`, for tests that need a fixed instant.
pub fn duration_until_sfn(from: SystemTime, target_sfn: u16) -> Duration {
    let current = sfn_at(from);
    let target = target_sfn % SFN_CYCLE as u16;
    if current == target {
        return Duration::ZERO;
    }
    let frames_ahead = (u32::from(target) + SFN_CYCLE - u32::from(current)) % SFN_CYCLE;
    Duration::from_millis(u64::from(frames_ahead) * RADIO_FRAME_MS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sfn_advances_one_per_radio_frame_and_wraps_at_1024() {
        let base = UNIX_EPOCH;
        assert_eq!(sfn_at(base), 0);
        assert_eq!(sfn_at(base + Duration::from_millis(10)), 1);
        assert_eq!(
            sfn_at(base + Duration::from_millis(19)),
            1,
            "within frame 1"
        );
        assert_eq!(sfn_at(base + Duration::from_millis(20)), 2);
        // 1024 frames is 10.24 s, after which the counter is back at 0.
        assert_eq!(sfn_at(base + Duration::from_millis(10_240)), 0);
        assert_eq!(sfn_at(base + Duration::from_millis(10_250)), 1);
    }

    #[test]
    fn a_clock_before_the_epoch_yields_frame_zero_rather_than_failing() {
        // A paging decision must not fail on a clock reading.
        let before = UNIX_EPOCH - Duration::from_secs(60);
        assert_eq!(sfn_at(before), 0);
    }

    #[test]
    fn the_mib_carries_the_six_most_significant_bits() {
        assert_eq!(sfn_msb6(0), 0);
        assert_eq!(sfn_msb6(15), 0, "frames 0-15 share one MSB6 value");
        assert_eq!(sfn_msb6(16), 1);
        assert_eq!(sfn_msb6(1023), 63);
        // And it fits the 6-bit field.
        for sfn in 0..1024u16 {
            assert!(sfn_msb6(sfn) <= 63, "sfn {sfn} overflowed the MIB field");
        }
    }

    #[test]
    fn the_distance_between_sfns_is_circular() {
        assert_eq!(sfn_distance(0, 0), 0);
        assert_eq!(sfn_distance(5, 3), 2);
        assert_eq!(sfn_distance(3, 5), 2);
        // The wrap: 1023 and 1 are two frames apart, not 1022.
        assert_eq!(sfn_distance(1023, 1), 2);
        assert_eq!(sfn_distance(1, 1023), 2);
        // Antipodal.
        assert_eq!(sfn_distance(0, 512), 512);
    }

    #[test]
    fn the_wait_until_a_future_frame_is_ten_milliseconds_per_frame() {
        let base = UNIX_EPOCH; // SFN 0
        assert_eq!(duration_until_sfn(base, 0), Duration::ZERO);
        assert_eq!(duration_until_sfn(base, 1), Duration::from_millis(10));
        assert_eq!(duration_until_sfn(base, 64), Duration::from_millis(640));
        // A target already passed waits for the next cycle rather than going
        // backwards, which a signed subtraction would do.
        let at_frame_100 = base + Duration::from_millis(1_000);
        assert_eq!(sfn_at(at_frame_100), 100);
        assert_eq!(
            duration_until_sfn(at_frame_100, 50),
            Duration::from_millis((1024 - 50) as u64 * 10)
        );
    }

    #[test]
    fn the_mib_cross_check_tolerates_transport_delay_but_not_a_wrong_clock() {
        // Same coarse frame: consistent.
        assert!(msb6_matches(sfn_msb6(100), 100, 2));
        // A PDU that crossed a 16-frame boundary in flight: still consistent,
        // because the MSB6 granularity is itself 16 frames.
        assert!(msb6_matches(sfn_msb6(95), 96, 2));
        // A clock half a cycle out is NOT consistent -- which is the whole point
        // of having the check.
        assert!(!msb6_matches(sfn_msb6(100), 612, 2));
        // Nor is one a few coarse steps out.
        assert!(!msb6_matches(sfn_msb6(100), 200, 2));
    }

    #[test]
    fn the_cross_check_holds_across_the_sfn_wrap() {
        // MSB6 goes 63 -> 0 at the wrap; the check must not read that as a
        // 63-step jump.
        assert!(msb6_matches(sfn_msb6(1020), 2, 4));
    }

    #[test]
    fn the_live_clock_agrees_with_the_instant_based_derivation() {
        // The property that makes two processes agree: `current_sfn` is only
        // `sfn_at(now)`, so there is no second time base to drift.
        let now = SystemTime::now();
        let live = current_sfn();
        let derived = sfn_at(now);
        assert!(
            sfn_distance(live, derived) <= 1,
            "live {live} and derived {derived} must agree to within a frame"
        );
    }
}
