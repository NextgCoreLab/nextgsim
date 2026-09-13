//! NGAP guard timers (TS 38.413 §8.3.3.4, §8.4.1.2, §8.7.1.3)
//!
//! NGAP procedures that wait on the AMF need a bound, or a peer that never answers
//! leaves the gNB holding state forever: a UE context stuck in `Releasing`, a handover
//! preparation that never completes, an NG Setup that is never retried. TS 38.413 names
//! the timers (TNGRELOCoverall, TNGRELOCprep) without fixing their values, which are
//! the operator's.
//!
//! # Why the deadline is passed in
//!
//! Every method takes `now` rather than reading the clock. The task loop supplies
//! `Instant::now()`; a test supplies a base instant and adds to it, so "the timer
//! expired" is an ordinary assertion with no sleeping, no `tokio::time::pause`, and no
//! dependence on how long the test host takes to schedule a task. That is what
//! criterion 7 of #42 asks for by "a controllable clock".

use std::time::{Duration, Instant};

/// A guard timer's identity and the action its expiry implies.
///
/// Equality is on the whole variant including its key, so restarting or cancelling a
/// timer for one UE cannot touch another's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardTimer {
    /// TNGRELOCoverall — a UE Context Release Request was sent and the AMF has not
    /// answered with a UE Context Release Command. On expiry the gNB releases the
    /// context locally rather than leaving it in `Releasing` forever
    /// (TS 38.413 §8.3.3.4).
    UeContextRelease {
        /// The UE whose release is outstanding.
        ue_id: i32,
    },
    /// TNGRELOCprep — handover preparation is in flight and neither a Handover Command
    /// nor a Handover Preparation Failure has arrived. On expiry the preparation is
    /// cancelled and the UE stays on the source cell (TS 38.413 §8.4.1.2).
    HandoverPreparation {
        /// The UE being handed over.
        ue_id: i32,
    },
    /// The NG Setup retry an NG Setup Failure's Time to Wait IE gates. The retry must
    /// not fire before the indicated time (TS 38.413 §8.7.1.3).
    NgSetupRetry {
        /// The AMF to retry towards.
        amf_id: i32,
    },
}

/// Pending guard timers, ordered only by when they are asked for.
///
/// A `Vec` and a linear scan: the population is one entry per in-flight procedure per
/// UE, and the scan runs once per poll rather than once per message.
#[derive(Debug, Default)]
pub struct GuardTimers {
    entries: Vec<(Instant, GuardTimer)>,
}

impl GuardTimers {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Start (or restart) `timer`, due `after` from `now`.
    ///
    /// Restarting REPLACES the pending entry rather than adding a second one. Two
    /// entries for one timer would both expire, and the second would act on a procedure
    /// the first already cleaned up — for `UeContextRelease` that means releasing a UE
    /// id that has since been reused.
    pub fn start(&mut self, now: Instant, after: Duration, timer: GuardTimer) {
        self.entries.retain(|(_, pending)| *pending != timer);
        self.entries.push((now + after, timer));
    }

    /// Cancel `timer` if it is pending. Returns whether it was.
    ///
    /// The return value matters at the call sites: a Handover Command for a UE with no
    /// pending TNGRELOCprep is a response to a preparation this gNB does not think it
    /// started, which is worth a log line rather than silence.
    pub fn cancel(&mut self, timer: GuardTimer) -> bool {
        let before = self.entries.len();
        self.entries.retain(|(_, pending)| *pending != timer);
        before != self.entries.len()
    }

    /// Remove and return every timer due at or before `now`.
    ///
    /// Draining rather than peeking: an expiry that stayed pending would fire again on
    /// the next poll, so a UE that could not be released would be "released" once per
    /// second for the life of the process.
    pub fn expired(&mut self, now: Instant) -> Vec<GuardTimer> {
        let mut fired = Vec::new();
        self.entries.retain(|(deadline, timer)| {
            if *deadline <= now {
                fired.push(*timer);
                false
            } else {
                true
            }
        });
        fired
    }

    /// Whether `timer` is pending.
    pub fn is_pending(&self, timer: GuardTimer) -> bool {
        self.entries.iter().any(|(_, pending)| *pending == timer)
    }

    /// How many timers are pending.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether nothing is pending.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timer_fires_only_once_its_deadline_has_passed() {
        let mut timers = GuardTimers::new();
        let base = Instant::now();
        let timer = GuardTimer::UeContextRelease { ue_id: 7 };
        timers.start(base, Duration::from_secs(5), timer);

        assert!(timers.is_pending(timer));
        assert!(
            timers.expired(base + Duration::from_secs(4)).is_empty(),
            "a timer must not fire early"
        );
        assert_eq!(timers.expired(base + Duration::from_secs(5)), vec![timer]);
        assert!(
            timers.expired(base + Duration::from_secs(60)).is_empty(),
            "an expiry is drained, or it fires again on every later poll"
        );
        assert!(!timers.is_pending(timer));
    }

    #[test]
    fn cancelling_removes_the_timer_and_reports_whether_it_was_pending() {
        let mut timers = GuardTimers::new();
        let base = Instant::now();
        let timer = GuardTimer::HandoverPreparation { ue_id: 3 };
        timers.start(base, Duration::from_secs(10), timer);

        assert!(timers.cancel(timer), "it was pending");
        assert!(!timers.cancel(timer), "and now it is not");
        assert!(timers.expired(base + Duration::from_secs(30)).is_empty());
    }

    /// Restarting must not leave two entries behind: the second expiry would act on a
    /// procedure the first already finished.
    #[test]
    fn restarting_replaces_rather_than_duplicates() {
        let mut timers = GuardTimers::new();
        let base = Instant::now();
        let timer = GuardTimer::UeContextRelease { ue_id: 1 };
        timers.start(base, Duration::from_secs(5), timer);
        timers.start(base + Duration::from_secs(1), Duration::from_secs(5), timer);
        assert_eq!(timers.len(), 1);

        // The deadline is the LATER one, so the original 5s point has not passed.
        assert!(timers.expired(base + Duration::from_secs(5)).is_empty());
        assert_eq!(
            timers.expired(base + Duration::from_secs(6)),
            vec![timer],
            "and it fires exactly once"
        );
    }

    /// The key is part of the identity: one UE's timer must not cancel another's.
    #[test]
    fn timers_are_keyed_per_ue_and_per_kind() {
        let mut timers = GuardTimers::new();
        let base = Instant::now();
        let release_1 = GuardTimer::UeContextRelease { ue_id: 1 };
        let release_2 = GuardTimer::UeContextRelease { ue_id: 2 };
        let prep_1 = GuardTimer::HandoverPreparation { ue_id: 1 };
        for t in [release_1, release_2, prep_1] {
            timers.start(base, Duration::from_secs(5), t);
        }
        assert_eq!(timers.len(), 3);

        assert!(timers.cancel(release_1));
        assert!(!timers.is_pending(release_1));
        assert!(
            timers.is_pending(release_2) && timers.is_pending(prep_1),
            "cancelling one UE's release must not touch the other UE or the other kind"
        );
    }

    #[test]
    fn every_due_timer_is_returned_from_one_poll() {
        let mut timers = GuardTimers::new();
        let base = Instant::now();
        timers.start(
            base,
            Duration::from_secs(1),
            GuardTimer::UeContextRelease { ue_id: 1 },
        );
        timers.start(
            base,
            Duration::from_secs(2),
            GuardTimer::HandoverPreparation { ue_id: 2 },
        );
        timers.start(
            base,
            Duration::from_secs(30),
            GuardTimer::NgSetupRetry { amf_id: 0 },
        );

        let fired = timers.expired(base + Duration::from_secs(3));
        assert_eq!(fired.len(), 2, "both due timers, in one poll");
        assert!(fired.contains(&GuardTimer::UeContextRelease { ue_id: 1 }));
        assert!(fired.contains(&GuardTimer::HandoverPreparation { ue_id: 2 }));
        assert!(timers.is_pending(GuardTimer::NgSetupRetry { amf_id: 0 }));
    }
}
