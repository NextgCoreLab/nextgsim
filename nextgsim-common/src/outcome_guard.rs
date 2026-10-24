//! A spawned task's outcome, guaranteed to reach the task loop that spawned it.
//!
//! The gNB SCTP task and the UE RLS task run DNS lookups and SCTP connects in
//! spawned tasks, and record "an attempt is in flight" until the outcome comes
//! back (#311). A spawned task that ended without reporting, for example by
//! panicking, left that record set for good: the target never reconnected, or
//! re-resolution stopped (#342 review). [`OutcomeGuard`] sends a fallback
//! outcome when it is dropped without having sent the real one, which happens
//! on a panic as the task's future unwinds, so the loop's state heals itself.

use tokio::sync::mpsc::UnboundedSender;

/// Sends exactly one outcome on `tx`: the one given to [`Self::send`], or
/// `fallback` if the guard is dropped first.
pub struct OutcomeGuard<T> {
    tx: UnboundedSender<T>,
    fallback: Option<T>,
}

impl<T> OutcomeGuard<T> {
    /// A guard that reports `fallback` unless [`Self::send`] runs.
    pub fn new(tx: UnboundedSender<T>, fallback: T) -> Self {
        Self {
            tx,
            fallback: Some(fallback),
        }
    }

    /// Report `outcome` and disarm the fallback. A receiver that has gone away
    /// is not an error: the loop that would act on it has ended.
    pub fn send(mut self, outcome: T) {
        self.fallback = None;
        let _ = self.tx.send(outcome);
    }
}

impl<T> Drop for OutcomeGuard<T> {
    fn drop(&mut self) {
        if let Some(fallback) = self.fallback.take() {
            let _ = self.tx.send(fallback);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_real_outcome_is_sent_once_and_disarms_the_fallback() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        OutcomeGuard::new(tx, "fallback").send("done");
        assert_eq!(rx.try_recv().ok(), Some("done"));
        assert!(rx.try_recv().is_err(), "no fallback after a real outcome");
    }

    /// #342 review: a task that panics before reporting still reports.
    #[tokio::test]
    async fn a_panicking_task_still_reports_the_fallback() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let guard = OutcomeGuard::new(tx, "fallback");
        let handle = tokio::spawn(async move {
            let _guard = guard;
            panic!("the attempt panicked");
        });
        assert!(handle.await.is_err(), "the task panicked");
        assert_eq!(rx.recv().await, Some("fallback"));
    }
}
