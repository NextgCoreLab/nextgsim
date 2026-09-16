//! Test-only helpers shared across this crate's unit tests.
//!
//! The `tracing` capture exists because several honesty invariants here are
//! about what the runtime *emits* — "a default start must not claim TS 23.586
//! compliance" (issue #55), "a default build must not advertise an active Rel-18
//! NR Sidelink capability" (issue #54) — and asserting those over the source text
//! would be asserting about the wrong thing. The workspace had no such harness
//! before #55; it lives here so there is exactly one copy rather than one per
//! module that needs it.
//!
//! Two hazards are encoded here rather than left to each caller to rediscover:
//!
//! 1. **`with_default` installs a *thread-local* dispatcher.** A `tokio::spawn`ed
//!    task can be polled on a worker thread where the capture is not installed,
//!    and the buffer comes back empty — which satisfies an "absence" assertion
//!    for entirely the wrong reason. So the task is driven directly on a
//!    current-thread runtime.
//! 2. **An absence assertion needs a positive control.** These helpers return the
//!    captured text and assert nothing; callers are expected to pin something
//!    that *must* be present before concluding anything from what is missing.
//! 3. **`tracing`'s callsite-interest cache is PROCESS-GLOBAL, while
//!    `with_default` is per-THREAD.** Registering or dropping a dispatcher
//!    rebuilds that cache for every callsite (`tracing_core::callsite::
//!    register_dispatch`), so two captures running concurrently rebuild each
//!    other's — and a callsite re-evaluated at the instant no dispatcher is
//!    registered is cached as `Interest::never()` and stays off until the next
//!    registration. One capture then loses a line while another capture's guard
//!    was being dropped. [`capture_logs`] therefore takes [`CAPTURE_LOCK`] so only
//!    one capture exists at a time.
//!
//!    This is not hypothetical: CI's sidelink job failed on issue #136's merge
//!    commit with the ranging task's *stopped* line captured and its *started* line
//!    missing — two `info!` calls in the same function, on the same thread, under
//!    the same subscriber. The extra tests in that commit changed the timing; the
//!    hazard was already there.

use std::io::Write;
use std::sync::{Arc, Mutex};

/// Serialises log captures **and every task run that shares their callsites**.
///
/// Declared beside the helper that takes it rather than inside a `mod tests`,
/// because what it guards is process-global state (`tracing`'s callsite-interest
/// cache) and a second lock elsewhere would guard nothing. See hazard 3.
static CAPTURE_LOCK: Mutex<()> = Mutex::new(());

/// Take [`CAPTURE_LOCK`] for the caller's scope.
///
/// **Any test that drives a task whose `info!` lines a capture test asserts on must
/// hold this**, not just the captures: the interest cache those lines live in is
/// process-global, so a concurrent run of the same task can leave a callsite cached
/// as `never` while a capture is reading it. Serialising the captures alone was
/// tried first and still flaked 2 runs in 12 (issue #136).
///
/// Poisoning is tolerated for the same reason [`capture_logs`] tolerates it.
pub(crate) fn hold_capture_lock() -> std::sync::MutexGuard<'static, ()> {
    CAPTURE_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

use nextgsim_common::config::UeConfig;
use tokio::sync::mpsc;

use crate::tasks::{
    AppMessage, NasMessage, RlsMessage, RrcMessage, Task, TaskHandle, TaskMessage, UeTaskBase,
};

/// A [`UeTaskBase`] over the default `UeConfig`, with every peer channel's
/// receiver dropped.
///
/// Suitable for any task under test that does not need a peer to answer: sends
/// into these handles fail silently, which is what an isolated task should see.
pub(crate) fn task_base() -> UeTaskBase {
    let (app_tx, _app_rx) = mpsc::channel::<TaskMessage<AppMessage>>(1);
    let (nas_tx, _nas_rx) = mpsc::channel::<TaskMessage<NasMessage>>(1);
    let (rrc_tx, _rrc_rx) = mpsc::channel::<TaskMessage<RrcMessage>>(1);
    let (rls_tx, _rls_rx) = mpsc::channel::<TaskMessage<RlsMessage>>(1);
    UeTaskBase {
        config: Arc::new(UeConfig::default()),
        app_tx: TaskHandle::new(app_tx),
        nas_tx: TaskHandle::new(nas_tx),
        rrc_tx: TaskHandle::new(rrc_tx),
        rls_tx: TaskHandle::new(rls_tx),
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        sixg: None,
        rel18: None,
    }
}

/// A [`UeTaskBase`] over `config`, with every peer channel's receiver dropped.
///
/// Same shape as [`task_base`], for a task whose behaviour is driven by its
/// configuration rather than by its peers.
///
/// Gated on `sidelink` because its only callers today are the SL-PRS ranging tests
/// (issue #136), which are; widen the gate when something else needs it, rather
/// than carrying it as dead code in a default build.
#[cfg(feature = "sidelink")]
pub(crate) fn task_base_with_config(config: UeConfig) -> UeTaskBase {
    UeTaskBase {
        config: Arc::new(config),
        ..task_base()
    }
}

/// A `MakeWriter` that appends every formatted log record to a shared buffer.
#[derive(Clone, Default)]
pub(crate) struct CapturedLog(Arc<Mutex<Vec<u8>>>);

impl CapturedLog {
    /// Everything written so far, as text.
    pub(crate) fn text(&self) -> String {
        String::from_utf8_lossy(&self.0.lock().expect("log buffer not poisoned")).into_owned()
    }
}

impl Write for CapturedLog {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .expect("log buffer not poisoned")
            .extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tracing_subscriber::fmt::MakeWriter<'_> for CapturedLog {
    type Writer = Self;

    fn make_writer(&self) -> Self::Writer {
        self.clone()
    }
}

/// Runs `body` with a capturing subscriber installed and returns everything it
/// logged.
///
/// `body` is a plain closure, so anything async inside it must be driven on a
/// runtime created *inside* the closure — see the thread-local hazard in the
/// module docs. [`capture_task_logs`] does that correctly; prefer it.
pub(crate) fn capture_logs<F: FnOnce()>(body: F) -> String {
    let captured = CapturedLog::default();
    let subscriber = tracing_subscriber::fmt()
        .with_writer(captured.clone())
        .with_max_level(tracing::Level::TRACE)
        .with_ansi(false)
        .finish();

    // Held for the whole capture: see hazard 3. Poisoning is tolerated rather than
    // propagated -- one panicking capture test must not turn every other one into a
    // failure whose message is about a mutex.
    let _guard = hold_capture_lock();
    tracing::subscriber::with_default(subscriber, body);

    captured.text()
}

/// Drives `task` from start to exit under a capturing subscriber and returns
/// everything it logged.
///
/// The task ends because the sender is dropped immediately: `rx.recv()` yields
/// `None` and the run loop breaks, so the task starts and stops without needing
/// a shutdown message. That covers exactly the startup and shutdown lines an
/// operator sees, which is what the honesty invariants are about.
pub(crate) fn capture_task_logs<T: Task>(mut task: T) -> String {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime builds");

    capture_logs(|| {
        runtime.block_on(async {
            let (tx, rx) = mpsc::channel::<TaskMessage<T::Message>>(1);
            drop(tx);
            task.run(rx).await;
        });
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every capture must see its OWN events and all of them, even when several
    /// captures are attempted at once. Serialised by [`CAPTURE_LOCK`]; this drives
    /// the concurrency the lock exists for.
    ///
    /// It cannot fail deterministically without the lock -- the hazard is a race on
    /// process-global state, and the CI failure it stands for was not reproducible on
    /// demand. What it does do is exercise the path under contention, which nothing
    /// else here did.
    #[test]
    fn concurrent_captures_each_see_their_own_events_in_full() {
        const FIRST: &str = "capture probe one";
        const SECOND: &str = "capture probe two";

        let (a, b) = std::thread::scope(|scope| {
            let first = scope.spawn(|| {
                capture_logs(|| {
                    tracing::info!("{}", FIRST);
                    tracing::info!("{} again", FIRST);
                })
            });
            let second = scope.spawn(|| {
                capture_logs(|| {
                    tracing::info!("{}", SECOND);
                    tracing::info!("{} again", SECOND);
                })
            });
            (
                first.join().expect("first capture thread"),
                second.join().expect("second capture thread"),
            )
        });

        assert_eq!(
            a.matches(FIRST).count(),
            2,
            "the first capture must hold both of its own events; captured: {a:?}"
        );
        assert!(
            !a.contains(SECOND),
            "and none of the other thread's; captured: {a:?}"
        );
        assert_eq!(
            b.matches(SECOND).count(),
            2,
            "the second capture must hold both of its own events; captured: {b:?}"
        );
        assert!(
            !b.contains(FIRST),
            "and none of the other thread's; captured: {b:?}"
        );
    }
}
