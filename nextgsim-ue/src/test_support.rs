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

use std::io::Write;
use std::sync::{Arc, Mutex};

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
