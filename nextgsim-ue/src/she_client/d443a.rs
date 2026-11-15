//! Demo D443A (nextgsim#471): the UE's EEC client and `she-offload`.
//! **Non-normative.**
//!
//! When the demo is active, `she_enabled` is set and the UE config has an
//! `eec:` block, the SHE client task installs one EEC client per process (the
//! first UE instance). It does nothing until asked:
//!
//! - `nr-cli <ue> she-offload job [count=<n>] [interval-ms=<ms>]`: run `n`
//!   jobs (default 1), one after the other. Each asks the EES for a selection
//!   (TS 24.558 EEC registration with `easSelReqInd`) through the PDU session
//!   and runs the job on the first `discoveredEas`, or on-device on a 404
//!   ([`nextgsim_she::d443a::eec`]). Each job logs one `JOB job=<n> ...` line.
//! - `nr-cli <ue> she-offload status`: jobs run, on time, missed, on-device,
//!   and the last job's line.
//! - `nr-cli <ue> demo d443a-edge-budget set mode=static`: audited; the mode
//!   is read at each job and logged with it. Both modes behave the same in the
//!   UE: the ranking the mode selects is eesd's.
//!
//! With the demo off (no block, a kill switch off, or a build without the
//! feature) none of this exists: `she-offload` is an unknown command as before.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextgsim_common::config::EecClientConfig;
use nextgsim_demo::demo_log;
use nextgsim_she::d443a::{eec, ees_client, KEY};

/// The most jobs one `she-offload job` may ask for.
pub const MAX_JOBS: u32 = 1000;

#[derive(Debug, Default, Clone)]
struct Stats {
    jobs: u64,
    on_time: u64,
    missed: u64,
    on_device: u64,
    last: Option<String>,
}

#[derive(Debug)]
struct Client {
    config: EecClientConfig,
    sbi: tokio::sync::OnceCell<Arc<nextgsim_sbi_client::SbiClient>>,
    state: Mutex<State>,
}

#[derive(Debug, Default)]
struct State {
    next_job: u32,
    running: bool,
    stats: Stats,
}

static CLIENT: Mutex<Option<Arc<Client>>> = Mutex::new(None);

/// Serialises the tests that touch [`CLIENT`]. Declared beside it (one lock
/// per process-global).
/// A tokio mutex, so the async test may hold it across its awaits.
#[cfg(test)]
pub(crate) static TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// Install the client if demo D443A is active and `config` is present. A UE
/// without the demo is unchanged.
pub fn start(config: Option<&EecClientConfig>) {
    let _ = install(nextgsim_demo::active_mode(KEY).is_some(), config);
}

/// [`start`] with the demo state given: whether a client was installed.
fn install(active: bool, config: Option<&EecClientConfig>) -> bool {
    if !active {
        return false;
    }
    let Some(config) = config.cloned() else {
        demo_log!(
            warn,
            KEY,
            "demo enabled but the UE config has no eec block: no EEC client runs"
        );
        return false;
    };
    if let Err(e) = eec::validate(&config) {
        demo_log!(error, KEY, "eec block refused: {e}; no EEC client runs");
        return false;
    }
    let Ok(mut slot) = CLIENT.lock() else {
        return false;
    };
    if slot.is_some() {
        return false; // another UE instance of this process runs the client
    }
    demo_log!(
        info,
        KEY,
        "EEC {} -> EES {} via {} (AC {} naming {:?}, reqComp {}, deadline {} ms)",
        config.eec_id,
        config.ees,
        config.device.as_deref().unwrap_or("the routing table"),
        config.ac_id,
        config.eass,
        config.req_comp,
        config.deadline_ms
    );
    *slot = Some(Arc::new(Client {
        config,
        sbi: tokio::sync::OnceCell::new(),
        state: Mutex::new(State::default()),
    }));
    true
}

/// A parsed `she-offload` command.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Command {
    Jobs { count: u32, interval_ms: u64 },
    Status,
}

fn parse(args: &[&str]) -> Result<Command, String> {
    match args.first().map(|w| w.to_ascii_lowercase()).as_deref() {
        Some("status") if args.len() == 1 => Ok(Command::Status),
        Some("job") => {
            let (mut count, mut interval_ms) = (1u32, 0u64);
            for a in &args[1..] {
                match a.split_once('=') {
                    Some(("count", v)) => {
                        count = v
                            .parse()
                            .ok()
                            .filter(|n| (1..=MAX_JOBS).contains(n))
                            .ok_or_else(|| format!("count={v}: 1..={MAX_JOBS}"))?;
                    }
                    Some(("interval-ms", v)) => {
                        interval_ms = v
                            .parse()
                            .ok()
                            .filter(|n| *n <= 60_000)
                            .ok_or_else(|| format!("interval-ms={v}: 0..=60000"))?;
                    }
                    _ => return Err(format!("unknown argument {a:?}")),
                }
            }
            Ok(Command::Jobs { count, interval_ms })
        }
        _ => {
            Err("usage: she-offload job [count=<n>] [interval-ms=<ms>] | she-offload status".into())
        }
    }
}

/// Answer `she-offload ...` when the demo is active; `None` otherwise (and
/// for any other command), so the UE parses it as before.
pub fn route_cli(command: &str) -> Option<String> {
    route_cli_with(nextgsim_demo::active_mode(KEY).is_some(), command)
}

fn route_cli_with(active: bool, command: &str) -> Option<String> {
    let mut words = command.split_whitespace();
    if !words.next()?.eq_ignore_ascii_case("she-offload") || !active {
        return None;
    }
    let p = nextgsim_demo::prefix(KEY);
    let args: Vec<&str> = words.collect();
    let cmd = match parse(&args) {
        Ok(c) => c,
        Err(e) => return Some(format!("{p} {e}")),
    };
    let Some(client) = CLIENT.lock().ok().and_then(|c| c.clone()) else {
        return Some(format!(
            "{p} no EEC client in this UE (needs demo {KEY} enabled, she_enabled and an eec block)"
        ));
    };
    Some(match cmd {
        Command::Status => {
            let s = client.state.lock().ok()?;
            let st = &s.stats;
            format!(
                "{p} eec={} ees={} device={} mode={} running={} jobs={} on_time={} missed={} \
                 on_device={} last={}",
                client.config.eec_id,
                client.config.ees,
                client.config.device.as_deref().unwrap_or("-"),
                nextgsim_demo::active_mode(KEY).as_deref().unwrap_or("-"),
                s.running,
                st.jobs,
                st.on_time,
                st.missed,
                st.on_device,
                st.last.as_deref().unwrap_or("-")
            )
        }
        Command::Jobs { count, interval_ms } => {
            let first = {
                let Ok(mut s) = client.state.lock() else {
                    return None;
                };
                if s.running {
                    return Some(format!("{p} jobs are running; try again when they finish"));
                }
                s.running = true;
                // Job numbers wrap (u32, as in the job frame) rather than
                // overflow.
                let first = s.next_job.wrapping_add(1);
                s.next_job = s.next_job.wrapping_add(count);
                first
            };
            tokio::spawn(run_jobs(client, first, count, interval_ms));
            format!("{p} {count} job(s) started from job={first}; see she-offload status")
        }
    })
}

async fn run_jobs(client: Arc<Client>, first: u32, count: u32, interval_ms: u64) {
    let cfg = &client.config;
    let sbi = client
        .sbi
        .get_or_try_init(|| async {
            ees_client(
                cfg.oauth2.as_ref(),
                cfg.device.as_deref(),
                Duration::from_secs(3),
            )
            .await
            .map(Arc::new)
        })
        .await;
    for i in 0..count {
        let job = first.wrapping_add(i);
        let mode = nextgsim_demo::active_mode(KEY).unwrap_or_else(|| "-".into());
        let outcome = match &sbi {
            Ok(sbi) => eec::offload(sbi, cfg, job).await,
            Err(e) => eec::Outcome::OnDevice {
                reason: format!("eec-client ({e})"),
            },
        };
        let line = outcome.line(job, cfg.deadline_ms, &mode);
        demo_log!(info, KEY, "{line}");
        if let Ok(mut s) = client.state.lock() {
            let st = &mut s.stats;
            st.jobs += 1;
            match &outcome {
                eec::Outcome::OnDevice { .. } => st.on_device += 1,
                o if o.missed() => st.missed += 1,
                _ => st.on_time += 1,
            }
            st.last = Some(line);
        }
        if interval_ms > 0 && i + 1 < count {
            tokio::time::sleep(Duration::from_millis(interval_ms)).await;
        }
    }
    if let Ok(mut s) = client.state.lock() {
        s.running = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(ees: &str) -> EecClientConfig {
        serde_yaml::from_str(&format!(
            "ees: \"{ees}\"\neec_id: eec-471\ndevice: lo\neass: [eas-local]\njob_timeout_ms: 200\n"
        ))
        .unwrap()
    }

    fn clear() {
        *CLIENT
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    }

    fn reset() -> tokio::sync::MutexGuard<'static, ()> {
        let g = TEST_LOCK.blocking_lock();
        clear();
        g
    }

    async fn reset_async() -> tokio::sync::MutexGuard<'static, ()> {
        let g = TEST_LOCK.lock().await;
        clear();
        g
    }

    #[test]
    fn she_offload_parses_its_words_and_bounds_471() {
        assert_eq!(
            parse(&["job"]),
            Ok(Command::Jobs {
                count: 1,
                interval_ms: 0
            })
        );
        assert_eq!(
            parse(&["JOB", "count=10", "interval-ms=500"]),
            Ok(Command::Jobs {
                count: 10,
                interval_ms: 500
            })
        );
        assert_eq!(parse(&["status"]), Ok(Command::Status));
        for bad in [
            &["job", "count=0"][..],
            &["job", "count=1001"],
            &["job", "interval-ms=x"],
            &["job", "bogus=1"],
            &["status", "x"],
            &["run"],
            &[],
        ] {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }

    /// Demo off: nothing is installed and `she-offload` is not this module's,
    /// so the UE answers it as the unknown command it was. Other commands are
    /// never this module's.
    #[test]
    fn with_the_demo_off_she_offload_does_not_exist_471() {
        let _g = reset();
        assert!(!install(false, Some(&cfg("http://127.0.0.1:9"))));
        assert!(CLIENT.lock().unwrap().is_none());
        assert_eq!(route_cli_with(false, "she-offload job"), None);
        assert_eq!(route_cli_with(false, "she-offload status"), None);
        assert_eq!(route_cli_with(true, "status"), None);
        assert_eq!(route_cli_with(true, "deregister normal"), None);
    }

    /// m1: the production entry points read the process's demo state. This
    /// test process installs no demo runtime, so the demo is off: `start`
    /// installs nothing and `route_cli` leaves `she-offload` to the UE parser,
    /// even with a usable `eec:` block.
    #[test]
    fn the_real_gates_read_the_process_demo_state_471() {
        let _g = reset();
        assert!(nextgsim_demo::installed().is_none(), "no runtime in tests");
        start(Some(&cfg("http://127.0.0.1:9")));
        assert!(CLIENT.lock().unwrap().is_none());
        assert_eq!(route_cli("she-offload status"), None);
        assert_eq!(route_cli("she-offload job count=2"), None);
    }

    /// Demo on without an `eec:` block (or with one that cannot run): no
    /// client, and `she-offload` says why.
    #[test]
    fn with_the_demo_on_but_no_usable_eec_block_no_client_runs_471() {
        let _g = reset();
        assert!(!install(true, None));
        let mut bad = cfg("http://127.0.0.1:9");
        bad.eass.clear();
        assert!(!install(true, Some(&bad)));
        let out = route_cli_with(true, "she-offload status").unwrap();
        assert!(
            out.starts_with("[demo:d443a-edge-budget] no EEC client"),
            "{out}"
        );
        let out = route_cli_with(true, "she-offload frob").unwrap();
        assert!(out.contains("usage: she-offload"), "{out}");
    }

    /// Demo on with a client: `she-offload job count=2` runs two jobs in the
    /// background (here against an unreachable EES: both on-device), and
    /// `status` reports them; one client per process.
    #[tokio::test]
    async fn she_offload_runs_jobs_and_reports_them_471() {
        let _g = reset_async().await;
        assert!(install(true, Some(&cfg("http://127.0.0.1:9"))));
        assert!(
            !install(true, Some(&cfg("http://127.0.0.1:9"))),
            "one per process"
        );
        let out = route_cli_with(true, "she-offload job count=2").unwrap();
        assert!(out.contains("2 job(s) started from job=1"), "{out}");
        let busy = route_cli_with(true, "she-offload job").unwrap();
        assert!(busy.contains("jobs are running"), "{busy}");
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        let status = loop {
            let s = route_cli_with(true, "she-offload status").unwrap();
            if s.contains("running=false") {
                break s;
            }
            assert!(tokio::time::Instant::now() < deadline, "{s}");
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert!(
            status.contains("jobs=2 on_time=0 missed=0 on_device=2")
                && status.contains("last=JOB job=2 on-device fallback reason=ees-unreachable"),
            "{status}"
        );
        let out = route_cli_with(true, "she-offload job").unwrap();
        assert!(out.contains("from job=3"), "{out}");
        // The job counter wraps instead of overflowing.
        let client = CLIENT.lock().unwrap().clone().unwrap();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while client.state.lock().unwrap().running {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        client.state.lock().unwrap().next_job = u32::MAX - 1;
        let out = route_cli_with(true, "she-offload job count=3").unwrap();
        assert!(
            out.contains("3 job(s) started from job=4294967295"),
            "{out}"
        );
        assert_eq!(client.state.lock().unwrap().next_job, 1);
        while client.state.lock().unwrap().running {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let st = route_cli_with(true, "she-offload status").unwrap();
        assert!(st.contains("last=JOB job=1 "), "{st}");
        *CLIENT.lock().unwrap() = None;
    }
}
