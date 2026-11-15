//! `nr-she-eas`: an EAS sidecar of demo D443A (nextgsim#471). Non-normative.
//! See `nextgsim_she::d443a::eas` for what it does and its config.
//!
//! ```text
//! nr-she-eas -c eas.yaml                               # register, reflect STAMP, serve jobs
//! nr-she-eas -c eas.yaml --load 10                     # the load act: on SIGUSR1, avlComp -> 10
//! nr-she-eas -c eas.yaml --load 10 --load-after 30     # ... or 30 s after the first registration
//! nr-she-eas -c eas.yaml --load 10 --no-kpi-patch      # revert arm: load applied, the EES not told
//! ```
//!
//! The sidecar is demo-only: it refuses to start unless the demo is enabled,
//! and an unknown key or mode fails startup (`DemoConfigError`). It opens two
//! UDP listeners, the STAMP Session-Reflector and the job listener, both on
//! the configured `address` only (loopback by default; the `dn` address in
//! the demo), and no published port. It listens for no HTTP.

use std::process::ExitCode;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use nextgsim_demo::{demo_log, DemoRegistry, DemoRuntime, DemoSpec};
use nextgsim_she::d443a::eas::{Check, EasConfig, Registrar};
use nextgsim_she::d443a::{ees_client, job, stamp, KEY, MODES};
use tokio::net::UdpSocket;

fn registry() -> DemoRegistry {
    DemoRegistry::new(&[DemoSpec {
        key: KEY,
        modes: MODES,
        requires: &[],
    }])
}

#[derive(Debug, PartialEq, Eq)]
struct Args {
    config: String,
    load: Option<u32>,
    load_after_s: Option<u64>,
    no_kpi_patch: bool,
}

const USAGE: &str =
    "usage: nr-she-eas -c <eas.yaml> [--load <avlComp> [--load-after <s>] [--no-kpi-patch]]";

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut a = Args {
        config: String::new(),
        load: None,
        load_after_s: None,
        no_kpi_patch: false,
    };
    let mut config = None;
    let mut it = args.iter().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().cloned().ok_or_else(|| USAGE.to_string());
        match arg.as_str() {
            "-c" | "--config" => config = Some(value()?),
            "--load" => {
                let v = value()?;
                a.load = Some(
                    v.parse::<u32>()
                        .ok()
                        .filter(|n| *n > 0)
                        .ok_or_else(|| format!("--load {v}: a positive avlComp"))?,
                );
            }
            "--load-after" => {
                let v = value()?;
                a.load_after_s = Some(
                    v.parse()
                        .map_err(|_| format!("--load-after {v}: seconds"))?,
                );
            }
            "--no-kpi-patch" => a.no_kpi_patch = true,
            _ => return Err(USAGE.to_string()),
        }
    }
    if a.load.is_none() && (a.load_after_s.is_some() || a.no_kpi_patch) {
        return Err(format!(
            "--load-after and --no-kpi-patch need --load; {USAGE}"
        ));
    }
    a.config = config.ok_or(USAGE)?;
    Ok(a)
}

/// Parse the config and check the demo gate, without installing anything.
fn prepare(text: &str) -> Result<(EasConfig, DemoRuntime, String), String> {
    let config: EasConfig = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    config.validate()?;
    let root: serde_yaml::Value = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let runtime = DemoRuntime::from_root(&root, registry()).map_err(|e| e.to_string())?;
    let Some(mode) = runtime.mode(KEY) else {
        return Err(format!(
            "nr-she-eas runs only with demo {KEY} enabled (demo.enabled and demo.{KEY}.enabled)"
        ));
    };
    Ok((config, runtime, mode))
}

#[tokio::main]
async fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!("nr-she-eas: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let argv: Vec<String> = std::env::args().collect();
    let args = parse_args(&argv)?;
    let path = &args.config;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let (config, runtime, mode) = prepare(&text).map_err(|e| format!("{path}: {e}"))?;
    nextgsim_demo::install(runtime).map_err(|e| format!("{path}: {e}"))?;

    let reflector = bind_when_available(config.stamp_addr()).await?;
    let jobs = bind_when_available(config.job_addr()).await?;
    demo_log!(
        warn,
        KEY,
        "EAS {} (mode {mode}): unauthenticated STAMP Session-Reflector on {} (TS 33.501 T.5 \
         departure) and the job listener on {}; no published port",
        config.eas_id,
        config.stamp_addr(),
        config.job_addr()
    );
    tokio::spawn(stamp::run(reflector, Arc::default()));
    let avl = Arc::new(AtomicU32::new(config.avl_comp));
    tokio::spawn(job::serve(
        jobs,
        config.eas_id.clone(),
        avl.clone(),
        Arc::default(),
    ));

    let client = ees_client(config.ees.oauth2.as_ref(), None, Duration::from_secs(3)).await?;
    let mut reg = Registrar::new(client, &config.ees.api_root, config.ad());
    let interval = Duration::from_secs(config.check_interval_s.max(1));
    let mut usr1 = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::user_defined1())
        .map_err(|e| format!("SIGUSR1: {e}"))?;
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
        .map_err(|e| format!("SIGTERM: {e}"))?;
    let mut load_at: Option<tokio::time::Instant> = None;
    let mut load_pending = args.load;
    let mut ticker = tokio::time::interval(interval);
    loop {
        let load_due = async {
            match load_at {
                Some(t) => tokio::time::sleep_until(t).await,
                None => std::future::pending().await,
            }
        };
        tokio::select! {
            _ = ticker.tick() => {
                match reg.ensure().await {
                    Check::Registered(_) => {
                        if let (Some(after), None) = (args.load_after_s, load_at) {
                            if load_pending.is_some() {
                                load_at = Some(tokio::time::Instant::now() + Duration::from_secs(after));
                            }
                        }
                    }
                    Check::Alive => {}
                    Check::Failed(why) => demo_log!(warn, KEY, "EAS-REG eas={}: {why}; retrying", config.eas_id),
                }
            }
            _ = usr1.recv() => {
                if let Some(l) = load_pending.take() {
                    apply_load(&mut reg, &avl, l, args.no_kpi_patch).await;
                } else {
                    demo_log!(info, KEY, "SIGUSR1: no --load pending");
                }
            }
            () = load_due => {
                load_at = None;
                if let Some(l) = load_pending.take() {
                    apply_load(&mut reg, &avl, l, args.no_kpi_patch).await;
                }
            }
            _ = term.recv() => break,
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    if let Err(e) = reg.deregister().await {
        demo_log!(warn, KEY, "EAS-DEREG eas={}: {e}", config.eas_id);
    }
    Ok(())
}

/// The load act: the EAS has less compute from now on; unless
/// `no_kpi_patch`, the EES is told (merge-patch of `svcKpi.avlComp`).
async fn apply_load(reg: &mut Registrar, avl: &AtomicU32, load: u32, no_kpi_patch: bool) {
    let old = avl.swap(load, Ordering::Relaxed);
    if no_kpi_patch {
        demo_log!(
            warn,
            KEY,
            "LOAD eas={} avlComp {old} -> {load} locally; --no-kpi-patch: the EES is NOT told \
             (revert arm)",
            reg.ad().eas_id
        );
        return;
    }
    demo_log!(
        info,
        KEY,
        "LOAD eas={} avlComp {old} -> {load}",
        reg.ad().eas_id
    );
    if let Err(e) = reg.patch_avl_comp(load).await {
        demo_log!(
            warn,
            KEY,
            "EAS-KPI-PATCH eas={} failed: {e}",
            reg.ad().eas_id
        );
    }
}

/// Bind `addr`, waiting up to [`BIND_WAIT`] for it to appear (a `dn` address
/// can come up after the container starts).
async fn bind_when_available(addr: std::net::SocketAddr) -> Result<UdpSocket, String> {
    let deadline = tokio::time::Instant::now() + BIND_WAIT;
    loop {
        match UdpSocket::bind(addr).await {
            Ok(s) => return Ok(s),
            Err(e)
                if e.kind() == std::io::ErrorKind::AddrNotAvailable
                    && tokio::time::Instant::now() < deadline =>
            {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) => return Err(format!("bind {addr}: {e}")),
        }
    }
}

const BIND_WAIT: Duration = Duration::from_secs(120);

#[cfg(test)]
mod tests {
    use super::*;

    const ON: &str = "eas_id: eas-local\nmax_resp_time_ms: 10\navl_comp: 50\n\
        ees: { api_root: \"http://172.23.0.31:7777\" }\n\
        demo:\n  enabled: true\n  d443a-edge-budget:\n    enabled: true\n    mode: measured-budget\n";

    fn argv(s: &str) -> Vec<String> {
        std::iter::once("nr-she-eas")
            .chain(s.split_whitespace())
            .map(str::to_string)
            .collect()
    }

    /// The sidecar is demo-only: no block, the kill switch off or the demo's
    /// own flag off refuse to start; `mode: bogus` fails with UnknownMode.
    #[test]
    fn the_sidecar_refuses_to_start_unless_the_demo_is_enabled_471() {
        for text in [
            ON.split("demo:").next().unwrap().to_string(),
            ON.replace("  enabled: true\n  d443a", "  enabled: false\n  d443a"),
            ON.replace("    enabled: true\n", "    enabled: false\n"),
        ] {
            let err = prepare(&text).expect_err(&text);
            assert!(
                err.contains("runs only with demo d443a-edge-budget enabled"),
                "{err}"
            );
        }
        let err = prepare(&ON.replace("measured-budget", "bogus")).expect_err("bogus");
        assert!(
            err.contains("unknown mode \"bogus\" for demo \"d443a-edge-budget\""),
            "{err}"
        );
        assert!(prepare(&format!("{ON}bogus_field: 1\n")).is_err());
        for mode in MODES {
            let (_, _, m) = prepare(&ON.replace("measured-budget", mode)).unwrap();
            assert_eq!(m, *mode);
        }
    }

    /// Without `address:` both listeners bind loopback; a wildcard address is
    /// refused (it would also be the endPt the EES hands out).
    #[test]
    fn the_listeners_default_to_loopback_and_never_a_wildcard_471() {
        let (c, _, _) = prepare(ON).unwrap();
        assert_eq!(c.stamp_addr(), "127.0.0.1:862".parse().unwrap());
        assert_eq!(c.job_addr(), "127.0.0.1:9471".parse().unwrap());
        for wild in ["0.0.0.0", "\"::\""] {
            let err = prepare(&format!("address: {wild}\n{ON}")).expect_err(wild);
            assert!(err.contains("wildcard"), "{err}");
        }
        let (c, _, _) = prepare(&format!("address: 10.60.0.11\nstamp_port: 8620\n{ON}")).unwrap();
        assert_eq!(c.stamp_addr(), "10.60.0.11:8620".parse().unwrap());
    }

    #[test]
    fn the_load_knobs_parse_and_need_a_load_471() {
        assert_eq!(
            parse_args(&argv("-c e.yaml --load 10 --load-after 30 --no-kpi-patch")).unwrap(),
            Args {
                config: "e.yaml".into(),
                load: Some(10),
                load_after_s: Some(30),
                no_kpi_patch: true,
            }
        );
        assert!(parse_args(&argv("-c e.yaml --no-kpi-patch")).is_err());
        assert!(parse_args(&argv("-c e.yaml --load-after 3")).is_err());
        assert!(parse_args(&argv("-c e.yaml --load 0")).is_err());
        assert!(parse_args(&argv("--load 3")).is_err());
        assert_eq!(parse_args(&argv("-c e.yaml")).unwrap().load, None);
    }
}
