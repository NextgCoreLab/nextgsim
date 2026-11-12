//! `nr-fl-server`: the FL application server (AF) of demo D445A (nextgsim#480).
//! Non-normative. See `nextgsim_fl::d445a::server` for what it does.
//!
//! ```text
//! nr-fl-server -c af.yaml                 # every build with demo-d445a-fl-t8
//! nr-fl-server -c af.yaml --no-t8         # revert arm, revert-hooks builds only
//! nr-fl-server -c af.yaml --freeze-global # revert arm, revert-hooks builds only
//! ```
//!
//! ```yaml
//! listen: 10.46.0.10:7480       # default 127.0.0.1:7480; never a published port
//! t8_listen: 10.46.0.10:7481    # default 127.0.0.1:7481
//! nef: { api_root: "http://nef:7777" }
//! seed: 480
//! holds:
//!   - { before_round: 4, until: member-excluded, timeout_s: 180 }
//!   - { before_round: 7, until: members-full, timeout_s: 300 }
//! demo:
//!   enabled: true
//!   d445a-fl-t8: { enabled: true, mode: t8-exclusion }
//! ```
//!
//! The AF is demo-only: it refuses to start unless the demo is enabled, and an
//! unknown key or mode fails startup (`DemoConfigError`). Its mode is config plus
//! restart. After the last arm it keeps its listeners open and idles, so the
//! clients stay connected while the logs are collected.

use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;

use nextgsim_demo::{DemoRegistry, DemoRuntime, DemoSpec};
use nextgsim_fl::d445a::server::{Af, Knobs, ServerConfig};
use nextgsim_fl::d445a::{KEY, MODES};

fn registry() -> DemoRegistry {
    DemoRegistry::new(&[DemoSpec {
        key: KEY,
        modes: MODES,
        requires: &[],
    }])
}

#[derive(Debug, PartialEq)]
struct Args {
    config: String,
    knobs: Knobs,
}

const USAGE: &str = "usage: nr-fl-server -c <af.yaml> [--no-t8] [--freeze-global] (the last two in revert-hooks builds only)";

fn parse_args(args: &[String]) -> Result<Args, String> {
    let mut config = None;
    let mut knobs = Knobs::default();
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "-c" | "--config" => config = Some(it.next().ok_or(USAGE)?.clone()),
            "--no-t8" | "--freeze-global" if !cfg!(feature = "revert-hooks") => {
                return Err(format!(
                    "{a} is a revert arm and needs a revert-hooks build"
                ))
            }
            "--no-t8" => knobs.no_t8 = true,
            "--freeze-global" => knobs.freeze_global = true,
            _ => return Err(USAGE.to_string()),
        }
    }
    Ok(Args {
        config: config.ok_or(USAGE)?,
        knobs,
    })
}

/// Parse the config and check the demo gate, without installing anything.
/// Fails unless demo d445a-fl-t8 is enabled: the AF is demo-only.
fn prepare(text: &str) -> Result<(ServerConfig, DemoRuntime, String), String> {
    let config: ServerConfig = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let root: serde_yaml::Value = serde_yaml::from_str(text).map_err(|e| e.to_string())?;
    let runtime = DemoRuntime::from_root(&root, registry()).map_err(|e| e.to_string())?;
    let Some(mode) = runtime.mode(KEY) else {
        return Err(format!(
            "nr-fl-server runs only with demo {KEY} enabled (demo.enabled and demo.{KEY}.enabled)"
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
            tracing::error!("nr-fl-server: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let args = parse_args(&args)?;
    let path = &args.config;
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let (config, runtime, mode) = prepare(&text).map_err(|e| format!("{path}: {e}"))?;
    nextgsim_demo::install(runtime).map_err(|e| format!("{path}: {e}"))?;
    for addr in [config.listen, config.t8_listen] {
        if addr.ip().is_unspecified() {
            tracing::warn!(
                "[demo:{KEY}] binding the wildcard {addr}: acceptable only inside a private \
                 network with no published port"
            );
        }
    }
    let fl = bind_when_available(config.listen).await?;
    let t8 = if args.knobs.no_t8 {
        None
    } else {
        Some(bind_when_available(config.t8_listen).await?)
    };
    let af = Af::new(config, &mode, args.knobs).await?;
    af.run(fl, t8).await;
    tracing::info!("[demo:{KEY}] idling with the listeners open (SIGTERM to stop)");
    std::future::pending::<()>().await;
    Ok(())
}

/// Bind `addr`, waiting up to [`BIND_WAIT`] for it to appear (a pod or network
/// address can come up after the container starts).
async fn bind_when_available(addr: SocketAddr) -> Result<tokio::net::TcpListener, String> {
    let deadline = tokio::time::Instant::now() + BIND_WAIT;
    loop {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => return Ok(l),
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

    const ON: &str = "nef: { api_root: \"http://nef:7777\" }\ndemo:\n  enabled: true\n  d445a-fl-t8:\n    enabled: true\n    mode: t8-exclusion\n";

    /// The AF is demo-only: with no `demo:` block, the kill switch off, or the
    /// demo's own flag off, it refuses to start; an unknown mode fails with
    /// `UnknownMode` and no fallback.
    #[test]
    fn the_af_refuses_to_start_unless_the_demo_is_enabled() {
        for text in [
            "nef: { api_root: \"http://nef:7777\" }\n",
            &ON.replace("  enabled: true\n  d445a", "  enabled: false\n  d445a"),
            &ON.replace("    enabled: true\n", "    enabled: false\n"),
        ] {
            let err = prepare(text).expect_err(text);
            assert!(
                err.contains("runs only with demo d445a-fl-t8 enabled"),
                "{err}"
            );
        }
        let err = prepare(&ON.replace("t8-exclusion", "bogus")).expect_err("bogus");
        assert!(
            err.contains("unknown mode \"bogus\" for demo \"d445a-fl-t8\""),
            "{err}"
        );
        assert!(
            prepare(&format!("{ON}bogus_field: 1\n")).is_err(),
            "unknown fields fail"
        );
        let (_, _, mode) = prepare(&ON.replace("t8-exclusion", "poisoned-ue")).unwrap();
        assert_eq!(mode, "poisoned-ue");
    }

    /// Without `listen:` / `t8_listen:` both listeners bind loopback.
    #[test]
    fn the_default_listeners_are_loopback() {
        let (c, _, _) = prepare(ON).unwrap();
        assert_eq!(c.listen, SocketAddr::from(([127, 0, 0, 1], 7480)));
        assert_eq!(c.t8_listen, SocketAddr::from(([127, 0, 0, 1], 7481)));
    }

    /// The revert knobs exist only in revert-hooks builds.
    #[test]
    fn the_revert_knobs_need_revert_hooks() {
        let a = |v: &[&str]| parse_args(&v.iter().map(ToString::to_string).collect::<Vec<_>>());
        assert_eq!(
            a(&["x", "-c", "af.yaml"]).unwrap(),
            Args {
                config: "af.yaml".into(),
                knobs: Knobs::default()
            }
        );
        let r = a(&["x", "-c", "af.yaml", "--no-t8", "--freeze-global"]);
        if cfg!(feature = "revert-hooks") {
            assert_eq!(
                r.unwrap().knobs,
                Knobs {
                    no_t8: true,
                    freeze_global: true
                }
            );
        } else {
            assert!(r.unwrap_err().contains("needs a revert-hooks build"));
        }
        assert!(a(&["x"]).is_err());
        assert!(a(&["x", "-c", "af.yaml", "--bogus"]).is_err());
    }
}
