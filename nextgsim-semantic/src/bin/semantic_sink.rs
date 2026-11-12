//! `semantic-sink`: the DN-side end of demo D444A (nextgsim#477). Non-normative.
//!
//! Receives the UE application's feature datagrams over the PDU session (it
//! runs behind the UPF, in its network namespace), scores each with the same
//! seeded model the UE uses, and logs one row per epoch: datagrams received
//! and lost, IP bytes, rate, accuracy, and, for PCA epochs, the top-variance
//! raw and mean-pool baselines at equal bytes on the same samples.
//!
//! ```text
//! semantic-sink -c sink.yaml
//! ```
//!
//! ```yaml
//! listen: 10.45.0.1:5444   # default 127.0.0.1:5444; never a published port
//! seed: 444
//! rate_hz: 50
//! epoch_datagrams: 500
//! demo:
//!   enabled: true
//!   d444a-ambr-adapt: { enabled: true, mode: semantic }
//! ```
//!
//! The sink is demo-only: it refuses to start unless the demo is enabled, and
//! an unknown key or mode fails startup (`DemoConfigError`). Its mode only
//! labels the rows (config plus restart).

use std::net::SocketAddr;
use std::process::ExitCode;
use std::time::Duration;

use nextgsim_demo::{demo_log, DemoRegistry, DemoSpec};
use nextgsim_semantic::rate_adapt::{
    D444aModel, EpochScorer, FeatureDatagram, D444A_KEY, D444A_MODES, DEFAULT_SEED,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SinkConfig {
    #[serde(default = "default_listen")]
    listen: SocketAddr,
    #[serde(default = "default_seed")]
    seed: u64,
    #[serde(default = "default_rate_hz")]
    rate_hz: u32,
    #[serde(default = "default_epoch")]
    epoch_datagrams: u64,
    /// Read by `nextgsim_demo`, not here.
    #[serde(default)]
    #[allow(dead_code)]
    demo: serde_yaml::Value,
}

fn default_listen() -> SocketAddr {
    SocketAddr::from(([127, 0, 0, 1], 5444))
}
fn default_seed() -> u64 {
    DEFAULT_SEED
}
fn default_rate_hz() -> u32 {
    50
}
fn default_epoch() -> u64 {
    500
}

fn registry() -> DemoRegistry {
    DemoRegistry::new(&[DemoSpec {
        key: D444A_KEY,
        modes: D444A_MODES,
        requires: &[],
    }])
}

fn config_path() -> Result<String, String> {
    let args: Vec<String> = std::env::args().collect();
    match args.as_slice() {
        [_, flag, path] if flag == "-c" || flag == "--config" => Ok(path.clone()),
        _ => Err("usage: semantic-sink -c <sink.yaml>".to_string()),
    }
}

#[tokio::main(flavor = "current_thread")]
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
            tracing::error!("semantic-sink: {e}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), String> {
    let path = config_path()?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let config: SinkConfig = serde_yaml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
    nextgsim_demo::init_from_yaml(&text, registry()).map_err(|e| format!("{path}: {e}"))?;
    if nextgsim_demo::active_mode(D444A_KEY).is_none() {
        return Err(format!(
            "{path}: semantic-sink runs only with demo {D444A_KEY} enabled (demo.enabled and \
             demo.{D444A_KEY}.enabled)"
        ));
    }
    let socket = bind_when_available(config.listen).await?;
    let scorer = EpochScorer::new(
        D444aModel::new(config.seed),
        config.epoch_datagrams,
        config.rate_hz,
    );
    demo_log!(
        info,
        D444A_KEY,
        "semantic-sink listening on {} (seed {}, {} datagrams per epoch at {} Hz)",
        config.listen,
        config.seed,
        config.epoch_datagrams,
        config.rate_hz
    );
    serve(socket, scorer, &config).await
}

/// Bind `listen`, waiting up to [`BIND_WAIT`] for its address to appear: in the
/// UPF's network namespace, 10.45.0.1 exists only once the UPF has brought its
/// TUN up, which can be after this container starts.
async fn bind_when_available(listen: SocketAddr) -> Result<tokio::net::UdpSocket, String> {
    let deadline = tokio::time::Instant::now() + BIND_WAIT;
    loop {
        match tokio::net::UdpSocket::bind(listen).await {
            Ok(s) => return Ok(s),
            Err(e)
                if e.kind() == std::io::ErrorKind::AddrNotAvailable
                    && tokio::time::Instant::now() < deadline =>
            {
                tokio::time::sleep(Duration::from_secs(1)).await;
            }
            Err(e) => return Err(format!("bind {listen}: {e}")),
        }
    }
}

/// How long [`bind_when_available`] waits.
const BIND_WAIT: Duration = Duration::from_secs(180);

async fn serve(
    socket: tokio::net::UdpSocket,
    mut scorer: EpochScorer,
    config: &SinkConfig,
) -> Result<(), String> {
    // Close an epoch the sender stopped in, after two epochs of silence.
    let idle = Duration::from_secs_f64(
        2.0 * config.epoch_datagrams as f64 / f64::from(config.rate_hz.max(1)),
    );
    let mut buf = vec![0u8; 2048];
    let mut undecodable = 0u64;
    loop {
        let mode = nextgsim_demo::active_mode(D444A_KEY).unwrap_or_default();
        match tokio::time::timeout(idle, socket.recv_from(&mut buf)).await {
            Err(_) => {
                if let Some(row) = scorer.close(&mode) {
                    demo_log!(info, D444A_KEY, "{row} (closed after {idle:?} idle)");
                }
            }
            Ok(Err(e)) => return Err(format!("recv: {e}")),
            Ok(Ok((len, from))) => match FeatureDatagram::decode(&buf[..len]) {
                Ok(d) => {
                    if let Some(row) = scorer.on_datagram(&d, len, &mode) {
                        demo_log!(info, D444A_KEY, "{row}");
                    }
                }
                Err(e) => {
                    undecodable += 1;
                    tracing::debug!(
                        "[demo:{D444A_KEY}] {len} B from {from}: {e} ({undecodable} so far)"
                    );
                }
            },
        }
    }
}
