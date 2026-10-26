//! `lte-ue`: the nextgsim LTE UE.
//!
//! ```bash
//! lte-ue -c config/lte-ue.yaml
//! lte-ue detach [--switch-off] [--timeout SECS]   # ask the running UE to detach
//! lte-ue bearer-modify <EBI> <GBR-kbps> [--timeout SECS]
//! lte-ue bearer-release <EBI> [--timeout SECS]
//! ```
//!
//! With a `subscriber` configured the UE attaches (TS 24.301 §5.5.1) on the
//! cell it selects; without one it selects a cell, reads its system
//! information and camps (TS 36.304 §5.2), because there is no USIM to attach
//! with.
//!
//! A running UE listens on a Unix control socket (`LTE_UE_CONTROL_SOCKET`,
//! default [`DEFAULT_CONTROL_SOCKET`]). `lte-ue detach` connects to it, and the
//! UE runs a UE-initiated detach (TS 24.301 §5.5.2.2, #365). The command exits
//! 0 once the UE is EMM-DEREGISTERED (after DETACH ACCEPT, or at once for a
//! switch-off detach), 1 otherwise.
//!
//! `lte-ue bearer-modify` and `lte-ue bearer-release` run a UE-requested bearer
//! resource modification (TS 24.301 §6.5.4, #434) of a dedicated EPS bearer:
//! new QoS (MBR = GBR = the rate, both ways) for its traffic flows, or the
//! release of all of them. They exit 0 once the network's MODIFY, or
//! DEACTIVATE, EPS BEARER CONTEXT REQUEST for it was accepted (`modified`,
//! `released`), 1 on a refusal, a BEARER RESOURCE MODIFICATION REJECT or a
//! T3481 abort. The socket protocol is in [`nextgsim_lte_ue::control`].

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::process::ExitCode;

use std::sync::{Arc, Mutex};

use clap::Parser;
use nextgsim_eps_nas::ie::PdnAddress;
use nextgsim_lte_ue::config::TunSettings;
use nextgsim_lte_ue::nas::EpsNasStatus;
use nextgsim_lte_ue::rrc::UeRrcState;
use nextgsim_ue::tun::{TunConfig, TunInterface};
use tokio::sync::mpsc;

/// The longest control command line read; the commands are a few words.
const MAX_CONTROL_LINE: u64 = 256;

/// The control socket a running UE listens on unless `LTE_UE_CONTROL_SOCKET`
/// names another.
const DEFAULT_CONTROL_SOCKET: &str = "/run/nextgsim/lte-ue.sock";

/// nextgsim LTE UE - E-UTRA UE simulator
#[derive(Parser, Debug)]
#[command(name = "lte-ue", author, version, about, long_about = None)]
struct Args {
    /// Path to the UE configuration file (YAML); runs the UE
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: Option<String>,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(clap::Subcommand, Debug)]
enum Command {
    /// Ask the running UE to detach (TS 24.301 §5.5.2.2) and wait until it is
    /// EMM-DEREGISTERED.
    Detach {
        /// Detach with the switch-off bit set: the network sends no DETACH ACCEPT
        #[arg(long)]
        switch_off: bool,
        /// Seconds to wait for the detach to complete
        #[arg(long, default_value_t = 60)]
        timeout: u64,
        /// The UE's control socket
        #[arg(long, value_name = "PATH")]
        socket: Option<String>,
    },
    /// Ask the running UE for new QoS for a dedicated EPS bearer (TS 24.301
    /// §6.5.4): MBR = GBR = the rate, uplink and downlink. Waits until the
    /// network's MODIFY EPS BEARER CONTEXT REQUEST was accepted.
    BearerModify {
        /// The dedicated bearer's EPS bearer identity
        ebi: u8,
        /// The guaranteed (and maximum) bit rate, kbit/s
        gbr_kbps: u32,
        /// Seconds to wait for the network
        #[arg(long, default_value_t = nextgsim_lte_ue::control::DEFAULT_TIMEOUT_SECS)]
        timeout: u64,
        /// The UE's control socket
        #[arg(long, value_name = "PATH")]
        socket: Option<String>,
    },
    /// Ask the running UE to release a dedicated EPS bearer (TS 24.301
    /// §6.5.4, every packet filter deleted, ESM cause #36). Waits until the
    /// network's DEACTIVATE EPS BEARER CONTEXT REQUEST was accepted.
    BearerRelease {
        /// The dedicated bearer's EPS bearer identity
        ebi: u8,
        /// Seconds to wait for the network
        #[arg(long, default_value_t = nextgsim_lte_ue::control::DEFAULT_TIMEOUT_SECS)]
        timeout: u64,
        /// The UE's control socket
        #[arg(long, value_name = "PATH")]
        socket: Option<String>,
    },
}

/// The control socket path: `LTE_UE_CONTROL_SOCKET`, else the default.
fn control_socket_path(arg: Option<String>) -> String {
    arg.or_else(|| std::env::var("LTE_UE_CONTROL_SOCKET").ok())
        .unwrap_or_else(|| DEFAULT_CONTROL_SOCKET.to_string())
}

/// `lte-ue detach` / `bearer-modify` / `bearer-release`: send the command
/// `line` and print the UE's answer; success when it is `done`.
async fn control_client(
    name: &str,
    line: String,
    done: &str,
    timeout: u64,
    socket: Option<String>,
) -> ExitCode {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let path = control_socket_path(socket);
    let stream = match tokio::net::UnixStream::connect(&path).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("lte-ue {name}: cannot reach the UE at {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let (read, mut write) = stream.into_split();
    if let Err(e) = write.write_all(format!("{line}\n").as_bytes()).await {
        eprintln!("lte-ue {name}: {e}");
        return ExitCode::FAILURE;
    }
    let mut answer = String::new();
    let mut read = BufReader::new(read);
    let wait = std::time::Duration::from_secs(timeout + 5);
    match tokio::time::timeout(wait, read.read_line(&mut answer)).await {
        Ok(Ok(_)) => {
            let answer = answer.trim();
            println!("{answer}");
            if answer == done {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Ok(Err(e)) => {
            eprintln!("lte-ue {name}: {e}");
            ExitCode::FAILURE
        }
        Err(_) => {
            eprintln!("lte-ue {name}: no answer from the UE");
            ExitCode::FAILURE
        }
    }
}

/// Serve the control socket (#365, #434): each connection carries one command
/// line ([`nextgsim_lte_ue::control`]) and gets one answer line.
async fn control_server(
    path: String,
    status: Option<Arc<Mutex<EpsNasStatus>>>,
    commands: mpsc::UnboundedSender<nextgsim_lte_ue::task::ControlCommand>,
) {
    if let Some(dir) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    // A stale socket from an earlier run is replaced; anything else at the path
    // is left alone, and the UE runs without a control socket (#369 review).
    if let Ok(meta) = std::fs::symlink_metadata(&path) {
        use std::os::unix::fs::FileTypeExt;
        if !meta.file_type().is_socket() {
            log::warn!(
                "{path} exists and is not a socket; no control socket, `lte-ue detach` \
                 will not reach this UE"
            );
            return;
        }
        let _ = std::fs::remove_file(&path);
    }
    let listener = match tokio::net::UnixListener::bind(&path) {
        Ok(l) => l,
        Err(e) => {
            log::warn!("no control socket at {path} ({e}); `lte-ue detach` will not reach this UE");
            return;
        }
    };
    log::info!("control socket {path}");
    loop {
        let Ok((stream, _)) = listener.accept().await else {
            continue;
        };
        let status = status.clone();
        let commands = commands.clone();
        tokio::spawn(async move { serve_control(stream, status, commands).await });
    }
}

async fn serve_control(
    stream: tokio::net::UnixStream,
    status: Option<Arc<Mutex<EpsNasStatus>>>,
    commands: mpsc::UnboundedSender<nextgsim_lte_ue::task::ControlCommand>,
) {
    use nextgsim_eps_nas::EmmState;
    use tokio::io::AsyncReadExt;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    // One short command line; a peer that sends more is cut off (#369 review).
    if BufReader::new(read.take(MAX_CONTROL_LINE))
        .read_line(&mut line)
        .await
        .is_err()
    {
        return;
    }
    use nextgsim_lte_ue::control::{parse_line, run_bearer_command, ControlRequest};
    let answer = match (parse_line(&line), status) {
        (Err(e), _) => e,
        (Ok(ControlRequest::Detach { .. }), None) => {
            "error: no subscriber configured, nothing to detach".to_string()
        }
        (Ok(ControlRequest::Bearer { .. }), None) => {
            "error: no subscriber configured, no EPS bearers".to_string()
        }
        (Ok(ControlRequest::Bearer { cmd, timeout }), Some(status)) => {
            run_bearer_command(&status, &commands, cmd, timeout).await
        }
        (
            Ok(ControlRequest::Detach {
                switch_off,
                timeout,
            }),
            Some(status),
        ) => {
            let state = || status.lock().map(|s| s.state).ok();
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout);
            let mut asked: Option<tokio::time::Instant> = None;
            loop {
                let now = tokio::time::Instant::now();
                match state() {
                    Some(EmmState::Detached) => break "detached".to_string(),
                    Some(EmmState::DeregisteredInitiated) => {}
                    // Ask (again, at most once a second): RRC does not act
                    // on a detach while a connection is being set up.
                    _ if asked.is_none_or(|t| now >= t + std::time::Duration::from_secs(1)) => {
                        let _ = commands
                            .send(nextgsim_lte_ue::task::ControlCommand::Detach { switch_off });
                        asked = Some(now);
                    }
                    _ => {}
                }
                if now >= deadline {
                    break format!("error: not detached within {timeout} s (EMM {:?})", state());
                }
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    };
    let _ = write.write_all(format!("{answer}\n").as_bytes()).await;
}

#[tokio::main]
async fn main() -> ExitCode {
    let _ = tracing_log::LogTracer::init();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let args = Args::parse();
    match args.command {
        Some(Command::Detach {
            switch_off,
            timeout,
            socket,
        }) => {
            let line = format!(
                "detach{} {timeout}",
                if switch_off { " switch-off" } else { "" }
            );
            return control_client("detach", line, "detached", timeout, socket).await;
        }
        Some(Command::BearerModify {
            ebi,
            gbr_kbps,
            timeout,
            socket,
        }) => {
            let line = format!("bearer-modify {ebi} {gbr_kbps} {timeout}");
            return control_client("bearer-modify", line, "modified", timeout, socket).await;
        }
        Some(Command::BearerRelease {
            ebi,
            timeout,
            socket,
        }) => {
            let line = format!("bearer-release {ebi} {timeout}");
            return control_client("bearer-release", line, "released", timeout, socket).await;
        }
        None => {}
    }
    let Some(config_file) = args.config_file else {
        eprintln!("lte-ue: -c <config> is required to run the UE");
        return ExitCode::FAILURE;
    };
    let cfg = match nextgsim_lte_ue::config::load_with_env_and_validate(&config_file) {
        Ok(c) => c,
        Err(e) => {
            log::error!("{config_file}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let socket =
        match tokio::net::UdpSocket::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::UNSPECIFIED), 0))
            .await
        {
            Ok(s) => s,
            Err(e) => {
                log::error!("cannot bind the RLS socket: {e}");
                return ExitCode::FAILURE;
            }
        };
    let (state_tx, _state_rx) = tokio::sync::watch::channel(UeRrcState::NoCell);
    let (stop_tx, stop_rx) = tokio::sync::watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = stop_tx.send(true);
    });
    let (ip, ul_tx, dl_rx) = nextgsim_lte_ue::task::IpIo::channel(256);
    let mut status = None;
    let nas: Box<dyn nextgsim_lte_ue::nas::NasLayer> = match cfg.eps_nas() {
        Some(n) => {
            log::info!("USIM IMSI {}: EPS NAS enabled", n.imsi);
            let nas = nextgsim_lte_ue::nas::EpsNas::new(n);
            status = Some(nas.status());
            if let Some(tun) = cfg.tun.clone() {
                tokio::spawn(tun_bridge(tun, nas.status(), ul_tx, dl_rx));
            }
            Box::new(nas)
        }
        None => {
            log::info!("no subscriber configured: camping only");
            Box::new(nextgsim_lte_ue::nas::NoNas)
        }
    };
    let (control_tx, control_rx) = mpsc::unbounded_channel();
    tokio::spawn(control_server(
        control_socket_path(None),
        status,
        control_tx,
    ));
    match nextgsim_lte_ue::task::run_with_control(
        &cfg, nas, socket, state_tx, stop_rx, ip, control_rx,
    )
    .await
    {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("RLS socket failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The IPv4 PDN address of the default bearer NAS holds now, with its EBI.
fn bearer_ipv4(status: &Mutex<EpsNasStatus>) -> Option<(u8, Ipv4Addr)> {
    let b = status.lock().ok()?.bearer.clone()?;
    match b.pdn_address {
        PdnAddress::Ipv4(a) | PdnAddress::Ipv4v6(_, a) => Some((b.ebi, a)),
        other => {
            log::warn!("default bearer PDN address {other:?}: no IPv4 for the TUN interface");
            None
        }
    }
}

/// Bridge the default bearer to a TUN interface: wait for NAS to activate
/// the default EPS bearer with an IPv4 PDN address (TS 24.301 §6.4.1.3),
/// give the interface that address, and pump packets both ways.
///
/// The interface outlives the bearer. When a later attach (after a SERVICE
/// REJECT, or a network-initiated detach) activates a default bearer with a
/// different PDN address, the same interface takes the new address, so its
/// name -- and whatever the host routes through it -- stays.
async fn tun_bridge(
    settings: TunSettings,
    status: Arc<Mutex<EpsNasStatus>>,
    ul_tx: mpsc::Sender<Vec<u8>>,
    mut dl_rx: mpsc::Receiver<Vec<u8>>,
) {
    let (ebi, address) = loop {
        if let Some(b) = bearer_ipv4(&status) {
            break b;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    let netmask = Ipv4Addr::new(255, 255, 255, 255);
    let config = TunConfig {
        name: Some(settings.name.unwrap_or_else(|| format!("lteuetun{ebi}"))),
        address: Some(address),
        netmask: Some(netmask),
        configure_routing: settings.configure_routing,
        psi: Some(ebi),
        ..TunConfig::default()
    };
    let tun = match TunInterface::create_and_configure(&config, i32::from(ebi)).await {
        Ok(t) => Arc::new(t),
        Err(e) => {
            log::error!("cannot create the TUN interface (CAP_NET_ADMIN needed): {e}");
            return;
        }
    };
    log::info!("default bearer EBI {ebi}: TUN interface up with {address}");
    let reader = tun.clone();
    tokio::spawn(async move {
        while let Ok(p) = reader.recv().await {
            if ul_tx.send(p).await.is_err() {
                return;
            }
        }
    });
    let mut current = address;
    let mut poll = tokio::time::interval(std::time::Duration::from_millis(200));
    loop {
        tokio::select! {
            p = dl_rx.recv() => {
                let Some(p) = p else { return };
                if let Err(e) = tun.send(&p).await {
                    log::warn!("TUN write failed: {e}");
                }
            }
            _ = poll.tick() => {
                // A detached UE has no PDN connection: its TUN goes down (#365).
                let detached = status
                    .lock()
                    .is_ok_and(|s| s.state == nextgsim_eps_nas::EmmState::Detached);
                if detached {
                    match tun.set_up(false) {
                        Ok(()) => log::info!("detached: TUN interface {} down", tun.name()),
                        Err(e) => log::error!("cannot bring the TUN interface down: {e}"),
                    }
                    return;
                }
                let Some((ebi, a)) = bearer_ipv4(&status) else { continue };
                if a == current {
                    continue;
                }
                match tun.set_ipv4_address(a, netmask) {
                    Ok(()) => {
                        log::info!(
                            "default bearer EBI {ebi} re-established: TUN interface {} now {a} (was {current})",
                            tun.name()
                        );
                        current = a;
                    }
                    Err(e) => log::error!("cannot re-address the TUN interface: {e}"),
                }
            }
        }
    }
}
