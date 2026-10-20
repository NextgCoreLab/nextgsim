//! `lte-ue`: the nextgsim LTE UE.
//!
//! ```bash
//! lte-ue -c config/lte-ue.yaml
//! ```
//!
//! With a `subscriber` configured the UE attaches (TS 24.301 §5.5.1) on the
//! cell it selects; without one it selects a cell, reads its system
//! information and camps (TS 36.304 §5.2), because there is no USIM to attach
//! with.

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

/// nextgsim LTE UE - E-UTRA UE simulator
#[derive(Parser, Debug)]
#[command(name = "lte-ue", author, version, about, long_about = None)]
struct Args {
    /// Path to the UE configuration file (YAML)
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: String,
}

#[tokio::main]
async fn main() -> ExitCode {
    let _ = tracing_log::LogTracer::init();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();

    let args = Args::parse();
    let cfg = match nextgsim_lte_ue::config::load_and_validate(&args.config_file) {
        Ok(c) => c,
        Err(e) => {
            log::error!("{}: {e}", args.config_file);
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
    let nas: Box<dyn nextgsim_lte_ue::nas::NasLayer> = match cfg.eps_nas() {
        Some(n) => {
            log::info!("USIM IMSI {}: EPS NAS enabled", n.imsi);
            let nas = nextgsim_lte_ue::nas::EpsNas::new(n);
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
    match nextgsim_lte_ue::task::run(&cfg, nas, socket, state_tx, stop_rx, ip).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("RLS socket failed: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Bridge the default bearer to a TUN interface: wait for NAS to activate
/// the default EPS bearer with an IPv4 PDN address (TS 24.301 §6.4.1.3),
/// give the interface that address, and pump packets both ways.
async fn tun_bridge(
    settings: TunSettings,
    status: Arc<Mutex<EpsNasStatus>>,
    ul_tx: mpsc::Sender<Vec<u8>>,
    mut dl_rx: mpsc::Receiver<Vec<u8>>,
) {
    let (ebi, address) = loop {
        let bearer = status.lock().ok().and_then(|s| s.bearer.clone());
        if let Some(b) = bearer {
            match b.pdn_address {
                PdnAddress::Ipv4(a) | PdnAddress::Ipv4v6(_, a) => break (b.ebi, a),
                other => {
                    log::warn!(
                        "default bearer PDN address {other:?}: no IPv4 for the TUN interface"
                    );
                    return;
                }
            }
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    };
    let config = TunConfig {
        name: Some(settings.name.unwrap_or_else(|| format!("lteuetun{ebi}"))),
        address: Some(address),
        netmask: Some(Ipv4Addr::new(255, 255, 255, 255)),
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
    while let Some(p) = dl_rx.recv().await {
        if let Err(e) = tun.send(&p).await {
            log::warn!("TUN write failed: {e}");
        }
    }
}
