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

use clap::Parser;
use nextgsim_lte_ue::rrc::UeRrcState;

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
    let nas: Box<dyn nextgsim_lte_ue::nas::NasLayer> = match cfg.eps_nas() {
        Some(n) => {
            log::info!("USIM IMSI {}: EPS NAS enabled", n.imsi);
            Box::new(nextgsim_lte_ue::nas::EpsNas::new(n))
        }
        None => {
            log::info!("no subscriber configured: camping only");
            Box::new(nextgsim_lte_ue::nas::NoNas)
        }
    };
    match nextgsim_lte_ue::task::run(&cfg, nas, socket, state_tx, stop_rx).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            log::error!("RLS socket failed: {e}");
            ExitCode::FAILURE
        }
    }
}
