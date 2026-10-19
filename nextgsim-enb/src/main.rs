//! `lte-enb`: the nextgsim LTE eNB.
//!
//! ```bash
//! lte-enb -c config/enb.yaml
//! ```

use std::process::ExitCode;

use clap::Parser;

/// nextgsim eNB - LTE E-UTRAN NodeB simulator
#[derive(Parser, Debug)]
#[command(name = "lte-enb")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the eNB configuration file (YAML)
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: String,
}

fn init_logging() {
    let _ = tracing_log::LogTracer::init();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt().with_env_filter(filter).try_init();
}

#[tokio::main]
async fn main() -> ExitCode {
    init_logging();
    let args = Args::parse();
    let cfg = match nextgsim_enb::load_and_validate_enb_config(&args.config_file) {
        Ok(c) => c,
        Err(e) => {
            log::error!("{}: {e}", args.config_file);
            return ExitCode::FAILURE;
        }
    };
    log::info!(
        "eNB ID {:#x} ({:?}), PLMN {}-{}, TAC {}",
        cfg.enb_id,
        cfg.enb_id_type,
        cfg.plmn.mcc,
        cfg.plmn.mnc,
        cfg.tac
    );
    run(cfg).await
}

#[cfg(all(target_os = "linux", feature = "kernel-sctp"))]
async fn run(cfg: nextgsim_enb::EnbConfig) -> ExitCode {
    use nextgsim_enb::s1_task::{establish_s1, kernel::KernelConnector, serve};
    use std::net::SocketAddr;

    let mme = &cfg.mme_configs[0];
    if cfg.mme_configs.len() > 1 {
        log::warn!(
            "only the first of {} MMEs is used in this build",
            cfg.mme_configs.len()
        );
    }
    let remote = SocketAddr::new(mme.address, mme.port);
    let mut connector = KernelConnector::default();
    loop {
        tokio::select! {
            (_, mut transport) = establish_s1(&cfg, &mut connector, remote) => {
                tokio::select! {
                    () = serve(&mut transport) => {}
                    _ = tokio::signal::ctrl_c() => return ExitCode::SUCCESS,
                }
                // TS 36.412 §7: the eNB (re-)establishes the association.
                tokio::time::sleep(std::time::Duration::from_secs(cfg.s1_setup_retry_secs)).await;
            }
            _ = tokio::signal::ctrl_c() => return ExitCode::SUCCESS,
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "kernel-sctp")))]
async fn run(_cfg: nextgsim_enb::EnbConfig) -> ExitCode {
    log::error!(
        "lte-enb was built without the `kernel-sctp` feature. S1-MME is SCTP \
         (TS 36.412 §7) and an MME accepts nothing else; rebuild on Linux with \
         libsctp: cargo build -p nextgsim-enb --features kernel-sctp"
    );
    ExitCode::FAILURE
}
