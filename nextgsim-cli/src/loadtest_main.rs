//! Multi-UE Load Testing Tool
//!
//! Item 125: Multi-UE load test for `NextGCore` 5G Core

mod loadtest;

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "nr-loadtest")]
#[command(author, version)]
#[command(about = "Multi-UE load testing tool for NextGCore 5G Core")]
struct Args {
    /// Number of UEs to simulate
    #[arg(long, default_value = "10")]
    ues: u32,

    /// Registration rate (UEs/second, 0 = burst)
    #[arg(long, default_value = "5")]
    rate: u32,

    /// gNB address
    #[arg(long, default_value = "127.0.0.100")]
    gnb_addr: String,

    /// AMF address
    #[arg(long, default_value = "127.0.0.5")]
    amf_addr: String,

    /// Base IMSI (incremented per UE)
    #[arg(long, default_value = "999700000000001")]
    base_imsi: String,

    /// DNN for PDU session
    #[arg(long, default_value = "internet")]
    dnn: String,

    /// S-NSSAI SST
    #[arg(long, default_value = "1")]
    sst: u8,

    /// Test duration limit in seconds (0 = unlimited)
    #[arg(long, default_value = "0")]
    duration: u64,

    /// Skip PDU session establishment
    #[arg(long)]
    skip_pdu: bool,

    /// Enable ping test after PDU session
    #[arg(long)]
    ping: bool,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or(&args.log_level)
    ).init();

    let config = loadtest::LoadTestConfig {
        num_ues: args.ues,
        rate: args.rate,
        gnb_addr: args.gnb_addr,
        amf_addr: args.amf_addr,
        base_imsi: args.base_imsi,
        dnn: args.dnn,
        sst: args.sst,
        duration_secs: args.duration,
        enable_pdu_session: !args.skip_pdu,
        enable_ping: args.ping,
        ..Default::default()
    };

    let report = loadtest::run_load_test(config).await;

    // Exit with error if success rate < 90%
    let reg_total = report.registration_success + report.registration_failure;
    if reg_total > 0 && (report.registration_success as f64 / reg_total as f64) < 0.9 {
        std::process::exit(1);
    }

    Ok(())
}
