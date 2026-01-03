//! nextgsim UE (User Equipment) simulator
//!
//! This is the main binary for the 5G UE simulator. It implements:
//! - CLI argument parsing
//! - Configuration loading and validation
//! - Task spawning and lifecycle management
//! - Graceful shutdown handling
//!
//! # Usage
//!
//! ```bash
//! nr-ue -c config/ue.yaml
//! nr-ue -c config/ue.yaml -i 001010000000001
//! nr-ue -c config/ue.yaml -n 10 -t 100
//! ```
//!
//! # Reference
//!
//! Based on UERANSIM's `src/ue.cpp` implementation.

use std::process::ExitCode;
use std::time::Duration;

use anyhow::{bail, Context, Result};
use clap::Parser;
use tokio::signal;
use tokio::sync::watch;
use tracing::{error, info, warn};

use nextgsim_common::config::UeConfig;
use nextgsim_ue::{
    AppMessage, AppTask, NasMessage, RlsMessage, RlsTask, RrcMessage,
    Task, TaskManager, TaskMessage, UeTaskBase, DEFAULT_CHANNEL_CAPACITY,
};

/// nextgsim UE - 5G User Equipment Simulator
#[derive(Parser, Debug)]
#[command(name = "nr-ue")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the UE configuration file (YAML)
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: String,

    /// Use specified IMSI number instead of the one in config file
    /// Format: 15 digits or "imsi-" prefix followed by 15 digits
    #[arg(short = 'i', long = "imsi", value_name = "IMSI")]
    imsi: Option<String>,

    /// Generate specified number of UEs starting from the given IMSI (1-512)
    #[arg(short = 'n', long = "num-of-UE", value_name = "NUM")]
    num_ue: Option<u32>,

    /// Starting delay in milliseconds for each of the UEs
    #[arg(short = 't', long = "tempo", value_name = "TEMPO")]
    tempo: Option<u64>,

    /// Disable CLI command interface for this instance
    #[arg(short = 'l', long = "disable-cmd")]
    disable_cmd: bool,

    /// Do not auto configure routing for UE TUN interface
    #[arg(short = 'r', long = "no-routing-config")]
    no_routing_config: bool,
}

/// Parsed and validated CLI options
#[derive(Debug, Clone)]
pub struct UeOptions {
    /// Path to configuration file
    pub config_file: String,
    /// Override IMSI (without "imsi-" prefix)
    pub imsi: Option<String>,
    /// Number of UEs to spawn
    pub num_ue: u32,
    /// Delay between UE starts in milliseconds
    pub tempo: u64,
    /// Whether CLI is disabled
    pub disable_cmd: bool,
    /// Whether to skip routing configuration
    pub no_routing_config: bool,
}

impl TryFrom<Args> for UeOptions {
    type Error = anyhow::Error;

    fn try_from(args: Args) -> Result<Self> {
        // Validate and normalize IMSI
        let imsi = if let Some(imsi_arg) = args.imsi {
            let normalized = normalize_imsi(&imsi_arg)?;
            Some(normalized)
        } else {
            None
        };

        // Validate number of UEs
        let num_ue = if let Some(n) = args.num_ue {
            if n == 0 {
                bail!("Number of UEs must be at least 1");
            }
            if n > 512 {
                bail!("Number of UEs cannot exceed 512");
            }
            n
        } else {
            1
        };

        // Validate tempo
        let tempo = args.tempo.unwrap_or(0);

        Ok(UeOptions {
            config_file: args.config_file,
            imsi,
            num_ue,
            tempo,
            disable_cmd: args.disable_cmd,
            no_routing_config: args.no_routing_config,
        })
    }
}

/// Normalizes an IMSI string by removing the "imsi-" prefix if present.
/// Validates that the result is a valid 15-digit IMSI.
fn normalize_imsi(imsi: &str) -> Result<String> {
    let normalized = if imsi.len() > 5
        && imsi.starts_with("imsi-")
    {
        imsi[5..].to_string()
    } else {
        imsi.to_string()
    };

    // Validate IMSI format (15 digits)
    if normalized.len() != 15 {
        bail!(
            "Invalid IMSI '{}': must be exactly 15 digits (got {} digits)",
            normalized,
            normalized.len()
        );
    }

    if !normalized.chars().all(|c| c.is_ascii_digit()) {
        bail!("Invalid IMSI '{}': must contain only digits", normalized);
    }

    Ok(normalized)
}

/// Loads and validates a UE configuration from a YAML file.
fn load_ue_config(path: &str) -> Result<UeConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read configuration file: {}", path))?;

    let config: UeConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse configuration file: {}", path))?;

    Ok(config)
}

/// Validates a UE configuration.
fn validate_ue_config(config: &UeConfig) -> Result<()> {
    // Validate HPLMN
    if config.hplmn.mcc == 0 || config.hplmn.mcc > 999 {
        bail!(
            "Invalid HPLMN MCC {}: must be between 001 and 999",
            config.hplmn.mcc
        );
    }
    if config.hplmn.mnc > 999 {
        bail!(
            "Invalid HPLMN MNC {}: must be between 00 and 999",
            config.hplmn.mnc
        );
    }

    // Validate gNB search list
    if config.gnb_search_list.is_empty() {
        bail!("At least one gNB address must be specified in gnb_search_list");
    }

    // Validate key (must not be all zeros)
    if config.key.iter().all(|&b| b == 0) {
        bail!("Subscriber key K cannot be all zeros");
    }

    Ok(())
}

/// Loads and validates a UE configuration in one step.
fn load_and_validate_ue_config(path: &str) -> Result<UeConfig> {
    let config = load_ue_config(path)?;
    validate_ue_config(&config)?;
    Ok(config)
}

/// Application state for a single UE instance
struct UeApp {
    /// UE instance ID (0-based index for multi-UE)
    instance_id: u32,
    /// Task manager for lifecycle management
    task_manager: TaskManager,
    /// Shutdown signal receiver
    #[allow(dead_code)]
    shutdown_rx: watch::Receiver<bool>,
}

impl UeApp {
    /// Creates a new UE application with the given configuration
    async fn new(config: UeConfig, instance_id: u32, disable_cmd: bool) -> Result<Self> {
        // Log configuration summary
        if let Some(ref supi) = config.supi {
            info!("UE {}: SUPI={}", instance_id, supi);
        }
        info!(
            "UE {}: HPLMN={}-{}, gNB search list: {:?}",
            instance_id, config.hplmn.mcc, config.hplmn.mnc, config.gnb_search_list
        );

        // Create TaskManager with all channels
        let (task_manager, app_rx, nas_rx, rrc_rx, rls_rx) =
            TaskManager::new(config.clone(), DEFAULT_CHANNEL_CAPACITY);

        let task_base = task_manager.task_base();
        let shutdown_rx = task_manager.shutdown_receiver();

        // Generate node name for CLI
        let node_name = format!("ue{}", instance_id);

        // Spawn all tasks
        Self::spawn_tasks(
            task_base.clone(),
            app_rx,
            nas_rx,
            rrc_rx,
            rls_rx,
            disable_cmd,
            node_name,
        );

        Ok(Self {
            instance_id,
            task_manager,
            shutdown_rx,
        })
    }

    /// Spawns all UE tasks
    fn spawn_tasks(
        task_base: UeTaskBase,
        app_rx: tokio::sync::mpsc::Receiver<TaskMessage<AppMessage>>,
        nas_rx: tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>,
        rrc_rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
        rls_rx: tokio::sync::mpsc::Receiver<TaskMessage<RlsMessage>>,
        disable_cmd: bool,
        node_name: String,
    ) {
        // Spawn App task (handles status updates and CLI commands)
        let app_task_base = task_base.clone();
        tokio::spawn(async move {
            let mut app_task = if disable_cmd {
                AppTask::new_without_cli(app_task_base)
            } else {
                let mut task = AppTask::new(app_task_base);
                // Initialize CLI server
                match task.init_cli_server(vec![node_name]).await {
                    Ok(port) => {
                        if port > 0 {
                            info!("CLI server listening on port {}", port);
                        }
                    }
                    Err(e) => {
                        warn!("Failed to initialize CLI server: {}", e);
                    }
                }
                task
            };
            app_task.run(app_rx).await;
        });
        info!("App task spawned");

        // Spawn NAS task (handles MM and SM procedures)
        let nas_task_base = task_base.clone();
        tokio::spawn(async move {
            Self::run_nas_task(nas_task_base, nas_rx).await
        });
        info!("NAS task spawned");

        // Spawn RRC task (handles RRC state machine and procedures)
        let rrc_task_base = task_base.clone();
        tokio::spawn(async move {
            Self::run_rrc_task(rrc_task_base, rrc_rx).await
        });
        info!("RRC task spawned");

        // Spawn RLS task (handles cell search and gNB communication)
        let mut rls_task = RlsTask::from_ue_config(task_base);
        tokio::spawn(async move {
            rls_task.run(rls_rx).await;
        });
        info!("RLS task spawned");
    }

    /// Runs the NAS task (handles MM and SM procedures)
    async fn run_nas_task(
        _task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>,
    ) {
        info!("NAS task started");

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NasMessage::NasNotify => {
                            info!("NAS notify received");
                        }
                        NasMessage::NasDelivery { pdu } => {
                            info!("NAS delivery: len={}", pdu.len());
                            // Process NAS PDU
                        }
                        NasMessage::RrcConnectionSetup => {
                            info!("RRC connection setup complete");
                            // Trigger registration procedure
                        }
                        NasMessage::RrcConnectionRelease => {
                            info!("RRC connection released");
                        }
                        NasMessage::RrcEstablishmentFailure => {
                            warn!("RRC establishment failure");
                        }
                        NasMessage::RadioLinkFailure => {
                            warn!("Radio link failure");
                        }
                        NasMessage::Paging { paging_tmsi } => {
                            info!("Paging received: {} TMSIs", paging_tmsi.len());
                        }
                        NasMessage::ActiveCellChanged { previous_tai } => {
                            info!("Active cell changed from TAI: {:?}", previous_tai);
                        }
                        NasMessage::RrcFallbackIndication => {
                            info!("RRC fallback indication");
                        }
                        NasMessage::UplinkDataDelivery { psi, data } => {
                            info!("Uplink data delivery: psi={}, len={}", psi, data.len());
                        }
                        NasMessage::PerformMmCycle => {
                            // Perform MM state machine cycle
                        }
                        NasMessage::NasTimerExpire { timer_id } => {
                            info!("NAS timer {} expired", timer_id);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => {
                    info!("NAS task received shutdown signal");
                    break;
                }
                None => {
                    info!("NAS task channel closed");
                    break;
                }
            }
        }

        info!("NAS task stopped");
    }

    /// Runs the RRC task (handles RRC state machine and procedures)
    async fn run_rrc_task(
        _task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
    ) {
        info!("RRC task started");

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        RrcMessage::LocalReleaseConnection { treat_barred } => {
                            info!("Local release connection, treat_barred={}", treat_barred);
                        }
                        RrcMessage::UplinkNasDelivery { pdu_id, pdu } => {
                            info!("Uplink NAS delivery: pdu_id={}, len={}", pdu_id, pdu.len());
                        }
                        RrcMessage::RrcNotify => {
                            info!("RRC notify received");
                        }
                        RrcMessage::PerformUac { access_category, access_identities } => {
                            info!("Perform UAC: category={}, identities={}", access_category, access_identities);
                        }
                        RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu } => {
                            info!("Downlink RRC: cell_id={}, channel={:?}, len={}", cell_id, channel, pdu.len());
                        }
                        RrcMessage::SignalChanged { cell_id, dbm } => {
                            info!("Signal changed: cell_id={}, dbm={}", cell_id, dbm);
                        }
                        RrcMessage::RadioLinkFailure { cause } => {
                            warn!("Radio link failure: {:?}", cause);
                        }
                        RrcMessage::TriggerCycle => {
                            // Trigger RRC state machine cycle
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => {
                    info!("RRC task received shutdown signal");
                    break;
                }
                None => {
                    info!("RRC task channel closed");
                    break;
                }
            }
        }

        info!("RRC task stopped");
    }

    /// Runs the main event loop until shutdown
    #[allow(dead_code)]
    async fn run(&mut self) -> Result<()> {
        info!("UE {} started, waiting for shutdown signal...", self.instance_id);

        // Wait for shutdown signal (Ctrl+C or SIGTERM)
        tokio::select! {
            _ = signal::ctrl_c() => {
                info!("Received Ctrl+C, initiating shutdown...");
            }
            _ = async {
                loop {
                    if *self.shutdown_rx.borrow() {
                        break;
                    }
                    self.shutdown_rx.changed().await.ok();
                }
            } => {
                info!("Received shutdown signal from task manager");
            }
        }

        Ok(())
    }

    /// Performs graceful shutdown of all tasks
    async fn shutdown(mut self) -> Result<()> {
        info!("UE {}: Initiating graceful shutdown...", self.instance_id);

        match self.task_manager.shutdown().await {
            Ok(()) => {
                info!("UE {}: All tasks shut down successfully", self.instance_id);
                Ok(())
            }
            Err(e) => {
                warn!("UE {}: Some tasks failed during shutdown: {}", self.instance_id, e);
                // Still return Ok since we're shutting down anyway
                Ok(())
            }
        }
    }
}

/// Initializes the tracing subscriber for logging.
fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(filter)
        .init();
}

#[tokio::main]
async fn main() -> ExitCode {
    // Initialize logging
    init_logging();

    // Parse command line arguments
    let args = Args::parse();

    // Print banner
    println!("nextgsim UE - 5G User Equipment Simulator");
    println!("==========================================");

    // Validate and convert arguments
    let options = match UeOptions::try_from(args) {
        Ok(opts) => opts,
        Err(e) => {
            error!("Invalid arguments: {:#}", e);
            return ExitCode::FAILURE;
        }
    };

    // Run the UE application
    match run_ue(options).await {
        Ok(()) => {
            info!("UE exited successfully");
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!("UE failed: {:#}", e);
            ExitCode::FAILURE
        }
    }
}

/// Main UE execution logic
async fn run_ue(options: UeOptions) -> Result<()> {
    // Load and validate configuration
    info!("Loading configuration from: {}", options.config_file);
    let mut base_config = load_and_validate_ue_config(&options.config_file)?;

    // Apply IMSI override if specified
    if let Some(ref imsi) = options.imsi {
        info!("Overriding IMSI with: {}", imsi);
        base_config.supi = Some(nextgsim_common::Supi::imsi(imsi.clone()));
    }

    // Log configuration summary
    if let Some(ref supi) = base_config.supi {
        info!("Base SUPI: {}", supi);
    }
    info!(
        "HPLMN: {}-{}",
        base_config.hplmn.mcc, base_config.hplmn.mnc
    );
    info!("gNB search list: {:?}", base_config.gnb_search_list);
    info!("Number of UEs: {}", options.num_ue);
    if options.tempo > 0 {
        info!("UE start tempo: {}ms", options.tempo);
    }
    if options.disable_cmd {
        info!("CLI command interface: disabled");
    }
    if options.no_routing_config {
        info!("TUN routing configuration: disabled");
    }

    // Create and run UE instances
    let mut ue_apps: Vec<UeApp> = Vec::new();

    for i in 0..options.num_ue {
        // Clone config for each UE instance
        let mut config = base_config.clone();

        // Increment IMSI for multi-UE scenarios
        if options.num_ue > 1 {
            if let Some(ref supi) = base_config.supi {
                let new_imsi = increment_imsi(&supi.to_string(), i as u64)?;
                config.supi = Some(nextgsim_common::Supi::imsi(new_imsi));
            }
        }

        // Apply tempo delay between UE starts
        if i > 0 && options.tempo > 0 {
            info!("Waiting {}ms before starting UE {}...", options.tempo, i);
            tokio::time::sleep(Duration::from_millis(options.tempo)).await;
        }

        // Create UE instance
        info!("Starting UE instance {}...", i);
        let ue_app = UeApp::new(config, i, options.disable_cmd).await?;
        ue_apps.push(ue_app);
    }

    info!("All {} UE instance(s) started", options.num_ue);

    // Wait for shutdown signal
    signal::ctrl_c().await?;
    info!("Received Ctrl+C, shutting down all UEs...");

    // Shutdown all UE instances
    for ue_app in ue_apps {
        ue_app.shutdown().await?;
    }

    Ok(())
}

/// Increments an IMSI string by a given offset.
/// IMSI is a 15-digit string, and we increment the numeric value.
fn increment_imsi(imsi: &str, offset: u64) -> Result<String> {
    // Remove "imsi-" prefix if present
    let digits = if imsi.starts_with("imsi-") {
        &imsi[5..]
    } else {
        imsi
    };

    // Parse as u64 and increment
    let value: u64 = digits.parse()
        .with_context(|| format!("Invalid IMSI format: {}", imsi))?;
    let new_value = value + offset;

    // Format back to 15 digits
    Ok(format!("{:015}", new_value))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    #[test]
    fn test_cli_parsing() {
        Args::command().debug_assert();
    }

    #[test]
    fn test_config_file_required() {
        let args = Args::parse_from(["nr-ue", "-c", "config/ue.yaml"]);
        assert_eq!(args.config_file, "config/ue.yaml");
    }

    #[test]
    fn test_imsi_override() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-i", "001010000000001"]);
        assert_eq!(args.imsi, Some("001010000000001".to_string()));
    }

    #[test]
    fn test_imsi_with_prefix() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-i", "imsi-001010000000001"]);
        assert_eq!(args.imsi, Some("imsi-001010000000001".to_string()));
    }

    #[test]
    fn test_num_ue() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-n", "10"]);
        assert_eq!(args.num_ue, Some(10));
    }

    #[test]
    fn test_tempo() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-t", "100"]);
        assert_eq!(args.tempo, Some(100));
    }

    #[test]
    fn test_disable_cmd() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-l"]);
        assert!(args.disable_cmd);
    }

    #[test]
    fn test_no_routing_config() {
        let args = Args::parse_from(["nr-ue", "-c", "config.yaml", "-r"]);
        assert!(args.no_routing_config);
    }

    #[test]
    fn test_all_options() {
        let args = Args::parse_from([
            "nr-ue",
            "-c", "config.yaml",
            "-i", "001010000000001",
            "-n", "5",
            "-t", "200",
            "-l",
            "-r",
        ]);
        assert_eq!(args.config_file, "config.yaml");
        assert_eq!(args.imsi, Some("001010000000001".to_string()));
        assert_eq!(args.num_ue, Some(5));
        assert_eq!(args.tempo, Some(200));
        assert!(args.disable_cmd);
        assert!(args.no_routing_config);
    }

    #[test]
    fn test_normalize_imsi_plain() {
        let result = normalize_imsi("001010000000001").unwrap();
        assert_eq!(result, "001010000000001");
    }

    #[test]
    fn test_normalize_imsi_with_prefix() {
        let result = normalize_imsi("imsi-001010000000001").unwrap();
        assert_eq!(result, "001010000000001");
    }

    #[test]
    fn test_normalize_imsi_invalid_length() {
        let result = normalize_imsi("12345");
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_imsi_non_digits() {
        let result = normalize_imsi("00101000000000a");
        assert!(result.is_err());
    }

    #[test]
    fn test_ue_options_from_args_defaults() {
        let args = Args {
            config_file: "config.yaml".to_string(),
            imsi: None,
            num_ue: None,
            tempo: None,
            disable_cmd: false,
            no_routing_config: false,
        };
        let options = UeOptions::try_from(args).unwrap();
        assert_eq!(options.config_file, "config.yaml");
        assert_eq!(options.imsi, None);
        assert_eq!(options.num_ue, 1);
        assert_eq!(options.tempo, 0);
        assert!(!options.disable_cmd);
        assert!(!options.no_routing_config);
    }

    #[test]
    fn test_ue_options_num_ue_zero() {
        let args = Args {
            config_file: "config.yaml".to_string(),
            imsi: None,
            num_ue: Some(0),
            tempo: None,
            disable_cmd: false,
            no_routing_config: false,
        };
        let result = UeOptions::try_from(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_ue_options_num_ue_too_large() {
        let args = Args {
            config_file: "config.yaml".to_string(),
            imsi: None,
            num_ue: Some(513),
            tempo: None,
            disable_cmd: false,
            no_routing_config: false,
        };
        let result = UeOptions::try_from(args);
        assert!(result.is_err());
    }

    #[test]
    fn test_ue_options_valid_num_ue() {
        let args = Args {
            config_file: "config.yaml".to_string(),
            imsi: None,
            num_ue: Some(512),
            tempo: None,
            disable_cmd: false,
            no_routing_config: false,
        };
        let options = UeOptions::try_from(args).unwrap();
        assert_eq!(options.num_ue, 512);
    }

    #[test]
    fn test_increment_imsi() {
        let result = increment_imsi("001010000000001", 0).unwrap();
        assert_eq!(result, "001010000000001");

        let result = increment_imsi("001010000000001", 1).unwrap();
        assert_eq!(result, "001010000000002");

        let result = increment_imsi("001010000000001", 10).unwrap();
        assert_eq!(result, "001010000000011");
    }

    #[test]
    fn test_increment_imsi_with_prefix() {
        let result = increment_imsi("imsi-001010000000001", 5).unwrap();
        assert_eq!(result, "001010000000006");
    }
}
