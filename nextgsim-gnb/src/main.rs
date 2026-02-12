//! nextgsim gNB (gNodeB) simulator
//!
//! This is the main binary for the 5G gNB simulator. It implements:
//! - CLI argument parsing
//! - Configuration loading and validation
//! - Task spawning and lifecycle management
//! - Graceful shutdown handling
//!
//! # Usage
//!
//! ```bash
//! nr-gnb -c config/gnb.yaml
//! ```
//!
//! # Reference
//!
//! Based on UERANSIM's `src/gnb.cpp` implementation.

use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use tokio::signal;
use tokio::sync::watch;
use tracing::{error, info, warn};

use nextgsim_gnb::{
    load_and_validate_gnb_config, AppTask, GtpTask, NgapTask, RlsTask, RrcTask, SctpTask,
    SctpMessage, Task, TaskError, TaskManager, TaskMessage,
    DEFAULT_CHANNEL_CAPACITY, NGAP_PPID,
};

/// nextgsim gNB - 5G gNodeB Simulator
#[derive(Parser, Debug)]
#[command(name = "nr-gnb")]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Path to the gNB configuration file (YAML)
    #[arg(short = 'c', long = "config", value_name = "FILE")]
    config_file: String,

    /// Disable CLI command interface
    #[arg(short = 'l', long = "disable-cmd")]
    disable_cmd: bool,
}

/// Application state for the gNB
struct GnbApp {
    /// Task manager for lifecycle management
    task_manager: TaskManager,
    /// Shutdown signal receiver
    shutdown_rx: watch::Receiver<bool>,
}

impl GnbApp {
    /// Creates a new gNB application with the given configuration file
    async fn new(config_path: &str, disable_cmd: bool) -> Result<Self> {
        // Load and validate configuration
        info!("Loading configuration from: {}", config_path);
        let config = load_and_validate_gnb_config(config_path)
            .with_context(|| format!("Failed to load configuration from {config_path}"))?;

        info!(
            "Configuration loaded: NCI={:#x}, PLMN={}-{}, TAC={}",
            config.nci, config.plmn.mcc, config.plmn.mnc, config.tac
        );
        info!(
            "Network interfaces: link={}, ngap={}, gtp={}",
            config.link_ip, config.ngap_ip, config.gtp_ip
        );
        info!("AMF configurations: {} AMF(s) configured", config.amf_configs.len());

        // Create TaskManager with all channels
        let (task_manager, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx) =
            TaskManager::new(config.clone(), DEFAULT_CHANNEL_CAPACITY);

        let task_base = task_manager.task_base();
        let shutdown_rx = task_manager.shutdown_receiver();

        // Spawn all tasks
        Self::spawn_tasks(
            &task_manager,
            task_base.clone(),
            app_rx,
            ngap_rx,
            rrc_rx,
            gtp_rx,
            rls_rx,
            sctp_rx,
            disable_cmd,
        );

        Ok(Self {
            task_manager,
            shutdown_rx,
        })
    }

    /// Spawns all gNB tasks
    #[allow(clippy::too_many_arguments)]
    fn spawn_tasks(
        _task_manager: &TaskManager,
        task_base: nextgsim_gnb::GnbTaskBase,
        app_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::AppMessage>>,
        ngap_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::NgapMessage>>,
        rrc_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::RrcMessage>>,
        gtp_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::GtpMessage>>,
        rls_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::RlsMessage>>,
        sctp_rx: tokio::sync::mpsc::Receiver<TaskMessage<nextgsim_gnb::SctpMessage>>,
        disable_cmd: bool,
    ) {
        // Spawn App task (handles status updates and CLI commands)
        let app_task_base = task_base.clone();
        tokio::spawn(async move {
            let mut app_task = if disable_cmd {
                AppTask::new_without_cli(app_task_base)
            } else {
                let mut task = AppTask::new(app_task_base);
                // Initialize CLI server
                match task.init_cli_server("gnb".to_string()).await {
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
            Ok::<(), TaskError>(())
        });
        info!("App task spawned");

        // Spawn NGAP task
        let mut ngap_task = NgapTask::new(task_base.clone());
        tokio::spawn(async move {
            ngap_task.run(ngap_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("NGAP task spawned");

        // Spawn RRC task
        let mut rrc_task = RrcTask::new(task_base.clone());
        tokio::spawn(async move {
            rrc_task.run(rrc_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("RRC task spawned");

        // Spawn GTP task
        let mut gtp_task = GtpTask::new(task_base.clone());
        tokio::spawn(async move {
            gtp_task.run(gtp_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("GTP task spawned");

        // Spawn RLS task
        let mut rls_task = RlsTask::new(task_base.clone());
        tokio::spawn(async move {
            rls_task.run(rls_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("RLS task spawned");

        // Spawn SCTP task
        let mut sctp_task = SctpTask::new(task_base.clone());
        tokio::spawn(async move {
            sctp_task.run(sctp_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("SCTP task spawned");
    }

    /// Initiates connections to all configured AMFs
    async fn connect_to_amfs(&self) -> Result<()> {
        let task_base = self.task_manager.task_base();
        let config = &task_base.config;

        for (i, amf_config) in config.amf_configs.iter().enumerate() {
            let client_id = i as i32;
            info!(
                "Initiating connection to AMF {}: {}:{}",
                client_id, amf_config.address, amf_config.port
            );

            let msg = SctpMessage::ConnectionRequest {
                client_id,
                local_address: config.ngap_ip.to_string(),
                local_port: 0, // Let OS assign port
                remote_address: amf_config.address.to_string(),
                remote_port: amf_config.port,
                ppid: NGAP_PPID,
            };

            if let Err(e) = task_base.sctp_tx.send(msg).await {
                error!("Failed to send connection request for AMF {}: {}", client_id, e);
            }
        }

        Ok(())
    }

    /// Runs the main event loop until shutdown
    async fn run(&mut self) -> Result<()> {
        info!("gNB started, waiting for shutdown signal...");

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
        info!("Initiating graceful shutdown...");

        match self.task_manager.shutdown().await {
            Ok(()) => {
                info!("All tasks shut down successfully");
                Ok(())
            }
            Err(e) => {
                warn!("Some tasks failed during shutdown: {}", e);
                // Still return Ok since we're shutting down anyway
                Ok(())
            }
        }
    }
}

/// Initializes the tracing subscriber for logging
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
    println!("nextgsim gNB - 5G gNodeB Simulator");
    println!("==================================");

    // Create and run the gNB application
    match run_gnb(args).await {
        Ok(()) => {
            info!("gNB exited successfully");
            ExitCode::SUCCESS
        }
        Err(e) => {
            error!("gNB failed: {:#}", e);
            ExitCode::FAILURE
        }
    }
}

/// Main gNB execution logic
async fn run_gnb(args: Args) -> Result<()> {
    // Create the gNB application
    let mut app = GnbApp::new(&args.config_file, args.disable_cmd).await?;

    // Connect to configured AMFs
    app.connect_to_amfs().await?;

    // Run the main event loop
    app.run().await?;

    // Perform graceful shutdown
    app.shutdown().await?;

    Ok(())
}
