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
use tracing::{debug, error, info, warn};

use std::net::Ipv4Addr;

use nextgsim_common::config::UeConfig;
use nextgsim_rls::RrcChannel;
use nextgsim_ue::tun::{TunAppMessage, TunMessage, TunTask, TunTaskConfig};
#[cfg(feature = "nextgsim-fl")]
use nextgsim_ue::FlParticipantTask;
#[cfg(feature = "nextgsim-isac")]
use nextgsim_ue::IsacSensorTask;
#[cfg(feature = "nextgsim-nwdaf")]
use nextgsim_ue::NwdafReporterTask;
#[cfg(feature = "nextgsim-semantic")]
use nextgsim_ue::SemanticCodecTask;
#[cfg(feature = "nextgsim-she")]
use nextgsim_ue::SheClientTask;
use nextgsim_ue::{
    AppMessage, AppTask, MintMessage, MintTask, NasMessage, RangingTask, RlsMessage, RlsTask,
    RrcMessage, SidelinkMessage, SidelinkTask, Task, TaskManager, TaskMessage, UeRel18Receivers,
    UeTaskBase, DEFAULT_CHANNEL_CAPACITY,
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
    let normalized = if imsi.len() > 5 && imsi.starts_with("imsi-") {
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
        bail!("Invalid IMSI '{normalized}': must contain only digits");
    }

    Ok(normalized)
}

/// Loads and validates a UE configuration from a YAML file.
fn load_ue_config(path: &str) -> Result<UeConfig> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read configuration file: {path}"))?;

    let config: UeConfig = serde_yaml::from_str(&contents)
        .with_context(|| format!("Failed to parse configuration file: {path}"))?;

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

    // Validate the SUCI protection configuration (scheme, routing indicator,
    // home network public key id and key material) by performing a trial
    // build, so misconfiguration fails fast instead of inside the NAS task.
    if let Err(e) = nextgsim_ue::nas::mm::build_suci(config) {
        bail!("Invalid SUCI configuration: {e}");
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
        let (mut task_manager, app_rx, nas_rx, rrc_rx, rls_rx) =
            TaskManager::new(config.clone(), DEFAULT_CHANNEL_CAPACITY);

        // Install the Rel-18 task handles (Ranging, MINT, Sidelink) before any
        // task base clone so every task can reach the Rel-18 actors.
        let rel18_rxs = task_manager.init_rel18_tasks(DEFAULT_CHANNEL_CAPACITY);

        let task_base = task_manager.task_base();
        let shutdown_rx = task_manager.shutdown_receiver();

        // Generate node name for CLI
        let node_name = format!("ue{instance_id}");

        // Spawn all tasks
        Self::spawn_tasks(
            task_base.clone(),
            app_rx,
            nas_rx,
            rrc_rx,
            rls_rx,
            rel18_rxs,
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
    #[allow(unused_mut)]
    #[allow(clippy::too_many_arguments)]
    fn spawn_tasks(
        mut task_base: UeTaskBase,
        app_rx: tokio::sync::mpsc::Receiver<TaskMessage<AppMessage>>,
        nas_rx: tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>,
        rrc_rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
        rls_rx: tokio::sync::mpsc::Receiver<TaskMessage<RlsMessage>>,
        rel18_rxs: UeRel18Receivers,
        disable_cmd: bool,
        node_name: String,
    ) {
        // Create TUN task channel
        let (tun_tx, tun_rx) = tokio::sync::mpsc::channel::<TunMessage>(64);
        let (tun_app_tx, mut tun_app_rx) = tokio::sync::mpsc::channel::<TunAppMessage>(64);

        // Spawn TUN task (handles TUN interface for user plane data)
        let tun_config = TunTaskConfig::default();
        tokio::spawn(async move {
            let mut tun_task = TunTask::new(tun_config, tun_app_tx);
            tun_task.run(tun_rx).await;
        });
        info!("TUN task spawned");

        // Spawn TUN app message handler (logs TUN events)
        let rls_tx_for_tun = task_base.rls_tx.clone();
        let nas_tx_for_tun = task_base.nas_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = tun_app_rx.recv().await {
                match msg {
                    TunAppMessage::InterfaceCreated { psi, name } => {
                        info!("TUN interface created: {} for PSI {}", name, psi);
                    }
                    TunAppMessage::InterfaceDestroyed { psi } => {
                        info!("TUN interface destroyed for PSI {}", psi);
                    }
                    TunAppMessage::UplinkData { psi, data } => {
                        info!("TUN uplink data: PSI={}, len={}", psi, data.len());
                        // Trigger Service Request if UE is in IDLE state
                        // (NAS task will check actual state and only send if needed)
                        let _ = nas_tx_for_tun
                            .send(NasMessage::InitiateServiceRequest)
                            .await;
                        // Forward uplink data to RLS for transmission to gNB
                        let _ = rls_tx_for_tun
                            .send(RlsMessage::DataPduDelivery { psi, pdu: data })
                            .await;
                    }
                    TunAppMessage::Error { message } => {
                        warn!("TUN error: {}", message);
                    }
                }
            }
        });

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
        tokio::spawn(async move { Self::run_nas_task(nas_task_base, nas_rx, tun_tx).await });
        info!("NAS task spawned");

        // Spawn RRC task (handles RRC state machine and procedures)
        let rrc_task_base = task_base.clone();
        tokio::spawn(async move { Self::run_rrc_task(rrc_task_base, rrc_rx).await });
        info!("RRC task spawned");

        // Spawn Rel-18 5G-Advanced tasks (Ranging TS 23.586, MINT TS 23.761,
        // Sidelink). The handles were installed by the TaskManager before the
        // task base was cloned, so all tasks can reach these actors.
        let UeRel18Receivers {
            ranging_rx,
            mint_rx,
            sidelink_rx,
        } = rel18_rxs;
        let mut ranging_task = RangingTask::new(task_base.clone());
        tokio::spawn(async move { ranging_task.run(ranging_rx).await });
        info!("Ranging task spawned (Rel-18, TS 23.586)");
        let mut mint_task = MintTask::new(task_base.clone());
        tokio::spawn(async move { mint_task.run(mint_rx).await });
        info!("MINT task spawned (Rel-18, TS 23.761)");
        let mut sidelink_task = SidelinkTask::new(task_base.clone());
        tokio::spawn(async move { sidelink_task.run(sidelink_rx).await });
        info!("Sidelink task spawned (Rel-18 NR Sidelink)");

        // Spawn 6G AI-native network function tasks (Rel-20)
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        {
            let any_6g = false
                || cfg!(feature = "nextgsim-she") && task_base.config.she_enabled
                || cfg!(feature = "nextgsim-nwdaf") && task_base.config.ai_ml_enabled
                || cfg!(feature = "nextgsim-isac") && task_base.config.isac_enabled
                || cfg!(feature = "nextgsim-fl") && task_base.config.federated_learning_enabled
                || cfg!(feature = "nextgsim-semantic") && task_base.config.semantic_comm_enabled;

            if any_6g {
                let sixg_rxs = task_base.init_6g_tasks(DEFAULT_CHANNEL_CAPACITY);

                #[cfg(feature = "nextgsim-she")]
                if task_base.config.she_enabled {
                    let mut she_task = SheClientTask::new(task_base.clone());
                    tokio::spawn(async move {
                        she_task.run(sixg_rxs.she_client_rx).await;
                    });
                    info!("SHE Client task spawned");
                }

                #[cfg(feature = "nextgsim-nwdaf")]
                if task_base.config.ai_ml_enabled {
                    let mut nwdaf_task = NwdafReporterTask::new(task_base.clone());
                    tokio::spawn(async move {
                        nwdaf_task.run(sixg_rxs.nwdaf_reporter_rx).await;
                    });
                    info!("NWDAF Reporter task spawned");
                }

                #[cfg(feature = "nextgsim-isac")]
                if task_base.config.isac_enabled {
                    let mut isac_task = IsacSensorTask::new(task_base.clone());
                    tokio::spawn(async move {
                        isac_task.run(sixg_rxs.isac_sensor_rx).await;
                    });
                    info!("ISAC Sensor task spawned");
                }

                #[cfg(feature = "nextgsim-fl")]
                if task_base.config.federated_learning_enabled {
                    let mut fl_task = FlParticipantTask::new(task_base.clone());
                    tokio::spawn(async move {
                        fl_task.run(sixg_rxs.fl_participant_rx).await;
                    });
                    info!("FL Participant task spawned");
                }

                #[cfg(feature = "nextgsim-semantic")]
                if task_base.config.semantic_comm_enabled {
                    let mut semantic_task = SemanticCodecTask::new(task_base.clone());
                    tokio::spawn(async move {
                        semantic_task.run(sixg_rxs.semantic_codec_rx).await;
                    });
                    info!("Semantic Codec task spawned");
                }
            }
        }

        // Spawn RLS task (handles cell search and gNB communication)
        let mut rls_task = RlsTask::from_ue_config(task_base);
        tokio::spawn(async move {
            rls_task.run(rls_rx).await;
        });
        info!("RLS task spawned");
    }

    /// Runs the NAS task (handles MM and SM procedures)
    async fn run_nas_task(
        task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>,
        tun_tx: tokio::sync::mpsc::Sender<TunMessage>,
    ) {
        use nextgsim_nas::ies::ie1::ServiceType;
        use nextgsim_nas::ies::RegistrationType;
        use nextgsim_ue::nas::mm::{CmState, MmOrchestrator, MmUeIdentity};
        use nextgsim_ue::nas::sm::{SmOrchestrator, SmSessionParams};
        use nextgsim_ue::rrc::cell_selection::{Plmn, PlmnSelector};

        info!("NAS task started");

        // MM procedure orchestrator: owns the MM state machine, the NAS
        // timers and the NAS security context (TS 24.501 Section 5)
        let suci = match build_suci_from_config(&task_base.config) {
            Ok(suci) => suci,
            Err(e) => {
                // No fallback to the null scheme: a UE configured for ECIES
                // must not reveal its SUPI (TS 33.501 Section 6.12)
                error!("Cannot build SUCI from configuration: {e}; NAS task aborting");
                return;
            }
        };
        let identity = MmUeIdentity::from_config(&task_base.config, suci);
        let mut orch = MmOrchestrator::new(identity);

        // SM procedure orchestrator: owns the per-PSI session state, PSI /
        // PTI allocation and the SM timers (TS 24.501 Section 6). The UE only
        // accepts the spec-conformant PDU Session Establishment Accept wire
        // format (TS 24.501 Table 8.3.2.1.1); there is no legacy compat path.
        let mut sm_orch = SmOrchestrator::new();
        let sm_session_params = SmSessionParams::from_config(&task_base.config);

        // MINT secondary-subscription driver (Rel-18, TS 23.761). Inert unless
        // the UE config enables MINT with secondary SUPIs; it owns each
        // secondary subscription's MM/security context and SM sessions. Only
        // the primary subscription's sessions (subscription_index 0) are
        // established by the main `sm_orch`; secondary-SUPI sessions are routed
        // here per the DNN→subscription map.
        let mut mint_secondary =
            nextgsim_ue::nas::mm::MintSecondary::from_config(&task_base.config);
        if mint_secondary.is_active() {
            info!(
                "MINT enabled: {} secondary subscription(s), disaster_roaming={}",
                "configured",
                mint_secondary.disaster_roaming()
            );
        }

        // TS 23.122 PLMN selector (automatic mode), consuming the MM
        // orchestrator's forbidden-PLMN list
        let home_plmn = Plmn::new(
            task_base.config.hplmn.mcc,
            task_base.config.hplmn.mnc,
            task_base.config.hplmn.long_mnc,
        );
        let mut plmn_selector = PlmnSelector::new(home_plmn);
        // TS 23.122 §3.1: the EHPLMN list is read from the USIM (EF_EHPLMN). In
        // this simulation the USIM provides the configured HPLMN; when no
        // explicit EF_EHPLMN list is configured the HPLMN derived from the IMSI
        // is the (single) equivalent-HPLMN entry, so seed the selector with it.
        plmn_selector.set_ehplmn_list(vec![home_plmn]);

        let mut pdu_counter: u32 = 0;

        // UAV tracking-report cadence (Rel-18, TS 23.256). `uav_report_ticks`
        // counts NAS timer ticks once the aerial UE is registered; the report
        // position is the configured flight altitude at a fixed simulated
        // coordinate so the AMF geofence is exercised end to end.
        let mut uav_report_ticks: u64 = 0;
        let uav_report_position: (f64, f64, f64) = task_base
            .config
            .uav_config
            .as_ref()
            .map(|u| (37.7749, -122.4194, u.max_altitude_meters))
            .unwrap_or((0.0, 0.0, 0.0));

        // 1-second tick driving the NAS timers (T3510/T3511/T3502/T3512/
        // ..., T3580/T3581/T3582, SM back-off and the higher-priority PLMN
        // periodic search)
        let mut timer_tick = tokio::time::interval(Duration::from_secs(1));
        timer_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
            _ = timer_tick.tick() => {
                let outs = orch.tick();
                process_mm_outputs(
                    outs,
                    &mut orch,
                    &mut sm_orch,
                    &sm_session_params,
                    &mut plmn_selector,
                    &task_base,
                    &tun_tx,
                    &mut pdu_counter,
                )
                .await;
                let sm_outs = sm_orch.tick();
                process_sm_outputs(sm_outs, &mut orch, &task_base, &tun_tx, &mut pdu_counter)
                    .await;

                // MINT secondary subscriptions (Rel-18, TS 23.761): start the
                // secondary registrations once the primary is registered, then
                // drive their MM/SM timers each tick.
                if mint_secondary.is_active() && orch.state().is_registered() {
                    let started = mint_secondary.start_secondary_registrations();
                    for (index, outs) in started {
                        process_secondary_mm_outputs(
                            index, outs, &mut mint_secondary, &task_base, &tun_tx,
                            &mut pdu_counter,
                        )
                        .await;
                    }
                    let ticks = mint_secondary.tick();
                    for (index, outs) in ticks {
                        process_secondary_mm_outputs(
                            index, outs, &mut mint_secondary, &task_base, &tun_tx,
                            &mut pdu_counter,
                        )
                        .await;
                    }
                }

                // UAV tracking report (Rel-18, TS 23.256): once the aerial UE
                // is registered, periodically report its position so the AMF
                // can run the geofence and enforce flight authorization. The
                // configured position is reported as a Remote ID / flight
                // position; the gNB relays it to the AMF as Uplink NAS Transport.
                if let Some(uav) = task_base.config.uav_config.as_ref().filter(|u| u.is_aerial_ue) {
                    if orch.state().is_registered() {
                        uav_report_ticks += 1;
                        // Report at ~5 s cadence after registration.
                        if uav_report_ticks.is_multiple_of(5) {
                            let caa_id = uav.uav_id.clone().unwrap_or_default();
                            // Default flight position: the configured max
                            // altitude at a fixed simulated coordinate, so the
                            // report exercises the AMF geofence end to end.
                            let (lat, lon, alt) = uav_report_position;
                            if let Some(pdu) =
                                orch.build_uav_tracking_report(&caa_id, lat, lon, alt, 1)
                            {
                                pdu_counter += 1;
                                let _ = task_base
                                    .rrc_tx
                                    .send(RrcMessage::UplinkNasDelivery {
                                        pdu_id: pdu_counter,
                                        pdu: pdu.into(),
                                    })
                                    .await;
                            }
                        }
                    }
                }

                // Higher-priority PLMN periodic search (TS 23.122 4.4.3.3)
                if plmn_selector.tick() {
                    perform_plmn_selection(
                        &mut orch,
                        &mut plmn_selector,
                        home_plmn,
                        &task_base,
                        &mut pdu_counter,
                    )
                    .await;
                }
            }
            recv = rx.recv() => match recv {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NasMessage::NasNotify => {
                            info!("NAS notify received");
                        }
                        NasMessage::NasDelivery { pdu } => {
                            info!("NAS delivery: len={}", pdu.len());

                            // All downlink NAS arrives as 5GMM (EPD 0x7E):
                            // 5GSM messages are carried inside DL NAS
                            // Transport and dispatched to the SM
                            // orchestrator via MmOutput::NotHandled. The
                            // legacy raw-5GSM-over-RRC path was removed:
                            // a strict peer always wraps N1 SM containers
                            // (TS 24.501 Section 5.4.5).
                            if pdu.len() >= 4 && pdu.data()[0] == 0x7E {
                                // MINT (Rel-18, TS 23.761): on the shared radio
                                // connection, route the downlink to a secondary
                                // subscription's MM context when it owns the
                                // in-flight procedure; otherwise to the primary.
                                if let Some(index) = mint_secondary.owns_pending_downlink() {
                                    process_secondary_downlink(
                                        index,
                                        pdu.data(),
                                        &mut mint_secondary,
                                        &task_base,
                                        &tun_tx,
                                        &mut pdu_counter,
                                    )
                                    .await;
                                } else {
                                    let outs = orch.handle_downlink(pdu.data());
                                    process_mm_outputs(
                                        outs,
                                        &mut orch,
                                        &mut sm_orch,
                                        &sm_session_params,
                                        &mut plmn_selector,
                                        &task_base,
                                        &tun_tx,
                                        &mut pdu_counter,
                                    )
                                    .await;
                                }
                            } else if pdu.len() >= 4 {
                                warn!(
                                    "Discarding NAS PDU with EPD 0x{:02x}: raw 5GSM over RRC is not accepted",
                                    pdu.data()[0]
                                );
                            } else {
                                warn!("NAS PDU too short: {} bytes", pdu.len());
                            }
                        }
                        NasMessage::RrcConnectionSetup => {
                            info!("RRC connection setup complete");
                            orch.state_mut().switch_cm_state(CmState::Connected);
                        }
                        NasMessage::RrcConnectionRelease => {
                            info!("RRC connection released");
                            orch.state_mut().switch_cm_state(CmState::Idle);
                        }
                        NasMessage::RrcEstablishmentFailure => {
                            warn!("RRC establishment failure");
                        }
                        NasMessage::RadioLinkFailure => {
                            warn!("Radio link failure");
                            orch.state_mut().switch_cm_state(CmState::Idle);
                        }
                        NasMessage::Paging { paging_tmsi } => {
                            info!("Paging received: {} TMSIs", paging_tmsi.len());

                            // If registered and CM-IDLE, run the service request
                            // procedure (the 5G-S-TMSI is derived from the
                            // stored 5G-GUTI by the orchestrator)
                            if orch.state().is_registered() && orch.state().is_idle() {
                                let outs = orch.start_service_request(
                                    ServiceType::MobileTerminatedServices,
                                    None,
                                );
                                process_mm_outputs(
                                    outs,
                                    &mut orch,
                                    &mut sm_orch,
                                    &sm_session_params,
                                    &mut plmn_selector,
                                    &task_base,
                                    &tun_tx,
                                    &mut pdu_counter,
                                )
                                .await;
                            }
                        }
                        NasMessage::ActiveCellChanged { previous_tai } => {
                            info!("Active cell changed from TAI: {:?}", previous_tai);
                            // Mobility registration update trigger
                            // (TS 24.501 Section 5.5.1.3.2)
                            if orch.state().is_registered() {
                                let outs = orch.start_registration(
                                    RegistrationType::MobilityRegistrationUpdating,
                                );
                                process_mm_outputs(
                                    outs,
                                    &mut orch,
                                    &mut sm_orch,
                                    &sm_session_params,
                                    &mut plmn_selector,
                                    &task_base,
                                    &tun_tx,
                                    &mut pdu_counter,
                                )
                                .await;
                            }
                        }
                        NasMessage::RrcFallbackIndication => {
                            info!("RRC fallback indication");
                        }
                        NasMessage::UplinkDataDelivery { psi, data } => {
                            // This is actually downlink data (from network to UE)
                            // Write it to the TUN interface
                            debug!("Downlink data received: psi={}, len={}", psi, data.len());
                            let _ = tun_tx.send(TunMessage::WriteData { psi, data }).await;
                        }
                        NasMessage::InitiateServiceRequest => {
                            // Data-triggered Service Request: UE has data to send while in IDLE
                            if orch.state().is_registered() && orch.state().is_idle() {
                                info!("Initiating Service Request (data-triggered, IDLE -> CONNECTED)");
                                let outs =
                                    orch.start_service_request(ServiceType::Data, Some(0x2000));
                                process_mm_outputs(
                                    outs,
                                    &mut orch,
                                    &mut sm_orch,
                                    &sm_session_params,
                                    &mut plmn_selector,
                                    &task_base,
                                    &tun_tx,
                                    &mut pdu_counter,
                                )
                                .await;
                            } else if orch.state().is_registered() && orch.state().is_connected() {
                                debug!("Service Request not needed: already in CM-CONNECTED");
                            } else {
                                warn!(
                                    "Cannot send Service Request: not registered (RM={}, CM={})",
                                    orch.state().rm_state(),
                                    orch.state().cm_state()
                                );
                            }
                        }
                        NasMessage::PerformMmCycle => {
                            info!("PerformMmCycle received, MM state: {}", orch.state());

                            // If we're deregistered, start the initial registration
                            if orch.state().is_deregistered() {
                                // Wait a bit for gNB to complete NG Setup with AMF
                                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                                let outs = orch
                                    .start_registration(RegistrationType::InitialRegistration);
                                process_mm_outputs(
                                    outs,
                                    &mut orch,
                                    &mut sm_orch,
                                    &sm_session_params,
                                    &mut plmn_selector,
                                    &task_base,
                                    &tun_tx,
                                    &mut pdu_counter,
                                )
                                .await;
                            }
                        }
                        NasMessage::NasTimerExpire { timer_id } => {
                            info!("NAS timer {} expired", timer_id);
                        }
                        NasMessage::InitiatePduSessionEstablishment {
                            psi: _,
                            pti: _,
                            session_type,
                            apn,
                        } => {
                            // PSI and PTI are allocated by the SM
                            // orchestrator (TS 24.501 6.1.3.2 / 9.6); the
                            // CLI-provided hints are ignored
                            use nextgsim_nas::messages::sm::{PduSessionTypeValue, SscModeValue};
                            let requested_type = match session_type.as_str() {
                                "IPv6" => PduSessionTypeValue::Ipv6,
                                "IPv4v6" => PduSessionTypeValue::Ipv4v6,
                                "Unstructured" => PduSessionTypeValue::Unstructured,
                                "Ethernet" => PduSessionTypeValue::Ethernet,
                                _ => PduSessionTypeValue::Ipv4,
                            };
                            let dnn = apn.clone().or_else(|| {
                                sm_session_params.first().and_then(|p| p.dnn.clone())
                            });
                            // An XR DNN (xr/xr-split/xr-haptic) implies a
                            // requested XR 5QI so the SM orchestrator marks the
                            // session as XR (mirrors the SMF DNN→5QI mapping).
                            let requested_5qi = dnn.as_deref().and_then(|d| {
                                match d.to_ascii_lowercase().as_str() {
                                    "xr" | "xr-cloud" | "xr-gaming" => Some(82),
                                    "xr-split" => Some(84),
                                    "xr-haptic" => Some(85),
                                    _ => None,
                                }
                            });
                            // MINT (Rel-18, TS 23.761): a CLI-triggered session
                            // also routes to the subscription mapped to its DNN.
                            let subscription_index = task_base
                                .config
                                .mint_config
                                .as_ref()
                                .map_or(0, |m| m.subscription_for_dnn(dnn.as_deref()));
                            let params = SmSessionParams {
                                session_type: requested_type,
                                ssc_mode: SscModeValue::SscMode1,
                                dnn,
                                s_nssai: sm_session_params
                                    .first()
                                    .and_then(|p| p.s_nssai.clone()),
                                emergency: false,
                                requested_5qi,
                                subscription_index,
                            };
                            info!(
                                "Initiating PDU session establishment: type={}, dnn={:?}",
                                session_type, params.dnn
                            );
                            let outs = sm_orch.start_establishment(params);
                            process_sm_outputs(
                                outs,
                                &mut orch,
                                &task_base,
                                &tun_tx,
                                &mut pdu_counter,
                            )
                            .await;
                        }
                        NasMessage::InitiatePduSessionRelease { psi, pti: _ } => {
                            info!("Initiating PDU session release: PSI={}", psi);
                            let outs = sm_orch.start_release(psi);
                            process_sm_outputs(
                                outs,
                                &mut orch,
                                &task_base,
                                &tun_tx,
                                &mut pdu_counter,
                            )
                            .await;
                        }
                        NasMessage::InitiateDeregistration { switch_off } => {
                            info!("Initiating deregistration: switch_off={}", switch_off);
                            // Release the PDU sessions locally (the 5GMM
                            // deregistration implicitly releases them,
                            // TS 24.501 5.5.2.1)
                            let sm_outs = sm_orch.release_all_locally();
                            process_sm_outputs(
                                sm_outs,
                                &mut orch,
                                &task_base,
                                &tun_tx,
                                &mut pdu_counter,
                            )
                            .await;
                            let outs = orch.start_deregistration(switch_off);
                            process_mm_outputs(
                                outs,
                                &mut orch,
                                &mut sm_orch,
                                &sm_session_params,
                                &mut plmn_selector,
                                &task_base,
                                &tun_tx,
                                &mut pdu_counter,
                            )
                            .await;
                        }
                        NasMessage::InitiateEmergencyRegistration => {
                            info!("Initiating emergency registration");

                            use nextgsim_nas::messages::mm::{
                                Ie5gsMobileIdentity, MobileIdentityType,
                            };
                            use nextgsim_ue::nas::mm::{
                                EmergencyRegistrationProcedure, MmState, MmSubState,
                            };

                            let mut emerg_proc = EmergencyRegistrationProcedure::new();

                            // Use IMEI as identity if SUCI is not available (no USIM).
                            // Here we always build a SUCI — a full implementation would
                            // fall back to IMEI when no USIM is present.
                            let suci_data = match build_suci_from_config(&task_base.config) {
                                Ok(suci_data) => suci_data,
                                Err(e) => {
                                    warn!("Emergency registration aborted: cannot build SUCI: {e}");
                                    continue;
                                }
                            };
                            let mobile_identity =
                                Ie5gsMobileIdentity::new(MobileIdentityType::Suci, suci_data);

                            match emerg_proc.initiate(
                                MmState::Deregistered,
                                MmSubState::DeregisteredNormalService,
                                None,
                                mobile_identity,
                                false,
                            ) {
                                Ok((req, new_sub_state)) => {
                                    let mut nas_pdu = Vec::new();
                                    req.encode(&mut nas_pdu);
                                    info!(
                                        "Sending Emergency Registration Request, len={}, new_sub_state={:?}",
                                        nas_pdu.len(), new_sub_state
                                    );
                                    orch.state_mut().switch_mm_state(new_sub_state);
                                    pdu_counter += 1;
                                    let _ = task_base
                                        .rrc_tx
                                        .send(RrcMessage::UplinkNasDelivery {
                                            pdu_id: pdu_counter,
                                            pdu: nas_pdu.into(),
                                        })
                                        .await;
                                }
                                Err(e) => {
                                    warn!("Emergency registration initiation failed: {}", e);
                                }
                            }
                        }
                        NasMessage::DownlinkDataDelivery { psi, data } => {
                            // Forward downlink data to TUN interface
                            debug!("Downlink data delivery: psi={}, len={}", psi, data.len());
                            let _ = tun_tx.send(TunMessage::WriteData { psi, data }).await;
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
        }

        info!("NAS task stopped");
    }

    /// Runs the RRC task (handles RRC state machine and procedures)
    async fn run_rrc_task(
        task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
    ) {
        use nextgsim_common::OctetString;
        use nextgsim_ue::rrc::{RrcState, RrcStateMachine, RrcStateTransition};
        use std::collections::HashMap;

        info!("RRC task started");

        // Cell tracking
        let mut discovered_cells: HashMap<i32, i32> = HashMap::new(); // cell_id -> dbm
        let mut serving_cell: Option<i32> = None;
        let mut rrc_state_machine = RrcStateMachine::new();
        let mut registration_triggered = false;
        // Wave-6 I5: KgNB handed down by the NAS plane once NAS security is
        // active; consumed by the AS SecurityModeCommand path below when the
        // I5_UE_AS_SECURITY wire gate is on.
        let mut pending_kgnb: Option<[u8; 32]> = None;

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        RrcMessage::LocalReleaseConnection { treat_barred } => {
                            info!("Local release connection, treat_barred={}", treat_barred);
                            serving_cell = None;
                            registration_triggered = false;
                            if rrc_state_machine.can_transition(RrcStateTransition::Release) {
                                let _ = rrc_state_machine.transition(RrcStateTransition::Release);
                            }
                        }
                        RrcMessage::UplinkNasDelivery { pdu_id, pdu } => {
                            info!("Uplink NAS delivery: pdu_id={}, len={}", pdu_id, pdu.len());
                            // Forward to RLS for transmission
                            if serving_cell.is_some() {
                                // Check if we have an RRC connection established
                                // If connected, wrap in UL Information Transfer
                                // If not (initial access), send raw NAS PDU for gNB fallback
                                let wrapped_pdu = if rrc_state_machine.state().is_connected() {
                                    // Wrap NAS PDU in UL Information Transfer format
                                    // Format: [0x08 (msg type), 0x00 (padding), NAS PDU...]
                                    let mut ul_info_transfer = Vec::with_capacity(pdu.len() + 2);
                                    ul_info_transfer.push(0x08); // UL Information Transfer
                                    ul_info_transfer.push(0x00); // Padding
                                    ul_info_transfer.extend_from_slice(pdu.data());
                                    ul_info_transfer.into()
                                } else {
                                    // Initial access: send raw NAS PDU
                                    // gNB will auto-create UE context and send Initial UE Message
                                    pdu
                                };

                                let _ = task_base
                                    .rls_tx
                                    .send(RlsMessage::RrcPduDelivery {
                                        channel: RrcChannel::UlDcch,
                                        pdu_id,
                                        pdu: wrapped_pdu,
                                    })
                                    .await;
                            } else {
                                warn!("Cannot send uplink NAS: no serving cell");
                            }
                        }
                        RrcMessage::RrcNotify => {
                            info!("RRC notify received");
                        }
                        RrcMessage::PerformUac {
                            access_category,
                            access_identities,
                        } => {
                            info!(
                                "Perform UAC: category={}, identities={}",
                                access_category, access_identities
                            );
                        }
                        RrcMessage::DownlinkRrcDelivery {
                            cell_id,
                            channel,
                            pdu,
                        } => {
                            info!(
                                "Downlink RRC: cell_id={}, channel={:?}, len={}",
                                cell_id,
                                channel,
                                pdu.len()
                            );

                            // Receiving a downlink RRC message means RRC connection is established
                            // Transition to Connected state if we're in Idle
                            if rrc_state_machine.state().is_idle()
                                && rrc_state_machine
                                    .can_transition(RrcStateTransition::SetupComplete)
                            {
                                let _ =
                                    rrc_state_machine.transition(RrcStateTransition::SetupComplete);
                                info!(
                                    "RRC state: {} -> {} (connection established)",
                                    RrcState::Idle,
                                    rrc_state_machine.state()
                                );
                            }

                            // Wave-6 I5 (TS 38.331 §5.3.4): when the AS-security
                            // wire gate is on, recognise the AS
                            // SecurityModeCommand on SRB1 (DL-DCCH) BEFORE the
                            // raw-NAS heuristic below — its leading byte 0x20 is
                            // otherwise misread as a raw NAS PDU and dropped.
                            // Default-off: the block is skipped and the
                            // matched-sim path is byte-for-byte unchanged.
                            if nextgsim_ue::rrc::I5_UE_AS_SECURITY && channel == RrcChannel::DlDcch
                            {
                                use nextgsim_rrc::procedures::security_mode::{
                                    decode_security_mode_command, encode_security_mode_complete,
                                    SecurityModeCompleteParams,
                                };
                                use nextgsim_ue::rrc::{
                                    AsSecurityContext, CipheringAlgorithm, IntegrityAlgorithm,
                                };
                                if let Ok(smc) = decode_security_mode_command(pdu.data()) {
                                    match (
                                        smc.security_algorithms.integrity_algorithm,
                                        pending_kgnb,
                                    ) {
                                        (Some(integ), Some(kgnb)) => {
                                            let ciph = CipheringAlgorithm::from(
                                                smc.security_algorithms.ciphering_algorithm,
                                            );
                                            if ciph == CipheringAlgorithm::Nea3 {
                                                warn!("AS SMC selected NEA3 (unsupported keystream); not activating");
                                            } else {
                                                // Derive + install the AS keys (byte-identical to
                                                // the gNB's derive_rrc_up_key); SRB PDCP
                                                // integrity/ciphering enforcement rides these keys.
                                                let _as_ctx = AsSecurityContext::derive_from_kgnb(
                                                    &kgnb,
                                                    ciph,
                                                    IntegrityAlgorithm::from(integ),
                                                    0x4601,
                                                );
                                                if let Ok(bytes) = encode_security_mode_complete(
                                                    &SecurityModeCompleteParams {
                                                        rrc_transaction_id: smc.rrc_transaction_id,
                                                    },
                                                ) {
                                                    let _ = task_base
                                                        .rls_tx
                                                        .send(RlsMessage::RrcPduDelivery {
                                                            channel: RrcChannel::UlDcch,
                                                            pdu_id: 0,
                                                            pdu: OctetString::from_slice(&bytes),
                                                        })
                                                        .await;
                                                }
                                                info!(
                                                    "AS security activated (tid {}); SecurityModeComplete sent",
                                                    smc.rrc_transaction_id
                                                );
                                            }
                                        }
                                        _ => warn!(
                                            "AS SMC received but no KgNB / integrity algorithm; not activating (fail-closed)"
                                        ),
                                    }
                                    continue;
                                }
                            }

                            // Check if this is DL Information Transfer (0x04) which wraps NAS PDU
                            // Format: [0x04 (msg type), 0x00 (padding), 0x00 (padding), NAS PDU...]
                            let nas_pdu = if pdu.len() >= 3 && pdu.data()[0] == 0x04 {
                                // Extract NAS PDU from DL Information Transfer
                                OctetString::from_slice(&pdu.data()[3..])
                            } else {
                                // Assume raw NAS PDU
                                pdu
                            };

                            // Forward to NAS for processing
                            let _ = task_base
                                .nas_tx
                                .send(NasMessage::NasDelivery { pdu: nas_pdu })
                                .await;
                        }
                        RrcMessage::SignalChanged { cell_id, dbm } => {
                            info!("Signal changed: cell_id={}, dbm={}", cell_id, dbm);

                            // Track the cell
                            discovered_cells.insert(cell_id, dbm);

                            // If we don't have a serving cell yet and we're in IDLE state, select one
                            if serving_cell.is_none() && rrc_state_machine.state().is_idle() {
                                // Select the cell with best signal (or just the first one for now)
                                let best_cell = discovered_cells
                                    .iter()
                                    .max_by_key(|(_, &signal)| signal)
                                    .map(|(&id, _)| id);

                                if let Some(best_cell_id) = best_cell {
                                    info!("Selecting cell {} as serving cell", best_cell_id);
                                    serving_cell = Some(best_cell_id);

                                    // Tell RLS to camp on this cell
                                    let _ = task_base
                                        .rls_tx
                                        .send(RlsMessage::AssignCurrentCell {
                                            cell_id: best_cell_id,
                                        })
                                        .await;

                                    // Trigger registration if not already done
                                    if !registration_triggered {
                                        info!("Triggering NAS registration procedure");
                                        registration_triggered = true;
                                        let _ =
                                            task_base.nas_tx.send(NasMessage::PerformMmCycle).await;
                                    }
                                }
                            }
                        }
                        RrcMessage::RadioLinkFailure { cause } => {
                            warn!("Radio link failure: {:?}", cause);
                            // Remove the lost cell from our tracking
                            if let Some(cell_id) = serving_cell {
                                discovered_cells.remove(&cell_id);
                                serving_cell = None;
                                registration_triggered = false;
                            }
                            if rrc_state_machine
                                .can_transition(RrcStateTransition::RadioLinkFailure)
                            {
                                let _ = rrc_state_machine
                                    .transition(RrcStateTransition::RadioLinkFailure);
                            }
                            // Notify NAS of the failure
                            let _ = task_base.nas_tx.send(NasMessage::RadioLinkFailure).await;
                        }
                        RrcMessage::AsSecurityKey { kgnb } => {
                            // Wave-6 I5: store the KgNB from the NAS plane; used
                            // to derive K_RRCint/K_RRCenc on the AS SMC.
                            debug!("Received KgNB for AS security from NAS plane");
                            pending_kgnb = Some(kgnb);
                        }
                        RrcMessage::TriggerCycle => {
                            // Trigger RRC state machine cycle
                        }
                        RrcMessage::NtnTimingAdvanceReceived {
                            common_ta_us,
                            k_offset,
                            ..
                        } => {
                            info!("UE: NTN TA={}us, k_offset={}", common_ta_us, k_offset);
                        }
                        // 6G message routing - log and drop in main binary (handled by lib RrcTask)
                        #[cfg(feature = "nextgsim-she")]
                        RrcMessage::SixgInferenceRequest { model_id, .. } => {
                            debug!(
                                "6G inference request for model {}, not handled in main binary",
                                model_id
                            );
                        }
                        #[cfg(feature = "nextgsim-isac")]
                        RrcMessage::SixgSensingMeasurement {
                            measurement_type, ..
                        } => {
                            debug!(
                                "6G sensing measurement type {}, not handled in main binary",
                                measurement_type
                            );
                        }
                        #[cfg(feature = "nextgsim-semantic")]
                        RrcMessage::SixgSemanticData { content_type, .. } => {
                            debug!(
                                "6G semantic data type {}, not handled in main binary",
                                content_type
                            );
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
        info!(
            "UE {} started, waiting for shutdown signal...",
            self.instance_id
        );

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
                warn!(
                    "UE {}: Some tasks failed during shutdown: {}",
                    self.instance_id, e
                );
                // Still return Ok since we're shutting down anyway
                Ok(())
            }
        }
    }
}

/// Deliver MM orchestrator outputs: transmit NAS PDUs, react to procedure
/// outcomes and dispatch plain 5GMM messages the orchestrator does not own.
#[allow(clippy::too_many_arguments)]
async fn process_mm_outputs(
    outputs: Vec<nextgsim_ue::nas::mm::MmOutput>,
    orch: &mut nextgsim_ue::nas::mm::MmOrchestrator,
    sm_orch: &mut nextgsim_ue::nas::sm::SmOrchestrator,
    sm_session_params: &[nextgsim_ue::nas::sm::SmSessionParams],
    plmn_selector: &mut nextgsim_ue::rrc::cell_selection::PlmnSelector,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    use nextgsim_ue::nas::mm::MmOutput;
    use nextgsim_ue::rrc::cell_selection::Plmn;

    let home_plmn = Plmn::new(
        task_base.config.hplmn.mcc,
        task_base.config.hplmn.mnc,
        task_base.config.hplmn.long_mnc,
    );

    for output in outputs {
        match output {
            MmOutput::SendNasPdu(nas_pdu) => {
                *pdu_counter += 1;
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::UplinkNasDelivery {
                        pdu_id: *pdu_counter,
                        pdu: nas_pdu.into(),
                    })
                    .await;
            }
            MmOutput::RegistrationSucceeded => {
                info!("Registration completed, MM state: {}", orch.state());
                notify_rel18_registration(task_base, true).await;
                // TS 23.122: record the registered PLMN (the camped cell
                // broadcasts the configured PLMN in this simulation)
                plmn_selector.set_registered_plmn(Some(home_plmn));

                // Establish the configured default PDU sessions via the SM
                // orchestrator (PSI/PTI allocation, T3580, UL NAS Transport
                // wrapping all happen there)
                if sm_orch.active_sessions().is_empty() {
                    // Give the core a moment to finish the registration
                    // path before requesting the PDU session
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    // MINT (Rel-18, TS 23.761): the primary subscription only
                    // establishes sessions routed to it (subscription_index 0);
                    // secondary-SUPI sessions are established by the MINT driver
                    // under their own SUPI's context.
                    let primary_params: Vec<_> = sm_session_params
                        .iter()
                        .filter(|p| p.subscription_index == 0)
                        .cloned()
                        .collect();
                    let outs = sm_orch.establish_default_sessions(&primary_params);
                    process_sm_outputs(outs, orch, task_base, tun_tx, pdu_counter).await;
                }
            }
            MmOutput::EquivalentPlmnsUpdated(bcd_plmns) => {
                // TS 23.122 §4.4.3.1.1: the equivalent-PLMN list signalled in
                // Registration Accept is consulted (after the RPLMN) by
                // subsequent automatic PLMN selection. Convert the BCD triplets
                // into PLMNs and hand them to the selector.
                use nextgsim_ue::rrc::cell_selection::plmn_from_bcd;
                let plmns: Vec<Plmn> = bcd_plmns.iter().map(plmn_from_bcd).collect();
                if plmns.is_empty() {
                    info!("Equivalent-PLMN list cleared by the network");
                } else {
                    info!(
                        "Equivalent-PLMN list updated: {}",
                        plmns
                            .iter()
                            .map(ToString::to_string)
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }
                plmn_selector.set_equivalent_plmns(plmns);
            }
            MmOutput::RegistrationFailed(cause) => {
                warn!("Registration failed: {:?} (#{})", cause, cause as u8);
                plmn_selector.set_registered_plmn(None);
                notify_rel18_registration(task_base, false).await;
            }
            MmOutput::AuthenticationRejected => {
                warn!("Network rejected authentication: USIM considered invalid");
                notify_rel18_registration(task_base, false).await;
            }
            MmOutput::PlmnSearchNeeded => {
                // TS 23.122 Section 4.4.3.1: run PLMN selection with the
                // MM orchestrator's forbidden-PLMN list
                perform_plmn_selection(orch, plmn_selector, home_plmn, task_base, pdu_counter)
                    .await;
            }
            MmOutput::NotHandled(plain) => {
                handle_unmanaged_mm_message(plain, orch, sm_orch, task_base, tun_tx, pdu_counter)
                    .await;
            }
            MmOutput::AsSecurityKgnb(kgnb) => {
                // Wave-6 I5: hand the KgNB to the RRC plane for AS-security key
                // derivation on the next AS SecurityModeCommand (TS 33.501 §6.7).
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::AsSecurityKey { kgnb })
                    .await;
            }
        }
    }
}

/// Deliver SM orchestrator outputs: protect and transmit UL NAS Transport
/// PDUs through the MM security context and manage the TUN interfaces for
/// session lifecycle events.
async fn process_sm_outputs(
    outputs: Vec<nextgsim_ue::nas::sm::SmOutput>,
    orch: &mut nextgsim_ue::nas::mm::MmOrchestrator,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    use nextgsim_ue::nas::sm::SmOutput;

    for output in outputs {
        match output {
            SmOutput::SendNasPdu(plain) => {
                // 5GSM messages travel inside UL NAS Transport and are
                // protected with the MM NAS security context
                // (TS 24.501 Sections 4.4 and 5.4.5)
                let pdu = orch.protect_if_active(plain);
                *pdu_counter += 1;
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::UplinkNasDelivery {
                        pdu_id: *pdu_counter,
                        pdu: pdu.into(),
                    })
                    .await;
            }
            SmOutput::SessionEstablished { psi, ipv4 } => {
                info!("PDU session {psi} is now ACTIVE (IPv4: {ipv4:?})");
                if let Some(ip) = ipv4 {
                    let address = Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]);
                    info!("Creating TUN interface for PDU session {psi} with IP {address}");
                    let _ = tun_tx
                        .send(TunMessage::CreateInterface {
                            psi: i32::from(psi),
                            address,
                            netmask: Ipv4Addr::new(255, 255, 255, 0),
                        })
                        .await;
                }
            }
            SmOutput::SessionEstablishmentFailed { psi, cause } => {
                warn!(
                    "PDU session {psi} establishment failed: {:?} (#{})",
                    cause, cause as u8
                );
            }
            SmOutput::SessionReleased { psi } => {
                info!("PDU session {psi} released");
                let _ = tun_tx
                    .send(TunMessage::DestroyInterface {
                        psi: i32::from(psi),
                    })
                    .await;
            }
            SmOutput::SessionModified { psi } => {
                info!("PDU session {psi} modification completed");
            }
            SmOutput::SessionModificationFailed { psi, cause } => {
                warn!(
                    "PDU session {psi} modification failed: {:?} (#{})",
                    cause, cause as u8
                );
            }
        }
    }
}

/// Deliver a MINT secondary subscription's MM outputs (Rel-18, TS 23.761):
/// transmit its NAS PDUs over the shared radio connection, and on its
/// registration success report the subscription to the MINT task and establish
/// the PDU sessions routed to this subscription's SUPI under its own context.
async fn process_secondary_mm_outputs(
    index: u8,
    outputs: Vec<nextgsim_ue::nas::mm::MmOutput>,
    mint_secondary: &mut nextgsim_ue::nas::mm::MintSecondary,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    use nextgsim_ue::nas::mm::MmOutput;

    for output in outputs {
        match output {
            MmOutput::SendNasPdu(nas_pdu) => {
                *pdu_counter += 1;
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::UplinkNasDelivery {
                        pdu_id: *pdu_counter,
                        pdu: nas_pdu.into(),
                    })
                    .await;
            }
            MmOutput::RegistrationSucceeded => {
                info!("MINT: secondary subscription {index} registered");
                // Report the secondary registration to the MINT task so it
                // tracks per-SUPI state (TS 23.761).
                if let Some(ref rel18) = task_base.rel18 {
                    let supi = mint_secondary
                        .get_mut(index)
                        .map(|s| s.supi.clone())
                        .unwrap_or_default();
                    let _ = supi; // SUPI is tracked by index in the MINT task
                    let _ = rel18
                        .mint_tx
                        .send(MintMessage::RegistrationUpdate {
                            subscription_index: index,
                            registered: true,
                            serving_plmn: Some((
                                task_base.config.hplmn.mcc,
                                task_base.config.hplmn.mnc,
                            )),
                            guti: None,
                        })
                        .await;
                }
                // Establish this subscription's PDU sessions (routed by DNN to
                // this SUPI) under its own MM/security context, so the SMF
                // resolves SUPI-N's subscription in UDM/UDR.
                let params = nextgsim_ue::nas::mm::MintSecondary::session_params_for(
                    &task_base.config,
                    index,
                );
                if let Some(sub) = mint_secondary.get_mut(index) {
                    if sub.sm.active_sessions().is_empty() && !params.is_empty() {
                        tokio::time::sleep(Duration::from_millis(500)).await;
                        let sm_outs = sub.sm.establish_default_sessions(&params);
                        process_secondary_sm_outputs(
                            index,
                            sm_outs,
                            mint_secondary,
                            task_base,
                            tun_tx,
                            pdu_counter,
                        )
                        .await;
                    }
                }
            }
            MmOutput::EquivalentPlmnsUpdated(plmns) => {
                // A MINT secondary subscription does not drive the primary
                // PLMN selector; its equivalent-PLMN list is tracked per-SUPI
                // by the secondary context only (TS 23.761 / TS 23.122).
                tracing::debug!(
                    "MINT: secondary subscription {index} equivalent-PLMN list ({} entries)",
                    plmns.len()
                );
            }
            MmOutput::RegistrationFailed(cause) => {
                warn!("MINT: secondary subscription {index} registration failed: {cause:?}");
            }
            MmOutput::AuthenticationRejected => {
                warn!("MINT: secondary subscription {index} authentication rejected");
            }
            MmOutput::PlmnSearchNeeded => {
                warn!("MINT: secondary subscription {index} requested PLMN search (ignored)");
            }
            MmOutput::NotHandled(plain) => {
                // Dispatch a plain 5GMM message the secondary orchestrator does
                // not own (e.g. DL NAS Transport carrying its 5GSM responses).
                if plain.len() >= 3
                    && plain[2] == u8::from(nextgsim_nas::enums::MmMessageType::DlNasTransport)
                {
                    use nextgsim_nas::messages::mm::DlNasTransport;
                    if let Ok(dl) = DlNasTransport::decode(&mut &plain[3..]) {
                        if let Some(sub) = mint_secondary.get_mut(index) {
                            let sm_outs = sub.sm.handle_dl_nas_transport(&dl);
                            process_secondary_sm_outputs(
                                index,
                                sm_outs,
                                mint_secondary,
                                task_base,
                                tun_tx,
                                pdu_counter,
                            )
                            .await;
                        }
                    }
                }
            }
            MmOutput::AsSecurityKgnb(_kgnb) => {
                // A MINT secondary subscription shares the primary UE's AS/RRC
                // security context; it never drives an independent AS
                // SecurityModeCommand, so its KgNB is not plumbed to RRC.
                tracing::debug!(
                    "MINT: secondary subscription {index} KgNB derived (not used for AS security)"
                );
            }
        }
    }
}

/// Process a downlink NAS PDU for a MINT secondary subscription's MM context.
async fn process_secondary_downlink(
    index: u8,
    pdu: &[u8],
    mint_secondary: &mut nextgsim_ue::nas::mm::MintSecondary,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    let outs = match mint_secondary.get_mut(index) {
        Some(sub) => sub.mm.handle_downlink(pdu),
        None => return,
    };
    process_secondary_mm_outputs(index, outs, mint_secondary, task_base, tun_tx, pdu_counter).await;
}

/// Deliver a MINT secondary subscription's SM outputs: protect the UL NAS
/// Transport with that subscription's MM security context and transmit it,
/// and manage TUN interfaces for its session lifecycle (Rel-18, TS 23.761).
async fn process_secondary_sm_outputs(
    index: u8,
    outputs: Vec<nextgsim_ue::nas::sm::SmOutput>,
    mint_secondary: &mut nextgsim_ue::nas::mm::MintSecondary,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    use nextgsim_ue::nas::sm::SmOutput;

    for output in outputs {
        match output {
            SmOutput::SendNasPdu(plain) => {
                let pdu = match mint_secondary.get_mut(index) {
                    Some(sub) => sub.mm.protect_if_active(plain),
                    None => return,
                };
                *pdu_counter += 1;
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::UplinkNasDelivery {
                        pdu_id: *pdu_counter,
                        pdu: pdu.into(),
                    })
                    .await;
            }
            SmOutput::SessionEstablished { psi, ipv4 } => {
                info!(
                    "MINT: secondary subscription {index} PDU session {psi} ACTIVE (IPv4: {ipv4:?})"
                );
                // Report the session to the MINT task (per-SUPI session count).
                if let Some(ref rel18) = task_base.rel18 {
                    let _ = rel18
                        .mint_tx
                        .send(MintMessage::SessionUpdate {
                            subscription_index: index,
                            psi,
                            active: true,
                        })
                        .await;
                }
                if let Some(ip) = ipv4 {
                    let address = Ipv4Addr::new(ip[0], ip[1], ip[2], ip[3]);
                    let _ = tun_tx
                        .send(TunMessage::CreateInterface {
                            psi: i32::from(psi),
                            address,
                            netmask: Ipv4Addr::new(255, 255, 255, 0),
                        })
                        .await;
                }
            }
            SmOutput::SessionEstablishmentFailed { psi, cause } => {
                warn!(
                    "MINT: secondary subscription {index} PDU session {psi} establishment failed: {cause:?}"
                );
            }
            SmOutput::SessionReleased { psi } => {
                info!("MINT: secondary subscription {index} PDU session {psi} released");
                if let Some(ref rel18) = task_base.rel18 {
                    let _ = rel18
                        .mint_tx
                        .send(MintMessage::SessionUpdate {
                            subscription_index: index,
                            psi,
                            active: false,
                        })
                        .await;
                }
                let _ = tun_tx
                    .send(TunMessage::DestroyInterface {
                        psi: i32::from(psi),
                    })
                    .await;
            }
            SmOutput::SessionModified { psi } => {
                info!("MINT: secondary subscription {index} PDU session {psi} modified");
            }
            SmOutput::SessionModificationFailed { psi, cause } => {
                warn!(
                    "MINT: secondary subscription {index} PDU session {psi} modification failed: {cause:?}"
                );
            }
        }
    }
}

/// Run TS 23.122 automatic PLMN selection: feed the selector with the MM
/// orchestrator's forbidden-PLMN list and either trigger a registration
/// attempt on the selected PLMN or enter limited service.
async fn perform_plmn_selection(
    orch: &mut nextgsim_ue::nas::mm::MmOrchestrator,
    plmn_selector: &mut nextgsim_ue::rrc::cell_selection::PlmnSelector,
    home_plmn: nextgsim_ue::rrc::cell_selection::Plmn,
    task_base: &UeTaskBase,
    pdu_counter: &mut u32,
) {
    use nextgsim_nas::ies::RegistrationType;
    use nextgsim_ue::nas::mm::MmOutput;
    use nextgsim_ue::rrc::cell_selection::plmn_from_bcd;

    // Consume the MM orchestrator's forbidden-PLMN list (TS 24.501 5.3.13)
    let forbidden: Vec<_> = orch.forbidden_plmns().iter().map(plmn_from_bcd).collect();
    plmn_selector.set_forbidden(forbidden);
    plmn_selector.set_registered_plmn(None);

    // Available PLMNs: in this simulation every cell in the gNB search
    // list broadcasts the configured home PLMN
    let available = [home_plmn];
    match plmn_selector.select(&available) {
        Some(plmn) => {
            info!("PLMN selection chose {plmn}: attempting registration");
            let outs = orch.start_registration(RegistrationType::InitialRegistration);
            for out in outs {
                if let MmOutput::SendNasPdu(nas_pdu) = out {
                    *pdu_counter += 1;
                    let _ = task_base
                        .rrc_tx
                        .send(RrcMessage::UplinkNasDelivery {
                            pdu_id: *pdu_counter,
                            pdu: nas_pdu.into(),
                        })
                        .await;
                }
            }
        }
        None => {
            warn!(
                "PLMN selection found no allowed PLMN ({} forbidden): limited service",
                plmn_selector.is_forbidden(home_plmn)
            );
        }
    }
}

/// Notify the Rel-18 actors of a registration-state change: the MINT task
/// tracks the primary subscription's registration (TS 23.761) and, when
/// ranging is configured, the Sidelink task starts/stops discovery.
async fn notify_rel18_registration(task_base: &UeTaskBase, registered: bool) {
    let Some(ref rel18) = task_base.rel18 else {
        return;
    };
    let serving_plmn = if registered {
        Some((task_base.config.hplmn.mcc, task_base.config.hplmn.mnc))
    } else {
        None
    };
    let _ = rel18
        .mint_tx
        .send(MintMessage::RegistrationUpdate {
            subscription_index: 0,
            registered,
            serving_plmn,
            guti: None,
        })
        .await;
    let ranging_enabled = task_base
        .config
        .ranging_config
        .as_ref()
        .is_some_and(|c| c.enabled);
    if ranging_enabled {
        let msg = if registered {
            SidelinkMessage::StartDiscovery
        } else {
            SidelinkMessage::StopDiscovery
        };
        let _ = rel18.sidelink_tx.send(msg).await;
    }
}

/// Handle plain (already security-unwrapped) 5GMM messages that are not
/// part of the orchestrator's procedures: identity request, DL NAS
/// transport, configuration update.
async fn handle_unmanaged_mm_message(
    plain: Vec<u8>,
    orch: &mut nextgsim_ue::nas::mm::MmOrchestrator,
    sm_orch: &mut nextgsim_ue::nas::sm::SmOrchestrator,
    task_base: &UeTaskBase,
    tun_tx: &tokio::sync::mpsc::Sender<TunMessage>,
    pdu_counter: &mut u32,
) {
    use nextgsim_nas::enums::MmMessageType;
    use nextgsim_nas::ies::RegistrationType;
    use nextgsim_nas::messages::mm::{
        DlNasTransport, IdentityRequest, IdentityResponse, Ie5gsMobileIdentity, MobileIdentityType,
    };
    use nextgsim_ue::nas::mm::{MmOutput, MmSubState};

    let Ok(msg_type) = MmMessageType::try_from(plain[2]) else {
        warn!("Unknown NAS message type: 0x{:02x}", plain[2]);
        return;
    };

    match msg_type {
        MmMessageType::IdentityRequest => {
            // Decode Identity Request (decode consumes the full message)
            match IdentityRequest::decode(&mut plain.as_slice()) {
                Ok(id_req) => {
                    info!("Identity Request - Type: {:?}", id_req.identity_type.value);

                    // TS 24.501 §5.4.3.3 b): reuse the stored SUCI while T3519 is
                    // running; otherwise build a fresh one, store it and arm
                    // T3519 so the frequency limit (TS 33.501 §6.12.4) applies.
                    let suci_data = if let Some(stored) = orch.stored_suci_while_t3519_running() {
                        debug!("Reusing stored SUCI for Identity Response (T3519 running)");
                        stored
                    } else {
                        match build_suci_from_config(&task_base.config) {
                            Ok(fresh) => {
                                orch.store_suci_and_arm_t3519(fresh.clone());
                                fresh
                            }
                            Err(e) => {
                                warn!("Cannot answer Identity Request: SUCI build failed: {e}");
                                return;
                            }
                        }
                    };
                    let mobile_identity =
                        Ie5gsMobileIdentity::new(MobileIdentityType::Suci, suci_data);

                    let id_response = IdentityResponse::new(mobile_identity);
                    let mut nas_pdu = Vec::new();
                    id_response.encode(&mut nas_pdu);
                    let nas_pdu = orch.protect_if_active(nas_pdu);

                    info!("Sending Identity Response, PDU len={}", nas_pdu.len());
                    *pdu_counter += 1;
                    let _ = task_base
                        .rrc_tx
                        .send(RrcMessage::UplinkNasDelivery {
                            pdu_id: *pdu_counter,
                            pdu: nas_pdu.into(),
                        })
                        .await;
                }
                Err(e) => {
                    warn!("Failed to decode Identity Request: {:?}", e);
                }
            }
        }
        MmMessageType::DlNasTransport => {
            use nextgsim_nas::ies::ie1::PayloadContainerType;
            use nextgsim_nas::messages::mm::UlNasTransport;
            use nextgsim_ue::nas::mm::UePolicyReaction;
            // DL NAS Transport (TS 24.501 8.2.11): decode through the codec.
            // A "UE policy container" (0b0101) is UPDP (TS 24.501 Annex D) and
            // is handled on the MM plane BEFORE the SM hand-off; everything
            // else (N1 SM information, etc.) goes to the SM orchestrator.
            let header_len = 3; // EPD + SecHdr + MsgType
            match DlNasTransport::decode(&mut &plain[header_len..]) {
                Ok(dl) => {
                    debug!(
                        "DL NAS Transport: container type {:?}, PSI {:?}, len={}",
                        dl.payload_container_type,
                        dl.pdu_session_id,
                        dl.payload_container.len()
                    );
                    if dl.payload_container_type == PayloadContainerType::UePolicyContainer {
                        // MANAGE UE POLICY COMMAND: decode, store sections and
                        // answer COMPLETE/REJECT (Wave-6 E8).
                        match orch.handle_ue_policy_command(&dl.payload_container) {
                            UePolicyReaction::Reply(updp) => {
                                let ul = UlNasTransport::new(
                                    PayloadContainerType::UePolicyContainer,
                                    updp,
                                );
                                let mut nas_pdu = Vec::new();
                                ul.encode(&mut nas_pdu);
                                let nas_pdu = orch.protect_if_active(nas_pdu);
                                info!(
                                    "Sending UE policy delivery reply ({} sections stored), len={}",
                                    orch.ue_policy_section_count(),
                                    nas_pdu.len()
                                );
                                *pdu_counter += 1;
                                let _ = task_base
                                    .rrc_tx
                                    .send(RrcMessage::UplinkNasDelivery {
                                        pdu_id: *pdu_counter,
                                        pdu: nas_pdu.into(),
                                    })
                                    .await;
                            }
                            UePolicyReaction::Ignore => {
                                warn!("UE policy container undecodable and unrecoverable: dropped");
                            }
                        }
                    } else {
                        let outs = sm_orch.handle_dl_nas_transport(&dl);
                        process_sm_outputs(outs, orch, task_base, tun_tx, pdu_counter).await;
                    }
                }
                Err(e) => {
                    warn!("Failed to decode DL NAS Transport: {e:?}");
                }
            }
        }
        MmMessageType::ConfigurationUpdateCommand => {
            use nextgsim_ue::nas::mm::{
                ConfigUpdateProcedure, ConfigurationUpdateCommand as CuCmd,
                ConfigurationUpdateComplete as CuComplete,
            };
            let header_len = 3; // EPD + SecHdr + MsgType
            let cmd = if plain.len() > header_len {
                CuCmd::decode(&mut &plain[header_len..]).unwrap_or_default()
            } else {
                CuCmd::default()
            };
            let result = ConfigUpdateProcedure::process_command(&cmd);

            if let Some(ref new_guti) = result.new_guti {
                info!(
                    "ConfigurationUpdate: new GUTI received (type={:?})",
                    new_guti.identity_type
                );
            }
            if let Some(t) = result.new_t3512_secs {
                info!("ConfigurationUpdate: T3512 updated to {}s", t);
            }

            // Send ConfigurationUpdateComplete if the ACK bit was set
            if result.send_complete {
                let mut nas_pdu = Vec::new();
                CuComplete::new().encode(&mut nas_pdu);
                let nas_pdu = orch.protect_if_active(nas_pdu);
                info!("Sending ConfigurationUpdateComplete, len={}", nas_pdu.len());
                *pdu_counter += 1;
                let _ = task_base
                    .rrc_tx
                    .send(RrcMessage::UplinkNasDelivery {
                        pdu_id: *pdu_counter,
                        pdu: nas_pdu.into(),
                    })
                    .await;
            }

            // A re-registration request is a mobility registration trigger
            if result.re_register {
                info!("ConfigurationUpdate: re-registration requested");
                orch.state_mut()
                    .switch_mm_state(MmSubState::RegisteredUpdateNeeded);
                let outs = orch.start_registration(RegistrationType::MobilityRegistrationUpdating);
                for out in outs {
                    if let MmOutput::SendNasPdu(nas_pdu) = out {
                        *pdu_counter += 1;
                        let _ = task_base
                            .rrc_tx
                            .send(RrcMessage::UplinkNasDelivery {
                                pdu_id: *pdu_counter,
                                pdu: nas_pdu.into(),
                            })
                            .await;
                    }
                }
            }
        }
        _ => {
            info!("Unhandled NAS message type: {:?}", msg_type);
        }
    }
}

/// Build the SUCI 5GS mobile identity IE contents from the UE configuration
/// (TS 24.501 Section 9.11.3.4).
///
/// Delegates to [`nextgsim_ue::nas::mm::build_suci`], which selects the
/// protection scheme (null / ECIES Profile A / ECIES Profile B) from
/// `config.protection_scheme` and conceals the MSIN with a fresh ephemeral
/// key pair per call. Errors are fatal for identity signalling: there is no
/// silent fallback to the null scheme.
fn build_suci_from_config(
    config: &nextgsim_common::config::UeConfig,
) -> Result<Vec<u8>, nextgsim_ue::nas::mm::SuciBuildError> {
    nextgsim_ue::nas::mm::build_suci(config)
}

/// Initializes the tracing subscriber for logging.
fn init_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

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
    info!("HPLMN: {}-{}", base_config.hplmn.mcc, base_config.hplmn.mnc);
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
    let digits = if let Some(stripped) = imsi.strip_prefix("imsi-") {
        stripped
    } else {
        imsi
    };

    // Parse as u64 and increment
    let value: u64 = digits
        .parse()
        .with_context(|| format!("Invalid IMSI format: {imsi}"))?;
    let new_value = value + offset;

    // Format back to 15 digits
    Ok(format!("{new_value:015}"))
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
            "-c",
            "config.yaml",
            "-i",
            "001010000000001",
            "-n",
            "5",
            "-t",
            "200",
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

    // -- SUCI encoding tests --
    //
    // The full SUCI wire-format coverage (null scheme byte layout, routing
    // indicator, ECIES Profile A/B concealment + TS 33.501 Annex C vectors,
    // error paths) lives in `nextgsim_ue::nas::mm::suci`; these verify the
    // delegation and the config-driven scheme selection.

    #[test]
    fn test_build_suci_from_config_null_scheme() {
        use nextgsim_common::config::{OpType, UeConfig};
        use nextgsim_common::types::{Plmn, Supi, SupiType};

        let config = UeConfig {
            hplmn: Plmn {
                mcc: 999,
                mnc: 70,
                long_mnc: false,
            },
            supi: Some(Supi {
                supi_type: SupiType::Imsi,
                value: "999700000000001".to_string(),
            }),
            key: [0x11; 16],
            op: [0x22; 16],
            op_type: OpType::Opc,
            ..Default::default()
        };
        let suci = build_suci_from_config(&config).expect("null-scheme SUCI");
        // type(1) | PLMN(3) | RI(2) | scheme(1) | key id(1) | MSIN TBCD(5)
        assert_eq!(
            suci,
            vec![0x01, 0x99, 0xF9, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10]
        );
    }

    #[test]
    fn test_build_suci_from_config_invalid_scheme_fails() {
        use nextgsim_common::config::{OpType, UeConfig};
        use nextgsim_common::types::Plmn;

        let config = UeConfig {
            hplmn: Plmn {
                mcc: 999,
                mnc: 70,
                long_mnc: false,
            },
            supi: None,
            protection_scheme: 7, // unsupported
            key: [0x11; 16],
            op: [0x22; 16],
            op_type: OpType::Opc,
            ..Default::default()
        };
        // No fallback to the null scheme on bad configuration: both the
        // builder and the startup validation must reject it
        assert!(build_suci_from_config(&config).is_err());
        assert!(validate_ue_config(&config).is_err());
    }

    #[test]
    fn test_build_suci_from_config_no_supi_fallback() {
        use nextgsim_common::config::{OpType, UeConfig};
        use nextgsim_common::types::Plmn;

        let config = UeConfig {
            hplmn: Plmn {
                mcc: 999,
                mnc: 70,
                long_mnc: false,
            },
            supi: None, // no SUPI
            key: [0x11; 16],
            op: [0x22; 16],
            op_type: OpType::Opc,
            ..Default::default()
        };
        // Without a SUPI the builder falls back to the default MSIN
        let suci = build_suci_from_config(&config).expect("default-MSIN SUCI");
        assert_eq!(
            &suci[8..],
            &[0x00, 0x00, 0x00, 0x00, 0x10],
            "default MSIN 0000000001 in TBCD"
        );
    }
}
