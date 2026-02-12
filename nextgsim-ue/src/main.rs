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
use nextgsim_ue::{
    AppMessage, AppTask, NasMessage, RlsMessage, RlsTask, RrcMessage,
    Task, TaskManager, TaskMessage, UeTaskBase, DEFAULT_CHANNEL_CAPACITY,
};
use nextgsim_ue::tun::{TunMessage, TunTask, TunTaskConfig, TunAppMessage};

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
        let node_name = format!("ue{instance_id}");

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
                        let _ = nas_tx_for_tun.send(NasMessage::InitiateServiceRequest).await;
                        // Forward uplink data to RLS for transmission to gNB
                        let _ = rls_tx_for_tun.send(RlsMessage::DataPduDelivery { psi, pdu: data }).await;
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
        tokio::spawn(async move {
            Self::run_nas_task(nas_task_base, nas_rx, tun_tx).await
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
        task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>,
        tun_tx: tokio::sync::mpsc::Sender<TunMessage>,
    ) {
        use nextgsim_ue::nas::mm::{MmStateMachine, MmSubState};
        use nextgsim_nas::messages::mm::{RegistrationRequest, Ie5gsMobileIdentity, MobileIdentityType};
        use nextgsim_nas::messages::mm::{IdentityRequest, IdentityResponse};
        use nextgsim_nas::messages::mm::authentication::{AuthenticationRequest as NasAuthRequest, AuthenticationResponse as NasAuthResponse};
        use nextgsim_nas::messages::mm::security_mode::{SecurityModeCommand as NasSecModeCmd, SecurityModeComplete as NasSecModeComplete};
        use nextgsim_nas::ies::{Ie5gsRegistrationType, FollowOnRequest, RegistrationType};
        use nextgsim_nas::security::NasKeySetIdentifier;
        use nextgsim_nas::enums::{MmMessageType, SmMessageType};
        use nextgsim_crypto::milenage::{Milenage, compute_opc};
        use nextgsim_crypto::kdf::derive_res_star;
        use nextgsim_common::config::OpType;

        use bytes::BufMut;

        info!("NAS task started");

        // Initialize MM state machine
        let mut mm_state = MmStateMachine::new();
        let mut pdu_counter: u32 = 0;
        let mut registration_sent = false;
        let mut pdu_session_requested = false;
        let mut pti_counter: u8 = 1; // Procedure Transaction Identity counter

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NasMessage::NasNotify => {
                            info!("NAS notify received");
                        }
                        NasMessage::NasDelivery { pdu } => {
                            info!("NAS delivery: len={}", pdu.len());

                            // Decode the NAS message based on EPD
                            if pdu.len() >= 4 {
                                let epd = pdu.data()[0];

                                // Check if this is a 5GMM message (EPD=0x7E) or 5GSM message (EPD=0x2E)
                                if epd == 0x2E {
                                    // 5GSM (Session Management) message
                                    // Format: EPD (0x2E) + PSI + PTI + Message Type + IEs
                                    let psi = pdu.data()[1];
                                    let pti = pdu.data()[2];
                                    let sm_msg_type = pdu.data()[3];

                                    info!("Received 5GSM message: PSI={}, PTI={}, type=0x{:02x}", psi, pti, sm_msg_type);

                                    if let Ok(sm_type) = SmMessageType::try_from(sm_msg_type) {
                                        match sm_type {
                                            SmMessageType::PduSessionEstablishmentAccept => {
                                                info!("PDU Session Establishment Accept received!");
                                                // Parse IP address from the response (simplified)
                                                // In real implementation, decode QoS rules, session AMBR, etc.
                                                let mut ue_ip: Option<Ipv4Addr> = None;
                                                if pdu.len() > 8 {
                                                    // Look for PDU address IE (IEI = 0x29)
                                                    let pdu_data = pdu.data();
                                                    for i in 4..pdu.len().saturating_sub(5) {
                                                        if pdu_data[i] == 0x29 {
                                                            // PDU address IE found
                                                            let len = pdu_data[i + 1] as usize;
                                                            if len >= 5 && i + 2 + len <= pdu.len() {
                                                                let pdu_type = pdu_data[i + 2];
                                                                if pdu_type == 0x01 { // IPv4
                                                                    ue_ip = Some(Ipv4Addr::new(
                                                                        pdu_data[i + 3],
                                                                        pdu_data[i + 4],
                                                                        pdu_data[i + 5],
                                                                        pdu_data[i + 6]));
                                                                    info!("PDU Session {} established with IP: {}", psi, ue_ip.unwrap());
                                                                }
                                                            }
                                                            break;
                                                        }
                                                    }
                                                }
                                                info!("PDU Session {} is now ACTIVE", psi);

                                                // Create TUN interface for this PDU session
                                                if let Some(ip) = ue_ip {
                                                    info!("Creating TUN interface for PDU Session {} with IP {}", psi, ip);
                                                    let _ = tun_tx.send(TunMessage::CreateInterface {
                                                        psi: psi as i32,
                                                        address: ip,
                                                        netmask: Ipv4Addr::new(255, 255, 255, 0),
                                                    }).await;
                                                }
                                            }
                                            SmMessageType::PduSessionEstablishmentReject => {
                                                let cause = if pdu.len() > 4 { pdu.data()[4] } else { 0 };
                                                warn!("PDU Session Establishment Reject: cause={}", cause);
                                            }
                                            SmMessageType::PduSessionReleaseCommand => {
                                                // Network requests PDU session release
                                                info!("PDU Session Release Command received: PSI={}", psi);

                                                // Deactivate TUN interface for this session
                                                let _ = tun_tx.send(TunMessage::WriteData {
                                                    psi: psi as i32,
                                                    data: vec![].into(), // empty = signal close
                                                }).await;

                                                // Send PDU Session Release Complete
                                                let release_complete = nextgsim_nas::messages::sm::PduSessionReleaseComplete::new(psi, pti);
                                                let mut nas_pdu = Vec::new();
                                                release_complete.encode(&mut nas_pdu);

                                                info!("Sending PDU Session Release Complete: PSI={}, len={}", psi, nas_pdu.len());

                                                pdu_counter += 1;
                                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                    pdu_id: pdu_counter,
                                                    pdu: nas_pdu.into(),
                                                }).await;

                                                if psi == 1 {
                                                    pdu_session_requested = false;
                                                }
                                            }
                                            SmMessageType::PduSessionModificationCommand => {
                                                // Network modifies PDU session parameters
                                                info!("PDU Session Modification Command received: PSI={}", psi);

                                                // Send PDU Session Modification Complete
                                                let mod_complete = nextgsim_nas::messages::sm::PduSessionModificationComplete::new(psi, pti);
                                                let mut nas_pdu = Vec::new();
                                                mod_complete.encode(&mut nas_pdu);

                                                info!("Sending PDU Session Modification Complete: PSI={}, len={}", psi, nas_pdu.len());

                                                pdu_counter += 1;
                                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                    pdu_id: pdu_counter,
                                                    pdu: nas_pdu.into(),
                                                }).await;
                                            }
                                            SmMessageType::PduSessionReleaseReject => {
                                                warn!("PDU Session Release Reject: PSI={}", psi);
                                            }
                                            SmMessageType::PduSessionModificationReject => {
                                                warn!("PDU Session Modification Reject: PSI={}", psi);
                                            }
                                            _ => {
                                                info!("Unhandled 5GSM message type: {:?}", sm_type);
                                            }
                                        }
                                    } else {
                                        warn!("Unknown 5GSM message type: 0x{:02x}", sm_msg_type);
                                    }
                                } else if epd == 0x7E {
                                    // 5GMM (Mobility Management) message
                                    let msg_type_byte = pdu.data()[2];
                                    if let Ok(msg_type) = MmMessageType::try_from(msg_type_byte) {
                                        info!("Received NAS message type: {:?} (0x{:02x})", msg_type, msg_type_byte);

                                    match msg_type {
                                        MmMessageType::IdentityRequest => {
                                            // Decode Identity Request
                                            let mut buf = pdu.data();
                                            match IdentityRequest::decode(&mut buf) {
                                                Ok(id_req) => {
                                                    info!("Identity Request - Type: {:?}", id_req.identity_type.value);

                                                    // Build SUCI for Identity Response
                                                    let suci_data = build_suci_null_scheme(
                                                        "999", "70", // PLMN MCC/MNC
                                                        "0000000001" // MSIN
                                                    );
                                                    let mobile_identity = Ie5gsMobileIdentity::new(
                                                        MobileIdentityType::Suci,
                                                        suci_data,
                                                    );

                                                    // Build Identity Response
                                                    let id_response = IdentityResponse::new(mobile_identity);
                                                    let mut nas_pdu = Vec::new();
                                                    id_response.encode(&mut nas_pdu);

                                                    info!("Sending Identity Response, PDU len={}", nas_pdu.len());

                                                    // Send to RRC for transmission
                                                    pdu_counter += 1;
                                                    let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                        pdu_id: pdu_counter,
                                                        pdu: nas_pdu.into(),
                                                    }).await;
                                                }
                                                Err(e) => {
                                                    warn!("Failed to decode Identity Request: {:?}", e);
                                                }
                                            }
                                        }
                                        MmMessageType::AuthenticationRequest => {
                                            // Decode Authentication Request (skip 3-byte header)
                                            let header_len = 3; // EPD + SecHdr + MsgType
                                            match NasAuthRequest::decode(&mut &pdu.data()[header_len..]) {
                                                Ok(auth_req) => {
                                                    info!("Authentication Request received: ngKSI={}", auth_req.ng_ksi.ksi);

                                                    if let (Some(rand_ie), Some(autn_ie)) = (&auth_req.rand, &auth_req.autn) {
                                                        // 5G-AKA authentication
                                                        let rand = &rand_ie.value;
                                                        let autn = &autn_ie.value;
                                                        info!("5G-AKA: RAND={:02x?}, AUTN len={}", &rand[..4], autn.len());

                                                        // Get K and OPc from UE config
                                                        let config = &task_base.config;
                                                        let opc = match config.op_type {
                                                            OpType::Opc => config.op,
                                                            OpType::Op => compute_opc(&config.key, &config.op),
                                                        };

                                                        // Run Milenage: compute RES, CK, IK, AK
                                                        let m = Milenage::new(&config.key, &opc);
                                                        let res = m.f2(rand);
                                                        let ck = m.f3(rand);
                                                        let ik = m.f4(rand);
                                                        let ak = m.f5(rand);

                                                        // Verify AUTN: AUTN = SQN⊕AK || AMF || MAC
                                                        if autn.len() >= 16 {
                                                            // Extract SQN⊕AK, AMF, MAC from AUTN
                                                            let mut sqn_xor_ak = [0u8; 6];
                                                            sqn_xor_ak.copy_from_slice(&autn[0..6]);
                                                            let mut amf_from_autn = [0u8; 2];
                                                            amf_from_autn.copy_from_slice(&autn[6..8]);
                                                            let mac_from_autn = &autn[8..16];

                                                            // Recover SQN = (SQN⊕AK) ⊕ AK
                                                            let mut sqn = [0u8; 6];
                                                            for i in 0..6 {
                                                                sqn[i] = sqn_xor_ak[i] ^ ak[i];
                                                            }

                                                            // Verify MAC: expected_mac = f1(K, RAND, SQN, AMF)
                                                            let expected_mac = m.f1(rand, &sqn, &amf_from_autn);
                                                            if expected_mac == mac_from_autn {
                                                                info!("AUTN MAC verified successfully");

                                                                // Compute RES* = KDF(CK||IK, FC=0x6B, SN_name, RAND, RES)
                                                                let sn_name = format!(
                                                                    "5G:mnc{:03}.mcc{:03}.3gppnetwork.org",
                                                                    config.hplmn.mnc, config.hplmn.mcc
                                                                );
                                                                let res_star = derive_res_star(
                                                                    &ck, &ik,
                                                                    sn_name.as_bytes(),
                                                                    rand,
                                                                    &res,
                                                                );
                                                                info!("RES* computed: {:02x?}", &res_star[..4]);

                                                                // Build and send Authentication Response
                                                                let auth_response = NasAuthResponse::with_res_star(res_star.to_vec());
                                                                let mut nas_pdu = Vec::new();
                                                                auth_response.encode(&mut nas_pdu);

                                                                info!("Sending Authentication Response, PDU len={}", nas_pdu.len());
                                                                pdu_counter += 1;
                                                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                                    pdu_id: pdu_counter,
                                                                    pdu: nas_pdu.into(),
                                                                }).await;
                                                            } else {
                                                                warn!("AUTN MAC verification failed! expected={:02x?}, got={:02x?}",
                                                                    expected_mac, mac_from_autn);
                                                            }
                                                        } else {
                                                            warn!("AUTN too short: {} bytes", autn.len());
                                                        }
                                                    } else {
                                                        warn!("Authentication Request missing RAND or AUTN (EAP-AKA not supported)");
                                                    }
                                                }
                                                Err(e) => {
                                                    warn!("Failed to decode Authentication Request: {:?}", e);
                                                }
                                            }
                                        }
                                        MmMessageType::SecurityModeCommand => {
                                            // Decode Security Mode Command (skip 3-byte header)
                                            let header_len = 3;
                                            match NasSecModeCmd::decode(&mut &pdu.data()[header_len..]) {
                                                Ok(smc) => {
                                                    info!(
                                                        "Security Mode Command: enc_alg={}, int_alg={}, ngKSI={}",
                                                        smc.selected_nas_security_algorithms.ciphering,
                                                        smc.selected_nas_security_algorithms.integrity,
                                                        smc.ng_ksi.ksi
                                                    );

                                                    // Send Security Mode Complete
                                                    let smc_complete = NasSecModeComplete::new();
                                                    let mut nas_pdu = Vec::new();
                                                    smc_complete.encode(&mut nas_pdu);

                                                    info!("Sending Security Mode Complete, PDU len={}", nas_pdu.len());
                                                    pdu_counter += 1;
                                                    let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                        pdu_id: pdu_counter,
                                                        pdu: nas_pdu.into(),
                                                    }).await;
                                                }
                                                Err(e) => {
                                                    warn!("Failed to decode Security Mode Command: {:?}", e);
                                                }
                                            }
                                        }
                                        MmMessageType::RegistrationAccept => {
                                            info!("Received Registration Accept!");
                                            mm_state.switch_mm_state(MmSubState::Registered);
                                            info!("UE is now REGISTERED, MM state: {}", mm_state);

                                            // Trigger PDU Session Establishment after registration
                                            if !pdu_session_requested {
                                                tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
                                                info!("Initiating PDU Session Establishment");

                                                // Build PDU Session Establishment Request
                                                // Format: SM Header (4 bytes) + IEs
                                                let psi: u8 = 1; // PDU Session Identity
                                                let pti = pti_counter;
                                                pti_counter = pti_counter.wrapping_add(1);
                                                if pti_counter == 0 { pti_counter = 1; }

                                                let mut nas_pdu = Vec::new();

                                                // SM Header: EPD (0x2E) + PSI + PTI + Message Type (0xC1)
                                                nas_pdu.put_u8(0x2E); // EPD: 5GSM
                                                nas_pdu.put_u8(psi);  // PDU Session ID
                                                nas_pdu.put_u8(pti);  // PTI
                                                nas_pdu.put_u8(0xC1); // Message Type: PDU Session Establishment Request

                                                // Mandatory IE: Integrity protection maximum data rate (9.11.4.7)
                                                // IEI is not present for mandatory IEs
                                                nas_pdu.put_u8(0xFF); // Max data rate UL: full rate
                                                nas_pdu.put_u8(0xFF); // Max data rate DL: full rate

                                                // Optional IE: PDU session type (9.11.4.11)
                                                nas_pdu.put_u8(0x91); // IEI for PDU session type
                                                nas_pdu.put_u8(0x01); // IPv4

                                                // Optional IE: SSC mode (9.11.4.16)
                                                nas_pdu.put_u8(0xA1); // IEI for SSC mode
                                                nas_pdu.put_u8(0x01); // SSC mode 1

                                                info!("Sending PDU Session Establishment Request: PSI={}, PTI={}, len={}",
                                                      psi, pti, nas_pdu.len());

                                                pdu_session_requested = true;

                                                // Send to RRC for transmission
                                                pdu_counter += 1;
                                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                                    pdu_id: pdu_counter,
                                                    pdu: nas_pdu.into(),
                                                }).await;
                                            }
                                        }
                                        MmMessageType::RegistrationReject => {
                                            warn!("Received Registration Reject");
                                            mm_state.switch_mm_state(MmSubState::Deregistered);
                                            registration_sent = false;
                                        }
                                        MmMessageType::ServiceAccept => {
                                            info!("Service Accept received - UE is now CONNECTED");
                                            mm_state.switch_mm_state(MmSubState::Registered);
                                            mm_state.switch_cm_state(nextgsim_ue::nas::mm::CmState::Connected);
                                        }
                                        MmMessageType::ServiceReject => {
                                            let cause = if pdu.len() > 3 { pdu.data()[3] } else { 0 };
                                            warn!("Service Reject received: cause={}", cause);
                                            mm_state.switch_mm_state(MmSubState::Registered);
                                        }
                                        MmMessageType::DlNasTransport => {
                                            // DL NAS Transport contains an embedded SM message
                                            // Parse payload container to extract inner NAS PDU
                                            if pdu.len() > 7 {
                                                let header_len = 3; // EPD + SecHdr + MsgType
                                                let payload_type = pdu.data()[header_len]; // Payload container type
                                                if payload_type == 0x01 { // N1 SM information
                                                    let container_len = u16::from_be_bytes([
                                                        pdu.data()[header_len + 1],
                                                        pdu.data()[header_len + 2],
                                                    ]) as usize;
                                                    if container_len > 0 && header_len + 3 + container_len <= pdu.len() {
                                                        // Forward the embedded SM PDU back through NAS delivery
                                                        let inner_pdu = pdu.data()[header_len + 3..header_len + 3 + container_len].to_vec();
                                                        info!("DL NAS Transport: forwarding embedded SM PDU, len={}", inner_pdu.len());
                                                        let _ = task_base.nas_tx.send(
                                                            NasMessage::NasDelivery { pdu: inner_pdu.into() }
                                                        ).await;
                                                    }
                                                }
                                            }
                                        }
                                        _ => {
                                            info!("Unhandled NAS message type: {:?}", msg_type);
                                        }
                                    }
                                    } else {
                                        warn!("Unknown NAS message type: 0x{:02x}", msg_type_byte);
                                    }
                                } else {
                                    warn!("Unknown EPD: 0x{:02x}", epd);
                                }
                            } else {
                                warn!("NAS PDU too short: {} bytes", pdu.len());
                            }
                        }
                        NasMessage::RrcConnectionSetup => {
                            info!("RRC connection setup complete");
                            mm_state.switch_cm_state(nextgsim_ue::nas::mm::CmState::Connected);
                        }
                        NasMessage::RrcConnectionRelease => {
                            info!("RRC connection released");
                            mm_state.switch_cm_state(nextgsim_ue::nas::mm::CmState::Idle);
                            registration_sent = false;
                        }
                        NasMessage::RrcEstablishmentFailure => {
                            warn!("RRC establishment failure");
                        }
                        NasMessage::RadioLinkFailure => {
                            warn!("Radio link failure");
                            mm_state.switch_cm_state(nextgsim_ue::nas::mm::CmState::Idle);
                            registration_sent = false;
                        }
                        NasMessage::Paging { paging_tmsi } => {
                            info!("Paging received: {} TMSIs", paging_tmsi.len());

                            // If registered and CM-IDLE, send Service Request to transition to CONNECTED
                            if mm_state.is_registered() && mm_state.is_idle() {
                                info!("Sending Service Request (paging-triggered)");

                                use nextgsim_nas::messages::mm::ServiceRequest;
                                use nextgsim_nas::ies::ie1::{IeServiceType, ServiceType};

                                // Build 5G-S-TMSI: type (1) + AMF Set ID/Pointer (2) + TMSI (4)
                                let tmsi_data = if let Some(first) = paging_tmsi.first() {
                                    let mut data = vec![0xF4]; // TMSI type indicator
                                    // AMF Set ID (10 bits) + AMF Pointer (6 bits) = 2 bytes
                                    let set_ptr = ((first.amf_set_id & 0x3FF) << 6) | (first.amf_pointer as u16 & 0x3F);
                                    data.extend_from_slice(&set_ptr.to_be_bytes());
                                    data.extend_from_slice(&first.tmsi.to_be_bytes());
                                    data
                                } else {
                                    vec![0xF4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]
                                };

                                let tmsi = Ie5gsMobileIdentity::new(
                                    MobileIdentityType::Tmsi,
                                    tmsi_data,
                                );

                                let svc_req = ServiceRequest::new(
                                    NasKeySetIdentifier::no_key(),
                                    IeServiceType::new(ServiceType::MobileTerminatedServices),
                                    tmsi,
                                );

                                let mut nas_pdu = Vec::new();
                                svc_req.encode(&mut nas_pdu);

                                info!("Sending Service Request, PDU len={}", nas_pdu.len());
                                mm_state.switch_mm_state(MmSubState::ServiceRequestInitiated);

                                pdu_counter += 1;
                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                    pdu_id: pdu_counter,
                                    pdu: nas_pdu.into(),
                                }).await;
                            }
                        }
                        NasMessage::ActiveCellChanged { previous_tai } => {
                            info!("Active cell changed from TAI: {:?}", previous_tai);
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
                            if mm_state.is_registered() && mm_state.is_idle() {
                                info!("Initiating Service Request (data-triggered, IDLE -> CONNECTED)");

                                use nextgsim_nas::messages::mm::ServiceRequest;
                                use nextgsim_nas::ies::ie1::{IeServiceType, ServiceType};

                                // Build 5G-S-TMSI with default values (real implementation
                                // would use the stored GUTI from registration)
                                let tmsi_data = vec![0xF4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00];

                                let tmsi = Ie5gsMobileIdentity::new(
                                    MobileIdentityType::Tmsi,
                                    tmsi_data,
                                );

                                let svc_req = ServiceRequest::new(
                                    NasKeySetIdentifier::no_key(),
                                    IeServiceType::new(ServiceType::Data),
                                    tmsi,
                                );

                                let mut nas_pdu = Vec::new();
                                svc_req.encode(&mut nas_pdu);

                                info!("Sending Service Request (data), PDU len={}", nas_pdu.len());
                                mm_state.switch_mm_state(MmSubState::ServiceRequestInitiated);

                                pdu_counter += 1;
                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                    pdu_id: pdu_counter,
                                    pdu: nas_pdu.into(),
                                }).await;
                            } else if mm_state.is_registered() && mm_state.is_connected() {
                                debug!("Service Request not needed: already in CM-CONNECTED");
                            } else {
                                warn!("Cannot send Service Request: not registered (RM={}, CM={})",
                                    mm_state.rm_state(), mm_state.cm_state());
                            }
                        }
                        NasMessage::PerformMmCycle => {
                            info!("PerformMmCycle received, MM state: {}", mm_state);

                            // If we're deregistered and haven't sent registration, send it
                            if mm_state.is_deregistered() && !registration_sent {
                                // Wait a bit for gNB to complete NG Setup with AMF
                                tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
                                info!("Initiating initial registration");

                                // Build SUCI (SUPI Concealed Identity) - for testing, use null scheme
                                // Format: Type (SUCI=0x01) | PLMN | Routing Indicator | Scheme | Scheme Output
                                let suci_data = build_suci_null_scheme(
                                    "999", "70", // PLMN MCC/MNC
                                    "0000000001" // MSIN
                                );
                                let mobile_identity = Ie5gsMobileIdentity::new(
                                    MobileIdentityType::Suci,
                                    suci_data,
                                );

                                // Build Registration Request
                                let reg_request = RegistrationRequest::new(
                                    Ie5gsRegistrationType::new(FollowOnRequest::NoPending, RegistrationType::InitialRegistration),
                                    NasKeySetIdentifier::no_key(),
                                    mobile_identity,
                                );

                                // Encode the Registration Request
                                let mut nas_pdu = Vec::new();
                                reg_request.encode(&mut nas_pdu);

                                info!("Sending Registration Request, PDU len={}", nas_pdu.len());

                                // Update state
                                mm_state.switch_mm_state(MmSubState::RegisteredInitiated);
                                registration_sent = true;

                                // Send to RRC for transmission
                                pdu_counter += 1;
                                let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                    pdu_id: pdu_counter,
                                    pdu: nas_pdu.into(),
                                }).await;
                            }
                        }
                        NasMessage::NasTimerExpire { timer_id } => {
                            info!("NAS timer {} expired", timer_id);
                        }
                        NasMessage::InitiatePduSessionEstablishment { psi, pti, session_type, apn } => {
                            info!(
                                "Initiating PDU session establishment: PSI={}, PTI={}, type={}, apn={:?}",
                                psi, pti, session_type, apn
                            );

                            // Build PDU Session Establishment Request NAS PDU
                            let mut nas_pdu = Vec::new();

                            // SM Header: EPD (0x2E) + PSI + PTI + Message Type (0xC1)
                            nas_pdu.put_u8(0x2E); // EPD: 5GSM
                            nas_pdu.put_u8(psi);  // PDU Session ID
                            nas_pdu.put_u8(pti);  // PTI
                            nas_pdu.put_u8(0xC1); // Message Type: PDU Session Establishment Request

                            // Mandatory IE: Integrity protection maximum data rate (9.11.4.7)
                            nas_pdu.put_u8(0xFF); // Max data rate UL: full rate
                            nas_pdu.put_u8(0xFF); // Max data rate DL: full rate

                            // Optional IE: PDU session type (9.11.4.11)
                            let pdu_type = match session_type.as_str() {
                                "IPv6" => 0x02,
                                "IPv4v6" => 0x03,
                                _ => 0x01, // IPv4
                            };
                            nas_pdu.put_u8(0x91); // IEI for PDU session type
                            nas_pdu.put_u8(pdu_type);

                            // Optional IE: SSC mode (9.11.4.16)
                            nas_pdu.put_u8(0xA1); // IEI for SSC mode
                            nas_pdu.put_u8(0x01); // SSC mode 1

                            info!("Sending PDU Session Establishment Request: PSI={}, PTI={}, type={}, len={}",
                                  psi, pti, session_type, nas_pdu.len());

                            pdu_counter += 1;
                            let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                pdu_id: pdu_counter,
                                pdu: nas_pdu.into(),
                            }).await;
                        }
                        NasMessage::InitiatePduSessionRelease { psi, pti } => {
                            info!("Initiating PDU session release: PSI={}, PTI={}", psi, pti);

                            // Build PDU Session Release Request NAS PDU
                            let release_req = nextgsim_nas::messages::sm::PduSessionReleaseRequest::new(psi, pti);
                            let mut nas_pdu = Vec::new();
                            release_req.encode(&mut nas_pdu);

                            info!("Sending PDU Session Release Request: PSI={}, PTI={}, len={}", psi, pti, nas_pdu.len());

                            pdu_counter += 1;
                            let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                pdu_id: pdu_counter,
                                pdu: nas_pdu.into(),
                            }).await;
                        }
                        NasMessage::InitiateDeregistration { switch_off } => {
                            info!("Initiating deregistration: switch_off={}", switch_off);

                            use nextgsim_nas::messages::mm::DeregistrationRequestUeOriginating;
                            use nextgsim_nas::ies::ie1::{
                                IeDeRegistrationType, DeRegistrationAccessType,
                                SwitchOff as NasSwitchOff, ReRegistrationRequired,
                            };

                            let switch_off_val = if switch_off {
                                NasSwitchOff::SwitchOff
                            } else {
                                NasSwitchOff::NormalDeRegistration
                            };

                            let dereg_type = IeDeRegistrationType::new(
                                DeRegistrationAccessType::ThreeGppAccess,
                                ReRegistrationRequired::NotRequired,
                                switch_off_val,
                            );

                            let suci_data = build_suci_null_scheme("999", "70", "0000000001");
                            let mobile_identity = Ie5gsMobileIdentity::new(
                                MobileIdentityType::Suci,
                                suci_data,
                            );

                            let dereg_req = DeregistrationRequestUeOriginating::new(
                                dereg_type,
                                NasKeySetIdentifier::no_key(),
                                mobile_identity,
                            );

                            let mut nas_pdu = Vec::new();
                            dereg_req.encode(&mut nas_pdu);

                            info!("Sending Deregistration Request: switch_off={}, len={}", switch_off, nas_pdu.len());

                            mm_state.switch_mm_state(MmSubState::DeregisteredInitiated);

                            pdu_counter += 1;
                            let _ = task_base.rrc_tx.send(RrcMessage::UplinkNasDelivery {
                                pdu_id: pdu_counter,
                                pdu: nas_pdu.into(),
                            }).await;
                        }
                        NasMessage::DownlinkDataDelivery { psi, data } => {
                            // Forward downlink data to TUN interface
                            debug!("Downlink data delivery: psi={}, len={}", psi, data.len());
                            let _ = tun_tx.send(TunMessage::WriteData {
                                psi,
                                data,
                            }).await;
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
        task_base: UeTaskBase,
        mut rx: tokio::sync::mpsc::Receiver<TaskMessage<RrcMessage>>,
    ) {
        use std::collections::HashMap;
        use nextgsim_ue::rrc::{RrcStateMachine, RrcStateTransition, RrcState};
        use nextgsim_common::OctetString;

        info!("RRC task started");

        // Cell tracking
        let mut discovered_cells: HashMap<i32, i32> = HashMap::new(); // cell_id -> dbm
        let mut serving_cell: Option<i32> = None;
        let mut rrc_state_machine = RrcStateMachine::new();
        let mut registration_triggered = false;

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

                                let _ = task_base.rls_tx.send(RlsMessage::RrcPduDelivery {
                                    channel: RrcChannel::UlDcch,
                                    pdu_id,
                                    pdu: wrapped_pdu,
                                }).await;
                            } else {
                                warn!("Cannot send uplink NAS: no serving cell");
                            }
                        }
                        RrcMessage::RrcNotify => {
                            info!("RRC notify received");
                        }
                        RrcMessage::PerformUac { access_category, access_identities } => {
                            info!("Perform UAC: category={}, identities={}", access_category, access_identities);
                        }
                        RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu } => {
                            info!("Downlink RRC: cell_id={}, channel={:?}, len={}", cell_id, channel, pdu.len());

                            // Receiving a downlink RRC message means RRC connection is established
                            // Transition to Connected state if we're in Idle
                            if rrc_state_machine.state().is_idle()
                                && rrc_state_machine.can_transition(RrcStateTransition::SetupComplete) {
                                    let _ = rrc_state_machine.transition(RrcStateTransition::SetupComplete);
                                    info!("RRC state: {} -> {} (connection established)",
                                          RrcState::Idle, rrc_state_machine.state());
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
                            let _ = task_base.nas_tx.send(NasMessage::NasDelivery { pdu: nas_pdu }).await;
                        }
                        RrcMessage::SignalChanged { cell_id, dbm } => {
                            info!("Signal changed: cell_id={}, dbm={}", cell_id, dbm);

                            // Track the cell
                            discovered_cells.insert(cell_id, dbm);

                            // If we don't have a serving cell yet and we're in IDLE state, select one
                            if serving_cell.is_none() && rrc_state_machine.state().is_idle() {
                                // Select the cell with best signal (or just the first one for now)
                                let best_cell = discovered_cells.iter()
                                    .max_by_key(|(_, &signal)| signal)
                                    .map(|(&id, _)| id);

                                if let Some(best_cell_id) = best_cell {
                                    info!("Selecting cell {} as serving cell", best_cell_id);
                                    serving_cell = Some(best_cell_id);

                                    // Tell RLS to camp on this cell
                                    let _ = task_base.rls_tx.send(RlsMessage::AssignCurrentCell {
                                        cell_id: best_cell_id
                                    }).await;

                                    // Trigger registration if not already done
                                    if !registration_triggered {
                                        info!("Triggering NAS registration procedure");
                                        registration_triggered = true;
                                        let _ = task_base.nas_tx.send(NasMessage::PerformMmCycle).await;
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
                            if rrc_state_machine.can_transition(RrcStateTransition::RadioLinkFailure) {
                                let _ = rrc_state_machine.transition(RrcStateTransition::RadioLinkFailure);
                            }
                            // Notify NAS of the failure
                            let _ = task_base.nas_tx.send(NasMessage::RadioLinkFailure).await;
                        }
                        RrcMessage::TriggerCycle => {
                            // Trigger RRC state machine cycle
                        }
                        RrcMessage::NtnTimingAdvanceReceived { common_ta_us, k_offset, .. } => {
                            info!("UE: NTN TA={}us, k_offset={}", common_ta_us, k_offset);
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

/// Build SUCI with null protection scheme (scheme ID 0)
/// Returns the raw SUCI data bytes for the 5GS Mobile Identity IE
///
/// Format: Type octet | SUPI Format | PLMN | Routing Indicator | Protection Scheme | Home Network Public Key ID | Scheme Output (MSIN)
fn build_suci_null_scheme(mcc: &str, mnc: &str, msin: &str) -> Vec<u8> {
    let mut data = Vec::new();

    // Type octet: SUCI = 0x01, SUPI format = IMSI (0)
    // bits 2-0: type (001 = SUCI)
    // bits 6-4: SUPI format (000 = IMSI)
    // bit 3: spare
    // bit 7: spare
    data.push(0x01);

    // PLMN (MCC + MNC) in BCD format (3 bytes)
    // MCC digit 2 | MCC digit 1
    // MNC digit 3 | MCC digit 3
    // MNC digit 2 | MNC digit 1
    let mcc_bytes: Vec<u8> = mcc.chars().filter_map(|c| c.to_digit(10).map(|d| d as u8)).collect();
    let mnc_bytes: Vec<u8> = mnc.chars().filter_map(|c| c.to_digit(10).map(|d| d as u8)).collect();

    // MCC digit 2 (high nibble) | MCC digit 1 (low nibble)
    let byte1 = (mcc_bytes.get(1).copied().unwrap_or(0xF) << 4) | mcc_bytes.first().copied().unwrap_or(0xF);
    data.push(byte1);

    // MNC digit 3 (high nibble) | MCC digit 3 (low nibble)
    // For 2-digit MNC, digit 3 is 0xF
    let mnc_digit3 = if mnc_bytes.len() > 2 { mnc_bytes[2] } else { 0xF };
    let byte2 = (mnc_digit3 << 4) | mcc_bytes.get(2).copied().unwrap_or(0xF);
    data.push(byte2);

    // MNC digit 2 (high nibble) | MNC digit 1 (low nibble)
    let byte3 = (mnc_bytes.get(1).copied().unwrap_or(0xF) << 4) | mnc_bytes.first().copied().unwrap_or(0xF);
    data.push(byte3);

    // Routing indicator (4 digits in BCD, 2 bytes) - use 0000 for default
    data.push(0x00); // digits 2,1
    data.push(0xF0); // digits 4,3 (F = filler)

    // Protection scheme ID (4 bits) + Home network public key ID (4 bits)
    // Null scheme = 0, key ID = 0
    data.push(0x00);

    // Scheme output = MSIN in BCD (for null scheme, this is the unprotected MSIN)
    // MSIN digits in BCD pairs
    let msin_bytes: Vec<u8> = msin.chars().filter_map(|c| c.to_digit(10).map(|d| d as u8)).collect();
    for chunk in msin_bytes.chunks(2) {
        let low = chunk.first().copied().unwrap_or(0xF);
        let high = chunk.get(1).copied().unwrap_or(0xF);
        data.push((high << 4) | low);
    }

    data
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
    let digits = if let Some(stripped) = imsi.strip_prefix("imsi-") {
        stripped
    } else {
        imsi
    };

    // Parse as u64 and increment
    let value: u64 = digits.parse()
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
