//! NGAP Task Implementation
//!
//! This module implements the NGAP task for the gNB. The NGAP task handles:
//! - NG Setup procedure with AMF(s)
//! - UE context management
//! - NAS message routing between RRC and AMF
//! - PDU session resource management
//!
//! # Message Flow
//!
//! ```text
//! SCTP Task ---> NGAP Task ---> RRC Task (NAS delivery)
//!                    |
//!                    +-------> GTP Task (PDU sessions)
//!                    |
//!                    +-------> App Task (status updates)
//! ```

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::tasks::{
    AppMessage, GnbTaskBase, GtpMessage, GtpUeContextUpdate, HandoverInitiation, NgapMessage,
    PduSessionResource, RrcMessage, SctpMessage, StatusType, StatusUpdate, Task, TaskMessage,
    UeReleaseRequestCause, NGAP_PPID,
};
use nextgsim_common::OctetString;

use super::amf_context::{AmfIdentity, AmfState, NgapAmfContext};
use super::mbs_context::{GnbMbsContext, MbsSessionManager, MulticastTunnelInfo, Tmgi};
use super::ue_context::{AsSecurityContext, NgapPduSession, NgapUeContext};
use super::up_security::{self, DrbSecurityDecision, UpSecurityRefusal};
use crate::mbs_ngap::{GnbMbsSession, NgapMbsManager};
use crate::rrc::meas::a3_meas_config_params;
use crate::rrc::system_info::sib19_params;
use crate::rrc::transaction::RrcProcedure;
use nextgsim_ngap::procedures::mbs::{
    encode_multicast_session_activation_response, encode_multicast_session_deactivation_response,
    parse_multicast_group_paging, parse_multicast_session_activation_request,
    parse_multicast_session_deactivation_request, MbsSessionId,
};
use nextgsim_rrc::procedures::handover_preparation::{
    encode_handover_preparation_information, HandoverPreparationParams,
};
// The single-DRB builder is not imported here: `establish_drb` goes through
// `build_multi_drb_reconfiguration_params` for one bearer as well as two (issue #44),
// so the only remaining caller of the single-DRB one is a test, which imports it
// itself rather than leaving an unused name in the production scope.
use nextgsim_rrc::procedures::rrc_reconfiguration::{
    build_multi_drb_reconfiguration_params, encode_rrc_reconfiguration, DrbIntegrityProtection,
    DrbSpec, RrcReconfigurationParams,
};
use nextgsim_rrc::procedures::rrc_reestablishment::{phys_cell_id_from_nci, SIMULATED_C_RNTI};
use nextgsim_rrc::procedures::security_mode::{
    encode_security_mode_command, CipheringAlgorithmType, IntegrityAlgorithmType,
    SecurityAlgorithms, SecurityModeCommandParams,
};

use nextgsim_ngap::codec::{decode_ngap_pdu, encode_ngap_pdu, NGAP_PDU};
use nextgsim_ngap::procedures::amf_status_indication::{
    decode_amf_status_indication, AmfStatusIndicationData,
};
use std::time::{Duration, Instant};

use crate::ngap::timers::{GuardTimer, GuardTimers};
use nextgsim_ngap::procedures::error_indication::{
    decode_error_indication, encode_error_indication, error_indication_abstract_syntax_error,
    error_indication_transfer_syntax_error, CriticalityDiagnosticsInfo, ErrorIndicationData,
    ErrorIndicationParams, TriggeringMessageValue,
};
use nextgsim_ngap::procedures::handover::{
    decode_handover_command, decode_handover_preparation_failure, decode_handover_request,
    encode_handover_cancel, encode_handover_notify, encode_handover_request_acknowledge,
    encode_handover_required, encode_source_to_target_container, HandoverCancelParams,
    HandoverCause, HandoverCommandData, HandoverNotifyParams, HandoverPreparationFailureData,
    HandoverRequestAcknowledgeParams, HandoverRequestData, HandoverRequestSetupItem,
    HandoverRequiredParams, HandoverSecurityContext, HandoverTypeValue,
    NrCgiValue as HandoverNrCgiValue, PduSessionResourceAdmittedItem,
    PduSessionResourceFailedToSetupHoAckItem, PduSessionResourceHoRequiredItem,
    SourceToTargetContainerParams, TaiValue, TargetIdValue,
    UserLocationInfoNr as HandoverUserLocationInfoNr,
};
use nextgsim_ngap::procedures::initial_context_setup::{
    decode_initial_context_setup_request, encode_initial_context_setup_failure,
    encode_initial_context_setup_response, InitialContextSetupFailureParams,
    InitialContextSetupResponseParams,
};
use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi, NrCgi, Tai, UserLocationInfoNr};
use nextgsim_ngap::procedures::nas_non_delivery_indication::{
    encode_nas_non_delivery_indication, NasNonDeliveryIndicationParams,
};
use nextgsim_ngap::procedures::ng_reset::{
    decode_amf_configuration_update, decode_ng_reset, encode_amf_configuration_update_acknowledge,
    encode_ng_reset_acknowledge, AmfConfigurationUpdateAcknowledgeParams, NgResetAcknowledgeParams,
    NgResetScope, UeAssociation,
};
use nextgsim_ngap::procedures::ng_setup::{
    NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause,
};
use nextgsim_ngap::procedures::overload::{
    decode_overload_start, decode_overload_stop, OverloadData,
};
use nextgsim_ngap::procedures::paging::{decode_paging, PagingData, UePagingIdentityValue};
use nextgsim_ngap::procedures::path_switch::{
    decode_path_switch_request_acknowledge, decode_path_switch_request_failure,
    encode_path_switch_request, PathSwitchRequestAcknowledgeData, PathSwitchRequestFailureData,
    PathSwitchRequestParams, PathSwitchSessionItem, UeSecurityCapabilityBits,
};
use nextgsim_ngap::procedures::pdu_session_resource::{
    // Modify
    decode_pdu_session_resource_modify_request,
    // Release
    decode_pdu_session_resource_release_command,
    decode_pdu_session_resource_setup_request,
    encode_pdu_session_resource_modify_response,
    encode_pdu_session_resource_release_response,
    encode_pdu_session_resource_setup_response,
    PduSessionResourceFailedToModifyItem,
    PduSessionResourceFailedToSetupItem,
    PduSessionResourceModifyRequestData,
    PduSessionResourceModifyResponseItem,
    PduSessionResourceModifyResponseParams,
    PduSessionResourceReleaseCommandData,
    PduSessionResourceReleaseResponseParams,
    PduSessionResourceReleasedItem,
    PduSessionResourceSetupItem,
    PduSessionResourceSetupRequestData,
    PduSessionResourceSetupResponseItem,
    PduSessionResourceSetupResponseParams,
};
use nextgsim_ngap::procedures::pdu_session_resource_notify::{
    encode_pdu_session_resource_notify, NotifiedQosFlow, NotifiedSession, NotifyCause,
    PduSessionResourceNotifyParams, ReleasedSession,
};
use nextgsim_ngap::procedures::ran_configuration_update::{
    decode_ran_configuration_update_acknowledge, decode_ran_configuration_update_failure,
    encode_ran_configuration_update, RanConfigurationUpdateParams,
};
use nextgsim_ngap::procedures::transfer::{
    decode_modify_request_transfer, decode_release_command_transfer, decode_setup_request_transfer,
    encode_handover_ack_transfer, encode_handover_required_transfer,
    encode_modify_response_transfer, encode_modify_unsuccessful_transfer,
    encode_release_response_transfer, encode_setup_response_transfer,
    encode_setup_unsuccessful_transfer, GtpTunnelInfo, HandoverAckTransferParams,
    ModifyResponseTransferParams, SetupResponseTransferParams, UpSecurityPolicy, UpSecurityResult,
};
use nextgsim_ngap::procedures::ue_context_modification::{
    decode_ue_context_modification_request, encode_ue_context_modification_failure,
    encode_ue_context_modification_response, UeContextModificationFailureCause,
    UeContextModificationRequestData,
};
use nextgsim_ngap::procedures::ue_context_release::{
    decode_ue_context_release_command, encode_ue_context_release_complete,
    encode_ue_context_release_request, UeContextReleaseCompleteParams,
    UeContextReleaseRequestParams,
};
use nextgsim_ngap::procedures::{
    build_ng_setup_request,
    // NAS Transport
    decode_downlink_nas_transport,
    // Initial UE Message
    encode_initial_ue_message,
    encode_uplink_nas_transport,
    is_ng_setup_failure,
    is_ng_setup_response,
    parse_ng_setup_failure,
    parse_ng_setup_response,
    AllowedSnssai,
    BroadcastPlmnItem,
    DownlinkNasTransportData,
    GnbId,
    InitialUeMessageParams,
    NgSetupRequestParams,
    PagingDrx,
    RrcEstablishmentCauseValue,
    SNssai,
    SupportedTaItem,
    UeContextRequestValue,
    UplinkNasTransportParams,
};

/// NGAP Task for managing AMF communication and UE contexts
pub struct NgapTask {
    /// Task base with handles to other tasks
    task_base: GnbTaskBase,
    /// AMF contexts indexed by client ID
    amf_contexts: HashMap<i32, NgapAmfContext>,
    /// UE contexts indexed by UE ID
    ue_contexts: HashMap<i32, NgapUeContext>,
    /// Counter for generating RAN UE NGAP IDs
    ran_ue_ngap_id_counter: i64,
    /// Counter for generating downlink TEIDs
    #[allow(dead_code)]
    downlink_teid_counter: u32,
    /// Whether the NGAP task is initialized (at least one AMF ready)
    is_initialized: bool,
    /// MBS session manager (Rel-17)
    mbs_sessions: MbsSessionManager,
    /// MBS session state driven by the NG-C wire path (TS 38.413 §9.2.9).
    ///
    /// Separate from `mbs_sessions` above, which is keyed by the AMF-assigned
    /// numeric session id and is fed by the internal `NgapMessage::Mbs*`
    /// channel. An MBS procedure arriving on SCTP carries only
    /// `id-MBS-SessionID` (299) — a TMGI, never a numeric id — so it cannot key
    /// that map, and this one is keyed by TMGI.
    mbs_ngap_sessions: NgapMbsManager,
    /// NGAP guard timers: TNGRELOCoverall, TNGRELOCprep and the NG Setup retry gated by
    /// an NG Setup Failure's Time to Wait (TS 38.413 §8.3.3.4, §8.4.1.2, §8.7.1.3).
    guard_timers: GuardTimers,
    /// SCTP client IDs of TNL associations this node opened because an AMF
    /// Configuration Update asked it to (TS 38.413 §9.2.6.5, issue #41).
    ///
    /// Keyed by endpoint so a later remove request can be matched to the
    /// association it actually names. Separate from `amf_contexts` because the
    /// question "did I open this on request" is not answerable from a context that
    /// looks identical to a configured one.
    tnla_client_ids: HashMap<std::net::IpAddr, i32>,
    /// Next SCTP client ID for a dynamically added TNL association.
    ///
    /// Starts well above the configured range: `main.rs` assigns client IDs from
    /// the `amf_configs` INDEX (0, 1, 2 ...), so a dynamic ID drawn from the same
    /// space would collide with a configured AMF and the two would share a context.
    next_dynamic_tnla_id: i32,
}

/// The first SCTP client ID used for a dynamically added TNL association.
///
/// Configured AMFs take 0..n from their `amf_configs` index, so this leaves room
/// for far more configured AMFs than any deployment has.
const DYNAMIC_TNLA_CLIENT_ID_BASE: i32 = 10_000;

/// The NGAP SCTP port to use for a TNL association the AMF named without one
/// (TS 38.412 §7: NGAP has no port IE, so the AMF can only give an address).
const DEFAULT_NGAP_PORT: u16 = 38412;

impl NgapTask {
    /// Creates a new NGAP task
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            amf_contexts: HashMap::new(),
            ue_contexts: HashMap::new(),
            ran_ue_ngap_id_counter: 0,
            downlink_teid_counter: 0,
            is_initialized: false,
            mbs_sessions: MbsSessionManager::new(),
            mbs_ngap_sessions: NgapMbsManager::new(),
            guard_timers: GuardTimers::new(),
            tnla_client_ids: HashMap::new(),
            next_dynamic_tnla_id: DYNAMIC_TNLA_CLIENT_ID_BASE,
        }
    }

    /// Generates a new RAN UE NGAP ID
    fn next_ran_ue_ngap_id(&mut self) -> i64 {
        self.ran_ue_ngap_id_counter += 1;
        self.ran_ue_ngap_id_counter
    }

    /// Generates a new downlink TEID
    #[allow(dead_code)]
    fn next_downlink_teid(&mut self) -> u32 {
        self.downlink_teid_counter += 1;
        self.downlink_teid_counter
    }

    // ========================================================================
    // AMF Context Management
    // ========================================================================

    /// Creates an AMF context for a new connection
    fn create_amf_context(&mut self, client_id: i32) {
        let ctx = NgapAmfContext::new(client_id);
        self.amf_contexts.insert(client_id, ctx);
        debug!("Created AMF context for client_id: {}", client_id);
    }

    /// Finds an AMF context by client ID
    #[allow(dead_code)]
    fn find_amf_context(&self, client_id: i32) -> Option<&NgapAmfContext> {
        self.amf_contexts.get(&client_id)
    }

    /// Finds a mutable AMF context by client ID
    #[allow(dead_code)]
    fn find_amf_context_mut(&mut self, client_id: i32) -> Option<&mut NgapAmfContext> {
        self.amf_contexts.get_mut(&client_id)
    }

    /// Selects an AMF for a new UE by slice support and capacity, treating
    /// multiple TNLAs of one AMF as a single AMF and load-balancing across them
    /// (TS 38.413 §8.6.1.2, TS 38.412 §7, TS 23.501 §5.15.5.2.1).
    ///
    /// Candidate TNLAs are `Ready`, not marked `unavailable` by an AMF Status
    /// Indication (TS 38.413 §8.7.6), and — when the UE requested one or more
    /// S-NSSAIs — advertise support for *every* requested slice on the serving
    /// `plmn`. Candidates are grouped into logical AMFs by GUAMI identity so
    /// two associations for the same AMF are not double-counted; the
    /// highest-capacity AMF wins, and among that AMF's TNLAs the least-loaded
    /// association is chosen. An empty `requested` list imposes no slice
    /// constraint.
    ///
    /// If no Ready AMF serves the requested slices, selection falls back to the
    /// highest-capacity Ready AMF so the UE can still attach (the AMF then
    /// applies its own NSSAI policy) — this also preserves the pre-slice
    /// behaviour for deployments whose AMFs advertise no slice support.
    ///
    /// Returns the client_id of the chosen TNLA (the transport association the
    /// UE binds to), or `None` if no AMF is available.
    fn select_amf(&self, plmn: &[u8; 3], requested: &[SNssai]) -> Option<i32> {
        // A candidate TNLA is Ready, available, and (if a slice was requested)
        // serves every requested S-NSSAI on the serving PLMN.
        let is_candidate = |ctx: &&NgapAmfContext| {
            // `is_selectable`, not `is_ready`: an OVERLOADED AMF is still a
            // candidate, and the traffic-load-reduction percentage throttles it
            // proportionally at admission (issue #41). Excluding it here applied a
            // 100% reduction whatever the AMF asked for.
            ctx.is_selectable()
                && !ctx.unavailable
                && (requested.is_empty() || requested.iter().all(|s| ctx.supports_snssai(plmn, s)))
        };

        // Group candidate TNLAs into logical AMFs by GUAMI identity so that
        // multiple TNLAs serving one AMF (TS 38.412 §7) are one selection
        // target. A TNLA with no served GUAMI yet is its own singleton AMF.
        #[derive(PartialEq, Eq, Hash)]
        enum AmfKey {
            Guami(AmfIdentity),
            Association(i32),
        }
        let mut instances: HashMap<AmfKey, Vec<i32>> = HashMap::new();
        for ctx in self.amf_contexts.values().filter(is_candidate) {
            let key = match ctx.amf_identity() {
                Some(id) => AmfKey::Guami(id),
                None => AmfKey::Association(ctx.ctx_id),
            };
            instances.entry(key).or_default().push(ctx.ctx_id);
        }

        if instances.is_empty() {
            if !requested.is_empty() {
                warn!(
                    "No Ready AMF advertises the requested S-NSSAI {:?} for PLMN {:02x?}; falling back to capacity-only AMF selection",
                    requested, plmn
                );
                return self.select_amf(plmn, &[]);
            }
            return None;
        }

        // Highest-capacity AMF wins (deterministic tie-break: lowest member
        // TNLA id); then load-balance across that AMF's TNLAs by choosing the
        // least-loaded association (tie-break: lowest TNLA id).
        let best = instances.values().max_by_key(|tnlas| {
            let capacity = tnlas
                .iter()
                .filter_map(|id| self.amf_contexts.get(id))
                .map(|ctx| ctx.relative_capacity)
                .max()
                .unwrap_or(0);
            let lowest_tnla = tnlas.iter().min().copied().unwrap_or(i32::MAX);
            (capacity, std::cmp::Reverse(lowest_tnla))
        })?;

        best.iter()
            .copied()
            .min_by_key(|id| (self.ue_load(*id), *id))
    }

    /// Number of UE contexts currently bound to the given AMF association
    /// (TNLA). Used to spread new UEs across the TNLAs of a multi-TNLA AMF.
    fn ue_load(&self, amf_ctx_id: i32) -> usize {
        self.ue_contexts
            .values()
            .filter(|ue| ue.amf_ctx_id == amf_ctx_id)
            .count()
    }

    /// Groups AMF associations (TNLAs) by GUAMI identity so that multiple TNLAs
    /// serving the same AMF (TS 38.412 §7) are reported as one AMF. Only
    /// associations that carry a served GUAMI (post-NG-Setup) appear. Returned
    /// as (identity, sorted TNLA ids), ordered by lowest TNLA id.
    #[allow(dead_code)]
    fn amf_instances(&self) -> Vec<(AmfIdentity, Vec<i32>)> {
        let mut groups: HashMap<AmfIdentity, Vec<i32>> = HashMap::new();
        for ctx in self.amf_contexts.values() {
            if let Some(id) = ctx.amf_identity() {
                groups.entry(id).or_default().push(ctx.ctx_id);
            }
        }
        let mut out: Vec<(AmfIdentity, Vec<i32>)> = groups
            .into_iter()
            .map(|(id, mut tnlas)| {
                tnlas.sort();
                (id, tnlas)
            })
            .collect();
        out.sort_by_key(|(_, tnlas)| tnlas.first().copied().unwrap_or(i32::MAX));
        out
    }

    // ========================================================================
    // UE Context Management
    // ========================================================================

    /// Creates a UE context for a new UE
    fn create_ue_context(&mut self, ue_id: i32, amf_ctx_id: i32) -> Option<i64> {
        let amf_ctx = self.amf_contexts.get_mut(&amf_ctx_id)?;
        let stream_id = amf_ctx.allocate_stream()?;
        let ran_ue_ngap_id = self.next_ran_ue_ngap_id();

        let ctx = NgapUeContext::new(ue_id, ran_ue_ngap_id, amf_ctx_id, stream_id);
        self.ue_contexts.insert(ue_id, ctx);

        debug!(
            "Created UE context: ue_id={}, ran_ue_ngap_id={}, amf_ctx_id={}, stream={}",
            ue_id, ran_ue_ngap_id, amf_ctx_id, stream_id
        );

        Some(ran_ue_ngap_id)
    }

    /// Finds a UE context by UE ID
    #[allow(dead_code)]
    fn find_ue_context(&self, ue_id: i32) -> Option<&NgapUeContext> {
        self.ue_contexts.get(&ue_id)
    }

    /// Finds a mutable UE context by UE ID
    #[allow(dead_code)]
    fn find_ue_context_mut(&mut self, ue_id: i32) -> Option<&mut NgapUeContext> {
        self.ue_contexts.get_mut(&ue_id)
    }

    /// Finds a UE context by RAN UE NGAP ID
    #[allow(dead_code)]
    fn find_ue_by_ran_id(&self, ran_ue_ngap_id: i64) -> Option<&NgapUeContext> {
        self.ue_contexts
            .values()
            .find(|ctx| ctx.ran_ue_ngap_id == ran_ue_ngap_id)
    }

    /// Finds a UE context by AMF UE NGAP ID
    #[allow(dead_code)]
    fn find_ue_by_amf_id(&self, amf_ue_ngap_id: i64) -> Option<&NgapUeContext> {
        self.ue_contexts
            .values()
            .find(|ctx| ctx.amf_ue_ngap_id == Some(amf_ue_ngap_id))
    }

    /// Deletes a UE context
    fn delete_ue_context(&mut self, ue_id: i32) {
        if let Some(ctx) = self.ue_contexts.remove(&ue_id) {
            // Release the stream back to the AMF
            if let Some(amf_ctx) = self.amf_contexts.get_mut(&ctx.amf_ctx_id) {
                amf_ctx.release_stream(ctx.stream_id);
            }
            debug!("Deleted UE context: ue_id={}", ue_id);
        }
    }

    // ========================================================================
    // SCTP Event Handlers
    // ========================================================================

    /// Handles SCTP association up event
    ///
    /// Public because it is a real message-handler entry point
    /// (`NgapMessage::SctpAssociationUp`) also driven directly by the in-process
    /// paging harness (`tests/src/paging_mt_service_request.rs`), which brings
    /// the AMF association up the way the SCTP task does rather than mutating
    /// the AMF context state from outside.
    pub async fn handle_association_up(
        &mut self,
        client_id: i32,
        association_id: i32,
        in_streams: u16,
        out_streams: u16,
    ) {
        info!(
            "SCTP association up: client_id={}, association_id={}, in={}, out={}",
            client_id, association_id, in_streams, out_streams
        );

        // Create or update AMF context
        if !self.amf_contexts.contains_key(&client_id) {
            self.create_amf_context(client_id);
        }

        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            ctx.on_association_up(association_id, in_streams, out_streams);
        }

        // Send NG Setup Request
        self.send_ng_setup_request(client_id).await;
    }

    /// Handles SCTP association down event
    async fn handle_association_down(&mut self, client_id: i32) {
        info!("SCTP association down: client_id={}", client_id);

        // Update AMF context
        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            ctx.on_association_down();
        }

        // Release all UEs associated with this AMF
        let ue_ids: Vec<i32> = self
            .ue_contexts
            .values()
            .filter(|ctx| ctx.amf_ctx_id == client_id)
            .map(|ctx| ctx.ue_id)
            .collect();

        for ue_id in ue_ids {
            self.delete_ue_context(ue_id);
            // Notify RRC of AN release
            self.send_an_release(ue_id).await;
        }

        // Update initialization status
        self.update_initialization_status().await;
    }

    // ========================================================================
    // NG Setup Procedure
    // ========================================================================

    /// Sends NG Setup Request to an AMF
    async fn send_ng_setup_request(&mut self, amf_id: i32) {
        let config = &self.task_base.config;

        // Build PLMN identity bytes
        let plmn_bytes = config.plmn.encode();

        // Build supported TA list
        let slice_support: Vec<SNssai> = config
            .nssai
            .iter()
            .map(|s| SNssai {
                sst: s.sst,
                sd: s.sd,
            })
            .collect();

        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: plmn_bytes,
                gnb_id_value: (config.nci >> (36 - config.gnb_id_length)) as u32,
                gnb_id_length: config.gnb_id_length,
            },
            ran_node_name: Some("nextgsim-gnb".to_string()),
            supported_ta_list: vec![SupportedTaItem {
                tac: [
                    ((config.tac >> 16) & 0xFF) as u8,
                    ((config.tac >> 8) & 0xFF) as u8,
                    (config.tac & 0xFF) as u8,
                ],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: plmn_bytes,
                    slice_support_list: if slice_support.is_empty() {
                        vec![SNssai { sst: 1, sd: None }] // Default slice
                    } else {
                        slice_support
                    },
                }],
            }],
            default_paging_drx: PagingDrx::V128,
        };

        match build_ng_setup_request(&params) {
            Ok(pdu) => {
                match encode_ngap_pdu(&pdu) {
                    Ok(bytes) => {
                        // Mark AMF as waiting for response
                        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                            ctx.on_ng_setup_sent();
                        }

                        // Send via SCTP
                        self.send_ngap_non_ue(amf_id, 0, bytes).await;
                        info!("Sent NG Setup Request to AMF {}", amf_id);
                    }
                    Err(e) => {
                        error!("Failed to encode NG Setup Request: {}", e);
                    }
                }
            }
            Err(e) => {
                error!("Failed to build NG Setup Request: {}", e);
            }
        }
    }

    /// Handles NG Setup Response
    fn handle_ng_setup_response(&mut self, amf_id: i32, pdu_bytes: &[u8]) -> bool {
        let pdu = match decode_ngap_pdu(pdu_bytes) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to decode NGAP PDU: {}", e);
                return false;
            }
        };

        if !is_ng_setup_response(&pdu) {
            return false;
        }

        match parse_ng_setup_response(&pdu) {
            Ok(response) => {
                info!(
                    "Received NG Setup Response from AMF {}: name={}",
                    amf_id, response.amf_name
                );

                if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                    ctx.on_ng_setup_response(response);
                }

                // The NTN configuration is no longer forwarded from here (issue #56).
                //
                // This used to send `RrcMessage::NtnTimingAdvanceConfig` so the RRC
                // task could store parameters nothing read. Two things were wrong
                // with the shape, not just with the missing reader: the cell's NTN
                // configuration does not depend on an AMF answer (an NTN cell is an
                // NTN cell before NG Setup completes, and a UE reads SIB19 while
                // still in idle), and the message carried no ephemeris, without which
                // TS 38.300 §16.14.2.2's pre-compensation cannot be computed.
                //
                // `RrcTask::new` now derives the SIB19 parameters from the same
                // `config.ntn_config` block directly. The log line stays, because
                // "this gNB is running as an NTN cell" is worth one line at startup.
                if let Some(ref ntn) = self.task_base.config.ntn_config {
                    info!(
                        "NTN mode active: satellite_type={}, propagation_delay={}us, \
                         k_offset={}; the serving-cell ephemeris and Common TA are \
                         broadcast in SIB19 (TS 38.300 §16.4)",
                        ntn.satellite_type, ntn.propagation_delay_us, ntn.k_offset
                    );
                }

                true
            }
            Err(e) => {
                error!("Failed to parse NG Setup Response: {}", e);
                false
            }
        }
    }

    /// Handles NG Setup Failure
    fn handle_ng_setup_failure(&mut self, amf_id: i32, pdu_bytes: &[u8]) -> bool {
        let pdu = match decode_ngap_pdu(pdu_bytes) {
            Ok(p) => p,
            Err(e) => {
                error!("Failed to decode NGAP PDU: {}", e);
                return false;
            }
        };

        if !is_ng_setup_failure(&pdu) {
            return false;
        }

        match parse_ng_setup_failure(&pdu) {
            Ok(failure) => {
                warn!(
                    "Received NG Setup Failure from AMF {}: cause={:?}",
                    amf_id, failure.cause
                );

                // TS 38.413 §8.7.1.3: with a Time to Wait IE the gNB "shall wait at
                // least for the indicated time before re-initiating the NG Setup
                // procedure towards the same AMF". Scheduled as a guard timer rather
                // than left to SCTP reconnection, which is not gated by the IE at all
                // and would retry sooner than the AMF asked.
                if let Some(time_to_wait) = failure.time_to_wait {
                    let wait = time_to_wait.as_duration();
                    info!(
                        "AMF {} requested a {:?} wait before retry ({:?}); NG Setup retry scheduled",
                        amf_id, wait, time_to_wait
                    );
                    self.guard_timers.start(
                        Instant::now(),
                        wait,
                        GuardTimer::NgSetupRetry { amf_id },
                    );
                } else {
                    // No IE: no wait is mandated, so the existing SCTP reconnection
                    // path governs. Not retried here, because retrying immediately
                    // against an AMF that just refused is its own kind of wrong.
                    debug!("NG Setup Failure from AMF {amf_id} carried no Time to Wait");
                }

                if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
                    ctx.on_association_down(); // Reset to not connected
                }
                true
            }
            Err(e) => {
                error!("Failed to parse NG Setup Failure: {}", e);
                false
            }
        }
    }

    // ========================================================================
    // NAS Message Routing
    // ========================================================================

    /// Handles Downlink NAS Transport from AMF
    async fn handle_downlink_nas_transport(
        &mut self,
        amf_id: i32,
        stream: u16,
        dl_nas: DownlinkNasTransportData,
    ) {
        info!(
            "Downlink NAS Transport: amf_ue_ngap_id={}, ran_ue_ngap_id={}, nas_pdu_len={}",
            dl_nas.amf_ue_ngap_id,
            dl_nas.ran_ue_ngap_id,
            dl_nas.nas_pdu.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_ctx = self
            .ue_contexts
            .values_mut()
            .find(|ctx| ctx.ran_ue_ngap_id == dl_nas.ran_ue_ngap_id as i64);

        let ue_id = match ue_ctx {
            Some(ctx) => {
                // Update AMF-UE-NGAP-ID if not set
                if ctx.amf_ue_ngap_id.is_none() {
                    ctx.amf_ue_ngap_id = Some(dl_nas.amf_ue_ngap_id as i64);
                    info!(
                        "Updated UE context: ue_id={}, amf_ue_ngap_id={}",
                        ctx.ue_id, dl_nas.amf_ue_ngap_id
                    );
                }
                ctx.ue_id
            }
            None => {
                warn!(
                    "No UE context found for RAN-UE-NGAP-ID {}; reporting NAS non-delivery",
                    dl_nas.ran_ue_ngap_id
                );
                // TS 38.413 §8.6.4: the DL NAS message cannot be delivered (no UE
                // context), so report non-delivery to the AMF with the original
                // NAS-PDU rather than dropping it silently.
                self.send_nas_non_delivery_indication(
                    amf_id,
                    stream,
                    &dl_nas,
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownLocalUeNgapId),
                )
                .await;
                return;
            }
        };

        // Log the NAS PDU content for debugging
        debug!(
            "NAS PDU (first 16 bytes): {:02x?}",
            &dl_nas.nas_pdu[..dl_nas.nas_pdu.len().min(16)]
        );

        // Forward NAS PDU to RRC for delivery to UE
        let msg = RrcMessage::NasDelivery {
            ue_id,
            pdu: OctetString::from_slice(&dl_nas.nas_pdu),
        };

        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send NAS delivery to RRC: {}", e);
            // TS 38.413 §8.6.4: RRC could not accept the NAS-PDU for delivery to
            // the UE — report non-delivery to the AMF.
            self.send_nas_non_delivery_indication(
                amf_id,
                stream,
                &dl_nas,
                NgSetupFailureCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost),
            )
            .await;
        } else {
            info!(
                "Forwarded NAS PDU to RRC: ue_id={}, nas_len={}",
                ue_id,
                dl_nas.nas_pdu.len()
            );
        }
    }

    /// Encode + send a NAS NON DELIVERY INDICATION (TS 38.413 §8.6.4) carrying
    /// the original NAS-PDU and a cause, UE-associated on `stream`.
    async fn send_nas_non_delivery_indication(
        &self,
        amf_id: i32,
        stream: u16,
        dl_nas: &DownlinkNasTransportData,
        cause: NgSetupFailureCause,
    ) {
        let params = NasNonDeliveryIndicationParams {
            amf_ue_ngap_id: dl_nas.amf_ue_ngap_id,
            ran_ue_ngap_id: dl_nas.ran_ue_ngap_id,
            nas_pdu: dl_nas.nas_pdu.clone(),
            cause,
        };
        match encode_nas_non_delivery_indication(&params) {
            Ok(bytes) => {
                let _ = self.send_ngap_ue_associated(amf_id, stream, bytes).await;
            }
            Err(e) => error!("Failed to encode NAS Non Delivery Indication: {}", e),
        }
    }

    /// Handles Initial Context Setup Request from AMF
    ///
    /// Select the NR ciphering + integrity algorithms from the UE Security
    /// Capabilities (TS 38.413 §9.3.1.86 bitmaps: MSB / 0x8000 = 128-xEA1).
    /// gNB priority: prefer algorithm 2 (AES) > 1 (SNOW3G) > 3 (ZUC) > 0 (null).
    /// Integrity never falls back to NIA0 (TS 33.501 §5.11.2): NIA2 is
    /// mandatory-to-support, so an empty integrity bitmap defaults to NIA2.
    fn select_as_algorithms(
        caps: &nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue,
    ) -> ((u8, CipheringAlgorithmType), (u8, IntegrityAlgorithmType)) {
        let ciph = {
            let m = caps.nr_encryption_algorithms;
            if m & 0x4000 != 0 {
                (2, CipheringAlgorithmType::Nea2)
            } else if m & 0x8000 != 0 {
                (1, CipheringAlgorithmType::Nea1)
            } else if m & 0x2000 != 0 {
                (3, CipheringAlgorithmType::Nea3)
            } else {
                (0, CipheringAlgorithmType::Nea0)
            }
        };
        let integ = {
            let m = caps.nr_integrity_algorithms;
            if m & 0x4000 != 0 {
                (2, IntegrityAlgorithmType::Nia2)
            } else if m & 0x8000 != 0 {
                (1, IntegrityAlgorithmType::Nia1)
            } else if m & 0x2000 != 0 {
                (3, IntegrityAlgorithmType::Nia3)
            } else {
                (2, IntegrityAlgorithmType::Nia2)
            }
        };
        (ciph, integ)
    }

    /// Establish AS security at Initial Context Setup (TS 33.501 §6.7 / §6.9,
    /// TS 38.331 §5.3.4): select algorithms from the UE Security Capabilities,
    /// derive the AS keys (K_RRCenc/int, K_UPenc/int) from KgNB (SecurityKey IE)
    /// per TS 33.501 Annex A.8, store the security context, and send the RRC
    /// SecurityModeCommand to the UE on SRB1.
    async fn activate_as_security(
        &mut self,
        ue_id: i32,
        kgnb: [u8; 32],
        caps: &nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue,
    ) {
        let ((ciph_id, ciph_alg), (int_id, int_alg)) = Self::select_as_algorithms(caps);

        // Derive the four AS keys from KgNB (TS 33.501 Annex A.8). Shared with the
        // handover re-keying path (issue #39), so a re-keyed UE's four keys come from the
        // same four calls the initial activation made.
        let sec_ctx = AsSecurityContext::from_kgnb(kgnb, ciph_id, int_id);
        let (srb_k_int, srb_k_enc) = (sec_ctx.k_rrc_int, sec_ctx.k_rrc_enc);
        // Wave-6 C4-final: allocate the SecurityModeCommand tid from THIS UE's
        // per-context allocator (TS 38.331 §5.3.4 / §6.3.2). Pinned to 0 on the
        // wire while C5_TYPED_DCCH_DISPATCH is off (the UE DL-DCCH dispatcher is
        // still the legacy nibble matcher, on which a non-zero tid shuffles the
        // routed leading-byte nibble); becomes the per-UE 0..3 cycle once C5
        // types both DL dispatchers.
        let rrc_transaction_id =
            if let Some(ctx) = self.ue_contexts.values_mut().find(|c| c.ue_id == ue_id) {
                ctx.as_security = Some(sec_ctx);
                ctx.transactions.allocate(RrcProcedure::SecurityMode)
            } else {
                0
            };
        info!(
            "AS security established for UE {}: ciphering=NEA{}, integrity=NIA{}",
            ue_id, ciph_id, int_id
        );

        // Encode + send the RRC SecurityModeCommand (SRB1, DL-DCCH).
        let params = SecurityModeCommandParams {
            rrc_transaction_id,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: ciph_alg,
                integrity_algorithm: Some(int_alg),
            },
        };
        match encode_security_mode_command(&params) {
            Ok(pdu) => {
                let msg = RrcMessage::SecurityModeCommand {
                    ue_id,
                    pdu: OctetString::from_slice(&pdu),
                };
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to hand RRC SecurityModeCommand to RRC task: {}", e);
                }
            }
            Err(e) => error!("Failed to encode RRC SecurityModeCommand: {}", e),
        }

        // The RRC task verifies an RRCReestablishmentRequest's shortMAC-I with
        // K_RRCint (TS 38.331 §5.3.7.2) and answers with the AS context's
        // nextHopChainingCount, neither of which it can reach on its own: the AS
        // context lives here. Hand it the subset it needs (issue #37).
        //
        // NCC 0 is the spec's own initial value, not a placeholder: TS 33.501
        // §6.8.2.1.1 gives the KgNB established at Initial Context Setup an NCC of 0. A
        // fresh {NH, NCC} pair arrives in a Handover Request or a Path Switch Request
        // Acknowledge, which issue #39 wired.
        let reestablishment_security = RrcMessage::AsSecurityForReestablishment {
            ue_id,
            k_rrc_int: srb_k_int,
            // K_RRCenc too (issue #31): the RRC plane sends and receives SRB1
            // PDUs, so it is the plane that has to cipher them.
            k_rrc_enc: srb_k_enc,
            integrity_alg_id: int_id,
            ciphering_alg_id: ciph_id,
            c_rnti: SIMULATED_C_RNTI,
            phys_cell_id: phys_cell_id_from_nci(self.task_base.config.nci),
            next_hop_chaining_count: 0,
        };
        if let Err(e) = self.task_base.rrc_tx.send(reestablishment_security).await {
            error!(
                "Failed to hand the re-establishment security context to the RRC task: {}",
                e
            );
        }
    }

    /// Establishes UE security context (KgNB), stores security capabilities,
    /// forwards piggybacked NAS PDU to UE, and sends response back to AMF.
    async fn handle_initial_context_setup_request(
        &mut self,
        amf_id: i32,
        stream: u16,
        ics_req: nextgsim_ngap::procedures::initial_context_setup::InitialContextSetupRequestData,
    ) {
        info!(
            "Initial Context Setup Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, security_key_len={}",
            ics_req.amf_ue_ngap_id,
            ics_req.ran_ue_ngap_id,
            ics_req.security_key.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_ctx = self
            .ue_contexts
            .values_mut()
            .find(|ctx| ctx.ran_ue_ngap_id == ics_req.ran_ue_ngap_id as i64);

        let ue_id = match ue_ctx {
            Some(ctx) => {
                // Update AMF-UE-NGAP-ID
                if ctx.amf_ue_ngap_id.is_none() {
                    ctx.amf_ue_ngap_id = Some(ics_req.amf_ue_ngap_id as i64);
                }
                // Store the AMF-provided UE Security Capabilities and UE-AMBR as
                // the baseline; UE Context Modification may later replace them.
                ctx.ue_security_capabilities = Some(ics_req.ue_security_capabilities.clone());
                if let Some(ambr) = ics_req.ue_aggregate_max_bit_rate.clone() {
                    ctx.ue_ambr = Some(ambr);
                }
                // Transition UE state
                ctx.on_initial_context_setup();
                info!(
                    "UE context updated: ue_id={}, state={}, amf_ue_ngap_id={}",
                    ctx.ue_id, ctx.state, ics_req.amf_ue_ngap_id
                );
                ctx.ue_id
            }
            None => {
                warn!(
                    "No UE context found for RAN-UE-NGAP-ID {}",
                    ics_req.ran_ue_ngap_id
                );
                return;
            }
        };

        // TS 33.501 §6.7 / TS 38.331 §5.3.4: establish AS security — derive the
        // AS keys from KgNB, select algorithms from the UE Security
        // Capabilities, and send the RRC SecurityModeCommand — before
        // completing the context.
        self.activate_as_security(
            ue_id,
            ics_req.security_key,
            &ics_req.ue_security_capabilities,
        )
        .await;

        // Forward piggybacked NAS PDU (e.g., SecurityModeCommand) to UE via RRC
        if let Some(nas_pdu) = &ics_req.nas_pdu {
            debug!(
                "Forwarding piggybacked NAS PDU ({} bytes) to UE {}",
                nas_pdu.len(),
                ue_id
            );
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!(
                    "Failed to forward NAS PDU from InitialContextSetup to RRC: {}",
                    e
                );
            }
        }

        // Transition to Active state (security context established)
        if let Some(ctx) = self.ue_contexts.values_mut().find(|ctx| ctx.ue_id == ue_id) {
            ctx.on_context_setup_complete();
            info!("UE {} context setup complete, state={}", ue_id, ctx.state);
        }

        // amfg-01: set up any PDU sessions carried inside the ICS Request
        // (PDUSessionResourceSetupListCxtReq). Real AMFs (e.g. Open5GS) deliver
        // the first PDU session this way during registration+PDU, so the N3
        // tunnel must be established here, not only on the dedicated PDU Session
        // Resource Setup procedure.
        let carried_sessions = ics_req.pdu_session_setup_list.len();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);
        let mut cxt_res_items = Vec::new();
        let mut cxt_failed_items = Vec::new();
        for item in &ics_req.pdu_session_setup_list {
            match self.setup_one_pdu_session(ue_id, item, gnb_ip).await {
                Ok(resp) => cxt_res_items.push(resp),
                Err(failed) => cxt_failed_items.push(failed),
            }
        }

        // If the ICS Request carried PDU sessions but none could be set up,
        // answer with InitialContextSetupFailure (TS 38.413 §8.3.1.4) instead of
        // a bare success.
        if carried_sessions > 0 && cxt_res_items.is_empty() {
            warn!(
                "ICS carried {} PDU session(s) but none set up; sending InitialContextSetupFailure",
                carried_sessions
            );
            let fail_params = InitialContextSetupFailureParams {
                amf_ue_ngap_id: ics_req.amf_ue_ngap_id,
                ran_ue_ngap_id: ics_req.ran_ue_ngap_id,
                cause: NgSetupFailureCause::RadioNetwork(
                    RadioNetworkCause::RadioResourcesNotAvailable,
                ),
            };
            match encode_initial_context_setup_failure(&fail_params) {
                Ok(bytes) => {
                    let _ = self.send_ngap_ue_associated(amf_id, stream, bytes).await;
                }
                Err(e) => error!("Failed to encode InitialContextSetupFailure: {}", e),
            }
            return;
        }

        // Send Initial Context Setup Response (with the CxtRes / FailedToSetup
        // lists when the ICS Request carried PDU sessions).
        let response_params = InitialContextSetupResponseParams {
            amf_ue_ngap_id: ics_req.amf_ue_ngap_id,
            ran_ue_ngap_id: ics_req.ran_ue_ngap_id,
            setup_list: if cxt_res_items.is_empty() {
                None
            } else {
                Some(cxt_res_items)
            },
            failed_list: if cxt_failed_items.is_empty() {
                None
            } else {
                Some(cxt_failed_items)
            },
        };
        match encode_initial_context_setup_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!(
                    "Initial Context Setup Response sent for RAN-UE-NGAP-ID {}",
                    ics_req.ran_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Initial Context Setup Response: {}", e);
            }
        }
    }

    /// Handle an inbound UE CONTEXT MODIFICATION REQUEST (TS 38.413 §8.3.4).
    ///
    /// Applies any updated UE Security Capabilities and replaced UE-AMBR to the
    /// stored UE context, and — when a new Security Key IE is present — re-derives
    /// the AS keys (TS 33.501 §6.9.2) by reusing [`Self::activate_as_security`].
    /// Replies with a UE CONTEXT MODIFICATION RESPONSE on success, or a UE CONTEXT
    /// MODIFICATION FAILURE — never an Error Indication — on error. Context state
    /// is mutated only after every failure precondition has passed, so a failed
    /// procedure leaves the UE context unchanged (TS 38.413 §8.3.4).
    ///
    /// Simulator simplification: the AS re-key is signalled to the UE by reusing
    /// [`Self::activate_as_security`]'s RRC Security Mode Command (TS 38.331
    /// §5.3.4), rather than the spec-conformant key-change-on-the-fly carrier — an
    /// RRCReconfiguration with `masterKeyUpdate` (TS 38.331 §5.3.5.3). This mirrors
    /// how Initial Context Setup treats the Security Key IE as the KgNB and is
    /// deliberate: `masterKeyUpdate` is modelled on neither the gNB RRC builder nor
    /// the UE. The four AS keys are still correctly re-derived from the new KgNB.
    async fn handle_ue_context_modification_request(
        &mut self,
        amf_id: i32,
        stream: u16,
        ucm_req: UeContextModificationRequestData,
    ) {
        info!(
            "UE Context Modification Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, rekey={}",
            ucm_req.amf_ue_ngap_id,
            ucm_req.ran_ue_ngap_id,
            ucm_req.security_key.is_some()
        );

        // Phase 1 (read-only): locate the UE by RAN-UE-NGAP-ID and resolve the
        // capabilities a possible AS re-key would use — the request's caps when
        // present, otherwise the capabilities stored at Initial Context Setup. No
        // context state is mutated here so that a failed procedure leaves the UE
        // context unchanged.
        let (ue_id, stored_caps) = match self
            .ue_contexts
            .values()
            .find(|ctx| ctx.ran_ue_ngap_id == ucm_req.ran_ue_ngap_id as i64)
        {
            Some(ctx) => (ctx.ue_id, ctx.ue_security_capabilities.clone()),
            None => {
                warn!(
                    "UE Context Modification for unknown RAN-UE-NGAP-ID {}; replying with UE CONTEXT MODIFICATION FAILURE",
                    ucm_req.ran_ue_ngap_id
                );
                self.send_ue_context_modification_failure(
                    amf_id,
                    stream,
                    ucm_req.amf_ue_ngap_id,
                    ucm_req.ran_ue_ngap_id,
                    UeContextModificationFailureCause::RadioNetwork(
                        RadioNetworkCause::UnknownLocalUeNgapId,
                    ),
                )
                .await;
                return;
            }
        };

        let rekey_caps = ucm_req.ue_security_capabilities.clone().or(stored_caps);

        // Precondition: a Security Key re-key needs the UE Security Capabilities
        // to select the AS algorithms. Fail BEFORE mutating any context state so
        // the procedure is atomic.
        if ucm_req.security_key.is_some() && rekey_caps.is_none() {
            warn!(
                "UE Context Modification for UE {ue_id} carried a Security Key but no UE Security Capabilities are available; replying with UE CONTEXT MODIFICATION FAILURE"
            );
            self.send_ue_context_modification_failure(
                amf_id,
                stream,
                ucm_req.amf_ue_ngap_id,
                ucm_req.ran_ue_ngap_id,
                UeContextModificationFailureCause::Protocol(ProtocolCause::SemanticError),
            )
            .await;
            return;
        }

        // Phase 2 (commit): every failure precondition has passed — apply the
        // non-security updates (replaced UE-AMBR and updated UE Security
        // Capabilities, TS 38.413 §8.3.4.2) to the stored UE context.
        if let Some(ctx) = self.ue_contexts.values_mut().find(|ctx| ctx.ue_id == ue_id) {
            if ctx.amf_ue_ngap_id.is_none() {
                ctx.amf_ue_ngap_id = Some(ucm_req.amf_ue_ngap_id as i64);
            }
            if let Some(ambr) = ucm_req.ue_aggregate_max_bit_rate.clone() {
                ctx.ue_ambr = Some(ambr);
            }
            if let Some(caps) = ucm_req.ue_security_capabilities.clone() {
                ctx.ue_security_capabilities = Some(caps);
            }
        }

        // AS re-keying (TS 33.501 §6.9.2): re-derive the AS key set from the new
        // KgNB. `rekey_caps` is guaranteed present here by the precondition above.
        if let (Some(new_kgnb), Some(caps)) = (ucm_req.security_key, rekey_caps.as_ref()) {
            self.activate_as_security(ue_id, new_kgnb, caps).await;
            info!("AS re-keying completed for UE {ue_id} via UE Context Modification");
        }

        // Success: UE CONTEXT MODIFICATION RESPONSE.
        match encode_ue_context_modification_response(
            ucm_req.amf_ue_ngap_id,
            ucm_req.ran_ue_ngap_id,
        ) {
            Ok(bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, bytes).await;
                info!(
                    "UE Context Modification Response sent for RAN-UE-NGAP-ID {}",
                    ucm_req.ran_ue_ngap_id
                );
            }
            Err(e) => error!("Failed to encode UE Context Modification Response: {e}"),
        }
    }

    /// Encode + send a UE-associated UE CONTEXT MODIFICATION FAILURE.
    async fn send_ue_context_modification_failure(
        &self,
        amf_id: i32,
        stream: u16,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
        cause: UeContextModificationFailureCause,
    ) {
        match encode_ue_context_modification_failure(amf_ue_ngap_id, ran_ue_ngap_id, &cause) {
            Ok(bytes) => {
                let _ = self.send_ngap_ue_associated(amf_id, stream, bytes).await;
            }
            Err(e) => error!("Failed to encode UE Context Modification Failure: {e}"),
        }
    }

    /// amfg-01: set up a single PDU session and return its per-session result.
    ///
    /// Decodes the APER PDUSessionResourceSetupRequestTransfer (TS 38.413
    /// §9.3.4.1), allocates the gNB DL F-TEID, stores the session in the UE
    /// context, creates the GTP-U N3 tunnel, forwards any piggybacked NAS, and
    /// builds the PDUSessionResourceSetupResponseTransfer (§9.3.4.2).
    ///
    /// Shared by the dedicated PDU Session Resource Setup procedure and the PDU
    /// sessions carried inside an INITIAL CONTEXT SETUP REQUEST
    /// (`PDUSessionResourceSetupListCxtReq`, TS 38.413 §8.3.1.2). Returns the
    /// success item, or a FailedToSetup item (with an APER unsuccessful-transfer)
    /// on any error.
    /// TS 38.331 §5.3.5.6 / TS 38.300 §16.1.4: establish the user-plane DRB for
    /// a PDU session by sending an RRCReconfiguration carrying a
    /// RadioBearerConfig (one DRB-ToAddMod with an SDAP-Config keyed to the PDU
    /// session id and the accepted QFIs in mappedQoS-FlowsToAdd) plus the
    /// matching CellGroupConfig (one RLC bearer). Requires AS security to be
    /// active (established at Initial Context Setup, C5).
    ///
    /// # With `sdap-dataplane`: TWO DRBs, and the QFI decides which (issue #44)
    ///
    /// TS 37.324 §5.1 has an SDAP entity map each QoS flow onto a DRB. With the
    /// feature on, the session gets a default DRB and a GBR DRB, each carrying the
    /// subset of QFIs the policy in `nextgsim_gtp::qfi_drb` assigns it, and the UE
    /// is **told** the mapping through each DRB's `mappedQoS-FlowsToAdd`.
    ///
    /// The default DRB keeps the identity the PSI produced, so a session with one
    /// flow is numbered identically with the feature on or off.
    async fn establish_drb(
        &mut self,
        ue_id: i32,
        psi: u8,
        accepted_flows: &[(u8, Option<u16>)],
        integrity_protection: DrbIntegrityProtection,
    ) {
        if !self
            .ue_contexts
            .values()
            .any(|c| c.ue_id == ue_id && c.as_security.is_some())
        {
            warn!(
                "Establishing DRB for PDU session {} on UE {} without active AS security",
                psi, ue_id
            );
        }
        let qfis: Vec<u8> = accepted_flows.iter().map(|(qfi, _)| *qfi).collect();
        // One DRB per PDU session; DRB identity 1..=32, DTCH LCID above the SRBs.
        let drb_id = Self::drb_identity_for(psi);
        let lcid = (3 + drb_id).min(32);
        // Wave-6 C4-final: allocate the RRCReconfiguration tid from THIS UE's
        // per-context allocator (TS 38.331 §5.3.5 / §6.3.2). Pinned to 0 on the
        // wire while C5_TYPED_DCCH_DISPATCH is off (a non-zero tid shuffles the
        // leading-byte nibble the UE's legacy DL-DCCH dispatcher routes on, e.g.
        // tid 2 -> byte0 0x04 -> misrouted into the DL-information-transfer arm);
        // becomes the per-UE 0..3 cycle once C5 types both DL dispatchers.
        let rrc_transaction_id = self
            .ue_contexts
            .values_mut()
            .find(|c| c.ue_id == ue_id)
            .map(|c| c.transactions.allocate(RrcProcedure::Reconfiguration))
            .unwrap_or(0);
        // Configure the UE's A3 REPORTING measurement on this reconfiguration
        // (issue #170). This is the message that establishes the user plane, so it
        // is the first one a connected UE gets that can carry a `measConfig` — and
        // a UE that is about to carry traffic is exactly the UE whose mobility
        // reporting has to be on the gNB's margin rather than a local default.
        //
        // `None` when the configured margin is not signallable: the DRB half of
        // this message still goes out. See `rrc::meas::a3_meas_config_params`.
        let meas_config = a3_meas_config_params(&self.task_base.config);

        // The DRB set this session gets. With `sdap-dataplane` the QFI→DRB policy
        // decides it (issue #44); without, it is the single bearer keyed by PSI that
        // the pre-SDAP path has always sent, byte for byte.
        let specs = self.drb_specs_for(psi, accepted_flows, drb_id, lcid, integrity_protection);
        let params = match build_multi_drb_reconfiguration_params(
            rrc_transaction_id,
            psi,
            &specs,
            meas_config,
        ) {
            Ok(p) => p,
            Err(e) => {
                error!(
                    "Failed to build DRB reconfiguration for PDU session {}: {}",
                    psi, e
                );
                return;
            }
        };
        // Carry the serving cell's `ntn-Config` on this reconfiguration for the
        // connected-mode NTN update path (TS 38.300 §16.14.2.2, issue #56).
        //
        // This message, not a later one: it is what takes the UE into carrying
        // traffic, and §16.14.2.2 says a UE without a valid ephemeris and Common TA
        // "shall not transmit". A UE that acquired SIB19 in idle already has the
        // configuration; this is the refresh that keeps it valid past
        // `ntn-UlSyncValidityDuration` without requiring it to keep reading BCCH.
        //
        // Derived from the same `sib19_params(&config)` the RRC task's broadcast uses,
        // so the two cannot hand a UE two different ephemerides. `None` on a
        // terrestrial cell, where the field is simply absent.
        let params = RrcReconfigurationParams {
            ntn_config: sib19_params(&self.task_base.config),
            ..params
        };
        match encode_rrc_reconfiguration(&params) {
            Ok(pdu) => {
                let msg = RrcMessage::RrcReconfiguration {
                    ue_id,
                    pdu: OctetString::from_slice(&pdu),
                };
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to hand RRCReconfiguration to RRC task: {}", e);
                } else {
                    info!(
                        "Sent RRCReconfiguration establishing {} DRB(s) for PDU session {}: {} (all QFIs {:?})",
                        specs.len(),
                        psi,
                        specs
                            .iter()
                            .map(|s| format!("DRB {} <- QFIs {:?}", s.drb_id, s.qfis))
                            .collect::<Vec<_>>()
                            .join(", "),
                        qfis
                    );
                }
            }
            Err(e) => error!(
                "Failed to encode RRCReconfiguration for PDU session {}: {}",
                psi, e
            ),
        }
    }

    /// The DRB set one PDU session gets, and which QFIs ride each (issue #44).
    ///
    /// Without `sdap-dataplane`: exactly the single bearer the pre-SDAP path sent —
    /// all QFIs on one default DRB with the PSI-derived identity and LCID. The
    /// `build_multi_drb_*` builders are byte-identical to the single-DRB ones for a
    /// lone bearer (pinned by
    /// `the_multi_drb_builder_matches_the_single_drb_one_for_a_lone_bearer`), so the
    /// message is unchanged and the frozen golden vectors still describe it.
    ///
    /// With the feature: two bearers, split by the 5QI's resource type, and the
    /// GBR one is **omitted when no admitted flow maps to it** — a DRB with no QoS
    /// flow is an RLC and PDCP entity nothing would ever feed, and signalling it
    /// would have the UE build the same pair of idle entities.
    fn drb_specs_for(
        &self,
        psi: u8,
        accepted_flows: &[(u8, Option<u16>)],
        drb_id: u8,
        lcid: u8,
        integrity_protection: DrbIntegrityProtection,
    ) -> Vec<DrbSpec> {
        #[cfg(not(feature = "sdap-dataplane"))]
        {
            let _ = psi;
            vec![DrbSpec {
                drb_id,
                lcid,
                qfis: accepted_flows.iter().map(|(qfi, _)| *qfi).collect(),
                default_drb: true,
                integrity_protection,
            }]
        }
        #[cfg(feature = "sdap-dataplane")]
        {
            use nextgsim_gtp::qfi_drb::{allocate_drbs, DrbChoice, QfiDrbMap};

            let mut map = QfiDrbMap::new();
            for &(qfi, five_qi) in accepted_flows {
                map.admit_flow(qfi, five_qi);
            }
            let alloc = allocate_drbs(psi);
            debug_assert_eq!(
                alloc.default_drb_id, drb_id,
                "the default DRB must keep the identity the PSI produced, or a \
                 session's bearer is renumbered by a cargo feature"
            );
            debug_assert_eq!(alloc.default_lcid, lcid);

            let mut specs = vec![DrbSpec {
                drb_id: alloc.default_drb_id,
                lcid: alloc.default_lcid,
                qfis: map.qfis_for(DrbChoice::Default),
                default_drb: true,
                integrity_protection,
            }];
            let gbr_qfis = map.qfis_for(DrbChoice::Gbr);
            if !gbr_qfis.is_empty() {
                specs.push(DrbSpec {
                    drb_id: alloc.gbr_drb_id,
                    lcid: alloc.gbr_lcid,
                    qfis: gbr_qfis,
                    default_drb: false,
                    integrity_protection,
                });
            }
            specs
        }
    }

    /// The DRB identity carrying one PDU session.
    ///
    /// This simulator maps one DRB per PDU session, so the identity is the PSI —
    /// clamped into `DRB-Identity`'s 1..=32 range (TS 38.331 §6.3.2). Named because
    /// **two** places need the same answer: the RRCReconfiguration that establishes
    /// the DRB, and the BEARER input to user-plane ciphering, which TS 33.501
    /// Annex D.3.1.2 defines as the radio bearer identity minus one. The UE derives
    /// BEARER from the `drb-Identity` it was sent, so a gNB that derived it from the
    /// PSI instead would agree only by coincidence.
    fn drb_identity_for(psi: u8) -> u8 {
        psi.clamp(1, 32)
    }

    /// Whether this build can protect a DRB at all: the `up-security` feature.
    ///
    /// A `const` read by [`up_security::resolve`] rather than a `cfg!` inside it, so
    /// the refusal and protection paths are both compiled and unit-tested in every
    /// build and only the *wiring* is feature-gated.
    const UP_SECURITY_AVAILABLE: bool = cfg!(feature = "up-security");

    /// Resolve the SMF's user-plane security policy for one session and log what the
    /// gNB will do about it (issue #32, TS 33.501 §6.6.1).
    fn resolve_up_security(
        &self,
        ue_id: i32,
        psi: u8,
        requested: Option<UpSecurityPolicy>,
    ) -> Result<(UpSecurityPolicy, DrbSecurityDecision), UpSecurityRefusal> {
        let policy = requested.unwrap_or_else(|| {
            debug!(
                "PDU session {psi} on UE {ue_id}: the SMF sent no SecurityIndication,                  applying the locally configured policy"
            );
            UpSecurityPolicy::locally_configured_default()
        });
        // With no AS security context there are no negotiated algorithms, so the
        // null identities are the truth -- and a `required` policy then refuses,
        // which is right: nothing has been keyed.
        let (ciph, integ) = self
            .ue_contexts
            .values()
            .find(|c| c.ue_id == ue_id)
            .and_then(|c| c.as_security.as_ref())
            .map(|s| (s.ciphering_alg_id, s.integrity_alg_id))
            .unwrap_or((up_security::NULL_ALGORITHM, up_security::NULL_ALGORITHM));

        let decision = up_security::resolve(&policy, ciph, integ, Self::UP_SECURITY_AVAILABLE)?;
        info!(
            "PDU session {psi} on UE {ue_id}: user-plane security integrity={}              ciphering={} (policy integrity={:?} confidentiality={:?}, NEA{ciph}/NIA{integ})",
            decision.integrity, decision.ciphering, policy.integrity, policy.confidentiality
        );
        if !up_security::honours_confidentiality_policy(&policy, decision) {
            info!(
                "PDU session {psi} on UE {ue_id}: the SMF asked for no user-plane                  confidentiality, but PDCP-Config.cipheringDisabled is absent from this                  codec, so the DRB is ciphered anyway (issue #32 ceiling)"
            );
        }
        Ok((policy, decision))
    }

    /// Hand a DRB's user-plane keys to the RLS task, which owns the PDCP entities.
    ///
    /// A no-op without the `up-security` feature, where there is no `PdcpSecurity`
    /// to install and no `InstallDrbSecurity` variant to send.
    #[allow(unused_variables)]
    async fn install_drb_security(&self, ue_id: i32, psi: u8, decision: DrbSecurityDecision) {
        #[cfg(feature = "up-security")]
        {
            use nextgsim_pdcp::{PdcpSecurity, UpSecurity, DIRECTION_DOWNLINK};

            let security = if !decision.any() {
                None
            } else {
                let Some(as_ctx) = self
                    .ue_contexts
                    .values()
                    .find(|c| c.ue_id == ue_id)
                    .and_then(|c| c.as_security.as_ref())
                else {
                    // Unreachable via `resolve_up_security`, which reports the null
                    // algorithms when there is no context and so decides nothing to
                    // apply. Logged rather than asserted because a future caller
                    // could reach it, and an unprotected DRB is the safe outcome
                    // only if somebody is told.
                    warn!(
                        "PDU session {psi} on UE {ue_id}: user-plane security was                          decided but there is no AS security context to key it with;                          leaving the DRB unprotected"
                    );
                    return;
                };
                match UpSecurity::new(
                    as_ctx.k_up_enc,
                    as_ctx.k_up_int,
                    if decision.ciphering {
                        as_ctx.ciphering_alg_id
                    } else {
                        up_security::NULL_ALGORITHM
                    },
                    decision.integrity.then_some(as_ctx.integrity_alg_id),
                ) {
                    Ok(sec) => Some(Box::new(PdcpSecurity {
                        security: sec,
                        // BEARER is the radio bearer identity minus one
                        // (TS 33.501 Annex D.3.1.2), and the identity is the one
                        // `establish_drb` puts on the wire -- not the PSI, which the
                        // UE never sees in `drb-Identity`.
                        bearer: Self::drb_identity_for(psi).saturating_sub(1),
                        tx_direction: DIRECTION_DOWNLINK,
                    })),
                    Err(e) => {
                        error!("PDU session {psi} on UE {ue_id}: refusing to key the DRB: {e}");
                        return;
                    }
                }
            };
            let msg = crate::tasks::RlsMessage::InstallDrbSecurity {
                ue_id,
                psi: psi as i32,
                // The DEFAULT DRB, which is the bearer `bearer` above was derived from
                // and the only one this function keys (issue #44). A session's GBR DRB
                // is left unprotected by design for now: `PdcpSecurity::bearer` is
                // per-bearer, so the second DRB needs its own binding with its own
                // BEARER, and installing THIS one on it would cipher it with a BEARER
                // the UE does not use -- silently failing every MAC-I on that DRB
                // instead of leaving it visibly unprotected.
                drb_id: Self::drb_identity_for(psi) as i32,
                security,
            };
            if let Err(e) = self.task_base.rls_tx.send(msg).await {
                error!("Failed to install DRB security on the RLS task: {e}");
            }
        }
    }

    /// Key the DRB's PDCP entity and send the RRCReconfiguration that establishes it.
    ///
    /// Shared by the PDU Session Resource Setup path and the handover-in admission path
    /// (issue #39): a handover-in *is* a session setup whose QoS, tunnel and security
    /// policy arrived by a different route, so it must allocate the same way. Two copies
    /// of this is how a handover comes to establish a DRB the gNB cannot decipher.
    ///
    /// Keys before the reconfiguration, deliberately: the RRCReconfiguration tells the
    /// UE to start protecting, so the gNB's own entity has to be able to verify by the
    /// time the UE's first protected uplink PDU arrives.
    /// `accepted_flows` is `(qfi, five_qi)` per admitted QoS flow. The 5QI rides
    /// along because it is what the QFI→DRB policy decides on (issue #44): a flow's
    /// resource type comes from its 5QI, and the QFI alone says nothing about
    /// whether the flow is GBR. `None` for a dynamic 5QI.
    async fn key_and_establish_drb(
        &mut self,
        ue_id: i32,
        psi: u8,
        accepted_flows: &[(u8, Option<u16>)],
        decision: DrbSecurityDecision,
    ) {
        self.install_drb_security(ue_id, psi, decision).await;
        self.establish_drb(
            ue_id,
            psi,
            accepted_flows,
            if decision.integrity {
                DrbIntegrityProtection::Enabled
            } else {
                DrbIntegrityProtection::Disabled
            },
        )
        .await;
    }

    async fn setup_one_pdu_session(
        &mut self,
        ue_id: i32,
        item: &PduSessionResourceSetupItem,
        gnb_ip: std::net::IpAddr,
    ) -> Result<PduSessionResourceSetupResponseItem, PduSessionResourceFailedToSetupItem> {
        let psi = item.pdu_session_id;

        // Decode the APER PDUSessionResourceSetupRequestTransfer (TS 38.413 §9.3.4.1)
        let request = match decode_setup_request_transfer(&item.transfer) {
            Ok(req) => req,
            Err(e) => {
                warn!(
                    "PDU Session {}: failed to decode SetupRequestTransfer ({} bytes): {}",
                    psi,
                    item.transfer.len(),
                    e
                );
                let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::Protocol(
                    ProtocolCause::TransferSyntaxError,
                ))
                .unwrap_or_default();
                return Err(PduSessionResourceFailedToSetupItem {
                    pdu_session_id: psi,
                    transfer,
                });
            }
        };

        // TS 33.501 §6.6.1: a `required` policy this gNB cannot satisfy fails the
        // session. Checked BEFORE any TEID is allocated or GTP is told anything, so
        // a refused session leaves no state behind to clean up.
        let (policy, decision) =
            match self.resolve_up_security(ue_id, psi, request.security_indication) {
                Ok(resolved) => resolved,
                Err(refusal) => {
                    warn!("PDU Session {psi} on UE {ue_id} refused: {refusal}");
                    let cause = NgSetupFailureCause::RadioNetwork(match refusal {
                        UpSecurityRefusal::IntegrityNotPossible => {
                            RadioNetworkCause::UpIntegrityProtectionNotPossible
                        }
                        UpSecurityRefusal::ConfidentialityNotPossible => {
                            RadioNetworkCause::UpConfidentialityProtectionNotPossible
                        }
                    });
                    let transfer = encode_setup_unsuccessful_transfer(&cause).unwrap_or_default();
                    return Err(PduSessionResourceFailedToSetupItem {
                        pdu_session_id: psi,
                        transfer,
                    });
                }
            };

        let upf_teid = request.ul_tunnel.teid;
        let upf_addr = request.ul_tunnel.address;
        // First QoS flow is the default flow for the session
        let qfi = request.qos_flows[0].qfi;
        // The 5QI rides with the QFI because it is what the QFI→DRB policy decides
        // on (issue #44): a flow's resource type comes from its 5QI, and the QFI
        // alone says nothing about whether the flow is GBR. Built here, above the
        // `SessionCreate`, because the GTP task's SDAP entity needs the same set the
        // RRCReconfiguration below describes -- the two must agree on the mapping or
        // the gNB sends an SDU down a bearer the UE was never told about.
        let accepted_flows: Vec<(u8, Option<u16>)> = request
            .qos_flows
            .iter()
            .map(|f| (f.qfi, f.five_qi))
            .collect();

        // Allocate gNB DL TEID for the N3 tunnel
        let gnb_teid = self.next_downlink_teid();

        info!(
            "PDU Session {}: UPF TEID=0x{:08x}, UPF addr={}, QFIs={:?}, gNB TEID=0x{:08x}",
            psi,
            upf_teid,
            upf_addr,
            request.qos_flows.iter().map(|f| f.qfi).collect::<Vec<_>>(),
            gnb_teid
        );

        // Store PDU session in UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.add_pdu_session(NgapPduSession {
                psi,
                qfi: Some(qfi),
                uplink_teid: gnb_teid,
                downlink_teid: upf_teid,
                upf_address: upf_addr,
                up_security_policy: policy,
                up_security: decision,
            });
        }

        // Send GTP SessionCreate to GTP task
        let resource = PduSessionResource {
            psi: psi as i32,
            qfi: Some(qfi),
            qos_flows: accepted_flows.clone(),
            // uplink_teid = UPF's N3 TEID (where the gNB sends uplink G-PDUs);
            // downlink_teid = gNB's own TEID (where the UPF sends downlink). See gtp/task.rs:148-149.
            uplink_teid: upf_teid,
            downlink_teid: gnb_teid,
            upf_address: upf_addr,
        };
        let msg = GtpMessage::SessionCreate { ue_id, resource };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send SessionCreate to GTP: {}", e);
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
            }
            let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::RadioResourcesNotAvailable,
            ))
            .unwrap_or_default();
            return Err(PduSessionResourceFailedToSetupItem {
                pdu_session_id: psi,
                transfer,
            });
        }
        info!(
            "GTP session created for PSI={}, gNB TEID=0x{:08x}",
            psi, gnb_teid
        );

        // If NAS PDU present, forward to RRC
        if let Some(ref nas_pdu) = item.nas_pdu {
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!("Failed to forward NAS PDU from Setup Request to RRC: {}", e);
            }
        }

        // TS 38.331 §5.3.5.6: establish the PDU session's user-plane DRB via an
        // RRCReconfiguration carrying the accepted QoS flows (QFIs).
        let accepted_qfis: Vec<u8> = request.qos_flows.iter().map(|f| f.qfi).collect();
        // `accepted_flows` is the same set the `SessionCreate` above carried, by
        // construction: one binding shared by both, so the GTP task's SDAP mapping and
        // the mapping the UE is told cannot drift apart.
        self.key_and_establish_drb(ue_id, psi, &accepted_flows, decision)
            .await;

        // Build the APER PDUSessionResourceSetupResponseTransfer (TS 38.413
        // §9.3.4.2) with the real gNB F-TEID and the accepted QoS flows
        let transfer_params = SetupResponseTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: gnb_ip,
                teid: gnb_teid,
            },
            accepted_qfis,
            failed_qos_flows: vec![],
        };
        match encode_setup_response_transfer(&transfer_params) {
            Ok(response_transfer) => Ok(PduSessionResourceSetupResponseItem {
                pdu_session_id: psi,
                transfer: response_transfer,
            }),
            Err(e) => {
                error!(
                    "Failed to encode SetupResponseTransfer for PSI {}: {}",
                    psi, e
                );
                let transfer = encode_setup_unsuccessful_transfer(&NgSetupFailureCause::Misc(
                    nextgsim_ngap::procedures::ng_setup::MiscCause::Unspecified,
                ))
                .unwrap_or_default();
                Err(PduSessionResourceFailedToSetupItem {
                    pdu_session_id: psi,
                    transfer,
                })
            }
        }
    }

    /// Handles PDU Session Resource Setup Request from AMF
    async fn handle_pdu_session_resource_setup(
        &mut self,
        amf_id: i32,
        stream: u16,
        setup_req: PduSessionResourceSetupRequestData,
    ) {
        info!(
            "PDU Session Resource Setup Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            setup_req.amf_ue_ngap_id,
            setup_req.ran_ue_ngap_id,
            setup_req.pdu_session_resource_setup_list.len()
        );

        // Find UE context by RAN-UE-NGAP-ID
        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values_mut()
                .find(|ctx| ctx.ran_ue_ngap_id == setup_req.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => {
                    if ctx.amf_ue_ngap_id.is_none() {
                        ctx.amf_ue_ngap_id = Some(setup_req.amf_ue_ngap_id as i64);
                    }
                    ctx.ue_id
                }
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Resource Setup",
                        setup_req.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        let mut setup_response_items = Vec::new();
        let mut failed_items = Vec::new();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        for item in &setup_req.pdu_session_resource_setup_list {
            match self.setup_one_pdu_session(ue_id, item, gnb_ip).await {
                Ok(resp) => setup_response_items.push(resp),
                Err(failed) => failed_items.push(failed),
            }
        }

        // Build and send PDU Session Resource Setup Response
        let response_params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: setup_req.amf_ue_ngap_id,
            ran_ue_ngap_id: setup_req.ran_ue_ngap_id,
            setup_list: if setup_response_items.is_empty() {
                None
            } else {
                Some(setup_response_items)
            },
            failed_list: if failed_items.is_empty() {
                None
            } else {
                Some(failed_items)
            },
        };

        match encode_pdu_session_resource_setup_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Setup Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Setup Response: {}",
                    e
                );
            }
        }
    }

    /// Handles PDU Session Resource Modify Request from AMF
    async fn handle_pdu_session_resource_modify(
        &mut self,
        amf_id: i32,
        stream: u16,
        modify_req: PduSessionResourceModifyRequestData,
    ) {
        info!(
            "PDU Session Resource Modify Request: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            modify_req.amf_ue_ngap_id,
            modify_req.ran_ue_ngap_id,
            modify_req.pdu_session_resource_modify_list.len()
        );

        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values()
                .find(|ctx| ctx.ran_ue_ngap_id == modify_req.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => ctx.ue_id,
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Modify",
                        modify_req.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        let mut modify_response_items = Vec::new();
        let mut failed_items = Vec::new();
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        for item in &modify_req.pdu_session_resource_modify_list {
            let psi = item.pdu_session_id;

            // Decode the APER PDUSessionResourceModifyRequestTransfer (§9.3.4.3)
            let request = match decode_modify_request_transfer(&item.transfer) {
                Ok(req) => req,
                Err(e) => {
                    warn!(
                        "PDU Session {} modify: failed to decode ModifyRequestTransfer: {}",
                        psi, e
                    );
                    if let Ok(transfer) = encode_modify_unsuccessful_transfer(
                        &NgSetupFailureCause::Protocol(ProtocolCause::TransferSyntaxError),
                    ) {
                        failed_items.push(PduSessionResourceFailedToModifyItem {
                            pdu_session_id: psi,
                            transfer,
                        });
                    }
                    continue;
                }
            };

            // The session must already exist to be modified
            let existing = self
                .ue_contexts
                .get(&ue_id)
                .and_then(|ctx| ctx.get_pdu_session(psi).cloned());
            let Some(existing) = existing else {
                warn!("PDU Session {} modify: session not established", psi);
                if let Ok(transfer) = encode_modify_unsuccessful_transfer(
                    &NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownPduSessionId),
                ) {
                    failed_items.push(PduSessionResourceFailedToModifyItem {
                        pdu_session_id: psi,
                        transfer,
                    });
                }
                continue;
            };

            // Apply UL tunnel modification if requested, otherwise keep current
            let (upf_teid, upf_addr) = match request.new_ul_tunnel {
                Some(tunnel) => (tunnel.teid, tunnel.address),
                None => (existing.downlink_teid, existing.upf_address),
            };
            let qfi = request
                .qos_flows_add_or_modify
                .first()
                .copied()
                .or(existing.qfi)
                .unwrap_or(1);

            // Re-allocate the gNB DL TEID only when the UL tunnel changed
            let gnb_teid = if request.new_ul_tunnel.is_some() {
                self.next_downlink_teid()
            } else {
                existing.uplink_teid
            };

            info!(
                "PDU Session {} modify: UPF TEID=0x{:08x}, gNB TEID=0x{:08x}",
                psi, upf_teid, gnb_teid
            );

            // Update UE context
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
                ctx.add_pdu_session(NgapPduSession {
                    psi,
                    qfi: Some(qfi),
                    uplink_teid: gnb_teid,
                    downlink_teid: upf_teid,
                    upf_address: upf_addr,
                    // The Modify Request Transfer carries no `SecurityIndication`
                    // (TS 38.413 §9.3.4.3), so the session keeps the policy it was
                    // set up with. Carried across from `existing` explicitly:
                    // rebuilding the session with the local default would silently
                    // downgrade a `required` session on any QoS change.
                    up_security_policy: existing.up_security_policy,
                    up_security: existing.up_security,
                });
            }

            // Send SessionModify to GTP task
            let resource = PduSessionResource {
                psi: psi as i32,
                qfi: Some(qfi),
                // The Modify Request Transfer's `qosFlowAddOrModifyRequestList`
                // decodes to QFIs alone (`Vec<u8>`) -- this codec does not carry the
                // per-flow `QosFlowLevelQosParameters`, so there is no 5QI to rebuild
                // a mapping from. `None` each, which puts every modified flow on the
                // default DRB (TS 37.324 §5.3.1's rule for a flow with no explicit
                // mapping). That is a real ceiling: a GBR flow ADDED by a Modify would
                // land on the default bearer rather than the GBR one until the codec
                // surfaces the 5QI. Stated rather than papered over with a guessed
                // 5QI, which would put flows on a GBR bearer the UE was never told
                // about — the reconfiguration this path does not send.
                qos_flows: request
                    .qos_flows_add_or_modify
                    .iter()
                    .map(|&qfi| (qfi, None))
                    .collect(),
                // uplink_teid = UPF's N3 TEID (where the gNB sends uplink G-PDUs);
                // downlink_teid = gNB's own TEID (where the UPF sends downlink). See gtp/task.rs:148-149.
                uplink_teid: upf_teid,
                downlink_teid: gnb_teid,
                upf_address: upf_addr,
            };
            let msg = GtpMessage::SessionModify { ue_id, resource };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionModify to GTP: {}", e);
                if let Ok(transfer) =
                    encode_modify_unsuccessful_transfer(&NgSetupFailureCause::RadioNetwork(
                        RadioNetworkCause::RadioResourcesNotAvailable,
                    ))
                {
                    failed_items.push(PduSessionResourceFailedToModifyItem {
                        pdu_session_id: psi,
                        transfer,
                    });
                }
                continue;
            }

            // Forward NAS PDU to RRC if present
            if let Some(ref nas_pdu) = item.nas_pdu {
                let msg = RrcMessage::NasDelivery {
                    ue_id,
                    pdu: OctetString::from_slice(nas_pdu),
                };
                let _ = self.task_base.rrc_tx.send(msg).await;
            }

            // Build the APER PDUSessionResourceModifyResponseTransfer (§9.3.4.4)
            let transfer_params = ModifyResponseTransferParams {
                dl_tunnel: request.new_ul_tunnel.map(|_| GtpTunnelInfo {
                    address: gnb_ip,
                    teid: gnb_teid,
                }),
                ul_tunnel: None,
                modified_qfis: request.qos_flows_add_or_modify.clone(),
                failed_qos_flows: vec![],
            };
            match encode_modify_response_transfer(&transfer_params) {
                Ok(response_transfer) => {
                    modify_response_items.push(PduSessionResourceModifyResponseItem {
                        pdu_session_id: psi,
                        transfer: response_transfer,
                    });
                }
                Err(e) => {
                    error!(
                        "Failed to encode ModifyResponseTransfer for PSI {}: {}",
                        psi, e
                    );
                }
            }
        }

        let response_params = PduSessionResourceModifyResponseParams {
            amf_ue_ngap_id: modify_req.amf_ue_ngap_id,
            ran_ue_ngap_id: modify_req.ran_ue_ngap_id,
            modify_list: if modify_response_items.is_empty() {
                None
            } else {
                Some(modify_response_items)
            },
            failed_list: if failed_items.is_empty() {
                None
            } else {
                Some(failed_items)
            },
        };

        match encode_pdu_session_resource_modify_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Modify Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Modify Response: {}",
                    e
                );
            }
        }
    }

    /// Handles PDU Session Resource Release Command from AMF
    async fn handle_pdu_session_resource_release(
        &mut self,
        amf_id: i32,
        stream: u16,
        release_cmd: PduSessionResourceReleaseCommandData,
    ) {
        info!(
            "PDU Session Resource Release Command: amf_ue_ngap_id={}, ran_ue_ngap_id={}, {} items",
            release_cmd.amf_ue_ngap_id,
            release_cmd.ran_ue_ngap_id,
            release_cmd.pdu_session_resource_to_release_list.len()
        );

        let ue_id = {
            let ue_ctx = self
                .ue_contexts
                .values()
                .find(|ctx| ctx.ran_ue_ngap_id == release_cmd.ran_ue_ngap_id as i64);
            match ue_ctx {
                Some(ctx) => ctx.ue_id,
                None => {
                    warn!(
                        "No UE context for RAN-UE-NGAP-ID {} in PDU Session Release",
                        release_cmd.ran_ue_ngap_id
                    );
                    return;
                }
            }
        };

        // Forward NAS PDU if present
        if let Some(ref nas_pdu) = release_cmd.nas_pdu {
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: OctetString::from_slice(nas_pdu),
            };
            let _ = self.task_base.rrc_tx.send(msg).await;
        }

        let mut released_items = Vec::new();

        for item in &release_cmd.pdu_session_resource_to_release_list {
            let psi = item.pdu_session_id;

            // Decode the APER PDUSessionResourceReleaseCommandTransfer cause (§9.3.4.11)
            match decode_release_command_transfer(&item.transfer) {
                Ok(cause) => {
                    info!("PDU Session {} release cause: {:?}", psi, cause);
                }
                Err(e) => {
                    warn!(
                        "PDU Session {}: failed to decode ReleaseCommandTransfer: {}",
                        psi, e
                    );
                }
            }

            // Remove from UE context
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
            }

            // Release GTP session
            let msg = GtpMessage::SessionRelease {
                ue_id,
                psi: psi as i32,
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionRelease to GTP: {}", e);
            }

            info!("PDU Session {} released for ue_id={}", psi, ue_id);

            // Real APER PDUSessionResourceReleaseResponseTransfer (§9.3.4.12):
            // a single SEQUENCE preamble octet
            let transfer = match encode_release_response_transfer() {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("Failed to encode ReleaseResponseTransfer: {}", e);
                    continue;
                }
            };
            released_items.push(PduSessionResourceReleasedItem {
                pdu_session_id: psi,
                transfer,
            });
        }

        let response_params = PduSessionResourceReleaseResponseParams {
            amf_ue_ngap_id: release_cmd.amf_ue_ngap_id,
            ran_ue_ngap_id: release_cmd.ran_ue_ngap_id,
            released_list: released_items,
        };

        match encode_pdu_session_resource_release_response(&response_params) {
            Ok(response_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, response_bytes)
                    .await;
                info!("PDU Session Resource Release Response sent to AMF");
            }
            Err(e) => {
                error!(
                    "Failed to encode PDU Session Resource Release Response: {}",
                    e
                );
            }
        }
    }

    /// Handles UE Context Release Command from AMF
    async fn handle_ue_context_release_command(
        &mut self,
        amf_id: i32,
        stream: u16,
        release_cmd: nextgsim_ngap::procedures::ue_context_release::UeContextReleaseCommandData,
    ) {
        use nextgsim_ngap::procedures::ue_context_release::UeNgapIds;

        info!(
            "UE Context Release Command: ids={:?}, cause={:?}",
            release_cmd.ue_ngap_ids, release_cmd.cause
        );

        // Find UE by AMF or RAN NGAP ID
        let ue_id = match &release_cmd.ue_ngap_ids {
            UeNgapIds::Pair {
                amf_ue_ngap_id,
                ran_ue_ngap_id,
            } => self
                .ue_contexts
                .values()
                .find(|ctx| {
                    ctx.ran_ue_ngap_id == *ran_ue_ngap_id as i64
                        || ctx.amf_ue_ngap_id == Some(*amf_ue_ngap_id as i64)
                })
                .map(|ctx| ctx.ue_id),
            UeNgapIds::AmfOnly(amf_id_val) => self
                .ue_contexts
                .values()
                .find(|ctx| ctx.amf_ue_ngap_id == Some(*amf_id_val as i64))
                .map(|ctx| ctx.ue_id),
        };

        let ue_id = match ue_id {
            Some(id) => id,
            None => {
                warn!("No UE context found for UE Context Release Command");
                return;
            }
        };

        // The AMF answered, so TNGRELOCoverall has done its job (TS 38.413 §8.3.3.4).
        // Cancelled rather than left to expire: an expiry after the release completed
        // would act on a ue_id this release is about to free, and ids are reused.
        self.guard_timers
            .cancel(GuardTimer::UeContextRelease { ue_id });

        // Release all PDU sessions for this UE via GTP
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }

        // Notify RRC
        self.send_an_release(ue_id).await;

        // Get NGAP IDs before deletion
        let (amf_ue_ngap_id, ran_ue_ngap_id) = {
            let ctx = self.ue_contexts.get(&ue_id);
            match ctx {
                Some(c) => (
                    c.amf_ue_ngap_id.unwrap_or(0) as u64,
                    c.ran_ue_ngap_id as u32,
                ),
                None => (0, 0),
            }
        };

        // Delete UE context
        self.delete_ue_context(ue_id);

        // Send UE Context Release Complete
        let complete_params = UeContextReleaseCompleteParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
        };

        match encode_ue_context_release_complete(&complete_params) {
            Ok(complete_bytes) => {
                self.send_ngap_ue_associated(amf_id, stream, complete_bytes)
                    .await;
                info!(
                    "UE Context Release Complete sent to AMF for ue_id={}",
                    ue_id
                );
            }
            Err(e) => {
                error!("Failed to encode UE Context Release Complete: {}", e);
            }
        }
    }

    /// Handles Initial NAS delivery from RRC (Initial UE Message)
    async fn handle_initial_nas_delivery(
        &mut self,
        ue_id: i32,
        pdu: OctetString,
        rrc_establishment_cause: i64,
        _s_tmsi: Option<crate::tasks::GutiMobileIdentity>,
        s_nssai_list: Vec<nextgsim_common::SNssai>,
    ) {
        debug!(
            "Initial NAS delivery: ue_id={}, cause={}, pdu_len={}, s_nssai_count={}",
            ue_id,
            rrc_establishment_cause,
            pdu.len(),
            s_nssai_list.len()
        );

        // The UE's requested S-NSSAI(s) (from RRCSetupComplete) drive slice-aware
        // AMF selection and are echoed as the Allowed NSSAI in the Initial UE
        // Message (TS 38.413 §8.6.1.2). Map the common S-NSSAI onto the NGAP type.
        let requested_nssai: Vec<SNssai> = s_nssai_list
            .iter()
            .map(|s| SNssai {
                sst: s.sst,
                sd: s.sd,
            })
            .collect();
        let serving_plmn = self.task_base.config.plmn.encode();

        // Select an AMF for this UE, honouring the requested slice(s)
        let amf_id = match self.select_amf(&serving_plmn, &requested_nssai) {
            Some(id) => id,
            None => {
                warn!("No AMF available for UE {}", ue_id);
                return;
            }
        };

        // TS 38.413 §8.7.7: honour the AMFTrafficLoadReductionIndication the
        // selected AMF gave in OVERLOAD START. Before this the percentage was
        // stored and read only in a log line, so an AMF that asked for a 75%
        // reduction kept receiving 100% of the gNB's Initial UE Messages
        // (issue #41).
        //
        // The UE context is deliberately NOT created before this check: creating
        // one and then abandoning it leaks a RAN UE NGAP ID and a stream for a UE
        // the AMF never heard of.
        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
            if !ctx.admit_initial_ue_message() {
                warn!(
                    "Initial UE Message for UE {} blocked: AMF[{}] asked for a {}% \
                     traffic reduction (TS 38.413 §8.7.7)",
                    ue_id,
                    amf_id,
                    ctx.traffic_load_reduction.unwrap_or(0)
                );
                return;
            }
        }

        // Create UE context
        let ran_ue_ngap_id = match self.create_ue_context(ue_id, amf_id) {
            Some(id) => id,
            None => {
                error!("Failed to create UE context for UE {}", ue_id);
                return;
            }
        };

        // Get stream for this UE
        let stream = self
            .ue_contexts
            .get(&ue_id)
            .map(|ctx| ctx.stream_id)
            .unwrap_or(1); // Use stream 1 for UE-associated signaling

        // Build User Location Information
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        // Convert RRC establishment cause to NGAP value
        let rrc_cause = match rrc_establishment_cause {
            0 => RrcEstablishmentCauseValue::Emergency,
            1 => RrcEstablishmentCauseValue::HighPriorityAccess,
            2 => RrcEstablishmentCauseValue::MtAccess,
            3 => RrcEstablishmentCauseValue::MoSignalling,
            4 => RrcEstablishmentCauseValue::MoData,
            5 => RrcEstablishmentCauseValue::MoVoiceCall,
            6 => RrcEstablishmentCauseValue::MoVideoCall,
            7 => RrcEstablishmentCauseValue::MoSms,
            8 => RrcEstablishmentCauseValue::MpsHighPriorityAccess,
            9 => RrcEstablishmentCauseValue::McsHighPriorityAccess,
            _ => RrcEstablishmentCauseValue::MoSignalling, // Default
        };

        // Convert GutiMobileIdentity to FiveGSTmsi if provided
        let five_g_s_tmsi = _s_tmsi.map(|guti| {
            // Convert 5G-TMSI (u32) to bytes
            let tmsi_bytes = guti.tmsi.to_be_bytes();
            FiveGSTmsi {
                amf_set_id: guti.amf_set_id,
                amf_pointer: guti.amf_pointer,
                five_g_tmsi: tmsi_bytes,
            }
        });

        let params = InitialUeMessageParams {
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            nas_pdu: pdu.data().to_vec(),
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: Tai {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
                time_stamp: None,
            },
            rrc_establishment_cause: rrc_cause,
            five_g_s_tmsi,
            amf_set_id: None,
            ue_context_request: Some(UeContextRequestValue::Requested),
            // Carry the UE's requested slice(s) as the Allowed NSSAI so the AMF
            // sees the slice context used for selection (TS 38.413 §8.6.1.2).
            allowed_nssai: if requested_nssai.is_empty() {
                None
            } else {
                Some(
                    requested_nssai
                        .iter()
                        .map(|s| AllowedSnssai {
                            sst: s.sst,
                            sd: s.sd,
                        })
                        .collect(),
                )
            },
        };

        match encode_initial_ue_message(&params) {
            Ok(bytes) => {
                info!(
                    "Sending Initial UE Message: ue_id={}, ran_ue_ngap_id={}, amf_id={}, stream={}, len={}",
                    ue_id, ran_ue_ngap_id, amf_id, stream, bytes.len()
                );
                self.send_ngap_ue_associated(amf_id, stream, bytes).await;
            }
            Err(e) => {
                error!("Failed to encode Initial UE Message: {}", e);
                // Clean up UE context on failure
                self.delete_ue_context(ue_id);
                return;
            }
        }

        // Notify GTP task of new UE context
        self.send_gtp_ue_context_update(ue_id, None).await;
    }

    /// Handles Uplink NAS delivery from RRC
    async fn handle_uplink_nas_delivery(&mut self, ue_id: i32, pdu: OctetString) {
        debug!(
            "Uplink NAS delivery: ue_id={}, pdu_len={}",
            ue_id,
            pdu.len()
        );

        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("UE context not found for uplink NAS: ue_id={}", ue_id);
                return;
            }
        };

        let amf_ue_ngap_id = match ctx.amf_ue_ngap_id {
            Some(id) => id as u64,
            None => {
                warn!("AMF UE NGAP ID not set for UE {}", ue_id);
                return;
            }
        };

        let ran_ue_ngap_id = ctx.ran_ue_ngap_id as u32;
        let amf_ctx_id = ctx.amf_ctx_id;
        let stream = ctx.stream_id;

        // Build User Location Information
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = UplinkNasTransportParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            nas_pdu: pdu.data().to_vec(),
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: Tai {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
                time_stamp: None,
            },
        };

        match encode_uplink_nas_transport(&params) {
            Ok(bytes) => {
                info!(
                    "Sending Uplink NAS Transport: ue_id={}, ran_ue_ngap_id={}, amf_ue_ngap_id={}, amf_ctx_id={}, stream={}, len={}",
                    ue_id, ran_ue_ngap_id, amf_ue_ngap_id, amf_ctx_id, stream, bytes.len()
                );
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
            }
            Err(e) => {
                error!("Failed to encode Uplink NAS Transport: {}", e);
            }
        }
    }

    /// Delivers downlink NAS to RRC
    #[allow(dead_code)]
    async fn deliver_downlink_nas(&self, ue_id: i32, pdu: OctetString) {
        let msg = RrcMessage::NasDelivery { ue_id, pdu };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to deliver downlink NAS to RRC: {}", e);
        }
    }

    // ========================================================================
    // PDU Session Management
    // ========================================================================

    /// Handles PDU Session Resource Setup from AMF
    #[allow(dead_code)]
    async fn handle_pdu_session_setup(
        &mut self,
        ue_id: i32,
        psi: u8,
        qfi: Option<u8>,
        uplink_teid: u32,
        upf_address: std::net::IpAddr,
    ) {
        let downlink_teid = self.next_downlink_teid();

        // Add session to UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            let session = NgapPduSession {
                psi,
                qfi,
                uplink_teid,
                downlink_teid,
                upf_address,
                // No `SecurityIndication` reaches this path -- it adds a session
                // from a handover or a context transfer, not from a Setup Request --
                // so the local default applies and nothing is protected until an
                // SMF states a policy. Stated rather than inherited, because there
                // is no source session here to inherit from.
                up_security_policy: UpSecurityPolicy::locally_configured_default(),
                up_security: DrbSecurityDecision::default(),
            };
            ctx.add_pdu_session(session);
        }

        // Notify GTP task
        let resource = PduSessionResource {
            psi: psi as i32,
            qfi,
            // This path adds a session from a handover or context transfer and never
            // sees a `QosFlowSetupInfo`, so the only flow it can name is the default
            // one it was handed -- with no 5QI, since nothing told it one. Consistent
            // with the local security default the same function applies just above:
            // where no policy arrived, state the conservative one rather than invent.
            qos_flows: qfi.map(|qfi| (qfi, None)).into_iter().collect(),
            uplink_teid,
            downlink_teid,
            upf_address,
        };

        let msg = GtpMessage::SessionCreate { ue_id, resource };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send PDU session create to GTP: {}", e);
        }

        info!(
            "PDU session setup: ue_id={}, psi={}, ul_teid={:08x}, dl_teid={:08x}",
            ue_id, psi, uplink_teid, downlink_teid
        );
    }

    /// Handles PDU Session Resource Release from AMF
    #[allow(dead_code)]
    async fn handle_pdu_session_release(&mut self, ue_id: i32, psi: u8) {
        // Remove session from UE context
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.remove_pdu_session(psi);
        }

        // Notify GTP task
        let msg = GtpMessage::SessionRelease {
            ue_id,
            psi: psi as i32,
        };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send PDU session release to GTP: {}", e);
        }

        info!("PDU session release: ue_id={}, psi={}", ue_id, psi);
    }

    // ========================================================================
    // Radio Link Failure
    // ========================================================================

    /// Handles radio link failure notification from RRC
    async fn handle_radio_link_failure(&mut self, ue_id: i32) {
        info!("Radio link failure: ue_id={}", ue_id);
        self.handle_ue_context_release_request(ue_id, UeReleaseRequestCause::RadioLinkFailure)
            .await;
    }

    /// Handles UE Context Release Request (from App or RRC)
    /// Sends a PDU Session Resource Notify (TS 38.413 §8.3.5, issue #98).
    ///
    /// The conformant way for the gNB to tell the AMF that specific PDU session
    /// resources are released, or that specific QoS flows are no longer fulfilled,
    /// without the AMF having asked. Per-SESSION granularity, unlike
    /// `UE CONTEXT RELEASE REQUEST`, which tears down the whole UE context and so
    /// takes down sessions on unaffected UPFs with it.
    ///
    /// There is **no response message** for this procedure, so nothing is queued
    /// on a guard timer and no context state changes here. The gNB has already
    /// released whatever it is reporting; this tells the 5GC so the SMF and AMF
    /// stop believing otherwise. That also means delivery is best-effort: an AMF
    /// that is down when this is sent never learns, and there is no retry (see the
    /// ceiling in the spec).
    async fn send_pdu_session_resource_notify(
        &mut self,
        ue_id: i32,
        released_sessions: Vec<(u8, NotifyCause)>,
        notified_sessions: Vec<(u8, Vec<NotifiedQosFlow>)>,
    ) {
        let Some(ctx) = self.ue_contexts.get(&ue_id) else {
            warn!("UE context not found for PDU Session Resource Notify: ue_id={ue_id}");
            return;
        };
        let amf_ctx_id = ctx.amf_ctx_id;
        let ran_ue_ngap_id = ctx.ran_ue_ngap_id;
        let stream = ctx.stream_id;
        // The Notify is UE-associated and AMF-UE-NGAP-ID is MANDATORY, so a UE the
        // AMF has not yet given an ID cannot be the subject of one. Reported rather
        // than sent with a fabricated ID, which the AMF would fail to resolve.
        let Some(amf_ue_ngap_id) = ctx.amf_ue_ngap_id else {
            warn!(
                "Cannot send PDU Session Resource Notify for ue_id={ue_id}: no \
                 AMF-UE-NGAP-ID yet, and the IE is mandatory"
            );
            return;
        };

        let params = PduSessionResourceNotifyParams {
            amf_ue_ngap_id: amf_ue_ngap_id as u64,
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            notified_sessions: notified_sessions
                .into_iter()
                .map(|(pdu_session_id, notified_flows)| NotifiedSession {
                    pdu_session_id,
                    notified_flows,
                    // Per-flow release within a surviving session is not driven by
                    // any caller yet; the encoder supports it.
                    released_flows: Vec::new(),
                })
                .collect(),
            released_sessions: released_sessions
                .into_iter()
                .map(|(pdu_session_id, cause)| ReleasedSession {
                    pdu_session_id,
                    cause,
                })
                .collect(),
        };

        match encode_pdu_session_resource_notify(&params) {
            Ok(bytes) => {
                info!(
                    "Sending PDU Session Resource Notify: ue_id={ue_id}, \
                     ran_ue_ngap_id={ran_ue_ngap_id}, amf_ue_ngap_id={amf_ue_ngap_id}, \
                     released={}, notified={}",
                    params.released_sessions.len(),
                    params.notified_sessions.len()
                );
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
            }
            // The encoder refuses a Notify that reports nothing, which is a caller
            // bug rather than a peer problem -- so it is logged and dropped rather
            // than escalated.
            Err(e) => error!("Failed to encode PDU Session Resource Notify: {e}"),
        }
    }

    async fn handle_ue_context_release_request(
        &mut self,
        ue_id: i32,
        cause: UeReleaseRequestCause,
    ) {
        info!(
            "UE context release request: ue_id={}, cause={:?}",
            ue_id, cause
        );

        let ctx = match self.ue_contexts.get(&ue_id) {
            Some(c) => c,
            None => {
                warn!("UE context not found for release: ue_id={}", ue_id);
                return;
            }
        };

        let amf_ctx_id = ctx.amf_ctx_id;
        let ran_ue_ngap_id = ctx.ran_ue_ngap_id;
        let amf_ue_ngap_id = ctx.amf_ue_ngap_id;
        let stream = ctx.stream_id;

        // Send UE Context Release Request to AMF if we have the AMF UE NGAP ID
        if let Some(amf_id) = amf_ue_ngap_id {
            // Map the local trigger to the NGAP cause (TS 38.413 §9.3.1.2)
            let ngap_cause = match cause {
                UeReleaseRequestCause::UserTriggered => {
                    NgSetupFailureCause::Nas(NasCause::NormalRelease)
                }
                UeReleaseRequestCause::RadioLinkFailure => {
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost)
                }
                UeReleaseRequestCause::RanOriginated => NgSetupFailureCause::RadioNetwork(
                    RadioNetworkCause::ReleaseDueToNgranGeneratedReason,
                ),
            };

            let params = UeContextReleaseRequestParams {
                amf_ue_ngap_id: amf_id as u64,
                ran_ue_ngap_id: ran_ue_ngap_id as u32,
                cause: ngap_cause,
            };

            match encode_ue_context_release_request(&params) {
                Ok(bytes) => {
                    info!(
                        "Sending UE Context Release Request: ue_id={}, ran_ue_ngap_id={}, amf_ue_ngap_id={}, amf_ctx_id={}, cause={:?}",
                        ue_id, ran_ue_ngap_id, amf_id, amf_ctx_id, cause
                    );
                    self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                        .await;

                    // Keep the context in Releasing state; cleanup happens when the
                    // AMF answers with UE Context Release Command -- or when
                    // TNGRELOCoverall expires, which is what stops "waiting for the
                    // AMF" from meaning "forever" (TS 38.413 §8.3.3.4).
                    if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                        ctx.on_context_release();
                    }
                    let overall =
                        Duration::from_secs(self.task_base.config.ngap_tngreloc_overall_secs);
                    self.guard_timers.start(
                        Instant::now(),
                        overall,
                        GuardTimer::UeContextRelease { ue_id },
                    );
                    return;
                }
                Err(e) => {
                    error!("Failed to encode UE Context Release Request: {}", e);
                    // Fall through to local cleanup
                }
            }
        } else {
            info!(
                "UE context release without AMF UE NGAP ID: ue_id={}, ran_ue_ngap_id={}",
                ue_id, ran_ue_ngap_id
            );
        }

        // Local cleanup (no AMF association for this UE, or encoding failed).
        self.local_release_ue(ue_id).await;
    }

    /// Locally release a UE's NG-associated resources without sending any further
    /// NGAP message to the AMF: delete the NGAP UE context, tell RRC to release
    /// the access-network resources, and tell GTP-U to tear down the bearers.
    ///
    /// Used for AMF-less release, and for local release on a received ERROR
    /// INDICATION (TS 38.413 §10.6, Handling of AP ID) — where echoing another
    /// NGAP message back to the AMF would be wrong.
    async fn local_release_ue(&mut self, ue_id: i32) {
        self.delete_ue_context(ue_id);
        self.send_an_release(ue_id).await;
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }
    }

    /// Handle an inbound ERROR INDICATION (TS 38.413 §8.7.5).
    ///
    /// A received Error Indication is processed locally and MUST NOT be answered
    /// with a further Error Indication — doing so risks an on-wire ping-pong. When
    /// it reports AP IDs that identify a UE-associated logical NG-connection, that
    /// connection's resources are released locally (TS 38.413 §10.6, Handling of
    /// AP ID). A non-UE-associated Error Indication (or one whose AP IDs match no
    /// known UE) is logged and dropped.
    async fn handle_error_indication(&mut self, amf_id: i32, err_ind: ErrorIndicationData) {
        info!(
            "Received ERROR INDICATION from AMF[{}]: amf_ue_ngap_id={:?}, ran_ue_ngap_id={:?}, cause={:?}",
            amf_id, err_ind.amf_ue_ngap_id, err_ind.ran_ue_ngap_id, err_ind.cause
        );

        let ue_id = self
            .ue_contexts
            .values()
            .find(|ctx| {
                matches!(err_ind.ran_ue_ngap_id, Some(r) if ctx.ran_ue_ngap_id == r as i64)
                    || matches!(err_ind.amf_ue_ngap_id, Some(a) if ctx.amf_ue_ngap_id == Some(a as i64))
            })
            .map(|ctx| ctx.ue_id);

        match ue_id {
            Some(id) => {
                warn!(
                    "ERROR INDICATION references UE {}; locally releasing its NG connection (TS 38.413 §10.6)",
                    id
                );
                self.local_release_ue(id).await;
            }
            None => {
                debug!(
                    "ERROR INDICATION is not associated with a known UE context; processed with no local release and no reply"
                );
            }
        }
    }

    // ========================================================================
    // NGAP guard timers (TS 38.413 §8.3.3.4, §8.4.1.2, §8.7.1.3)
    // ========================================================================

    /// Act on every guard timer due at `now`.
    ///
    /// `now` is a parameter so the whole expiry path is testable without sleeping; the
    /// run loop passes `Instant::now()`.
    async fn process_expired_guard_timers(&mut self, now: Instant) {
        for timer in self.guard_timers.expired(now) {
            match timer {
                GuardTimer::UeContextRelease { ue_id } => {
                    // TS 38.413 §8.3.3.4: the AMF never answered the UE Context Release
                    // Request. Release locally rather than leave the context in
                    // `Releasing` forever -- every one that stays holds a
                    // RAN-UE-NGAP-ID, and the id space is finite.
                    warn!(
                        "TNGRELOCoverall expired for UE[{ue_id}]: no UE Context Release \
                         Command from the AMF, releasing locally"
                    );
                    self.local_release_ue(ue_id).await;
                }
                GuardTimer::HandoverPreparation { ue_id } => {
                    // TS 38.413 §8.4.1.2: on expiry the source cancels the preparation.
                    // Cancelling means SENDING Handover Cancel (§8.4.5), not just
                    // dropping the wait: the AMF and the target may already hold
                    // resources for this handover, and only the cancel releases them.
                    // `encode_handover_cancel` existed in nextgsim-ngap with no
                    // production caller -- this is it.
                    warn!(
                        "TNGRELOCprep expired for UE[{ue_id}]: cancelling handover \
                         preparation, UE stays on the source cell"
                    );
                    self.send_handover_cancel(ue_id).await;
                }
                GuardTimer::NgSetupRetry { amf_id } => {
                    // TS 38.413 §8.7.1.3: the Time to Wait has elapsed, so NG Setup may
                    // be re-initiated toward this AMF.
                    let still_known = self.amf_contexts.contains_key(&amf_id);
                    if still_known {
                        info!("Time to Wait elapsed, re-initiating NG Setup toward AMF {amf_id}");
                        self.send_ng_setup_request(amf_id).await;
                    } else {
                        debug!(
                            "Time to Wait elapsed for AMF {amf_id}, which is no longer \
                             configured; no retry"
                        );
                    }
                }
            }
        }
    }

    /// Send Handover Cancel for `ue_id` (TS 38.413 §8.4.5).
    ///
    /// The cause is `tNGRELOCprep-expiry`, which is the value TS 38.413 defines for
    /// exactly this: a preparation the source abandoned because its own supervision
    /// timer ran out. (`tNGRELOCoverall-expiry` is the sibling value and belongs to the
    /// other timer, so using it here would misreport which one fired.)
    async fn send_handover_cancel(&mut self, ue_id: i32) {
        let Some(ctx) = self.ue_contexts.get(&ue_id) else {
            debug!("Handover Cancel for UE[{ue_id}]: the context is already gone");
            return;
        };
        let (Some(amf_ue_ngap_id), ran_ue_ngap_id, amf_ctx_id, stream) = (
            ctx.amf_ue_ngap_id,
            ctx.ran_ue_ngap_id,
            ctx.amf_ctx_id,
            ctx.stream_id,
        ) else {
            // Without an AMF UE NGAP ID there is no UE-associated signalling connection
            // to cancel on, and the IE is mandatory.
            warn!(
                "Handover Cancel for UE[{ue_id}] skipped: no AMF UE NGAP ID, so the \
                 mandatory IE cannot be filled"
            );
            return;
        };

        let params = HandoverCancelParams {
            amf_ue_ngap_id: amf_ue_ngap_id as u64,
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            cause: HandoverCause::RadioNetwork(RadioNetworkCause::TngrelocPrepExpiry),
        };
        match encode_handover_cancel(&params) {
            Ok(bytes) => {
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
                info!(
                    "Sent Handover Cancel: ue_id={ue_id}, ran_ue_ngap_id={ran_ue_ngap_id}, \
                     cause=tNGRELOCprep-expiry"
                );
            }
            Err(e) => error!("Failed to encode Handover Cancel for UE[{ue_id}]: {e}"),
        }
    }

    // ========================================================================
    // NGAP PDU Handling
    // ========================================================================

    /// Handles received NGAP PDU from SCTP
    ///
    /// Public because it is a real message-handler entry point
    /// (`NgapMessage::ReceiveNgapPdu`) also driven directly by the in-process
    /// paging harness (`tests/src/paging_mt_service_request.rs`), which feeds it
    /// an AMF-encoded PDU so the NGAP decode is production code rather than a
    /// hand-built `PagingData`.
    pub async fn handle_ngap_pdu(&mut self, client_id: i32, stream: u16, pdu: OctetString) {
        debug!(
            "Received NGAP PDU: client_id={}, stream={}, len={}",
            client_id,
            stream,
            pdu.len()
        );

        let pdu_bytes = pdu.data();

        // Check AMF state
        let amf_state = self
            .amf_contexts
            .get(&client_id)
            .map(|ctx| ctx.state)
            .unwrap_or(AmfState::NotConnected);

        match amf_state {
            AmfState::WaitingNgSetup => {
                // Expecting NG Setup Response or Failure
                if self.handle_ng_setup_response(client_id, pdu_bytes) {
                    self.update_initialization_status().await;
                } else if self.handle_ng_setup_failure(client_id, pdu_bytes) {
                    // Already handled
                } else {
                    warn!("Unexpected NGAP PDU while waiting for NG Setup response");
                }
            }
            AmfState::Ready | AmfState::Overloaded => {
                // Handle operational messages
                // Try to decode as Downlink NAS Transport
                if let Ok(dl_nas) = decode_downlink_nas_transport(pdu_bytes) {
                    self.handle_downlink_nas_transport(client_id, stream, dl_nas)
                        .await;
                } else if let Ok(setup_req) = decode_pdu_session_resource_setup_request(pdu_bytes) {
                    self.handle_pdu_session_resource_setup(client_id, stream, setup_req)
                        .await;
                } else if let Ok(modify_req) = decode_pdu_session_resource_modify_request(pdu_bytes)
                {
                    self.handle_pdu_session_resource_modify(client_id, stream, modify_req)
                        .await;
                } else if let Ok(release_cmd) =
                    decode_pdu_session_resource_release_command(pdu_bytes)
                {
                    self.handle_pdu_session_resource_release(client_id, stream, release_cmd)
                        .await;
                } else if let Ok(ue_release_cmd) = decode_ue_context_release_command(pdu_bytes) {
                    self.handle_ue_context_release_command(client_id, stream, ue_release_cmd)
                        .await;
                } else if let Ok(ho_cmd) = decode_handover_command(pdu_bytes) {
                    self.handle_handover_command(client_id, stream, ho_cmd)
                        .await;
                } else if let Ok(ho_req) = decode_handover_request(pdu_bytes) {
                    self.handle_handover_request(client_id, stream, ho_req)
                        .await;
                } else if let Ok(ho_fail) = decode_handover_preparation_failure(pdu_bytes) {
                    self.handle_handover_preparation_failure(client_id, stream, ho_fail)
                        .await;
                } else if let Ok(ps_ack) = decode_path_switch_request_acknowledge(pdu_bytes) {
                    self.handle_path_switch_request_acknowledge(client_id, stream, ps_ack)
                        .await;
                } else if let Ok(ps_fail) = decode_path_switch_request_failure(pdu_bytes) {
                    self.handle_path_switch_request_failure(client_id, stream, ps_fail)
                        .await;
                } else if let Ok(ics_req) = decode_initial_context_setup_request(pdu_bytes) {
                    self.handle_initial_context_setup_request(client_id, stream, ics_req)
                        .await;
                } else if let Ok(ucm_req) = decode_ue_context_modification_request(pdu_bytes) {
                    self.handle_ue_context_modification_request(client_id, stream, ucm_req)
                        .await;
                } else if let Ok(reset) = decode_ng_reset(pdu_bytes) {
                    self.handle_ng_reset(client_id, stream, reset).await;
                } else if let Ok(update) = decode_amf_configuration_update(pdu_bytes) {
                    self.handle_amf_configuration_update(client_id, stream, update)
                        .await;
                } else if let Ok(paging) = decode_paging(pdu_bytes) {
                    self.handle_paging(client_id, paging).await;
                } else if let Ok(overload) = decode_overload_start(pdu_bytes) {
                    self.handle_overload_start(client_id, overload).await;
                } else if decode_overload_stop(pdu_bytes).is_ok() {
                    self.handle_overload_stop(client_id).await;
                } else if let Ok(status) = decode_amf_status_indication(pdu_bytes) {
                    self.handle_amf_status_indication(client_id, status).await;
                } else if decode_ran_configuration_update_acknowledge(pdu_bytes).is_ok() {
                    info!("Received RAN Configuration Update Acknowledge from AMF[{client_id}]");
                } else if let Ok(fail) = decode_ran_configuration_update_failure(pdu_bytes) {
                    warn!(
                        "Received RAN Configuration Update Failure from AMF[{}]: cause={:?}, time_to_wait={:?}",
                        client_id, fail.cause, fail.time_to_wait
                    );
                } else if let Ok(err_ind) = decode_error_indication(pdu_bytes) {
                    // TS 38.413 §8.7.5: a received Error Indication is processed
                    // locally and MUST NOT be answered with another Error
                    // Indication (which would create an on-wire ping-pong).
                    self.handle_error_indication(client_id, err_ind).await;
                } else if self.handle_mbs_pdu(client_id, pdu_bytes).await {
                    // An MBS session procedure (71/72/74) was decoded and
                    // applied to `mbs_ngap_sessions`; see `handle_mbs_pdu`.
                } else {
                    // amfg-05: this operational PDU could not be routed to a
                    // handler. Per TS 38.413 §8.7.5 / §10, answer with an NGAP
                    // Error Indication rather than silently dropping it.
                    self.handle_unroutable_pdu(client_id, pdu_bytes).await;
                }
            }
            _ => {
                warn!("Received NGAP PDU in unexpected AMF state: {:?}", amf_state);
            }
        }
    }

    // ========================================================================
    // NG Reset / AMF Configuration Update (amfg-07)
    // ========================================================================

    /// Handles an inbound NG Reset (TS 38.413 §8.7.4.2).
    ///
    /// Clears the indicated UE-associated logical NG-connections — all of them
    /// for a `NG Interface` reset, or the listed subset for a
    /// `Part of NG Interface` reset — and replies with an NG Reset Acknowledge
    /// echoing the associations actually released.
    async fn handle_ng_reset(
        &mut self,
        client_id: i32,
        stream: u16,
        reset: nextgsim_ngap::procedures::ng_reset::NgResetData,
    ) {
        info!(
            "NG Reset received from AMF[{}]: scope={:?}, cause={:?}",
            client_id, reset.scope, reset.cause
        );

        // Collect the UE ids to release, recording the released associations.
        let mut ue_ids: Vec<i32> = Vec::new();
        let mut released: Vec<UeAssociation> = Vec::new();

        match &reset.scope {
            NgResetScope::All => {
                for ctx in self.ue_contexts.values() {
                    if ctx.amf_ctx_id == client_id {
                        ue_ids.push(ctx.ue_id);
                        released.push(UeAssociation {
                            amf_ue_ngap_id: ctx.amf_ue_ngap_id.map(|v| v as u64),
                            ran_ue_ngap_id: Some(ctx.ran_ue_ngap_id as u32),
                        });
                    }
                }
            }
            NgResetScope::Part(list) => {
                for assoc in list {
                    // Match by RAN UE NGAP ID first, then AMF UE NGAP ID.
                    let found = self.ue_contexts.values().find(|ctx| {
                        ctx.amf_ctx_id == client_id
                            && ((assoc.ran_ue_ngap_id.is_some()
                                && Some(ctx.ran_ue_ngap_id as u32) == assoc.ran_ue_ngap_id)
                                || (assoc.amf_ue_ngap_id.is_some()
                                    && ctx.amf_ue_ngap_id.map(|v| v as u64)
                                        == assoc.amf_ue_ngap_id))
                    });
                    if let Some(ctx) = found {
                        ue_ids.push(ctx.ue_id);
                        released.push(UeAssociation {
                            amf_ue_ngap_id: ctx.amf_ue_ngap_id.map(|v| v as u64),
                            ran_ue_ngap_id: Some(ctx.ran_ue_ngap_id as u32),
                        });
                    } else {
                        // Echo back the requested association even if unknown,
                        // so the AMF can complete its bookkeeping.
                        released.push(assoc.clone());
                    }
                }
            }
        }

        for ue_id in ue_ids {
            self.release_ue_for_reset(ue_id).await;
        }

        // Build the Acknowledge. Always include the (possibly empty) released
        // list so the AMF sees exactly which associations were cleared.
        let params = NgResetAcknowledgeParams {
            released: Some(released),
        };
        match encode_ng_reset_acknowledge(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(client_id, stream, data).await;
                info!("Sent NG Reset Acknowledge to AMF[{}]", client_id);
            }
            Err(e) => error!("Failed to encode NG Reset Acknowledge: {}", e),
        }
    }

    /// Releases a UE context as part of an NG Reset: drop the NGAP context and
    /// notify the RRC and GTP tasks (no UE Context Release Complete is sent —
    /// NG Reset is acknowledged at the interface level).
    async fn release_ue_for_reset(&mut self, ue_id: i32) {
        self.delete_ue_context(ue_id);
        self.send_an_release(ue_id).await;
        let msg = GtpMessage::UeContextRelease { ue_id };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context release to GTP: {}", e);
        }
    }

    /// Handle an inbound OVERLOAD START (TS 38.413 §8.7.7): mark the AMF as
    /// overloaded and record the traffic-load-reduction indication. Because
    /// `select_amf` only picks `Ready` AMFs, an overloaded AMF is automatically
    /// diverted from new UE routing. No reply is sent (Overload Start is a
    /// class-2 procedure), and it is no longer answered with an Error Indication.
    async fn handle_overload_start(&mut self, amf_id: i32, overload: OverloadData) {
        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
            ctx.on_overload_start();
            ctx.traffic_load_reduction = overload.traffic_load_reduction;
            info!(
                "AMF[{}] entered OVERLOAD (state={:?}, traffic_load_reduction={:?})",
                amf_id, ctx.state, ctx.traffic_load_reduction
            );
        } else {
            warn!("OVERLOAD START for unknown AMF association {}", amf_id);
        }
    }

    /// Handle an inbound OVERLOAD STOP (TS 38.413 §8.7.8): clear the overload
    /// state so the AMF is eligible for UE routing again.
    async fn handle_overload_stop(&mut self, amf_id: i32) {
        if let Some(ctx) = self.amf_contexts.get_mut(&amf_id) {
            ctx.on_overload_stop();
            ctx.traffic_load_reduction = None;
            info!("AMF[{}] cleared OVERLOAD (state={:?})", amf_id, ctx.state);
        } else {
            warn!("OVERLOAD STOP for unknown AMF association {}", amf_id);
        }
    }

    /// Handle an inbound AMF STATUS INDICATION (TS 38.413 §8.7.6): mark every AMF
    /// context that serves one of the reported unavailable GUAMIs as unavailable
    /// for UE selection, so `select_amf` reselects toward another (backup) AMF.
    /// No reply is sent, and it is no longer answered with an Error Indication.
    /// Migrating already-connected UEs to the backup AMF is out of scope here;
    /// they reselect on their next Initial UE Message.
    async fn handle_amf_status_indication(&mut self, amf_id: i32, status: AmfStatusIndicationData) {
        if status.unavailable_guami_list.is_empty() {
            warn!("AMF Status Indication from AMF[{amf_id}] with empty unavailable GUAMI list");
            return;
        }
        let mut marked = 0;
        for ctx in self.amf_contexts.values_mut() {
            let hit = status.unavailable_guami_list.iter().any(|u| {
                ctx.served_guami_list.iter().any(|s| {
                    u.guami.plmn_identity == s.guami.plmn_identity
                        && u.guami.amf_region_id == s.guami.amf_region_id
                        && u.guami.amf_set_id == s.guami.amf_set_id
                        && u.guami.amf_pointer == s.guami.amf_pointer
                })
            });
            if hit {
                ctx.unavailable = true;
                marked += 1;
            }
        }
        info!(
            "AMF Status Indication from AMF[{}]: {} unavailable GUAMI(s) -> {} AMF context(s) excluded from selection",
            amf_id,
            status.unavailable_guami_list.len(),
            marked
        );
    }

    /// Handles an inbound AMF Configuration Update (TS 38.413 §8.7.3).
    ///
    /// Updates the stored AMF configuration (name, relative capacity, served
    /// GUAMI count) and replies with an AMF Configuration Update Acknowledge.
    /// Initiate a RAN CONFIGURATION UPDATE toward the AMF (TS 38.413 §8.7.2) to
    /// refresh this gNB's advertised RAN node name, TA/slice list and paging DRX.
    /// The AMF replies with a RAN CONFIGURATION UPDATE ACKNOWLEDGE or FAILURE,
    /// both handled in `handle_ngap_pdu`.
    ///
    /// Triggered by the `ran-config-update` CLI command via
    /// `NgapMessage::SendRanConfigurationUpdate` (issue #41). Operator-driven
    /// rather than automatic, and that is not a shortcut: TS 38.413 §8.7.2 sends
    /// this when the NG-RAN node's application-level configuration CHANGES, and
    /// nothing changes a simulator's configuration except an operator. An automatic
    /// trigger would have to invent a configuration change to have something to
    /// report.
    async fn send_ran_configuration_update(&mut self, amf_id: i32) {
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let slice_support: Vec<SNssai> = config
            .nssai
            .iter()
            .map(|s| SNssai {
                sst: s.sst,
                sd: s.sd,
            })
            .collect();
        let params = RanConfigurationUpdateParams {
            ran_node_name: Some("nextgsim-gnb".to_string()),
            supported_ta_list: Some(vec![SupportedTaItem {
                tac: [
                    ((config.tac >> 16) & 0xFF) as u8,
                    ((config.tac >> 8) & 0xFF) as u8,
                    (config.tac & 0xFF) as u8,
                ],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: plmn_bytes,
                    slice_support_list: if slice_support.is_empty() {
                        vec![SNssai { sst: 1, sd: None }]
                    } else {
                        slice_support
                    },
                }],
            }]),
            default_paging_drx: Some(PagingDrx::V128),
        };
        match encode_ran_configuration_update(&params) {
            Ok(bytes) => {
                self.send_ngap_non_ue(amf_id, 0, bytes).await;
                info!("Sent RAN Configuration Update to AMF[{amf_id}]");
            }
            Err(e) => error!("Failed to encode RAN Configuration Update: {e}"),
        }
    }

    async fn handle_amf_configuration_update(
        &mut self,
        client_id: i32,
        stream: u16,
        update: nextgsim_ngap::procedures::ng_reset::AmfConfigurationUpdateData,
    ) {
        info!(
            "AMF Configuration Update from AMF[{}]: name={:?}, served_guami={}, capacity={:?}, \
             plmn_support={}",
            client_id,
            update.amf_name,
            update.served_guami_list.len(),
            update.relative_amf_capacity,
            update.plmn_support_list.len()
        );

        if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
            if let Some(ref name) = update.amf_name {
                ctx.amf_name = Some(name.clone());
            }
            if let Some(cap) = update.relative_amf_capacity {
                ctx.relative_capacity = cap;
            }
            // TS 38.413 §8.7.3: apply the updated served-GUAMI and PLMN-support
            // lists when present (an empty list means the IE was absent).
            if !update.served_guami_list.is_empty() {
                ctx.served_guami_list = update.served_guami_list.clone();
            }
            if !update.plmn_support_list.is_empty() {
                ctx.plmn_support_list = update.plmn_support_list.clone();
            }
        }

        // TS 38.413 §8.7.3: act on the TNL-association lists, and report the
        // outcome. Before this they were not even parsed, so an AMF that asked the
        // gNB to add a second TNLA got a bare acknowledge -- which reads as "done"
        // -- and then balanced traffic onto an association that did not exist
        // (issue #41).
        let (tnla_setup, tnla_failed_to_setup) = self.apply_tnla_changes(client_id, &update).await;

        let params = AmfConfigurationUpdateAcknowledgeParams {
            tnla_setup,
            tnla_failed_to_setup,
        };
        match encode_amf_configuration_update_acknowledge(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(client_id, stream, data).await;
                info!(
                    "Sent AMF Configuration Update Acknowledge to AMF[{}]",
                    client_id
                );
            }
            Err(e) => error!(
                "Failed to encode AMF Configuration Update Acknowledge: {}",
                e
            ),
        }
    }

    /// Allocates the next SCTP client ID for a dynamically added TNL association.
    fn next_tnla_client_id(&mut self) -> i32 {
        let id = self.next_dynamic_tnla_id;
        self.next_dynamic_tnla_id += 1;
        id
    }

    /// Applies an AMF Configuration Update's TNL-association lists
    /// (TS 38.413 §8.7.3, §9.2.6.5, issue #41).
    ///
    /// Returns `(setup, failed_to_setup)` for the Acknowledge.
    ///
    /// # What "add" and "remove" mean here
    ///
    /// The gNB's TNLAs are SCTP associations it initiated, so adding one is a
    /// `SctpMessage::ConnectionRequest` to the address the AMF named and removing
    /// one is a `ConnectionClose`. Both are real actions, not bookkeeping.
    ///
    /// # Why an add is reported as SET UP before the association is confirmed
    ///
    /// `ConnectionRequest` is asynchronous: the SCTP task answers later with
    /// `AssociationSetup` or nothing. The Acknowledge cannot wait for that without
    /// holding the procedure open past its point. So an accepted request is
    /// reported in the setup list and a *rejected* one — an address this gNB cannot
    /// use, or a queue it could not reach — in the failed list. That is a real
    /// distinction rather than optimism: the failure cases below are the ones the
    /// gNB knows about at Acknowledge time, and they are the ones an AMF can act
    /// on. An association that is accepted here and then fails to come up shows up
    /// as an association that never reaches `Ready`, which `select_amf` already
    /// excludes.
    async fn apply_tnla_changes(
        &mut self,
        client_id: i32,
        update: &nextgsim_ngap::procedures::ng_reset::AmfConfigurationUpdateData,
    ) -> (
        Vec<nextgsim_ngap::procedures::ng_reset::AmfTnlAssociationAddress>,
        Vec<nextgsim_ngap::procedures::ng_reset::AmfTnlAssociationAddress>,
    ) {
        let mut setup = Vec::new();
        let mut failed = Vec::new();

        // REMOVE first. An update that removes an address and adds it back with a
        // different usage would otherwise close the association it had just asked
        // for.
        for address in &update.tnla_to_remove {
            // `tnla_client_ids` is the SINGLE source of truth for "did this gNB open
            // this association on request". A second check against the AMF context's
            // own record was redundant with it -- a revert round proved so by making
            // that check always say yes and changing nothing -- so the context record
            // is updated as bookkeeping and the decision rests here.
            //
            // The association carrying this very message is NOT in this map (it came
            // from `amf_configs`), which is what stops a remove request tearing down
            // the procedure mid-flight.
            match self.tnla_client_ids.remove(&address.endpoint) {
                Some(id) => {
                    info!(
                        "AMF[{client_id}] asked to remove TNL association {}; closing \
                         it (AMF[{id}])",
                        address.endpoint
                    );
                    if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
                        ctx.on_tnla_removed(&address.endpoint);
                    }
                    let msg = SctpMessage::ConnectionClose { client_id: id };
                    if let Err(e) = self.task_base.sctp_tx.send(msg).await {
                        error!("Failed to close TNL association {}: {e}", address.endpoint);
                    }
                    self.amf_contexts.remove(&id);
                }
                None => warn!(
                    "AMF[{client_id}] asked to remove TNL association {}, which this \
                     gNB never opened on request; ignoring rather than closing \
                     something else",
                    address.endpoint
                ),
            }
        }

        for item in &update.tnla_to_add {
            let endpoint = item.address.endpoint;
            // Adding the association that carries this message is a no-op the AMF
            // may legitimately send (it is describing its own pool); reported as
            // set up, since it demonstrably is.
            // Already opened on a previous request: report it as set up (it
            // demonstrably is) rather than opening a second association to the same
            // endpoint.
            if self.tnla_client_ids.contains_key(&endpoint) {
                setup.push(item.address);
                continue;
            }

            let new_id = self.next_tnla_client_id();
            let amf_port = self
                .task_base
                .config
                .amf_configs
                .first()
                .map_or(DEFAULT_NGAP_PORT, |a| a.port);
            let msg = SctpMessage::ConnectionRequest {
                client_id: new_id,
                local_address: self.task_base.config.ngap_ip.to_string(),
                local_port: 0,
                remote_address: endpoint.to_string(),
                remote_port: amf_port,
                ppid: NGAP_PPID,
            };
            match self.task_base.sctp_tx.send(msg).await {
                Ok(()) => {
                    info!(
                        "AMF[{client_id}] asked to add TNL association {endpoint} \
                         (usage={:?}, weight={}); connecting as AMF[{new_id}]",
                        item.usage, item.weight_factor
                    );
                    self.create_amf_context(new_id);
                    self.tnla_client_ids.insert(endpoint, new_id);
                    if let Some(ctx) = self.amf_contexts.get_mut(&client_id) {
                        ctx.on_tnla_established(endpoint);
                    }
                    setup.push(item.address);
                }
                Err(e) => {
                    warn!("Cannot add TNL association {endpoint}: {e}");
                    failed.push(item.address);
                }
            }
        }

        // UPDATE carries only usage and weight, neither of which this gNB acts on:
        // it has no per-association traffic split to weight and no usage-based
        // routing. Reported at INFO and NOT acknowledged as a setup, because
        // acknowledging an update it did not apply is the misreport this issue is
        // about.
        for item in &update.tnla_to_update {
            info!(
                "AMF[{client_id}] updated TNL association {} (usage={:?}, weight={:?}); \
                 recorded, not acted on: this gNB has no per-association traffic \
                 split to weight",
                item.address.endpoint, item.usage, item.weight_factor
            );
        }

        (setup, failed)
    }

    // ========================================================================
    // Paging (amfg-06)
    // ========================================================================

    /// Handles an inbound PAGING from the AMF (TS 38.413 §8.6.1 / §9.2.3.1).
    ///
    /// Matches the TAIListForPaging against this gNB's served TAI and, on a
    /// match, triggers RRC Paging toward the air interface carrying the
    /// 5G-S-TMSI. PagingPriority / PagingDRX are honored on a best-effort
    /// (logged) basis.
    async fn handle_paging(&mut self, client_id: i32, paging: PagingData) {
        let config = &self.task_base.config;
        let served_plmn = config.plmn.encode();
        let served_tac = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        // Collect the TAIs from the paging list that this gNB serves.
        let matching: Vec<&nextgsim_ngap::procedures::initial_ue_message::Tai> = paging
            .tai_list_for_paging
            .iter()
            .filter(|tai| tai.plmn_identity == served_plmn && tai.tac == served_tac)
            .collect();

        if matching.is_empty() {
            debug!(
                "PAGING from AMF[{}] not for a served TAI (served plmn={:02x?} tac={:02x?}); ignoring",
                client_id, served_plmn, served_tac
            );
            return;
        }

        if paging.paging_priority.is_some() || paging.paging_drx.is_some() {
            debug!(
                "PAGING QoS hints: priority={:?}, drx={:?}",
                paging.paging_priority, paging.paging_drx
            );
        }

        // Serialize the 5G-S-TMSI for the RRC Paging record.
        let ue_paging_tmsi = serialize_five_g_s_tmsi(&paging.ue_paging_identity);

        // Serialize the matching TAIs (plmn(3) + tac(3) per entry).
        let mut tai_list_for_paging = Vec::with_capacity(matching.len() * 6);
        for tai in &matching {
            tai_list_for_paging.extend_from_slice(&tai.plmn_identity);
            tai_list_for_paging.extend_from_slice(&tai.tac);
        }

        info!(
            "PAGING from AMF[{}]: matched {} served TAI(s); triggering RRC Paging",
            client_id,
            matching.len()
        );

        let msg = RrcMessage::Paging {
            ue_paging_tmsi,
            tai_list_for_paging,
            // The (default)PagingDRX the AMF signalled, converted to radio frames
            // -- the `T` of TS 38.304 §7.1. Before issue #99 this IE was decoded,
            // logged and dropped, so the paging occasion could not follow what the
            // network asked for.
            drx_cycle_frames: paging.paging_drx.map(|drx| drx.radio_frames()),
        };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to forward Paging to RRC: {}", e);
        }
    }

    // ========================================================================
    // Handover Procedures
    // ========================================================================

    /// Handles Handover Command from AMF (source gNB side)
    /// AMF sends this after receiving `HandoverRequestAcknowledge` from target gNB.
    /// Source gNB must forward the transparent container to UE via RRC and
    /// release resources after UE completes handover.
    async fn handle_handover_command(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ho_cmd: HandoverCommandData,
    ) {
        info!(
            "Handover Command received: amf_ue_ngap_id={}, ran_ue_ngap_id={}, type={:?}",
            ho_cmd.amf_ue_ngap_id, ho_cmd.ran_ue_ngap_id, ho_cmd.handover_type
        );

        // Find the UE by RAN UE NGAP ID
        let ue_id = self
            .find_ue_by_ran_id(ho_cmd.ran_ue_ngap_id as i64)
            .map(|ctx| ctx.ue_id);

        if let Some(ue_id) = ue_id {
            // Preparation completed, so TNGRELOCprep is done (TS 38.413 §8.4.1.2).
            if !self
                .guard_timers
                .cancel(GuardTimer::HandoverPreparation { ue_id })
            {
                warn!(
                    "Handover Command for UE[{ue_id}] with no handover preparation \
                     pending: either it already timed out or this gNB never started one"
                );
            }
            // Forward the Target-to-Source Transparent Container to UE via RRC
            // This contains the RRC Reconfiguration with mobility control info
            let container = OctetString::from_slice(&ho_cmd.target_to_source_transparent_container);
            let msg = RrcMessage::NasDelivery {
                ue_id,
                pdu: container,
            };
            if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                error!("Failed to forward handover command to RRC: {}", e);
            }

            // Release PDU sessions that failed handover
            if let Some(ref release_list) = ho_cmd.pdu_session_resource_to_release_list {
                for item in release_list {
                    let msg = GtpMessage::SessionRelease {
                        ue_id,
                        psi: item.pdu_session_id as i32,
                    };
                    if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                        error!(
                            "Failed to release PDU session {}: {}",
                            item.pdu_session_id, e
                        );
                    }
                }
            }

            info!(
                "Handover Command processed for UE[{}], forwarded to RRC",
                ue_id
            );
        } else {
            warn!(
                "Handover Command for unknown RAN UE NGAP ID: {}",
                ho_cmd.ran_ue_ngap_id
            );
        }
    }

    /// Handles Handover Request from AMF (target gNB side)
    /// AMF sends this to the target gNB to prepare handover resources.
    /// Target gNB must allocate resources and respond with `HandoverRequestAcknowledge`.
    async fn handle_handover_request(
        &mut self,
        client_id: i32,
        stream: u16,
        ho_req: HandoverRequestData,
    ) {
        info!(
            "Handover Request received (target gNB): amf_ue_ngap_id={}, type={:?}, {} requested session(s)",
            ho_req.amf_ue_ngap_id,
            ho_req.handover_type,
            ho_req.pdu_sessions.len()
        );

        // Allocate a new UE context for the incoming handover
        let Some(ran_ue_ngap_id) = self.create_ue_context(
            self.ue_contexts.len() as i32 + 1000, // Handover UE IDs start from 1000
            client_id,
        ) else {
            error!(
                "Failed to create UE context for handover, amf_ue_ngap_id={}",
                ho_req.amf_ue_ngap_id
            );
            return;
        };
        let Some(ue_id) = self.find_ue_by_ran_id(ran_ue_ngap_id).map(|ctx| ctx.ue_id) else {
            error!("Handover UE context vanished immediately after creation");
            return;
        };
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.amf_ue_ngap_id = Some(ho_req.amf_ue_ngap_id as i64);
        }

        // TS 33.501 §6.9.2.3.1: derive this target's KgNB* BEFORE admitting any
        // session, because the admitted DRBs are keyed from it. Doing it after would
        // hand `install_drb_security` the source's keys, and every PDCP MAC on this
        // cell would fail with nothing to say why.
        self.adopt_handover_security_context(ue_id, ho_req.security_context);

        // Admit the requested sessions -- each one allocating a real DL F-TEID, a GTP-U
        // tunnel and a DRB through the SAME path a PDU Session Resource Setup uses
        // (TS 38.413 §8.4.2.2: "shall attempt to execute the requested PDU session
        // configuration and associated security"). Before this the gNB admitted a
        // hardcoded session 1 with a `vec![0x00]` transfer and allocated nothing.
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);
        let mut admitted_list = Vec::new();
        let mut failed_list = Vec::new();
        for session in &ho_req.pdu_sessions {
            match self
                .admit_one_handover_session(ue_id, session, gnb_ip)
                .await
            {
                Ok(item) => admitted_list.push(item),
                Err(item) => failed_list.push(item),
            }
        }
        if admitted_list.is_empty() && !ho_req.pdu_sessions.is_empty() {
            warn!(
                "Handover for amf_ue_ngap_id={}: no requested session could be admitted",
                ho_req.amf_ue_ngap_id
            );
        }

        let target_to_source_container = ho_req.source_to_target_transparent_container.clone();

        match encode_handover_request_acknowledge(&HandoverRequestAcknowledgeParams {
            amf_ue_ngap_id: ho_req.amf_ue_ngap_id,
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            pdu_session_resource_admitted_list: admitted_list,
            pdu_session_resource_failed_list: if failed_list.is_empty() {
                None
            } else {
                Some(failed_list)
            },
            target_to_source_transparent_container: target_to_source_container,
        }) {
            Ok(data) => {
                let delivered = self.send_ngap_ue_associated(client_id, stream, data).await;
                if !delivered {
                    // #169: the acknowledge ENCODED but never reached the SCTP task, so the
                    // AMF has not been told this target admitted anything. Arming here would
                    // have the target report a HANDOVER NOTIFY for a handover the AMF has no
                    // record of -- exactly the hazard the arming was placed after the
                    // acknowledge to avoid. The guard used to cover only the encode, and
                    // `send_ngap_ue_associated` swallowed its own send error, so this branch
                    // was indistinguishable from success.
                    error!(
                        "Handover Request Acknowledge for UE[{ue_id}] could not be delivered \
                         to the AMF: NOT arming arrival detection (TS 38.413 §8.4.3)"
                    );
                    return;
                }
                info!(
                    "Sent Handover Request Acknowledge: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                    ho_req.amf_ue_ngap_id, ran_ue_ngap_id
                );
                // Arm the arrival detection (TS 38.413 §8.4.3, issue #156). Only once the
                // acknowledge is encoded AND away: an admission the AMF was never told
                // about must not have the target report an arrival for it.
                if let Err(e) = self
                    .task_base
                    .rrc_tx
                    .send(RrcMessage::ExpectHandoverArrival { ue_id })
                    .await
                {
                    error!("Could not arm the handover-arrival detection for UE[{ue_id}]: {e}");
                }
            }
            Err(e) => {
                error!("Failed to encode Handover Request Acknowledge: {}", e);
            }
        }
    }

    /// Admit one PDU session on a handover-in, allocating the same resources a PDU
    /// Session Resource Setup would (TS 38.413 §8.4.2.2, issue #39).
    ///
    /// The `handoverRequestTransfer` **contains** a
    /// `PDUSessionResourceSetupRequestTransfer`, so this decodes it with the same
    /// function and resolves the same user-plane security policy — including the
    /// `SecurityIndication` (issue #32). A handover-in that took a different path would
    /// be a second place for the QoS, tunnel and security decisions to be made
    /// differently.
    async fn admit_one_handover_session(
        &mut self,
        ue_id: i32,
        session: &HandoverRequestSetupItem,
        gnb_ip: std::net::IpAddr,
    ) -> Result<PduSessionResourceAdmittedItem, PduSessionResourceFailedToSetupHoAckItem> {
        let psi = session.pdu_session_id;
        let failed =
            |cause: NgSetupFailureCause| PduSessionResourceFailedToSetupHoAckItem {
                pdu_session_id: psi,
                handover_resource_allocation_unsuccessful_transfer:
                    encode_setup_unsuccessful_transfer(&cause).unwrap_or_default(),
            };

        let request = match decode_setup_request_transfer(&session.transfer) {
            Ok(req) => req,
            Err(e) => {
                warn!("Handover session {psi}: undecodable handoverRequestTransfer: {e}");
                return Err(failed(NgSetupFailureCause::Protocol(
                    ProtocolCause::TransferSyntaxError,
                )));
            }
        };

        let (policy, decision) =
            match self.resolve_up_security(ue_id, psi, request.security_indication) {
                Ok(resolved) => resolved,
                Err(refusal) => {
                    warn!("Handover session {psi} refused: {refusal}");
                    return Err(failed(NgSetupFailureCause::RadioNetwork(match refusal {
                        UpSecurityRefusal::IntegrityNotPossible => {
                            RadioNetworkCause::UpIntegrityProtectionNotPossible
                        }
                        UpSecurityRefusal::ConfidentialityNotPossible => {
                            RadioNetworkCause::UpConfidentialityProtectionNotPossible
                        }
                    })));
                }
            };

        let upf_teid = request.ul_tunnel.teid;
        let upf_addr = request.ul_tunnel.address;
        let qfi = request.qos_flows[0].qfi;
        // See the same binding in `setup_one_pdu_session`: hoisted above the
        // `SessionCreate` so the GTP task's SDAP entity and the RRCReconfiguration
        // below describe one mapping. A handover-in that got this wrong would hand the
        // UE a bearer set the target gNB does not route to.
        let accepted_flows: Vec<(u8, Option<u16>)> = request
            .qos_flows
            .iter()
            .map(|f| (f.qfi, f.five_qi))
            .collect();
        let gnb_teid = self.next_downlink_teid();

        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.add_pdu_session(NgapPduSession {
                psi,
                qfi: Some(qfi),
                uplink_teid: gnb_teid,
                downlink_teid: upf_teid,
                upf_address: upf_addr,
                up_security_policy: policy,
                up_security: decision,
            });
        }

        let resource = PduSessionResource {
            psi: psi as i32,
            qfi: Some(qfi),
            qos_flows: accepted_flows.clone(),
            uplink_teid: upf_teid,
            downlink_teid: gnb_teid,
            upf_address: upf_addr,
        };
        if let Err(e) = self
            .task_base
            .gtp_tx
            .send(GtpMessage::SessionCreate { ue_id, resource })
            .await
        {
            error!("Handover session {psi}: failed to create the GTP session: {e}");
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(psi);
            }
            return Err(failed(NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::RadioResourcesNotAvailable,
            )));
        }

        let accepted_qfis: Vec<u8> = request.qos_flows.iter().map(|f| f.qfi).collect();
        self.key_and_establish_drb(ue_id, psi, &accepted_flows, decision)
            .await;

        let transfer = match encode_handover_ack_transfer(&HandoverAckTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: gnb_ip,
                teid: gnb_teid,
            },
            admitted_qfis: accepted_qfis.clone(),
            failed_qos_flows: vec![],
            // What this target ACTUALLY applied, not what was asked for -- the AMF has
            // to learn whether the new serving node honoured the policy (issue #32).
            security_result: decision.any().then_some(UpSecurityResult {
                integrity_performed: decision.integrity,
                confidentiality_performed: decision.ciphering,
            }),
        }) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!("Handover session {psi}: failed to encode the ack transfer: {e}");
                return Err(failed(NgSetupFailureCause::Misc(
                    nextgsim_ngap::procedures::ng_setup::MiscCause::Unspecified,
                )));
            }
        };
        info!(
            "Handover session {psi} admitted: gNB TEID=0x{gnb_teid:08x}, QFIs={accepted_qfis:?}, \
             integrity={} ciphering={}",
            decision.integrity, decision.ciphering
        );
        Ok(PduSessionResourceAdmittedItem {
            pdu_session_id: psi,
            handover_request_ack_transfer: transfer,
        })
    }

    /// Adopt the AMF-supplied `{NH, NCC}` and derive this target's `KgNB*`
    /// (TS 33.501 §6.9.2.3.1, Annex A.11; issue #39).
    ///
    /// Vertical when the AMF sent a context, horizontal from the UE's current `KgNB`
    /// when it did not — and the log says which, because only the vertical case gives
    /// forward security against a compromised source gNB, and a silent fallback would
    /// lose that with nothing to show.
    ///
    /// With neither a context nor an existing `KgNB` there is nothing to chain from, and
    /// this says so rather than deriving from zeros: a `KgNB*` built on an invented key
    /// would make every DRB on this cell fail its MAC with no indication that a key was
    /// the problem.
    fn adopt_handover_security_context(
        &mut self,
        ue_id: i32,
        context: Option<HandoverSecurityContext>,
    ) {
        use nextgsim_crypto::kdf::{derive_kgnb_star, KgnbStarChaining};

        let target_pci = phys_cell_id_from_nci(self.task_base.config.nci);
        let target_arfcn = self.task_base.config.dl_arfcn;
        let existing = self
            .ue_contexts
            .values()
            .find(|c| c.ue_id == ue_id)
            .and_then(|c| c.as_security.as_ref())
            .map(|s| s.kgnb);

        let (chaining, source_key) = match (context, existing) {
            (Some(ctx), _) => {
                info!("Handover for UE {ue_id}: adopting {ctx}");
                (KgnbStarChaining::Vertical, ctx.next_hop_nh)
            }
            (None, Some(kgnb)) => (KgnbStarChaining::Horizontal, kgnb),
            (None, None) => {
                warn!(
                    "Handover for UE {ue_id}: the AMF sent no SecurityContext and this \
                     target holds no KgNB, so no KgNB* can be derived. The admitted DRBs \
                     will be unprotected."
                );
                return;
            }
        };

        let kgnb_star = derive_kgnb_star(&source_key, target_pci, target_arfcn);
        info!(
            "Handover for UE {ue_id}: derived KgNB* {chaining:?} for PCI {target_pci}, \
             ARFCN-DL {target_arfcn}"
        );
        if chaining == KgnbStarChaining::Horizontal {
            warn!(
                "Handover for UE {ue_id}: KgNB* derived HORIZONTALLY -- the source gNB \
                 can compute this key, so forward security is not provided \
                 (TS 33.501 §6.9.2.3.1)"
            );
        }

        // Re-derive the AS keys from KgNB* and install them, which is what makes the
        // adoption real rather than a log line: the RRC and UP keys the target protects
        // with all hang off this KgNB.
        let ((ciph_id, _), (int_id, _)) = match self
            .ue_contexts
            .values()
            .find(|c| c.ue_id == ue_id)
            .and_then(|c| c.as_security.as_ref())
        {
            // Keep the algorithms the UE already negotiated: a handover re-keys, it does
            // not renegotiate (TS 33.501 §6.9.2.3.1).
            Some(sec) => ((sec.ciphering_alg_id, ()), (sec.integrity_alg_id, ())),
            // A fresh handover-in with no prior context: the algorithms come with the
            // UE Security Capabilities in a later Initial Context Setup, so NEA0/NIA0
            // until then rather than a guess.
            None => ((0u8, ()), (0u8, ())),
        };
        let derived = AsSecurityContext::from_kgnb(kgnb_star, ciph_id, int_id);
        if let Some(ctx) = self.ue_contexts.values_mut().find(|c| c.ue_id == ue_id) {
            ctx.as_security = Some(derived);
        }
    }

    /// Handles Handover Preparation Failure from AMF (source gNB side)
    /// AMF sends this when the target gNB rejected the handover.
    async fn handle_handover_preparation_failure(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ho_fail: HandoverPreparationFailureData,
    ) {
        warn!(
            "Handover Preparation Failure: amf_ue_ngap_id={}, ran_ue_ngap_id={}, cause={:?}",
            ho_fail.amf_ue_ngap_id, ho_fail.ran_ue_ngap_id, ho_fail.cause
        );

        // Handover failed - UE stays on source gNB, no action needed beyond stopping
        // the guard timer: the AMF answered, so TNGRELOCprep must not also fire and
        // cancel a preparation that is already over.
        if let Some(ue_id) = self
            .find_ue_by_ran_id(ho_fail.ran_ue_ngap_id as i64)
            .map(|ctx| ctx.ue_id)
        {
            info!("Handover preparation failed for UE[{ue_id}], staying on source cell");
            self.guard_timers
                .cancel(GuardTimer::HandoverPreparation { ue_id });
        }
    }

    /// Initiates a handover by sending HANDOVER REQUIRED to the AMF (source gNB side,
    /// TS 38.413 §8.4.1.1).
    ///
    /// Reachable since issue #39: `NgapMessage::InitiateHandover` is what drives it, from
    /// the RRC plane's measurement-report and NWDAF paths. Before this it carried
    /// `#[allow(dead_code)]` and no caller.
    ///
    /// Returns `false` when nothing was sent, so a caller can report a handover that did
    /// not start rather than logging one that did.
    async fn initiate_handover(&mut self, request: HandoverInitiation) -> bool {
        let ue_id = request.ue_id;
        let Some(ctx) = self.ue_contexts.values().find(|c| c.ue_id == ue_id) else {
            warn!("Cannot initiate handover for unknown UE[{ue_id}]");
            return false;
        };
        let Some(amf_ue_ngap_id) = ctx.amf_ue_ngap_id.map(|id| id as u64) else {
            warn!("UE[{ue_id}] has no AMF UE NGAP ID, cannot hand over");
            return false;
        };
        let (amf_ctx_id, ran_ue_ngap_id) = (ctx.amf_ctx_id, ctx.ran_ue_ngap_id as u32);

        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let target_tac_bytes = [
            ((request.target_tac >> 16) & 0xFF) as u8,
            ((request.target_tac >> 8) & 0xFF) as u8,
            (request.target_tac & 0xFF) as u8,
        ];

        // A real source-to-target transparent container (TS 38.413 §9.3.1.20): the UE's
        // own capabilities in an RRC `HandoverPreparationInformation`, the cell the
        // source means as the target, and where the UE has been. The `vec![0x00]` this
        // replaces told the target nothing at all.
        let rrc_container =
            match encode_handover_preparation_information(&HandoverPreparationParams {
                nr_capability: request.ue_nr_capability.clone(),
            }) {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("UE[{ue_id}]: cannot build the HandoverPreparationInformation: {e}");
                    return false;
                }
            };
        let source_container =
            match encode_source_to_target_container(&SourceToTargetContainerParams {
                rrc_container,
                target_cell: HandoverNrCgiValue {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: request.target_cell_identity,
                },
                source_cell: HandoverNrCgiValue {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci & 0xF_FFFF_FFFF,
                },
                time_in_source_cell_s: request.time_in_source_cell_s,
            }) {
                Ok(bytes) => bytes,
                Err(e) => {
                    error!("UE[{ue_id}]: cannot build the source-to-target container: {e}");
                    return false;
                }
            };

        // Every session the UE actually has. An empty list means the UE has no user
        // plane to move, which is a real state -- the placeholder "default entry" this
        // replaces asked the target to admit a session that did not exist.
        let pdu_sessions: Vec<PduSessionResourceHoRequiredItem> = self
            .ue_contexts
            .values()
            .find(|c| c.ue_id == ue_id)
            .map(|ctx| {
                ctx.pdu_sessions
                    .values()
                    .map(|sess| PduSessionResourceHoRequiredItem {
                        pdu_session_id: sess.psi,
                        // TS 38.413 §9.3.4.12: the transfer's only member is
                        // `directForwardingPathAvailability`, and this gNB forwards
                        // nothing during a handover -- so an all-absent transfer is the
                        // truthful encoding, not a placeholder.
                        handover_required_transfer: encode_handover_required_transfer(false)
                            .unwrap_or_default(),
                    })
                    .collect()
            })
            .unwrap_or_default();
        if pdu_sessions.is_empty() {
            // `PDUSessionResourceListHORqd` is `SIZE(1..maxnoofPDUSessions)`, so an empty
            // list is not encodable at all -- and refusing is the honest answer anyway: a
            // UE with no PDU session has no user plane to move, and the old code's "add at
            // least a default entry" asked the target to admit a session that did not
            // exist.
            //
            // Found by a test: the encoder failed silently and `initiate_handover` still
            // reported success, which is precisely the no-op-that-logs-success this issue
            // is about.
            warn!(
                "UE[{ue_id}] has no PDU session, so there is nothing to hand over; \
                 refusing rather than sending an unencodable HANDOVER REQUIRED"
            );
            return false;
        }

        info!(
            "Initiating handover for UE[{ue_id}] to gNB {} cell {:#x} ({} session(s))",
            request.target_gnb_id,
            request.target_cell_identity,
            pdu_sessions.len()
        );
        self.send_handover_required(
            amf_ctx_id,
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            &plmn_bytes,
            request.target_gnb_id,
            &target_tac_bytes,
            &source_container,
            &pdu_sessions,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    /// Returns whether the message actually reached the wire, so a caller cannot report
    /// a handover that never started (issue #39).
    async fn send_handover_required(
        &mut self,
        amf_client_id: i32,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
        plmn_bytes: &[u8; 3],
        target_gnb_id: u32,
        target_tac_bytes: &[u8; 3],
        source_container: &[u8],
        pdu_sessions: &[PduSessionResourceHoRequiredItem],
    ) -> bool {
        let params = HandoverRequiredParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            handover_type: HandoverTypeValue::Intra5gs,
            cause: HandoverCause::RadioNetwork(
                nextgsim_ngap::procedures::ng_setup::RadioNetworkCause::HandoverDesirableForRadioReason,
            ),
            target_id: TargetIdValue::TargetRanNodeId {
                global_ran_node_id: target_gnb_id.to_be_bytes().to_vec(),
                selected_tai: TaiValue {
                    plmn_identity: *plmn_bytes,
                    tac: *target_tac_bytes,
                },
            },
            direct_forwarding_path_availability: None,
            pdu_session_resource_list: pdu_sessions.to_vec(),
            source_to_target_transparent_container: source_container.to_vec(),
        };

        let sent = match encode_handover_required(&params) {
            Ok(data) => {
                self.send_ngap_ue_associated(amf_client_id, 1, data).await;
                info!(
                    "Sent Handover Required: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                    amf_ue_ngap_id, ran_ue_ngap_id
                );
                // TNGRELOCprep starts when preparation is actually on the wire, not
                // when it was decided: an encode failure leaves nothing to supervise.
                if let Some(ue_id) = self
                    .find_ue_by_ran_id(ran_ue_ngap_id as i64)
                    .map(|ctx| ctx.ue_id)
                {
                    let prep = Duration::from_secs(self.task_base.config.ngap_tngreloc_prep_secs);
                    self.guard_timers.start(
                        Instant::now(),
                        prep,
                        GuardTimer::HandoverPreparation { ue_id },
                    );
                }
                true
            }
            Err(e) => {
                error!("Failed to encode Handover Required: {}", e);
                false
            }
        };
        sent
    }

    /// Sends Handover Notify to AMF (target gNB side)
    /// Called after UE has completed handover to target cell
    async fn send_handover_notify(
        &self,
        amf_client_id: i32,
        amf_ue_ngap_id: u64,
        ran_ue_ngap_id: u32,
    ) {
        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = HandoverNotifyParams {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
            user_location_info: HandoverUserLocationInfoNr {
                nr_cgi: HandoverNrCgiValue {
                    plmn_identity: plmn_bytes,
                    nr_cell_identity: config.nci,
                },
                tai: TaiValue {
                    plmn_identity: plmn_bytes,
                    tac: tac_bytes,
                },
            },
        };

        match encode_handover_notify(&params) {
            Ok(data) => {
                self.send_ngap_ue_associated(amf_client_id, 1, data).await;
                info!(
                    "Sent Handover Notify: amf_ue_ngap_id={}, ran_ue_ngap_id={}",
                    amf_ue_ngap_id, ran_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Handover Notify: {}", e);
            }
        }
    }

    // ========================================================================
    // Path Switch Procedure (TS 38.413 Section 8.4.4)
    // ========================================================================

    /// Sends a Path Switch Request to the AMF (target gNB side, Xn handover).
    ///
    /// Requests the 5GC to switch the DL GTP-U termination point to this gNB
    /// for all PDU sessions of the given UE. The per-session
    /// `PathSwitchRequestTransfer` carries the real DL F-TEID allocated here.
    async fn send_path_switch_request(&mut self, ue_id: i32, source_amf_ue_ngap_id: u64) {
        let gnb_ip = self
            .task_base
            .config
            .gtp_advertise_ip
            .unwrap_or(self.task_base.config.gtp_ip);

        let (amf_ctx_id, ran_ue_ngap_id, stream, sessions) = {
            let ctx = match self.ue_contexts.get(&ue_id) {
                Some(c) => c,
                None => {
                    warn!("Cannot send Path Switch Request for unknown UE[{}]", ue_id);
                    return;
                }
            };
            let sessions: Vec<PathSwitchSessionItem> = ctx
                .pdu_sessions
                .values()
                .map(|sess| PathSwitchSessionItem {
                    pdu_session_id: sess.psi,
                    dl_tunnel: GtpTunnelInfo {
                        address: gnb_ip,
                        teid: sess.uplink_teid,
                    },
                    accepted_qfis: vec![sess.qfi.unwrap_or(1)],
                })
                .collect();
            (ctx.amf_ctx_id, ctx.ran_ue_ngap_id, ctx.stream_id, sessions)
        };

        if sessions.is_empty() {
            warn!(
                "Cannot send Path Switch Request for UE[{}]: no PDU sessions",
                ue_id
            );
            return;
        }

        let config = &self.task_base.config;
        let plmn_bytes = config.plmn.encode();
        let tac_bytes = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let params = PathSwitchRequestParams {
            ran_ue_ngap_id: ran_ue_ngap_id as u32,
            source_amf_ue_ngap_id,
            nr_cgi: NrCgi {
                plmn_identity: plmn_bytes,
                nr_cell_identity: config.nci,
            },
            tai: Tai {
                plmn_identity: plmn_bytes,
                tac: tac_bytes,
            },
            ue_security_capabilities: UeSecurityCapabilityBits::default(),
            sessions,
        };

        match encode_path_switch_request(&params) {
            Ok(bytes) => {
                self.send_ngap_ue_associated(amf_ctx_id, stream, bytes)
                    .await;
                info!(
                    "Sent Path Switch Request: ue_id={}, ran_ue_ngap_id={}, source_amf_ue_ngap_id={}",
                    ue_id, ran_ue_ngap_id, source_amf_ue_ngap_id
                );
            }
            Err(e) => {
                error!("Failed to encode Path Switch Request: {}", e);
            }
        }
    }

    /// Handles Path Switch Request Acknowledge from AMF.
    ///
    /// Switches the UL GTP-U tunnels to the (possibly new) UPF endpoints, re-resolves
    /// each session's user-plane security policy (issue #32), and **adopts the fresh
    /// `{NH, NCC}`** by deriving this node's `KgNB*` from it (TS 33.501 §6.9.2.3.1,
    /// issue #39).
    ///
    /// That last clause used to be in this comment and not in the code — the body
    /// performed no key adoption at all. It does now, through the same
    /// `adopt_handover_security_context` the HANDOVER REQUEST path uses, so a path switch
    /// and an N2 handover cannot chain differently.
    async fn handle_path_switch_request_acknowledge(
        &mut self,
        _client_id: i32,
        _stream: u16,
        ack: PathSwitchRequestAcknowledgeData,
    ) {
        info!(
            "Path Switch Request Acknowledge: amf_ue_ngap_id={}, ran_ue_ngap_id={}, ncc={}, {} switched sessions",
            ack.amf_ue_ngap_id,
            ack.ran_ue_ngap_id,
            ack.next_hop_chaining_count,
            ack.switched_sessions.len()
        );

        let ue_id = match self.find_ue_by_ran_id(ack.ran_ue_ngap_id as i64) {
            Some(ctx) => ctx.ue_id,
            None => {
                warn!(
                    "Path Switch Request Acknowledge for unknown RAN-UE-NGAP-ID {}",
                    ack.ran_ue_ngap_id
                );
                return;
            }
        };

        // Adopt the AMF-assigned ID (it can change across the path switch)
        if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
            ctx.amf_ue_ngap_id = Some(ack.amf_ue_ngap_id as i64);
        }

        // Adopt the fresh {NH, NCC} the AMF sent and re-key from it (TS 33.501
        // §6.9.2.3.1, issue #39). The same function the HANDOVER REQUEST path uses, so a
        // path switch and an N2 handover cannot chain differently -- and this is what the
        // doc comment above used to claim while the body did nothing.
        self.adopt_handover_security_context(
            ue_id,
            Some(HandoverSecurityContext {
                next_hop_chaining_count: ack.next_hop_chaining_count,
                next_hop_nh: ack.next_hop_nh,
            }),
        );

        // Update UL tunnels for switched sessions and notify the GTP task
        for session in &ack.switched_sessions {
            // TS 38.413 §9.3.4.10: the Acknowledge transfer restates the SMF's
            // user-plane security policy to the *new* serving node, which is this gNB
            // after an Xn handover. Re-resolved against this gNB's own negotiated
            // algorithms, because the source node's decision was made with keys and a
            // build this one does not necessarily share (issue #32).
            if let Some(policy) = session.security_indication {
                match self.resolve_up_security(ue_id, session.pdu_session_id, Some(policy)) {
                    Ok((policy, decision)) => {
                        if let Some(s) = self
                            .ue_contexts
                            .get_mut(&ue_id)
                            .and_then(|ctx| ctx.pdu_sessions.get_mut(&session.pdu_session_id))
                        {
                            s.up_security_policy = policy;
                            s.up_security = decision;
                        }
                        self.install_drb_security(ue_id, session.pdu_session_id, decision)
                            .await;
                    }
                    Err(refusal) => {
                        // The path switch has already been acknowledged, so there is
                        // no response left to refuse in. Warned and the session left
                        // as it was rather than protected on a policy this gNB cannot
                        // meet -- reporting success here would be the lie #32 is
                        // about, and silently downgrading is the other one.
                        warn!(
                            "Path switch for PSI {} carries a user-plane security policy \
                             this gNB cannot satisfy: {refusal}",
                            session.pdu_session_id
                        );
                    }
                }
            }

            let updated = self.ue_contexts.get_mut(&ue_id).and_then(|ctx| {
                ctx.pdu_sessions.get_mut(&session.pdu_session_id).map(|s| {
                    if let Some(tunnel) = session.ul_tunnel {
                        s.downlink_teid = tunnel.teid;
                        s.upf_address = tunnel.address;
                    }
                    s.clone()
                })
            });

            if let Some(s) = updated {
                let resource = PduSessionResource {
                    psi: s.psi as i32,
                    qfi: s.qfi,
                    // Empty, and that is the correct value here rather than a gap: a
                    // path switch moves the N3 tunnel and touches no QoS flow, and the
                    // GTP task treats a modify's flow list as a delta (issue #44), so
                    // an empty one leaves the session's SDAP mapping exactly as the
                    // setup built it. Restating `s.qfi` here would be worse than
                    // useless -- `NgapPduSession` keeps no 5QI, so it would re-admit
                    // the default flow with `None` and demote a GBR flow to the
                    // default DRB the UE is not using for it.
                    qos_flows: Vec::new(),
                    // NgapPduSession stores UPF TEID in downlink_teid and gNB TEID in
                    // uplink_teid (internal convention); the GTP task expects the opposite
                    // (uplink_teid = UPF dest). Map across the boundary here.
                    uplink_teid: s.downlink_teid,
                    downlink_teid: s.uplink_teid,
                    upf_address: s.upf_address,
                };
                let msg = GtpMessage::SessionModify { ue_id, resource };
                if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                    error!(
                        "Failed to send SessionModify after path switch for PSI {}: {}",
                        session.pdu_session_id, e
                    );
                }
            } else {
                warn!(
                    "Path switch acknowledged unknown PDU session {}",
                    session.pdu_session_id
                );
            }
        }

        info!("Path switch completed for UE[{}]", ue_id);
    }

    /// Handles Path Switch Request Failure from AMF.
    ///
    /// Per TS 38.413 §8.4.4.3 the gNB releases the UE-associated resources
    /// when the path switch is rejected.
    async fn handle_path_switch_request_failure(
        &mut self,
        _client_id: i32,
        _stream: u16,
        failure: PathSwitchRequestFailureData,
    ) {
        warn!(
            "Path Switch Request Failure: amf_ue_ngap_id={}, ran_ue_ngap_id={}, released sessions={:?}",
            failure.amf_ue_ngap_id, failure.ran_ue_ngap_id, failure.released_sessions
        );

        let ue_id = match self.find_ue_by_ran_id(failure.ran_ue_ngap_id as i64) {
            Some(ctx) => ctx.ue_id,
            None => {
                warn!(
                    "Path Switch Request Failure for unknown RAN-UE-NGAP-ID {}",
                    failure.ran_ue_ngap_id
                );
                return;
            }
        };

        // Release the sessions the 5GC reported as released (with cause)
        for (psi, cause) in &failure.released_sessions {
            info!(
                "Releasing PDU session {} after path switch failure (cause: {:?})",
                psi, cause
            );
            if let Some(ctx) = self.ue_contexts.get_mut(&ue_id) {
                ctx.remove_pdu_session(*psi);
            }
            let msg = GtpMessage::SessionRelease {
                ue_id,
                psi: *psi as i32,
            };
            if let Err(e) = self.task_base.gtp_tx.send(msg).await {
                error!("Failed to send SessionRelease to GTP: {}", e);
            }
        }

        // The UE context at the target gNB is released after a failed switch
        self.handle_ue_context_release_request(ue_id, UeReleaseRequestCause::RanOriginated)
            .await;
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Handles an operational NGAP PDU that could not be routed to a handler.
    ///
    /// Per TS 38.413 §8.7.5 (Error Indication) and §10 (handling of
    /// unknown/erroneous messages) the NG-RAN node answers with an Error
    /// Indication. We distinguish two cases:
    ///   * the PDU did not APER-decode at all → transfer-syntax-error (no
    ///     CriticalityDiagnostics, since the procedure code is unknown);
    ///   * the PDU decoded but no handler exists for the procedure →
    ///     abstract-syntax-error (ignore-and-notify) with CriticalityDiagnostics
    ///     carrying the offending procedure code.
    async fn handle_unroutable_pdu(&self, amf_id: i32, pdu_bytes: &[u8]) {
        let params = unroutable_error_params(pdu_bytes);
        match decode_ngap_pdu(pdu_bytes) {
            Ok(_) => warn!(
                "Unsupported NGAP procedure from AMF[{}]; sending Error Indication \
                 (abstract syntax error)",
                amf_id
            ),
            Err(e) => warn!(
                "Undecodable NGAP PDU from AMF[{}] ({}); sending Error Indication \
                 (transfer syntax error)",
                amf_id, e
            ),
        }
        self.send_error_indication(amf_id, params).await;
    }

    /// Encodes and sends an NGAP Error Indication to the AMF.
    ///
    /// Error Indication is non-UE-associated here (the offending PDU could not
    /// be tied to a known UE context), so it is sent on stream 0.
    async fn send_error_indication(&self, amf_id: i32, params: ErrorIndicationParams) {
        match encode_error_indication(&params) {
            Ok(data) => {
                self.send_ngap_non_ue(amf_id, 0, data).await;
            }
            Err(e) => {
                error!("Failed to encode Error Indication: {}", e);
            }
        }
    }

    /// Sends an NGAP PDU for non-UE-associated signaling (stream 0)
    async fn send_ngap_non_ue(&self, amf_id: i32, stream: u16, data: Vec<u8>) {
        let msg = SctpMessage::SendMessage {
            client_id: amf_id,
            stream,
            buffer: OctetString::from_slice(&data),
        };

        if let Err(e) = self.task_base.sctp_tx.send(msg).await {
            error!("Failed to send NGAP PDU to SCTP: {}", e);
        }
    }

    /// Sends an NGAP PDU for UE-associated signaling (stream > 0).
    ///
    /// Returns whether the PDU reached the SCTP task. Most callers have nothing to do
    /// differently either way and ignore it — but a caller that goes on to act as though
    /// the AMF has been told something does need to know, and used to have no way to ask
    /// (issue #169).
    async fn send_ngap_ue_associated(&self, amf_id: i32, stream: u16, data: Vec<u8>) -> bool {
        debug!(
            "Sending UE-associated NGAP PDU: amf_id={}, stream={}, len={}",
            amf_id,
            stream,
            data.len()
        );
        let msg = SctpMessage::SendMessage {
            client_id: amf_id,
            stream,
            buffer: OctetString::from_slice(&data),
        };

        if let Err(e) = self.task_base.sctp_tx.send(msg).await {
            error!("Failed to send NGAP PDU to SCTP: {}", e);
            return false;
        }
        true
    }

    /// Sends AN release to RRC
    async fn send_an_release(&self, ue_id: i32) {
        let msg = RrcMessage::AnRelease { ue_id };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            error!("Failed to send AN release to RRC: {}", e);
        }
    }

    /// Sends GTP UE context update
    async fn send_gtp_ue_context_update(&self, ue_id: i32, amf_ue_ngap_id: Option<i64>) {
        let update = GtpUeContextUpdate {
            ue_id,
            amf_ue_ngap_id,
        };
        let msg = GtpMessage::UeContextUpdate { ue_id, update };
        if let Err(e) = self.task_base.gtp_tx.send(msg).await {
            error!("Failed to send UE context update to GTP: {}", e);
        }
    }

    /// Updates initialization status and notifies App task
    async fn update_initialization_status(&mut self) {
        let any_ready = self
            .amf_contexts
            .values()
            .any(super::amf_context::NgapAmfContext::is_ready);

        if any_ready != self.is_initialized {
            self.is_initialized = any_ready;

            // Notify App task
            let msg = AppMessage::StatusUpdate(StatusUpdate {
                status_type: StatusType::NgapIsUp,
                value: any_ready,
            });
            if let Err(e) = self.task_base.app_tx.send(msg).await {
                error!("Failed to send status update to App: {}", e);
            }

            // If initialized, notify RRC to power on radio
            if any_ready {
                let msg = RrcMessage::RadioPowerOn;
                if let Err(e) = self.task_base.rrc_tx.send(msg).await {
                    error!("Failed to send radio power on to RRC: {}", e);
                }
                info!("NGAP initialized, radio powered on");
            }
        }
    }

    // ========================================================================
    // MBS (Multicast/Broadcast Service) Procedures (Rel-17)
    // ========================================================================

    /// Routes an inbound MBS session procedure into [`NgapMbsManager`]
    /// (TS 38.413 §9.2.9), returning whether this PDU was one.
    ///
    /// Called from `handle_ngap_pdu` after the UE-associated and NG-interface
    /// procedures and before `handle_unroutable_pdu`, so a PDU that is not an
    /// MBS procedure still earns its Error Indication. Returning `false` rather
    /// than sending one here keeps that single decision in one place.
    ///
    /// The three procedures handled are the AMF-initiated ones the RAN must
    /// answer or act on:
    ///
    /// - **71** `id-MulticastSessionActivation`: record the session and reply
    ///   `MulticastSessionActivationResponse` echoing `id-MBS-SessionID`.
    /// - **72** `id-MulticastSessionDeactivation`: tear the session down and
    ///   reply `MulticastSessionDeactivationResponse`.
    /// - **74** `id-MulticastGroupPaging`: page the group if this gNB serves one
    ///   of the TAIs named. Criticality is *ignore* for the area list, and the
    ///   procedure has **no** response message, so nothing is sent back.
    ///
    /// 73 `id-MulticastSessionUpdate` and 68 `id-BroadcastSessionSetup` are not
    /// handled: both carry `MBS-SessionTNLInfo5GC`, an MB-UPF shared-tunnel
    /// descriptor this node has no user-plane path for, so answering them
    /// successfully would claim a delivery capability that does not exist. They
    /// fall through to the Error Indication, which is the conformant reply to an
    /// unsupported procedure (TS 38.413 §8.7.5).
    async fn handle_mbs_pdu(&mut self, amf_id: i32, pdu_bytes: &[u8]) -> bool {
        let Ok(pdu) = decode_ngap_pdu(pdu_bytes) else {
            return false;
        };

        if let Ok(req) = parse_multicast_session_activation_request(&pdu) {
            self.handle_multicast_session_activation(amf_id, &req.mbs_session_id)
                .await;
            return true;
        }

        if let Ok(req) = parse_multicast_session_deactivation_request(&pdu) {
            self.handle_multicast_session_deactivation(amf_id, &req.mbs_session_id)
                .await;
            return true;
        }

        if let Ok(paging) = parse_multicast_group_paging(&pdu) {
            self.handle_ngap_multicast_group_paging(amf_id, &paging)
                .await;
            return true;
        }

        false
    }

    /// The cell identity `NgapMbsManager` tracks MBS activations against.
    ///
    /// The simulated gNB radiates exactly one cell, so the served cell is the
    /// one derived from the configured NCI — the same value the RRC layer uses
    /// as its `physCellId`, so "active in this cell" means the same thing on
    /// both sides of the stack.
    fn served_mbs_cell_id(&self) -> i32 {
        i32::from(phys_cell_id_from_nci(self.task_base.config.nci))
    }

    /// Applies a `MulticastSessionActivationRequest` and answers it
    /// (TS 38.413 §9.2.9.1 / §9.2.9.2).
    async fn handle_multicast_session_activation(
        &mut self,
        amf_id: i32,
        mbs_session_id: &MbsSessionId,
    ) {
        let cell_id = self.served_mbs_cell_id();

        // A multicast session, not broadcast: the activation procedures of
        // §9.2.9 are the multicast ones (broadcast uses 66-68), so per-UE
        // membership applies and `ue_join` must be accepted.
        let mut session =
            GnbMbsSession::new_multicast(mbs_session_id.tmgi_hex(), mbs_session_id.tmgi);
        session.activate_cell(cell_id);

        let session = self.mbs_ngap_sessions.start_session(session);
        info!(
            "MulticastSessionActivationRequest from AMF[{}]: TMGI={} activated in cell {} \
             ({} MBS session(s) now active)",
            amf_id,
            session.tmgi_hex(),
            cell_id,
            self.mbs_ngap_sessions.session_count()
        );

        match encode_multicast_session_activation_response(mbs_session_id) {
            Ok(data) => {
                // MBS session procedures are non-UE-associated, so stream 0
                // (TS 38.412 §7).
                self.send_ngap_non_ue(amf_id, 0, data).await;
            }
            Err(e) => {
                error!("Failed to encode MulticastSessionActivationResponse: {}", e);
            }
        }
    }

    /// Applies a `MulticastSessionDeactivationRequest` and answers it
    /// (TS 38.413 §9.2.9.3 / §9.2.9.4).
    ///
    /// A TMGI this gNB never activated is still answered with a successful
    /// outcome: the AMF's intent -- that the session not be delivered here --
    /// already holds, so a failure would tell it to retry a teardown that has
    /// nothing left to tear down.
    async fn handle_multicast_session_deactivation(
        &mut self,
        amf_id: i32,
        mbs_session_id: &MbsSessionId,
    ) {
        match self.mbs_ngap_sessions.stop_session(&mbs_session_id.tmgi) {
            Some(session) => info!(
                "MulticastSessionDeactivationRequest from AMF[{}]: TMGI={} deactivated \
                 ({} UE(s) had joined)",
                amf_id,
                session.tmgi_hex(),
                session.joined_ues.len()
            ),
            None => warn!(
                "MulticastSessionDeactivationRequest from AMF[{}] for TMGI={}, which is not \
                 active here; acknowledging anyway",
                amf_id,
                mbs_session_id.tmgi_hex()
            ),
        }

        match encode_multicast_session_deactivation_response(mbs_session_id) {
            Ok(data) => self.send_ngap_non_ue(amf_id, 0, data).await,
            Err(e) => {
                error!(
                    "Failed to encode MulticastSessionDeactivationResponse: {}",
                    e
                );
            }
        }
    }

    /// Applies a `MulticastGroupPaging` (TS 38.413 §9.2.9.5).
    ///
    /// The procedure has no response message. The gNB pages only if it serves
    /// one of the TAIs in `MulticastGroupPagingAreaList` and is actually
    /// radiating the paged session, since paging for a group this cell does not
    /// carry would wake UEs that have nothing to receive.
    async fn handle_ngap_multicast_group_paging(
        &mut self,
        amf_id: i32,
        paging: &nextgsim_ngap::procedures::mbs::MulticastGroupPagingData,
    ) {
        let config = &self.task_base.config;
        let served_plmn = config.plmn.encode();
        let served_tac = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];

        let serves_a_paged_tai = paging
            .paging_tais
            .iter()
            .any(|tai| tai.plmn_identity == served_plmn && tai.tac == served_tac);

        if !serves_a_paged_tai {
            debug!(
                "MulticastGroupPaging from AMF[{}] for TMGI={} names no served TAI \
                 (served plmn={:02x?} tac={:02x?}); ignoring",
                amf_id,
                paging.mbs_session_id.tmgi_hex(),
                served_plmn,
                served_tac
            );
            return;
        }

        let cell_id = self.served_mbs_cell_id();
        let is_radiating = self
            .mbs_ngap_sessions
            .sessions_for_cell(cell_id)
            .iter()
            .any(|s| s.tmgi == paging.mbs_session_id.tmgi);

        if !is_radiating {
            warn!(
                "MulticastGroupPaging from AMF[{}] for TMGI={}, which is not active in cell {}; \
                 nothing to page",
                amf_id,
                paging.mbs_session_id.tmgi_hex(),
                cell_id
            );
            return;
        }

        let joined = self
            .mbs_ngap_sessions
            .get(&paging.mbs_session_id.tmgi)
            .map_or(0, |s| s.joined_ues.len());

        info!(
            "MulticastGroupPaging from AMF[{}]: TMGI={} is active in cell {} with {} joined UE(s)",
            amf_id,
            paging.mbs_session_id.tmgi_hex(),
            cell_id,
            joined
        );
    }

    /// Handles MBS Session Activation Request from AMF
    async fn handle_mbs_session_activation_request(
        &mut self,
        session_id: u32,
        tmgi_bytes: [u8; 6],
        is_broadcast: bool,
        multicast_ip: Option<std::net::IpAddr>,
        qfi: u8,
    ) {
        info!(
            "MBS Session Activation Request: session_id={}, tmgi={:02x?}, broadcast={}, qfi={}",
            session_id, tmgi_bytes, is_broadcast, qfi
        );

        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        let mut mbs_ctx = GnbMbsContext::new(session_id, tmgi, is_broadcast);

        // Create multicast tunnel info if provided
        if let Some(mcast_ip) = multicast_ip {
            let tnl_info = MulticastTunnelInfo {
                multicast_ip: mcast_ip,
                source_ip: None,
                teid: self.next_downlink_teid(),
                qfi,
            };
            mbs_ctx.activate(Some(tnl_info));
        } else {
            mbs_ctx.activate(None);
        }

        // Add session to manager
        if self.mbs_sessions.add_session(mbs_ctx) {
            info!(
                "MBS session {} activated successfully (TMGI={:02x?})",
                session_id, tmgi_bytes
            );
        } else {
            warn!(
                "Failed to activate MBS session {}, already exists",
                session_id
            );
        }
    }

    /// Handles MBS Session Deactivation Request from AMF
    async fn handle_mbs_session_deactivation_request(&mut self, session_id: u32) {
        info!(
            "MBS Session Deactivation Request: session_id={}",
            session_id
        );

        if let Some(mut session) = self.mbs_sessions.remove_session(session_id) {
            session.deactivate();
            info!("MBS session {} deactivated", session_id);
        } else {
            warn!("MBS session {} not found for deactivation", session_id);
        }
    }

    /// Handles Multicast Group Paging from AMF
    async fn handle_multicast_group_paging(&mut self, tmgi_bytes: [u8; 6], area_scope: Vec<u32>) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "Multicast Group Paging: tmgi={:02x?}, area_scope={:?}",
            tmgi_bytes, area_scope
        );

        // Find the MBS session
        if let Some(session) = self.mbs_sessions.get_session_by_tmgi(&tmgi) {
            if session.is_active() {
                info!(
                    "Paging for MBS session {}, {} joined UEs",
                    session.session_id,
                    session.ue_count()
                );
                // In a real implementation, would trigger RRC paging for interested UEs
                // For now, just log the paging request
            } else {
                warn!(
                    "MBS session {} is not active (state: {:?})",
                    session.session_id, session.state
                );
            }
        } else {
            warn!("MBS session with TMGI {:02x?} not found", tmgi_bytes);
        }
    }

    /// Handles MBS UE Join Request from RRC
    async fn handle_mbs_ue_join_request(&mut self, ue_id: i32, tmgi_bytes: [u8; 6]) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "MBS UE Join Request: ue_id={}, tmgi={:02x?}",
            ue_id, tmgi_bytes
        );

        if let Some(session) = self.mbs_sessions.get_session_by_tmgi_mut(&tmgi) {
            if session.add_ue(ue_id) {
                info!(
                    "UE {} joined MBS session {}, total UEs: {}",
                    ue_id,
                    session.session_id,
                    session.ue_count()
                );

                // Send MulticastSessionUpdateRequest to AMF to inform of UE join
                // In a real implementation, would encode and send NGAP message
                debug!(
                    "Would send MulticastSessionUpdateRequest to AMF for session {} (UE {} joined)",
                    session.session_id, ue_id
                );
            } else {
                debug!("UE {} already in MBS session {}", ue_id, session.session_id);
            }
        } else {
            warn!(
                "Cannot join MBS session: TMGI {:02x?} not found",
                tmgi_bytes
            );
        }
    }

    /// Handles MBS UE Leave Request from RRC
    async fn handle_mbs_ue_leave_request(&mut self, ue_id: i32, tmgi_bytes: [u8; 6]) {
        let tmgi = Tmgi::from_bytes(&tmgi_bytes);
        info!(
            "MBS UE Leave Request: ue_id={}, tmgi={:02x?}",
            ue_id, tmgi_bytes
        );

        if let Some(session) = self.mbs_sessions.get_session_by_tmgi_mut(&tmgi) {
            if session.remove_ue(ue_id) {
                info!(
                    "UE {} left MBS session {}, remaining UEs: {}",
                    ue_id,
                    session.session_id,
                    session.ue_count()
                );

                // Send MulticastSessionUpdateRequest to AMF to inform of UE leave
                // In a real implementation, would encode and send NGAP message
                debug!(
                    "Would send MulticastSessionUpdateRequest to AMF for session {} (UE {} left)",
                    session.session_id, ue_id
                );
            } else {
                debug!("UE {} not in MBS session {}", ue_id, session.session_id);
            }
        } else {
            warn!(
                "Cannot leave MBS session: TMGI {:02x?} not found",
                tmgi_bytes
            );
        }
    }
}

/// Builds the Error Indication parameters for an NGAP PDU that the gNB could
/// not route (TS 38.413 §8.7.5 / §10).
///
/// * undecodable PDU → transfer-syntax-error (no CriticalityDiagnostics);
/// * decoded but unsupported procedure → abstract-syntax-error
///   (ignore-and-notify) with CriticalityDiagnostics carrying the procedure
///   code and triggering-message kind.
fn unroutable_error_params(pdu_bytes: &[u8]) -> ErrorIndicationParams {
    match decode_ngap_pdu(pdu_bytes) {
        Ok(pdu) => {
            let (proc_code, trigger) = ngap_procedure_code(&pdu);
            let mut p = error_indication_abstract_syntax_error(None, None, false);
            p.criticality_diagnostics = Some(CriticalityDiagnosticsInfo {
                procedure_code: Some(proc_code),
                triggering_message: Some(trigger),
                procedure_criticality: None,
                ies_criticality_diagnostics: Vec::new(),
            });
            p
        }
        Err(_) => error_indication_transfer_syntax_error(None, None),
    }
}

/// Serializes a 5G-S-TMSI UE paging identity into the canonical 48-bit form
/// for an RRC Paging record (TS 23.003 §2.10.1): AMF Set ID (10 bits) +
/// AMF Pointer (6 bits) packed into 2 octets, followed by the 32-bit 5G-TMSI.
fn serialize_five_g_s_tmsi(identity: &UePagingIdentityValue) -> Vec<u8> {
    match identity {
        UePagingIdentityValue::FiveGSTmsi(tmsi) => {
            let mut out = Vec::with_capacity(6);
            // AMF Set ID is 10 bits, AMF Pointer is 6 bits -> 16 bits.
            let packed = ((tmsi.amf_set_id & 0x3FF) << 6) | (tmsi.amf_pointer as u16 & 0x3F);
            out.extend_from_slice(&packed.to_be_bytes());
            out.extend_from_slice(&tmsi.five_g_tmsi);
            out
        }
    }
}

/// Extracts the NGAP procedure code and triggering-message kind from a decoded
/// PDU, for populating CriticalityDiagnostics in an Error Indication
/// (TS 38.413 §9.3.1.3).
fn ngap_procedure_code(pdu: &NGAP_PDU) -> (u8, TriggeringMessageValue) {
    match pdu {
        NGAP_PDU::InitiatingMessage(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::InitiatingMessage,
        ),
        NGAP_PDU::SuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::SuccessfulOutcome,
        ),
        NGAP_PDU::UnsuccessfulOutcome(m) => (
            m.procedure_code.0,
            TriggeringMessageValue::UnsuccessfulOutcome,
        ),
    }
}

// ============================================================================
// Task Implementation
// ============================================================================

#[async_trait::async_trait]
impl Task for NgapTask {
    type Message = NgapMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NGAP task started");

        // Guard timers are polled rather than each armed as its own task: one tick
        // handles every pending timer, and the granularity bounds the overshoot (a
        // timer fires within one tick of its deadline, never before it -- `expired`
        // compares deadlines, so the tick rate cannot make one fire early).
        let mut guard_tick = tokio::time::interval(Duration::from_millis(500));
        guard_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            let msg = tokio::select! {
                _ = guard_tick.tick() => {
                    self.process_expired_guard_timers(Instant::now()).await;
                    continue;
                }
                msg = rx.recv() => msg,
            };
            match msg {
                Some(TaskMessage::Message(msg)) => match msg {
                    NgapMessage::SctpAssociationUp {
                        client_id,
                        association_id,
                        in_streams,
                        out_streams,
                    } => {
                        self.handle_association_up(
                            client_id,
                            association_id,
                            in_streams,
                            out_streams,
                        )
                        .await;
                    }
                    NgapMessage::SctpAssociationDown { client_id } => {
                        self.handle_association_down(client_id).await;
                    }
                    NgapMessage::ReceiveNgapPdu {
                        client_id,
                        stream,
                        pdu,
                    } => {
                        self.handle_ngap_pdu(client_id, stream, pdu).await;
                    }
                    NgapMessage::InitialNasDelivery {
                        ue_id,
                        pdu,
                        rrc_establishment_cause,
                        s_tmsi,
                        s_nssai_list,
                    } => {
                        self.handle_initial_nas_delivery(
                            ue_id,
                            pdu,
                            rrc_establishment_cause,
                            s_tmsi,
                            s_nssai_list,
                        )
                        .await;
                    }
                    NgapMessage::UplinkNasDelivery { ue_id, pdu } => {
                        self.handle_uplink_nas_delivery(ue_id, pdu).await;
                    }
                    NgapMessage::RadioLinkFailure { ue_id } => {
                        self.handle_radio_link_failure(ue_id).await;
                    }
                    NgapMessage::InitiateHandover(request) => {
                        // The production caller `initiate_handover` never had (issue #39).
                        if !self.initiate_handover(*request).await {
                            // Reported rather than swallowed: a handover that did not
                            // start must not look like one that did, which is the
                            // no-op-that-logs-success failure this issue is about.
                            warn!("Handover initiation did not send a HANDOVER REQUIRED");
                        }
                    }
                    NgapMessage::HandoverAccessCompleted { ue_id } => {
                        // TS 38.413 §8.4.3: the target tells the AMF the UE has arrived,
                        // which is what makes the AMF switch the user plane. The
                        // production caller `send_handover_notify` never had.
                        let target = self
                            .ue_contexts
                            .values()
                            .find(|c| c.ue_id == ue_id)
                            .map(|c| (c.amf_ctx_id, c.amf_ue_ngap_id, c.ran_ue_ngap_id as u32));
                        match target {
                            Some((amf_ctx_id, Some(amf_ue_ngap_id), ran_ue_ngap_id)) => {
                                self.send_handover_notify(
                                    amf_ctx_id,
                                    amf_ue_ngap_id as u64,
                                    ran_ue_ngap_id,
                                )
                                .await;
                            }
                            Some((_, None, _)) => warn!(
                                "UE[{ue_id}] arrived on this cell but has no AMF UE NGAP ID, \
                                 so no HANDOVER NOTIFY can be sent"
                            ),
                            None => warn!("Handover access completed for unknown UE[{ue_id}]"),
                        }
                    }
                    NgapMessage::SendPathSwitchRequest {
                        ue_id,
                        source_amf_ue_ngap_id,
                    } => {
                        // TS 38.413 §8.4.4, the Xn counterpart. The production caller
                        // `send_path_switch_request` never had.
                        self.send_path_switch_request(ue_id, source_amf_ue_ngap_id)
                            .await;
                    }
                    NgapMessage::SendRanConfigurationUpdate { amf_id } => {
                        // Per-association: RAN CONFIGURATION UPDATE is sent on the
                        // NG-C interface instance whose configuration it describes,
                        // so `None` fans out to every Ready AMF rather than picking
                        // one arbitrarily.
                        let targets: Vec<i32> = match amf_id {
                            Some(id) => vec![id],
                            None => self
                                .amf_contexts
                                .values()
                                .filter(|c| c.is_ready())
                                .map(|c| c.ctx_id)
                                .collect(),
                        };
                        for id in targets {
                            self.send_ran_configuration_update(id).await;
                        }
                    }
                    NgapMessage::PduSessionResourceNotify {
                        ue_id,
                        released_sessions,
                        notified_sessions,
                    } => {
                        self.send_pdu_session_resource_notify(
                            ue_id,
                            released_sessions,
                            notified_sessions,
                        )
                        .await;
                    }
                    NgapMessage::UeContextReleaseRequest { ue_id, cause } => {
                        self.handle_ue_context_release_request(ue_id, cause).await;
                    }
                    NgapMessage::NtnTimingInfoReceived {
                        satellite_type,
                        satellite_id,
                        propagation_delay_us,
                        common_ta_us,
                        k_offset,
                    } => {
                        info!(
                                "NTN timing info: sat_type={}, sat_id={}, delay={}us, TA={}us, k_offset={}",
                                satellite_type, satellite_id, propagation_delay_us, common_ta_us, k_offset
                            );
                    }
                    NgapMessage::MbsSessionActivationRequest {
                        session_id,
                        tmgi,
                        is_broadcast,
                        multicast_ip,
                        qfi,
                    } => {
                        self.handle_mbs_session_activation_request(
                            session_id,
                            tmgi,
                            is_broadcast,
                            multicast_ip,
                            qfi,
                        )
                        .await;
                    }
                    NgapMessage::MbsSessionDeactivationRequest { session_id } => {
                        self.handle_mbs_session_deactivation_request(session_id)
                            .await;
                    }
                    NgapMessage::MulticastGroupPaging { tmgi, area_scope } => {
                        self.handle_multicast_group_paging(tmgi, area_scope).await;
                    }
                    NgapMessage::MbsUeJoinRequest { ue_id, tmgi } => {
                        self.handle_mbs_ue_join_request(ue_id, tmgi).await;
                    }
                    NgapMessage::MbsUeLeaveRequest { ue_id, tmgi } => {
                        self.handle_mbs_ue_leave_request(ue_id, tmgi).await;
                    }
                },
                Some(TaskMessage::Shutdown) => {
                    info!("NGAP task received shutdown signal");
                    break;
                }
                None => {
                    info!("NGAP task channel closed");
                    break;
                }
            }
        }

        info!(
            "NGAP task stopped, {} AMF contexts, {} UE contexts",
            self.amf_contexts.len(),
            self.ue_contexts.len()
        );
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;
    use nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
            ntn_config: None,
            mbs_enabled: false,
            prose_enabled: false,
            lcs_enabled: false,
            snpn_config: None,
            ..Default::default()
        }
    }

    #[test]
    fn test_select_as_algorithms_prefers_aes_and_derives_distinct_keys() {
        use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
        use nextgsim_ngap::procedures::initial_context_setup::UeSecurityCapabilitiesValue;

        // Caps: NEA1|NEA2 (0xC000), NIA1|NIA2 (0xC000) — MSB (0x8000) = 128-xEA1.
        let caps = UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: 0xC000,
            nr_integrity_algorithms: 0xC000,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        };
        let ((ciph_id, ciph), (int_id, integ)) = NgapTask::select_as_algorithms(&caps);
        // gNB prefers AES (NEA2/NIA2).
        assert_eq!(ciph_id, 2);
        assert_eq!(int_id, 2);
        assert_eq!(ciph, CipheringAlgorithmType::Nea2);
        assert_eq!(integ, IntegrityAlgorithmType::Nia2);

        // Empty bitmaps: no encryption -> NEA0, but integrity never falls back
        // to NIA0 (TS 33.501 §5.11.2) — defaults to mandatory-to-support NIA2.
        let caps0 = UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: 0,
            nr_integrity_algorithms: 0,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        };
        let ((c0, _), (i0, _)) = NgapTask::select_as_algorithms(&caps0);
        assert_eq!(c0, 0);
        assert_eq!(i0, 2);

        // AS keys derived from KgNB are distinct per algorithm-type distinguisher
        // and non-zero (TS 33.501 Annex A.8).
        let kgnb = [0x11u8; 32];
        let k_rrc_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcEnc, 2);
        let k_rrc_int = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::RrcInt, 2);
        let k_up_enc = derive_rrc_up_key(&kgnb, AlgorithmTypeDistinguisher::UpEnc, 2);
        assert_ne!(k_rrc_enc, [0u8; 16]);
        assert_ne!(k_rrc_enc, k_rrc_int);
        assert_ne!(k_rrc_enc, k_up_enc);
    }

    #[test]
    fn test_gnb_drb_reconfiguration_is_valid_and_carries_qfis() {
        use nextgsim_rrc::codec::decode_rrc;
        use nextgsim_rrc::codec::generated::DL_DCCH_Message;
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            build_drb_reconfiguration_params, is_rrc_reconfiguration,
        };

        // The single-DRB shape `establish_drb` used before issue #44 and still produces
        // byte for byte without `sdap-dataplane`: PDU session 5, DRB 5, LCID 8, accepted
        // QFIs 1 & 9, and the A3 measConfig from THIS gNB's configuration (issue #170).
        // The result must be a decodable DL-DCCH RRCReconfiguration.
        let params = build_drb_reconfiguration_params(
            0,
            5,
            5,
            8,
            &[1, 9],
            true,
            DrbIntegrityProtection::Disabled,
            a3_meas_config_params(&test_config()),
        )
        .unwrap();
        let bytes = encode_rrc_reconfiguration(&params).unwrap();
        assert!(!bytes.is_empty());
        let msg: DL_DCCH_Message = decode_rrc(&bytes).unwrap();
        assert!(is_rrc_reconfiguration(&msg));
    }

    /// `establish_drb`'s message must carry the gNB's A3 margin (issue #170).
    ///
    /// The PRODUCTION-caller assertion: `build_drb_reconfiguration_params` can take
    /// a measConfig, and this is what says the live path passes one. Without it the
    /// feature would be "correct but unreachable" — the gNB would still send a
    /// measConfig-less reconfiguration and every test above would pass.
    #[tokio::test]
    async fn establish_drb_configures_the_ues_a3_reporting_margin() {
        use nextgsim_rrc::procedures::meas_config::read_a3_meas_configs;
        use nextgsim_rrc::procedures::rrc_reconfiguration::decode_rrc_reconfiguration;

        let mut config = test_config();
        // A margin no default in this tree holds, so a reproduced default cannot
        // pass for a signalled one.
        config.cho_a3_offset_db = 11.0;
        config.cho_hysteresis_db = 3.5;
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        // One flow with no 5QI, so the QFI→DRB policy puts it on the default DRB and
        // this measConfig assertion sees one bearer with `sdap-dataplane` on or off.
        task.establish_drb(1, 1, &[(1, None)], DrbIntegrityProtection::Disabled)
            .await;

        let pdu = loop {
            match rrc_rx.try_recv() {
                Ok(TaskMessage::Message(RrcMessage::RrcReconfiguration { pdu, .. })) => break pdu,
                Ok(_) => continue,
                Err(_) => panic!("establish_drb must hand an RRCReconfiguration to the RRC task"),
            }
        };

        let data = decode_rrc_reconfiguration(pdu.data()).expect("a decodable reconfiguration");
        let signalled = data
            .meas_config
            .expect("the live DRB reconfiguration must carry a measConfig");
        let read = read_a3_meas_configs(&signalled);
        assert_eq!(read.len(), 1, "one A3 reporting binding");
        assert_eq!(
            read[0].meas_id, 1,
            "measId 1, which REPLACES the UE's pre-signalling default"
        );
        assert_eq!(read[0].a3_offset_db, 11, "the gNB's configured 11 dB");
        assert_eq!(
            read[0].hysteresis_half_db, 7,
            "the gNB's configured 3.5 dB, in 0.5 dB units"
        );
    }

    /// And a margin TS 38.331 cannot carry costs the UE its measConfig and
    /// NOTHING else: the DRB half of the same message still goes out.
    #[tokio::test]
    async fn an_unsignallable_margin_still_establishes_the_drb() {
        use nextgsim_rrc::procedures::rrc_reconfiguration::decode_rrc_reconfiguration;

        let mut config = test_config();
        config.cho_a3_offset_db = 99.0;
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        // One flow with no 5QI, so the QFI→DRB policy puts it on the default DRB and
        // this measConfig assertion sees one bearer with `sdap-dataplane` on or off.
        task.establish_drb(1, 1, &[(1, None)], DrbIntegrityProtection::Disabled)
            .await;

        let pdu = loop {
            match rrc_rx.try_recv() {
                Ok(TaskMessage::Message(RrcMessage::RrcReconfiguration { pdu, .. })) => break pdu,
                Ok(_) => continue,
                Err(_) => {
                    panic!("the DRB must still be established when the margin is unsignallable")
                }
            }
        };
        let data = decode_rrc_reconfiguration(pdu.data()).expect("a decodable reconfiguration");
        assert!(
            data.meas_config.is_none(),
            "an unsignallable margin must be omitted, not clamped"
        );
        assert!(
            data.radio_bearer_config.is_some(),
            "and the DRB half must survive"
        );
    }

    #[test]
    fn test_ngap_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NgapTask::new(task_base);
        assert!(task.amf_contexts.is_empty());
        assert!(task.ue_contexts.is_empty());
        assert!(!task.is_initialized);
    }

    #[test]
    fn test_ngap_task_amf_context_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        assert!(task.amf_contexts.contains_key(&1));
        let ctx = task.find_amf_context(1).unwrap();
        assert_eq!(ctx.ctx_id, 1);
        assert_eq!(ctx.state, AmfState::NotConnected);
    }

    #[test]
    fn test_ngap_task_ue_context_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        // Create AMF context first
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        // Create UE context
        let ran_id = task.create_ue_context(10, 1);
        assert!(ran_id.is_some());
        assert_eq!(ran_id.unwrap(), 1); // First RAN UE NGAP ID

        let ue_ctx = task.find_ue_context(10).unwrap();
        assert_eq!(ue_ctx.ue_id, 10);
        assert_eq!(ue_ctx.amf_ctx_id, 1);
    }

    #[test]
    fn test_ngap_task_select_amf() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);
        let plmn = [0x00, 0xf1, 0x10];

        // No AMF available
        assert!(task.select_amf(&plmn, &[]).is_none());

        // Add AMF but not ready
        task.create_amf_context(1);
        assert!(task.select_amf(&plmn, &[]).is_none());

        // Make AMF ready
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 100;
        }
        assert_eq!(task.select_amf(&plmn, &[]), Some(1));
    }

    /// Slice-aware AMF selection (TS 38.413 §8.6.1.2, issue #41): a UE that
    /// requests S-NSSAI X is routed to an AMF serving X even when another AMF
    /// has higher capacity, and an empty request falls back to capacity.
    #[test]
    fn test_select_amf_slice_aware() {
        use nextgsim_ngap::procedures::{PlmnSupportItem, SNssai};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        let plmn = [0x00, 0xf1, 0x10];

        // AMF 1: higher capacity, serves only SST=1.
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 200;
            ctx.plmn_support_list = vec![PlmnSupportItem {
                plmn_identity: plmn,
                slice_support_list: vec![SNssai { sst: 1, sd: None }],
            }];
        }
        // AMF 2: lower capacity, serves only SST=2.
        task.create_amf_context(2);
        if let Some(ctx) = task.find_amf_context_mut(2) {
            ctx.on_association_up(101, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 100;
            ctx.plmn_support_list = vec![PlmnSupportItem {
                plmn_identity: plmn,
                slice_support_list: vec![SNssai { sst: 2, sd: None }],
            }];
        }

        // Requesting SST=2 must route to AMF 2 despite AMF 1's higher capacity.
        assert_eq!(
            task.select_amf(&plmn, &[SNssai { sst: 2, sd: None }]),
            Some(2)
        );
        // Requesting SST=1 routes to AMF 1.
        assert_eq!(
            task.select_amf(&plmn, &[SNssai { sst: 1, sd: None }]),
            Some(1)
        );
        // No slice constraint → highest capacity (AMF 1).
        assert_eq!(task.select_amf(&plmn, &[]), Some(1));
        // A slice no AMF serves → capacity-only fallback (AMF 1), still attaches.
        assert_eq!(
            task.select_amf(&plmn, &[SNssai { sst: 9, sd: None }]),
            Some(1)
        );
    }

    /// AMF identity by GUAMI (TS 38.412 §7, issue #41 criterion 8): two SCTP
    /// associations advertising the same GUAMI are one AMF reached over two
    /// TNLAs; a distinct GUAMI is a separate AMF.
    #[test]
    fn test_amf_instances_group_tnlas_by_guami() {
        use nextgsim_ngap::procedures::{Guami, ServedGuamiItem};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        let plmn = [0x00, 0xf1, 0x10];

        // Two associations (TNLAs) advertising the SAME GUAMI = one AMF.
        for client in [1, 2] {
            task.create_amf_context(client);
            if let Some(ctx) = task.find_amf_context_mut(client) {
                ctx.on_association_up(100 + client, 4, 4);
                ctx.state = AmfState::Ready;
                ctx.served_guami_list = vec![ServedGuamiItem {
                    guami: Guami {
                        plmn_identity: plmn,
                        amf_region_id: 1,
                        amf_set_id: 2,
                        amf_pointer: 3,
                    },
                    backup_amf_name: None,
                }];
            }
        }
        // A third association for a DIFFERENT AMF (different AMF Pointer).
        task.create_amf_context(3);
        if let Some(ctx) = task.find_amf_context_mut(3) {
            ctx.on_association_up(103, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.served_guami_list = vec![ServedGuamiItem {
                guami: Guami {
                    plmn_identity: plmn,
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 9,
                },
                backup_amf_name: None,
            }];
        }

        let instances = task.amf_instances();
        assert_eq!(instances.len(), 2, "two GUAMIs => two logical AMFs");
        let (id0, tnlas0) = &instances[0];
        assert_eq!(id0.amf_pointer, 3);
        assert_eq!(
            tnlas0,
            &vec![1, 2],
            "same-GUAMI associations collapse to one AMF with two TNLAs"
        );
        let (id1, tnlas1) = &instances[1];
        assert_eq!(id1.amf_pointer, 9);
        assert_eq!(tnlas1, &vec![3]);
    }

    /// Multiple-TNLA load balancing (TS 38.412 §7, issue #41 criterion 8):
    /// new UEs spread across the TNLAs of one AMF instead of piling on one.
    #[test]
    fn test_select_amf_load_balances_across_tnlas() {
        use nextgsim_ngap::procedures::{Guami, ServedGuamiItem};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        let plmn = [0x00, 0xf1, 0x10];

        // Two TNLAs of the SAME AMF (same GUAMI), equal capacity.
        for client in [1, 2] {
            task.create_amf_context(client);
            if let Some(ctx) = task.find_amf_context_mut(client) {
                ctx.on_association_up(100 + client, 4, 4);
                ctx.state = AmfState::Ready;
                ctx.relative_capacity = 100;
                ctx.served_guami_list = vec![ServedGuamiItem {
                    guami: Guami {
                        plmn_identity: plmn,
                        amf_region_id: 1,
                        amf_set_id: 2,
                        amf_pointer: 3,
                    },
                    backup_amf_name: None,
                }];
            }
        }

        // First UE lands on the lower TNLA id (both idle).
        let first = task.select_amf(&plmn, &[]).unwrap();
        assert_eq!(first, 1);
        task.create_ue_context(10, first);
        // Next selection load-balances onto the other TNLA of the same AMF.
        let second = task.select_amf(&plmn, &[]).unwrap();
        assert_eq!(
            second, 2,
            "second UE load-balanced onto the other TNLA of the same AMF"
        );
        task.create_ue_context(11, second);
        // Both TNLAs now at load 1 → tie breaks back to the lower id.
        assert_eq!(task.select_amf(&plmn, &[]).unwrap(), 1);
    }

    #[test]
    fn test_ngap_task_ran_ue_ngap_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        let id1 = task.next_ran_ue_ngap_id();
        let id2 = task.next_ran_ue_ngap_id();
        let id3 = task.next_ran_ue_ngap_id();

        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
        assert_eq!(id3, 3);
    }

    #[test]
    fn test_ngap_task_ue_context_deletion() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NgapTask::new(task_base);

        // Setup
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(10, 1);

        assert!(task.find_ue_context(10).is_some());

        // Delete
        task.delete_ue_context(10);
        assert!(task.find_ue_context(10).is_none());
    }

    // ------------------------------------------------------------------
    // amfg-06: inbound Paging
    // ------------------------------------------------------------------

    use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi as PagingTmsi, Tai};
    use nextgsim_ngap::procedures::paging::PagingData;

    fn sample_tmsi() -> PagingTmsi {
        PagingTmsi {
            amf_set_id: 0x155,
            amf_pointer: 0x2A,
            five_g_tmsi: [0xDE, 0xAD, 0xBE, 0xEF],
        }
    }

    #[test]
    fn test_serialize_five_g_s_tmsi_packing() {
        let id = UePagingIdentityValue::FiveGSTmsi(sample_tmsi());
        let bytes = serialize_five_g_s_tmsi(&id);
        assert_eq!(bytes.len(), 6);
        // packed = (0x155 << 6) | 0x2A = 0x556A
        assert_eq!(&bytes[0..2], &[0x55, 0x6A]);
        assert_eq!(&bytes[2..6], &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[tokio::test]
    async fn test_paging_matching_tai_triggers_rrc() {
        let config = test_config();
        let served_plmn = config.plmn.encode();
        let served_tac = [
            ((config.tac >> 16) & 0xFF) as u8,
            ((config.tac >> 8) & 0xFF) as u8,
            (config.tac & 0xFF) as u8,
        ];
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        let paging = PagingData {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(sample_tmsi()),
            tai_list_for_paging: vec![Tai {
                plmn_identity: served_plmn,
                tac: served_tac,
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
        };
        task.handle_paging(1, paging).await;

        match rrc_rx.try_recv() {
            Ok(TaskMessage::Message(RrcMessage::Paging {
                ue_paging_tmsi,
                tai_list_for_paging,
                drx_cycle_frames,
            })) => {
                assert_eq!(ue_paging_tmsi.len(), 6);
                assert_eq!(tai_list_for_paging.len(), 6); // one matching TAI
                assert_eq!(&tai_list_for_paging[0..3], &served_plmn);
                assert_eq!(&tai_list_for_paging[3..6], &served_tac);
                // The AMF's (default)PagingDRX must REACH the RRC layer -- it was
                // decoded and dropped before issue #99. This fixture sends none,
                // so the absence is what carries through.
                assert_eq!(drx_cycle_frames, None);
            }
            other => panic!("expected RRC Paging, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_paging_non_matching_tai_is_ignored() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        let paging = PagingData {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(sample_tmsi()),
            // A TAI this gNB does not serve.
            tai_list_for_paging: vec![Tai {
                plmn_identity: [0x99, 0x99, 0x99],
                tac: [0x12, 0x34, 0x56],
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
        };
        task.handle_paging(1, paging).await;

        assert!(
            rrc_rx.try_recv().is_err(),
            "no RRC Paging should be emitted for an unserved TAI"
        );
    }

    // ------------------------------------------------------------------
    // amfg-07: NG Reset handler clears exactly the listed UE contexts
    // ------------------------------------------------------------------

    use nextgsim_ngap::procedures::ng_reset::NgResetData;

    #[tokio::test]
    async fn test_ng_reset_partial_clears_listed_contexts() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }

        // Two UE contexts on AMF[1]; give them distinct RAN UE NGAP IDs.
        let ran1 = task.create_ue_context(10, 1).expect("ue1");
        let ran2 = task.create_ue_context(20, 1).expect("ue2");
        assert_ne!(ran1, ran2);

        // Reset only UE[10] by its RAN UE NGAP ID.
        let reset = NgResetData {
            cause: None,
            scope: NgResetScope::Part(vec![UeAssociation {
                amf_ue_ngap_id: None,
                ran_ue_ngap_id: Some(ran1 as u32),
            }]),
        };
        task.handle_ng_reset(1, 0, reset).await;

        assert!(
            task.find_ue_context(10).is_none(),
            "UE[10] should be cleared"
        );
        assert!(task.find_ue_context(20).is_some(), "UE[20] must remain");
    }

    #[tokio::test]
    async fn test_ng_reset_all_clears_all_contexts_for_amf() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);

        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(10, 1).expect("ue1");
        task.create_ue_context(20, 1).expect("ue2");

        let reset = NgResetData {
            cause: None,
            scope: NgResetScope::All,
        };
        task.handle_ng_reset(1, 0, reset).await;

        assert!(task.find_ue_context(10).is_none());
        assert!(task.find_ue_context(20).is_none());
    }

    // ------------------------------------------------------------------
    // amfg-05: Error Indication generation for unroutable PDUs
    // ------------------------------------------------------------------

    use nextgsim_ngap::codec::ID_ERROR_INDICATION;
    use nextgsim_ngap::procedures::error_indication::decode_error_indication;
    use nextgsim_ngap::procedures::ng_setup::{NgSetupFailureCause, ProtocolCause};

    /// A garbage byte stream that does not APER-decode as an NGAP PDU must
    /// yield a transfer-syntax-error Error Indication with no
    /// CriticalityDiagnostics (the procedure code is unknown).
    #[test]
    fn test_unroutable_undecodable_yields_transfer_syntax_error() {
        let garbage = [0xFFu8, 0xFE, 0xAB, 0xCD, 0x12, 0x34];
        let params = unroutable_error_params(&garbage);
        assert!(matches!(
            params.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::TransferSyntaxError
            ))
        ));
        assert!(params.criticality_diagnostics.is_none());

        // Round-trip: the produced Error Indication encodes and decodes.
        let bytes = encode_error_indication(&params).expect("encode");
        let decoded = decode_error_indication(&bytes).expect("decode");
        assert!(matches!(
            decoded.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::TransferSyntaxError
            ))
        ));
    }

    /// A well-formed-but-unhandled NGAP PDU must yield an
    /// abstract-syntax-error (ignore-and-notify) Error Indication whose
    /// CriticalityDiagnostics carries the offending procedure code.
    #[test]
    fn test_unroutable_decodable_yields_abstract_syntax_error_with_proc_code() {
        // Use an Error Indication PDU (procedure code 9) as a stand-in for a
        // valid-but-unhandled inbound procedure: it decodes cleanly yet has no
        // gNB inbound handler.
        let valid_pdu_bytes =
            encode_error_indication(&ErrorIndicationParams::default()).expect("encode sample pdu");

        let params = unroutable_error_params(&valid_pdu_bytes);
        assert!(matches!(
            params.cause,
            Some(NgSetupFailureCause::Protocol(
                ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify
            ))
        ));
        let diag = params
            .criticality_diagnostics
            .as_ref()
            .expect("criticality diagnostics present");
        assert_eq!(diag.procedure_code, Some(ID_ERROR_INDICATION));
        assert_eq!(
            diag.triggering_message,
            Some(TriggeringMessageValue::InitiatingMessage)
        );

        // Strict APER round-trip of the generated Error Indication.
        let bytes = encode_error_indication(&params).expect("encode");
        let decoded = decode_error_indication(&bytes).expect("decode");
        assert_eq!(
            decoded
                .criticality_diagnostics
                .and_then(|d| d.procedure_code),
            Some(ID_ERROR_INDICATION)
        );
    }

    /// ngap_procedure_code reports the correct triggering-message kind.
    #[test]
    fn test_ngap_procedure_code_initiating() {
        let bytes = encode_error_indication(&ErrorIndicationParams::default()).expect("encode");
        let pdu = decode_ngap_pdu(&bytes).expect("decode");
        let (code, trigger) = ngap_procedure_code(&pdu);
        assert_eq!(code, ID_ERROR_INDICATION);
        assert_eq!(trigger, TriggeringMessageValue::InitiatingMessage);
    }

    // ---- UE Context Modification (issue #40) --------------------------------

    /// Procedure code for UE Context Modification (TS 38.413).
    const ID_UE_CONTEXT_MODIFICATION: u8 = 40;

    /// Seed an NGAP task with a Ready AMF and one UE context; returns the task
    /// (with `sctp_rx` bound to observe replies) and the UE's RAN-UE-NGAP-ID.
    fn seed_task_with_ue() -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
        i64,
    ) {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        let ran = task.create_ue_context(10, 1).expect("ue context");
        (task, sctp_rx, ran)
    }

    fn caps(nea: u16, nia: u16) -> UeSecurityCapabilitiesValue {
        UeSecurityCapabilitiesValue {
            nr_encryption_algorithms: nea,
            nr_integrity_algorithms: nia,
            eutra_encryption_algorithms: None,
            eutra_integrity_algorithms: None,
        }
    }

    fn encode_ucm_request(data: &UeContextModificationRequestData) -> Vec<u8> {
        nextgsim_ngap::procedures::ue_context_modification::encode_ue_context_modification_request(
            data,
        )
        .expect("encode inbound request")
    }

    /// Decode a UE Context Modification RESPONSE and return the echoed
    /// (AMF-UE-NGAP-ID, RAN-UE-NGAP-ID).
    fn response_ids(bytes: &[u8]) -> (u64, u32) {
        use nextgsim_ngap::codec::generated::{
            SuccessfulOutcomeValue, UEContextModificationResponseProtocolIEs_EntryValue as RespIe,
        };
        match decode_ngap_pdu(bytes).expect("decode response") {
            NGAP_PDU::SuccessfulOutcome(o) => match o.value {
                SuccessfulOutcomeValue::Id_UEContextModification(r) => {
                    let mut amf = None;
                    let mut ran = None;
                    for ie in &r.protocol_i_es.0 {
                        match &ie.value {
                            RespIe::Id_AMF_UE_NGAP_ID(id) => amf = Some(id.0),
                            RespIe::Id_RAN_UE_NGAP_ID(id) => ran = Some(id.0),
                            _ => {}
                        }
                    }
                    (amf.expect("amf id"), ran.expect("ran id"))
                }
                other => panic!("expected UEContextModification response, got {other:?}"),
            },
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    /// Decode a UE Context Modification FAILURE and return its Cause.
    fn failure_cause(bytes: &[u8]) -> NgSetupFailureCause {
        use nextgsim_ngap::codec::generated::{
            UEContextModificationFailureProtocolIEs_EntryValue as FailIe, UnsuccessfulOutcomeValue,
        };
        use nextgsim_ngap::procedures::ue_context_release::parse_cause;
        match decode_ngap_pdu(bytes).expect("decode failure") {
            NGAP_PDU::UnsuccessfulOutcome(o) => match o.value {
                UnsuccessfulOutcomeValue::Id_UEContextModification(f) => {
                    for ie in &f.protocol_i_es.0 {
                        if let FailIe::Id_Cause(c) = &ie.value {
                            return parse_cause(c);
                        }
                    }
                    panic!("no Cause IE in failure");
                }
                other => panic!("expected UEContextModification failure, got {other:?}"),
            },
            other => panic!("expected UnsuccessfulOutcome, got {other:?}"),
        }
    }

    /// A UE CONTEXT MODIFICATION REQUEST carrying a new Security Key re-derives
    /// the AS keys and the gNB replies with a RESPONSE (not an Error Indication).
    #[tokio::test]
    async fn test_ue_context_modification_rekeys_and_responds() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();

        // Establish a baseline AS security context.
        task.activate_as_security(10, [0x11u8; 32], &caps(0xC000, 0xC000))
            .await;
        let old = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();

        // Inbound request with a *new* KgNB and updated capabilities.
        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: Some(caps(0xC000, 0xC000)),
            security_key: Some([0x22u8; 32]),
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        // (1) The reply is a UE CONTEXT MODIFICATION RESPONSE, not an Error
        // Indication, and it echoes the request's AMF/RAN IDs.
        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_ne!(stream, 0, "UE-associated reply must not use stream 0");
                let bytes = buffer.data();
                match decode_ngap_pdu(bytes).expect("decode reply") {
                    NGAP_PDU::SuccessfulOutcome(o) => {
                        assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
                    }
                    other => panic!("expected UE Ctx Mod Response, got {other:?}"),
                }
                assert_eq!(response_ids(bytes), (555, ran as u32));
            }
            other => panic!("expected an outbound SCTP SendMessage, got {other:?}"),
        }

        // (2) The AS keys were refreshed from the new KgNB.
        let new = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();
        assert_eq!(new.kgnb, [0x22u8; 32]);
        assert_ne!(new.k_rrc_enc, old.k_rrc_enc, "AS keys must be re-derived");
    }

    /// A UE CONTEXT MODIFICATION REQUEST without a Security Key applies the
    /// replaced UE-AMBR, replies with a RESPONSE, and does not touch AS keys.
    #[tokio::test]
    async fn test_ue_context_modification_updates_ambr_without_rekey() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        task.activate_as_security(10, [0x11u8; 32], &caps(0xC000, 0xC000))
            .await;
        let old = task
            .find_ue_context(10)
            .unwrap()
            .as_security
            .clone()
            .unwrap();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: Some(
                nextgsim_ngap::procedures::initial_context_setup::UeAggregateMaxBitRate {
                    dl: 300_000_000,
                    ul: 100_000_000,
                },
            ),
            ue_security_capabilities: None,
            security_key: None,
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let pdu = decode_ngap_pdu(buffer.data()).expect("decode reply");
                assert!(matches!(pdu, NGAP_PDU::SuccessfulOutcome(_)));
            }
            other => panic!("expected a Response, got {other:?}"),
        }

        let ctx = task.find_ue_context(10).unwrap();
        let ambr = ctx.ue_ambr.clone().expect("UE-AMBR applied");
        assert_eq!(ambr.dl, 300_000_000);
        assert_eq!(ambr.ul, 100_000_000);
        // No re-key requested → AS keys unchanged.
        let new = ctx.as_security.clone().unwrap();
        assert_eq!(new.kgnb, old.kgnb);
        assert_eq!(new.k_rrc_enc, old.k_rrc_enc);
    }

    /// A UE CONTEXT MODIFICATION REQUEST for an unknown UE is answered with a
    /// UE CONTEXT MODIFICATION FAILURE — never an Error Indication.
    #[tokio::test]
    async fn test_ue_context_modification_unknown_ue_replies_failure() {
        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: 9999, // no UE has this RAN-UE-NGAP-ID
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: None,
            security_key: Some([0x22u8; 32]),
        };
        let inbound = encode_ucm_request(&data);

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let bytes = buffer.data();
                match decode_ngap_pdu(bytes).expect("decode reply") {
                    NGAP_PDU::UnsuccessfulOutcome(o) => {
                        assert_eq!(o.procedure_code.0, ID_UE_CONTEXT_MODIFICATION);
                    }
                    // An Error Indication would decode as an InitiatingMessage
                    // (procedure code 9) — explicitly rejected here.
                    other => panic!("expected UE Ctx Mod Failure, got {other:?}"),
                }
                assert_eq!(
                    failure_cause(bytes),
                    NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UnknownLocalUeNgapId)
                );
            }
            other => panic!("expected a Failure, got {other:?}"),
        }
    }

    /// A Security Key with no resolvable UE Security Capabilities is answered
    /// with a FAILURE (Protocol/SemanticError), and — crucially — the procedure
    /// is atomic: a co-carried UE-AMBR is NOT applied and AS keys are untouched.
    #[tokio::test]
    async fn test_ue_context_modification_no_caps_rekey_fails_atomically() {
        // A freshly-created UE context has no stored UE Security Capabilities
        // (they are set at Initial Context Setup, which we deliberately skip).
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        assert!(task
            .find_ue_context(10)
            .unwrap()
            .ue_security_capabilities
            .is_none());

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            // Co-carried UE-AMBR must NOT be committed when the procedure fails.
            ue_aggregate_max_bit_rate: Some(
                nextgsim_ngap::procedures::initial_context_setup::UeAggregateMaxBitRate {
                    dl: 1,
                    ul: 1,
                },
            ),
            ue_security_capabilities: None,
            security_key: Some([0x22u8; 32]),
        };
        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&encode_ucm_request(&data)))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let bytes = buffer.data();
                assert!(matches!(
                    decode_ngap_pdu(bytes).expect("decode reply"),
                    NGAP_PDU::UnsuccessfulOutcome(_)
                ));
                assert_eq!(
                    failure_cause(bytes),
                    NgSetupFailureCause::Protocol(ProtocolCause::SemanticError)
                );
            }
            other => panic!("expected a Failure, got {other:?}"),
        }

        // Atomicity: nothing was applied on the failed procedure.
        let ctx = task.find_ue_context(10).unwrap();
        assert!(
            ctx.ue_ambr.is_none(),
            "UE-AMBR must not be applied on failure"
        );
        assert!(
            ctx.as_security.is_none(),
            "AS keys must be untouched on failure"
        );
    }

    /// Updated UE Security Capabilities carried by the request are stored on the
    /// UE context (TS 38.413 §8.3.4.2).
    #[tokio::test]
    async fn test_ue_context_modification_stores_updated_caps() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();

        let data = UeContextModificationRequestData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32,
            ue_aggregate_max_bit_rate: None,
            ue_security_capabilities: Some(caps(0xA000, 0x8000)),
            security_key: None,
        };
        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&encode_ucm_request(&data)))
            .await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                assert!(matches!(
                    decode_ngap_pdu(buffer.data()).expect("decode reply"),
                    NGAP_PDU::SuccessfulOutcome(_)
                ));
            }
            other => panic!("expected a Response, got {other:?}"),
        }

        let stored = task
            .find_ue_context(10)
            .unwrap()
            .ue_security_capabilities
            .clone()
            .expect("UE Security Capabilities applied");
        // Assert the NR fields specifically (parse always fills the E-UTRA
        // fields, so full-struct equality would be brittle).
        assert_eq!(stored.nr_encryption_algorithms, 0xA000);
        assert_eq!(stored.nr_integrity_algorithms, 0x8000);
    }

    // ---- Error Indication + NAS Non Delivery (issue #42) --------------------

    /// A received ERROR INDICATION referencing a known UE locally releases that
    /// UE's NG connection and does NOT echo another Error Indication back to the
    /// AMF (TS 38.413 §8.7.5 / §10.6).
    #[tokio::test]
    async fn test_error_indication_releases_ue_and_does_not_echo() {
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        assert!(task.find_ue_context(10).is_some());

        let params = ErrorIndicationParams {
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: Some(ran as u32),
            cause: Some(NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::UnknownLocalUeNgapId,
            )),
            criticality_diagnostics: None,
        };
        let inbound = encode_error_indication(&params).expect("encode error indication");

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        // The UE context was locally released ...
        assert!(
            task.find_ue_context(10).is_none(),
            "UE context must be locally released on Error Indication"
        );
        // ... and NO Error Indication (or any NGAP PDU) was echoed to the AMF.
        assert!(
            sctp_rx.try_recv().is_err(),
            "must not reply to a received Error Indication (ping-pong)"
        );
    }

    /// A received ERROR INDICATION whose AP IDs match no UE is processed with no
    /// local release and, crucially, no reply.
    #[tokio::test]
    async fn test_error_indication_unknown_ue_is_silently_processed() {
        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();

        let params = ErrorIndicationParams {
            amf_ue_ngap_id: None,
            ran_ue_ngap_id: Some(4242), // matches no UE
            cause: Some(NgSetupFailureCause::Protocol(ProtocolCause::SemanticError)),
            criticality_diagnostics: None,
        };
        let inbound = encode_error_indication(&params).expect("encode");

        task.handle_ngap_pdu(1, 2, OctetString::from_slice(&inbound))
            .await;

        assert!(task.find_ue_context(10).is_some(), "unrelated UE untouched");
        assert!(sctp_rx.try_recv().is_err(), "no reply to Error Indication");
    }

    /// When RRC cannot accept the NAS-PDU for delivery to the UE (send failure),
    /// the gNB still reports a NAS NON DELIVERY INDICATION (TS 38.413 §8.6.4).
    #[tokio::test]
    async fn test_downlink_nas_rrc_failure_emits_non_delivery() {
        // `seed_task_with_ue` drops the RRC receiver, so `rrc_tx.send` fails —
        // exercising the RRC-send-failure non-delivery path for a *known* UE.
        let (mut task, mut sctp_rx, ran) = seed_task_with_ue();
        let nas_pdu = vec![0x7e, 0x00, 0x55];
        let dl_nas = DownlinkNasTransportData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: ran as u32, // matches the seeded UE
            nas_pdu: nas_pdu.clone(),
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            allowed_nssai: None,
        };

        task.handle_downlink_nas_transport(1, 3, dl_nas).await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                assert_eq!(nas_non_delivery_pdu(buffer.data()), nas_pdu);
            }
            other => panic!("expected NAS Non Delivery on RRC failure, got {other:?}"),
        }
    }

    /// A DOWNLINK NAS TRANSPORT for an unknown UE is reported to the AMF as a NAS
    /// NON DELIVERY INDICATION carrying the original NAS-PDU (TS 38.413 §8.6.4).
    #[tokio::test]
    async fn test_downlink_nas_unknown_ue_emits_non_delivery() {
        use nextgsim_ngap::procedures::nas_non_delivery_indication::is_nas_non_delivery_indication;

        let (mut task, mut sctp_rx, _ran) = seed_task_with_ue();
        let nas_pdu = vec![0x7e, 0x00, 0x42, 0x11, 0x22];
        let dl_nas = DownlinkNasTransportData {
            amf_ue_ngap_id: 555,
            ran_ue_ngap_id: 4242, // no UE has this RAN-UE-NGAP-ID
            nas_pdu: nas_pdu.clone(),
            old_amf: None,
            ran_paging_priority: None,
            index_to_rfsp: None,
            allowed_nssai: None,
        };

        task.handle_downlink_nas_transport(1, 3, dl_nas).await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_ne!(
                    stream, 0,
                    "UE-associated non-delivery must not use stream 0"
                );
                let pdu = decode_ngap_pdu(buffer.data()).expect("decode reply");
                assert!(
                    is_nas_non_delivery_indication(&pdu),
                    "expected NAS Non Delivery Indication, got {pdu:?}"
                );
                // The original NAS-PDU is echoed back to the AMF.
                let echoed = nas_non_delivery_pdu(buffer.data());
                assert_eq!(echoed, nas_pdu);
            }
            other => panic!("expected NAS Non Delivery Indication, got {other:?}"),
        }
    }

    /// Extract the NAS-PDU IE from a NAS Non Delivery Indication's bytes.
    fn nas_non_delivery_pdu(bytes: &[u8]) -> Vec<u8> {
        use nextgsim_ngap::codec::generated::{
            InitiatingMessageValue, NASNonDeliveryIndicationProtocolIEs_EntryValue as Ie,
        };
        match decode_ngap_pdu(bytes).expect("decode") {
            NGAP_PDU::InitiatingMessage(m) => match m.value {
                InitiatingMessageValue::Id_NASNonDeliveryIndication(x) => x
                    .protocol_i_es
                    .0
                    .iter()
                    .find_map(|ie| match &ie.value {
                        Ie::Id_NAS_PDU(p) => Some(p.0.clone()),
                        _ => None,
                    })
                    .expect("NAS-PDU IE present"),
                other => panic!("wrong variant: {other:?}"),
            },
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    // ---- Overload Start / Stop (issue #41) ----------------------------------

    /// OVERLOAD START drives the AMF context into `Overloaded` (which excludes it
    /// from `select_amf`) and records the traffic-load-reduction; OVERLOAD STOP
    /// clears it. Neither is echoed with an Error Indication (TS 38.413 §8.7.7/8).
    #[tokio::test]
    async fn test_overload_start_stop_marks_amf_without_echo() {
        use nextgsim_ngap::procedures::overload::{encode_overload_start, encode_overload_stop};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        // OVERLOAD START → Overloaded + reduction stored, no reply.
        let start = encode_overload_start(Some(75)).expect("encode start");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&start))
            .await;
        {
            let ctx = task.find_amf_context_mut(1).unwrap();
            assert_eq!(ctx.state, AmfState::Overloaded);
            assert_eq!(ctx.traffic_load_reduction, Some(75));
        }
        assert!(
            sctp_rx.try_recv().is_err(),
            "Overload Start must not be answered with an Error Indication"
        );

        // OVERLOAD STOP → Ready, reduction cleared, no reply.
        let stop = encode_overload_stop().expect("encode stop");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&stop))
            .await;
        {
            let ctx = task.find_amf_context_mut(1).unwrap();
            assert_eq!(ctx.state, AmfState::Ready);
            assert_eq!(ctx.traffic_load_reduction, None);
        }
        assert!(
            sctp_rx.try_recv().is_err(),
            "Overload Stop must not be answered with an Error Indication"
        );
    }

    /// AMF STATUS INDICATION marking a served GUAMI unavailable excludes that AMF
    /// from `select_amf`, with no Error Indication echo (TS 38.413 §8.7.6).
    #[tokio::test]
    async fn test_amf_status_indication_excludes_amf_from_selection() {
        use nextgsim_ngap::procedures::amf_status_indication::{
            encode_amf_status_indication, AmfStatusIndicationParams, UnavailableGuamiItem,
        };
        use nextgsim_ngap::procedures::initial_context_setup::GuamiValue;
        use nextgsim_ngap::procedures::{Guami, ServedGuamiItem};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
            ctx.relative_capacity = 100;
            ctx.served_guami_list = vec![ServedGuamiItem {
                guami: Guami {
                    plmn_identity: [0x00, 0xf1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 3,
                },
                backup_amf_name: None,
            }];
        }
        // Ready + available → this AMF is selectable.
        assert_eq!(task.select_amf(&[0x00, 0xf1, 0x10], &[]), Some(1));

        // AMF Status Indication marks that GUAMI unavailable.
        let params = AmfStatusIndicationParams {
            unavailable_guami_list: vec![UnavailableGuamiItem {
                guami: GuamiValue {
                    plmn_identity: [0x00, 0xf1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 3,
                },
                backup_amf_name: Some("backup-amf".into()),
            }],
        };
        let inbound = encode_amf_status_indication(&params).expect("encode");
        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&inbound))
            .await;

        // The AMF is excluded from selection, and nothing was echoed.
        assert!(task.find_amf_context_mut(1).unwrap().unavailable);
        assert_eq!(
            task.select_amf(&[0x00, 0xf1, 0x10], &[]),
            None,
            "an AMF with an unavailable GUAMI must not be selected"
        );
        assert!(
            sctp_rx.try_recv().is_err(),
            "AMF Status Indication must not be answered with an Error Indication"
        );
    }

    /// AMF CONFIGURATION UPDATE applies the served-GUAMI and PLMN-support lists
    /// (not just their counts) to the stored AMF context and is acknowledged
    /// (TS 38.413 §8.7.3).
    #[tokio::test]
    async fn test_amf_configuration_update_applies_guami_and_plmn_lists() {
        use nextgsim_ngap::procedures::ng_reset::AmfConfigurationUpdateData;
        use nextgsim_ngap::procedures::{Guami, PlmnSupportItem, SNssai, ServedGuamiItem};

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        let update = AmfConfigurationUpdateData {
            amf_name: Some("amf-new".into()),
            served_guami_list: vec![ServedGuamiItem {
                guami: Guami {
                    plmn_identity: [0x00, 0xf1, 0x10],
                    amf_region_id: 1,
                    amf_set_id: 2,
                    amf_pointer: 3,
                },
                backup_amf_name: None,
            }],
            relative_amf_capacity: Some(200),
            plmn_support_list: vec![PlmnSupportItem {
                plmn_identity: [0x00, 0xf1, 0x10],
                slice_support_list: vec![SNssai { sst: 1, sd: None }],
            }],
            // No TNLA change: this test is about the GUAMI/PLMN/capacity half.
            tnla_to_add: Vec::new(),
            tnla_to_remove: Vec::new(),
            tnla_to_update: Vec::new(),
        };
        task.handle_amf_configuration_update(1, 0, update).await;

        let ctx = task.find_amf_context_mut(1).unwrap();
        assert_eq!(ctx.amf_name.as_deref(), Some("amf-new"));
        assert_eq!(ctx.relative_capacity, 200);
        assert_eq!(ctx.served_guami_list.len(), 1);
        assert_eq!(ctx.served_guami_list[0].guami.amf_set_id, 2);
        assert_eq!(ctx.plmn_support_list.len(), 1);
        assert_eq!(ctx.plmn_support_list[0].slice_support_list[0].sst, 1);

        // An AMF Configuration Update Acknowledge was sent.
        assert!(
            sctp_rx.try_recv().is_ok(),
            "must reply with an AMF Configuration Update Acknowledge"
        );
    }

    /// The gNB can initiate a RAN CONFIGURATION UPDATE (TS 38.413 §8.7.2): the
    /// send path emits a well-formed RAN Configuration Update PDU on stream 0.
    #[tokio::test]
    async fn test_send_ran_configuration_update_emits_pdu() {
        use nextgsim_ngap::procedures::ran_configuration_update::is_ran_configuration_update;

        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        task.send_ran_configuration_update(1).await;

        match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_eq!(
                    stream, 0,
                    "RAN Config Update is non-UE-associated (stream 0)"
                );
                let pdu = decode_ngap_pdu(buffer.data()).expect("decode");
                assert!(
                    is_ran_configuration_update(&pdu),
                    "expected a RAN Configuration Update, got {pdu:?}"
                );
            }
            other => panic!("expected a RAN Configuration Update PDU, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // #42 criteria 4-6: NGAP guard timers
    // (TS 38.413 §8.3.3.4 TNGRELOCoverall, §8.4.1.2 TNGRELOCprep,
    //  §8.7.1.3 NG Setup Failure Time to Wait)
    //
    // Every expiry is driven by passing an explicit `now` to
    // `process_expired_guard_timers`, so these tests neither sleep nor depend on
    // how promptly the host schedules a timer -- the "controllable clock"
    // criterion 7(c) asks for.
    // ------------------------------------------------------------------

    use crate::ngap::timers::GuardTimer;
    use crate::ngap::UeState;
    use nextgsim_ngap::procedures::ng_setup::TimeToWaitValue;
    use std::time::{Duration, Instant};

    /// A task with one ready AMF and one UE context that has an AMF UE NGAP ID, i.e.
    /// a UE whose UE-associated signalling connection is usable.
    fn task_with_ue(
        ue_id: i32,
    ) -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::SctpMessage>>,
    ) {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(ue_id, 1).expect("ue context");
        if let Some(ctx) = task.find_ue_context_mut(ue_id) {
            ctx.amf_ue_ngap_id = Some(4242);
        }
        (task, sctp_rx)
    }

    // ========================================================================
    // Inter-gNB handover (issue #39)
    // ========================================================================

    /// A task with every receiver alive, including the SCTP one the handover tests read.
    ///
    /// Separate from `task_with_live_receivers` rather than replacing it: that helper's
    /// four-tuple is destructured by the issue #32 tests, and widening it there would be
    /// churn for no gain.
    #[allow(clippy::type_complexity)]
    fn task_with_sctp(
        ue_id: i32,
    ) -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::RrcMessage>>,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::GtpMessage>>,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::SctpMessage>>,
    ) {
        let (task_base, _app_rx, _ngap_rx, rrc_rx, gtp_rx, _rls_rx, sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(ue_id, 1).expect("ue context");
        if let Some(ctx) = task.find_ue_context_mut(ue_id) {
            ctx.amf_ue_ngap_id = Some(7777);
        }
        (task, rrc_rx, gtp_rx, sctp_rx)
    }

    /// Every NGAP PDU the task put on the SCTP channel.
    fn drain_sctp(
        rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::SctpMessage>>,
    ) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let TaskMessage::Message(SctpMessage::SendMessage { buffer, .. }) = msg {
                out.push(buffer.data().to_vec());
            }
        }
        out
    }

    /// A HANDOVER REQUEST asking this target to admit `psis`, with a security context.
    fn handover_request(psis: &[u8], security: Option<HandoverSecurityContext>) -> NGAP_PDU {
        use nextgsim_ngap::procedures::handover::{
            build_handover_request, HandoverRequestParams, HandoverRequestSetupItem,
        };
        use nextgsim_ngap::procedures::pdu_session_resource::SnssaiValue;
        use nextgsim_ngap::procedures::transfer::{
            encode_setup_request_transfer, QosFlowSetupInfo, SetupRequestTransferData,
        };
        let pdu_sessions = psis
            .iter()
            .map(|&psi| HandoverRequestSetupItem {
                pdu_session_id: psi,
                s_nssai: SnssaiValue { sst: 1, sd: None },
                transfer: encode_setup_request_transfer(&SetupRequestTransferData {
                    ambr_dl: Some(1_000_000),
                    ambr_ul: Some(1_000_000),
                    ul_tunnel: GtpTunnelInfo {
                        address: "10.45.0.1".parse().unwrap(),
                        // A per-session UPF TEID, so a handler that admitted one session
                        // and reused its tunnel for the rest would be visible.
                        teid: 0x1000 + u32::from(psi),
                    },
                    pdu_session_type: 0,
                    qos_flows: vec![QosFlowSetupInfo {
                        qfi: psi,
                        five_qi: Some(9),
                        arp_priority_level: 8,
                    }],
                    security_indication: None,
                })
                .expect("encode the inner transfer"),
            })
            .collect();
        build_handover_request(&HandoverRequestParams {
            amf_ue_ngap_id: 7777,
            handover_type: HandoverTypeValue::Intra5gs,
            cause: HandoverCause::RadioNetwork(RadioNetworkCause::HandoverDesirableForRadioReason),
            source_to_target_transparent_container: vec![0x11, 0x22],
            pdu_sessions,
            security_context: security,
        })
        .expect("build the handover request")
    }

    /// #39, criteria 1 and 2: the target admits **each** requested session, allocating a
    /// real DL F-TEID and a GTP-U tunnel per session, and answers with a decodable
    /// Handover Request Acknowledge Transfer — not the hardcoded session 1 with
    /// `vec![0x00]`.
    #[tokio::test]
    async fn the_target_admits_every_requested_session_with_a_real_ack_transfer() {
        use nextgsim_ngap::procedures::handover::parse_handover_request_acknowledge;
        use nextgsim_ngap::procedures::transfer::decode_handover_ack_transfer;

        let (mut task, _rrc_rx, mut gtp_rx, mut sctp_rx) = task_with_sctp(1);
        // Two sessions with different ids, because a handler that admitted the first
        // twice -- or a hardcoded 1 -- would satisfy a single-session test.
        let pdu = handover_request(
            &[5, 6],
            Some(HandoverSecurityContext {
                next_hop_chaining_count: 1,
                next_hop_nh: [0xA5; 32],
            }),
        );
        let ho_req = nextgsim_ngap::procedures::handover::parse_handover_request(&pdu)
            .expect("parse the request");

        task.handle_handover_request(1, 8, ho_req).await;

        // The Acknowledge admits both, each with a real transfer naming its own tunnel.
        let sent = drain_sctp(&mut sctp_rx);
        let ack = sent
            .iter()
            .find_map(|bytes| {
                nextgsim_ngap::codec::decode_ngap_pdu(bytes)
                    .ok()
                    .and_then(|pdu| parse_handover_request_acknowledge(&pdu).ok())
            })
            .expect("a Handover Request Acknowledge must be sent");
        assert_eq!(ack.pdu_session_resource_admitted_list.len(), 2);
        let mut teids = Vec::new();
        for item in &ack.pdu_session_resource_admitted_list {
            assert!(
                [5u8, 6].contains(&item.pdu_session_id),
                "admitted {} which was never requested",
                item.pdu_session_id
            );
            let data = decode_handover_ack_transfer(&item.handover_request_ack_transfer)
                .unwrap_or_else(|e| {
                    panic!(
                        "session {}'s ack transfer must be a real transfer, not vec![0x00]: {e}",
                        item.pdu_session_id
                    )
                });
            teids.push(data.dl_tunnel.teid);
        }
        assert_eq!(teids.len(), 2);
        assert_ne!(
            teids[0], teids[1],
            "each admitted session needs its OWN DL F-TEID, or the UPF sends both \
             sessions' downlink to one tunnel"
        );

        // And GTP was told to create a session for each -- the allocation the old handler
        // performed for none of them.
        let mut created: Vec<i32> = Vec::new();
        while let Ok(msg) = gtp_rx.try_recv() {
            if let TaskMessage::Message(GtpMessage::SessionCreate { resource, .. }) = msg {
                created.push(resource.psi);
            }
        }
        created.sort_unstable();
        assert_eq!(
            created,
            vec![5, 6],
            "a handover-in must allocate a data plane, or the UE arrives with none"
        );
    }

    // ------------------------------------------------------------------
    // #169: the handover-arrival arming, over the channel
    // ------------------------------------------------------------------

    /// The `ue_id` of the UE a handover-in created, found by the AMF id the request carried.
    ///
    /// Not the pre-existing fixture UE: `handle_handover_request` allocates ids from 1000 up.
    fn handover_ue_id(task: &NgapTask) -> i32 {
        task.ue_contexts
            .values()
            .find(|ctx| ctx.amf_ue_ngap_id == Some(7777) && ctx.ue_id >= 1000)
            .map(|ctx| ctx.ue_id)
            .expect("the handover-in must have created a UE context")
    }

    /// Every `ExpectHandoverArrival` on the RRC channel.
    fn armed_arrivals(
        rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::RrcMessage>>,
    ) -> Vec<i32> {
        let mut out = Vec::new();
        while let Ok(msg) = rx.try_recv() {
            if let TaskMessage::Message(RrcMessage::ExpectHandoverArrival { ue_id }) = msg {
                out.push(ue_id);
            }
        }
        out
    }

    /// #169: the arming reaches the RRC task **over the channel**, for the right UE.
    ///
    /// #156 gave the target an arrival report and every test of it drove the RRC half by
    /// calling the handler. Nothing asserted the NGAP task sends anything at all, so
    /// deleting the send left the suite green — the same class as #151's R12/R13 and #158's
    /// R4, both recorded in LEARNINGS: the helper is tested and the wiring is not.
    #[tokio::test]
    async fn the_target_arms_arrival_detection_over_the_channel() {
        use nextgsim_ngap::procedures::handover::parse_handover_request_acknowledge;

        let (mut task, mut rrc_rx, _gtp_rx, mut sctp_rx) = task_with_sctp(1);
        let pdu = handover_request(&[5], None);
        let ho_req =
            nextgsim_ngap::procedures::handover::parse_handover_request(&pdu).expect("parse");

        task.handle_handover_request(1, 8, ho_req).await;

        // POSITIVE CONTROL: the acknowledge really went out. Without it, "arrival was
        // armed" says nothing about the order it was armed in.
        let sent = drain_sctp(&mut sctp_rx);
        assert!(
            sent.iter()
                .any(|bytes| nextgsim_ngap::codec::decode_ngap_pdu(bytes)
                    .ok()
                    .and_then(|pdu| parse_handover_request_acknowledge(&pdu).ok())
                    .is_some()),
            "precondition: the Handover Request Acknowledge must have been sent"
        );

        assert_eq!(
            armed_arrivals(&mut rrc_rx),
            vec![handover_ue_id(&task)],
            "the NGAP task must arm arrival detection for the handover UE on the RRC \
             channel; nothing asserted this send existed before #169"
        );
    }

    /// #169: an acknowledge that could not be DELIVERED arms nothing.
    ///
    /// This is the half that was broken rather than merely untested. The comment at the
    /// arming site says "an admission the AMF was never told about must not have the target
    /// report an arrival for it", but the guard covered only the ENCODE, and
    /// `send_ngap_ue_associated` swallowed its own send error — so a target whose
    /// acknowledge never left still armed, and would later report a HANDOVER NOTIFY for a
    /// handover the AMF has no record of.
    ///
    /// The SCTP receiver is dropped to make the send fail, which is the only way this
    /// branch is reachable from outside: `build_handover_request_acknowledge` cannot fail
    /// (it returns `Ok` unconditionally) and both transparent containers are unconstrained
    /// OCTET STRINGs, so the `Err(e)` encode arm cannot be provoked through the handler.
    /// That is stated here rather than asserted around, because a test that claimed to
    /// cover the encode arm would be claiming something untrue.
    #[tokio::test]
    async fn an_acknowledge_that_could_not_be_delivered_arms_nothing() {
        let (mut task, mut rrc_rx, _gtp_rx, sctp_rx) = task_with_sctp(1);
        // The SCTP task is gone: `sctp_tx.send` now fails, so the AMF is never told.
        drop(sctp_rx);

        let pdu = handover_request(&[5], None);
        let ho_req =
            nextgsim_ngap::procedures::handover::parse_handover_request(&pdu).expect("parse");

        task.handle_handover_request(1, 8, ho_req).await;

        // POSITIVE CONTROL first: the handler really ran the admission, so an empty RRC
        // channel means "deliberately not armed" and not "never got there". LEARNINGS: a
        // negative assertion is satisfied by every path that never arrives.
        let ue_id = handover_ue_id(&task);
        assert!(
            task.ue_contexts
                .get(&ue_id)
                .is_some_and(|ctx| ctx.amf_ue_ngap_id == Some(7777)),
            "precondition: the handover-in must have been processed as far as the acknowledge"
        );

        assert!(
            armed_arrivals(&mut rrc_rx).is_empty(),
            "an acknowledge the AMF never received must not arm arrival detection: the \
             target would report a HANDOVER NOTIFY for a handover nobody asked it to admit"
        );
    }

    /// #39: the target derives its `KgNB*` from the AMF's `{NH, NCC}` — vertically — and
    /// the resulting AS keys differ from the source's.
    #[tokio::test]
    async fn the_target_re_keys_vertically_from_the_amfs_next_hop() {
        use nextgsim_crypto::kdf::derive_kgnb_star;

        let (mut task, _rrc_rx, _gtp_rx, _sctp_rx) = task_with_sctp(1);
        let nh = [0xA5u8; 32];
        let pdu = handover_request(
            &[5],
            Some(HandoverSecurityContext {
                next_hop_chaining_count: 3,
                next_hop_nh: nh,
            }),
        );
        let ho_req = nextgsim_ngap::procedures::handover::parse_handover_request(&pdu).unwrap();
        task.handle_handover_request(1, 8, ho_req).await;

        // The handover UE is the one with the AMF id from the request.
        // The handover UE, not the pre-existing one: `handle_handover_request` allocates
        // ids from 1000 up.
        let ctx = task
            .ue_contexts
            .values()
            .find(|c| c.ue_id >= 1000)
            .expect("the handover UE context must exist");
        let sec = ctx
            .as_security
            .as_ref()
            .expect("the target must hold an AS security context after re-keying");

        let expected_kgnb = derive_kgnb_star(
            &nh,
            phys_cell_id_from_nci(test_config().nci),
            test_config().dl_arfcn,
        );
        assert_eq!(
            sec.kgnb, expected_kgnb,
            "KgNB* must be derived from the AMF's NH, bound to this cell's PCI and \
             ARFCN-DL (TS 33.501 Annex A.11)"
        );
        assert_ne!(
            sec.kgnb, nh,
            "and it must not be the NH used unchanged: that is not a derivation"
        );
        // The four AS keys hang off it, which is what makes the adoption real.
        assert_eq!(
            sec.k_rrc_int,
            AsSecurityContext::from_kgnb(expected_kgnb, sec.ciphering_alg_id, sec.integrity_alg_id)
                .k_rrc_int
        );
    }

    /// #39, criterion 3: `NgapMessage::InitiateHandover` drives a HANDOVER REQUIRED whose
    /// source-to-target container is a real one, carrying the UE's capabilities and the
    /// target cell — not `vec![0x00]`.
    #[tokio::test]
    async fn initiate_handover_sends_a_handover_required_with_a_real_container() {
        use nextgsim_ngap::procedures::handover::{
            decode_source_to_target_container, parse_handover_required,
        };
        use nextgsim_rrc::procedures::handover_preparation::decode_handover_preparation_information;

        let (mut task, _rrc_rx, _gtp_rx, mut sctp_rx) = task_with_sctp(3);
        // A PDU session, because `PDUSessionResourceListHORqd` is `SIZE(1..)` and a UE
        // with none has no user plane to move.
        if let Some(ctx) = task.find_ue_context_mut(3) {
            ctx.add_pdu_session(NgapPduSession {
                psi: 5,
                qfi: Some(9),
                uplink_teid: 1,
                downlink_teid: 2,
                upf_address: "10.45.0.1".parse().unwrap(),
                up_security_policy: UpSecurityPolicy::locally_configured_default(),
                up_security: DrbSecurityDecision::default(),
            });
        }
        let capability = vec![0xCA, 0xFE, 0xBA, 0xBE];
        assert!(
            task.initiate_handover(HandoverInitiation {
                ue_id: 3,
                target_gnb_id: 0x99,
                target_tac: 7,
                target_cell_identity: 0x0001_2345_6789,
                ue_nr_capability: Some(capability.clone()),
                time_in_source_cell_s: 42,
            })
            .await,
            "a connected UE with an AMF id must produce a HANDOVER REQUIRED"
        );

        let sent = drain_sctp(&mut sctp_rx);
        let required = sent
            .iter()
            .find_map(|bytes| {
                nextgsim_ngap::codec::decode_ngap_pdu(bytes)
                    .ok()
                    .and_then(|pdu| parse_handover_required(&pdu).ok())
            })
            .expect("a HANDOVER REQUIRED must be sent");

        let container =
            decode_source_to_target_container(&required.source_to_target_transparent_container)
                .expect("the container must be a real one, not vec![0x00]");
        assert_eq!(
            container.target_cell.nr_cell_identity, 0x0001_2345_6789,
            "the target must be told which cell the source meant"
        );
        assert_eq!(
            container.visited_cells.first().map(|c| c.nr_cell_identity),
            Some(test_config().nci & 0xF_FFFF_FFFF),
            "and where the UE has been"
        );
        assert_eq!(
            decode_handover_preparation_information(&container.rrc_container)
                .expect("the RRCContainer must be a real HandoverPreparationInformation")
                .nr_capability,
            Some(capability),
            "the target needs the UE's capabilities before it can configure anything"
        );
    }

    /// A UE with **no PDU session** is refused, because
    /// `PDUSessionResourceListHORqd` is `SIZE(1..maxnoofPDUSessions)` and an empty list is
    /// not encodable.
    ///
    /// Found by a test failing for the wrong reason: the encoder failed silently and
    /// `initiate_handover` reported success anyway — the no-op-that-logs-success this
    /// issue is about. The old code went further and invented a "default entry" for
    /// session 1.
    #[tokio::test]
    async fn initiate_handover_refuses_a_ue_with_no_pdu_session() {
        let (mut task, _rrc_rx, _gtp_rx, mut sctp_rx) = task_with_sctp(4);
        assert!(
            task.find_ue_context(4)
                .is_some_and(|c| c.pdu_session_count() == 0),
            "precondition: the UE has no PDU session"
        );
        assert!(
            !task
                .initiate_handover(HandoverInitiation {
                    ue_id: 4,
                    target_gnb_id: 1,
                    target_tac: 1,
                    target_cell_identity: 1,
                    ue_nr_capability: None,
                    time_in_source_cell_s: 0,
                })
                .await,
            "a UE with no session has nothing to hand over"
        );
        assert!(
            drain_sctp(&mut sctp_rx).is_empty(),
            "and nothing must go on the wire, least of all an unencodable message"
        );
    }

    /// #39, criterion 7: the Path Switch Request Acknowledge adopts the fresh `{NH, NCC}`
    /// and re-keys from it — which the doc comment used to claim while the body did
    /// nothing.
    ///
    /// A revert round removing the adoption stayed green against the issue #32 policy
    /// test, because that one is about the user-plane policy and not about keys.
    #[tokio::test]
    async fn the_path_switch_acknowledge_re_keys_from_the_fresh_next_hop() {
        use nextgsim_crypto::kdf::derive_kgnb_star;
        use nextgsim_ngap::procedures::path_switch::SwitchedSessionItem;

        let (mut task, _rrc_rx, _gtp_rx, _sctp_rx) = task_with_sctp(1);
        let ran_ue_ngap_id = task
            .find_ue_context(1)
            .map(|c| c.ran_ue_ngap_id as u32)
            .expect("ran id");
        let nh = [0x77u8; 32];

        task.handle_path_switch_request_acknowledge(
            1,
            8,
            PathSwitchRequestAcknowledgeData {
                amf_ue_ngap_id: 4242,
                ran_ue_ngap_id,
                next_hop_chaining_count: 4,
                next_hop_nh: nh,
                switched_sessions: vec![SwitchedSessionItem {
                    pdu_session_id: 1,
                    ul_tunnel: None,
                    security_indication: None,
                }],
            },
        )
        .await;

        let expected = derive_kgnb_star(
            &nh,
            phys_cell_id_from_nci(test_config().nci),
            test_config().dl_arfcn,
        );
        assert_eq!(
            task.find_ue_context(1)
                .and_then(|c| c.as_security.as_ref())
                .map(|s| s.kgnb),
            Some(expected),
            "a path switch must adopt the AMF's fresh NH and derive KgNB* from it \
             (TS 33.501 §6.9.2.3.1) -- the same derivation the HANDOVER REQUEST path uses"
        );
    }

    /// A UE with no AMF UE NGAP ID cannot be handed over, and that is reported rather
    /// than logged as a handover that started.
    #[tokio::test]
    async fn initiate_handover_reports_a_ue_it_cannot_hand_over() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        task.create_ue_context(5, 1).expect("ue context");
        // No `amf_ue_ngap_id`.
        assert!(
            !task
                .initiate_handover(HandoverInitiation {
                    ue_id: 5,
                    target_gnb_id: 1,
                    target_tac: 1,
                    target_cell_identity: 1,
                    ue_nr_capability: None,
                    time_in_source_cell_s: 0,
                })
                .await,
            "a UE the AMF does not know cannot be handed over"
        );
        assert!(
            !task
                .initiate_handover(HandoverInitiation {
                    ue_id: 999,
                    target_gnb_id: 1,
                    target_tac: 1,
                    target_cell_identity: 1,
                    ue_nr_capability: None,
                    time_in_source_cell_s: 0,
                })
                .await,
            "nor an unknown one"
        );
    }

    // ========================================================================
    // User-plane security wiring (issue #32)
    // ========================================================================

    /// An NGAP task whose GTP, RRC and RLS receivers stay alive.
    ///
    /// `task_with_ue` drops them, which makes every `send` fail — and
    /// `setup_one_pdu_session` reports a send failure as
    /// `RadioResourcesNotAvailable`, so a security test built on it would see the
    /// wrong refusal and pass for the wrong reason. Found exactly that way.
    #[allow(clippy::type_complexity)]
    fn task_with_live_receivers(
        ue_id: i32,
    ) -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::RrcMessage>>,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::GtpMessage>>,
        tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::RlsMessage>>,
    ) {
        let (task_base, _app_rx, _ngap_rx, rrc_rx, gtp_rx, rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(ue_id, 1).expect("ue context");
        if let Some(ctx) = task.find_ue_context_mut(ue_id) {
            ctx.amf_ue_ngap_id = Some(4242);
        }
        (task, rrc_rx, gtp_rx, rls_rx)
    }

    use nextgsim_ngap::procedures::transfer::MaxIntegrityProtectedDataRate;

    /// A Setup Request item whose transfer carries `policy`.
    fn setup_item_with_policy(
        psi: u8,
        policy: Option<nextgsim_ngap::procedures::transfer::UpSecurityPolicy>,
    ) -> nextgsim_ngap::procedures::pdu_session_resource::PduSessionResourceSetupItem {
        use nextgsim_ngap::procedures::pdu_session_resource::{
            PduSessionResourceSetupItem, SnssaiValue,
        };
        use nextgsim_ngap::procedures::transfer::{
            encode_setup_request_transfer, QosFlowSetupInfo, SetupRequestTransferData,
        };
        let transfer = encode_setup_request_transfer(&SetupRequestTransferData {
            ambr_dl: Some(1_000_000),
            ambr_ul: Some(1_000_000),
            ul_tunnel: GtpTunnelInfo {
                address: "10.45.0.1".parse().unwrap(),
                teid: 0x1234,
            },
            pdu_session_type: 0,
            qos_flows: vec![QosFlowSetupInfo {
                qfi: 1,
                five_qi: Some(9),
                arp_priority_level: 8,
            }],
            security_indication: policy,
        })
        .expect("encode the setup transfer");
        PduSessionResourceSetupItem {
            pdu_session_id: psi,
            nas_pdu: None,
            s_nssai: SnssaiValue { sst: 1, sd: None },
            transfer,
        }
    }

    /// Give a UE the AS security context an Initial Context Setup would install.
    fn key_ue(task: &mut NgapTask, ue_id: i32, ciphering_alg_id: u8, integrity_alg_id: u8) {
        let ctx = task.find_ue_context_mut(ue_id).expect("ue context");
        ctx.as_security = Some(AsSecurityContext {
            kgnb: [0x33; 32],
            k_rrc_enc: [0x01; 16],
            k_rrc_int: [0x02; 16],
            k_up_enc: [0x03; 16],
            k_up_int: [0x04; 16],
            ciphering_alg_id,
            integrity_alg_id,
        });
    }

    fn policy(
        integrity: nextgsim_ngap::procedures::transfer::UpProtectionPolicy,
        confidentiality: nextgsim_ngap::procedures::transfer::UpProtectionPolicy,
    ) -> nextgsim_ngap::procedures::transfer::UpSecurityPolicy {
        UpSecurityPolicy {
            integrity,
            confidentiality,
            max_integrity_protected_data_rate: None,
        }
    }

    /// #32, criterion 4: a `required` policy the gNB cannot satisfy fails the PDU
    /// session, with the cause TS 38.413 §9.3.1.2 gives for it — and it fails
    /// *before* any state is created.
    #[tokio::test]
    async fn a_required_up_policy_the_gnb_cannot_satisfy_refuses_the_session() {
        use nextgsim_ngap::procedures::transfer::{
            decode_setup_unsuccessful_transfer, UpProtectionPolicy,
        };
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(21);
        // NIA0/NEA0 selected: neither protection is possible whatever the build.
        key_ue(&mut task, 21, 0, 0);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();

        let item = setup_item_with_policy(
            1,
            Some(policy(
                UpProtectionPolicy::Required,
                UpProtectionPolicy::Preferred,
            )),
        );
        let err = task
            .setup_one_pdu_session(21, &item, gnb_ip)
            .await
            .expect_err("a required integrity policy with NIA0 must fail the session");
        assert_eq!(err.pdu_session_id, 1);
        assert_eq!(
            decode_setup_unsuccessful_transfer(&err.transfer).expect("a real cause"),
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UpIntegrityProtectionNotPossible),
            "the SMF must be told WHICH protection was impossible, or it cannot relax \
             the right half of its policy"
        );
        assert!(
            task.find_ue_context(21)
                .is_some_and(|c| c.pdu_session_count() == 0),
            "a refused session must leave no PDU session behind"
        );

        // The confidentiality half yields the other cause.
        let item = setup_item_with_policy(
            2,
            Some(policy(
                UpProtectionPolicy::Preferred,
                UpProtectionPolicy::Required,
            )),
        );
        let err = task
            .setup_one_pdu_session(21, &item, gnb_ip)
            .await
            .expect_err("a required confidentiality policy with NEA0 must fail");
        assert_eq!(
            decode_setup_unsuccessful_transfer(&err.transfer).expect("a real cause"),
            NgSetupFailureCause::RadioNetwork(
                RadioNetworkCause::UpConfidentialityProtectionNotPossible
            )
        );
    }

    /// The positive control for the refusal above: a session the gNB CAN satisfy is
    /// established, and the SMF's policy is stored on it.
    ///
    /// Without this, a `setup_one_pdu_session` that refused every session carrying a
    /// `SecurityIndication` would pass the test above. `preferred` deliberately,
    /// because it is satisfiable in **both** feature arms — see
    /// `without_the_feature_a_required_policy_is_refused` for the other half.
    #[tokio::test]
    async fn a_satisfiable_policy_establishes_the_session_and_is_stored_on_it() {
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(22);
        key_ue(&mut task, 22, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();

        let requested = policy(UpProtectionPolicy::Preferred, UpProtectionPolicy::Preferred);
        let item = setup_item_with_policy(3, Some(requested));
        task.setup_one_pdu_session(22, &item, gnb_ip)
            .await
            .expect("a preferred policy never refuses");

        let session = task
            .find_ue_context(22)
            .and_then(|c| c.get_pdu_session(3).cloned())
            .expect("the session must be established");
        assert_eq!(
            (
                session.up_security_policy.integrity,
                session.up_security_policy.confidentiality
            ),
            (requested.integrity, requested.confidentiality),
            "the SMF's policy must reach the session context, not be discarded"
        );
        assert_eq!(
            session.up_security_policy.max_integrity_protected_data_rate,
            Some(MaxIntegrityProtectedDataRate::MaximumUeRate),
            "the conditional rate IE is present whenever integrity is wanted, and an \
             unsupplied one encodes as maximum-UE-rate rather than vanishing"
        );
        // Whether it is *applied* depends on the build; whether it was *asked for*
        // does not, which is why the two are separate fields.
        assert_eq!(
            session.up_security.integrity,
            cfg!(feature = "up-security"),
            "the decision must follow this build's capability"
        );
        assert_eq!(session.up_security.ciphering, cfg!(feature = "up-security"));
    }

    /// Without the `up-security` feature the gNB cannot protect a DRB, so a
    /// `required` policy is **refused** (TS 33.501 §6.6.1) rather than established
    /// unprotected.
    ///
    /// This does not take the default build's user plane down: `nextgcore`'s SMF
    /// sends `security_indication: None`, so the local `preferred` default applies
    /// and nothing is refused. Pinned here so that stays a checked fact rather than
    /// an assumption — if a core starts sending `required`, this is the test that
    /// says what happens.
    #[cfg(not(feature = "up-security"))]
    #[tokio::test]
    async fn without_the_feature_a_required_policy_is_refused() {
        use nextgsim_ngap::procedures::transfer::{
            decode_setup_unsuccessful_transfer, UpProtectionPolicy,
        };
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(26);
        // Real algorithms: the only reason protection is impossible is the build.
        key_ue(&mut task, 26, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        let item = setup_item_with_policy(
            7,
            Some(policy(
                UpProtectionPolicy::Required,
                UpProtectionPolicy::Preferred,
            )),
        );
        let err = task
            .setup_one_pdu_session(26, &item, gnb_ip)
            .await
            .expect_err("a build that cannot protect must not claim it did");
        assert_eq!(
            decode_setup_unsuccessful_transfer(&err.transfer).expect("a real cause"),
            NgSetupFailureCause::RadioNetwork(RadioNetworkCause::UpIntegrityProtectionNotPossible)
        );
    }

    /// With the feature on, the same `required` policy is satisfied and both
    /// protections are applied. The other half of the pair above: together they show
    /// the refusal follows the *capability* and not the policy.
    #[cfg(feature = "up-security")]
    #[tokio::test]
    async fn with_the_feature_a_required_policy_is_satisfied_and_both_protections_apply() {
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(27);
        key_ue(&mut task, 27, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        let item = setup_item_with_policy(
            8,
            Some(policy(
                UpProtectionPolicy::Required,
                UpProtectionPolicy::Required,
            )),
        );
        task.setup_one_pdu_session(27, &item, gnb_ip)
            .await
            .expect("NEA2/NIA2 with the feature on can satisfy a required policy");
        assert_eq!(
            task.find_ue_context(27)
                .and_then(|c| c.get_pdu_session(8))
                .map(|s| s.up_security),
            Some(DrbSecurityDecision {
                integrity: true,
                ciphering: true
            })
        );
    }

    /// A session the SMF sent no `SecurityIndication` for gets the locally configured
    /// policy, and is still established.
    #[tokio::test]
    async fn a_session_with_no_security_indication_gets_the_local_default() {
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(23);
        key_ue(&mut task, 23, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();

        task.setup_one_pdu_session(23, &setup_item_with_policy(4, None), gnb_ip)
            .await
            .expect("a session with no policy must still come up");
        assert_eq!(
            task.find_ue_context(23)
                .and_then(|c| c.get_pdu_session(4))
                .map(|s| s.up_security_policy),
            Some(UpSecurityPolicy::locally_configured_default()),
            "no SecurityIndication means the local policy, not an unprotected session"
        );
    }

    /// A `not-needed` integrity policy is honoured: nothing is protected, and the
    /// session comes up. Pinned separately from the resolver's own unit test because
    /// this is the path that reaches the RRC reconfiguration.
    #[tokio::test]
    async fn a_not_needed_integrity_policy_leaves_the_drb_without_a_mac_i() {
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(24);
        key_ue(&mut task, 24, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();

        let item = setup_item_with_policy(
            5,
            Some(policy(
                UpProtectionPolicy::NotNeeded,
                UpProtectionPolicy::Preferred,
            )),
        );
        task.setup_one_pdu_session(24, &item, gnb_ip)
            .await
            .expect("not-needed never refuses");
        let decision = task
            .find_ue_context(24)
            .and_then(|c| c.get_pdu_session(5))
            .map(|s| s.up_security)
            .expect("the session must exist");
        assert!(
            !decision.integrity,
            "a `not-needed` integrity policy must not append a MAC-I"
        );
    }

    /// A PDU Session Modify keeps the session's user-plane security policy.
    ///
    /// The Modify Request Transfer has no `SecurityIndication` (TS 38.413 §9.3.4.3),
    /// and the modify path rebuilds the session record — so without carrying the
    /// policy across, any QoS change would silently downgrade a protected session to
    /// the local default.
    #[tokio::test]
    async fn a_pdu_session_modify_keeps_the_sessions_security_policy() {
        use nextgsim_ngap::codec::generated::{
            PDUSessionResourceModifyRequestTransfer,
            PDUSessionResourceModifyRequestTransferProtocolIEs,
        };
        use nextgsim_ngap::procedures::pdu_session_resource::{
            PduSessionResourceModifyRequestData, PduSessionResourceModifyRequestItem,
        };
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;

        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(30);
        key_ue(&mut task, 30, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        // `not-needed` integrity: distinguishable from the local `preferred` default
        // in BOTH feature arms, which a `required` policy would not be.
        let requested = policy(UpProtectionPolicy::NotNeeded, UpProtectionPolicy::Preferred);
        task.setup_one_pdu_session(30, &setup_item_with_policy(11, Some(requested)), gnb_ip)
            .await
            .expect("setup");
        let ran_ue_ngap_id = task
            .find_ue_context(30)
            .map(|c| c.ran_ue_ngap_id as u32)
            .expect("ran id");

        // An empty container: every IE in the Modify Request Transfer is optional, so
        // this is a legal "nothing to change" modify and it still rebuilds the record.
        let transfer =
            nextgsim_ngap::codec::encode_aper(&PDUSessionResourceModifyRequestTransfer {
                protocol_i_es: PDUSessionResourceModifyRequestTransferProtocolIEs(vec![]),
            })
            .expect("encode an empty modify transfer");

        task.handle_pdu_session_resource_modify(
            1,
            8,
            PduSessionResourceModifyRequestData {
                amf_ue_ngap_id: 4242,
                ran_ue_ngap_id,
                pdu_session_resource_modify_list: vec![PduSessionResourceModifyRequestItem {
                    pdu_session_id: 11,
                    nas_pdu: None,
                    transfer,
                }],
            },
        )
        .await;

        let after = task
            .find_ue_context(30)
            .and_then(|c| c.get_pdu_session(11))
            .map(|s| s.up_security_policy)
            .expect("the session must survive the modify");
        assert_eq!(
            after.integrity,
            UpProtectionPolicy::NotNeeded,
            "a modify must not reset the session's policy to the local default"
        );
        assert_ne!(
            after.integrity,
            UpSecurityPolicy::locally_configured_default().integrity,
            "and the test must be able to tell the two apart"
        );
    }

    /// The Path Switch Request Acknowledge's policy is applied to the session, not
    /// only logged (TS 38.413 §9.3.4.10).
    #[tokio::test]
    async fn the_path_switch_acknowledges_policy_reaches_the_session() {
        use nextgsim_ngap::procedures::path_switch::SwitchedSessionItem;
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        let (mut task, _rrc_rx, _gtp_rx, _rls_rx) = task_with_live_receivers(25);
        key_ue(&mut task, 25, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        // Establish a session with NO policy, so the Acknowledge is the only source.
        task.setup_one_pdu_session(25, &setup_item_with_policy(6, None), gnb_ip)
            .await
            .expect("setup");
        let ran_ue_ngap_id = task
            .find_ue_context(25)
            .map(|c| c.ran_ue_ngap_id as u32)
            .expect("ran id");

        // `preferred`/`not-needed` so the Acknowledge is adopted in both feature arms;
        // the refusal path a `required` policy would take is covered separately.
        let switched = policy(UpProtectionPolicy::Preferred, UpProtectionPolicy::NotNeeded);
        task.handle_path_switch_request_acknowledge(
            1,
            8,
            PathSwitchRequestAcknowledgeData {
                amf_ue_ngap_id: 4242,
                ran_ue_ngap_id,
                next_hop_chaining_count: 1,
                next_hop_nh: [0x77; 32],
                switched_sessions: vec![SwitchedSessionItem {
                    pdu_session_id: 6,
                    ul_tunnel: None,
                    security_indication: Some(switched),
                }],
            },
        )
        .await;

        let adopted = task
            .find_ue_context(25)
            .and_then(|c| c.get_pdu_session(6))
            .map(|s| s.up_security_policy)
            .expect("the session must still exist");
        assert_eq!(
            (adopted.integrity, adopted.confidentiality),
            (switched.integrity, switched.confidentiality),
            "the target gNB must adopt the policy the 5GC restated, or an Xn handover \
             silently drops user-plane protection"
        );
        assert_ne!(
            (adopted.integrity, adopted.confidentiality),
            (
                UpSecurityPolicy::locally_configured_default().integrity,
                UpSecurityPolicy::locally_configured_default().confidentiality
            ),
            "and the adopted policy must differ from the local default the session \
             started with, or this test could not tell the two apart"
        );
    }

    /// #32, criterion 3 and 6 on the gNB side: setting up a protected session both
    /// hands the RLS task the `K_UPenc`/`K_UPint` binding **and** tells the UE to
    /// protect, in the same procedure.
    ///
    /// Reads what the task actually sent on both channels rather than the decision it
    /// recorded: a `up_security` field set to `true` with nothing on either channel is
    /// exactly the failure this issue is about.
    #[cfg(feature = "up-security")]
    #[tokio::test]
    async fn a_protected_session_keys_the_rls_entity_and_signals_the_ue() {
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            decode_rrc_reconfiguration, drb_integrity_protection, DrbIntegrityProtection,
        };
        let (mut task, mut rrc_rx, _gtp_rx, mut rls_rx) = task_with_live_receivers(28);
        key_ue(&mut task, 28, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        let item = setup_item_with_policy(
            9,
            Some(policy(
                UpProtectionPolicy::Required,
                UpProtectionPolicy::Required,
            )),
        );
        task.setup_one_pdu_session(28, &item, gnb_ip)
            .await
            .expect("setup");

        // 1. The RLS task was handed a binding for this DRB.
        let mut installed = None;
        while let Ok(msg) = rls_rx.try_recv() {
            if let TaskMessage::Message(crate::tasks::RlsMessage::InstallDrbSecurity {
                ue_id,
                psi,
                drb_id,
                security,
            }) = msg
            {
                installed = Some((ue_id, psi, drb_id, security));
            }
        }
        let (ue_id, psi, drb_id, security) =
            installed.expect("the RLS task must be handed the DRB's user-plane keys");
        assert_eq!((ue_id, psi), (28, 9));
        assert_eq!(
            drb_id, 9,
            "the keys must name the DRB whose identity BEARER was derived from \
             (issue #44), not merely the session"
        );
        let security = security.expect("a protected DRB must carry a real binding");
        assert!(
            security.security.integrity_protected(),
            "a `required` integrity policy must install a MAC-I"
        );
        assert_eq!(
            security.security.ciphering_alg_id(),
            2,
            "and the negotiated NEA, not the null one"
        );
        assert_eq!(
            security.bearer, 8,
            "BEARER is the radio bearer identity minus one (TS 33.501 Annex D.3.1.2)"
        );
        assert_eq!(
            security.tx_direction,
            nextgsim_pdcp::DIRECTION_DOWNLINK,
            "a gNB transmits downlink; the wrong bit fails every MAC with no other symptom"
        );

        // 2. The RRCReconfiguration told the UE to protect the same DRB.
        let mut reconfig = None;
        while let Ok(msg) = rrc_rx.try_recv() {
            if let TaskMessage::Message(crate::tasks::RrcMessage::RrcReconfiguration {
                pdu, ..
            }) = msg
            {
                reconfig = Some(pdu);
            }
        }
        let pdu = reconfig.expect("the UE must be sent an RRCReconfiguration");
        let data = decode_rrc_reconfiguration(pdu.data()).expect("real UPER");
        let rbc: nextgsim_rrc::codec::generated::RadioBearerConfig =
            nextgsim_rrc::codec::decode_rrc(
                &data.radio_bearer_config.expect("a radio bearer config"),
            )
            .expect("decode the radio bearer config");
        assert_eq!(
            drb_integrity_protection(&rbc, 9),
            DrbIntegrityProtection::Enabled,
            "the UE must be TOLD to integrity-protect, or it never will"
        );
    }

    /// The negative half: an unprotected session installs nothing and signals
    /// nothing. Without this, a gNB that keyed and signalled unconditionally would
    /// pass the test above.
    #[cfg(feature = "up-security")]
    #[tokio::test]
    async fn a_not_needed_policy_neither_keys_integrity_nor_signals_it() {
        use nextgsim_ngap::procedures::transfer::UpProtectionPolicy;
        use nextgsim_rrc::procedures::rrc_reconfiguration::{
            decode_rrc_reconfiguration, drb_integrity_protection, DrbIntegrityProtection,
        };
        let (mut task, mut rrc_rx, _gtp_rx, mut rls_rx) = task_with_live_receivers(29);
        key_ue(&mut task, 29, 2, 2);
        let gnb_ip: std::net::IpAddr = "10.0.0.1".parse().unwrap();
        let item = setup_item_with_policy(
            10,
            Some(policy(
                UpProtectionPolicy::NotNeeded,
                UpProtectionPolicy::NotNeeded,
            )),
        );
        task.setup_one_pdu_session(29, &item, gnb_ip)
            .await
            .expect("setup");

        let mut integrity_installed = None;
        while let Ok(msg) = rls_rx.try_recv() {
            if let TaskMessage::Message(crate::tasks::RlsMessage::InstallDrbSecurity {
                security,
                ..
            }) = msg
            {
                integrity_installed =
                    Some(security.is_some_and(|s| s.security.integrity_protected()));
            }
        }
        assert_eq!(
            integrity_installed,
            Some(false),
            "a `not-needed` integrity policy must not install a MAC-I -- ciphering is \
             still installed, which is the stated cipheringDisabled ceiling"
        );

        let mut reconfig = None;
        while let Ok(msg) = rrc_rx.try_recv() {
            if let TaskMessage::Message(crate::tasks::RrcMessage::RrcReconfiguration {
                pdu, ..
            }) = msg
            {
                reconfig = Some(pdu);
            }
        }
        let data = decode_rrc_reconfiguration(reconfig.expect("a reconfiguration").data())
            .expect("real UPER");
        let rbc: nextgsim_rrc::codec::generated::RadioBearerConfig =
            nextgsim_rrc::codec::decode_rrc(
                &data.radio_bearer_config.expect("a radio bearer config"),
            )
            .expect("decode");
        assert_eq!(
            drb_integrity_protection(&rbc, 10),
            DrbIntegrityProtection::Disabled,
            "and the UE must not be told to protect a DRB the gNB will not verify"
        );
    }

    /// Criterion 4: a UE Context Release Request with no Command in reply must not leave
    /// the context in `Releasing` forever.
    #[tokio::test]
    async fn tngreloc_overall_expiry_releases_a_stuck_ue_context() {
        let (mut task, _sctp_rx) = task_with_ue(11);

        task.handle_ue_context_release_request(11, UeReleaseRequestCause::UserTriggered)
            .await;
        assert!(
            task.guard_timers
                .is_pending(GuardTimer::UeContextRelease { ue_id: 11 }),
            "sending the request must arm TNGRELOCoverall"
        );
        assert_eq!(
            task.find_ue_context(11).map(|c| c.state),
            Some(UeState::Releasing),
            "precondition: the context is waiting on the AMF"
        );

        let overall = Duration::from_secs(test_config().ngap_tngreloc_overall_secs);
        let base = Instant::now();
        task.process_expired_guard_timers(base).await;
        assert!(
            task.find_ue_context(11).is_some(),
            "it must not be released before the timer expires"
        );

        task.process_expired_guard_timers(base + overall).await;
        assert!(
            task.find_ue_context(11).is_none(),
            "on expiry the context is released locally, or it holds a RAN-UE-NGAP-ID \
             forever and the id space is finite"
        );
    }

    /// The AMF answering must disarm the timer, or the expiry would later act on a
    /// ue_id that has been freed and possibly reused.
    #[tokio::test]
    async fn a_release_command_cancels_tngreloc_overall() {
        use nextgsim_ngap::procedures::ue_context_release::{
            UeContextReleaseCommandData, UeNgapIds,
        };
        let (mut task, _sctp_rx) = task_with_ue(12);

        task.handle_ue_context_release_request(12, UeReleaseRequestCause::UserTriggered)
            .await;
        assert!(task
            .guard_timers
            .is_pending(GuardTimer::UeContextRelease { ue_id: 12 }));

        let ran_ue_ngap_id = task
            .find_ue_context(12)
            .map(|c| c.ran_ue_ngap_id)
            .expect("ue context");
        task.handle_ue_context_release_command(
            1,
            0,
            UeContextReleaseCommandData {
                ue_ngap_ids: UeNgapIds::Pair {
                    amf_ue_ngap_id: 4242,
                    ran_ue_ngap_id: ran_ue_ngap_id as u32,
                },
                cause: NgSetupFailureCause::Nas(NasCause::NormalRelease),
            },
        )
        .await;
        assert!(
            !task
                .guard_timers
                .is_pending(GuardTimer::UeContextRelease { ue_id: 12 }),
            "the AMF answered, so the guard timer must be cancelled"
        );
    }

    /// Criterion 5: handover preparation that is never answered is cancelled, and
    /// cancelling means SENDING Handover Cancel (TS 38.413 §8.4.5) -- the AMF and target
    /// may hold resources that only the cancel releases.
    #[tokio::test]
    async fn tngreloc_prep_expiry_cancels_the_handover() {
        let (mut task, mut sctp_rx) = task_with_ue(13);
        let ran_ue_ngap_id = task
            .find_ue_context(13)
            .map(|c| c.ran_ue_ngap_id)
            .expect("ue context") as u32;

        task.send_handover_required(
            1,
            4242,
            ran_ue_ngap_id,
            &[0x00, 0xf1, 0x10],
            0x1234,
            &[0x00, 0x00, 0x01],
            &[0x00],
            &[],
        )
        .await;
        assert!(
            task.guard_timers
                .is_pending(GuardTimer::HandoverPreparation { ue_id: 13 }),
            "sending Handover Required must arm TNGRELOCprep"
        );
        // Drain the Handover Required so the assertion below is about the cancel.
        while sctp_rx.try_recv().is_ok() {}

        let prep = Duration::from_secs(test_config().ngap_tngreloc_prep_secs);
        let base = Instant::now();
        task.process_expired_guard_timers(base + prep - Duration::from_millis(1))
            .await;
        assert!(
            sctp_rx.try_recv().is_err(),
            "nothing may be sent before the timer expires"
        );

        task.process_expired_guard_timers(base + prep).await;
        let sent = sctp_rx.try_recv().expect("a Handover Cancel must be sent");
        let bytes = match sent {
            TaskMessage::Message(crate::tasks::SctpMessage::SendMessage { buffer, .. }) => {
                buffer.data().to_vec()
            }
            other => panic!("expected an NGAP PDU to the SCTP task, got {other:?}"),
        };
        let cancel = nextgsim_ngap::procedures::handover::decode_handover_cancel(&bytes)
            .expect("the PDU must decode as Handover Cancel");
        assert_eq!(cancel.amf_ue_ngap_id, 4242);
        assert_eq!(cancel.ran_ue_ngap_id, ran_ue_ngap_id);
        assert!(
            task.find_ue_context(13).is_some(),
            "the UE stays on the source cell: cancelling a handover does not release it"
        );
    }

    /// A Handover Command disarms TNGRELOCprep, so a completed handover cannot later be
    /// cancelled by its own guard timer.
    #[tokio::test]
    async fn a_handover_command_cancels_tngreloc_prep() {
        let (mut task, mut sctp_rx) = task_with_ue(14);
        let ran_ue_ngap_id = task
            .find_ue_context(14)
            .map(|c| c.ran_ue_ngap_id)
            .expect("ue context") as u32;
        task.send_handover_required(
            1,
            4242,
            ran_ue_ngap_id,
            &[0x00, 0xf1, 0x10],
            0x1234,
            &[0x00, 0x00, 0x01],
            &[0x00],
            &[],
        )
        .await;
        while sctp_rx.try_recv().is_ok() {}

        task.handle_handover_command(
            1,
            0,
            HandoverCommandData {
                amf_ue_ngap_id: 4242,
                ran_ue_ngap_id,
                handover_type: HandoverTypeValue::Intra5gs,
                target_to_source_transparent_container: vec![0x01, 0x02],
                pdu_session_resource_handover_list: None,
                pdu_session_resource_to_release_list: None,
            },
        )
        .await;
        assert!(
            !task
                .guard_timers
                .is_pending(GuardTimer::HandoverPreparation { ue_id: 14 }),
            "preparation completed, so its guard timer must be cancelled"
        );

        // And no Handover Cancel is produced later.
        while sctp_rx.try_recv().is_ok() {}
        task.process_expired_guard_timers(Instant::now() + Duration::from_secs(3600))
            .await;
        assert!(
            sctp_rx.try_recv().is_err(),
            "a completed handover must never be cancelled by its own guard timer"
        );
    }

    /// Criterion 6: an NG Setup Failure carrying Time to Wait schedules a retry, and the
    /// retry does not fire before the indicated time.
    #[tokio::test]
    async fn ng_setup_failure_time_to_wait_gates_the_retry() {
        use nextgsim_ngap::procedures::ng_setup::{build_ng_setup_failure, NgSetupFailureParams};
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
        }

        let failure = build_ng_setup_failure(&NgSetupFailureParams {
            cause: NgSetupFailureCause::Misc(
                nextgsim_ngap::procedures::ng_setup::MiscCause::ControlProcessingOverload,
            ),
            time_to_wait: Some(TimeToWaitValue::V10s),
        })
        .expect("build failure");
        let bytes = nextgsim_ngap::codec::encode_ngap_pdu(&failure).expect("encode");
        assert!(
            task.handle_ng_setup_failure(1, &bytes),
            "the failure must be recognised"
        );
        assert!(
            task.guard_timers
                .is_pending(GuardTimer::NgSetupRetry { amf_id: 1 }),
            "Time to Wait must schedule a retry rather than being logged and dropped"
        );

        while sctp_rx.try_recv().is_ok() {}
        let base = Instant::now();
        task.process_expired_guard_timers(base + Duration::from_secs(9))
            .await;
        assert!(
            sctp_rx.try_recv().is_err(),
            "TS 38.413 §8.7.1.3: the retry must wait AT LEAST the indicated 10s"
        );

        task.process_expired_guard_timers(base + Duration::from_secs(10))
            .await;
        assert!(
            sctp_rx.try_recv().is_ok(),
            "once the wait has elapsed the NG Setup is re-initiated"
        );
    }

    /// The Time to Wait mapping is the whole of the IE's meaning, so it is asserted
    /// value by value rather than at one sample.
    #[test]
    fn time_to_wait_maps_to_the_seconds_ts_38_413_defines() {
        for (value, secs) in [
            (TimeToWaitValue::V1s, 1),
            (TimeToWaitValue::V2s, 2),
            (TimeToWaitValue::V5s, 5),
            (TimeToWaitValue::V10s, 10),
            (TimeToWaitValue::V20s, 20),
            (TimeToWaitValue::V60s, 60),
        ] {
            assert_eq!(value.as_duration(), Duration::from_secs(secs), "{value:?}");
        }
    }

    /// An NG Setup Failure with NO Time to Wait must not schedule anything: there is no
    /// mandated wait, and retrying on a timer this gNB invented would be worse than
    /// leaving the existing SCTP reconnection path to govern.
    #[tokio::test]
    async fn ng_setup_failure_without_time_to_wait_schedules_no_retry() {
        use nextgsim_ngap::procedures::ng_setup::{build_ng_setup_failure, NgSetupFailureParams};
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);

        let failure = build_ng_setup_failure(&NgSetupFailureParams {
            cause: NgSetupFailureCause::Misc(
                nextgsim_ngap::procedures::ng_setup::MiscCause::Unspecified,
            ),
            time_to_wait: None,
        })
        .expect("build failure");
        let bytes = nextgsim_ngap::codec::encode_ngap_pdu(&failure).expect("encode");
        assert!(task.handle_ng_setup_failure(1, &bytes));
        assert!(
            task.guard_timers.is_empty(),
            "no Time to Wait means no mandated wait and so no scheduled retry"
        );
    }

    // ========================================================================
    // PDU Session Resource Notify (issue #98, TS 38.413 §8.3.5)
    // ========================================================================

    /// Pops the next NGAP PDU handed to SCTP.
    fn next_ngap_pdu(
        rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<crate::tasks::SctpMessage>>,
    ) -> Vec<u8> {
        loop {
            match rx.try_recv() {
                Ok(TaskMessage::Message(crate::tasks::SctpMessage::SendMessage {
                    buffer, ..
                })) => return buffer.data().to_vec(),
                Ok(_) => continue,
                Err(e) => panic!("expected an NGAP PDU on SCTP, got none: {e}"),
            }
        }
    }

    /// #98: a released PDU session is reported to the AMF at PER-SESSION
    /// granularity, and the emitted bytes decode as a real Notify.
    ///
    /// Asserted by DECODING what went to SCTP, not by inspecting the params: the
    /// whole point of the issue is that the procedure reaches the wire.
    #[tokio::test]
    async fn a_released_pdu_session_is_reported_as_a_notify_on_the_wire() {
        use nextgsim_ngap::procedures::pdu_session_resource_notify::decode_pdu_session_resource_notify;

        let (mut task, mut sctp_rx) = task_with_ue(11);

        task.send_pdu_session_resource_notify(
            11,
            vec![(5, NotifyCause::TransportResourceUnavailable)],
            Vec::new(),
        )
        .await;

        let bytes = next_ngap_pdu(&mut sctp_rx);
        let decoded = decode_pdu_session_resource_notify(&bytes)
            .expect("the emitted PDU must decode as a PDU Session Resource Notify");
        assert_eq!(decoded.amf_ue_ngap_id, 4242, "the AMF's own UE identity");
        assert_eq!(decoded.released_sessions.len(), 1);
        assert_eq!(decoded.released_sessions[0].pdu_session_id, 5);
        assert_eq!(
            decoded.released_sessions[0].cause,
            NotifyCause::TransportResourceUnavailable
        );
        assert!(
            decoded.notified_sessions.is_empty(),
            "reporting a release must not also claim a QoS-flow change"
        );
    }

    /// A surviving session with a not-fulfilled QoS flow is reported without
    /// releasing anything -- the distinction the two lists exist to make.
    #[tokio::test]
    async fn an_unfulfilled_qos_flow_is_reported_without_releasing_the_session() {
        use nextgsim_ngap::procedures::pdu_session_resource_notify::{
            decode_pdu_session_resource_notify, NotificationCauseValue,
        };

        let (mut task, mut sctp_rx) = task_with_ue(11);

        task.send_pdu_session_resource_notify(
            11,
            Vec::new(),
            vec![(
                7,
                vec![NotifiedQosFlow {
                    qos_flow_identifier: 3,
                    notification_cause: NotificationCauseValue::NotFulfilled,
                }],
            )],
        )
        .await;

        let bytes = next_ngap_pdu(&mut sctp_rx);
        let decoded = decode_pdu_session_resource_notify(&bytes).expect("decodes");
        assert!(
            decoded.released_sessions.is_empty(),
            "the session survives; only the flow's state changed"
        );
        assert_eq!(decoded.notified_sessions.len(), 1);
        assert_eq!(decoded.notified_sessions[0].pdu_session_id, 7);
        assert_eq!(
            decoded.notified_sessions[0].notified_flows[0].notification_cause,
            NotificationCauseValue::NotFulfilled
        );
    }

    /// The sender is reachable from the `NgapMessage` dispatch, not only by a
    /// direct call. A sender with no dispatch arm is unreachable in the running
    /// binary -- the recorded failure mode of a correct fix in an unreachable place.
    #[tokio::test]
    async fn the_notify_is_reachable_through_the_ngap_message_dispatch() {
        use nextgsim_ngap::procedures::pdu_session_resource_notify::decode_pdu_session_resource_notify;

        let (mut task, mut sctp_rx) = task_with_ue(11);
        let (tx, rx) = tokio::sync::mpsc::channel::<TaskMessage<NgapMessage>>(4);

        tx.send(TaskMessage::Message(
            NgapMessage::PduSessionResourceNotify {
                ue_id: 11,
                released_sessions: vec![(9, NotifyCause::RadioResourcesNotAvailable)],
                notified_sessions: Vec::new(),
            },
        ))
        .await
        .expect("queued");
        tx.send(TaskMessage::Shutdown).await.expect("queued");
        drop(tx);

        task.run(rx).await;

        let bytes = next_ngap_pdu(&mut sctp_rx);
        let decoded = decode_pdu_session_resource_notify(&bytes)
            .expect("the dispatched message must reach the wire as a Notify");
        assert_eq!(decoded.released_sessions[0].pdu_session_id, 9);
        assert_eq!(
            decoded.released_sessions[0].cause,
            NotifyCause::RadioResourcesNotAvailable
        );
    }

    /// A UE with no AMF-UE-NGAP-ID cannot be the subject of a Notify: the IE is
    /// mandatory, and a fabricated value is one the AMF would fail to resolve. The
    /// sender must report and send NOTHING rather than emit an unresolvable PDU.
    #[tokio::test]
    async fn a_ue_without_an_amf_ue_ngap_id_emits_no_notify() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        task.create_ue_context(12, 1).expect("ue context");
        // Deliberately NOT setting amf_ue_ngap_id.

        task.send_pdu_session_resource_notify(
            12,
            vec![(1, NotifyCause::TransportResourceUnavailable)],
            Vec::new(),
        )
        .await;

        assert!(
            sctp_rx.try_recv().is_err(),
            "no PDU may be emitted for a UE the AMF has not identified"
        );
    }

    /// An unknown UE emits nothing either, rather than panicking or sending a PDU
    /// with a fabricated RAN-UE-NGAP-ID.
    #[tokio::test]
    async fn an_unknown_ue_emits_no_notify() {
        let (mut task, mut sctp_rx) = task_with_ue(11);
        task.send_pdu_session_resource_notify(
            999,
            vec![(1, NotifyCause::TransportResourceUnavailable)],
            Vec::new(),
        )
        .await;
        assert!(sctp_rx.try_recv().is_err());
    }

    // ========================================================================
    // AMF interface management remainder (issue #41)
    // ========================================================================

    /// #41, criterion 3: an AMF that asked for a traffic reduction must actually
    /// receive less. The percentage used to be stored and read only in a log line.
    ///
    /// The gate is deterministic, so this asserts an EXACT count rather than a
    /// statistical range — a probabilistic gate would make this test a coin flip.
    #[test]
    fn a_traffic_load_reduction_throttles_initial_ue_messages_in_proportion() {
        let mut ctx = NgapAmfContext::new(1);
        ctx.on_association_up(1, 4, 4);
        ctx.state = AmfState::Ready;

        // No overload: everything is admitted.
        for i in 0..100 {
            assert!(
                ctx.admit_initial_ue_message(),
                "attempt {i} must be admitted with no reduction in force"
            );
        }

        // 75% reduction: exactly 25 of the next 100 attempts get through.
        ctx.on_overload_start();
        ctx.traffic_load_reduction = Some(75);
        let admitted = (0..100).filter(|_| ctx.admit_initial_ue_message()).count();
        assert_eq!(
            admitted, 25,
            "a 75% reduction must admit 25 of 100, not 0 and not 100"
        );

        // 1% reduction admits 99 -- the boundary that a naive "block if reduction
        // > 0" gate would get wrong by blocking everything.
        let mut ctx = NgapAmfContext::new(2);
        ctx.state = AmfState::Ready;
        ctx.on_overload_start();
        ctx.traffic_load_reduction = Some(1);
        assert_eq!(
            (0..100).filter(|_| ctx.admit_initial_ue_message()).count(),
            99,
            "a 1% reduction must barely throttle"
        );

        // 99% admits 1 -- the other boundary.
        let mut ctx = NgapAmfContext::new(3);
        ctx.state = AmfState::Ready;
        ctx.on_overload_start();
        ctx.traffic_load_reduction = Some(99);
        assert_eq!(
            (0..100).filter(|_| ctx.admit_initial_ue_message()).count(),
            1,
            "a 99% reduction must still let one through: the IE is a REDUCTION, \
             not a bar"
        );
    }

    /// OVERLOAD STOP restores full admission and does not leave a partial block
    /// behind.
    #[test]
    fn overload_stop_restores_full_admission() {
        let mut ctx = NgapAmfContext::new(1);
        ctx.state = AmfState::Ready;
        ctx.on_overload_start();
        ctx.traffic_load_reduction = Some(90);
        // Consume a few attempts so the credit is mid-cycle.
        for _ in 0..5 {
            ctx.admit_initial_ue_message();
        }
        ctx.on_overload_stop();
        ctx.traffic_load_reduction = None;
        for i in 0..20 {
            assert!(
                ctx.admit_initial_ue_message(),
                "attempt {i} after OVERLOAD STOP must be admitted"
            );
        }
    }

    /// The gate is on the ADMISSION path, not just on the context: an Initial UE
    /// Message for a new UE must not reach SCTP while the AMF is throttling, and
    /// no UE context may be created for it either -- creating one and abandoning it
    /// leaks a RAN UE NGAP ID and a stream for a UE the AMF never heard of.
    #[tokio::test]
    async fn a_throttled_initial_ue_message_reaches_neither_sctp_nor_a_ue_context() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
            ctx.on_overload_start();
            // 100 is out of the wire range (1..99) and means "stop"; the gate
            // clamps it rather than looping on zero credit.
            ctx.traffic_load_reduction = Some(100);
        }

        task.handle_initial_nas_delivery(
            77,
            OctetString::from_slice(&[0x7e, 0x00, 0x41]),
            0,
            None,
            Vec::new(),
        )
        .await;

        assert!(
            sctp_rx.try_recv().is_err(),
            "a fully throttled AMF must receive no Initial UE Message"
        );
        assert!(
            task.find_ue_context(77).is_none(),
            "no UE context may be created for a UE whose Initial UE Message was \
             never sent"
        );
    }

    /// The positive control for the test above, and it exists because a revert
    /// round needed it.
    ///
    /// Disabling the admission gate left that test GREEN: an `Overloaded` AMF was
    /// excluded from `select_amf` outright (the old `is_ready()` predicate), so the
    /// Initial UE Message was blocked for a completely different reason and the
    /// gate could never run at all. This asserts the SAME overloaded AMF DOES get
    /// the message when its reduction admits one — so "blocked" above means the
    /// gate blocked it, not that no AMF was selectable.
    #[tokio::test]
    async fn an_overloaded_amf_still_receives_the_messages_its_reduction_admits() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
            ctx.on_overload_start();
            // 1% reduction: the very first attempt is admitted.
            ctx.traffic_load_reduction = Some(1);
        }
        assert_eq!(
            task.find_amf_context(1).map(|c| c.state),
            Some(AmfState::Overloaded),
            "precondition: the AMF really is overloaded"
        );

        task.handle_initial_nas_delivery(
            78,
            OctetString::from_slice(&[0x7e, 0x00, 0x41]),
            0,
            None,
            Vec::new(),
        )
        .await;

        assert!(
            sctp_rx.try_recv().is_ok(),
            "an overloaded AMF asking for a 1% reduction must still be SELECTED and \
             still receive this message; excluding it from selection would apply a \
             100% reduction whatever it asked for"
        );
        assert!(
            task.find_ue_context(78).is_some(),
            "and the admitted UE must get its context"
        );
    }

    /// #41, criterion 5: a TNL association the AMF asks to ADD is connected, and
    /// the Acknowledge reports it in `AMF-TNLAssociationSetupList` rather than
    /// being a bare acknowledge that reads as "nothing to do".
    #[tokio::test]
    async fn an_amf_configuration_update_adds_a_tnl_association_and_reports_it() {
        use nextgsim_ngap::procedures::ng_reset::{
            parse_amf_configuration_update_acknowledge, AmfConfigurationUpdateData,
            AmfTnlAssociationAddress, AmfTnlAssociationToAdd, TnlAssociationUsage,
        };
        use std::net::{IpAddr, Ipv4Addr};

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        let endpoint = IpAddr::V4(Ipv4Addr::new(10, 20, 30, 40));
        let update = AmfConfigurationUpdateData {
            amf_name: None,
            served_guami_list: Vec::new(),
            relative_amf_capacity: None,
            plmn_support_list: Vec::new(),
            tnla_to_add: vec![AmfTnlAssociationToAdd {
                address: AmfTnlAssociationAddress { endpoint },
                usage: Some(TnlAssociationUsage::Both),
                weight_factor: 50,
            }],
            tnla_to_remove: Vec::new(),
            tnla_to_update: Vec::new(),
        };
        task.handle_amf_configuration_update(1, 0, update).await;

        // A ConnectionRequest to the named endpoint, and then the Acknowledge.
        let mut connect_seen = false;
        let mut ack_bytes = None;
        while let Ok(TaskMessage::Message(msg)) = sctp_rx.try_recv() {
            match msg {
                crate::tasks::SctpMessage::ConnectionRequest {
                    remote_address,
                    remote_port,
                    ..
                } => {
                    assert_eq!(remote_address, "10.20.30.40");
                    assert_eq!(remote_port, 38412, "the NGAP SCTP port");
                    connect_seen = true;
                }
                crate::tasks::SctpMessage::SendMessage { buffer, .. } => {
                    ack_bytes = Some(buffer.data().to_vec());
                }
                _ => {}
            }
        }
        assert!(
            connect_seen,
            "the gNB must actually open the TNL association the AMF named, not \
             just record it"
        );

        let ack = parse_amf_configuration_update_acknowledge(
            &ack_bytes.expect("an Acknowledge must be sent"),
        )
        .expect("the Acknowledge must decode");
        assert_eq!(
            ack.tnla_setup,
            vec![AmfTnlAssociationAddress { endpoint }],
            "the Acknowledge must report the association it set up; a bare \
             acknowledge reads as 'nothing to do' and the AMF would then balance \
             traffic onto an association it was never told about"
        );
        assert!(ack.tnla_failed_to_setup.is_empty());
    }

    /// An AMF asking to remove an association this gNB never established must not
    /// cause it to close something else -- notably not the association carrying
    /// the very message.
    #[tokio::test]
    async fn removing_an_unknown_tnl_association_closes_nothing() {
        use nextgsim_ngap::procedures::ng_reset::{
            AmfConfigurationUpdateData, AmfTnlAssociationAddress,
        };
        use std::net::{IpAddr, Ipv4Addr};

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        let update = AmfConfigurationUpdateData {
            amf_name: None,
            served_guami_list: Vec::new(),
            relative_amf_capacity: None,
            plmn_support_list: Vec::new(),
            tnla_to_add: Vec::new(),
            tnla_to_remove: vec![AmfTnlAssociationAddress {
                endpoint: IpAddr::V4(Ipv4Addr::new(9, 9, 9, 9)),
            }],
            tnla_to_update: Vec::new(),
        };
        task.handle_amf_configuration_update(1, 0, update).await;

        let mut closes = 0;
        while let Ok(TaskMessage::Message(msg)) = sctp_rx.try_recv() {
            if matches!(msg, crate::tasks::SctpMessage::ConnectionClose { .. }) {
                closes += 1;
            }
        }
        assert_eq!(
            closes, 0,
            "an unknown address must close nothing; the association carrying this \
             very message is the one that would be torn down"
        );
        assert!(
            task.find_amf_context(1).is_some(),
            "and the AMF context must survive"
        );
    }

    /// #41, criterion 6: RAN CONFIGURATION UPDATE has a real send path, reachable
    /// from the `NgapMessage` dispatch. It used to carry `#[allow(dead_code)]` with
    /// no caller at all.
    #[tokio::test]
    async fn a_ran_configuration_update_is_sent_from_the_ngap_dispatch() {
        use nextgsim_ngap::procedures::ran_configuration_update::decode_ran_configuration_update;

        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, mut sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 4, 4);
            ctx.state = AmfState::Ready;
        }

        let (tx, rx) = tokio::sync::mpsc::channel::<TaskMessage<NgapMessage>>(4);
        tx.send(TaskMessage::Message(
            NgapMessage::SendRanConfigurationUpdate { amf_id: None },
        ))
        .await
        .expect("queued");
        tx.send(TaskMessage::Shutdown).await.expect("queued");
        drop(tx);
        task.run(rx).await;

        let mut sent = None;
        while let Ok(TaskMessage::Message(msg)) = sctp_rx.try_recv() {
            if let crate::tasks::SctpMessage::SendMessage { buffer, .. } = msg {
                sent = Some(buffer.data().to_vec());
            }
        }
        let bytes = sent.expect("a RAN Configuration Update must reach SCTP");
        let decoded = decode_ran_configuration_update(&bytes)
            .expect("and must decode as a RAN Configuration Update");
        assert_eq!(
            decoded.ran_node_name.as_deref(),
            Some("nextgsim-gnb"),
            "the update must carry this node's identity"
        );
    }

    // ========================================================================
    // MBS session procedures on the wire (TS 38.413 §9.2.9, issue #185)
    // ========================================================================

    /// A `MulticastSessionActivationRequest` as built by a real nextgcore AMF
    /// (`nextgcore_ngap::builder::build_multicast_session_activation_request`
    /// for TMGI 00F110010203). Byte 1 = 0x47 = procedure 71.
    const AMF_MBS_ACTIVATION: &[u8] = &[
        0x00, 0x47, 0x00, 0x1A, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x30, 0x00, 0x08, 0x07, 0x00, 0x00, 0xF1, 0x10, 0x01, 0x02, 0x03,
    ];

    /// Its deactivation counterpart (procedure 72 = 0x48), same TMGI.
    const AMF_MBS_DEACTIVATION: &[u8] = &[
        0x00, 0x48, 0x00, 0x1A, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x31, 0x00, 0x08, 0x07, 0x00, 0x00, 0xF1, 0x10, 0x01, 0x02, 0x03,
    ];

    /// `MulticastGroupPaging` (procedure 74 = 0x4A) for the same TMGI, naming
    /// TAC 000001 under PLMN 00F110 — which `test_config` serves.
    const AMF_MBS_GROUP_PAGING: &[u8] = &[
        0x00, 0x4A, 0x40, 0x1B, 0x00, 0x00, 0x02, 0x01, 0x2B, 0x00, 0x07, 0x00, 0x00, 0xF1, 0x10,
        0x01, 0x02, 0x03, 0x01, 0x33, 0x40, 0x09, 0x00, 0x00, 0x00, 0x00, 0xF1, 0x10, 0x00, 0x00,
        0x01,
    ];

    const AMF_MBS_TMGI: [u8; 6] = [0x00, 0xF1, 0x10, 0x01, 0x02, 0x03];

    /// An NGAP task with one AMF in the Ready state and no UE.
    fn seed_task_ready() -> (
        NgapTask,
        tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
    ) {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, sctp_rx) =
            GnbTaskBase::new(test_config(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = NgapTask::new(task_base);
        task.create_amf_context(1);
        if let Some(ctx) = task.find_amf_context_mut(1) {
            ctx.on_association_up(100, 8, 8);
            ctx.state = AmfState::Ready;
        }
        (task, sctp_rx)
    }

    /// Criteria 1, 2 and 3 together: a real AMF's activation request arriving on
    /// the SCTP receive path reaches `NgapMbsManager`, and the gNB answers with a
    /// `MulticastSessionActivationResponse` whose `MBS-SessionID` echoes the
    /// request's.
    ///
    /// The session-count and TMGI assertions are only reachable if
    /// `handle_ngap_pdu` routed the PDU into the manager -- before this change
    /// the same bytes fell through to `handle_unroutable_pdu` and the reply was
    /// an Error Indication.
    #[tokio::test]
    async fn an_amf_mbs_activation_reaches_the_manager_and_is_answered() {
        let (mut task, mut sctp_rx) = seed_task_ready();
        assert_eq!(task.mbs_ngap_sessions.session_count(), 0);

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(AMF_MBS_ACTIVATION))
            .await;

        // (1) The state machine saw it, keyed by the TMGI the AMF sent.
        assert_eq!(task.mbs_ngap_sessions.session_count(), 1);
        let session = task
            .mbs_ngap_sessions
            .get(&AMF_MBS_TMGI)
            .expect("the activated session must be keyed by its TMGI");
        assert_eq!(session.tmgi, AMF_MBS_TMGI);
        assert!(session.is_active(), "activation must leave it Active");
        assert!(
            session
                .active_cells
                .contains(&i32::from(phys_cell_id_from_nci(0x000000010))),
            "the session must be radiating in this gNB's served cell"
        );

        // (2) The reply is a MulticastSessionActivationResponse echoing the TMGI.
        let reply = match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, stream, .. })) => {
                assert_eq!(
                    stream, 0,
                    "MBS session procedures are non-UE-associated (TS 38.412 §7)"
                );
                buffer.data().to_vec()
            }
            other => panic!("expected an outbound SCTP SendMessage, got {other:?}"),
        };

        match decode_ngap_pdu(&reply).expect("the reply must be a valid NGAP PDU") {
            NGAP_PDU::SuccessfulOutcome(o) => {
                assert_eq!(
                    o.procedure_code.0, 71,
                    "the outcome must be id-MulticastSessionActivation"
                );
            }
            other => panic!("expected a SuccessfulOutcome, got {other:?}"),
        }

        // Decoded through the public codec so the echo is asserted on the wire
        // bytes, not on a struct this test built.
        let echoed =
            nextgsim_ngap::procedures::mbs::decode_multicast_session_activation_response(&reply)
                .expect("the reply must decode as a MulticastSessionActivationResponse");
        assert_eq!(
            echoed.mbs_session_id.tmgi, AMF_MBS_TMGI,
            "the response's MBS-SessionID must echo the request's"
        );
    }

    /// A deactivation request tears the session down and is answered with a
    /// `MulticastSessionDeactivationResponse` (TS 38.413 §9.2.9.3/§9.2.9.4).
    #[tokio::test]
    async fn an_amf_mbs_deactivation_tears_the_session_down_and_is_answered() {
        let (mut task, mut sctp_rx) = seed_task_ready();

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(AMF_MBS_ACTIVATION))
            .await;
        assert_eq!(task.mbs_ngap_sessions.session_count(), 1);
        let _ = sctp_rx.try_recv(); // the activation response

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(AMF_MBS_DEACTIVATION))
            .await;

        assert_eq!(
            task.mbs_ngap_sessions.session_count(),
            0,
            "deactivation must remove the session"
        );

        let reply = match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                buffer.data().to_vec()
            }
            other => panic!("expected an outbound SCTP SendMessage, got {other:?}"),
        };
        match decode_ngap_pdu(&reply).expect("valid NGAP PDU") {
            NGAP_PDU::SuccessfulOutcome(o) => assert_eq!(
                o.procedure_code.0, 72,
                "the outcome must be id-MulticastSessionDeactivation"
            ),
            other => panic!("expected a SuccessfulOutcome, got {other:?}"),
        }
    }

    /// `MulticastGroupPaging` is accepted for an active session and, per
    /// TS 38.413 §9.2.9.5, answered with nothing at all — in particular NOT with
    /// an Error Indication, which is what an unrouted PDU would have produced.
    #[tokio::test]
    async fn an_amf_multicast_group_paging_is_accepted_without_a_reply() {
        let (mut task, mut sctp_rx) = seed_task_ready();

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(AMF_MBS_ACTIVATION))
            .await;
        let _ = sctp_rx.try_recv(); // the activation response

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(AMF_MBS_GROUP_PAGING))
            .await;

        // The session is untouched by paging, and nothing went back on the wire.
        assert_eq!(task.mbs_ngap_sessions.session_count(), 1);
        match sctp_rx.try_recv() {
            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {}
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                let pdu = decode_ngap_pdu(buffer.data());
                panic!("MulticastGroupPaging has no response message, but the gNB sent {pdu:?}");
            }
            other => panic!("unexpected SCTP traffic: {other:?}"),
        }
    }

    /// The MBS dispatch does not swallow procedures it does not implement.
    ///
    /// `BroadcastSessionSetupRequest` (procedure 68) carries an MB-UPF shared
    /// tunnel this node has no user-plane path for, so it must still fall through
    /// to the Error Indication rather than being silently accepted.
    #[tokio::test]
    async fn an_unsupported_mbs_procedure_still_earns_an_error_indication() {
        let (mut task, mut sctp_rx) = seed_task_ready();

        // Procedure 68 with an empty IE container: enough to decode as a
        // BroadcastSessionSetupRequest, which this node does not handle.
        let pdu = NGAP_PDU::InitiatingMessage(nextgsim_ngap::codec::InitiatingMessage {
            procedure_code: nextgsim_ngap::codec::ProcedureCode(68),
            criticality: nextgsim_ngap::codec::Criticality(
                nextgsim_ngap::codec::Criticality::REJECT,
            ),
            value: nextgsim_ngap::codec::InitiatingMessageValue::Id_BroadcastSessionSetup(
                nextgsim_ngap::codec::BroadcastSessionSetupRequest {
                    protocol_i_es: nextgsim_ngap::codec::BroadcastSessionSetupRequestProtocolIEs(
                        vec![],
                    ),
                },
            ),
        });
        let bytes = encode_ngap_pdu(&pdu).expect("procedure 68 must encode");

        task.handle_ngap_pdu(1, 0, OctetString::from_slice(&bytes))
            .await;

        assert_eq!(
            task.mbs_ngap_sessions.session_count(),
            0,
            "an unsupported broadcast setup must not create a session"
        );

        let reply = match sctp_rx.try_recv() {
            Ok(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. })) => {
                buffer.data().to_vec()
            }
            other => panic!("expected an Error Indication, got {other:?}"),
        };
        let err = decode_error_indication(&reply)
            .expect("the reply to an unsupported procedure must be an Error Indication");
        assert_eq!(
            err.criticality_diagnostics
                .as_ref()
                .and_then(|d| d.procedure_code),
            Some(68),
            "the diagnostics must name the procedure that was not supported"
        );
    }
}
