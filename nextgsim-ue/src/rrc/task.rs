//! RRC Task Implementation for UE
//!
//! This module implements the RRC (Radio Resource Control) task for the UE,
//! handling cell selection, RRC connection management, and handover.
//!
//! # Reference
//! - 3GPP TS 38.331: NR; RRC protocol specification
//! - 3GPP TS 38.304: UE procedures in Idle mode and RRC Inactive state
//! - UERANSIM: src/ue/rrc/task.cpp

use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

use crate::rrc::cell_selection::{
    CellChangeEvent, CellSelector, MibInfo, Plmn as CellPlmn, Sib1Info,
};
use crate::rrc::conditional_handover::{handover_command_for, CondReconfigStore};
use crate::rrc::handover::{
    build_reconfiguration_complete, parse_handover_command, HandoverCommand, HandoverManager,
};
use crate::rrc::measurement::{
    EutraCellKey, MeasConfig, MeasEventType, MeasurementManager, ReportTriggerConfig,
    ReportTriggerType,
};
use crate::rrc::reestablishment::{
    ReestablishmentProcedure, ReestablishmentState, ReestablishmentTrigger,
};
use crate::rrc::resume::ResumeProcedure;
use crate::rrc::security::{
    as_security_enabled, compute_short_mac_i, AsSecurityContext, AsSecurityError,
    CipheringAlgorithm, IntegrityAlgorithm, DIRECTION_DOWNLINK, SMC_PDCP_COUNT, SRB1_BEARER,
};
use crate::rrc::state::{RrcState, RrcStateMachine};
#[cfg(feature = "nextgsim-she")]
use crate::tasks::SheClientMessage;
#[cfg(feature = "nextgsim-isac")]
use crate::tasks::{IsacMeasurementType, IsacSensorMessage};
use crate::tasks::{NasMessage, RlfCause, RlsMessage, RrcMessage, Task, TaskMessage, UeTaskBase};
#[cfg(feature = "nextgsim-semantic")]
use crate::tasks::{SemanticCodecMessage, SemanticTaskType};
use nextgsim_common::frame_clock;
use nextgsim_common::OctetString;
use nextgsim_common::Plmn;
use nextgsim_pdcp::srb_security::MAC_I_LEN;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::codec::{decode_rrc, BCCH_DL_SCH_Message, CellGroupConfig, RadioBearerConfig};
use nextgsim_rrc::procedures::conditional_handover::decode_cho_config;
use nextgsim_rrc::procedures::measurement_report::{
    db_to_rsrq_range, db_to_sinr_range, dbm_to_eutra_rsrp_range, dbm_to_rsrp_range,
    encode_measurement_report, MeasCellResults, MeasResult2Nr, MeasResultCellNr, MeasResultEutra,
    MeasResultNr, MeasResultServFreqNr, MeasurementReportError, MeasurementReportParams,
};
use nextgsim_rrc::procedures::paging::{
    decode_paging, PagedUeIdentity, FIVE_G_S_TMSI_LEN, MAX_PAGE_RECORDS,
};
use nextgsim_rrc::procedures::paging_occasion::{
    paging_occasion, ue_id_from_s_tmsi, PagingCycleConfig, PagingOccasion,
};
use nextgsim_rrc::procedures::rrc_reestablishment::{
    decode_rrc_reestablishment, encode_rrc_reestablishment_complete,
    encode_rrc_reestablishment_request, phys_cell_id_from_nci, ReestablishmentCauseValue,
    ReestablishmentUeIdentity, RrcReestablishmentCompleteParams, RrcReestablishmentRequestParams,
};
use nextgsim_rrc::procedures::rrc_release::{decode_rrc_release, CellReselectionPrioritiesParams};
use nextgsim_rrc::procedures::rrc_setup::{
    decode_rrc_setup, encode_rrc_setup_complete, encode_rrc_setup_request,
    RrcEstablishmentCause as AsnEstablishmentCause, RrcSetupCompleteParams, RrcSetupData,
    RrcSetupRequestParams, SNssai as RrcSNssai, UeIdentity,
};
use nextgsim_rrc::procedures::scell_config::decode_scell_config;
use nextgsim_rrc::procedures::security_mode::{
    decode_security_mode_command, encode_security_mode_complete, encode_security_mode_failure,
    SecurityModeCommandData, SecurityModeCompleteParams, SecurityModeFailureParams,
};
use nextgsim_rrc::procedures::system_information::{
    decode_mib, is_system_information, parse_sib1, parse_system_information, CellBarredStatus,
    IntraFreqReselection, PlmnIdentity as SibPlmnIdentity,
};
use nextgsim_rrc::procedures::ue_capability::{
    build_minimal_nr_capability_container, decode_ue_capability_enquiry,
    encode_ue_capability_information, RatType, UeCapabilityInformationParams,
    UeCapabilityRatContainer,
};

/// Minimum acceptance window after a paging frame, in radio frames (issue #99).
///
/// The gNB transmits *in* the frame; the PDU then crosses a UDP transport, a
/// channel and a task queue before this UE decodes it. A tolerance smaller than
/// that delay drops conformant paging, and under load the delay is tens of
/// milliseconds rather than one frame -- a 20 ms window was tight enough to be
/// exceeded by scheduling jitter alone.
const PAGING_OCCASION_MIN_TOLERANCE_FRAMES: u16 = 4;

/// The acceptance window as a fraction of the DRX cycle: `T / 8`.
///
/// Proportional so the check's strictness is comparable across cycles -- a
/// 40 ms window is generous against a 320 ms cycle and tight against a 2.56 s one
/// -- with [`PAGING_OCCASION_MIN_TOLERANCE_FRAMES`] as the absolute floor for
/// transport delay. It still rejects seven eighths of the cycle, so a paging from
/// a foreign occasion is refused rather than the check being decorative.
const PAGING_OCCASION_TOLERANCE_DIVISOR: u16 = 8;

/// C-RNTI recorded in the AS security context for re-establishment ShortMAC-I
/// (TS 38.331 §5.3.7.4). The sim has no MAC-layer C-RNTI allocation, so a fixed
/// non-zero value is used; it only feeds the re-establishment MAC and is never
/// signalled on the wire.
const AS_SECURITY_C_RNTI: u16 = 0x4601;

/// Simplified DL/UL-DCCH envelope code: first byte 0x06 marks a UE capability
/// transfer message; the remaining bytes are the real ASN.1 UPER encoding of
/// UECapabilityEnquiry (DL) / UECapabilityInformation (UL).
const RRC_MSG_TYPE_UE_CAPABILITY: u8 = 0x06;

/// Default NR band advertised by the simulated UE (n78, 3.5 GHz TDD)
const DEFAULT_NR_BAND: u16 = 78;

/// Simplified DL-DCCH envelope code for an RRCReconfiguration carrying a
/// conditional-reconfiguration container: `[0x0D][transaction id][container]`.
///
/// The same hand-rolled framing family as the handover (0x00) and DAPS (0x10)
/// reconfigurations this task already parses; issue #107 covers replacing all of
/// them with real UPER, which for `conditionalReconfiguration` also needs a
/// Rel-16 RRC schema (issue #105).
const RECONFIGURATION_WITH_CHO: u8 = 0x0D;

/// Where the CHO container starts in such a message, after the envelope code and
/// the transaction identifier.
const CHO_CONTAINER_OFFSET: usize = 2;

/// Simplified DL-DCCH envelope code for an RRCReconfiguration carrying a
/// secondary-cell configuration: `[0x0E][transaction id][UPER CellGroupConfig]`.
///
/// Unlike the CHO container, the payload here is a **real UPER encoding**:
/// `sCellToAddModList` and `sCellToReleaseList` are Rel-15 IEs, so the vendored
/// schema models them (see `nextgsim_rrc::procedures::scell_config`). Only the
/// envelope is hand-rolled, the same arrangement as the UE capability transfer
/// (0x06). Issue #107 covers the envelope.
const RECONFIGURATION_WITH_SCELL: u8 = 0x0E;

/// Where the `CellGroupConfig` starts in such a message.
const SCELL_CONTAINER_OFFSET: usize = 2;

/// `measId` of the default inter-RAT B1 measurement, installed when
/// `UeConfig::eutra_neighbours` is non-empty. Distinct from the default A3
/// `measId` 1.
const DEFAULT_B1_MEAS_ID: u8 = 2;

/// `q-RxLevMin` assumed for a cell whose SIB1 omits `cellSelectionInfo`
/// (the IE is `OPTIONAL — Cond Standalone` in TS 38.331 §6.3.2).
const DEFAULT_Q_RX_LEV_MIN: i8 = -70;

/// Converts a broadcast SIB1 PLMN identity into the cell-selection PLMN type.
///
/// `None` when the MCC is absent (TS 38.331 allows it to be omitted, meaning
/// "same as the previous entry" — which this UE cannot resolve from a single
/// entry) or the digit count is not one a PLMN can have.
fn broadcast_plmn(identity: &SibPlmnIdentity) -> Option<CellPlmn> {
    let mcc_digits = identity.mcc.as_ref()?;
    let mcc =
        u16::from(mcc_digits[0]) * 100 + u16::from(mcc_digits[1]) * 10 + u16::from(mcc_digits[2]);
    let mnc = match identity.mnc.len() {
        2 => u16::from(identity.mnc[0]) * 10 + u16::from(identity.mnc[1]),
        3 => {
            u16::from(identity.mnc[0]) * 100
                + u16::from(identity.mnc[1]) * 10
                + u16::from(identity.mnc[2])
        }
        _ => return None,
    };
    // A two-digit MNC is NOT a three-digit one with a leading zero: 001-01 and
    // 001-001 are different networks, so the digit count is carried through.
    Some(CellPlmn::new(mcc, mnc, identity.mnc.len() == 3))
}

/// UAC barring configuration per 3GPP TS 38.331
/// Represents the uac-BarringInfoSetList from SIB1
#[derive(Debug, Clone)]
pub struct UacBarringConfig {
    /// Barring factor (0..95 in steps of 5, percent probability of being barred)
    /// 0 means no barring, 95 means 95% of attempts are barred
    pub barring_factor_percent: u8,
    /// Barring time in seconds (range: 5, 10, 20, 30, 60, 120, 240, 512)
    pub barring_time_secs: u16,
    /// Bitmask of access categories subject to barring (bit 0 = category 0, etc.)
    pub barring_for_access_category: u32,
}

impl Default for UacBarringConfig {
    fn default() -> Self {
        Self {
            barring_factor_percent: 0,
            barring_time_secs: 5,
            barring_for_access_category: 0,
        }
    }
}

/// RRC cycle interval in milliseconds
const RRC_CYCLE_INTERVAL_MS: u64 = 2500;

/// Cell selection interval in milliseconds
const CELL_SELECTION_INTERVAL_MS: u64 = 1000;

/// T300: RRCSetupRequest supervision timer default (ms), TS 38.331 §5.3.3.2.
/// SIB1 broadcasts the operative value (default 1000 ms); this is the fallback.
const T300_DEFAULT_MS: u64 = 1000;

/// UE-side NTN timing state
#[derive(Debug, Clone)]
pub struct UeNtnTiming {
    pub common_ta_us: u64,
    pub k_offset: u16,
    pub autonomous_ta: bool,
    pub max_doppler_hz: f64,
}

/// SRB1 state recorded from a decoded RRCSetup (Wave-6 C2).
///
/// Per TS 38.331 §5.3.3.4 the UE shall apply the RRCSetup's
/// radioBearerConfig and masterCellGroup; per §5.3.5.6.3 the RRCSetup
/// establishes SRB1 (SRB-Identity 1). All subsequent DCCH signalling
/// (SecurityModeCommand, RRCReconfiguration, ...) rides SRB1.
#[derive(Debug, Clone)]
pub struct Srb1Config {
    /// rrc-TransactionIdentifier of the RRCSetup that established SRB1
    pub rrc_transaction_id: u8,
    /// Decoded radioBearerConfig (carries srb-ToAddModList with SRB1)
    pub radio_bearer_config: RadioBearerConfig,
    /// Decoded masterCellGroup (`OCTET STRING (CONTAINING CellGroupConfig)`),
    /// when parseable — carries the RLC-BearerConfig (LCID 1) serving SRB1
    pub cell_group_config: Option<CellGroupConfig>,
}

/// RRC Task for managing cell selection and RRC connections
pub struct RrcTask {
    task_base: UeTaskBase,
    /// RRC state machine
    state_machine: RrcStateMachine,
    /// Cell selector for cell selection/reselection
    cell_selector: CellSelector,
    /// Measurement manager for connected state measurements
    measurement_manager: MeasurementManager,
    /// Handover manager for mobility
    handover_manager: HandoverManager,
    /// PDU ID counter for RRC messages
    pdu_id_counter: u32,
    /// Current serving cell ID
    serving_cell_id: Option<i32>,
    /// The last (PCI, RSRP dBm) pair reported to NAS for positioning (issue #46).
    ///
    /// Held so the report is sent on a CHANGE rather than once per RRC cycle. `None`
    /// means nothing has been reported yet, which is distinct from "the same as last
    /// time" -- the first measurement after a cell change must always go out.
    reported_serving_measurement: Option<(u16, i32)>,
    /// Pending NAS PDU for initial message
    initial_nas_pdu: Option<OctetString>,
    /// RRC establishment cause
    establishment_cause: i64,
    /// Last cell selection attempt time
    last_cell_selection: Option<Instant>,
    /// NTN timing advance state (if operating via satellite)
    ntn_timing: Option<UeNtnTiming>,
    /// UAC barring configuration from SIB1
    uac_barring: UacBarringConfig,
    /// RRC re-establishment procedure state
    reestablishment_proc: ReestablishmentProcedure,
    /// RRC resume procedure state
    resume_proc: ResumeProcedure,
    /// AS security context (set after AS Security Mode Command); required for
    /// ShortMAC-I derivation in re-establishment (TS 38.331 §5.3.7)
    as_security: Option<AsSecurityContext>,
    /// KgNB handed down from the NAS plane once NAS security is active
    /// (TS 33.501 §6.9.4.1: KgNB = KDF(KAMF, uplink NAS COUNT)). Consumed by the
    /// AS SecurityModeCommand handler to derive K_RRCint/K_RRCenc (Wave-6 I5).
    pending_kgnb: Option<[u8; 32]>,
    /// SRB1 configuration decoded from the RRCSetup (Wave-6 C2); `Some` once
    /// the RRCSetup's radioBearerConfig established SRB1 (TS 38.331
    /// §5.3.5.6.3) — recorded BEFORE any DL-DCCH (SRB1) message is handled
    srb1_config: Option<Srb1Config>,
    /// T300 establishment guard (TS 38.331 §5.3.3.2): deadline by which an
    /// RRCSetup must arrive after an RRCSetupRequest. `Some` while establishment
    /// is in flight; cleared on RRCSetup reception or on expiry.
    t300_deadline: Option<tokio::time::Instant>,
    /// Cells whose system information came from a real BCCH broadcast, so the
    /// simulated fallback must not overwrite it with the UE's own assumptions.
    cells_with_broadcast_si: std::collections::HashSet<i32>,
    /// Stored conditional reconfigurations (TS 38.331 §5.3.5.13), populated only
    /// when `UeConfig::conditional_handover` is set. Evaluated on every
    /// measurement tick; a triggered candidate executes through
    /// `handover_manager`.
    cond_reconfig: CondReconfigStore,
    /// The rrc-TransactionIdentifier of the RRCReconfiguration that carried the
    /// stored CHO container. §5.3.5.13.5 applies the candidate's own
    /// `condRRCReconfig`, whose transaction identifier this simulator's container
    /// does not carry (issue #107), so the carrying message's is echoed instead.
    cho_transaction_id: u8,
    /// The UE's own 5G-S-TMSI, handed down by the NAS plane once a 5G-GUTI is
    /// assigned (`RrcMessage::PagingIdentity`). PCCH `PagingRecord`s are matched
    /// against it (TS 38.331 §5.3.2.3). `None` until the UE has a GUTI, and no
    /// paging record can match then — the AS has no identity to compare.
    paging_s_tmsi: Option<[u8; FIVE_G_S_TMSI_LEN]>,
    /// Configured secondary cells (TS 38.331 §5.3.5.5.9), `sCellIndex` -> the
    /// simulator cell id its `physCellId` names. Released individually by
    /// `sCellToReleaseList` and wholesale on going to RRC_IDLE (§5.3.11).
    ///
    /// A `BTreeMap` because the *lowest* configured index is the one handed to
    /// the measurement manager: A6 is defined against the SCell of the
    /// `measObjectNR` associated with the event (§5.5.4.7), and this simulator's
    /// measurement configuration carries no measurement object, so it can model
    /// one A6 reference and not one per event. That is the ceiling of #112's
    /// "SCell half of CA, not the whole feature".
    configured_scells: std::collections::BTreeMap<u8, i32>,
}

/// Converts this UE's measurement report into the UPER `MeasurementReport` of
/// TS 38.331 §5.5.5 (issue #107).
///
/// A free function rather than a method: it is pure, and keeping it out of the
/// task means the encoding can be tested without constructing an `RrcTask`.
///
/// # The two quantity systems
///
/// The UE measures in **dBm**; `MeasurementReport` carries **RSRP-Range**
/// (`INTEGER (0..127)`, TS 38.133 Table 10.1.6.1-1). `dbm_to_rsrp_range` does the
/// conversion, and it is the one already in `nextgsim-rrc` rather than a second
/// copy — the gNB's parser uses `rsrp_range_to_dbm` to invert it, so a second
/// mapping is how the two ends would disagree.
///
/// # An absent measurement is absent, not zero
///
/// `RSRP-Range` 0 means "below -156 dBm", a real and very weak measurement, so a
/// cell whose RSRP the UE does not have reports **no** SSB results rather than
/// range 0. The old byte format substituted -120 dBm, which reported a plausible
/// level the UE had never measured.
fn build_uper_measurement_report(
    report: &crate::rrc::measurement::MeasurementReport,
) -> Result<Vec<u8>, MeasurementReportError> {
    fn cell_results(cell: &crate::rrc::measurement::CellMeasResult) -> Option<MeasCellResults> {
        // Only the quantities the UE actually has. All three absent means no
        // ssb-Results at all, which is what "not measured" looks like on the wire.
        let rsrp = cell.rsrp.map(|dbm| dbm_to_rsrp_range(f64::from(dbm)));
        let rsrq = cell.rsrq.map(|db| db_to_rsrq_range(f64::from(db)));
        let sinr = cell.sinr.map(|db| db_to_sinr_range(f64::from(db)));
        if rsrp.is_none() && rsrq.is_none() && sinr.is_none() {
            return None;
        }
        Some(MeasCellResults { rsrp, rsrq, sinr })
    }

    fn meas_result_nr(cell: &crate::rrc::measurement::CellMeasResult) -> MeasResultNr {
        MeasResultNr {
            // PhysCellId is INTEGER (0..1007); the UE's own `pci` is a u32 local
            // index, so a value past the ASN.1 bound would fail the encode. Reduced
            // the same way `phys_cell_id_from_nci` does, for the same reason.
            phys_cell_id: Some((cell.pci % 1008) as u16),
            cell_results: MeasResultCellNr {
                ssb_results: cell_results(cell),
                csi_rs_results: None,
            },
            rs_index_results: None,
        }
    }

    let serv_freq_results = vec![MeasResultServFreqNr {
        // The serving cell is always servCellIndex 0 here: this simulator models
        // one serving cell plus an optional SCell for measurement reference only
        // (issue #112), and the SCell is never the report's serving entry.
        serv_cell_index: 0,
        meas_result_serving_cell: meas_result_nr(&report.serving_cell),
        // `measResultBestNeighCell` is left absent rather than filled with the
        // strongest neighbour: it is the best neighbour ON THE SERVING FREQUENCY,
        // and the neighbour list below already carries every measured cell. Filling
        // it would report the same cell twice.
        meas_result_best_neigh_cell: None,
    }];

    // `measResultNeighCells` is a CHOICE, so a report carries NR results or E-UTRA
    // results and never both (TS 38.331 §5.5.5). An E-UTRA report is a B1/B2
    // report; anything else is intra-NR.
    let (neigh_freq_results, eutra_neigh_results) = if report.eutra_neighbor_cells.is_empty() {
        let nr = if report.neighbor_cells.is_empty() {
            Vec::new()
        } else {
            vec![MeasResult2Nr {
                ssb_frequency_arfcn: None,
                ref_freq_csi_rs: None,
                meas_result_list: report.neighbor_cells.iter().map(meas_result_nr).collect(),
            }]
        };
        (nr, Vec::new())
    } else {
        (
            Vec::new(),
            report
                .eutra_neighbor_cells
                .iter()
                .map(|c| {
                    MeasResultEutra::with_rsrp(
                        c.cell.pci,
                        // An E-UTRA neighbour with no RSRP reports range 0, which
                        // TS 36.133 defines as "below -140 dBm" -- the honest value
                        // for a cell the UE detected but could not measure.
                        c.rsrp
                            .map_or(0, |dbm| dbm_to_eutra_rsrp_range(f64::from(dbm))),
                    )
                })
                .collect(),
        )
    };

    encode_measurement_report(&MeasurementReportParams {
        meas_id: report.meas_id,
        serv_freq_results,
        neigh_freq_results,
        eutra_neigh_results,
        enhanced_quantities: None,
    })
}

impl RrcTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        // Get HPLMN from config
        let hplmn = task_base.config.hplmn;
        let selected_plmn = CellPlmn::new(hplmn.mcc, hplmn.mnc, hplmn.long_mnc);

        let mut cell_selector = CellSelector::new();
        cell_selector.set_selected_plmn(Some(selected_plmn));
        // SNPN (Rel-17, TS 23.501 §5.30): when configured, gate cell selection
        // on the SNPN NID so only the matching SNPN cell is selectable.
        if let Some(ref snpn) = task_base.config.snpn_config {
            cell_selector.set_required_nid(Some(snpn.nid.clone()));
        }

        Self {
            task_base,
            state_machine: RrcStateMachine::new(),
            cell_selector,
            measurement_manager: MeasurementManager::new(),
            handover_manager: HandoverManager::new(),
            pdu_id_counter: 0,
            serving_cell_id: None,
            reported_serving_measurement: None,
            initial_nas_pdu: None,
            establishment_cause: 3, // mo-Data
            last_cell_selection: None,
            ntn_timing: None,
            uac_barring: UacBarringConfig::default(),
            reestablishment_proc: ReestablishmentProcedure::new(),
            resume_proc: ResumeProcedure::new(),
            as_security: None,
            pending_kgnb: None,
            srb1_config: None,
            t300_deadline: None,
            cells_with_broadcast_si: std::collections::HashSet::new(),
            cond_reconfig: CondReconfigStore::new(),
            cho_transaction_id: 0,
            paging_s_tmsi: None,
            configured_scells: std::collections::BTreeMap::new(),
        }
    }

    /// Returns the SRB1 configuration recorded from the RRCSetup, if the
    /// serving gNB established SRB1 (Wave-6 C2, TS 38.331 §5.3.5.6.3).
    pub fn srb1_config(&self) -> Option<&Srb1Config> {
        self.srb1_config.as_ref()
    }

    /// Installs the AS security context (called once the AS Security Mode
    /// procedure derives KRRCint). Enables re-establishment with a real
    /// ShortMAC-I per TS 38.331 §5.3.7.4.
    pub fn set_as_security_context(&mut self, ctx: AsSecurityContext) {
        self.as_security = Some(ctx);
    }

    /// Installs the KgNB derived by the NAS plane once NAS security is active
    /// (TS 33.501 §6.9.4.1). The AS SecurityModeCommand handler consumes it to
    /// derive the RRC keys (Wave-6 I5). Called via `RrcMessage::AsSecurityKey`.
    pub fn set_pending_kgnb(&mut self, kgnb: [u8; 32]) {
        self.pending_kgnb = Some(kgnb);
    }

    /// Installs (or, with `None`, deletes) the 5G-S-TMSI the AS matches PCCH
    /// paging records against. Called via `RrcMessage::PagingIdentity` when the
    /// NAS plane is assigned or drops a 5G-GUTI (TS 24.501 §9.11.3.4).
    ///
    /// Public because it is a real message-handler entry point also driven
    /// directly by the in-process paging harness
    /// (`tests/src/paging_mt_service_request.rs`).
    pub fn set_paging_identity(&mut self, s_tmsi: Option<[u8; FIVE_G_S_TMSI_LEN]>) {
        match s_tmsi {
            Some(tmsi) => debug!("Paging identity installed: 5G-S-TMSI {:02x?}", tmsi),
            None => debug!("Paging identity deleted"),
        }
        self.paging_s_tmsi = s_tmsi;
    }

    /// Returns the installed AS security context, if AS security has been
    /// activated (test/observability hook, Wave-6 I5).
    #[cfg(test)]
    pub fn as_security(&self) -> Option<&AsSecurityContext> {
        self.as_security.as_ref()
    }

    /// Handle the AS SecurityModeCommand (TS 38.331 §5.3.4, TS 33.501 §6.7):
    /// derive K_RRCint/K_RRCenc from the pending KgNB and the gNB-selected
    /// algorithms (byte-identical to the gNB's own `derive_rrc_up_key`),
    /// install the AS security context, and reply SecurityModeComplete on SRB1.
    ///
    /// Fail-closed (TS 33.501 §6.7.2): with no integrity algorithm, an
    /// unsupported cipher (NEA3 has no keystream in the sim), or no KgNB yet,
    /// the UE does NOT activate AS security and sends no SecurityModeComplete
    /// (the network's SMC transaction times out). Only reached when the
    /// `UeConfig::as_security_enabled` wire gate is on.
    async fn handle_as_security_mode_command(
        &mut self,
        smc: SecurityModeCommandData,
        protected_bytes: &[u8],
        mac_i: [u8; 4],
    ) {
        let tid = smc.rrc_transaction_id;

        // Integrity protection is mandatory for AS security.
        let Some(integ_alg) = smc.security_algorithms.integrity_algorithm else {
            warn!(
                "AS SecurityModeCommand (tid {tid}) carries no integrity algorithm; not activating"
            );
            return;
        };
        let integrity = IntegrityAlgorithm::from(integ_alg);
        let ciphering = CipheringAlgorithm::from(smc.security_algorithms.ciphering_algorithm);

        // NEA3 is accepted. It used to be refused here with "unsupported
        // keystream", on the strength of a comment that was never true:
        // `zuc::nea3_encrypt` is complete and `nextgsim-nas` has used it for NAS
        // ciphering all along (issue #31, criterion 6). Refusing it aborted
        // activation against any peer that legitimately selected NEA3, which both
        // ends advertise.

        let Some(kgnb) = self.pending_kgnb else {
            warn!(
                "AS SecurityModeCommand (tid {tid}) received before KgNB is available \
                 (NAS security not active?); not activating (TS 33.501 §6.7)"
            );
            return;
        };

        let ctx =
            AsSecurityContext::derive_from_kgnb(&kgnb, ciphering, integrity, AS_SECURITY_C_RNTI);

        // TS 38.331 §5.3.4.2: verify the SecurityModeCommand's integrity with the
        // FRESHLY DERIVED K_RRCint before replying. On failure the UE continues
        // with its previous configuration and answers SecurityModeFailure
        // (§5.3.4.4).
        //
        // Before this, the command was accepted unverified: any peer that could put
        // bytes on SRB1 could activate AS security with keys of its choosing, and
        // the UE logged "AS security activated" for it.
        //
        // The SMC is integrity protected and NOT ciphered (§5.3.4.2), so the MAC-I
        // is verified over the plaintext with a null cipher — which is why the
        // integrity-only context below is built rather than reusing `ctx` whole.
        match self.verify_security_mode_command(&ctx, protected_bytes, mac_i) {
            Ok(()) => {}
            Err(e) => {
                warn!(
                    "AS SecurityModeCommand (tid {tid}) failed integrity verification \
                     ({e}); keeping the previous configuration and answering \
                     SecurityModeFailure (TS 38.331 §5.3.4.4)"
                );
                self.send_security_mode_failure(tid).await;
                return;
            }
        }

        info!(
            "AS security activated: integrity=NIA{}, ciphering=NEA{}, tid={}",
            integrity.id(),
            ciphering.id(),
            tid
        );
        self.set_as_security_context(ctx);
        self.send_security_mode_complete(tid).await;
    }

    /// Verifies a SecurityModeCommand's MAC-I with the freshly derived
    /// `K_RRCint` (TS 38.331 §5.3.4.2, issue #31 criterion 3).
    ///
    /// The SMC is integrity protected but **not ciphered**, so the check is run
    /// with a null cipher over the plaintext. A `Nia0` command is refused rather
    /// than accepted: NIA0's MAC is all zeros, so "the MAC verified" would prove
    /// nothing, and TS 33.501 §5.11.1 does not permit NIA0 for SRBs outside
    /// unauthenticated emergency service.
    fn verify_security_mode_command(
        &self,
        ctx: &AsSecurityContext,
        protected_bytes: &[u8],
        mac_i: [u8; 4],
    ) -> Result<(), AsSecurityError> {
        if ctx.integrity_algorithm == IntegrityAlgorithm::Nia0 {
            return Err(AsSecurityError::IntegrityCheckFailed);
        }
        // Integrity-only: the SMC is not ciphered, so the verifying context uses
        // NEA0 whatever cipher the command selected for later PDUs.
        let integrity_only = AsSecurityContext {
            ciphering_algorithm: CipheringAlgorithm::Nea0,
            ..ctx.clone()
        };
        let expected = integrity_only.compute_rrc_mac_i(
            SMC_PDCP_COUNT,
            SRB1_BEARER,
            DIRECTION_DOWNLINK,
            protected_bytes,
        );
        if expected == mac_i {
            Ok(())
        } else {
            Err(AsSecurityError::IntegrityCheckFailed)
        }
    }

    /// Sends a `SecurityModeFailure` (UL-DCCH, SRB1, TS 38.331 §5.3.4.4).
    ///
    /// Unprotected, deliberately: the UE has just established that it cannot agree
    /// with the gNB on keys, so protecting the refusal with those keys would make
    /// it unverifiable too.
    async fn send_security_mode_failure(&mut self, tid: u8) {
        match encode_security_mode_failure(&SecurityModeFailureParams {
            rrc_transaction_id: tid,
        }) {
            Ok(pdu) => {
                warn!("Sending AS SecurityModeFailure (tid={tid})");
                self.send_uplink_rrc(RrcChannel::UlDcch, OctetString::from_slice(&pdu))
                    .await;
            }
            Err(e) => error!("Failed to encode SecurityModeFailure: {e}"),
        }
    }

    /// Build and send the RRC SecurityModeComplete (UL-DCCH, SRB1) echoing the
    /// SecurityModeCommand transaction id (TS 38.331 §5.3.4.3).
    async fn send_security_mode_complete(&mut self, tid: u8) {
        match encode_security_mode_complete(&SecurityModeCompleteParams {
            rrc_transaction_id: tid,
        }) {
            Ok(bytes) => {
                info!("Sending AS SecurityModeComplete (tid={tid})");
                self.send_uplink_rrc(RrcChannel::UlDcch, OctetString::from_slice(&bytes))
                    .await;
            }
            Err(e) => error!("Failed to encode SecurityModeComplete: {e}"),
        }
    }

    /// Get the next PDU ID for RRC message tracking
    fn next_pdu_id(&mut self) -> u32 {
        self.pdu_id_counter = self.pdu_id_counter.wrapping_add(1);
        if self.pdu_id_counter == 0 {
            self.pdu_id_counter = 1;
        }
        self.pdu_id_counter
    }

    /// Check if the given cell is the active serving cell
    fn is_active_cell(&self, cell_id: i32) -> bool {
        self.serving_cell_id == Some(cell_id)
    }

    /// Perform the RRC cycle (cell selection in idle, measurements in connected)
    ///
    /// Public because it is a real message-handler entry point
    /// (`RrcMessage::TriggerCycle`) also driven directly by the in-process
    /// strict-peer harness (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub async fn perform_cycle(&mut self) {
        match self.state_machine.state() {
            RrcState::Idle | RrcState::Inactive => {
                self.perform_cell_selection().await;
            }
            RrcState::Connected => {
                // In connected state, perform measurements for handover
                self.perform_measurements().await;
            }
        }
    }

    /// Clears the `CELL_SELECTION_INTERVAL_MS` throttle so the next
    /// [`Self::perform_cycle`] re-evaluates immediately.
    ///
    /// Test-only. `perform_cell_selection` deliberately rate-limits itself, so
    /// without this a reselection test would have to sleep for the interval
    /// between every signal change — and a test that sleeps for real time is the
    /// kind that becomes flaky under load. The alternative was to make the
    /// interval configurable in production for a test's benefit, which is worse.
    #[cfg(test)]
    fn force_cell_selection_now(&mut self) {
        self.last_cell_selection = None;
    }

    /// Perform cell selection
    async fn perform_cell_selection(&mut self) {
        // Check if enough time has passed since last selection
        if let Some(last) = self.last_cell_selection {
            if last.elapsed() < Duration::from_millis(CELL_SELECTION_INTERVAL_MS) {
                return;
            }
        }
        self.last_cell_selection = Some(Instant::now());

        // Run cell selection algorithm
        if let Some(selected_cell) = self.cell_selector.perform_cell_selection() {
            let old_cell = self.serving_cell_id;
            let old_tai = self.get_current_tai();

            // Update serving cell
            self.serving_cell_id = Some(selected_cell.cell_id);

            info!(
                "Cell selection complete: cell_id={}, plmn={}, tac={}, category={:?}",
                selected_cell.cell_id,
                selected_cell.plmn,
                selected_cell.tac,
                selected_cell.category
            );

            // Notify RLS of the new serving cell
            if let Err(e) = self
                .task_base
                .rls_tx
                .send(RlsMessage::AssignCurrentCell {
                    cell_id: selected_cell.cell_id,
                })
                .await
            {
                error!("Failed to notify RLS of cell change: {}", e);
            }

            // If cell changed, notify NAS
            if old_cell != Some(selected_cell.cell_id) {
                if let Err(e) = self
                    .task_base
                    .nas_tx
                    .send(NasMessage::ActiveCellChanged {
                        previous_tai: old_tai,
                        // Report the PLMNs the UE can currently see so NAS can
                        // run TS 23.122 §4.4.3 selection over the real radio
                        // rather than a hardcoded home-PLMN list.
                        available_plmns: self.cell_selector.available_plmns(),
                        // The camped cell's own PLMN, which is what makes the
                        // REGISTERED PLMN knowable (issue #49). Always Some here,
                        // and that is a property of cell selection rather than an
                        // assumption: `perform_cell_selection` skips any cell whose
                        // SIB1 has not been read, so a selected cell's PLMN is one
                        // the cell BROADCAST rather than one the UE assumed.
                        serving_plmn: Some(selected_cell.plmn),
                        // TS 38.304 §4.4: a SUITABLE cell gives normal service, an
                        // ACCEPTABLE one only limited service. NAS decides which
                        // registration to attempt from this (issue #50).
                        cell_category: selected_cell.category,
                    })
                    .await
                {
                    error!("Failed to notify NAS of cell change: {}", e);
                }
            }
        }
    }

    /// Get the current TAI (Tracking Area Identity)
    fn get_current_tai(&self) -> nextgsim_common::types::Tai {
        if let Some(cell_id) = self.serving_cell_id {
            if let Some(cell) = self.cell_selector.get_cell(cell_id) {
                return nextgsim_common::types::Tai {
                    plmn: Plmn::new(
                        cell.sib1.plmn.mcc,
                        cell.sib1.plmn.mnc,
                        cell.sib1.plmn.long_mnc,
                    ),
                    tac: cell.sib1.tac,
                };
            }
        }
        nextgsim_common::types::Tai::default()
    }

    /// Perform measurements for handover (in connected state)
    async fn perform_measurements(&mut self) {
        // Update measurement manager with serving cell
        self.measurement_manager
            .set_serving_cell(self.serving_cell_id);

        // Update measurements from cell selector
        let cells = self.cell_selector.cells();
        for (&cell_id, cell) in cells.iter() {
            self.measurement_manager
                .update_measurement(cell_id, cell.dbm);
        }

        // Inter-RAT (E-UTRA) measurements, for events B1/B2. Configured rather
        // than measured: RLS carries NR cells only.
        self.update_eutra_measurements();

        // Evaluate measurement events
        self.measurement_manager.evaluate_events();

        // Report the serving cell to NAS, which is where an LPP E-CID request is
        // answered from (issue #46).
        self.report_serving_cell_measurement().await;

        // Process any pending measurement reports
        let reports = self.measurement_manager.take_pending_reports();
        for report in reports {
            self.send_measurement_report(&report).await;
        }

        // Conditional reconfiguration evaluation (TS 38.331 §5.3.5.13.4): a
        // stored candidate whose execution condition has held for its
        // time-to-trigger executes without the network ordering it.
        self.evaluate_conditional_handover().await;
    }

    /// Evaluate the stored conditional reconfigurations and execute a triggered
    /// candidate (TS 38.331 §5.3.5.13.4, §5.3.5.13.5).
    ///
    /// Nothing happens while a handover is already in progress: the UE is between
    /// cells, and §5.3.5.13.5's selection is over candidates of the source cell.
    async fn evaluate_conditional_handover(&mut self) {
        if self.cond_reconfig.candidate_count() == 0 || self.handover_manager.is_in_progress() {
            return;
        }
        let Some(triggered) = self.cond_reconfig.evaluate(&self.measurement_manager) else {
            return;
        };
        let Some(source_cell_id) = self.serving_cell_id else {
            return;
        };

        info!(
            "Executing conditional handover: candidate {}/{} -> cell {}",
            triggered.id.config_id,
            triggered.id.candidate_index,
            triggered.candidate.target_cell.phys_cell_id
        );
        let command = handover_command_for(&triggered.candidate, self.cho_transaction_id);
        // §5.3.5.3: applying a reconfigurationWithSync releases every stored
        // candidate, whether or not the execution below succeeds -- they were
        // prepared by the source cell.
        self.cond_reconfig.clear();
        self.handle_handover_command(source_cell_id, command).await;
    }

    /// Stored conditional-reconfiguration candidates, for tests and status
    /// reporting.
    pub fn cho_candidate_count(&self) -> usize {
        self.cond_reconfig.candidate_count()
    }

    /// Send measurement report to the network via RRC
    async fn send_measurement_report(
        &mut self,
        report: &crate::rrc::measurement::MeasurementReport,
    ) {
        // A real UPER `MeasurementReport` on UL-DCCH (TS 38.331 §5.5.5,
        // issue #107). This used to be a hand-built byte format --
        // `[0x0B, meas_id, pci_hi, pci_lo, rsrp, n, ...]` -- whose own comment said
        // "In real implementation, this would be proper ASN.1 encoding", while
        // `nextgsim-rrc`'s complete `measurement_report` codec sat with zero
        // callers. The gNB's parser flips in the SAME change, or measurements
        // stop arriving.
        let pdu = match build_uper_measurement_report(report) {
            Ok(bytes) => OctetString::from_slice(&bytes),
            Err(e) => {
                // Not a silent drop: a report that cannot be encoded is a
                // measurement the network will never see, and the event that
                // triggered it has already been consumed.
                // An inter-RAT (B1/B2) report reaches here every time: the
                // `measResultListEUTRA` arm of `measResultNeighCells` is past the
                // extension marker and `asn1-codecs` 0.7.2 refuses to encode an
                // extended CHOICE (issue #117). Named in the log, because
                // "measurement lost" without the reason reads as a transient fault
                // rather than a codec ceiling.
                error!(
                    "Failed to encode MeasurementReport (meas_id={}): {e}. The \
                     measurement is LOST. An inter-RAT (B1/B2) report cannot be \
                     encoded by this codec version at all -- see issue #117",
                    report.meas_id
                );
                return;
            }
        };
        // `triggered_cells` is the event's cellsTriggeredList (TS 38.331
        // §5.5.4.1); the neighbour list above leads with those cells, so the
        // count says how many of the reported neighbours the event fired on.
        info!(
            "Sending measurement report: meas_id={}, serving_rsrp={:?}, neighbors={}, triggered={}",
            report.meas_id,
            report.serving_cell.rsrp,
            report.neighbor_cells.len(),
            report.triggered_cells.len()
        );
        self.send_uplink_rrc(RrcChannel::UlDcch, pdu).await;
    }

    /// Configure measurements (called when receiving RRC Reconfiguration with measConfig)
    #[allow(dead_code)]
    fn configure_measurements(&mut self, config: MeasConfig) {
        info!(
            "Adding measurement config: meas_id={}, event={:?}",
            config.meas_id, config.trigger_config.trigger_type
        );
        self.measurement_manager.add_config(config);
    }

    /// Setup default A3 measurement for handover
    fn setup_default_measurements(&mut self) {
        // Default A3 event configuration for handover
        let config = MeasConfig {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            quantity: crate::rrc::measurement::MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::A3),
                threshold: None,
                threshold1: None,
                threshold2: None,
                a3_offset: Some(3), // Neighbor 3dB better than serving
                a6_offset: None,
                hysteresis: 2,        // 1dB hysteresis
                time_to_trigger: 640, // 640ms
            },
            report_amount: 8,
            report_interval: 480,
            max_report_cells: 4,
        };
        self.measurement_manager.add_config(config);

        // A configured inter-RAT neighbour list needs something evaluating it, or
        // the measurements are stored and never read. TS 38.331 §5.5.4.8: B1
        // enters when an E-UTRA neighbour beats b1-ThresholdEUTRA.
        if !self.task_base.config.eutra_neighbours.is_empty() {
            let threshold = self.task_base.config.eutra_b1_threshold_dbm;
            self.measurement_manager.add_config(MeasConfig {
                meas_id: DEFAULT_B1_MEAS_ID,
                meas_object_id: 2,
                report_config_id: 2,
                quantity: crate::rrc::measurement::MeasQuantity::SsRsrp,
                trigger_config: ReportTriggerConfig {
                    trigger_type: ReportTriggerType::Event(MeasEventType::B1),
                    threshold: Some(threshold),
                    threshold1: None,
                    threshold2: None,
                    a3_offset: None,
                    a6_offset: None,
                    hysteresis: 2,        // 1 dB
                    time_to_trigger: 640, // as for the A3 measId
                },
                report_amount: 8,
                report_interval: 480,
                max_report_cells: 4,
            });
            info!(
                "Inter-RAT measurement configured: {} E-UTRA neighbour(s), \
                 b1-ThresholdEUTRA {} dBm",
                self.task_base.config.eutra_neighbours.len(),
                threshold
            );
        }
    }

    /// Feed the configured inter-RAT (E-UTRA) neighbours into the measurement
    /// manager, so events B1 and B2 have a measurement source.
    ///
    /// A static stand-in, not a radio: RLS models NR cells only, so the level is
    /// whatever the configuration says and never changes. The offsets are set on
    /// every cycle alongside it because they are part of the B1/B2 inequalities
    /// and the manager stores them per cell and per carrier, not per report.
    fn update_eutra_measurements(&mut self) {
        for neighbour in &self.task_base.config.eutra_neighbours {
            let cell = EutraCellKey::new(neighbour.earfcn, neighbour.pci);
            self.measurement_manager
                .update_eutra_measurement(cell, neighbour.rsrp_dbm);
            self.measurement_manager
                .set_eutra_cell_offset(cell, neighbour.cell_individual_offset_db);
            self.measurement_manager
                .set_eutra_frequency_offset(neighbour.earfcn, neighbour.frequency_offset_db);
        }
    }

    /// Report the serving cell's PCI and level to NAS, when either has changed.
    ///
    /// The NAS task answers LPP location requests (TS 37.355 E-CID) and has no
    /// measurements of its own -- they live here, in the `MeasurementManager`. Sending
    /// only on a change keeps this to a trickle instead of one message per RRC cycle,
    /// and means the cached value at the NAS end is never older than the last time the
    /// radio actually moved.
    ///
    /// Nothing is sent while the serving cell has no measurement: NAS then reports no
    /// RSRP at all, which is the honest answer, rather than a level left over from a
    /// cell the UE is no longer on.
    async fn report_serving_cell_measurement(&mut self) {
        let Some(cell_id) = self.serving_cell_id else {
            return;
        };
        let Some(rsrp_dbm) = self.measurement_manager.rsrp(cell_id) else {
            return;
        };
        // The RLS cell id is the PCI throughout this simulator (see
        // `MeasurementManager::update_measurement`), and physCellId is INTEGER(0..503)
        // on the wire. A cell id outside that is reported once and then not sent,
        // because truncating it would name a DIFFERENT cell in the report.
        let Ok(phys_cell_id) = u16::try_from(cell_id) else {
            warn!("Serving cell id {cell_id} is negative; not reported to NAS for positioning");
            return;
        };
        if self.reported_serving_measurement == Some((phys_cell_id, rsrp_dbm)) {
            return;
        }
        self.reported_serving_measurement = Some((phys_cell_id, rsrp_dbm));
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::ServingCellMeasurement {
                phys_cell_id,
                rsrp_dbm,
            })
            .await
        {
            error!("Failed to report serving cell measurement to NAS: {}", e);
        }
    }

    /// Handle signal change from RLS
    ///
    /// Public because it is a real message-handler entry point
    /// (`RrcMessage::SignalChanged`) also driven directly by the in-process
    /// strict-peer harness (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub async fn handle_signal_changed(&mut self, cell_id: i32, dbm: i32) {
        let event = self.cell_selector.handle_signal_change(cell_id, dbm);

        match event {
            CellChangeEvent::CellDetected(id) => {
                debug!("Cell detected: cell_id={}", id);
                // In simulation, we need to provide system info for the cell
                // For now, use default values that make the cell selectable
                self.provide_simulated_system_info(id);
            }
            CellChangeEvent::CellLost(id) => {
                debug!("Cell lost: cell_id={}", id);
                // Remove from measurement manager
                self.measurement_manager.remove_measurement(id);
            }
            CellChangeEvent::ActiveCellLost(cell_info) => {
                warn!(
                    "Active cell lost: cell_id={}, triggering cell selection",
                    cell_info.cell_id
                );
                self.serving_cell_id = None;
                self.measurement_manager.set_serving_cell(None);
                // Trigger immediate cell selection
                self.last_cell_selection = None;
            }
            CellChangeEvent::SignalUpdated(id, dbm) => {
                debug!("Signal updated: cell_id={}, dbm={}", id, dbm);
                // Update measurement manager
                self.measurement_manager.update_measurement(id, dbm);
            }
            CellChangeEvent::None => {}
        }
    }

    /// Provide simulated system information for a cell.
    ///
    /// A stand-in for the real broadcast: it invents the cell's PLMN (this UE's
    /// own HPLMN), TAC and NCI, so the cell cannot fail to look suitable — the UE
    /// is checking values it made up. Kept because heartbeat-only discovery has
    /// nothing else to go on, and skipped in two cases:
    ///
    /// - the cell has broadcast real system information (#21), which must not be
    ///   overwritten by an assumption;
    /// - `require_broadcast_sib1` is configured, in which case the UE waits for
    ///   the real broadcast instead and an un-broadcasting cell never becomes
    ///   selectable.
    fn provide_simulated_system_info(&mut self, cell_id: i32) {
        if self.cells_with_broadcast_si.contains(&cell_id) {
            debug!("Cell {cell_id} broadcasts its own system info; not simulating it");
            return;
        }
        if self.task_base.config.require_broadcast_sib1 {
            debug!(
                "Cell {cell_id} has no broadcast SIB1 yet and require_broadcast_sib1 is set: \
                 waiting for the real broadcast"
            );
            return;
        }
        // Get HPLMN from config to use for the cell
        let hplmn = self.task_base.config.hplmn;

        // Update MIB - cell not barred
        let mib = MibInfo {
            has_mib: true,
            is_barred: false,
            is_intra_freq_reselect_allowed: true,
        };
        self.cell_selector.update_mib(cell_id, mib);

        // Update SIB1 with PLMN matching our home network
        let sib1 = Sib1Info {
            has_sib1: true,
            is_reserved: false,
            nci: cell_id as i64,
            tac: 1, // Default TAC
            plmn: CellPlmn::new(hplmn.mcc, hplmn.mnc, hplmn.long_mnc),
            q_rx_lev_min: -70, // Reasonable minimum
            q_rx_lev_min_offset: None,
            q_qual_min: None,
            // SNPN (Rel-17, TS 23.501 §5.30): the simulated cell broadcasts the
            // configured SNPN NID (the gNB SIB1 npn-IdentityInfoList in a real
            // deployment); None for a public cell.
            nid: self
                .task_base
                .config
                .snpn_config
                .as_ref()
                .map(|s| s.nid.clone()),
        };
        self.cell_selector.update_sib1(cell_id, sib1);

        debug!(
            "Simulated system info provided for cell {}: plmn={}-{}",
            cell_id, hplmn.mcc, hplmn.mnc
        );
    }

    /// Handle downlink RRC message from RLS
    ///
    /// Public because it is a real message-handler entry point
    /// (`RrcMessage::DownlinkRrcDelivery`) also driven directly by the
    /// in-process strict-peer harness (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub async fn handle_downlink_rrc(
        &mut self,
        cell_id: i32,
        channel: RrcChannel,
        pdu: OctetString,
    ) {
        if pdu.is_empty() {
            warn!("Empty downlink RRC PDU");
            return;
        }

        debug!(
            "Downlink RRC: cell_id={}, channel={:?}, len={}",
            cell_id,
            channel,
            pdu.len()
        );

        match channel {
            RrcChannel::DlCcch => {
                self.handle_dl_ccch_message(cell_id, &pdu).await;
            }
            RrcChannel::DlDcch => {
                self.handle_dl_dcch_message(cell_id, &pdu).await;
            }
            RrcChannel::Pcch => {
                self.handle_pcch_message(cell_id, &pdu).await;
            }
            RrcChannel::BcchBch => self.handle_broadcast_mib(cell_id, &pdu),
            RrcChannel::BcchDlSch => self.handle_broadcast_bcch_dl_sch(cell_id, &pdu),
            _ => {
                warn!("Unexpected downlink channel: {:?}", channel);
            }
        }
    }

    /// Handles a broadcast MIB on BCCH-BCH (TS 38.331 §5.2.1).
    ///
    /// The decoded `cellBarred` and `intraFreqReselection` replace the values the
    /// UE previously assumed for the cell.
    fn handle_broadcast_mib(&mut self, cell_id: i32, pdu: &OctetString) {
        let mib = match decode_mib(pdu.data()) {
            Ok(mib) => mib,
            Err(e) => {
                warn!("Failed to decode broadcast MIB from cell {cell_id}: {e}");
                return;
            }
        };
        let barred = mib.cell_barred == CellBarredStatus::Barred;
        let local_sfn = frame_clock::current_sfn();
        debug!(
            "MIB from cell {cell_id}: barred={barred}, sfn(msb6)={}, local sfn={local_sfn}",
            mib.system_frame_number
        );

        // The broadcast SFN is the CROSS-CHECK on the shared frame clock
        // (issue #99). Both ends derive the SFN from the wall clock, which is an
        // assumption rather than a protocol; the MIB's 6 MSBs are the one place it
        // can be tested at runtime, so a mismatch is reported loudly instead of
        // leaving paging to fail silently at the occasion check.
        if !frame_clock::msb6_matches(
            mib.system_frame_number,
            local_sfn,
            PAGING_OCCASION_MIN_TOLERANCE_FRAMES,
        ) {
            warn!(
                "Frame clock disagreement with cell {cell_id}: MIB says SFN>>4 = {}, this UE \
                 derives {} from its own clock. Paging occasions will not line up; the two \
                 processes are not sharing a clock (different hosts?).",
                mib.system_frame_number,
                frame_clock::sfn_msb6(local_sfn)
            );
        }
        self.cells_with_broadcast_si.insert(cell_id);
        self.cell_selector.update_mib(
            cell_id,
            MibInfo {
                has_mib: true,
                is_barred: barred,
                is_intra_freq_reselect_allowed: mib.intra_freq_reselection
                    == IntraFreqReselection::Allowed,
            },
        );
    }

    /// Dispatches a BCCH-DL-SCH PDU on the arm of its `BCCH-DL-SCH-Message`
    /// CHOICE (issue #50).
    ///
    /// SIB1 and `SystemInformation` (which carries SIB2/SIB3/SIB4) share this
    /// channel. Before SIB2/3/4 existed the handler called `decode_sib1`
    /// unconditionally, so every `SystemInformation` broadcast would have been
    /// logged as a SIB1 decode failure -- a warning per SI period, and the
    /// reselection parameters silently never read.
    fn handle_broadcast_bcch_dl_sch(&mut self, cell_id: i32, pdu: &OctetString) {
        let msg: BCCH_DL_SCH_Message = match decode_rrc(pdu.data()) {
            Ok(msg) => msg,
            Err(e) => {
                warn!("Failed to decode BCCH-DL-SCH PDU from cell {cell_id}: {e}");
                return;
            }
        };
        if is_system_information(&msg) {
            self.handle_broadcast_system_information(cell_id, &msg);
        } else {
            self.handle_broadcast_sib1_message(cell_id, &msg);
        }
    }

    /// Handles a broadcast `SystemInformation` carrying SIB2/SIB3/SIB4
    /// (TS 38.331 §6.3.1, issue #50).
    ///
    /// This is where the UE stops assuming its reselection parameters. Before
    /// this, `Q_hyst` and `Treselection` were the compile-time constants
    /// `DEFAULT_Q_HYST_DB` and `CELL_RESELECTION_TIME_TO_TRIGGER_MS`, and there
    /// was no per-cell `Qoffset` or reselection priority at all.
    ///
    /// Only the SERVING cell's SI is applied. A neighbour's SIB2 describes that
    /// neighbour's reselection behaviour, and applying it would let whichever cell
    /// broadcast most recently set this UE's hysteresis. While no cell is camped
    /// the first SI read is applied, because a UE that has not camped yet has no
    /// serving cell to prefer and needs some parameters to camp with.
    fn handle_broadcast_system_information(&mut self, cell_id: i32, msg: &BCCH_DL_SCH_Message) {
        let serving = self.serving_cell_id;
        if serving.is_some_and(|serving| serving != cell_id) {
            debug!(
                "Ignoring SystemInformation from non-serving cell {cell_id} \
                 (camped on {serving:?}): a neighbour's SIB2 describes that \
                 neighbour's reselection behaviour, not this UE's"
            );
            return;
        }

        let si = match parse_system_information(msg) {
            Ok(si) => si,
            Err(e) => {
                warn!("Failed to parse SystemInformation from cell {cell_id}: {e}");
                return;
            }
        };
        if si.is_empty() {
            // Legal: a cell may schedule only SIB5..SIB9 in a message. Not a
            // warning, or a cell that broadcast SIB5 would log an error per period.
            debug!("SystemInformation from cell {cell_id} carried no SIB this UE models");
            return;
        }

        if let Some(sib2) = si.sib2.as_ref() {
            info!(
                "SIB2 from cell {cell_id}: q-Hyst={} dB, t-ReselectionNR={} s, \
                 cellReselectionPriority={} (reselection parameters now come from \
                 the network, not from this UE's constants)",
                sib2.q_hyst_db, sib2.t_reselection_s, sib2.cell_reselection_priority
            );
            self.cell_selector.apply_sib2(
                sib2.q_hyst_db,
                sib2.t_reselection_s,
                sib2.cell_reselection_priority,
            );
        }
        if let Some(sib3) = si.sib3.as_ref() {
            let offsets: Vec<(u16, i32)> = sib3
                .intra_freq_neighbours
                .iter()
                .map(|n| (n.phys_cell_id, n.q_offset_db))
                .collect();
            debug!(
                "SIB3 from cell {cell_id}: {} intra-frequency neighbour offsets",
                offsets.len()
            );
            self.cell_selector.apply_sib3(&offsets);
        }
        if let Some(sib4) = si.sib4.as_ref() {
            let carriers: Vec<(u32, u8)> = sib4
                .inter_freq_carriers
                .iter()
                .map(|c| (c.dl_carrier_freq, c.cell_reselection_priority))
                .collect();
            debug!(
                "SIB4 from cell {cell_id}: {} inter-frequency carrier priorities",
                carriers.len()
            );
            self.cell_selector.apply_sib4(&carriers);
        }
    }

    /// Handles a broadcast SIB1 on BCCH-DL-SCH (TS 38.331 §6.3.2).
    ///
    /// This is where the UE learns what the cell actually is: its PLMN, TAC and
    /// NR Cell Identity, rather than the values `provide_simulated_system_info`
    /// invents from the UE's own configuration.
    fn handle_broadcast_sib1_message(&mut self, cell_id: i32, msg: &BCCH_DL_SCH_Message) {
        let sib1 = match parse_sib1(msg) {
            Ok(sib1) => sib1,
            Err(e) => {
                warn!("Failed to parse broadcast SIB1 from cell {cell_id}: {e}");
                return;
            }
        };
        let Some(info) = sib1.plmn_identity_info_list.first() else {
            warn!("Broadcast SIB1 from cell {cell_id} carries no PLMN identity info");
            return;
        };
        let Some(plmn) = info.plmn_identity_list.first().and_then(broadcast_plmn) else {
            warn!("Broadcast SIB1 from cell {cell_id} carries no usable PLMN");
            return;
        };

        let q_rx_lev_min = sib1
            .cell_selection_info
            .as_ref()
            .map_or(DEFAULT_Q_RX_LEV_MIN, |info| info.q_rx_lev_min);
        info!(
            "SIB1 from cell {cell_id}: plmn={}-{}, tac={:?}, nci={:#x}",
            plmn.mcc, plmn.mnc, info.tracking_area_code, info.cell_identity
        );
        self.cells_with_broadcast_si.insert(cell_id);
        self.cell_selector.update_sib1(
            cell_id,
            Sib1Info {
                has_sib1: true,
                is_reserved: false,
                nci: info.cell_identity as i64,
                tac: info.tracking_area_code.unwrap_or(0),
                plmn,
                q_rx_lev_min,
                q_rx_lev_min_offset: None,
                q_qual_min: None,
                // SIB1 npn-IdentityInfoList (Rel-16 SNPN) is not in the Rel-15
                // schema this tree compiles, so a broadcast cell is public here.
                nid: None,
            },
        );
    }

    /// Handles a PCCH `Paging` message (TS 38.331 §5.3.2.3).
    ///
    /// The UE monitors PCCH in RRC_IDLE and RRC_INACTIVE only (TS 38.304 §7.1);
    /// a paging message received while RRC_CONNECTED is discarded, since a
    /// connected UE is reached over its own SRB and answering it would start a
    /// service request for a connection it already has.
    ///
    /// Each `PagingRecord` is compared with the UE's own 5G-S-TMSI, and only the
    /// records that match are reported to NAS. A message with no matching record
    /// is a normal event — PCCH is a broadcast channel, so most paging a UE
    /// receives is for somebody else.
    /// This UE's paging occasion (TS 38.304 §7.1), or `None` before it has a
    /// 5G-S-TMSI to derive `UE_ID` from.
    ///
    /// The DRX cycle comes from configuration, which has to match the cell's:
    /// this UE never receives `PCCH-Config` because SIB1's `pcch-Config` is not
    /// modelled, so the two ends agree by configuration rather than by signalling.
    /// Recorded here because a mismatch would look like paging being dropped at
    /// random.
    fn paging_occasion(&self) -> Option<PagingOccasion> {
        let s_tmsi = self.paging_s_tmsi?;
        let t = self.task_base.config.paging_default_cycle_frames;
        let config = PagingCycleConfig::with_default_spreading(t).ok()?;
        Some(paging_occasion(ue_id_from_s_tmsi(&s_tmsi), &config))
    }

    /// Whether the current frame is within this UE's paging occasion.
    fn is_own_paging_occasion(&self) -> bool {
        self.is_own_paging_occasion_at(frame_clock::current_sfn())
    }

    /// Whether `sfn` is within this UE's paging occasion.
    ///
    /// The SFN is a parameter so the decision is deterministically testable; the
    /// live-clock caller above is the only production path.
    ///
    /// A UE with no 5G-S-TMSI has no occasion, and answers `true`: it cannot be
    /// the addressee of a matching record anyway (the identity check has already
    /// returned), so refusing here would only mask that.
    pub(crate) fn is_own_paging_occasion_at(&self, sfn: u16) -> bool {
        match self.paging_occasion() {
            Some(occasion) => {
                let tolerance = (occasion.t() / PAGING_OCCASION_TOLERANCE_DIVISOR)
                    .max(PAGING_OCCASION_MIN_TOLERANCE_FRAMES);
                occasion.is_within_occasion(sfn, tolerance)
            }
            None => true,
        }
    }

    async fn handle_pcch_message(&mut self, cell_id: i32, pdu: &OctetString) {
        let state = self.state_machine.state();
        if state == RrcState::Connected {
            debug!("PCCH Paging ignored: UE is RRC_CONNECTED");
            return;
        }

        let records = match decode_paging(pdu.data()) {
            Ok(records) => records,
            Err(e) => {
                warn!("Failed to decode PCCH Paging from cell {cell_id}: {e}");
                return;
            }
        };

        let Some(own_s_tmsi) = self.paging_s_tmsi else {
            debug!(
                "PCCH Paging with {} record(s) ignored: UE has no 5G-S-TMSI yet",
                records.len()
            );
            return;
        };

        let matched: Vec<[u8; FIVE_G_S_TMSI_LEN]> = records
            .iter()
            .filter_map(|record| match record.ue_identity {
                PagedUeIdentity::FiveGSTmsi(s_tmsi) if s_tmsi == own_s_tmsi => Some(s_tmsi),
                // A full I-RNTI pages a UE in RRC_INACTIVE by an identity the
                // NG-RAN allocated in the RRCRelease with suspendConfig, which
                // this UE never receives (RRC_INACTIVE is not reachable yet), so
                // it cannot be this UE.
                _ => None,
            })
            .collect();

        if matched.is_empty() {
            debug!(
                "PCCH Paging from cell {cell_id}: none of {} record(s) match this UE",
                records.len().min(MAX_PAGE_RECORDS)
            );
            return;
        }

        // TS 38.304 §7.1: a UE monitors ONE paging occasion per DRX cycle, so a
        // record addressed to this UE arriving outside its own occasion did not
        // come from a conformant network and is not acted on (issue #99). Before
        // this the UE accepted a matching record whenever it arrived, so the
        // occasion could not be got wrong.
        if !self.is_own_paging_occasion() {
            let occasion = self.paging_occasion();
            debug!(
                "PCCH Paging from cell {cell_id} matched this UE but arrived outside its \
                 paging occasion (SFN {}, PF residue {:?} mod {:?}); ignored",
                frame_clock::current_sfn(),
                occasion.map(|o| o.pf_residue()),
                occasion.map(|o| o.t())
            );
            return;
        }

        info!(
            "Paged by cell {cell_id} in {state}: 5G-S-TMSI {:02x?}",
            own_s_tmsi
        );

        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::Paging {
                paging_s_tmsi: matched,
            })
            .await
        {
            error!("Failed to deliver paging indication to NAS: {}", e);
        }
    }

    /// Handle DL-CCCH message (RRC Setup, RRC Reject)
    async fn handle_dl_ccch_message(&mut self, cell_id: i32, pdu: &OctetString) {
        let bytes = pdu.data();
        if bytes.is_empty() {
            return;
        }

        // Simplified parsing: check message type
        let msg_type = bytes[0] & 0x0F;

        match msg_type {
            0x00 => {
                // RRC Setup
                info!("Received RRC Setup from cell {}", cell_id);
                // If a re-establishment was in progress, this is the fallback path
                if self.reestablishment_proc.is_in_progress() {
                    self.reestablishment_proc.on_rrc_setup_fallback();
                }
                self.handle_rrc_setup(cell_id, pdu).await;
            }
            0x01 => {
                // RRC Reject
                warn!("Received RRC Reject from cell {}", cell_id);
                self.handle_rrc_reject(cell_id).await;
            }
            // No DL-CCCH arm for an RRCReestablishment. There was one, on low
            // nibble 0x02, and it never matched: the gNB's bespoke reply led with
            // 0x24, whose low nibble is 0x04. TS 38.331 §6.2.1 puts
            // RRCReestablishment on DL-DCCH/SRB1 anyway, which is where it is
            // handled now (issue #37).
            _ => {
                debug!("Unhandled DL-CCCH message type: {:#x}", msg_type);
            }
        }
    }

    /// Handle DL-DCCH message (DL Information Transfer, RRC Release, etc.)
    async fn handle_dl_dcch_message(&mut self, cell_id: i32, pdu: &OctetString) {
        if !self.is_active_cell(cell_id) {
            debug!("Ignoring DL-DCCH from non-active cell {}", cell_id);
            return;
        }

        // Wave-6 C2 model assertion: DCCH signalling rides SRB1, which the
        // RRCSetup must have established (TS 38.331 §5.3.5.6.3) and which
        // handle_rrc_setup records BEFORE any DL-DCCH message (e.g. the
        // SecurityModeCommand) is handled. Tolerated as a warning until
        // C5/C6 make DCCH framing fail-closed.
        if self.srb1_config.is_none() {
            warn!(
                "DL-DCCH message from cell {} but no SRB1 was recorded from \
                 the RRCSetup (peer sent no srb-ToAddModList?)",
                cell_id
            );
        }

        let bytes = pdu.data();
        if bytes.is_empty() {
            return;
        }

        // Wave-6 I5 (TS 38.331 §5.3.4): when the AS-security wire gate is on,
        // recognise the AS SecurityModeCommand by a typed DL-DCCH decode BEFORE
        // the legacy nibble dispatcher. The gNB sends the SMC as raw UPER whose
        // leading byte 0x20 has low-nibble 0x0 — which the legacy matcher below
        // would misroute to the RRCReconfiguration arm. Default-off: the
        // matched-sim path is unchanged (this whole block is skipped).
        if as_security_enabled(&self.task_base.config) && bytes.len() > MAC_I_LEN {
            // The SecurityModeCommand arrives INTEGRITY PROTECTED and unciphered
            // (TS 38.331 §5.3.4.2), so the PDCP payload is `smc_uper || MAC-I`.
            // Split explicitly rather than relying on the UPER decoder to tolerate
            // trailing octets: the MAC-I must be verified over exactly the bytes the
            // gNB signed, and "whatever the decoder did not consume" is not that.
            let (pdu, mac) = bytes.split_at(bytes.len() - MAC_I_LEN);
            if let Ok(smc) = decode_security_mode_command(pdu) {
                info!(
                    "Received AS SecurityModeCommand from cell {} (tid {})",
                    cell_id, smc.rrc_transaction_id
                );
                let mac_i: [u8; MAC_I_LEN] =
                    mac.try_into().expect("split_at yields exactly MAC_I_LEN");
                self.handle_as_security_mode_command(smc, pdu, mac_i).await;
                return;
            }
        }

        // RRCReestablishment (TS 38.331 §6.2.1: DL-DCCH / SRB1), decoded rather
        // than nibble-matched. Tried before the legacy matcher because its real
        // UPER leading byte would otherwise be misrouted.
        if let Ok(reestablishment) = decode_rrc_reestablishment(bytes) {
            self.handle_rrc_reestablishment(cell_id, &reestablishment)
                .await;
            return;
        }

        let msg_type = bytes[0] & 0x0F;

        match msg_type {
            0x04 => {
                // DL Information Transfer - forward NAS to NAS task
                if bytes.len() > 3 {
                    let nas_pdu = OctetString::from_slice(&bytes[3..]);
                    self.forward_nas_to_nas_task(nas_pdu).await;
                }
            }
            0x0D => {
                // RRC Release
                info!("Received RRC Release from cell {}", cell_id);
                // Dedicated cellReselectionPriorities, when the release carries
                // them (TS 38.331 §6.3.2, TS 38.304 §5.2.4.1, issue #50). Applied
                // BEFORE the release is processed, because the list governs the
                // idle mode the UE is about to enter.
                //
                // A decode failure is not reported as an error: the gNB falls
                // back to a hand-built byte PDU when UPER encoding fails, and that
                // PDU is a legitimate release this UE must still act on. What is
                // lost in that case is only the optional IE.
                if let Ok(release) = decode_rrc_release(bytes) {
                    self.apply_dedicated_reselection_priorities(
                        release.cell_reselection_priorities,
                    );
                }
                // If a resume was in progress, the network is rejecting it
                if self.resume_proc.is_in_progress() {
                    self.resume_proc
                        .on_release_received(&mut self.state_machine);
                } else {
                    self.handle_rrc_release().await;
                }
            }
            0x00 => {
                // RRC Reconfiguration
                debug!("Received RRC Reconfiguration from cell {}", cell_id);
                self.handle_rrc_reconfiguration(cell_id, pdu).await;
            }
            RRC_MSG_TYPE_UE_CAPABILITY => {
                // UECapabilityEnquiry: envelope byte + ASN.1 UPER message
                info!("Received UECapabilityEnquiry from cell {}", cell_id);
                self.handle_ue_capability_enquiry(&bytes[1..]).await;
            }
            0x08 => {
                // RRCResume (first byte 0x28; low nibble 0x08)
                // gNB encodes: bytes[0]=0x28, bytes[1]=transaction_id
                info!("Received RRCResume from cell {}", cell_id);
                let rrc_transaction_id = if bytes.len() > 1 { bytes[1] } else { 0 };

                match self.resume_proc.on_resume_received(
                    rrc_transaction_id,
                    None,
                    &mut self.state_machine,
                ) {
                    Ok(complete_params) => {
                        // Send RRCResumeComplete
                        // Encoding mirrors gNB: first byte indicates ResumeComplete
                        let mut rrc_complete = Vec::with_capacity(4);
                        rrc_complete.push(0x08); // RRCResumeComplete message type
                        rrc_complete.push(complete_params.rrc_transaction_id);
                        rrc_complete.push(0x00); // criticalExtensions placeholder
                        if let Some(nas) = complete_params.dedicated_nas_message {
                            rrc_complete.extend_from_slice(&nas);
                        }
                        let rrc_pdu = OctetString::from_slice(&rrc_complete);
                        self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;

                        // Notify NAS that connection is back
                        if let Err(e) = self
                            .task_base
                            .nas_tx
                            .send(NasMessage::RrcConnectionSetup)
                            .await
                        {
                            error!("Failed to notify NAS after RRC resume: {}", e);
                        }
                        info!("RRC resume complete on cell {}", cell_id);
                    }
                    Err(e) => {
                        warn!("RRCResume handling failed ({}), falling back to idle", e);
                        // Resume failed; ensure NAS is told the connection is gone
                        if let Err(ne) = self
                            .task_base
                            .nas_tx
                            .send(NasMessage::RrcEstablishmentFailure)
                            .await
                        {
                            error!("Failed to send RrcEstablishmentFailure: {}", ne);
                        }
                    }
                }
            }
            _ => {
                // Check if this is a raw NAS PDU (EPD = 0x7E or 0x2E)
                if bytes.len() >= 2 && (bytes[0] == 0x7E || bytes[0] == 0x2E) {
                    debug!("Received raw NAS PDU, forwarding to NAS task");
                    self.forward_nas_to_nas_task(pdu.clone()).await;
                } else {
                    debug!("Unhandled DL-DCCH message type: {:#x}", msg_type);
                }
            }
        }
    }

    /// Handle RRC Setup message
    async fn handle_rrc_setup(&mut self, cell_id: i32, pdu: &OctetString) {
        // RRCSetup received: stop the T300 establishment guard (TS 38.331
        // §5.3.3.4) before applying the configuration.
        self.stop_t300();

        // Wave-6 C2: decode the RRCSetup payload — per TS 38.331 §5.3.3.4
        // the UE shall apply the radioBearerConfig (SRB1 establishment,
        // §5.3.5.6.3) and the masterCellGroup. Decode failure is TOLERATED
        // (warn + verbatim legacy behavior below) so the matched sim stays
        // default-safe — no config flag.
        match decode_rrc_setup(pdu.data()) {
            Ok(setup) => self.apply_rrc_setup_config(&setup),
            Err(e) => warn!(
                "RRCSetup payload not decodable as ASN.1 DL-CCCH ({}); \
                 proceeding with legacy connection setup",
                e
            ),
        }

        // Transition to connected state
        if let Err(e) = self.state_machine.on_rrc_setup() {
            warn!("Failed to transition to connected state: {}", e);
            return;
        }

        self.serving_cell_id = Some(cell_id);
        info!("RRC connection established with cell {}", cell_id);

        // Set up default measurements for handover
        self.setup_default_measurements();
        self.measurement_manager.set_serving_cell(Some(cell_id));

        // Send RRC Setup Complete with initial NAS
        self.send_rrc_setup_complete().await;

        // Notify NAS of connection setup
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcConnectionSetup)
            .await
        {
            error!("Failed to notify NAS of RRC setup: {}", e);
        }
    }

    /// Applies the decoded RRCSetup configuration (Wave-6 C2, TS 38.331
    /// §5.3.3.4): records the transaction id, the SRB1 RadioBearerConfig and
    /// the decoded CellGroupConfig on the UE RRC state — BEFORE any SRB1
    /// (DL-DCCH) message can be handled.
    ///
    /// Tolerance: a decodable RRCSetup that does NOT establish SRB1 (e.g. the
    /// pre-Wave-6 placeholder `[0x20, 0x00, 0x04, 0x00]`) is warned about and
    /// records nothing; external behavior is unchanged.
    fn apply_rrc_setup_config(&mut self, setup: &RrcSetupData) {
        let radio_bearer_config: RadioBearerConfig = match decode_rrc(&setup.radio_bearer_config) {
            Ok(config) => config,
            Err(e) => {
                warn!(
                    "RRCSetup radioBearerConfig not decodable ({}); tolerated",
                    e
                );
                return;
            }
        };

        let has_srb1 = radio_bearer_config
            .srb_to_add_mod_list
            .as_ref()
            .is_some_and(|list| list.0.iter().any(|srb| srb.srb_identity.0 == 1));
        if !has_srb1 {
            warn!(
                "RRCSetup radioBearerConfig does not establish SRB1 \
                 (TS 38.331 §5.3.5.6.3 violation by the peer); tolerated, no SRB1 recorded"
            );
            return;
        }

        // masterCellGroup is OCTET STRING (CONTAINING CellGroupConfig).
        let cell_group_config: Option<CellGroupConfig> = match decode_rrc(&setup.master_cell_group)
        {
            Ok(config) => Some(config),
            Err(e) => {
                warn!(
                    "RRCSetup masterCellGroup not decodable as CellGroupConfig ({}); \
                     SRB1 recorded from radioBearerConfig only",
                    e
                );
                None
            }
        };

        let lcid = cell_group_config.as_ref().and_then(|cgc| {
            cgc.rlc_bearer_to_add_mod_list
                .as_ref()
                .and_then(|list| list.0.first())
                .map(|bearer| bearer.logical_channel_identity.0)
        });

        info!(
            "SRB1 established (LCID {}), rrc_transaction_id={}",
            lcid.unwrap_or(1),
            setup.rrc_transaction_id
        );

        self.srb1_config = Some(Srb1Config {
            rrc_transaction_id: setup.rrc_transaction_id,
            radio_bearer_config,
            cell_group_config,
        });
    }

    /// Send RRC Setup Complete message using proper ASN.1 UPER encoding
    async fn send_rrc_setup_complete(&mut self) {
        let nas_pdu = if let Some(pdu) = self.initial_nas_pdu.take() {
            pdu
        } else {
            warn!("send_rrc_setup_complete called but initial_nas_pdu is None — ignoring");
            return;
        };
        let nas_data = nas_pdu.data().to_vec();

        // RedCap (Reduced Capability) indication driven by UE config (Rel-17,
        // TS 38.331 §6.2.2). When set, the gNB caps this UE's serving
        // bandwidth and the core reduces the session-AMBR.
        let redcap_indication = self.task_base.config.redcap;
        if redcap_indication {
            info!("Signalling RedCap (Reduced Capability) indication in RRCSetupComplete");
        }

        // Advertise the UE's configured S-NSSAI(s) in RRCSetupComplete so the
        // gNB can perform slice-aware AMF selection (TS 38.331 §6.2.2,
        // TS 38.413 §8.6.1.2). An empty configured NSSAI signals none.
        let configured = &self.task_base.config.configured_nssai.slices;
        let s_nssai_list: Option<Vec<RrcSNssai>> = if configured.is_empty() {
            None
        } else {
            Some(
                configured
                    .iter()
                    .map(|s| RrcSNssai {
                        sst: s.sst,
                        sd: s.sd_as_u32(),
                    })
                    .collect(),
            )
        };
        if let Some(ref list) = s_nssai_list {
            info!(
                "Signalling {} configured S-NSSAI(s) in RRCSetupComplete",
                list.len()
            );
        }

        let params = RrcSetupCompleteParams {
            registered_amf: None,
            // Wave-6 C2/C4-interim: the echoed tid stays PINNED to 0 until C5
            // lands typed UL-DCCH dispatch on both peers. Echoing a received
            // tid of 2 would make the ASN.1 UL-DCCH leading byte 0x14, whose
            // low nibble 0x4 hits the gNB's bespoke fallback
            // handle_rrc_setup_complete arm if the ASN.1-first decode is ever
            // bypassed, mis-extracting garbage NAS from bytes[3..]. Unpinned
            // in C4-final (after C5); the gNB ignores the echoed tid today.
            rrc_transaction_id: 0,
            selected_plmn_identity: 1,
            guami_type: None,
            s_nssai_list,
            dedicated_nas_message: nas_data.clone(),
            ng_5g_s_tmsi_value: None,
            redcap_indication,
        };

        let pdu = match encode_rrc_setup_complete(&params) {
            Ok(bytes) => {
                debug!("ASN.1 RRCSetupComplete encoded: {} bytes", bytes.len());
                OctetString::from_slice(&bytes)
            }
            Err(e) => {
                warn!(
                    "ASN.1 RRCSetupComplete encoding failed ({}), using fallback",
                    e
                );
                let mut rrc_pdu = Vec::with_capacity(nas_data.len() + 3);
                rrc_pdu.push(0x04);
                rrc_pdu.push(0x00);
                rrc_pdu.push(0x01);
                rrc_pdu.extend_from_slice(&nas_data);
                OctetString::from_slice(&rrc_pdu)
            }
        };

        self.send_uplink_rrc(RrcChannel::UlDcch, pdu).await;
    }

    /// Handle RRC Reject message
    async fn handle_rrc_reject(&mut self, _cell_id: i32) {
        // RRCReject ends the establishment attempt: stop T300 (TS 38.331
        // §5.3.3.2 stops it on RRCReject, not only on RRCSetup) and drop the
        // pending initial NAS so a later spurious expiry can't fire.
        self.stop_t300();
        self.initial_nas_pdu = None;

        // Notify NAS of establishment failure
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcEstablishmentFailure)
            .await
        {
            error!("Failed to notify NAS of RRC reject: {}", e);
        }
    }

    /// Handle an `RRCReestablishment` (TS 38.331 §5.3.7.5): the network verified
    /// the `shortMAC-I` and restored the context.
    ///
    /// The reply arrives on **DL-DCCH / SRB1** and the completion goes out on
    /// UL-DCCH, per §6.2.1 — only the preceding request rides CCCH. Both were on
    /// CCCH before, in a bespoke framing neither end recognised.
    ///
    /// The `nextHopChainingCount` is recorded but not yet acted on: deriving the
    /// new KgNB from the {NH, NCC} pair (TS 33.501 §6.9.4.1) needs the NH, which
    /// only reaches a gNB through a Path Switch Request Acknowledge — a path that
    /// is not wired (issue #39). The count is logged so a mismatch is visible.
    async fn handle_rrc_reestablishment(
        &mut self,
        cell_id: i32,
        reestablishment: &nextgsim_rrc::procedures::rrc_reestablishment::RrcReestablishmentData,
    ) {
        info!(
            "Received RRCReestablishment from cell {} (tid {}, ncc {})",
            cell_id, reestablishment.rrc_transaction_id, reestablishment.next_hop_chaining_count
        );

        // If a re-establishment was not already in WaitingForResponse, the UE
        // could have crashed and recovered; treat cell as found first.
        if self.reestablishment_proc.state() == ReestablishmentState::CellSearch {
            let _ = self.reestablishment_proc.on_cell_found();
        }

        match self.reestablishment_proc.on_reestablishment_received(
            reestablishment.rrc_transaction_id,
            &mut self.state_machine,
        ) {
            Ok(complete_params) => {
                match encode_rrc_reestablishment_complete(&RrcReestablishmentCompleteParams {
                    rrc_transaction_id: complete_params.rrc_transaction_id,
                }) {
                    Ok(bytes) => {
                        self.send_uplink_rrc(RrcChannel::UlDcch, OctetString::from_slice(&bytes))
                            .await;
                    }
                    Err(e) => {
                        warn!("RRCReestablishmentComplete encoding failed: {}", e);
                        return;
                    }
                }

                // Notify NAS that connection is restored
                if let Err(e) = self
                    .task_base
                    .nas_tx
                    .send(NasMessage::RrcConnectionSetup)
                    .await
                {
                    error!("Failed to notify NAS after re-establishment: {}", e);
                }
                self.serving_cell_id = Some(cell_id);
                info!("RRC re-establishment complete on cell {}", cell_id);
            }
            Err(e) => {
                warn!("RRCReestablishment handling failed: {}", e);
            }
        }
    }

    /// Applies (or clears) the dedicated `cellReselectionPriorities` an RRCRelease
    /// carried (TS 38.331 §6.3.2, TS 38.304 §5.2.4.1, issue #50).
    ///
    /// Three distinct cases, and conflating any two of them is the trap:
    ///
    /// - **IE absent** — the network said nothing about priorities, so the UE keeps
    ///   whatever it has. NOT the same as being told to forget them.
    /// - **IE present with entries** — dedicated priorities that override the
    ///   broadcast ones while they are valid.
    /// - **IE present and empty** — TS 38.304 has the UE DELETE its stored
    ///   dedicated priorities and fall back to broadcast. This is why an empty list
    ///   cannot be treated as absent.
    fn apply_dedicated_reselection_priorities(
        &mut self,
        priorities: Option<CellReselectionPrioritiesParams>,
    ) {
        let Some(priorities) = priorities else {
            debug!("RRCRelease carried no cellReselectionPriorities; keeping current priorities");
            return;
        };
        let carriers: Vec<(u32, u8)> = priorities
            .freq_priority_list_nr
            .iter()
            .map(|f| (f.carrier_freq, f.priority))
            .collect();
        if carriers.is_empty() {
            info!(
                "RRCRelease carried an EMPTY cellReselectionPriorities: deleting the \
                 dedicated priorities and falling back to broadcast (TS 38.304 §5.2.4.1)"
            );
        } else {
            info!(
                "RRCRelease assigned {} dedicated carrier reselection priorities",
                carriers.len()
            );
        }
        self.cell_selector
            .apply_dedicated_carrier_priorities(&carriers);
    }

    /// Handle RRC Release message
    async fn handle_rrc_release(&mut self) {
        // Transition to idle state
        if let Err(e) = self.state_machine.on_rrc_release() {
            warn!("Failed to transition to idle state: {}", e);
        }

        // Conditional reconfigurations belong to the released connection: the
        // candidates were prepared by a cell the UE is no longer connected to,
        // and TS 38.331 §5.3.11 releases the RRC configuration on release.
        self.cond_reconfig.clear();
        // Secondary cells are part of that configuration (§5.3.5.5.9), so they
        // go the same way: A6 has no reference in RRC_IDLE.
        self.release_all_scells();

        info!("RRC connection released");

        // Notify NAS of connection release
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcConnectionRelease)
            .await
        {
            error!("Failed to notify NAS of RRC release: {}", e);
        }
    }

    /// Handle RRC Reconfiguration message (for handover)
    async fn handle_rrc_reconfiguration(&mut self, cell_id: i32, pdu: &OctetString) {
        let bytes = pdu.data();

        // Conditional reconfiguration (TS 38.331 §5.3.5.13.3): the message
        // carries candidate target configurations rather than an immediate
        // handover command.
        if bytes.first() == Some(&RECONFIGURATION_WITH_CHO) {
            self.handle_conditional_reconfiguration(bytes).await;
            return;
        }

        // Secondary cell addition/release (TS 38.331 §5.3.5.5.9).
        if bytes.first() == Some(&RECONFIGURATION_WITH_SCELL) {
            self.handle_scell_reconfiguration(bytes).await;
            return;
        }

        // Check if this is a handover reconfiguration
        if let Some(ho_command) = parse_handover_command(bytes) {
            info!(
                "Received handover command: target_pci={} (local cell resolved later)",
                ho_command.target_cell.pci
            );
            self.handle_handover_command(cell_id, ho_command).await;
            return;
        }

        // Regular reconfiguration (no handover)
        debug!("RRC Reconfiguration (no handover) - sending RRC Reconfiguration Complete");

        // Extract transaction ID (simplified)
        let transaction_id = if bytes.len() > 1 { bytes[1] } else { 0 };

        // Build RRC Reconfiguration Complete
        let rrc_pdu = OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
        self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
    }

    /// Handle an RRCReconfiguration carrying a conditional-reconfiguration
    /// container (TS 38.331 §5.3.5.13.3).
    ///
    /// The container is decoded whatever the configuration says, so a malformed
    /// one is reported rather than silently ignored, and the message is always
    /// acknowledged with an RRCReconfigurationComplete — the reconfiguration
    /// itself is accepted. Whether the candidates are *stored* is what
    /// `UeConfig::conditional_handover` gates.
    async fn handle_conditional_reconfiguration(&mut self, bytes: &[u8]) {
        let transaction_id = bytes.get(1).copied().unwrap_or(0);

        match decode_cho_config(&bytes[CHO_CONTAINER_OFFSET.min(bytes.len())..]) {
            Ok(config) => {
                if self.task_base.config.conditional_handover {
                    self.cho_transaction_id = transaction_id;
                    let stored = self.cond_reconfig.add_config(&config);
                    info!(
                        "Conditional reconfiguration stored: config {} now holds {} candidate(s), {} not armed",
                        config.config_id,
                        stored,
                        self.cond_reconfig.unevaluated_count()
                    );
                } else {
                    info!(
                        "Conditional reconfiguration ignored: config {} carries {} candidate(s) \
                         but conditional_handover is off",
                        config.config_id,
                        config.candidate_cells.len()
                    );
                }
            }
            Err(e) => warn!(
                "Failed to decode conditional reconfiguration container: {}",
                e
            ),
        }

        let rrc_pdu = OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
        self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
    }

    /// Handle an RRCReconfiguration carrying a secondary-cell configuration
    /// (TS 38.331 §5.3.5.5.9).
    ///
    /// Releases are applied before additions, which is the order §5.3.5.5.9
    /// itself uses, so a message that releases index 1 and adds a new cell at
    /// index 1 ends with the new cell rather than nothing.
    ///
    /// A cell the UE has never measured is still recorded: RLS may report it on a
    /// later heartbeat, and A6 simply cannot enter until it does — the same
    /// posture as a CHO candidate targeting an unmeasured cell.
    async fn handle_scell_reconfiguration(&mut self, bytes: &[u8]) {
        let transaction_id = bytes.get(1).copied().unwrap_or(0);

        match decode_scell_config(&bytes[SCELL_CONTAINER_OFFSET.min(bytes.len())..]) {
            Ok(config) => {
                for index in &config.to_release {
                    match self.configured_scells.remove(index) {
                        Some(cell_id) => {
                            info!("SCell released: sCellIndex={}, cell={}", index, cell_id)
                        }
                        None => warn!(
                            "sCellToReleaseList names sCellIndex {}, which is not configured",
                            index
                        ),
                    }
                }
                for addition in &config.to_add {
                    // physCellId is the simulator cell id throughout the UE
                    // measurement path (see `candidate_cell_id`).
                    let cell_id = i32::from(addition.phys_cell_id);
                    self.configured_scells.insert(addition.scell_index, cell_id);
                    info!(
                        "SCell configured: sCellIndex={}, physCellId={}, measured={}",
                        addition.scell_index,
                        addition.phys_cell_id,
                        self.measurement_manager.rsrp(cell_id).is_some()
                    );
                }
                self.apply_scell_reference();
            }
            Err(e) => warn!("Failed to decode secondary cell configuration: {}", e),
        }

        let rrc_pdu = OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
        self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
    }

    /// Point the measurement manager's A6 reference at the lowest-index
    /// configured SCell, or at nothing when none is configured.
    ///
    /// See `configured_scells` for why it is the lowest index rather than a
    /// per-event one.
    fn apply_scell_reference(&mut self) {
        let scell = self
            .configured_scells
            .values()
            .next()
            .copied()
            .filter(|cell_id| Some(*cell_id) != self.serving_cell_id);
        self.measurement_manager.set_scell(scell);
    }

    /// Release every configured secondary cell (TS 38.331 §5.3.11: the UE
    /// releases the RRC configuration on going to RRC_IDLE).
    fn release_all_scells(&mut self) {
        if !self.configured_scells.is_empty() {
            info!(
                "Releasing {} configured SCell(s) on leaving RRC_CONNECTED",
                self.configured_scells.len()
            );
            self.configured_scells.clear();
        }
        self.measurement_manager.set_scell(None);
    }

    /// The simulator cell ids of the configured secondary cells, lowest
    /// `sCellIndex` first — for tests and status reporting.
    pub fn configured_scells(&self) -> Vec<i32> {
        self.configured_scells.values().copied().collect()
    }

    /// Handle handover command from RRC Reconfiguration
    async fn handle_handover_command(&mut self, source_cell_id: i32, mut command: HandoverCommand) {
        // Resolve the target's `physCellId` to a cell this UE can actually hear
        // (issue #107). The command names the target by PCI, which is the 3GPP
        // identity; the simulator's cell index is not on the wire and must not be,
        // because a UE handed an index it cannot verify would trust a number the
        // network invented. `phys_cell_id_of` derives each known cell's PCI from
        // the NCI its SIB1 broadcast, which is the convention both ends share.
        let target_pci = command.target_cell.pci;
        command.target_cell.cell_id =
            self.cell_selector.cells().keys().copied().find(|id| {
                self.cell_selector.phys_cell_id_of(*id) == Some((target_pci % 1008) as u16)
            });
        let target_cell_id = command.target_cell.cell_id;
        let transaction_id = command.transaction_id;

        // Start handover in the handover manager
        self.handover_manager
            .start_handover(source_cell_id, command);

        // Check if we have signal to the target cell. An UNRESOLVED PCI takes the
        // failure branch: a target the UE cannot identify is a target it cannot
        // reach, which is exactly what TargetCellUnreachable means.
        if target_cell_id.is_some_and(|id| self.cell_selector.has_signal_to_cell(id)) {
            // Start synchronization
            self.handover_manager.start_synchronization();

            // In simulation, we assume sync is instant
            self.handover_manager.sync_complete();

            // Complete handover
            if let Some(new_cell_id) = self.handover_manager.complete() {
                // Update serving cell
                let old_cell_id = self.serving_cell_id;
                self.serving_cell_id = Some(new_cell_id);

                // Update measurement manager
                self.measurement_manager.set_serving_cell(Some(new_cell_id));

                // §5.3.5.5.2: applying a reconfigurationWithSync replaces the
                // cell group configuration, so the source cell's secondary cells
                // are gone. The target names its own in a later reconfiguration.
                self.release_all_scells();

                // Notify RLS of new serving cell
                if let Err(e) = self
                    .task_base
                    .rls_tx
                    .send(RlsMessage::AssignCurrentCell {
                        cell_id: new_cell_id,
                    })
                    .await
                {
                    error!("Failed to notify RLS of handover: {}", e);
                }

                info!(
                    "Handover successful: {} -> {}",
                    old_cell_id.unwrap_or(-1),
                    new_cell_id
                );

                // Send RRC Reconfiguration Complete
                let rrc_pdu =
                    OctetString::from_slice(&build_reconfiguration_complete(transaction_id));
                self.send_uplink_rrc(RrcChannel::UlDcch, rrc_pdu).await;
            }
        } else {
            // Target cell not reachable - handover failure
            warn!(
                "Handover failed: target PCI {} not in coverage (local cell {:?})",
                target_pci, target_cell_id
            );
            if let Some(source_cell) = self
                .handover_manager
                .fail(crate::rrc::handover::HandoverFailureCause::TargetCellUnreachable)
            {
                // Stay on source cell
                self.serving_cell_id = Some(source_cell);
            }

            // Trigger RRC re-establishment
            self.handle_handover_failure().await;
        }
    }

    /// Handle handover failure - initiate re-establishment
    async fn handle_handover_failure(&mut self) {
        warn!("Initiating RRC re-establishment after handover failure");

        // Notify NAS of radio link failure
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RadioLinkFailure)
            .await
        {
            error!("Failed to notify NAS of handover failure: {}", e);
        }
    }

    /// Handles a UECapabilityEnquiry (TS 38.331 §5.6.1) and responds with
    /// UECapabilityInformation carrying a real UE-NR-Capability container.
    async fn handle_ue_capability_enquiry(&mut self, uper_bytes: &[u8]) {
        let enquiry = match decode_ue_capability_enquiry(uper_bytes) {
            Ok(data) => data,
            Err(e) => {
                warn!("Failed to decode UECapabilityEnquiry: {}", e);
                return;
            }
        };

        // Provide a capability container for each requested RAT we support
        // (NR only for this UE)
        let mut containers = Vec::new();
        for rat in &enquiry.rat_types {
            if *rat == RatType::Nr {
                match build_minimal_nr_capability_container(DEFAULT_NR_BAND) {
                    Ok(container) => containers.push(UeCapabilityRatContainer {
                        rat_type: RatType::Nr,
                        container,
                    }),
                    Err(e) => {
                        error!("Failed to build UE-NR-Capability container: {}", e);
                    }
                }
            } else {
                debug!(
                    "UECapabilityEnquiry for unsupported RAT {:?} — skipping",
                    rat
                );
            }
        }

        let params = UeCapabilityInformationParams {
            rrc_transaction_id: enquiry.rrc_transaction_id,
            containers,
        };

        match encode_ue_capability_information(&params) {
            Ok(uper) => {
                let mut rrc_pdu = Vec::with_capacity(uper.len() + 1);
                rrc_pdu.push(RRC_MSG_TYPE_UE_CAPABILITY);
                rrc_pdu.extend_from_slice(&uper);
                info!(
                    "Sending UECapabilityInformation (transaction {}, {} containers)",
                    params.rrc_transaction_id,
                    params.containers.len()
                );
                self.send_uplink_rrc(RrcChannel::UlDcch, OctetString::from_slice(&rrc_pdu))
                    .await;
            }
            Err(e) => {
                error!("Failed to encode UECapabilityInformation: {}", e);
            }
        }
    }

    /// Forward NAS PDU to NAS task
    async fn forward_nas_to_nas_task(&self, pdu: OctetString) {
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::NasDelivery { pdu })
            .await
        {
            error!("Failed to forward NAS to NAS task: {}", e);
        }
    }

    /// Handle uplink NAS delivery from NAS task
    ///
    /// Public because it is a real message-handler entry point
    /// (`RrcMessage::UplinkNasDelivery`) also driven directly by the
    /// in-process strict-peer harness (`tests/src/rrc_handshake.rs`, Wave-6 C3).
    pub async fn handle_uplink_nas_delivery(&mut self, pdu_id: u32, pdu: OctetString) {
        // If not connected, this is initial NAS - start connection establishment
        if self.state_machine.state() == RrcState::Idle {
            self.initial_nas_pdu = Some(pdu.clone());
            self.start_connection_establishment(pdu).await;
        } else if self.state_machine.state() == RrcState::Connected {
            // Build UL Information Transfer
            let mut rrc_pdu = Vec::with_capacity(pdu.len() + 2);
            rrc_pdu.push(0x08); // ULInformationTransfer message type
            rrc_pdu.push(0x00); // Critical extensions
            rrc_pdu.extend_from_slice(pdu.data());

            let ul_info = OctetString::from_slice(&rrc_pdu);
            self.send_uplink_rrc_with_id(RrcChannel::UlDcch, pdu_id, ul_info)
                .await;
        }
    }

    /// Start RRC connection establishment using proper ASN.1 UPER encoding
    async fn start_connection_establishment(&mut self, _nas_pdu: OctetString) {
        if self.state_machine.state() != RrcState::Idle {
            warn!("Cannot start connection establishment: not in idle state");
            self.handle_establishment_failure().await;
            return;
        }

        if self.serving_cell_id.is_none() {
            warn!("Cannot start connection establishment: no serving cell");
            self.handle_establishment_failure().await;
            return;
        }

        info!("Starting RRC connection establishment");

        let random_id: u64 = rand::random::<u64>() & 0x7FFFFFFFFF; // 39-bit value

        let establishment_cause = match self.establishment_cause {
            0 => AsnEstablishmentCause::Emergency,
            1 => AsnEstablishmentCause::HighPriorityAccess,
            2 => AsnEstablishmentCause::MtAccess,
            3 => AsnEstablishmentCause::MoSignalling,
            5 => AsnEstablishmentCause::MoVoiceCall,
            6 => AsnEstablishmentCause::MoVideoCall,
            7 => AsnEstablishmentCause::MoSms,
            8 => AsnEstablishmentCause::MpsPriorityAccess,
            9 => AsnEstablishmentCause::McsPriorityAccess,
            _ => AsnEstablishmentCause::MoData, // 4 or default
        };

        let params = RrcSetupRequestParams {
            ue_identity: UeIdentity::RandomValue(random_id),
            establishment_cause,
        };

        let pdu = match encode_rrc_setup_request(&params) {
            Ok(bytes) => {
                debug!(
                    "ASN.1 RRCSetupRequest encoded: {} bytes, random_id={:x}",
                    bytes.len(),
                    random_id
                );
                OctetString::from_slice(&bytes)
            }
            Err(e) => {
                warn!(
                    "ASN.1 RRCSetupRequest encoding failed ({}), using fallback",
                    e
                );
                let mut rrc_pdu = Vec::with_capacity(8);
                rrc_pdu.push(0x00);
                rrc_pdu.extend_from_slice(&random_id.to_be_bytes()[3..8]);
                rrc_pdu.push(self.establishment_cause as u8);
                OctetString::from_slice(&rrc_pdu)
            }
        };

        self.send_uplink_rrc(RrcChannel::UlCcch, pdu).await;

        // Start the T300 establishment guard (TS 38.331 §5.3.3.2): if no
        // RRCSetup arrives before it expires, the run loop aborts the attempt.
        self.start_t300();
    }

    /// Handle establishment failure
    async fn handle_establishment_failure(&self) {
        if let Err(e) = self
            .task_base
            .nas_tx
            .send(NasMessage::RrcEstablishmentFailure)
            .await
        {
            error!("Failed to notify NAS of establishment failure: {}", e);
        }
    }

    /// Starts the T300 establishment guard on transmission of an
    /// RRCSetupRequest (TS 38.331 §5.3.3.2).
    fn start_t300(&mut self) {
        self.t300_deadline =
            Some(tokio::time::Instant::now() + Duration::from_millis(T300_DEFAULT_MS));
        debug!("T300 started ({} ms)", T300_DEFAULT_MS);
    }

    /// Stops T300 on reception of RRCSetup (TS 38.331 §5.3.3.4).
    fn stop_t300(&mut self) {
        if self.t300_deadline.take().is_some() {
            debug!("T300 stopped (RRCSetup received)");
        }
    }

    /// True while the T300 establishment guard is running.
    #[cfg(test)]
    fn t300_running(&self) -> bool {
        self.t300_deadline.is_some()
    }

    /// Handles T300 expiry (TS 38.331 §5.3.3.2): the RRC connection
    /// establishment attempt is abandoned — the guard is cleared, the pending
    /// initial NAS is dropped, and the upper layers (NAS) are informed of the
    /// establishment failure so registration can be retried or failed. The RRC
    /// state machine stays in RRC_IDLE (establishment does not leave Idle until
    /// an RRCSetup is received), so no state transition is required here.
    async fn on_t300_expiry(&mut self) {
        warn!("T300 expired: RRC connection establishment failed (TS 38.331 §5.3.3.2)");
        self.t300_deadline = None;
        self.initial_nas_pdu = None;
        self.handle_establishment_failure().await;
    }

    /// Handle radio link failure
    async fn handle_radio_link_failure(&mut self, cause: RlfCause) {
        warn!("Radio link failure: {:?}", cause);

        // Map RLF cause to re-establishment trigger
        let trigger = match cause {
            RlfCause::PduIdExists | RlfCause::PduIdFull | RlfCause::SignalLostToConnectedCell => {
                ReestablishmentTrigger::RadioLinkFailure
            }
        };

        // Per TS 38.331 §5.3.7.2 re-establishment is only initiated when AS
        // security has been activated; otherwise the UE goes to RRC_IDLE.
        let security = if self.state_machine.state().has_connection_context()
            && !self.reestablishment_proc.is_in_progress()
        {
            self.as_security.clone()
        } else {
            None
        };
        if let Some(security) = security {
            let c_rnti = security.c_rnti;

            // Target cell identity: the cell we re-establish on (the serving
            // cell in this simulation, looked up from its SIB1 NCI)
            let target_cell_identity = self
                .serving_cell_id
                .and_then(|id| self.cell_selector.get_cell(id))
                .map(|cell| cell.sib1.nci as u64)
                .unwrap_or(0);
            // The PCI is derived from that same NCI rather than from the UE's own
            // cell index, which is a local counter the network cannot hold. See
            // `phys_cell_id_from_nci`: with a real broadcast SIB1 both ends derive
            // the same number, and without one the gNB answers with an RRCSetup,
            // which is §5.3.3.1's behaviour for an unresolvable identity.
            let pci = phys_cell_id_from_nci(target_cell_identity);

            // ShortMAC-I derived from the AS security context (TS 38.331 §5.3.7.4)
            let short_mac_i = match compute_short_mac_i(&security, pci, target_cell_identity) {
                Ok(mac) => mac,
                Err(e) => {
                    error!("ShortMAC-I derivation failed: {} — going to idle", e);
                    let _ = self.state_machine.on_rrc_release();
                    self.serving_cell_id = None;
                    if let Err(ne) = self
                        .task_base
                        .nas_tx
                        .send(NasMessage::RadioLinkFailure)
                        .await
                    {
                        error!("Failed to notify NAS of radio link failure: {}", ne);
                    }
                    return;
                }
            };

            match self.reestablishment_proc.initiate(
                trigger,
                c_rnti,
                pci,
                short_mac_i,
                &mut self.state_machine,
            ) {
                Ok(params) => {
                    info!(
                        "RLF: sending RRCReestablishmentRequest (cause={}, c_rnti={:#06x}, pci={})",
                        params.trigger, params.c_rnti, params.pci
                    );
                    // Immediately mark cell as found (re-use serving cell if still visible)
                    let _ = self.reestablishment_proc.on_cell_found();

                    // A real UPER UL-CCCH RRCReestablishmentRequest (TS 38.331
                    // §6.2.2). The bespoke framing this replaced put 0x05 in the
                    // first byte, which the gNB's dispatcher read as an
                    // RRCSetupRequest -- and which actually DECODES as one, so
                    // the network answered a re-establishment with an RRCSetup
                    // built on a fabricated context and never saw the shortMAC-I
                    // at all (issue #37).
                    let cause = match params.trigger {
                        ReestablishmentTrigger::ReconfigurationFailure => {
                            ReestablishmentCauseValue::ReconfigurationFailure
                        }
                        ReestablishmentTrigger::HandoverFailure => {
                            ReestablishmentCauseValue::HandoverFailure
                        }
                        _ => ReestablishmentCauseValue::OtherFailure,
                    };
                    match encode_rrc_reestablishment_request(&RrcReestablishmentRequestParams {
                        ue_identity: ReestablishmentUeIdentity {
                            c_rnti: params.c_rnti,
                            phys_cell_id: params.pci,
                            short_mac_i: params.short_mac_i,
                        },
                        reestablishment_cause: cause,
                    }) {
                        Ok(bytes) => {
                            self.send_uplink_rrc(
                                RrcChannel::UlCcch,
                                OctetString::from_slice(&bytes),
                            )
                            .await;
                        }
                        Err(e) => {
                            error!(
                                "RRCReestablishmentRequest encoding failed: {} -- going to idle",
                                e
                            );
                            let _ = self.state_machine.on_rrc_release();
                            self.serving_cell_id = None;
                            if let Err(ne) = self
                                .task_base
                                .nas_tx
                                .send(NasMessage::RadioLinkFailure)
                                .await
                            {
                                error!("Failed to notify NAS of radio link failure: {}", ne);
                            }
                        }
                    }
                }
                Err(e) => {
                    warn!(
                        "Could not initiate re-establishment: {} — falling back to idle",
                        e
                    );
                    let _ = self.state_machine.on_rrc_release();
                    self.serving_cell_id = None;
                    if let Err(ne) = self
                        .task_base
                        .nas_tx
                        .send(NasMessage::RadioLinkFailure)
                        .await
                    {
                        error!("Failed to notify NAS of radio link failure: {}", ne);
                    }
                }
            }
        } else {
            // Already idle, re-establishment in progress, or AS security never
            // activated (TS 38.331 §5.3.7.2: go to RRC_IDLE) — just notify NAS
            let _ = self.state_machine.on_rrc_release();
            self.serving_cell_id = None;
            if let Err(e) = self
                .task_base
                .nas_tx
                .send(NasMessage::RadioLinkFailure)
                .await
            {
                error!("Failed to notify NAS of radio link failure: {}", e);
            }
        }
    }

    /// Perform Unified Access Control check per 3GPP TS 38.331 Section 5.3.14
    ///
    /// Returns true if access is allowed, false if barred.
    fn perform_uac_check(&self, access_category: i32, access_identities: u32) -> bool {
        // Access categories 0 (MT access) and 2 (emergency) are never barred
        if access_category == 0 || access_category == 2 {
            return true;
        }

        // Check if access identities indicate high-priority (bit 0 = priority access)
        // Access identity 0 is always allowed per TS 38.331
        if access_identities & 0x01 != 0 {
            return true;
        }

        // Check if this access category is subject to barring
        if access_category >= 0 && (access_category as u32) < 32 {
            let category_mask = 1u32 << (access_category as u32);
            if self.uac_barring.barring_for_access_category & category_mask == 0 {
                // This category is not subject to barring
                return true;
            }
        }

        // Apply barring factor check
        if self.uac_barring.barring_factor_percent == 0 {
            return true; // No barring configured
        }

        // Generate random number 0..99 and compare against barring factor
        let random: u8 = rand::random::<u8>() % 100;
        let allowed = random >= self.uac_barring.barring_factor_percent;

        if !allowed {
            info!(
                "UAC barred: category={}, identities={:#x}, factor={}%, barring_time={}s",
                access_category,
                access_identities,
                self.uac_barring.barring_factor_percent,
                self.uac_barring.barring_time_secs
            );
        }

        allowed
    }

    /// Handle local release connection request from NAS
    async fn handle_local_release_connection(&mut self, _treat_barred: bool) {
        info!("Local release connection requested");

        // Upper-layer abort of the connection: stop T300 if establishment was
        // in flight (TS 38.331 §5.3.3.2 stops T300 on abort by upper layers) so
        // the guard can't fire a spurious expiry after the attempt is dropped.
        self.stop_t300();

        // Transition to idle
        let _ = self.state_machine.on_rrc_release();
        self.serving_cell_id = None;

        // Reset STI in RLS
        if let Err(e) = self.task_base.rls_tx.send(RlsMessage::ResetSti).await {
            error!("Failed to send ResetSti to RLS: {}", e);
        }
    }

    /// Send uplink RRC message
    async fn send_uplink_rrc(&mut self, channel: RrcChannel, pdu: OctetString) {
        let pdu_id = self.next_pdu_id();
        self.send_uplink_rrc_with_id(channel, pdu_id, pdu).await;
    }

    /// Route AI/ML inference request to SHE Client task
    #[cfg(feature = "nextgsim-she")]
    async fn route_6g_inference(&self, model_id: String, input_data: Vec<f32>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let msg = SheClientMessage::InferenceRequest {
                model_id,
                input: input_data,
                input_shape: vec![],
                deadline_ms: 0,
                response_tx: None,
            };
            if let Err(e) = sixg.she_client_tx.send(msg).await {
                error!("Failed to route inference to SHE Client: {}", e);
            }
        } else {
            warn!("6G tasks not initialized, dropping inference request");
        }
    }

    /// Route sensing measurement to ISAC Sensor task
    #[cfg(feature = "nextgsim-isac")]
    async fn route_6g_sensing(&self, measurement_type: String, measurements: Vec<f32>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let meas_type = match measurement_type.as_str() {
                "toa" | "ToA" => IsacMeasurementType::ToA,
                "aoa" | "AoA" => IsacMeasurementType::AoA,
                "doppler" | "Doppler" => IsacMeasurementType::Doppler,
                "csi" | "CSI" => IsacMeasurementType::Csi,
                _ => IsacMeasurementType::MultiPath,
            };
            let msg = IsacSensorMessage::SensingMeasurement {
                measurement_type: meas_type,
                data: measurements,
                timestamp_ms: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("value expected")
                    .as_millis() as u64,
            };
            if let Err(e) = sixg.isac_sensor_tx.send(msg).await {
                error!("Failed to route sensing data to ISAC: {}", e);
            }
        } else {
            warn!("6G tasks not initialized, dropping sensing measurement");
        }
    }

    /// Route semantic communication data to Semantic Codec task
    #[cfg(feature = "nextgsim-semantic")]
    async fn route_6g_semantic(&self, content_type: String, data: Vec<u8>) {
        if let Some(ref sixg) = self.task_base.sixg {
            let task_type = match content_type.as_str() {
                "image" => SemanticTaskType::ImageClassification,
                "object" => SemanticTaskType::ObjectDetection,
                "speech" => SemanticTaskType::SpeechRecognition,
                "sensor" => SemanticTaskType::SensorFusion,
                "video" => SemanticTaskType::VideoAnalytics,
                "text" => SemanticTaskType::TextUnderstanding,
                _ => SemanticTaskType::Custom(0),
            };
            // Convert raw bytes to f32 features for the codec
            let features: Vec<f32> = data.iter().map(|&b| b as f32 / 255.0).collect();
            let dims = vec![features.len()];
            let msg = SemanticCodecMessage::Encode {
                task_type,
                data: features,
                dimensions: dims,
                channel_quality: None,
                response_tx: None,
            };
            if let Err(e) = sixg.semantic_codec_tx.send(msg).await {
                error!("Failed to route semantic data to codec: {}", e);
            }
        } else {
            warn!("6G tasks not initialized, dropping semantic data");
        }
    }

    /// Send uplink RRC message with specific PDU ID
    async fn send_uplink_rrc_with_id(&self, channel: RrcChannel, pdu_id: u32, pdu: OctetString) {
        let msg = RlsMessage::RrcPduDelivery {
            channel,
            pdu_id,
            pdu,
        };
        if let Err(e) = self.task_base.rls_tx.send(msg).await {
            error!("Failed to send RRC message to RLS: {}", e);
        }
    }
}

/// Resolves when the T300 deadline is reached; pends forever when T300 is not
/// running, so the RRC run loop's `select!` arm only fires on a real expiry.
async fn wait_t300(deadline: Option<tokio::time::Instant>) {
    match deadline {
        Some(d) => tokio::time::sleep_until(d).await,
        None => std::future::pending::<()>().await,
    }
}

#[async_trait::async_trait]
impl Task for RrcTask {
    type Message = RrcMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("RRC task started");

        let mut cycle_timer = interval(Duration::from_millis(RRC_CYCLE_INTERVAL_MS));

        loop {
            tokio::select! {
                Some(msg) = rx.recv() => {
                    match msg {
                        TaskMessage::Message(rrc_msg) => match rrc_msg {
                            RrcMessage::LocalReleaseConnection { treat_barred } => {
                                self.handle_local_release_connection(treat_barred).await;
                            }
                            RrcMessage::UplinkNasDelivery { pdu_id, pdu } => {
                                self.handle_uplink_nas_delivery(pdu_id, pdu).await;
                            }
                            RrcMessage::RrcNotify => {
                                debug!("RRC notify received");
                            }
                            RrcMessage::SetSelectedPlmn { plmn } => {
                                // TS 23.122 §4.4.3 (issue #49): the PLMN NAS chose
                                // becomes the one cell selection searches for. Before
                                // this the RRC selector kept the configured home
                                // PLMN forever, so a UE that selected another
                                // available PLMN still could not camp on its cells.
                                info!("Cell selection will now search for PLMN {plmn}");
                                self.cell_selector.set_selected_plmn(Some(plmn));
                            }
                            RrcMessage::AsSecurityKey { kgnb } => {
                                debug!("Received KgNB for AS security from NAS plane");
                                self.set_pending_kgnb(kgnb);
                            }
                            RrcMessage::PerformUac { access_category, access_identities } => {
                                let allowed = self.perform_uac_check(access_category, access_identities);
                                debug!("UAC check: category={}, identities={}, allowed={}", access_category, access_identities, allowed);
                                if !allowed {
                                    // Notify NAS that access is barred
                                    if let Err(e) = self.task_base.nas_tx.send(NasMessage::RrcEstablishmentFailure).await {
                                        error!("Failed to notify NAS of UAC barring: {}", e);
                                    }
                                }
                            }
                            RrcMessage::PagingIdentity { s_tmsi } => {
                                self.set_paging_identity(s_tmsi);
                            }
                            RrcMessage::DownlinkRrcDelivery { cell_id, channel, pdu } => {
                                self.handle_downlink_rrc(cell_id, channel, pdu).await;
                            }
                            RrcMessage::SignalChanged { cell_id, dbm } => {
                                self.handle_signal_changed(cell_id, dbm).await;
                            }
                            RrcMessage::RadioLinkFailure { cause } => {
                                self.handle_radio_link_failure(cause).await;
                            }
                            RrcMessage::TriggerCycle => {
                                self.perform_cycle().await;
                            }
                            RrcMessage::NtnTimingAdvanceReceived {
                                common_ta_us, k_offset, autonomous_ta, max_doppler_hz,
                            } => {
                                info!(
                                    "UE RRC: NTN timing advance received: TA={}us, k_offset={}, autonomous={}, doppler={}Hz",
                                    common_ta_us, k_offset, autonomous_ta, max_doppler_hz
                                );
                                self.ntn_timing = Some(UeNtnTiming {
                                    common_ta_us,
                                    k_offset,
                                    autonomous_ta,
                                    max_doppler_hz,
                                });
                            }
                            // 6G message routing
                            #[cfg(feature = "nextgsim-she")]
                            RrcMessage::SixgInferenceRequest { model_id, input_data } => {
                                self.route_6g_inference(model_id, input_data).await;
                            }
                            #[cfg(feature = "nextgsim-isac")]
                            RrcMessage::SixgSensingMeasurement { measurement_type, measurements } => {
                                self.route_6g_sensing(measurement_type, measurements).await;
                            }
                            #[cfg(feature = "nextgsim-semantic")]
                            RrcMessage::SixgSemanticData { content_type, data } => {
                                self.route_6g_semantic(content_type, data).await;
                            }
                        },
                        TaskMessage::Shutdown => {
                            info!("RRC task received shutdown signal");
                            break;
                        }
                    }
                }
                _ = cycle_timer.tick() => {
                    self.perform_cycle().await;
                }
                _ = wait_t300(self.t300_deadline) => {
                    self.on_t300_expiry().await;
                }
            }
        }

        info!(
            "RRC task stopped in {:?} state with {} cells",
            self.state_machine.state(),
            self.cell_selector.cells().len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::config::UeConfig;

    fn test_config() -> UeConfig {
        UeConfig::default()
    }

    #[test]
    fn test_rrc_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let task = RrcTask::new(task_base);
        assert_eq!(task.state_machine.state(), RrcState::Idle);
        assert!(task.serving_cell_id.is_none());
    }

    #[test]
    fn test_pdu_id_generation() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert_eq!(task.next_pdu_id(), 1);
        assert_eq!(task.next_pdu_id(), 2);
        assert_eq!(task.next_pdu_id(), 3);
    }

    #[test]
    fn test_is_active_cell() {
        let config = test_config();
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        assert!(!task.is_active_cell(1));

        task.serving_cell_id = Some(1);
        assert!(task.is_active_cell(1));
        assert!(!task.is_active_cell(2));
    }

    // ========================================================================
    // Wave-6 C2: UE-side RRCSetup ASN.1 decode + tolerance verification.
    // The golden literal is the C1 hand-derived RRCSetup(SRB1, tid 0) — see
    // nextgsim-rrc rrc_setup.rs `golden_rrc_setup_srb1_bytes` for the
    // bit-by-bit derivation from tools/rrc-15.6.0.asn1.
    // ========================================================================

    use nextgsim_rrc::procedures::rrc_setup::decode_rrc_setup_complete;

    /// C1 golden RRCSetup: SRB1 RadioBearerConfig + CellGroupConfig(LCID 1),
    /// tid 0 (hand-derived, strict-peer fixture).
    const GOLDEN_RRC_SETUP_SRB1_TID0: [u8; 8] = [0x20, 0x40, 0x00, 0x22, 0x00, 0x04, 0x00, 0x00];

    /// The pre-Wave-6 gNB placeholder RRCSetup: valid ASN.1 (empty
    /// RadioBearerConfig, 1-byte garbage masterCellGroup) but establishes NO
    /// SRB1 — the C2 tolerance fixture.
    const LEGACY_PLACEHOLDER_RRC_SETUP: [u8; 4] = [0x20, 0x00, 0x04, 0x00];

    fn run_async<F: std::future::Future>(fut: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(fut)
    }

    /// Pops the next uplink RRC PDU handed to the RLS (skipping non-PDU RLS
    /// traffic such as cell assignment).
    fn next_uplink_rrc(
        rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) -> (RrcChannel, OctetString) {
        loop {
            match rx.try_recv() {
                Ok(TaskMessage::Message(RlsMessage::RrcPduDelivery { channel, pdu, .. })) => {
                    return (channel, pdu)
                }
                Ok(_) => continue,
                Err(e) => panic!("expected an uplink RRC PDU, got none: {e}"),
            }
        }
    }

    /// Camps the UE on cell 1 via the real cell-detection path and hands it
    /// the initial NAS so a real RRCSetupRequest goes out (the same
    /// real-handler flow as the strict-peer harness in
    /// tests/src/rrc_handshake.rs). Returns the initial NAS bytes.
    async fn camp_and_request(
        task: &mut RrcTask,
        rls_rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) -> Vec<u8> {
        task.handle_signal_changed(1, -60).await;
        task.perform_cycle().await;

        let nas = vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D];
        task.handle_uplink_nas_delivery(1, OctetString::from_slice(&nas))
            .await;
        let (ch, _setup_req) = next_uplink_rrc(rls_rx);
        assert_eq!(ch, RrcChannel::UlCcch, "RRCSetupRequest must go on UL-CCCH");
        nas
    }

    /// On camping, the RRC task reports the radio's available PLMNs to NAS in
    /// ActiveCellChanged (issue #49, TS 23.122 §4.4.3) — the production caller
    /// of CellSelector::available_plmns — so NAS selects over the real radio
    /// rather than a hardcoded home-PLMN list.
    #[test]
    fn test_active_cell_changed_reports_available_plmns() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_signal_changed(1, -60).await;
            task.perform_cycle().await; // camps on cell 1 → emits ActiveCellChanged

            let mut reported: Option<Vec<CellPlmn>> = None;
            while let Ok(msg) = nas_rx.try_recv() {
                if let TaskMessage::Message(NasMessage::ActiveCellChanged {
                    available_plmns, ..
                }) = msg
                {
                    reported = Some(available_plmns);
                }
            }
            let reported = reported.expect("ActiveCellChanged emitted on camp");
            assert!(
                !reported.is_empty(),
                "available PLMNs reported to NAS for selection (TS 23.122 §4.4.3)"
            );
        });
    }

    /// The RRC task reports the serving cell's PCI and level to NAS (issue #46),
    /// which is what an LPP E-CID positioning report is made of. Before this the NAS
    /// task had no measurements at all, so a `ProvideLocationInformation` could only
    /// have carried invented ones.
    #[test]
    fn test_serving_cell_measurement_is_reported_to_nas() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, mut rls_rx) =
            UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            // The report comes from the CONNECTED-state measurement cycle, which is
            // also when an LPP request can arrive (it travels over NAS).
            camp_and_request(&mut task, &mut rls_rx).await;
            task.handle_downlink_rrc(
                1,
                RrcChannel::DlCcch,
                OctetString::from_slice(&GOLDEN_RRC_SETUP_SRB1_TID0),
            )
            .await;
            assert_eq!(task.state_machine.state(), RrcState::Connected);

            task.handle_signal_changed(1, -71).await;
            task.perform_cycle().await;

            let mut reported = None;
            while let Ok(msg) = nas_rx.try_recv() {
                if let TaskMessage::Message(NasMessage::ServingCellMeasurement {
                    phys_cell_id,
                    rsrp_dbm,
                }) = msg
                {
                    reported = Some((phys_cell_id, rsrp_dbm));
                }
            }
            assert_eq!(
                reported,
                Some((1, -71)),
                "the camped cell's id doubles as its PCI, and the level is the one RLS \
                 reported"
            );
        });
    }

    /// The report goes out on a CHANGE, not once per RRC cycle: an unchanged serving
    /// cell would otherwise put a message on the NAS channel every 2.5 s forever.
    #[test]
    fn test_serving_cell_measurement_is_not_repeated_unchanged() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, mut rls_rx) =
            UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            camp_and_request(&mut task, &mut rls_rx).await;
            task.handle_downlink_rrc(
                1,
                RrcChannel::DlCcch,
                OctetString::from_slice(&GOLDEN_RRC_SETUP_SRB1_TID0),
            )
            .await;

            let count_reports = |rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<NasMessage>>| {
                let mut n = 0;
                while let Ok(msg) = rx.try_recv() {
                    if matches!(
                        msg,
                        TaskMessage::Message(NasMessage::ServingCellMeasurement { .. })
                    ) {
                        n += 1;
                    }
                }
                n
            };

            task.handle_signal_changed(1, -71).await;
            task.perform_cycle().await;
            assert_eq!(
                count_reports(&mut nas_rx),
                1,
                "the first measurement is sent"
            );

            // Two more cycles with the level unchanged.
            task.perform_cycle().await;
            task.perform_cycle().await;
            assert_eq!(
                count_reports(&mut nas_rx),
                0,
                "an unchanged measurement is not re-sent"
            );

            // A changed level IS sent, so the suppression is a comparison and not a
            // one-shot latch.
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            assert_eq!(count_reports(&mut nas_rx), 1, "a changed level is sent");
        });
    }

    /// The RRC task also reports the CAMPED cell's own PLMN (issue #49), which is
    /// what makes the registered PLMN knowable — it used to be recorded as the
    /// configured home PLMN unconditionally, which is why `PlmnSelector` could never
    /// tell an HPLMN from a VPLMN.
    #[test]
    fn test_active_cell_changed_reports_the_serving_cell_plmn() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_signal_changed(1, -60).await;
            task.perform_cycle().await;

            let mut serving: Option<Option<CellPlmn>> = None;
            while let Ok(msg) = nas_rx.try_recv() {
                if let TaskMessage::Message(NasMessage::ActiveCellChanged {
                    serving_plmn, ..
                }) = msg
                {
                    serving = Some(serving_plmn);
                }
            }
            let serving = serving
                .expect("ActiveCellChanged emitted on camp")
                .expect("a camped cell has read SIB1, so its PLMN is known");
            // The camped cell broadcasts the configured PLMN in this fixture, so the
            // value is checkable rather than merely present.
            let expected = crate::rrc::cell_selection::Plmn::new(
                test_config().hplmn.mcc,
                test_config().hplmn.mnc,
                test_config().hplmn.long_mnc,
            );
            assert_eq!(
                serving, expected,
                "the reported serving PLMN must be the one the cell broadcast"
            );
        });
    }

    /// `SetSelectedPlmn` makes the NAS selection drive cell selection (issue #49):
    /// after it, the RRC selector searches for the chosen PLMN and no longer for the
    /// configured home one.
    #[test]
    fn test_set_selected_plmn_redirects_cell_selection() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        let home = crate::rrc::cell_selection::Plmn::new(
            test_config().hplmn.mcc,
            test_config().hplmn.mnc,
            test_config().hplmn.long_mnc,
        );
        assert_eq!(
            task.cell_selector.selected_plmn(),
            Some(home),
            "RrcTask::new pins the configured home PLMN"
        );

        // A different PLMN, as automatic selection would choose after a rejection.
        let chosen = crate::rrc::cell_selection::Plmn::new(262, 30, true);
        task.cell_selector.set_selected_plmn(Some(chosen));
        assert_eq!(
            task.cell_selector.selected_plmn(),
            Some(chosen),
            "cell selection must now search for the PLMN NAS chose"
        );
    }

    /// T300 establishment guard (issue #30, TS 38.331 §5.3.3.2 / §5.3.3.4):
    /// started when the RRCSetupRequest is sent, stopped when RRCSetup arrives.
    #[test]
    fn test_t300_started_on_setup_request_stopped_on_rrc_setup() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            assert!(
                !task.t300_running(),
                "T300 not running before establishment"
            );

            camp_and_request(&mut task, &mut rls_rx).await;
            assert!(
                task.t300_running(),
                "T300 started when the RRCSetupRequest is sent"
            );

            // RRCSetup arrives → Connected, T300 stopped.
            task.handle_downlink_rrc(
                1,
                RrcChannel::DlCcch,
                OctetString::from_slice(&GOLDEN_RRC_SETUP_SRB1_TID0),
            )
            .await;
            assert_eq!(task.state_machine.state(), RrcState::Connected);
            assert!(
                !task.t300_running(),
                "T300 stopped when RRCSetup is received"
            );
        });
    }

    /// T300 expiry (issue #30, TS 38.331 §5.3.3.2): abandons the establishment
    /// attempt — the guard clears, the pending initial NAS is dropped, the RRC
    /// stays in RRC_IDLE, and NAS is told of the establishment failure.
    #[test]
    fn test_t300_expiry_aborts_establishment_and_notifies_nas() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, mut rls_rx) =
            UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            camp_and_request(&mut task, &mut rls_rx).await;
            assert!(task.t300_running());
            assert!(
                task.initial_nas_pdu.is_some(),
                "initial NAS pending during establishment"
            );

            task.on_t300_expiry().await;

            assert!(!task.t300_running(), "T300 cleared on expiry");
            assert!(
                task.initial_nas_pdu.is_none(),
                "pending initial NAS dropped on expiry"
            );
            assert_eq!(
                task.state_machine.state(),
                RrcState::Idle,
                "RRC stays Idle (establishment never left Idle)"
            );

            let mut got_failure = false;
            while let Ok(msg) = nas_rx.try_recv() {
                if let TaskMessage::Message(NasMessage::RrcEstablishmentFailure) = msg {
                    got_failure = true;
                }
            }
            assert!(
                got_failure,
                "NAS informed of RrcEstablishmentFailure on T300 expiry"
            );
        });
    }

    /// T300 is also stopped when the establishment attempt ends by RRCReject or
    /// by an upper-layer abort (TS 38.331 §5.3.3.2), not only on RRCSetup — so
    /// no spurious expiry fires afterwards.
    #[test]
    fn test_t300_stopped_on_reject_and_local_release() {
        // RRCReject (DL-CCCH msg type 0x01) stops T300.
        {
            let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) =
                UeTaskBase::new(test_config(), 32);
            let mut task = RrcTask::new(task_base);
            run_async(async {
                camp_and_request(&mut task, &mut rls_rx).await;
                assert!(task.t300_running());
                task.handle_downlink_rrc(1, RrcChannel::DlCcch, OctetString::from_slice(&[0x01]))
                    .await;
                assert!(!task.t300_running(), "T300 stopped on RRCReject");
            });
        }
        // Upper-layer abort (LocalReleaseConnection) stops T300.
        {
            let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) =
                UeTaskBase::new(test_config(), 32);
            let mut task = RrcTask::new(task_base);
            run_async(async {
                camp_and_request(&mut task, &mut rls_rx).await;
                assert!(task.t300_running());
                task.handle_local_release_connection(false).await;
                assert!(
                    !task.t300_running(),
                    "T300 stopped on upper-layer establishment abort"
                );
            });
        }
    }

    // ========================================================================
    // Conditional handover (issue #20, TS 38.331 §5.3.5.13)
    // ========================================================================

    use crate::rrc::TriggeringCell;
    use nextgsim_common::config::EutraNeighbourConfig;
    use nextgsim_rrc::procedures::conditional_handover::{
        encode_cho_config, A3Offset, ChoCandidateCell, ChoCondition, ChoConfig,
        ChoTargetCellConfig, EventA3Condition, Hysteresis, TimeToTrigger,
    };
    use nextgsim_rrc::procedures::scell_config::{encode_scell_config, ScellConfig};

    /// A container arming one A3 candidate on cell 2 with no time-to-trigger,
    /// wrapped in the DL-DCCH envelope the UE's reconfiguration handler expects.
    fn cho_reconfiguration_pdu(transaction_id: u8, target_pci: u16) -> OctetString {
        let config = ChoConfig {
            config_id: 3,
            candidate_cells: vec![ChoCandidateCell {
                candidate_index: 0,
                condition: ChoCondition::EventA3(EventA3Condition {
                    a3_offset: A3Offset::new(3.0).unwrap(),
                    hysteresis: Hysteresis::new(1.0).unwrap(),
                    time_to_trigger: TimeToTrigger::Ms0,
                    use_rsrp: true,
                }),
                target_cell: ChoTargetCellConfig {
                    phys_cell_id: target_pci,
                    ssb_frequency_arfcn: 620000,
                    ssb_subcarrier_spacing_khz: 30,
                    nr_cell_identity: None,
                    plmn_identity: None,
                    rrc_reconfiguration: None,
                },
                priority: 0,
            }],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        };

        let mut pdu = vec![RECONFIGURATION_WITH_CHO, transaction_id];
        pdu.extend_from_slice(&encode_cho_config(&config).expect("encode CHO container"));
        OctetString::from_slice(&pdu)
    }

    /// Camps, connects (golden RRCSetup) and returns with the UE in Connected on
    /// cell 1 at -90 dBm.
    async fn connect_on_cell_one(
        task: &mut RrcTask,
        rls_rx: &mut mpsc::Receiver<TaskMessage<RlsMessage>>,
    ) {
        camp_and_request(task, rls_rx).await;
        task.handle_downlink_rrc(
            1,
            RrcChannel::DlCcch,
            OctetString::from_slice(&GOLDEN_RRC_SETUP_SRB1_TID0),
        )
        .await;
        assert_eq!(task.state_machine.state(), RrcState::Connected);
        // Drain the RRCSetupComplete so a test can read the next uplink message
        // as the answer to what it sends.
        let _setup_complete = next_uplink_rrc(rls_rx);
        task.handle_signal_changed(1, -90).await;
    }

    // ========================================================================
    // Inter-RAT measurement source (issue #113, TS 38.331 §5.5.4.8/.9)
    // ========================================================================

    /// The whole point of #113: a B1 event that enters from the UE's *configured*
    /// inter-RAT neighbour list, with no test call to `update_eutra_measurement`
    /// and no test-installed measurement configuration either — the B1 `measId`
    /// comes from `setup_default_measurements` because the list is non-empty.
    #[test]
    fn a_b1_event_enters_from_the_configured_inter_rat_neighbours() {
        let mut config = test_config();
        config.eutra_b1_threshold_dbm = -100;
        config.eutra_neighbours = vec![
            EutraNeighbourConfig {
                earfcn: 1850,
                pci: 42,
                rsrp_dbm: -95, // -95 - 1 > -100: over the bar
                cell_individual_offset_db: 0,
                frequency_offset_db: 0,
            },
            EutraNeighbourConfig {
                earfcn: 1850,
                pci: 43,
                rsrp_dbm: -115, // well below it
                cell_individual_offset_db: 0,
                frequency_offset_db: 0,
            },
        ];
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            // Two cycles: the first starts the 640 ms time-to-trigger run.
            task.perform_cycle().await;
            task.measurement_manager
                .add_config(zero_ttt_b1_config(&task.task_base.config));
            task.perform_cycle().await;

            assert_eq!(
                task.measurement_manager
                    .triggered_cells(ZERO_TTT_B1_MEAS_ID),
                vec![TriggeringCell::Eutra(EutraCellKey::new(1850, 42))],
                "the stronger configured neighbour crossed b1-ThresholdEUTRA; \
                 the weaker one did not"
            );
        });
    }

    /// A `measId` copy of the default B1 configuration with the time-to-trigger
    /// removed, so a test does not have to sleep 640 ms. The *measurement source*
    /// is still the configuration, which is what #113 is about; only the
    /// trigger timing is shortened.
    const ZERO_TTT_B1_MEAS_ID: u8 = 20;

    fn zero_ttt_b1_config(config: &UeConfig) -> MeasConfig {
        MeasConfig {
            meas_id: ZERO_TTT_B1_MEAS_ID,
            meas_object_id: 2,
            report_config_id: 2,
            quantity: crate::rrc::measurement::MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::B1),
                threshold: Some(config.eutra_b1_threshold_dbm),
                threshold1: None,
                threshold2: None,
                a3_offset: None,
                a6_offset: None,
                hysteresis: 2,
                time_to_trigger: 0,
            },
            report_amount: 0,
            report_interval: 0,
            max_report_cells: 4,
        }
    }

    /// A non-empty neighbour list installs the default B1 `measId` on its own: a
    /// measurement source nothing evaluates would be inert.
    #[test]
    fn configured_inter_rat_neighbours_install_a_default_b1_meas_id() {
        for neighbours in [0usize, 1] {
            let mut config = test_config();
            config.eutra_neighbours = (0..neighbours)
                .map(|_| EutraNeighbourConfig {
                    earfcn: 1850,
                    pci: 42,
                    rsrp_dbm: -95,
                    cell_individual_offset_db: 0,
                    frequency_offset_db: 0,
                })
                .collect();
            let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
            let mut task = RrcTask::new(task_base);

            task.setup_default_measurements();

            // measId 1 is the default A3; measId 2 the default B1.
            assert_eq!(
                task.measurement_manager.config_count(),
                1 + neighbours,
                "{neighbours} configured neighbour(s) -> B1 measId present = {}",
                neighbours > 0
            );
        }
    }

    /// The per-carrier Ofn and per-cell Ocn from the configuration are part of the
    /// B1 inequality (B1-1 is Mn + Ofn + Ocn - Hys > Thresh). The same level with
    /// the offsets present crosses the bar and without them does not, so the test
    /// fails if the offsets never reach the measurement manager.
    #[test]
    fn the_configured_inter_rat_offsets_are_applied() {
        let cell = EutraCellKey::new(1850, 7);

        // Ofn 3 + Ocn 3: -104 + 6 - 1 = -99 > -100, so it enters.
        // Both zero:      -104 + 0 - 1 = -105, so it does not.
        for (ofn, ocn, expect_triggered) in [(3, 3, true), (0, 0, false)] {
            let mut config = test_config();
            config.eutra_b1_threshold_dbm = -100;
            config.eutra_neighbours = vec![EutraNeighbourConfig {
                earfcn: cell.earfcn,
                pci: cell.pci,
                rsrp_dbm: -104,
                cell_individual_offset_db: ocn,
                frequency_offset_db: ofn,
            }];
            let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
            let mut task = RrcTask::new(task_base);

            run_async(async {
                connect_on_cell_one(&mut task, &mut rls_rx).await;
                task.measurement_manager
                    .add_config(zero_ttt_b1_config(&task.task_base.config));
                task.perform_cycle().await;

                let stored = task
                    .measurement_manager
                    .eutra_measurements()
                    .get(&cell)
                    .expect("the configured neighbour is measured");
                assert_eq!(stored.rsrp, Some(-104));
                assert_eq!(
                    stored.cell_individual_offset, ocn,
                    "Ocn reached the measurement manager"
                );

                let triggered = task
                    .measurement_manager
                    .triggered_cells(ZERO_TTT_B1_MEAS_ID);
                if expect_triggered {
                    assert_eq!(
                        triggered,
                        vec![TriggeringCell::Eutra(cell)],
                        "Ofn {ofn} + Ocn {ocn} lifts -104 dBm over the -99 dBm bar"
                    );
                } else {
                    assert!(
                        triggered.is_empty(),
                        "with no offsets -104 dBm is below the bar"
                    );
                }
            });
        }
    }

    /// With no configured neighbours nothing is measured, so B1/B2 cannot enter —
    /// the pre-#113 behaviour, unchanged.
    #[test]
    fn no_configured_inter_rat_neighbours_means_no_inter_rat_measurements() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.measurement_manager
                .add_config(zero_ttt_b1_config(&task.task_base.config));
            task.perform_cycle().await;

            assert!(task.measurement_manager.eutra_measurements().is_empty());
            assert!(task
                .measurement_manager
                .triggered_cells(ZERO_TTT_B1_MEAS_ID)
                .is_empty());
        });
    }

    /// An RRCReconfiguration adding one secondary cell, in the DL-DCCH envelope
    /// the UE's reconfiguration handler expects.
    fn scell_reconfiguration_pdu(transaction_id: u8, config: &ScellConfig) -> OctetString {
        let mut pdu = vec![RECONFIGURATION_WITH_SCELL, transaction_id];
        pdu.extend_from_slice(&encode_scell_config(config).expect("encode SCell container"));
        OctetString::from_slice(&pdu)
    }

    /// An A6 measurement configuration: `a6-Offset` 3 dB, hysteresis 1 dB, no
    /// time-to-trigger. A neighbour must beat the SCell by 4 dB to enter.
    fn a6_meas_config() -> MeasConfig {
        MeasConfig {
            meas_id: 6,
            meas_object_id: 1,
            report_config_id: 1,
            quantity: crate::rrc::measurement::MeasQuantity::SsRsrp,
            trigger_config: ReportTriggerConfig {
                trigger_type: ReportTriggerType::Event(MeasEventType::A6),
                threshold: None,
                threshold1: None,
                threshold2: None,
                a3_offset: None,
                a6_offset: Some(3),
                hysteresis: 2, // 1 dB
                time_to_trigger: 0,
            },
            report_amount: 0,
            report_interval: 0,
            max_report_cells: 4,
        }
    }

    /// The whole point of #112: A6 measures against the SCell (TS 38.331
    /// §5.5.4.7), and until an RRCReconfiguration names one the event cannot
    /// enter however strong a neighbour is. The SCell here is configured over the
    /// wire, not by calling `set_scell`.
    #[test]
    fn an_scell_from_a_reconfiguration_makes_event_a6_enter() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.measurement_manager.add_config(a6_meas_config());

            // Cell 2 will be the SCell at -95 dBm; cell 3 beats it by 5 dB.
            task.handle_signal_changed(2, -95).await;
            task.handle_signal_changed(3, -90).await;
            task.perform_cycle().await;
            assert!(
                task.measurement_manager.triggered_cells(6).is_empty(),
                "with no SCell configured, A6 has no reference and cannot enter"
            );

            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(4, &ScellConfig::add_one(1, 2)),
            )
            .await;
            assert_eq!(
                task.configured_scells(),
                vec![2],
                "the reconfiguration configured cell 2 as the SCell"
            );
            let (channel, pdu) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(channel, RrcChannel::UlDcch);
            assert_eq!(
                pdu.data(),
                &build_reconfiguration_complete(4)[..],
                "RRCReconfigurationComplete echoes the transaction id"
            );

            task.perform_cycle().await;
            assert_eq!(
                task.measurement_manager.triggered_cells(6),
                vec![TriggeringCell::Nr(3)],
                "cell 3 at -90 dBm beats the -91 dBm A6 bar over the SCell"
            );
        });
    }

    /// §5.3.5.5.9: `sCellToReleaseList` removes it again, and A6 stops being
    /// evaluable — the neighbour has not moved.
    #[test]
    fn releasing_the_scell_stops_event_a6_being_evaluable() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.measurement_manager.add_config(a6_meas_config());
            task.handle_signal_changed(2, -95).await;
            task.handle_signal_changed(3, -90).await;
            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(1, &ScellConfig::add_one(1, 2)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);
            task.perform_cycle().await;
            assert_eq!(
                task.measurement_manager.triggered_cells(6),
                vec![TriggeringCell::Nr(3)]
            );

            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(2, &ScellConfig::release_one(1)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);

            assert!(task.configured_scells().is_empty());
            assert_eq!(task.measurement_manager.scell_id(), None);
            task.perform_cycle().await;
            assert!(
                task.measurement_manager.triggered_cells(6).is_empty(),
                "no SCell, no A6 -- cell 3 is still at -90 dBm"
            );
        });
    }

    /// §5.3.11: going to RRC_IDLE releases the RRC configuration, secondary
    /// cells included.
    #[test]
    fn going_to_rrc_idle_releases_the_scell() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.handle_signal_changed(2, -95).await;
            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(1, &ScellConfig::add_one(1, 2)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);
            assert_eq!(task.configured_scells(), vec![2]);

            task.handle_rrc_release().await;

            assert_eq!(task.state_machine.state(), RrcState::Idle);
            assert!(task.configured_scells().is_empty());
            assert_eq!(task.measurement_manager.scell_id(), None);
        });
    }

    /// A release naming an unconfigured index is reported, not silently applied
    /// to whatever SCell happens to be there.
    #[test]
    fn releasing_an_unconfigured_scell_index_leaves_the_configured_one_alone() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.handle_signal_changed(2, -95).await;
            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(1, &ScellConfig::add_one(1, 2)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);

            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(2, &ScellConfig::release_one(9)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);

            assert_eq!(
                task.configured_scells(),
                vec![2],
                "sCellIndex 9 was never configured, so index 1 survives"
            );
        });
    }

    /// A cell is not its own SCell: a configuration naming the serving cell
    /// records it but must not make A6 measure the PCell against itself, which
    /// would let any neighbour above the PCell trigger a handover event that
    /// §5.5.4.7 says is about the secondary cell.
    #[test]
    fn the_serving_cell_is_not_accepted_as_its_own_scell_reference() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.measurement_manager.add_config(a6_meas_config());
            task.handle_signal_changed(3, -50).await; // far stronger than the PCell

            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(1, &ScellConfig::add_one(1, 1)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);

            assert_eq!(task.measurement_manager.scell_id(), None);
            task.perform_cycle().await;
            assert!(
                task.measurement_manager.triggered_cells(6).is_empty(),
                "the serving cell is not an A6 reference, whatever cell 3 does"
            );
        });
    }

    /// A malformed container must not be applied and must not take the existing
    /// configuration down with it; the reconfiguration is still acknowledged.
    #[test]
    fn a_malformed_scell_container_changes_nothing() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.handle_signal_changed(2, -95).await;
            task.handle_rrc_reconfiguration(
                1,
                &scell_reconfiguration_pdu(1, &ScellConfig::add_one(1, 2)),
            )
            .await;
            let _complete = next_uplink_rrc(&mut rls_rx);
            assert_eq!(task.configured_scells(), vec![2]);

            let pdu = OctetString::from_slice(&[RECONFIGURATION_WITH_SCELL, 7, 0xFF]);
            task.handle_rrc_reconfiguration(1, &pdu).await;

            assert_eq!(
                task.configured_scells(),
                vec![2],
                "a container that does not decode leaves the SCell configured"
            );
            let (_, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(complete.data(), &build_reconfiguration_complete(7)[..]);
        });
    }

    /// The container is decoded and acknowledged either way, but the candidates
    /// are only stored when the runtime switch is on — with it off, mobility is
    /// unchanged.
    #[test]
    fn a_cho_container_arms_the_store_only_when_the_switch_is_on() {
        for enabled in [false, true] {
            let mut config = test_config();
            config.conditional_handover = enabled;
            let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
            let mut task = RrcTask::new(task_base);

            run_async(async {
                connect_on_cell_one(&mut task, &mut rls_rx).await;
                task.handle_rrc_reconfiguration(1, &cho_reconfiguration_pdu(5, 2))
                    .await;

                assert_eq!(
                    task.cho_candidate_count(),
                    usize::from(enabled),
                    "candidates stored only with conditional_handover = {enabled}"
                );

                // The reconfiguration itself is accepted in both cases.
                let (channel, pdu) = next_uplink_rrc(&mut rls_rx);
                assert_eq!(channel, RrcChannel::UlDcch);
                assert_eq!(
                    pdu.data(),
                    &build_reconfiguration_complete(5)[..],
                    "RRCReconfigurationComplete echoes the transaction id"
                );
            });
        }
    }

    /// The end of the path: a candidate that arrived over the wire executes a
    /// handover once its own cell beats the serving cell, with no handover
    /// command from the network.
    #[test]
    fn a_stored_candidate_hands_the_ue_over_without_a_network_command() {
        let mut config = test_config();
        config.conditional_handover = true;
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.handle_rrc_reconfiguration(1, &cho_reconfiguration_pdu(9, 2))
                .await;
            let _complete = next_uplink_rrc(&mut rls_rx);
            assert_eq!(task.cho_candidate_count(), 1);

            // Cell 2 is visible but no better than the serving cell: nothing fires.
            task.handle_signal_changed(2, -91).await;
            task.perform_cycle().await;
            assert_eq!(
                task.serving_cell_id,
                Some(1),
                "a candidate below the A3 bar must not trigger"
            );
            assert_eq!(task.cho_candidate_count(), 1, "and stays stored");

            // Cell 2 pulls ahead by more than a3-Offset + hysteresis.
            task.handle_signal_changed(2, -80).await;
            task.perform_cycle().await;

            assert_eq!(
                task.serving_cell_id,
                Some(2),
                "the UE executed the conditional handover itself"
            );
            assert!(
                task.handover_manager.last_handover_duration().is_some(),
                "the handover ran through the handover manager"
            );
            assert_eq!(
                task.cho_candidate_count(),
                0,
                "§5.3.5.3: applying the reconfigurationWithSync releases the candidates"
            );

            // An RRCReconfigurationComplete for the executed candidate, carrying
            // the transaction id of the message that armed it.
            let mut saw_complete = false;
            while let Ok(msg) = rls_rx.try_recv() {
                if let TaskMessage::Message(RlsMessage::RrcPduDelivery { pdu, .. }) = msg {
                    if pdu.data() == &build_reconfiguration_complete(9)[..] {
                        saw_complete = true;
                    }
                }
            }
            assert!(
                saw_complete,
                "the handover is completed towards the network"
            );
        });
    }

    /// §5.3.11: going to RRC_IDLE removes every entry in VarConditionalReconfig.
    #[test]
    fn releasing_the_connection_drops_the_stored_candidates() {
        let mut config = test_config();
        config.conditional_handover = true;
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            task.handle_rrc_reconfiguration(1, &cho_reconfiguration_pdu(1, 2))
                .await;
            assert_eq!(task.cho_candidate_count(), 1);

            task.handle_rrc_release().await;
            assert_eq!(task.cho_candidate_count(), 0);
        });
    }

    /// A container the UE cannot decode is reported, not stored, and the
    /// reconfiguration is still acknowledged.
    #[test]
    fn a_malformed_cho_container_stores_nothing() {
        let mut config = test_config();
        config.conditional_handover = true;
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(config, 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            connect_on_cell_one(&mut task, &mut rls_rx).await;
            // Header claims two candidates and carries none.
            let pdu = OctetString::from_slice(&[RECONFIGURATION_WITH_CHO, 4, 3, 2, 0]);
            task.handle_rrc_reconfiguration(1, &pdu).await;

            assert_eq!(task.cho_candidate_count(), 0);
            let (_, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(complete.data(), &build_reconfiguration_complete(4)[..]);
        });
    }

    /// C2 gate: feeding the C1 golden RRCSetup bytes transitions the UE to
    /// Connected, records SRB1 (srb-Identity 1, LCID 1) BEFORE any DL-DCCH
    /// handling, and emits an RRCSetupComplete with tid 0.
    #[test]
    fn test_rrc_setup_golden_srb1_decoded_connected_tid0_echo() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            let nas = camp_and_request(&mut task, &mut rls_rx).await;

            task.handle_downlink_rrc(
                1,
                RrcChannel::DlCcch,
                OctetString::from_slice(&GOLDEN_RRC_SETUP_SRB1_TID0),
            )
            .await;

            assert_eq!(task.state_machine.state(), RrcState::Connected);

            // SRB1 recorded — and recorded BEFORE any SRB1-labelled DL-DCCH
            // (e.g. SecurityModeCommand) could be handled.
            let srb1 = task
                .srb1_config
                .as_ref()
                .expect("SRB1 must be recorded from the golden RRCSetup");
            assert_eq!(srb1.rrc_transaction_id, 0);
            let srbs = srb1
                .radio_bearer_config
                .srb_to_add_mod_list
                .as_ref()
                .expect("srb-ToAddModList present");
            assert_eq!(srbs.0.len(), 1);
            assert_eq!(srbs.0[0].srb_identity.0, 1, "SRB-Identity must be 1");
            let cgc = srb1
                .cell_group_config
                .as_ref()
                .expect("masterCellGroup must decode as CellGroupConfig");
            let bearer = &cgc
                .rlc_bearer_to_add_mod_list
                .as_ref()
                .expect("rlc-BearerToAddModList present")
                .0[0];
            assert_eq!(bearer.logical_channel_identity.0, 1, "SRB1 LCID must be 1");

            // RRCSetupComplete emitted on UL-DCCH, tid echo pinned to 0
            // (Wave-6 C4-interim; unpinned after C5), NAS intact.
            let (ch, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(ch, RrcChannel::UlDcch);
            let decoded =
                decode_rrc_setup_complete(complete.data()).expect("ASN.1 RRCSetupComplete");
            assert_eq!(
                decoded.rrc_transaction_id, 0,
                "tid echo pinned to 0 until C5"
            );
            assert_eq!(decoded.dedicated_nas_message, nas);
        });
    }

    /// C2 tolerance gate: the pre-Wave-6 placeholder RRCSetup (no SRB1) must
    /// produce IDENTICAL external behavior — Connected + RRCSetupComplete —
    /// with a warning and no SRB1 recorded.
    #[test]
    fn test_rrc_setup_legacy_placeholder_tolerated() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            let nas = camp_and_request(&mut task, &mut rls_rx).await;

            task.handle_downlink_rrc(
                1,
                RrcChannel::DlCcch,
                OctetString::from_slice(&LEGACY_PLACEHOLDER_RRC_SETUP),
            )
            .await;

            // Identical external behavior: Connected + SetupComplete...
            assert_eq!(task.state_machine.state(), RrcState::Connected);
            let (ch, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(ch, RrcChannel::UlDcch);
            let decoded =
                decode_rrc_setup_complete(complete.data()).expect("ASN.1 RRCSetupComplete");
            assert_eq!(decoded.rrc_transaction_id, 0);
            assert_eq!(decoded.dedicated_nas_message, nas);

            // ...but no SRB1 recorded (warn logged on the tolerance path).
            assert!(
                task.srb1_config.is_none(),
                "placeholder RRCSetup establishes no SRB1"
            );
        });
    }

    // ========================================================================
    // Wave-6 I5: UE AS-security activation. The SMC is built with the gNB's
    // OWN encoder (nextgsim-rrc `encode_security_mode_command`, called by
    // nextgsim-gnb `activate_as_security`) and the derived keys are checked
    // against the gNB's OWN `derive_rrc_up_key` — a genuine strict-peer oracle
    // over the shared libs, and the SecurityModeComplete is decoded by the
    // gNB's OWN typed UL-DCCH dispatcher (`dispatch_ul_dcch`, C5).
    // ========================================================================

    use nextgsim_crypto::kdf::{derive_rrc_up_key, AlgorithmTypeDistinguisher};
    use nextgsim_rrc::procedures::dcch_dispatch::{dispatch_ul_dcch, UlDcchMessage};
    use nextgsim_rrc::procedures::security_mode::{
        decode_security_mode_complete, encode_security_mode_command, CipheringAlgorithmType,
        IntegrityAlgorithmType, SecurityAlgorithms, SecurityModeCommandParams,
    };

    const TEST_KGNB: [u8; 32] = [0x11u8; 32];

    fn gnb_smc_bytes(tid: u8) -> Vec<u8> {
        // The exact algorithms nextgsim-gnb selects at Initial Context Setup
        // for the matched sim: NEA0 ciphering, NIA2 integrity.
        encode_security_mode_command(&SecurityModeCommandParams {
            rrc_transaction_id: tid,
            security_algorithms: SecurityAlgorithms {
                ciphering_algorithm: CipheringAlgorithmType::Nea0,
                integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
            },
        })
        .expect("gNB encodes SecurityModeCommand")
    }

    /// The MAC-I the **gNB** appends to a SecurityModeCommand (issue #31).
    ///
    /// Computed here the way the gNB computes it — through the shared
    /// `nextgsim-pdcp` layer, with the keys derived from the same KgNB — so these
    /// tests fail if the gNB's protect and the UE's verify ever disagree. That is
    /// the whole point of hoisting the layer.
    fn gnb_smc_mac_i(pdu: &[u8]) -> [u8; MAC_I_LEN] {
        use nextgsim_pdcp::srb_security::SrbSecurity;
        let k_rrc_int = derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcInt, 2);
        let k_rrc_enc = derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcEnc, 0);
        // Integrity only: the SMC is not ciphered (TS 38.331 §5.3.4.2).
        SrbSecurity::new(k_rrc_enc, k_rrc_int, 0, 2)
            .expect("legal ids")
            .compute_mac_i(SMC_PDCP_COUNT, SRB1_BEARER, DIRECTION_DOWNLINK, pdu)
    }

    /// Hands the UE a SecurityModeCommand exactly as the gNB would: the UPER PDU
    /// followed by its MAC-I.
    async fn deliver_gnb_smc(task: &mut RrcTask, tid: u8) {
        let pdu = gnb_smc_bytes(tid);
        let mac = gnb_smc_mac_i(&pdu);
        let smc = decode_security_mode_command(&pdu).expect("decode SMC");
        task.handle_as_security_mode_command(smc, &pdu, mac).await;
    }

    #[test]
    fn test_as_smc_derives_keys_and_completes_strict_peer() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            // NAS plane handed us KgNB (TS 33.501 §6.9.4.1).
            task.set_pending_kgnb(TEST_KGNB);

            deliver_gnb_smc(&mut task, 0).await;

            // AS security context installed; keys byte-identical to the gNB's
            // own derive_rrc_up_key(KgNB, ...) (TS 33.501 Annex A.8).
            let ctx = task.as_security().expect("AS security must be active");
            assert_eq!(
                ctx.k_rrc_int,
                derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcInt, 2)
            );
            assert_eq!(
                ctx.k_rrc_enc,
                derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcEnc, 0)
            );

            // SecurityModeComplete emitted on UL-DCCH, decodable by the gNB's
            // OWN typed UL-DCCH dispatcher (C5) with the echoed tid.
            let (ch, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(ch, RrcChannel::UlDcch);
            match dispatch_ul_dcch(complete.data()).expect("gNB typed dispatch") {
                UlDcchMessage::SecurityModeComplete(d) => assert_eq!(d.rrc_transaction_id, 0),
                other => panic!("expected SecurityModeComplete, got {other:?}"),
            }
            assert_eq!(
                decode_security_mode_complete(complete.data())
                    .unwrap()
                    .rrc_transaction_id,
                0
            );
        });
    }

    #[test]
    fn test_as_smc_echoes_tid() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            task.set_pending_kgnb(TEST_KGNB);
            deliver_gnb_smc(&mut task, 3).await;
            let (_ch, complete) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(
                decode_security_mode_complete(complete.data())
                    .unwrap()
                    .rrc_transaction_id,
                3,
                "SecurityModeComplete must echo the SMC transaction id (TS 38.331 §5.3.4.3)"
            );
        });
    }

    /// #31, criterion 1: AS security activation is controlled by CONFIG, not by a
    /// compile-time constant.
    ///
    /// It used to be `pub const I5_UE_AS_SECURITY: bool = false;`, so **no shipping
    /// build could activate AS security at all**, whatever the operator set. This
    /// asserts the predicate follows the config in both directions — a revert round
    /// that pinned it back to `false` must fail here.
    #[test]
    fn as_security_activation_follows_the_config_not_a_constant() {
        use crate::rrc::security::as_security_enabled;

        let off = test_config();
        assert!(
            !as_security_enabled(&off),
            "the default must stay off: enabling it drops all UL-DCCH traffic while \
             the gNB dispatches on raw bytes"
        );

        let on = nextgsim_common::config::UeConfig {
            as_security_enabled: true,
            ..test_config()
        };
        assert!(
            as_security_enabled(&on),
            "an operator that turned it ON must get it: a hard-coded false made the \
             configuration unactionable"
        );
    }

    /// #31, criterion 3: a SecurityModeCommand whose MAC-I does not verify is
    /// **refused**, the previous configuration is kept, and the UE answers
    /// `SecurityModeFailure` (TS 38.331 §5.3.4.2, §5.3.4.4).
    ///
    /// Before this the command was accepted unverified: anything that could put
    /// bytes on SRB1 could activate AS security with keys of its choosing, and the
    /// UE logged "AS security activated" for it.
    #[test]
    fn a_security_mode_command_with_a_forged_mac_is_refused_with_a_failure() {
        use nextgsim_rrc::procedures::security_mode::decode_security_mode_failure;

        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.set_pending_kgnb(TEST_KGNB);

            let pdu = gnb_smc_bytes(2);
            let mut mac = gnb_smc_mac_i(&pdu);
            mac[0] ^= 0xFF; // forged
            let smc = decode_security_mode_command(&pdu).expect("decode SMC");
            task.handle_as_security_mode_command(smc, &pdu, mac).await;

            assert!(
                task.as_security().is_none(),
                "a command that failed integrity must NOT activate AS security"
            );
            let (ch, failure) = next_uplink_rrc(&mut rls_rx);
            assert_eq!(ch, RrcChannel::UlDcch);
            assert_eq!(
                decode_security_mode_failure(failure.data()).expect("a SecurityModeFailure"),
                2,
                "the failure must echo the refused command's transaction id, or the \
                 gNB cannot tell which command failed"
            );
        });
    }

    /// The positive control for the test above: the SAME command with its real
    /// MAC-I activates and answers Complete. Without it, "refused" could mean the
    /// handler refuses everything.
    #[test]
    fn a_security_mode_command_with_a_valid_mac_activates_and_completes() {
        use nextgsim_rrc::procedures::security_mode::decode_security_mode_failure;

        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.set_pending_kgnb(TEST_KGNB);
            deliver_gnb_smc(&mut task, 2).await;

            assert!(
                task.as_security().is_some(),
                "the valid command must activate"
            );
            let (_ch, reply) = next_uplink_rrc(&mut rls_rx);
            assert!(
                decode_security_mode_failure(reply.data()).is_err(),
                "and the reply must be a Complete, not a Failure"
            );
            assert_eq!(
                decode_security_mode_complete(reply.data())
                    .expect("SecurityModeComplete")
                    .rrc_transaction_id,
                2
            );
        });
    }

    /// NIA0 is refused: its MAC is all zeros, so "the MAC verified" would prove
    /// nothing, and TS 33.501 §5.11.1 does not permit NIA0 for SRBs outside
    /// unauthenticated emergency service.
    #[test]
    fn a_security_mode_command_selecting_nia0_is_refused() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.set_pending_kgnb(TEST_KGNB);
            let pdu = encode_security_mode_command(&SecurityModeCommandParams {
                rrc_transaction_id: 0,
                security_algorithms: SecurityAlgorithms {
                    ciphering_algorithm: CipheringAlgorithmType::Nea0,
                    integrity_algorithm: Some(IntegrityAlgorithmType::Nia0),
                },
            })
            .expect("encodes");
            let smc = decode_security_mode_command(&pdu).expect("decode");
            // An all-zero MAC, which is what NIA0 produces -- so this would verify
            // if NIA0 were accepted.
            task.handle_as_security_mode_command(smc, &pdu, [0u8; MAC_I_LEN])
                .await;

            assert!(
                task.as_security().is_none(),
                "NIA0 must not activate AS security: an all-zero MAC verifies for \
                 anyone"
            );
            assert!(
                rls_rx.try_recv().is_ok(),
                "and the refusal must be answered, not silent"
            );
        });
    }

    /// NEA3 is accepted now (criterion 6). It used to abort activation here.
    #[test]
    fn a_security_mode_command_selecting_nea3_activates() {
        use crate::rrc::security::DIRECTION_UPLINK;
        use nextgsim_pdcp::srb_security::SrbSecurity;

        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.set_pending_kgnb(TEST_KGNB);
            let pdu = encode_security_mode_command(&SecurityModeCommandParams {
                rrc_transaction_id: 1,
                security_algorithms: SecurityAlgorithms {
                    ciphering_algorithm: CipheringAlgorithmType::Nea3,
                    integrity_algorithm: Some(IntegrityAlgorithmType::Nia2),
                },
            })
            .expect("encodes");
            // The gNB's MAC for a NEA3 command: integrity is still NIA2 and the SMC
            // is unciphered, so the ciphering choice does not enter the MAC.
            let k_int = derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcInt, 2);
            let k_enc = derive_rrc_up_key(&TEST_KGNB, AlgorithmTypeDistinguisher::RrcEnc, 3);
            let mac = SrbSecurity::new(k_enc, k_int, 0, 2)
                .expect("ids")
                .compute_mac_i(SMC_PDCP_COUNT, SRB1_BEARER, DIRECTION_DOWNLINK, &pdu);
            let smc = decode_security_mode_command(&pdu).expect("decode");
            task.handle_as_security_mode_command(smc, &pdu, mac).await;

            let ctx = task
                .as_security()
                .expect("NEA3 must activate, not abort (issue #31 criterion 6)");
            assert_eq!(ctx.ciphering_algorithm, CipheringAlgorithm::Nea3);
            // And the activated context can actually protect a PDU with NEA3.
            let protected = ctx
                .protect_srb(1, SRB1_BEARER, DIRECTION_UPLINK, &[0x20, 0x08])
                .expect("NEA3 must protect");
            assert_eq!(
                ctx.unprotect_srb(1, SRB1_BEARER, DIRECTION_UPLINK, &protected)
                    .expect("and verify"),
                vec![0x20, 0x08]
            );
            let _ = next_uplink_rrc(&mut rls_rx);
        });
    }

    #[test]
    fn test_as_smc_without_kgnb_fails_closed() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            // No KgNB installed → must NOT activate and must send nothing.
            deliver_gnb_smc(&mut task, 1).await;

            assert!(task.as_security().is_none(), "no KgNB → no activation");
            assert!(
                rls_rx.try_recv().is_err(),
                "fail-closed: no SecurityModeComplete without KgNB"
            );
        });
    }

    #[test]
    fn test_as_security_key_message_stores_kgnb() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        assert!(task.pending_kgnb.is_none());
        task.set_pending_kgnb(TEST_KGNB);
        assert_eq!(task.pending_kgnb, Some(TEST_KGNB));
    }

    // ========================================================================
    // Broadcast system information (#21, TS 38.331 §5.2.1)
    // ========================================================================

    /// The gNB's own encoders build the fixtures, so the test exercises the pair
    /// the two sides actually use rather than a hand-rolled PDU.
    fn broadcast_si(nci: u64, tac: u32, mcc: u16, mnc: u16, long_mnc: bool) -> (Vec<u8>, Vec<u8>) {
        use nextgsim_rrc::procedures::system_information::{
            encode_mib, encode_sib1, CellBarredStatus, CellSelectionInfo, DmrsTypeAPosition,
            IntraFreqReselection, MibParams, PdcchConfigSib1Params, PlmnIdentityInfo, Sib1Params,
            SubCarrierSpacingCommon,
        };

        let mib = encode_mib(&MibParams {
            system_frame_number: 0,
            sub_carrier_spacing_common: SubCarrierSpacingCommon::Scs30Or120,
            ssb_subcarrier_offset: 0,
            dmrs_type_a_position: DmrsTypeAPosition::Pos2,
            pdcch_config_sib1: PdcchConfigSib1Params {
                coreset_zero: 0,
                search_space_zero: 0,
            },
            cell_barred: CellBarredStatus::NotBarred,
            intra_freq_reselection: IntraFreqReselection::Allowed,
        })
        .expect("MIB");

        let mnc_digits = if long_mnc {
            vec![
                ((mnc / 100) % 10) as u8,
                ((mnc / 10) % 10) as u8,
                (mnc % 10) as u8,
            ]
        } else {
            vec![((mnc / 10) % 10) as u8, (mnc % 10) as u8]
        };
        let sib1 = encode_sib1(&Sib1Params {
            cell_selection_info: Some(CellSelectionInfo {
                q_rx_lev_min: -70,
                q_rx_lev_min_offset: None,
                q_rx_lev_min_sul: None,
                q_qual_min: None,
                q_qual_min_offset: None,
            }),
            plmn_identity_info_list: vec![PlmnIdentityInfo {
                plmn_identity_list: vec![SibPlmnIdentity {
                    mcc: Some([
                        ((mcc / 100) % 10) as u8,
                        ((mcc / 10) % 10) as u8,
                        (mcc % 10) as u8,
                    ]),
                    mnc: mnc_digits,
                }],
                tracking_area_code: Some(tac),
                cell_identity: nci,
            }],
            ims_emergency_support: false,
            ecall_over_ims_support: false,
            ue_timers_and_constants: None,
            intra_freq_reselection_redcap: false,
        })
        .expect("SIB1");

        (mib, sib1)
    }

    /// A broadcast SIB1 replaces the UE's assumptions with what the cell actually
    /// advertises: PLMN, TAC and NCI all come off the air.
    #[test]
    fn a_broadcast_sib1_supplies_the_cells_real_identity() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        let (mib, sib1) = broadcast_si(0xABCD, 42, 262, 2, false);

        run_async(async {
            // The cell is discovered from its RLS heartbeat first, which is also
            // what installs the fabricated system info the broadcast must replace.
            task.handle_signal_changed(7, -60).await;
            task.handle_downlink_rrc(7, RrcChannel::BcchBch, OctetString::from_slice(&mib))
                .await;
            task.handle_downlink_rrc(7, RrcChannel::BcchDlSch, OctetString::from_slice(&sib1))
                .await;
        });

        let cell = task
            .cell_selector
            .get_cell(7)
            .expect("the cell must be known from its broadcast");
        assert!(cell.mib.has_mib && !cell.mib.is_barred);
        assert!(cell.sib1.has_sib1);
        assert_eq!(cell.sib1.nci, 0xABCD, "the NCI the cell broadcast");
        assert_eq!(cell.sib1.tac, 42, "the TAC the cell broadcast");
        assert_eq!(
            cell.sib1.plmn,
            CellPlmn::new(262, 2, false),
            "a PLMN this UE did NOT configure -- it came off the air -- with the \
             broadcast's TWO-digit MNC preserved (262-02 is not 262-002)"
        );
        assert_eq!(
            cell.sib1.q_rx_lev_min, -70,
            "q-RxLevMin from the broadcast cellSelectionInfo"
        );
    }

    /// The broadcast MIB decides whether the cell is barred. The fabricated one
    /// always says "not barred", so only a real broadcast can produce a barred
    /// cell — which is why this asserts on `Barred` rather than on `has_mib`.
    #[test]
    fn a_broadcast_mib_can_bar_a_cell_the_fallback_called_usable() {
        use nextgsim_rrc::procedures::system_information::{
            encode_mib, CellBarredStatus, DmrsTypeAPosition, IntraFreqReselection, MibParams,
            PdcchConfigSib1Params, SubCarrierSpacingCommon,
        };

        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        let barred = encode_mib(&MibParams {
            system_frame_number: 0,
            sub_carrier_spacing_common: SubCarrierSpacingCommon::Scs30Or120,
            ssb_subcarrier_offset: 0,
            dmrs_type_a_position: DmrsTypeAPosition::Pos2,
            pdcch_config_sib1: PdcchConfigSib1Params {
                coreset_zero: 0,
                search_space_zero: 0,
            },
            cell_barred: CellBarredStatus::Barred,
            intra_freq_reselection: IntraFreqReselection::NotAllowed,
        })
        .expect("MIB");

        run_async(async {
            task.handle_signal_changed(11, -60).await;
            assert!(
                !task.cell_selector.get_cell(11).unwrap().mib.is_barred,
                "precondition: the fabricated MIB says the cell is usable"
            );
            task.handle_downlink_rrc(11, RrcChannel::BcchBch, OctetString::from_slice(&barred))
                .await;
        });

        let mib = &task.cell_selector.get_cell(11).unwrap().mib;
        assert!(mib.is_barred, "the broadcast MIB bars the cell");
        assert!(
            !mib.is_intra_freq_reselect_allowed,
            "and forbids intra-frequency reselection"
        );
    }

    /// A SIB1 whose PLMN identity omits the MCC cannot be resolved from a single
    /// entry (TS 38.331 lets it mean "same as the previous entry"), so it must not
    /// be stored as PLMN 000 — a UE would then treat the cell as a network that
    /// does not exist.
    #[test]
    fn a_broadcast_plmn_without_an_mcc_is_not_stored() {
        use nextgsim_rrc::procedures::system_information::{
            encode_sib1, PlmnIdentityInfo, Sib1Params,
        };

        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        let sib1 = encode_sib1(&Sib1Params {
            cell_selection_info: None,
            plmn_identity_info_list: vec![PlmnIdentityInfo {
                plmn_identity_list: vec![SibPlmnIdentity {
                    mcc: None,
                    mnc: vec![0, 1],
                }],
                tracking_area_code: Some(1),
                cell_identity: 0x99,
            }],
            ims_emergency_support: false,
            ecall_over_ims_support: false,
            ue_timers_and_constants: None,
            intra_freq_reselection_redcap: false,
        })
        .expect("SIB1");

        run_async(async {
            task.handle_signal_changed(12, -60).await;
            task.handle_downlink_rrc(12, RrcChannel::BcchDlSch, OctetString::from_slice(&sib1))
                .await;
        });

        assert!(
            !task.cells_with_broadcast_si.contains(&12),
            "an unusable PLMN must not count as system information"
        );
        assert_ne!(
            task.cell_selector.get_cell(12).unwrap().sib1.nci,
            0x99,
            "and its cell identity must not be stored either"
        );
    }

    /// Once a cell has broadcast its system information, the simulated fallback
    /// must not overwrite it with the UE's own assumptions.
    #[test]
    fn the_simulated_fallback_does_not_overwrite_a_broadcast() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        let (_, sib1) = broadcast_si(0x1234, 9, 262, 2, false);

        run_async(async {
            task.handle_signal_changed(3, -60).await;
            task.handle_downlink_rrc(3, RrcChannel::BcchDlSch, OctetString::from_slice(&sib1))
                .await;
        });
        task.provide_simulated_system_info(3);

        let cell = task.cell_selector.get_cell(3).expect("known cell");
        assert_eq!(
            cell.sib1.nci, 0x1234,
            "still the broadcast NCI, not the cell id"
        );
        assert_eq!(
            cell.sib1.tac, 9,
            "still the broadcast TAC, not the default 1"
        );
    }

    /// With `require_broadcast_sib1` set, a cell that broadcasts nothing never
    /// gets fabricated system information, so it cannot be selected.
    #[test]
    fn requiring_a_broadcast_sib1_suppresses_the_simulated_fallback() {
        let config = UeConfig {
            require_broadcast_sib1: true,
            ..test_config()
        };
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);

        run_async(async { task.handle_signal_changed(5, -60).await });

        let cell = task
            .cell_selector
            .get_cell(5)
            .expect("the cell is still detected from its heartbeat");
        assert!(
            !cell.sib1.has_sib1,
            "no broadcast, no fabricated SIB1, so the cell is not selectable"
        );
    }

    /// Default configuration keeps the pre-#21 behaviour: the fallback fabricates
    /// system information so heartbeat-only discovery still selects a cell.
    #[test]
    fn by_default_the_simulated_fallback_still_applies() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async { task.handle_signal_changed(5, -60).await });

        let cell = task.cell_selector.get_cell(5).expect("detected cell");
        assert!(cell.sib1.has_sib1);
        assert_eq!(cell.sib1.nci, 5, "the fabricated NCI is the cell id");
    }

    #[test]
    fn an_undecodable_broadcast_is_dropped_without_marking_the_cell() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_signal_changed(4, -60).await;
            task.handle_downlink_rrc(
                4,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&[0xFF, 0xFF, 0xFF]),
            )
            .await;
        });

        assert!(
            !task.cells_with_broadcast_si.contains(&4),
            "garbage must not count as a broadcast"
        );
    }

    // ========================================================================
    // Paging (#35): PCCH monitoring in RRC_IDLE (TS 38.331 §5.3.2.3)
    // ========================================================================

    use nextgsim_rrc::procedures::paging::{encode_paging, PagingRecordParams};

    /// This UE's own 5G-S-TMSI, as the NAS plane derives it from the 5G-GUTI.
    const OWN_S_TMSI: [u8; 6] = [0x55, 0x6A, 0xDE, 0xAD, 0xBE, 0xEF];

    /// A 5G-S-TMSI whose paging occasion is the CURRENT radio frame.
    ///
    /// Issue #99 made the UE drop a matching record that arrives outside its own
    /// paging occasion (TS 38.304 §7.1), so an identity-matching test needs an
    /// identity that is actually being paged now -- otherwise it asserts the
    /// occasion check by accident and fails on 127 frames out of 128.
    ///
    /// Chosen by searching identities rather than by controlling the clock: `UE_ID`
    /// determines the frame, so this is deterministic against whatever the clock
    /// reads.
    fn s_tmsi_paged_in_the_current_frame(config: &UeConfig) -> [u8; 6] {
        let cycle = PagingCycleConfig::with_default_spreading(config.paging_default_cycle_frames)
            .expect("the configured paging cycle must be valid");
        let now = frame_clock::current_sfn();
        for candidate in 0u32..=0xFFFF {
            let s_tmsi = [
                0x55,
                0x6A,
                0xDE,
                0xAD,
                (candidate >> 8) as u8,
                (candidate & 0xFF) as u8,
            ];
            let occasion = paging_occasion(ue_id_from_s_tmsi(&s_tmsi), &cycle);
            if occasion.is_paging_frame(now) {
                return s_tmsi;
            }
        }
        panic!("no identity pages in frame {now}, which cannot happen for N = T");
    }

    /// Another subscriber's 5G-S-TMSI: same AMF (identical first two octets),
    /// different 5G-TMSI — so a comparison that only checks the AMF part, or
    /// one that ignores the identity altogether, cannot pass by accident.
    const OTHER_S_TMSI: [u8; 6] = [0x55, 0x6A, 0x00, 0x00, 0x00, 0x01];

    const PAGING_CELL: i32 = 1;

    fn pcch_paging(identities: &[[u8; 6]]) -> OctetString {
        let records: Vec<PagingRecordParams> = identities
            .iter()
            .map(|s_tmsi| PagingRecordParams::five_g_s_tmsi(*s_tmsi))
            .collect();
        OctetString::from_slice(&encode_paging(&records).expect("encode PCCH Paging"))
    }

    /// Pops the next paging indication the RRC layer handed to NAS.
    fn try_take_paging(
        nas_rx: &mut mpsc::Receiver<TaskMessage<NasMessage>>,
    ) -> Option<Vec<[u8; FIVE_G_S_TMSI_LEN]>> {
        while let Ok(msg) = nas_rx.try_recv() {
            if let TaskMessage::Message(NasMessage::Paging { paging_s_tmsi }) = msg {
                return Some(paging_s_tmsi);
            }
        }
        None
    }

    #[test]
    fn a_paging_record_matching_the_ues_own_5g_s_tmsi_reaches_nas() {
        let config = test_config();
        let own = s_tmsi_paged_in_the_current_frame(&config);
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(own));

        run_async(async {
            task.handle_downlink_rrc(PAGING_CELL, RrcChannel::Pcch, pcch_paging(&[own]))
                .await;
        });

        assert_eq!(
            try_take_paging(&mut nas_rx),
            Some(vec![own]),
            "the matched 5G-S-TMSI must be reported to NAS"
        );
    }

    /// PCCH is a broadcast channel, so a UE reads paging for other subscribers
    /// constantly. Answering one would start a service request the network
    /// never asked this UE for.
    #[test]
    fn a_paging_record_for_another_subscriber_is_not_reported() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(OWN_S_TMSI));

        run_async(async {
            task.handle_downlink_rrc(PAGING_CELL, RrcChannel::Pcch, pcch_paging(&[OTHER_S_TMSI]))
                .await;
        });

        assert!(
            try_take_paging(&mut nas_rx).is_none(),
            "a non-matching paging record must not reach NAS"
        );
    }

    /// Only the matching record of a multi-UE paging message is reported: the
    /// AS filters, so NAS never sees another subscriber's identity.
    #[test]
    fn only_the_matching_record_of_a_multi_ue_paging_message_is_reported() {
        let config = test_config();
        let own = s_tmsi_paged_in_the_current_frame(&config);
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config, 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(own));

        run_async(async {
            task.handle_downlink_rrc(
                PAGING_CELL,
                RrcChannel::Pcch,
                pcch_paging(&[OTHER_S_TMSI, own, OTHER_S_TMSI]),
            )
            .await;
        });

        assert_eq!(try_take_paging(&mut nas_rx), Some(vec![own]));
    }

    /// A UE with no 5G-S-TMSI has no occasion to be inside, and must not be
    /// gated: the identity check in `handle_pcch_message` has already returned by
    /// then, so gating here would only be able to mask that.
    #[test]
    fn a_ue_without_a_paging_identity_is_not_gated_by_the_occasion_check() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let task = RrcTask::new(task_base);
        // No `set_paging_identity`.
        for sfn in [0u16, 1, 17, 511, 1023] {
            assert!(
                task.is_own_paging_occasion_at(sfn),
                "an identity-less UE must not be gated at SFN {sfn}"
            );
        }
    }

    /// CRITERION 5 (issue #99), at the unit level: the occasion decision itself.
    ///
    /// The end-to-end version lives in `tests/src/paging_mt_service_request.rs`;
    /// this pins the predicate, including that it is not simply always true.
    #[test]
    fn the_occasion_check_admits_the_ues_own_frame_and_rejects_a_foreign_one() {
        let config = test_config();
        let own = s_tmsi_paged_in_the_current_frame(&config);
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(config.clone(), 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(own));

        assert!(
            task.is_own_paging_occasion(),
            "an identity chosen for the current frame must be in its occasion"
        );

        // An identity whose paging frame is several frames away must not be.
        let cycle = PagingCycleConfig::with_default_spreading(config.paging_default_cycle_frames)
            .expect("valid cycle");
        let now = frame_clock::current_sfn();
        let foreign = (0u32..=0xFFFF)
            .map(|candidate| {
                [
                    0x55,
                    0x6A,
                    0xDE,
                    0xAD,
                    (candidate >> 8) as u8,
                    (candidate & 0xFF) as u8,
                ]
            })
            .find(|s_tmsi| {
                let occasion = paging_occasion(ue_id_from_s_tmsi(s_tmsi), &cycle);
                // Clear the whole acceptance window, not just one frame.
                let window = (occasion.t() / 8).max(4) + 1;
                !(0..=window).any(|back| occasion.is_paging_frame(now.wrapping_sub(back)))
            })
            .expect("some identity pages in another frame");
        task.set_paging_identity(Some(foreign));
        assert!(
            !task.is_own_paging_occasion(),
            "an identity paged in another frame must be outside this occasion"
        );
    }

    /// TS 38.304 §7.1: PCCH is monitored in RRC_IDLE and RRC_INACTIVE only.
    #[test]
    fn a_connected_ue_ignores_its_own_paging_record() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(OWN_S_TMSI));
        task.state_machine
            .transition(crate::rrc::state::RrcStateTransition::SetupComplete)
            .expect("Idle -> Connected");

        run_async(async {
            task.handle_downlink_rrc(PAGING_CELL, RrcChannel::Pcch, pcch_paging(&[OWN_S_TMSI]))
                .await;
        });

        assert!(
            try_take_paging(&mut nas_rx).is_none(),
            "a connected UE is reached on its own SRB, not by paging"
        );
    }

    /// Before a 5G-GUTI is assigned the UE has no paging identity, so no record
    /// can be for it — including one carrying all-zero octets.
    #[test]
    fn a_ue_without_a_paging_identity_reports_nothing() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_downlink_rrc(
                PAGING_CELL,
                RrcChannel::Pcch,
                pcch_paging(&[[0u8; 6], OWN_S_TMSI]),
            )
            .await;
        });

        assert!(
            try_take_paging(&mut nas_rx).is_none(),
            "no identity installed → no match"
        );
    }

    /// A deleted paging identity (deregistration) stops paging responses.
    #[test]
    fn a_cleared_paging_identity_stops_matching() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(OWN_S_TMSI));
        task.set_paging_identity(None);

        run_async(async {
            task.handle_downlink_rrc(PAGING_CELL, RrcChannel::Pcch, pcch_paging(&[OWN_S_TMSI]))
                .await;
        });

        assert!(try_take_paging(&mut nas_rx).is_none());
    }

    #[test]
    fn an_undecodable_pcch_pdu_is_dropped_without_reporting() {
        let (task_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 16);
        let mut task = RrcTask::new(task_base);
        task.set_paging_identity(Some(OWN_S_TMSI));

        run_async(async {
            task.handle_downlink_rrc(
                PAGING_CELL,
                RrcChannel::Pcch,
                OctetString::from_slice(&[0xFF, 0xFF, 0xFF]),
            )
            .await;
        });

        assert!(try_take_paging(&mut nas_rx).is_none());
    }

    // ========================================================================
    // Idle-mode cell reselection from broadcast parameters (issue #50)
    // ========================================================================

    /// Encodes a `SystemInformation` carrying SIB2 (and optionally SIB3
    /// neighbour offsets), the way the gNB broadcasts it.
    fn broadcast_reselection_si(
        q_hyst_db: i32,
        t_reselection_s: u8,
        neighbours: &[(u16, i32)],
    ) -> Vec<u8> {
        use nextgsim_rrc::procedures::system_information::{
            encode_system_information, IntraFreqNeighbour, Sib2Params, Sib3Params,
            SystemInformationParams,
        };
        encode_system_information(&SystemInformationParams {
            sib2: Some(Sib2Params {
                q_hyst_db,
                t_reselection_s,
                cell_reselection_priority: 6,
                q_rx_lev_min: -70,
                s_intra_search_p: 31,
                thresh_serving_low_p: 4,
            }),
            sib3: if neighbours.is_empty() {
                None
            } else {
                Some(Sib3Params {
                    intra_freq_neighbours: neighbours
                        .iter()
                        .map(|&(phys_cell_id, q_offset_db)| IntraFreqNeighbour {
                            phys_cell_id,
                            q_offset_db,
                        })
                        .collect(),
                })
            },
            sib4: None,
        })
        .expect("the reselection SI must encode")
    }

    /// #50, criterion 2 and criterion 3: a camped idle UE takes `Q_hyst` and
    /// `Treselection` from BROADCAST SIB2 rather than from the compile-time
    /// constants, and this runs on the production `RrcTask` path — not a
    /// test-only construction of the reselection logic.
    #[test]
    fn a_camped_ue_takes_q_hyst_and_t_reselection_from_broadcast_sib2() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        // Values that are NOT the constants, so the assertion cannot pass on the
        // defaults: DEFAULT_Q_HYST_DB is 4 and the default t_reselection is 1000 ms.
        let si = broadcast_reselection_si(10, 3, &[]);

        run_async(async {
            task.handle_signal_changed(1, -60).await;
            task.perform_cycle().await;
            assert_eq!(
                task.serving_cell_id,
                Some(1),
                "must camp before broadcasting"
            );
            task.handle_downlink_rrc(1, RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                .await;
        });

        let params = task.cell_selector.reselection_params();
        assert!(
            params.from_broadcast,
            "the parameters must be marked as coming from the network"
        );
        assert_eq!(
            params.q_hyst, 10,
            "q-Hyst must come from SIB2, not DEFAULT_Q_HYST_DB"
        );
        assert_eq!(
            params.t_reselection, 3000,
            "t-ReselectionNR is in SECONDS on the wire and milliseconds here"
        );
        assert_eq!(params.serving_priority, Some(6));
    }

    /// The R-criterion itself: a neighbour better ranked than the serving cell by
    /// more than the BROADCAST `Q_hyst`, held for longer than the BROADCAST
    /// `Treselection`, causes a reselection — and one held for less does not.
    ///
    /// `t_reselection_s: 0` makes the time-to-trigger elapse immediately, so the
    /// test asserts the criterion rather than sleeping. A separate assertion
    /// covers the not-yet-elapsed case with a non-zero timer.
    #[test]
    fn a_camped_ue_reselects_a_neighbour_that_beats_r_s_for_t_reselection() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        // q-Hyst 6 dB, Treselection 0 s.
        let si = broadcast_reselection_si(6, 0, &[]);

        run_async(async {
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            assert_eq!(task.serving_cell_id, Some(1));
            task.handle_downlink_rrc(1, RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                .await;

            // R_s = -80 + 6 = -74. A neighbour at -76 does NOT beat it.
            task.handle_signal_changed(2, -76).await;
            task.force_cell_selection_now();
            task.perform_cycle().await;
            assert_eq!(
                task.serving_cell_id,
                Some(1),
                "a neighbour inside the broadcast hysteresis must not win: \
                 R_n = -76 is not > R_s = -80 + 6"
            );

            // A neighbour at -70 does: R_n = -70 > R_s = -74.
            task.handle_signal_changed(2, -70).await;
            task.force_cell_selection_now();
            task.perform_cycle().await;
            assert_eq!(
                task.serving_cell_id,
                Some(2),
                "R_n = -70 beats R_s = -74 and Treselection is 0 s, so the UE \
                 must reselect"
            );
        });
    }

    /// `Treselection` is honoured: the same margin that reselected above does
    /// nothing while the candidate has not been better for long enough.
    #[test]
    fn a_better_neighbour_does_not_win_before_t_reselection_elapses() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        // Same 6 dB hysteresis, but a 7 s timer that cannot elapse during the test.
        let si = broadcast_reselection_si(6, 7, &[]);

        run_async(async {
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            task.handle_downlink_rrc(1, RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                .await;

            task.handle_signal_changed(2, -70).await;
            // Twice, because the first evaluation only ARMS the candidate.
            task.force_cell_selection_now();
            task.perform_cycle().await;
            task.force_cell_selection_now();
            task.perform_cycle().await;
            assert_eq!(
                task.serving_cell_id,
                Some(1),
                "the neighbour is better ranked but has not held it for the \
                 broadcast Treselection of 7 s"
            );
        });
    }

    /// The `Qoffset` term of `R_n`, which did not exist before: the SAME signal
    /// levels that reselect with no offset must NOT reselect once SIB3 gives the
    /// neighbour a positive `q-OffsetCell`.
    ///
    /// This is the assertion the old `best_dbm > current_dbm + q_hyst`
    /// comparison could not make, because it had no per-cell term at all.
    #[test]
    fn a_broadcast_q_offset_cell_makes_a_neighbour_less_attractive() {
        // The neighbour's PCI is derived from the NCI its SIB1 broadcast, the
        // same way the gNB derives it (the #37 convention). Its PLMN must be the
        // UE's own configured one (the default config's 0-00), or the cell is
        // out-of-PLMN and cell selection never considers it suitable -- which
        // would make this test pass for the wrong reason.
        let hplmn = test_config().hplmn;
        let (mib2, sib1_cell2) = broadcast_si(0x2000, 42, hplmn.mcc, hplmn.mnc, hplmn.long_mnc);
        let neighbour_pci = phys_cell_id_from_nci(0x2000);

        // Control: no SIB3 offset, so R_n = -70 > R_s = -80 + 6 and it reselects.
        let (task_base, _a, _n, _r, _l) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            let si = broadcast_reselection_si(6, 0, &[]);
            task.handle_downlink_rrc(1, RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                .await;
            task.handle_signal_changed(2, -70).await;
            task.handle_downlink_rrc(2, RrcChannel::BcchBch, OctetString::from_slice(&mib2))
                .await;
            task.handle_downlink_rrc(
                2,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&sib1_cell2),
            )
            .await;
            task.force_cell_selection_now();
            task.perform_cycle().await;
        });
        assert_eq!(
            task.serving_cell_id,
            Some(2),
            "control: with no q-OffsetCell the neighbour wins"
        );

        // Now the same levels, with the neighbour given a +8 dB q-OffsetCell:
        // R_n = -70 - 8 = -78, which no longer beats R_s = -74.
        let (task_base, _a, _n, _r, _l) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        run_async(async {
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            let si = broadcast_reselection_si(6, 0, &[(neighbour_pci, 8)]);
            task.handle_downlink_rrc(1, RrcChannel::BcchDlSch, OctetString::from_slice(&si))
                .await;
            task.handle_signal_changed(2, -70).await;
            task.handle_downlink_rrc(2, RrcChannel::BcchBch, OctetString::from_slice(&mib2))
                .await;
            task.handle_downlink_rrc(
                2,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&sib1_cell2),
            )
            .await;
            task.force_cell_selection_now();
            task.perform_cycle().await;
        });
        assert_eq!(
            task.cell_selector.q_offset_of(2),
            8,
            "the broadcast offset must be keyed to this cell by its derived PCI"
        );
        assert_eq!(
            task.serving_cell_id,
            Some(1),
            "R_n = -70 - 8 = -78 does not beat R_s = -80 + 6 = -74, so the \
             broadcast q-OffsetCell must keep the UE on cell 1"
        );
    }

    /// A re-broadcast SIB3 REPLACES the stored offsets rather than merging into
    /// them, so an offset the network withdrew stops being applied.
    ///
    /// Found by a revert round: making `apply_sib3` merge instead of replace left
    /// every test green, because none of them re-broadcast a SIB3 with an entry
    /// removed. A merge would keep applying a `Qoffset` the cell no longer
    /// advertises, which is a stale-state defect no round trip can see.
    #[test]
    fn a_withdrawn_q_offset_cell_stops_being_applied() {
        let hplmn = test_config().hplmn;
        let (mib2, sib1_cell2) = broadcast_si(0x2000, 42, hplmn.mcc, hplmn.mnc, hplmn.long_mnc);
        let neighbour_pci = phys_cell_id_from_nci(0x2000);

        let (task_base, _a, _n, _r, _l) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_signal_changed(1, -80).await;
            task.perform_cycle().await;
            // The neighbour must be DISCOVERED before its broadcast can be
            // recorded against it: `q_offset_of` keys on the PCI derived from the
            // cell's SIB1, and a cell the UE has never heard has no entry to
            // derive from.
            task.handle_signal_changed(2, -70).await;
            task.handle_downlink_rrc(2, RrcChannel::BcchBch, OctetString::from_slice(&mib2))
                .await;
            task.handle_downlink_rrc(
                2,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&sib1_cell2),
            )
            .await;

            // First broadcast names the neighbour with a +8 dB offset.
            let with_offset = broadcast_reselection_si(6, 0, &[(neighbour_pci, 8)]);
            task.handle_downlink_rrc(
                1,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&with_offset),
            )
            .await;
            assert_eq!(
                task.cell_selector.q_offset_of(2),
                8,
                "positive control: the offset must be applied before its \
                 withdrawal can be asserted"
            );

            // A later broadcast names a DIFFERENT neighbour, withdrawing the
            // first one's offset.
            let other_pci = neighbour_pci.wrapping_add(1) % 1008;
            let withdrawn = broadcast_reselection_si(6, 0, &[(other_pci, 12)]);
            task.handle_downlink_rrc(
                1,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&withdrawn),
            )
            .await;
        });

        assert_eq!(
            task.cell_selector.q_offset_of(2),
            0,
            "the withdrawn q-OffsetCell must no longer be applied: SIB3 carries \
             the cell's COMPLETE neighbour list, so merging would keep an offset \
             the network stopped advertising"
        );
    }

    /// A neighbour's `SystemInformation` must not set this UE's reselection
    /// parameters: a SIB2 describes the reselection behaviour of the cell that
    /// broadcast it. Otherwise whichever cell broadcast most recently would own
    /// the camped UE's hysteresis.
    #[test]
    fn a_non_serving_cells_system_information_is_ignored() {
        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        run_async(async {
            task.handle_signal_changed(1, -60).await;
            task.perform_cycle().await;
            assert_eq!(task.serving_cell_id, Some(1));

            // The serving cell's own SI is applied...
            let serving_si = broadcast_reselection_si(10, 3, &[]);
            task.handle_downlink_rrc(
                1,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&serving_si),
            )
            .await;
            assert_eq!(task.cell_selector.reselection_params().q_hyst, 10);

            // ...and a neighbour's is not.
            let neighbour_si = broadcast_reselection_si(24, 7, &[]);
            task.handle_downlink_rrc(
                2,
                RrcChannel::BcchDlSch,
                OctetString::from_slice(&neighbour_si),
            )
            .await;
            assert_eq!(
                task.cell_selector.reselection_params().q_hyst,
                10,
                "a neighbour's SIB2 must not overwrite the serving cell's"
            );
        });
    }

    /// A dedicated `cellReselectionPriorities` in RRCRelease is applied, and an
    /// EMPTY one deletes the dedicated state rather than being treated as absent
    /// (TS 38.304 §5.2.4.1).
    #[test]
    fn a_dedicated_reselection_priority_list_is_applied_and_an_empty_one_clears_it() {
        use nextgsim_rrc::procedures::rrc_release::FreqPriorityNrParams;

        let (task_base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);

        // Absent: nothing stored, nothing dedicated.
        task.apply_dedicated_reselection_priorities(None);
        let (priorities, dedicated) = task.cell_selector.carrier_priorities();
        assert!(priorities.is_empty() && !dedicated);

        // Present with entries: stored and marked dedicated.
        task.apply_dedicated_reselection_priorities(Some(CellReselectionPrioritiesParams {
            freq_priority_list_nr: vec![FreqPriorityNrParams {
                carrier_freq: 632628,
                priority: 5,
            }],
            t320: None,
        }));
        let (priorities, dedicated) = task.cell_selector.carrier_priorities();
        assert_eq!(priorities.get(&632628), Some(&5));
        assert!(
            dedicated,
            "an assigned list is dedicated, overriding broadcast"
        );

        // Present and EMPTY: deletes the dedicated state. Distinct from absent,
        // which is why the first case above is asserted separately.
        task.apply_dedicated_reselection_priorities(Some(CellReselectionPrioritiesParams {
            freq_priority_list_nr: Vec::new(),
            t320: None,
        }));
        let (priorities, dedicated) = task.cell_selector.carrier_priorities();
        assert!(
            priorities.is_empty() && !dedicated,
            "an empty dedicated list must delete the stored priorities and fall \
             back to broadcast, not store an empty override"
        );
    }

    // ========================================================================
    // MeasurementReport is real UPER now (issue #107, criterion 4)
    // ========================================================================

    fn measurement_report_fixture() -> crate::rrc::measurement::MeasurementReport {
        use crate::rrc::measurement::{CellMeasResult, MeasurementReport};
        MeasurementReport {
            meas_id: 3,
            serving_cell: CellMeasResult {
                pci: 1,
                nci: None,
                rsrp: Some(-90),
                rsrq: None,
                sinr: None,
            },
            neighbor_cells: vec![CellMeasResult {
                pci: 2,
                nci: None,
                rsrp: Some(-80),
                rsrq: None,
                sinr: None,
            }],
            eutra_neighbor_cells: Vec::new(),
            triggered_cells: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    /// The UE emits a real UPER `MeasurementReport`, and the **gNB's own parser**
    /// reads back the levels it put in.
    ///
    /// This is the criterion that matters: the two ends had to flip together, and a
    /// test that only round-tripped through this crate would pass with the gNB still
    /// expecting the hand-rolled bytes. `nextgsim-gnb` is not a dependency of
    /// `nextgsim-ue`, so the shared surface asserted here is the codec both call —
    /// `decode_measurement_report` is exactly what the gNB's
    /// `parse_measurement_report` calls, and its own test asserts the dBm result.
    #[test]
    fn the_ue_emits_a_uper_measurement_report_the_shared_codec_decodes() {
        use nextgsim_rrc::procedures::measurement_report::{
            decode_measurement_report, rsrp_range_to_dbm,
        };

        let report = measurement_report_fixture();
        let bytes = build_uper_measurement_report(&report).expect("the report must encode");

        // Not the legacy format: that started with 0x0B and was 9 bytes here.
        assert_ne!(
            bytes.first(),
            Some(&0x0Bu8),
            "the hand-rolled leading byte must be gone"
        );

        let decoded = decode_measurement_report(&bytes).expect("the gNB's codec must decode it");
        assert_eq!(decoded.meas_id.0, 3);

        let serving = decoded
            .serv_freq_results
            .first()
            .expect("a serving result must be present");
        assert_eq!(serving.serv_cell_index, 0);
        assert_eq!(serving.meas_result_serving_cell.phys_cell_id, Some(1));
        let serving_rsrp = serving
            .meas_result_serving_cell
            .cell_results
            .ssb_results
            .as_ref()
            .and_then(|r| r.rsrp)
            .expect("the serving RSRP must be carried");
        assert_eq!(
            rsrp_range_to_dbm(serving_rsrp).round() as i32,
            -90,
            "the level must survive dBm -> RSRP-Range -> dBm"
        );

        let neigh = decoded
            .neigh_freq_results
            .first()
            .expect("the neighbour frequency entry must be present");
        assert_eq!(neigh.meas_result_list.len(), 1);
        assert_eq!(neigh.meas_result_list[0].phys_cell_id, Some(2));
        assert_eq!(
            rsrp_range_to_dbm(
                neigh.meas_result_list[0]
                    .cell_results
                    .ssb_results
                    .as_ref()
                    .and_then(|r| r.rsrp)
                    .expect("neighbour RSRP")
            )
            .round() as i32,
            -80
        );
        assert!(
            decoded.eutra_neigh_results.is_empty(),
            "an intra-NR report must not also carry E-UTRA results: \
             measResultNeighCells is a CHOICE"
        );
    }

    /// The **wiring**: `send_measurement_report` puts the UPER bytes on UL-DCCH.
    ///
    /// The test above exercises `build_uper_measurement_report` directly, so it
    /// passes even if the sender never calls it — a revert round proved exactly
    /// that by replacing the sender's call with legacy bytes and staying green.
    /// This is the recorded pattern where the helper is tested and the wiring is
    /// not, and the better the helper's test the more convincing the illusion.
    #[test]
    fn send_measurement_report_puts_the_uper_bytes_on_ul_dcch() {
        use nextgsim_rrc::procedures::measurement_report::decode_measurement_report;

        let (task_base, _app_rx, _nas_rx, _rrc_rx, mut rls_rx) = UeTaskBase::new(test_config(), 32);
        let mut task = RrcTask::new(task_base);
        let report = measurement_report_fixture();

        run_async(async {
            task.send_measurement_report(&report).await;
        });

        let (channel, pdu) = next_uplink_rrc(&mut rls_rx);
        assert_eq!(
            channel,
            RrcChannel::UlDcch,
            "a MeasurementReport rides UL-DCCH"
        );
        let decoded = decode_measurement_report(pdu.data())
            .expect("what the sender actually emitted must be a UPER MeasurementReport");
        assert_eq!(decoded.meas_id.0, report.meas_id);
        assert_ne!(
            pdu.data().first(),
            Some(&0x0Bu8),
            "and not the legacy hand-rolled format"
        );
    }

    /// A cell the UE detected but could not measure reports **no** ssb-Results.
    ///
    /// `RSRP-Range` 0 is a real measurement meaning "below -156 dBm", so
    /// substituting it for "unknown" would report an implausibly weak level as a
    /// measurement. The old byte format substituted **-120 dBm**, a plausible level
    /// the UE had never measured, which is worse still.
    #[test]
    fn an_unmeasured_cell_reports_no_results_rather_than_range_zero() {
        use nextgsim_rrc::procedures::measurement_report::decode_measurement_report;

        let mut report = measurement_report_fixture();
        report.serving_cell.rsrp = None;
        report.neighbor_cells[0].rsrp = None;

        let bytes = build_uper_measurement_report(&report).expect("encodes");
        let decoded = decode_measurement_report(&bytes).expect("decodes");

        assert!(
            decoded.serv_freq_results[0]
                .meas_result_serving_cell
                .cell_results
                .ssb_results
                .is_none(),
            "an unmeasured serving cell must carry no ssb-Results, not range 0"
        );
        assert!(
            decoded.neigh_freq_results[0].meas_result_list[0]
                .cell_results
                .ssb_results
                .is_none(),
            "and neither must an unmeasured neighbour"
        );
    }

    /// An inter-RAT (B1/B2) report **cannot be encoded by this codec version**, and
    /// that is issue #117's ceiling reached from a new direction.
    ///
    /// `measResultNeighCells` is a CHOICE whose `measResultListEUTRA` alternative
    /// sits past the extension marker, and `asn1-codecs` 0.7.2's
    /// `encode_choice_idx_common` returns `EncodeNotSupported` for exactly that.
    /// #113 pinned the refusal at the codec (`the_eutra_choice_arm_cannot_be_uper_encoded_by_this_codec`);
    /// this pins its **consequence at the UE**: a B1/B2 report is not sent at all.
    ///
    /// Asserted rather than left to be discovered, because the alternative is a
    /// measurement that vanishes with a log line. `send_measurement_report` names
    /// #117 in that log so an operator is told why. When #117 is resolved this test
    /// must be REPLACED by the success assertion, not deleted — the inter-RAT arm is
    /// the thing it is about.
    #[test]
    fn an_inter_rat_report_cannot_be_encoded_by_this_codec_version() {
        use crate::rrc::measurement::{EutraCellKey, EutraMeasResult};

        let mut report = measurement_report_fixture();
        report.eutra_neighbor_cells = vec![EutraMeasResult {
            cell: EutraCellKey {
                earfcn: 1850,
                pci: 42,
            },
            rsrp: Some(-100),
            cell_individual_offset: 0,
        }];

        let err = build_uper_measurement_report(&report)
            .expect_err("issue #117: the extended CHOICE arm cannot be encoded");
        let text = err.to_string();
        assert!(
            text.contains("extended choice") || text.contains("EncodeNotSupported"),
            "the failure must be the codec's extended-CHOICE refusal (issue #117), \
             not some other encode error that would hide it: {text}"
        );

        // The positive control: the SAME report without the E-UTRA results encodes
        // fine, so the refusal is about the extension arm and not about the report.
        report.eutra_neighbor_cells.clear();
        assert!(
            build_uper_measurement_report(&report).is_ok(),
            "an intra-NR report must still encode; otherwise the assertion above \
             says nothing about the E-UTRA arm"
        );
    }
}
