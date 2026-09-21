//! RRC Reconfiguration Procedure
//!
//! Implements the RRC Reconfiguration procedure as defined in 3GPP TS 38.331 Section 5.3.5.
//! This procedure is used to modify an RRC connection, including radio bearer configuration,
//! measurement configuration, and cell group configuration.
//!
//! The procedure consists of two messages:
//! 1. `RRCReconfiguration` - gNB → UE: Network request to modify RRC connection
//! 2. `RRCReconfigurationComplete` - UE → gNB: Confirmation of reconfiguration

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use crate::procedures::meas_config::{build_a3_meas_config, A3MeasConfigParams, MeasConfigError};
// The connected-mode NTN update path (issue #56): `ntn-Config` reaches a
// connected UE in a `SystemInformation` carried by
// `dedicatedSystemInformationDelivery`. See [`RrcReconfigurationParams::ntn_config`].
// The dedicated sidelink configuration (issue #141): `sl-ConfigDedicatedNR-r16` lives
// in the v1610 extension. See [`RrcReconfigurationParams::sl_config`].
use crate::procedures::sidelink_ue_information::{
    build_sl_config_dedicated, read_sl_config_dedicated, SidelinkRrcError, SlConfigDedicatedParams,
};
use crate::procedures::system_information::{
    build_system_information, parse_system_information, Sib19Data, Sib19Params,
    SystemInformationError, SystemInformationParams,
};
use thiserror::Error;

/// Errors that can occur during RRC Reconfiguration procedures
#[derive(Debug, Error)]
pub enum RrcReconfigurationError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// A `dedicatedSystemInformationDelivery` payload that could not be built or
    /// read (issue #56).
    ///
    /// Its own variant for the same reason as [`Self::MeasConfig`]: a caller that
    /// cannot signal an NTN update must still be able to send the bearer half of
    /// the reconfiguration, and telling the two failures apart is what makes that
    /// decision possible.
    #[error("Invalid dedicatedSystemInformationDelivery: {0}")]
    DedicatedSystemInformation(#[from] SystemInformationError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),

    /// A `measConfig` whose field values TS 38.331 does not allow (issue #170).
    ///
    /// Its own variant rather than folded into `InvalidFieldValue`: a caller that
    /// wants to decline arming measurements but still send the bearer half of the
    /// reconfiguration has to be able to tell the two apart.
    #[error("Invalid measConfig: {0}")]
    MeasConfig(#[from] MeasConfigError),

    /// An `sl-ConfigDedicatedNR` that could not be built (issue #141).
    ///
    /// Its own variant for the same reason `MeasConfig` is: a gNB that mis-configured a
    /// sidelink `t400` must still be able to send the bearer half of the
    /// reconfiguration, and the caller can only decide that if it can tell which half
    /// failed.
    #[error("Invalid sl-ConfigDedicatedNR: {0}")]
    SlConfig(#[from] SidelinkRrcError),
}

// ============================================================================
// RRC Reconfiguration
// ============================================================================

/// Parameters for building an RRC Reconfiguration message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (encoded as bytes, optional)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (encoded as bytes, optional)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (encoded as bytes, optional)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
    /// `masterKeyUpdate`: the IE that tells the UE to re-derive `KgNB*`
    /// (TS 38.331 §5.3.5.7, TS 33.501 §6.9.2.3.1; issue #39).
    ///
    /// `None` means the UE keeps its current keys, which is right for a reconfiguration
    /// that is not a handover. Present on a handover command, and its
    /// `keySetChangeIndicator` is what distinguishes a **vertical** re-key (from a fresh
    /// NH the AMF supplied) from a **horizontal** one (from the current `KgNB`). Getting
    /// that bit wrong makes the two ends derive different keys and every PDCP MAC on the
    /// target fail, with nothing to say a key was the problem.
    pub master_key_update: Option<MasterKeyUpdateParams>,
    /// `measConfig`: the measurement configuration the UE is to apply
    /// (TS 38.331 §5.5.2, §6.3.2; issue #170).
    ///
    /// `None` leaves the UE's measurement configuration alone, which is right for a
    /// reconfiguration that changes only bearers. `Some` carries the gNB's A3
    /// margin, and is what makes the reporting trigger the UE evaluates the same
    /// one the gNB decides handovers on — before this the field was hardcoded
    /// `None`, so the two were independent with no wire path between them.
    pub meas_config: Option<A3MeasConfigParams>,
    /// The serving cell's `ntn-Config`, for the connected-mode NTN update path
    /// (TS 38.300 §16.14.2.2, issue #56).
    ///
    /// §16.14.2.2: "In connected mode, the UE shall be able to continuously update
    /// the Timing Advance and frequency pre-compensation." A connected UE is not
    /// obliged to keep reading BCCH, so the ephemeris has to be able to reach it on
    /// SRB1 as well as on the broadcast.
    ///
    /// # Why this rides `dedicatedSystemInformationDelivery` and not a dedicated IE
    ///
    /// The obvious-looking home, `ServingCellConfigCommon.ntn-Config-r17`, is inside
    /// a `[[ ]]` extension-addition group of that SEQUENCE. The vendored
    /// `asn1-compiler` does not generate SEQUENCE extension additions, so that field
    /// does not exist in the generated tree — and the vendored codec's
    /// `encode_sequence_header_common` still returns `EncodeNotSupported` for an
    /// extended SEQUENCE, so it could not be encoded even if it did.
    ///
    /// `RRCReconfiguration-v1530-IEs.dedicatedSystemInformationDelivery` is in the
    /// ROOT of its SEQUENCE and is an `OCTET STRING (CONTAINING SystemInformation)`,
    /// and TS 38.331's own field description says it "is used to transfer SIB6,
    /// SIB7, SIB8, **SIB19**, SIB20, SIB21, SIB25, SIB26, SIB27 to the UE". So this
    /// is not a workaround for the codec gap — it is a carrier the spec designates
    /// for exactly this IE, which happens also to be reachable. The UE applies it by
    /// "perform[ing] the action upon reception of System Information as specified in
    /// 5.2.2.4" (§5.3.5.3), i.e. the same handler the broadcast feeds, so one
    /// application path serves both.
    pub ntn_config: Option<Sib19Params>,
    /// `sl-ConfigDedicatedNR-r16`: the dedicated sidelink configuration the network
    /// grants (TS 38.331 §5.3.5.3, §6.3.2; issue #141).
    ///
    /// `None` leaves the UE's sidelink configuration alone. `Some` is a `setup`: the UE
    /// applies it and may transmit sidelink on the configured carriers. Before issue #141
    /// this field did not exist and the IE was never sent, so a UE running PC5 did so
    /// with no network grant at all — the gNB was unaware sidelink was in use.
    ///
    /// Carried in the `v1610` extension, which is why a reconfiguration that carries only
    /// this still has to build the whole `v1530 -> v1540 -> v1560 -> v1610` chain: the
    /// intermediate extensions exist purely to reach it.
    pub sl_config: Option<SlConfigDedicatedParams>,
}

/// `masterKeyUpdate` as the network sets it and the UE reads it
/// (TS 38.331 §6.3.2 `MasterKeyUpdate`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasterKeyUpdateParams {
    /// `keySetChangeIndicator`: `true` for a **vertical** derivation from a fresh NH,
    /// `false` for a **horizontal** one from the current `KgNB`.
    pub key_set_change_indicator: bool,
    /// `nextHopChainingCount` (0..=7) the derivation chains on.
    pub next_hop_chaining_count: u8,
}

/// Parsed RRC Reconfiguration data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Radio Bearer Configuration (raw bytes)
    pub radio_bearer_config: Option<Vec<u8>>,
    /// Secondary Cell Group Configuration (raw bytes)
    pub secondary_cell_group: Option<Vec<u8>>,
    /// Master Cell Group Configuration (raw bytes)
    pub master_cell_group: Option<Vec<u8>>,
    /// Full configuration indicator
    pub full_config: bool,
    /// `masterKeyUpdate`, when the message carried one (issue #39).
    pub master_key_update: Option<MasterKeyUpdateParams>,
    /// The **decoded** `measConfig`, when the message carried one (issue #170).
    ///
    /// The whole IE rather than the A3 parameters alone: a `measConfig` may carry
    /// removals, a `quantityConfig` or a periodical report, and a receiver that
    /// wanted any of those would have to re-decode the message to see them.
    /// [`read_a3_meas_configs`] extracts the A3 bindings from it.
    pub meas_config: Option<MeasConfig>,
    /// The serving cell's NTN configuration, when the message carried a
    /// `dedicatedSystemInformationDelivery` containing a SIB19 (issue #56).
    ///
    /// Already parsed rather than left as bytes, for the same reason `meas_config`
    /// is: the receiver CONSUMES it — it derives a timing advance from it — so
    /// handing back an octet string would only add a place to lose it.
    pub ntn_config: Option<Sib19Data>,
    /// The decoded `sl-ConfigDedicatedNR-r16`, when the message carried a `setup`
    /// (issue #141).
    ///
    /// `None` covers both "the IE was absent" and "the IE was a `release`", because the
    /// UE's action is the same in each: it holds no granted sidelink configuration.
    /// [`RrcReconfigurationData::sl_config_released`] distinguishes them for a caller
    /// that needs to, and the UE's own state is what records whether it once had one.
    pub sl_config: Option<SlConfigDedicatedParams>,
    /// Whether the message carried an `sl-ConfigDedicatedNR-r16` **`release`**.
    ///
    /// Separate from `sl_config: None` because the two mean different things on the wire:
    /// absent leaves the UE's configuration alone (TS 38.331 §5.3.5.3 acts only on IEs
    /// present), while `release` revokes it. A UE that treated a release as an absence
    /// would keep transmitting sidelink the network had just withdrawn.
    pub sl_config_released: bool,
}

/// Build an RRC Reconfiguration message
pub fn build_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<DL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Decode radio bearer config if provided
    let radio_bearer_config = if let Some(ref bytes) = params.radio_bearer_config {
        Some(decode_rrc::<RadioBearerConfig>(bytes)?)
    } else {
        None
    };

    // Build the IEs
    let rrc_reconfiguration_ies = RRCReconfiguration_IEs {
        radio_bearer_config,
        secondary_cell_group: params
            .secondary_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_IEsSecondaryCellGroup(b.clone())),
        // A real generated-codec `MeasConfig` (issue #170). This was hardcoded
        // `None` with the comment "Simplified - not including MeasConfig for now",
        // which made the gNB's A3 margin unreachable by the UE: it ran its
        // reporting trigger off a hard-coded local default while the gNB decided
        // handovers on the configured one.
        meas_config: params
            .meas_config
            .as_ref()
            .map(build_a3_meas_config)
            .transpose()?,
        late_non_critical_extension: None,
        // `ntn_config` joins the condition since issue #56: it lives in the v1530
        // extension, so a reconfiguration that carries ONLY an NTN update -- which
        // is exactly the connected-mode ephemeris refresh of TS 38.300 §16.14.2.2 --
        // would otherwise have the extension omitted and the update silently
        // dropped.
        // `sl_config` joins the condition since issue #141, for exactly the reason
        // `ntn_config` did: it lives further down the extension chain (v1610), so a
        // reconfiguration that carries ONLY a sidelink grant would otherwise have the
        // whole chain omitted and the grant silently dropped.
        non_critical_extension: if params.master_cell_group.is_some()
            || params.full_config
            || params.ntn_config.is_some()
            || params.sl_config.is_some()
        {
            Some(build_v1530_extension(params)?)
        } else {
            None
        },
    };

    let rrc_reconfiguration = RRCReconfiguration {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCReconfigurationCriticalExtensions::RrcReconfiguration(
            rrc_reconfiguration_ies,
        ),
    };

    let message_type = DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(
        rrc_reconfiguration,
    ));

    Ok(DL_DCCH_Message {
        message: message_type,
    })
}

/// Wraps a SIB19 as the `SystemInformation` that
/// `dedicatedSystemInformationDelivery` contains (TS 38.331 §6.2.2, issue #56).
///
/// `OCTET STRING (CONTAINING SystemInformation)` means the octets are a complete,
/// separately-encoded `SystemInformation` UPER PDU — so this reuses
/// `build_system_information`, the same builder the broadcast path uses, rather
/// than assembling a second one. That is what keeps the dedicated delivery and the
/// broadcast byte-identical in content: the UE's SIB19 handler cannot tell them
/// apart, which is precisely §5.3.5.3's "perform the action upon reception of
/// System Information".
fn build_ntn_dedicated_si(ntn_config: &Sib19Params) -> Result<Vec<u8>, SystemInformationError> {
    let si = build_system_information(&SystemInformationParams {
        sib19: Some(*ntn_config),
        ..Default::default()
    })?;
    // The contained type is `SystemInformation`, not `BCCH-DL-SCH-Message`: the
    // channel wrapper belongs to the broadcast, and a receiver decoding this octet
    // string expects the inner message.
    let BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(inner)) =
        &si.message
    else {
        return Err(SystemInformationError::InvalidMessageType {
            expected: "SystemInformation".to_string(),
            actual: "SystemInformationBlockType1".to_string(),
        });
    };
    Ok(encode_rrc(inner)?)
}

/// Reads a `dedicatedSystemInformationDelivery` back, returning the SIB19 it
/// carried if any (issue #56).
///
/// A payload that does not decode, or that carries no SIB19, yields `None` rather
/// than an error: the field legally carries SIB6/7/8/20/21/25/26/27 too, and a
/// reconfiguration that delivered a SIB20 must not fail to apply its bearer half
/// because this NTN reader found nothing for itself.
fn read_ntn_dedicated_si(bytes: &[u8]) -> Option<Sib19Data> {
    let inner: SystemInformation = decode_rrc(bytes).ok()?;
    let msg = BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(inner)),
    };
    parse_system_information(&msg).ok()?.sib19
}

/// Build the v1530 extension for RRC Reconfiguration
fn build_v1530_extension(
    params: &RrcReconfigurationParams,
) -> Result<RRCReconfiguration_v1530_IEs, RrcReconfigurationError> {
    Ok(RRCReconfiguration_v1530_IEs {
        master_cell_group: params
            .master_cell_group
            .as_ref()
            .map(|b| RRCReconfiguration_v1530_IEsMasterCellGroup(b.clone())),
        full_config: if params.full_config {
            Some(RRCReconfiguration_v1530_IEsFullConfig(
                RRCReconfiguration_v1530_IEsFullConfig::TRUE,
            ))
        } else {
            None
        },
        dedicated_nas_message_list: None,
        master_key_update: params.master_key_update.map(|u| MasterKeyUpdate {
            key_set_change_indicator: MasterKeyUpdateKeySetChangeIndicator(
                u.key_set_change_indicator,
            ),
            next_hop_chaining_count: NextHopChainingCount(u.next_hop_chaining_count),
            // The `nas-Container` carries an intra-5GC N2 handover's NAS security
            // parameters. This gNB performs no NAS-level key change on handover, so
            // sending one would describe a transform neither end applies.
            nas_container: None,
        }),
        dedicated_sib1_delivery: None,
        // The connected-mode NTN update (issue #56). TS 38.331's field description
        // names SIB19 among this field's permitted payloads.
        dedicated_system_information_delivery: params
            .ntn_config
            .as_ref()
            .map(build_ntn_dedicated_si)
            .transpose()?
            .map(RRCReconfiguration_v1530_IEsDedicatedSystemInformationDelivery),
        other_config: None,
        // The chain down to v1610, built only when something down there needs carrying
        // (issue #141). v1540 and v1560 hold nothing this gNB sets; they exist here
        // solely as the hops to `sl-ConfigDedicatedNR-r16`.
        non_critical_extension: params
            .sl_config
            .as_ref()
            .map(build_v1610_chain)
            .transpose()?,
    })
}

/// Builds `v1540 -> v1560 -> v1610` to carry an `sl-ConfigDedicatedNR-r16` `setup`
/// (TS 38.331 §6.2.2, issue #141).
///
/// Every field in the two intervening extensions is left absent: this gNB sets no
/// `otherConfig-v1540`, no MR-DC secondary cell group and no `sk-Counter`, so the hops
/// cost two optional bitmaps and nothing else.
///
/// A `setup` rather than an ever-`release`: `release` is reachable through
/// [`RrcReconfigurationParams::sl_config`] being `None`, which omits the IE — and TS
/// 38.331 §5.3.5.3 has the UE act only on IEs that are present, so omission leaves the
/// UE's configuration alone. Revoking a grant needs the explicit `release` arm, which no
/// caller in this tree has cause to send yet and so is not built here; the *parser*
/// reads it, because a conformant peer may send one.
fn build_v1610_chain(
    sl_config: &SlConfigDedicatedParams,
) -> Result<RRCReconfiguration_v1540_IEs, RrcReconfigurationError> {
    let v1610 = RRCReconfiguration_v1610_IEs {
        other_config_v1610: None,
        bap_config_r16: None,
        iab_ip_address_configuration_list_r16: None,
        conditional_reconfiguration_r16: None,
        daps_source_release_r16: None,
        t316_r16: None,
        need_for_gaps_config_nr_r16: None,
        on_demand_sib_request_r16: None,
        dedicated_pos_sys_info_delivery_r16: None,
        sl_config_dedicated_nr_r16: Some(
            RRCReconfiguration_v1610_IEsSl_ConfigDedicatedNR_r16::Setup(build_sl_config_dedicated(
                sl_config,
            )?),
        ),
        // The E-UTRA sidelink configuration. This is an NR-only simulator with no
        // LTE V2X carrier, so a value here would describe a carrier that does not exist.
        sl_config_dedicated_eutra_info_r16: None,
        target_cell_smtc_scg_r16: None,
        non_critical_extension: None,
    };
    let v1560 = RRCReconfiguration_v1560_IEs {
        mrdc_secondary_cell_group_config: None,
        radio_bearer_config2: None,
        sk_counter: None,
        non_critical_extension: Some(v1610),
    };
    Ok(RRCReconfiguration_v1540_IEs {
        other_config_v1540: None,
        non_critical_extension: Some(v1560),
    })
}

/// Reads an `sl-ConfigDedicatedNR-r16` out of a reconfiguration's extension chain
/// (issue #141).
///
/// Returns `(setup_params, was_released)`. A chain that stops short of v1610 — which is
/// every reconfiguration this tree sent before issue #141 — yields `(None, false)`,
/// i.e. "no sidelink IE was present", which leaves the UE's configuration alone.
fn read_sl_config_from_chain(
    ies: &RRCReconfiguration_IEs,
) -> (Option<SlConfigDedicatedParams>, bool) {
    let Some(v1610) = ies
        .non_critical_extension
        .as_ref()
        .and_then(|v1530| v1530.non_critical_extension.as_ref())
        .and_then(|v1540| v1540.non_critical_extension.as_ref())
        .and_then(|v1560| v1560.non_critical_extension.as_ref())
    else {
        return (None, false);
    };
    match v1610.sl_config_dedicated_nr_r16.as_ref() {
        Some(RRCReconfiguration_v1610_IEsSl_ConfigDedicatedNR_r16::Setup(config)) => {
            (Some(read_sl_config_dedicated(config)), false)
        }
        Some(RRCReconfiguration_v1610_IEsSl_ConfigDedicatedNR_r16::Release(_)) => (None, true),
        None => (None, false),
    }
}

// ============================================================================
// amfg-04: structured RadioBearerConfig / DRB / SDAP / CellGroupConfig builders
// ============================================================================
//
// TS 38.331 §5.3.5.6.5/§6.3.2: an RRCReconfiguration that establishes a PDU
// session's user plane carries a RadioBearerConfig with one DRB-ToAddMod per
// PDU session; the DRB's cn-Association is an SDAP-Config keyed to the PDU
// session id with the accepted QoS flows (QFIs) mapped via
// mappedQoS-FlowsToAdd, plus a matching CellGroupConfig (one RLC bearer per
// DRB). Previously these were passed as opaque pre-encoded bytes and only the
// first QoS flow was honoured; these constructors build the real structured
// config from the full accepted-QFI set.

/// Build a `RadioBearerConfig` carrying a single DRB for one PDU session.
///
/// * `pdu_session_id` — the 5GS PDU session id the DRB serves.
/// * `drb_id` — the data radio bearer identity (1..=32).
/// * `qfis` — every accepted QoS Flow Identifier to map in SDAP
///   (`mappedQoS-FlowsToAdd`); when empty the field is omitted (the ASN.1
///   SEQUENCE-OF has a lower bound of 1).
/// * `default_drb` — whether this DRB is the SDAP default DRB for the session.
///
/// The SDAP header for DL and UL is set to PRESENT, which is required when QoS
/// flow remapping / QFI carriage is in use.
///
/// # The SDAP header is ONE octet
///
/// This comment used to say "3-byte SDAP header", and issue #44's criterion 2
/// inherited the wrong number from it. **TS 37.324 §6.2.2 defines one octet**: D/C,
/// then RQI (downlink) or R (uplink), then a **6-bit** QFI. A six-bit QFI is
/// precisely why `QFI ::= INTEGER (1..maxNrofQFIs)` and
/// `QosFlowIdentifier ::= INTEGER (0..63)` are what they are, two fields down in
/// this very function.
///
/// Corrected here rather than only in the issue, because this comment is where the
/// error propagated from — and a receiver built to skip three octets would read two
/// octets of payload as header, which is the exact interop failure the criterion
/// was filed about. The layout lives in `nextgsim_pdcp::sdap`.
pub fn build_drb_radio_bearer_config(
    pdu_session_id: u8,
    drb_id: u8,
    qfis: &[u8],
    default_drb: bool,
    integrity_protection: DrbIntegrityProtection,
) -> RadioBearerConfig {
    let drb = build_drb_to_add_mod(
        pdu_session_id,
        drb_id,
        qfis,
        default_drb,
        integrity_protection,
    );
    RadioBearerConfig {
        srb_to_add_mod_list: None,
        srb3_to_release: None,
        drb_to_add_mod_list: Some(DRB_ToAddModList(vec![drb])),
        drb_to_release_list: None,
        security_config: None,
    }
}

/// One `DRB-ToAddMod` with its `SDAP-Config` and `PDCP-Config`.
///
/// Factored out of [`build_drb_radio_bearer_config`] so the single-DRB and
/// multi-DRB builders produce byte-identical DRB entries (issue #44). That matters
/// for more than tidiness: with two constructors, a DRB built by the multi-DRB path
/// could differ in some absent-OPTIONAL detail from one built by the single path,
/// and the golden byte vectors would only be testing one of them.
fn build_drb_to_add_mod(
    pdu_session_id: u8,
    drb_id: u8,
    qfis: &[u8],
    default_drb: bool,
    integrity_protection: DrbIntegrityProtection,
) -> DRB_ToAddMod {
    let mapped_qo_s_flows_to_add = if qfis.is_empty() {
        None
    } else {
        Some(SDAP_ConfigMappedQoS_FlowsToAdd(
            qfis.iter().map(|q| QFI(*q)).collect(),
        ))
    };

    let sdap_config = SDAP_Config {
        pdu_session: PDU_SessionID(pdu_session_id),
        sdap_header_dl: SDAP_ConfigSdap_HeaderDL(SDAP_ConfigSdap_HeaderDL::PRESENT),
        sdap_header_ul: SDAP_ConfigSdap_HeaderUL(SDAP_ConfigSdap_HeaderUL::PRESENT),
        default_drb: SDAP_ConfigDefaultDRB(default_drb),
        mapped_qo_s_flows_to_add,
        mapped_qo_s_flows_to_release: None,
    };

    // `PDCP-Config.drb` is present only when there is something to say in it.
    // Integrity protection is the one thing this simulator configures there
    // (issue #32), so an unprotected DRB still gets the minimal config it had
    // before -- which is what keeps the golden bytes below unchanged when the
    // user plane is not protected.
    let drb = match integrity_protection {
        DrbIntegrityProtection::Disabled => None,
        DrbIntegrityProtection::Enabled => Some(PDCP_ConfigDrb {
            discard_timer: None,
            pdcp_sn_size_ul: None,
            pdcp_sn_size_dl: None,
            // Mandatory in the SEQUENCE, and `notUsed` is the truth: every ROHC
            // profile is advertised unsupported by this crate's UE capabilities.
            header_compression: PDCP_ConfigDrbHeaderCompression::NotUsed(
                PDCP_ConfigDrbHeaderCompression_notUsed,
            ),
            integrity_protection: Some(PDCP_ConfigDrbIntegrityProtection(
                PDCP_ConfigDrbIntegrityProtection::ENABLED,
            )),
            status_report_required: None,
            out_of_order_delivery: None,
        }),
    };
    let pdcp_config = PDCP_Config {
        drb,
        more_than_one_rlc: None,
        t_reordering: None,
    };

    DRB_ToAddMod {
        cn_association: Some(DRB_ToAddModCnAssociation::Sdap_Config(sdap_config)),
        drb_identity: DRB_Identity(drb_id),
        reestablish_pdcp: None,
        recover_pdcp: None,
        pdcp_config: Some(pdcp_config),
    }
}

/// One DRB of a PDU session, as the network configures it (issue #44).
///
/// A session has more than one when its QoS flows do not all belong on the same
/// bearer — TS 37.324 §5.1 maps each flow onto a DRB, and the mapping is what
/// `mappedQoS-FlowsToAdd` carries per DRB. The **policy** that decides which flow
/// goes where is `nextgsim_gtp::qfi_drb`, deliberately not here: this crate builds
/// the message that *states* the mapping, and inventing one would put a second
/// policy in the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DrbSpec {
    /// `drb-Identity` (1..=32).
    pub drb_id: u8,
    /// `logicalChannelIdentity` of the DRB's RLC bearer.
    pub lcid: u8,
    /// Every QFI mapped to this DRB (`mappedQoS-FlowsToAdd`). When empty the field
    /// is omitted, because the ASN.1 SEQUENCE-OF has a lower bound of 1.
    pub qfis: Vec<u8>,
    /// `defaultDRB`: the bearer an unmapped flow falls to (§5.3.1).
    ///
    /// Exactly one DRB of a session should set it. Not enforced here — the builder
    /// states what it is given — but
    /// [`build_multi_drb_radio_bearer_config`] refuses a list with none or several,
    /// because a session with two default DRBs has no defined behaviour for an
    /// unmapped flow.
    pub default_drb: bool,
    /// Whether this DRB's PDCP entity integrity-protects user data.
    pub integrity_protection: DrbIntegrityProtection,
}

/// Build a `RadioBearerConfig` carrying **every** DRB of one PDU session
/// (issue #44).
///
/// The multi-DRB generalisation of [`build_drb_radio_bearer_config`], which
/// remains as the single-DRB path so the golden byte vectors below keep testing the
/// exact message the pre-SDAP data path sends.
///
/// Refuses a `specs` list that is empty, or that does not name **exactly one**
/// default DRB: an unmapped QoS flow goes to the default DRB (TS 37.324 §5.3.1), so
/// a session with none has nowhere to put one and a session with two has a choice
/// nothing resolves. Reported rather than defaulted, because picking one silently
/// is how the two ends come to disagree about which bearer that is.
pub fn build_multi_drb_radio_bearer_config(
    pdu_session_id: u8,
    specs: &[DrbSpec],
) -> Result<RadioBearerConfig, RrcReconfigurationError> {
    if specs.is_empty() {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "a PDU session needs at least one DRB".to_string(),
        ));
    }
    let defaults = specs.iter().filter(|s| s.default_drb).count();
    if defaults != 1 {
        return Err(RrcReconfigurationError::InvalidFieldValue(format!(
            "a PDU session must have exactly one default DRB, not {defaults}: an \
             unmapped QoS flow has nowhere else to go (TS 37.324 §5.3.1)"
        )));
    }
    // Two DRBs with one identity would share an RLC entity and interleave two
    // flows' sequence-number spaces -- a silent corruption, so it is refused.
    for (index, spec) in specs.iter().enumerate() {
        if specs[..index].iter().any(|s| s.drb_id == spec.drb_id) {
            return Err(RrcReconfigurationError::InvalidFieldValue(format!(
                "DRB identity {} is used twice in one PDU session",
                spec.drb_id
            )));
        }
        if specs[..index].iter().any(|s| s.lcid == spec.lcid) {
            return Err(RrcReconfigurationError::InvalidFieldValue(format!(
                "logical channel identity {} is used twice in one PDU session",
                spec.lcid
            )));
        }
    }

    let drbs: Vec<DRB_ToAddMod> = specs
        .iter()
        .map(|spec| {
            build_drb_to_add_mod(
                pdu_session_id,
                spec.drb_id,
                &spec.qfis,
                spec.default_drb,
                spec.integrity_protection,
            )
        })
        .collect();

    Ok(RadioBearerConfig {
        srb_to_add_mod_list: None,
        srb3_to_release: None,
        drb_to_add_mod_list: Some(DRB_ToAddModList(drbs)),
        drb_to_release_list: None,
        security_config: None,
    })
}

/// Build a `CellGroupConfig` (master cell group) with one RLC bearer for the
/// given DRB. `lcid` is the logical channel identity carrying the DRB.
pub fn build_cell_group_config(drb_id: u8, lcid: u8) -> CellGroupConfig {
    build_multi_drb_cell_group_config(&[(drb_id, lcid)])
}

/// Build a `CellGroupConfig` with one RLC bearer per `(drb_id, lcid)` pair
/// (issue #44).
///
/// One RLC bearer per DRB, which is what TS 38.331 §6.3.2 requires: a DRB with no
/// `RLC-BearerConfig` naming it is configured at the SDAP and PDCP layers and has
/// no logical channel to ride, so nothing would carry it.
pub fn build_multi_drb_cell_group_config(bearers: &[(u8, u8)]) -> CellGroupConfig {
    let rlc_bearers: Vec<RLC_BearerConfig> = bearers
        .iter()
        .map(|&(drb_id, lcid)| RLC_BearerConfig {
            logical_channel_identity: LogicalChannelIdentity(lcid),
            served_radio_bearer: Some(RLC_BearerConfigServedRadioBearer::Drb_Identity(
                DRB_Identity(drb_id),
            )),
            reestablish_rlc: None,
            rlc_config: None,
            mac_logical_channel_config: None,
        })
        .collect();

    CellGroupConfig {
        cell_group_id: CellGroupId(0),
        // `None` for an empty list, not an empty SEQUENCE-OF: the ASN.1 has a
        // lower bound of 1, so an empty vector would not encode.
        rlc_bearer_to_add_mod_list: if rlc_bearers.is_empty() {
            None
        } else {
            Some(CellGroupConfigRlc_BearerToAddModList(rlc_bearers))
        },
        rlc_bearer_to_release_list: None,
        mac_cell_group_config: None,
        physical_cell_group_config: None,
        sp_cell_config: None,
        s_cell_to_add_mod_list: None,
        s_cell_to_release_list: None,
    }
}

/// Read every DRB of a `RadioBearerConfig` back as the mapping it states
/// (issue #44).
///
/// The UE half: it is **told** which QFIs ride which bearer rather than
/// recomputing the gNB's policy, which is what lets the policy change on the
/// network side without a matching UE change. Returns one entry per
/// `DRB-ToAddMod` whose `cn-Association` is an `SDAP-Config` for `pdu_session_id`.
///
/// The `lcid` is **not** in a `RadioBearerConfig` — it is in the `CellGroupConfig`
/// — so it comes back as 0 here and the caller pairs the two. Stated rather than
/// silently zero, because an LCID of 0 is not a value (`INTEGER (1..32)`).
pub fn read_drb_specs(rbc: &RadioBearerConfig, pdu_session_id: u8) -> Vec<DrbSpec> {
    let Some(list) = rbc.drb_to_add_mod_list.as_ref() else {
        return Vec::new();
    };
    list.0
        .iter()
        .filter_map(|drb| {
            let DRB_ToAddModCnAssociation::Sdap_Config(sdap) = drb.cn_association.as_ref()? else {
                // An `eps-BearerIdentity` association is an EPS bearer, not a 5GS
                // QoS-flow mapping, so it has no QFIs to read.
                return None;
            };
            if sdap.pdu_session.0 != pdu_session_id {
                return None;
            }
            Some(DrbSpec {
                drb_id: drb.drb_identity.0,
                // See the doc comment: the LCID lives in the CellGroupConfig.
                lcid: 0,
                qfis: sdap
                    .mapped_qo_s_flows_to_add
                    .as_ref()
                    .map(|m| m.0.iter().map(|q| q.0).collect())
                    .unwrap_or_default(),
                default_drb: sdap.default_drb.0,
                integrity_protection: drb_integrity_protection(rbc, drb.drb_identity.0),
            })
        })
        .collect()
}

/// The `logicalChannelIdentity` serving `drb_id`, from a `CellGroupConfig`.
///
/// `None` when no RLC bearer names that DRB — which is a configuration the UE
/// cannot act on, rather than a default to invent: a DRB with no logical channel
/// has nothing to ride.
pub fn lcid_for_drb(cgc: &CellGroupConfig, drb_id: u8) -> Option<u8> {
    cgc.rlc_bearer_to_add_mod_list
        .as_ref()?
        .0
        .iter()
        .find(|b| {
            matches!(
                b.served_radio_bearer.as_ref(),
                Some(RLC_BearerConfigServedRadioBearer::Drb_Identity(d)) if d.0 == drb_id
            )
        })
        .map(|b| b.logical_channel_identity.0)
}

/// amfg-04: build a fully-structured `RrcReconfigurationParams` for bringing up
/// one PDU session's DRB. The structured `RadioBearerConfig` and
/// `CellGroupConfig` are UPER-encoded into the existing byte-carrying param
/// fields (`radio_bearer_config` is decoded back and embedded structurally by
/// `build_rrc_reconfiguration`; `master_cell_group` is carried as the
/// `OCTET STRING (CONTAINING CellGroupConfig)` masterCellGroup IE). The opaque
/// byte path is preserved for callers that already have pre-encoded config.
///
/// `meas_config` is the measurement configuration to carry, or `None` to leave the
/// UE's alone (issue #170). Threaded through rather than defaulted here, because
/// the margin is the **gNB's** configuration and this crate has no access to it —
/// defaulting would put a second A3 margin in the tree, which is the defect #170
/// is about.
//
// `too_many_arguments` (8, over the 7 threshold) is allowed rather than fixed by
// bundling into a params struct. Every argument is a distinct IE that goes on the
// wire, and the two that could be confused for each other are already newtyped
// (`DrbIntegrityProtection` for exactly that reason — see its own doc). A wrapper
// struct would add a type whose only job is to be destructured one line later, and
// the eight call sites would each gain a field name without gaining a check.
#[allow(clippy::too_many_arguments)]
pub fn build_drb_reconfiguration_params(
    rrc_transaction_id: u8,
    pdu_session_id: u8,
    drb_id: u8,
    lcid: u8,
    qfis: &[u8],
    default_drb: bool,
    integrity_protection: DrbIntegrityProtection,
    meas_config: Option<A3MeasConfigParams>,
) -> Result<RrcReconfigurationParams, RrcReconfigurationError> {
    let rbc = build_drb_radio_bearer_config(
        pdu_session_id,
        drb_id,
        qfis,
        default_drb,
        integrity_protection,
    );
    let cgc = build_cell_group_config(drb_id, lcid);

    let radio_bearer_config = encode_rrc(&rbc)?;
    let master_cell_group = encode_rrc(&cgc)?;

    Ok(RrcReconfigurationParams {
        rrc_transaction_id,
        radio_bearer_config: Some(radio_bearer_config),
        secondary_cell_group: None,
        master_cell_group: Some(master_cell_group),
        full_config: false,
        // A DRB-establishing reconfiguration is not a handover, so the UE keeps its keys.
        master_key_update: None,
        meas_config,
        // Left to the caller (issue #56). This builder takes no cell configuration,
        // so it has no ephemeris to signal; the gNB's RRC task adds the NTN update
        // to the params it gets back when the cell is an NTN one. Baking `None` in
        // here rather than inventing a default keeps a TN cell's reconfiguration
        // byte-identical to what it was.
        ntn_config: None,
        // Likewise left to the caller (issue #141): a sidelink grant answers a UE's
        // `SidelinkUEInformation`, which this DRB builder knows nothing about.
        sl_config: None,
    })
}

/// Build the `RrcReconfigurationParams` establishing **every** DRB of one PDU
/// session (issue #44).
///
/// The multi-DRB generalisation of [`build_drb_reconfiguration_params`]. Both the
/// `RadioBearerConfig` and the `CellGroupConfig` carry one entry per DRB, so the UE
/// receives the whole QFI→DRB mapping in one message and is told it rather than
/// deriving it.
///
/// Identical to the single-DRB function when `specs` has one element — asserted by
/// `the_multi_drb_builder_matches_the_single_drb_one_for_a_lone_bearer`, which is
/// what lets the pre-SDAP golden byte vectors go on testing the live path.
pub fn build_multi_drb_reconfiguration_params(
    rrc_transaction_id: u8,
    pdu_session_id: u8,
    specs: &[DrbSpec],
    meas_config: Option<A3MeasConfigParams>,
) -> Result<RrcReconfigurationParams, RrcReconfigurationError> {
    let rbc = build_multi_drb_radio_bearer_config(pdu_session_id, specs)?;
    let bearers: Vec<(u8, u8)> = specs.iter().map(|s| (s.drb_id, s.lcid)).collect();
    let cgc = build_multi_drb_cell_group_config(&bearers);

    Ok(RrcReconfigurationParams {
        rrc_transaction_id,
        radio_bearer_config: Some(encode_rrc(&rbc)?),
        secondary_cell_group: None,
        master_cell_group: Some(encode_rrc(&cgc)?),
        full_config: false,
        master_key_update: None,
        meas_config,
        // See `build_drb_reconfiguration_params`: the caller supplies the NTN
        // update, because this builder holds no cell configuration (issue #56).
        ntn_config: None,
        // Likewise the sidelink grant (issue #141).
        sl_config: None,
    })
}

/// Whether a DRB's PDCP entity integrity-protects user data
/// (TS 38.331 `PDCP-Config.drb.integrityProtection`, TS 33.501 §6.6.1).
///
/// An enum and not a `bool` because it sits next to `default_drb` in the DRB
/// builders' signatures, and two adjacent booleans is how a caller silently swaps
/// "this is the default DRB" for "protect this DRB".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DrbIntegrityProtection {
    /// No `integrityProtection` IE: the DRB carries no MAC-I.
    #[default]
    Disabled,
    /// `integrityProtection: enabled`.
    Enabled,
}

/// Read a DRB's configured integrity protection out of a `RadioBearerConfig`.
///
/// The UE half of the signalling: the gNB decides the policy from the SMF's
/// `SecurityIndication` and puts it here, and this is how the UE learns it rather
/// than being configured to match. Returns [`DrbIntegrityProtection::Disabled`]
/// when the DRB, its `PDCP-Config`, or the IE itself is absent — all three mean the
/// same thing on the wire (§6.3.2, `Cond ConnectedTo5GC1`: the IE is present only
/// when protection is on).
pub fn drb_integrity_protection(rbc: &RadioBearerConfig, drb_id: u8) -> DrbIntegrityProtection {
    let Some(list) = rbc.drb_to_add_mod_list.as_ref() else {
        return DrbIntegrityProtection::Disabled;
    };
    let protected = list
        .0
        .iter()
        .find(|drb| drb.drb_identity.0 == drb_id)
        .and_then(|drb| drb.pdcp_config.as_ref())
        .and_then(|cfg| cfg.drb.as_ref())
        .and_then(|drb| drb.integrity_protection.as_ref())
        .is_some();
    if protected {
        DrbIntegrityProtection::Enabled
    } else {
        DrbIntegrityProtection::Disabled
    }
}

/// Parse an RRC Reconfiguration from a DL-DCCH message
pub fn parse_rrc_reconfiguration(
    msg: &DL_DCCH_Message,
) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let rrc_reconfiguration = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::RrcReconfiguration(reconfig) => reconfig,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfiguration".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_reconfiguration.critical_extensions {
        RRCReconfigurationCriticalExtensions::RrcReconfiguration(ies) => ies,
        RRCReconfigurationCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfiguration".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Encode radio bearer config back to bytes if present
    let radio_bearer_config = if let Some(ref config) = ies.radio_bearer_config {
        Some(encode_rrc(config)?)
    } else {
        None
    };

    // Extract secondary cell group
    let secondary_cell_group = ies.secondary_cell_group.as_ref().map(|scg| scg.0.clone());

    // Extract master cell group, full_config, masterKeyUpdate and the NTN update
    // from the v1530 extension
    let (master_cell_group, full_config, master_key_update, ntn_config) =
        if let Some(ref ext) = ies.non_critical_extension {
            let mcg = ext.master_cell_group.as_ref().map(|m| m.0.clone());
            let fc = ext.full_config.is_some();
            let mku = ext
                .master_key_update
                .as_ref()
                .map(|u| MasterKeyUpdateParams {
                    key_set_change_indicator: u.key_set_change_indicator.0,
                    next_hop_chaining_count: u.next_hop_chaining_count.0,
                });
            // The connected-mode NTN update (issue #56). `None` for a delivery that
            // carried some other SIB, which is legal -- see `read_ntn_dedicated_si`.
            let ntn = ext
                .dedicated_system_information_delivery
                .as_ref()
                .and_then(|d| read_ntn_dedicated_si(&d.0));
            (mcg, fc, mku, ntn)
        } else {
            (None, false, None, None)
        };

    // The dedicated sidelink configuration (issue #141). Read from the whole `ies`
    // rather than from `ext` above, because it sits four extensions further down
    // (v1530 -> v1540 -> v1560 -> v1610) and the walk is worth having in one named
    // place.
    let (sl_config, sl_config_released) = read_sl_config_from_chain(ies);

    Ok(RrcReconfigurationData {
        rrc_transaction_id: rrc_reconfiguration.rrc_transaction_identifier.0,
        radio_bearer_config,
        secondary_cell_group,
        master_cell_group,
        full_config,
        master_key_update,
        // Cloned rather than re-encoded, unlike `radio_bearer_config` above: the
        // measurement configuration is CONSUMED by the receiver (it installs the
        // margin), not forwarded, so a byte round trip would only add a way to
        // lose it (issue #170).
        meas_config: ies.meas_config.clone(),
        ntn_config,
        // The dedicated sidelink grant (issue #141), from four extensions further down
        // the chain than the NTN update above.
        sl_config,
        sl_config_released,
    })
}

// ============================================================================
// RRC Reconfiguration Complete
// ============================================================================

/// Parameters for building an RRC Reconfiguration Complete message
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
}

/// Parsed RRC Reconfiguration Complete data
#[derive(Debug, Clone)]
pub struct RrcReconfigurationCompleteData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
}

/// Build an RRC Reconfiguration Complete message
pub fn build_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<UL_DCCH_Message, RrcReconfigurationError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReconfigurationError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    let rrc_reconfiguration_complete_ies = RRCReconfigurationComplete_IEs {
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let rrc_reconfiguration_complete = RRCReconfigurationComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions:
            RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(
                rrc_reconfiguration_complete_ies,
            ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(
        rrc_reconfiguration_complete,
    ));

    Ok(UL_DCCH_Message {
        message: message_type,
    })
}

/// Parse an RRC Reconfiguration Complete from a UL-DCCH message
pub fn parse_rrc_reconfiguration_complete(
    msg: &UL_DCCH_Message,
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let rrc_reconfiguration_complete = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::RrcReconfigurationComplete(complete) => complete,
            _ => {
                return Err(RrcReconfigurationError::InvalidMessageType {
                    expected: "RRCReconfigurationComplete".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Verify we have the expected critical extensions variant
    match &rrc_reconfiguration_complete.critical_extensions {
        RRCReconfigurationCompleteCriticalExtensions::RrcReconfigurationComplete(_) => {}
        RRCReconfigurationCompleteCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReconfigurationError::InvalidMessageType {
                expected: "rrcReconfigurationComplete".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    Ok(RrcReconfigurationCompleteData {
        rrc_transaction_id: rrc_reconfiguration_complete.rrc_transaction_identifier.0,
    })
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Reconfiguration to bytes
pub fn encode_rrc_reconfiguration(
    params: &RrcReconfigurationParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration from bytes
pub fn decode_rrc_reconfiguration(
    bytes: &[u8],
) -> Result<RrcReconfigurationData, RrcReconfigurationError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration(&msg)
}

/// Build and encode an RRC Reconfiguration Complete to bytes
pub fn encode_rrc_reconfiguration_complete(
    params: &RrcReconfigurationCompleteParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    let msg = build_rrc_reconfiguration_complete(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Reconfiguration Complete from bytes
pub fn decode_rrc_reconfiguration_complete(
    bytes: &[u8],
) -> Result<RrcReconfigurationCompleteData, RrcReconfigurationError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_reconfiguration_complete(&msg)
}

/// Check if a DL-DCCH message is an RRC Reconfiguration
pub fn is_rrc_reconfiguration(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcReconfiguration(_))
    )
}

/// Check if a UL-DCCH message is an RRC Reconfiguration Complete
pub fn is_rrc_reconfiguration_complete(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcReconfigurationComplete(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Multi-DRB signalling for the SDAP QFI->DRB mapping (issue #44).
    // ========================================================================

    /// The default DRB of a session, non-GBR flows on it.
    fn default_spec(drb_id: u8, lcid: u8, qfis: Vec<u8>) -> DrbSpec {
        DrbSpec {
            drb_id,
            lcid,
            qfis,
            default_drb: true,
            integrity_protection: DrbIntegrityProtection::Disabled,
        }
    }

    /// The GBR DRB of a session.
    fn gbr_spec(drb_id: u8, lcid: u8, qfis: Vec<u8>) -> DrbSpec {
        DrbSpec {
            drb_id,
            lcid,
            qfis,
            default_drb: false,
            integrity_protection: DrbIntegrityProtection::Disabled,
        }
    }

    /// The equivalence that lets the pre-SDAP golden byte vectors go on testing the
    /// live path: for ONE bearer, the multi-DRB builders produce byte-identical
    /// output to the single-DRB ones.
    ///
    /// Without this, the golden vectors would be testing a function the data path no
    /// longer calls — the "correct but unreachable" trap, inverted.
    #[test]
    fn the_multi_drb_builder_matches_the_single_drb_one_for_a_lone_bearer() {
        for integrity in [
            DrbIntegrityProtection::Disabled,
            DrbIntegrityProtection::Enabled,
        ] {
            let single = build_drb_radio_bearer_config(1, 1, &[9], true, integrity);
            let multi = build_multi_drb_radio_bearer_config(
                1,
                &[DrbSpec {
                    drb_id: 1,
                    lcid: 4,
                    qfis: vec![9],
                    default_drb: true,
                    integrity_protection: integrity,
                }],
            )
            .expect("one default DRB is a legal session");
            assert_eq!(multi, single, "{integrity:?}: the structures must be equal");
            assert_eq!(
                encode_rrc(&multi).expect("encode"),
                encode_rrc(&single).expect("encode"),
                "{integrity:?}: and byte-identical, or the golden vectors test a \
                 function the data path no longer calls"
            );
        }
        // The same for the cell group.
        assert_eq!(
            encode_rrc(&build_multi_drb_cell_group_config(&[(1, 4)])).expect("encode"),
            encode_rrc(&build_cell_group_config(1, 4)).expect("encode")
        );
    }

    /// Criterion 1 on the wire: two QFIs on ONE PDU session reach the UE mapped to
    /// two DIFFERENT DRBs, and the mapping survives a byte round trip.
    #[test]
    fn two_qfis_on_one_session_are_signalled_on_distinct_drbs() {
        let specs = [default_spec(1, 4, vec![9, 5]), gbr_spec(17, 20, vec![1, 2])];
        let params = build_multi_drb_reconfiguration_params(0, 1, &specs, None)
            .expect("two DRBs with one default is legal");
        let bytes = encode_rrc_reconfiguration(&params).expect("encode");

        // Byte round trip first: this message now carries two DRBs and two RLC
        // bearers, so a length-determinant error in either SEQUENCE-OF would
        // corrupt whatever follows.
        let msg: DL_DCCH_Message = decode_rrc(&bytes).expect("decode");
        assert_eq!(
            encode_rrc(&msg).expect("re-encode"),
            bytes,
            "a two-DRB reconfiguration must byte round trip"
        );

        let data = decode_rrc_reconfiguration(&bytes).expect("parse");
        let rbc: RadioBearerConfig =
            decode_rrc(&data.radio_bearer_config.expect("present")).expect("decode RBC");
        let cgc: CellGroupConfig =
            decode_rrc(&data.master_cell_group.expect("present")).expect("decode CGC");

        let read = read_drb_specs(&rbc, 1);
        assert_eq!(read.len(), 2, "both DRBs must reach the UE");

        // The QFI sets are disjoint and land on different bearers -- criterion 1.
        let drb_for = |qfi: u8| -> u8 {
            read.iter()
                .find(|s| s.qfis.contains(&qfi))
                .unwrap_or_else(|| panic!("QFI {qfi} must be mapped"))
                .drb_id
        };
        assert_ne!(
            drb_for(9),
            drb_for(1),
            "QFI 9 and QFI 1 are on one PDU session and MUST map to distinct DRBs"
        );
        assert_eq!(drb_for(9), 1, "the non-GBR flows on the default DRB");
        assert_eq!(drb_for(1), 17, "and the GBR flows on the second");
        assert_eq!(drb_for(5), 1, "QFI 5 rides with QFI 9");
        assert_eq!(drb_for(2), 17);

        // Exactly one default DRB, and it is the one carrying the unmapped-flow
        // fallback.
        let defaults: Vec<u8> = read
            .iter()
            .filter(|s| s.default_drb)
            .map(|s| s.drb_id)
            .collect();
        assert_eq!(
            defaults,
            vec![1],
            "exactly one default DRB (TS 37.324 §5.3.1)"
        );

        // And each DRB has its own logical channel, or nothing carries it.
        assert_eq!(lcid_for_drb(&cgc, 1), Some(4));
        assert_eq!(lcid_for_drb(&cgc, 17), Some(20));
        assert_eq!(
            lcid_for_drb(&cgc, 9),
            None,
            "a DRB the cell group does not name has no logical channel"
        );
    }

    /// A session with no default DRB, or two, is REFUSED: an unmapped QoS flow goes
    /// to the default DRB (§5.3.1), so neither shape has defined behaviour.
    #[test]
    fn a_session_without_exactly_one_default_drb_is_refused() {
        // None.
        assert!(build_multi_drb_radio_bearer_config(
            1,
            &[gbr_spec(1, 4, vec![1]), gbr_spec(17, 20, vec![2])]
        )
        .is_err());
        // Two.
        assert!(build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), default_spec(17, 20, vec![5])]
        )
        .is_err());
        // Empty.
        assert!(build_multi_drb_radio_bearer_config(1, &[]).is_err());
        // And exactly one is accepted, so the check is a check and not a refusal of
        // everything.
        assert!(build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), gbr_spec(17, 20, vec![1])]
        )
        .is_ok());
    }

    /// A repeated DRB identity or LCID is refused. Two bearers sharing an identity
    /// would share an RLC entity and interleave two flows' sequence-number spaces —
    /// a silent corruption, not a failure, which is why it is caught at the builder.
    #[test]
    fn a_repeated_drb_identity_or_lcid_is_refused() {
        assert!(build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), gbr_spec(1, 20, vec![1])]
        )
        .is_err());
        assert!(build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), gbr_spec(17, 4, vec![1])]
        )
        .is_err());
    }

    /// A DRB with no QFIs omits `mappedQoS-FlowsToAdd` rather than encoding an empty
    /// SEQUENCE-OF, whose ASN.1 lower bound is 1. This is the shape a GBR DRB has
    /// when a session happens to admit no GBR flow.
    #[test]
    fn a_drb_with_no_qfis_omits_the_mapping_list() {
        let rbc = build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), gbr_spec(17, 20, vec![])],
        )
        .expect("legal");
        let bytes = encode_rrc(&rbc).expect("an empty QFI set must still encode");
        let decoded: RadioBearerConfig = decode_rrc(&bytes).expect("decode");
        assert_eq!(encode_rrc(&decoded).expect("re-encode"), bytes);

        let read = read_drb_specs(&decoded, 1);
        let gbr = read.iter().find(|s| s.drb_id == 17).expect("present");
        assert!(
            gbr.qfis.is_empty(),
            "no QFIs, and no empty SEQUENCE-OF either"
        );
    }

    /// `read_drb_specs` ignores a DRB belonging to a DIFFERENT PDU session, so a UE
    /// with two sessions does not read one session's mapping into the other.
    #[test]
    fn a_drb_of_another_session_is_not_read() {
        let mut rbc = build_multi_drb_radio_bearer_config(
            1,
            &[default_spec(1, 4, vec![9]), gbr_spec(17, 20, vec![1])],
        )
        .expect("legal");
        // Re-point the GBR DRB at session 2.
        if let Some(DRB_ToAddModCnAssociation::Sdap_Config(sdap)) = rbc
            .drb_to_add_mod_list
            .as_mut()
            .expect("present")
            .0
            .iter_mut()
            .find(|d| d.drb_identity.0 == 17)
            .and_then(|d| d.cn_association.as_mut())
        {
            sdap.pdu_session = PDU_SessionID(2);
        }
        let read = read_drb_specs(&rbc, 1);
        assert_eq!(read.len(), 1, "only session 1's DRB");
        assert_eq!(read[0].drb_id, 1);
        // And session 2's is readable on its own.
        assert_eq!(read_drb_specs(&rbc, 2).len(), 1);
    }

    /// Per-DRB integrity protection is read back per DRB, not per session: the two
    /// bearers of one session can differ, and a reader that took the first one's
    /// answer for both would key one entity wrongly.
    #[test]
    fn integrity_protection_is_read_back_per_drb() {
        let rbc = build_multi_drb_radio_bearer_config(
            1,
            &[
                DrbSpec {
                    integrity_protection: DrbIntegrityProtection::Disabled,
                    ..default_spec(1, 4, vec![9])
                },
                DrbSpec {
                    integrity_protection: DrbIntegrityProtection::Enabled,
                    ..gbr_spec(17, 20, vec![1])
                },
            ],
        )
        .expect("legal");
        let bytes = encode_rrc(&rbc).expect("encode");
        let decoded: RadioBearerConfig = decode_rrc(&bytes).expect("decode");
        let read = read_drb_specs(&decoded, 1);
        let by_id = |id: u8| {
            read.iter()
                .find(|s| s.drb_id == id)
                .expect("present")
                .integrity_protection
        };
        assert_eq!(by_id(1), DrbIntegrityProtection::Disabled);
        assert_eq!(by_id(17), DrbIntegrityProtection::Enabled);
    }

    // ========================================================================
    // Handover command golden bytes (issue #107, criterion 6)
    // ========================================================================

    /// Hand-derived handover command: `RRCReconfiguration`, tid 0, whose
    /// `masterCellGroup` carries a `reconfigurationWithSync` to physCellId 16 with
    /// `newUE-Identity` 1 and `t304` 1000 ms, and `fullConfig` set.
    ///
    /// Derivation of the framing from `tools/rrc-19.3.0.asn1`:
    ///
    /// ```text
    /// bit 0      DL-DCCH-MessageType CHOICE, 2 alternatives -> 1 bit. c1 = 0
    /// bits 1..4  c1 CHOICE, 16 alternatives -> 4 bits.
    ///            rrcReconfiguration = 0 -> 0000
    /// bits 5..6  RRC-TransactionIdentifier, INTEGER (0..3) -> 2 bits. tid 0 -> 00
    /// bit 7      RRCReconfiguration criticalExtensions CHOICE -> 1 bit.
    ///            rrcReconfiguration = 0
    ///            => byte 0 = 0b0_0000_00_0 = 0x00
    /// ```
    ///
    /// The tail is the `RRCReconfiguration-IEs` preamble, the v1530 extension
    /// carrying `masterCellGroup` and `fullConfig`, and the embedded
    /// `CellGroupConfig`. Those inner layers are pinned by this literal and
    /// cross-checked by `golden_handover_command_cross_decode` rather than derived
    /// bit by bit — stated plainly, because a wrong derivation in a comment is worse
    /// than an honest "cross-checked".
    const GOLDEN_HANDOVER_COMMAND_TID0: [u8; 13] = [
        0x00, 0x0E, 0x00, 0x48, 0x20, 0x42, 0x40, 0x00, 0x20, 0x50, 0x00, 0x03, 0x40,
    ];

    fn golden_handover_params(rrc_transaction_id: u8) -> HandoverCommandParams {
        HandoverCommandParams {
            rrc_transaction_id,
            target_phys_cell_id: 16,
            new_ue_identity: 1,
            t304_ms: 1000,
            full_config: true,
            master_key_update: None,
        }
    }

    #[test]
    fn golden_handover_command_bytes() {
        let bytes = encode_handover_command(&golden_handover_params(0)).expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_HANDOVER_COMMAND_TID0.to_vec(),
            "the handover command must match the hand-derived UPER bytes, not a \
             round trip through this codec"
        );

        // The tid occupies bits 5..6 of byte 0 only: tid 2 -> 0b0_0000_10_0 = 0x04.
        let bytes_tid2 = encode_handover_command(&golden_handover_params(2)).expect("encode");
        let mut expected = GOLDEN_HANDOVER_COMMAND_TID0;
        expected[0] = 0x04;
        assert_eq!(bytes_tid2, expected.to_vec());
    }

    #[test]
    fn golden_handover_command_cross_decode() {
        let data = decode_handover_command(&GOLDEN_HANDOVER_COMMAND_TID0)
            .expect("the golden must decode as a handover command");
        assert_eq!(data.rrc_transaction_id, 0);
        assert_eq!(data.target_phys_cell_id, 16);
        assert_eq!(data.new_ue_identity, 1);
        assert_eq!(data.t304_ms, 1000);
        assert!(data.full_config);

        // And the reconfigurationWithSync really is there, which is what makes this
        // a handover command rather than a plain reconfiguration.
        let reconf = decode_rrc_reconfiguration(&GOLDEN_HANDOVER_COMMAND_TID0).expect("decode");
        let cgc: CellGroupConfig = decode_rrc(
            reconf
                .master_cell_group
                .as_ref()
                .expect("masterCellGroup present"),
        )
        .expect("decode CellGroupConfig");
        let sync = cgc
            .sp_cell_config
            .expect("spCellConfig present")
            .reconfiguration_with_sync
            .expect("reconfigurationWithSync present");
        assert_eq!(
            sync.sp_cell_config_common
                .expect("spCellConfigCommon present")
                .phys_cell_id
                .expect("physCellId present")
                .0,
            16
        );
    }

    /// A target PCI past the ASN.1 bound is refused rather than encoded.
    #[test]
    fn an_out_of_range_target_phys_cell_id_is_refused() {
        let mut params = golden_handover_params(0);
        params.target_phys_cell_id = 1008;
        assert!(encode_handover_command(&params).is_err());
    }

    /// `t304` enumerates non-uniform values, so a value outside the set is refused
    /// rather than rounded: it is the timer that decides when the UE declares
    /// handover failure.
    #[test]
    fn a_non_enumerated_t304_is_refused_not_rounded() {
        for legal in [50u16, 100, 150, 200, 500, 1000, 2000, 10000] {
            let mut params = golden_handover_params(0);
            params.t304_ms = legal;
            assert!(
                encode_handover_command(&params).is_ok(),
                "t304 {legal} ms is enumerated and must encode"
            );
            assert_eq!(
                t304_ms(t304_index(legal).expect("index")).expect("ms"),
                legal
            );
        }
        for illegal in [0u16, 300, 250, 1500, 9999] {
            let mut params = golden_handover_params(0);
            params.t304_ms = illegal;
            assert!(
                encode_handover_command(&params).is_err(),
                "t304 {illegal} ms is NOT enumerated and must be refused, not \
                 rounded to a deadline nobody configured"
            );
        }
    }

    // ========================================================================
    // RRC Reconfiguration Tests
    // ========================================================================

    fn create_test_reconfiguration_params() -> RrcReconfigurationParams {
        RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0x00, 0x01, 0x02]), // Sample cell group config
            full_config: false,
            master_key_update: None,
            meas_config: None,
            ntn_config: None,
            sl_config: None,
        }
    }

    /// The A3 `measConfig` a default-configured gNB signals: 3 dB offset, 1 dB
    /// hysteresis (issue #170).
    fn test_meas_config_params() -> A3MeasConfigParams {
        A3MeasConfigParams {
            meas_id: 1,
            meas_object_id: 1,
            report_config_id: 1,
            ssb_frequency_arfcn: 632_628,
            ssb_subcarrier_spacing_khz: 30,
            a3_offset_db: 3.0,
            hysteresis_db: 1.0,
            time_to_trigger_ms: 640,
            report_interval_ms: 480,
            report_amount: Some(8),
            max_report_cells: 4,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let msg = build_rrc_reconfiguration(&params).unwrap();
        let result = parse_rrc_reconfiguration(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.master_cell_group, params.master_cell_group);
        assert_eq!(data.full_config, params.full_config);
    }

    #[test]
    fn test_rrc_reconfiguration_with_full_config() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 2,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: Some(vec![0xAA, 0xBB]),
            full_config: true,
            master_key_update: None,
            meas_config: None,
            ntn_config: None,
            sl_config: None,
        };

        let msg = build_rrc_reconfiguration(&params).unwrap();
        let data = parse_rrc_reconfiguration(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert!(data.full_config);
        assert_eq!(data.master_cell_group, Some(vec![0xAA, 0xBB]));
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration() {
        let params = create_test_reconfiguration_params();
        let encoded = encode_rrc_reconfiguration(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_reconfiguration() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
            master_key_update: None,
            meas_config: None,
            ntn_config: None,
            sl_config: None,
        };

        let result = build_rrc_reconfiguration(&params);
        assert!(result.is_err());
    }

    // ========================================================================
    // RRC Reconfiguration Complete Tests
    // ========================================================================

    fn create_test_reconfiguration_complete_params() -> RrcReconfigurationCompleteParams {
        RrcReconfigurationCompleteParams {
            rrc_transaction_id: 0,
        }
    }

    #[test]
    fn test_build_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_reconfiguration_complete(&msg));
    }

    #[test]
    fn test_parse_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let msg = build_rrc_reconfiguration_complete(&params).unwrap();
        let result = parse_rrc_reconfiguration_complete(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_encode_decode_rrc_reconfiguration_complete() {
        let params = create_test_reconfiguration_complete_params();
        let encoded = encode_rrc_reconfiguration_complete(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_reconfiguration_complete(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id_complete() {
        let params = RrcReconfigurationCompleteParams {
            rrc_transaction_id: 4, // Invalid: must be 0-3
        };

        let result = build_rrc_reconfiguration_complete(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_rrc_reconfiguration_complete_all_transaction_ids() {
        // Test all valid transaction IDs (0-3)
        for id in 0..=3 {
            let params = RrcReconfigurationCompleteParams {
                rrc_transaction_id: id,
            };
            let msg = build_rrc_reconfiguration_complete(&params).unwrap();
            let data = parse_rrc_reconfiguration_complete(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }

    // ========================================================================
    // amfg-04: structured DRB / SDAP / CellGroup tests
    // ========================================================================

    /// #32, criterion 3 (the signalling half): the DRB's integrity protection is
    /// carried in `PDCP-Config.drb.integrityProtection` and read back off the wire,
    /// so the UE is *told* the policy rather than configured to match it.
    #[test]
    fn drb_integrity_protection_survives_a_uper_round_trip() {
        for (configured, expected) in [
            (
                DrbIntegrityProtection::Disabled,
                DrbIntegrityProtection::Disabled,
            ),
            (
                DrbIntegrityProtection::Enabled,
                DrbIntegrityProtection::Enabled,
            ),
        ] {
            let rbc = build_drb_radio_bearer_config(2, 1, &[9], true, configured);
            let bytes = encode_rrc(&rbc).expect("encode");
            let decoded: RadioBearerConfig = decode_rrc(&bytes).expect("decode");
            assert_eq!(
                drb_integrity_protection(&decoded, 1),
                expected,
                "the {configured:?} setting must survive the wire"
            );
        }
        // And the two encodings differ, or the IE is not actually on the wire.
        assert_ne!(
            encode_rrc(&build_drb_radio_bearer_config(
                2,
                1,
                &[9],
                true,
                DrbIntegrityProtection::Enabled
            ))
            .unwrap(),
            encode_rrc(&build_drb_radio_bearer_config(
                2,
                1,
                &[9],
                true,
                DrbIntegrityProtection::Disabled
            ))
            .unwrap(),
            "an `integrityProtection` that changed no bytes would be signalled nowhere"
        );
    }

    /// An unprotected DRB's encoding is unchanged by issue #32, which is what keeps
    /// the goldens below meaningful and the default build byte-identical.
    #[test]
    fn an_unprotected_drb_omits_the_pdcp_drb_config_entirely() {
        let rbc = build_drb_radio_bearer_config(2, 1, &[9], true, DrbIntegrityProtection::Disabled);
        let drb = &rbc.drb_to_add_mod_list.as_ref().unwrap().0[0];
        assert!(
            drb.pdcp_config.as_ref().unwrap().drb.is_none(),
            "PDCP-Config.drb must stay absent when there is nothing to configure in it"
        );
    }

    /// `drb_integrity_protection` answers about the DRB it was asked about, not the
    /// first one in the list. Pinned because with one DRB per session the two are
    /// indistinguishable, and a multi-DRB reconfiguration would silently protect the
    /// wrong bearer.
    #[test]
    fn drb_integrity_protection_is_read_per_drb_and_not_from_the_first() {
        let protected =
            build_drb_radio_bearer_config(1, 1, &[1], true, DrbIntegrityProtection::Enabled);
        let plain =
            build_drb_radio_bearer_config(2, 2, &[2], false, DrbIntegrityProtection::Disabled);
        let mut merged = protected.clone();
        merged.drb_to_add_mod_list = Some(DRB_ToAddModList(vec![
            protected.drb_to_add_mod_list.unwrap().0[0].clone(),
            plain.drb_to_add_mod_list.unwrap().0[0].clone(),
        ]));
        assert_eq!(
            drb_integrity_protection(&merged, 1),
            DrbIntegrityProtection::Enabled
        );
        assert_eq!(
            drb_integrity_protection(&merged, 2),
            DrbIntegrityProtection::Disabled
        );
        assert_eq!(
            drb_integrity_protection(&merged, 3),
            DrbIntegrityProtection::Disabled,
            "a DRB that is not in the list is not protected"
        );
    }

    #[test]
    fn test_build_drb_radio_bearer_config_maps_all_qfis() {
        let qfis = [1u8, 5, 9];
        let rbc =
            build_drb_radio_bearer_config(2, 1, &qfis, true, DrbIntegrityProtection::Disabled);

        let drb_list = rbc
            .drb_to_add_mod_list
            .as_ref()
            .expect("DRB-ToAddModList present");
        assert_eq!(drb_list.0.len(), 1);
        let drb = &drb_list.0[0];
        assert_eq!(drb.drb_identity.0, 1);

        let sdap = match drb.cn_association.as_ref().expect("cn-Association present") {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config cn-Association"),
        };
        assert_eq!(sdap.pdu_session.0, 2);
        assert!(sdap.default_drb.0);
        let mapped = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .expect("mappedQoS-FlowsToAdd present");
        let mapped_vals: Vec<u8> = mapped.0.iter().map(|q| q.0).collect();
        assert_eq!(mapped_vals, vec![1, 5, 9]);
    }

    #[test]
    fn test_build_drb_radio_bearer_config_empty_qfis_omits_mapping() {
        let rbc = build_drb_radio_bearer_config(1, 1, &[], false, DrbIntegrityProtection::Disabled);
        let drb = &rbc.drb_to_add_mod_list.as_ref().unwrap().0[0];
        let sdap = match drb.cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert!(sdap.mapped_qo_s_flows_to_add.is_none());
    }

    #[test]
    fn test_radio_bearer_config_uper_roundtrip() {
        let qfis = [1u8, 2, 3];
        let rbc =
            build_drb_radio_bearer_config(4, 2, &qfis, true, DrbIntegrityProtection::Disabled);
        let bytes = encode_rrc(&rbc).expect("encode RadioBearerConfig");
        let decoded: RadioBearerConfig = decode_rrc(&bytes).expect("decode RadioBearerConfig");
        assert_eq!(decoded, rbc);

        // Confirm the decoded SDAP carries the same QFIs.
        let drb = &decoded.drb_to_add_mod_list.unwrap().0[0];
        let sdap = match drb.cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .unwrap()
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1, 2, 3]);
    }

    #[test]
    fn test_cell_group_config_uper_roundtrip() {
        let cgc = build_cell_group_config(2, 4);
        let bytes = encode_rrc(&cgc).expect("encode CellGroupConfig");
        let decoded: CellGroupConfig = decode_rrc(&bytes).expect("decode CellGroupConfig");
        assert_eq!(decoded, cgc);

        let rlc_list = decoded
            .rlc_bearer_to_add_mod_list
            .expect("RLC bearer list present");
        assert_eq!(rlc_list.0.len(), 1);
        assert_eq!(rlc_list.0[0].logical_channel_identity.0, 4);
        match rlc_list.0[0].served_radio_bearer.as_ref().unwrap() {
            RLC_BearerConfigServedRadioBearer::Drb_Identity(d) => assert_eq!(d.0, 2),
            _ => panic!("expected served DRB identity"),
        }
    }

    // ========================================================================
    // Wave-6 C5 — hand-derived golden byte vectors (TS 38.331 §6.2.1/§5.3.5,
    // UPER per X.691). Derived BY HAND from tools/rrc-19.3.0.asn1, NOT produced
    // by the encoder — the reviewer re-derives every bit below.
    // ========================================================================

    /// Minimal RRCReconfiguration on DL-DCCH, tid 0, all IEs absent, 13 bits:
    ///
    /// ```text
    /// DL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, rrcReconfiguration = index 0        0000
    /// RRCReconfiguration ::= SEQUENCE { tid, criticalExtensions } (no ext)
    ///   rrc-TransactionIdentifier — 2 bits, 0                               00
    ///   criticalExtensions CHOICE {rrcReconfiguration, future} — 1 bit       0
    /// RRCReconfiguration-IEs ::= SEQUENCE { 5 OPTIONAL fields } (no ext)
    ///   presence (radioBearer|secondaryCG|measConfig|lateNC|nonCrit)     00000
    /// = 0 0000 00 0 00000  (13 bits) + 3 pad = 0000 0000 | 0000 0000 -> 0x00 0x00
    /// ```
    const GOLDEN_RECONFIG_MINIMAL_TID0: [u8; 2] = [0x00, 0x00];

    /// RRCReconfigurationComplete on UL-DCCH, tid 0, 10 bits:
    ///
    /// ```text
    /// UL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                       0
    ///   c1 CHOICE (16 alts) — 4 bits, rrcReconfigurationComplete = index 1 0001
    /// RRCReconfigurationComplete ::= SEQUENCE { tid, critExt } (no ext)
    ///   rrc-TransactionIdentifier — 2 bits, 0                               00
    ///   criticalExtensions CHOICE {rrcReconfigurationComplete, future} 1 bit 0
    /// RRCReconfigurationComplete-IEs ::= SEQUENCE { 2 OPTIONAL fields }(no ext)
    ///   presence (lateNonCritical|nonCritical)                              00
    /// = 0 0001 00 0 00  (10 bits) + 6 pad = 0000 1000 | 0000 0000 -> 0x08 0x00
    /// ```
    const GOLDEN_RECONFIG_COMPLETE_TID0: [u8; 2] = [0x08, 0x00];

    #[test]
    fn golden_rrc_reconfiguration_minimal_bytes() {
        let bytes = encode_rrc_reconfiguration(&RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
            master_key_update: None,
            meas_config: None,
            ntn_config: None,
            sl_config: None,
        })
        .expect("encode minimal RRCReconfiguration");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_MINIMAL_TID0.to_vec(),
            "minimal RRCReconfiguration(tid 0) must match the hand-derived UPER bytes"
        );
    }

    /// The golden bytes above are the `measConfig`-ABSENT encoding, and the
    /// presence bit that says so is the third of the five in
    /// `RRCReconfiguration-IEs`. So adding a `measConfig` must change the bytes —
    /// if it did not, `optional_idx = 2` would not be being written and the field
    /// would be silently dropped on the wire (issue #170).
    ///
    /// A POSITIVE assertion on the difference rather than on a hand-derived vector:
    /// the message now embeds a whole `MeasConfig`, and the decoder is the
    /// authority on that layout (the byte round trip in `meas_config.rs` pins it).
    #[test]
    fn a_meas_config_changes_the_reconfiguration_bytes_and_is_read_back() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 0,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
            master_key_update: None,
            meas_config: Some(test_meas_config_params()),
            ntn_config: None,
            sl_config: None,
        };
        let bytes = encode_rrc_reconfiguration(&params).expect("encode with a measConfig");
        assert_ne!(
            bytes,
            GOLDEN_RECONFIG_MINIMAL_TID0.to_vec(),
            "a measConfig must reach the wire"
        );

        // And it survives: the margin the gNB configured is the margin decoded.
        let data = decode_rrc_reconfiguration(&bytes).expect("decode");
        let signalled = data.meas_config.expect("the measConfig must be carried");
        let read = crate::procedures::meas_config::read_a3_meas_configs(&signalled);
        assert_eq!(read.len(), 1, "one A3 binding");
        assert_eq!(read[0].meas_id, 1);
        assert_eq!(read[0].a3_offset_db, 3, "3 dB, in whole dB");
        assert_eq!(read[0].hysteresis_half_db, 2, "1 dB, in 0.5 dB units");
    }

    /// The whole message — `measConfig` and all — byte round trips.
    ///
    /// Load-bearing in the way #117's session learnt: an open-type framing bug
    /// round-tripped to PLAUSIBLE garbage that both `is_ok()` and a structure
    /// comparison accepted. Here the risk is the same in shape: `measConfig` is a
    /// deeply nested OPTIONAL whose presence bit sits among four others, and a
    /// decoder that consumed the wrong number of bits would mis-read whatever
    /// followed. Nothing follows it in this vector, so the DRB one below carries a
    /// `masterCellGroup` after it.
    #[test]
    fn a_reconfiguration_carrying_a_meas_config_byte_round_trips() {
        for (label, params) in [
            (
                "measConfig alone",
                RrcReconfigurationParams {
                    rrc_transaction_id: 0,
                    radio_bearer_config: None,
                    secondary_cell_group: None,
                    master_cell_group: None,
                    full_config: false,
                    master_key_update: None,
                    meas_config: Some(test_meas_config_params()),
                    ntn_config: None,
                    sl_config: None,
                },
            ),
            (
                // A `masterCellGroup` AFTER the measConfig: if the decoder stopped
                // at the wrong bit, this is the field that would come back wrong.
                "measConfig then masterCellGroup",
                RrcReconfigurationParams {
                    rrc_transaction_id: 3,
                    radio_bearer_config: None,
                    secondary_cell_group: None,
                    master_cell_group: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
                    full_config: true,
                    master_key_update: None,
                    meas_config: Some(test_meas_config_params()),
                    ntn_config: None,
                    sl_config: None,
                },
            ),
        ] {
            let bytes = encode_rrc_reconfiguration(&params).expect("encode");
            let msg: DL_DCCH_Message = decode_rrc(&bytes).expect("decode");
            assert_eq!(
                encode_rrc(&msg).expect("re-encode"),
                bytes,
                "{label} must byte round trip"
            );
            // And the field after it is intact, which byte identity alone would not
            // distinguish from a symmetric bug.
            let data = decode_rrc_reconfiguration(&bytes).expect("parse");
            assert_eq!(
                data.master_cell_group, params.master_cell_group,
                "{label}: the masterCellGroup after the measConfig"
            );
            assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        }
    }

    /// A margin TS 38.331 cannot carry fails the BUILD rather than encoding
    /// something out of constraint. Named so the gNB's "decline the measConfig,
    /// send the rest" behaviour has something to rest on.
    #[test]
    fn an_unsignallable_meas_config_fails_the_build() {
        let result = encode_rrc_reconfiguration(&RrcReconfigurationParams {
            meas_config: Some(A3MeasConfigParams {
                a3_offset_db: 99.0,
                ..test_meas_config_params()
            }),
            ..create_test_reconfiguration_params()
        });
        assert!(matches!(
            result,
            Err(RrcReconfigurationError::MeasConfig(_))
        ));
    }

    #[test]
    fn golden_rrc_reconfiguration_complete_bytes() {
        let bytes = encode_rrc_reconfiguration_complete(&RrcReconfigurationCompleteParams {
            rrc_transaction_id: 0,
        })
        .expect("encode RRCReconfigurationComplete");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_COMPLETE_TID0.to_vec(),
            "RRCReconfigurationComplete(tid 0) must match the hand-derived UPER bytes"
        );
    }

    #[test]
    fn test_build_drb_reconfiguration_params_full_message_roundtrip() {
        let qfis = [1u8, 6, 7];
        let params = build_drb_reconfiguration_params(
            1,
            5,
            1,
            4,
            &qfis,
            true,
            DrbIntegrityProtection::Disabled,
            // No measConfig: this test is about the DRB half, and the golden
            // byte vectors below are the no-measConfig encoding.
            None,
        )
        .expect("build structured reconfig params");

        // Build the full RRCReconfiguration and round-trip through UPER.
        let encoded = encode_rrc_reconfiguration(&params).expect("encode RRCReconfiguration");
        let data = decode_rrc_reconfiguration(&encoded).expect("decode RRCReconfiguration");

        // The carried radio_bearer_config decodes back to the structured DRB.
        let rbc_bytes = data
            .radio_bearer_config
            .expect("radio_bearer_config present after roundtrip");
        let rbc: RadioBearerConfig = decode_rrc(&rbc_bytes).expect("decode RBC");
        let drb_list = rbc.drb_to_add_mod_list.expect("DRB list");
        assert_eq!(drb_list.0.len(), 1);
        let sdap = match drb_list.0[0].cn_association.as_ref().unwrap() {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert_eq!(sdap.pdu_session.0, 5);
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .unwrap()
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1, 6, 7]);

        // The masterCellGroup octet string decodes back to the structured cell group.
        let mcg_bytes = data.master_cell_group.expect("master_cell_group present");
        let cgc: CellGroupConfig = decode_rrc(&mcg_bytes).expect("decode CGC");
        assert_eq!(
            cgc.rlc_bearer_to_add_mod_list.unwrap().0[0]
                .logical_channel_identity
                .0,
            4
        );
    }

    // ========================================================================
    // Wave-6 H6 — hand-derived golden UPER byte vectors for the LIVE DRB
    // RRCReconfiguration path (TS 38.331 §5.3.5 / §6.2.2 / §6.3.2, UPER per
    // ITU-T X.691). Authored with the H5 dual-derivation method
    // (`.context/GOLDEN-VECTOR-METHOD.md`):
    //
    //   * Derivation A = the per-byte bit tables in the doc comments below,
    //     hand-derived from `tools/rrc-19.3.0.asn1`.
    //   * Derivation B = the independent `bderive_*` UPER recompute below — a
    //     clean-room bit writer built directly from X.691 + the ASN.1, NOT the
    //     production `encode_rrc`. `golden_h6_derivation_b_matches_frozen`
    //     asserts B reproduces every frozen array (catches a symmetric
    //     encoder/decoder bug the round-trip tests cannot see).
    //   * The production encoder is the THIRD, independent check (tier-1
    //     `golden_*_bytes` tests).
    //
    // Exact production shape: `establish_drb` (nextgsim-gnb ngap/task.rs) for a
    // PDU session with id 1 uses drb_id = psi.clamp(1,32) = 1, lcid =
    // (3+drb_id).min(32) = 4, tid pinned 0, one accepted QFI. It calls
    // `build_drb_reconfiguration_params(0, 1, 1, 4, &[qfi], true)`, which
    // `encode_rrc`s the RadioBearerConfig and CellGroupConfig standalone (those
    // two standalone encodings are the `params.radio_bearer_config` and
    // masterCellGroup octet-string contents) before encoding the whole message.
    // ========================================================================

    /// Independent UNALIGNED-PER bit writer for derivation B (H5 method).
    /// Deliberately NOT the generated encoder — it emits the bitstream directly
    /// from the X.691 rules so that agreeing with `encode_rrc` proves the
    /// golden bytes are spec-derived, not a self-consistent codec artefact.
    struct BitW {
        bits: Vec<u8>,
    }
    impl BitW {
        fn new() -> Self {
            Self { bits: Vec::new() }
        }
        /// One raw bit.
        fn bit(&mut self, b: u8) {
            self.bits.push(b & 1);
        }
        /// Non-negative binary integer into `n` bits, MSB first (X.691 3.7.19).
        fn nbits(&mut self, value: u64, n: u32) {
            for i in (0..n).rev() {
                self.bits.push(((value >> i) & 1) as u8);
            }
        }
        /// X.691 §13.2 constrained whole number, UNALIGNED variant: a
        /// minimal-width bit field of `ceil(log2(range))` bits, never aligned;
        /// a range of 1 emits nothing (§13.2.1).
        fn cint(&mut self, value: u64, lo: u64, hi: u64) {
            let range = hi - lo + 1;
            if range == 1 {
                return;
            }
            let width = u64::BITS - (range - 1).leading_zeros();
            self.nbits(value - lo, width);
        }
        /// Unconstrained OCTET STRING, UNALIGNED (X.691 §11.9 + §17): a general
        /// length determinant (single octet `0nnnnnnn` for len < 128, no
        /// alignment) then the octets placed directly bit-by-bit.
        fn octet_string(&mut self, data: &[u8]) {
            assert!(data.len() < 128, "vectors use the single-octet length form");
            self.nbits(data.len() as u64, 8);
            for &b in data {
                self.nbits(u64::from(b), 8);
            }
        }
        /// Pad the trailing partial octet with zero bits (final message only).
        fn bytes(&self) -> Vec<u8> {
            let mut out = Vec::new();
            let mut i = 0;
            while i < self.bits.len() {
                let mut v = 0u8;
                for j in 0..8 {
                    v <<= 1;
                    if i + j < self.bits.len() {
                        v |= self.bits[i + j];
                    }
                }
                out.push(v);
                i += 8;
            }
            out
        }
    }

    /// Derivation B — RadioBearerConfig carrying one DRB (SDAP), inline body.
    fn bderive_drb_radio_bearer_config(w: &mut BitW, psi: u64, drb_id: u64, qfis: &[u64]) {
        // RadioBearerConfig ::= SEQUENCE {5 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b0_0100, 5); // presence: only drb-ToAddModList
                              // DRB-ToAddModList ::= SEQUENCE (SIZE(1..maxDRB=29)) OF
        w.cint(1, 1, 29); // one element
                          // DRB-ToAddMod ::= SEQUENCE {4 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b1001, 4); // presence: cnAssociation + pdcp-Config
                            // cnAssociation CHOICE {eps-BearerIdentity, sdap-Config} (2 alts)
        w.cint(1, 0, 1); // sdap-Config = index 1
                         // SDAP-Config ::= SEQUENCE {2 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b10, 2); // presence: only mappedQoS-FlowsToAdd
        w.cint(psi, 0, 255); // pdu-Session (PDU-SessionID 0..255)
        w.cint(0, 0, 1); // sdap-HeaderDL: present = index 0
        w.cint(0, 0, 1); // sdap-HeaderUL: present = index 0
        w.bit(1); // defaultDRB = TRUE (BOOLEAN)
                  // mappedQoS-FlowsToAdd ::= SEQUENCE (SIZE(1..maxNrofQFIs=64)) OF QFI
        w.cint(qfis.len() as u64, 1, 64);
        for &q in qfis {
            w.cint(q, 0, 63); // QFI (0..maxQFI=63)
        }
        w.cint(drb_id, 1, 32); // drb-Identity (mandatory, after the OPTIONAL cnAssociation)
                               // pdcp-Config ::= SEQUENCE {3 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b000, 3); // presence: drb / moreThanOneRLC / t-Reordering all absent
    }

    /// Derivation B — CellGroupConfig with one RLC bearer, inline body.
    fn bderive_cell_group_config(w: &mut BitW, lcid: u64, served_is_drb: bool, served_id: u64) {
        // CellGroupConfig ::= SEQUENCE {7 OPTIONAL, ..., ext-group} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b100_0000, 7); // presence: only rlc-BearerToAddModList
        w.cint(0, 0, 3); // cellGroupId (0..maxSecondaryCellGroups=3)
                         // rlc-BearerToAddModList ::= SEQUENCE (SIZE(1..maxLC-ID=32)) OF
        w.cint(1, 1, 32); // one element
                          // RLC-BearerConfig ::= SEQUENCE {4 OPTIONAL, ...} (extensible)
        w.bit(0); // extension bit
        w.nbits(0b1000, 4); // presence: only servedRadioBearer
        w.cint(lcid, 1, 32); // logicalChannelIdentity (1..32)
                             // servedRadioBearer CHOICE {srb-Identity, drb-Identity} (2 alts)
        w.cint(u64::from(served_is_drb), 0, 1);
        if served_is_drb {
            w.cint(served_id, 1, 32); // DRB-Identity (1..32)
        } else {
            w.cint(served_id, 1, 3); // SRB-Identity (1..3)
        }
    }

    fn bderive_drb_reconfiguration(
        tid: u64,
        psi: u64,
        drb_id: u64,
        lcid: u64,
        qfis: &[u64],
    ) -> Vec<u8> {
        let mut w = BitW::new();
        // DL-DCCH-MessageType CHOICE {c1, messageClassExtension} (2 alts)
        w.cint(0, 0, 1); // c1 = index 0
                         // c1 CHOICE (16 alternatives, spare-padded, not extensible)
        w.cint(0, 0, 15); // rrcReconfiguration = index 0
                          // RRCReconfiguration ::= SEQUENCE {tid, criticalExtensions}
        w.cint(tid, 0, 3); // rrc-TransactionIdentifier (0..3)
                           // criticalExtensions CHOICE {rrcReconfiguration, future} (2 alts)
        w.cint(0, 0, 1); // rrcReconfiguration = index 0
                         // RRCReconfiguration-IEs ::= SEQUENCE {5 OPTIONAL} (NOT extensible)
        w.nbits(0b1_0001, 5); // radioBearerConfig + nonCriticalExtension present
        bderive_drb_radio_bearer_config(&mut w, psi, drb_id, qfis);
        // nonCriticalExtension = RRCReconfiguration-v1530-IEs ::= SEQUENCE {8 OPTIONAL}
        w.nbits(0b1000_0000, 8); // only masterCellGroup present
                                 // masterCellGroup ::= OCTET STRING (CONTAINING CellGroupConfig)
        let mut cg = BitW::new();
        bderive_cell_group_config(&mut cg, lcid, true, drb_id);
        w.octet_string(&cg.bytes());
        w.bytes()
    }

    fn bderive_srb1_radio_bearer_config() -> Vec<u8> {
        let mut w = BitW::new();
        // RadioBearerConfig ::= SEQUENCE {5 OPTIONAL, ...} (extensible)
        w.bit(0);
        w.nbits(0b1_0000, 5); // only srb-ToAddModList
                              // SRB-ToAddModList ::= SEQUENCE (SIZE(1..2)) OF
        w.cint(1, 1, 2); // one element
                         // SRB-ToAddMod ::= SEQUENCE {3 OPTIONAL, ...} (extensible)
        w.bit(0);
        w.nbits(0b000, 3);
        w.cint(1, 1, 3); // srb-Identity = 1
        w.bytes()
    }

    /// DRB RadioBearerConfig standalone (the live `params.radio_bearer_config`),
    /// UPER, 52 bits (psi=1, drb_id=1, one QFI=1):
    ///
    /// ```text
    /// RadioBearerConfig ::= SEQUENCE {5 OPT, ...}   ext=0, presence 0 0100
    /// DRB-ToAddModList SIZE(1..29) — 5-bit len, count 1 -> 0 0000
    /// DRB-ToAddMod ::= SEQUENCE {4 OPT, ...}        ext=0, presence 1 0 0 1
    ///   cnAssociation CHOICE {eps,sdap} — 1 bit, sdap = idx 1        1
    ///   SDAP-Config ::= SEQUENCE {2 OPT, ...}       ext=0, presence 1 0
    ///     pdu-Session INTEGER(0..255) — 8 bits, 1        0000 0001
    ///     sdap-HeaderDL {present,absent} — 1 bit, present            0
    ///     sdap-HeaderUL {present,absent} — 1 bit, present            0
    ///     defaultDRB BOOLEAN — 1 bit, TRUE                           1
    ///     mappedQoS-FlowsToAdd SIZE(1..64) — 6-bit len, count 1 000000
    ///       QFI INTEGER(0..63) — 6 bits, value 1              000001
    ///   drb-Identity INTEGER(1..32) — 5 bits, value 1 -> off 0  0 0000
    ///   PDCP-Config ::= SEQUENCE {3 OPT, ...}       ext=0, presence 000
    /// = 0001 0000 0000 1001 1010 0000 0001 0010 0000 0000 0010 0000
    ///   + 4 pad -> 10 09 A0 12 00 20 00
    /// ```
    const GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1: [u8; 7] = [0x10, 0x09, 0xA0, 0x12, 0x00, 0x20, 0x00];

    /// DRB CellGroupConfig standalone (the live masterCellGroup octet-string
    /// contents), UPER, 31 bits (drb_id=1, lcid=4):
    ///
    /// ```text
    /// CellGroupConfig ::= SEQUENCE {7 OPT, ...}   ext=0, presence 1 000000
    /// cellGroupId INTEGER(0..3) — 2 bits, 0                          0 0
    /// rlc-BearerToAddModList SIZE(1..32) — 5-bit len, count 1  0 0000
    /// RLC-BearerConfig ::= SEQUENCE {4 OPT, ...}  ext=0, presence 1 000
    ///   logicalChannelIdentity INTEGER(1..32) — 5 bits, 4 -> off 3 00011
    ///   servedRadioBearer CHOICE {srb,drb} — 1 bit, drb = idx 1        1
    ///     drb-Identity INTEGER(1..32) — 5 bits, 1 -> off 0        0 0000
    /// = 0100 0000 0000 0000 1000 0001 1100 000 + 1 pad -> 40 00 81 C0
    /// ```
    const GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1: [u8; 4] = [0x40, 0x00, 0x81, 0xC0];

    /// Full DL-DCCH RRCReconfiguration establishing one DRB, tid 0, 113 bits —
    /// the exact `establish_drb(psi=1)` shape (drb_id 1, lcid 4, QFI 1):
    ///
    /// ```text
    /// DL-DCCH-MessageType CHOICE {c1, mce} — 1 bit, c1                    0
    /// c1 CHOICE (16 alts) — 4 bits, rrcReconfiguration = idx 0        0000
    /// rrc-TransactionIdentifier INTEGER(0..3) — 2 bits, 0               0 0
    /// criticalExtensions CHOICE {rrcReconfiguration, future} — 1 bit      0
    /// RRCReconfiguration-IEs ::= SEQUENCE {5 OPT} (NOT ext)
    ///   presence (radioBearer|secondaryCG|meas|lateNC|nonCrit)     1 0001
    /// radioBearerConfig (inline) = the 52 GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1
    ///   bits, UNPADDED:              0 00100 00000 0 1001 1 0 10 00000001
    ///                                0 0 1 000000 000001 00000 0 000
    /// nonCriticalExtension = RRCReconfiguration-v1530-IEs {8 OPT} (NOT ext)
    ///   presence (mcg|full|nas|mku|sib1|sysinfo|other|nonCrit)  1 0000000
    /// masterCellGroup OCTET STRING (CONTAINING CellGroupConfig):
    ///   length determinant, 1 octet (<128), value 4          0000 0100
    ///   contents = GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1 (4 octets, unaligned)
    ///                          0100 0000 0000 0000 1000 0001 1100 0000
    /// = 113 bits + 7 pad ->
    ///   00 88 80 4D 00 90 01 00 40 02 20 00 40 E0 00
    /// ```
    const GOLDEN_RECONFIG_DRB_PSI1: [u8; 15] = [
        0x00, 0x88, 0x80, 0x4D, 0x00, 0x90, 0x01, 0x00, 0x40, 0x02, 0x20, 0x00, 0x40, 0xE0, 0x00,
    ];

    /// Derivation B reproduces every H6 frozen vector (independent of both the
    /// hand bit tables above AND the production `encode_rrc`).
    #[test]
    fn golden_h6_derivation_b_matches_frozen() {
        assert_eq!(
            bderive_srb1_radio_bearer_config(),
            [0x40, 0x00],
            "derivation B: SRB1 RadioBearerConfig"
        );
        assert_eq!(
            {
                let mut w = BitW::new();
                bderive_cell_group_config(&mut w, 4, true, 1);
                w.bytes()
            },
            GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1.to_vec(),
            "derivation B: DRB CellGroupConfig"
        );
        assert_eq!(
            {
                let mut w = BitW::new();
                bderive_drb_radio_bearer_config(&mut w, 1, 1, &[1]);
                w.bytes()
            },
            GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1.to_vec(),
            "derivation B: DRB RadioBearerConfig"
        );
        assert_eq!(
            bderive_drb_reconfiguration(0, 1, 1, 4, &[1]),
            GOLDEN_RECONFIG_DRB_PSI1.to_vec(),
            "derivation B: DRB RRCReconfiguration"
        );
    }

    /// Tier 1 — encoder golden: the production `encode_rrc` output for the
    /// standalone DRB RadioBearerConfig must equal the frozen bytes.
    #[test]
    fn golden_drb_radio_bearer_config_bytes() {
        let bytes = encode_rrc(&build_drb_radio_bearer_config(
            1,
            1,
            &[1],
            true,
            DrbIntegrityProtection::Disabled,
        ))
        .expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1.to_vec(),
            "DRB RadioBearerConfig must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden: the frozen DRB RadioBearerConfig bytes decode
    /// to exactly the builder's struct (guards decoder drift independently).
    #[test]
    fn golden_drb_radio_bearer_config_cross_decode() {
        let decoded: RadioBearerConfig =
            decode_rrc(&GOLDEN_DRB_RADIO_BEARER_CONFIG_PSI1).expect("decode RBC");
        assert_eq!(
            decoded,
            build_drb_radio_bearer_config(1, 1, &[1], true, DrbIntegrityProtection::Disabled)
        );
    }

    /// Tier 1 — encoder golden for the standalone DRB CellGroupConfig.
    #[test]
    fn golden_drb_cell_group_config_bytes() {
        let bytes = encode_rrc(&build_cell_group_config(1, 4)).expect("encode");
        assert_eq!(
            bytes,
            GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1.to_vec(),
            "DRB CellGroupConfig must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden for the standalone DRB CellGroupConfig.
    #[test]
    fn golden_drb_cell_group_config_cross_decode() {
        let decoded: CellGroupConfig =
            decode_rrc(&GOLDEN_DRB_CELL_GROUP_CONFIG_PSI1).expect("decode CGC");
        assert_eq!(decoded, build_cell_group_config(1, 4));
    }

    /// Tier 1 — encoder golden for the FULL live DRB RRCReconfiguration
    /// message (`establish_drb(psi=1)`): flipping any encoder bit fails here.
    #[test]
    fn golden_drb_rrc_reconfiguration_bytes() {
        let params = build_drb_reconfiguration_params(
            0,
            1,
            1,
            4,
            &[1],
            true,
            DrbIntegrityProtection::Disabled,
            // The frozen vector is the measConfig-ABSENT encoding, so this is
            // `None` deliberately: a measConfig would change every byte after the
            // third presence bit. `a_meas_config_changes_the_reconfiguration_bytes`
            // is what asserts the present form differs.
            None,
        )
        .expect("build DRB params");
        let bytes = encode_rrc_reconfiguration(&params).expect("encode RRCReconfiguration");
        assert_eq!(
            bytes,
            GOLDEN_RECONFIG_DRB_PSI1.to_vec(),
            "DRB RRCReconfiguration must match the hand-derived UPER bytes"
        );
    }

    /// Tier 2 — decoder golden for the full DRB RRCReconfiguration: the frozen
    /// bytes decode to the same DL-DCCH message the builder produces, and the
    /// SDAP mapping / masterCellGroup survive intact.
    #[test]
    fn golden_drb_rrc_reconfiguration_cross_decode() {
        let params = build_drb_reconfiguration_params(
            0,
            1,
            1,
            4,
            &[1],
            true,
            DrbIntegrityProtection::Disabled,
            None,
        )
        .expect("build DRB params");
        let expected = build_rrc_reconfiguration(&params).expect("build message");
        let decoded: DL_DCCH_Message =
            decode_rrc(&GOLDEN_RECONFIG_DRB_PSI1).expect("decode DL-DCCH");
        assert_eq!(decoded, expected);

        // Spec-shape assertions on the decoded message (TS 38.331 §5.3.5.6).
        let data = parse_rrc_reconfiguration(&decoded).expect("parse");
        assert_eq!(data.rrc_transaction_id, 0);
        let rbc: RadioBearerConfig =
            decode_rrc(&data.radio_bearer_config.expect("radioBearerConfig")).expect("RBC");
        let drb = &rbc.drb_to_add_mod_list.expect("DRB list").0[0];
        assert_eq!(drb.drb_identity.0, 1);
        let sdap = match drb.cn_association.as_ref().expect("cn-Association") {
            DRB_ToAddModCnAssociation::Sdap_Config(s) => s,
            _ => panic!("expected SDAP-Config"),
        };
        assert_eq!(sdap.pdu_session.0, 1);
        let mapped: Vec<u8> = sdap
            .mapped_qo_s_flows_to_add
            .as_ref()
            .expect("mappedQoS-FlowsToAdd")
            .0
            .iter()
            .map(|q| q.0)
            .collect();
        assert_eq!(mapped, vec![1]);
        let cgc: CellGroupConfig =
            decode_rrc(&data.master_cell_group.expect("masterCellGroup")).expect("CGC");
        let bearer = &cgc.rlc_bearer_to_add_mod_list.expect("RLC list").0[0];
        assert_eq!(bearer.logical_channel_identity.0, 4);
        match bearer
            .served_radio_bearer
            .as_ref()
            .expect("servedRadioBearer")
        {
            RLC_BearerConfigServedRadioBearer::Drb_Identity(d) => assert_eq!(d.0, 1),
            _ => panic!("expected served DRB identity"),
        }
    }

    // ========================================================================
    // Connected-mode NTN update (issue #56)
    // ========================================================================

    use crate::procedures::ntn_timing::{EphemerisStateVector, NtnServingCellConfig};

    /// The same 600 km overhead LEO geometry the SIB19 and timing tests use, so the
    /// number this reconfiguration carries is the number those modules
    /// hand-compute.
    fn test_ntn_sib19_params() -> Sib19Params {
        Sib19Params {
            ntn_config: NtnServingCellConfig {
                epoch_sfn: 512,
                epoch_subframe: 3,
                ul_sync_validity_index: 5,
                cell_specific_k_offset: 478,
                ta_common: 1_000_000,
                ta_common_drift: Some(-500),
                ephemeris: EphemerisStateVector::from_ecef(
                    [6_378_137.0 + 600_000.0, 0.0, 0.0],
                    [1000.0, 0.0, 0.0],
                )
                .expect("within the ASN.1 ranges"),
            },
            t_service: None,
            distance_thresh: None,
        }
    }

    /// Criterion 4: `ntn-Config` reaches a connected UE in an RRCReconfiguration,
    /// through real UPER, with the ephemeris intact.
    #[test]
    fn an_rrc_reconfiguration_carries_the_ntn_config_to_a_connected_ue() {
        let params = RrcReconfigurationParams {
            ntn_config: Some(test_ntn_sib19_params()),
            ..create_test_reconfiguration_params()
        };
        let bytes = encode_rrc_reconfiguration(&params).expect("encodes");
        let data = decode_rrc_reconfiguration(&bytes).expect("decodes");

        let sib19 = data
            .ntn_config
            .expect("the reconfiguration must carry the NTN update");
        let cfg = sib19
            .ntn_config
            .expect("and the delivered SIB19 must carry an ntn-Config");
        assert_eq!(
            cfg,
            test_ntn_sib19_params().ntn_config,
            "every NTN-Config field must survive the dedicated delivery: the UE \
             re-derives its timing advance from these numbers"
        );
        // The value is usable, not merely present: the same non-zero TA the
        // broadcast path yields.
        let ue = [6_378_137.0, 0.0, 0.0];
        assert!(
            cfg.autonomous_ta_us(ue) > 8000.0,
            "the TA derived from the connected-mode update must be the same non-zero \
             ~8075 us as the broadcast's, or the two paths disagree"
        );
    }

    /// An NTN update ALONE must still emit the v1530 extension. A reconfiguration
    /// carrying only an ephemeris refresh is exactly the connected-mode update of
    /// TS 38.300 §16.14.2.2, and the extension's emit condition used to depend only
    /// on `masterCellGroup`/`fullConfig`.
    #[test]
    fn a_reconfiguration_carrying_only_an_ntn_update_still_emits_the_extension() {
        let params = RrcReconfigurationParams {
            rrc_transaction_id: 1,
            radio_bearer_config: None,
            secondary_cell_group: None,
            master_cell_group: None,
            full_config: false,
            master_key_update: None,
            meas_config: None,
            ntn_config: Some(test_ntn_sib19_params()),
            sl_config: None,
        };
        let data = decode_rrc_reconfiguration(&encode_rrc_reconfiguration(&params).unwrap())
            .expect("decodes");
        assert!(
            data.master_cell_group.is_none() && !data.full_config,
            "nothing but the NTN update is set"
        );
        assert!(
            data.ntn_config.and_then(|s| s.ntn_config).is_some(),
            "an NTN-only reconfiguration must still carry the update; if the v1530 \
             extension is omitted the ephemeris is silently dropped"
        );
    }

    /// A reconfiguration with no NTN update must be byte-identical to what it was
    /// before this field existed: a terrestrial cell's signalling must not grow.
    #[test]
    fn a_reconfiguration_without_an_ntn_update_carries_no_dedicated_si() {
        let params = create_test_reconfiguration_params();
        let data = decode_rrc_reconfiguration(&encode_rrc_reconfiguration(&params).unwrap())
            .expect("decodes");
        assert!(
            data.ntn_config.is_none(),
            "a TN cell's reconfiguration must carry no dedicatedSystemInformationDelivery"
        );

        // And the encoded bytes are unchanged by the field's existence: the same
        // params with `ntn_config: None` must produce the same PDU as one built
        // before the field was added, which is what this equality pins.
        let explicit_none = RrcReconfigurationParams {
            ntn_config: None,
            sl_config: None,
            ..create_test_reconfiguration_params()
        };
        assert_eq!(
            encode_rrc_reconfiguration(&params).unwrap(),
            encode_rrc_reconfiguration(&explicit_none).unwrap()
        );
    }

    /// A `dedicatedSystemInformationDelivery` carrying some OTHER SIB is legal, and
    /// must not make the reconfiguration fail to parse — only report no NTN update.
    #[test]
    fn a_dedicated_delivery_of_another_sib_yields_no_ntn_update_rather_than_an_error() {
        use crate::procedures::system_information::{Sib3Params, SystemInformationParams};
        // A SystemInformation carrying SIB3 instead of SIB19, encoded the way the
        // contained type requires.
        let si = build_system_information(&SystemInformationParams {
            sib3: Some(Sib3Params::default()),
            ..Default::default()
        })
        .expect("a SIB3 SystemInformation builds");
        let BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(inner)) =
            &si.message
        else {
            panic!("expected a SystemInformation");
        };
        let payload = encode_rrc(inner).expect("encodes");
        assert!(
            read_ntn_dedicated_si(&payload).is_none(),
            "a delivery carrying SIB3 must report no NTN update"
        );

        // And garbage must not panic or error either: the bearer half of a
        // reconfiguration has to keep applying.
        assert!(read_ntn_dedicated_si(&[0xff, 0xff, 0xff, 0xff]).is_none());
        assert!(read_ntn_dedicated_si(&[]).is_none());
    }

    /// An unencodable ephemeris must fail the BUILD rather than emit a
    /// reconfiguration whose NTN update is quietly absent.
    #[test]
    fn an_unencodable_ntn_config_fails_the_reconfiguration_build() {
        let bad = Sib19Params {
            ntn_config: NtnServingCellConfig {
                cell_specific_k_offset: 0, // INTEGER(1..1023)
                ..test_ntn_sib19_params().ntn_config
            },
            ..test_ntn_sib19_params()
        };
        let err = encode_rrc_reconfiguration(&RrcReconfigurationParams {
            ntn_config: Some(bad),
            ..create_test_reconfiguration_params()
        })
        .expect_err("an unencodable NTN config must be refused");
        assert!(
            matches!(err, RrcReconfigurationError::DedicatedSystemInformation(_)),
            "and refused as a dedicated-SI failure, so a caller can still choose to \
             send the bearer half: got {err:?}"
        );
    }

    /// The dedicated delivery and the broadcast must carry the SAME bytes for the
    /// same configuration. If they diverged, a UE would derive one timing advance
    /// from BCCH and a different one from SRB1.
    #[test]
    fn the_dedicated_delivery_and_the_broadcast_carry_identical_sib19_content() {
        use crate::procedures::system_information::decode_system_information;
        let params = test_ntn_sib19_params();

        let dedicated = build_ntn_dedicated_si(&params).expect("dedicated builds");
        let from_dedicated = read_ntn_dedicated_si(&dedicated).expect("and reads back");

        let broadcast = crate::procedures::system_information::encode_system_information(
            &SystemInformationParams {
                sib19: Some(params),
                ..Default::default()
            },
        )
        .expect("broadcast encodes");
        let from_broadcast = decode_system_information(&broadcast)
            .expect("decodes")
            .sib19
            .expect("carries SIB19");

        assert_eq!(
            from_dedicated, from_broadcast,
            "the connected-mode update and the broadcast must deliver the same \
             configuration; a UE reading both must not get two different answers"
        );
    }
}

// ============================================================================
// Handover command: RRCReconfiguration with reconfigurationWithSync
// (TS 38.331 §5.3.5.5.2, §6.3.2 — issue #107, criterion 3)
// ============================================================================

/// `t304` in milliseconds, and the ASN.1 enumeration index carrying it
/// (TS 38.331 §6.3.2 `ReconfigurationWithSync.t304`).
///
/// A table rather than arithmetic: the values are not a uniform step (50, 100,
/// 150, 200, 500, 1000, 2000, 10000), so any computed mapping is wrong above 200.
const T304_MS_TO_INDEX: [(u16, u8); 8] = [
    (50, 0),
    (100, 1),
    (150, 2),
    (200, 3),
    (500, 4),
    (1000, 5),
    (2000, 6),
    (10000, 7),
];

/// Maps a `t304` in milliseconds onto its enumeration index, refusing a value the
/// enumeration does not contain.
///
/// Refused rather than rounded to the nearest legal value: `t304` is the timer
/// that decides when the UE declares handover failure, and a silently adjusted
/// one makes a handover-failure test fire at a time nobody configured.
pub fn t304_index(ms: u16) -> Result<u8, RrcReconfigurationError> {
    T304_MS_TO_INDEX
        .iter()
        .find(|(value, _)| *value == ms)
        .map(|(_, index)| *index)
        .ok_or_else(|| {
            RrcReconfigurationError::InvalidFieldValue(format!(
                "t304 {ms} ms is not one of the values TS 38.331 enumerates \
                 (50, 100, 150, 200, 500, 1000, 2000, 10000)"
            ))
        })
}

/// The inverse of [`t304_index`], in milliseconds.
pub fn t304_ms(index: u8) -> Result<u16, RrcReconfigurationError> {
    T304_MS_TO_INDEX
        .iter()
        .find(|(_, value)| *value == index)
        .map(|(ms, _)| *ms)
        .ok_or_else(|| {
            RrcReconfigurationError::InvalidFieldValue(format!(
                "t304 index {index} is out of range"
            ))
        })
}

/// `ss-PBCH-BlockPower` broadcast in a `ServingCellConfigCommon`.
///
/// A PHY parameter this simulator does not model (there is no PBCH). -20 dBm is
/// inside the `INTEGER (-60..50)` range and unremarkable; documented here rather
/// than made a knob nothing reads, the same treatment `mib_params` gives the
/// MIB's physical-layer fields.
const SS_PBCH_BLOCK_POWER_DBM: i8 = -20;

/// Parameters for a handover command (issue #107, criterion 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandoverCommandParams {
    /// RRC-TransactionIdentifier (0..3)
    pub rrc_transaction_id: u8,
    /// `physCellId` of the target cell (0..1007). This is how the target is
    /// identified on the wire — the simulator's own cell index has no IE, so the
    /// receiving UE resolves the PCI against the cells it can hear.
    pub target_phys_cell_id: u16,
    /// `newUE-Identity`: the C-RNTI the UE is to use in the target cell.
    pub new_ue_identity: u16,
    /// `t304` in milliseconds; must be one of the enumerated values.
    pub t304_ms: u16,
    /// `fullConfig`: the UE releases its stored configuration and applies this one
    /// whole.
    pub full_config: bool,
    /// `masterKeyUpdate`: tells the UE to re-derive `KgNB*` for the target
    /// (issue #39). `None` leaves the UE on its current keys, which is what this
    /// simulator did before — and what made a handover lose forward security silently.
    pub master_key_update: Option<MasterKeyUpdateParams>,
}

/// What a handover command carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HandoverCommandData {
    /// RRC-TransactionIdentifier
    pub rrc_transaction_id: u8,
    /// `physCellId` of the target cell
    pub target_phys_cell_id: u16,
    /// `newUE-Identity` (C-RNTI in the target)
    pub new_ue_identity: u16,
    /// `t304` in milliseconds
    pub t304_ms: u16,
    /// Whether `fullConfig` was set
    pub full_config: bool,
    /// `masterKeyUpdate`, when the command carried one.
    pub master_key_update: Option<MasterKeyUpdateParams>,
}

/// Builds the `CellGroupConfig` carrying a `reconfigurationWithSync` for a
/// handover to `target_phys_cell_id` (TS 38.331 §6.3.2).
///
/// `reconfigurationWithSync` is what makes an `RRCReconfiguration` a **handover
/// command** rather than a plain reconfiguration: TS 38.331 §5.3.5.5.2 has the UE
/// perform the synchronous reconfiguration — reset MAC, re-establish RLC/PDCP and
/// access the target cell — only when this IE is present.
pub fn build_handover_cell_group_config(
    target_phys_cell_id: u16,
    new_ue_identity: u16,
    t304_ms: u16,
) -> Result<CellGroupConfig, RrcReconfigurationError> {
    if target_phys_cell_id > 1007 {
        return Err(RrcReconfigurationError::InvalidFieldValue(format!(
            "physCellId is INTEGER (0..1007), got {target_phys_cell_id}"
        )));
    }

    let sp_cell_config_common = ServingCellConfigCommon {
        phys_cell_id: Some(PhysCellId(target_phys_cell_id)),
        // The target's frequency configuration is deliberately absent. This
        // simulator's RLS presents one carrier, so a target cell is always on the
        // UE's current frequency and `downlinkConfigCommon` would carry a
        // `frequencyInfoDL` the UE already has. Absent is honest; a fabricated
        // ARFCN would be a value the UE might act on.
        downlink_config_common: None,
        uplink_config_common: None,
        supplementary_uplink_config: None,
        n_timing_advance_offset: None,
        ssb_positions_in_burst: None,
        ssb_periodicity_serving_cell: None,
        // Mandatory PHY fields with no counterpart in a simulator with no PHY,
        // fixed at the values a 30 kHz FR1 cell would use — the same treatment
        // `nextgsim-gnb`'s `mib_params` gives the MIB's physical-layer fields, and
        // documented for the same reason.
        dmrs_type_a_position: ServingCellConfigCommonDmrs_TypeA_Position(
            ServingCellConfigCommonDmrs_TypeA_Position::POS2,
        ),
        lte_crs_to_match_around: None,
        rate_match_pattern_to_add_mod_list: None,
        rate_match_pattern_to_release_list: None,
        ssb_subcarrier_spacing: None,
        tdd_ul_dl_configuration_common: None,
        ss_pbch_block_power: ServingCellConfigCommonSs_PBCH_BlockPower(SS_PBCH_BLOCK_POWER_DBM),
    };

    let sp_cell_config = SpCellConfig {
        serv_cell_index: None,
        reconfiguration_with_sync: Some(ReconfigurationWithSync {
            sp_cell_config_common: Some(sp_cell_config_common),
            new_ue_identity: RNTI_Value(new_ue_identity),
            t304: ReconfigurationWithSyncT304(t304_index(t304_ms)?),
            // No dedicated RACH preamble: this simulator has no RACH, so a
            // contention-free preamble would name a resource that does not exist.
            rach_config_dedicated: None,
        }),
        rlf_timers_and_constants: None,
        rlm_in_sync_out_of_sync_threshold: None,
        sp_cell_config_dedicated: None,
    };

    Ok(CellGroupConfig {
        cell_group_id: CellGroupId(0),
        rlc_bearer_to_add_mod_list: None,
        rlc_bearer_to_release_list: None,
        mac_cell_group_config: None,
        physical_cell_group_config: None,
        sp_cell_config: Some(sp_cell_config),
        s_cell_to_add_mod_list: None,
        s_cell_to_release_list: None,
    })
}

/// Builds the `RrcReconfigurationParams` for a handover command.
pub fn build_handover_command_params(
    params: &HandoverCommandParams,
) -> Result<RrcReconfigurationParams, RrcReconfigurationError> {
    let cgc = build_handover_cell_group_config(
        params.target_phys_cell_id,
        params.new_ue_identity,
        params.t304_ms,
    )?;
    Ok(RrcReconfigurationParams {
        rrc_transaction_id: params.rrc_transaction_id,
        radio_bearer_config: None,
        secondary_cell_group: None,
        master_cell_group: Some(encode_rrc(&cgc)?),
        full_config: params.full_config,
        master_key_update: params.master_key_update,
        // A handover command carries no `measConfig` here (issue #170). The target
        // cell's measurement configuration is what a conformant one would carry,
        // and this gNB has one carrier: the frequency the UE already measures is
        // the frequency it would be told to measure, so signalling it again would
        // re-send the same configuration with a `reconfigurationWithSync` beside
        // it. If this ever becomes multi-carrier, this is the site.
        meas_config: None,
        // No NTN update on a handover command (issue #56). TS 38.331's `EpochTime`
        // description is explicit that on handover the ephemeris reference point and
        // the SFN are the TARGET cell's, so signalling this cell's `ntn-Config` in a
        // handover command would hand the UE a configuration referenced to the wrong
        // clock -- worse than sending none, because the UE would apply it. The
        // target's own SIB19 broadcast is what the UE reads after the switch.
        ntn_config: None,
        // No sidelink grant on a handover command either (issue #141), for a related
        // reason: sidelink resources are the SOURCE cell's, so carrying them into a
        // handover would grant the UE carriers the target has not allocated. A UE that
        // needs sidelink after the switch re-sends its `SidelinkUEInformation` to the
        // target, which is what TS 38.331 §5.8.3.2 requires on entering a new cell.
        sl_config: None,
    })
}

/// Builds and encodes a handover command to UPER bytes.
pub fn encode_handover_command(
    params: &HandoverCommandParams,
) -> Result<Vec<u8>, RrcReconfigurationError> {
    encode_rrc_reconfiguration(&build_handover_command_params(params)?)
}

/// Decodes a handover command, i.e. an `RRCReconfiguration` whose
/// `masterCellGroup` carries a `reconfigurationWithSync`.
///
/// Returns `None` for an `RRCReconfiguration` that is **not** a handover command —
/// one with no `masterCellGroup`, or one whose cell group has no
/// `reconfigurationWithSync`. That is not an error: a DRB-establishing
/// reconfiguration is a perfectly good message that simply is not a handover, and
/// a receiver has to be able to tell them apart.
pub fn decode_handover_command(bytes: &[u8]) -> Option<HandoverCommandData> {
    let data = decode_rrc_reconfiguration(bytes).ok()?;
    let master_cell_group = data.master_cell_group.as_ref()?;
    let cgc: CellGroupConfig = decode_rrc(master_cell_group).ok()?;
    let sync = cgc.sp_cell_config?.reconfiguration_with_sync?;
    let target_phys_cell_id = sync.sp_cell_config_common?.phys_cell_id?.0;
    Some(HandoverCommandData {
        rrc_transaction_id: data.rrc_transaction_id,
        target_phys_cell_id,
        new_ue_identity: sync.new_ue_identity.0,
        // A t304 index outside the enumeration cannot occur from a successful
        // decode (the ASN.1 type bounds it), so an unmappable one is treated as
        // "not a usable handover command" rather than silently defaulted to a
        // timer the network never set.
        t304_ms: t304_ms(sync.t304.0).ok()?,
        full_config: data.full_config,
        master_key_update: data.master_key_update,
    })
}
