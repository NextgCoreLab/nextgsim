//! System Information Procedures
//!
//! Implements MIB (Master Information Block) and SIB1 (System Information Block Type 1)
//! structures as defined in 3GPP TS 38.331.
//!
//! MIB is broadcast on BCCH-BCH and contains essential system information for initial access.
//! SIB1 is broadcast on BCCH-DL-SCH and contains cell access and selection information.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during System Information procedures
#[derive(Debug, Error)]
pub enum SystemInformationError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

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
}

// ============================================================================
// MIB (Master Information Block)
// ============================================================================

/// Subcarrier spacing for common control channels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubCarrierSpacingCommon {
    /// 15 kHz or 60 kHz (FR1 or FR2)
    Scs15Or60,
    /// 30 kHz or 120 kHz (FR1 or FR2)
    Scs30Or120,
}

/// DMRS Type-A position
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmrsTypeAPosition {
    /// Position 2
    Pos2,
    /// Position 3
    Pos3,
}

/// Cell barred status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellBarredStatus {
    /// Cell is barred
    Barred,
    /// Cell is not barred
    NotBarred,
}

/// Intra-frequency reselection status
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntraFreqReselection {
    /// Intra-frequency reselection allowed
    Allowed,
    /// Intra-frequency reselection not allowed
    NotAllowed,
}

/// PDCCH configuration for SIB1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PdcchConfigSib1Params {
    /// CORESET zero configuration (0-15)
    pub coreset_zero: u8,
    /// Search space zero configuration (0-15)
    pub search_space_zero: u8,
}

/// Parameters for building a MIB message
#[derive(Debug, Clone)]
pub struct MibParams {
    /// System Frame Number (6 bits, 0-63)
    pub system_frame_number: u8,
    /// Subcarrier spacing for common control channels
    pub sub_carrier_spacing_common: SubCarrierSpacingCommon,
    /// SSB subcarrier offset (0-15)
    pub ssb_subcarrier_offset: u8,
    /// DMRS Type-A position
    pub dmrs_type_a_position: DmrsTypeAPosition,
    /// PDCCH configuration for SIB1
    pub pdcch_config_sib1: PdcchConfigSib1Params,
    /// Cell barred status
    pub cell_barred: CellBarredStatus,
    /// Intra-frequency reselection status
    pub intra_freq_reselection: IntraFreqReselection,
}

/// Parsed MIB data
#[derive(Debug, Clone)]
pub struct MibData {
    /// System Frame Number (6 bits)
    pub system_frame_number: u8,
    /// Subcarrier spacing for common control channels
    pub sub_carrier_spacing_common: SubCarrierSpacingCommon,
    /// SSB subcarrier offset
    pub ssb_subcarrier_offset: u8,
    /// DMRS Type-A position
    pub dmrs_type_a_position: DmrsTypeAPosition,
    /// PDCCH configuration for SIB1
    pub pdcch_config_sib1: PdcchConfigSib1Params,
    /// Cell barred status
    pub cell_barred: CellBarredStatus,
    /// Intra-frequency reselection status
    pub intra_freq_reselection: IntraFreqReselection,
}

/// Build a MIB message
pub fn build_mib(params: &MibParams) -> Result<BCCH_BCH_Message, SystemInformationError> {
    // Validate system frame number (6 bits)
    if params.system_frame_number > 63 {
        return Err(SystemInformationError::InvalidFieldValue(
            "System Frame Number must be 0-63 (6 bits)".to_string(),
        ));
    }

    // Validate SSB subcarrier offset
    if params.ssb_subcarrier_offset > 15 {
        return Err(SystemInformationError::InvalidFieldValue(
            "SSB Subcarrier Offset must be 0-15".to_string(),
        ));
    }

    // Validate PDCCH config values
    if params.pdcch_config_sib1.coreset_zero > 15 {
        return Err(SystemInformationError::InvalidFieldValue(
            "CORESET Zero must be 0-15".to_string(),
        ));
    }
    if params.pdcch_config_sib1.search_space_zero > 15 {
        return Err(SystemInformationError::InvalidFieldValue(
            "Search Space Zero must be 0-15".to_string(),
        ));
    }

    // Build system frame number (6 bits)
    let mut sfn_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..6).rev() {
        sfn_bv.push((params.system_frame_number >> i) & 1 == 1);
    }

    // Build spare bit (1 bit)
    let mut spare_bv: BitVec<u8, Msb0> = BitVec::new();
    spare_bv.push(false);

    let mib = MIB {
        system_frame_number: MIBSystemFrameNumber(sfn_bv),
        sub_carrier_spacing_common: match params.sub_carrier_spacing_common {
            SubCarrierSpacingCommon::Scs15Or60 => {
                MIBSubCarrierSpacingCommon(MIBSubCarrierSpacingCommon::SCS15OR60)
            }
            SubCarrierSpacingCommon::Scs30Or120 => {
                MIBSubCarrierSpacingCommon(MIBSubCarrierSpacingCommon::SCS30OR120)
            }
        },
        ssb_subcarrier_offset: MIBSsb_SubcarrierOffset(params.ssb_subcarrier_offset),
        dmrs_type_a_position: match params.dmrs_type_a_position {
            DmrsTypeAPosition::Pos2 => MIBDmrs_TypeA_Position(MIBDmrs_TypeA_Position::POS2),
            DmrsTypeAPosition::Pos3 => MIBDmrs_TypeA_Position(MIBDmrs_TypeA_Position::POS3),
        },
        pdcch_config_sib1: PDCCH_ConfigSIB1 {
            control_resource_set_zero: ControlResourceSetZero(
                params.pdcch_config_sib1.coreset_zero,
            ),
            search_space_zero: SearchSpaceZero(params.pdcch_config_sib1.search_space_zero),
        },
        cell_barred: match params.cell_barred {
            CellBarredStatus::Barred => MIBCellBarred(MIBCellBarred::BARRED),
            CellBarredStatus::NotBarred => MIBCellBarred(MIBCellBarred::NOT_BARRED),
        },
        intra_freq_reselection: match params.intra_freq_reselection {
            IntraFreqReselection::Allowed => {
                MIBIntraFreqReselection(MIBIntraFreqReselection::ALLOWED)
            }
            IntraFreqReselection::NotAllowed => {
                MIBIntraFreqReselection(MIBIntraFreqReselection::NOT_ALLOWED)
            }
        },
        spare: MIBSpare(spare_bv),
    };

    Ok(BCCH_BCH_Message {
        message: BCCH_BCH_MessageType::Mib(mib),
    })
}

/// Parse a MIB from a BCCH-BCH message
pub fn parse_mib(msg: &BCCH_BCH_Message) -> Result<MibData, SystemInformationError> {
    let mib = match &msg.message {
        BCCH_BCH_MessageType::Mib(mib) => mib,
        BCCH_BCH_MessageType::MessageClassExtension(_) => {
            return Err(SystemInformationError::InvalidMessageType {
                expected: "MIB".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Parse system frame number
    let system_frame_number = bitvec_to_u8(&mib.system_frame_number.0);

    // Parse subcarrier spacing
    let sub_carrier_spacing_common = match mib.sub_carrier_spacing_common.0 {
        MIBSubCarrierSpacingCommon::SCS15OR60 => SubCarrierSpacingCommon::Scs15Or60,
        MIBSubCarrierSpacingCommon::SCS30OR120 => SubCarrierSpacingCommon::Scs30Or120,
        _ => SubCarrierSpacingCommon::Scs15Or60, // Default fallback
    };

    // Parse DMRS Type-A position
    let dmrs_type_a_position = match mib.dmrs_type_a_position.0 {
        MIBDmrs_TypeA_Position::POS2 => DmrsTypeAPosition::Pos2,
        MIBDmrs_TypeA_Position::POS3 => DmrsTypeAPosition::Pos3,
        _ => DmrsTypeAPosition::Pos2, // Default fallback
    };

    // Parse cell barred status
    let cell_barred = match mib.cell_barred.0 {
        MIBCellBarred::BARRED => CellBarredStatus::Barred,
        MIBCellBarred::NOT_BARRED => CellBarredStatus::NotBarred,
        _ => CellBarredStatus::NotBarred, // Default fallback
    };

    // Parse intra-frequency reselection
    let intra_freq_reselection = match mib.intra_freq_reselection.0 {
        MIBIntraFreqReselection::ALLOWED => IntraFreqReselection::Allowed,
        MIBIntraFreqReselection::NOT_ALLOWED => IntraFreqReselection::NotAllowed,
        _ => IntraFreqReselection::Allowed, // Default fallback
    };

    Ok(MibData {
        system_frame_number,
        sub_carrier_spacing_common,
        ssb_subcarrier_offset: mib.ssb_subcarrier_offset.0,
        dmrs_type_a_position,
        pdcch_config_sib1: PdcchConfigSib1Params {
            coreset_zero: mib.pdcch_config_sib1.control_resource_set_zero.0,
            search_space_zero: mib.pdcch_config_sib1.search_space_zero.0,
        },
        cell_barred,
        intra_freq_reselection,
    })
}

/// Helper function to convert `BitVec` to u8
fn bitvec_to_u8(bv: &BitVec<u8, Msb0>) -> u8 {
    let mut value: u8 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u8);
    }
    value
}

// ============================================================================
// SIB1 (System Information Block Type 1)
// ============================================================================

/// Cell selection information for SIB1
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellSelectionInfo {
    /// Minimum required RX level (Q-RxLevMin)
    pub q_rx_lev_min: i8,
    /// Offset to minimum RX level (optional, 1-8)
    pub q_rx_lev_min_offset: Option<u8>,
    /// Minimum required RX level for SUL (optional)
    pub q_rx_lev_min_sul: Option<i8>,
    /// Minimum required quality level (optional)
    pub q_qual_min: Option<i8>,
    /// Offset to minimum quality level (optional, 1-8)
    pub q_qual_min_offset: Option<u8>,
}

/// PLMN Identity (MCC + MNC)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlmnIdentity {
    /// Mobile Country Code (3 digits)
    pub mcc: Option<[u8; 3]>,
    /// Mobile Network Code (2-3 digits)
    pub mnc: Vec<u8>,
}

/// PLMN Identity Information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlmnIdentityInfo {
    /// List of PLMN identities
    pub plmn_identity_list: Vec<PlmnIdentity>,
    /// Tracking Area Code (24 bits)
    pub tracking_area_code: Option<u32>,
    /// Cell Identity (36 bits)
    pub cell_identity: u64,
}

/// UE timers and constants broadcast in SIB1 (TS 38.331 §6.3.2, §7.1.1)
///
/// All timer values are in milliseconds; counters are plain counts. Values
/// must be members of the enumerated sets defined in TS 38.331 (e.g. t300 in
/// {100, 200, 300, 400, 600, 1000, 1500, 2000} ms), otherwise the build fails.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UeTimersAndConstantsParams {
    /// T300: RRCSetupRequest supervision timer (ms)
    pub t300_ms: u16,
    /// T301: RRCReestablishmentRequest supervision timer (ms)
    pub t301_ms: u16,
    /// T310: radio link failure detection timer (ms)
    pub t310_ms: u16,
    /// N310: consecutive out-of-sync indications before starting T310
    pub n310: u8,
    /// T311: re-establishment cell selection timer (ms)
    pub t311_ms: u16,
    /// N311: consecutive in-sync indications stopping T310
    pub n311: u8,
    /// T319: RRCResumeRequest supervision timer (ms)
    pub t319_ms: u16,
}

impl Default for UeTimersAndConstantsParams {
    fn default() -> Self {
        // Common deployment defaults per TS 38.331 §7.1.1
        Self {
            t300_ms: 1000,
            t301_ms: 1000,
            t310_ms: 1000,
            n310: 10,
            t311_ms: 30000,
            n311: 1,
            t319_ms: 1000,
        }
    }
}

/// Parameters for building a SIB1 message
#[derive(Debug, Clone)]
pub struct Sib1Params {
    /// Cell selection information (optional for non-standalone)
    pub cell_selection_info: Option<CellSelectionInfo>,
    /// PLMN identity info list
    pub plmn_identity_info_list: Vec<PlmnIdentityInfo>,
    /// IMS emergency support
    pub ims_emergency_support: bool,
    /// eCall over IMS support
    pub ecall_over_ims_support: bool,
    /// UE timers and constants (optional)
    pub ue_timers_and_constants: Option<UeTimersAndConstantsParams>,
    /// `intraFreqReselectionRedCap` (Rel-17, TS 38.331 §6.3.2 SIB1-v1700-IEs):
    /// controls whether RedCap UEs are allowed to perform intra-frequency cell
    /// reselection.
    ///
    /// NOT WIRE-CONFORMANT (Wave 4 honest-defer): the `rrc-15.6.0` SIB1 schema
    /// predates the Rel-17 field, so this flag is broadcast as a sim-internal
    /// private marker TLV inside the opaque SIB1 `lateNonCriticalExtension`
    /// OCTET STRING (see [`SIB1_REDCAP_LNCE_TAG`]). A real UE will not parse it;
    /// conformant SIB1 RedCap signalling requires a Rel-17 RRC codec.
    pub intra_freq_reselection_redcap: bool,
}

/// Private, sim-internal marker tag for the `intraFreqReselectionRedCap` TLV
/// inside the opaque SIB1 `lateNonCriticalExtension` OCTET STRING. NOT a 3GPP
/// IEI; chosen in the 0xF0-0xFF private range (it previously aliased NAS IEI
/// 0x52). Kept consistent with `REDCAP_LNCE_TAG` in the `rrc_setup` module.
/// Conformant SIB1 RedCap signalling requires a Rel-17 RRC codec.
const SIB1_REDCAP_LNCE_TAG: u8 = 0xFE;

/// Parsed SIB1 data
#[derive(Debug, Clone)]
pub struct Sib1Data {
    /// Cell selection information
    pub cell_selection_info: Option<CellSelectionInfo>,
    /// PLMN identity info list
    pub plmn_identity_info_list: Vec<PlmnIdentityInfo>,
    /// IMS emergency support
    pub ims_emergency_support: bool,
    /// eCall over IMS support
    pub ecall_over_ims_support: bool,
    /// UE timers and constants
    pub ue_timers_and_constants: Option<UeTimersAndConstantsParams>,
    /// `intraFreqReselectionRedCap` (Rel-17), recovered from the SIB1
    /// `lateNonCriticalExtension` octet container (see [`Sib1Params`]).
    pub intra_freq_reselection_redcap: bool,
}

/// Build a SIB1 message
pub fn build_sib1(params: &Sib1Params) -> Result<BCCH_DL_SCH_Message, SystemInformationError> {
    if params.plmn_identity_info_list.is_empty() {
        return Err(SystemInformationError::InvalidFieldValue(
            "PLMN Identity Info List cannot be empty".to_string(),
        ));
    }

    // Build cell selection info if present
    let cell_selection_info =
        params
            .cell_selection_info
            .as_ref()
            .map(|csi| SIB1CellSelectionInfo {
                q_rx_lev_min: Q_RxLevMin(csi.q_rx_lev_min),
                q_rx_lev_min_offset: csi
                    .q_rx_lev_min_offset
                    .map(SIB1CellSelectionInfoQ_RxLevMinOffset),
                q_rx_lev_min_sul: csi.q_rx_lev_min_sul.map(Q_RxLevMin),
                q_qual_min: csi.q_qual_min.map(Q_QualMin),
                q_qual_min_offset: csi
                    .q_qual_min_offset
                    .map(SIB1CellSelectionInfoQ_QualMinOffset),
            });

    // Build PLMN identity info list
    let plmn_identity_list: Vec<PLMN_IdentityInfo> = params
        .plmn_identity_info_list
        .iter()
        .map(build_plmn_identity_info)
        .collect();

    let cell_access_related_info = CellAccessRelatedInfo {
        plmn_identity_list: PLMN_IdentityInfoList(plmn_identity_list),
        cell_reserved_for_other_use: None,
    };

    let sib1 = SIB1 {
        cell_selection_info,
        cell_access_related_info,
        conn_est_failure_control: None,
        si_scheduling_info: None,
        serving_cell_config_common: None,
        ims_emergency_support: if params.ims_emergency_support {
            Some(SIB1Ims_EmergencySupport(SIB1Ims_EmergencySupport::TRUE))
        } else {
            None
        },
        e_call_over_ims_support: if params.ecall_over_ims_support {
            Some(SIB1ECallOverIMS_Support(SIB1ECallOverIMS_Support::TRUE))
        } else {
            None
        },
        ue_timers_and_constants: params
            .ue_timers_and_constants
            .as_ref()
            .map(build_ue_timers_and_constants)
            .transpose()?,
        uac_barring_info: None,
        use_full_resume_id: None,
        // intraFreqReselectionRedCap (Rel-17): broadcast as a minimal TLV in the
        // spec-legal lateNonCriticalExtension OCTET STRING when allowed.
        late_non_critical_extension: if params.intra_freq_reselection_redcap {
            Some(SIB1LateNonCriticalExtension(vec![
                SIB1_REDCAP_LNCE_TAG,
                1, // length
                1, // value: intra-freq reselection allowed for RedCap
            ]))
        } else {
            None
        },
        non_critical_extension: None,
    };

    Ok(BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(
            BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(sib1),
        ),
    })
}

/// Helper function to build PLMN Identity Info
fn build_plmn_identity_info(info: &PlmnIdentityInfo) -> PLMN_IdentityInfo {
    let plmn_list: Vec<PLMN_Identity> = info
        .plmn_identity_list
        .iter()
        .map(|plmn| {
            let mcc = plmn.mcc.map(|digits| {
                MCC(vec![
                    MCC_MNC_Digit(digits[0]),
                    MCC_MNC_Digit(digits[1]),
                    MCC_MNC_Digit(digits[2]),
                ])
            });
            let mnc: Vec<MCC_MNC_Digit> = plmn.mnc.iter().map(|&d| MCC_MNC_Digit(d)).collect();
            PLMN_Identity { mcc, mnc: MNC(mnc) }
        })
        .collect();

    // Build tracking area code (24 bits)
    let tac = info.tracking_area_code.map(|tac_val| {
        let mut tac_bv: BitVec<u8, Msb0> = BitVec::new();
        for i in (0..24).rev() {
            tac_bv.push((tac_val >> i) & 1 == 1);
        }
        TrackingAreaCode(tac_bv)
    });

    // Build cell identity (36 bits)
    let mut cell_id_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..36).rev() {
        cell_id_bv.push((info.cell_identity >> i) & 1 == 1);
    }

    PLMN_IdentityInfo {
        plmn_identity_list: PLMN_IdentityInfoPlmn_IdentityList(plmn_list),
        tracking_area_code: tac,
        ranac: None,
        cell_identity: CellIdentity(cell_id_bv),
        cell_reserved_for_operator_use: PLMN_IdentityInfoCellReservedForOperatorUse(
            PLMN_IdentityInfoCellReservedForOperatorUse::NOT_RESERVED,
        ),
    }
}

/// Parse a SIB1 from a BCCH-DL-SCH message
pub fn parse_sib1(msg: &BCCH_DL_SCH_Message) -> Result<Sib1Data, SystemInformationError> {
    let sib1 = match &msg.message {
        BCCH_DL_SCH_MessageType::C1(c1) => match c1 {
            BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(sib1) => sib1,
            BCCH_DL_SCH_MessageType_c1::SystemInformation(_) => {
                return Err(SystemInformationError::InvalidMessageType {
                    expected: "SIB1".to_string(),
                    actual: "SystemInformation".to_string(),
                })
            }
        },
        BCCH_DL_SCH_MessageType::MessageClassExtension(_) => {
            return Err(SystemInformationError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    // Parse cell selection info
    let cell_selection_info = sib1
        .cell_selection_info
        .as_ref()
        .map(|csi| CellSelectionInfo {
            q_rx_lev_min: csi.q_rx_lev_min.0,
            q_rx_lev_min_offset: csi.q_rx_lev_min_offset.as_ref().map(|v| v.0),
            q_rx_lev_min_sul: csi.q_rx_lev_min_sul.as_ref().map(|v| v.0),
            q_qual_min: csi.q_qual_min.as_ref().map(|v| v.0),
            q_qual_min_offset: csi.q_qual_min_offset.as_ref().map(|v| v.0),
        });

    // Parse PLMN identity info list
    let plmn_identity_info_list: Vec<PlmnIdentityInfo> = sib1
        .cell_access_related_info
        .plmn_identity_list
        .0
        .iter()
        .map(parse_plmn_identity_info)
        .collect();

    // Parse IMS emergency support
    let ims_emergency_support = sib1.ims_emergency_support.is_some();

    // Parse eCall over IMS support
    let ecall_over_ims_support = sib1.e_call_over_ims_support.is_some();

    // Parse UE timers and constants
    let ue_timers_and_constants = sib1
        .ue_timers_and_constants
        .as_ref()
        .map(parse_ue_timers_and_constants)
        .transpose()?;

    // intraFreqReselectionRedCap (Rel-17): recovered from the SIB1
    // lateNonCriticalExtension octet container TLV.
    let intra_freq_reselection_redcap = sib1
        .late_non_critical_extension
        .as_ref()
        .map(|lnce| parse_redcap_lnce(&lnce.0))
        .unwrap_or(false);

    Ok(Sib1Data {
        cell_selection_info,
        plmn_identity_info_list,
        ims_emergency_support,
        ecall_over_ims_support,
        ue_timers_and_constants,
        intra_freq_reselection_redcap,
    })
}

/// Scan a SIB1 `lateNonCriticalExtension` octet container for the
/// `intraFreqReselectionRedCap` TLV (`SIB1_REDCAP_LNCE_TAG`, len, value).
/// Returns true when present and set.
fn parse_redcap_lnce(bytes: &[u8]) -> bool {
    let mut i = 0;
    while i + 1 < bytes.len() {
        let tag = bytes[i];
        let len = bytes[i + 1] as usize;
        let val_start = i + 2;
        if val_start + len > bytes.len() {
            break;
        }
        if tag == SIB1_REDCAP_LNCE_TAG {
            return bytes.get(val_start).copied().unwrap_or(0) != 0;
        }
        i = val_start + len;
    }
    false
}

/// Build the generated `UE_TimersAndConstants` from millisecond/count values
fn build_ue_timers_and_constants(
    params: &UeTimersAndConstantsParams,
) -> Result<UE_TimersAndConstants, SystemInformationError> {
    let invalid = |field: &str, value: u32| {
        SystemInformationError::InvalidFieldValue(format!(
            "{field} value {value} is not in the TS 38.331 enumerated set"
        ))
    };

    let t300 = match params.t300_ms {
        100 => UE_TimersAndConstantsT300::MS100,
        200 => UE_TimersAndConstantsT300::MS200,
        300 => UE_TimersAndConstantsT300::MS300,
        400 => UE_TimersAndConstantsT300::MS400,
        600 => UE_TimersAndConstantsT300::MS600,
        1000 => UE_TimersAndConstantsT300::MS1000,
        1500 => UE_TimersAndConstantsT300::MS1500,
        2000 => UE_TimersAndConstantsT300::MS2000,
        v => return Err(invalid("t300", v as u32)),
    };
    let t301 = match params.t301_ms {
        100 => UE_TimersAndConstantsT301::MS100,
        200 => UE_TimersAndConstantsT301::MS200,
        300 => UE_TimersAndConstantsT301::MS300,
        400 => UE_TimersAndConstantsT301::MS400,
        600 => UE_TimersAndConstantsT301::MS600,
        1000 => UE_TimersAndConstantsT301::MS1000,
        1500 => UE_TimersAndConstantsT301::MS1500,
        2000 => UE_TimersAndConstantsT301::MS2000,
        v => return Err(invalid("t301", v as u32)),
    };
    let t310 = match params.t310_ms {
        0 => UE_TimersAndConstantsT310::MS0,
        50 => UE_TimersAndConstantsT310::MS50,
        100 => UE_TimersAndConstantsT310::MS100,
        200 => UE_TimersAndConstantsT310::MS200,
        500 => UE_TimersAndConstantsT310::MS500,
        1000 => UE_TimersAndConstantsT310::MS1000,
        2000 => UE_TimersAndConstantsT310::MS2000,
        v => return Err(invalid("t310", v as u32)),
    };
    let n310 = match params.n310 {
        1 => UE_TimersAndConstantsN310::N1,
        2 => UE_TimersAndConstantsN310::N2,
        3 => UE_TimersAndConstantsN310::N3,
        4 => UE_TimersAndConstantsN310::N4,
        6 => UE_TimersAndConstantsN310::N6,
        8 => UE_TimersAndConstantsN310::N8,
        10 => UE_TimersAndConstantsN310::N10,
        20 => UE_TimersAndConstantsN310::N20,
        v => return Err(invalid("n310", v as u32)),
    };
    let t311 = match params.t311_ms {
        1000 => UE_TimersAndConstantsT311::MS1000,
        3000 => UE_TimersAndConstantsT311::MS3000,
        5000 => UE_TimersAndConstantsT311::MS5000,
        10000 => UE_TimersAndConstantsT311::MS10000,
        15000 => UE_TimersAndConstantsT311::MS15000,
        20000 => UE_TimersAndConstantsT311::MS20000,
        30000 => UE_TimersAndConstantsT311::MS30000,
        v => return Err(invalid("t311", v as u32)),
    };
    let n311 = match params.n311 {
        1 => UE_TimersAndConstantsN311::N1,
        2 => UE_TimersAndConstantsN311::N2,
        3 => UE_TimersAndConstantsN311::N3,
        4 => UE_TimersAndConstantsN311::N4,
        5 => UE_TimersAndConstantsN311::N5,
        6 => UE_TimersAndConstantsN311::N6,
        8 => UE_TimersAndConstantsN311::N8,
        10 => UE_TimersAndConstantsN311::N10,
        v => return Err(invalid("n311", v as u32)),
    };
    let t319 = match params.t319_ms {
        100 => UE_TimersAndConstantsT319::MS100,
        200 => UE_TimersAndConstantsT319::MS200,
        300 => UE_TimersAndConstantsT319::MS300,
        400 => UE_TimersAndConstantsT319::MS400,
        600 => UE_TimersAndConstantsT319::MS600,
        1000 => UE_TimersAndConstantsT319::MS1000,
        1500 => UE_TimersAndConstantsT319::MS1500,
        2000 => UE_TimersAndConstantsT319::MS2000,
        v => return Err(invalid("t319", v as u32)),
    };

    Ok(UE_TimersAndConstants {
        t300: UE_TimersAndConstantsT300(t300),
        t301: UE_TimersAndConstantsT301(t301),
        t310: UE_TimersAndConstantsT310(t310),
        n310: UE_TimersAndConstantsN310(n310),
        t311: UE_TimersAndConstantsT311(t311),
        n311: UE_TimersAndConstantsN311(n311),
        t319: UE_TimersAndConstantsT319(t319),
    })
}

/// Parse the generated `UE_TimersAndConstants` into millisecond/count values
fn parse_ue_timers_and_constants(
    timers: &UE_TimersAndConstants,
) -> Result<UeTimersAndConstantsParams, SystemInformationError> {
    let invalid = |field: &'static str| {
        SystemInformationError::InvalidFieldValue(format!("Unknown {field} enumerated value"))
    };

    let t300_ms = match timers.t300.0 {
        UE_TimersAndConstantsT300::MS100 => 100,
        UE_TimersAndConstantsT300::MS200 => 200,
        UE_TimersAndConstantsT300::MS300 => 300,
        UE_TimersAndConstantsT300::MS400 => 400,
        UE_TimersAndConstantsT300::MS600 => 600,
        UE_TimersAndConstantsT300::MS1000 => 1000,
        UE_TimersAndConstantsT300::MS1500 => 1500,
        UE_TimersAndConstantsT300::MS2000 => 2000,
        _ => return Err(invalid("t300")),
    };
    let t301_ms = match timers.t301.0 {
        UE_TimersAndConstantsT301::MS100 => 100,
        UE_TimersAndConstantsT301::MS200 => 200,
        UE_TimersAndConstantsT301::MS300 => 300,
        UE_TimersAndConstantsT301::MS400 => 400,
        UE_TimersAndConstantsT301::MS600 => 600,
        UE_TimersAndConstantsT301::MS1000 => 1000,
        UE_TimersAndConstantsT301::MS1500 => 1500,
        UE_TimersAndConstantsT301::MS2000 => 2000,
        _ => return Err(invalid("t301")),
    };
    let t310_ms = match timers.t310.0 {
        UE_TimersAndConstantsT310::MS0 => 0,
        UE_TimersAndConstantsT310::MS50 => 50,
        UE_TimersAndConstantsT310::MS100 => 100,
        UE_TimersAndConstantsT310::MS200 => 200,
        UE_TimersAndConstantsT310::MS500 => 500,
        UE_TimersAndConstantsT310::MS1000 => 1000,
        UE_TimersAndConstantsT310::MS2000 => 2000,
        _ => return Err(invalid("t310")),
    };
    let n310 = match timers.n310.0 {
        UE_TimersAndConstantsN310::N1 => 1,
        UE_TimersAndConstantsN310::N2 => 2,
        UE_TimersAndConstantsN310::N3 => 3,
        UE_TimersAndConstantsN310::N4 => 4,
        UE_TimersAndConstantsN310::N6 => 6,
        UE_TimersAndConstantsN310::N8 => 8,
        UE_TimersAndConstantsN310::N10 => 10,
        UE_TimersAndConstantsN310::N20 => 20,
        _ => return Err(invalid("n310")),
    };
    let t311_ms = match timers.t311.0 {
        UE_TimersAndConstantsT311::MS1000 => 1000,
        UE_TimersAndConstantsT311::MS3000 => 3000,
        UE_TimersAndConstantsT311::MS5000 => 5000,
        UE_TimersAndConstantsT311::MS10000 => 10000,
        UE_TimersAndConstantsT311::MS15000 => 15000,
        UE_TimersAndConstantsT311::MS20000 => 20000,
        UE_TimersAndConstantsT311::MS30000 => 30000,
        _ => return Err(invalid("t311")),
    };
    let n311 = match timers.n311.0 {
        UE_TimersAndConstantsN311::N1 => 1,
        UE_TimersAndConstantsN311::N2 => 2,
        UE_TimersAndConstantsN311::N3 => 3,
        UE_TimersAndConstantsN311::N4 => 4,
        UE_TimersAndConstantsN311::N5 => 5,
        UE_TimersAndConstantsN311::N6 => 6,
        UE_TimersAndConstantsN311::N8 => 8,
        UE_TimersAndConstantsN311::N10 => 10,
        _ => return Err(invalid("n311")),
    };
    let t319_ms = match timers.t319.0 {
        UE_TimersAndConstantsT319::MS100 => 100,
        UE_TimersAndConstantsT319::MS200 => 200,
        UE_TimersAndConstantsT319::MS300 => 300,
        UE_TimersAndConstantsT319::MS400 => 400,
        UE_TimersAndConstantsT319::MS600 => 600,
        UE_TimersAndConstantsT319::MS1000 => 1000,
        UE_TimersAndConstantsT319::MS1500 => 1500,
        UE_TimersAndConstantsT319::MS2000 => 2000,
        _ => return Err(invalid("t319")),
    };

    Ok(UeTimersAndConstantsParams {
        t300_ms,
        t301_ms,
        t310_ms,
        n310,
        t311_ms,
        n311,
        t319_ms,
    })
}

/// Helper function to parse PLMN Identity Info
fn parse_plmn_identity_info(info: &PLMN_IdentityInfo) -> PlmnIdentityInfo {
    let plmn_identity_list: Vec<PlmnIdentity> = info
        .plmn_identity_list
        .0
        .iter()
        .map(|plmn| {
            let mcc = plmn.mcc.as_ref().map(|mcc_val| {
                let mut digits = [0u8; 3];
                for (i, digit) in mcc_val.0.iter().enumerate().take(3) {
                    digits[i] = digit.0;
                }
                digits
            });
            let mnc: Vec<u8> = plmn.mnc.0.iter().map(|d| d.0).collect();
            PlmnIdentity { mcc, mnc }
        })
        .collect();

    // Parse tracking area code (24 bits)
    let tracking_area_code = info
        .tracking_area_code
        .as_ref()
        .map(|tac| bitvec_to_u32(&tac.0));

    // Parse cell identity (36 bits)
    let cell_identity = bitvec_to_u64(&info.cell_identity.0);

    PlmnIdentityInfo {
        plmn_identity_list,
        tracking_area_code,
        cell_identity,
    }
}

/// Helper function to convert `BitVec` to u32
fn bitvec_to_u32(bv: &BitVec<u8, Msb0>) -> u32 {
    let mut value: u32 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u32);
    }
    value
}

/// Helper function to convert `BitVec` to u64
fn bitvec_to_u64(bv: &BitVec<u8, Msb0>) -> u64 {
    let mut value: u64 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u64);
    }
    value
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a MIB to bytes
pub fn encode_mib(params: &MibParams) -> Result<Vec<u8>, SystemInformationError> {
    let msg = build_mib(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a MIB from bytes
pub fn decode_mib(bytes: &[u8]) -> Result<MibData, SystemInformationError> {
    let msg: BCCH_BCH_Message = decode_rrc(bytes)?;
    parse_mib(&msg)
}

/// Build and encode a SIB1 to bytes
pub fn encode_sib1(params: &Sib1Params) -> Result<Vec<u8>, SystemInformationError> {
    let msg = build_sib1(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a SIB1 from bytes
pub fn decode_sib1(bytes: &[u8]) -> Result<Sib1Data, SystemInformationError> {
    let msg: BCCH_DL_SCH_Message = decode_rrc(bytes)?;
    parse_sib1(&msg)
}

/// Check if a BCCH-BCH message is a MIB
pub fn is_mib(msg: &BCCH_BCH_Message) -> bool {
    matches!(&msg.message, BCCH_BCH_MessageType::Mib(_))
}

/// Check if a BCCH-DL-SCH message is a SIB1
pub fn is_sib1(msg: &BCCH_DL_SCH_Message) -> bool {
    matches!(
        &msg.message,
        BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(_))
    )
}

// ============================================================================
// SIB2 / SIB3 / SIB4 — idle-mode reselection parameters (TS 38.331 §6.3.1)
// ============================================================================
//
// SIB1 tells the UE what the cell IS. These three tell it when to LEAVE:
//
// - **SIB2** carries the cell-common reselection parameters — `q-Hyst`,
//   `t-ReselectionNR` and the serving frequency's `cellReselectionPriority` —
//   i.e. the `Q_hyst` and `Treselection` terms of the TS 38.304 §5.2.4.6
//   R-criterion.
// - **SIB3** carries `intraFreqNeighCellList`, whose per-cell `q-OffsetCell` is
//   the `Qoffset` term of `R_n`. Entries are keyed by `physCellId`.
// - **SIB4** carries `interFreqCarrierFreqList`, giving each other carrier its
//   own `cellReselectionPriority`.
//
// All three ride a `SystemInformation` message on BCCH-DL-SCH — the *same*
// channel SIB1 uses, but the other arm of the `BCCH-DL-SCH-Message` CHOICE. A
// receiver therefore has to dispatch on the arm rather than assume SIB1.
//
// Note none of these is on an extension arm of that CHOICE, so unlike SIB19
// (issue #56) they encode fine with this codec version — see issue #117 for why
// that distinction matters.

/// `Q-Hyst` in dB, and the ASN.1 enumeration index that carries it
/// (TS 38.331 §6.3.1 `q-Hyst`).
///
/// Read off the schema's enumeration rather than computed: the values are *not*
/// a uniform step. They go 0..6 in 1 dB steps and then 8, 10, 12 … 24 in 2 dB
/// steps, so `index == dB` holds only up to 6 and any arithmetic mapping is
/// wrong above it.
const Q_HYST_DB_TO_INDEX: [(i32, u8); 16] = [
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (4, 4),
    (5, 5),
    (6, 6),
    (8, 7),
    (10, 8),
    (12, 9),
    (14, 10),
    (16, 11),
    (18, 12),
    (20, 13),
    (22, 14),
    (24, 15),
];

/// `Q-OffsetRange` in dB, and its ASN.1 enumeration index (TS 38.331 §6.3.4).
///
/// Same non-uniformity as `Q-Hyst`, and signed: -24..-6 in 2 dB steps, then
/// -5..5 in 1 dB steps, then 6..24 in 2 dB steps. Index 15 is 0 dB.
const Q_OFFSET_DB_TO_INDEX: [(i32, u8); 31] = [
    (-24, 0),
    (-22, 1),
    (-20, 2),
    (-18, 3),
    (-16, 4),
    (-14, 5),
    (-12, 6),
    (-10, 7),
    (-8, 8),
    (-6, 9),
    (-5, 10),
    (-4, 11),
    (-3, 12),
    (-2, 13),
    (-1, 14),
    (0, 15),
    (1, 16),
    (2, 17),
    (3, 18),
    (4, 19),
    (5, 20),
    (6, 21),
    (8, 22),
    (10, 23),
    (12, 24),
    (14, 25),
    (16, 26),
    (18, 27),
    (20, 28),
    (22, 29),
    (24, 30),
];

/// The highest `cellReselectionPriority` the ASN.1 type allows
/// (`INTEGER (0..7)`, TS 38.331 §6.3.4).
pub const CELL_RESELECTION_PRIORITY_MAX: u8 = 7;

/// The highest `t-ReselectionNR` the ASN.1 type allows, in seconds
/// (`INTEGER (0..7)`, TS 38.331 §6.3.1).
pub const T_RESELECTION_MAX_S: u8 = 7;

/// Maps a `Q-Hyst` in dB onto its enumeration index, refusing a value the
/// enumeration does not contain.
///
/// Refuses rather than rounding to the nearest legal value: a silently adjusted
/// hysteresis would make a reselection test that seems to configure 7 dB
/// actually configure 6 or 8, and the discrepancy would surface as a flaky
/// margin comparison rather than as a configuration error.
pub fn q_hyst_index(db: i32) -> Result<u8, SystemInformationError> {
    Q_HYST_DB_TO_INDEX
        .iter()
        .find(|(value, _)| *value == db)
        .map(|(_, index)| *index)
        .ok_or_else(|| {
            SystemInformationError::InvalidFieldValue(format!(
                "q-Hyst {db} dB is not one of the values TS 38.331 enumerates \
                 (0-6 in 1 dB steps, then 8-24 in 2 dB steps)"
            ))
        })
}

/// The inverse of [`q_hyst_index`].
pub fn q_hyst_db(index: u8) -> Result<i32, SystemInformationError> {
    Q_HYST_DB_TO_INDEX
        .iter()
        .find(|(_, value)| *value == index)
        .map(|(db, _)| *db)
        .ok_or_else(|| {
            SystemInformationError::InvalidFieldValue(format!(
                "q-Hyst index {index} is out of range"
            ))
        })
}

/// Maps a `Q-OffsetRange` in dB onto its enumeration index. Refuses an absent
/// value for the same reason as [`q_hyst_index`].
pub fn q_offset_index(db: i32) -> Result<u8, SystemInformationError> {
    Q_OFFSET_DB_TO_INDEX
        .iter()
        .find(|(value, _)| *value == db)
        .map(|(_, index)| *index)
        .ok_or_else(|| {
            SystemInformationError::InvalidFieldValue(format!(
                "q-OffsetCell {db} dB is not one of the values TS 38.331 \
                 enumerates (-24..-6 and 6..24 in 2 dB steps, -5..5 in 1 dB steps)"
            ))
        })
}

/// The inverse of [`q_offset_index`].
pub fn q_offset_db(index: u8) -> Result<i32, SystemInformationError> {
    Q_OFFSET_DB_TO_INDEX
        .iter()
        .find(|(_, value)| *value == index)
        .map(|(db, _)| *db)
        .ok_or_else(|| {
            SystemInformationError::InvalidFieldValue(format!(
                "q-OffsetCell index {index} is out of range"
            ))
        })
}

/// Parameters for the SIB2 this cell broadcasts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sib2Params {
    /// `q-Hyst` in dB — the `Q_hyst` term added to the SERVING cell's measured
    /// level in `R_s` (TS 38.304 §5.2.4.6).
    pub q_hyst_db: i32,
    /// `t-ReselectionNR` in seconds — how long a neighbour must stay better
    /// ranked before the UE reselects to it.
    pub t_reselection_s: u8,
    /// `cellReselectionPriority` of the serving frequency (0..7).
    pub cell_reselection_priority: u8,
    /// `q-RxLevMin` for intra-frequency reselection, in units of 2 dBm.
    pub q_rx_lev_min: i8,
    /// `s-IntraSearchP`, the threshold below which the UE starts measuring
    /// intra-frequency neighbours, in units of 2 dB (0..31).
    pub s_intra_search_p: u8,
    /// `threshServingLowP`, in units of 2 dB (0..31).
    pub thresh_serving_low_p: u8,
}

/// Parsed SIB2 data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sib2Data {
    /// `q-Hyst` in dB
    pub q_hyst_db: i32,
    /// `t-ReselectionNR` in seconds
    pub t_reselection_s: u8,
    /// Serving frequency's `cellReselectionPriority`
    pub cell_reselection_priority: u8,
    /// `q-RxLevMin` in units of 2 dBm
    pub q_rx_lev_min: i8,
    /// `s-IntraSearchP` in units of 2 dB
    pub s_intra_search_p: u8,
    /// `threshServingLowP` in units of 2 dB
    pub thresh_serving_low_p: u8,
}

/// One `intraFreqNeighCellList` entry: a neighbour's PCI and its `Qoffset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IntraFreqNeighbour {
    /// `physCellId` (0..1007) of the neighbour
    pub phys_cell_id: u16,
    /// `q-OffsetCell` in dB — SUBTRACTED from this neighbour's measured level
    /// in `R_n` (TS 38.304 §5.2.4.6), so a positive value makes the neighbour
    /// LESS attractive.
    pub q_offset_db: i32,
}

/// Parameters for the SIB3 this cell broadcasts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sib3Params {
    /// Intra-frequency neighbours and their per-cell offsets. Up to 16, the
    /// `IntraFreqNeighCellList` size bound.
    pub intra_freq_neighbours: Vec<IntraFreqNeighbour>,
}

/// Parsed SIB3 data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sib3Data {
    /// Intra-frequency neighbours and their per-cell offsets
    pub intra_freq_neighbours: Vec<IntraFreqNeighbour>,
}

/// One `interFreqCarrierFreqList` entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterFreqCarrier {
    /// `dl-CarrierFreq` (ARFCN)
    pub dl_carrier_freq: u32,
    /// `cellReselectionPriority` for this carrier (0..7)
    pub cell_reselection_priority: u8,
    /// `threshX-HighP`, in units of 2 dB (0..31)
    pub thresh_x_high_p: u8,
    /// `threshX-LowP`, in units of 2 dB (0..31)
    pub thresh_x_low_p: u8,
    /// `q-RxLevMin`, in units of 2 dBm
    pub q_rx_lev_min: i8,
    /// `t-ReselectionNR` for this carrier, in seconds
    pub t_reselection_s: u8,
}

/// Parameters for the SIB4 this cell broadcasts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sib4Params {
    /// Other carriers and their reselection priorities
    pub inter_freq_carriers: Vec<InterFreqCarrier>,
}

/// Parsed SIB4 data.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Sib4Data {
    /// Other carriers and their reselection priorities
    pub inter_freq_carriers: Vec<InterFreqCarrier>,
}

/// Which SIBs a `SystemInformation` message should carry.
///
/// A struct of options rather than a list of an enum, so the same SIB cannot be
/// requested twice — `sib-TypeAndInfo` is a SEQUENCE OF and would happily carry
/// two `sib2` entries, which a receiver would then have to arbitrate between.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SystemInformationParams {
    /// SIB2, when broadcast
    pub sib2: Option<Sib2Params>,
    /// SIB3, when broadcast
    pub sib3: Option<Sib3Params>,
    /// SIB4, when broadcast
    pub sib4: Option<Sib4Params>,
}

/// Parsed `SystemInformation` contents.
///
/// Every field is optional because a `SystemInformation` message carries
/// whichever SIBs the cell chose to schedule together; absence means "not in
/// this message", not "not broadcast by this cell".
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SystemInformationData {
    /// SIB2, when this message carried one
    pub sib2: Option<Sib2Data>,
    /// SIB3, when this message carried one
    pub sib3: Option<Sib3Data>,
    /// SIB4, when this message carried one
    pub sib4: Option<Sib4Data>,
}

impl SystemInformationData {
    /// Whether this message carried anything this codec understands.
    ///
    /// A `SystemInformation` carrying only SIB5..SIB9 decodes successfully and
    /// yields an empty value; a caller that treated that as a parse failure
    /// would log an error for a perfectly legal message.
    pub fn is_empty(&self) -> bool {
        self.sib2.is_none() && self.sib3.is_none() && self.sib4.is_none()
    }
}

/// Build a `SystemInformation` message carrying the requested SIBs.
pub fn build_system_information(
    params: &SystemInformationParams,
) -> Result<BCCH_DL_SCH_Message, SystemInformationError> {
    let mut entries: Vec<SystemInformation_IEsSib_TypeAndInfo_Entry> = Vec::new();

    if let Some(sib2) = params.sib2.as_ref() {
        entries.push(SystemInformation_IEsSib_TypeAndInfo_Entry::Sib2(
            build_sib2(sib2)?,
        ));
    }
    if let Some(sib3) = params.sib3.as_ref() {
        entries.push(SystemInformation_IEsSib_TypeAndInfo_Entry::Sib3(
            build_sib3(sib3)?,
        ));
    }
    if let Some(sib4) = params.sib4.as_ref() {
        entries.push(SystemInformation_IEsSib_TypeAndInfo_Entry::Sib4(
            build_sib4(sib4)?,
        ));
    }

    // `sib-TypeAndInfo` is SIZE (1..maxSIB): an empty list is not encodable, and
    // a message carrying no SIB would be meaningless anyway.
    if entries.is_empty() {
        return Err(SystemInformationError::InvalidFieldValue(
            "SystemInformation must carry at least one SIB".to_string(),
        ));
    }

    Ok(BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(
            SystemInformation {
                critical_extensions: SystemInformationCriticalExtensions::SystemInformation(
                    SystemInformation_IEs {
                        sib_type_and_info: SystemInformation_IEsSib_TypeAndInfo(entries),
                        late_non_critical_extension: None,
                        non_critical_extension: None,
                    },
                ),
            },
        )),
    })
}

fn build_sib2(params: &Sib2Params) -> Result<SIB2, SystemInformationError> {
    if params.cell_reselection_priority > CELL_RESELECTION_PRIORITY_MAX {
        return Err(SystemInformationError::InvalidFieldValue(format!(
            "cellReselectionPriority {} exceeds the ASN.1 maximum {}",
            params.cell_reselection_priority, CELL_RESELECTION_PRIORITY_MAX
        )));
    }
    if params.t_reselection_s > T_RESELECTION_MAX_S {
        return Err(SystemInformationError::InvalidFieldValue(format!(
            "t-ReselectionNR {} s exceeds the ASN.1 maximum {} s",
            params.t_reselection_s, T_RESELECTION_MAX_S
        )));
    }

    Ok(SIB2 {
        cell_reselection_info_common: SIB2CellReselectionInfoCommon {
            nrof_ss_blocks_to_average: None,
            abs_thresh_ss_blocks_consolidation: None,
            range_to_best_cell: None,
            q_hyst: SIB2CellReselectionInfoCommonQ_Hyst(q_hyst_index(params.q_hyst_db)?),
            speed_state_reselection_pars: None,
        },
        cell_reselection_serving_freq_info: SIB2CellReselectionServingFreqInfo {
            s_non_intra_search_p: None,
            s_non_intra_search_q: None,
            thresh_serving_low_p: ReselectionThreshold(params.thresh_serving_low_p),
            thresh_serving_low_q: None,
            cell_reselection_priority: CellReselectionPriority(params.cell_reselection_priority),
            cell_reselection_sub_priority: None,
        },
        intra_freq_cell_reselection_info: SIB2IntraFreqCellReselectionInfo {
            q_rx_lev_min: Q_RxLevMin(params.q_rx_lev_min),
            q_rx_lev_min_sul: None,
            q_qual_min: None,
            s_intra_search_p: ReselectionThreshold(params.s_intra_search_p),
            s_intra_search_q: None,
            t_reselection_nr: T_Reselection(params.t_reselection_s),
            frequency_band_list: None,
            frequency_band_list_sul: None,
            p_max: None,
            smtc: None,
            ss_rssi_measurement: None,
            ssb_to_measure: None,
            derive_ssb_index_from_cell: SIB2IntraFreqCellReselectionInfoDeriveSSB_IndexFromCell(
                true,
            ),
        },
    })
}

fn build_sib3(params: &Sib3Params) -> Result<SIB3, SystemInformationError> {
    let neighbours = params
        .intra_freq_neighbours
        .iter()
        .map(|n| {
            Ok(IntraFreqNeighCellInfo {
                phys_cell_id: PhysCellId(n.phys_cell_id),
                q_offset_cell: Q_OffsetRange(q_offset_index(n.q_offset_db)?),
                q_rx_lev_min_offset_cell: None,
                q_rx_lev_min_offset_cell_sul: None,
                q_qual_min_offset_cell: None,
            })
        })
        .collect::<Result<Vec<_>, SystemInformationError>>()?;

    Ok(SIB3 {
        // `IntraFreqNeighCellList` is SIZE (1..maxCellIntra): an empty list is
        // not encodable, so an empty request omits the field rather than
        // encoding a zero-length list.
        intra_freq_neigh_cell_list: if neighbours.is_empty() {
            None
        } else {
            Some(IntraFreqNeighCellList(neighbours))
        },
        intra_freq_black_cell_list: None,
        late_non_critical_extension: None,
    })
}

fn build_sib4(params: &Sib4Params) -> Result<SIB4, SystemInformationError> {
    if params.inter_freq_carriers.is_empty() {
        return Err(SystemInformationError::InvalidFieldValue(
            "SIB4 interFreqCarrierFreqList is SIZE (1..maxFreq): a SIB4 with no \
             carrier is not encodable, so omit the SIB instead"
                .to_string(),
        ));
    }

    let carriers = params
        .inter_freq_carriers
        .iter()
        .map(|c| {
            if c.cell_reselection_priority > CELL_RESELECTION_PRIORITY_MAX {
                return Err(SystemInformationError::InvalidFieldValue(format!(
                    "interFreq cellReselectionPriority {} exceeds the ASN.1 maximum {}",
                    c.cell_reselection_priority, CELL_RESELECTION_PRIORITY_MAX
                )));
            }
            if c.t_reselection_s > T_RESELECTION_MAX_S {
                return Err(SystemInformationError::InvalidFieldValue(format!(
                    "interFreq t-ReselectionNR {} s exceeds the ASN.1 maximum {} s",
                    c.t_reselection_s, T_RESELECTION_MAX_S
                )));
            }
            Ok(InterFreqCarrierFreqInfo {
                dl_carrier_freq: ARFCN_ValueNR(c.dl_carrier_freq),
                frequency_band_list: None,
                frequency_band_list_sul: None,
                nrof_ss_blocks_to_average: None,
                abs_thresh_ss_blocks_consolidation: None,
                smtc: None,
                // A PHY parameter this simulator does not model; 30 kHz matches
                // the FR1 cell the MIB advertises in `mib_params`.
                ssb_subcarrier_spacing: SubcarrierSpacing(SubcarrierSpacing::K_HZ30),
                ssb_to_measure: None,
                derive_ssb_index_from_cell: InterFreqCarrierFreqInfoDeriveSSB_IndexFromCell(true),
                ss_rssi_measurement: None,
                q_rx_lev_min: Q_RxLevMin(c.q_rx_lev_min),
                q_rx_lev_min_sul: None,
                q_qual_min: None,
                p_max: None,
                t_reselection_nr: T_Reselection(c.t_reselection_s),
                t_reselection_nr_sf: None,
                thresh_x_high_p: ReselectionThreshold(c.thresh_x_high_p),
                thresh_x_low_p: ReselectionThreshold(c.thresh_x_low_p),
                thresh_x_q: None,
                cell_reselection_priority: Some(CellReselectionPriority(
                    c.cell_reselection_priority,
                )),
                cell_reselection_sub_priority: None,
                q_offset_freq: None,
                inter_freq_neigh_cell_list: None,
                inter_freq_black_cell_list: None,
            })
        })
        .collect::<Result<Vec<_>, SystemInformationError>>()?;

    Ok(SIB4 {
        inter_freq_carrier_freq_list: InterFreqCarrierFreqList(carriers),
        late_non_critical_extension: None,
    })
}

/// Parse a `SystemInformation` message, extracting the SIBs this codec models.
///
/// SIB5..SIB9 entries are skipped rather than refused: they are legal in the
/// same message, and a cell that schedules SIB5 alongside SIB2 should not make
/// the SIB2 unreadable.
pub fn parse_system_information(
    msg: &BCCH_DL_SCH_Message,
) -> Result<SystemInformationData, SystemInformationError> {
    let BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(si)) =
        &msg.message
    else {
        return Err(SystemInformationError::InvalidMessageType {
            expected: "SystemInformation".to_string(),
            actual: "SystemInformationBlockType1 or messageClassExtension".to_string(),
        });
    };

    let SystemInformationCriticalExtensions::SystemInformation(ies) = &si.critical_extensions
    else {
        return Err(SystemInformationError::InvalidMessageType {
            expected: "systemInformation critical extension".to_string(),
            actual: "criticalExtensionsFuture".to_string(),
        });
    };

    let mut data = SystemInformationData::default();
    for entry in &ies.sib_type_and_info.0 {
        match entry {
            SystemInformation_IEsSib_TypeAndInfo_Entry::Sib2(sib2) => {
                data.sib2 = Some(parse_sib2(sib2)?);
            }
            SystemInformation_IEsSib_TypeAndInfo_Entry::Sib3(sib3) => {
                data.sib3 = Some(parse_sib3(sib3)?);
            }
            SystemInformation_IEsSib_TypeAndInfo_Entry::Sib4(sib4) => {
                data.sib4 = Some(parse_sib4(sib4)?);
            }
            _ => {}
        }
    }
    Ok(data)
}

fn parse_sib2(sib2: &SIB2) -> Result<Sib2Data, SystemInformationError> {
    Ok(Sib2Data {
        q_hyst_db: q_hyst_db(sib2.cell_reselection_info_common.q_hyst.0)?,
        t_reselection_s: sib2.intra_freq_cell_reselection_info.t_reselection_nr.0,
        cell_reselection_priority: sib2
            .cell_reselection_serving_freq_info
            .cell_reselection_priority
            .0,
        q_rx_lev_min: sib2.intra_freq_cell_reselection_info.q_rx_lev_min.0,
        s_intra_search_p: sib2.intra_freq_cell_reselection_info.s_intra_search_p.0,
        thresh_serving_low_p: sib2
            .cell_reselection_serving_freq_info
            .thresh_serving_low_p
            .0,
    })
}

fn parse_sib3(sib3: &SIB3) -> Result<Sib3Data, SystemInformationError> {
    let mut intra_freq_neighbours = Vec::new();
    if let Some(list) = sib3.intra_freq_neigh_cell_list.as_ref() {
        for info in &list.0 {
            intra_freq_neighbours.push(IntraFreqNeighbour {
                phys_cell_id: info.phys_cell_id.0,
                q_offset_db: q_offset_db(info.q_offset_cell.0)?,
            });
        }
    }
    Ok(Sib3Data {
        intra_freq_neighbours,
    })
}

fn parse_sib4(sib4: &SIB4) -> Result<Sib4Data, SystemInformationError> {
    let mut inter_freq_carriers = Vec::new();
    for info in &sib4.inter_freq_carrier_freq_list.0 {
        inter_freq_carriers.push(InterFreqCarrier {
            dl_carrier_freq: info.dl_carrier_freq.0,
            // `cellReselectionPriority` is OPTIONAL per TS 38.331: a carrier with
            // none has "no priority provided" and TS 38.304 §5.2.4.1 says the UE
            // shall not consider it for reselection. Modelled as priority 0 with
            // the consequence noted, because the UE side treats 0 as the lowest
            // priority and a carrier at the lowest priority is never chosen over
            // the serving one -- the same outcome, without a second Option to
            // thread through the selector.
            cell_reselection_priority: info.cell_reselection_priority.as_ref().map_or(0, |p| p.0),
            thresh_x_high_p: info.thresh_x_high_p.0,
            thresh_x_low_p: info.thresh_x_low_p.0,
            q_rx_lev_min: info.q_rx_lev_min.0,
            t_reselection_s: info.t_reselection_nr.0,
        });
    }
    Ok(Sib4Data {
        inter_freq_carriers,
    })
}

/// Build and encode a `SystemInformation` message to bytes.
pub fn encode_system_information(
    params: &SystemInformationParams,
) -> Result<Vec<u8>, SystemInformationError> {
    let msg = build_system_information(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a `SystemInformation` message from bytes.
pub fn decode_system_information(
    bytes: &[u8],
) -> Result<SystemInformationData, SystemInformationError> {
    let msg: BCCH_DL_SCH_Message = decode_rrc(bytes)?;
    parse_system_information(&msg)
}

/// Whether a BCCH-DL-SCH message is a `SystemInformation` (as opposed to a SIB1).
///
/// The two share a channel, so a receiver must dispatch on this rather than
/// assume: before SIB2/3/4 existed the UE's BCCH-DL-SCH handler called
/// `decode_sib1` unconditionally, which would report a decode failure for every
/// `SystemInformation` broadcast.
pub fn is_system_information(msg: &BCCH_DL_SCH_Message) -> bool {
    matches!(
        &msg.message,
        BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // MIB Tests
    // ========================================================================

    fn create_test_mib_params() -> MibParams {
        MibParams {
            system_frame_number: 10,
            sub_carrier_spacing_common: SubCarrierSpacingCommon::Scs15Or60,
            ssb_subcarrier_offset: 5,
            dmrs_type_a_position: DmrsTypeAPosition::Pos2,
            pdcch_config_sib1: PdcchConfigSib1Params {
                coreset_zero: 0,
                search_space_zero: 0,
            },
            cell_barred: CellBarredStatus::NotBarred,
            intra_freq_reselection: IntraFreqReselection::Allowed,
        }
    }

    #[test]
    fn test_build_mib() {
        let params = create_test_mib_params();
        let result = build_mib(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_mib(&msg));
    }

    #[test]
    fn test_parse_mib() {
        let params = create_test_mib_params();
        let msg = build_mib(&params).unwrap();
        let result = parse_mib(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.system_frame_number, params.system_frame_number);
        assert_eq!(
            data.sub_carrier_spacing_common,
            params.sub_carrier_spacing_common
        );
        assert_eq!(data.ssb_subcarrier_offset, params.ssb_subcarrier_offset);
        assert_eq!(data.dmrs_type_a_position, params.dmrs_type_a_position);
        assert_eq!(data.cell_barred, params.cell_barred);
        assert_eq!(data.intra_freq_reselection, params.intra_freq_reselection);
    }

    #[test]
    fn test_encode_decode_mib() {
        let params = create_test_mib_params();
        let encoded = encode_mib(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_mib(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.system_frame_number, params.system_frame_number);
    }

    #[test]
    fn test_mib_invalid_sfn() {
        let params = MibParams {
            system_frame_number: 64, // Invalid: must be 0-63
            ..create_test_mib_params()
        };
        let result = build_mib(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_mib_invalid_ssb_offset() {
        let params = MibParams {
            ssb_subcarrier_offset: 16, // Invalid: must be 0-15
            ..create_test_mib_params()
        };
        let result = build_mib(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_mib_all_scs_options() {
        for scs in [
            SubCarrierSpacingCommon::Scs15Or60,
            SubCarrierSpacingCommon::Scs30Or120,
        ] {
            let params = MibParams {
                sub_carrier_spacing_common: scs,
                ..create_test_mib_params()
            };
            let msg = build_mib(&params).unwrap();
            let data = parse_mib(&msg).unwrap();
            assert_eq!(data.sub_carrier_spacing_common, scs);
        }
    }

    #[test]
    fn test_mib_cell_barred_options() {
        for status in [CellBarredStatus::Barred, CellBarredStatus::NotBarred] {
            let params = MibParams {
                cell_barred: status,
                ..create_test_mib_params()
            };
            let msg = build_mib(&params).unwrap();
            let data = parse_mib(&msg).unwrap();
            assert_eq!(data.cell_barred, status);
        }
    }

    // ========================================================================
    // SIB1 Tests
    // ========================================================================

    fn create_test_sib1_params() -> Sib1Params {
        Sib1Params {
            ue_timers_and_constants: None,
            cell_selection_info: Some(CellSelectionInfo {
                q_rx_lev_min: -70,
                q_rx_lev_min_offset: None,
                q_rx_lev_min_sul: None,
                q_qual_min: None,
                q_qual_min_offset: None,
            }),
            plmn_identity_info_list: vec![PlmnIdentityInfo {
                plmn_identity_list: vec![PlmnIdentity {
                    mcc: Some([0, 0, 1]),
                    mnc: vec![0, 1],
                }],
                tracking_area_code: Some(0x000001),
                cell_identity: 0x123456789,
            }],
            ims_emergency_support: false,
            ecall_over_ims_support: false,
            intra_freq_reselection_redcap: false,
        }
    }

    #[test]
    fn test_sib1_intra_freq_reselection_redcap_roundtrip() {
        // intraFreqReselectionRedCap set: survives UPER encode/decode via the
        // SIB1 lateNonCriticalExtension octet container.
        let params = Sib1Params {
            intra_freq_reselection_redcap: true,
            ..create_test_sib1_params()
        };
        let bytes = encode_sib1(&params).unwrap();
        let data = decode_sib1(&bytes).unwrap();
        assert!(
            data.intra_freq_reselection_redcap,
            "intraFreqReselectionRedCap must round-trip"
        );

        // Not set: default decode yields false.
        let params_off = create_test_sib1_params();
        let bytes_off = encode_sib1(&params_off).unwrap();
        let data_off = decode_sib1(&bytes_off).unwrap();
        assert!(!data_off.intra_freq_reselection_redcap);
    }

    #[test]
    fn test_build_sib1() {
        let params = create_test_sib1_params();
        let result = build_sib1(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_sib1(&msg));
    }

    #[test]
    fn test_parse_sib1() {
        let params = create_test_sib1_params();
        let msg = build_sib1(&params).unwrap();
        let result = parse_sib1(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert!(data.cell_selection_info.is_some());
        assert_eq!(data.plmn_identity_info_list.len(), 1);
        assert_eq!(data.ims_emergency_support, params.ims_emergency_support);
    }

    #[test]
    fn test_encode_decode_sib1() {
        let params = create_test_sib1_params();
        let encoded = encode_sib1(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_sib1(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.plmn_identity_info_list.len(), 1);
    }

    #[test]
    fn test_sib1_empty_plmn_list() {
        let params = Sib1Params {
            plmn_identity_info_list: vec![], // Invalid: cannot be empty
            ..create_test_sib1_params()
        };
        let result = build_sib1(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_sib1_with_ims_support() {
        let params = Sib1Params {
            ims_emergency_support: true,
            ecall_over_ims_support: true,
            ..create_test_sib1_params()
        };
        let msg = build_sib1(&params).unwrap();
        let data = parse_sib1(&msg).unwrap();
        assert!(data.ims_emergency_support);
        assert!(data.ecall_over_ims_support);
    }

    #[test]
    fn test_sib1_multiple_plmns() {
        let params = Sib1Params {
            plmn_identity_info_list: vec![PlmnIdentityInfo {
                plmn_identity_list: vec![
                    PlmnIdentity {
                        mcc: Some([0, 0, 1]),
                        mnc: vec![0, 1],
                    },
                    PlmnIdentity {
                        mcc: Some([3, 1, 0]),
                        mnc: vec![2, 6, 0],
                    },
                ],
                tracking_area_code: Some(0x000001),
                cell_identity: 0x123456789,
            }],
            ..create_test_sib1_params()
        };
        let msg = build_sib1(&params).unwrap();
        let data = parse_sib1(&msg).unwrap();
        assert_eq!(data.plmn_identity_info_list[0].plmn_identity_list.len(), 2);
    }

    #[test]
    fn test_sib1_with_ue_timers_and_constants_roundtrip() {
        let timers = UeTimersAndConstantsParams {
            t300_ms: 400,
            t301_ms: 600,
            t310_ms: 2000,
            n310: 6,
            t311_ms: 5000,
            n311: 2,
            t319_ms: 1500,
        };
        let params = Sib1Params {
            ue_timers_and_constants: Some(timers),
            ..create_test_sib1_params()
        };
        let bytes = encode_sib1(&params).unwrap();
        let data = decode_sib1(&bytes).unwrap();
        assert_eq!(data.ue_timers_and_constants, Some(timers));
    }

    #[test]
    fn test_sib1_default_ue_timers_roundtrip() {
        let params = Sib1Params {
            ue_timers_and_constants: Some(UeTimersAndConstantsParams::default()),
            ..create_test_sib1_params()
        };
        let msg = build_sib1(&params).unwrap();
        let data = parse_sib1(&msg).unwrap();
        assert_eq!(
            data.ue_timers_and_constants,
            Some(UeTimersAndConstantsParams::default())
        );
    }

    #[test]
    fn test_sib1_rejects_invalid_timer_value() {
        let params = Sib1Params {
            ue_timers_and_constants: Some(UeTimersAndConstantsParams {
                t300_ms: 123, // not in the enumerated set
                ..UeTimersAndConstantsParams::default()
            }),
            ..create_test_sib1_params()
        };
        assert!(build_sib1(&params).is_err());
    }

    // ========================================================================
    // SIB2 / SIB3 / SIB4 Tests (issue #50)
    // ========================================================================

    fn test_sib2_params() -> Sib2Params {
        Sib2Params {
            q_hyst_db: 4,
            t_reselection_s: 1,
            cell_reselection_priority: 6,
            q_rx_lev_min: -70,
            s_intra_search_p: 31,
            thresh_serving_low_p: 4,
        }
    }

    #[test]
    fn a_system_information_carrying_sib2_round_trips() {
        let params = SystemInformationParams {
            sib2: Some(test_sib2_params()),
            sib3: None,
            sib4: None,
        };
        let bytes = encode_system_information(&params).expect("SIB2 encodes");
        let decoded = decode_system_information(&bytes).expect("SIB2 decodes");
        assert_eq!(
            decoded.sib2.expect("sib2 present"),
            Sib2Data {
                q_hyst_db: 4,
                t_reselection_s: 1,
                cell_reselection_priority: 6,
                q_rx_lev_min: -70,
                s_intra_search_p: 31,
                thresh_serving_low_p: 4,
            }
        );
    }

    #[test]
    fn a_system_information_carrying_sib3_round_trips_the_per_cell_offsets() {
        let params = SystemInformationParams {
            sib2: None,
            sib3: Some(Sib3Params {
                intra_freq_neighbours: vec![
                    IntraFreqNeighbour {
                        phys_cell_id: 1,
                        q_offset_db: 0,
                    },
                    // A NEGATIVE offset and a POSITIVE one, because the sign
                    // decides the direction of the R_n adjustment and a codec
                    // that dropped it would still round trip a 0.
                    IntraFreqNeighbour {
                        phys_cell_id: 511,
                        q_offset_db: -6,
                    },
                    IntraFreqNeighbour {
                        phys_cell_id: 1007,
                        q_offset_db: 12,
                    },
                ],
            }),
            sib4: None,
        };
        let bytes = encode_system_information(&params).expect("SIB3 encodes");
        let decoded = decode_system_information(&bytes).expect("SIB3 decodes");
        let sib3 = decoded.sib3.expect("sib3 present");
        assert_eq!(sib3.intra_freq_neighbours.len(), 3);
        assert_eq!(
            sib3.intra_freq_neighbours[0],
            IntraFreqNeighbour {
                phys_cell_id: 1,
                q_offset_db: 0
            }
        );
        assert_eq!(
            sib3.intra_freq_neighbours[1],
            IntraFreqNeighbour {
                phys_cell_id: 511,
                q_offset_db: -6
            }
        );
        assert_eq!(
            sib3.intra_freq_neighbours[2],
            IntraFreqNeighbour {
                phys_cell_id: 1007,
                q_offset_db: 12
            }
        );
    }

    #[test]
    fn a_system_information_carrying_sib4_round_trips_the_carrier_priorities() {
        let params = SystemInformationParams {
            sib2: None,
            sib3: None,
            sib4: Some(Sib4Params {
                inter_freq_carriers: vec![InterFreqCarrier {
                    dl_carrier_freq: 632628,
                    cell_reselection_priority: 3,
                    thresh_x_high_p: 10,
                    thresh_x_low_p: 4,
                    q_rx_lev_min: -70,
                    t_reselection_s: 2,
                }],
            }),
        };
        let bytes = encode_system_information(&params).expect("SIB4 encodes");
        let decoded = decode_system_information(&bytes).expect("SIB4 decodes");
        let sib4 = decoded.sib4.expect("sib4 present");
        assert_eq!(sib4.inter_freq_carriers.len(), 1);
        assert_eq!(sib4.inter_freq_carriers[0].dl_carrier_freq, 632628);
        assert_eq!(sib4.inter_freq_carriers[0].cell_reselection_priority, 3);
        assert_eq!(sib4.inter_freq_carriers[0].t_reselection_s, 2);
    }

    /// The three SIBs schedule together in one message, which is the case a
    /// receiver that dispatched on "the first entry" would get wrong.
    #[test]
    fn one_system_information_can_carry_sib2_sib3_and_sib4_together() {
        let params = SystemInformationParams {
            sib2: Some(test_sib2_params()),
            sib3: Some(Sib3Params {
                intra_freq_neighbours: vec![IntraFreqNeighbour {
                    phys_cell_id: 7,
                    q_offset_db: -3,
                }],
            }),
            sib4: Some(Sib4Params {
                inter_freq_carriers: vec![InterFreqCarrier {
                    dl_carrier_freq: 500000,
                    cell_reselection_priority: 1,
                    thresh_x_high_p: 8,
                    thresh_x_low_p: 2,
                    q_rx_lev_min: -70,
                    t_reselection_s: 0,
                }],
            }),
        };
        let bytes = encode_system_information(&params).expect("all three encode");
        let decoded = decode_system_information(&bytes).expect("all three decode");
        assert!(decoded.sib2.is_some(), "sib2 lost");
        assert!(decoded.sib3.is_some(), "sib3 lost");
        assert!(decoded.sib4.is_some(), "sib4 lost");
        assert!(!decoded.is_empty());
    }

    /// A `SystemInformation` and a SIB1 share BCCH-DL-SCH, so the receiver has to
    /// tell them apart by the CHOICE arm. Before SIB2/3/4 existed the UE called
    /// `decode_sib1` on every BCCH-DL-SCH PDU.
    #[test]
    fn system_information_and_sib1_are_distinguishable_on_the_shared_channel() {
        let si = build_system_information(&SystemInformationParams {
            sib2: Some(test_sib2_params()),
            sib3: None,
            sib4: None,
        })
        .expect("SI builds");
        let sib1 = build_sib1(&create_test_sib1_params()).expect("SIB1 builds");

        assert!(is_system_information(&si));
        assert!(!is_sib1(&si));
        assert!(is_sib1(&sib1));
        assert!(!is_system_information(&sib1));

        // And parsing one as the other is refused rather than yielding garbage.
        assert!(parse_system_information(&sib1).is_err());
        assert!(parse_sib1(&si).is_err());
    }

    /// `q-Hyst` and `q-OffsetRange` enumerate non-uniform dB steps, so an
    /// arithmetic mapping is wrong above the break. These pin the break points
    /// against the schema rather than against the encoder's own output.
    #[test]
    fn the_q_hyst_mapping_matches_the_asn_1_enumeration_including_its_2_db_steps() {
        // 1 dB steps up to 6: index == dB.
        for db in 0..=6 {
            assert_eq!(q_hyst_index(db).expect("legal"), db as u8, "q-Hyst {db} dB");
        }
        // Then 2 dB steps, where index != dB.
        assert_eq!(q_hyst_index(8).expect("legal"), 7);
        assert_eq!(q_hyst_index(24).expect("legal"), 15);
        // 7 dB is NOT in the enumeration, and is refused rather than rounded.
        assert!(
            q_hyst_index(7).is_err(),
            "7 dB must be refused, not rounded to 6 or 8"
        );
        assert!(q_hyst_index(25).is_err());
        assert!(q_hyst_index(-1).is_err());
        for (db, index) in Q_HYST_DB_TO_INDEX {
            assert_eq!(q_hyst_db(index).expect("legal"), db);
        }
    }

    #[test]
    fn the_q_offset_mapping_is_signed_and_matches_the_asn_1_enumeration() {
        assert_eq!(
            q_offset_index(0).expect("legal"),
            15,
            "0 dB is index 15, not 0"
        );
        assert_eq!(q_offset_index(-24).expect("legal"), 0);
        assert_eq!(q_offset_index(24).expect("legal"), 30);
        // -5..5 are 1 dB steps; outside that the steps are 2 dB, so -7 and 7 are
        // not enumerated.
        assert!(q_offset_index(-7).is_err());
        assert!(q_offset_index(7).is_err());
        for (db, index) in Q_OFFSET_DB_TO_INDEX {
            assert_eq!(q_offset_db(index).expect("legal"), db);
        }
    }

    /// The ASN.1 size bounds are `SIZE (1..)`, so an empty list is not
    /// encodable. Each is handled where it lands rather than producing a codec
    /// panic: SIB3 omits the optional field, SIB4 refuses (its list is mandatory).
    #[test]
    fn empty_lists_are_handled_rather_than_encoded_as_zero_length() {
        let sib3_only_empty = SystemInformationParams {
            sib2: None,
            sib3: Some(Sib3Params::default()),
            sib4: None,
        };
        let bytes = encode_system_information(&sib3_only_empty).expect("an empty SIB3 encodes");
        let decoded = decode_system_information(&bytes).expect("decodes");
        assert!(decoded
            .sib3
            .expect("sib3 present")
            .intra_freq_neighbours
            .is_empty());

        let sib4_empty = SystemInformationParams {
            sib2: None,
            sib3: None,
            sib4: Some(Sib4Params::default()),
        };
        assert!(encode_system_information(&sib4_empty).is_err());

        // And a message with no SIB at all is refused: sib-TypeAndInfo is SIZE (1..).
        assert!(encode_system_information(&SystemInformationParams::default()).is_err());
    }

    /// Out-of-range values are refused at build time rather than reaching the
    /// codec, which would panic or silently truncate.
    #[test]
    fn out_of_range_priorities_and_timers_are_refused() {
        let mut params = test_sib2_params();
        params.cell_reselection_priority = CELL_RESELECTION_PRIORITY_MAX + 1;
        assert!(encode_system_information(&SystemInformationParams {
            sib2: Some(params.clone()),
            sib3: None,
            sib4: None,
        })
        .is_err());

        let mut params = test_sib2_params();
        params.t_reselection_s = T_RESELECTION_MAX_S + 1;
        assert!(encode_system_information(&SystemInformationParams {
            sib2: Some(params),
            sib3: None,
            sib4: None,
        })
        .is_err());
    }
}
