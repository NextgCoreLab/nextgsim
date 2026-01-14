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
            control_resource_set_zero: ControlResourceSetZero(params.pdcch_config_sib1.coreset_zero),
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

/// Helper function to convert BitVec to u8
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
}

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
}

/// Build a SIB1 message
pub fn build_sib1(params: &Sib1Params) -> Result<BCCH_DL_SCH_Message, SystemInformationError> {
    if params.plmn_identity_info_list.is_empty() {
        return Err(SystemInformationError::InvalidFieldValue(
            "PLMN Identity Info List cannot be empty".to_string(),
        ));
    }

    // Build cell selection info if present
    let cell_selection_info = params.cell_selection_info.as_ref().map(|csi| {
        SIB1CellSelectionInfo {
            q_rx_lev_min: Q_RxLevMin(csi.q_rx_lev_min),
            q_rx_lev_min_offset: csi.q_rx_lev_min_offset.map(SIB1CellSelectionInfoQ_RxLevMinOffset),
            q_rx_lev_min_sul: csi.q_rx_lev_min_sul.map(Q_RxLevMin),
            q_qual_min: csi.q_qual_min.map(Q_QualMin),
            q_qual_min_offset: csi.q_qual_min_offset.map(SIB1CellSelectionInfoQ_QualMinOffset),
        }
    });

    // Build PLMN identity info list
    let plmn_identity_list: Vec<PLMN_IdentityInfo> = params
        .plmn_identity_info_list
        .iter()
        .map(|info| build_plmn_identity_info(info))
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
        ue_timers_and_constants: None,
        uac_barring_info: None,
        use_full_resume_id: None,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    Ok(BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(sib1)),
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
    let cell_selection_info = sib1.cell_selection_info.as_ref().map(|csi| CellSelectionInfo {
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
        .map(|info| parse_plmn_identity_info(info))
        .collect();

    // Parse IMS emergency support
    let ims_emergency_support = sib1.ims_emergency_support.is_some();

    // Parse eCall over IMS support
    let ecall_over_ims_support = sib1.e_call_over_ims_support.is_some();

    Ok(Sib1Data {
        cell_selection_info,
        plmn_identity_info_list,
        ims_emergency_support,
        ecall_over_ims_support,
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
    let tracking_area_code = info.tracking_area_code.as_ref().map(|tac| bitvec_to_u32(&tac.0));

    // Parse cell identity (36 bits)
    let cell_identity = bitvec_to_u64(&info.cell_identity.0);

    PlmnIdentityInfo {
        plmn_identity_list,
        tracking_area_code,
        cell_identity,
    }
}

/// Helper function to convert BitVec to u32
fn bitvec_to_u32(bv: &BitVec<u8, Msb0>) -> u32 {
    let mut value: u32 = 0;
    for bit in bv.iter() {
        value = (value << 1) | (*bit as u32);
    }
    value
}

/// Helper function to convert BitVec to u64
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
        assert_eq!(data.sub_carrier_spacing_common, params.sub_carrier_spacing_common);
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
        for scs in [SubCarrierSpacingCommon::Scs15Or60, SubCarrierSpacingCommon::Scs30Or120] {
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
        }
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
            plmn_identity_info_list: vec![
                PlmnIdentityInfo {
                    plmn_identity_list: vec![
                        PlmnIdentity { mcc: Some([0, 0, 1]), mnc: vec![0, 1] },
                        PlmnIdentity { mcc: Some([3, 1, 0]), mnc: vec![2, 6, 0] },
                    ],
                    tracking_area_code: Some(0x000001),
                    cell_identity: 0x123456789,
                },
            ],
            ..create_test_sib1_params()
        };
        let msg = build_sib1(&params).unwrap();
        let data = parse_sib1(&msg).unwrap();
        assert_eq!(data.plmn_identity_info_list[0].plmn_identity_list.len(), 2);
    }
}
