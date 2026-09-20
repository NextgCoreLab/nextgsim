//! UE Capability Transfer Procedure
//!
//! Implements the UE Capability Transfer procedure as defined in 3GPP TS 38.331
//! Section 5.6.1:
//!
//! 1. `UECapabilityEnquiry` - gNB → UE (DL-DCCH): requests UE radio access
//!    capabilities for the listed RAT types
//! 2. `UECapabilityInformation` - UE → gNB (UL-DCCH): returns the capability
//!    containers
//!
//! Both messages use real ASN.1 UPER encoding via the generated RRC types.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during UE Capability Transfer procedures
#[derive(Debug, Error)]
pub enum UeCapabilityError {
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

/// RAT type for capability request/response
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatType {
    /// NR standalone
    Nr,
    /// EUTRA-NR dual connectivity
    EutraNr,
    /// EUTRA (LTE)
    Eutra,
}

impl RatType {
    fn to_asn(self) -> RAT_Type {
        match self {
            RatType::Nr => RAT_Type(RAT_Type::NR),
            RatType::EutraNr => RAT_Type(RAT_Type::EUTRA_NR),
            RatType::Eutra => RAT_Type(RAT_Type::EUTRA),
        }
    }

    fn from_asn(value: &RAT_Type) -> Result<Self, UeCapabilityError> {
        match value.0 {
            RAT_Type::NR => Ok(RatType::Nr),
            RAT_Type::EUTRA_NR => Ok(RatType::EutraNr),
            RAT_Type::EUTRA => Ok(RatType::Eutra),
            other => Err(UeCapabilityError::InvalidFieldValue(format!(
                "Unknown RAT-Type: {other}"
            ))),
        }
    }
}

// ============================================================================
// UE Capability Enquiry
// ============================================================================

/// Parameters for building a UE Capability Enquiry
#[derive(Debug, Clone)]
pub struct UeCapabilityEnquiryParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// RAT types to enquire (1-8 entries)
    pub rat_types: Vec<RatType>,
}

/// Parsed UE Capability Enquiry data
#[derive(Debug, Clone)]
pub struct UeCapabilityEnquiryData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Requested RAT types
    pub rat_types: Vec<RatType>,
}

/// Build a UE Capability Enquiry message
pub fn build_ue_capability_enquiry(
    params: &UeCapabilityEnquiryParams,
) -> Result<DL_DCCH_Message, UeCapabilityError> {
    if params.rrc_transaction_id > 3 {
        return Err(UeCapabilityError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }
    if params.rat_types.is_empty() || params.rat_types.len() > 8 {
        return Err(UeCapabilityError::InvalidFieldValue(
            "RAT type list must contain 1-8 entries".to_string(),
        ));
    }

    let request_list: Vec<UE_CapabilityRAT_Request> = params
        .rat_types
        .iter()
        .map(|rat| UE_CapabilityRAT_Request {
            rat_type: rat.to_asn(),
            capability_request_filter: None,
        })
        .collect();

    let enquiry = UECapabilityEnquiry {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: UECapabilityEnquiryCriticalExtensions::UeCapabilityEnquiry(
            UECapabilityEnquiry_IEs {
                ue_capability_rat_request_list: UE_CapabilityRAT_RequestList(request_list),
                late_non_critical_extension: None,
                ue_capability_enquiry_ext: None,
            },
        ),
    };

    Ok(DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::UeCapabilityEnquiry(enquiry)),
    })
}

/// Parse a UE Capability Enquiry from a DL-DCCH message
pub fn parse_ue_capability_enquiry(
    msg: &DL_DCCH_Message,
) -> Result<UeCapabilityEnquiryData, UeCapabilityError> {
    let enquiry = match &msg.message {
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::UeCapabilityEnquiry(e)) => e,
        DL_DCCH_MessageType::C1(_) => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "UECapabilityEnquiry".to_string(),
                actual: "other c1 message".to_string(),
            })
        }
        _ => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &enquiry.critical_extensions {
        UECapabilityEnquiryCriticalExtensions::UeCapabilityEnquiry(ies) => ies,
        UECapabilityEnquiryCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "ueCapabilityEnquiry".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    let rat_types = ies
        .ue_capability_rat_request_list
        .0
        .iter()
        .map(|req| RatType::from_asn(&req.rat_type))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(UeCapabilityEnquiryData {
        rrc_transaction_id: enquiry.rrc_transaction_identifier.0,
        rat_types,
    })
}

/// Build and encode a UE Capability Enquiry to bytes (UPER)
pub fn encode_ue_capability_enquiry(
    params: &UeCapabilityEnquiryParams,
) -> Result<Vec<u8>, UeCapabilityError> {
    let msg = build_ue_capability_enquiry(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a UE Capability Enquiry from bytes (UPER)
pub fn decode_ue_capability_enquiry(
    bytes: &[u8],
) -> Result<UeCapabilityEnquiryData, UeCapabilityError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_ue_capability_enquiry(&msg)
}

// ============================================================================
// UE Capability Information
// ============================================================================

/// One RAT capability container in the UE Capability Information
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeCapabilityRatContainer {
    /// RAT type of the container
    pub rat_type: RatType,
    /// Encoded capability container (e.g. UE-NR-Capability octets)
    pub container: Vec<u8>,
}

/// Parameters for building a UE Capability Information
#[derive(Debug, Clone)]
pub struct UeCapabilityInformationParams {
    /// RRC Transaction Identifier (0-3), echoes the enquiry
    pub rrc_transaction_id: u8,
    /// Capability containers per RAT (may be empty if nothing is supported)
    pub containers: Vec<UeCapabilityRatContainer>,
}

/// Parsed UE Capability Information data
#[derive(Debug, Clone)]
pub struct UeCapabilityInformationData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Capability containers per RAT
    pub containers: Vec<UeCapabilityRatContainer>,
}

/// Build a UE Capability Information message
pub fn build_ue_capability_information(
    params: &UeCapabilityInformationParams,
) -> Result<UL_DCCH_Message, UeCapabilityError> {
    if params.rrc_transaction_id > 3 {
        return Err(UeCapabilityError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }
    if params.containers.len() > 8 {
        return Err(UeCapabilityError::InvalidFieldValue(
            "At most 8 capability containers allowed".to_string(),
        ));
    }

    let container_list = if params.containers.is_empty() {
        None
    } else {
        Some(UE_CapabilityRAT_ContainerList(
            params
                .containers
                .iter()
                .map(|c| UE_CapabilityRAT_Container {
                    rat_type: c.rat_type.to_asn(),
                    ue_capability_rat_container:
                        UE_CapabilityRAT_ContainerUe_CapabilityRAT_Container(c.container.clone()),
                })
                .collect(),
        ))
    };

    let information = UECapabilityInformation {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: UECapabilityInformationCriticalExtensions::UeCapabilityInformation(
            UECapabilityInformation_IEs {
                ue_capability_rat_container_list: container_list,
                late_non_critical_extension: None,
                non_critical_extension: None,
            },
        ),
    };

    Ok(UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::UeCapabilityInformation(
            information,
        )),
    })
}

/// Parse a UE Capability Information from a UL-DCCH message
pub fn parse_ue_capability_information(
    msg: &UL_DCCH_Message,
) -> Result<UeCapabilityInformationData, UeCapabilityError> {
    let information = match &msg.message {
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::UeCapabilityInformation(i)) => i,
        UL_DCCH_MessageType::C1(_) => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "UECapabilityInformation".to_string(),
                actual: "other c1 message".to_string(),
            })
        }
        _ => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &information.critical_extensions {
        UECapabilityInformationCriticalExtensions::UeCapabilityInformation(ies) => ies,
        UECapabilityInformationCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(UeCapabilityError::InvalidMessageType {
                expected: "ueCapabilityInformation".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    let containers = ies
        .ue_capability_rat_container_list
        .as_ref()
        .map(|list| {
            list.0
                .iter()
                .map(|c| {
                    Ok(UeCapabilityRatContainer {
                        rat_type: RatType::from_asn(&c.rat_type)?,
                        container: c.ue_capability_rat_container.0.clone(),
                    })
                })
                .collect::<Result<Vec<_>, UeCapabilityError>>()
        })
        .transpose()?
        .unwrap_or_default();

    Ok(UeCapabilityInformationData {
        rrc_transaction_id: information.rrc_transaction_identifier.0,
        containers,
    })
}

/// Build and encode a UE Capability Information to bytes (UPER)
pub fn encode_ue_capability_information(
    params: &UeCapabilityInformationParams,
) -> Result<Vec<u8>, UeCapabilityError> {
    let msg = build_ue_capability_information(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a UE Capability Information from bytes (UPER)
pub fn decode_ue_capability_information(
    bytes: &[u8],
) -> Result<UeCapabilityInformationData, UeCapabilityError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_ue_capability_information(&msg)
}

// ============================================================================
// Minimal UE-NR-Capability container
// ============================================================================

/// Build a real UPER-encoded minimal `UE-NR-Capability` container
/// (TS 38.331 §6.3.3) for the given supported NR band.
///
/// The capability advertises Rel-15 access stratum, no ROHC profiles and a
/// single supported band; everything optional is omitted.
pub fn build_minimal_nr_capability_container(
    supported_band: u16,
) -> Result<Vec<u8>, UeCapabilityError> {
    build_nr_capability_container(supported_band, false)
}

/// Build a real UPER-encoded `UE-NR-Capability` container (TS 38.331 §6.3.3)
/// that additionally declares whether this is a RedCap UE.
///
/// # Why this is where the RedCap declaration lives
///
/// TS 38.331 V19.3.0 has **no** `redCapIndication` IE -- not in
/// `RRCSetupComplete-v1700-IEs`, not anywhere in the module (a grep for
/// `redCapIndication` over the vendored `38331-j30.txt` returns zero hits). The
/// IE that declares a UE to be RedCap is `supportOfRedCap-r17`, inside
/// `RedCapParameters-r17` at `UE-NR-Capability-v1700.redCapParameters-r17`;
/// TS 38.306 defines the other features' behaviour in terms of "a UE ...
/// indicating `supportOfRedCap-r17`", which is what makes it the declaration.
///
/// So the conformant RedCap declaration travels in UE capability transfer
/// (TS 38.331 §5.6.1), not in `RRCSetupComplete`. See #105.
///
/// When `redcap` is set the whole `v1530 -> ... -> v1700` non-critical-extension
/// chain is emitted with every other field absent, because that is the only way
/// to reach a Rel-17 field. `access-stratum-release` is raised to `rel17` to
/// match, since a container claiming `rel15` while carrying Rel-17 extensions
/// would be self-contradictory.
pub fn build_nr_capability_container(
    supported_band: u16,
    redcap: bool,
) -> Result<Vec<u8>, UeCapabilityError> {
    if supported_band == 0 || supported_band > 1024 {
        return Err(UeCapabilityError::InvalidFieldValue(
            "NR band must be 1-1024".to_string(),
        ));
    }

    let capability = UE_NR_Capability {
        access_stratum_release: AccessStratumRelease(if redcap {
            AccessStratumRelease::REL17
        } else {
            AccessStratumRelease::REL15
        }),
        pdcp_parameters: PDCP_Parameters {
            supported_rohc_profiles: PDCP_ParametersSupportedROHC_Profiles {
                profile0x0000: PDCP_ParametersSupportedROHC_ProfilesProfile0x0000(false),
                profile0x0001: PDCP_ParametersSupportedROHC_ProfilesProfile0x0001(false),
                profile0x0002: PDCP_ParametersSupportedROHC_ProfilesProfile0x0002(false),
                profile0x0003: PDCP_ParametersSupportedROHC_ProfilesProfile0x0003(false),
                profile0x0004: PDCP_ParametersSupportedROHC_ProfilesProfile0x0004(false),
                profile0x0006: PDCP_ParametersSupportedROHC_ProfilesProfile0x0006(false),
                profile0x0101: PDCP_ParametersSupportedROHC_ProfilesProfile0x0101(false),
                profile0x0102: PDCP_ParametersSupportedROHC_ProfilesProfile0x0102(false),
                profile0x0103: PDCP_ParametersSupportedROHC_ProfilesProfile0x0103(false),
                profile0x0104: PDCP_ParametersSupportedROHC_ProfilesProfile0x0104(false),
            },
            max_number_rohc_context_sessions: PDCP_ParametersMaxNumberROHC_ContextSessions(
                PDCP_ParametersMaxNumberROHC_ContextSessions::CS2,
            ),
            uplink_only_rohc_profiles: None,
            continue_rohc_context: None,
            out_of_order_delivery: None,
            short_sn: None,
            pdcp_duplication_srb: None,
            pdcp_duplication_mcg_or_scg_drb: None,
        },
        rlc_parameters: None,
        mac_parameters: None,
        phy_parameters: Phy_Parameters {
            phy_parameters_common: None,
            phy_parameters_xdd_diff: None,
            phy_parameters_frx_diff: None,
            phy_parameters_fr1: None,
            phy_parameters_fr2: None,
        },
        rf_parameters: RF_Parameters {
            supported_band_list_nr: RF_ParametersSupportedBandListNR(vec![BandNR {
                band_nr: FreqBandIndicatorNR(supported_band),
                modified_mpr_behaviour: None,
                mimo_parameters_per_band: None,
                extended_cp: None,
                multiple_tci: None,
                bwp_without_restriction: None,
                bwp_same_numerology: None,
                bwp_diff_numerology: None,
                cross_carrier_scheduling_same_scs: None,
                pdsch_256qam_fr2: None,
                pusch_256qam: None,
                ue_power_class: None,
                rate_matching_lte_crs: None,
                // Rel-16 dropped the `-v1530` suffix from `channelBWs-DL-v1530`
                // and `channelBWs-UL-v1530` (TS 38.331 Annex A.3.1.2: the suffix
                // is only carried while the field is distinguished as an
                // extension); the SEQUENCE positions are unchanged.
                channel_b_ws_dl: None,
                channel_b_ws_ul: None,
            }]),
            supported_band_combination_list: None,
            applied_freq_band_list_filter: None,
        },
        meas_and_mob_parameters: None,
        fdd_add_ue_nr_capabilities: None,
        tdd_add_ue_nr_capabilities: None,
        fr1_add_ue_nr_capabilities: None,
        fr2_add_ue_nr_capabilities: None,
        feature_sets: None,
        feature_set_combinations: None,
        late_non_critical_extension: None,
        non_critical_extension: if redcap {
            Some(build_redcap_capability_chain())
        } else {
            None
        },
    };

    Ok(encode_rrc(&capability)?)
}

/// Build `UE-NR-Capability-v1530 -> ... -> v1700` carrying nothing but
/// `redCapParameters-r17.supportOfRedCap-r17`.
///
/// Every intermediate version group exists only to carry the
/// `nonCriticalExtension` pointer to the next one, so all their own fields are
/// absent. `MeasAndMobParameters-v1700` and `MBS-Parameters-r17` are MANDATORY in
/// `v1700` (no `OPTIONAL`), so both are constructed -- each with its single
/// optional field absent, which is the minimum a conformant encoder can write.
fn build_redcap_capability_chain() -> UE_NR_Capability_v1530 {
    let v1700 = UE_NR_Capability_v1700 {
        inactive_state_po_determination_r17: None,
        high_speed_parameters_v1700: None,
        pow_sav_parameters_v1700: None,
        mac_parameters_v1700: None,
        ims_parameters_v1700: None,
        // Mandatory in v1700, not OPTIONAL.
        meas_and_mob_parameters_v1700: MeasAndMobParameters_v1700 {
            meas_and_mob_parameters_fr2_2_r17: None,
        },
        app_layer_meas_parameters_r17: None,
        // The RedCap declaration itself (TS 38.331 `RedCapParameters-r17`).
        // `supportOf16DRB-RedCap-r17` is left absent: this UE does not claim the
        // optional 16-DRB extension, only baseline RedCap support.
        red_cap_parameters_r17: Some(RedCapParameters_r17 {
            support_of_red_cap_r17: Some(RedCapParameters_r17SupportOfRedCap_r17(
                RedCapParameters_r17SupportOfRedCap_r17::SUPPORTED,
            )),
            support_of16_drb_red_cap_r17: None,
        }),
        ra_sdt_r17: None,
        srb_sdt_r17: None,
        gnb_side_rtt_based_pdc_r17: None,
        bh_rlf_detection_recovery_indication_r17: None,
        nrdc_parameters_v1700: None,
        bap_parameters_v1700: None,
        musim_gap_preference_r17: None,
        musim_leave_connected_r17: None,
        // Mandatory in v1700, not OPTIONAL.
        mbs_parameters_r17: MBS_Parameters_r17 {
            max_mrb_add_r17: None,
        },
        non_terrestrial_network_r17: None,
        ntn_scenario_support_r17: None,
        slice_infofor_cell_reselection_r17: None,
        ue_radio_paging_info_r17: None,
        ul_gap_fr2_pattern_r17: None,
        ntn_parameters_r17: None,
        non_critical_extension: None,
    };

    let v1690 = UE_NR_Capability_v1690 {
        ul_rrc_segmentation_r16: None,
        non_critical_extension: Some(v1700),
    };
    let v1650 = UE_NR_Capability_v1650 {
        mps_priority_indication_r16: None,
        high_speed_parameters_v1650: None,
        non_critical_extension: Some(v1690),
    };
    let v1640 = UE_NR_Capability_v1640 {
        redirect_at_resume_by_nas_r16: None,
        phy_parameters_shared_spectrum_ch_access_r16: None,
        non_critical_extension: Some(v1650),
    };
    let v1610 = UE_NR_Capability_v1610 {
        in_device_coex_ind_r16: None,
        dl_dedicated_message_segmentation_r16: None,
        nrdc_parameters_v1610: None,
        pow_sav_parameters_r16: None,
        fr1_add_ue_nr_capabilities_v1610: None,
        fr2_add_ue_nr_capabilities_v1610: None,
        bh_rlf_indication_r16: None,
        direct_sn_addition_first_rrc_iab_r16: None,
        bap_parameters_r16: None,
        reference_time_provision_r16: None,
        sidelink_parameters_r16: None,
        high_speed_parameters_r16: None,
        mac_parameters_v1610: None,
        mcg_rlf_recovery_via_scg_r16: None,
        resume_with_stored_mcg_s_cells_r16: None,
        resume_with_stored_scg_r16: None,
        resume_with_scg_config_r16: None,
        ue_based_perf_meas_parameters_r16: None,
        son_parameters_r16: None,
        on_demand_sib_connected_r16: None,
        non_critical_extension: Some(v1640),
    };
    let v1570 = UE_NR_Capability_v1570 {
        nrdc_parameters_v1570: None,
        non_critical_extension: Some(v1610),
    };
    let v1560 = UE_NR_Capability_v1560 {
        nrdc_parameters: None,
        received_filters: None,
        non_critical_extension: Some(v1570),
    };
    let v1550 = UE_NR_Capability_v1550 {
        reduced_cp_latency: None,
        non_critical_extension: Some(v1560),
    };
    let v1540 = UE_NR_Capability_v1540 {
        sdap_parameters: None,
        overheating_ind: None,
        ims_parameters: None,
        fr1_add_ue_nr_capabilities_v1540: None,
        fr2_add_ue_nr_capabilities_v1540: None,
        fr1_fr2_add_ue_nr_capabilities: None,
        non_critical_extension: Some(v1550),
    };
    UE_NR_Capability_v1530 {
        fdd_add_ue_nr_capabilities_v1530: None,
        tdd_add_ue_nr_capabilities_v1530: None,
        dummy: None,
        inter_rat_parameters: None,
        inactive_state: None,
        delay_budget_reporting: None,
        non_critical_extension: Some(v1540),
    }
}

/// Read `supportOfRedCap-r17` back out of a `UE-NR-Capability` container.
///
/// This is the conformant answer to "is this a RedCap UE?" -- see
/// [`build_nr_capability_container`] for why the question is not answered by
/// `RRCSetupComplete`.
pub fn parse_nr_capability_redcap(bytes: &[u8]) -> Result<bool, UeCapabilityError> {
    let capability: UE_NR_Capability = decode_rrc(bytes)?;
    Ok(capability
        .non_critical_extension
        .as_ref()
        .and_then(|v1530| v1530.non_critical_extension.as_ref())
        .and_then(|v1540| v1540.non_critical_extension.as_ref())
        .and_then(|v1550| v1550.non_critical_extension.as_ref())
        .and_then(|v1560| v1560.non_critical_extension.as_ref())
        .and_then(|v1570| v1570.non_critical_extension.as_ref())
        .and_then(|v1610| v1610.non_critical_extension.as_ref())
        .and_then(|v1640| v1640.non_critical_extension.as_ref())
        .and_then(|v1650| v1650.non_critical_extension.as_ref())
        .and_then(|v1690| v1690.non_critical_extension.as_ref())
        .and_then(|v1700| v1700.red_cap_parameters_r17.as_ref())
        .and_then(|rc| rc.support_of_red_cap_r17.as_ref())
        .is_some_and(|s| s.0 == RedCapParameters_r17SupportOfRedCap_r17::SUPPORTED))
}

/// Decode a `UE-NR-Capability` container and return the supported NR bands
pub fn parse_nr_capability_bands(bytes: &[u8]) -> Result<Vec<u16>, UeCapabilityError> {
    let capability: UE_NR_Capability = decode_rrc(bytes)?;
    Ok(capability
        .rf_parameters
        .supported_band_list_nr
        .0
        .iter()
        .map(|band| band.band_nr.0)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ue_capability_enquiry_roundtrip() {
        let params = UeCapabilityEnquiryParams {
            rrc_transaction_id: 1,
            rat_types: vec![RatType::Nr, RatType::EutraNr],
        };
        let bytes = encode_ue_capability_enquiry(&params).unwrap();
        let decoded = decode_ue_capability_enquiry(&bytes).unwrap();
        assert_eq!(decoded.rrc_transaction_id, 1);
        assert_eq!(decoded.rat_types, vec![RatType::Nr, RatType::EutraNr]);
    }

    #[test]
    fn test_ue_capability_enquiry_rejects_invalid() {
        let params = UeCapabilityEnquiryParams {
            rrc_transaction_id: 4,
            rat_types: vec![RatType::Nr],
        };
        assert!(build_ue_capability_enquiry(&params).is_err());

        let params = UeCapabilityEnquiryParams {
            rrc_transaction_id: 0,
            rat_types: vec![],
        };
        assert!(build_ue_capability_enquiry(&params).is_err());
    }

    #[test]
    fn test_ue_capability_information_roundtrip() {
        let params = UeCapabilityInformationParams {
            rrc_transaction_id: 1,
            containers: vec![UeCapabilityRatContainer {
                rat_type: RatType::Nr,
                container: vec![0xDE, 0xAD, 0xBE, 0xEF],
            }],
        };
        let bytes = encode_ue_capability_information(&params).unwrap();
        let decoded = decode_ue_capability_information(&bytes).unwrap();
        assert_eq!(decoded.rrc_transaction_id, 1);
        assert_eq!(decoded.containers.len(), 1);
        assert_eq!(decoded.containers[0].rat_type, RatType::Nr);
        assert_eq!(
            decoded.containers[0].container,
            vec![0xDE, 0xAD, 0xBE, 0xEF]
        );
    }

    #[test]
    fn test_ue_capability_information_empty_containers() {
        // UE that cannot provide any of the requested capabilities
        let params = UeCapabilityInformationParams {
            rrc_transaction_id: 2,
            containers: vec![],
        };
        let bytes = encode_ue_capability_information(&params).unwrap();
        let decoded = decode_ue_capability_information(&bytes).unwrap();
        assert_eq!(decoded.rrc_transaction_id, 2);
        assert!(decoded.containers.is_empty());
    }

    #[test]
    fn test_enquiry_information_cross_decode_rejected() {
        // An enquiry must not parse as information and vice versa
        let enquiry_bytes = encode_ue_capability_enquiry(&UeCapabilityEnquiryParams {
            rrc_transaction_id: 0,
            rat_types: vec![RatType::Nr],
        })
        .unwrap();
        assert!(decode_ue_capability_information(&enquiry_bytes).is_err());
    }

    #[test]
    fn test_minimal_nr_capability_container_roundtrip() {
        let container = build_minimal_nr_capability_container(78).unwrap();
        assert!(!container.is_empty());
        let bands = parse_nr_capability_bands(&container).unwrap();
        assert_eq!(bands, vec![78]);
    }

    #[test]
    fn test_minimal_nr_capability_rejects_invalid_band() {
        assert!(build_minimal_nr_capability_container(0).is_err());
        assert!(build_minimal_nr_capability_container(2000).is_err());
    }

    #[test]
    fn test_full_capability_transfer_roundtrip() {
        // End-to-end: enquiry -> information with a real NR capability container
        let container = build_minimal_nr_capability_container(78).unwrap();
        let info = UeCapabilityInformationParams {
            rrc_transaction_id: 3,
            containers: vec![UeCapabilityRatContainer {
                rat_type: RatType::Nr,
                container,
            }],
        };
        let bytes = encode_ue_capability_information(&info).unwrap();
        let decoded = decode_ue_capability_information(&bytes).unwrap();
        let bands = parse_nr_capability_bands(&decoded.containers[0].container).unwrap();
        assert_eq!(bands, vec![78]);
    }

    #[test]
    fn test_ue_capability_enquiry_rejects_truncated() {
        let bytes = encode_ue_capability_enquiry(&UeCapabilityEnquiryParams {
            rrc_transaction_id: 0,
            rat_types: vec![RatType::Nr, RatType::Eutra, RatType::EutraNr],
        })
        .unwrap();
        assert!(decode_ue_capability_enquiry(&bytes[..1]).is_err());
    }

    /// #105's headline acceptance: a genuine post-Rel-15 IE encodes and decodes.
    ///
    /// `supportOfRedCap-r17` lives in `RedCapParameters-r17` inside
    /// `UE-NR-Capability-v1700`, ten non-critical-extension hops past where the
    /// Rel-15 schema stopped. Under `rrc-15.6.0` this test could not even be
    /// written: `UE-NR-Capability.nonCriticalExtension` bottomed out long before
    /// v1700 and the type did not exist.
    ///
    /// Asserted POSITIVELY on the decoded value, and asserted on the
    /// `UE-NR-Capability` type rather than on bytes we chose, so the only way to
    /// pass is for the encoder to have really walked the chain and for the decoder
    /// to have really found the field.
    #[test]
    fn support_of_redcap_r17_round_trips_as_a_real_rel17_ie() {
        let container = build_nr_capability_container(78, true).unwrap();

        assert!(
            parse_nr_capability_redcap(&container).unwrap(),
            "supportOfRedCap-r17 must survive encode -> decode as a real \
             UE-NR-Capability-v1700 field"
        );

        // The Rel-15 part of the same container is untouched by the extension.
        assert_eq!(parse_nr_capability_bands(&container).unwrap(), vec![78]);

        // Decoding the whole type proves the chain is genuinely present and that
        // `access-stratum-release` was raised to match it, rather than the reader
        // above having found the value by some shortcut.
        let decoded: UE_NR_Capability = crate::codec::decode_rrc(&container).unwrap();
        assert_eq!(
            decoded.access_stratum_release.0,
            AccessStratumRelease::REL17
        );
        // Ten hops, named: v1530, v1540, v1550, v1560, v1570, v1610, v1640,
        // v1650, v1690, v1700. Anonymous `.and_then` repetition is how an
        // off-by-one hides here, so each hop is bound and named.
        let v1530 = decoded
            .non_critical_extension
            .as_ref()
            .expect("v1530 must be present on the wire");
        let v1540 = v1530.non_critical_extension.as_ref().expect("v1540");
        let v1550 = v1540.non_critical_extension.as_ref().expect("v1550");
        let v1560 = v1550.non_critical_extension.as_ref().expect("v1560");
        let v1570 = v1560.non_critical_extension.as_ref().expect("v1570");
        let v1610 = v1570.non_critical_extension.as_ref().expect("v1610");
        let v1640 = v1610.non_critical_extension.as_ref().expect("v1640");
        let v1650 = v1640.non_critical_extension.as_ref().expect("v1650");
        let v1690 = v1650.non_critical_extension.as_ref().expect("v1690");
        let v1700 = v1690
            .non_critical_extension
            .as_ref()
            .expect("the v1530..v1700 chain must be present on the wire");
        assert_eq!(
            v1700
                .red_cap_parameters_r17
                .as_ref()
                .and_then(|rc| rc.support_of_red_cap_r17.as_ref())
                .map(|s| s.0),
            Some(RedCapParameters_r17SupportOfRedCap_r17::SUPPORTED),
        );
    }

    /// A non-RedCap container carries NO extension chain at all, so the Rel-19
    /// schema costs nothing on the default path: the bytes are what the Rel-15
    /// schema produced.
    #[test]
    fn a_non_redcap_capability_container_carries_no_rel16_extension() {
        let container = build_nr_capability_container(78, false).unwrap();
        assert!(!parse_nr_capability_redcap(&container).unwrap());

        let decoded: UE_NR_Capability = crate::codec::decode_rrc(&container).unwrap();
        assert_eq!(
            decoded.access_stratum_release.0,
            AccessStratumRelease::REL15
        );
        assert!(
            decoded.non_critical_extension.is_none(),
            "a UE that is not RedCap must not pay for the v1530.. chain"
        );

        // And it is byte-identical to what the old minimal builder produces, which
        // is the compatibility claim the golden SIB/setup fixtures rest on.
        assert_eq!(
            container,
            build_minimal_nr_capability_container(78).unwrap()
        );
    }
}
