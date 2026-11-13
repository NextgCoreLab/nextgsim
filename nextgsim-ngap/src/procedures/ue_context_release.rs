//! UE Context Release Procedure
//!
//! Implements the UE Context Release procedure as defined in 3GPP TS 38.413 Section 8.3.3.
//! This procedure is used to release the UE context at the NG-RAN node.
//!
//! The procedure consists of:
//! - UE Context Release Request: Sent by NG-RAN to AMF to request release
//! - UE Context Release Command: Sent by AMF to NG-RAN to command release
//! - UE Context Release Complete: Sent by NG-RAN to AMF to confirm release

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::ng_setup::{
    MiscCause, NasCause, NgSetupFailureCause, ProtocolCause, RadioNetworkCause, TransportCause,
};
use thiserror::Error;

/// Errors that can occur during UE Context Release procedures
#[derive(Debug, Error)]
pub enum UeContextReleaseError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] NgapCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Missing mandatory IE
    #[error("Missing mandatory IE: {0}")]
    MissingMandatoryIe(String),

    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
}

/// Cause values for UE Context Release
pub type UeContextReleaseCause = NgSetupFailureCause;

// ============================================================================
// UE Context Release Request
// ============================================================================

/// Parameters for building a UE Context Release Request
#[derive(Debug, Clone)]
pub struct UeContextReleaseRequestParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// The UE's PDU sessions, sent as the PDU Session Resource List (IE 133,
    /// TS 38.413 §9.2.2.4) when non-empty. TS 23.502 §4.2.6 step 1: the NG-RAN
    /// lists the UE's active sessions so the AMF can release their user plane;
    /// leave empty for a UE with none (the IE is then absent, SIZE(1..256)).
    pub pdu_session_ids: Vec<u8>,
    /// Cause of release request
    pub cause: UeContextReleaseCause,
}

/// Parsed UE Context Release Request data
#[derive(Debug, Clone)]
pub struct UeContextReleaseRequestData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// PDU Session Resource List (IE 133), empty when absent
    pub pdu_session_ids: Vec<u8>,
    /// Cause of release request
    pub cause: UeContextReleaseCause,
}

/// Build a UE Context Release Request PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Request
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_request(
    params: &UeContextReleaseRequestParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = Vec::new();

    // IE: AMF-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(
            params.amf_ue_ngap_id,
        )),
    });

    // IE: RAN-UE-NGAP-ID (mandatory)
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(
            params.ran_ue_ngap_id,
        )),
    });

    // IE: PDU Session Resource List (optional, criticality reject), in
    // object-set order between the UE IDs and the Cause.
    if !params.pdu_session_ids.is_empty() {
        protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PDU_SESSION_RESOURCE_LIST_CXT_REL_REQ),
            criticality: Criticality(Criticality::REJECT),
            value:
                UEContextReleaseRequestProtocolIEs_EntryValue::Id_PDUSessionResourceListCxtRelReq(
                    PDUSessionResourceListCxtRelReq(
                        params
                            .pdu_session_ids
                            .iter()
                            .map(|&psi| PDUSessionResourceItemCxtRelReq {
                                pdu_session_id: PDUSessionID(psi),
                                ie_extensions: None,
                            })
                            .collect(),
                    ),
                ),
        });
    }

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(UEContextReleaseRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let request = UEContextReleaseRequest {
        protocol_i_es: UEContextReleaseRequestProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE_REQUEST),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_UEContextReleaseRequest(request),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// Parse a UE Context Release Request from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseRequestData)` - The parsed request data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_request(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseRequestData, UeContextReleaseError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let request = match &initiating_message.value {
        InitiatingMessageValue::Id_UEContextReleaseRequest(req) => req,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseRequest".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut cause: Option<UeContextReleaseCause> = None;
    let mut pdu_session_ids = Vec::new();

    for ie in &request.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
            UEContextReleaseRequestProtocolIEs_EntryValue::Id_PDUSessionResourceListCxtRelReq(
                list,
            ) => {
                pdu_session_ids = list.0.iter().map(|i| i.pdu_session_id.0).collect();
            }
        }
    }

    Ok(UeContextReleaseRequestData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        pdu_session_ids,
        cause: cause
            .ok_or_else(|| UeContextReleaseError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

// ============================================================================
// UE Context Release Command
// ============================================================================

/// UE NGAP IDs - can be either a pair or just AMF ID
#[derive(Debug, Clone)]
pub enum UeNgapIds {
    /// Both AMF and RAN UE NGAP IDs
    Pair {
        /// AMF UE NGAP ID
        amf_ue_ngap_id: u64,
        /// RAN UE NGAP ID
        ran_ue_ngap_id: u32,
    },
    /// Only AMF UE NGAP ID (when RAN ID is unknown)
    AmfOnly(u64),
}

/// Parameters for building a UE Context Release Command
#[derive(Debug, Clone)]
pub struct UeContextReleaseCommandParams {
    /// UE NGAP IDs
    pub ue_ngap_ids: UeNgapIds,
    /// Cause of release
    pub cause: UeContextReleaseCause,
}

/// Parsed UE Context Release Command data
#[derive(Debug, Clone)]
pub struct UeContextReleaseCommandData {
    /// UE NGAP IDs
    pub ue_ngap_ids: UeNgapIds,
    /// Cause of release
    pub cause: UeContextReleaseCause,
}

/// Build a UE Context Release Command PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Command
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_command(
    params: &UeContextReleaseCommandParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = Vec::new();

    // IE: UE-NGAP-IDs (mandatory)
    let ue_ngap_ids = build_ue_ngap_ids(&params.ue_ngap_ids);
    protocol_ies.push(UEContextReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_NGAP_I_DS),
        criticality: Criticality(Criticality::REJECT),
        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_NGAP_IDs(ue_ngap_ids),
    });

    // IE: Cause (mandatory)
    let cause = build_cause(&params.cause);
    protocol_ies.push(UEContextReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(cause),
    });

    let command = UEContextReleaseCommand {
        protocol_i_es: UEContextReleaseCommandProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UEContextRelease(command),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

fn build_ue_ngap_ids(ids: &UeNgapIds) -> UE_NGAP_IDs {
    match ids {
        UeNgapIds::Pair {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
        } => UE_NGAP_IDs::UE_NGAP_ID_pair(UE_NGAP_ID_pair {
            amf_ue_ngap_id: AMF_UE_NGAP_ID(*amf_ue_ngap_id),
            ran_ue_ngap_id: RAN_UE_NGAP_ID(*ran_ue_ngap_id),
            ie_extensions: None,
        }),
        UeNgapIds::AmfOnly(amf_id) => UE_NGAP_IDs::AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(*amf_id)),
    }
}

/// Parse a UE Context Release Command from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseCommandData)` - The parsed command data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_command(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseCommandData, UeContextReleaseError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let command = match &initiating_message.value {
        InitiatingMessageValue::Id_UEContextRelease(cmd) => cmd,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseCommand".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ue_ngap_ids: Option<UeNgapIds> = None;
    let mut cause: Option<UeContextReleaseCause> = None;

    for ie in &command.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_NGAP_IDs(ids) => {
                ue_ngap_ids = Some(parse_ue_ngap_ids(ids)?);
            }
            UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(c) => {
                cause = Some(parse_cause(c));
            }
        }
    }

    Ok(UeContextReleaseCommandData {
        ue_ngap_ids: ue_ngap_ids
            .ok_or_else(|| UeContextReleaseError::MissingMandatoryIe("UE-NGAP-IDs".to_string()))?,
        cause: cause
            .ok_or_else(|| UeContextReleaseError::MissingMandatoryIe("Cause".to_string()))?,
    })
}

fn parse_ue_ngap_ids(ids: &UE_NGAP_IDs) -> Result<UeNgapIds, UeContextReleaseError> {
    match ids {
        UE_NGAP_IDs::UE_NGAP_ID_pair(pair) => Ok(UeNgapIds::Pair {
            amf_ue_ngap_id: pair.amf_ue_ngap_id.0,
            ran_ue_ngap_id: pair.ran_ue_ngap_id.0,
        }),
        UE_NGAP_IDs::AMF_UE_NGAP_ID(amf_id) => Ok(UeNgapIds::AmfOnly(amf_id.0)),
        UE_NGAP_IDs::Choice_Extensions(_) => Err(UeContextReleaseError::InvalidIeValue(
            "Unsupported UE-NGAP-IDs choice extension".to_string(),
        )),
    }
}

// ============================================================================
// UE Context Release Complete
// ============================================================================

/// Parameters for building a UE Context Release Complete
#[derive(Debug, Clone)]
pub struct UeContextReleaseCompleteParams {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Information on Recommended Cells and RAN Nodes for Paging (TS 38.413
    /// §9.3.1.100), optional. `None` sends the two IDs only, byte-identical to
    /// the message before the IE was supported.
    pub recommended_paging: Option<RecommendedPagingInfo>,
}

/// Parsed UE Context Release Complete data
#[derive(Debug, Clone)]
pub struct UeContextReleaseCompleteData {
    /// AMF UE NGAP ID
    pub amf_ue_ngap_id: u64,
    /// RAN UE NGAP ID
    pub ran_ue_ngap_id: u32,
    /// Information on Recommended Cells and RAN Nodes for Paging, when present
    /// and of the shapes [`RecommendedPagingInfo`] models.
    pub recommended_paging: Option<RecommendedPagingInfo>,
}

/// The most cells or RAN nodes a recommendation may list
/// (`maxnoofRecommendedCells`, `maxnoofRecommendedRANNodes`, TS 38.413 §9.4.7).
pub const MAX_RECOMMENDED_PAGING_ITEMS: usize = 16;

/// Information on Recommended Cells and RAN Nodes for Paging (TS 38.413
/// §9.3.1.100). Both lists are mandatory, so each holds 1..=16 entries.
///
/// The order is the RAN's and is significant (§9.3.1.71, §9.3.1.101): visited
/// entries in the order the UE visited them, most recent first, each followed
/// by the non-visited entries associated with it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedPagingInfo {
    /// *Recommended Cells for Paging* (§9.3.1.71).
    pub cells: Vec<RecommendedCell>,
    /// *Recommended RAN Nodes for Paging* (§9.3.1.101).
    pub ran_nodes: Vec<RecommendedRanNode>,
}

/// One *Recommended Cell Item* (§9.3.1.71): an NR CGI (§9.3.1.7) as the
/// NG-RAN CGI (§9.3.1.73), and, for a visited cell, the time stayed in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecommendedCell {
    /// The cell's PLMN Identity, as encoded on the wire (3 octets, TS 38.413
    /// §9.3.3.5).
    pub plmn: [u8; 3],
    /// The NR Cell Identity, 36 bits.
    pub nci: u64,
    /// *Time Stayed in Cell* in seconds (0..=4095; a longer stay is sent as
    /// 4095). Included for visited cells only.
    pub time_stayed_s: Option<u16>,
}

/// One *Recommended RAN Node Item* (§9.3.1.101) whose AMF Paging Target is a
/// Global RAN Node ID (§9.3.1.5) of the Global gNB ID kind (§9.3.1.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecommendedRanNode {
    /// The gNB's PLMN Identity, as encoded on the wire.
    pub plmn: [u8; 3],
    /// The gNB ID, the leftmost `gnb_id_len` bits of its cells' NCIs.
    pub gnb_id: u32,
    /// The gNB ID length in bits, 22..=32.
    pub gnb_id_len: u8,
}

/// The `width` low bits of `value`, most significant first.
fn msb_bits(value: u64, width: usize) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0> {
    (0..width).rev().map(|i| (value >> i) & 1 == 1).collect()
}

/// A bit string read back as an unsigned integer, most significant bit first.
fn bits_value(bits: &bitvec::slice::BitSlice<u8, bitvec::order::Msb0>) -> u64 {
    bits.iter().fold(0u64, |v, b| (v << 1) | u64::from(*b))
}

/// The generated IE for `info`, checked against TS 38.413's bounds: 1..=16
/// cells and nodes, a 36-bit NCI, a 22..32-bit gNB ID that fits its length,
/// and a time stayed of at most 4095 s.
fn build_recommended_paging(
    info: &RecommendedPagingInfo,
) -> Result<InfoOnRecommendedCellsAndRANNodesForPaging, UeContextReleaseError> {
    let bad = |what: String| UeContextReleaseError::InvalidIeValue(what);
    for (name, n) in [
        ("cells", info.cells.len()),
        ("RAN nodes", info.ran_nodes.len()),
    ] {
        if !(1..=MAX_RECOMMENDED_PAGING_ITEMS).contains(&n) {
            return Err(bad(format!(
                "recommended paging lists {n} {name}; TS 38.413 §9.3.1.100 needs 1..={MAX_RECOMMENDED_PAGING_ITEMS}"
            )));
        }
    }
    let mut cells = Vec::with_capacity(info.cells.len());
    for c in &info.cells {
        if c.nci > 0xF_FFFF_FFFF {
            return Err(bad(format!(
                "recommended cell NCI {:#x} is wider than 36 bits",
                c.nci
            )));
        }
        if let Some(t) = c.time_stayed_s {
            if t > 4095 {
                return Err(bad(format!("time stayed in cell {t} s is above 4095")));
            }
        }
        cells.push(RecommendedCellItem {
            ngran_cgi: NGRAN_CGI::NR_CGI(NR_CGI {
                plmn_identity: PLMNIdentity(c.plmn.to_vec()),
                nr_cell_identity: NRCellIdentity(msb_bits(c.nci, 36)),
                ie_extensions: None,
            }),
            time_stayed_in_cell: c.time_stayed_s.map(RecommendedCellItemTimeStayedInCell),
            ie_extensions: None,
        });
    }
    let mut nodes = Vec::with_capacity(info.ran_nodes.len());
    for n in &info.ran_nodes {
        if !(22..=32).contains(&n.gnb_id_len) {
            return Err(bad(format!(
                "recommended gNB ID length {} is outside 22..32 (TS 38.413 §9.3.1.6)",
                n.gnb_id_len
            )));
        }
        if u64::from(n.gnb_id) >> n.gnb_id_len != 0 {
            return Err(bad(format!(
                "recommended gNB ID {:#x} does not fit {} bits",
                n.gnb_id, n.gnb_id_len
            )));
        }
        nodes.push(RecommendedRANNodeItem {
            amf_paging_target: AMFPagingTarget::GlobalRANNodeID(GlobalRANNodeID::GlobalGNB_ID(
                GlobalGNB_ID {
                    plmn_identity: PLMNIdentity(n.plmn.to_vec()),
                    gnb_id: GNB_ID::GNB_ID(GNB_ID_gNB_ID(msb_bits(
                        u64::from(n.gnb_id),
                        usize::from(n.gnb_id_len),
                    ))),
                    ie_extensions: None,
                },
            )),
            ie_extensions: None,
        });
    }
    Ok(InfoOnRecommendedCellsAndRANNodesForPaging {
        recommended_cells_for_paging: RecommendedCellsForPaging {
            recommended_cell_list: RecommendedCellList(cells),
            ie_extensions: None,
        },
        recommend_ran_nodes_for_paging: RecommendedRANNodesForPaging {
            recommended_ran_node_list: RecommendedRANNodeList(nodes),
            ie_extensions: None,
        },
        ie_extensions: None,
    })
}

/// The IE read back into [`RecommendedPagingInfo`]. Entries of a kind it does
/// not model (an E-UTRA CGI, an ng-eNB, N3IWF or TAI target, a choice
/// extension) are left out; `None` when either list is then empty.
fn parse_recommended_paging(
    ie: &InfoOnRecommendedCellsAndRANNodesForPaging,
) -> Option<RecommendedPagingInfo> {
    let plmn = |p: &PLMNIdentity| <[u8; 3]>::try_from(p.0.as_slice()).ok();
    let cells: Vec<RecommendedCell> = ie
        .recommended_cells_for_paging
        .recommended_cell_list
        .0
        .iter()
        .filter_map(|c| match &c.ngran_cgi {
            NGRAN_CGI::NR_CGI(cgi) => Some(RecommendedCell {
                plmn: plmn(&cgi.plmn_identity)?,
                nci: bits_value(&cgi.nr_cell_identity.0),
                time_stayed_s: c.time_stayed_in_cell.as_ref().map(|t| t.0),
            }),
            _ => None,
        })
        .collect();
    let ran_nodes: Vec<RecommendedRanNode> = ie
        .recommend_ran_nodes_for_paging
        .recommended_ran_node_list
        .0
        .iter()
        .filter_map(|n| match &n.amf_paging_target {
            AMFPagingTarget::GlobalRANNodeID(GlobalRANNodeID::GlobalGNB_ID(g)) => match &g.gnb_id {
                GNB_ID::GNB_ID(id) => Some(RecommendedRanNode {
                    plmn: plmn(&g.plmn_identity)?,
                    gnb_id: bits_value(&id.0) as u32,
                    gnb_id_len: id.0.len() as u8,
                }),
                _ => None,
            },
            _ => None,
        })
        .collect();
    (!cells.is_empty() && !ran_nodes.is_empty())
        .then_some(RecommendedPagingInfo { cells, ran_nodes })
}

/// Build a UE Context Release Complete PDU
///
/// # Arguments
/// * `params` - Parameters for the UE Context Release Complete
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(UeContextReleaseError)` - If construction fails
pub fn build_ue_context_release_complete(
    params: &UeContextReleaseCompleteParams,
) -> Result<NGAP_PDU, UeContextReleaseError> {
    let mut protocol_ies = vec![
        // IE: AMF-UE-NGAP-ID (mandatory)
        UEContextReleaseCompleteProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_AMF_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(
                AMF_UE_NGAP_ID(params.amf_ue_ngap_id),
            ),
        },
        // IE: RAN-UE-NGAP-ID (mandatory)
        UEContextReleaseCompleteProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RAN_UE_NGAP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(
                RAN_UE_NGAP_ID(params.ran_ue_ngap_id),
            ),
        },
    ];
    // IE: Information on Recommended Cells and RAN Nodes for Paging (optional,
    // criticality ignore, TS 38.413 §9.2.2.6), after the two IDs as the
    // message's IE table orders it.
    if let Some(info) = &params.recommended_paging {
        protocol_ies.push(UEContextReleaseCompleteProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_INFO_ON_RECOMMENDED_CELLS_AND_RAN_NODES_FOR_PAGING),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_InfoOnRecommendedCellsAndRANNodesForPaging(
                build_recommended_paging(info)?,
            ),
        });
    }

    let complete = UEContextReleaseComplete {
        protocol_i_es: UEContextReleaseCompleteProtocolIEs(protocol_ies),
    };

    let successful_outcome = SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_UEContextRelease(complete),
    };

    Ok(NGAP_PDU::SuccessfulOutcome(successful_outcome))
}

/// Parse a UE Context Release Complete from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(UeContextReleaseCompleteData)` - The parsed complete data
/// * `Err(UeContextReleaseError)` - If parsing fails
pub fn parse_ue_context_release_complete(
    pdu: &NGAP_PDU,
) -> Result<UeContextReleaseCompleteData, UeContextReleaseError> {
    let successful_outcome = match pdu {
        NGAP_PDU::SuccessfulOutcome(outcome) => outcome,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "SuccessfulOutcome".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let complete = match &successful_outcome.value {
        SuccessfulOutcomeValue::Id_UEContextRelease(comp) => comp,
        _ => {
            return Err(UeContextReleaseError::InvalidMessageType {
                expected: "UEContextReleaseComplete".to_string(),
                actual: format!("{:?}", successful_outcome.value),
            })
        }
    };

    let mut amf_ue_ngap_id: Option<u64> = None;
    let mut ran_ue_ngap_id: Option<u32> = None;
    let mut recommended_paging: Option<RecommendedPagingInfo> = None;

    for ie in &complete.protocol_i_es.0 {
        match &ie.value {
            UEContextReleaseCompleteProtocolIEs_EntryValue::Id_AMF_UE_NGAP_ID(id) => {
                amf_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseCompleteProtocolIEs_EntryValue::Id_RAN_UE_NGAP_ID(id) => {
                ran_ue_ngap_id = Some(id.0);
            }
            UEContextReleaseCompleteProtocolIEs_EntryValue::Id_InfoOnRecommendedCellsAndRANNodesForPaging(info) => {
                recommended_paging = parse_recommended_paging(info);
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(UeContextReleaseCompleteData {
        amf_ue_ngap_id: amf_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("AMF-UE-NGAP-ID".to_string())
        })?,
        ran_ue_ngap_id: ran_ue_ngap_id.ok_or_else(|| {
            UeContextReleaseError::MissingMandatoryIe("RAN-UE-NGAP-ID".to_string())
        })?,
        recommended_paging,
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a generated NGAP `Cause` from the typed cause representation.
///
/// Shared across procedures that carry a Cause IE (UE context release,
/// transfer containers, path switch failure handling).
pub fn build_cause(cause: &UeContextReleaseCause) -> Cause {
    match cause {
        NgSetupFailureCause::RadioNetwork(rn) => Cause::RadioNetwork(build_radio_network_cause(rn)),
        NgSetupFailureCause::Transport(t) => Cause::Transport(build_transport_cause(t)),
        NgSetupFailureCause::Nas(n) => Cause::Nas(build_nas_cause(n)),
        NgSetupFailureCause::Protocol(p) => Cause::Protocol(build_protocol_cause(p)),
        NgSetupFailureCause::Misc(m) => Cause::Misc(build_misc_cause(m)),
    }
}

fn build_radio_network_cause(cause: &RadioNetworkCause) -> CauseRadioNetwork {
    let value = match cause {
        RadioNetworkCause::Unspecified => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::TxnrelocoverallExpiry => CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY,
        RadioNetworkCause::SuccessfulHandover => CauseRadioNetwork::SUCCESSFUL_HANDOVER,
        RadioNetworkCause::ReleaseDueToNgranGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON
        }
        RadioNetworkCause::ReleaseDueTo5gcGeneratedReason => {
            CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON
        }
        RadioNetworkCause::HandoverCancelled => CauseRadioNetwork::HANDOVER_CANCELLED,
        RadioNetworkCause::PartialHandover => CauseRadioNetwork::PARTIAL_HANDOVER,
        RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem => {
            CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM
        }
        RadioNetworkCause::HoTargetNotAllowed => CauseRadioNetwork::HO_TARGET_NOT_ALLOWED,
        RadioNetworkCause::TngrelocOverallExpiry => CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY,
        RadioNetworkCause::TngrelocPrepExpiry => CauseRadioNetwork::TNGRELOCPREP_EXPIRY,
        RadioNetworkCause::CellNotAvailable => CauseRadioNetwork::CELL_NOT_AVAILABLE,
        RadioNetworkCause::UnknownTargetId => CauseRadioNetwork::UNKNOWN_TARGET_ID,
        RadioNetworkCause::NoRadioResourcesAvailableInTargetCell => {
            CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL
        }
        RadioNetworkCause::UnknownLocalUeNgapId => CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID,
        RadioNetworkCause::InconsistentRemoteUeNgapId => {
            CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID
        }
        RadioNetworkCause::HandoverDesirableForRadioReason => {
            CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON
        }
        RadioNetworkCause::TimeCriticalHandover => CauseRadioNetwork::TIME_CRITICAL_HANDOVER,
        RadioNetworkCause::ResourceOptimisationHandover => {
            CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER
        }
        RadioNetworkCause::ReduceLoadInServingCell => {
            CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL
        }
        RadioNetworkCause::UserInactivity => CauseRadioNetwork::USER_INACTIVITY,
        RadioNetworkCause::RadioConnectionWithUeLost => {
            CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST
        }
        RadioNetworkCause::RadioResourcesNotAvailable => {
            CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE
        }
        RadioNetworkCause::InvalidQosCombination => CauseRadioNetwork::INVALID_QOS_COMBINATION,
        RadioNetworkCause::FailureInRadioInterfaceProcedure => {
            CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE
        }
        RadioNetworkCause::InteractionWithOtherProcedure => {
            CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE
        }
        RadioNetworkCause::UnknownPduSessionId => CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID,
        RadioNetworkCause::UnkownQosFlowId => CauseRadioNetwork::UNKOWN_QOS_FLOW_ID,
        RadioNetworkCause::MultipleQosFlowIdInstances => {
            CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES
        }
        RadioNetworkCause::UnknownMappedUeNgapId => CauseRadioNetwork::UNSPECIFIED,
        RadioNetworkCause::UpIntegrityProtectionNotPossible => {
            CauseRadioNetwork::UP_INTEGRITY_PROTECTION_NOT_POSSIBLE
        }
        RadioNetworkCause::UpConfidentialityProtectionNotPossible => {
            CauseRadioNetwork::UP_CONFIDENTIALITY_PROTECTION_NOT_POSSIBLE
        }
        RadioNetworkCause::Other(v) => *v,
    };
    CauseRadioNetwork(value)
}

fn build_transport_cause(cause: &TransportCause) -> CauseTransport {
    let value = match cause {
        TransportCause::TransportResourceUnavailable => {
            CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE
        }
        TransportCause::Unspecified => CauseTransport::UNSPECIFIED,
        TransportCause::Other(v) => *v,
    };
    CauseTransport(value)
}

fn build_nas_cause(cause: &NasCause) -> CauseNas {
    let value = match cause {
        NasCause::NormalRelease => CauseNas::NORMAL_RELEASE,
        NasCause::AuthenticationFailure => CauseNas::AUTHENTICATION_FAILURE,
        NasCause::Deregister => CauseNas::DEREGISTER,
        NasCause::Unspecified => CauseNas::UNSPECIFIED,
        NasCause::Other(v) => *v,
    };
    CauseNas(value)
}

fn build_protocol_cause(cause: &ProtocolCause) -> CauseProtocol {
    let value = match cause {
        ProtocolCause::TransferSyntaxError => CauseProtocol::TRANSFER_SYNTAX_ERROR,
        ProtocolCause::AbstractSyntaxErrorReject => CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT,
        ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
        }
        ProtocolCause::MessageNotCompatibleWithReceiverState => {
            CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE
        }
        ProtocolCause::SemanticError => CauseProtocol::SEMANTIC_ERROR,
        ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage => {
            CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE
        }
        ProtocolCause::Unspecified => CauseProtocol::UNSPECIFIED,
        ProtocolCause::Other(v) => *v,
    };
    CauseProtocol(value)
}

fn build_misc_cause(cause: &MiscCause) -> CauseMisc {
    let value = match cause {
        MiscCause::ControlProcessingOverload => CauseMisc::CONTROL_PROCESSING_OVERLOAD,
        MiscCause::NotEnoughUserPlaneProcessingResources => {
            CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES
        }
        MiscCause::HardwareFailure => CauseMisc::HARDWARE_FAILURE,
        MiscCause::OmIntervention => CauseMisc::OM_INTERVENTION,
        MiscCause::UnknownPlmnOrSnpn => CauseMisc::UNKNOWN_PLMN_OR_SNPN,
        MiscCause::Unspecified => CauseMisc::UNSPECIFIED,
        MiscCause::Other(v) => *v,
    };
    CauseMisc(value)
}

/// Parse a generated NGAP `Cause` into the typed cause representation.
pub fn parse_cause(cause: &Cause) -> UeContextReleaseCause {
    match cause {
        Cause::RadioNetwork(rn) => NgSetupFailureCause::RadioNetwork(parse_radio_network_cause(rn)),
        Cause::Transport(t) => NgSetupFailureCause::Transport(parse_transport_cause(t)),
        Cause::Nas(n) => NgSetupFailureCause::Nas(parse_nas_cause(n)),
        Cause::Protocol(p) => NgSetupFailureCause::Protocol(parse_protocol_cause(p)),
        Cause::Misc(m) => NgSetupFailureCause::Misc(parse_misc_cause(m)),
        Cause::Choice_Extensions(_) => NgSetupFailureCause::Misc(MiscCause::Unspecified),
    }
}

fn parse_radio_network_cause(cause: &CauseRadioNetwork) -> RadioNetworkCause {
    match cause.0 {
        CauseRadioNetwork::UNSPECIFIED => RadioNetworkCause::Unspecified,
        CauseRadioNetwork::TXNRELOCOVERALL_EXPIRY => RadioNetworkCause::TxnrelocoverallExpiry,
        CauseRadioNetwork::SUCCESSFUL_HANDOVER => RadioNetworkCause::SuccessfulHandover,
        CauseRadioNetwork::RELEASE_DUE_TO_NGRAN_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueToNgranGeneratedReason
        }
        CauseRadioNetwork::RELEASE_DUE_TO_5GC_GENERATED_REASON => {
            RadioNetworkCause::ReleaseDueTo5gcGeneratedReason
        }
        CauseRadioNetwork::HANDOVER_CANCELLED => RadioNetworkCause::HandoverCancelled,
        CauseRadioNetwork::PARTIAL_HANDOVER => RadioNetworkCause::PartialHandover,
        CauseRadioNetwork::HO_FAILURE_IN_TARGET_5GC_NGRAN_NODE_OR_TARGET_SYSTEM => {
            RadioNetworkCause::HoFailureInTarget5gcNgranNodeOrTargetSystem
        }
        CauseRadioNetwork::HO_TARGET_NOT_ALLOWED => RadioNetworkCause::HoTargetNotAllowed,
        CauseRadioNetwork::TNGRELOCOVERALL_EXPIRY => RadioNetworkCause::TngrelocOverallExpiry,
        CauseRadioNetwork::TNGRELOCPREP_EXPIRY => RadioNetworkCause::TngrelocPrepExpiry,
        CauseRadioNetwork::CELL_NOT_AVAILABLE => RadioNetworkCause::CellNotAvailable,
        CauseRadioNetwork::UNKNOWN_TARGET_ID => RadioNetworkCause::UnknownTargetId,
        CauseRadioNetwork::NO_RADIO_RESOURCES_AVAILABLE_IN_TARGET_CELL => {
            RadioNetworkCause::NoRadioResourcesAvailableInTargetCell
        }
        CauseRadioNetwork::UNKNOWN_LOCAL_UE_NGAP_ID => RadioNetworkCause::UnknownLocalUeNgapId,
        CauseRadioNetwork::INCONSISTENT_REMOTE_UE_NGAP_ID => {
            RadioNetworkCause::InconsistentRemoteUeNgapId
        }
        CauseRadioNetwork::HANDOVER_DESIRABLE_FOR_RADIO_REASON => {
            RadioNetworkCause::HandoverDesirableForRadioReason
        }
        CauseRadioNetwork::TIME_CRITICAL_HANDOVER => RadioNetworkCause::TimeCriticalHandover,
        CauseRadioNetwork::RESOURCE_OPTIMISATION_HANDOVER => {
            RadioNetworkCause::ResourceOptimisationHandover
        }
        CauseRadioNetwork::REDUCE_LOAD_IN_SERVING_CELL => {
            RadioNetworkCause::ReduceLoadInServingCell
        }
        CauseRadioNetwork::USER_INACTIVITY => RadioNetworkCause::UserInactivity,
        CauseRadioNetwork::RADIO_CONNECTION_WITH_UE_LOST => {
            RadioNetworkCause::RadioConnectionWithUeLost
        }
        CauseRadioNetwork::RADIO_RESOURCES_NOT_AVAILABLE => {
            RadioNetworkCause::RadioResourcesNotAvailable
        }
        CauseRadioNetwork::INVALID_QOS_COMBINATION => RadioNetworkCause::InvalidQosCombination,
        CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE => {
            RadioNetworkCause::FailureInRadioInterfaceProcedure
        }
        CauseRadioNetwork::INTERACTION_WITH_OTHER_PROCEDURE => {
            RadioNetworkCause::InteractionWithOtherProcedure
        }
        CauseRadioNetwork::UNKNOWN_PDU_SESSION_ID => RadioNetworkCause::UnknownPduSessionId,
        CauseRadioNetwork::UNKOWN_QOS_FLOW_ID => RadioNetworkCause::UnkownQosFlowId,
        CauseRadioNetwork::MULTIPLE_QOS_FLOW_ID_INSTANCES => {
            RadioNetworkCause::MultipleQosFlowIdInstances
        }
        CauseRadioNetwork::UP_INTEGRITY_PROTECTION_NOT_POSSIBLE => {
            RadioNetworkCause::UpIntegrityProtectionNotPossible
        }
        CauseRadioNetwork::UP_CONFIDENTIALITY_PROTECTION_NOT_POSSIBLE => {
            RadioNetworkCause::UpConfidentialityProtectionNotPossible
        }
        other => RadioNetworkCause::Other(other),
    }
}

fn parse_transport_cause(cause: &CauseTransport) -> TransportCause {
    match cause.0 {
        CauseTransport::TRANSPORT_RESOURCE_UNAVAILABLE => {
            TransportCause::TransportResourceUnavailable
        }
        CauseTransport::UNSPECIFIED => TransportCause::Unspecified,
        other => TransportCause::Other(other),
    }
}

fn parse_nas_cause(cause: &CauseNas) -> NasCause {
    match cause.0 {
        CauseNas::NORMAL_RELEASE => NasCause::NormalRelease,
        CauseNas::AUTHENTICATION_FAILURE => NasCause::AuthenticationFailure,
        CauseNas::DEREGISTER => NasCause::Deregister,
        CauseNas::UNSPECIFIED => NasCause::Unspecified,
        other => NasCause::Other(other),
    }
}

fn parse_protocol_cause(cause: &CauseProtocol) -> ProtocolCause {
    match cause.0 {
        CauseProtocol::TRANSFER_SYNTAX_ERROR => ProtocolCause::TransferSyntaxError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT => ProtocolCause::AbstractSyntaxErrorReject,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY => {
            ProtocolCause::AbstractSyntaxErrorIgnoreAndNotify
        }
        CauseProtocol::MESSAGE_NOT_COMPATIBLE_WITH_RECEIVER_STATE => {
            ProtocolCause::MessageNotCompatibleWithReceiverState
        }
        CauseProtocol::SEMANTIC_ERROR => ProtocolCause::SemanticError,
        CauseProtocol::ABSTRACT_SYNTAX_ERROR_FALSELY_CONSTRUCTED_MESSAGE => {
            ProtocolCause::AbstractSyntaxErrorFalselyConstructedMessage
        }
        CauseProtocol::UNSPECIFIED => ProtocolCause::Unspecified,
        other => ProtocolCause::Other(other),
    }
}

fn parse_misc_cause(cause: &CauseMisc) -> MiscCause {
    match cause.0 {
        CauseMisc::CONTROL_PROCESSING_OVERLOAD => MiscCause::ControlProcessingOverload,
        CauseMisc::NOT_ENOUGH_USER_PLANE_PROCESSING_RESOURCES => {
            MiscCause::NotEnoughUserPlaneProcessingResources
        }
        CauseMisc::HARDWARE_FAILURE => MiscCause::HardwareFailure,
        CauseMisc::OM_INTERVENTION => MiscCause::OmIntervention,
        CauseMisc::UNKNOWN_PLMN_OR_SNPN => MiscCause::UnknownPlmnOrSnpn,
        CauseMisc::UNSPECIFIED => MiscCause::Unspecified,
        other => MiscCause::Other(other),
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a UE Context Release Request to bytes
pub fn encode_ue_context_release_request(
    params: &UeContextReleaseRequestParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_request(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Request from bytes
pub fn decode_ue_context_release_request(
    bytes: &[u8],
) -> Result<UeContextReleaseRequestData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_request(&pdu)
}

/// Build and encode a UE Context Release Command to bytes
pub fn encode_ue_context_release_command(
    params: &UeContextReleaseCommandParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_command(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Command from bytes
pub fn decode_ue_context_release_command(
    bytes: &[u8],
) -> Result<UeContextReleaseCommandData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_command(&pdu)
}

/// Build and encode a UE Context Release Complete to bytes
pub fn encode_ue_context_release_complete(
    params: &UeContextReleaseCompleteParams,
) -> Result<Vec<u8>, UeContextReleaseError> {
    let pdu = build_ue_context_release_complete(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a UE Context Release Complete from bytes
pub fn decode_ue_context_release_complete(
    bytes: &[u8],
) -> Result<UeContextReleaseCompleteData, UeContextReleaseError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_ue_context_release_complete(&pdu)
}

/// Check if an NGAP PDU is a UE Context Release Request
pub fn is_ue_context_release_request(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UEContextReleaseRequest(_))
    )
}

/// Check if an NGAP PDU is a UE Context Release Command
pub fn is_ue_context_release_command(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_UEContextRelease(_))
    )
}

/// Check if an NGAP PDU is a UE Context Release Complete
pub fn is_ue_context_release_complete(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::SuccessfulOutcome(outcome)
            if matches!(outcome.value, SuccessfulOutcomeValue::Id_UEContextRelease(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_request_params() -> UeContextReleaseRequestParams {
        UeContextReleaseRequestParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            pdu_session_ids: Vec::new(),
            cause: UeContextReleaseCause::RadioNetwork(RadioNetworkCause::UserInactivity),
        }
    }

    fn create_test_command_params() -> UeContextReleaseCommandParams {
        UeContextReleaseCommandParams {
            ue_ngap_ids: UeNgapIds::Pair {
                amf_ue_ngap_id: 12345,
                ran_ue_ngap_id: 1,
            },
            cause: UeContextReleaseCause::Nas(NasCause::NormalRelease),
        }
    }

    fn create_test_complete_params() -> UeContextReleaseCompleteParams {
        UeContextReleaseCompleteParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 1,
            recommended_paging: None,
        }
    }

    #[test]
    fn test_build_ue_context_release_request() {
        let params = create_test_request_params();
        let result = build_ue_context_release_request(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_request(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_request() {
        let params = create_test_request_params();
        let result = encode_ue_context_release_request(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_request_roundtrip() {
        let params = create_test_request_params();

        let encoded = encode_ue_context_release_request(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_request(&encoded).expect("Failed to decode");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
        assert_eq!(parsed.cause, params.cause);
    }

    #[test]
    fn test_build_ue_context_release_command() {
        let params = create_test_command_params();
        let result = build_ue_context_release_command(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_command(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_command() {
        let params = create_test_command_params();
        let result = encode_ue_context_release_command(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_command_roundtrip() {
        let params = create_test_command_params();

        let encoded = encode_ue_context_release_command(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_command(&encoded).expect("Failed to decode");

        match (&parsed.ue_ngap_ids, &params.ue_ngap_ids) {
            (
                UeNgapIds::Pair {
                    amf_ue_ngap_id: parsed_amf,
                    ran_ue_ngap_id: parsed_ran,
                },
                UeNgapIds::Pair {
                    amf_ue_ngap_id: params_amf,
                    ran_ue_ngap_id: params_ran,
                },
            ) => {
                assert_eq!(parsed_amf, params_amf);
                assert_eq!(parsed_ran, params_ran);
            }
            _ => panic!("Expected Pair variant"),
        }
        assert_eq!(parsed.cause, params.cause);
    }

    #[test]
    fn test_ue_context_release_command_amf_only() {
        let params = UeContextReleaseCommandParams {
            ue_ngap_ids: UeNgapIds::AmfOnly(12345),
            cause: UeContextReleaseCause::Nas(NasCause::Deregister),
        };

        let encoded = encode_ue_context_release_command(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_command(&encoded).expect("Failed to decode");

        match parsed.ue_ngap_ids {
            UeNgapIds::AmfOnly(amf_id) => assert_eq!(amf_id, 12345),
            _ => panic!("Expected AmfOnly variant"),
        }
    }

    #[test]
    fn test_build_ue_context_release_complete() {
        let params = create_test_complete_params();
        let result = build_ue_context_release_complete(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        assert!(is_ue_context_release_complete(&pdu));
    }

    #[test]
    fn test_encode_ue_context_release_complete() {
        let params = create_test_complete_params();
        let result = encode_ue_context_release_complete(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_ue_context_release_complete_roundtrip() {
        let params = create_test_complete_params();

        let encoded = encode_ue_context_release_complete(&params).expect("Failed to encode");
        let parsed = decode_ue_context_release_complete(&encoded).expect("Failed to decode");

        assert_eq!(parsed.amf_ue_ngap_id, params.amf_ue_ngap_id);
        assert_eq!(parsed.ran_ue_ngap_id, params.ran_ue_ngap_id);
    }

    #[test]
    fn test_various_causes() {
        // Test with different cause types
        let causes = vec![
            UeContextReleaseCause::RadioNetwork(RadioNetworkCause::UserInactivity),
            UeContextReleaseCause::RadioNetwork(RadioNetworkCause::RadioConnectionWithUeLost),
            UeContextReleaseCause::Nas(NasCause::NormalRelease),
            UeContextReleaseCause::Nas(NasCause::Deregister),
            UeContextReleaseCause::Transport(TransportCause::Unspecified),
            UeContextReleaseCause::Protocol(ProtocolCause::Unspecified),
            UeContextReleaseCause::Misc(MiscCause::Unspecified),
        ];

        for cause in causes {
            let params = UeContextReleaseRequestParams {
                amf_ue_ngap_id: 12345,
                ran_ue_ngap_id: 1,
                pdu_session_ids: Vec::new(),
                cause: cause.clone(),
            };

            let encoded = encode_ue_context_release_request(&params).expect("Failed to encode");
            let parsed = decode_ue_context_release_request(&encoded).expect("Failed to decode");

            assert_eq!(parsed.cause, cause);
        }
    }

    /// IE 133, PDU Session Resource List (TS 38.413 §9.2.2.4), between the UE IDs
    /// and the Cause in object-set order, and absent when the UE has no session.
    /// The entry's octets are derived by hand from X.691 APER:
    ///   id 133            -> 00 85
    ///   criticality reject-> 00   (2 bits, then the open type aligns)
    ///   open-type length  -> 03
    ///   value: SEQUENCE OF SIZE(1..256): count-1 = 0 in 8 bits -> 00
    ///          item: ext bit 0, iE-Extensions absent 0, then PDUSessionID
    ///          (0..255, range 256: one octet, octet-aligned) -> 00 05
    #[test]
    fn release_request_carries_the_pdu_session_list_in_order() {
        let mut params = create_test_request_params();
        params.pdu_session_ids = vec![5];
        let pdu = build_ue_context_release_request(&params).expect("builds");
        let NGAP_PDU::InitiatingMessage(ref m) = pdu else {
            panic!()
        };
        let InitiatingMessageValue::Id_UEContextReleaseRequest(ref req) = m.value else {
            panic!()
        };
        let ids: Vec<u16> = req.protocol_i_es.0.iter().map(|e| e.id.0).collect();
        assert_eq!(ids, vec![10, 85, 133, 15]);
        let entry = crate::codec::encode_aper(&req.protocol_i_es.0[2]).expect("entry");
        assert_eq!(entry, [0x00, 0x85, 0x00, 0x03, 0x00, 0x00, 0x05]);
        let bytes = encode_ngap_pdu(&pdu).unwrap();
        let back = parse_ue_context_release_request(&decode_ngap_pdu(&bytes).unwrap()).unwrap();
        assert_eq!(back.pdu_session_ids, vec![5]);

        params.pdu_session_ids.clear();
        let pdu = build_ue_context_release_request(&params).expect("builds");
        let NGAP_PDU::InitiatingMessage(ref m) = pdu else {
            panic!()
        };
        let InitiatingMessageValue::Id_UEContextReleaseRequest(ref req) = m.value else {
            panic!()
        };
        assert!(
            req.protocol_i_es.0.iter().all(|e| e.id.0 != 133),
            "absent when empty"
        );
    }

    /// nextgcore amfd's `NEXTGSIM_RELEASE_COMPLETE_WITH_RECOMMENDATION`
    /// (`nextgcore-amfd/src/ngap_asn1.rs`, #1448): the 63 octets its decode test
    /// pins, captured from a scratch encoding through this crate's generated
    /// types. Copied verbatim, so this encoder and amfd's decoder are pinned to
    /// one vector (nextgsim#462).
    const NEXTGCORE_PINNED_RELEASE_COMPLETE: [u8; 63] = [
        0x20, 0x29, 0x00, 0x3b, 0x00, 0x00, 0x03, 0x00, 0x0a, 0x40, 0x02, 0x00, 0x07, 0x00, 0x55,
        0x40, 0x02, 0x00, 0x03, 0x00, 0x20, 0x40, 0x28, 0x01, 0x40, 0x00, 0xf1, 0x10, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x1e, 0x00, 0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x02, 0x00, 0x40,
        0x00, 0x00, 0xf1, 0x10, 0x50, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0xf1, 0x10, 0x50, 0x00,
        0x00, 0x00, 0x02,
    ];

    /// The content amfd's vector documents: AMF UE NGAP ID 7, RAN UE NGAP ID 3;
    /// cells NR CGI 001/01 NCI 0x10 (visited, 30 s) then NCI 0x20 (non-visited);
    /// RAN nodes gNB 1 (visited) then gNB 2, 32-bit gNB IDs.
    fn pinned_recommendation() -> RecommendedPagingInfo {
        const PLMN: [u8; 3] = [0x00, 0xF1, 0x10];
        RecommendedPagingInfo {
            cells: vec![
                RecommendedCell {
                    plmn: PLMN,
                    nci: 0x10,
                    time_stayed_s: Some(30),
                },
                RecommendedCell {
                    plmn: PLMN,
                    nci: 0x20,
                    time_stayed_s: None,
                },
            ],
            ran_nodes: vec![
                RecommendedRanNode {
                    plmn: PLMN,
                    gnb_id: 1,
                    gnb_id_len: 32,
                },
                RecommendedRanNode {
                    plmn: PLMN,
                    gnb_id: 2,
                    gnb_id_len: 32,
                },
            ],
        }
    }

    /// #462: the recommendation encodes to exactly the octets nextgcore amfd's
    /// decode test pins (TS 38.413 §9.3.1.100, both lists, in the §9.3.1.71 /
    /// §9.3.1.101 order), and decodes back to the same content.
    #[test]
    fn the_recommendation_encodes_to_the_vector_amfd_decodes() {
        let params = UeContextReleaseCompleteParams {
            amf_ue_ngap_id: 7,
            ran_ue_ngap_id: 3,
            recommended_paging: Some(pinned_recommendation()),
        };
        let bytes = encode_ue_context_release_complete(&params).expect("encode");
        assert_eq!(
            bytes,
            NEXTGCORE_PINNED_RELEASE_COMPLETE.to_vec(),
            "{bytes:02x?}"
        );
        let parsed = decode_ue_context_release_complete(&bytes).expect("decode");
        assert_eq!((parsed.amf_ue_ngap_id, parsed.ran_ue_ngap_id), (7, 3));
        assert_eq!(parsed.recommended_paging, Some(pinned_recommendation()));
    }

    /// #462: without a recommendation the message is the two IDs only, the
    /// octets it had before the IE was supported (the pinned vector with its
    /// third IE taken out: count 2, length 15).
    #[test]
    fn without_a_recommendation_the_release_complete_is_unchanged() {
        let params = UeContextReleaseCompleteParams {
            amf_ue_ngap_id: 7,
            ran_ue_ngap_id: 3,
            recommended_paging: None,
        };
        let bytes = encode_ue_context_release_complete(&params).expect("encode");
        assert_eq!(
            bytes,
            vec![
                0x20, 0x29, 0x00, 0x0f, 0x00, 0x00, 0x02, 0x00, 0x0a, 0x40, 0x02, 0x00, 0x07, 0x00,
                0x55, 0x40, 0x02, 0x00, 0x03
            ]
        );
        assert_eq!(
            decode_ue_context_release_complete(&bytes)
                .unwrap()
                .recommended_paging,
            None
        );
    }

    /// #462: the bounds TS 38.413 sets are refused at encode, not truncated:
    /// an empty or 17-entry list, a 37-bit NCI, a gNB ID length outside 22..32
    /// or a gNB ID wider than its length, and a time stayed above 4095 s.
    /// 16 entries and the 22- and 32-bit lengths encode and round-trip.
    #[test]
    fn a_recommendation_out_of_bounds_is_refused() {
        let encode = |info: RecommendedPagingInfo| {
            encode_ue_context_release_complete(&UeContextReleaseCompleteParams {
                amf_ue_ngap_id: 1,
                ran_ue_ngap_id: 1,
                recommended_paging: Some(info),
            })
        };
        let base = pinned_recommendation();
        let mut cases: Vec<(&str, RecommendedPagingInfo)> = Vec::new();
        let mut v = base.clone();
        v.cells.clear();
        cases.push(("no cell", v));
        let mut v = base.clone();
        v.ran_nodes.clear();
        cases.push(("no node", v));
        let mut v = base.clone();
        v.cells = vec![base.cells[1]; 17];
        cases.push(("17 cells", v));
        let mut v = base.clone();
        v.ran_nodes = vec![base.ran_nodes[1]; 17];
        cases.push(("17 nodes", v));
        let mut v = base.clone();
        v.cells[0].nci = 1 << 36;
        cases.push(("37-bit NCI", v));
        let mut v = base.clone();
        v.cells[0].time_stayed_s = Some(4096);
        cases.push(("4096 s", v));
        for len in [21, 33] {
            let mut v = base.clone();
            v.ran_nodes[0].gnb_id_len = len;
            cases.push(("gNB ID length", v));
        }
        let mut v = base.clone();
        v.ran_nodes[0].gnb_id = 1 << 22;
        v.ran_nodes[0].gnb_id_len = 22;
        cases.push(("gNB ID wider than 22 bits", v));
        for (name, info) in cases {
            assert!(
                matches!(encode(info), Err(UeContextReleaseError::InvalidIeValue(_))),
                "{name} must be refused"
            );
        }

        let mut v = base.clone();
        v.cells = (1..=16)
            .map(|i| RecommendedCell {
                plmn: base.cells[0].plmn,
                nci: i << 4,
                time_stayed_s: Some(4095),
            })
            .collect();
        v.ran_nodes = (1..=16)
            .map(|i| RecommendedRanNode {
                plmn: base.cells[0].plmn,
                gnb_id: (i as u32) | 0x20_0000,
                gnb_id_len: if i % 2 == 0 { 22 } else { 32 },
            })
            .collect();
        let bytes = encode(v.clone()).expect("16 entries and both length bounds encode");
        assert_eq!(
            decode_ue_context_release_complete(&bytes)
                .unwrap()
                .recommended_paging,
            Some(v)
        );
    }
}
