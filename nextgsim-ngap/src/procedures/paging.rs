//! Paging Procedure
//!
//! Implements the Paging procedure as defined in 3GPP TS 38.413 Section 8.7.2.
//! This procedure is used by the AMF to request the NG-RAN node to page a UE.
//! The Paging message is sent from AMF to gNB to initiate paging for a UE.

use crate::codec::generated::*;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use crate::procedures::initial_ue_message::{
    build_nr_cgi, build_tai, parse_nr_cgi, parse_tai, FiveGSTmsi, NrCgi, Tai,
};
use bitvec::prelude::*;
use thiserror::Error;

/// Errors that can occur during Paging procedures
#[derive(Debug, Error)]
pub enum PagingError {
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

/// Paging Priority values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingPriorityValue {
    /// Priority level 1 (highest)
    PrioLevel1,
    /// Priority level 2
    PrioLevel2,
    /// Priority level 3
    PrioLevel3,
    /// Priority level 4
    PrioLevel4,
    /// Priority level 5
    PrioLevel5,
    /// Priority level 6
    PrioLevel6,
    /// Priority level 7
    PrioLevel7,
    /// Priority level 8 (lowest)
    PrioLevel8,
}

impl From<PagingPriorityValue> for PagingPriority {
    fn from(priority: PagingPriorityValue) -> Self {
        let value = match priority {
            PagingPriorityValue::PrioLevel1 => PagingPriority::PRIOLEVEL1,
            PagingPriorityValue::PrioLevel2 => PagingPriority::PRIOLEVEL2,
            PagingPriorityValue::PrioLevel3 => PagingPriority::PRIOLEVEL3,
            PagingPriorityValue::PrioLevel4 => PagingPriority::PRIOLEVEL4,
            PagingPriorityValue::PrioLevel5 => PagingPriority::PRIOLEVEL5,
            PagingPriorityValue::PrioLevel6 => PagingPriority::PRIOLEVEL6,
            PagingPriorityValue::PrioLevel7 => PagingPriority::PRIOLEVEL7,
            PagingPriorityValue::PrioLevel8 => PagingPriority::PRIOLEVEL8,
        };
        PagingPriority(value)
    }
}

impl TryFrom<PagingPriority> for PagingPriorityValue {
    type Error = PagingError;

    fn try_from(priority: PagingPriority) -> Result<Self, Self::Error> {
        match priority.0 {
            PagingPriority::PRIOLEVEL1 => Ok(PagingPriorityValue::PrioLevel1),
            PagingPriority::PRIOLEVEL2 => Ok(PagingPriorityValue::PrioLevel2),
            PagingPriority::PRIOLEVEL3 => Ok(PagingPriorityValue::PrioLevel3),
            PagingPriority::PRIOLEVEL4 => Ok(PagingPriorityValue::PrioLevel4),
            PagingPriority::PRIOLEVEL5 => Ok(PagingPriorityValue::PrioLevel5),
            PagingPriority::PRIOLEVEL6 => Ok(PagingPriorityValue::PrioLevel6),
            PagingPriority::PRIOLEVEL7 => Ok(PagingPriorityValue::PrioLevel7),
            PagingPriority::PRIOLEVEL8 => Ok(PagingPriorityValue::PrioLevel8),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingPriority value: {}",
                priority.0
            ))),
        }
    }
}

/// Paging Origin values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingOriginValue {
    /// Paging originated from non-3GPP access
    Non3gpp,
}

impl From<PagingOriginValue> for PagingOrigin {
    fn from(_: PagingOriginValue) -> Self {
        PagingOrigin(PagingOrigin::NON_3GPP)
    }
}

impl TryFrom<PagingOrigin> for PagingOriginValue {
    type Error = PagingError;

    fn try_from(origin: PagingOrigin) -> Result<Self, Self::Error> {
        match origin.0 {
            PagingOrigin::NON_3GPP => Ok(PagingOriginValue::Non3gpp),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingOrigin value: {}",
                origin.0
            ))),
        }
    }
}

/// Paging DRX values as defined in 3GPP TS 38.413
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingDrxValue {
    /// 32 radio frames
    V32,
    /// 64 radio frames
    V64,
    /// 128 radio frames
    V128,
    /// 256 radio frames
    V256,
}

impl PagingDrxValue {
    /// The DRX cycle in radio frames, which is the `T` of TS 38.304 §7.1.
    ///
    /// Exists so the paging occasion can actually be derived from what the AMF
    /// signalled: before this the IE was decoded and logged and nothing consumed
    /// it (issue #99).
    pub fn radio_frames(self) -> u16 {
        match self {
            Self::V32 => 32,
            Self::V64 => 64,
            Self::V128 => 128,
            Self::V256 => 256,
        }
    }
}

impl From<PagingDrxValue> for PagingDRX {
    fn from(drx: PagingDrxValue) -> Self {
        match drx {
            PagingDrxValue::V32 => PagingDRX(PagingDRX::V32),
            PagingDrxValue::V64 => PagingDRX(PagingDRX::V64),
            PagingDrxValue::V128 => PagingDRX(PagingDRX::V128),
            PagingDrxValue::V256 => PagingDRX(PagingDRX::V256),
        }
    }
}

impl TryFrom<PagingDRX> for PagingDrxValue {
    type Error = PagingError;

    fn try_from(drx: PagingDRX) -> Result<Self, Self::Error> {
        match drx.0 {
            PagingDRX::V32 => Ok(PagingDrxValue::V32),
            PagingDRX::V64 => Ok(PagingDrxValue::V64),
            PagingDRX::V128 => Ok(PagingDrxValue::V128),
            PagingDRX::V256 => Ok(PagingDrxValue::V256),
            _ => Err(PagingError::InvalidIeValue(format!(
                "Unknown PagingDRX value: {}",
                drx.0
            ))),
        }
    }
}

/// UE Paging Identity - identifies the UE to be paged
#[derive(Debug, Clone)]
pub enum UePagingIdentityValue {
    /// 5G-S-TMSI based identity
    FiveGSTmsi(FiveGSTmsi),
}

/// Next Paging Area Scope (TS 38.413 §9.3.1.72): whether the AMF's next paging
/// attempt pages the same area as this one or a different one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NextPagingAreaScopeValue {
    /// The next attempt pages the same area.
    Same,
    /// The next attempt pages a different area.
    Changed,
}

/// Paging Attempt Information (TS 38.413 §9.3.1.72): which attempt over NG this
/// PAGING is, of how many the AMF intends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PagingAttemptInfoValue {
    /// Paging Attempt Count, 1..16: "increased by one at each new paging
    /// attempt" (TS 38.300 §9.2.5).
    pub paging_attempt_count: u8,
    /// Intended Number of Paging Attempts, 1..16.
    pub intended_number_of_paging_attempts: u8,
    /// Next Paging Area Scope, if the AMF signalled one.
    pub next_paging_area_scope: Option<NextPagingAreaScopeValue>,
}

/// The cell of one Recommended Cell Item: an NG-RAN CGI (TS 38.413 §9.3.1.73).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendedCellCgi {
    /// An NR CGI: the PLMN and the 36-bit NR Cell Identity.
    Nr {
        /// PLMN Identity (3 octets).
        plmn_identity: [u8; 3],
        /// NR Cell Identity (36 bits).
        nr_cell_identity: u64,
    },
    /// An E-UTRA CGI: the PLMN and the 28-bit E-UTRA Cell Identity.
    Eutra {
        /// PLMN Identity (3 octets).
        plmn_identity: [u8; 3],
        /// E-UTRA Cell Identity (28 bits).
        eutra_cell_identity: u32,
    },
}

/// One Recommended Cell Item (TS 38.413 §9.3.1.71).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecommendedCellValue {
    /// The cell.
    pub cgi: RecommendedCellCgi,
    /// Time Stayed in Cell, seconds (0..4095), present for visited cells.
    pub time_stayed_in_cell: Option<u16>,
}

/// Assistance Data for Paging (TS 38.413 §9.3.1.69), the parts this codec
/// carries: the Assistance Data for Recommended Cells (§9.3.1.70, its
/// Recommended Cells for Paging list, §9.3.1.71) and the Paging Attempt
/// Information (§9.3.1.72). The NPN and CE-capable-UE extensions are not
/// carried and are ignored when received.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AssistanceDataForPagingValue {
    /// The Recommended Cells for Paging, in the order received (the most
    /// recently visited cell first, TS 38.413 §9.3.1.71). Empty when the IE
    /// carried no Assistance Data for Recommended Cells.
    pub recommended_cells: Vec<RecommendedCellValue>,
    /// The Paging Attempt Information, when present.
    pub paging_attempt_information: Option<PagingAttemptInfoValue>,
}

/// Parameters for building a Paging message
#[derive(Debug, Clone)]
pub struct PagingParams {
    /// UE Paging Identity (mandatory)
    pub ue_paging_identity: UePagingIdentityValue,
    /// TAI List for Paging (mandatory) - list of TAIs where the UE should be paged
    pub tai_list_for_paging: Vec<Tai>,
    /// Paging DRX (optional)
    pub paging_drx: Option<PagingDrxValue>,
    /// Paging Priority (optional)
    pub paging_priority: Option<PagingPriorityValue>,
    /// Paging Origin (optional)
    pub paging_origin: Option<PagingOriginValue>,
    /// Assistance Data for Paging (optional, TS 38.413 §9.3.1.69)
    pub assistance_data: Option<AssistanceDataForPagingValue>,
}

/// Parsed Paging message data
#[derive(Debug, Clone)]
pub struct PagingData {
    /// UE Paging Identity
    pub ue_paging_identity: UePagingIdentityValue,
    /// TAI List for Paging
    pub tai_list_for_paging: Vec<Tai>,
    /// Paging DRX (optional)
    pub paging_drx: Option<PagingDrxValue>,
    /// Paging Priority (optional)
    pub paging_priority: Option<PagingPriorityValue>,
    /// Paging Origin (optional)
    pub paging_origin: Option<PagingOriginValue>,
    /// Assistance Data for Paging (optional, TS 38.413 §9.3.1.69)
    pub assistance_data: Option<AssistanceDataForPagingValue>,
}

// ============================================================================
// Paging Message Builder
// ============================================================================

/// Build a Paging PDU
///
/// # Arguments
/// * `params` - Parameters for the Paging message
///
/// # Returns
/// * `Ok(NGAP_PDU)` - The constructed PDU
/// * `Err(PagingError)` - If construction fails
pub fn build_paging(params: &PagingParams) -> Result<NGAP_PDU, PagingError> {
    let mut protocol_ies = Vec::new();

    // IE: UEPagingIdentity (mandatory)
    let ue_paging_identity = build_ue_paging_identity(&params.ue_paging_identity);
    protocol_ies.push(PagingProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_UE_PAGING_IDENTITY),
        criticality: Criticality(Criticality::IGNORE),
        value: PagingProtocolIEs_EntryValue::Id_UEPagingIdentity(ue_paging_identity),
    });

    // IE: PagingDRX (optional)
    if let Some(drx) = params.paging_drx {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_DRX),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingDRX(drx.into()),
        });
    }

    // IE: TAIListForPaging (mandatory)
    let tai_list = build_tai_list_for_paging(&params.tai_list_for_paging);
    protocol_ies.push(PagingProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_TAI_LIST_FOR_PAGING),
        criticality: Criticality(Criticality::IGNORE),
        value: PagingProtocolIEs_EntryValue::Id_TAIListForPaging(tai_list),
    });

    // IE: PagingPriority (optional)
    if let Some(priority) = params.paging_priority {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_PRIORITY),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingPriority(priority.into()),
        });
    }

    // IE: PagingOrigin (optional)
    if let Some(origin) = params.paging_origin {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_PAGING_ORIGIN),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_PagingOrigin(origin.into()),
        });
    }

    // IE: AssistanceDataForPaging (optional), after Paging Origin as in the
    // TS 38.413 §9.2.4.1 tabular order.
    if let Some(assistance) = &params.assistance_data {
        protocol_ies.push(PagingProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_ASSISTANCE_DATA_FOR_PAGING),
            criticality: Criticality(Criticality::IGNORE),
            value: PagingProtocolIEs_EntryValue::Id_AssistanceDataForPaging(
                build_assistance_data_for_paging(assistance)?,
            ),
        });
    }

    let paging = Paging {
        protocol_i_es: PagingProtocolIEs(protocol_ies),
    };

    let initiating_message = InitiatingMessage {
        procedure_code: ProcedureCode(ID_PAGING),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_Paging(paging),
    };

    Ok(NGAP_PDU::InitiatingMessage(initiating_message))
}

/// The ASN.1 Assistance Data for Paging of `value`. Refuses what the ASN.1
/// bounds refuse (TS 38.413 §9.3.1.71-72), so a builder caller learns it here
/// rather than from an opaque encoder error.
fn build_assistance_data_for_paging(
    value: &AssistanceDataForPagingValue,
) -> Result<AssistanceDataForPaging, PagingError> {
    let cells = if value.recommended_cells.is_empty() {
        None
    } else {
        if value.recommended_cells.len() > MAX_RECOMMENDED_CELLS {
            return Err(PagingError::InvalidIeValue(format!(
                "{} Recommended Cells for Paging; at most {MAX_RECOMMENDED_CELLS}",
                value.recommended_cells.len()
            )));
        }
        let mut items = Vec::with_capacity(value.recommended_cells.len());
        for cell in &value.recommended_cells {
            let ngran_cgi = match &cell.cgi {
                RecommendedCellCgi::Nr {
                    plmn_identity,
                    nr_cell_identity,
                } => {
                    if *nr_cell_identity >= 1 << 36 {
                        return Err(PagingError::InvalidIeValue(format!(
                            "NR Cell Identity {nr_cell_identity:#x} is wider than 36 bits"
                        )));
                    }
                    NGRAN_CGI::NR_CGI(build_nr_cgi(&NrCgi {
                        plmn_identity: *plmn_identity,
                        nr_cell_identity: *nr_cell_identity,
                    }))
                }
                RecommendedCellCgi::Eutra {
                    plmn_identity,
                    eutra_cell_identity,
                } => {
                    if *eutra_cell_identity >= 1 << 28 {
                        return Err(PagingError::InvalidIeValue(format!(
                            "E-UTRA Cell Identity {eutra_cell_identity:#x} is wider than 28 bits"
                        )));
                    }
                    let mut bv: BitVec<u8, Msb0> = BitVec::new();
                    for i in (0..28).rev() {
                        bv.push((eutra_cell_identity >> i) & 1 == 1);
                    }
                    NGRAN_CGI::EUTRA_CGI(EUTRA_CGI {
                        plmn_identity: PLMNIdentity(plmn_identity.to_vec()),
                        eutra_cell_identity: EUTRACellIdentity(bv),
                        ie_extensions: None,
                    })
                }
            };
            let time_stayed_in_cell = match cell.time_stayed_in_cell {
                Some(t) if t > 4095 => {
                    return Err(PagingError::InvalidIeValue(format!(
                        "Time Stayed in Cell {t} s; at most 4095"
                    )))
                }
                other => other.map(RecommendedCellItemTimeStayedInCell),
            };
            items.push(RecommendedCellItem {
                ngran_cgi,
                time_stayed_in_cell,
                ie_extensions: None,
            });
        }
        Some(AssistanceDataForRecommendedCells {
            recommended_cells_for_paging: RecommendedCellsForPaging {
                recommended_cell_list: RecommendedCellList(items),
                ie_extensions: None,
            },
            ie_extensions: None,
        })
    };
    let paging_attempt_information = match value.paging_attempt_information {
        None => None,
        Some(pai) => {
            for (name, v) in [
                ("Paging Attempt Count", pai.paging_attempt_count),
                (
                    "Intended Number of Paging Attempts",
                    pai.intended_number_of_paging_attempts,
                ),
            ] {
                if !(1..=16).contains(&v) {
                    return Err(PagingError::InvalidIeValue(format!(
                        "{name} {v}; the range is 1..16"
                    )));
                }
            }
            Some(PagingAttemptInformation {
                paging_attempt_count: PagingAttemptCount(pai.paging_attempt_count),
                intended_number_of_paging_attempts: IntendedNumberOfPagingAttempts(
                    pai.intended_number_of_paging_attempts,
                ),
                next_paging_area_scope: pai.next_paging_area_scope.map(|scope| {
                    NextPagingAreaScope(match scope {
                        NextPagingAreaScopeValue::Same => NextPagingAreaScope::SAME,
                        NextPagingAreaScopeValue::Changed => NextPagingAreaScope::CHANGED,
                    })
                }),
                ie_extensions: None,
            })
        }
    };
    Ok(AssistanceDataForPaging {
        assistance_data_for_recommended_cells: cells,
        paging_attempt_information,
        ie_extensions: None,
    })
}

/// maxnoofRecommendedCells (TS 38.413 §9.3.1.71).
pub const MAX_RECOMMENDED_CELLS: usize = 16;

fn build_ue_paging_identity(identity: &UePagingIdentityValue) -> UEPagingIdentity {
    match identity {
        UePagingIdentityValue::FiveGSTmsi(tmsi) => {
            let five_g_s_tmsi = build_five_g_s_tmsi(tmsi);
            UEPagingIdentity::FiveG_S_TMSI(five_g_s_tmsi)
        }
    }
}

fn build_five_g_s_tmsi(tmsi: &FiveGSTmsi) -> FiveG_S_TMSI {
    // AMF Set ID is 10 bits
    let mut amf_set_id_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..10).rev() {
        amf_set_id_bv.push((tmsi.amf_set_id >> i) & 1 == 1);
    }

    // AMF Pointer is 6 bits
    let mut amf_pointer_bv: BitVec<u8, Msb0> = BitVec::new();
    for i in (0..6).rev() {
        amf_pointer_bv.push((tmsi.amf_pointer >> i) & 1 == 1);
    }

    FiveG_S_TMSI {
        amf_set_id: AMFSetID(amf_set_id_bv),
        amf_pointer: AMFPointer(amf_pointer_bv),
        five_g_tmsi: FiveG_TMSI(tmsi.five_g_tmsi.to_vec()),
        ie_extensions: None,
    }
}

fn build_tai_list_for_paging(tai_list: &[Tai]) -> TAIListForPaging {
    let items: Vec<TAIListForPagingItem> = tai_list
        .iter()
        .map(|tai| TAIListForPagingItem {
            tai: build_tai(tai),
            ie_extensions: None,
        })
        .collect();

    TAIListForPaging(items)
}

// ============================================================================
// Paging Message Parser
// ============================================================================

/// Parse a Paging message from an NGAP PDU
///
/// # Arguments
/// * `pdu` - The NGAP PDU to parse
///
/// # Returns
/// * `Ok(PagingData)` - The parsed message data
/// * `Err(PagingError)` - If parsing fails
pub fn parse_paging(pdu: &NGAP_PDU) -> Result<PagingData, PagingError> {
    let initiating_message = match pdu {
        NGAP_PDU::InitiatingMessage(msg) => msg,
        _ => {
            return Err(PagingError::InvalidMessageType {
                expected: "InitiatingMessage".to_string(),
                actual: format!("{pdu:?}"),
            })
        }
    };

    let paging = match &initiating_message.value {
        InitiatingMessageValue::Id_Paging(msg) => msg,
        _ => {
            return Err(PagingError::InvalidMessageType {
                expected: "Paging".to_string(),
                actual: format!("{:?}", initiating_message.value),
            })
        }
    };

    let mut ue_paging_identity: Option<UePagingIdentityValue> = None;
    let mut tai_list_for_paging: Option<Vec<Tai>> = None;
    let mut paging_drx: Option<PagingDrxValue> = None;
    let mut paging_priority: Option<PagingPriorityValue> = None;
    let mut paging_origin: Option<PagingOriginValue> = None;
    let mut assistance_data: Option<AssistanceDataForPagingValue> = None;

    for ie in &paging.protocol_i_es.0 {
        match &ie.value {
            PagingProtocolIEs_EntryValue::Id_UEPagingIdentity(identity) => {
                ue_paging_identity = Some(parse_ue_paging_identity(identity)?);
            }
            PagingProtocolIEs_EntryValue::Id_TAIListForPaging(list) => {
                tai_list_for_paging = Some(parse_tai_list_for_paging(list));
            }
            PagingProtocolIEs_EntryValue::Id_PagingDRX(drx) => {
                paging_drx = Some(drx.clone().try_into()?);
            }
            PagingProtocolIEs_EntryValue::Id_PagingPriority(priority) => {
                paging_priority = Some(priority.clone().try_into()?);
            }
            PagingProtocolIEs_EntryValue::Id_PagingOrigin(origin) => {
                paging_origin = Some(origin.clone().try_into()?);
            }
            PagingProtocolIEs_EntryValue::Id_AssistanceDataForPaging(assistance) => {
                assistance_data = Some(parse_assistance_data_for_paging(assistance));
            }
            _ => {
                // Ignore other IEs
            }
        }
    }

    Ok(PagingData {
        ue_paging_identity: ue_paging_identity
            .ok_or_else(|| PagingError::MissingMandatoryIe("UEPagingIdentity".to_string()))?,
        tai_list_for_paging: tai_list_for_paging
            .ok_or_else(|| PagingError::MissingMandatoryIe("TAIListForPaging".to_string()))?,
        paging_drx,
        paging_priority,
        paging_origin,
        assistance_data,
    })
}

/// The Assistance Data for Paging as received. The IE's criticality is
/// `ignore` (TS 38.413 §9.2.4.1) and it "may be used" (§8.5.1.2), so an IE
/// that decodes is never a reason to fail the PAGING: an extension value this
/// codec does not know reads as absent, and a cell whose identity has the
/// wrong bit length (which the decoder cannot produce) is dropped rather than
/// read as cell 0. An IE whose bytes do not decode at all is a transfer syntax
/// error (TS 38.413 §10.2): it fails the whole PDU decode in `decode_paging`,
/// exactly as before this IE was read.
fn parse_assistance_data_for_paging(
    value: &AssistanceDataForPaging,
) -> AssistanceDataForPagingValue {
    let recommended_cells = value
        .assistance_data_for_recommended_cells
        .iter()
        .flat_map(|cells| {
            cells
                .recommended_cells_for_paging
                .recommended_cell_list
                .0
                .iter()
        })
        .filter_map(|item| {
            let cgi = match &item.ngran_cgi {
                NGRAN_CGI::NR_CGI(nr) if nr.nr_cell_identity.0.len() == 36 => {
                    let NrCgi {
                        plmn_identity,
                        nr_cell_identity,
                    } = parse_nr_cgi(nr);
                    RecommendedCellCgi::Nr {
                        plmn_identity,
                        nr_cell_identity,
                    }
                }
                NGRAN_CGI::EUTRA_CGI(eutra) if eutra.eutra_cell_identity.0.len() == 28 => {
                    RecommendedCellCgi::Eutra {
                        plmn_identity: eutra.plmn_identity.0.as_slice().try_into().ok()?,
                        eutra_cell_identity: eutra
                            .eutra_cell_identity
                            .0
                            .iter()
                            .fold(0u32, |acc, bit| (acc << 1) | u32::from(*bit)),
                    }
                }
                _ => return None,
            };
            Some(RecommendedCellValue {
                cgi,
                time_stayed_in_cell: item.time_stayed_in_cell.as_ref().map(|t| t.0),
            })
        })
        .collect();
    let paging_attempt_information =
        value
            .paging_attempt_information
            .as_ref()
            .map(|pai| PagingAttemptInfoValue {
                paging_attempt_count: pai.paging_attempt_count.0,
                intended_number_of_paging_attempts: pai.intended_number_of_paging_attempts.0,
                // An extension value this codec does not know reads as absent.
                next_paging_area_scope: pai.next_paging_area_scope.as_ref().and_then(|scope| {
                    match scope.0 {
                        NextPagingAreaScope::SAME => Some(NextPagingAreaScopeValue::Same),
                        NextPagingAreaScope::CHANGED => Some(NextPagingAreaScopeValue::Changed),
                        _ => None,
                    }
                }),
            });
    AssistanceDataForPagingValue {
        recommended_cells,
        paging_attempt_information,
    }
}

fn parse_ue_paging_identity(
    identity: &UEPagingIdentity,
) -> Result<UePagingIdentityValue, PagingError> {
    match identity {
        UEPagingIdentity::FiveG_S_TMSI(tmsi) => {
            Ok(UePagingIdentityValue::FiveGSTmsi(parse_five_g_s_tmsi(tmsi)))
        }
        _ => Err(PagingError::InvalidIeValue(
            "Unsupported UEPagingIdentity type".to_string(),
        )),
    }
}

fn parse_five_g_s_tmsi(tmsi: &FiveG_S_TMSI) -> FiveGSTmsi {
    // AMF Set ID is 10 bits
    let amf_set_id = if tmsi.amf_set_id.0.len() >= 10 {
        let mut value: u16 = 0;
        for (i, bit) in tmsi.amf_set_id.0.iter().take(10).enumerate() {
            if *bit {
                value |= 1 << (9 - i);
            }
        }
        value
    } else {
        0
    };

    // AMF Pointer is 6 bits
    let amf_pointer = if tmsi.amf_pointer.0.len() >= 6 {
        let mut value: u8 = 0;
        for (i, bit) in tmsi.amf_pointer.0.iter().take(6).enumerate() {
            if *bit {
                value |= 1 << (5 - i);
            }
        }
        value
    } else {
        0
    };

    let five_g_tmsi: [u8; 4] = tmsi
        .five_g_tmsi
        .0
        .as_slice()
        .try_into()
        .unwrap_or([0, 0, 0, 0]);

    FiveGSTmsi {
        amf_set_id,
        amf_pointer,
        five_g_tmsi,
    }
}

fn parse_tai_list_for_paging(list: &TAIListForPaging) -> Vec<Tai> {
    list.0.iter().map(|item| parse_tai(&item.tai)).collect()
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode a Paging message to bytes
///
/// # Arguments
/// * `params` - Parameters for the Paging message
///
/// # Returns
/// * `Ok(Vec<u8>)` - The encoded bytes
/// * `Err(PagingError)` - If building or encoding fails
pub fn encode_paging(params: &PagingParams) -> Result<Vec<u8>, PagingError> {
    let pdu = build_paging(params)?;
    Ok(encode_ngap_pdu(&pdu)?)
}

/// Decode and parse a Paging message from bytes
///
/// # Arguments
/// * `bytes` - The encoded bytes
///
/// # Returns
/// * `Ok(PagingData)` - The parsed message data
/// * `Err(PagingError)` - If decoding or parsing fails
pub fn decode_paging(bytes: &[u8]) -> Result<PagingData, PagingError> {
    let pdu = decode_ngap_pdu(bytes)?;
    parse_paging(&pdu)
}

/// Check if an NGAP PDU is a Paging message
pub fn is_paging(pdu: &NGAP_PDU) -> bool {
    matches!(
        pdu,
        NGAP_PDU::InitiatingMessage(msg)
            if matches!(msg.value, InitiatingMessageValue::Id_Paging(_))
    )
}

#[cfg(test)]
mod tests {
    /// The DRX cycle in radio frames is the `T` of TS 38.304 §7.1, so a wrong
    /// mapping puts every paged UE's occasion in the wrong frame (issue #99).
    #[test]
    fn the_drx_cycle_converts_to_radio_frames() {
        assert_eq!(PagingDrxValue::V32.radio_frames(), 32);
        assert_eq!(PagingDrxValue::V64.radio_frames(), 64);
        assert_eq!(PagingDrxValue::V128.radio_frames(), 128);
        assert_eq!(PagingDrxValue::V256.radio_frames(), 256);
    }

    use super::*;

    fn create_test_params() -> PagingParams {
        PagingParams {
            ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(FiveGSTmsi {
                amf_set_id: 1,
                amf_pointer: 0,
                five_g_tmsi: [0x00, 0x00, 0x00, 0x01],
            }),
            tai_list_for_paging: vec![Tai {
                plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                tac: [0x00, 0x00, 0x01],           // TAC = 1
            }],
            paging_drx: None,
            paging_priority: None,
            paging_origin: None,
            assistance_data: None,
        }
    }

    /// PAGINGs as nextgcore's amfd puts them on the wire (nextgsim#464), captured
    /// from `nextgcore-amfd` at nextgcore `9e3d841a` (`ngap_asn1.rs`
    /// `build_paging_with_assistance_asn1` / `build_paging_with_attempt_asn1` /
    /// `build_paging_asn1`): 5G-S-TMSI set 1, pointer 0, TMSI 1; TAI 999-70 TAC 1.
    ///
    /// Demo d440c-nwdaf-paging, NWDAF-steered attempt 1 of 5, next scope
    /// `changed`, Recommended Cells [999-70 NCI 0x10, 999-70 NCI 0x20]
    /// (`paging_assistance_with_cells_aper`).
    const AMFD_PAGING_CELLS_ATTEMPT_1: &str = "001840330000030073400700080000000001006740070099f907000001000b401660100099f90700000001000099f90700000002040220";
    /// Base `paging.strategy` staged attempt 2 of 5, next scope `same`, no cells.
    const AMFD_PAGING_ATTEMPT_2: &str =
        "001840200000030073400700080000000001006740070099f907000001000b4003241200";
    /// Attempt 5 of 5: the last, so no Next Paging Area Scope.
    const AMFD_PAGING_ATTEMPT_5_LAST: &str =
        "001840200000030073400700080000000001006740070099f907000001000b4003204200";
    /// The default `registration-area` PAGING: no Assistance Data.
    const AMFD_PAGING_DEFAULT: &str = "001840190000020073400700080000000001006740070099f907000001";

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    const PLMN_999_70: [u8; 3] = [0x99, 0xF9, 0x07];

    fn nr_cell(nci: u64) -> RecommendedCellValue {
        RecommendedCellValue {
            cgi: RecommendedCellCgi::Nr {
                plmn_identity: PLMN_999_70,
                nr_cell_identity: nci,
            },
            time_stayed_in_cell: None,
        }
    }

    /// nextgsim#464: the gNB reads amfd's Assistance Data for Paging off amfd's
    /// own bytes: the Recommended Cells in order and the Paging Attempt
    /// Information (TS 38.413 §9.3.1.69-72). Before, the IE was skipped and
    /// `PagingData` had nowhere to put it.
    #[test]
    fn amfd_paging_assistance_decodes_cells_in_order_and_the_attempt() {
        let parsed = decode_paging(&unhex(AMFD_PAGING_CELLS_ATTEMPT_1)).expect("decode");
        assert_eq!(
            parsed.assistance_data,
            Some(AssistanceDataForPagingValue {
                recommended_cells: vec![nr_cell(0x10), nr_cell(0x20)],
                paging_attempt_information: Some(PagingAttemptInfoValue {
                    paging_attempt_count: 1,
                    intended_number_of_paging_attempts: 5,
                    next_paging_area_scope: Some(NextPagingAreaScopeValue::Changed),
                }),
            })
        );
        // The rest of the PAGING is untouched by the new IE.
        assert_eq!(parsed.tai_list_for_paging.len(), 1);
        assert_eq!(parsed.tai_list_for_paging[0].plmn_identity, PLMN_999_70);
        assert_eq!(parsed.tai_list_for_paging[0].tac, [0, 0, 1]);
    }

    /// A staged attempt carries only the Paging Attempt Information; the last
    /// attempt carries no Next Paging Area Scope; the default PAGING carries no
    /// Assistance Data at all, and parses exactly as before.
    #[test]
    fn amfd_paging_attempt_only_last_attempt_and_default() {
        let a2 = decode_paging(&unhex(AMFD_PAGING_ATTEMPT_2)).expect("decode");
        assert_eq!(
            a2.assistance_data,
            Some(AssistanceDataForPagingValue {
                recommended_cells: vec![],
                paging_attempt_information: Some(PagingAttemptInfoValue {
                    paging_attempt_count: 2,
                    intended_number_of_paging_attempts: 5,
                    next_paging_area_scope: Some(NextPagingAreaScopeValue::Same),
                }),
            })
        );
        let a5 = decode_paging(&unhex(AMFD_PAGING_ATTEMPT_5_LAST)).expect("decode");
        assert_eq!(
            a5.assistance_data
                .and_then(|a| a.paging_attempt_information)
                .map(|p| (p.paging_attempt_count, p.next_paging_area_scope)),
            Some((5, None))
        );
        let none = decode_paging(&unhex(AMFD_PAGING_DEFAULT)).expect("decode");
        assert_eq!(none.assistance_data, None);
    }

    /// This builder encodes the IE byte-identically to amfd, so tests that build
    /// a PAGING here exercise the bytes the gNB really receives.
    #[test]
    fn the_builder_encodes_assistance_exactly_as_amfd_does() {
        let mut params = create_test_params();
        params.tai_list_for_paging[0].plmn_identity = PLMN_999_70;
        params.assistance_data = Some(AssistanceDataForPagingValue {
            recommended_cells: vec![nr_cell(0x10), nr_cell(0x20)],
            paging_attempt_information: Some(PagingAttemptInfoValue {
                paging_attempt_count: 1,
                intended_number_of_paging_attempts: 5,
                next_paging_area_scope: Some(NextPagingAreaScopeValue::Changed),
            }),
        });
        assert_eq!(
            encode_paging(&params).expect("encode"),
            unhex(AMFD_PAGING_CELLS_ATTEMPT_1)
        );
        for (vector, count, scope) in [
            (
                AMFD_PAGING_ATTEMPT_2,
                2,
                Some(NextPagingAreaScopeValue::Same),
            ),
            (AMFD_PAGING_ATTEMPT_5_LAST, 5, None),
        ] {
            params.assistance_data = Some(AssistanceDataForPagingValue {
                recommended_cells: vec![],
                paging_attempt_information: Some(PagingAttemptInfoValue {
                    paging_attempt_count: count,
                    intended_number_of_paging_attempts: 5,
                    next_paging_area_scope: scope,
                }),
            });
            assert_eq!(
                encode_paging(&params).expect("encode"),
                unhex(vector),
                "{vector}"
            );
        }
        params.assistance_data = None;
        assert_eq!(
            encode_paging(&params).expect("encode"),
            unhex(AMFD_PAGING_DEFAULT)
        );
    }

    /// AMFD_PAGING_ATTEMPT_2 with its Next Paging Area Scope replaced by an
    /// extension value this codec does not know (index `ext_index` past the root
    /// {same, changed}): the PDU and IE lengths grow by one octet.
    fn attempt_2_with_scope_extension(ext_index: u8) -> Vec<u8> {
        // ...TAI IE | id 11, criticality ignore, length 4 | the value.
        // Value bits: ADP ext 0, opt {cells 0, PAI 1, ext 0}; PAI ext 0, opt
        // {scope 1, ext 0}; count ext 0, 4 bits 1; intended ext 0, 4 bits 4;
        // scope ext 1, then a normally small number (0 + 6 bits) = ext_index.
        let mut bytes =
            unhex("001840210000030073400700080000000001006740070099f907000001000b400424124000");
        // The 6-bit index sits in the last 5 bits of the third value octet and
        // the first bit of the fourth.
        let n = bytes.len();
        bytes[n - 2] |= ext_index >> 1;
        bytes[n - 1] = (ext_index & 1) << 7;
        bytes
    }

    /// The raw Next Paging Area Scope the generated decoder reads from `bytes`.
    fn raw_scope(bytes: &[u8]) -> Option<u8> {
        let NGAP_PDU::InitiatingMessage(msg) = decode_ngap_pdu(bytes).unwrap() else {
            panic!("not an InitiatingMessage")
        };
        let InitiatingMessageValue::Id_Paging(paging) = msg.value else {
            panic!("not a PAGING")
        };
        paging
            .protocol_i_es
            .0
            .iter()
            .find_map(|ie| match &ie.value {
                PagingProtocolIEs_EntryValue::Id_AssistanceDataForPaging(a) => a
                    .paging_attempt_information
                    .as_ref()
                    .and_then(|p| p.next_paging_area_scope.as_ref())
                    .map(|s| s.0),
                _ => None,
            })
    }

    /// Criticality ignore at work: a Next Paging Area Scope extension value
    /// this codec does not know reads as absent, and the rest of the PAGING,
    /// the attempt included, is decoded unchanged.
    #[test]
    fn an_unknown_next_paging_area_scope_extension_reads_as_absent() {
        let base = decode_paging(&unhex(AMFD_PAGING_ATTEMPT_2)).expect("decode");
        // The two fixtures really are two different values, neither in the root.
        let raw: Vec<_> = [0, 1]
            .iter()
            .map(|&i| raw_scope(&attempt_2_with_scope_extension(i)))
            .collect();
        assert_ne!(raw[0], raw[1], "{raw:?}");
        for r in &raw {
            assert!(!matches!(r, Some(0) | Some(1) | None), "{raw:?}");
        }
        for ext_index in [0, 1] {
            let parsed = decode_paging(&attempt_2_with_scope_extension(ext_index))
                .unwrap_or_else(|e| panic!("extension index {ext_index}: {e}"));
            assert_eq!(
                parsed.assistance_data,
                Some(AssistanceDataForPagingValue {
                    recommended_cells: vec![],
                    paging_attempt_information: Some(PagingAttemptInfoValue {
                        paging_attempt_count: 2,
                        intended_number_of_paging_attempts: 5,
                        next_paging_area_scope: None,
                    }),
                }),
                "extension index {ext_index}"
            );
            assert_eq!(parsed.tai_list_for_paging.len(), 1);
            assert_eq!(
                parsed.tai_list_for_paging[0].plmn_identity,
                base.tai_list_for_paging[0].plmn_identity
            );
            assert_eq!(
                parsed.tai_list_for_paging[0].tac,
                base.tai_list_for_paging[0].tac
            );
            match (&parsed.ue_paging_identity, &base.ue_paging_identity) {
                (UePagingIdentityValue::FiveGSTmsi(a), UePagingIdentityValue::FiveGSTmsi(b)) => {
                    assert_eq!(
                        (a.amf_set_id, a.amf_pointer, a.five_g_tmsi),
                        (b.amf_set_id, b.amf_pointer, b.five_g_tmsi)
                    );
                }
            }
        }
    }

    /// Undecodable IE bytes are a transfer syntax error (TS 38.413 §10.2), not a
    /// criticality case: the PDU decode fails, as it did before this IE was read.
    #[test]
    fn undecodable_assistance_bytes_fail_the_pdu_decode() {
        let bytes =
            unhex("001840200000030073400700080000000001006740070099f907000001000b4003ffffff");
        assert!(decode_paging(&bytes).is_err());
    }

    /// Cell identities wider than their BIT STRING (36 bits NR, 28 bits E-UTRA,
    /// TS 38.413 §9.3.1.7, §9.3.1.9) are refused by name, not truncated.
    #[test]
    fn the_builder_refuses_over_wide_cell_identities() {
        let mut params = create_test_params();
        for (cgi, widest) in [
            (
                (|id: u64| RecommendedCellCgi::Nr {
                    plmn_identity: PLMN_999_70,
                    nr_cell_identity: id,
                }) as fn(u64) -> RecommendedCellCgi,
                (1u64 << 36) - 1,
            ),
            (
                (|id: u64| RecommendedCellCgi::Eutra {
                    plmn_identity: PLMN_999_70,
                    eutra_cell_identity: id as u32,
                }) as fn(u64) -> RecommendedCellCgi,
                (1u64 << 28) - 1,
            ),
        ] {
            let with = |id| {
                Some(AssistanceDataForPagingValue {
                    recommended_cells: vec![RecommendedCellValue {
                        cgi: cgi(id),
                        time_stayed_in_cell: None,
                    }],
                    paging_attempt_information: None,
                })
            };
            params.assistance_data = with(widest);
            assert!(encode_paging(&params).is_ok(), "{widest:#x} fits");
            params.assistance_data = with(widest + 1);
            match encode_paging(&params) {
                Err(PagingError::InvalidIeValue(m)) => assert!(m.contains("wider than"), "{m}"),
                other => panic!("{:#x}: expected InvalidIeValue, got {other:?}", widest + 1),
            }
        }
    }

    /// An E-UTRA cell and Time Stayed in Cell survive the round trip, and the
    /// ASN.1 bounds (16 cells, 4095 s, attempt 1..16) are refused at build time.
    #[test]
    fn assistance_round_trips_eutra_and_time_stayed_and_refuses_out_of_range() {
        let mut params = create_test_params();
        let eutra = RecommendedCellValue {
            cgi: RecommendedCellCgi::Eutra {
                plmn_identity: PLMN_999_70,
                eutra_cell_identity: 0x0AB_CDEF,
            },
            time_stayed_in_cell: Some(4095),
        };
        let assistance = AssistanceDataForPagingValue {
            recommended_cells: vec![
                RecommendedCellValue {
                    time_stayed_in_cell: Some(17),
                    ..nr_cell(0xF_FFFF_FFFF)
                },
                eutra,
            ],
            paging_attempt_information: None,
        };
        params.assistance_data = Some(assistance.clone());
        let parsed = decode_paging(&encode_paging(&params).expect("encode")).expect("decode");
        assert_eq!(parsed.assistance_data, Some(assistance));

        params.assistance_data = Some(AssistanceDataForPagingValue {
            recommended_cells: vec![nr_cell(0x10); MAX_RECOMMENDED_CELLS + 1],
            paging_attempt_information: None,
        });
        assert!(matches!(
            encode_paging(&params),
            Err(PagingError::InvalidIeValue(_))
        ));
        params.assistance_data = Some(AssistanceDataForPagingValue {
            recommended_cells: vec![RecommendedCellValue {
                time_stayed_in_cell: Some(4096),
                ..nr_cell(0x10)
            }],
            paging_attempt_information: None,
        });
        assert!(matches!(
            encode_paging(&params),
            Err(PagingError::InvalidIeValue(_))
        ));
        for (count, intended) in [(0, 5), (1, 17)] {
            params.assistance_data = Some(AssistanceDataForPagingValue {
                recommended_cells: vec![],
                paging_attempt_information: Some(PagingAttemptInfoValue {
                    paging_attempt_count: count,
                    intended_number_of_paging_attempts: intended,
                    next_paging_area_scope: None,
                }),
            });
            assert!(matches!(
                encode_paging(&params),
                Err(PagingError::InvalidIeValue(_))
            ));
        }
    }

    #[test]
    fn test_build_paging() {
        let params = create_test_params();
        let result = build_paging(&params);
        assert!(result.is_ok());

        let pdu = result.unwrap();
        match pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_PAGING);
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_encode_paging() {
        let params = create_test_params();
        let result = encode_paging(&params);
        assert!(result.is_ok());

        let bytes = result.unwrap();
        assert!(!bytes.is_empty());
    }

    #[test]
    fn test_paging_roundtrip() {
        let params = create_test_params();

        // Build and encode
        let pdu = build_paging(&params).expect("Failed to build message");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Decode
        let decoded_pdu = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify structure
        match decoded_pdu {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, ID_PAGING);
                match msg.value {
                    InitiatingMessageValue::Id_Paging(paging_msg) => {
                        // Verify we have the expected IEs (at least 2 mandatory)
                        assert!(paging_msg.protocol_i_es.0.len() >= 2);
                    }
                    _ => panic!("Expected Paging"),
                }
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    #[test]
    fn test_paging_parse_roundtrip() {
        let params = create_test_params();

        // Build, encode, decode, and parse
        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        // Verify parsed data matches original params
        match (&parsed.ue_paging_identity, &params.ue_paging_identity) {
            (
                UePagingIdentityValue::FiveGSTmsi(parsed_tmsi),
                UePagingIdentityValue::FiveGSTmsi(params_tmsi),
            ) => {
                assert_eq!(parsed_tmsi.amf_set_id, params_tmsi.amf_set_id);
                assert_eq!(parsed_tmsi.amf_pointer, params_tmsi.amf_pointer);
                assert_eq!(parsed_tmsi.five_g_tmsi, params_tmsi.five_g_tmsi);
            }
        }

        assert_eq!(
            parsed.tai_list_for_paging.len(),
            params.tai_list_for_paging.len()
        );
        assert_eq!(
            parsed.tai_list_for_paging[0].plmn_identity,
            params.tai_list_for_paging[0].plmn_identity
        );
        assert_eq!(
            parsed.tai_list_for_paging[0].tac,
            params.tai_list_for_paging[0].tac
        );
    }

    #[test]
    fn test_paging_with_optional_ies() {
        let mut params = create_test_params();
        params.paging_drx = Some(PagingDrxValue::V128);
        params.paging_priority = Some(PagingPriorityValue::PrioLevel1);
        params.paging_origin = Some(PagingOriginValue::Non3gpp);

        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.paging_drx, Some(PagingDrxValue::V128));
        assert_eq!(
            parsed.paging_priority,
            Some(PagingPriorityValue::PrioLevel1)
        );
        assert_eq!(parsed.paging_origin, Some(PagingOriginValue::Non3gpp));
    }

    #[test]
    fn test_paging_with_multiple_tais() {
        let mut params = create_test_params();
        params.tai_list_for_paging = vec![
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x01],
            },
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x02],
            },
            Tai {
                plmn_identity: [0x00, 0xF1, 0x10],
                tac: [0x00, 0x00, 0x03],
            },
        ];

        let encoded = encode_paging(&params).expect("Failed to encode");
        let parsed = decode_paging(&encoded).expect("Failed to decode and parse");

        assert_eq!(parsed.tai_list_for_paging.len(), 3);
        assert_eq!(parsed.tai_list_for_paging[0].tac, [0x00, 0x00, 0x01]);
        assert_eq!(parsed.tai_list_for_paging[1].tac, [0x00, 0x00, 0x02]);
        assert_eq!(parsed.tai_list_for_paging[2].tac, [0x00, 0x00, 0x03]);
    }

    #[test]
    fn test_paging_priority_conversion() {
        // Test all priority values
        let priorities = [
            PagingPriorityValue::PrioLevel1,
            PagingPriorityValue::PrioLevel2,
            PagingPriorityValue::PrioLevel3,
            PagingPriorityValue::PrioLevel4,
            PagingPriorityValue::PrioLevel5,
            PagingPriorityValue::PrioLevel6,
            PagingPriorityValue::PrioLevel7,
            PagingPriorityValue::PrioLevel8,
        ];

        for priority in priorities {
            let paging_priority: PagingPriority = priority.into();
            let converted_back: PagingPriorityValue = paging_priority.try_into().unwrap();
            assert_eq!(priority, converted_back);
        }
    }

    #[test]
    fn test_paging_drx_conversion() {
        // Test all DRX values
        let drx_values = [
            PagingDrxValue::V32,
            PagingDrxValue::V64,
            PagingDrxValue::V128,
            PagingDrxValue::V256,
        ];

        for drx in drx_values {
            let paging_drx: PagingDRX = drx.into();
            let converted_back: PagingDrxValue = paging_drx.try_into().unwrap();
            assert_eq!(drx, converted_back);
        }
    }

    #[test]
    fn test_is_paging() {
        let params = create_test_params();
        let pdu = build_paging(&params).expect("Failed to build message");
        assert!(is_paging(&pdu));
    }
}
