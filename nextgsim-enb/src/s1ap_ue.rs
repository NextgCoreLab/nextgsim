//! UE-associated S1AP the eNB drives in this increment: Initial UE Message
//! (TS 36.413 §8.6.2.1), Uplink NAS Transport (§8.6.2.3) and the receive side
//! of Downlink NAS Transport (§8.6.2.2), with the §10 error handling and the
//! UE-associated Error Indication of §10.6 / §8.7.2.
//!
//! Pure functions, like [`crate::s1ap`]: PDUs in, PDUs and decisions out.

use nextgsim_common::Plmn;
use nextgsim_s1ap::codec::{self, *};

use crate::s1ap::{DiagnosedIe, MAX_KNOWN_PROCEDURE_CODE};

/// `ENB-UE-S1AP-ID ::= INTEGER (0..16777215)` (§9.2.3.4).
pub const MAX_ENB_UE_S1AP_ID: u32 = (1 << 24) - 1;

/// Initial UE Message / Uplink / Downlink NAS Transport are all
/// `CRITICALITY ignore` procedures (§9.3.2 S1AP-PDU-Descriptions).
const NAS_PROCEDURE_CRITICALITY: u8 = Criticality::IGNORE;

fn plmn_identity(p: &Plmn) -> PLMNidentity {
    TBCD_STRING(p.encode().to_vec())
}

fn bit_string(value: u32, bits: u32) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0> {
    (0..bits).rev().map(|i| (value >> i) & 1 == 1).collect()
}

/// Where the UE is: the TAI and E-UTRAN CGI the eNB reports with its NAS
/// (§9.2.3.16 TAI, §9.2.1.38 E-UTRAN CGI).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UeLocation {
    /// The PLMN the UE selected (RRCConnectionSetupComplete
    /// `selectedPLMN-Identity`), which both identities are built from.
    pub plmn: Plmn,
    /// Tracking Area Code of the cell.
    pub tac: u16,
    /// 28-bit E-UTRAN Cell Identity.
    pub cell_identity: u32,
}

fn tai(l: &UeLocation) -> TAI {
    TAI {
        plm_nidentity: plmn_identity(&l.plmn),
        tac: TAC(l.tac.to_be_bytes().to_vec()),
        ie_extensions: None,
    }
}

fn ecgi(l: &UeLocation) -> EUTRAN_CGI {
    EUTRAN_CGI {
        plm_nidentity: plmn_identity(&l.plmn),
        cell_id: CellIdentity(bit_string(l.cell_identity, 28)),
        ie_extensions: None,
    }
}

/// Build INITIAL UE MESSAGE (§9.1.7.1), IEs in the message-table order with
/// the criticality `InitialUEMessage-IEs` assigns each.
///
/// `rrc_cause` is the RRC establishment cause index; TS 36.331 §6.2.2
/// EstablishmentCause and §9.2.1.3a RRC Establishment Cause share it.
/// `s_tmsi` is included when the UE identified itself with one in
/// RRCConnectionRequest (§8.6.2.1).
pub fn build_initial_ue_message(
    enb_ue_s1ap_id: u32,
    nas: &[u8],
    location: &UeLocation,
    rrc_cause: u8,
    s_tmsi: Option<(u8, u32)>,
) -> S1AP_PDU {
    let mut ies = vec![
        InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                enb_ue_s1ap_id,
            )),
        },
        InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_NAS_PDU),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(nas.to_vec())),
        },
        InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_TAI),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_TAI(tai(location)),
        },
        InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_EUTRAN_CGI),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_EUTRAN_CGI(ecgi(location)),
        },
        InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_RRC_ESTABLISHMENT_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_RRC_Establishment_Cause(
                RRC_Establishment_Cause(rrc_cause),
            ),
        },
    ];
    if let Some((mmec, m_tmsi)) = s_tmsi {
        ies.push(InitialUEMessageProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_S_TMSI),
            criticality: Criticality(Criticality::REJECT),
            value: InitialUEMessageProtocolIEs_EntryValue::Id_S_TMSI(S_TMSI {
                mmec: MME_Code(vec![mmec]),
                m_tmsi: M_TMSI(m_tmsi.to_be_bytes().to_vec()),
                ie_extensions: None,
            }),
        });
    }
    S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_INITIAL_UE_MESSAGE),
        criticality: Criticality(NAS_PROCEDURE_CRITICALITY),
        value: InitiatingMessageValue::Id_initialUEMessage(InitialUEMessage {
            protocol_i_es: InitialUEMessageProtocolIEs(ies),
        }),
    })
}

/// Build UPLINK NAS TRANSPORT (§9.1.7.3).
pub fn build_uplink_nas_transport(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    nas: &[u8],
    location: &UeLocation,
) -> S1AP_PDU {
    let ies = vec![
        UplinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UplinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                mme_ue_s1ap_id,
            )),
        },
        UplinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UplinkNASTransportProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                enb_ue_s1ap_id,
            )),
        },
        UplinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_NAS_PDU),
            criticality: Criticality(Criticality::REJECT),
            value: UplinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(nas.to_vec())),
        },
        UplinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_EUTRAN_CGI),
            criticality: Criticality(Criticality::IGNORE),
            value: UplinkNASTransportProtocolIEs_EntryValue::Id_EUTRAN_CGI(ecgi(location)),
        },
        UplinkNASTransportProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_TAI),
            criticality: Criticality(Criticality::IGNORE),
            value: UplinkNASTransportProtocolIEs_EntryValue::Id_TAI(tai(location)),
        },
    ];
    S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UPLINK_NAS_TRANSPORT),
        criticality: Criticality(NAS_PROCEDURE_CRITICALITY),
        value: InitiatingMessageValue::Id_uplinkNASTransport(UplinkNASTransport {
            protocol_i_es: UplinkNASTransportProtocolIEs(ies),
        }),
    })
}

/// Build a UE-associated ERROR INDICATION (§9.1.8.2): the UE S1AP IDs it
/// concerns, then Cause and, when given, Criticality Diagnostics. Message-table
/// order: MME UE S1AP ID, eNB UE S1AP ID, Cause, Criticality Diagnostics.
pub fn build_ue_error_indication(
    mme_ue_s1ap_id: Option<u32>,
    enb_ue_s1ap_id: Option<u32>,
    cause: Cause,
    procedure: Option<(u8, u8, u8)>,
    ies: &[DiagnosedIe],
) -> S1AP_PDU {
    let mut pdu = crate::s1ap::build_error_indication(cause, procedure, ies);
    if let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_ErrorIndication(ei),
        ..
    }) = &mut pdu
    {
        let mut ids = Vec::new();
        if let Some(id) = mme_ue_s1ap_id {
            ids.push(ErrorIndicationProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
                criticality: Criticality(Criticality::IGNORE),
                value: ErrorIndicationProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(id)),
            });
        }
        if let Some(id) = enb_ue_s1ap_id {
            ids.push(ErrorIndicationProtocolIEs_Entry {
                id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
                criticality: Criticality(Criticality::IGNORE),
                value: ErrorIndicationProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(id)),
            });
        }
        ids.append(&mut ei.protocol_i_es.0);
        ei.protocol_i_es.0 = ids;
    }
    pdu
}

/// Build UE CONTEXT RELEASE REQUEST (§9.1.4.5) with a Radio Network cause:
/// the eNB asks the MME to release the UE (§8.3.2), e.g. on radio loss.
pub fn build_ue_context_release_request(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    radio_network_cause: u8,
) -> S1AP_PDU {
    let ies = vec![
        UEContextReleaseRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(
                MME_UE_S1AP_ID(mme_ue_s1ap_id),
            ),
        },
        UEContextReleaseRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::REJECT),
            value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(
                ENB_UE_S1AP_ID(enb_ue_s1ap_id),
            ),
        },
        UEContextReleaseRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseRequestProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(
                CauseRadioNetwork(radio_network_cause),
            )),
        },
    ];
    S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE_REQUEST),
        // uEContextReleaseRequest: CRITICALITY ignore (§9.3.2).
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_UEContextReleaseRequest(UEContextReleaseRequest {
            protocol_i_es: UEContextReleaseRequestProtocolIEs(ies),
        }),
    })
}

/// Build UE CONTEXT RELEASE COMPLETE (§9.1.4.7), the successful outcome of
/// the UE Context Release procedure (§8.3.3.2).
pub fn build_ue_context_release_complete(mme_ue_s1ap_id: u32, enb_ue_s1ap_id: u32) -> S1AP_PDU {
    let ies = vec![
        UEContextReleaseCompleteProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(
                MME_UE_S1AP_ID(mme_ue_s1ap_id),
            ),
        },
        UEContextReleaseCompleteProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: UEContextReleaseCompleteProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(
                ENB_UE_S1AP_ID(enb_ue_s1ap_id),
            ),
        },
    ];
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        // uEContextRelease: CRITICALITY reject (§9.3.2).
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_UEContextRelease(UEContextReleaseComplete {
            protocol_i_es: UEContextReleaseCompleteProtocolIEs(ies),
        }),
    })
}

/// The UE S1AP IDs of a UE CONTEXT RELEASE COMMAND (§9.2.3.18 UE S1AP IDs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReleaseIds {
    /// `uE-S1AP-ID-pair`.
    Pair {
        /// MME UE S1AP ID.
        mme: u32,
        /// eNB UE S1AP ID.
        enb: u32,
    },
    /// `mME-UE-S1AP-ID` only.
    Mme(u32),
}

/// What the eNB does with an S1AP PDU received once the interface is
/// operational.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum S1Inbound {
    /// DOWNLINK NAS TRANSPORT to relay to the UE. `error_indication` is owed
    /// for notify-criticality IEs not comprehended (§10.3.4.2).
    DownlinkNas {
        /// MME UE S1AP ID.
        mme_ue_s1ap_id: u32,
        /// eNB UE S1AP ID.
        enb_ue_s1ap_id: u32,
        /// The NAS PDU.
        nas: Vec<u8>,
        /// Error Indication to send alongside, if any.
        error_indication: Option<Vec<u8>>,
    },
    /// UE CONTEXT RELEASE COMMAND (§8.3.3).
    ReleaseCommand {
        /// Which UE.
        ids: ReleaseIds,
    },
    /// Nothing to relay; send this Error Indication (§10).
    ErrorIndication(Vec<u8>),
    /// A PDU this build does not act on, for the log.
    NotHandled(String),
}

fn encode_or_log(pdu: &S1AP_PDU) -> Option<Vec<u8>> {
    match codec::encode_s1ap_pdu(pdu) {
        Ok(b) => Some(b),
        Err(e) => {
            log::error!("cannot encode ERROR INDICATION: {e}");
            None
        }
    }
}

fn protocol(v: u8) -> Cause {
    Cause::Protocol(CauseProtocol(v))
}

fn handle_downlink_nas(dl: &DownlinkNASTransport, unknown: &[UnknownIe]) -> S1Inbound {
    let mut mme = None;
    let mut enb = None;
    let mut nas = None;
    for ie in &dl.protocol_i_es.0 {
        match &ie.value {
            DownlinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => mme = Some(v.0),
            DownlinkNASTransportProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb = Some(v.0),
            DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(v) => nas = Some(v.0.clone()),
            _ => {}
        }
    }
    let procedure = Some((
        ID_DOWNLINK_NAS_TRANSPORT,
        TriggeringMessage::INITIATING_MESSAGE,
        NAS_PROCEDURE_CRITICALITY,
    ));
    // §10.3.4.2, Class 2 EP: a `reject` IE not comprehended terminates the
    // procedure and is reported in an Error Indication. Absent criticality is
    // treated as reject.
    let rejected: Vec<DiagnosedIe> = unknown
        .iter()
        .filter(|u| {
            u.criticality
                .as_ref()
                .is_none_or(|c| c.0 == Criticality::REJECT)
        })
        .filter_map(|u| {
            Some(DiagnosedIe {
                id: u16::try_from(u.id).ok()?,
                criticality: Criticality::REJECT,
                type_of_error: TypeOfError::NOT_UNDERSTOOD,
            })
        })
        .collect();
    // §10.3.5: a missing mandatory IE whose criticality is reject (all three
    // here) terminates the procedure, reported as "missing".
    let mut missing = Vec::new();
    for (present, id) in [
        (mme.is_some(), ID_MME_UE_S1AP_ID),
        (enb.is_some(), ID_E_NB_UE_S1AP_ID),
        (nas.is_some(), ID_NAS_PDU),
    ] {
        if !present {
            missing.push(DiagnosedIe {
                id,
                criticality: Criticality::REJECT,
                type_of_error: TypeOfError::MISSING,
            });
        }
    }
    if !rejected.is_empty() || !missing.is_empty() {
        let mut diag = rejected;
        diag.extend(missing);
        return match encode_or_log(&build_ue_error_indication(
            mme,
            enb,
            protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
            procedure,
            &diag,
        )) {
            Some(ei) => S1Inbound::ErrorIndication(ei),
            None => S1Inbound::NotHandled("DOWNLINK NAS TRANSPORT rejected".into()),
        };
    }
    let (Some(mme_ue_s1ap_id), Some(enb_ue_s1ap_id), Some(nas)) = (mme, enb, nas) else {
        unreachable!("checked above");
    };
    let notify: Vec<DiagnosedIe> = unknown
        .iter()
        .filter(|u| u.criticality.as_ref().map(|c| c.0) == Some(Criticality::NOTIFY))
        .filter_map(|u| {
            Some(DiagnosedIe {
                id: u16::try_from(u.id).ok()?,
                criticality: Criticality::NOTIFY,
                type_of_error: TypeOfError::NOT_UNDERSTOOD,
            })
        })
        .collect();
    let error_indication = (!notify.is_empty())
        .then(|| {
            encode_or_log(&build_ue_error_indication(
                Some(mme_ue_s1ap_id),
                Some(enb_ue_s1ap_id),
                protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY),
                procedure,
                &notify,
            ))
        })
        .flatten();
    S1Inbound::DownlinkNas {
        mme_ue_s1ap_id,
        enb_ue_s1ap_id,
        nas,
        error_indication,
    }
}

fn handle_release_command(cmd: &UEContextReleaseCommand, unknown: &[UnknownIe]) -> S1Inbound {
    let ids = cmd.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_S1AP_IDs(
            UE_S1AP_IDs::UE_S1AP_ID_pair(p),
        ) => Some(ReleaseIds::Pair {
            mme: p.mme_ue_s1ap_id.0,
            enb: p.enb_ue_s1ap_id.0,
        }),
        UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_S1AP_IDs(
            UE_S1AP_IDs::MME_UE_S1AP_ID(m),
        ) => Some(ReleaseIds::Mme(m.0)),
        _ => None,
    });
    let procedure = Some((
        ID_UE_CONTEXT_RELEASE,
        TriggeringMessage::INITIATING_MESSAGE,
        Criticality::REJECT,
    ));
    let rejected: Vec<DiagnosedIe> = unknown
        .iter()
        .filter(|u| {
            u.criticality
                .as_ref()
                .is_none_or(|c| c.0 == Criticality::REJECT)
        })
        .filter_map(|u| {
            Some(DiagnosedIe {
                id: u16::try_from(u.id).ok()?,
                criticality: Criticality::REJECT,
                type_of_error: TypeOfError::NOT_UNDERSTOOD,
            })
        })
        .collect();
    // §10.3.4.2: "If a message initiating a procedure that does not have a
    // message to report unsuccessful outcome is received containing ... 'Reject
    // IE' which the receiving node does not comprehend, the receiving node shall
    // terminate the procedure and initiate the Error Indication procedure."
    // UE Context Release has no unsuccessful outcome; §10.3.5 treats a missing
    // UE S1AP IDs (mandatory, reject) the same way.
    let missing = ids.is_none();
    if !rejected.is_empty() || missing {
        let mut diag = rejected;
        if missing {
            diag.push(DiagnosedIe {
                id: ID_UE_S1AP_I_DS,
                criticality: Criticality::REJECT,
                type_of_error: TypeOfError::MISSING,
            });
        }
        let (mme, enb) = match ids {
            Some(ReleaseIds::Pair { mme, enb }) => (Some(mme), Some(enb)),
            Some(ReleaseIds::Mme(m)) => (Some(m), None),
            None => (None, None),
        };
        return encode_or_log(&build_ue_error_indication(
            mme,
            enb,
            protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
            procedure,
            &diag,
        ))
        .map_or_else(
            || S1Inbound::NotHandled("UE CONTEXT RELEASE COMMAND rejected".into()),
            S1Inbound::ErrorIndication,
        );
    }
    S1Inbound::ReleaseCommand {
        ids: ids.expect("checked above"),
    }
}

/// Classify an S1AP PDU received on an operational S1 interface.
pub fn handle_s1_pdu(bytes: &[u8]) -> S1Inbound {
    let (pdu, unknown) = match codec::decode_s1ap_pdu_with_unknown_ies(bytes) {
        Ok(v) => v,
        Err(e) => {
            let header = crate::s1ap::peek_header(bytes);
            return match header {
                // §10.3.4.1: an unknown procedure code; ignore → nothing,
                // otherwise report it with the procedure IEs.
                Some((kind, code, crit)) if code > MAX_KNOWN_PROCEDURE_CODE => {
                    if crit == Criticality::IGNORE {
                        S1Inbound::NotHandled(format!("unknown procedure {code} (ignore)"))
                    } else {
                        let cause = if crit == Criticality::REJECT {
                            CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT
                        } else {
                            CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
                        };
                        encode_or_log(&build_ue_error_indication(
                            None,
                            None,
                            protocol(cause),
                            Some((code, kind, crit)),
                            &[],
                        ))
                        .map_or_else(
                            || S1Inbound::NotHandled("unknown procedure".into()),
                            S1Inbound::ErrorIndication,
                        )
                    }
                }
                // §10.2: transfer syntax error.
                _ => encode_or_log(&build_ue_error_indication(
                    None,
                    None,
                    protocol(CauseProtocol::TRANSFER_SYNTAX_ERROR),
                    None,
                    &[],
                ))
                .map_or_else(
                    || S1Inbound::NotHandled(format!("undecodable S1AP PDU: {e}")),
                    S1Inbound::ErrorIndication,
                ),
            };
        }
    };
    match &pdu {
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_downlinkNASTransport(dl),
            ..
        }) => handle_downlink_nas(dl, &unknown),
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_UEContextRelease(cmd),
            ..
        }) => handle_release_command(cmd, &unknown),
        S1AP_PDU::InitiatingMessage(m) => S1Inbound::NotHandled(format!(
            "initiating message, procedure {} (not supported in this build)",
            m.procedure_code.0
        )),
        S1AP_PDU::SuccessfulOutcome(m) => S1Inbound::NotHandled(format!(
            "successful outcome, procedure {}",
            m.procedure_code.0
        )),
        S1AP_PDU::UnsuccessfulOutcome(m) => S1Inbound::NotHandled(format!(
            "unsuccessful outcome, procedure {}",
            m.procedure_code.0
        )),
    }
}

/// The Error Indication owed for a UE-associated PDU naming an eNB UE S1AP ID
/// this eNB has no context for.
pub fn unknown_enb_ue_s1ap_id_error(mme: u32, enb: u32) -> Option<Vec<u8>> {
    // §10.6: "If a node receives a first ... message that contains an eNB UE
    // S1AP ID that is not known ... the receiving node shall initiate an Error
    // Indication procedure ... with the cause value Unknown eNB UE S1AP ID".
    encode_or_log(&build_ue_error_indication(
        Some(mme),
        Some(enb),
        Cause::RadioNetwork(CauseRadioNetwork(CauseRadioNetwork::UNKNOWN_ENB_UE_S1AP_ID)),
        None,
        &[],
    ))
}

#[cfg(test)]
mod tests;
