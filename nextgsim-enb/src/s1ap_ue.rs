//! UE-associated S1AP the eNB drives: Initial UE Message (TS 36.413
//! §8.6.2.1), Uplink NAS Transport (§8.6.2.3) and the receive side of
//! Downlink NAS Transport (§8.6.2.2); from #186 increment 3 the Initial
//! Context Setup procedure (§8.3.1); with the §10 error handling and the
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
    /// INITIAL CONTEXT SETUP REQUEST to act on (§8.3.1.2).
    InitialContextSetup(Box<InitialContextSetup>),
    /// INITIAL CONTEXT SETUP FAILURE owed for a request the eNB cannot
    /// process (§10.3.4.2 / §10.3.5 for a class 1 procedure).
    InitialContextSetupFailure {
        /// The UE's eNB UE S1AP ID, for the stream.
        enb_ue_s1ap_id: u32,
        /// The encoded failure.
        pdu: Vec<u8>,
    },
    /// S1AP PAGING (§8.5), to broadcast on the cell when its TAI is listed.
    Paging(PagingRequest),
    /// E-RAB SETUP REQUEST (§8.2.1, #283).
    ErabSetup(Box<ErabSetupRequest>),
    /// E-RAB MODIFY REQUEST (§8.2.2, #283).
    ErabModify(Box<ErabModifyRequest>),
    /// E-RAB RELEASE COMMAND (§8.2.3, #283).
    ErabRelease(Box<ErabReleaseCommand>),
    /// Nothing to relay; send this Error Indication (§10).
    ErrorIndication(Vec<u8>),
    /// A PDU this build does not act on, for the log.
    NotHandled(String),
}

/// The paged UE's identity (§9.2.3.13 UE Paging Identity).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PagingIdentity {
    /// `s-TMSI`: MMEC and M-TMSI.
    STmsi {
        /// MME Code.
        mmec: u8,
        /// M-TMSI.
        m_tmsi: u32,
    },
    /// `iMSI`, as decimal digits (§9.2.3.11: TBCD, two digits per octet).
    Imsi(Vec<u8>),
}

/// A decoded S1AP PAGING (§9.1.6), the IEs the eNB acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PagingRequest {
    /// UE Identity Index value (§9.2.3.10, IMSI mod 1024), for the paging
    /// occasion.
    pub identity_index: u16,
    /// Who is paged.
    pub identity: PagingIdentity,
    /// Paging for the CS domain (CN Domain, §9.2.3.22); otherwise PS.
    pub cs_domain: bool,
    /// The TAIs to page in, as (PLMN, TAC).
    pub tais: Vec<(Plmn, u16)>,
}

/// IMSI digits from its TBCD encoding: low nibble first, a 0xF filler ends it.
fn imsi_digits(tbcd: &[u8]) -> Vec<u8> {
    let mut digits = Vec::with_capacity(tbcd.len() * 2);
    for b in tbcd {
        for nibble in [b & 0x0F, b >> 4] {
            if nibble > 9 {
                return digits;
            }
            digits.push(nibble);
        }
    }
    digits
}

fn handle_paging(paging: &Paging) -> S1Inbound {
    let mut index = None;
    let mut identity = None;
    let mut cs_domain = None;
    let mut tais = None;
    for ie in &paging.protocol_i_es.0 {
        match &ie.value {
            PagingProtocolIEs_EntryValue::Id_UEIdentityIndexValue(v) => {
                index = Some(bits_value(&v.0) as u16);
            }
            PagingProtocolIEs_EntryValue::Id_UEPagingID(UEPagingID::S_TMSI(t)) => {
                identity = Some(PagingIdentity::STmsi {
                    mmec: t.mmec.0.first().copied().unwrap_or(0),
                    m_tmsi: t
                        .m_tmsi
                        .0
                        .iter()
                        .fold(0u32, |acc, b| (acc << 8) | u32::from(*b)),
                });
            }
            PagingProtocolIEs_EntryValue::Id_UEPagingID(UEPagingID::IMSI(i)) => {
                identity = Some(PagingIdentity::Imsi(imsi_digits(&i.0)));
            }
            PagingProtocolIEs_EntryValue::Id_CNDomain(d) => {
                cs_domain = Some(d.0 == CNDomain::CS);
            }
            PagingProtocolIEs_EntryValue::Id_TAIList(list) => {
                tais = Some(
                    list.0
                        .iter()
                        .filter_map(|e| {
                            let TAIList_EntryValue::Id_TAIItem(item) = &e.value;
                            let plmn = <[u8; 3]>::try_from(item.tai.plm_nidentity.0.as_slice())
                                .ok()
                                .map(Plmn::decode)?;
                            let tac = <[u8; 2]>::try_from(item.tai.tac.0.as_slice())
                                .ok()
                                .map(u16::from_be_bytes)?;
                            Some((plmn, tac))
                        })
                        .collect(),
                );
            }
            _ => {}
        }
    }
    // §9.1.6: all four are mandatory. Paging is a class 2 procedure whose
    // receiver has nothing to answer; a PAGING missing one cannot be acted on.
    match (index, identity, cs_domain, tais) {
        (Some(identity_index), Some(identity), Some(cs_domain), Some(tais)) => {
            S1Inbound::Paging(PagingRequest {
                identity_index,
                identity,
                cs_domain,
                tais,
            })
        }
        _ => S1Inbound::NotHandled("PAGING without a mandatory IE".into()),
    }
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

// ---------------------------------------------------------------------------
// Initial Context Setup (§8.3.1)
// ---------------------------------------------------------------------------

/// One E-RAB of INITIAL CONTEXT SETUP REQUEST (§9.1.4.1, E-RAB To Be Setup
/// Item).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErabToSetup {
    /// E-RAB ID (= EPS bearer identity).
    pub e_rab_id: u8,
    /// QCI.
    pub qci: u8,
    /// Whether the GBR QoS Information IE is present (§9.2.1.15).
    pub gbr_info: bool,
    /// The SGW's S1-U transport layer address, as octets (4 IPv4, 16 IPv6).
    pub sgw_address: Vec<u8>,
    /// The SGW's S1-U GTP-TEID.
    pub sgw_teid: u32,
    /// NAS-PDU to deliver to the UE (the ATTACH ACCEPT, during attach).
    pub nas: Option<Vec<u8>>,
}

/// INITIAL CONTEXT SETUP REQUEST, the parts the eNB acts on.
#[derive(Clone, PartialEq, Eq)]
pub struct InitialContextSetup {
    /// MME UE S1AP ID.
    pub mme_ue_s1ap_id: u32,
    /// eNB UE S1AP ID.
    pub enb_ue_s1ap_id: u32,
    /// UE AMBR downlink, bit/s.
    pub ue_ambr_dl: u64,
    /// UE AMBR uplink, bit/s.
    pub ue_ambr_ul: u64,
    /// The E-RABs.
    pub erabs: Vec<ErabToSetup>,
    /// Encryption Algorithms (§9.2.1.40): bit 0 (most significant) = 128-EEA1,
    /// bit 1 = 128-EEA2, bit 2 = 128-EEA3; EEA0 is always supported.
    pub eea_bitmap: u16,
    /// Integrity Protection Algorithms, same layout for EIA1..EIA3.
    pub eia_bitmap: u16,
    /// Security Key: `K_eNB` (§9.2.1.41).
    pub security_key: [u8; 32],
    /// Notify-criticality IEs not comprehended, reported in the response's
    /// Criticality Diagnostics (§10.3.4.2).
    pub notify: Vec<DiagnosedIe>,
}

impl std::fmt::Debug for InitialContextSetup {
    /// Never prints the Security Key.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InitialContextSetup")
            .field("mme_ue_s1ap_id", &self.mme_ue_s1ap_id)
            .field("enb_ue_s1ap_id", &self.enb_ue_s1ap_id)
            .field("erabs", &self.erabs)
            .field("eea_bitmap", &format_args!("{:#06x}", self.eea_bitmap))
            .field("eia_bitmap", &format_args!("{:#06x}", self.eia_bitmap))
            .field("security_key", &"<redacted>")
            .finish()
    }
}

impl InitialContextSetup {
    /// Whether 128-EEA`alg` is in the UE's capabilities (EEA0 always is).
    pub fn ue_supports_eea(&self, alg: u8) -> bool {
        alg == 0 || (1..=3).contains(&alg) && self.eea_bitmap & (0x8000 >> (alg - 1)) != 0
    }

    /// Whether 128-EIA`alg` is in the UE's capabilities (EIA0 is not signalled).
    pub fn ue_supports_eia(&self, alg: u8) -> bool {
        (1..=3).contains(&alg) && self.eia_bitmap & (0x8000 >> (alg - 1)) != 0
    }
}

fn bits_value(b: &bitvec::slice::BitSlice<u8, bitvec::order::Msb0>) -> u32 {
    b.iter()
        .take(16)
        .fold(0u32, |acc, x| (acc << 1) | u32::from(*x))
        << 16u32.saturating_sub(b.len().min(16) as u32)
}

fn bit_octets(b: &bitvec::slice::BitSlice<u8, bitvec::order::Msb0>) -> Vec<u8> {
    b.chunks(8)
        .map(|c| c.iter().fold(0u8, |acc, x| (acc << 1) | u8::from(*x)) << (8 - c.len()))
        .collect()
}

/// The procedure tuple of Initial Context Setup for Criticality Diagnostics.
const ICS_PROCEDURE: (u8, u8, u8) = (
    ID_INITIAL_CONTEXT_SETUP,
    TriggeringMessage::INITIATING_MESSAGE,
    Criticality::REJECT,
);

fn handle_initial_context_setup(
    req: &InitialContextSetupRequest,
    unknown: &[UnknownIe],
) -> S1Inbound {
    let mut mme = None;
    let mut enb = None;
    let mut ambr = None;
    let mut erabs = None;
    let mut caps = None;
    let mut key = None;
    for ie in &req.protocol_i_es.0 {
        match &ie.value {
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => {
                mme = Some(v.0)
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => {
                enb = Some(v.0)
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_uEaggregateMaximumBitrate(a) => {
                ambr = Some((
                    a.u_eaggregate_maximum_bit_rate_dl.0,
                    a.u_eaggregate_maximum_bit_rate_ul.0,
                ));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_E_RABToBeSetupListCtxtSUReq(l) => {
                erabs = Some(
                    l.0.iter()
                        .map(|e| {
                            let E_RABToBeSetupListCtxtSUReq_EntryValue::Id_E_RABToBeSetupItemCtxtSUReq(i) =
                                &e.value;
                            ErabToSetup {
                                e_rab_id: i.e_rab_id.0,
                                qci: i.e_ra_blevel_qo_s_parameters.qci.0,
                                gbr_info: i.e_ra_blevel_qo_s_parameters.gbr_qos_information.is_some(),
                                sgw_address: bit_octets(&i.transport_layer_address.0),
                                sgw_teid: u32::from_be_bytes(
                                    i.gtp_teid.0.as_slice().try_into().unwrap_or([0; 4]),
                                ),
                                nas: i.nas_pdu.as_ref().map(|n| n.0.clone()),
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_UESecurityCapabilities(c) => {
                caps = Some((
                    bits_value(&c.encryption_algorithms.0) as u16,
                    bits_value(&c.integrity_protection_algorithms.0) as u16,
                ));
            }
            InitialContextSetupRequestProtocolIEs_EntryValue::Id_SecurityKey(k) => {
                let o = bit_octets(&k.0);
                key = <[u8; 32]>::try_from(o.as_slice()).ok();
            }
            _ => {}
        }
    }
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
    // §10.3.5: every mandatory IE of this message is criticality reject.
    let mut missing = Vec::new();
    for (present, id) in [
        (mme.is_some(), ID_MME_UE_S1AP_ID),
        (enb.is_some(), ID_E_NB_UE_S1AP_ID),
        (ambr.is_some(), ID_U_EAGGREGATE_MAXIMUM_BITRATE),
        (erabs.is_some(), ID_E_RAB_TO_BE_SETUP_LIST_CTXT_SU_REQ),
        (caps.is_some(), ID_UE_SECURITY_CAPABILITIES),
        (key.is_some(), ID_SECURITY_KEY),
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
        // Class 1 (§10.3.4.2, §10.3.5): terminate and answer with the
        // unsuccessful outcome -- which needs both UE S1AP IDs; without them
        // only an Error Indication can say anything.
        return match (mme, enb) {
            (Some(m), Some(e)) => encode_or_log(&build_initial_context_setup_failure(
                m,
                e,
                protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
                &diag,
            ))
            .map_or_else(
                || S1Inbound::NotHandled("INITIAL CONTEXT SETUP rejected".into()),
                |pdu| S1Inbound::InitialContextSetupFailure {
                    enb_ue_s1ap_id: e,
                    pdu,
                },
            ),
            _ => encode_or_log(&build_ue_error_indication(
                mme,
                enb,
                protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
                Some(ICS_PROCEDURE),
                &diag,
            ))
            .map_or_else(
                || S1Inbound::NotHandled("INITIAL CONTEXT SETUP rejected".into()),
                S1Inbound::ErrorIndication,
            ),
        };
    }
    let notify = unknown
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
    let (Some(mme), Some(enb), Some((dl, ul)), Some(erabs), Some((eea, eia)), Some(key)) =
        (mme, enb, ambr, erabs, caps, key)
    else {
        unreachable!("checked above");
    };
    S1Inbound::InitialContextSetup(Box::new(InitialContextSetup {
        mme_ue_s1ap_id: mme,
        enb_ue_s1ap_id: enb,
        ue_ambr_dl: dl,
        ue_ambr_ul: ul,
        erabs,
        eea_bitmap: eea,
        eia_bitmap: eia,
        security_key: key,
        notify,
    }))
}

fn criticality_diagnostics(procedure: (u8, u8, u8), ies: &[DiagnosedIe]) -> CriticalityDiagnostics {
    CriticalityDiagnostics {
        procedure_code: Some(ProcedureCode(procedure.0)),
        triggering_message: Some(TriggeringMessage(procedure.1)),
        procedure_criticality: Some(Criticality(procedure.2)),
        i_es_criticality_diagnostics: (!ies.is_empty()).then(|| {
            CriticalityDiagnostics_IE_List(
                ies.iter()
                    .map(|ie| CriticalityDiagnostics_IE_Item {
                        ie_criticality: Criticality(ie.criticality),
                        ie_id: ProtocolIE_ID(ie.id),
                        type_of_error: TypeOfError(ie.type_of_error),
                        ie_extensions: None,
                    })
                    .collect(),
            )
        }),
        ie_extensions: None,
    }
}

/// One established E-RAB for INITIAL CONTEXT SETUP RESPONSE: its ID and the
/// eNB's S1-U endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ErabSetup {
    /// E-RAB ID.
    pub e_rab_id: u8,
    /// eNB S1-U address.
    pub address: std::net::IpAddr,
    /// eNB S1-U GTP-TEID.
    pub teid: u32,
}

fn transport_layer_address(ip: std::net::IpAddr) -> TransportLayerAddress {
    let octets = match ip {
        std::net::IpAddr::V4(a) => a.octets().to_vec(),
        std::net::IpAddr::V6(a) => a.octets().to_vec(),
    };
    TransportLayerAddress(
        octets
            .iter()
            .flat_map(|o| (0..8).rev().map(move |i| (o >> i) & 1 == 1))
            .collect(),
    )
}

/// Build INITIAL CONTEXT SETUP RESPONSE (§9.1.4.2), IEs in message-table
/// order with their `InitialContextSetupResponseIEs` criticalities (all
/// ignore), with Criticality Diagnostics when notify IEs were not comprehended.
pub fn build_initial_context_setup_response(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    erabs: &[ErabSetup],
    notify: &[DiagnosedIe],
) -> S1AP_PDU {
    let mut ies = vec![
        InitialContextSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(
                MME_UE_S1AP_ID(mme_ue_s1ap_id),
            ),
        },
        InitialContextSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(
                ENB_UE_S1AP_ID(enb_ue_s1ap_id),
            ),
        },
        InitialContextSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_SETUP_LIST_CTXT_SU_RES),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_E_RABSetupListCtxtSURes(
                E_RABSetupListCtxtSURes(
                    erabs
                        .iter()
                        .map(|e| E_RABSetupListCtxtSURes_Entry {
                            id: ProtocolIE_ID(ID_E_RAB_SETUP_ITEM_CTXT_SU_RES),
                            criticality: Criticality(Criticality::IGNORE),
                            value: E_RABSetupListCtxtSURes_EntryValue::Id_E_RABSetupItemCtxtSURes(
                                E_RABSetupItemCtxtSURes {
                                    e_rab_id: E_RAB_ID(e.e_rab_id),
                                    transport_layer_address: transport_layer_address(e.address),
                                    gtp_teid: GTP_TEID(e.teid.to_be_bytes().to_vec()),
                                    ie_extensions: None,
                                },
                            ),
                        })
                        .collect(),
                ),
            ),
        },
    ];
    if !notify.is_empty() {
        ies.push(InitialContextSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupResponseProtocolIEs_EntryValue::Id_CriticalityDiagnostics(
                criticality_diagnostics(ICS_PROCEDURE, notify),
            ),
        });
    }
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_INITIAL_CONTEXT_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_InitialContextSetup(InitialContextSetupResponse {
            protocol_i_es: InitialContextSetupResponseProtocolIEs(ies),
        }),
    })
}

/// Build INITIAL CONTEXT SETUP FAILURE (§9.1.4.3), with Criticality
/// Diagnostics when `diag` names the IEs that caused it.
pub fn build_initial_context_setup_failure(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    cause: Cause,
    diag: &[DiagnosedIe],
) -> S1AP_PDU {
    let mut ies = vec![
        InitialContextSetupFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(
                MME_UE_S1AP_ID(mme_ue_s1ap_id),
            ),
        },
        InitialContextSetupFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(
                ENB_UE_S1AP_ID(enb_ue_s1ap_id),
            ),
        },
        InitialContextSetupFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CAUSE),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_Cause(cause),
        },
    ];
    if !diag.is_empty() {
        ies.push(InitialContextSetupFailureProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
            criticality: Criticality(Criticality::IGNORE),
            value: InitialContextSetupFailureProtocolIEs_EntryValue::Id_CriticalityDiagnostics(
                criticality_diagnostics(ICS_PROCEDURE, diag),
            ),
        });
    }
    S1AP_PDU::UnsuccessfulOutcome(UnsuccessfulOutcome {
        procedure_code: ProcedureCode(ID_INITIAL_CONTEXT_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: UnsuccessfulOutcomeValue::Id_InitialContextSetup(InitialContextSetupFailure {
            protocol_i_es: InitialContextSetupFailureProtocolIEs(ies),
        }),
    })
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
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_InitialContextSetup(req),
            ..
        }) => handle_initial_context_setup(req, &unknown),
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_Paging(paging),
            ..
        }) => handle_paging(paging),
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_E_RABSetup(req),
            ..
        }) => handle_erab_setup(req, &unknown),
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_E_RABModify(req),
            ..
        }) => handle_erab_modify(req, &unknown),
        S1AP_PDU::InitiatingMessage(InitiatingMessage {
            value: InitiatingMessageValue::Id_E_RABRelease(cmd),
            ..
        }) => handle_erab_release(cmd, &unknown),
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

// ===========================================================================
// E-RAB management (TS 36.413 §8.2, nextgsim #283)
// ===========================================================================

/// E-RAB SETUP REQUEST (§9.1.3.1), the parts the eNB acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErabSetupRequest {
    /// MME UE S1AP ID.
    pub mme_ue_s1ap_id: u32,
    /// eNB UE S1AP ID.
    pub enb_ue_s1ap_id: u32,
    /// The E-RABs to set up, each with the NAS-PDU for the UE.
    pub erabs: Vec<ErabToSetup>,
}

/// One item of E-RAB MODIFY REQUEST (§9.1.3.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErabToModify {
    /// E-RAB ID.
    pub e_rab_id: u8,
    /// The new QCI.
    pub qci: u8,
    /// Whether the GBR QoS Information IE is present (§9.2.1.15).
    pub gbr_info: bool,
    /// NAS-PDU for the UE (the MODIFY EPS BEARER CONTEXT REQUEST).
    pub nas: Vec<u8>,
}

/// E-RAB MODIFY REQUEST (§9.1.3.3), the parts the eNB acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErabModifyRequest {
    /// MME UE S1AP ID.
    pub mme_ue_s1ap_id: u32,
    /// eNB UE S1AP ID.
    pub enb_ue_s1ap_id: u32,
    /// The E-RABs to modify.
    pub erabs: Vec<ErabToModify>,
}

/// E-RAB RELEASE COMMAND (§9.1.3.5), the parts the eNB acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ErabReleaseCommand {
    /// MME UE S1AP ID.
    pub mme_ue_s1ap_id: u32,
    /// eNB UE S1AP ID.
    pub enb_ue_s1ap_id: u32,
    /// The E-RAB IDs to release.
    pub erabs: Vec<u8>,
    /// NAS-PDU for the UE (the DEACTIVATE EPS BEARER CONTEXT REQUEST).
    pub nas: Option<Vec<u8>>,
}

/// The Error Indication for an E-RAB procedure missing a mandatory IE, or with
/// a reject-criticality IE not comprehended (§10.3.4.2, §10.3.5): E-RAB Setup,
/// Modify and Release have no unsuccessful outcome to carry it.
fn erab_ie_errors(
    unknown: &[UnknownIe],
    mandatory: &[(bool, u16)],
    mme: Option<u32>,
    enb: Option<u32>,
    procedure_code: u8,
    name: &str,
) -> Option<S1Inbound> {
    let mut diag: Vec<DiagnosedIe> = unknown
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
    for (present, id) in mandatory {
        if !present {
            diag.push(DiagnosedIe {
                id: *id,
                criticality: Criticality::REJECT,
                type_of_error: TypeOfError::MISSING,
            });
        }
    }
    if diag.is_empty() {
        return None;
    }
    Some(
        encode_or_log(&build_ue_error_indication(
            mme,
            enb,
            protocol(CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT),
            Some((
                procedure_code,
                TriggeringMessage::INITIATING_MESSAGE,
                Criticality::REJECT,
            )),
            &diag,
        ))
        .map_or_else(
            || S1Inbound::NotHandled(format!("{name} rejected")),
            S1Inbound::ErrorIndication,
        ),
    )
}

fn handle_erab_setup(req: &E_RABSetupRequest, unknown: &[UnknownIe]) -> S1Inbound {
    let (mut mme, mut enb, mut erabs) = (None, None, None);
    for ie in &req.protocol_i_es.0 {
        match &ie.value {
            E_RABSetupRequestProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => mme = Some(v.0),
            E_RABSetupRequestProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb = Some(v.0),
            E_RABSetupRequestProtocolIEs_EntryValue::Id_E_RABToBeSetupListBearerSUReq(l) => {
                erabs = Some(
                    l.0.iter()
                        .map(|e| {
                            let E_RABToBeSetupListBearerSUReq_EntryValue::Id_E_RABToBeSetupItemBearerSUReq(i) =
                                &e.value;
                            ErabToSetup {
                                e_rab_id: i.e_rab_id.0,
                                qci: i.e_ra_blevel_qo_s_parameters.qci.0,
                                gbr_info: i.e_ra_blevel_qo_s_parameters.gbr_qos_information.is_some(),
                                sgw_address: bit_octets(&i.transport_layer_address.0),
                                sgw_teid: u32::from_be_bytes(
                                    i.gtp_teid.0.as_slice().try_into().unwrap_or([0; 4]),
                                ),
                                nas: Some(i.nas_pdu.0.clone()),
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
            _ => {}
        }
    }
    if let Some(err) = erab_ie_errors(
        unknown,
        &[
            (mme.is_some(), ID_MME_UE_S1AP_ID),
            (enb.is_some(), ID_E_NB_UE_S1AP_ID),
            (erabs.is_some(), ID_E_RAB_TO_BE_SETUP_LIST_BEARER_SU_REQ),
        ],
        mme,
        enb,
        ID_E_RAB_SETUP,
        "E-RAB SETUP REQUEST",
    ) {
        return err;
    }
    let (Some(mme), Some(enb), Some(erabs)) = (mme, enb, erabs) else {
        unreachable!("checked above");
    };
    S1Inbound::ErabSetup(Box::new(ErabSetupRequest {
        mme_ue_s1ap_id: mme,
        enb_ue_s1ap_id: enb,
        erabs,
    }))
}

fn handle_erab_modify(req: &E_RABModifyRequest, unknown: &[UnknownIe]) -> S1Inbound {
    let (mut mme, mut enb, mut erabs) = (None, None, None);
    for ie in &req.protocol_i_es.0 {
        match &ie.value {
            E_RABModifyRequestProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => mme = Some(v.0),
            E_RABModifyRequestProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb = Some(v.0),
            E_RABModifyRequestProtocolIEs_EntryValue::Id_E_RABToBeModifiedListBearerModReq(l) => {
                erabs = Some(
                    l.0.iter()
                        .map(|e| {
                            let E_RABToBeModifiedListBearerModReq_EntryValue::Id_E_RABToBeModifiedItemBearerModReq(i) =
                                &e.value;
                            ErabToModify {
                                e_rab_id: i.e_rab_id.0,
                                qci: i.e_rab_level_qo_s_parameters.qci.0,
                                gbr_info: i.e_rab_level_qo_s_parameters.gbr_qos_information.is_some(),
                                nas: i.nas_pdu.0.clone(),
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }
            _ => {}
        }
    }
    if let Some(err) = erab_ie_errors(
        unknown,
        &[
            (mme.is_some(), ID_MME_UE_S1AP_ID),
            (enb.is_some(), ID_E_NB_UE_S1AP_ID),
            (erabs.is_some(), ID_E_RAB_TO_BE_MODIFIED_LIST_BEARER_MOD_REQ),
        ],
        mme,
        enb,
        ID_E_RAB_MODIFY,
        "E-RAB MODIFY REQUEST",
    ) {
        return err;
    }
    let (Some(mme), Some(enb), Some(erabs)) = (mme, enb, erabs) else {
        unreachable!("checked above");
    };
    S1Inbound::ErabModify(Box::new(ErabModifyRequest {
        mme_ue_s1ap_id: mme,
        enb_ue_s1ap_id: enb,
        erabs,
    }))
}

fn handle_erab_release(cmd: &E_RABReleaseCommand, unknown: &[UnknownIe]) -> S1Inbound {
    let (mut mme, mut enb, mut erabs, mut nas) = (None, None, None, None);
    for ie in &cmd.protocol_i_es.0 {
        match &ie.value {
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => mme = Some(v.0),
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb = Some(v.0),
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_E_RABToBeReleasedList(l) => {
                erabs = Some(
                    l.0.iter()
                        .map(|e| {
                            let E_RABList_EntryValue::Id_E_RABItem(i) = &e.value;
                            i.e_rab_id.0
                        })
                        .collect::<Vec<_>>(),
                );
            }
            E_RABReleaseCommandProtocolIEs_EntryValue::Id_NAS_PDU(n) => nas = Some(n.0.clone()),
            _ => {}
        }
    }
    if let Some(err) = erab_ie_errors(
        unknown,
        &[
            (mme.is_some(), ID_MME_UE_S1AP_ID),
            (enb.is_some(), ID_E_NB_UE_S1AP_ID),
            (erabs.is_some(), ID_E_RAB_TO_BE_RELEASED_LIST),
        ],
        mme,
        enb,
        ID_E_RAB_RELEASE,
        "E-RAB RELEASE COMMAND",
    ) {
        return err;
    }
    let (Some(mme), Some(enb), Some(erabs)) = (mme, enb, erabs) else {
        unreachable!("checked above");
    };
    S1Inbound::ErabRelease(Box::new(ErabReleaseCommand {
        mme_ue_s1ap_id: mme,
        enb_ue_s1ap_id: enb,
        erabs,
        nas,
    }))
}

/// E-RAB List (§9.2.1.36) of failed E-RABs and their causes, or `None` when
/// empty (the list has at least one item).
fn failed_list(failed: &[(u8, Cause)]) -> Option<E_RABList> {
    (!failed.is_empty()).then(|| {
        E_RABList(
            failed
                .iter()
                .map(|(id, cause)| E_RABList_Entry {
                    id: ProtocolIE_ID(ID_E_RAB_ITEM),
                    criticality: Criticality(Criticality::IGNORE),
                    value: E_RABList_EntryValue::Id_E_RABItem(E_RABItem {
                        e_rab_id: E_RAB_ID(*id),
                        cause: cause.clone(),
                        ie_extensions: None,
                    }),
                })
                .collect(),
        )
    })
}

/// Build E-RAB SETUP RESPONSE (§9.1.3.2): IEs in message-table order, all
/// criticality ignore; each list only when non-empty.
pub fn build_erab_setup_response(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    setup: &[ErabSetup],
    failed: &[(u8, Cause)],
) -> S1AP_PDU {
    let ignore = Criticality(Criticality::IGNORE);
    let mut ies = vec![
        E_RABSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABSetupResponseProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                mme_ue_s1ap_id,
            )),
        },
        E_RABSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABSetupResponseProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                enb_ue_s1ap_id,
            )),
        },
    ];
    if !setup.is_empty() {
        ies.push(E_RABSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_SETUP_LIST_BEARER_SU_RES),
            criticality: ignore.clone(),
            value: E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABSetupListBearerSURes(
                E_RABSetupListBearerSURes(
                    setup
                        .iter()
                        .map(|e| E_RABSetupListBearerSURes_Entry {
                            id: ProtocolIE_ID(ID_E_RAB_SETUP_ITEM_BEARER_SU_RES),
                            criticality: ignore.clone(),
                            value:
                                E_RABSetupListBearerSURes_EntryValue::Id_E_RABSetupItemBearerSURes(
                                    E_RABSetupItemBearerSURes {
                                        e_rab_id: E_RAB_ID(e.e_rab_id),
                                        transport_layer_address: transport_layer_address(e.address),
                                        gtp_teid: GTP_TEID(e.teid.to_be_bytes().to_vec()),
                                        ie_extensions: None,
                                    },
                                ),
                        })
                        .collect(),
                ),
            ),
        });
    }
    if let Some(l) = failed_list(failed) {
        ies.push(E_RABSetupResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_FAILED_TO_SETUP_LIST_BEARER_SU_RES),
            criticality: ignore,
            value: E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABFailedToSetupListBearerSURes(
                l,
            ),
        });
    }
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_E_RAB_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_E_RABSetup(E_RABSetupResponse {
            protocol_i_es: E_RABSetupResponseProtocolIEs(ies),
        }),
    })
}

/// Build E-RAB MODIFY RESPONSE (§9.1.3.4).
pub fn build_erab_modify_response(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    modified: &[u8],
    failed: &[(u8, Cause)],
) -> S1AP_PDU {
    let ignore = Criticality(Criticality::IGNORE);
    let mut ies = vec![
        E_RABModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABModifyResponseProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                mme_ue_s1ap_id,
            )),
        },
        E_RABModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABModifyResponseProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                enb_ue_s1ap_id,
            )),
        },
    ];
    if !modified.is_empty() {
        ies.push(E_RABModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_MODIFY_LIST_BEARER_MOD_RES),
            criticality: ignore.clone(),
            value: E_RABModifyResponseProtocolIEs_EntryValue::Id_E_RABModifyListBearerModRes(
                E_RABModifyListBearerModRes(
                    modified
                        .iter()
                        .map(|id| E_RABModifyListBearerModRes_Entry {
                            id: ProtocolIE_ID(ID_E_RAB_MODIFY_ITEM_BEARER_MOD_RES),
                            criticality: ignore.clone(),
                            value: E_RABModifyListBearerModRes_EntryValue::Id_E_RABModifyItemBearerModRes(
                                E_RABModifyItemBearerModRes {
                                    e_rab_id: E_RAB_ID(*id),
                                    ie_extensions: None,
                                },
                            ),
                        })
                        .collect(),
                ),
            ),
        });
    }
    if let Some(l) = failed_list(failed) {
        ies.push(E_RABModifyResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_FAILED_TO_MODIFY_LIST),
            criticality: ignore,
            value: E_RABModifyResponseProtocolIEs_EntryValue::Id_E_RABFailedToModifyList(l),
        });
    }
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_E_RAB_MODIFY),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_E_RABModify(E_RABModifyResponse {
            protocol_i_es: E_RABModifyResponseProtocolIEs(ies),
        }),
    })
}

/// Build E-RAB RELEASE RESPONSE (§9.1.3.6).
pub fn build_erab_release_response(
    mme_ue_s1ap_id: u32,
    enb_ue_s1ap_id: u32,
    released: &[u8],
    failed: &[(u8, Cause)],
) -> S1AP_PDU {
    let ignore = Criticality(Criticality::IGNORE);
    let mut ies = vec![
        E_RABReleaseResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_MME_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABReleaseResponseProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                mme_ue_s1ap_id,
            )),
        },
        E_RABReleaseResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_NB_UE_S1AP_ID),
            criticality: ignore.clone(),
            value: E_RABReleaseResponseProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                enb_ue_s1ap_id,
            )),
        },
    ];
    if !released.is_empty() {
        ies.push(E_RABReleaseResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_RELEASE_LIST_BEARER_REL_COMP),
            criticality: ignore.clone(),
            value: E_RABReleaseResponseProtocolIEs_EntryValue::Id_E_RABReleaseListBearerRelComp(
                E_RABReleaseListBearerRelComp(
                    released
                        .iter()
                        .map(|id| E_RABReleaseListBearerRelComp_Entry {
                            id: ProtocolIE_ID(ID_E_RAB_RELEASE_ITEM_BEARER_REL_COMP),
                            criticality: ignore.clone(),
                            value: E_RABReleaseListBearerRelComp_EntryValue::Id_E_RABReleaseItemBearerRelComp(
                                E_RABReleaseItemBearerRelComp {
                                    e_rab_id: E_RAB_ID(*id),
                                    ie_extensions: None,
                                },
                            ),
                        })
                        .collect(),
                ),
            ),
        });
    }
    if let Some(l) = failed_list(failed) {
        ies.push(E_RABReleaseResponseProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_RAB_FAILED_TO_RELEASE_LIST),
            criticality: ignore,
            value: E_RABReleaseResponseProtocolIEs_EntryValue::Id_E_RABFailedToReleaseList(l),
        });
    }
    S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        procedure_code: ProcedureCode(ID_E_RAB_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: SuccessfulOutcomeValue::Id_E_RABRelease(E_RABReleaseResponse {
            protocol_i_es: E_RABReleaseResponseProtocolIEs(ies),
        }),
    })
}

#[cfg(test)]
mod tests;
