//! S1AP procedures the eNB drives: S1 Setup (TS 36.413 §8.7.3) and the Error
//! Indication it owes the MME under the error-handling rules of §10.
//!
//! Everything here is pure -- PDUs in, PDUs and decisions out -- so the whole
//! procedure logic is testable without a transport. [`crate::s1_task`] owns the
//! SCTP association and the timing.

use std::time::Duration;

use bitvec::prelude::*;
use nextgsim_common::Plmn;
use nextgsim_s1ap::codec::{self, *};

use crate::config::{EnbConfig, EnbIdType};

/// S1 Setup's procedure criticality (`s1Setup ... CRITICALITY reject`).
const S1_SETUP_CRITICALITY: u8 = Criticality::REJECT;
/// Error Indication's procedure criticality (`errorIndication ... CRITICALITY ignore`).
const ERROR_INDICATION_CRITICALITY: u8 = Criticality::IGNORE;
/// Highest procedure code TS 36.413 V19.2.0 defines (`id-S1Removal`). Codes
/// 0..=67 are contiguous in §9.3.6, so anything above is one this eNB does not
/// comprehend (§10.3.4.1).
pub const MAX_KNOWN_PROCEDURE_CODE: u8 = ID_S1_REMOVAL;

fn plmn_identity(p: &Plmn) -> PLMNidentity {
    TBCD_STRING(p.encode().to_vec())
}

fn decode_plmn(p: &PLMNidentity) -> Option<Plmn> {
    <[u8; 3]>::try_from(p.0.as_slice()).ok().map(Plmn::decode)
}

/// The leftmost-bit-first BIT STRING of `value` in `bits` bits.
fn bit_string(value: u32, bits: u32) -> BitVec<u8, Msb0> {
    (0..bits).rev().map(|i| (value >> i) & 1 == 1).collect()
}

fn enb_id(c: &EnbConfig) -> ENB_ID {
    let bits = bit_string(c.enb_id, c.enb_id_type.bits());
    match c.enb_id_type {
        EnbIdType::Macro => ENB_ID::MacroENB_ID(ENB_ID_macroENB_ID(bits)),
        EnbIdType::Home => ENB_ID::HomeENB_ID(ENB_ID_homeENB_ID(bits)),
        EnbIdType::ShortMacro => ENB_ID::Short_macroENB_ID(ENB_ID_short_macroENB_ID(bits)),
        EnbIdType::LongMacro => ENB_ID::Long_macroENB_ID(ENB_ID_long_macroENB_ID(bits)),
    }
}

/// Build S1 SETUP REQUEST (TS 36.413 §9.1.8.4) from a validated config.
///
/// IEs are emitted in the message table's order, each with the criticality
/// its `S1SetupRequestIEs` object assigns.
pub fn build_s1_setup_request(c: &EnbConfig) -> S1AP_PDU {
    let mut ies = vec![S1SetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_GLOBAL_ENB_ID),
        criticality: Criticality(Criticality::REJECT),
        value: S1SetupRequestProtocolIEs_EntryValue::Id_Global_ENB_ID(Global_ENB_ID {
            plm_nidentity: plmn_identity(&c.plmn),
            enb_id: enb_id(c),
            ie_extensions: None,
        }),
    }];
    if let Some(name) = &c.enb_name {
        ies.push(S1SetupRequestProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_E_N_BNAME),
            criticality: Criticality(Criticality::IGNORE),
            value: S1SetupRequestProtocolIEs_EntryValue::Id_eNBname(ENBname(name.clone())),
        });
    }
    let tas = c
        .effective_supported_tas()
        .iter()
        .map(|ta| SupportedTAs_Item {
            tac: TAC(ta.tac.to_be_bytes().to_vec()),
            broadcast_plm_ns: BPLMNs(ta.broadcast_plmns.iter().map(plmn_identity).collect()),
            ie_extensions: None,
        })
        .collect();
    ies.push(S1SetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_SUPPORTED_T_AS),
        criticality: Criticality(Criticality::REJECT),
        value: S1SetupRequestProtocolIEs_EntryValue::Id_SupportedTAs(SupportedTAs(tas)),
    });
    ies.push(S1SetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_DEFAULT_PAGING_DRX),
        criticality: Criticality(Criticality::IGNORE),
        value: S1SetupRequestProtocolIEs_EntryValue::Id_DefaultPagingDRX(PagingDRX(
            c.paging_drx.index(),
        )),
    });
    S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_S1_SETUP),
        criticality: Criticality(S1_SETUP_CRITICALITY),
        value: InitiatingMessageValue::Id_S1Setup(S1SetupRequest {
            protocol_i_es: S1SetupRequestProtocolIEs(ies),
        }),
    })
}

/// One Served GUMMEIs item (TS 36.413 §9.1.8.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ServedGummei {
    /// Served PLMNs.
    pub plmns: Vec<Plmn>,
    /// Served MME group IDs.
    pub group_ids: Vec<u16>,
    /// Served MME codes.
    pub mme_codes: Vec<u8>,
}

/// What the eNB stores from S1 SETUP RESPONSE for the life of the TNL
/// association (§8.7.3.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MmeInfo {
    /// Served GUMMEIs.
    pub served_gummeis: Vec<ServedGummei>,
    /// MME Name, if sent.
    pub mme_name: Option<String>,
    /// Relative MME Capacity. Mandatory, but criticality `ignore`, so a
    /// response without it is still accepted (§10.3.5).
    pub relative_capacity: Option<u8>,
}

/// The eNB's decision on what the MME sent while S1 Setup was outstanding.
#[derive(Debug, Clone, PartialEq)]
pub enum SetupOutcome {
    /// S1 SETUP RESPONSE was accepted; the S1 interface is operational.
    Accepted {
        /// Stored MME data.
        mme: MmeInfo,
        /// An Error Indication owed under §10.3.4.2/§10.3.5 ("Ignore IE and
        /// Notify Sender" on a response), already encoded.
        error_indication: Option<Vec<u8>>,
    },
    /// S1 SETUP FAILURE. §8.7.3.3: with Time To Wait, wait at least that long
    /// before re-initiating towards the same MME.
    Failed {
        /// Cause, when present (mandatory, criticality `ignore`).
        cause: Option<Cause>,
        /// Time To Wait, when present.
        time_to_wait: Option<Duration>,
        /// Error Indication owed for notify-criticality IEs, if any.
        error_indication: Option<Vec<u8>>,
    },
    /// The response is unusable: an IE marked `reject` was not comprehended or
    /// is missing. §10.3.4.2/§10.3.5 make the procedure unsuccessfully
    /// terminated with local error handling -- no message to the MME.
    UnsuccessfullyTerminated {
        /// Why, for the log.
        reason: String,
    },
    /// Not an S1 Setup outcome. Carries the Error Indication §10 requires, if
    /// any (transfer syntax error, or a procedure code not comprehended with
    /// criticality reject/notify).
    NotSetupOutcome {
        /// What arrived, for the log.
        description: String,
        /// Error Indication to send, if one is owed.
        error_indication: Option<Vec<u8>>,
    },
}

/// TimeToWait ENUMERATED → duration (TS 36.413 §9.2.1.61).
pub fn time_to_wait_duration(t: &TimeToWait) -> Option<Duration> {
    let secs = match t.0 {
        TimeToWait::V1S => 1,
        TimeToWait::V2S => 2,
        TimeToWait::V5S => 5,
        TimeToWait::V10S => 10,
        TimeToWait::V20S => 20,
        TimeToWait::V60S => 60,
        _ => return None,
    };
    Some(Duration::from_secs(secs))
}

/// An IE to report in Criticality Diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DiagnosedIe {
    /// IE id.
    pub id: u16,
    /// Its criticality (as received, or as specified for a missing IE).
    pub criticality: u8,
    /// `TypeOfError`.
    pub type_of_error: u8,
}

/// Build ERROR INDICATION (TS 36.413 §8.7.2, §9.1.8.2) carrying `cause` and,
/// when given, Criticality Diagnostics (§9.2.1.21).
pub fn build_error_indication(
    cause: Cause,
    procedure: Option<(u8, u8, u8)>,
    ies: &[DiagnosedIe],
) -> S1AP_PDU {
    let mut entries = vec![ErrorIndicationProtocolIEs_Entry {
        id: ProtocolIE_ID(ID_CAUSE),
        criticality: Criticality(Criticality::IGNORE),
        value: ErrorIndicationProtocolIEs_EntryValue::Id_Cause(cause),
    }];
    if procedure.is_some() || !ies.is_empty() {
        let list = (!ies.is_empty()).then(|| {
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
        });
        entries.push(ErrorIndicationProtocolIEs_Entry {
            id: ProtocolIE_ID(ID_CRITICALITY_DIAGNOSTICS),
            criticality: Criticality(Criticality::IGNORE),
            value: ErrorIndicationProtocolIEs_EntryValue::Id_CriticalityDiagnostics(
                CriticalityDiagnostics {
                    procedure_code: procedure.map(|p| ProcedureCode(p.0)),
                    triggering_message: procedure.map(|p| TriggeringMessage(p.1)),
                    procedure_criticality: procedure.map(|p| Criticality(p.2)),
                    i_es_criticality_diagnostics: list,
                    ie_extensions: None,
                },
            ),
        });
    }
    S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_ERROR_INDICATION),
        criticality: Criticality(ERROR_INDICATION_CRITICALITY),
        value: InitiatingMessageValue::Id_ErrorIndication(ErrorIndication {
            protocol_i_es: ErrorIndicationProtocolIEs(entries),
        }),
    })
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

/// The APER header of an S1AP-PDU: the root CHOICE index (0 initiating,
/// 1 successful, 2 unsuccessful), the procedure code and its criticality.
///
/// The root CHOICE is extensible with three root alternatives: one extension
/// bit and a 2-bit index, then the octet-aligned `ProcedureCode` (INTEGER
/// 0..255) and the 2-bit `Criticality`. Reading it does not depend on the
/// rest of the message decoding, which is what §10.3.4.1 needs when the
/// procedure code is the thing not comprehended.
pub fn peek_header(bytes: &[u8]) -> Option<(u8, u8, u8)> {
    let b0 = *bytes.first()?;
    if b0 & 0x80 != 0 {
        return None; // root-extension alternative: none defined
    }
    let kind = (b0 >> 5) & 0x03;
    if kind > 2 {
        return None;
    }
    let code = *bytes.get(1)?;
    let crit = bytes.get(2)? >> 6;
    Some((kind, code, crit))
}

fn protocol_cause(v: u8) -> Cause {
    Cause::Protocol(CauseProtocol(v))
}

/// Apply TS 36.413 §10.2 / §10.3.4.1 to a PDU that is not, or does not decode
/// as, an S1 Setup outcome.
fn not_setup_outcome(bytes: &[u8], description: String) -> SetupOutcome {
    let error_indication = match peek_header(bytes) {
        // §10.3.4.1: a procedure code not comprehended.
        Some((kind, code, crit)) if code > MAX_KNOWN_PROCEDURE_CODE => match crit {
            c if c == Criticality::IGNORE => None,
            c => {
                let cause = if c == Criticality::REJECT {
                    CauseProtocol::ABSTRACT_SYNTAX_ERROR_REJECT
                } else {
                    CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY
                };
                // "it shall include the Procedure Code IE, the Triggering
                // Message IE, and the Procedure Criticality IE".
                encode_or_log(&build_error_indication(
                    protocol_cause(cause),
                    Some((code, kind, c)),
                    &[],
                ))
            }
        },
        _ => None,
    };
    SetupOutcome::NotSetupOutcome {
        description,
        error_indication,
    }
}

fn notify_ies(unknown: &[UnknownIe]) -> Vec<DiagnosedIe> {
    unknown
        .iter()
        .filter(|u| u.criticality.as_ref().map(|c| c.0) == Some(Criticality::NOTIFY))
        .filter_map(|u| {
            Some(DiagnosedIe {
                id: u16::try_from(u.id).ok()?,
                criticality: Criticality::NOTIFY,
                type_of_error: TypeOfError::NOT_UNDERSTOOD,
            })
        })
        .collect()
}

fn notify_error_indication(triggering: u8, ies: &[DiagnosedIe]) -> Option<Vec<u8>> {
    if ies.is_empty() {
        return None;
    }
    encode_or_log(&build_error_indication(
        protocol_cause(CauseProtocol::ABSTRACT_SYNTAX_ERROR_IGNORE_AND_NOTIFY),
        Some((ID_S1_SETUP, triggering, S1_SETUP_CRITICALITY)),
        ies,
    ))
}

/// A `reject` IE that was not comprehended, if any. Absent criticality is
/// treated as `reject`: the conservative reading when the sender gave none.
fn first_reject(unknown: &[UnknownIe]) -> Option<i128> {
    unknown
        .iter()
        .find(|u| {
            u.criticality
                .as_ref()
                .is_none_or(|c| c.0 == Criticality::REJECT)
        })
        .map(|u| u.id)
}

fn served_gummeis(list: &ServedGUMMEIs) -> Result<Vec<ServedGummei>, String> {
    list.0
        .iter()
        .map(|item| {
            let plmns = item
                .served_plm_ns
                .0
                .iter()
                .map(|p| decode_plmn(p).ok_or_else(|| "malformed served PLMN".to_string()))
                .collect::<Result<Vec<_>, _>>()?;
            let group_ids = item
                .served_group_i_ds
                .0
                .iter()
                .map(|g| {
                    <[u8; 2]>::try_from(g.0.as_slice())
                        .map(u16::from_be_bytes)
                        .map_err(|_| "malformed MME group ID".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mme_codes = item
                .served_mme_cs
                .0
                .iter()
                .map(|c| {
                    c.0.first()
                        .copied()
                        .filter(|_| c.0.len() == 1)
                        .ok_or_else(|| "malformed MME code".to_string())
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(ServedGummei {
                plmns,
                group_ids,
                mme_codes,
            })
        })
        .collect()
}

fn handle_response(resp: &S1SetupResponse, unknown: &[UnknownIe]) -> SetupOutcome {
    if let Some(id) = first_reject(unknown) {
        return SetupOutcome::UnsuccessfullyTerminated {
            reason: format!("S1 SETUP RESPONSE carries IE {id} (reject) not comprehended"),
        };
    }
    let mut gummeis = None;
    let mut mme_name = None;
    let mut relative_capacity = None;
    for ie in &resp.protocol_i_es.0 {
        match &ie.value {
            S1SetupResponseProtocolIEs_EntryValue::Id_ServedGUMMEIs(g) => gummeis = Some(g),
            S1SetupResponseProtocolIEs_EntryValue::Id_MMEname(n) => mme_name = Some(n.0.clone()),
            S1SetupResponseProtocolIEs_EntryValue::Id_RelativeMMECapacity(c) => {
                relative_capacity = Some(c.0);
            }
            _ => {}
        }
    }
    // §10.3.5: Served GUMMEIs is mandatory with criticality reject.
    let Some(gummeis) = gummeis else {
        return SetupOutcome::UnsuccessfullyTerminated {
            reason: "S1 SETUP RESPONSE is missing Served GUMMEIs (reject)".into(),
        };
    };
    let served_gummeis = match served_gummeis(gummeis) {
        Ok(g) => g,
        Err(e) => {
            return SetupOutcome::UnsuccessfullyTerminated {
                reason: format!("S1 SETUP RESPONSE Served GUMMEIs: {e}"),
            }
        }
    };
    // Relative MME Capacity is mandatory with criticality `ignore`: §10.3.5
    // has the receiver ignore that it is missing and continue -- no report.
    SetupOutcome::Accepted {
        mme: MmeInfo {
            served_gummeis,
            mme_name,
            relative_capacity,
        },
        error_indication: notify_error_indication(
            TriggeringMessage::SUCCESSFUL_OUTCOME,
            &notify_ies(unknown),
        ),
    }
}

fn handle_failure(fail: &S1SetupFailure, unknown: &[UnknownIe]) -> SetupOutcome {
    if let Some(id) = first_reject(unknown) {
        return SetupOutcome::UnsuccessfullyTerminated {
            reason: format!("S1 SETUP FAILURE carries IE {id} (reject) not comprehended"),
        };
    }
    let mut cause = None;
    let mut time_to_wait = None;
    for ie in &fail.protocol_i_es.0 {
        match &ie.value {
            S1SetupFailureProtocolIEs_EntryValue::Id_Cause(c) => cause = Some(c.clone()),
            S1SetupFailureProtocolIEs_EntryValue::Id_TimeToWait(t) => {
                time_to_wait = time_to_wait_duration(t);
            }
            S1SetupFailureProtocolIEs_EntryValue::Id_CriticalityDiagnostics(_) => {}
        }
    }
    SetupOutcome::Failed {
        cause,
        time_to_wait,
        error_indication: notify_error_indication(
            TriggeringMessage::UNSUCCESSFULL_OUTCOME,
            &notify_ies(unknown),
        ),
    }
}

/// Classify what the MME sent while S1 Setup is outstanding.
pub fn handle_setup_reply(bytes: &[u8]) -> SetupOutcome {
    let (pdu, unknown) = match codec::decode_s1ap_pdu_with_unknown_ies(bytes) {
        Ok(v) => v,
        Err(e) => {
            // A known procedure that does not decode is a transfer syntax
            // error (§10.2); an unknown procedure code is §10.3.4.1's case.
            let unknown_code =
                peek_header(bytes).is_some_and(|(_, code, _)| code > MAX_KNOWN_PROCEDURE_CODE);
            if unknown_code {
                return not_setup_outcome(bytes, format!("unknown procedure ({e})"));
            }
            return SetupOutcome::NotSetupOutcome {
                description: format!("undecodable S1AP PDU: {e}"),
                error_indication: encode_or_log(&build_error_indication(
                    protocol_cause(CauseProtocol::TRANSFER_SYNTAX_ERROR),
                    None,
                    &[],
                )),
            };
        }
    };
    match &pdu {
        S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            value: SuccessfulOutcomeValue::Id_S1Setup(resp),
            ..
        }) => handle_response(resp, &unknown),
        S1AP_PDU::UnsuccessfulOutcome(UnsuccessfulOutcome {
            value: UnsuccessfulOutcomeValue::Id_S1Setup(fail),
            ..
        }) => handle_failure(fail, &unknown),
        other => SetupOutcome::NotSetupOutcome {
            description: format!("{} while S1 Setup outstanding", describe(other)),
            error_indication: None,
        },
    }
}

fn describe(pdu: &S1AP_PDU) -> String {
    match pdu {
        S1AP_PDU::InitiatingMessage(m) => {
            format!("initiating message, procedure {}", m.procedure_code.0)
        }
        S1AP_PDU::SuccessfulOutcome(m) => {
            format!("successful outcome, procedure {}", m.procedure_code.0)
        }
        S1AP_PDU::UnsuccessfulOutcome(m) => {
            format!("unsuccessful outcome, procedure {}", m.procedure_code.0)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests;
