//! NRPPa positioning procedures, TS 38.455
//!
//! NRPPa (NR Positioning Protocol A) runs between the LMF and an NG-RAN node.
//! It is NOT an NGAP procedure: NGAP merely tunnels it as an opaque
//! `NRPPa-PDU ::= OCTET STRING` (TS 38.413 §9.3.3.17) inside the four transport
//! procedures of TS 38.413 §8.14:
//!
//! | NGAP procedure code | Message | Direction |
//! |---|---|---|
//! | 8  `id-DownlinkUEAssociatedNRPPaTransport`     | `DownlinkUEAssociatedNRPPaTransport`     | AMF -> RAN |
//! | 50 `id-UplinkUEAssociatedNRPPaTransport`       | `UplinkUEAssociatedNRPPaTransport`       | RAN -> AMF |
//! | 5  `id-DownlinkNonUEAssociatedNRPPaTransport`  | `DownlinkNonUEAssociatedNRPPaTransport`  | AMF -> RAN |
//! | 47 `id-UplinkNonUEAssociatedNRPPaTransport`    | `UplinkNonUEAssociatedNRPPaTransport`    | RAN -> AMF |
//!
//! Note the codes are 5/8/47/50, NOT the 7/45/46 an adjacent reading of the
//! procedure-code list suggests: 7 is `DownlinkRANStatusTransfer`, 45 is
//! `UETNLABindingRelease` and 46 is `UplinkNASTransport`. Getting these wrong
//! routes an NRPPa transport into an unrelated handler, which is exactly what a
//! generated codec's OPEN-type key dispatch catches (`Key <id> Not Found`).
//!
//! This module decodes the tunnelled NRPPa PDU, answers the three procedures a
//! conformant gNB must answer, and re-wraps the answer for the uplink transport:
//!
//! | NRPPa procedure code | Procedure | Clause |
//! |---|---|---|
//! | 2  `id-e-CIDMeasurementInitiation`    | E-CID Measurement Initiation   | §8.2.1 |
//! | 9  `id-positioningInformationExchange`| Positioning Information Exchange | §8.2.6 |
//! | 16 `id-tRPInformationExchange`        | TRP Information Exchange       | §8.2.8 |
//!
//! # The NRPPa PDU wrapper has FOUR root fields, not three
//!
//! This is the one structural divergence from NGAP and the easiest way to get
//! the wire format wrong. NGAP's `InitiatingMessage` is
//! `{ procedureCode, criticality, value }`; NRPPa's (TS 38.455 §9.3.3) is
//! `{ procedureCode, criticality, nrppatransactionID, value }`, with
//! `NRPPATransactionID ::= INTEGER (0..32767)` encoded BETWEEN `criticality` and
//! the open-type `value`. Its range is 32768, which is `<= 65536`, so X.691
//! §13.2.6 puts it in TWO ALIGNED OCTETS rather than the 15 bits its upper bound
//! suggests. The generated types carry the field, so the codec gets both its
//! presence and its width right by construction rather than by a hand-written
//! encoder remembering to emit it.
//!
//! # Which transport carries which procedure
//!
//! E-CID Measurement Initiation and Positioning Information Exchange are about a
//! specific target UE, so they travel UE-associated (NGAP 8 down / 50 up) and their
//! answer must be returned on the same UE's NGAP association. TRP Information
//! Exchange asks about the node's transmission-reception points and names no UE, so
//! it travels non-UE-associated (NGAP 5 down / 47 up). Answering one on the other transport
//! would leave the LMF unable to correlate the response, which is why
//! [`NrppaResponse`] reports the association class it requires rather than
//! letting the caller assume.
//!
//! # Why the response is a SuccessfulOutcome and not a Report
//!
//! TS 38.455 §8.2.1.2 is a class-1 (request/response) procedure: the NG-RAN node
//! answers an E-CID MEASUREMENT INITIATION REQUEST with an E-CID MEASUREMENT
//! INITIATION RESPONSE, carrying the measurements when the report
//! characteristics are `onDemand`. The separate E-CID MEASUREMENT REPORT
//! (procedure 4) is a class-2 message used for SUBSEQUENT periodic reports once
//! a periodic measurement has been set up; it is not the answer to the request.
//! So this module answers with the response.

use crate::codec::nrppa_generated as nrppa;
use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NgapCodecError};
use asn1_codecs::aper::AperCodec;
use asn1_codecs::PerCodecData;
use thiserror::Error;

use crate::codec::generated::{
    DownlinkNonUEAssociatedNRPPaTransport, DownlinkUEAssociatedNRPPaTransport, InitiatingMessage,
    InitiatingMessageValue, NRPPa_PDU as NgapNrppaPdu, ProcedureCode as NgapProcedureCode,
    RoutingID, UplinkNonUEAssociatedNRPPaTransportProtocolIEs,
    UplinkNonUEAssociatedNRPPaTransportProtocolIEs_Entry,
    UplinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue,
    UplinkUEAssociatedNRPPaTransportProtocolIEs, UplinkUEAssociatedNRPPaTransportProtocolIEs_Entry,
    UplinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue, NGAP_PDU,
};

// ---------------------------------------------------------------------------
// NRPPa procedure codes (TS 38.455 §9.3.7)
// ---------------------------------------------------------------------------

/// `id-e-CIDMeasurementInitiation` — E-CID Measurement Initiation (§8.2.1).
pub const PROC_E_CID_MEASUREMENT_INITIATION: u8 = 2;
/// `id-positioningInformationExchange` — Positioning Information Exchange (§8.2.6).
pub const PROC_POSITIONING_INFORMATION_EXCHANGE: u8 = 9;
/// `id-tRPInformationExchange` — TRP Information Exchange (§8.2.8).
pub const PROC_TRP_INFORMATION_EXCHANGE: u8 = 16;

// ---------------------------------------------------------------------------
// NRPPa ProtocolIE-IDs used here (TS 38.455 §9.3.7)
// ---------------------------------------------------------------------------

/// `id-LMF-UE-Measurement-ID`.
pub const IE_LMF_UE_MEASUREMENT_ID: u16 = 2;
/// `id-ReportCharacteristics`.
pub const IE_REPORT_CHARACTERISTICS: u16 = 3;
/// `id-MeasurementQuantities`.
pub const IE_MEASUREMENT_QUANTITIES: u16 = 5;
/// `id-RAN-UE-Measurement-ID`.
pub const IE_RAN_UE_MEASUREMENT_ID: u16 = 6;
/// `id-E-CID-MeasurementResult`.
pub const IE_E_CID_MEASUREMENT_RESULT: u16 = 7;
/// `id-TRPInformationTypeListTRPReq`.
pub const IE_TRP_INFORMATION_TYPE_LIST_TRP_REQ: u16 = 29;
/// `id-TRPInformationListTRPResp`.
pub const IE_TRP_INFORMATION_LIST_TRP_RESP: u16 = 30;
/// `id-TRPList`.
pub const IE_TRP_LIST: u16 = 47;
/// `id-SFNInitialisationTime`.
pub const IE_SFN_INITIALISATION_TIME: u16 = 54;

// ---------------------------------------------------------------------------
// NGAP procedure codes for the four NRPPa transports (TS 38.413 §9.3.1.2)
// ---------------------------------------------------------------------------

/// `id-DownlinkNonUEAssociatedNRPPaTransport`.
pub const NGAP_PROC_DL_NON_UE_NRPPA: u8 = 5;
/// `id-DownlinkUEAssociatedNRPPaTransport`.
pub const NGAP_PROC_DL_UE_NRPPA: u8 = 8;
/// `id-UplinkNonUEAssociatedNRPPaTransport`.
pub const NGAP_PROC_UL_NON_UE_NRPPA: u8 = 47;
/// `id-UplinkUEAssociatedNRPPaTransport`.
pub const NGAP_PROC_UL_UE_NRPPA: u8 = 50;

/// `Criticality ::= ENUMERATED { reject, ignore, notify }` (TS 38.455 §9.3.6).
/// The three NRPPa values share the NGAP encoding but are a distinct type.
const CRITICALITY_REJECT: u8 = 0;
const CRITICALITY_IGNORE: u8 = 1;

/// Errors raised while handling an NRPPa procedure.
#[derive(Debug, Error)]
pub enum NrppaError {
    /// Codec error while encoding or decoding NGAP.
    #[error("NGAP codec error: {0}")]
    Ngap(#[from] NgapCodecError),

    /// The tunnelled NRPPa PDU could not be decoded.
    #[error("NRPPa APER decode error: {0}")]
    Decode(String),

    /// The NRPPa PDU could not be encoded.
    #[error("NRPPa APER encode error: {0}")]
    Encode(String),

    /// The NGAP PDU is not one of the NRPPa transport procedures.
    #[error("not an NRPPa transport PDU")]
    NotNrppaTransport,

    /// The NRPPa procedure is one this gNB does not implement.
    #[error("unsupported NRPPa procedure code {0}")]
    UnsupportedProcedure(u8),

    /// A mandatory IE was absent from the request.
    #[error("missing mandatory NRPPa IE: {0}")]
    MissingMandatoryIe(&'static str),
}

/// Which NGAP transport an NRPPa PDU arrived on, and therefore which uplink
/// transport its answer must use.
///
/// TS 38.413 §8.14: a UE-associated NRPPa PDU is bound to a UE's NGAP
/// association and carries the two NGAP UE IDs; a non-UE-associated one carries
/// a `RoutingID` identifying the LMF instead. The two are not interchangeable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NrppaAssociation {
    /// UE-associated (NGAP 8 down / 46 up). Carries the NGAP UE ID pair.
    UeAssociated {
        /// `AMF-UE-NGAP-ID` of the target UE.
        amf_ue_ngap_id: u64,
        /// `RAN-UE-NGAP-ID` of the target UE.
        ran_ue_ngap_id: u32,
    },
    /// Non-UE-associated (NGAP 7 down / 45 up). Carries the LMF `RoutingID`
    /// verbatim so the answer can be routed back to the originating LMF.
    NonUeAssociated {
        /// The `Routing-ID` octets exactly as received.
        routing_id: Vec<u8>,
    },
}

/// A decoded NRPPa request, with the transport it arrived on.
#[derive(Debug, Clone, PartialEq)]
pub struct NrppaRequest {
    /// The NGAP association class the request arrived on.
    pub association: NrppaAssociation,
    /// `NRPPATransactionID` — echoed unchanged in the answer so the LMF can
    /// correlate (TS 38.455 §9.2.5).
    pub transaction_id: u16,
    /// Which NRPPa procedure was requested.
    pub procedure: NrppaProcedure,
}

/// The NRPPa procedures this gNB answers.
#[derive(Debug, Clone, PartialEq)]
pub enum NrppaProcedure {
    /// E-CID Measurement Initiation Request (§8.2.1).
    ECidMeasurementInitiation {
        /// `LMF-UE-Measurement-ID`, mandatory, echoed in the response.
        lmf_ue_measurement_id: u8,
    },
    /// Positioning Information Request (§8.2.6). Every IE of this request is
    /// OPTIONAL (TS 38.455 §9.1.10), so there is nothing mandatory to extract:
    /// the request's arrival is the whole trigger.
    PositioningInformationExchange,
    /// TRP Information Request (§8.2.8).
    TrpInformationExchange {
        /// `TRPList`, OPTIONAL: when present, only these TRPs are asked about;
        /// when absent, the node reports all of its TRPs.
        requested_trps: Option<Vec<u16>>,
        /// `TRPInformationTypeListTRPReq`, mandatory: which information types
        /// the LMF wants per TRP.
        requested_types: Vec<u8>,
    },
}

/// An NRPPa answer, ready to be wrapped into an uplink NGAP transport.
#[derive(Debug, Clone, PartialEq)]
pub struct NrppaResponse {
    /// The association the answer must be sent on — the same class the request
    /// arrived on.
    pub association: NrppaAssociation,
    /// The APER-encoded NRPPa PDU to tunnel.
    pub nrppa_pdu: Vec<u8>,
}

/// The gNB-side facts an NRPPa answer is synthesised from.
///
/// These come from the gNB's own configuration and radio state; nothing here is
/// invented per-request, so two requests against an unchanged cell produce
/// identical measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct NrppaCellInfo {
    /// BCD-encoded PLMN identity, 3 octets (TS 38.413 §9.3.3.5).
    pub plmn_identity: [u8; 3],
    /// 36-bit NR Cell Identity, in the low 36 bits.
    pub nr_cell_identity: u64,
    /// Tracking Area Code, 3 octets.
    pub tac: [u8; 3],
    /// Serving-cell physical cell identity, `NR-PCI ::= INTEGER (0..1007)`.
    pub nr_pci: u16,
    /// Downlink `NR-ARFCN ::= INTEGER (0..3279165)`.
    pub nr_arfcn: u32,
    /// Measured SS-RSRP as `ValueRSRP-NR ::= INTEGER (0..127)`
    /// (TS 38.455 §9.2.36; maps to -156..-31 dBm per TS 38.133).
    pub ss_rsrp: u8,
    /// The node's TRP identities, `TRP-ID ::= INTEGER (0..65535)`.
    pub trp_ids: Vec<u16>,
}

// ---------------------------------------------------------------------------
// APER helpers for the tunnelled NRPPa PDU
// ---------------------------------------------------------------------------

fn encode_nrppa_pdu(pdu: &nrppa::NRPPA_PDU) -> Result<Vec<u8>, NrppaError> {
    let mut data = PerCodecData::new_aper();
    pdu.aper_encode(&mut data)
        .map_err(|e| NrppaError::Encode(format!("{e:?}")))?;
    Ok(data.into_bytes())
}

fn decode_nrppa_pdu(bytes: &[u8]) -> Result<nrppa::NRPPA_PDU, NrppaError> {
    let mut data = PerCodecData::from_slice_aper(bytes);
    nrppa::NRPPA_PDU::aper_decode(&mut data).map_err(|e| NrppaError::Decode(format!("{e:?}")))
}

// ---------------------------------------------------------------------------
// Downlink: unwrap the NGAP transport and decode the NRPPa request
// ---------------------------------------------------------------------------

/// Extract the NRPPa payload and association from a downlink NGAP transport.
///
/// Returns [`NrppaError::NotNrppaTransport`] when the PDU is a different NGAP
/// procedure, which is how the dispatch chain distinguishes "not mine" from
/// "mine but malformed".
fn unwrap_downlink(pdu: &NGAP_PDU) -> Result<(NrppaAssociation, Vec<u8>), NrppaError> {
    let NGAP_PDU::InitiatingMessage(init) = pdu else {
        return Err(NrppaError::NotNrppaTransport);
    };

    match &init.value {
        InitiatingMessageValue::Id_DownlinkUEAssociatedNRPPaTransport(dl) => {
            unwrap_dl_ue_associated(dl)
        }
        InitiatingMessageValue::Id_DownlinkNonUEAssociatedNRPPaTransport(dl) => {
            unwrap_dl_non_ue_associated(dl)
        }
        _ => Err(NrppaError::NotNrppaTransport),
    }
}

fn unwrap_dl_ue_associated(
    dl: &DownlinkUEAssociatedNRPPaTransport,
) -> Result<(NrppaAssociation, Vec<u8>), NrppaError> {
    use crate::codec::generated::DownlinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;

    let mut amf_ue_ngap_id = None;
    let mut ran_ue_ngap_id = None;
    let mut payload = None;

    for ie in &dl.protocol_i_es.0 {
        match &ie.value {
            V::Id_AMF_UE_NGAP_ID(id) => amf_ue_ngap_id = Some(id.0),
            V::Id_RAN_UE_NGAP_ID(id) => ran_ue_ngap_id = Some(id.0),
            V::Id_NRPPa_PDU(p) => payload = Some(p.0.clone()),
            _ => {}
        }
    }

    // All three IEs are mandatory in TS 38.413 §9.2.9.4, so a request missing
    // any of them cannot be answered on the right association.
    Ok((
        NrppaAssociation::UeAssociated {
            amf_ue_ngap_id: amf_ue_ngap_id
                .ok_or(NrppaError::MissingMandatoryIe("AMF-UE-NGAP-ID"))?,
            ran_ue_ngap_id: ran_ue_ngap_id
                .ok_or(NrppaError::MissingMandatoryIe("RAN-UE-NGAP-ID"))?,
        },
        payload.ok_or(NrppaError::MissingMandatoryIe("NRPPa-PDU"))?,
    ))
}

fn unwrap_dl_non_ue_associated(
    dl: &DownlinkNonUEAssociatedNRPPaTransport,
) -> Result<(NrppaAssociation, Vec<u8>), NrppaError> {
    use crate::codec::generated::DownlinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;

    let mut routing_id = None;
    let mut payload = None;

    for ie in &dl.protocol_i_es.0 {
        match &ie.value {
            V::Id_RoutingID(r) => routing_id = Some(r.0.clone()),
            V::Id_NRPPa_PDU(p) => payload = Some(p.0.clone()),
        }
    }

    Ok((
        NrppaAssociation::NonUeAssociated {
            routing_id: routing_id.ok_or(NrppaError::MissingMandatoryIe("RoutingID"))?,
        },
        payload.ok_or(NrppaError::MissingMandatoryIe("NRPPa-PDU"))?,
    ))
}

/// Decode a downlink NGAP NRPPa transport into a typed [`NrppaRequest`].
///
/// This is the entry point the gNB's NGAP dispatch calls. It fails with
/// [`NrppaError::NotNrppaTransport`] for any other NGAP procedure, so it can sit
/// in the `else if` chain without swallowing unrelated PDUs.
pub fn parse_downlink_nrppa_transport(pdu_bytes: &[u8]) -> Result<NrppaRequest, NrppaError> {
    let ngap = decode_ngap_pdu(pdu_bytes)?;
    let (association, payload) = unwrap_downlink(&ngap)?;
    let nrppa_pdu = decode_nrppa_pdu(&payload)?;

    let nrppa::NRPPA_PDU::InitiatingMessage(init) = &nrppa_pdu else {
        // A SuccessfulOutcome or UnsuccessfulOutcome arriving downlink is an
        // answer to something the gNB initiated, not a request to answer.
        return Err(NrppaError::Decode(
            "downlink NRPPa PDU is not an InitiatingMessage".to_string(),
        ));
    };

    let transaction_id = init.nrppatransaction_id.0;
    let procedure = match &init.value {
        nrppa::InitiatingMessageValue::Id_e_CIDMeasurementInitiation(req) => {
            parse_e_cid_request(req)?
        }
        nrppa::InitiatingMessageValue::Id_positioningInformationExchange(_) => {
            NrppaProcedure::PositioningInformationExchange
        }
        nrppa::InitiatingMessageValue::Id_tRPInformationExchange(req) => parse_trp_request(req)?,
        _ => return Err(NrppaError::UnsupportedProcedure(init.procedure_code.0)),
    };

    Ok(NrppaRequest {
        association,
        transaction_id,
        procedure,
    })
}

fn parse_e_cid_request(
    req: &nrppa::E_CIDMeasurementInitiationRequest,
) -> Result<NrppaProcedure, NrppaError> {
    use nrppa::E_CIDMeasurementInitiationRequestProtocolIEs_EntryValue as V;

    let mut lmf_ue_measurement_id = None;
    for ie in &req.protocol_i_es.0 {
        if let V::Id_LMF_UE_Measurement_ID(id) = &ie.value {
            lmf_ue_measurement_id = Some(id.0);
        }
    }

    Ok(NrppaProcedure::ECidMeasurementInitiation {
        lmf_ue_measurement_id: lmf_ue_measurement_id
            .ok_or(NrppaError::MissingMandatoryIe("LMF-UE-Measurement-ID"))?,
    })
}

fn parse_trp_request(req: &nrppa::TRPInformationRequest) -> Result<NrppaProcedure, NrppaError> {
    use nrppa::TRPInformationRequestProtocolIEs_EntryValue as V;

    let mut requested_trps = None;
    let mut requested_types = None;

    for ie in &req.protocol_i_es.0 {
        match &ie.value {
            V::Id_TRPList(list) => {
                requested_trps = Some(list.0.iter().map(|item| item.trp_id.0).collect());
            }
            V::Id_TRPInformationTypeListTRPReq(list) => {
                // Each element of TRPInformationTypeListTRPReq is itself a
                // ProtocolIE-Single-Container wrapping id-TRPInformationTypeItem
                // (IE 57), not a bare enumerated value.
                use nrppa::TRPInformationTypeListTRPReq_EntryValue as Item;
                requested_types = Some(
                    list.0
                        .iter()
                        .map(|entry| {
                            let Item::Id_TRPInformationTypeItem(ty) = &entry.value;
                            ty.0
                        })
                        .collect(),
                );
            }
        }
    }

    Ok(NrppaProcedure::TrpInformationExchange {
        requested_trps,
        requested_types: requested_types.ok_or(NrppaError::MissingMandatoryIe(
            "TRPInformationTypeListTRPReq",
        ))?,
    })
}

// ---------------------------------------------------------------------------
// Uplink: build the NRPPa answer
// ---------------------------------------------------------------------------

/// Build the `E-CID-MeasurementResult` a gNB reports for its serving cell.
///
/// TS 38.455 §9.2.2: `servingCell-ID` and `servingCellTAC` are mandatory; the
/// measurements ride in the OPTIONAL `measuredResults`. The SS-RSRP is carried in
/// the CHOICE's extension arm (`id-ResultSS-RSRP`, IE 32) because SS-RSRP is an
/// NR quantity and the CHOICE's root alternatives are all E-UTRA
/// (`valueAngleOfArrival-EUTRA`, `resultRSRP-EUTRA`, ...). An encoder that tried
/// to report NR RSRP in a root arm would be reporting an E-UTRA measurement.
fn build_e_cid_measurement_result(cell: &NrppaCellInfo) -> nrppa::E_CID_MeasurementResult {
    let mut cell_id = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::new();
    for bit in (0..36).rev() {
        cell_id.push((cell.nr_cell_identity >> bit) & 1 == 1);
    }

    let ss_rsrp_item = nrppa::ResultSS_RSRP_Item {
        nr_pci: nrppa::NR_PCI(cell.nr_pci),
        nr_arfcn: nrppa::NR_ARFCN(cell.nr_arfcn),
        cgi_nr: None,
        value_ss_rsrp_cell: Some(nrppa::ValueRSRP_NR(cell.ss_rsrp)),
        ss_rsrp_per_ssb: None,
        ie_extensions: None,
    };

    let measured_results =
        nrppa::MeasuredResults(vec![nrppa::MeasuredResultsValue::Choice_Extension(
            nrppa::MeasuredResultsValue_choice_Extension {
                id: nrppa::ProtocolIE_ID(32),
                criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                value: nrppa::MeasuredResultsValue_choice_ExtensionValue::Id_ResultSS_RSRP(
                    nrppa::ResultSS_RSRP(vec![ss_rsrp_item]),
                ),
            },
        )]);

    nrppa::E_CID_MeasurementResult {
        serving_cell_id: nrppa::NG_RAN_CGI {
            plmn_identity: nrppa::PLMN_Identity(cell.plmn_identity.to_vec()),
            ng_ra_ncell: nrppa::NG_RANCell::NR_CellID(nrppa::NRCellIdentifier(cell_id)),
            ie_extensions: None,
        },
        serving_cell_tac: nrppa::TAC(cell.tac.to_vec()),
        ng_ran_access_point_position: None,
        measured_results: Some(measured_results),
        ie_extensions: None,
    }
}

/// Build an `E-CIDMeasurementInitiationResponse` (§8.2.1, procedure 2,
/// successful outcome).
///
/// `LMF-UE-Measurement-ID` is echoed from the request and `RAN-UE-Measurement-ID`
/// is the node's own handle for the measurement; both are mandatory
/// (TS 38.455 §9.1.3). The measurement result is OPTIONAL in the schema but is
/// the point of the procedure, so it is always emitted.
fn build_e_cid_response(
    transaction_id: u16,
    lmf_ue_measurement_id: u8,
    ran_ue_measurement_id: u8,
    cell: &NrppaCellInfo,
) -> nrppa::NRPPA_PDU {
    use nrppa::E_CIDMeasurementInitiationResponseProtocolIEs_EntryValue as V;

    let ies = vec![
        nrppa::E_CIDMeasurementInitiationResponseProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_LMF_UE_MEASUREMENT_ID),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: V::Id_LMF_UE_Measurement_ID(nrppa::UE_Measurement_ID(lmf_ue_measurement_id)),
        },
        nrppa::E_CIDMeasurementInitiationResponseProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_RAN_UE_MEASUREMENT_ID),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: V::Id_RAN_UE_Measurement_ID(nrppa::UE_Measurement_ID(ran_ue_measurement_id)),
        },
        nrppa::E_CIDMeasurementInitiationResponseProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_E_CID_MEASUREMENT_RESULT),
            criticality: nrppa::Criticality(CRITICALITY_IGNORE),
            value: V::Id_E_CID_MeasurementResult(build_e_cid_measurement_result(cell)),
        },
    ];

    nrppa::NRPPA_PDU::SuccessfulOutcome(nrppa::SuccessfulOutcome {
        procedure_code: nrppa::ProcedureCode(PROC_E_CID_MEASUREMENT_INITIATION),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::SuccessfulOutcomeValue::Id_e_CIDMeasurementInitiation(
            nrppa::E_CIDMeasurementInitiationResponse {
                protocol_i_es: nrppa::E_CIDMeasurementInitiationResponseProtocolIEs(ies),
            },
        ),
    })
}

/// Build a `PositioningInformationResponse` (§8.2.6, procedure 9, successful
/// outcome).
///
/// Every IE of this message is OPTIONAL (TS 38.455 §9.1.11). The gNB has no SRS
/// configured for the target on an NRPPa-only path, so what it can honestly
/// supply is the SFN timing reference: `id-SFNInitialisationTime`
/// (`RelativeTime1900 ::= BIT STRING (SIZE(64))`), the NTP-format instant at
/// which SFN 0 of the serving cell began. That is precisely the datum an LMF
/// needs to relate a UE's reported timing to absolute time, so the response is
/// informative rather than empty.
fn build_positioning_information_response(
    transaction_id: u16,
    sfn_initialisation_time: u64,
) -> nrppa::NRPPA_PDU {
    use nrppa::PositioningInformationResponseProtocolIEs_EntryValue as V;

    let mut time_bits = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::new();
    for bit in (0..64).rev() {
        time_bits.push((sfn_initialisation_time >> bit) & 1 == 1);
    }

    let ies = vec![nrppa::PositioningInformationResponseProtocolIEs_Entry {
        id: nrppa::ProtocolIE_ID(IE_SFN_INITIALISATION_TIME),
        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
        value: V::Id_SFNInitialisationTime(nrppa::RelativeTime1900(time_bits)),
    }];

    nrppa::NRPPA_PDU::SuccessfulOutcome(nrppa::SuccessfulOutcome {
        procedure_code: nrppa::ProcedureCode(PROC_POSITIONING_INFORMATION_EXCHANGE),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::SuccessfulOutcomeValue::Id_positioningInformationExchange(
            nrppa::PositioningInformationResponse {
                protocol_i_es: nrppa::PositioningInformationResponseProtocolIEs(ies),
            },
        ),
    })
}

/// Build a `TRPInformationResponse` (§8.2.8, procedure 16, successful outcome).
///
/// `TRPInformationListTRPResp` is mandatory (TS 38.455 §9.1.29). Each entry
/// answers for one TRP with the information types the LMF asked for; the node
/// reports only the types it holds, which per §8.2.8.2 is the required behaviour
/// (the LMF's request is a request, not an assertion that every type exists).
/// This node can answer PCI (type 0), NR-CGI (type 1) and ARFCN (type 2) from its
/// own cell configuration.
fn build_trp_information_response(
    transaction_id: u16,
    requested_trps: Option<&[u16]>,
    requested_types: &[u8],
    cell: &NrppaCellInfo,
) -> nrppa::NRPPA_PDU {
    use nrppa::TRPInformationResponseProtocolIEs_EntryValue as V;

    // An absent TRPList means "all of this node's TRPs"; a present one filters
    // to the intersection, so an LMF asking about a TRP this node does not have
    // gets no entry for it rather than a fabricated one.
    //
    // `TRP-ID ::= INTEGER (1..maxnoTRPs, ...)` (TS 38.455 §9.2.23), so a
    // configured id of 0 is not encodable and is dropped here rather than shifted
    // into range: renumbering it would rename the reference point the LMF
    // correlates its measurements against.
    let trps: Vec<u16> = cell
        .trp_ids
        .iter()
        .copied()
        .filter(|id| *id >= 1)
        .filter(|id| requested_trps.is_none_or(|requested| requested.contains(id)))
        .collect();

    let mut cell_id = bitvec::vec::BitVec::<u8, bitvec::order::Msb0>::new();
    for bit in (0..36).rev() {
        cell_id.push((cell.nr_cell_identity >> bit) & 1 == 1);
    }

    let entries: Vec<nrppa::TRPInformationListTRPResp_Entry> = trps
        .iter()
        .map(|trp_id| {
            let mut items = Vec::new();
            for ty in requested_types {
                match ty {
                    // TRPInformationTypeItem: 0 = nR-PCI, 1 = nG-RAN-CGI,
                    // 2 = aRFCN (TS 38.455 §9.2.24).
                    0 => items.push(nrppa::TRPInformationTypeResponseItem::PCI_NR(
                        nrppa::TRPInformationTypeResponseItem_pCI_NR(cell.nr_pci),
                    )),
                    1 => items.push(nrppa::TRPInformationTypeResponseItem::CGI_NR(
                        nrppa::CGI_NR {
                            plmn_identity: nrppa::PLMN_Identity(cell.plmn_identity.to_vec()),
                            n_rcell_identifier: nrppa::NRCellIdentifier(cell_id.clone()),
                            ie_extensions: None,
                        },
                    )),
                    2 => items.push(nrppa::TRPInformationTypeResponseItem::ARFCN(
                        nrppa::TRPInformationTypeResponseItem_aRFCN(cell.nr_arfcn),
                    )),
                    // Types 3..7 (PRS configuration, SSB information, SFN
                    // initialisation time, spatial direction, geographical
                    // coordinates) describe state this node does not model, so
                    // they are omitted rather than answered with placeholders.
                    _ => {}
                }
            }

            nrppa::TRPInformationListTRPResp_Entry {
                trp_information: nrppa::TRPInformation {
                    trp_id: nrppa::TRP_ID(*trp_id),
                    trp_information_type_response_list: nrppa::TRPInformationTypeResponseList(
                        items,
                    ),
                    ie_extensions: None,
                },
                ie_extensions: None,
            }
        })
        .collect();

    let ies = vec![nrppa::TRPInformationResponseProtocolIEs_Entry {
        id: nrppa::ProtocolIE_ID(IE_TRP_INFORMATION_LIST_TRP_RESP),
        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
        value: V::Id_TRPInformationListTRPResp(nrppa::TRPInformationListTRPResp(entries)),
    }];

    nrppa::NRPPA_PDU::SuccessfulOutcome(nrppa::SuccessfulOutcome {
        procedure_code: nrppa::ProcedureCode(PROC_TRP_INFORMATION_EXCHANGE),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::SuccessfulOutcomeValue::Id_tRPInformationExchange(
            nrppa::TRPInformationResponse {
                protocol_i_es: nrppa::TRPInformationResponseProtocolIEs(ies),
            },
        ),
    })
}

/// Answer a decoded NRPPa request from the gNB's own cell state.
///
/// `ran_ue_measurement_id` is the node's handle for an E-CID measurement; the
/// caller allocates it so successive measurements for one UE are distinguishable.
/// `sfn_initialisation_time` is the NTP-format instant SFN 0 began, used by the
/// Positioning Information answer.
pub fn build_nrppa_response(
    request: &NrppaRequest,
    cell: &NrppaCellInfo,
    ran_ue_measurement_id: u8,
    sfn_initialisation_time: u64,
) -> Result<NrppaResponse, NrppaError> {
    let pdu = match &request.procedure {
        NrppaProcedure::ECidMeasurementInitiation {
            lmf_ue_measurement_id,
        } => build_e_cid_response(
            request.transaction_id,
            *lmf_ue_measurement_id,
            ran_ue_measurement_id,
            cell,
        ),
        NrppaProcedure::PositioningInformationExchange => {
            build_positioning_information_response(request.transaction_id, sfn_initialisation_time)
        }
        NrppaProcedure::TrpInformationExchange {
            requested_trps,
            requested_types,
        } => build_trp_information_response(
            request.transaction_id,
            requested_trps.as_deref(),
            requested_types,
            cell,
        ),
    };

    Ok(NrppaResponse {
        association: request.association.clone(),
        nrppa_pdu: encode_nrppa_pdu(&pdu)?,
    })
}

/// Wrap an NRPPa answer into the matching uplink NGAP transport and encode it.
///
/// TS 38.413 §8.14.3/§8.14.4: the uplink transport mirrors the downlink one, so a
/// UE-associated request is answered on NGAP 46 with the same UE ID pair and a
/// non-UE-associated one on NGAP 45 with the same `RoutingID`.
pub fn encode_uplink_nrppa_transport(response: &NrppaResponse) -> Result<Vec<u8>, NrppaError> {
    let pdu = match &response.association {
        NrppaAssociation::UeAssociated {
            amf_ue_ngap_id,
            ran_ue_ngap_id,
        } => {
            use crate::codec::generated::{AMF_UE_NGAP_ID, RAN_UE_NGAP_ID};
            use UplinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;

            let ies = vec![
                UplinkUEAssociatedNRPPaTransportProtocolIEs_Entry {
                    id: crate::codec::generated::ProtocolIE_ID(10),
                    criticality: crate::codec::generated::Criticality(CRITICALITY_REJECT),
                    value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(*amf_ue_ngap_id)),
                },
                UplinkUEAssociatedNRPPaTransportProtocolIEs_Entry {
                    id: crate::codec::generated::ProtocolIE_ID(85),
                    criticality: crate::codec::generated::Criticality(CRITICALITY_REJECT),
                    value: V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(*ran_ue_ngap_id)),
                },
                UplinkUEAssociatedNRPPaTransportProtocolIEs_Entry {
                    id: crate::codec::generated::ProtocolIE_ID(46),
                    criticality: crate::codec::generated::Criticality(CRITICALITY_REJECT),
                    value: V::Id_NRPPa_PDU(NgapNrppaPdu(response.nrppa_pdu.clone())),
                },
            ];

            NGAP_PDU::InitiatingMessage(InitiatingMessage {
                procedure_code: NgapProcedureCode(NGAP_PROC_UL_UE_NRPPA),
                criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
                value: InitiatingMessageValue::Id_UplinkUEAssociatedNRPPaTransport(
                    crate::codec::generated::UplinkUEAssociatedNRPPaTransport {
                        protocol_i_es: UplinkUEAssociatedNRPPaTransportProtocolIEs(ies),
                    },
                ),
            })
        }
        NrppaAssociation::NonUeAssociated { routing_id } => {
            use UplinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;

            let ies = vec![
                UplinkNonUEAssociatedNRPPaTransportProtocolIEs_Entry {
                    id: crate::codec::generated::ProtocolIE_ID(89),
                    criticality: crate::codec::generated::Criticality(CRITICALITY_REJECT),
                    value: V::Id_RoutingID(RoutingID(routing_id.clone())),
                },
                UplinkNonUEAssociatedNRPPaTransportProtocolIEs_Entry {
                    id: crate::codec::generated::ProtocolIE_ID(46),
                    criticality: crate::codec::generated::Criticality(CRITICALITY_REJECT),
                    value: V::Id_NRPPa_PDU(NgapNrppaPdu(response.nrppa_pdu.clone())),
                },
            ];

            NGAP_PDU::InitiatingMessage(InitiatingMessage {
                procedure_code: NgapProcedureCode(NGAP_PROC_UL_NON_UE_NRPPA),
                criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
                value: InitiatingMessageValue::Id_UplinkNonUEAssociatedNRPPaTransport(
                    crate::codec::generated::UplinkNonUEAssociatedNRPPaTransport {
                        protocol_i_es: UplinkNonUEAssociatedNRPPaTransportProtocolIEs(ies),
                    },
                ),
            })
        }
    };

    Ok(encode_ngap_pdu(&pdu)?)
}

#[cfg(test)]
mod tests;
