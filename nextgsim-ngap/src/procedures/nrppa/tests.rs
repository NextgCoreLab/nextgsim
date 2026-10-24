//! Tests for the NRPPa positioning procedures (TS 38.455).
//!
//! # Where the golden bytes come from
//!
//! A round trip against this module's own encoder proves only self-consistency,
//! so the header fixture below is derived BY HAND from X.691 rather than captured
//! from either encoder, and it is the shape nextgcore's independent hand-written
//! NRPPa codec produces (`nextgcore-asn1c/src/nrppa/pdu.rs`, whose
//! `InitiatingMessage::encode_aper` emits `procedureCode`, `criticality`,
//! `nrppatransactionID` and only then the fragmented open-type `value`). The
//! derivation is spelled out in [`ecid_request_header_bytes`] so a future reader
//! can check it against the spec instead of trusting the encoder that produced
//! it.

use super::*;

/// PLMN 001/01, NCI 0x0000000001, TAC 1, PCI 42, ARFCN 632628, SS-RSRP 80,
/// two TRPs. Fixed values so a measurement is reproducible.
fn test_cell() -> NrppaCellInfo {
    NrppaCellInfo {
        plmn_identity: [0x00, 0xf1, 0x10],
        nr_cell_identity: 1,
        tac: [0x00, 0x00, 0x01],
        nr_pci: 42,
        nr_arfcn: 632_628,
        ss_rsrp: 80,
        trp_ids: vec![1, 2],
    }
}

/// The first five octets of an APER-encoded NRPPa `E-CIDMeasurementInitiationRequest`
/// with `nrppatransactionID = 7`, derived by hand from X.691.
///
/// Each field here is APER-ALIGNED, which is what makes the header land on octet
/// boundaries rather than packing into a dense bit stream:
///
/// | Octet | Field | ASN.1 | X.691 rule | Value |
/// |---|---|---|---|---|
/// | 0 | `NRPPA-PDU` CHOICE | 3 alternatives, extensible | §23.5-23.6: 1 extension bit + 2-bit index = 3 bits, padded because the next field is octet-aligned | `0x00` — not extended, alternative 0 = `initiatingMessage` |
/// | 1 | `procedureCode` | `INTEGER (0..255)` | §13.2.5: range is exactly 256, so "one octet, aligned" | `0x02` = `id-e-CIDMeasurementInitiation` |
/// | 2 | `criticality` | `ENUMERATED {reject,ignore,notify}` | 2-bit index, padded before the next aligned field | `0x00` = `reject` |
/// | 3-4 | `nrppatransactionID` | `INTEGER (0..32767)` | §13.2.6: range 32768 <= 65536, so "two octets, aligned" | `0x0007` |
///
/// The load-bearing claim is that `nrppatransactionID` sits between `criticality`
/// and `value` at all: TS 38.455 §9.3.3 puts it there and NGAP has no such
/// field, so an encoder that omitted it would shift the whole open-type `value`
/// two octets earlier and a peer would decode garbage. Asserted as a PREFIX so
/// the test pins the header layout without also pinning the IE payload.
///
/// nextgcore's independent hand-written codec produces the same layout: its
/// `InitiatingMessage::encode_aper` writes `procedureCode`, `criticality`, then
/// `nrppa_transaction_id` through `encode_constrained_whole_number`, whose
/// `range <= 65536` arm likewise aligns and then writes 16 bits.
const ECID_REQUEST_HEADER_PREFIX: [u8; 5] = [0x00, 0x02, 0x00, 0x00, 0x07];

fn ecid_request_header_bytes() -> Vec<u8> {
    let request = build_ecid_request(7, 3);
    encode_nrppa_pdu(&request).expect("E-CID request encodes")
}

/// Build the `E-CIDMeasurementInitiationRequest` an LMF sends, so the gNB side
/// can be driven from a real request rather than from a hand-built struct.
///
/// Carries the three mandatory IEs of TS 38.455 §9.1.2: `LMF-UE-Measurement-ID`,
/// `ReportCharacteristics` (`onDemand`, so no `MeasurementPeriodicity` is
/// required) and `MeasurementQuantities`.
fn build_ecid_request(transaction_id: u16, lmf_ue_measurement_id: u16) -> nrppa::NRPPA_PDU {
    use nrppa::E_CIDMeasurementInitiationRequestProtocolIEs_EntryValue as V;

    // Each element of MeasurementQuantities is a ProtocolIE-Single-Container
    // wrapping id-MeasurementQuantities-Item (IE 11).
    let quantities = nrppa::MeasurementQuantities(vec![nrppa::MeasurementQuantities_Entry {
        id: nrppa::ProtocolIE_ID(11),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        value: nrppa::MeasurementQuantities_EntryValue::Id_MeasurementQuantities_Item(
            nrppa::MeasurementQuantities_Item {
                measurement_quantities_value: nrppa::MeasurementQuantitiesValue(
                    nrppa::MeasurementQuantitiesValue::R_SRP,
                ),
                ie_extensions: None,
            },
        ),
    }]);

    let ies = vec![
        nrppa::E_CIDMeasurementInitiationRequestProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_LMF_UE_MEASUREMENT_ID),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: V::Id_LMF_UE_Measurement_ID(nrppa::UE_Measurement_ID(lmf_ue_measurement_id)),
        },
        nrppa::E_CIDMeasurementInitiationRequestProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_REPORT_CHARACTERISTICS),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: V::Id_ReportCharacteristics(nrppa::ReportCharacteristics(
                nrppa::ReportCharacteristics::ON_DEMAND,
            )),
        },
        nrppa::E_CIDMeasurementInitiationRequestProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_MEASUREMENT_QUANTITIES),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: V::Id_MeasurementQuantities(quantities),
        },
    ];

    nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(PROC_E_CID_MEASUREMENT_INITIATION),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::InitiatingMessageValue::Id_e_CIDMeasurementInitiation(
            nrppa::E_CIDMeasurementInitiationRequest {
                protocol_i_es: nrppa::E_CIDMeasurementInitiationRequestProtocolIEs(ies),
            },
        ),
    })
}

/// Build the `TRPInformationRequest` an LMF sends (§8.2.8).
fn build_trp_request(
    transaction_id: u16,
    trp_ids: Option<Vec<u16>>,
    types: &[u8],
) -> nrppa::NRPPA_PDU {
    use nrppa::TRPInformationRequestProtocolIEs_EntryValue as V;

    let mut ies = Vec::new();
    if let Some(ids) = trp_ids {
        ies.push(nrppa::TRPInformationRequestProtocolIEs_Entry {
            id: nrppa::ProtocolIE_ID(IE_TRP_LIST),
            criticality: nrppa::Criticality(CRITICALITY_IGNORE),
            value: V::Id_TRPList(nrppa::TRPList(
                ids.into_iter()
                    .map(|trp_id| nrppa::TRPItem {
                        trp_id: nrppa::TRP_ID(trp_id),
                        ie_extensions: None,
                    })
                    .collect(),
            )),
        });
    }

    let type_entries = types
        .iter()
        .map(|ty| nrppa::TRPInformationTypeListTRPReq_Entry {
            id: nrppa::ProtocolIE_ID(57),
            criticality: nrppa::Criticality(CRITICALITY_REJECT),
            value: nrppa::TRPInformationTypeListTRPReq_EntryValue::Id_TRPInformationTypeItem(
                nrppa::TRPInformationTypeItem(*ty),
            ),
        })
        .collect();

    ies.push(nrppa::TRPInformationRequestProtocolIEs_Entry {
        id: nrppa::ProtocolIE_ID(IE_TRP_INFORMATION_TYPE_LIST_TRP_REQ),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        value: V::Id_TRPInformationTypeListTRPReq(nrppa::TRPInformationTypeListTRPReq(
            type_entries,
        )),
    });

    nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(PROC_TRP_INFORMATION_EXCHANGE),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::InitiatingMessageValue::Id_tRPInformationExchange(
            nrppa::TRPInformationRequest {
                protocol_i_es: nrppa::TRPInformationRequestProtocolIEs(ies),
            },
        ),
    })
}

/// Build a `PositioningInformationRequest` (§8.2.6). All its IEs are OPTIONAL, so
/// an empty container is a legal request and is what an LMF asking only for the
/// timing reference sends.
fn build_positioning_information_request(transaction_id: u16) -> nrppa::NRPPA_PDU {
    nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(PROC_POSITIONING_INFORMATION_EXCHANGE),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value: nrppa::InitiatingMessageValue::Id_positioningInformationExchange(
            nrppa::PositioningInformationRequest {
                protocol_i_es: nrppa::PositioningInformationRequestProtocolIEs(vec![]),
            },
        ),
    })
}

/// The `Routing ID` the downlink fixtures carry: an LMF's NF instance id, which
/// is what nextgcore amfd puts there.
const LMF_ROUTING_ID: &[u8] = b"7a31ef31-6d17-4ace-9aa7-d7697cbed513";

/// Wrap an NRPPa PDU into a downlink UE-associated NGAP transport (NGAP 8),
/// standing in for what the AMF relays from the LMF.
fn wrap_dl_ue_associated(nrppa_pdu: &nrppa::NRPPA_PDU, amf_id: u64, ran_id: u32) -> Vec<u8> {
    use crate::codec::generated::{
        Criticality as NgapCriticality, DownlinkUEAssociatedNRPPaTransportProtocolIEs,
        DownlinkUEAssociatedNRPPaTransportProtocolIEs_Entry as Entry,
        DownlinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V, ProtocolIE_ID as NgapIeId,
        AMF_UE_NGAP_ID, RAN_UE_NGAP_ID,
    };

    let payload = encode_nrppa_pdu(nrppa_pdu).expect("NRPPa PDU encodes");
    let ies = vec![
        Entry {
            id: NgapIeId(10),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(amf_id)),
        },
        Entry {
            id: NgapIeId(85),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(ran_id)),
        },
        Entry {
            id: NgapIeId(89),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_RoutingID(RoutingID(LMF_ROUTING_ID.to_vec())),
        },
        Entry {
            id: NgapIeId(46),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_NRPPa_PDU(NgapNrppaPdu(payload)),
        },
    ];

    let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: NgapProcedureCode(NGAP_PROC_DL_UE_NRPPA),
        criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
        value: InitiatingMessageValue::Id_DownlinkUEAssociatedNRPPaTransport(
            DownlinkUEAssociatedNRPPaTransport {
                protocol_i_es: DownlinkUEAssociatedNRPPaTransportProtocolIEs(ies),
            },
        ),
    });

    encode_ngap_pdu(&pdu).expect("DL UE-associated transport encodes")
}

/// Wrap an NRPPa PDU into a downlink non-UE-associated NGAP transport (NGAP 7).
fn wrap_dl_non_ue_associated(nrppa_pdu: &nrppa::NRPPA_PDU, routing_id: &[u8]) -> Vec<u8> {
    use crate::codec::generated::{
        Criticality as NgapCriticality, DownlinkNonUEAssociatedNRPPaTransportProtocolIEs,
        DownlinkNonUEAssociatedNRPPaTransportProtocolIEs_Entry as Entry,
        DownlinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V,
        ProtocolIE_ID as NgapIeId,
    };

    let payload = encode_nrppa_pdu(nrppa_pdu).expect("NRPPa PDU encodes");
    let ies = vec![
        Entry {
            id: NgapIeId(89),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_RoutingID(RoutingID(routing_id.to_vec())),
        },
        Entry {
            id: NgapIeId(46),
            criticality: NgapCriticality(CRITICALITY_REJECT),
            value: V::Id_NRPPa_PDU(NgapNrppaPdu(payload)),
        },
    ];

    let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: NgapProcedureCode(NGAP_PROC_DL_NON_UE_NRPPA),
        criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
        value: InitiatingMessageValue::Id_DownlinkNonUEAssociatedNRPPaTransport(
            DownlinkNonUEAssociatedNRPPaTransport {
                protocol_i_es: DownlinkNonUEAssociatedNRPPaTransportProtocolIEs(ies),
            },
        ),
    });

    encode_ngap_pdu(&pdu).expect("DL non-UE-associated transport encodes")
}

// ---------------------------------------------------------------------------
// The wire format: the NRPPa header diverges from NGAP
// ---------------------------------------------------------------------------

#[test]
fn ecid_request_header_matches_hand_derived_x691_bytes() {
    let bytes = ecid_request_header_bytes();
    assert_eq!(
        &bytes[..ECID_REQUEST_HEADER_PREFIX.len()],
        &ECID_REQUEST_HEADER_PREFIX,
        "NRPPa header octets must match the hand-derived X.691 layout; \
         got {:02x?}",
        &bytes[..6.min(bytes.len())]
    );
}

#[test]
fn the_transaction_id_occupies_octets_three_and_four_of_the_header() {
    // Two requests differing only in transactionID must differ ONLY in the two
    // octets the field occupies. This pins the field's POSITION, not merely its
    // presence: if it were absent, or encoded elsewhere, one of these three
    // assertions would fail.
    let seven = encode_nrppa_pdu(&build_ecid_request(7, 3)).expect("encodes");
    let big = encode_nrppa_pdu(&build_ecid_request(0x3039, 3)).expect("encodes");

    assert_eq!(
        seven.len(),
        big.len(),
        "a fixed two-octet field cannot change the PDU length"
    );
    assert_eq!(
        &seven[..3],
        &big[..3],
        "the choice, procedureCode and criticality octets must not move"
    );
    assert_eq!(&seven[3..5], &[0x00, 0x07], "transactionID 7 as two octets");
    assert_eq!(
        &big[3..5],
        &[0x30, 0x39],
        "transactionID 0x3039 as two octets"
    );
    assert_eq!(
        &seven[5..],
        &big[5..],
        "everything after the transactionID must be byte-identical"
    );
}

#[test]
fn transaction_id_round_trips_through_the_decoder() {
    let bytes = encode_nrppa_pdu(&build_ecid_request(12345, 3)).expect("encodes");
    let decoded = decode_nrppa_pdu(&bytes).expect("decodes");

    let nrppa::NRPPA_PDU::InitiatingMessage(init) = decoded else {
        panic!("expected an InitiatingMessage");
    };
    assert_eq!(init.nrppatransaction_id.0, 12345);
    assert_eq!(init.procedure_code.0, PROC_E_CID_MEASUREMENT_INITIATION);
}

// ---------------------------------------------------------------------------
// E-CID Measurement Initiation, TS 38.455 §8.2.1
// ---------------------------------------------------------------------------

#[test]
fn ecid_request_over_dl_ue_transport_yields_a_response_with_the_serving_cell_and_rsrp() {
    let cell = test_cell();
    let dl = wrap_dl_ue_associated(&build_ecid_request(7, 3), 0x0102_0304, 0x0506);

    let request = parse_downlink_nrppa_transport(&dl).expect("DL NRPPa transport parses");
    assert_eq!(
        request.association,
        NrppaAssociation::UeAssociated {
            amf_ue_ngap_id: 0x0102_0304,
            ran_ue_ngap_id: 0x0506,
            routing_id: LMF_ROUTING_ID.to_vec(),
        }
    );
    assert_eq!(request.transaction_id, 7);
    assert_eq!(
        request.procedure,
        NrppaProcedure::ECidMeasurementInitiation {
            lmf_ue_measurement_id: 3
        }
    );

    let response = build_nrppa_response(&request, &cell, 9, 0).expect("response builds");
    let uplink = encode_uplink_nrppa_transport(&response).expect("uplink encodes");

    // The answer must be an UplinkUEAssociatedNRPPaTransport on the same UE.
    let ngap = decode_ngap_pdu(&uplink).expect("uplink decodes as NGAP");
    let NGAP_PDU::InitiatingMessage(init) = &ngap else {
        panic!("expected an NGAP InitiatingMessage");
    };
    assert_eq!(init.procedure_code.0, NGAP_PROC_UL_UE_NRPPA);

    let InitiatingMessageValue::Id_UplinkUEAssociatedNRPPaTransport(ul) = &init.value else {
        panic!("expected UplinkUEAssociatedNRPPaTransport");
    };

    use crate::codec::generated::UplinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;
    let mut amf_id = None;
    let mut ran_id = None;
    let mut routing_id = None;
    let mut payload = None;
    for ie in &ul.protocol_i_es.0 {
        match &ie.value {
            V::Id_AMF_UE_NGAP_ID(id) => amf_id = Some(id.0),
            V::Id_RAN_UE_NGAP_ID(id) => ran_id = Some(id.0),
            V::Id_RoutingID(r) => routing_id = Some(r.0.clone()),
            V::Id_NRPPa_PDU(p) => payload = Some(p.0.clone()),
        }
    }
    assert_eq!(
        amf_id,
        Some(0x0102_0304),
        "the answer must name the same UE"
    );
    assert_eq!(ran_id, Some(0x0506));
    // #374: TS 38.413 §9.2.9.2 makes the Routing ID mandatory (reject); amfd
    // rejected every answer without it.
    assert_eq!(
        routing_id.as_deref(),
        Some(LMF_ROUTING_ID),
        "the Routing ID is echoed"
    );

    // Now the positive assertion that matters: the measurements survive.
    let inner = decode_nrppa_pdu(&payload.expect("NRPPa-PDU IE present")).expect("NRPPa decodes");
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = inner else {
        panic!("§8.2.1.2 requires a successful outcome, not an error");
    };
    assert_eq!(outcome.procedure_code.0, PROC_E_CID_MEASUREMENT_INITIATION);
    assert_eq!(
        outcome.nrppatransaction_id.0, 7,
        "the transactionID must be echoed so the LMF can correlate"
    );

    let nrppa::SuccessfulOutcomeValue::Id_e_CIDMeasurementInitiation(resp) = &outcome.value else {
        panic!("expected an E-CIDMeasurementInitiationResponse");
    };

    use nrppa::E_CIDMeasurementInitiationResponseProtocolIEs_EntryValue as RV;
    let mut lmf_id = None;
    let mut ran_measurement_id = None;
    let mut result = None;
    for ie in &resp.protocol_i_es.0 {
        match &ie.value {
            RV::Id_LMF_UE_Measurement_ID(id) => lmf_id = Some(id.0),
            RV::Id_RAN_UE_Measurement_ID(id) => ran_measurement_id = Some(id.0),
            RV::Id_E_CID_MeasurementResult(r) => result = Some(r.clone()),
            _ => {}
        }
    }
    assert_eq!(lmf_id, Some(3), "the LMF's measurement id must be echoed");
    assert_eq!(ran_measurement_id, Some(9));

    let result = result.expect("the response must carry an E-CID-MeasurementResult");

    // Serving Cell-ID: PLMN and the 36-bit NCI.
    assert_eq!(result.serving_cell_id.plmn_identity.0, cell.plmn_identity);
    let nrppa::NG_RANCell::NR_CellID(nci) = &result.serving_cell_id.ng_ra_ncell else {
        panic!("a gNB must report an NR cell identity, not an E-UTRA one");
    };
    assert_eq!(nci.0.len(), 36, "NRCellIdentifier is BIT STRING (SIZE(36))");
    let decoded_nci = nci.0.iter().fold(0u64, |acc, b| (acc << 1) | u64::from(*b));
    assert_eq!(decoded_nci, cell.nr_cell_identity);
    assert_eq!(result.serving_cell_tac.0, cell.tac);

    // And at least one radio quantity: SS-RSRP, in the CHOICE extension arm.
    let measured = result
        .measured_results
        .expect("the response must carry a radio quantity");
    let nrppa::MeasuredResultsValue::Choice_Extension(ext) = &measured.0[0] else {
        panic!("SS-RSRP is an NR quantity and rides in the CHOICE extension arm");
    };
    assert_eq!(ext.id.0, 32, "id-ResultSS-RSRP");
    let nrppa::MeasuredResultsValue_choice_ExtensionValue::Id_ResultSS_RSRP(rsrp) = &ext.value
    else {
        panic!("expected a ResultSS-RSRP");
    };
    assert_eq!(rsrp.0[0].nr_pci.0, cell.nr_pci);
    assert_eq!(rsrp.0[0].nr_arfcn.0, cell.nr_arfcn);
    assert_eq!(
        rsrp.0[0].value_ss_rsrp_cell.as_ref().map(|v| v.0),
        Some(cell.ss_rsrp),
        "the reported RSRP must be the measured value"
    );
}

// ---------------------------------------------------------------------------
// Positioning Information Exchange, TS 38.455 §8.2.6
// ---------------------------------------------------------------------------

#[test]
fn positioning_information_request_yields_a_response_carrying_the_sfn_timing_reference() {
    let cell = test_cell();
    let dl = wrap_dl_ue_associated(&build_positioning_information_request(11), 7, 8);

    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.procedure,
        NrppaProcedure::PositioningInformationExchange {
            requested_srs: None
        }
    );

    let sfn_time = 0x0123_4567_89ab_cdef;
    let response = build_nrppa_response(&request, &cell, 1, sfn_time).expect("response builds");
    let inner = decode_nrppa_pdu(&response.nrppa_pdu).expect("NRPPa decodes");

    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = inner else {
        panic!("§8.2.6 requires a PositioningInformationResponse");
    };
    assert_eq!(
        outcome.procedure_code.0,
        PROC_POSITIONING_INFORMATION_EXCHANGE
    );
    assert_eq!(outcome.nrppatransaction_id.0, 11);

    let nrppa::SuccessfulOutcomeValue::Id_positioningInformationExchange(resp) = &outcome.value
    else {
        panic!("expected a PositioningInformationResponse");
    };

    use nrppa::PositioningInformationResponseProtocolIEs_EntryValue as V;
    let V::Id_SFNInitialisationTime(time) = &resp.protocol_i_es.0[0].value else {
        panic!("the response must carry id-SFNInitialisationTime");
    };
    assert_eq!(
        time.0.len(),
        64,
        "RelativeTime1900 is BIT STRING (SIZE(64))"
    );
    let decoded = time
        .0
        .iter()
        .fold(0u64, |acc, b| (acc << 1) | u64::from(*b));
    assert_eq!(
        decoded, sfn_time,
        "the SFN initialisation time must survive the round trip intact"
    );
}

// ---------------------------------------------------------------------------
// TRP Information Exchange, TS 38.455 §8.2.8
// ---------------------------------------------------------------------------

#[test]
fn trp_information_request_over_non_ue_transport_returns_pci_cgi_and_arfcn_per_trp() {
    let cell = test_cell();
    let routing_id = vec![0xAA, 0xBB];
    // Types 0/1/2 = nR-PCI, nG-RAN-CGI, aRFCN (TS 38.455 §9.2.24).
    let dl = wrap_dl_non_ue_associated(&build_trp_request(21, None, &[0, 1, 2]), &routing_id);

    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.association,
        NrppaAssociation::NonUeAssociated {
            routing_id: routing_id.clone()
        },
        "TRP information names no UE, so it must stay non-UE-associated"
    );
    assert_eq!(
        request.procedure,
        NrppaProcedure::TrpInformationExchange {
            requested_trps: None,
            requested_types: vec![0, 1, 2],
        }
    );

    let response = build_nrppa_response(&request, &cell, 1, 0).expect("response builds");
    let uplink = encode_uplink_nrppa_transport(&response).expect("uplink encodes");

    let ngap = decode_ngap_pdu(&uplink).expect("decodes");
    let NGAP_PDU::InitiatingMessage(init) = &ngap else {
        panic!("expected an InitiatingMessage");
    };
    assert_eq!(
        init.procedure_code.0, NGAP_PROC_UL_NON_UE_NRPPA,
        "the answer must go back on the non-UE-associated transport"
    );

    let InitiatingMessageValue::Id_UplinkNonUEAssociatedNRPPaTransport(ul) = &init.value else {
        panic!("expected UplinkNonUEAssociatedNRPPaTransport");
    };

    use crate::codec::generated::UplinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V;
    let mut echoed_routing = None;
    let mut payload = None;
    for ie in &ul.protocol_i_es.0 {
        match &ie.value {
            V::Id_RoutingID(r) => echoed_routing = Some(r.0.clone()),
            V::Id_NRPPa_PDU(p) => payload = Some(p.0.clone()),
        }
    }
    assert_eq!(
        echoed_routing,
        Some(routing_id),
        "the RoutingID must be echoed so the answer reaches the originating LMF"
    );

    let inner = decode_nrppa_pdu(&payload.expect("payload present")).expect("NRPPa decodes");
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = inner else {
        panic!("§8.2.8 requires a TRPInformationResponse");
    };
    assert_eq!(outcome.procedure_code.0, PROC_TRP_INFORMATION_EXCHANGE);
    assert_eq!(outcome.nrppatransaction_id.0, 21);

    let nrppa::SuccessfulOutcomeValue::Id_tRPInformationExchange(resp) = &outcome.value else {
        panic!("expected a TRPInformationResponse");
    };
    use nrppa::TRPInformationResponseProtocolIEs_EntryValue as RV;
    let RV::Id_TRPInformationListTRPResp(list) = &resp.protocol_i_es.0[0].value else {
        panic!("the response must carry id-TRPInformationListTRPResp");
    };

    assert_eq!(list.0.len(), 2, "both of this node's TRPs must be reported");
    let reported: Vec<u16> = list.0.iter().map(|e| e.trp_information.trp_id.0).collect();
    assert_eq!(reported, cell.trp_ids);

    // Each TRP answers the three types that were asked for.
    for entry in &list.0 {
        let items = &entry.trp_information.trp_information_type_response_list.0;
        assert_eq!(items.len(), 3, "three requested types, three answers");

        let nrppa::TRPInformationTypeResponseItem::PCI_NR(pci) = &items[0] else {
            panic!("expected nR-PCI first");
        };
        assert_eq!(pci.0, cell.nr_pci);

        let nrppa::TRPInformationTypeResponseItem::CGI_NR(cgi) = &items[1] else {
            panic!("expected nG-RAN-CGI second");
        };
        assert_eq!(cgi.plmn_identity.0, cell.plmn_identity);
        let nci = cgi
            .n_rcell_identifier
            .0
            .iter()
            .fold(0u64, |acc, b| (acc << 1) | u64::from(*b));
        assert_eq!(nci, cell.nr_cell_identity);

        let nrppa::TRPInformationTypeResponseItem::ARFCN(arfcn) = &items[2] else {
            panic!("expected aRFCN third");
        };
        assert_eq!(arfcn.0, cell.nr_arfcn);
    }
}

#[test]
fn a_trp_list_filters_the_answer_to_the_trps_the_lmf_named() {
    let cell = test_cell();
    // Ask about TRP 2 (which the node has) and TRP 99 (which it does not).
    let dl = wrap_dl_non_ue_associated(&build_trp_request(3, Some(vec![2, 99]), &[0]), &[0x01]);

    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.procedure,
        NrppaProcedure::TrpInformationExchange {
            requested_trps: Some(vec![2, 99]),
            requested_types: vec![0],
        }
    );

    let response = build_nrppa_response(&request, &cell, 1, 0).expect("builds");
    let inner = decode_nrppa_pdu(&response.nrppa_pdu).expect("decodes");
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = inner else {
        panic!("expected a successful outcome");
    };
    let nrppa::SuccessfulOutcomeValue::Id_tRPInformationExchange(resp) = &outcome.value else {
        panic!("expected a TRPInformationResponse");
    };
    use nrppa::TRPInformationResponseProtocolIEs_EntryValue as RV;
    let RV::Id_TRPInformationListTRPResp(list) = &resp.protocol_i_es.0[0].value else {
        panic!("expected the TRP list");
    };

    let reported: Vec<u16> = list.0.iter().map(|e| e.trp_information.trp_id.0).collect();
    assert_eq!(
        reported,
        vec![2],
        "a TRP the node does not have must not be fabricated into the answer"
    );
}

#[test]
fn an_unrequested_information_type_is_omitted_rather_than_invented() {
    let cell = test_cell();
    // Type 5 is sFNInitTime, which this node does not model.
    let dl = wrap_dl_non_ue_associated(&build_trp_request(4, Some(vec![1]), &[0, 5]), &[0x01]);

    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    let response = build_nrppa_response(&request, &cell, 1, 0).expect("builds");
    let inner = decode_nrppa_pdu(&response.nrppa_pdu).expect("decodes");
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = inner else {
        panic!("expected a successful outcome");
    };
    let nrppa::SuccessfulOutcomeValue::Id_tRPInformationExchange(resp) = &outcome.value else {
        panic!("expected a TRPInformationResponse");
    };
    use nrppa::TRPInformationResponseProtocolIEs_EntryValue as RV;
    let RV::Id_TRPInformationListTRPResp(list) = &resp.protocol_i_es.0[0].value else {
        panic!("expected the TRP list");
    };

    let items = &list.0[0]
        .trp_information
        .trp_information_type_response_list
        .0;
    assert_eq!(
        items.len(),
        1,
        "only the type the node can answer should appear"
    );
    assert!(matches!(
        items[0],
        nrppa::TRPInformationTypeResponseItem::PCI_NR(_)
    ));
}

// ---------------------------------------------------------------------------
// Dispatch discrimination: the parser must not claim unrelated PDUs
// ---------------------------------------------------------------------------

#[test]
fn a_non_nrppa_ngap_pdu_is_not_claimed_as_an_nrppa_transport() {
    // A Paging PDU must fall through, or adding NRPPa to the dispatch chain
    // would steal PDUs from the handlers that own them.
    use crate::codec::generated::{Paging, PagingProtocolIEs};

    let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: NgapProcedureCode(24),
        criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
        value: InitiatingMessageValue::Id_Paging(Paging {
            protocol_i_es: PagingProtocolIEs(vec![]),
        }),
    });
    let bytes = encode_ngap_pdu(&pdu).expect("encodes");

    assert!(matches!(
        parse_downlink_nrppa_transport(&bytes),
        Err(NrppaError::NotNrppaTransport)
    ));
}

#[test]
fn an_nrppa_procedure_this_gnb_does_not_implement_is_reported_as_unsupported() {
    // Procedure 6, OTDOA Information Exchange, is a real NRPPa procedure that
    // this gNB does not answer. It must be reported as unsupported rather than
    // silently mis-parsed as one of the three that are implemented.
    let pdu = nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(6),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(1),
        value: nrppa::InitiatingMessageValue::Id_oTDOAInformationExchange(
            nrppa::OTDOAInformationRequest {
                protocol_i_es: nrppa::OTDOAInformationRequestProtocolIEs(vec![]),
            },
        ),
    });
    let dl = wrap_dl_ue_associated(&pdu, 1, 2);

    assert!(matches!(
        parse_downlink_nrppa_transport(&dl),
        Err(NrppaError::UnsupportedProcedure(6))
    ));
}

#[test]
fn a_downlink_transport_missing_the_nrppa_payload_is_rejected() {
    use crate::codec::generated::{
        Criticality as NgapCriticality, DownlinkNonUEAssociatedNRPPaTransportProtocolIEs,
        DownlinkNonUEAssociatedNRPPaTransportProtocolIEs_Entry as Entry,
        DownlinkNonUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V,
        ProtocolIE_ID as NgapIeId,
    };

    let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: NgapProcedureCode(NGAP_PROC_DL_NON_UE_NRPPA),
        criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
        value: InitiatingMessageValue::Id_DownlinkNonUEAssociatedNRPPaTransport(
            DownlinkNonUEAssociatedNRPPaTransport {
                protocol_i_es: DownlinkNonUEAssociatedNRPPaTransportProtocolIEs(vec![Entry {
                    id: NgapIeId(89),
                    criticality: NgapCriticality(CRITICALITY_REJECT),
                    value: V::Id_RoutingID(RoutingID(vec![0x01])),
                }]),
            },
        ),
    });
    let bytes = encode_ngap_pdu(&pdu).expect("encodes");

    assert!(matches!(
        parse_downlink_nrppa_transport(&bytes),
        Err(NrppaError::MissingMandatoryIe("NRPPa-PDU"))
    ));
}

// ---------------------------------------------------------------------------
// Multi-RTT, nextgsim #347: TS 38.455 §8.2.6 (SRS), §8.2.9, §8.2.10, §8.5.1
// ---------------------------------------------------------------------------

fn hex(s: &str) -> Vec<u8> {
    s.split_whitespace()
        .map(|b| u8::from_str_radix(b, 16).expect("hex"))
        .collect()
}

fn decode_hex_nrppa(s: &str) -> nrppa::NRPPA_PDU {
    decode_nrppa_pdu(&hex(s)).expect("lmfd's vector decodes")
}

fn test_srs() -> SrsAllocation {
    SrsAllocation {
        resource_set_id: 3,
        resource_type: SrsResourceType::SemiPersistent,
        sequence_id: 77,
        pci: 42,
        point_a: srs_point_a(632_628),
    }
}

/// nextgcore lmfd's POSITIONING INFORMATION REQUEST (transaction 1,
/// `nrppa_downlink.rs`): semi-persistent SRS, 20 MHz.
const LMFD_POSITIONING_INFORMATION_REQUEST: &str = "00 09 00 00 01 09 00 00 01 00 0c 40 02 01 08";
/// nextgcore lmfd's POSITIONING ACTIVATION REQUEST: semi-persistent set 3,
/// transaction 0.
const LMFD_POSITIONING_ACTIVATION_REQUEST: &str = "00 11 00 00 00 09 00 00 01 00 2c 00 02 01 80";

#[test]
fn an_srs_request_asks_for_the_srs_configuration() {
    let dl = wrap_dl_ue_associated(
        &decode_hex_nrppa(LMFD_POSITIONING_INFORMATION_REQUEST),
        7,
        8,
    );
    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.procedure,
        NrppaProcedure::PositioningInformationExchange {
            requested_srs: Some(SrsResourceType::SemiPersistent)
        }
    );
    // §8.2.6.2: the SRS Configuration and the SFN Initialisation Time.
    let answer = build_nrppa_answer(
        &request,
        &test_cell(),
        1,
        5,
        &NrppaDecision::PositioningInformation(Ok(test_srs())),
    )
    .expect("builds")
    .expect("answered");
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = decode_nrppa_pdu(&answer.nrppa_pdu).unwrap()
    else {
        panic!("a PositioningInformationResponse");
    };
    let nrppa::SuccessfulOutcomeValue::Id_positioningInformationExchange(resp) = outcome.value
    else {
        panic!("a PositioningInformationResponse");
    };
    use nrppa::PositioningInformationResponseProtocolIEs_EntryValue as V;
    let config = resp
        .protocol_i_es
        .0
        .iter()
        .find_map(|ie| match &ie.value {
            V::Id_SRSConfiguration(c) => Some(c.clone()),
            _ => None,
        })
        .expect("the SRS Configuration IE");
    assert!(resp
        .protocol_i_es
        .0
        .iter()
        .any(|ie| matches!(ie.value, V::Id_SFNInitialisationTime(_))));
    // What lmfd reads: the first semi-persistent positioning set's id.
    let carrier = &config.srs_carrier_list.0[0];
    let set = &carrier
        .active_ulbwp
        .srs_config
        .pos_srs_resource_set_list
        .as_ref()
        .expect("a positioning SRS resource set")
        .0[0];
    assert_eq!(set.possrs_resource_set_id.0, 3);
    assert!(matches!(
        set.posresource_set_type,
        nrppa::PosResourceSetType::Semi_persistent(_)
    ));
    assert_eq!(srs_configuration_sets(&config), vec![(3, Some(42))]);
}

fn srs_configuration_sets(config: &nrppa::SRSConfiguration) -> Vec<(u8, Option<u16>)> {
    multi_rtt::srs_configuration_resource_sets(config)
}

#[test]
fn an_srs_request_this_node_cannot_serve_is_a_positioning_information_failure() {
    let dl = wrap_dl_ue_associated(
        &decode_hex_nrppa(LMFD_POSITIONING_INFORMATION_REQUEST),
        7,
        8,
    );
    let request = parse_downlink_nrppa_transport(&dl).unwrap();
    let answer = build_nrppa_answer(
        &request,
        &test_cell(),
        1,
        0,
        &NrppaDecision::PositioningInformation(Err(
            NrppaRadioCause::RequestedItemTemporarilyNotAvailable,
        )),
    )
    .unwrap()
    .unwrap();
    let nrppa::NRPPA_PDU::UnsuccessfulOutcome(outcome) =
        decode_nrppa_pdu(&answer.nrppa_pdu).unwrap()
    else {
        panic!("a PositioningInformationFailure");
    };
    assert_eq!(
        outcome.procedure_code.0,
        PROC_POSITIONING_INFORMATION_EXCHANGE
    );
    assert_eq!(outcome.nrppatransaction_id.0, 1);
    // An SRS request answered from the cell alone would drop the SRS: refused.
    assert!(build_nrppa_response(&request, &test_cell(), 1, 0).is_err());
}

#[test]
fn a_positioning_activation_is_parsed_and_answered_with_its_activation_time() {
    let dl = wrap_dl_ue_associated(&decode_hex_nrppa(LMFD_POSITIONING_ACTIVATION_REQUEST), 7, 8);
    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.procedure,
        NrppaProcedure::PositioningActivation(SrsActivation::SemiPersistent { resource_set_id: 3 })
    );
    let at = CellTime { sfn: 513, slot: 17 };
    let answer = build_nrppa_answer(
        &request,
        &test_cell(),
        1,
        0,
        &NrppaDecision::PositioningActivation(Ok(at)),
    )
    .unwrap()
    .unwrap();
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = decode_nrppa_pdu(&answer.nrppa_pdu).unwrap()
    else {
        panic!("a PositioningActivationResponse");
    };
    assert_eq!(
        outcome.procedure_code.0,
        multi_rtt::PROC_POSITIONING_ACTIVATION
    );
    let nrppa::SuccessfulOutcomeValue::Id_positioningActivation(resp) = outcome.value else {
        panic!("a PositioningActivationResponse");
    };
    use nrppa::PositioningActivationResponseProtocolIEs_EntryValue as V;
    let mut seen = (None, None);
    for ie in resp.protocol_i_es.0 {
        match ie.value {
            V::Id_SystemFrameNumber(sfn) => seen.0 = Some(sfn.0),
            V::Id_SlotNumber(slot) => seen.1 = Some(slot.0),
            V::Id_CriticalityDiagnostics(_) => {}
        }
    }
    assert_eq!(seen, (Some(513), Some(17)));

    let failure = build_nrppa_answer(
        &request,
        &test_cell(),
        1,
        0,
        &NrppaDecision::PositioningActivation(Err(NrppaRadioCause::RequestedItemNotSupported)),
    )
    .unwrap()
    .unwrap();
    assert!(matches!(
        decode_nrppa_pdu(&failure.nrppa_pdu).unwrap(),
        nrppa::NRPPA_PDU::UnsuccessfulOutcome(_)
    ));
}

fn positioning_deactivation(abort: nrppa::AbortTransmission) -> nrppa::NRPPA_PDU {
    nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(multi_rtt::PROC_POSITIONING_DEACTIVATION),
        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
        nrppatransaction_id: nrppa::NRPPATransactionID(4),
        value: nrppa::InitiatingMessageValue::Id_positioningDeactivation(nrppa::PositioningDeactivation {
            protocol_i_es: nrppa::PositioningDeactivationProtocolIEs(vec![
                nrppa::PositioningDeactivationProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(53),
                    criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                    value: nrppa::PositioningDeactivationProtocolIEs_EntryValue::Id_AbortTransmission(abort),
                },
            ]),
        }),
    })
}

#[test]
fn a_positioning_deactivation_is_parsed_and_not_answered() {
    let dl = wrap_dl_ue_associated(
        &positioning_deactivation(nrppa::AbortTransmission::DeactivateSRSResourceSetID(
            nrppa::SRSResourceSetID(3),
        )),
        7,
        8,
    );
    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    assert_eq!(
        request.procedure,
        NrppaProcedure::PositioningDeactivation(SrsDeactivation::ResourceSet(3))
    );
    assert_eq!(
        build_nrppa_answer(
            &request,
            &test_cell(),
            1,
            0,
            &NrppaDecision::PositioningDeactivation
        )
        .unwrap(),
        None,
        "§8.2.10 is class 2"
    );
    let dl = wrap_dl_ue_associated(
        &positioning_deactivation(nrppa::AbortTransmission::ReleaseALL(
            nrppa::AbortTransmission_releaseALL,
        )),
        7,
        8,
    );
    assert_eq!(
        parse_downlink_nrppa_transport(&dl).unwrap().procedure,
        NrppaProcedure::PositioningDeactivation(SrsDeactivation::ReleaseAll)
    );
}

/// A MEASUREMENT REQUEST shaped as lmfd sends it (`nrppa_downlink.rs`): two
/// TRPs, onDemand, gNB-RxTxTimeDiff, the gNB's SRS Configuration.
fn lmfd_measurement_request(lmf_measurement_id: u32, srs: &SrsAllocation) -> nrppa::NRPPA_PDU {
    use nrppa::MeasurementRequestProtocolIEs_EntryValue as V;
    let ie = |id: u16, criticality: u8, value: V| nrppa::MeasurementRequestProtocolIEs_Entry {
        id: nrppa::ProtocolIE_ID(id),
        criticality: nrppa::Criticality(criticality),
        value,
    };
    nrppa::NRPPA_PDU::InitiatingMessage(nrppa::InitiatingMessage {
        procedure_code: nrppa::ProcedureCode(multi_rtt::PROC_MEASUREMENT),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(21),
        value: nrppa::InitiatingMessageValue::Id_Measurement(nrppa::MeasurementRequest {
            protocol_i_es: nrppa::MeasurementRequestProtocolIEs(vec![
                ie(
                    39,
                    CRITICALITY_REJECT,
                    V::Id_LMF_Measurement_ID(nrppa::Measurement_ID(lmf_measurement_id)),
                ),
                ie(
                    41,
                    CRITICALITY_REJECT,
                    V::Id_TRP_MeasurementRequestList(nrppa::TRP_MeasurementRequestList(
                        [1u16, 2]
                            .iter()
                            .map(|id| nrppa::TRP_MeasurementRequestItem {
                                trp_id: nrppa::TRP_ID(*id),
                                search_window_information: None,
                                ie_extensions: None,
                            })
                            .collect(),
                    )),
                ),
                ie(
                    3,
                    CRITICALITY_REJECT,
                    V::Id_ReportCharacteristics(nrppa::ReportCharacteristics(0)),
                ),
                ie(
                    52,
                    CRITICALITY_REJECT,
                    V::Id_TRPMeasurementQuantities(nrppa::TRPMeasurementQuantities(vec![
                        nrppa::TRPMeasurementQuantitiesList_Item {
                            trp_measurement_type: nrppa::TRPMeasurementType(0),
                            timing_reporting_granularity_factor: None,
                            ie_extensions: None,
                        },
                    ])),
                ),
                ie(
                    26,
                    CRITICALITY_IGNORE,
                    V::Id_SRSConfiguration(build_srs_configuration(srs)),
                ),
            ]),
        }),
    })
}

/// lmfd's decoding of a k0 report (`codec_glue.rs`): `(v - 985024) * T_c`.
fn lmfd_rx_tx_seconds(v: u32) -> f64 {
    (f64::from(v) - 985_024.0) * multi_rtt::T_C_SECONDS
}

#[test]
fn a_measurement_request_is_answered_with_a_gnb_rx_tx_per_trp() {
    let dl = wrap_dl_non_ue_associated(&lmfd_measurement_request(9, &test_srs()), &[0x00, 0x01]);
    let request = parse_downlink_nrppa_transport(&dl).expect("parses");
    let NrppaProcedure::Measurement(req) = &request.procedure else {
        panic!("a Measurement");
    };
    assert_eq!(
        *req,
        TrpMeasurementRequest {
            lmf_measurement_id: 9,
            trp_ids: vec![1, 2],
            on_demand: true,
            quantities: vec![TRP_MEASUREMENT_GNB_RX_TX_TIME_DIFF],
            srs_resource_sets: vec![(3, Some(42))],
        }
    );

    // 100 m: a 667 ns round trip.
    let rtt = 2.0 * 100.0 / 299_792_458.0;
    let at = CellTime { sfn: 7, slot: 3 };
    let measurements = [1u16, 2].map(|trp_id| TrpRxTxMeasurement {
        trp_id,
        rx_tx_seconds: rtt,
        time: at,
    });
    let answer = build_nrppa_answer(
        &request,
        &test_cell(),
        1,
        0,
        &NrppaDecision::Measurement(Ok((44, measurements.to_vec()))),
    )
    .unwrap()
    .unwrap();
    assert_eq!(
        answer.association, request.association,
        "non-UE-associated, same routing id"
    );
    let nrppa::NRPPA_PDU::SuccessfulOutcome(outcome) = decode_nrppa_pdu(&answer.nrppa_pdu).unwrap()
    else {
        panic!("a MeasurementResponse");
    };
    assert_eq!(outcome.procedure_code.0, multi_rtt::PROC_MEASUREMENT);
    assert_eq!(outcome.nrppatransaction_id.0, 21);
    let nrppa::SuccessfulOutcomeValue::Id_Measurement(resp) = outcome.value else {
        panic!("a MeasurementResponse");
    };
    use nrppa::MeasurementResponseProtocolIEs_EntryValue as V;
    let (mut lmf, mut ran, mut list) = (None, None, None);
    for ie in resp.protocol_i_es.0 {
        match ie.value {
            V::Id_LMF_Measurement_ID(id) => lmf = Some(id.0),
            V::Id_RAN_Measurement_ID(id) => ran = Some(id.0),
            V::Id_TRP_MeasurementResponseList(l) => list = Some(l),
            V::Id_CriticalityDiagnostics(_) => {}
        }
    }
    assert_eq!((lmf, ran), (Some(9), Some(44)));
    let list = list.expect("the TRP Measurement Response List");
    assert_eq!(
        list.0.iter().map(|i| i.trp_id.0).collect::<Vec<_>>(),
        vec![1, 2]
    );
    for item in &list.0 {
        let result = &item.measurement_result.0[0];
        let nrppa::TrpMeasuredResultsValue::GNB_RxTxTimeDiff(rx_tx) =
            &result.measured_results_value
        else {
            panic!("a gNB Rx-Tx time difference");
        };
        let nrppa::GNBRxTxTimeDiffMeas::K0(v) = &rx_tx.rx_tx_time_diff else {
            panic!("the k0 report");
        };
        let decoded = lmfd_rx_tx_seconds(v.0);
        assert!(
            (decoded - rtt).abs() <= multi_rtt::T_C_SECONDS,
            "{decoded} vs {rtt}"
        );
        assert_eq!(result.time_stamp.system_frame_number.0, 7);
        assert!(matches!(
            result.time_stamp.slot_index,
            nrppa::TimeStampSlotIndex::SCS_30(nrppa::TimeStampSlotIndex_sCS_30(3))
        ));
    }
}

#[test]
fn a_measurement_nothing_was_measured_for_is_a_measurement_failure() {
    let dl = wrap_dl_non_ue_associated(&lmfd_measurement_request(9, &test_srs()), &[0x00, 0x01]);
    let request = parse_downlink_nrppa_transport(&dl).unwrap();
    for decision in [
        NrppaDecision::Measurement(Err(NrppaRadioCause::RequestedItemTemporarilyNotAvailable)),
        NrppaDecision::Measurement(Ok((44, vec![]))),
    ] {
        let answer = build_nrppa_answer(&request, &test_cell(), 1, 0, &decision)
            .unwrap()
            .unwrap();
        let nrppa::NRPPA_PDU::UnsuccessfulOutcome(outcome) =
            decode_nrppa_pdu(&answer.nrppa_pdu).unwrap()
        else {
            panic!("§8.5.1.3: a MeasurementFailure");
        };
        let nrppa::UnsuccessfulOutcomeValue::Id_Measurement(failure) = outcome.value else {
            panic!("a MeasurementFailure");
        };
        assert!(failure.protocol_i_es.0.iter().any(|ie| matches!(
            ie.value,
            nrppa::MeasurementFailureProtocolIEs_EntryValue::Id_LMF_Measurement_ID(
                nrppa::Measurement_ID(9)
            )
        )));
    }
}

#[test]
fn the_k0_report_maps_zero_to_its_midpoint_and_stays_off_the_open_ends() {
    assert_eq!(gnb_rx_tx_k0_report(0.0), 985_024);
    assert_eq!(gnb_rx_tx_k0_report(multi_rtt::T_C_SECONDS), 985_025);
    assert_eq!(gnb_rx_tx_k0_report(1.0), 1_970_048);
    assert_eq!(gnb_rx_tx_k0_report(-1.0), 1);
}

#[test]
fn cell_time_counts_frames_and_30_khz_slots_from_sfn_0() {
    assert_eq!(
        CellTime::from_micros_since_sfn0(0),
        CellTime { sfn: 0, slot: 0 }
    );
    assert_eq!(
        CellTime::from_micros_since_sfn0(9_999),
        CellTime { sfn: 0, slot: 19 }
    );
    assert_eq!(
        CellTime::from_micros_since_sfn0(10_000),
        CellTime { sfn: 1, slot: 0 }
    );
    assert_eq!(
        CellTime::from_micros_since_sfn0(10_240_000 + 25_600),
        CellTime { sfn: 2, slot: 11 },
        "SFN wraps every 1024 frames"
    );
}

#[test]
fn point_a_is_half_the_carrier_below_its_centre() {
    // 51 PRBs * 12 * 30 kHz / 2 = 9180 kHz: 612 steps of 15 kHz at 3.5 GHz.
    assert_eq!(srs_point_a(632_628), 632_628 - 612);
    // 1836 steps of 5 kHz below 3 GHz.
    assert_eq!(srs_point_a(400_000), 400_000 - 1836);
}

/// #374: the Routing ID is mandatory in the downlink UE-associated transport
/// (TS 38.413 §9.2.9.1), and the answer must echo it, so a transport without
/// one is refused.
#[test]
fn a_ue_associated_transport_without_a_routing_id_is_refused() {
    use crate::codec::generated::{
        Criticality as NgapCriticality, DownlinkUEAssociatedNRPPaTransportProtocolIEs,
        DownlinkUEAssociatedNRPPaTransportProtocolIEs_Entry as Entry,
        DownlinkUEAssociatedNRPPaTransportProtocolIEs_EntryValue as V, ProtocolIE_ID as NgapIeId,
        AMF_UE_NGAP_ID, RAN_UE_NGAP_ID,
    };
    let payload = encode_nrppa_pdu(&build_ecid_request(7, 3)).unwrap();
    let pdu = NGAP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: NgapProcedureCode(NGAP_PROC_DL_UE_NRPPA),
        criticality: crate::codec::generated::Criticality(CRITICALITY_IGNORE),
        value: InitiatingMessageValue::Id_DownlinkUEAssociatedNRPPaTransport(
            DownlinkUEAssociatedNRPPaTransport {
                protocol_i_es: DownlinkUEAssociatedNRPPaTransportProtocolIEs(vec![
                    Entry {
                        id: NgapIeId(10),
                        criticality: NgapCriticality(CRITICALITY_REJECT),
                        value: V::Id_AMF_UE_NGAP_ID(AMF_UE_NGAP_ID(1)),
                    },
                    Entry {
                        id: NgapIeId(85),
                        criticality: NgapCriticality(CRITICALITY_REJECT),
                        value: V::Id_RAN_UE_NGAP_ID(RAN_UE_NGAP_ID(2)),
                    },
                    Entry {
                        id: NgapIeId(46),
                        criticality: NgapCriticality(CRITICALITY_REJECT),
                        value: V::Id_NRPPa_PDU(NgapNrppaPdu(payload)),
                    },
                ]),
            },
        ),
    });
    let dl = encode_ngap_pdu(&pdu).unwrap();
    assert!(matches!(
        parse_downlink_nrppa_transport(&dl),
        Err(NrppaError::MissingMandatoryIe("RoutingID"))
    ));
}

/// #376 review: each `resourceType` of `RequestedSRSTransmissionCharacteristics`
/// (`ENUMERATED {periodic, semi-persistent, aperiodic, ...}`, TS 38.455) maps to
/// its own SRS resource type.
#[test]
fn each_requested_srs_resource_type_is_parsed_as_itself() {
    let mut pdu = decode_hex_nrppa(LMFD_POSITIONING_INFORMATION_REQUEST);
    for (value, expected) in [
        (0, SrsResourceType::Periodic),
        (1, SrsResourceType::SemiPersistent),
        (2, SrsResourceType::Aperiodic),
    ] {
        if let nrppa::NRPPA_PDU::InitiatingMessage(init) = &mut pdu {
            if let nrppa::InitiatingMessageValue::Id_positioningInformationExchange(req) =
                &mut init.value
            {
                for ie in &mut req.protocol_i_es.0 {
                    if let nrppa::PositioningInformationRequestProtocolIEs_EntryValue::Id_RequestedSRSTransmissionCharacteristics(chars) = &mut ie.value {
                        chars.resource_type =
                            nrppa::RequestedSRSTransmissionCharacteristicsResourceType(value);
                    }
                }
            }
        }
        let request = parse_downlink_nrppa_transport(&wrap_dl_ue_associated(&pdu, 7, 8)).unwrap();
        assert_eq!(
            request.procedure,
            NrppaProcedure::PositioningInformationExchange {
                requested_srs: Some(expected)
            },
            "resourceType {value}"
        );
    }
}
