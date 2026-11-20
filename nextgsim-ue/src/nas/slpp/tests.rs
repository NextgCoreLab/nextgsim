//! SLPP endpoint tests (nextgcore #557).
//!
//! The LMF-side inputs are nextgcore's own hand-derived goldens, copied byte
//! for byte from `nextgcore-slpp-gen/src/tests.rs` (`REQUEST_CAPABILITIES`,
//! `REQUEST_LOCATION_INFORMATION`). nextgcore's lmfd builds exactly these
//! octets, so these tests are the per-PR half of the cross-repo interop check.
//! The UE's replies are pinned to octets derived here by hand. lmfd decodes
//! those same octets in its own tests, which pins the reverse direction.
//!
//! What this does not prove: that an AMF relays type-12 containers between
//! the two. That is the nightly cross-repo E2E's job.

use super::*;
use nextgsim_nas::messages::mm::DlNasTransport;

/// nextgcore-slpp-gen `REQUEST_CAPABILITIES`: SL-RTT, transaction 1.
const LMF_REQUEST_CAPABILITIES: [u8; 6] = [0x44, 0x02, 0x00, 0x40, 0x04, 0x00];

/// nextgcore-slpp-gen `REQUEST_LOCATION_INFORMATION`: transaction 2,
/// rangeMeasurementsRequired, SL-RTT with sl-PRS-RSRP-Request.
const LMF_REQUEST_LOCATION_INFORMATION: [u8; 10] =
    [0x44, 0x04, 0x22, 0x40, 0x08, 0x00, 0xA0, 0x08, 0x40, 0x00];

const OWN_ID: [u8; 3] = [0xA5, 0xA5, 0xA5];
const PEER_ID: [u8; 3] = [0x5A, 0x5A, 0x5A];

/// SL-RTT-ProvideCapabilities as this UE sends it: ext 0 | opts `00000` |
/// positioningModes length-1 `001`, bits `01` | band list count-1 in 10 bits
/// `0000000000` | band ext 0, opts `100`, n1 `00`. That is 27 bits.
const UE_SL_RTT_CAPABILITIES: [u8; 4] = [0x00, 0xA0, 0x02, 0x00];

/// The UE's ProvideCapabilities to [`LMF_REQUEST_CAPABILITIES`], its first
/// message (sequence number 0):
/// opts `1110010` (applicationLayerID, transactionID, sequenceNumber, body) |
/// len 3 + A5 A5 A5 | tid `00000001` | end `1` | seq `00000000` | c1 `0`,
/// index 1 `0001` | critExt `0` | IEs opts `00010000` | len 4 +
/// [`UE_SL_RTT_CAPABILITIES`]. That is 110 bits, plus 2 pad bits.
const UE_PROVIDE_CAPABILITIES: [u8; 14] = [
    0xE4, 0x07, 0x4B, 0x4B, 0x4A, 0x03, 0x00, 0x08, 0x40, 0x10, 0x02, 0x80, 0x08, 0x00,
];

/// The SL-RTT report against the peer at 50 m: ext 0 | opts `10` | ext 0 |
/// MeasList count-1 `00000000` | per-ARP count-1 `00` | element ext 0, opts
/// `100`, len 3 + 5A 5A 5A | CHOICE ext 0, single `0` | result ext 0, opts
/// `01000000` (Rx-Tx only, no RSRP) | k2 `010` | 246338 in 19 bits. That is
/// 83 bits.
const UE_SL_RTT_REPORT: [u8; 11] = [
    0x40, 0x01, 0x00, 0xD6, 0x96, 0x96, 0x82, 0x02, 0x78, 0x48, 0x40,
];

/// The UE's ProvideLocationInformation to [`LMF_REQUEST_LOCATION_INFORMATION`],
/// its second message (sequence number 1): opts `1110010` | len 3 + A5 A5 A5 |
/// tid `00000010` | end `1` | seq `00000001` | c1 `0`, index 5 `0101` |
/// critExt `0` | IEs opts `00010000` | len 11 + [`UE_SL_RTT_REPORT`]. That is
/// 166 bits, plus 2 pad bits.
const UE_PROVIDE_LOCATION_INFORMATION: [u8; 21] = [
    0xE4, 0x07, 0x4B, 0x4B, 0x4A, 0x05, 0x01, 0x28, 0x40, 0x2D, 0x00, 0x04, 0x03, 0x5A, 0x5A, 0x5A,
    0x08, 0x09, 0xE1, 0x21, 0x00,
];

fn peer_at_50_m() -> Vec<SlRttMeasurement> {
    vec![SlRttMeasurement::from_range(PEER_ID.to_vec(), 50.0)]
}

fn endpoint() -> SlppEndpoint {
    SlppEndpoint::new(Some(OWN_ID.to_vec()))
}

/// Wrap an SLPP message the way the AMF delivers it, in a DL NAS TRANSPORT
/// of container type 12, and unwrap it as the UE's NAS layer does.
fn via_dl_nas(slpp: &[u8]) -> Vec<u8> {
    let mut pdu = Vec::new();
    DlNasTransport::new(PayloadContainerType::SlppMessage, slpp.to_vec()).encode(&mut pdu);
    assert_eq!(pdu[3], 0x0C, "DL container type 12");
    let dl = DlNasTransport::decode(&mut &pdu[3..]).expect("DL NAS decodes");
    assert_eq!(dl.payload_container_type, PayloadContainerType::SlppMessage);
    dl.payload_container
}

/// Unwrap the UE's UL NAS TRANSPORT and return its SLPP payload.
fn slpp_of(uplink: SlppUplink) -> Vec<u8> {
    let SlppUplink::Send(pdu) = uplink else {
        panic!("the UE must answer");
    };
    assert_eq!(&pdu[..3], [0x7E, 0x00, 0x67], "plain UL NAS TRANSPORT");
    assert_eq!(pdu[3], 0x0C, "UL container type 12 (SLPP)");
    let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("UL NAS decodes");
    assert_eq!(ul.payload_container_type, PayloadContainerType::SlppMessage);
    ul.payload_container
}

fn ies_of_location(msg: &SLPP_Message) -> &ProvideLocationInformation_IEs {
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::ProvideLocationInformation(p))) =
        &msg.slpp_message_body
    else {
        panic!("not ProvideLocationInformation: {msg:?}");
    };
    let ProvideLocationInformationCriticalExtensions::ProvideLocationInformation(ies) =
        &p.critical_extensions
    else {
        panic!("criticalExtensionsFuture");
    };
    ies
}

fn error_cause(msg: &SLPP_Message) -> u8 {
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::Error(Error::CriticalExtensions(
        Error_criticalExtensions::Error(ies),
    )))) = &msg.slpp_message_body
    else {
        panic!("not an Error: {msg:?}");
    };
    ies.common_i_es_error
        .as_ref()
        .expect("a cause")
        .error_cause
        .0
}

// -- TS 38.133 table 10.4A.4.1.1-1 --------------------------------------------

#[test]
fn the_rx_tx_report_value_follows_the_k2_table_rows() {
    let tc = TC_SECONDS;
    // SL_RX-TX_TIME_DIFFERENCE_0000: T < -985024 Tc.
    assert_eq!(rx_tx_report_value_k2(-985_025.0 * tc), Some(0));
    // _0001: -985024 <= T < -985020.
    assert_eq!(rx_tx_report_value_k2(-985_023.0 * tc), Some(1));
    // _246256: -4 <= T < 0. _246257: 0 <= T < 4.
    assert_eq!(rx_tx_report_value_k2(-2.0 * tc), Some(246_256));
    assert_eq!(rx_tx_report_value_k2(0.0), Some(246_257));
    assert_eq!(rx_tx_report_value_k2(3.9 * tc), Some(246_257));
    // _492512: 985020 <= T < 985024. _492513: 985024 <= T.
    assert_eq!(rx_tx_report_value_k2(985_022.0 * tc), Some(492_512));
    assert_eq!(rx_tx_report_value_k2(985_024.0 * tc), Some(RX_TX_K2_MAX));
    assert_eq!(rx_tx_report_value_k2(1.0), Some(RX_TX_K2_MAX));
    assert_eq!(rx_tx_report_value_k2(f64::NAN), None);
}

#[test]
fn fifty_metres_reports_value_246338() {
    // 50 m / c = 166.78 ns = 327.9 Tc, inside 324 <= T < 328: value 246338,
    // the value both repos' goldens carry.
    let m = SlRttMeasurement::from_range(PEER_ID.to_vec(), 50.0);
    assert_eq!(
        rx_tx_report_value_k2(m.rx_tx_time_difference_s),
        Some(246_338)
    );
}

#[test]
fn the_application_layer_id_is_the_l2_id_in_three_octets() {
    assert_eq!(application_layer_id(0x5A_5A5A), Some(PEER_ID.to_vec()));
    assert_eq!(application_layer_id(1), Some(vec![0, 0, 1]));
    assert_eq!(application_layer_id(0x0100_0000), None);
}

// -- Interop with nextgcore's LMF goldens ------------------------------------

/// The capability transfer (§5.1.5). Counterpart: nextgcore lmfd decodes
/// UE_PROVIDE_CAPABILITIES in its SLPP tests.
#[test]
fn the_lmf_capability_request_is_answered_with_the_hand_derived_octets() {
    let mut ue = endpoint();
    let reply = slpp_of(ue.handle_nas_container(&via_dl_nas(&LMF_REQUEST_CAPABILITIES), None, &[]));
    assert_eq!(reply, UE_PROVIDE_CAPABILITIES);

    let msg = decode(&reply).expect("decodes");
    assert_eq!(
        msg.application_layer_id.as_ref().map(|a| &a.0[..]),
        Some(&OWN_ID[..])
    );
    assert_eq!(
        msg.transaction_id,
        Some(SLPP_MessageTransactionID(1)),
        "same tid"
    );
    assert!(msg.end_transaction.0, "§5.1.2: endTransaction TRUE");
    assert_eq!(msg.sequence_number, Some(SequenceNumber(0)), "§4.3.2");
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::ProvideCapabilities(pc))) =
        &msg.slpp_message_body
    else {
        panic!("not ProvideCapabilities");
    };
    let ProvideCapabilitiesCriticalExtensions::ProvideCapabilities(ies) = &pc.critical_extensions
    else {
        panic!("criticalExtensionsFuture");
    };
    assert!(
        ies.sl_tdoa_provide_capabilities.is_none(),
        "not requested, not supported"
    );
    let container = &ies.sl_rtt_provide_capabilities.as_ref().expect("sl-RTT").0;
    assert_eq!(container, &UE_SL_RTT_CAPABILITIES);
    let caps: SL_RTT_ProvideCapabilities = decode_ie(container).expect("caps");
    assert_eq!(caps.positioning_modes.0.len(), 2);
    assert!(!caps.positioning_modes.0[0], "not UE-based");
    assert!(caps.positioning_modes.0[1], "UE-assisted");
    assert_eq!(caps.sl_rtt_capability_band_list.0.len(), 1);
    assert_eq!(
        caps.sl_rtt_capability_band_list.0[0].sl_prs_rx_tx_time_diff_without_tx_time_stamp,
        Some(SL_RTT_CapabilityPerBandSl_PRS_RxTxTimeDiffWithoutTxTimeStamp(0)),
        "n1"
    );
}

/// The location information transfer (§5.3) after the capability transfer.
/// Counterpart: nextgcore lmfd decodes UE_PROVIDE_LOCATION_INFORMATION in its
/// SLPP tests.
#[test]
fn the_lmf_location_request_is_answered_with_the_hand_derived_octets() {
    let mut ue = endpoint();
    let _ = ue.handle_nas_container(&via_dl_nas(&LMF_REQUEST_CAPABILITIES), None, &[]);
    let reply = slpp_of(ue.handle_nas_container(
        &via_dl_nas(&LMF_REQUEST_LOCATION_INFORMATION),
        None,
        &peer_at_50_m(),
    ));
    assert_eq!(reply, UE_PROVIDE_LOCATION_INFORMATION);

    let msg = decode(&reply).expect("decodes");
    assert_eq!(msg.transaction_id, Some(SLPP_MessageTransactionID(2)));
    assert!(msg.end_transaction.0);
    assert_eq!(
        msg.sequence_number,
        Some(SequenceNumber(1)),
        "the second message"
    );
    let ies = ies_of_location(&msg);
    assert!(ies.common_i_es_provide_location_information.is_none());
    let rtt = &ies
        .sl_rtt_provide_location_information
        .as_ref()
        .expect("sl-RTT")
        .0;
    assert_eq!(rtt, &UE_SL_RTT_REPORT);
    let report: SL_RTT_ProvideLocationInformation = decode_ie(rtt).expect("report");
    let list = report
        .sl_rtt_signal_measurement_information
        .expect("measured")
        .sl_rtt_meas_list
        .0;
    assert_eq!(list.len(), 1);
    let element = &list[0].0[0];
    assert_eq!(
        element.application_layer_id.as_ref().map(|a| &a.0[..]),
        Some(&PEER_ID[..])
    );
    let SL_PRS_RxTxTimeDiffMeasResult::Single_SL_PRS_RxTxTimeDiff(r) =
        &element.sl_prs_rx_tx_time_diff_meas_result
    else {
        panic!("single");
    };
    assert_eq!(
        r.sl_prs_rx_tx_time_diff_result,
        Some(SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult::K2(
            SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult_k2(246_338)
        ))
    );
    assert!(
        r.sl_prs_rsrp_result.is_none(),
        "no RSRP is measured, none is invented"
    );
}

// -- §5.4 and the rest of the table ------------------------------------------

/// The LCS correlation id the AMF put in the DL Additional information comes
/// back as the UL routing information (TS 24.501 §5.4.5.2.2 c1).
#[test]
fn the_reply_echoes_the_lcs_correlation_id_as_routing_information() {
    let mut ue = endpoint();
    let SlppUplink::Send(pdu) = ue.handle_nas_container(
        &via_dl_nas(&LMF_REQUEST_CAPABILITIES),
        Some(b"corr-0001"),
        &[],
    ) else {
        panic!("answered");
    };
    let ul = UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
    assert_eq!(ul.additional_info.as_deref(), Some(&b"corr-0001"[..]));
    assert_eq!(ul.payload_container, UE_PROVIDE_CAPABILITIES);
}

#[test]
fn a_location_request_with_no_peer_reports_not_all_measurements_possible() {
    let mut ue = endpoint();
    let reply = ue
        .handle_downlink(&LMF_REQUEST_LOCATION_INFORMATION, &[])
        .expect("a reply");
    let msg = decode(&reply).expect("decodes");
    let ies = ies_of_location(&msg);
    let report: SL_RTT_ProvideLocationInformation = decode_ie(
        &ies.sl_rtt_provide_location_information
            .as_ref()
            .expect("sl-RTT")
            .0,
    )
    .expect("report");
    assert!(report.sl_rtt_signal_measurement_information.is_none());
    assert_eq!(
        report.sl_rtt_error,
        Some(SL_RTT_LocationInformationError(2)),
        "notAllRequestedMeasurementsPossible"
    );
}

fn location_request(location_information_type: u8) -> Vec<u8> {
    let common = CommonIEsRequestLocationInformation {
        location_information_type: LocationInformationType(location_information_type),
        periodical_reporting: None,
        additional_information: None,
        qos: None,
        environment: None,
        scheduled_location_time: None,
        location_coordinate_types: None,
        velocity_types: None,
    };
    let mut msg = decode(&LMF_REQUEST_LOCATION_INFORMATION).expect("decodes");
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::RequestLocationInformation(r))) =
        &mut msg.slpp_message_body
    else {
        unreachable!()
    };
    let RequestLocationInformationCriticalExtensions::RequestLocationInformation(ies) =
        &mut r.critical_extensions
    else {
        unreachable!()
    };
    ies.common_i_es_request_location_information = Some(
        RequestLocationInformation_IEsCommonIEsRequestLocationInformation(
            encode_ie(&common).expect("encodes"),
        ),
    );
    encode(&msg).expect("encodes")
}

#[test]
fn a_range_estimate_request_is_answered_with_requested_method_not_supported() {
    // rangeEstimateRequired (4): UE-based, which this UE does not offer.
    let mut ue = endpoint();
    let reply = ue
        .handle_downlink(&location_request(4), &peer_at_50_m())
        .expect("a reply");
    let msg = decode(&reply).expect("decodes");
    let ies = ies_of_location(&msg);
    assert!(ies.sl_rtt_provide_location_information.is_none());
    let common: CommonIEsProvideLocationInformation = decode_ie(
        &ies.common_i_es_provide_location_information
            .as_ref()
            .expect("common")
            .0,
    )
    .expect("common");
    assert_eq!(
        common.location_error.map(|e| e.locationfailurecause.0),
        Some(1),
        "requestedMethodNotSupported"
    );
}

#[test]
fn an_undecodable_message_gets_a_header_error_without_a_transaction() {
    let mut ue = endpoint();
    let reply = ue.handle_downlink(&[0xFF], &[]).expect("an Error");
    let msg = decode(&reply).expect("decodes");
    assert_eq!(error_cause(&msg), 1, "slppMessageHeaderError");
    assert!(msg.transaction_id.is_none(), "the tid was not decoded");
}

fn with_body(body: SLPP_MessageBody_c1, tid: u8, seq: Option<u8>, ack: bool) -> Vec<u8> {
    encode(&SLPP_Message {
        application_layer_id: None,
        transaction_id: Some(SLPP_MessageTransactionID(tid)),
        end_transaction: SLPP_MessageEndTransaction(false),
        sequence_number: seq.map(SequenceNumber),
        session_id: None,
        acknowledgement: ack.then_some(Acknowledgement {
            ack_requested: AcknowledgementAckRequested(true),
            ack_indicator: None,
        }),
        slpp_message_body: Some(SLPP_MessageBody::C1(body)),
        non_critical_extension: None,
    })
    .expect("encodes")
}

fn assistance_data() -> SLPP_MessageBody_c1 {
    SLPP_MessageBody_c1::ProvideAssistanceData(ProvideAssistanceData {
        critical_extensions: ProvideAssistanceDataCriticalExtensions::ProvideAssistanceData(
            ProvideAssistanceData_IEs {
                common_i_es_provide_assistance_data: None,
                common_sl_prs_methods_i_es_provide_assistance_data: None,
                sl_ao_a_provide_assistance_data: None,
                sl_rtt_provide_assistance_data: Some(
                    ProvideAssistanceData_IEsSl_RTT_ProvideAssistanceData(vec![0x00]),
                ),
                sl_tdoa_provide_assistance_data: None,
                sl_toa_provide_assistance_data: None,
                late_non_critical_extension: None,
                non_critical_extension: None,
            },
        ),
    })
}

#[test]
fn assistance_data_is_stored_and_needs_no_reply() {
    let mut ue = endpoint();
    assert!(ue
        .handle_downlink(&with_body(assistance_data(), 3, None, false), &[])
        .is_none());
    assert_eq!(
        ue.assistance_data()
            .and_then(|a| a.sl_rtt_provide_assistance_data.as_ref())
            .map(|c| c.0.clone()),
        Some(vec![0x00])
    );
}

#[test]
fn a_message_asking_for_an_acknowledgement_gets_one_even_when_repeated() {
    let mut ue = endpoint();
    let pad = with_body(assistance_data(), 3, Some(7), true);
    for round in 0..2 {
        let reply = ue.handle_downlink(&pad, &[]).expect("an acknowledgement");
        let msg = decode(&reply).expect("decodes");
        assert!(msg.slpp_message_body.is_none(), "ack-only, round {round}");
        assert_eq!(
            msg.acknowledgement.and_then(|a| a.ack_indicator),
            Some(SequenceNumber(7)),
            "§4.3.3 ackIndicator, round {round}"
        );
    }
}

#[test]
fn a_duplicate_sequence_number_is_discarded() {
    let mut ue = endpoint();
    let mut req = decode(&LMF_REQUEST_CAPABILITIES).expect("decodes");
    req.sequence_number = Some(SequenceNumber(9));
    let bytes = encode(&req).expect("encodes");
    assert!(
        ue.handle_downlink(&bytes, &[]).is_some(),
        "first copy answered"
    );
    assert!(
        ue.handle_downlink(&bytes, &[]).is_none(),
        "§4.3.2: duplicate discarded"
    );
}

#[test]
fn an_abort_stops_the_transaction_without_a_reply() {
    let mut ue = endpoint();
    let abort = SLPP_MessageBody_c1::Abort(Abort {
        critical_extensions: AbortCriticalExtensions::Abort(Abort_IEs {
            common_i_es_abort: None,
            late_non_critical_extension: None,
            non_critical_extension: None,
        }),
    });
    assert!(ue
        .handle_downlink(&with_body(abort, 5, None, false), &[])
        .is_none());
    assert!(ue.was_aborted(5));
    assert!(!ue.was_aborted(4));
}

#[test]
fn an_assistance_data_request_is_answered_with_an_error() {
    let mut ue = endpoint();
    let req = SLPP_MessageBody_c1::RequestAssistanceData(RequestAssistanceData {
        critical_extensions: RequestAssistanceDataCriticalExtensions::CriticalExtensionsFuture(
            RequestAssistanceDataCriticalExtensions_criticalExtensionsFuture {},
        ),
    });
    let reply = ue
        .handle_downlink(&with_body(req, 6, None, false), &[])
        .expect("an Error");
    let msg = decode(&reply).expect("decodes");
    assert_eq!(error_cause(&msg), 0, "undefined");
    assert_eq!(msg.transaction_id, Some(SLPP_MessageTransactionID(6)));
}

#[test]
fn an_error_is_never_answered() {
    let mut ue = endpoint();
    let err = SLPP_MessageBody_c1::Error(Error::CriticalExtensions(
        Error_criticalExtensions::Error(Error_IEs {
            common_i_es_error: None,
            late_non_critical_extension: None,
            non_critical_extension: None,
        }),
    ));
    assert!(ue
        .handle_downlink(&with_body(err, 2, None, false), &[])
        .is_none());
}
