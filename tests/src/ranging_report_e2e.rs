//! #139, moved onto SLPP by nextgcore #557: sidelink ranging end to end, from an
//! SL-PRS occasion to the bytes that leave the UE.
//!
//! Runs the REAL sidelink and ranging tasks and the REAL SLPP endpoint in one
//! process, and asserts the **SL Rx-Tx value recovered from the UL NAS TRANSPORT
//! the UE emits**, not a log line and not an in-process value read out of a task:
//!
//! ```text
//! SidelinkMessage::SlPrsOccasion
//!   → sidelink task: measure each configured anchor (geometry → RTT + carrier phase)
//!   → RangingMessage::{StartRanging, RttMeasurement, CarrierPhaseMeasurement}
//!   → ranging task: RTT distance, widelane ambiguity resolution
//!   → RangingMessage::ReportToLmf
//!   → NasMessage::SidelinkRttMeasurements
//!   → the SLPP endpoint holds them (in production: MmOrchestrator caches them)
//!   → the LMF's SLPP RequestLocationInformation arrives (nextgcore's golden octets)
//!   → UL NAS TRANSPORT (payload container type 12, SLPP) carrying
//!     ProvideLocationInformation with sl-RTT measurements (TS 38.355)
//! ```
//!
//! **This file only compiles under `--features sidelink`**, and the ranging pipeline
//! is additionally behind `ranging_config.enabled`, which is off by default. The
//! default `cargo test --workspace` therefore skips it; the `ranging` CI job runs it.
//!
//! # What this does NOT assert, and why
//!
//! That the LMF turns the measurement into a range. The LMF is nextgcore's `lmfd`, a
//! different product in a different repository with no build-time link to this one,
//! so the chain is asserted in two halves that meet at the bytes:
//!
//! * **here**, the UE answers nextgcore's own `REQUEST_LOCATION_INFORMATION` golden
//!   and the measurement leaves in a real NAS message of container type 12;
//! * **in nextgcore** (#557), `lmfd` decodes this UE's hand-derived reply octets
//!   (`nas::slpp`'s `UE_PROVIDE_LOCATION_INFORMATION`) and pairs the Rx-Tx halves.
//!
//! It also cannot tell a carrier-phase range from an RTT-only one any more: at the
//! k = 2 reporting step (4 Tc, about 0.6 m of one-way flight) 50.00 m and 49.95 m map
//! to the same report value. That is TS 38.133's resolution, not a loss here.

#![cfg(feature = "sidelink")]

use nextgsim_common::config::{RangingAnchor, RangingConfig, UeConfig};
use nextgsim_slpp::generated::{
    ProvideLocationInformationCriticalExtensions, SLPP_MessageBody, SLPP_MessageBody_c1,
    SL_PRS_RxTxTimeDiffMeasResult, SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult,
    SL_RTT_ProvideLocationInformation,
};
use nextgsim_ue::nas::slpp::{application_layer_id, SlppEndpoint, SlppUplink};
use nextgsim_ue::tasks::{
    NasMessage, RangingMessage, SidelinkMessage, Task, TaskHandle, TaskMessage, UeRel18Handles,
    UeTaskBase,
};
use nextgsim_ue::{RangingTask, SidelinkTask};
use tokio::sync::mpsc;

/// The peer this UE ranges against, and where it is.
///
/// A 3-4-5 triangle scaled to 30/40/50 m: the expected range is exactly 50 m, so a
/// wrong axis or a squared/unsquared slip cannot coincide with it.
const ANCHOR_UE_ID: u64 = 0x00A5_A5A5;
const ANCHOR_POSITION: [f64; 3] = [30.0, 40.0, 0.0];
const EXPECTED_RANGE_M: f64 = 50.0;

/// nextgcore lmfd's SLPP `RequestLocationInformation`, the golden
/// `REQUEST_LOCATION_INFORMATION` of `nextgcore-slpp-gen`: transaction 2,
/// rangeMeasurementsRequired, SL-RTT.
const LMF_REQUEST_LOCATION_INFORMATION: &[u8] =
    &[0x44, 0x04, 0x22, 0x40, 0x08, 0x00, 0xA0, 0x08, 0x40, 0x00];

/// TS 38.133 table 10.4A.4.1.1-1 (k = 2) for 50 m of one-way flight.
const EXPECTED_RX_TX_K2: u32 = 246_338;

fn ranging_config() -> UeConfig {
    UeConfig {
        ranging_config: Some(RangingConfig {
            enabled: true,
            own_position: [0.0, 0.0, 0.0],
            anchors: vec![RangingAnchor {
                ue_id: ANCHOR_UE_ID,
                position: ANCHOR_POSITION,
            }],
            ..RangingConfig::default()
        }),
        ..UeConfig::default()
    }
}

/// A task base with a live NAS receiver and the Rel-18 handles wired, plus the two
/// task handles and the NAS receiver the test drives.
struct Harness {
    sidelink: TaskHandle<SidelinkMessage>,
    ranging: TaskHandle<RangingMessage>,
    nas_rx: mpsc::Receiver<TaskMessage<NasMessage>>,
    sidelink_run: tokio::task::JoinHandle<()>,
    ranging_run: tokio::task::JoinHandle<()>,
}

fn spawn_harness() -> Harness {
    let (app_tx, _app_rx) = mpsc::channel(8);
    let (nas_tx, nas_rx) = mpsc::channel(16);
    let (rrc_tx, _rrc_rx) = mpsc::channel(8);
    let (rls_tx, _rls_rx) = mpsc::channel(8);
    let (sidelink_tx, sidelink_rx) = mpsc::channel(16);
    let (ranging_tx, ranging_rx) = mpsc::channel(16);
    let (mint_tx, _mint_rx) = mpsc::channel(8);

    let sidelink = TaskHandle::new(sidelink_tx);
    let ranging = TaskHandle::new(ranging_tx);
    let base = UeTaskBase::from_handles(
        std::sync::Arc::new(ranging_config()),
        TaskHandle::new(app_tx),
        TaskHandle::new(nas_tx),
        TaskHandle::new(rrc_tx),
        TaskHandle::new(rls_tx),
        Some(UeRel18Handles {
            ranging_tx: ranging.clone(),
            mint_tx: TaskHandle::new(mint_tx),
            sidelink_tx: sidelink.clone(),
        }),
    );

    let mut sidelink_task = SidelinkTask::new(base.clone());
    let mut ranging_task = RangingTask::new(base);
    Harness {
        sidelink,
        ranging,
        nas_rx,
        sidelink_run: tokio::spawn(async move { sidelink_task.run(sidelink_rx).await }),
        ranging_run: tokio::spawn(async move { ranging_task.run(ranging_rx).await }),
    }
}

/// The whole UE-side chain: an SL-PRS occasion becomes a range inside the NAS
/// message the UE puts on the wire.
#[tokio::test]
async fn an_sl_prs_occasion_becomes_a_range_in_the_ul_nas_transport_the_ue_sends() {
    let mut h = spawn_harness();

    // 1. The occasion. The sidelink task measures the anchor and reports to ranging.
    h.sidelink
        .send(SidelinkMessage::SlPrsOccasion {
            timestamp_ms: 1_000,
        })
        .await
        .expect("the sidelink task accepts the occasion");
    // Join the producer before asking for the report: its measurements and the report
    // request travel the same channel to the ranging task, so otherwise the report
    // could overtake them.
    h.sidelink.shutdown().await.expect("shutdown accepted");
    h.sidelink_run.await.expect("the sidelink task exits");

    // 2. The report, pushed toward the LMF.
    h.ranging
        .send(RangingMessage::ReportToLmf { response_tx: None })
        .await
        .expect("the ranging task accepts the report request");
    h.ranging.shutdown().await.expect("shutdown accepted");
    h.ranging_run.await.expect("the ranging task exits");

    let mut reported = None;
    while let Ok(msg) = h.nas_rx.try_recv() {
        if let TaskMessage::Message(NasMessage::SidelinkRttMeasurements { measurements }) = msg {
            reported = Some(measurements);
        }
    }
    let reported = reported.expect("the measurements must reach the NAS task");

    // 3. The SLPP endpoint on the NAS plane answers the LMF's request with what it
    //    holds. Driven directly rather than through `MmOrchestrator`, which only
    //    caches the measurements and delegates here; that boundary has its own test.
    let mut endpoint = SlppEndpoint::new(application_layer_id(1));
    let SlppUplink::Send(pdu) =
        endpoint.handle_nas_container(LMF_REQUEST_LOCATION_INFORMATION, None, &reported)
    else {
        panic!("an SLPP location request must be answered");
    };

    // 4. The assertion that matters: the measurement, read out of the bytes that left.
    let ul = nextgsim_nas::messages::mm::UlNasTransport::decode(&mut &pdu[3..])
        .expect("the UE emits a decodable UL NAS TRANSPORT");
    assert_eq!(
        ul.payload_container_type,
        nextgsim_nas::ies::ie1::PayloadContainerType::SlppMessage,
        "an AMF routes on the container type, so a report in the wrong one reaches \
         the wrong consumer"
    );
    let reply = nextgsim_slpp::decode(&ul.payload_container).expect("a decodable SLPP message");
    assert_eq!(
        reply.transaction_id.map(|t| t.0),
        Some(2),
        "the reply must echo the LMF's transaction so it can be matched"
    );
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::ProvideLocationInformation(p))) =
        reply.slpp_message_body
    else {
        panic!("expected a ProvideLocationInformation");
    };
    let ProvideLocationInformationCriticalExtensions::ProvideLocationInformation(ies) =
        p.critical_extensions
    else {
        panic!("criticalExtensionsFuture");
    };
    let report: SL_RTT_ProvideLocationInformation = nextgsim_slpp::decode_ie(
        &ies.sl_rtt_provide_location_information
            .expect("the SL-RTT report must be in the message that left the UE")
            .0,
    )
    .expect("a decodable SL-RTT report");
    let list = report
        .sl_rtt_signal_measurement_information
        .expect("measurements, not an error")
        .sl_rtt_meas_list
        .0;
    assert_eq!(list.len(), 1, "one anchor, one measurement");
    let element = &list[0].0[0];
    assert_eq!(
        element.application_layer_id.as_ref().map(|a| a.0.clone()),
        application_layer_id(ANCHOR_UE_ID),
        "the measurement must name the peer it was made against"
    );
    let SL_PRS_RxTxTimeDiffMeasResult::Single_SL_PRS_RxTxTimeDiff(result) =
        &element.sl_prs_rx_tx_time_diff_meas_result
    else {
        panic!("a single Rx-Tx result");
    };
    match &result.sl_prs_rx_tx_time_diff_result {
        Some(SL_PRS_RxTxTimeDiffResultSl_PRS_RxTxTimeDiffResult::K2(v)) => {
            assert_eq!(
                nextgsim_ue::nas::slpp::rx_tx_report_value_k2(EXPECTED_RANGE_M / 299_792_458.0),
                Some(EXPECTED_RX_TX_K2),
                "the expected value is the table mapping of the geometry"
            );
            assert_eq!(
                v.0, EXPECTED_RX_TX_K2,
                "the 30/40/50 m geometry must arrive as its one-way flight time"
            );
        }
        other => panic!("expected the k2 arm, got {other:?}"),
    }
}

/// The negative control. Without it, "the range arrived" could mean the UE reports a
/// range whatever happened — and the whole pipeline being default-off is the property
/// #55's honesty tests rest on.
#[tokio::test]
async fn a_ue_with_ranging_disabled_sends_no_range_at_all() {
    let mut config = ranging_config();
    config.ranging_config.as_mut().expect("configured").enabled = false;

    let (app_tx, _app_rx) = mpsc::channel(8);
    let (nas_tx, mut nas_rx) = mpsc::channel(16);
    let (rrc_tx, _rrc_rx) = mpsc::channel(8);
    let (rls_tx, _rls_rx) = mpsc::channel(8);
    let (sidelink_tx, sidelink_rx) = mpsc::channel(16);
    let (ranging_tx, ranging_rx) = mpsc::channel(16);
    let (mint_tx, _mint_rx) = mpsc::channel(8);
    let sidelink = TaskHandle::new(sidelink_tx);
    let ranging = TaskHandle::new(ranging_tx);
    let base = UeTaskBase::from_handles(
        std::sync::Arc::new(config.clone()),
        TaskHandle::new(app_tx),
        TaskHandle::new(nas_tx),
        TaskHandle::new(rrc_tx),
        TaskHandle::new(rls_tx),
        Some(UeRel18Handles {
            ranging_tx: ranging.clone(),
            mint_tx: TaskHandle::new(mint_tx),
            sidelink_tx: sidelink.clone(),
        }),
    );
    let mut sidelink_task = SidelinkTask::new(base.clone());
    let mut ranging_task = RangingTask::new(base);
    let sidelink_run = tokio::spawn(async move { sidelink_task.run(sidelink_rx).await });
    let ranging_run = tokio::spawn(async move { ranging_task.run(ranging_rx).await });

    sidelink
        .send(SidelinkMessage::SlPrsOccasion {
            timestamp_ms: 1_000,
        })
        .await
        .expect("accepted");
    sidelink.shutdown().await.expect("shutdown accepted");
    sidelink_run.await.expect("exits");
    ranging
        .send(RangingMessage::ReportToLmf { response_tx: None })
        .await
        .expect("accepted");
    ranging.shutdown().await.expect("shutdown accepted");
    ranging_run.await.expect("exits");

    while let Ok(msg) = nas_rx.try_recv() {
        assert!(
            !matches!(
                msg,
                TaskMessage::Message(NasMessage::SidelinkRttMeasurements { .. })
            ),
            "a disabled ranging config must produce no measurement"
        );
    }

    // And the SLPP reply says so rather than inventing a measurement.
    let mut endpoint = SlppEndpoint::new(application_layer_id(1));
    let SlppUplink::Send(pdu) =
        endpoint.handle_nas_container(LMF_REQUEST_LOCATION_INFORMATION, None, &[])
    else {
        panic!("answered");
    };
    let ul = nextgsim_nas::messages::mm::UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
    let reply = nextgsim_slpp::decode(&ul.payload_container).expect("decodes");
    let Some(SLPP_MessageBody::C1(SLPP_MessageBody_c1::ProvideLocationInformation(p))) =
        reply.slpp_message_body
    else {
        panic!("expected a ProvideLocationInformation");
    };
    let ProvideLocationInformationCriticalExtensions::ProvideLocationInformation(ies) =
        p.critical_extensions
    else {
        panic!("criticalExtensionsFuture");
    };
    let report: SL_RTT_ProvideLocationInformation =
        nextgsim_slpp::decode_ie(&ies.sl_rtt_provide_location_information.expect("sl-RTT").0)
            .expect("decodes");
    assert!(report.sl_rtt_signal_measurement_information.is_none());
    assert_eq!(
        report.sl_rtt_error.map(|e| e.0),
        Some(2),
        "notAllRequestedMeasurementsPossible, not an empty list"
    );
}
