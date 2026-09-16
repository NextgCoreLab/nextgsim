//! #139 — sidelink ranging end to end, from an SL-PRS occasion to the bytes that
//! leave the UE.
//!
//! Runs the REAL sidelink and ranging tasks and the REAL LPP endpoint in one
//! process, and asserts the **range recovered from the UL NAS TRANSPORT the UE
//! emits** — not a log line, and not an in-process value read out of a task:
//!
//! ```text
//! SidelinkMessage::SlPrsOccasion
//!   → sidelink task: measure each configured anchor (geometry → RTT + carrier phase)
//!   → RangingMessage::{StartRanging, RttMeasurement, CarrierPhaseMeasurement}
//!   → ranging task: RTT distance, widelane ambiguity resolution
//!   → RangingMessage::ReportToLmf
//!   → NasMessage::SidelinkRangingReport
//!   → the LPP endpoint holds it (in production: MmOrchestrator caches it)
//!   → an LMF's LPP RequestLocationInformation arrives
//!   → UL NAS TRANSPORT (payload container type LPP) carrying
//!     ProvideLocationInformation with the report in addition group 4
//! ```
//!
//! **This file only compiles under `--features sidelink`**, and the ranging pipeline
//! is additionally behind `ranging_config.enabled`, which is off by default. The
//! default `cargo test --workspace` therefore skips it; the `ranging` CI job runs it.
//!
//! # What this does NOT assert, and why
//!
//! The issue asks for "the range or position result **held at the LMF**". The LMF is
//! `nextgcore`'s `lmfd` — a different product, in a different repository, with **no
//! build-time link to this one** (neither workspace depends on the other; the only
//! mention is a comment about SCTP wire compatibility). CI checks out one repository,
//! so a test here cannot reach lmfd's context, and the Docker jobs that could run
//! both binaries are `workflow_dispatch`-only.
//!
//! So the chain is asserted in two halves that meet at the bytes:
//!
//! * **here** — the UE produces a report and it leaves in a real NAS message, with
//!   the range read back out of that message;
//! * **nextgcore #138** — `lmfd` files the range against the UE, and its LPP codec is
//!   held against *this* UE's hand-derived golden vector, decoding it to the same
//!   values and re-encoding to the same bytes.
//!
//! That is as far as an automated test can go without a cross-process harness. It is
//! stated rather than papered over, and it is why the startup wording below still
//! names what is not exercised.

#![cfg(feature = "sidelink")]

use nextgsim_common::config::{RangingAnchor, RangingConfig, UeConfig};
use nextgsim_ue::nas::lpp::{
    LppBody, LppEndpoint, LppMessage, LppTransactionId, LppUplink, ServingCellMeasurements,
    SidelinkRangingMethod,
};
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

/// An LMF's `RequestLocationInformation`, from the peer's own hand-derived reference
/// vector (see `nextgsim-ue/src/nas/lpp`'s `LMF_REQUEST_LOCATION_INFORMATION`):
/// `transactionID { locationServer, 0 }, endTransaction TRUE, ecid
/// requestedMeasurements '11000'B`.
const LMF_REQUEST_LOCATION_INFORMATION: &[u8] = &[0x90, 0x01, 0x20, 0x09, 0x30];

/// The serving-cell measurements an E-CID report is made of. The LPP layer has no
/// radio of its own, so these are supplied rather than measured -- the same
/// arrangement the UE's RRC layer uses in production.
fn serving_cell() -> ServingCellMeasurements {
    ServingCellMeasurements {
        phys_cell_id: 7,
        arfcn: 1850,
        system_frame_number: None,
        rsrp_result: Some(60),
        rsrq_result: None,
    }
}

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
    let base = UeTaskBase {
        config: std::sync::Arc::new(ranging_config()),
        app_tx: TaskHandle::new(app_tx),
        nas_tx: TaskHandle::new(nas_tx),
        rrc_tx: TaskHandle::new(rrc_tx),
        rls_tx: TaskHandle::new(rls_tx),
        rel18: Some(UeRel18Handles {
            ranging_tx: ranging.clone(),
            mint_tx: TaskHandle::new(mint_tx),
            sidelink_tx: sidelink.clone(),
        }),
    };

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
        if let TaskMessage::Message(NasMessage::SidelinkRangingReport { results }) = msg {
            reported = Some(results);
        }
    }
    let reported = reported.expect("the report must reach the NAS task");

    // 3. The LPP endpoint on the NAS plane answers an LMF's request with what it
    //    holds. Driven directly rather than through `MmOrchestrator`, which only
    //    caches the report and delegates here -- that boundary has its own test in
    //    the orchestrator (issue #137).
    let mut endpoint = LppEndpoint::new();
    let LppUplink::Send(pdu) =
        endpoint.handle_nas_container(LMF_REQUEST_LOCATION_INFORMATION, &serving_cell(), &reported)
    else {
        panic!("an LPP location request must be answered");
    };

    // 4. The assertion that matters: the range, read out of the bytes that left.
    let ul = nextgsim_nas::messages::mm::UlNasTransport::decode(&mut &pdu[3..])
        .expect("the UE emits a decodable UL NAS TRANSPORT");
    assert_eq!(
        ul.payload_container_type,
        nextgsim_nas::ies::ie1::PayloadContainerType::LppMessage,
        "an AMF routes on the container type, so a report in the wrong one reaches \
         the wrong consumer"
    );
    let reply = LppMessage::decode(&ul.payload_container).expect("a decodable LPP message");
    assert_eq!(
        reply.transaction_id,
        Some(LppTransactionId {
            initiator: nextgsim_ue::nas::lpp::Initiator::LocationServer,
            transaction_number: 0,
        }),
        "the reply must echo the server's transaction so it can be matched"
    );

    let Some(LppBody::ProvideLocationInformation { sidelink, .. }) = reply.body else {
        panic!("expected a ProvideLocationInformation");
    };
    let report = sidelink.expect("the ranging report must be in the message that left the UE");
    assert_eq!(report.results.len(), 1, "one anchor, one range");
    let result = report.results[0];
    assert_eq!(
        result.peer_layer2_id, ANCHOR_UE_ID as u32,
        "the range must name the peer it was measured against"
    );
    assert_eq!(
        result.range_cm,
        (EXPECTED_RANGE_M * 100.0) as u32,
        "the 30/40/50 m geometry must arrive as 5000 cm; the RTT-only estimate for \
         the same geometry is 4995 cm, so this also says the widelane resolved"
    );
    assert_eq!(
        result.method,
        SidelinkRangingMethod::CarrierPhase,
        "two carrier frequencies are configured by default, so the widelane \
         combination must be what produced this range"
    );
    assert!(
        result.measurement_count >= 1,
        "and it must rest on at least one measurement"
    );
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
    let base = UeTaskBase {
        config: std::sync::Arc::new(config.clone()),
        app_tx: TaskHandle::new(app_tx),
        nas_tx: TaskHandle::new(nas_tx),
        rrc_tx: TaskHandle::new(rrc_tx),
        rls_tx: TaskHandle::new(rls_tx),
        rel18: Some(UeRel18Handles {
            ranging_tx: ranging.clone(),
            mint_tx: TaskHandle::new(mint_tx),
            sidelink_tx: sidelink.clone(),
        }),
    };
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
                TaskMessage::Message(NasMessage::SidelinkRangingReport { .. })
            ),
            "a disabled ranging config must produce no report"
        );
    }

    // And the LPP reply carries no report either, so the encoding is what a UE
    // without sidelink ranging has always sent.
    let mut endpoint = LppEndpoint::new();
    let LppUplink::Send(pdu) =
        endpoint.handle_nas_container(LMF_REQUEST_LOCATION_INFORMATION, &serving_cell(), &[])
    else {
        panic!("answered");
    };
    let ul = nextgsim_nas::messages::mm::UlNasTransport::decode(&mut &pdu[3..]).expect("decodes");
    let reply = LppMessage::decode(&ul.payload_container).expect("decodes");
    assert!(
        matches!(
            reply.body,
            Some(LppBody::ProvideLocationInformation { sidelink: None, .. })
        ),
        "no report, not an empty one"
    );
}
