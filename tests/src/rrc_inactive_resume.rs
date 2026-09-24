//! RRC_INACTIVE resume, end to end, under the SHIPPED configuration (issue #201).
//!
//! # What was false
//!
//! A UE suspended to RRC_INACTIVE by the gNB CLI's `ue-suspend` could not resume.
//! Instead of an `RRCResumeRequest1` it fell back to RRC_IDLE and established
//! afresh, so the core saw a second `InitialUEMessage` with a NEW `ran_ue_ngap_id`
//! carrying a NAS PDU it had no security context to unwrap — observed live in
//! nextgcore's Docker E2E run 36045644289 as
//! `Unhandled NAS message type 0xf7 in Initial UE Message`.
//!
//! Three gates, each independently fatal, made that the outcome in **every shipped
//! configuration** (`config/ue.yaml` and `config/gnb.yaml` both set
//! `as_security_enabled: false`):
//!
//! 1. the UE's NAS plane handed the RRC plane its `KgNB` only when the wire gate
//!    was on (`send_security_mode_complete`, nextgsim-ue/src/nas/mm/orchestrator.rs),
//!    so RRC could derive no `K_RRCint`;
//! 2. the UE's RRC plane recorded an AS context only from an SMC it ACCEPTED, and
//!    an unprotected SMC is refused (§5.3.4.2), so `initiate_resume` found
//!    `self.as_security == None` and took its RRC_IDLE arm; and
//! 3. even with a `resumeMAC-I` to present, the gNB could not READ the request: its
//!    UL-CCCH byte-fallback ladder reached `handle_rrc_resume_request` only for a
//!    leading `0x28` — the pre-#151 bespoke framing — while a conformant
//!    `RRCResumeRequest1` is `UL-CCCH1` c1 index 0 and leads with `0x00`, which
//!    falls in the ladder's own `0x00..=0x1F` **RRCSetupRequest** range. So a
//!    resume was answered with an `RRCSetup` on a fabricated context, with the
//!    I-RNTI never looked up and the `resumeMAC-I` never verified.
//!
//! Gate 3 is not in the issue: it was found by encoding a real `RRCResumeRequest1`
//! and reading its leading byte. It is the same defect the re-establishment path
//! had before #151, one message class over.
//!
//! # Why this file drives the real stack
//!
//! Because #198 passed its tests while production stayed broken, by seeding the
//! state it then asserted on. Nothing here seeds: the AS keys arrive at the UE
//! because the **real** NGAP task processed a **captured core** Initial Context
//! Setup Request and its `activate_as_security` sent a real SecurityModeCommand;
//! the I-RNTI is the one the gNB allocated; the `resumeMAC-I` is computed by the UE
//! and verified by the gNB; and every identifier asserted on is **decoded off the
//! wire** from the PDU the gNB handed its SCTP task.
//!
//! The load-bearing assertion is the `ran_ue_ngap_id`: the test learns it from the
//! `InitialUEMessage` at registration and then requires the resumed NAS to arrive
//! in an `UplinkNASTransport` bearing that same value. It has no other way to know
//! it, and the defect's signature is precisely that this number changes.
//!
//! # References
//!
//! * TS 38.331 §5.3.13.3 — `RRCResumeRequest1` contents and the `resumeMAC-I`
//! * TS 38.331 §5.3.13.4 — `RRCResumeComplete` carries the triggering NAS
//! * TS 38.331 §5.3.8.3 — `RRCRelease` with `suspendConfig`
//! * TS 38.331 §5.3.13.5 — handling of a resume the UE cannot perform
//! * TS 38.300 §9.2.2.2 — a new RRC connection instead of resumption
//! * TS 33.501 Annex A.8 — the `K_RRCint` both ends derive from `KgNB`

use std::time::Duration;

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::rrc::connection::RrcConnectionManager;
use nextgsim_gnb::rrc::ue_context::RrcUeContextManager;
use nextgsim_gnb::tasks::{
    GnbTaskBase, NgapMessage, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    SctpMessage, Task, TaskMessage, DEFAULT_CHANNEL_CAPACITY,
};
use nextgsim_gnb::{NgapTask, RrcTask as GnbRrcTask};
use nextgsim_ngap::procedures::initial_ue_message::decode_initial_ue_message;
use nextgsim_ngap::procedures::nas_transport::decode_uplink_nas_transport;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::rrc_release::{
    decode_rrc_release, encode_rrc_release, RrcReleaseParams,
};
use nextgsim_rrc::procedures::rrc_resume::{
    decode_rrc_resume, decode_rrc_resume_complete, decode_rrc_resume_request1,
    encode_rrc_resume_request1, RrcResumeRequest1Params,
};
use nextgsim_rrc::procedures::rrc_setup::{
    decode_rrc_setup, decode_rrc_setup_complete, decode_rrc_setup_request,
};
use nextgsim_rrc::procedures::security_mode::{
    decode_security_mode_command, decode_security_mode_failure,
};
use nextgsim_rrc::procedures::suspend_config::{
    decode_suspend_config, encode_suspend_config, SuspendConfigParams,
};
use nextgsim_ue::{RrcTask as UeRrcTask, UeTaskBase};

/// The internal `ue_id` the RLS layer resolves this UE's STI to. Only this is
/// chosen by the test; every NGAP identifier asserted on comes off the wire.
const UE_ID: i32 = 3;
/// The AMF association everything runs over.
const AMF_CLIENT_ID: i32 = 1;
const NGAP_STREAM: u16 = 0;
const CELL_ID: i32 = 1;
/// The RAN UE NGAP ID the captured Initial Context Setup Request names, which must
/// equal the one the gNB allocates for the first UE of a fresh instance — otherwise
/// the AMF's request would find no context to key.
const CAPTURED_RAN_UE_NGAP_ID: i64 = 1;
/// The `KgNB` carried in the captured ICS Request's `SecurityKey` IE.
///
/// Handed to the UE's RRC plane as well, because TS 33.501 §6.2 has both ends derive
/// the same `KgNB` independently from KAMF: it never crosses the air, so there is no
/// wire hop to drive instead, and a test that used a different value here would be
/// asserting that two ends with DIFFERENT keys agree on a MAC.
const CAPTURED_KGNB: [u8; 32] = [0x11u8; 32];
/// `t380` in minutes: a legal `PeriodicRNAU-TimerValue`, long enough that it cannot
/// expire mid-test and turn a resume into an RNAU.
const T380_MINUTES: u16 = 20;

/// NG Setup Response as produced by the core's own ogs-ngap codec (AMFName
/// "nextgcore-amf", GUAMI 001-01, capacity 255, PLMN 001-01 with S-NSSAI sst=1).
///
/// The same captured vector `gnb_cli_reachability` and `paging_mt_service_request`
/// use, for the same reason: the AMF association must reach `Ready` on a real
/// decoded message, because `handle_ngap_pdu` refuses to route any operational PDU
/// until it does.
const CORE_NG_SETUP_RESPONSE: [u8; 55] = [
    0x20, 0x15, 0x00, 0x33, 0x00, 0x00, 0x04, 0x00, 0x01, 0x00, 0x0f, 0x06, 0x00, 0x6e, 0x65, 0x78,
    0x74, 0x67, 0x63, 0x6f, 0x72, 0x65, 0x2d, 0x61, 0x6d, 0x66, 0x00, 0x60, 0x00, 0x08, 0x00, 0x00,
    0x00, 0xf1, 0x10, 0x02, 0x00, 0x41, 0x00, 0x56, 0x40, 0x01, 0xff, 0x00, 0x50, 0x00, 0x08, 0x00,
    0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x08,
];

/// Initial Context Setup Request as produced by the strict core's ogs-ngap codec:
/// AMF-UE-NGAP-ID 1, RAN-UE-NGAP-ID 1, GUAMI 001-01, `SecurityKey` = `KgNB` of
/// 0x11 × 32, UE Security Capabilities 0x0000 (so the gNB's `select_as_algorithms`
/// picks NEA0 / NIA2), UE-AMBR 1 Gbps / 500 Mbps.
///
/// The same vector `nextgsim-ngap`'s `ics_request_from_core_decodes` pins. Used
/// here because this is what makes the AS keys arrive **in production's own way**:
/// the real NGAP task parses it and `handle_initial_context_setup_request` calls
/// `activate_as_security`, which derives the four keys from this `SecurityKey` and
/// sends the RRC SecurityModeCommand. A test that handed the UE a `KgNB` directly
/// would prove nothing about whether a registered UE ever gets one.
const CORE_ICS_REQUEST: [u8; 99] = [
    0x00, 0x0e, 0x00, 0x5f, 0x00, 0x00, 0x07, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x01, 0x00, 0x55, 0x00,
    0x02, 0x00, 0x01, 0x00, 0x1c, 0x00, 0x07, 0x00, 0x00, 0xf1, 0x10, 0x02, 0x00, 0x41, 0x00, 0x00,
    0x00, 0x02, 0x00, 0x01, 0x00, 0x77, 0x00, 0x09, 0x10, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x5e, 0x00, 0x20, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
    0x11, 0x11, 0x11, 0x11, 0x11, 0x00, 0x6e, 0x00, 0x0a, 0x0c, 0x3b, 0x9a, 0xca, 0x00, 0x30, 0x1d,
    0xcd, 0x65, 0x00,
];

/// Registration Request NAS PDU (5GMM EPD 0x7E, message type 0x41). Opaque to the
/// RAN, which relays it untouched.
fn registration_nas() -> Vec<u8> {
    vec![0x7E, 0x00, 0x41, 0x79, 0x00, 0x0D, 0x01, 0x00, 0xF1, 0x10]
}

/// The NAS PDU that triggers the resume, standing in for the Service Request a
/// suspended UE's NAS emits on MO data. Deliberately a DISTINCT byte string from
/// `registration_nas`, so the assertion that this one arrived cannot be satisfied
/// by the registration NAS being replayed.
fn service_request_nas() -> Vec<u8> {
    vec![0x7E, 0x00, 0x4C, 0x11, 0x22, 0x33, 0x44]
}

/// A gNB serving PLMN 001-01 — matching the captured NG Setup Response's served
/// PLMN and the UE's default HPLMN, so AMF selection and cell selection succeed.
///
/// **`as_security_enabled` is left at its shipped `false`.** Criterion 1 of #201 is
/// about the shipped configuration, so flipping it here would make the test pass by
/// avoiding the defect.
fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x0000_0000_0010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// The UE, likewise at the shipped default.
fn ue_config() -> UeConfig {
    UeConfig::default()
}

/// Asserts the premise the whole file rests on: both ends run with the AS-security
/// wire gate OFF, as shipped. If this ever fails, every other assertion here is
/// measuring a configuration nobody deploys.
#[test]
fn the_shipped_configuration_has_the_as_security_wire_gate_off() {
    assert!(
        !gnb_config().as_security_enabled,
        "the gNB under test must run with the SHIPPED as_security_enabled: false \
         (config/gnb.yaml), or this file proves nothing about a deployed gNB"
    );
    assert!(
        !ue_config().as_security_enabled,
        "the UE under test must run with the SHIPPED as_security_enabled: false \
         (config/ue.yaml)"
    );
}

/// Pops the next uplink RRC PDU the UE handed its RLS, skipping other RLS traffic.
fn next_ue_uplink_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<nextgsim_ue::TaskMessage<nextgsim_ue::RlsMessage>>,
) -> (RrcChannel, OctetString) {
    loop {
        match rx.try_recv() {
            Ok(nextgsim_ue::TaskMessage::Message(nextgsim_ue::RlsMessage::RrcPduDelivery {
                channel,
                pdu,
                ..
            })) => return (channel, pdu),
            Ok(_) => continue,
            Err(e) => panic!("expected an uplink RRC PDU from the UE, got none: {e}"),
        }
    }
}

/// Waits up to `budget` for the next downlink RRC PDU the gNB handed its RLS.
///
/// An `await` rather than a `try_recv` because the gNB's RRC task runs in its own
/// tokio task here: a synchronous poll would race its progress and make the test
/// depend on scheduler luck. Skips non-PDU RLS traffic (the system-information
/// broadcast its `run` loop emits on a timer).
async fn await_gnb_downlink_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
    budget: Duration,
) -> Option<(RrcChannel, OctetString)> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Some(TaskMessage::Message(GnbRlsMessage::DownlinkRrc {
                rrc_channel,
                data,
                ..
            }))) => return Some((rrc_channel, data)),
            Ok(Some(_)) => continue,
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

/// Waits for the next downlink RRC PDU that `wanted` accepts, skipping every other
/// PDU on the channel.
///
/// Matching on a PREDICATE rather than taking the head, because a live gNB RRC task
/// puts several unrelated PDUs on this channel: the MIB every 80 ms and SIB1 every
/// 160 ms on BCCH from its `run` loop's own timer, and a `UECapabilityEnquiry` on
/// DL-DCCH immediately after every RRCSetupComplete. A test that asserted on the
/// head would be asserting on whichever of those happened to arrive first.
///
/// The predicate is always a real DECODE of the message the step expects, so a
/// mismatch is reported as "the expected message never arrived" rather than as a
/// surprising channel.
async fn await_gnb_rrc_matching(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
    budget: Duration,
    mut wanted: impl FnMut(RrcChannel, &[u8]) -> bool,
) -> Option<(RrcChannel, OctetString)> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        let Some((channel, pdu)) = await_gnb_downlink_rrc(rx, Duration::from_millis(300)).await
        else {
            continue;
        };
        if wanted(channel, pdu.data()) {
            return Some((channel, pdu));
        }
    }
    None
}

/// Waits up to `budget` for a BROADCAST RRC PDU on `channel`.
///
/// A separate waiter because system information travels as
/// `RlsMessage::BroadcastRrc`, not `DownlinkRrc`: it has no `ue_id`, since a UE
/// reading BCCH is by definition not yet addressed. `await_gnb_downlink_rrc` would
/// therefore never see a SIB at all.
async fn await_gnb_broadcast_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
    budget: Duration,
    channel: RrcChannel,
) -> Option<OctetString> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(300), rx.recv()).await {
            Ok(Some(TaskMessage::Message(GnbRlsMessage::BroadcastRrc {
                rrc_channel,
                data,
                ..
            }))) if rrc_channel == channel => return Some(data),
            Ok(Some(_)) => continue,
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

/// Asserts that NO UE-dedicated downlink RRC PDU arrives within `budget`, and
/// returns the offending PDU if one does.
///
/// A positively-bounded absence: the budget is spent in full, and broadcast PDUs are
/// excluded by construction (they arrive as `BroadcastRrc`, which
/// `await_gnb_downlink_rrc` skips), so this cannot pass merely because the channel
/// was busy with system information.
async fn expect_no_gnb_dedicated_rrc(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
    budget: Duration,
) -> Option<(RrcChannel, OctetString)> {
    await_gnb_rrc_matching(rx, budget, |_, _| true).await
}

/// Waits up to `budget` for an NGAP PDU the gNB sent the AMF that `pick` accepts.
///
/// Needed because the NGAP task runs in its own tokio task: `try_recv` would race
/// its progress and make the test depend on scheduler luck.
async fn await_sctp_pdu<T>(
    rx: &mut tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
    budget: Duration,
    mut pick: impl FnMut(&[u8]) -> Option<T>,
) -> Option<T> {
    let deadline = tokio::time::Instant::now() + budget;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
            Ok(Some(TaskMessage::Message(SctpMessage::SendMessage { buffer, .. }))) => {
                if let Some(found) = pick(buffer.data()) {
                    return Some(found);
                }
            }
            Ok(Some(_)) => continue,
            Ok(None) => return None,
            Err(_) => continue,
        }
    }
    None
}

/// A registered, AS-keyed UE and the live ends needed to drive it further.
///
/// The gNB's RRC task is **spawned** rather than held, and reached only through
/// `rrc_tx` — its real inbox. That is deliberate: `RrcMessage::SuspendUe` and
/// `RrcMessage::AsSecurityForReestablishment` are dispatched by the task's own
/// `run` loop, and calling their handlers directly would be the seam that let #198
/// pass while production stayed broken. The UE's RRC task is held, because its
/// entry points (`handle_downlink_rrc`, `handle_uplink_nas_delivery`) are the real
/// message handlers its own loop calls and are public for exactly this use.
struct SuspendableUe {
    ue_rrc: UeRrcTask,
    /// The gNB RRC task's real inbox.
    gnb_rrc_tx: nextgsim_gnb::TaskHandle<GnbRrcMessage>,
    ue_rls_rx: tokio::sync::mpsc::Receiver<nextgsim_ue::TaskMessage<nextgsim_ue::RlsMessage>>,
    gnb_rls_rx: tokio::sync::mpsc::Receiver<TaskMessage<GnbRlsMessage>>,
    sctp_rx: tokio::sync::mpsc::Receiver<TaskMessage<SctpMessage>>,
    /// The RAN UE NGAP ID read out of the `InitialUEMessage` on the wire. The value
    /// criterion 2 requires to be unchanged across the suspension.
    wire_ran_ue_ngap_id: i64,
    handles: Vec<tokio::task::JoinHandle<()>>,
}

/// Drives a real registration through to Initial Context Setup, so the UE holds the
/// AS keys a resume needs and the gNB holds a UE-associated NG connection.
///
/// Every hop is production code reached through a real message on a real inbox.
/// Nothing is seeded:
///
/// ```text
/// SctpAssociationUp + captured NG Setup Response -> AMF context Ready
/// UE RRC RRCSetupRequest (real UPER, UL-CCCH)    -> gNB RRC RRCSetup (DL-CCCH)
/// UE RRC RRCSetupComplete (UL-DCCH)              -> NgapMessage::InitialNasDelivery
/// real NGAP task                                 -> InitialUEMessage on the wire
/// captured core ICS Request -> activate_as_security
///                           -> SecurityModeCommand   (to the UE, via gNB RRC)
///                           -> AsSecurityForReestablishment (to gNB RRC: its K_RRCint)
/// ```
///
/// Both AS-security messages travel the NGAP→RRC channel and are dispatched by the
/// gNB RRC task's own `run` loop; the test only relays the resulting air-interface
/// PDU to the UE.
async fn register_and_key_a_ue() -> SuspendableUe {
    let (base, _app_rx, ngap_rx, rrc_rx, _gtp_rx, gnb_rls_rx, sctp_rx) =
        GnbTaskBase::new(gnb_config(), DEFAULT_CHANNEL_CAPACITY);
    let mut gnb_rls_rx = gnb_rls_rx;
    let mut sctp_rx = sctp_rx;
    let ngap_tx = base.ngap_tx.clone();
    let gnb_rrc_tx = base.rrc_tx.clone();
    let mut handles = Vec::new();

    // Both gNB tasks run their OWN dispatch loops, so every message below is routed
    // by production rather than by a test-side call to a handler.
    let mut ngap_task = NgapTask::new(base.clone());
    handles.push(tokio::spawn(async move { ngap_task.run(ngap_rx).await }));
    let mut gnb_rrc_task = GnbRrcTask::new(base.clone());
    handles.push(tokio::spawn(async move { gnb_rrc_task.run(rrc_rx).await }));

    ngap_tx
        .send(NgapMessage::SctpAssociationUp {
            client_id: AMF_CLIENT_ID,
            association_id: 1,
            in_streams: 2,
            out_streams: 2,
        })
        .await
        .expect("the NGAP task accepts the association-up");
    assert!(
        await_sctp_pdu(&mut sctp_rx, Duration::from_secs(5), |bytes| {
            // NG Setup Request: initiatingMessage (0x00), procedure code 21 (0x15).
            (bytes.first() == Some(&0x00) && bytes.get(1) == Some(&0x15)).then_some(())
        })
        .await
        .is_some(),
        "the gNB must send an NG Setup Request before it can be Ready"
    );
    ngap_tx
        .send(NgapMessage::ReceiveNgapPdu {
            client_id: AMF_CLIENT_ID,
            stream: NGAP_STREAM,
            pdu: OctetString::from_slice(&CORE_NG_SETUP_RESPONSE),
        })
        .await
        .expect("the NGAP task accepts the NG Setup Response");

    // Unbar the cell through the real message, not a direct call.
    gnb_rrc_tx
        .send(GnbRrcMessage::RadioPowerOn)
        .await
        .expect("the gNB RRC task accepts RadioPowerOn");

    let (ue_base, _ue_app_rx, _ue_nas_rx, _ue_rrc_rx, ue_rls_rx) =
        UeTaskBase::new(ue_config(), DEFAULT_CHANNEL_CAPACITY);
    let mut ue_rls_rx = ue_rls_rx;
    let mut ue_rrc = UeRrcTask::new(ue_base);

    ue_rrc.handle_signal_changed(CELL_ID, -60).await;

    // The UE must read this cell's SIB1 BEFORE it resumes, and that is not a test
    // convenience -- it is load-bearing for the `resumeMAC-I`.
    //
    // `VarResumeMAC-Input` covers `targetCellIdentity` (the NCI of the cell being
    // resumed on) and `sourcePhysCellId` (derived from the NCI of the cell the
    // suspendConfig arrived in). The UE learns this cell's NCI from SIB1 and nowhere
    // else; without it `serving_cell_identity()` is 0, so the UE would MAC over cell
    // 0 while the gNB recomputed over its configured NCI and every resume would be
    // refused for the wrong reason.
    //
    // Taken from the gNB RRC task's OWN periodic broadcast (its `run` loop emits
    // MIB/SIB1 on a timer, TS 38.331 §5.2.1) and relayed on the real BCCH-DL-SCH
    // entry point, so the NCI the UE ends up with is the one this gNB actually
    // advertises rather than one the test asserted into place.
    let sib1 = await_gnb_broadcast_rrc(
        &mut gnb_rls_rx,
        Duration::from_secs(5),
        RrcChannel::BcchDlSch,
    )
    .await
    .expect(
        "the gNB RRC task must broadcast SIB1 on BCCH-DL-SCH, or the UE can never \
             learn this cell's NCI and no resumeMAC-I could ever match",
    );
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::BcchDlSch, sib1)
        .await;

    // RRC connection establishment (TS 38.331 §5.3.3).
    ue_rrc.perform_cycle().await;
    ue_rrc
        .handle_uplink_nas_delivery(1, OctetString::from_slice(&registration_nas()))
        .await;
    let (ch, setup_req) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch, "RRCSetupRequest rides UL-CCCH");

    send_uplink_rrc(&gnb_rrc_tx, RrcChannel::UlCcch, setup_req).await;
    let (ch, rrc_setup) =
        await_gnb_rrc_matching(&mut gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_setup(b).is_ok()
        })
        .await
        .expect("the gNB must answer the RRCSetupRequest with an RRCSetup");
    assert_eq!(ch, RrcChannel::DlCcch, "RRCSetup rides DL-CCCH");

    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (ch, setup_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch, "RRCSetupComplete rides UL-DCCH");

    // The hop that reaches NGAP.
    send_uplink_rrc(&gnb_rrc_tx, RrcChannel::UlDcch, setup_complete).await;

    // Learn the RAN UE NGAP ID from the WIRE. Production allocated it; the test has
    // no other way to know it, which is what makes the criterion-2 assertion real.
    let wire_ran_ue_ngap_id = await_sctp_pdu(&mut sctp_rx, Duration::from_secs(5), |bytes| {
        decode_initial_ue_message(bytes)
            .ok()
            .map(|d| i64::from(d.ran_ue_ngap_id))
    })
    .await
    .expect("the registration must produce an InitialUEMessage on the wire");
    assert_eq!(
        wire_ran_ue_ngap_id, CAPTURED_RAN_UE_NGAP_ID,
        "the first UE of a fresh gNB takes RAN UE NGAP ID 1, which is also the id the \
         captured ICS Request names -- they must agree or the AMF's request would find \
         no context"
    );

    // NAS security. The UE's NAS plane derives KgNB at Security Mode Complete and
    // hands it to RRC as `RrcMessage::AsSecurityKey` (nextgsim-ue/src/main.rs, on
    // `MmOutput::AsSecurityKgnb`). Delivered here through that same public entry
    // point, carrying the SAME KgNB the captured ICS Request gives the gNB -- which
    // is what TS 33.501 §6.2 guarantees: both ends derive it independently from KAMF
    // and it never crosses the air, so there is no wire hop to drive instead.
    ue_rrc.set_pending_kgnb(CAPTURED_KGNB);

    // The AMF's Initial Context Setup Request. The real NGAP task's
    // `activate_as_security` derives the four AS keys from its `SecurityKey` and
    // emits BOTH the SecurityModeCommand (bound for the UE) and
    // `AsSecurityForReestablishment` (the gNB RRC task's own copy of K_RRCint) onto
    // the RRC inbox, where the spawned task above dispatches them.
    ngap_tx
        .send(NgapMessage::ReceiveNgapPdu {
            client_id: AMF_CLIENT_ID,
            stream: NGAP_STREAM,
            pdu: OctetString::from_slice(&CORE_ICS_REQUEST),
        })
        .await
        .expect("the NGAP task accepts the Initial Context Setup Request");

    // The SecurityModeCommand as it leaves the gNB's send path. Unprotected, because
    // the wire gate is off -- which is the configuration under test.
    let (ch, smc) = await_gnb_rrc_matching(&mut gnb_rls_rx, Duration::from_secs(5), |_, b| {
        // Decoding as a BARE DL-DCCH-Message is itself part of the assertion: with the
        // wire gate off the SMC carries NO MAC-I, which is exactly the form the UE must
        // refuse on the wire and still record keys from. A protected SMC would be
        // `smc_uper || MAC-I` and would not decode here at all.
        decode_security_mode_command(b).is_ok()
    })
    .await
    .expect(
        "Initial Context Setup must produce an unprotected SecurityModeCommand on the \
         air; nothing means activate_as_security never ran",
    );
    assert_eq!(ch, RrcChannel::DlDcch, "the SMC rides DL-DCCH on SRB1");

    // The UE sees it. With the wire gate off it REFUSES the command (§5.3.4.2
    // requires integrity protection and this one carries none) -- and records the
    // inactive AS context anyway, which is the change under test.
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, smc)
        .await;
    let (ch, failure) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch);
    assert!(
        decode_security_mode_failure(failure.data()).is_ok(),
        "under the shipped config the UE must STILL refuse an unprotected SMC \
         (TS 38.331 §5.3.4.2): recording the resume keys must not activate SRB1 \
         protection"
    );
    send_uplink_rrc(&gnb_rrc_tx, RrcChannel::UlDcch, failure).await;

    SuspendableUe {
        ue_rrc,
        gnb_rrc_tx,
        ue_rls_rx,
        gnb_rls_rx,
        sctp_rx,
        wire_ran_ue_ngap_id,
        handles,
    }
}

/// Hands an uplink RRC PDU to the gNB RRC task through its real inbox.
async fn send_uplink_rrc(
    tx: &nextgsim_gnb::TaskHandle<GnbRrcMessage>,
    channel: RrcChannel,
    data: OctetString,
) {
    tx.send(GnbRrcMessage::UplinkRrc {
        ue_id: UE_ID,
        rrc_channel: channel,
        data,
    })
    .await
    .expect("the gNB RRC task accepts uplink RRC");
}

/// Suspends the UE to RRC_INACTIVE through `RrcMessage::SuspendUe` — the only
/// production trigger, and the one `ue-suspend` sends — and delivers the resulting
/// `RRCRelease`-with-`suspendConfig` to the UE.
///
/// Returns the I-RNTI the **gNB allocated**, decoded from the `suspendConfig` the
/// gNB put on the wire. Not a value the test chose.
async fn suspend_to_inactive(ue: &mut SuspendableUe) -> u64 {
    ue.gnb_rrc_tx
        .send(GnbRrcMessage::SuspendUe {
            ue_id: UE_ID,
            t380_minutes: Some(T380_MINUTES),
        })
        .await
        .expect("the gNB RRC task accepts SuspendUe");

    let (ch, release) =
        await_gnb_rrc_matching(&mut ue.gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_release(b).is_ok()
        })
        .await
        .expect(
            "SuspendUe must put an RRCRelease on the air; nothing means the gNB refused to \
         suspend and released instead",
        );
    assert_eq!(
        ch,
        RrcChannel::DlDcch,
        "an RRCRelease with suspendConfig rides DL-DCCH on SRB1 (TS 38.331 §6.2.2)"
    );

    // Read the I-RNTI off the wire, out of the gNB's own PDU.
    let decoded =
        decode_rrc_release(release.data()).expect("the gNB must emit a decodable RRCRelease");
    let suspend_bytes = decoded.suspend_config.as_ref().expect(
        "the release must carry a suspendConfig; without one this is a plain release \
         and the gNB never entered RRC_INACTIVE (TS 38.331 §5.3.8.3) -- which is what \
         `initiate_rrc_suspend` does when it has no AS keys",
    );
    let suspend_config =
        decode_suspend_config(suspend_bytes).expect("the suspendConfig must decode");
    let i_rnti = suspend_config.full_i_rnti;
    assert_ne!(
        i_rnti, 0,
        "the gNB must allocate a non-zero I-RNTI: 0 is the sentinel a lookup must not \
         match by accident"
    );
    assert_eq!(
        suspend_config.t380_minutes,
        Some(T380_MINUTES),
        "the t380 asked for must reach the air, or the UE could RNAU mid-test and the \
         resume under test would carry the wrong cause"
    );

    ue.ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, release)
        .await;

    i_rnti
}

/// **Criteria 1 and 2 of issue #201, wire-decoded.**
///
/// A UE suspended by `ue-suspend`'s own `RrcMessage::SuspendUe`, given uplink data,
/// sends an `RRCResumeRequest1` whose `resumeMAC-I` the gNB VERIFIES — under the
/// shipped configs — and the NAS riding the `RRCResumeComplete` reaches the AMF in
/// an `UplinkNASTransport` on the PRE-EXISTING NGAP context, with the
/// `ran_ue_ngap_id` unchanged and no second `InitialUEMessage`.
///
/// Every identifier asserted on is decoded from a PDU production emitted: the
/// I-RNTI from the gNB's `suspendConfig`, the `resumeMAC-I` from the UE's
/// `RRCResumeRequest1`, and the `ran_ue_ngap_id` from the NGAP PDUs on the SCTP
/// outbox.
///
/// Revert-verified three ways, one per gate — see the module header:
///
/// * make `store_inactive_as_context` (nextgsim-ue) return without writing
///   `inactive_as_security`, and this fails at the `RRCResumeRequest1` channel
///   assertion: the UE sends an `RRCSetupRequest` on UL-CCCH instead;
/// * restore the `as_security_enabled` gate on `MmOutput::AsSecurityKgnb`
///   (nextgsim-ue/src/nas/mm/orchestrator.rs) and it fails the same way, because
///   there is then no `KgNB` for the UE's RRC plane to derive from;
/// * remove the `decode_rrc_resume_request1` arm from `handle_ul_ccch_message`
///   (nextgsim-gnb) and it fails at the `RRCResume` assertion, because the gNB reads
///   the resume as a setup request and answers `RRCSetup` on DL-CCCH.
#[tokio::test]
async fn a_suspended_ue_resumes_on_the_pre_existing_ngap_context() {
    let mut ue = register_and_key_a_ue().await;
    let i_rnti = suspend_to_inactive(&mut ue).await;

    // MO data: the NAS hands RRC a Service Request, which in RRC_INACTIVE is the
    // resume trigger (TS 38.331 §5.3.13.2).
    let nas = service_request_nas();
    ue.ue_rrc
        .handle_uplink_nas_delivery(2, OctetString::from_slice(&nas))
        .await;

    // 1. The UE sends an `RRCResumeRequest1` -- NOT an `RRCSetupRequest`. This is
    //    the assertion the defect fails: the fallback produced UL-CCCH.
    let (ch, resume_req) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlCcch1,
        "a resume rides UL-CCCH1 (TS 38.331 §6.2.1); UL-CCCH would mean the UE fell \
         back to RRC_IDLE and is establishing afresh -- issue #201's symptom"
    );
    let decoded = decode_rrc_resume_request1(resume_req.data())
        .expect("the UE must emit a decodable RRCResumeRequest1");
    assert_eq!(
        decoded.resume_identity, i_rnti,
        "the UE must present the FULL I-RNTI the gNB allocated in its suspendConfig"
    );
    assert_ne!(
        decoded.resume_mac_i, 0,
        "the resumeMAC-I must be a real NIA2 MAC over VarResumeMAC-Input; 0 is what an \
         absent K_RRCint or an NIA0 context produces, and it would verify for anyone"
    );

    // 2. The gNB VERIFIES it and answers `RRCResume` on DL-DCCH. A refused resume
    //    sends nothing at all (§5.3.13.3 leaves the UE to T319), and a resume read as
    //    a setup request answers `RRCSetup` on DL-CCCH -- so an `RRCResume` on
    //    DL-DCCH is positive evidence that the MAC was recomputed and matched.
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlCcch1, resume_req).await;
    // Matched as a real `RRCResume` DECODE, so an `RRCSetup` -- what the gNB sent when
    // it read the resume as a setup request -- cannot satisfy this.
    let (ch, rrc_resume) =
        await_gnb_rrc_matching(&mut ue.gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_resume(b).is_ok()
        })
        .await
        .expect(
            "the gNB must answer a verified resume with a real RRCResume. Silence means \
             the resumeMAC-I did not verify against the stored context; an RRCSetup \
             instead means the gNB could not READ the RRCResumeRequest1 and fell through \
             to its RRCSetupRequest arm",
        );
    assert_eq!(
        ch,
        RrcChannel::DlDcch,
        "RRCResume rides DL-DCCH on the SRB1 a suspended UE already had (§6.2.2)"
    );

    // 3. The UE completes, and the NAS that triggered the resume rides the
    //    `RRCResumeComplete` (§5.3.13.4).
    ue.ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, rrc_resume)
        .await;
    let (ch, complete) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch, "RRCResumeComplete rides UL-DCCH");
    let complete_decoded = decode_rrc_resume_complete(complete.data())
        .expect("the UE must emit a decodable RRCResumeComplete");
    assert_eq!(
        complete_decoded.dedicated_nas_message.as_deref(),
        Some(nas.as_slice()),
        "the NAS that triggered the resume must ride the Complete (§5.3.13.4), \
         byte-for-byte"
    );

    // 4. **The criterion.** The gNB forwards it as an `UplinkNASTransport` on the
    //    pre-existing context, and the `ran_ue_ngap_id` DECODED OFF THE WIRE is the
    //    pre-suspension one.
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlDcch, complete).await;

    enum Seen {
        Uplink { ran_ue_ngap_id: u32, nas: Vec<u8> },
        SecondInitial { ran_ue_ngap_id: u32 },
    }
    let seen = await_sctp_pdu(&mut ue.sctp_rx, Duration::from_secs(5), |bytes| {
        if let Ok(d) = decode_uplink_nas_transport(bytes) {
            return Some(Seen::Uplink {
                ran_ue_ngap_id: d.ran_ue_ngap_id,
                nas: d.nas_pdu,
            });
        }
        // A second InitialUEMessage is the DEFECT, matched explicitly so the failure
        // names what happened instead of timing out with nothing to say.
        decode_initial_ue_message(bytes)
            .ok()
            .map(|d| Seen::SecondInitial {
                ran_ue_ngap_id: d.ran_ue_ngap_id,
            })
    })
    .await
    .expect(
        "the resumed NAS must reach the AMF; nothing at all on the wire means the gNB \
         dropped it",
    );

    match seen {
        Seen::Uplink {
            ran_ue_ngap_id,
            nas: wire_nas,
        } => {
            assert_eq!(
                i64::from(ran_ue_ngap_id),
                ue.wire_ran_ue_ngap_id,
                "the resumed UplinkNASTransport must carry the PRE-SUSPENSION \
                 ran_ue_ngap_id ({}), decoded off the wire -- a different value means a \
                 new RRC connection, not a resume",
                ue.wire_ran_ue_ngap_id
            );
            assert_eq!(
                wire_nas, nas,
                "and it must carry the Service Request the UE sent, unmodified: this is \
                 the PDU nextgcore #403's handle_service_request_nas needs"
            );
        }
        Seen::SecondInitial { ran_ue_ngap_id } => panic!(
            "the gNB sent a SECOND InitialUEMessage (ran_ue_ngap_id={ran_ue_ngap_id}) \
             instead of an UplinkNASTransport -- issue #201's exact symptom, which the \
             core answers with Service Reject #9"
        ),
    }

    for handle in ue.handles {
        handle.abort();
    }
}

/// **The negative half of criterion 3.** A resume presenting a WRONG `resumeMAC-I`
/// is refused: the gNB answers nothing and keeps its stored context.
///
/// The positive control is the test above, and the pair is what makes either one
/// mean something: without this, "the gNB answered an RRCResume" could mean it
/// answers anything; without that, "the gNB was silent" could mean it is always
/// silent. This test carries its own positive control too — the genuine request,
/// replayed after the forgery, still resumes.
///
/// The forgery is byte-identical to the real request except for the MAC, and is
/// built with the UE's own encoder, so the refusal cannot be an artefact of a
/// malformed PDU. It has to be the verification.
#[tokio::test]
async fn a_resume_with_a_wrong_resume_mac_i_is_rejected() {
    let mut ue = register_and_key_a_ue().await;
    let i_rnti = suspend_to_inactive(&mut ue).await;

    // The GENUINE request, so the forgery differs in exactly one field.
    ue.ue_rrc
        .handle_uplink_nas_delivery(2, OctetString::from_slice(&service_request_nas()))
        .await;
    let (ch, genuine) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch1);
    let genuine_decoded =
        decode_rrc_resume_request1(genuine.data()).expect("a decodable RRCResumeRequest1");
    assert_eq!(
        genuine_decoded.resume_identity, i_rnti,
        "the forgery must name a REAL suspended context, or it would be refused for \
         the wrong reason (UnknownIRnti rather than MacIMismatch)"
    );

    // Complement every bit of the MAC. A wrapping increment could collide at 0xFFFF;
    // the complement cannot equal the original for any 16-bit value.
    let forged_mac = !genuine_decoded.resume_mac_i;
    assert_ne!(forged_mac, genuine_decoded.resume_mac_i);
    let forged = encode_rrc_resume_request1(&RrcResumeRequest1Params {
        resume_identity: genuine_decoded.resume_identity,
        resume_mac_i: forged_mac,
        resume_cause: genuine_decoded.resume_cause,
    })
    .expect("the forged request encodes");

    send_uplink_rrc(
        &ue.gnb_rrc_tx,
        RrcChannel::UlCcch1,
        OctetString::from_slice(&forged),
    )
    .await;

    // Nothing is sent: §5.3.13.3 leaves the UE to fall back on T319, and no
    // RRCReject can ride a connection that was never resumed. The budget is spent in
    // full, and broadcast PDUs are skipped, so this is a bounded absence rather than
    // a race won.
    if let Some((rrc_channel, data)) =
        expect_no_gnb_dedicated_rrc(&mut ue.gnb_rls_rx, Duration::from_secs(1)).await
    {
        panic!(
            "a forged resumeMAC-I must be refused, but the gNB answered on \
             {rrc_channel:?} with {} bytes (leading {:#04x})",
            data.len(),
            data.data().first().copied().unwrap_or(0)
        );
    }

    // And the context is deliberately KEPT: a failed verification may be a genuine
    // UE on a stale key, so discarding on the first bad MAC would let anyone evict a
    // suspended UE by guessing an I-RNTI. Proven POSITIVELY -- the genuine request,
    // replayed after the forgery, still resumes.
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlCcch1, genuine).await;
    let (ch, _answer) =
        await_gnb_rrc_matching(&mut ue.gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_resume(b).is_ok()
        })
        .await
        .expect(
            "the genuine resume must still be honoured after a forgery: the stored \
             context must not have been evicted by the failed attempt",
        );
    assert_eq!(ch, RrcChannel::DlDcch);

    for handle in ue.handles {
        handle.abort();
    }
}

/// **Criterion 4.** When the UE does fall back to RRC_IDLE, the gNB is told and no
/// stale I-RNTI context is left behind.
///
/// The signal is a fresh `RRCSetupRequest` on the same `ue_id`: TS 38.331 §5.3.13.5
/// has a UE that cannot resume establish instead, and TS 38.300 §9.2.2.2 makes "a
/// new RRC connection instead of resumption of the previous RRC connection" the
/// network's cue. 3GPP defines no UE→network "I am abandoning this I-RNTI" message,
/// so this is the only signal there is.
///
/// Asserted through the gNB's own OBSERVABLE behaviour, not a private counter: after
/// the fallback the same I-RNTI must no longer resume. That is the property the leak
/// breaks — a retained context would still answer — and it needs no test-only
/// accessor.
///
/// Revert-verified: remove the `discard_suspended_for_ue` call from
/// `process_rrc_setup_request` (nextgsim-gnb/src/rrc/connection.rs) and the final
/// assertion fails, because the gNB answers the replayed resume with an `RRCResume`
/// off the context it should have dropped.
#[tokio::test]
async fn a_fallback_to_idle_leaves_no_stale_i_rnti_context() {
    let mut ue = register_and_key_a_ue().await;
    let i_rnti = suspend_to_inactive(&mut ue).await;

    // Capture a GENUINE resume for this I-RNTI without sending it. This is the probe
    // the final assertion replays, and taking it here -- while the context is still
    // live -- is what makes the two halves comparable.
    ue.ue_rrc
        .handle_uplink_nas_delivery(2, OctetString::from_slice(&service_request_nas()))
        .await;
    let (ch, genuine_resume) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlCcch1);
    assert_eq!(
        decode_rrc_resume_request1(genuine_resume.data())
            .expect("decodable")
            .resume_identity,
        i_rnti,
        "the probe must name the I-RNTI the gNB allocated"
    );

    // Precondition, asserted rather than assumed: this probe DOES resume while the
    // context is live. Without it the final assertion would pass for a gNB that had
    // never held a context at all.
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlCcch1, genuine_resume).await;
    let (ch, answer) =
        await_gnb_rrc_matching(&mut ue.gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_resume(b).is_ok()
        })
        .await
        .expect("the probe must resume while the context is live, or this test proves nothing");
    assert_eq!(ch, RrcChannel::DlDcch);

    // Suspend again, so there is a fresh context for the fallback to abandon. The
    // gNB allocates a NEW I-RNTI (§5.3.13.3 -- an I-RNTI is single-use), which is
    // exactly why a leak accumulates one entry per attempt.
    ue.ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlDcch, answer)
        .await;
    let (_ch, complete) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlDcch, complete).await;
    let second_i_rnti = suspend_to_inactive(&mut ue).await;
    assert_ne!(
        second_i_rnti, i_rnti,
        "an I-RNTI is single-use: the gNB must allocate a fresh one (§5.3.13.3)"
    );
    let second_resume = {
        ue.ue_rrc
            .handle_uplink_nas_delivery(3, OctetString::from_slice(&service_request_nas()))
            .await;
        let (ch, pdu) = next_ue_uplink_rrc(&mut ue.ue_rls_rx);
        assert_eq!(ch, RrcChannel::UlCcch1);
        pdu
    };

    // THE FALLBACK. A UE that could not resume establishes afresh (§5.3.13.5), on the
    // same `ue_id` -- which is what the RLS layer resolves from the UE's STI and what
    // survives a suspension, because a UE in RRC_INACTIVE keeps sending heartbeats.
    let (fallback_base, _a, _b, _c, fallback_rls_rx) =
        UeTaskBase::new(ue_config(), DEFAULT_CHANNEL_CAPACITY);
    let mut fallback_rls_rx = fallback_rls_rx;
    let mut fallback_ue = UeRrcTask::new(fallback_base);
    fallback_ue.handle_signal_changed(CELL_ID, -60).await;
    fallback_ue.perform_cycle().await;
    fallback_ue
        .handle_uplink_nas_delivery(1, OctetString::from_slice(&service_request_nas()))
        .await;
    let (ch, fresh_setup_req) = next_ue_uplink_rrc(&mut fallback_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlCcch,
        "a re-establishing UE sends an RRCSetupRequest on UL-CCCH"
    );
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlCcch, fresh_setup_req).await;
    let (ch, _rrc_setup) =
        await_gnb_rrc_matching(&mut ue.gnb_rls_rx, Duration::from_secs(5), |_, b| {
            decode_rrc_setup(b).is_ok()
        })
        .await
        .expect("the gNB must answer the fresh establishment with an RRCSetup");
    assert_eq!(ch, RrcChannel::DlCcch, "and that answer is an RRCSetup");

    // **The criterion.** The abandoned I-RNTI must no longer resume: the gNB has
    // discarded the context, so the same request now finds nothing and is answered
    // with silence (`ResumeRejection::UnknownIRnti`).
    send_uplink_rrc(&ue.gnb_rrc_tx, RrcChannel::UlCcch1, second_resume).await;
    if let Some((rrc_channel, data)) =
        expect_no_gnb_dedicated_rrc(&mut ue.gnb_rls_rx, Duration::from_secs(1)).await
    {
        panic!(
            "the gNB must discard the RRC_INACTIVE context (I-RNTI={second_i_rnti:#x}) of \
             a UE that established a NEW RRC connection (TS 38.300 §9.2.2.2), but it \
             answered the abandoned I-RNTI on {rrc_channel:?} with {} bytes -- the stale \
             context is still there, holding K_RRCint for a UE that has moved on \
             (issue #201, criterion 4)",
            data.len()
        );
    }

    for handle in ue.handles {
        handle.abort();
    }
}

/// The UE's half of criterion 4: a UE that cannot resume ESTABLISHES rather than
/// going quiet, so the gNB has something to react to.
///
/// Driven on a UE that is suspended but holds NO AS keys — the pre-#201 state of
/// every shipped UE — so the fallback arm of `initiate_resume` is the one under
/// test. It also carries the triggering NAS onward, which the silent fallback lost
/// until NAS retransmitted on T3517.
///
/// Revert-verified: make `leave_rrc_inactive_to_idle` (nextgsim-ue) stop calling
/// `start_connection_establishment` and this fails with no uplink PDU at all.
#[tokio::test]
async fn a_ue_that_cannot_resume_establishes_instead_of_going_quiet() {
    // No `set_pending_kgnb` and no SecurityModeCommand, so this UE has no AS keys of
    // either kind and `initiate_resume` must take its fallback arm.
    let (ue_base, _a, _b, _c, ue_rls_rx) = UeTaskBase::new(ue_config(), DEFAULT_CHANNEL_CAPACITY);
    let mut ue_rls_rx = ue_rls_rx;
    let mut ue_rrc = UeRrcTask::new(ue_base);
    ue_rrc.handle_signal_changed(CELL_ID, -60).await;
    ue_rrc.perform_cycle().await;

    // Reach RRC_CONNECTED, which a suspension has to come from. The RRCSetup is
    // built by the gNB's own production encoder rather than by hand.
    ue_rrc
        .handle_uplink_nas_delivery(1, OctetString::from_slice(&registration_nas()))
        .await;
    let (_ch, _setup_req) = next_ue_uplink_rrc(&mut ue_rls_rx);
    let rrc_setup = build_production_rrc_setup();
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (ch, _setup_complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlDcch,
        "the UE must reach RRC_CONNECTED before it can be suspended"
    );

    // Suspend it with a real `RRCRelease` carrying a real `suspendConfig`, both from
    // the production encoders. The gNB cannot be ASKED to suspend this UE -- with no
    // AS context `initiate_rrc_suspend` correctly refuses, which is issue #38's own
    // guard -- but a UE stranded in RRC_INACTIVE without keys is precisely the live
    // state #201 reports, because in a real run the gNB HAS keys and the UE does not.
    let suspend_bytes = encode_suspend_config(&SuspendConfigParams {
        full_i_rnti: 0x42,
        // The gNB's own default (`SuspendParams::default`, nextgsim-gnb), so the UE
        // under test is given a cycle a real gNB would signal.
        ran_paging_cycle_rf: 64,
        ran_notification_area: None,
        t380_minutes: Some(T380_MINUTES),
        next_hop_chaining_count: 0,
    })
    .expect("the suspendConfig encodes");
    let release = encode_rrc_release(&RrcReleaseParams {
        rrc_transaction_id: 0,
        cell_reselection_priorities: None,
        redirected_carrier_info: None,
        suspend_config: Some(suspend_bytes),
        deprioritisation_req: None,
        wait_time: None,
    })
    .expect("the RRCRelease encodes");
    ue_rrc
        .handle_downlink_rrc(
            CELL_ID,
            RrcChannel::DlDcch,
            OctetString::from_slice(&release),
        )
        .await;
    while ue_rls_rx.try_recv().is_ok() {}

    // MO data on a UE that cannot compute a resumeMAC-I.
    let nas = service_request_nas();
    ue_rrc
        .handle_uplink_nas_delivery(2, OctetString::from_slice(&nas))
        .await;

    // It must ESTABLISH, and the establishment must carry the pending NAS onward
    // rather than sitting silent until NAS retransmits.
    let (ch, out) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(
        ch,
        RrcChannel::UlCcch,
        "a UE that cannot resume must fall back to establishment (TS 38.331 \
         §5.3.13.5), which is how the gNB learns the I-RNTI is abandoned \
         (TS 38.300 §9.2.2.2)"
    );
    assert!(
        decode_rrc_setup_request(out.data()).is_ok(),
        "and it must be a real RRCSetupRequest the gNB can act on"
    );

    // The triggering NAS survives to the RRCSetupComplete: the resume kept it in
    // `initial_nas_pdu` for an RRCResumeComplete that never came, and on this path it
    // must ride the Complete instead.
    let rrc_setup = build_production_rrc_setup();
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::DlCcch, rrc_setup)
        .await;
    let (ch, complete) = next_ue_uplink_rrc(&mut ue_rls_rx);
    assert_eq!(ch, RrcChannel::UlDcch);
    let decoded = decode_rrc_setup_complete(complete.data()).expect("a decodable RRCSetupComplete");
    assert_eq!(
        decoded.dedicated_nas_message, nas,
        "the NAS that triggered the failed resume must ride the RRCSetupComplete: \
         dropping it loses the Service Request until NAS retransmits on T3517"
    );
}

/// An `RRCSetup` built by the gNB's own production encoder, for the two places above
/// that need one without a live gNB task.
///
/// Uses `RrcConnectionManager`'s real builder rather than hand-rolled bytes, so the
/// UE under test is answered by the same PDU a real gNB sends.
fn build_production_rrc_setup() -> OctetString {
    let mut conn_mgr = RrcConnectionManager::new();
    let mut ue_mgr = RrcUeContextManager::new();
    conn_mgr.set_barred(false);
    let result = conn_mgr
        .process_rrc_setup_request(&mut ue_mgr, UE_ID, 0x1234_5678, false, 3)
        .expect("the gNB's own setup path must produce an RRCSetup");
    result.rrc_setup_pdu
}
