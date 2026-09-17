//! #35 — In-process AMF→gNB→UE paging test (mobile-terminated service request).
//!
//! Drives the REAL gNB NGAP, gNB RRC and UE RRC task handlers end-to-end in one
//! process, with no mocks between them:
//!
//! ```text
//! AMF NGAP Paging PDU (nextgsim-ngap encoder, TS 38.413 §9.2.3.1)
//!   -> gNB handle_ngap_pdu (real NGAP decode + served-TAI match)
//!   -> RrcMessage::Paging carrying the serialised 5G-S-TMSI
//!   -> gNB handle_paging -> PCCH broadcast (TS 38.331 §5.3.2.2)
//!   -> UE handle_downlink_rrc(Pcch) -> record matched against the UE's own
//!      5G-S-TMSI (TS 38.331 §5.3.2.3)
//!   -> NasMessage::Paging, on which the UE's NAS loop starts the MT service
//!      request (TS 24.501 §5.6.1.1).
//! ```
//!
//! Before this landed the chain died in the middle: the gNB's `handle_paging`
//! logged and dropped, nothing ever produced on PCCH, and the UE's
//! `NasMessage::Paging` variant had no producer — so every mobile-terminated
//! event was silently lost even though the AMF-side NGAP trace looked correct.
//!
//! The AMF association is brought up through production code as well: an
//! `SctpAssociationUp` followed by a real captured NG Setup Response, because
//! the gNB only dispatches operational NGAP messages once the AMF context is
//! `Ready`.

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::{NgapTask, RrcTask as GnbRrcTask};
use nextgsim_ngap::procedures::initial_ue_message::{FiveGSTmsi, Tai};
use nextgsim_ngap::procedures::paging::{
    encode_paging as encode_ngap_paging, PagingParams, UePagingIdentityValue,
};
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::paging::{decode_paging, PagedUeIdentity};
use nextgsim_ue::{NasMessage, RrcTask as UeRrcTask, TaskMessage as UeTaskMessage, UeTaskBase};
use std::time::Duration;
use tokio::sync::mpsc;

const AMF_CLIENT_ID: i32 = 1;
const NGAP_STREAM: u16 = 0;
const CELL_ID: i32 = 1;

/// Served PLMN 001-01 as the gNB config encodes it, and TAC 1 — the TAI the
/// AMF pages, so the gNB's TAI match succeeds.
const SERVED_PLMN: [u8; 3] = [0x00, 0xF1, 0x10];
const SERVED_TAC: [u8; 3] = [0x00, 0x00, 0x01];

/// The paged UE's identity: AMF Set ID 0x155, AMF Pointer 0x2A, 5G-TMSI
/// 0xDEADBEEF.
const PAGED_AMF_SET_ID: u16 = 0x155;
const PAGED_AMF_POINTER: u8 = 0x2A;
const PAGED_TMSI: [u8; 4] = [0xDE, 0xAD, 0xBE, 0xEF];

/// The same identity as the 48-bit 5G-S-TMSI the air interface carries
/// (TS 23.003 §2.10.1): AMF Set ID (10 bits) and AMF Pointer (6 bits) packed
/// into 0x556A = (0x155 << 6) | 0x2A, then the 5G-TMSI.
///
/// Written out independently of the packing code so a test-side copy of the
/// same shift cannot agree with a wrong implementation.
const PAGED_S_TMSI: [u8; 6] = [0x55, 0x6A, 0xDE, 0xAD, 0xBE, 0xEF];

/// Another subscriber on the same AMF (identical first two octets, different
/// 5G-TMSI), so a comparison that stops at the AMF part still fails.
const OTHER_S_TMSI: [u8; 6] = [0x55, 0x6A, 0x00, 0x00, 0x00, 0x01];

/// NG Setup Response as produced by the core's own ogs-ngap codec (AMFName
/// "nextgcore-amf", GUAMI 001-01/region 2/set 1/pointer 1, capacity 255, PLMN
/// 001-01 with S-NSSAI sst=1). Same vector as the cross-codec fixture in
/// `nextgsim-ngap`'s `capture_tests`, reused here so the AMF context reaches
/// `Ready` on a real message rather than a test-only state mutation.
const CORE_NG_SETUP_RESPONSE: [u8; 55] = [
    0x20, 0x15, 0x00, 0x33, 0x00, 0x00, 0x04, 0x00, 0x01, 0x00, 0x0f, 0x06, 0x00, 0x6e, 0x65, 0x78,
    0x74, 0x67, 0x63, 0x6f, 0x72, 0x65, 0x2d, 0x61, 0x6d, 0x66, 0x00, 0x60, 0x00, 0x08, 0x00, 0x00,
    0x00, 0xf1, 0x10, 0x02, 0x00, 0x41, 0x00, 0x56, 0x40, 0x01, 0xff, 0x00, 0x50, 0x00, 0x08, 0x00,
    0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x08,
];

fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// An AMF Paging PDU for `tai`, encoded by the NGAP procedure builder.
fn amf_paging_pdu(tai: Tai) -> Vec<u8> {
    encode_ngap_paging(&PagingParams {
        ue_paging_identity: UePagingIdentityValue::FiveGSTmsi(FiveGSTmsi {
            amf_set_id: PAGED_AMF_SET_ID,
            amf_pointer: PAGED_AMF_POINTER,
            five_g_tmsi: PAGED_TMSI,
        }),
        tai_list_for_paging: vec![tai],
        paging_drx: None,
        paging_priority: None,
        paging_origin: None,
    })
    .expect("encode NGAP Paging")
}

/// A UE configuration whose paging cycle MATCHES the gNB's.
///
/// Not optional: TS 38.304 §7.1's `T` is the same term on both sides, and this
/// simulator does not signal `PCCH-Config` in SIB1, so the two ends agree by
/// configuration. A UE left on the 128-frame default while the gNB pages on rf32
/// computes a different paging frame and drops the paging -- which is what this
/// test found the first time it ran.
fn ue_config_matching_gnb_paging() -> UeConfig {
    UeConfig {
        paging_default_cycle_frames: PAGING_DRX_FRAMES,
        ..UeConfig::default()
    }
}

/// The DRX cycle these tests page on, in radio frames (TS 38.331 `rf32`).
///
/// The shortest `defaultPagingCycle` there is, chosen so the deferral to the UE's
/// paging occasion bounds at 320 ms rather than the 1.28 s of the rf128 default.
const PAGING_DRX_FRAMES: u16 = 32;

fn served_tai() -> Tai {
    Tai {
        plmn_identity: SERVED_PLMN,
        tac: SERVED_TAC,
    }
}

/// A real gNB NGAP task with an AMF association in `Ready` state, plus the RRC
/// channel its handlers emit on.
async fn ngap_task_with_ready_amf() -> (NgapTask, mpsc::Receiver<GnbTaskMessage<GnbRrcMessage>>) {
    let (task_base, _app_rx, _ngap_rx, rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
        GnbTaskBase::new(gnb_config(), 32);
    let mut task = NgapTask::new(task_base);

    // Association up → the gNB creates the AMF context and sends NG Setup
    // Request; the response then moves the context to Ready.
    task.handle_association_up(AMF_CLIENT_ID, 1, 2, 2).await;
    task.handle_ngap_pdu(
        AMF_CLIENT_ID,
        NGAP_STREAM,
        OctetString::from_slice(&CORE_NG_SETUP_RESPONSE),
    )
    .await;

    (task, rrc_rx)
}

/// Pops the next `RrcMessage::Paging` the NGAP task forwarded to RRC.
fn try_take_rrc_paging(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRrcMessage>>,
) -> Option<(Vec<u8>, Vec<u8>, Option<u16>)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRrcMessage::Paging {
            ue_paging_tmsi,
            tai_list_for_paging,
            drx_cycle_frames,
        }) = msg
        {
            return Some((ue_paging_tmsi, tai_list_for_paging, drx_cycle_frames));
        }
    }
    None
}

/// Pops the next broadcast PDU the gNB RRC task handed to its RLS.
fn try_take_broadcast(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
) -> Option<(RrcChannel, OctetString)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRlsMessage::BroadcastRrc {
            rrc_channel, data, ..
        }) = msg
        {
            return Some((rrc_channel, data));
        }
    }
    None
}

/// Waits up to `budget` for the gNB to broadcast.
///
/// ISSUE #99 CHANGED THE TIMING of this whole chain: the PCCH Paging is now
/// transmitted at the paged UE's paging frame (TS 38.304 §7.1) rather than
/// immediately, so a `try_recv` straight after `handle_paging` is racing the
/// occasion by design. The DRX cycle these tests use is rf32, which bounds the
/// wait at 320 ms.
async fn await_broadcast(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
    budget: Duration,
) -> Option<(RrcChannel, OctetString)> {
    let deadline = tokio::time::Instant::now() + budget;
    loop {
        if let Some(found) = try_take_broadcast(rx) {
            return Some(found);
        }
        if tokio::time::Instant::now() >= deadline {
            return None;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Pops the next paging indication the UE RRC task handed to its NAS.
fn try_take_ue_paging(rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>) -> Option<Vec<[u8; 6]>> {
    while let Ok(msg) = rx.try_recv() {
        if let UeTaskMessage::Message(NasMessage::Paging { paging_s_tmsi }) = msg {
            return Some(paging_s_tmsi);
        }
    }
    None
}

/// The whole chain: an AMF Paging PDU reaches an idle UE as a NAS paging
/// indication for its own 5G-S-TMSI.
#[tokio::test]
async fn an_amf_paging_reaches_the_idle_ue_as_a_nas_paging_indication() {
    let (mut ngap, mut gnb_rrc_rx) = ngap_task_with_ready_amf().await;

    // ---- NGAP leg: the AMF pages a TAI this gNB serves --------------------
    ngap.handle_ngap_pdu(
        AMF_CLIENT_ID,
        NGAP_STREAM,
        OctetString::from_slice(&amf_paging_pdu(served_tai())),
    )
    .await;

    let (ue_paging_tmsi, tai_list, _drx) =
        try_take_rrc_paging(&mut gnb_rrc_rx).expect("the NGAP layer must forward an RRC Paging");
    assert_eq!(
        ue_paging_tmsi, PAGED_S_TMSI,
        "the NGAP task must serialise the 5G-S-TMSI in TS 23.003 order"
    );
    assert_eq!(tai_list.len(), 6, "exactly one served TAI matched");

    // ---- gNB RRC leg: PCCH broadcast -------------------------------------
    let (gnb_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut gnb_rls_rx, _sctp_rx) =
        GnbTaskBase::new(gnb_config(), 32);
    let mut gnb_rrc = GnbRrcTask::new(gnb_base);

    // The paging occasion the gNB will use (issue #99), computed before the send
    // so the test knows how long the transmission is legitimately deferred.
    let schedule = gnb_rrc.schedule_paging(
        &PAGED_S_TMSI,
        Some(PAGING_DRX_FRAMES),
        nextgsim_common::frame_clock::current_sfn(),
    );
    gnb_rrc
        .handle_paging(ue_paging_tmsi, tai_list, Some(PAGING_DRX_FRAMES))
        .await;

    // CRITERION 4: nothing goes out before the occasion. Only asserted when the
    // occasion is not the current frame -- roughly 31 runs in 32, and claiming it
    // in the 1-in-32 case where the delay is genuinely zero would be false.
    if !schedule.delay.is_zero() {
        assert!(
            try_take_broadcast(&mut gnb_rls_rx).is_none(),
            "the PCCH Paging must not be transmitted before the UE's paging frame \
             (SFN {}, {} ms away)",
            schedule.paging_frame,
            schedule.delay.as_millis()
        );
    }

    let (channel, pcch_pdu) =
        await_broadcast(&mut gnb_rls_rx, schedule.delay + Duration::from_millis(300))
            .await
            .expect("the gNB must broadcast the paging at the UE's paging frame");
    assert_eq!(channel, RrcChannel::Pcch, "paging is a PCCH transmission");

    let records = decode_paging(pcch_pdu.data()).expect("a decodable PCCH Paging message");
    assert_eq!(
        records[0].ue_identity,
        PagedUeIdentity::FiveGSTmsi(PAGED_S_TMSI),
        "the broadcast record must carry the paged identity"
    );

    // ---- UE leg: PCCH monitor and identity match -------------------------
    let (ue_base, _ue_app_rx, mut ue_nas_rx, _ue_rrc_rx, _ue_rls_rx) =
        UeTaskBase::new(ue_config_matching_gnb_paging(), 32);
    let mut ue_rrc = UeRrcTask::new(ue_base);
    // What the UE's NAS plane hands down once a 5G-GUTI is assigned
    // (`RrcMessage::PagingIdentity`).
    ue_rrc.set_paging_identity(Some(PAGED_S_TMSI));

    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::Pcch, pcch_pdu)
        .await;

    assert_eq!(
        try_take_ue_paging(&mut ue_nas_rx),
        Some(vec![PAGED_S_TMSI]),
        "the UE must report the paging to NAS, which starts the MT service request"
    );
}

/// The negative half: the same broadcast reaching a UE with a different
/// 5G-S-TMSI must not start anything. PCCH is a broadcast channel, so this is
/// the common case rather than an edge case.
#[tokio::test]
async fn a_ue_with_a_different_5g_s_tmsi_is_not_paged_by_the_same_broadcast() {
    let (mut ngap, mut gnb_rrc_rx) = ngap_task_with_ready_amf().await;
    ngap.handle_ngap_pdu(
        AMF_CLIENT_ID,
        NGAP_STREAM,
        OctetString::from_slice(&amf_paging_pdu(served_tai())),
    )
    .await;
    let (ue_paging_tmsi, tai_list, _drx) =
        try_take_rrc_paging(&mut gnb_rrc_rx).expect("RRC Paging");

    let (gnb_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut gnb_rls_rx, _sctp_rx) =
        GnbTaskBase::new(gnb_config(), 32);
    let mut gnb_rrc = GnbRrcTask::new(gnb_base);
    let schedule = gnb_rrc.schedule_paging(
        &PAGED_S_TMSI,
        Some(PAGING_DRX_FRAMES),
        nextgsim_common::frame_clock::current_sfn(),
    );
    gnb_rrc
        .handle_paging(ue_paging_tmsi, tai_list, Some(PAGING_DRX_FRAMES))
        .await;
    let (_, pcch_pdu) =
        await_broadcast(&mut gnb_rls_rx, schedule.delay + Duration::from_millis(300))
            .await
            .expect("PCCH broadcast");

    let (ue_base, _ue_app_rx, mut ue_nas_rx, _ue_rrc_rx, _ue_rls_rx) =
        UeTaskBase::new(ue_config_matching_gnb_paging(), 32);
    let mut ue_rrc = UeRrcTask::new(ue_base);
    ue_rrc.set_paging_identity(Some(OTHER_S_TMSI));

    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::Pcch, pcch_pdu)
        .await;

    assert!(
        try_take_ue_paging(&mut ue_nas_rx).is_none(),
        "a UE that was not paged must not start a service request"
    );
}

/// A Paging for a TAI the gNB does not serve is dropped by the NGAP layer, so
/// nothing is broadcast at all — the UE half never runs.
#[tokio::test]
async fn a_paging_for_an_unserved_tai_never_reaches_the_air_interface() {
    let (mut ngap, mut gnb_rrc_rx) = ngap_task_with_ready_amf().await;

    let foreign_tai = Tai {
        plmn_identity: [0x62, 0xF2, 0x10],
        tac: [0x00, 0x00, 0x99],
    };
    ngap.handle_ngap_pdu(
        AMF_CLIENT_ID,
        NGAP_STREAM,
        OctetString::from_slice(&amf_paging_pdu(foreign_tai)),
    )
    .await;

    assert!(
        try_take_rrc_paging(&mut gnb_rrc_rx).is_none(),
        "a Paging for another tracking area must not reach RRC"
    );
}

/// CRITERION 5 (issue #99): a record that matches this UE but arrives outside its
/// own paging occasion produces no service request.
///
/// The occasion is chosen by picking the IDENTITY rather than by controlling the
/// clock: `UE_ID` determines the paging frame, so searching for an identity whose
/// frame is several frames away from the current one gives a deterministic
/// "foreign occasion" against the live clock. Both directions are asserted, since
/// a UE that dropped everything would pass the negative alone.
#[tokio::test]
async fn a_matching_record_outside_the_ues_paging_occasion_starts_no_service_request() {
    use nextgsim_rrc::procedures::paging::{encode_paging, PagingRecordParams};
    use nextgsim_rrc::procedures::paging_occasion::{
        paging_occasion, ue_id_from_s_tmsi, PagingCycleConfig,
    };

    let cycle = PagingCycleConfig::with_default_spreading(PAGING_DRX_FRAMES).expect("valid cycle");

    // The frame is PINNED, not read from the clock (issue #171). This used to take
    // `now = current_sfn()` and then deliver the PCCH message some microseconds-to-
    // milliseconds later, against a UE that re-read the clock. A radio frame is 10 ms, so
    // the delivery could land in a LATER frame than the one the identities were chosen for
    // -- and this test is the direction where that turns CORRECT behaviour into a failure:
    // `off_occasion` drifting INTO the occasion means the record is rightly accepted and
    // the negative assertion fails.
    //
    // It was worse than a 10 ms window, because the search only cleared frames BACKWARDS
    // (`now.wrapping_sub(back)`), so a drift forwards was not covered at all. With the
    // frame pinned, the occasion test below is exact and no window is needed.
    const PINNED_SFN: u16 = 137;
    let now = PINNED_SFN;

    // An identity whose paging frame is the pinned frame, and one the pinned frame's
    // acceptance window does not reach in either direction.
    let mut at_occasion: Option<[u8; 6]> = None;
    let mut off_occasion: Option<[u8; 6]> = None;
    for candidate in 0u32..4096 {
        let s_tmsi = [
            0x55,
            0x6A,
            0xDE,
            0xAD,
            (candidate >> 8) as u8,
            (candidate & 0xFF) as u8,
        ];
        let occasion = paging_occasion(ue_id_from_s_tmsi(&s_tmsi), &cycle);
        if at_occasion.is_none() && occasion.is_paging_frame(now) {
            at_occasion = Some(s_tmsi);
        }
        // Clear the UE's whole acceptance window (T/8, floor 4) on BOTH sides, plus a
        // frame. Symmetric now: a one-sided sweep was the second half of the defect.
        let window = (cycle.t() / 8).max(4) + 1;
        if off_occasion.is_none()
            && !(0..=window).any(|offset| {
                occasion.is_paging_frame(now.wrapping_sub(offset))
                    || occasion.is_paging_frame(now.wrapping_add(offset))
            })
        {
            off_occasion = Some(s_tmsi);
        }
        if at_occasion.is_some() && off_occasion.is_some() {
            break;
        }
    }
    let at_occasion = at_occasion.expect("some identity pages in the current frame");
    let off_occasion = off_occasion.expect("some identity pages in another frame");

    // ---- outside its occasion: dropped -----------------------------------
    let pdu = encode_paging(&[PagingRecordParams::five_g_s_tmsi(off_occasion)])
        .expect("encode PCCH Paging");
    let (ue_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) =
        UeTaskBase::new(ue_config_matching_gnb_paging(), 32);
    let mut ue_rrc = UeRrcTask::new(ue_base);
    ue_rrc.pin_sfn(PINNED_SFN);
    ue_rrc.set_paging_identity(Some(off_occasion));
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::Pcch, OctetString::from_slice(&pdu))
        .await;
    assert!(
        try_take_ue_paging(&mut nas_rx).is_none(),
        "a record matching this UE but delivered outside its paging occasion must \
         not start a service request"
    );

    // ---- inside its occasion: accepted (the positive control) ------------
    let pdu = encode_paging(&[PagingRecordParams::five_g_s_tmsi(at_occasion)])
        .expect("encode PCCH Paging");
    let (ue_base, _app_rx, mut nas_rx, _rrc_rx, _rls_rx) =
        UeTaskBase::new(ue_config_matching_gnb_paging(), 32);
    let mut ue_rrc = UeRrcTask::new(ue_base);
    ue_rrc.pin_sfn(PINNED_SFN);
    ue_rrc.set_paging_identity(Some(at_occasion));
    ue_rrc
        .handle_downlink_rrc(CELL_ID, RrcChannel::Pcch, OctetString::from_slice(&pdu))
        .await;
    assert_eq!(
        try_take_ue_paging(&mut nas_rx),
        Some(vec![at_occasion]),
        "a record delivered in this UE's own paging occasion must be acted on"
    );
}
