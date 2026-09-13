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
) -> Option<(Vec<u8>, Vec<u8>)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRrcMessage::Paging {
            ue_paging_tmsi,
            tai_list_for_paging,
        }) = msg
        {
            return Some((ue_paging_tmsi, tai_list_for_paging));
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

    let (ue_paging_tmsi, tai_list) =
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
    gnb_rrc.handle_paging(ue_paging_tmsi, tai_list).await;

    let (channel, pcch_pdu) =
        try_take_broadcast(&mut gnb_rls_rx).expect("the gNB must broadcast the paging");
    assert_eq!(channel, RrcChannel::Pcch, "paging is a PCCH transmission");

    let records = decode_paging(pcch_pdu.data()).expect("a decodable PCCH Paging message");
    assert_eq!(
        records[0].ue_identity,
        PagedUeIdentity::FiveGSTmsi(PAGED_S_TMSI),
        "the broadcast record must carry the paged identity"
    );

    // ---- UE leg: PCCH monitor and identity match -------------------------
    let (ue_base, _ue_app_rx, mut ue_nas_rx, _ue_rrc_rx, _ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 32);
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
    let (ue_paging_tmsi, tai_list) = try_take_rrc_paging(&mut gnb_rrc_rx).expect("RRC Paging");

    let (gnb_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut gnb_rls_rx, _sctp_rx) =
        GnbTaskBase::new(gnb_config(), 32);
    let mut gnb_rrc = GnbRrcTask::new(gnb_base);
    gnb_rrc.handle_paging(ue_paging_tmsi, tai_list).await;
    let (_, pcch_pdu) = try_take_broadcast(&mut gnb_rls_rx).expect("PCCH broadcast");

    let (ue_base, _ue_app_rx, mut ue_nas_rx, _ue_rrc_rx, _ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 32);
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
