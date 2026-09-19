//! #44 — SDAP on the user plane, end to end: the QFI picks the DRB, and a
//! **one-octet** SDAP header carries the QFI and RQI across the radio.
//!
//! Runs the REAL gNB and UE RLS tasks in one process over loopback UDP, with real
//! cell discovery, and carries user data on two DRBs of ONE PDU session:
//!
//! ```text
//! gNB RlsMessage::DownlinkData{psi 1, drb_id 1}  ──┐ default DRB (non-GBR QFIs)
//! gNB RlsMessage::DownlinkData{psi 1, drb_id 17} ──┤ GBR DRB
//!                                                  ▼  per-(UE, DRB) RLC entities
//!                                        UDP (RLS PduTransmission,
//!                                             payload = DRB identity)
//!                                                  ▼
//!                                   UE per-DRB RLC entities → reassembly
//!                                                  ▼  SDAP header stripped
//!                              NasMessage::UplinkDataDelivery{psi, data}
//! ```
//!
//! # What this covers, and why it is the test the issue needed
//!
//! Issue #44's criteria 1, 2, 4 and 6 are all statements about the **live data
//! path**, and none of them can be settled by a unit test:
//!
//! * **Criterion 1** — two flows with distinct QFIs on one PDU session map to
//!   distinct DRBs. The policy is unit-tested in `nextgsim_gtp::qfi_drb`; what needs
//!   an end-to-end test is that both DRBs of one session are ADDRESSABLE on the wire
//!   and that each flow's payload ROUTES to the bearer named for it and arrives
//!   intact, segmentation included. That is the reach of
//!   `two_drbs_of_one_session_carry_their_own_traffic` and no more.
//!
//!   The other half of criterion 1 — that the two bearers own independent
//!   sequence-number spaces — is pinned by
//!   `nextgsim_gnb::rls::task`'s
//!   `two_drbs_of_one_session_get_independent_sequence_number_spaces`, and NOT here.
//!   This harness cannot reach it: the gNB's `handle_downlink_data` submits one SDU
//!   and then drains `build_pdu` to exhaustion before returning, so two sequential
//!   `DownlinkData` messages are segmented, sent and reassembled one at a time. The
//!   two streams are never in flight together, which is the only state in which a
//!   shared entity interleaves them — so re-keying the entity maps on the PSI leaves
//!   every test in this file green. A unit test inside the gNB crate can submit to
//!   both bearers before draining either, and is therefore where that guard lives.
//! * **Criterion 2** — the header reaches the UE. The octet's layout is pinned by
//!   hand-computed byte assertions in `nextgsim_pdcp::sdap`; this asserts the gNB
//!   actually prepends one and the UE actually strips it, which is the part a
//!   codec test cannot reach.
//! * **Criterion 4** — the UE's receive entity strips the header and the payload
//!   arrives byte-identical on the right flow. A UE that did not strip would
//!   deliver an SDU one octet longer, with the header as its first byte — which is
//!   exactly the interop failure the issue was filed about, and which shows up here
//!   as a payload mismatch.
//! * **Criterion 6** — with the feature off the PSI-based path stays green. That is
//!   what `the_default_drb_carries_its_payload_unchanged` asserts, and it is the one
//!   test here compiled in **both** states deliberately: feature-off it IS the
//!   pre-SDAP path, feature-on it is the same path now carrying a header the UE
//!   strips, and either way the payload must arrive byte-identical.
//!
//! The uplink is covered separately and is NOT symmetric with the downlink — see
//! `the_uplink_header_is_added_by_the_ue_and_stripped_by_the_gnb` for why.
//!
//! # Why the payloads are what they are
//!
//! Every payload is distinct and none is a prefix of another, so a swap between the
//! two DRBs cannot look correct. One payload exceeds the 1500-byte grant so it is
//! segmented and reassembled by its own bearer's entity, which is what makes the
//! routing assertion cover a multi-PDU flow and not just a single datagram. And
//! each payload's first octet
//! is chosen NOT to be a plausible SDAP header (`0x80 | qfi`), so a UE that failed
//! to strip the header would fail the byte comparison rather than accidentally
//! passing it.

use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RlsTask as GnbRlsTask;
use nextgsim_gnb::Task as GnbTask;
use nextgsim_ue::rls::{RlsTask as UeRlsTask, RlsTaskConfig};
use nextgsim_ue::{
    NasMessage, RlsMessage as UeRlsMessage, Task as UeTask, TaskMessage as UeTaskMessage,
    UeTaskBase,
};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

/// The one PDU session under test. Two DRBs, one session — which is the shape
/// criterion 1 is about, and the shape this tree could not express before #44.
const PSI: i32 = 1;

/// A non-GBR QFI. 5QI 9 is best-effort, so the policy puts it on the default DRB.
const QFI_NON_GBR: u8 = 9;

/// A GBR QFI. 5QI 1 is conversational voice, so the policy puts it on the second
/// DRB. Distinct from [`QFI_NON_GBR`] so a mapping that sent both to one bearer
/// would be visible.
///
/// Only referenced by the feature-gated tests, because the second DRB only exists
/// with the feature on — which is the whole point of criterion 1.
#[cfg(feature = "sdap-dataplane")]
const QFI_GBR: u8 = 1;

const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

fn gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x000000010,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ..Default::default()
    }
}

/// A loopback address with a port the OS has just confirmed free.
///
/// The gNB RLS task binds a fixed address and the UE has to be told that address
/// up front, so the port cannot be left to the OS at bind time.
async fn free_loopback_addr() -> SocketAddr {
    let probe = UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("probe socket for a free port");
    probe.local_addr().expect("probe local address")
}

struct Harness {
    gnb_rls_tx: nextgsim_gnb::tasks::TaskHandle<GnbRlsMessage>,
    ue_rls_tx: nextgsim_ue::TaskHandle<UeRlsMessage>,
    gnb_rrc_rx: mpsc::Receiver<GnbTaskMessage<GnbRrcMessage>>,
    /// Where the gNB's RLS task hands a reassembled UPLINK SDU to GTP-U. The far
    /// end of the uplink path, and the only place that can say whether the UE's
    /// SDAP header was stripped before the SDU left the radio layer.
    ///
    /// Feature-gated with its reader: the only test that drives the uplink is
    /// itself gated, because without the feature there is no header to add or
    /// strip and the direction is unchanged from before #44. Gated rather than
    /// `#[allow(dead_code)]`, so the field disappears with its purpose instead of
    /// the warning being silenced while an unused channel is still held open.
    #[cfg(feature = "sdap-dataplane")]
    gnb_gtp_rx: mpsc::Receiver<GnbTaskMessage<nextgsim_gnb::tasks::GtpMessage>>,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rrc_rx: mpsc::Receiver<UeTaskMessage<nextgsim_ue::RrcMessage>>,
}

async fn start_harness() -> Harness {
    let gnb_addr = free_loopback_addr().await;

    // The GTP receiver is named only where it is read (see `Harness::gnb_gtp_rx`);
    // without the feature it is dropped here, which closes the channel the gNB's
    // RLS task would send an uplink SDU on -- harmless, because no test in that
    // state drives the uplink.
    #[cfg(feature = "sdap-dataplane")]
    let (gnb_base, _gnb_app_rx, _gnb_ngap_rx, gnb_rrc_rx, gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
        GnbTaskBase::new(gnb_config(), 64);
    #[cfg(not(feature = "sdap-dataplane"))]
    let (gnb_base, _gnb_app_rx, _gnb_ngap_rx, gnb_rrc_rx, _gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
        GnbTaskBase::new(gnb_config(), 64);
    let gnb_rls_tx = gnb_base.rls_tx.clone();
    let mut gnb_rls = GnbRlsTask::with_bind_address(gnb_base, gnb_addr);
    tokio::spawn(async move { gnb_rls.run(gnb_rls_rx).await });

    let (ue_base, _ue_app_rx, ue_nas_rx, ue_rrc_rx, ue_rls_rx) =
        UeTaskBase::new(UeConfig::default(), 64);
    let ue_rls_tx = ue_base.rls_tx.clone();
    let mut ue_rls = UeRlsTask::new(
        ue_base,
        RlsTaskConfig {
            gnb_search_list: vec![gnb_addr],
            bind_address: Some("127.0.0.1:0".parse().expect("UE bind address")),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_threshold: Duration::from_millis(2000),
        },
    );
    tokio::spawn(async move { ue_rls.run(ue_rls_rx).await });

    Harness {
        gnb_rls_tx,
        ue_rls_tx,
        gnb_rrc_rx,
        #[cfg(feature = "sdap-dataplane")]
        gnb_gtp_rx,
        ue_nas_rx,
        ue_rrc_rx,
    }
}

/// Waits until the gNB has discovered the UE and the UE has discovered the cell,
/// then makes that cell the UE's serving cell — the step the UE RRC task normally
/// performs, and the one that lets the UE accept user-plane PDUs.
async fn complete_discovery(h: &mut Harness) -> i32 {
    let gnb_saw_ue = tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = h.gnb_rrc_rx.recv().await {
            if let GnbTaskMessage::Message(GnbRrcMessage::SignalDetected { ue_id }) = msg {
                return ue_id;
            }
        }
        panic!("gNB RRC channel closed before the UE was detected");
    })
    .await
    .expect("the gNB must discover the UE");

    let cell_id = tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = h.ue_rrc_rx.recv().await {
            if let UeTaskMessage::Message(nextgsim_ue::RrcMessage::SignalChanged {
                cell_id, ..
            }) = msg
            {
                return cell_id;
            }
        }
        panic!("UE RRC channel closed before a cell was found");
    })
    .await
    .expect("the UE must discover the cell");

    h.ue_rls_tx
        .send(UeRlsMessage::AssignCurrentCell { cell_id })
        .await
        .expect("UE RLS task alive");

    gnb_saw_ue
}

/// Collects `count` downlink SDUs the UE handed up to NAS, as `(psi, bytes)`.
async fn collect_delivered(
    rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>,
    count: usize,
) -> Vec<(i32, Vec<u8>)> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(Duration::from_secs(10), async {
        while out.len() < count {
            match rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { psi, data })) => {
                    out.push((psi, data.data().to_vec()));
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s), got {}", out.len()));
    out
}

/// Payloads whose first octet cannot be mistaken for an SDAP header.
///
/// A conformant one-octet DL header is `0x80 | (rqi << 6) | qfi`, so every value in
/// `0x80..=0xFF` is a plausible header. Starting each payload at `0x0N` means a UE
/// that skipped a byte it should not have, or failed to skip one it should, changes
/// the payload in a way the byte comparison catches — rather than producing
/// something that still looks like a valid header followed by data.
fn payload(tag: u8, len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    v.push(tag & 0x0F);
    v.extend((1..len).map(|i| ((i as u32 * 7 + u32::from(tag)) % 251) as u8));
    v
}

// ============================================================================
// Criterion 6: the feature-OFF path is unchanged. Compiled in BOTH states.
// ============================================================================

/// What the gNB's RLS task expects in `DownlinkData.pdu`.
///
/// # Why a test has to do this at all
///
/// `RlsMessage::DownlinkData` is the boundary BELOW SDAP: on the live path the GTP
/// task has already prepended the header by the time it sends one, so with the
/// feature on the field carries an SDAP PDU and without it a bare SDU. A test that
/// injects at this boundary is standing in for the GTP task, so it owes the same
/// framing — injecting a bare SDU feature-on is not "testing the old path", it is
/// sending a malformed PDU that the UE correctly discards (its D/C bit reads as a
/// Control PDU).
///
/// Built with the SHIPPED encoder, so a change to the octet layout moves this
/// helper rather than leaving it encoding a stale format.
#[cfg(feature = "sdap-dataplane")]
fn framed(qfi: u8, rqi: bool, sdu: &[u8]) -> Vec<u8> {
    use nextgsim_pdcp::sdap::{build_dl_pdu, SdapHeader};
    build_dl_pdu(SdapHeader { qfi, rqi }, sdu).expect("a legal QFI")
}

/// Without the feature the field is the bare SDU, exactly as before #44.
#[cfg(not(feature = "sdap-dataplane"))]
fn framed(_qfi: u8, _rqi: bool, sdu: &[u8]) -> Vec<u8> {
    sdu.to_vec()
}

/// Tell the UE which QFIs ride which DRB, as an `RRCReconfiguration` would.
///
/// On the live path the UE learns this from the `SDAP-Config` in the DRB's
/// `DRB-ToAddMod`; this harness runs the RLS tasks alone, with no RRC, so the test
/// plays RRC's part. Without it the UE has no record of the session's second DRB and
/// falls back to treating the DRB identity as the PSI — which is the right fallback
/// for a bearer nothing configured, and the wrong answer here.
#[cfg(feature = "sdap-dataplane")]
async fn install_mapping(h: &Harness, psi: i32, drb_id: i32, qfis: Vec<u8>, default_drb: bool) {
    h.ue_rls_tx
        .send(UeRlsMessage::InstallSdapMapping {
            psi,
            drb_id,
            qfis,
            default_drb,
        })
        .await
        .expect("UE RLS task alive");
}

/// The existing PSI-based path still carries a payload byte-identically.
///
/// Compiled and run in both feature states deliberately: with the feature off it
/// IS the pre-SDAP path, and with it on it is the same path now carrying a header
/// the UE strips. Either way the payload NAS receives must be the payload the gNB
/// sent — which is the invariant an SDAP sublayer must not break, and the one
/// criterion 6 asks about.
#[tokio::test]
async fn the_default_drb_carries_its_payload_unchanged() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    // The session's default DRB, which the mapping has to name before the UE can
    // resolve its PSI. No-op without the feature.
    #[cfg(feature = "sdap-dataplane")]
    install_mapping(&h, PSI, PSI, vec![QFI_NON_GBR], true).await;

    let small = payload(0x1, 32);
    // Larger than the 1500-byte MAC grant, so RLC segments it and the receiving
    // bearer's own entity has to reassemble it. An SDAP header is prepended ONCE
    // to the whole SDU, not per segment, so a reassembly that dropped or repeated
    // a segment boundary would show up here as a corrupted payload.
    let large = payload(0x2, 3000);

    for sdu in [&small, &large] {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                // The session's default DRB. Its identity is the PSI
                // (`nextgsim_gtp::qfi_drb::allocate_drbs` keeps
                // `default_drb_id == psi.clamp(1, 32)`), so this line describes the
                // same bearer with the feature on or off — which is the
                // compatibility property the whole re-key rests on.
                drb_id: PSI,
                pdu: OctetString::from_slice(&framed(QFI_NON_GBR, false, sdu)),
            })
            .await
            .expect("gNB RLS task alive");
    }

    let delivered = collect_delivered(&mut h.ue_nas_rx, 2).await;
    let on_session: Vec<&Vec<u8>> = delivered
        .iter()
        .filter(|(psi, _)| *psi == PSI)
        .map(|(_, data)| data)
        .collect();

    assert_eq!(
        on_session,
        vec![&small, &large],
        "the payload NAS receives must be byte-identical to the one the gNB sent, \
         segmented one included -- an unstripped SDAP header would make each SDU one \
         octet longer and shift every byte"
    );
}

// ============================================================================
// Criteria 1, 2 and 4: only reachable with the feature on.
// ============================================================================

/// Criterion 1 end to end, as far as this harness reaches: both DRBs of ONE PDU
/// session are addressable, and traffic offered on each arrives intact — the right
/// payload, header-free, on the one PSI — with no cross-delivery.
///
/// This is addressing and routing, NOT bearer independence. The sequence-number
/// independence is pinned by `two_drbs_of_one_session_get_independent_sequence_number_spaces`
/// in `nextgsim-gnb/src/rls/task.rs`; see this module's header for why it cannot be
/// asserted from here (the gNB drains `build_pdu` per `DownlinkData`, so the two
/// flows are never in flight at once).
///
/// One payload is over the 1500-byte grant so it is segmented, which makes the
/// routing assertion a real one: a segment stream attributed to the wrong bearer
/// reassembles into something that is not this payload.
#[cfg(feature = "sdap-dataplane")]
#[tokio::test]
async fn two_drbs_of_one_session_carry_their_own_traffic() {
    use nextgsim_gtp::qfi_drb::allocate_drbs;

    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    let alloc = allocate_drbs(PSI as u8);
    assert_ne!(
        alloc.default_drb_id, alloc.gbr_drb_id,
        "precondition: the session's two DRBs must be distinct, or this test is \
         asserting about one bearer twice"
    );

    // Both bearers of the one session, as the RRCReconfiguration would state them:
    // the non-GBR QFI on the default DRB, the GBR one on the second. This is the
    // mapping criterion 1 is about, and the UE is TOLD it rather than deriving it.
    install_mapping(
        &h,
        PSI,
        i32::from(alloc.default_drb_id),
        vec![QFI_NON_GBR],
        true,
    )
    .await;
    install_mapping(&h, PSI, i32::from(alloc.gbr_drb_id), vec![QFI_GBR], false).await;

    let on_default = payload(0x3, 64);
    let on_gbr = payload(0x4, 3000);

    for (drb_id, qfi, sdu) in [
        (alloc.default_drb_id, QFI_NON_GBR, &on_default),
        (alloc.gbr_drb_id, QFI_GBR, &on_gbr),
    ] {
        let pdu = framed(qfi, false, sdu);
        assert_eq!(
            pdu.len(),
            sdu.len() + 1,
            "the SDAP header is ONE octet (TS 37.324 §6.2.2), not three"
        );
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: i32::from(drb_id),
                pdu: OctetString::from_slice(&pdu),
            })
            .await
            .expect("gNB RLS task alive");
    }

    let delivered = collect_delivered(&mut h.ue_nas_rx, 2).await;

    // Both SDUs belong to the SAME PDU session, so NAS sees one PSI for both --
    // which is the point: the two bearers are a radio-layer split, invisible above
    // SDAP.
    for (psi, _) in &delivered {
        assert_eq!(
            *psi, PSI,
            "both DRBs serve one PDU session, so NAS must see one PSI"
        );
    }

    let bodies: Vec<&Vec<u8>> = delivered.iter().map(|(_, data)| data).collect();
    assert!(
        bodies.contains(&&on_default),
        "the default DRB's payload must arrive intact and header-free"
    );
    assert!(
        bodies.contains(&&on_gbr),
        "and so must the GBR DRB's, segmented across several RLC PDUs and reassembled \
         by that bearer's own entity -- this says the second DRB is addressable and its \
         flow routes to it, which is what this harness can see. That the two bearers \
         also NUMBER independently is asserted by the gNB unit test \
         two_drbs_of_one_session_get_independent_sequence_number_spaces, because the \
         drain-per-message drive pattern here never puts both flows in flight at once"
    );
}

/// Criteria 2 and 4, as a POSITIVE assertion about the octet that crosses the
/// radio: the QFI and RQI the gNB set are the ones the UE decodes, and the SDU
/// behind them is untouched.
///
/// Distinct from the test above, which asserts the *payload* survives. This one
/// asserts the *header* carries information — a UE that stripped a fixed octet
/// without reading it would pass that test and fail this one.
#[cfg(feature = "sdap-dataplane")]
#[tokio::test]
async fn the_qfi_and_rqi_survive_the_radio() {
    use nextgsim_pdcp::sdap::{build_dl_pdu, decode_dl_pdu, SdapHeader};

    // Both RQI values and the six-bit boundary, because 63 is where a mask error
    // would let the QFI bleed into the RQI bit.
    for (qfi, rqi) in [
        (QFI_NON_GBR, false),
        (QFI_NON_GBR, true),
        (0u8, false),
        (63u8, true),
    ] {
        let sdu = payload(0x5, 48);
        let pdu = build_dl_pdu(SdapHeader { qfi, rqi }, &sdu).expect("a legal QFI");

        // The shipped DECODER is the authority on the layout, so the round trip is
        // driven off it rather than off a hand-written byte comparison here -- the
        // absolute layout is pinned by `nextgsim_pdcp::sdap`'s own hand-computed
        // vectors, and duplicating them would be a second place to get wrong.
        let (decoded, body) = decode_dl_pdu(&pdu).expect("the gNB's own header must decode");
        assert_eq!(
            decoded,
            SdapHeader { qfi, rqi },
            "QFI {qfi} / RQI {rqi} must survive the octet"
        );
        assert_eq!(body, &sdu[..], "and the SDU behind it must be untouched");
    }
}

/// The UPLINK half of criteria 2 and 4: the UE prepends a one-octet UL SDAP header
/// and the gNB strips it, so the SDU that reaches GTP-U is the one the UE sent.
///
/// This direction is not symmetric with the downlink and is the easier half to get
/// wrong. The UE has no per-packet classifier — no TFT or URSP matcher — so it uses
/// the session's default flow, and it only stamps a header once the network has told
/// it the mapping: an uplink packet that beat its `RRCReconfiguration` goes out
/// unstamped rather than being dropped, so that the first packet of a session is not
/// lost to a race. That fallback is why this test installs the mapping FIRST and then
/// asserts a byte-exact payload — without the install the SDU would still cross, be
/// read as a malformed header at the gNB, and vanish.
#[cfg(feature = "sdap-dataplane")]
#[tokio::test]
async fn the_uplink_header_is_added_by_the_ue_and_stripped_by_the_gnb() {
    use nextgsim_gnb::tasks::GtpMessage;

    let mut h = start_harness().await;
    let _ue_id = complete_discovery(&mut h).await;

    install_mapping(&h, PSI, PSI, vec![QFI_NON_GBR], true).await;

    // One small and one segmented, because the header goes on the SDU once and RLC
    // segments the SDAP PDU -- so a segmented uplink is where a header counted per
    // segment would show up.
    let small = payload(0x6, 40);
    let large = payload(0x7, 3000);

    for sdu in [&small, &large] {
        h.ue_rls_tx
            .send(UeRlsMessage::UplinkData {
                psi: PSI,
                data: OctetString::from_slice(sdu),
            })
            .await
            .expect("UE RLS task alive");
    }

    let mut received: Vec<Vec<u8>> = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), async {
        while received.len() < 2 {
            match h.gnb_gtp_rx.recv().await {
                Some(GnbTaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    received.push(pdu.data().to_vec());
                }
                Some(_) => continue,
                None => panic!("gNB GTP channel closed after {} SDU(s)", received.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "expected 2 uplink SDU(s) at the gNB, got {} -- an unstamped uplink is \
             discarded as a malformed SDAP header, so zero here means the UE never \
             learnt the mapping",
            received.len()
        )
    });

    assert_eq!(
        received,
        vec![small, large],
        "the SDU reaching GTP-U must be byte-identical to the one the UE sent: a \
         header the gNB failed to strip would make it one octet longer"
    );
}

/// Criterion 3: an over-budget flow is dropped rather than forwarded.
///
/// Driven at the enforcer rather than through the radio, deliberately: the drop
/// happens in the gNB's GTP task before anything reaches RLS, so an end-to-end
/// assertion would be "no SDU arrives", which is also what a lost UDP datagram
/// looks like. A positive assertion on the enforcer's verdict, plus the DSCP it
/// resolves, says what the data path acts on.
#[cfg(feature = "sdap-dataplane")]
#[test]
fn an_over_budget_flow_is_policed_and_its_dscp_resolved() {
    use nextgsim_gtp::{QfiDscpMapper, QosFlowEnforcer};

    let mut enforcer = QosFlowEnforcer::new();
    // 5QI 1 (conversational voice, GBR) with an 8 kbit/s ceiling.
    //
    // The packet is 4000 bytes and not 1500, which matters: `TokenBucket::from_kbps`
    // floors the burst at `.max(1500)` -- one MTU -- so a fresh bucket ALWAYS admits
    // a single MTU-sized packet however low the MBR. A test that offered 1500 bytes
    // would therefore pass for an enforcer that policed nothing, which is the trap
    // this comment exists to keep the next reader out of.
    enforcer.configure_flow(QFI_GBR, 1, 8, 0);

    let (allowed, dscp) = enforcer
        .enforce(QFI_GBR, 4000)
        .expect("a configured flow must have a verdict");
    assert!(
        !allowed,
        "4000 bytes against an 8 kbit/s MBR (a 1500-byte burst floor) must be refused \
         -- otherwise the ceiling polices nothing and the excess reaches the air"
    );
    // The DSCP is resolved either way: policing decides whether the packet goes, and
    // the DSCP decides how the transport treats the ones that do. Compared against a
    // SEPARATE mapper rather than a literal, so the assertion tracks the 5QI table
    // instead of freezing one of its entries here.
    let expected = QfiDscpMapper::new().resolve(QFI_GBR, 1);
    assert_eq!(
        dscp, expected,
        "the DSCP must come from QfiDscpMapper's 5QI table (5QI 1), not from the QFI"
    );

    // An unlimited flow on the SAME enforcer passes a packet of the same size, so the
    // refusal above is the rate limiter's verdict and not a blanket denial.
    enforcer.configure_flow(QFI_NON_GBR, 9, 0, 0);
    let (allowed, non_gbr_dscp) = enforcer
        .enforce(QFI_NON_GBR, 4000)
        .expect("a configured flow must have a verdict");
    assert!(allowed, "an unlimited flow must pass the same packet size");
    // And the two flows resolve to DIFFERENT DSCPs, which is what makes the marking
    // per-flow rather than per-session -- the whole point of criterion 3.
    assert_ne!(
        dscp, non_gbr_dscp,
        "a GBR and a non-GBR flow must not resolve to the same DSCP, or the transport \
         cannot tell them apart"
    );
}
