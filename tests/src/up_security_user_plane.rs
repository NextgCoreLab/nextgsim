//! #32 — DRB user-plane security end to end, over the real RLS transport.
//!
//! Runs the REAL gNB and UE RLS tasks in one process over loopback UDP with real
//! cell discovery, and carries user data in both directions through PDCP ciphering
//! and integrity protection:
//!
//! ```text
//! gNB RlsMessage::DownlinkData  →  PDCP (SN, COUNT, MAC-I over header+data,
//!                                        cipher(data || MAC-I))
//!                               →  RLC  →  UDP (RLS)
//!                               →  UE RLC reassembly
//!                               →  UE PDCP (decipher, verify, reorder)
//!                               →  NasMessage::UplinkDataDelivery
//! ```
//!
//! **This file only compiles under `--features up-security`**, which implies
//! `drb-pdcp`: without a PDCP entity there is no COUNT to key on and nothing to
//! protect. The default `cargo test --workspace` therefore skips it and the
//! `up-security` CI job runs it.
//!
//! # What makes this a real end-to-end test
//!
//! Delivery alone would not prove the traffic was protected — an
//! `install_drb_security` that dropped the binding on the floor would deliver
//! everything just as happily. So the mismatch case is asserted too: a gNB that
//! protects against a UE that does not **must fail**, because that is only possible
//! if the protection reached the wire.

#![cfg(feature = "up-security")]

use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_common::config::{GnbConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, GtpMessage, RlsMessage as GnbRlsMessage, RrcMessage as GnbRrcMessage,
    TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RlsTask as GnbRlsTask;
use nextgsim_gnb::Task as GnbTask;
use nextgsim_pdcp::{PdcpSecurity, UpSecurity, DIRECTION_DOWNLINK, DIRECTION_UPLINK};
use nextgsim_ue::rls::{RlsTask as UeRlsTask, RlsTaskConfig};
use nextgsim_ue::{
    NasMessage, RlsMessage as UeRlsMessage, Task as UeTask, TaskMessage as UeTaskMessage,
    UeTaskBase,
};
use tokio::net::UdpSocket;
use tokio::sync::mpsc;

const PSI: i32 = 1;
/// The DRB identity `PSI`'s session uses. Equal to the PSI because a session's
/// DEFAULT DRB keeps the identity the PSI produced
/// (`nextgsim_gtp::qfi_drb::allocate_drbs`), so this suite describes the same bearer
/// with `sdap-dataplane` on or off (issue #44) — and `BEARER` below stays right.
const DRB_ID: i32 = PSI;
/// BEARER for DRB 1: the radio bearer identity minus one (TS 33.501 Annex D.3.1.2).
const BEARER: u8 = 0;
/// The QoS flow this session's traffic rides (issue #44). 5QI 9 is best-effort
/// non-GBR, so the policy maps it to the session's **default** DRB — `DRB_ID`, the
/// only bearer this suite establishes.
const QFI_NON_GBR: u8 = 9;
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);
const DELIVERY_TIMEOUT: Duration = Duration::from_secs(10);
/// Long enough for a PDU that WILL be delivered to arrive, short enough that a
/// negative assertion does not stall the suite.
const DISCARD_TIMEOUT: Duration = Duration::from_secs(2);

/// The keys both ends derive from one `KgNB`. Fixed rather than derived here: this
/// test is about the data path, and `nextgsim-ue`'s own tests pin that the two
/// endpoints' derivations agree.
const K_UP_ENC: [u8; 16] = [
    0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF, 0x00,
];
const K_UP_INT: [u8; 16] = [
    0xAB, 0xCD, 0xEF, 0x01, 0x23, 0x45, 0x67, 0x89, 0x98, 0x76, 0x54, 0x32, 0x10, 0xFE, 0xDC, 0xBA,
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
    gnb_gtp_rx: mpsc::Receiver<GnbTaskMessage<GtpMessage>>,
    ue_nas_rx: mpsc::Receiver<UeTaskMessage<NasMessage>>,
    ue_rrc_rx: mpsc::Receiver<UeTaskMessage<nextgsim_ue::RrcMessage>>,
}

async fn start_harness() -> Harness {
    let gnb_addr = free_loopback_addr().await;

    let (gnb_base, _gnb_app_rx, _gnb_ngap_rx, gnb_rrc_rx, gnb_gtp_rx, gnb_rls_rx, _gnb_sctp_rx) =
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
        gnb_gtp_rx,
        ue_nas_rx,
        ue_rrc_rx,
    }
}

/// Waits for mutual discovery and makes the found cell the UE's serving cell,
/// returning the UE id the gNB assigned.
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

fn binding(integrity_alg_id: Option<u8>, tx_direction: u8) -> Box<PdcpSecurity> {
    Box::new(PdcpSecurity {
        security: UpSecurity::new(K_UP_ENC, K_UP_INT, 2, integrity_alg_id).expect("legal ids"),
        bearer: BEARER,
        tx_direction,
    })
}

/// Keys both ends of the DRB, as NGAP and RRC would after an Initial Context Setup.
///
/// With `sdap-dataplane` this also installs the QoS-flow-to-DRB mapping, because the
/// same `RRCReconfiguration` that carries the DRB's security also carries its
/// `SDAP-Config` — and the UPLINK half of this suite does not work without it. The UE
/// stamps an uplink SDAP header only for a session it has a mapping for; with none it
/// sends the SDU unstamped (a deliberate choice, so a packet that beats its
/// reconfiguration is not lost), and a gNB built with the feature then discards it as
/// an undecodable header. That is the whole uplink, silently gone — so the mapping is
/// not decoration here, it is what makes the session sendable.
///
/// The downlink needs no such thing: the UE resolves a DRB back to a PSI via
/// `psi_for_drb`, which falls back to the DRB identity, and `DRB_ID == PSI` here.
#[cfg(feature = "sdap-dataplane")]
async fn key_both_ends(h: &Harness, ue_id: i32, integrity_alg_id: Option<u8>) {
    install_security(h, ue_id, integrity_alg_id).await;
    h.ue_rls_tx
        .send(UeRlsMessage::InstallSdapMapping {
            psi: PSI,
            drb_id: DRB_ID,
            qfis: vec![QFI_NON_GBR],
            // The session's default DRB, which is the bearer the UE's uplink rides:
            // it has no per-packet classifier to pick another with.
            default_drb: true,
        })
        .await
        .expect("UE RLS task alive");
}

/// Without the feature there is no mapping to install, so keying is all there is —
/// byte-identical to what this suite did before #44.
#[cfg(not(feature = "sdap-dataplane"))]
async fn key_both_ends(h: &Harness, ue_id: i32, integrity_alg_id: Option<u8>) {
    install_security(h, ue_id, integrity_alg_id).await;
}

/// The keying itself, shared by both arms of `key_both_ends`.
async fn install_security(h: &Harness, ue_id: i32, integrity_alg_id: Option<u8>) {
    h.gnb_rls_tx
        .send(GnbRlsMessage::InstallDrbSecurity {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            security: Some(binding(integrity_alg_id, DIRECTION_DOWNLINK)),
        })
        .await
        .expect("gNB RLS task alive");
    h.ue_rls_tx
        .send(UeRlsMessage::InstallDrbSecurity {
            psi: PSI,
            drb_id: DRB_ID,
            security: Some(binding(integrity_alg_id, DIRECTION_UPLINK)),
        })
        .await
        .expect("UE RLS task alive");
}

/// What the gNB's RLS task expects in `DownlinkData.pdu`.
///
/// # Why a test about PDCP security has to know about SDAP
///
/// `RlsMessage::DownlinkData` is the boundary BELOW the SDAP sublayer: on the live
/// path the gNB's GTP task has already prepended the one-octet header (TS 37.324
/// §6.2.2) by the time it sends one. So with `sdap-dataplane` the field carries an
/// SDAP Data PDU and without it a bare SDU, and a test that injects here is standing
/// in for the GTP task — it owes the same framing. SDAP sits ABOVE PDCP (§4.2), so
/// the header is inside what PDCP ciphers and integrity-protects, which is exactly
/// what makes this the right place for the test to add it: the sublayer under test
/// sees no difference.
///
/// Sending a bare SDU feature-on is not "testing the unprotected framing", it is
/// sending a malformed PDU that the UE correctly refuses to deliver. Both symptoms
/// showed up here: `0xD0`-filled and `b"protected"`-style payloads whose first octet
/// has bit 8 set decode as a "valid" header and arrive one octet short, while a
/// payload starting `0x00` (`(0..64u8).collect()`) reads as a Control PDU and is
/// discarded outright (issue #44).
///
/// Built with the SHIPPED encoder, so a change to the octet layout moves this helper
/// rather than leaving it encoding a stale format.
#[cfg(feature = "sdap-dataplane")]
fn framed(qfi: u8, sdu: &[u8]) -> Vec<u8> {
    use nextgsim_pdcp::sdap::{build_dl_pdu, SdapHeader};
    // RQI clear: reflective QoS (TS 23.501 §5.7.5.3) is orthogonal to user-plane
    // security, and setting it would have the UE log a flow property this suite never
    // established.
    build_dl_pdu(SdapHeader { qfi, rqi: false }, sdu).expect("a legal QFI")
}

/// Without the feature the field is the bare SDU, exactly as before #44 — so what
/// this suite puts on the wire is byte-identical to what it always did.
#[cfg(not(feature = "sdap-dataplane"))]
fn framed(_qfi: u8, sdu: &[u8]) -> Vec<u8> {
    sdu.to_vec()
}

async fn collect_at_ue(
    rx: &mut mpsc::Receiver<UeTaskMessage<NasMessage>>,
    count: usize,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { data, .. })) => {
                    out.push(data.data().to_vec());
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the UE, got {}", out.len()));
    out
}

async fn collect_at_gnb(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GtpMessage>>,
    count: usize,
) -> Vec<Vec<u8>> {
    let mut out = Vec::with_capacity(count);
    tokio::time::timeout(DELIVERY_TIMEOUT, async {
        while out.len() < count {
            match rx.recv().await {
                Some(GnbTaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    out.push(pdu.data().to_vec());
                }
                Some(_) => continue,
                None => panic!("gNB GTP channel closed after {} SDU(s)", out.len()),
            }
        }
    })
    .await
    .unwrap_or_else(|_| panic!("expected {count} SDU(s) at the gNB, got {}", out.len()));
    out
}

/// CRITERION 3 and 7: DRB traffic in both directions arrives intact through
/// ciphering and integrity protection, over the real transport.
#[tokio::test]
async fn protected_drb_traffic_survives_in_both_directions() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;
    key_both_ends(&h, ue_id, Some(2)).await;

    // ---- downlink: gNB -> UE -------------------------------------------
    let downlink: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0xD0 + i; 60]).collect();
    for payload in &downlink {
        h.gnb_rls_tx
            .send(GnbRlsMessage::DownlinkData {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                pdu: OctetString::from_slice(&framed(QFI_NON_GBR, payload)),
            })
            .await
            .expect("gNB RLS task alive");
    }
    assert_eq!(
        collect_at_ue(&mut h.ue_nas_rx, downlink.len()).await,
        downlink,
        "protected downlink SDUs must reach NAS intact and in order"
    );

    // ---- uplink: UE -> gNB ---------------------------------------------
    let uplink: Vec<Vec<u8>> = (0..5u8).map(|i| vec![0x50 + i; 45]).collect();
    for payload in &uplink {
        h.ue_rls_tx
            .send(UeRlsMessage::UplinkData {
                psi: PSI,
                data: OctetString::from_slice(payload),
            })
            .await
            .expect("UE RLS task alive");
    }
    assert_eq!(
        collect_at_gnb(&mut h.gnb_gtp_rx, uplink.len()).await,
        uplink,
        "protected uplink SDUs must reach GTP-U intact and in order"
    );
}

/// The assertion that makes the test above mean something: a gNB that protects
/// against a UE that does not **must not deliver**.
///
/// If the binding were dropped on the floor, this would deliver happily — so this is
/// the only test here that can tell "protected" from "the keys were ignored".
#[tokio::test]
async fn an_unprotected_peer_cannot_read_a_protected_drb() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;

    // Only the gNB is keyed. The UE's entity has no security, so it will read the
    // ciphertext as payload -- and there is no MAC to check, so it delivers
    // garbage rather than discarding. That is exactly why `up-security` is a
    // build-time feature both ends must share.
    h.gnb_rls_tx
        .send(GnbRlsMessage::InstallDrbSecurity {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            security: Some(binding(Some(2), DIRECTION_DOWNLINK)),
        })
        .await
        .expect("gNB RLS task alive");

    let payload = vec![0xA5u8; 60];
    h.gnb_rls_tx
        .send(GnbRlsMessage::DownlinkData {
            ue_id,
            psi: PSI,
            // Framed like every other injection here even though this test's
            // assertion is negative: what the mismatch must be shown to break is the
            // REAL downlink, and a bare SDU would let the test pass for the wrong
            // reason — the UE rejecting malformed framing rather than failing to
            // decipher.
            drb_id: DRB_ID,
            pdu: OctetString::from_slice(&framed(QFI_NON_GBR, &payload)),
        })
        .await
        .expect("gNB RLS task alive");

    let delivered = tokio::time::timeout(DISCARD_TIMEOUT, async {
        loop {
            match h.ue_nas_rx.recv().await {
                Some(UeTaskMessage::Message(NasMessage::UplinkDataDelivery { data, .. })) => {
                    return data.data().to_vec()
                }
                Some(_) => continue,
                None => panic!("UE NAS channel closed"),
            }
        }
    })
    .await;

    match delivered {
        Ok(got) => assert_ne!(
            got, payload,
            "an unkeyed UE must not recover the plaintext -- if it did, the gNB never \
             ciphered and this whole issue is a no-op"
        ),
        // Nothing delivered at all is also a pass: the PDU was unreadable.
        Err(_) => {}
    }
}

/// The other mismatch, and the dangerous one: a **UE** that protects against a gNB
/// that does not.
///
/// The gNB has no MAC to verify, so it hands the ciphertext to GTP-U as if it were an
/// IP packet. Asserted so the failure mode is a checked fact rather than a surprise
/// in the field — this is the reason the feature is not a runtime switch.
#[tokio::test]
async fn an_unprotected_gnb_does_not_recover_a_protected_uplink() {
    let mut h = start_harness().await;
    let _ue_id = complete_discovery(&mut h).await;

    h.ue_rls_tx
        .send(UeRlsMessage::InstallDrbSecurity {
            psi: PSI,
            drb_id: DRB_ID,
            security: Some(binding(Some(2), DIRECTION_UPLINK)),
        })
        .await
        .expect("UE RLS task alive");

    let payload = vec![0x5Au8; 50];
    h.ue_rls_tx
        .send(UeRlsMessage::UplinkData {
            psi: PSI,
            data: OctetString::from_slice(&payload),
        })
        .await
        .expect("UE RLS task alive");

    let delivered = tokio::time::timeout(DISCARD_TIMEOUT, async {
        loop {
            match h.gnb_gtp_rx.recv().await {
                Some(GnbTaskMessage::Message(GtpMessage::DataPduDelivery { pdu, .. })) => {
                    return pdu.data().to_vec()
                }
                Some(_) => continue,
                None => panic!("gNB GTP channel closed"),
            }
        }
    })
    .await;

    match delivered {
        Ok(got) => assert_ne!(
            got, payload,
            "an unkeyed gNB must not recover the plaintext, or the UE never ciphered"
        ),
        Err(_) => {}
    }
}

/// A ciphering-only DRB (integrity `not-needed`) carries no MAC-I, and the whole
/// payload still arrives.
///
/// The regression this guards: treating "no integrity" as NIA0 would append four
/// zero octets on transmit and strip four octets of user payload on receive, and a
/// test that only checked *delivery* would not notice.
#[tokio::test]
async fn a_ciphering_only_drb_delivers_the_whole_payload() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;
    key_both_ends(&h, ue_id, None).await;

    let payload: Vec<u8> = (0..64u8).collect();
    h.gnb_rls_tx
        .send(GnbRlsMessage::DownlinkData {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            pdu: OctetString::from_slice(&framed(QFI_NON_GBR, &payload)),
        })
        .await
        .expect("gNB RLS task alive");
    assert_eq!(
        collect_at_ue(&mut h.ue_nas_rx, 1).await,
        vec![payload],
        "every octet must arrive: a DRB without integrity has no MAC-I to strip"
    );
}

/// Removing the binding returns the DRB to the clear on both ends, which is what a
/// released and re-established session needs so it cannot inherit stale keys.
#[tokio::test]
async fn removing_the_binding_returns_the_drb_to_the_clear() {
    let mut h = start_harness().await;
    let ue_id = complete_discovery(&mut h).await;
    key_both_ends(&h, ue_id, Some(2)).await;

    h.gnb_rls_tx
        .send(GnbRlsMessage::DownlinkData {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            pdu: OctetString::from_slice(&framed(QFI_NON_GBR, b"protected")),
        })
        .await
        .expect("gNB RLS task alive");
    assert_eq!(
        collect_at_ue(&mut h.ue_nas_rx, 1).await,
        vec![b"protected".to_vec()]
    );

    for _ in 0..1 {
        h.gnb_rls_tx
            .send(GnbRlsMessage::InstallDrbSecurity {
                ue_id,
                psi: PSI,
                drb_id: DRB_ID,
                security: None,
            })
            .await
            .expect("gNB RLS task alive");
        h.ue_rls_tx
            .send(UeRlsMessage::InstallDrbSecurity {
                psi: PSI,
                drb_id: DRB_ID,
                security: None,
            })
            .await
            .expect("UE RLS task alive");
    }

    h.gnb_rls_tx
        .send(GnbRlsMessage::DownlinkData {
            ue_id,
            psi: PSI,
            drb_id: DRB_ID,
            pdu: OctetString::from_slice(&framed(QFI_NON_GBR, b"in-the-clear")),
        })
        .await
        .expect("gNB RLS task alive");
    assert_eq!(
        collect_at_ue(&mut h.ue_nas_rx, 1).await,
        vec![b"in-the-clear".to_vec()],
        "both ends must drop protection together, or the DRB is dead"
    );
}
