//! Demo D444A (nextgsim#477) through the UE's real task wiring: a network
//! Session-AMBR change reported above NAS (#454) reaches the semantic codec
//! task through the App task, and the UL feature stream drops from 128 to 40
//! features on the wire.
//!
//! Its own test binary: the demo state is process-global and installed once.
#![cfg(feature = "demo-d444a-ambr-adapt")]

use std::time::Duration;

use nextgsim_common::config::{SemanticStreamConfig, UeConfig};
use nextgsim_nas::messages::sm::IeSessionAmbr;
use nextgsim_semantic::rate_adapt::{EncoderId, FeatureDatagram};
use nextgsim_ue::tasks::{AppMessage, UeStatusUpdate, DEFAULT_CHANNEL_CAPACITY};
use nextgsim_ue::wiring::UeWiring;
use tokio::net::UdpSocket;

async fn next_keep(sink: &UdpSocket) -> (EncoderId, usize) {
    let mut buf = [0u8; 2048];
    let (len, _) = tokio::time::timeout(Duration::from_secs(20), sink.recv_from(&mut buf))
        .await
        .expect("a datagram within 20 s")
        .unwrap();
    let d = FeatureDatagram::decode(&buf[..len]).unwrap();
    (d.encoder, d.values.len())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_network_session_ambr_change_cuts_the_stream_to_keep_40() {
    let sink = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    nextgsim_ue::demo::init(
        "demo:\n  enabled: true\n  d444a-ambr-adapt:\n    enabled: true\n    mode: semantic\n",
    )
    .unwrap();
    let config = UeConfig {
        semantic_comm_enabled: true,
        semantic_stream: Some(SemanticStreamConfig {
            sink: sink.local_addr().unwrap(),
            device: None,
            psi: 1,
            rate_hz: 50,
            epoch_datagrams: 50,
            seed: 444,
            follow_session_ambr: true,
        }),
        ..UeConfig::default()
    };
    let wiring = UeWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
    let base = wiring.task_base();
    let (_manager, _nas_rx) = wiring.spawn(None);

    assert_eq!(next_keep(&sink).await, (EncoderId::Pca, 128));

    // What the NAS task sends the App task for a PDU SESSION MODIFICATION
    // COMMAND with Session-AMBR UL 64 kbps (unit 1, value 64), DL 2 Mbps.
    base.app_tx
        .send(AppMessage::StatusUpdate(
            UeStatusUpdate::SessionParametersChanged {
                psi: 1,
                session_ambr: IeSessionAmbr::new(6, 2, 1, 64),
            },
        ))
        .await
        .unwrap();

    let mut seen = Vec::new();
    for _ in 0..200 {
        let k = next_keep(&sink).await;
        seen.push(k.1);
        if k == (EncoderId::Pca, 40) {
            return;
        }
    }
    panic!("the stream never dropped to keep 40: {seen:?}");
}
