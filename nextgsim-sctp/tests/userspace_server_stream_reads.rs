//! The userspace [`SctpServer`] must deliver every message on a stream, not only
//! the first.
//!
//! SCTP streams are long-lived (RFC 9260 §1.5.2): an NGAP or S1AP peer reuses
//! the same stream for every message of an association (stream 0 for
//! non-UE-associated signalling, TS 38.412 §7 / TS 36.412 §7). The server used
//! to read a stream only when `accept_stream()` first returned it, so the second
//! and every later message on that stream was never delivered.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use nextgsim_sctp::{
    SctpAssociation, SctpConfig, SctpServer, SctpServerConfig, ServerEvent, NGAP_PPID,
};
use tokio::{
    sync::mpsc,
    time::{sleep, timeout, Instant},
};

/// Received `(stream_id, payload)` pairs, in delivery order.
type Log = Arc<Mutex<Vec<(u16, Vec<u8>)>>>;

/// An SCTP server that records every message it delivers.
async fn spawn_recording_server() -> (std::net::SocketAddr, Log) {
    let mut server = SctpServer::bind(
        "127.0.0.1:0".parse().expect("addr"),
        SctpServerConfig::default(),
    )
    .await
    .expect("bind server");
    let addr = server.local_addr();
    let (tx, mut rx) = mpsc::unbounded_channel();
    server.set_event_sender(tx);
    let received: Log = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            if server.recv(Duration::from_millis(10)).await.is_err() {
                return;
            }
            while let Ok(ev) = rx.try_recv() {
                if let ServerEvent::DataReceived { message, .. } = ev {
                    log.lock()
                        .expect("server log")
                        .push((message.stream_id, message.data.to_vec()));
                }
            }
        }
    });
    (addr, received)
}

/// Wait until the server has delivered `n` messages, or 5 s pass.
async fn wait_for(log: &Log, n: usize) -> Vec<(u16, Vec<u8>)> {
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let got = log.lock().expect("server log").clone();
        if got.len() >= n || Instant::now() >= deadline {
            return got;
        }
        sleep(Duration::from_millis(10)).await;
    }
}

async fn connect(addr: std::net::SocketAddr) -> SctpAssociation {
    timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, SctpConfig::default()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect")
}

/// The messages delivered on `stream`, in order.
fn on_stream(got: &[(u16, Vec<u8>)], stream: u16) -> Vec<Vec<u8>> {
    got.iter()
        .filter(|(s, _)| *s == stream)
        .map(|(_, d)| d.clone())
        .collect()
}

#[tokio::test]
async fn every_message_on_one_stream_is_delivered_in_order() {
    let (addr, log) = spawn_recording_server().await;
    let mut client = connect(addr).await;

    for msg in [&b"first"[..], b"second", b"third"] {
        client
            .send_with_ppid(0, msg, NGAP_PPID)
            .await
            .expect("send");
    }

    let got = wait_for(&log, 3).await;
    assert_eq!(
        on_stream(&got, 0),
        vec![b"first".to_vec(), b"second".to_vec(), b"third".to_vec()],
        "all three messages on stream 0, in send order"
    );
}

#[tokio::test]
async fn interleaved_streams_each_deliver_every_message_in_order() {
    let (addr, log) = spawn_recording_server().await;
    let mut client = connect(addr).await;

    let plan: [(u16, &[u8]); 6] = [
        (0, b"a1"),
        (1, b"b1"),
        (0, b"a2"),
        (0, b"a3"),
        (1, b"b2"),
        (0, b"a4"),
    ];
    for (stream, msg) in plan {
        client
            .send_with_ppid(stream, msg, NGAP_PPID)
            .await
            .expect("send");
    }

    let got = wait_for(&log, plan.len()).await;
    assert_eq!(got.len(), plan.len(), "every message delivered: {got:?}");
    // SCTP orders within a stream, not across streams (RFC 9260 §6.6).
    assert_eq!(
        on_stream(&got, 0),
        vec![
            b"a1".to_vec(),
            b"a2".to_vec(),
            b"a3".to_vec(),
            b"a4".to_vec()
        ],
        "stream 0 in send order"
    );
    assert_eq!(
        on_stream(&got, 1),
        vec![b"b1".to_vec(), b"b2".to_vec()],
        "stream 1 in send order"
    );
}

#[tokio::test]
async fn a_message_sent_after_a_pause_on_an_accepted_stream_is_delivered() {
    // The case the old read loop could never reach: the stream is already
    // accepted, so `accept_stream()` returns nothing when the next message lands.
    let (addr, log) = spawn_recording_server().await;
    let mut client = connect(addr).await;

    client
        .send_with_ppid(0, b"setup", NGAP_PPID)
        .await
        .expect("send");
    assert_eq!(wait_for(&log, 1).await.len(), 1, "first message delivered");

    sleep(Duration::from_millis(100)).await;
    client
        .send_with_ppid(0, b"later", NGAP_PPID)
        .await
        .expect("send");

    let got = wait_for(&log, 2).await;
    assert_eq!(
        on_stream(&got, 0),
        vec![b"setup".to_vec(), b"later".to_vec()],
        "the later message on the same stream is delivered too"
    );
}

/// CRC32c (Castagnoli, reflected 0x82F63B78) over an SCTP packet whose checksum
/// field is zeroed, stored little-endian at offset 8 (RFC 9260 Appendix B).
fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0x82F6_3B78
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Whether a packet's chunks are all DATA chunks (type 0).
fn only_data_chunks(packet: &[u8]) -> bool {
    let mut off = 12;
    let mut any = false;
    while off + 4 <= packet.len() {
        let len = u16::from_be_bytes([packet[off + 2], packet[off + 3]]) as usize;
        if len < 4 || off + len > packet.len() || packet[off] != 0 {
            return false;
        }
        any = true;
        off += (len + 3) & !3;
    }
    any && off == packet.len()
}

/// A UDP relay that holds back client→server DATA-only packets until it has
/// `bundle` of them, then forwards them as ONE SCTP packet (same common header,
/// chunks concatenated, checksum recomputed). Everything else passes through.
async fn spawn_bundling_relay(server: std::net::SocketAddr, bundle: usize) -> std::net::SocketAddr {
    let sock = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("bind relay");
    let addr = sock.local_addr().expect("relay addr");
    tokio::spawn(async move {
        let mut client: Option<std::net::SocketAddr> = None;
        let mut held: Vec<Vec<u8>> = Vec::new();
        let mut buf = vec![0u8; 65536];
        loop {
            let Ok((n, from)) = sock.recv_from(&mut buf).await else {
                return;
            };
            let pkt = buf[..n].to_vec();
            if from == server {
                if let Some(c) = client {
                    let _ = sock.send_to(&pkt, c).await;
                }
                continue;
            }
            client = Some(from);
            if !only_data_chunks(&pkt) {
                let _ = sock.send_to(&pkt, server).await;
                continue;
            }
            held.push(pkt);
            if held.len() < bundle {
                continue;
            }
            let mut merged = held[0][..12].to_vec();
            for p in held.drain(..) {
                merged.extend_from_slice(&p[12..]);
            }
            merged[8..12].copy_from_slice(&[0, 0, 0, 0]);
            let sum = crc32c(&merged);
            merged[8..12].copy_from_slice(&sum.to_le_bytes());
            let _ = sock.send_to(&merged, server).await;
        }
    });
    addr
}

#[tokio::test]
async fn messages_bundled_in_one_packet_on_one_stream_are_all_delivered() {
    // RFC 9260 §6.10 lets a sender bundle several DATA chunks into one packet.
    // A read loop that takes one message per received datagram leaves the rest
    // queued in the stream until some unrelated packet arrives.
    let (server_addr, log) = spawn_recording_server().await;
    let relay = spawn_bundling_relay(server_addr, 3).await;
    let mut client = connect(relay).await;

    for msg in [&b"one"[..], b"two", b"three"] {
        client
            .send_with_ppid(0, msg, NGAP_PPID)
            .await
            .expect("send");
    }

    let got = wait_for(&log, 3).await;
    assert_eq!(
        on_stream(&got, 0),
        vec![b"one".to_vec(), b"two".to_vec(), b"three".to_vec()],
        "all three bundled messages delivered from the one packet"
    );
}

#[test]
fn crc32c_matches_the_rfc_3720_check_value() {
    // RFC 3720 B.4: 32 octets of zeros -> 0x8A9136AA.
    assert_eq!(crc32c(&[0u8; 32]), 0x8A91_36AA);
}
