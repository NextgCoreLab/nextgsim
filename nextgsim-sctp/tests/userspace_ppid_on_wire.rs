//! The userspace (sctp-proto over UDP) backend must put the caller's PPID on the
//! wire and report the true PPID on receive (#220).
//!
//! A UDP relay sits between an [`SctpAssociation`] (the gNB/eNB side) and an
//! [`SctpServer`] (the AMF/MME side) and records the Payload Protocol Identifier
//! of every DATA chunk it forwards, in each direction. The server echoes each
//! message back with the PPID it received, so one exchange checks the client's
//! send, the server's receive, the server's send and the client's receive.
//!
//! Before #220, sctp-proto mapped every PPID outside the five WebRTC values to
//! `PayloadProtocolIdentifier::Unknown`, which serialises as 58: NGAP (60, TS
//! 38.412 §7) and S1AP (18, TS 36.412 §7) both went out as 58, and the receive
//! side reported NGAP_PPID for anything it could not name.

use std::{
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::Duration,
};

use nextgsim_sctp::{
    SctpAssociation, SctpConfig, SctpServer, SctpServerConfig, ServerEvent, NGAP_PPID, S1AP_PPID,
};
use tokio::{net::UdpSocket, sync::mpsc, time::timeout};

/// Direction of a DATA chunk seen by the relay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    ClientToServer,
    ServerToClient,
}

/// PPIDs of the DATA chunks (type 0) in one SCTP packet (RFC 9260 §3.3.1).
fn data_chunk_ppids(packet: &[u8]) -> Vec<u32> {
    let mut ppids = Vec::new();
    // 12-octet common header, then chunks padded to 4 octets.
    let mut off = 12;
    while off + 4 <= packet.len() {
        let ty = packet[off];
        let len = u16::from_be_bytes([packet[off + 2], packet[off + 3]]) as usize;
        if len < 4 || off + len > packet.len() {
            break;
        }
        // DATA: type, flags, length, TSN (4), stream id (2), SSN (2), PPID (4).
        if ty == 0 && len >= 16 {
            ppids.push(u32::from_be_bytes([
                packet[off + 12],
                packet[off + 13],
                packet[off + 14],
                packet[off + 15],
            ]));
        }
        off += (len + 3) & !3;
    }
    ppids
}

/// A UDP relay recording every DATA chunk PPID it forwards.
async fn spawn_relay(server: SocketAddr) -> (SocketAddr, Arc<Mutex<Vec<(Dir, u32)>>>) {
    let sock = UdpSocket::bind("127.0.0.1:0").await.expect("bind relay");
    let addr = sock.local_addr().expect("relay addr");
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    tokio::spawn(async move {
        let mut client: Option<SocketAddr> = None;
        let mut buf = vec![0u8; 65536];
        loop {
            let Ok((n, from)) = sock.recv_from(&mut buf).await else {
                return;
            };
            let pkt = &buf[..n];
            let (dir, to) = if from == server {
                match client {
                    Some(c) => (Dir::ServerToClient, c),
                    None => continue,
                }
            } else {
                client = Some(from);
                (Dir::ClientToServer, server)
            };
            for ppid in data_chunk_ppids(pkt) {
                log.lock().expect("relay log").push((dir, ppid));
            }
            let _ = sock.send_to(pkt, to).await;
        }
    });
    (addr, seen)
}

/// An SCTP server that echoes every message on its stream with the PPID it
/// reported on receive, and records those received PPIDs.
async fn spawn_echo_server() -> (SocketAddr, Arc<Mutex<Vec<u32>>>) {
    let mut server = SctpServer::bind(
        "127.0.0.1:0".parse().expect("addr"),
        SctpServerConfig::default(),
    )
    .await
    .expect("bind server");
    let addr = server.local_addr();
    let (tx, mut rx) = mpsc::unbounded_channel();
    server.set_event_sender(tx);
    let received = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&received);
    tokio::spawn(async move {
        loop {
            if server.recv(Duration::from_millis(10)).await.is_err() {
                return;
            }
            while let Ok(ev) = rx.try_recv() {
                if let ServerEvent::DataReceived {
                    association_id,
                    message,
                } = ev
                {
                    log.lock().expect("server log").push(message.ppid);
                    let _ = server
                        .send_with_ppid(
                            association_id,
                            message.stream_id,
                            &message.data,
                            message.ppid,
                        )
                        .await;
                }
            }
        }
    });
    (addr, received)
}

/// Send `data` on `stream` with `ppid` and wait for the echo. Each message uses
/// its own stream so the two PPIDs are checked independently; repeated messages
/// on one stream are covered by `userspace_server_stream_reads.rs`.
async fn round_trip(
    client: &mut SctpAssociation,
    stream: u16,
    ppid: u32,
    data: &'static [u8],
) -> u32 {
    client
        .send_with_ppid(stream, data, ppid)
        .await
        .expect("send");
    let msg = timeout(Duration::from_secs(5), client.recv())
        .await
        .expect("echo within 5 s")
        .expect("recv")
        .expect("association open");
    assert_eq!(&msg.data[..], data, "the echo carries the same payload");
    msg.ppid
}

#[tokio::test]
async fn ngap_and_s1ap_ppids_are_carried_verbatim_both_ways() {
    let (server_addr, server_received) = spawn_echo_server().await;
    let (relay_addr, wire) = spawn_relay(server_addr).await;

    let mut client = timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(relay_addr, SctpConfig::default()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect");

    let ngap_echo = round_trip(&mut client, 0, NGAP_PPID, b"ngap").await;
    let s1ap_echo = round_trip(&mut client, 1, S1AP_PPID, b"s1ap").await;

    // Receive side, both ends.
    assert_eq!(
        *server_received.lock().expect("server log"),
        vec![NGAP_PPID, S1AP_PPID],
        "the server must report the PPID the client sent"
    );
    assert_eq!(
        ngap_echo, NGAP_PPID,
        "the client must report the NGAP PPID it received"
    );
    assert_eq!(
        s1ap_echo, S1AP_PPID,
        "the client must report the S1AP PPID it received"
    );

    // On the wire, both directions: 60 and 18, never 58.
    let wire = wire.lock().expect("relay log").clone();
    let c2s: Vec<u32> = wire
        .iter()
        .filter(|(d, _)| *d == Dir::ClientToServer)
        .map(|(_, p)| *p)
        .collect();
    let s2c: Vec<u32> = wire
        .iter()
        .filter(|(d, _)| *d == Dir::ServerToClient)
        .map(|(_, p)| *p)
        .collect();
    // Retransmissions may repeat a chunk; dedup consecutive repeats.
    let dedup = |mut v: Vec<u32>| {
        v.dedup();
        v
    };
    assert_eq!(
        dedup(c2s),
        vec![60, 18],
        "client -> server DATA chunk PPIDs on the wire"
    );
    assert_eq!(
        dedup(s2c),
        vec![60, 18],
        "server -> client DATA chunk PPIDs on the wire"
    );
}

#[test]
fn the_relay_parser_reads_the_ppid_field() {
    // Common header (12) + one DATA chunk: type 0, flags 3, length 20, TSN 1,
    // stream 0, SSN 0, PPID 60, 4 octets of user data.
    let mut pkt = vec![0u8; 12];
    pkt.extend_from_slice(&[
        0, 3, 0, 20, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 60, b'n', b'g', b'a', b'p',
    ]);
    assert_eq!(data_chunk_ppids(&pkt), vec![60]);
}
