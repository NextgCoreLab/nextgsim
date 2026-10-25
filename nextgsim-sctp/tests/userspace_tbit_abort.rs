//! A userspace association must end on the T-bit ABORT a restarted peer
//! answers its next packet with (#411).
//!
//! RFC 9260 §8.4 rule 8: a packet that matches no association is answered
//! with an ABORT whose Verification Tag is the received packet's and whose T
//! bit is set (nextgcore #1142 makes the AMF do this). §8.5.1 B: the receiver
//! accepts it when the tag is its peer's. Before nextgcore #1234's sctp-proto
//! (66128cd) the endpoint routed such a packet by the tag it carried, found no
//! association, and dropped it: the gNB went on retransmitting until
//! Association.Max.Retrans.

use std::{net::SocketAddr, time::Duration};

use nextgsim_sctp::{SctpAssociation, SctpConfig, SctpServer, SctpServerConfig};
use tokio::{
    net::UdpSocket,
    task::JoinHandle,
    time::{sleep, timeout, Instant},
};

/// A userspace SCTP server (the AMF stand-in) at `addr`, served until aborted.
async fn server_at(addr: SocketAddr) -> (SocketAddr, JoinHandle<()>) {
    let mut server = SctpServer::bind(addr, SctpServerConfig::default())
        .await
        .expect("bind server");
    let bound = server.local_addr();
    let task = tokio::spawn(async move {
        loop {
            let _ = server.recv(Duration::from_millis(10)).await;
        }
    });
    (bound, task)
}

/// CRC32c (Castagnoli, reflected), as RFC 9260 Appendix A computes it.
fn crc32c(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in data {
        crc ^= u32::from(b);
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

/// The ABORT of RFC 9260 §8.4 rule 8 for the received `packet`: ports
/// swapped, the received Verification Tag, one ABORT chunk (type 6) with the
/// T bit (flags bit 0) and no causes, and the CRC32c in little-endian order
/// over the packet with a zero checksum field (Appendix A).
fn reflected_abort(packet: &[u8], tag_override: Option<u32>) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&packet[2..4]); // source port: the packet's destination
    out.extend_from_slice(&packet[0..2]); // destination port: its source
    match tag_override {
        Some(t) => out.extend_from_slice(&t.to_be_bytes()),
        None => out.extend_from_slice(&packet[4..8]),
    }
    out.extend_from_slice(&[0, 0, 0, 0]);
    out.extend_from_slice(&[6, 0x01, 0x00, 0x04]);
    let crc = crc32c(&out);
    out[8..12].copy_from_slice(&crc.to_le_bytes());
    out
}

/// Short RTOs and Association.Max.Retrans 3: a loss by retransmission takes
/// about 1.5 s, against which the ABORT's effect is measured.
fn fast_config() -> SctpConfig {
    SctpConfig {
        rto_initial_ms: 300,
        rto_min_ms: 300,
        rto_max_ms: 400,
        max_association_retrans: 3,
        ..SctpConfig::default()
    }
}

/// Poll `assoc` until it closes or `limit` passes; returns the time it took.
async fn poll_until_closed(assoc: &mut SctpAssociation, limit: Duration) -> Option<Duration> {
    let start = Instant::now();
    while start.elapsed() < limit {
        assoc.poll().await.expect("poll");
        while assoc.try_recv().expect("try_recv").is_some() {}
        if assoc.is_closed() {
            return Some(start.elapsed());
        }
        sleep(Duration::from_millis(5)).await;
    }
    None
}

/// The peer restarted at its address: the stand-in endpoint there answers the
/// association's next packet with `answer(packet)`.
async fn restarted_peer_answering(
    assoc: &mut SctpAssociation,
    first: JoinHandle<()>,
    addr: SocketAddr,
    answer: impl Fn(&[u8]) -> Vec<u8>,
) {
    first.abort();
    let _ = first.await;
    let restarted = UdpSocket::bind(addr).await.expect("rebind at the address");
    assoc.send(0, b"initial-ue-message").await.expect("send");
    let mut buf = [0u8; 2048];
    let (n, from) = timeout(Duration::from_secs(2), restarted.recv_from(&mut buf))
        .await
        .expect("the association's DATA reaches the restarted peer")
        .expect("recv");
    restarted
        .send_to(&answer(&buf[..n]), from)
        .await
        .expect("send the answer");
}

#[tokio::test]
async fn a_restarted_peers_t_bit_abort_ends_the_association_at_once() {
    let (addr, first) = server_at("127.0.0.1:0".parse().unwrap()).await;
    let mut assoc = timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, fast_config()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect");
    restarted_peer_answering(&mut assoc, first, addr, |p| reflected_abort(p, None)).await;

    let took = poll_until_closed(&mut assoc, Duration::from_secs(3))
        .await
        .expect("the T-bit ABORT ends the association");
    // Well before the first T3-rtx expiry (300 ms), which a loss needs four of
    assert!(took < Duration::from_millis(250), "{took:?}");
    assert!(assoc.send(0, b"x").await.is_err());
}

/// §8.5.1 B: a T-bit ABORT is accepted only under the peer's tag. One with
/// another tag is not the peer's answer and leaves the association to its
/// retransmissions.
#[tokio::test]
async fn a_t_bit_abort_under_a_foreign_tag_is_ignored() {
    let (addr, first) = server_at("127.0.0.1:0".parse().unwrap()).await;
    let mut assoc = timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, fast_config()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect");
    restarted_peer_answering(&mut assoc, first, addr, |p| {
        let tag = u32::from_be_bytes([p[4], p[5], p[6], p[7]]);
        reflected_abort(p, Some(tag ^ 0x5A5A_5A5A))
    })
    .await;
    assert_eq!(
        poll_until_closed(&mut assoc, Duration::from_millis(250)).await,
        None,
        "a forged tag does not end the association"
    );
}
