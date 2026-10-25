//! A userspace association must notice a peer that restarted at the same
//! address (#338).
//!
//! sctp-proto answers no out-of-the-blue packet and its T3-rtx timer never
//! fails, so DATA to a restarted peer was retransmitted forever and the
//! association stayed up. RFC 9260 §8.1: after Association.Max.Retrans
//! consecutive retransmissions with no answer the peer is unreachable and the
//! association is closed.

use std::{net::SocketAddr, time::Duration};

use nextgsim_sctp::{SctpAssociation, SctpConfig, SctpServer, SctpServerConfig};
use tokio::{
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

/// Short RTOs and Association.Max.Retrans 3, so a loss shows in about 1.5 s.
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
        sleep(Duration::from_millis(10)).await;
    }
    None
}

#[tokio::test]
async fn a_peer_restarted_at_the_same_address_loses_the_association() {
    let (addr, first) = server_at("127.0.0.1:0".parse().unwrap()).await;
    let mut assoc = timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, fast_config()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect");

    // A live peer answers: DATA is SACKed and nothing is lost, well past the
    // time the loss below takes.
    for _ in 0..5 {
        assoc.send(0, b"ng-setup").await.expect("send");
        assert_eq!(
            poll_until_closed(&mut assoc, Duration::from_millis(800)).await,
            None,
            "a live peer keeps the association"
        );
    }

    // The peer restarts in place: same address, a fresh endpoint.
    first.abort();
    let _ = first.await;
    let (again, _second) = server_at(addr).await;
    assert_eq!(again, addr);

    assoc.send(0, b"initial-ue-message").await.expect("send");
    let took = poll_until_closed(&mut assoc, Duration::from_secs(15))
        .await
        .expect("the association is lost after Association.Max.Retrans");
    // Lost on the 4th T3-rtx expiry, one more than Association.Max.Retrans, at
    // RTO 300 + 400 + 400 + 400 ms
    // (1.5 s); a loss after only 3 would be 1.1 s
    assert!(took >= Duration::from_millis(1400), "{took:?}");
    assert!(
        assoc.send(0, b"x").await.is_err(),
        "a lost association no longer sends"
    );
}
