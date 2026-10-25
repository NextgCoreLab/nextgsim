//! A userspace association must notice a peer's graceful shutdown when it
//! happens, not on its next send (#411).
//!
//! RFC 9260 §9.2: the peer sends SHUTDOWN, the association answers with a
//! SHUTDOWN ACK, and the peer's SHUTDOWN COMPLETE ends it. sctp-proto closes
//! its association on that SHUTDOWN COMPLETE without an `AssociationLost`
//! event, so the association looked `Established` until a send failed with
//! "sending payload data in non-Established state". By then the gNB had lost
//! the UE's NAS message with it.

use std::{net::SocketAddr, time::Duration};

use nextgsim_sctp::{SctpAssociation, SctpConfig, SctpServer, SctpServerConfig, ServerEvent};
use tokio::{
    sync::mpsc,
    task::JoinHandle,
    time::{sleep, timeout, Instant},
};

/// A userspace SCTP server (the AMF stand-in) that shuts each association
/// down gracefully once `shutdown_after` has passed since it came up.
async fn server_shutting_down_after(shutdown_after: Duration) -> (SocketAddr, JoinHandle<()>) {
    let mut server = SctpServer::bind("127.0.0.1:0".parse().unwrap(), SctpServerConfig::default())
        .await
        .expect("bind server");
    let bound = server.local_addr();
    let (tx, mut rx) = mpsc::unbounded_channel();
    server.set_event_sender(tx);
    let task = tokio::spawn(async move {
        let mut due = Vec::new();
        loop {
            let _ = server.recv(Duration::from_millis(10)).await;
            while let Ok(event) = rx.try_recv() {
                if let ServerEvent::NewAssociation { association_id, .. } = event {
                    due.push((association_id, Instant::now() + shutdown_after));
                }
            }
            let now = Instant::now();
            for (id, _) in due.iter().filter(|(_, at)| *at <= now) {
                server.shutdown_association(*id).await.expect("SHUTDOWN");
            }
            due.retain(|(_, at)| *at > now);
        }
    });
    (bound, task)
}

/// Poll `assoc`, sending nothing, until it closes or `limit` passes.
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
async fn a_peers_shutdown_closes_the_association_with_nothing_sent() {
    let (addr, _server) = server_shutting_down_after(Duration::from_millis(500)).await;
    let mut assoc = timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, SctpConfig::default()),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect");
    assert!(assoc.is_established());

    // Idle: the UE has nothing to send. The peer's SHUTDOWN must still end it,
    // well inside the time a retransmission-based loss would take.
    let took = poll_until_closed(&mut assoc, Duration::from_secs(3))
        .await
        .expect("the peer's SHUTDOWN closes the association while idle");
    assert!(took < Duration::from_secs(2), "{took:?}");
    assert!(
        assoc.send(0, b"initial-ue-message").await.is_err(),
        "a shut-down association no longer sends"
    );
}
