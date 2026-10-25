//! The userspace association's outbound stream count is the one RFC 9260 §5.1.1
//! negotiates: this endpoint's requested OS, lowered to the peer's MIS (the
//! number of inbound streams the peer accepts). A stream identifier at or above
//! it is invalid at the peer (§6.5). nextgsim #335: the vendored sctp-proto
//! bounded it by the peer's OS instead, and the gNB used its configured count.

use std::time::Duration;

use nextgsim_sctp::{SctpAssociation, SctpConfig, SctpServer, SctpServerConfig};
use tokio::time::timeout;

/// A userspace SCTP server advertising `mis` inbound and `os` outbound streams,
/// driven in the background so the INIT / INIT ACK handshake completes.
async fn spawn_server(mis: u16, os: u16) -> std::net::SocketAddr {
    let mut server = SctpServer::bind(
        "127.0.0.1:0".parse().expect("addr"),
        SctpServerConfig {
            max_inbound_streams: mis,
            max_outbound_streams: os,
            ..SctpServerConfig::default()
        },
    )
    .await
    .expect("bind server");
    let addr = server.local_addr();
    tokio::spawn(async move { while server.recv(Duration::from_millis(10)).await.is_ok() {} });
    addr
}

async fn connect(addr: std::net::SocketAddr, os: u16, mis: u16) -> SctpAssociation {
    let config = SctpConfig {
        max_outbound_streams: os,
        max_inbound_streams: mis,
        ..SctpConfig::default()
    };
    timeout(
        Duration::from_secs(5),
        SctpAssociation::connect(addr, config),
    )
    .await
    .expect("handshake within 5 s")
    .expect("connect")
}

/// The AMF's MIS (2) is below both its own OS (16) and the gNB's OS (8): the gNB
/// may use 2 outbound streams. Before #335 it used min(8, 16) = 8.
#[tokio::test]
async fn the_outbound_stream_count_is_bounded_by_the_peers_mis() {
    let addr = spawn_server(2, 16).await;
    let client = connect(addr, 8, 8).await;
    assert_eq!(client.outbound_streams(), 2);
}

/// A peer MIS above the requested OS leaves the requested OS.
#[tokio::test]
async fn a_larger_peer_mis_keeps_the_requested_outbound_count() {
    let addr = spawn_server(10, 1).await;
    let client = connect(addr, 8, 8).await;
    assert_eq!(client.outbound_streams(), 8);
}
