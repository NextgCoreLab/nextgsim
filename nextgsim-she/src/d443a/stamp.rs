//! An RFC 8762 STAMP Session-Reflector: stateless, unauthenticated
//! (§4.3.1), the far end of upfd's N6 delay measurement (TS 29.244 §5.33.8.3).
//!
//! **Departure:** TS 23.501 §5.8.2.23 and TS 33.501 clause T.5 expect the
//! measurement protocol to be protected; unauthenticated STAMP is not. The
//! reflector therefore binds the internal `dn` (or pod) address only, and a
//! reflected packet is never larger than the test packet it answers, so it
//! cannot amplify.
//!
//! **No reflection loops.** Two reflectors must never answer each other: one
//! datagram spoofed from one to the other would otherwise bounce between them
//! at line rate. So the reflector answers only a Session-Sender test packet
//! (octets 24..44 are MBZ in RFC 8762 §4.2.1 and are what upfd sends, while
//! every reflection carries the sender's fields there), and never a source
//! port that is the STAMP port, its own port, or a privileged port (< 1024;
//! upfd's Session-Sender uses an ephemeral port).

use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use tokio::net::UdpSocket;

/// The STAMP well-known port (RFC 8762 §4.1), where upfd probes unless the
/// SMF provisions another.
pub const STAMP_PORT: u16 = 862;
/// An unauthenticated STAMP test packet, sent and reflected (RFC 8762
/// §4.2.1, §4.3.1): 44 octets.
pub const STAMP_LEN: usize = 44;
/// The reflector's Error Estimate: S = 0 (not synchronized), Z = 0 (NTP),
/// scale 0, multiplier 1 (RFC 4656 §4.1.2: never 0).
const ERROR_ESTIMATE: u16 = 0x0001;
/// The `Session-Sender TTL` written (labelled: not the received TTL).
const SENDER_TTL: u8 = 255;

/// The reflected packet for test packet `pkt`, received at `t2` and sent at
/// `t3` (NTP 64-bit), per RFC 8762 §4.3.1 (stateless: the reflector's
/// Sequence Number is the sender's). `None` for a datagram shorter than a
/// test packet, which is not answered.
pub fn reflect(pkt: &[u8], t2: u64, t3: u64) -> Option<[u8; STAMP_LEN]> {
    if !is_test_packet(pkt) {
        return None;
    }
    let mut r = [0u8; STAMP_LEN];
    // Sequence Number (stateless mode: copied), Timestamp T3, Error Estimate.
    r[0..4].copy_from_slice(&pkt[0..4]);
    r[4..12].copy_from_slice(&t3.to_be_bytes());
    r[12..14].copy_from_slice(&ERROR_ESTIMATE.to_be_bytes());
    // Receive Timestamp T2.
    r[16..24].copy_from_slice(&t2.to_be_bytes());
    // Session-Sender Sequence Number, Timestamp and Error Estimate.
    r[24..28].copy_from_slice(&pkt[0..4]);
    r[28..36].copy_from_slice(&pkt[4..12]);
    r[36..38].copy_from_slice(&pkt[12..14]);
    r[40] = SENDER_TTL;
    Some(r)
}

/// A Session-Sender test packet: at least 44 octets, with octets 24..44 MBZ
/// (RFC 8762 §4.2.1; upfd's `stamp_test_packet` leaves them zero). A
/// reflection carries the sender's Sequence Number, Timestamp and TTL there,
/// so it is never one.
pub fn is_test_packet(pkt: &[u8]) -> bool {
    pkt.len() >= STAMP_LEN && pkt[24..STAMP_LEN].iter().all(|b| *b == 0)
}

/// Whether a datagram from `from` may be answered by a reflector on
/// `own_port`: not from the STAMP port, not from its own port, not from a
/// privileged port. (A reflector sends from those; a Session-Sender does not.)
pub fn answers_source(from: SocketAddr, own_port: u16) -> bool {
    let p = from.port();
    p != STAMP_PORT && p != own_port && p >= 1024
}

/// What the reflector has done.
#[derive(Debug, Default)]
pub struct ReflectorStats {
    pub reflected: AtomicU64,
    pub dropped: AtomicU64,
}

/// Bind the reflector at `addr`.
pub async fn bind(addr: SocketAddr) -> std::io::Result<UdpSocket> {
    UdpSocket::bind(addr).await
}

/// Serve `socket` until it fails: every test packet is answered at once.
pub async fn run(socket: UdpSocket, stats: Arc<ReflectorStats>) {
    let own_port = socket.local_addr().map_or(STAMP_PORT, |a| a.port());
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = socket.recv_from(&mut buf).await else {
            return;
        };
        let t2 = super::ntp64(SystemTime::now());
        let reply = answers_source(from, own_port)
            .then(|| reflect(&buf[..n], t2, super::ntp64(SystemTime::now())))
            .flatten();
        match reply {
            Some(r) => {
                if socket.send_to(&r, from).await.is_ok() {
                    stats.reflected.fetch_add(1, Ordering::Relaxed);
                }
            }
            None => {
                stats.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}
