//! The demo's job exchange between the UE's AC and an EAS: one fixed 64-octet
//! UDP frame each way. **Non-normative stand-in:** the EAS "computes" by
//! sleeping `ceil(reqComp / avlComp)` ms, the same compute term eesd's
//! `measured-budget` ranking uses.
//!
//! A reply is the size of the request, so the listener cannot amplify. Frames
//! that are not exactly [`FRAME_LEN`] octets with the magic, version and kind
//! are dropped. Concurrent jobs are bounded ([`MAX_INFLIGHT`]); one beyond the
//! bound is dropped (the AC times out and counts a miss). The bound is
//! process-wide, not per source: anyone on the internal `dn` network can hold
//! every slot for up to [`MAX_COMPUTE_MS`] and so starve the demo's own job.
//! Accepted for a demo listener on an internal network (labelled).
//!
//! ```text
//!  0..4   magic "D443"        4  version 1        5  kind (1 job, 2 done)
//!  8..12  job number          12..16  reqComp (job) / compute ms (done)
//! 16..20  avlComp used (done) 20..60  easId, zero-padded (done)
//! ```

use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nextgsim_demo::demo_log;
use tokio::net::UdpSocket;
use tokio::sync::Semaphore;

use super::KEY;

/// The frame length, both ways.
pub const FRAME_LEN: usize = 64;
/// The job port when none is configured (TS 29.558 `EndPoint` has no port).
pub const DEFAULT_PORT: u16 = 9471;
/// The longest an `easId` may be in a reply frame.
pub const MAX_EAS_ID: usize = 40;
/// Jobs computed at once; more are dropped.
pub const MAX_INFLIGHT: usize = 64;
/// The longest a job may compute (a bound on what one datagram can cost).
pub const MAX_COMPUTE_MS: u32 = 10_000;

const MAGIC: &[u8; 4] = b"D443";
const VERSION: u8 = 1;
const KIND_JOB: u8 = 1;
const KIND_DONE: u8 = 2;

/// `ceil(req_comp / avl_comp)` ms, capped at [`MAX_COMPUTE_MS`]; `None` with
/// no compute available.
pub fn compute_ms(req_comp: u32, avl_comp: u32) -> Option<u32> {
    (avl_comp > 0).then(|| req_comp.div_ceil(avl_comp).min(MAX_COMPUTE_MS))
}

/// A job frame.
pub fn job_frame(job: u32, req_comp: u32) -> [u8; FRAME_LEN] {
    let mut f = [0u8; FRAME_LEN];
    f[0..4].copy_from_slice(MAGIC);
    f[4] = VERSION;
    f[5] = KIND_JOB;
    f[8..12].copy_from_slice(&job.to_be_bytes());
    f[12..16].copy_from_slice(&req_comp.to_be_bytes());
    f
}

/// A job, read from a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Job {
    pub job: u32,
    pub req_comp: u32,
}

/// A job's answer, read from a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Done {
    pub job: u32,
    pub compute_ms: u32,
    pub avl_comp: u32,
    pub eas_id: String,
}

fn header_ok(f: &[u8], kind: u8) -> bool {
    f.len() == FRAME_LEN && &f[0..4] == MAGIC && f[4] == VERSION && f[5] == kind
}

fn u32_at(f: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([f[at], f[at + 1], f[at + 2], f[at + 3]])
}

/// Parse a job frame.
pub fn parse_job(f: &[u8]) -> Option<Job> {
    header_ok(f, KIND_JOB).then(|| Job {
        job: u32_at(f, 8),
        req_comp: u32_at(f, 12),
    })
}

/// A done frame. An `eas_id` longer than [`MAX_EAS_ID`] is cut.
pub fn done_frame(d: &Done) -> [u8; FRAME_LEN] {
    let mut f = [0u8; FRAME_LEN];
    f[0..4].copy_from_slice(MAGIC);
    f[4] = VERSION;
    f[5] = KIND_DONE;
    f[8..12].copy_from_slice(&d.job.to_be_bytes());
    f[12..16].copy_from_slice(&d.compute_ms.to_be_bytes());
    f[16..20].copy_from_slice(&d.avl_comp.to_be_bytes());
    let id = d.eas_id.as_bytes();
    let n = id.len().min(MAX_EAS_ID);
    f[20..20 + n].copy_from_slice(&id[..n]);
    f
}

/// Parse a done frame.
pub fn parse_done(f: &[u8]) -> Option<Done> {
    if !header_ok(f, KIND_DONE) {
        return None;
    }
    let id = &f[20..20 + MAX_EAS_ID];
    let end = id.iter().position(|b| *b == 0).unwrap_or(MAX_EAS_ID);
    Some(Done {
        job: u32_at(f, 8),
        compute_ms: u32_at(f, 12),
        avl_comp: u32_at(f, 16),
        eas_id: String::from_utf8_lossy(&id[..end]).into_owned(),
    })
}

/// What the job listener has done.
#[derive(Debug, Default)]
pub struct JobStats {
    pub executed: AtomicU64,
    pub dropped: AtomicU64,
}

/// Serve jobs on `socket` as EAS `eas_id`, computing with the current
/// `avl_comp` (which applied load lowers), until the socket fails.
pub async fn serve(
    socket: UdpSocket,
    eas_id: String,
    avl_comp: Arc<AtomicU32>,
    stats: Arc<JobStats>,
) {
    let socket = Arc::new(socket);
    let slots = Arc::new(Semaphore::new(MAX_INFLIGHT));
    let mut buf = [0u8; 2048];
    loop {
        let Ok((n, from)) = socket.recv_from(&mut buf).await else {
            return;
        };
        let Some(job) = parse_job(&buf[..n]) else {
            stats.dropped.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        let Ok(slot) = slots.clone().try_acquire_owned() else {
            stats.dropped.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        let avl = avl_comp.load(Ordering::Relaxed);
        let Some(ms) = compute_ms(job.req_comp, avl) else {
            stats.dropped.fetch_add(1, Ordering::Relaxed);
            continue;
        };
        demo_log!(
            info,
            KEY,
            "EAS-EXEC job={} src={} eas={eas_id} reqComp={} avlComp={avl} compute={ms}ms",
            job.job,
            canonical(from.ip()),
            job.req_comp
        );
        stats.executed.fetch_add(1, Ordering::Relaxed);
        let reply = done_frame(&Done {
            job: job.job,
            compute_ms: ms,
            avl_comp: avl,
            eas_id: eas_id.clone(),
        });
        let socket = socket.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(u64::from(ms))).await;
            let _ = socket.send_to(&reply, from).await;
            drop(slot);
        });
    }
}

fn canonical(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6.to_canonical(),
        v4 => v4,
    }
}

/// A job's outcome at the AC.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Answer {
    pub done: Done,
    /// Send to answer, ms (rounded up).
    pub rtt_ms: u32,
}

/// A UDP socket toward `dst`, bound to `device` if given (`SO_BINDTODEVICE`),
/// connected so only `dst`'s datagrams are read. The local port is
/// ephemeral: this is a client socket, not a listener.
pub async fn client_socket(dst: SocketAddr, device: Option<&str>) -> std::io::Result<UdpSocket> {
    let local: SocketAddr = if dst.is_ipv4() {
        (std::net::Ipv4Addr::UNSPECIFIED, 0).into()
    } else {
        (std::net::Ipv6Addr::UNSPECIFIED, 0).into()
    };
    let socket = UdpSocket::bind(local).await?;
    if let Some(dev) = device {
        bind_to_device(&socket, dev)?;
    }
    socket.connect(dst).await?;
    Ok(socket)
}

#[cfg(any(target_os = "android", target_os = "fuchsia", target_os = "linux"))]
fn bind_to_device(socket: &UdpSocket, dev: &str) -> std::io::Result<()> {
    socket.bind_device(Some(dev.as_bytes()))
}

#[cfg(not(any(target_os = "android", target_os = "fuchsia", target_os = "linux")))]
fn bind_to_device(_socket: &UdpSocket, dev: &str) -> std::io::Result<()> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        format!("binding to {dev} needs SO_BINDTODEVICE (Linux)"),
    ))
}

/// Run job `job` at `dst` and wait up to `timeout` for its answer.
pub async fn run(
    dst: SocketAddr,
    device: Option<&str>,
    job: u32,
    req_comp: u32,
    timeout: Duration,
) -> Result<Answer, String> {
    let socket = client_socket(dst, device)
        .await
        .map_err(|e| format!("socket to {dst}: {e}"))?;
    let sent = Instant::now();
    socket
        .send(&job_frame(job, req_comp))
        .await
        .map_err(|e| format!("send to {dst}: {e}"))?;
    let deadline = tokio::time::Instant::from_std(sent) + timeout;
    let mut buf = [0u8; 2048];
    loop {
        let n = match tokio::time::timeout_at(deadline, socket.recv(&mut buf)).await {
            Err(_) => return Err(format!("no answer from {dst} in {timeout:?}")),
            Ok(Err(e)) => return Err(format!("receive from {dst}: {e}")),
            Ok(Ok(n)) => n,
        };
        if let Some(done) = parse_done(&buf[..n]).filter(|d| d.job == job) {
            let rtt = sent.elapsed();
            let rtt_ms = u32::try_from(rtt.as_micros().div_ceil(1000)).unwrap_or(u32::MAX);
            return Ok(Answer { done, rtt_ms });
        }
    }
}
