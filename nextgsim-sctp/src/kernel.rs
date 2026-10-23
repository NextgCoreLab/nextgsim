//! Kernel SCTP transport backend (Linux + lksctp/libsctp).
//!
//! This module provides [`KernelSctpAssociation`], a one-to-one (SOCK_STREAM)
//! SCTP client that speaks **real** SCTP on the wire (IP protocol 132) via the
//! Linux kernel SCTP stack and the lksctp `libsctp` helper library. This is the
//! backend that lets the nextgsim gNB associate with a real / Open5GS AMF;
//! the default userspace `sctp-proto`-over-UDP backend only interoperates with
//! the matching simulator.
//!
//! # Availability
//!
//! The whole module is gated `#[cfg(all(target_os = "linux", feature =
//! "kernel-sctp"))]`. It cannot be compiled or tested on macOS — it is written
//! to be compiled in a Linux Docker container and exercised in the exit gate.
//!
//! # Build / link requirements (Linux)
//!
//! * The kernel must have SCTP support (module `sctp`; `modprobe sctp`).
//! * `libsctp` must be available at link time. On Debian/Ubuntu install
//!   `libsctp-dev` (which provides `-lsctp` and `<netinet/sctp.h>`). The
//!   `#[link(name = "sctp")]` attribute on the `extern "C"` block below pulls
//!   in `-lsctp`.
//!
//! # PPID byte order (critical)
//!
//! NGAP requires PPID 60 on the wire (TS 38.412). lksctp's `sctp_sendmsg`
//! takes the PPID argument and places it on the wire **verbatim**, and the
//! field is defined to be in network byte order. Open5GS therefore calls
//! `htobe32(ppid)` before `sctp_sendmsg`. We mirror that exactly: we pass
//! `NGAP_PPID.to_be()` to `sctp_sendmsg`, and on receive we read
//! `sinfo.sinfo_ppid` (which the kernel hands back in network byte order) and
//! convert it with `u32::from_be` before comparing it against the PPID the
//! caller said the association carries (`connect`'s `expected_ppid`).

use std::io;
use std::net::SocketAddr;
use std::os::fd::{AsRawFd, RawFd};

use bytes::Bytes;
use libc::{c_int, c_void, size_t, sockaddr, socklen_t};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use tokio::io::unix::AsyncFd;
use tracing::{debug, info, trace, warn};

use crate::association::{ReceivedMessage, Result, SctpConfig, SctpError, NGAP_PPID};

// ===========================================================================
// libsctp (lksctp-tools) FFI
// ===========================================================================

// `sctp_sndrcvinfo` and `sctp_initmsg` are used straight from `libc` (#209):
// they were hand-transcribed here, independently of nextgcore's own copy, and
// comparing the two transcriptions is how nextgcore #429's unsigned
// `sinfo_assoc_id` was found. `sinfo_ppid` still arrives in network byte order
// (see the module-level note); that is a property of the kernel, not the type.

/// A zeroed `sctp_sndrcvinfo` out-parameter for `sctp_recvmsg`. `libc`'s
/// struct does not implement `Default`, so it is spelled out.
fn empty_sndrcvinfo() -> libc::sctp_sndrcvinfo {
    libc::sctp_sndrcvinfo {
        sinfo_stream: 0,
        sinfo_ssn: 0,
        sinfo_flags: 0,
        sinfo_ppid: 0,
        sinfo_context: 0,
        sinfo_timetolive: 0,
        sinfo_tsn: 0,
        sinfo_cumtsn: 0,
        sinfo_assoc_id: 0,
    }
}

/// lksctp `struct sctp_event_subscribe` (from `<netinet/sctp.h>`), used with
/// the `SCTP_EVENTS` socket option. `libc` does not define it, so it is the one
/// SCTP struct still transcribed; `event_subscribe_layout_matches_the_kernel`
/// pins it. Every field is a `__u8` flag (0/1). The
/// struct has grown across kernel versions; the kernel copies
/// `min(optlen, sizeof(kernel struct))` bytes and leaves the rest at its
/// default, so passing this stable prefix (through `sctp_stream_change_event`)
/// is forward-compatible. We only enable the three we depend on:
/// `sctp_data_io_event` (fills `sctp_sndrcvinfo` on recv — the PPID/stream),
/// and `sctp_association_event` / `sctp_shutdown_event` (so
/// [`KernelSctpAssociation::handle_notification_state`] can observe
/// COMM_LOST / SHUTDOWN and tear the association down).
///
/// ```c
/// struct sctp_event_subscribe {
///     __u8 sctp_data_io_event;
///     __u8 sctp_association_event;
///     __u8 sctp_address_event;
///     __u8 sctp_send_failure_event;
///     __u8 sctp_peer_error_event;
///     __u8 sctp_shutdown_event;
///     __u8 sctp_partial_delivery_event;
///     __u8 sctp_adaptation_layer_event;
///     __u8 sctp_authentication_event;
///     __u8 sctp_sender_dry_event;
///     __u8 sctp_stream_reset_event;
///     __u8 sctp_assoc_reset_event;
///     __u8 sctp_stream_change_event;
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SctpEventSubscribe {
    sctp_data_io_event: u8,
    sctp_association_event: u8,
    sctp_address_event: u8,
    sctp_send_failure_event: u8,
    sctp_peer_error_event: u8,
    sctp_shutdown_event: u8,
    sctp_partial_delivery_event: u8,
    sctp_adaptation_layer_event: u8,
    sctp_authentication_event: u8,
    sctp_sender_dry_event: u8,
    sctp_stream_reset_event: u8,
    sctp_assoc_reset_event: u8,
    sctp_stream_change_event: u8,
}

// SCTP socket-option level and names. `libc` defines `SOL_SCTP` for Android
// only, not Linux, so it stays local; the option names come from `libc` (#209).
/// `IPPROTO_SCTP` socket option level for `setsockopt` (== 132).
const SOL_SCTP: c_int = 132;
// `SCTP_INITMSG` requests outbound/inbound stream counts. `SCTP_EVENTS`
// subscribes to SCTP events. Critically, enabling `sctp_data_io_event` makes
// the kernel attach the legacy `SCTP_SNDRCV` ancillary message that lksctp's
// `sctp_recvmsg()` parses to fill `sctp_sndrcvinfo` (PPID + stream). This is
// the same setup Open5GS uses (`ogs_sctp_socket`). The newer
// `SCTP_RECVRCVINFO` delivers a *different* cmsg (`SCTP_RCVINFO`) that
// `sctp_recvmsg()` does not read, which left the received PPID/stream zeroed.
use libc::{SCTP_EVENTS, SCTP_INITMSG};

// libsctp helper functions. We link against `-lsctp` (libsctp-dev). These are
// the classic one-to-one helpers that carry the PPID/stream out-of-band so we
// do not have to assemble `cmsg` ancillary data by hand.
#[link(name = "sctp")]
extern "C" {
    /// `sctp_sendmsg` — send `len` bytes on `sd` with the given `ppid` (network
    /// byte order), `stream_no`, and SCTP flags. `to`/`tolen` may be NULL/0 on
    /// a connected one-to-one socket.
    // Both helpers return `int` in <netinet/sctp.h>, NOT `ssize_t` (#243).
    // Declaring `ssize_t` reads the whole of `rax`, whose upper half the
    // callee need not set: with Debian bookworm's libsctp1 a `-1`/EAGAIN came
    // back as 4294967295, was taken for a full-buffer message with a zeroed
    // sndrcvinfo (PPID 0), and the receive loop spun on it at 100% CPU.
    fn sctp_sendmsg(
        sd: c_int,
        msg: *const c_void,
        len: size_t,
        to: *const sockaddr,
        tolen: socklen_t,
        ppid: u32,
        flags: u32,
        stream_no: u16,
        timetolive: u32,
        context: u32,
    ) -> c_int;

    /// `sctp_recvmsg` — receive up to `len` bytes into `msg`, filling `sinfo`
    /// (stream, ppid in network byte order, flags, ...) and `msg_flags`.
    fn sctp_recvmsg(
        sd: c_int,
        msg: *mut c_void,
        len: size_t,
        from: *mut sockaddr,
        fromlen: *mut socklen_t,
        sinfo: *mut libc::sctp_sndrcvinfo,
        msg_flags: *mut c_int,
    ) -> c_int;
}

/// What one `sctp_recvmsg` produced (see `recv_step`).
enum RecvStep {
    /// A whole user message.
    Message(ReceivedMessage),
    /// A notification or a partial delivery was consumed; read again.
    Pending,
    /// The association has ended (orderly shutdown, SHUTDOWN or COMM_LOST).
    Closed,
}

/// The out-parameters of one `sctp_recvmsg`.
struct RawRecv {
    buf: Vec<u8>,
    n: usize,
    sinfo: libc::sctp_sndrcvinfo,
    msg_flags: c_int,
}

/// One `sctp_recvmsg` on `fd` into a `cap`-byte buffer. `Err` carries errno,
/// so EAGAIN surfaces as [`io::ErrorKind::WouldBlock`] — never as data.
fn raw_recvmsg(fd: RawFd, cap: usize) -> io::Result<RawRecv> {
    let mut buf = vec![0u8; cap];
    let mut sinfo = empty_sndrcvinfo();
    let mut msg_flags: c_int = 0;
    // SAFETY: `fd` is a valid SCTP socket; `buf` has `cap` bytes;
    // `sinfo`/`msg_flags` are valid out-params; from/fromlen NULL (no peer
    // address is needed on a connected one-to-one socket).
    let rc = unsafe {
        sctp_recvmsg(
            fd,
            buf.as_mut_ptr().cast::<c_void>(),
            cap as size_t,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::addr_of_mut!(sinfo),
            std::ptr::addr_of_mut!(msg_flags),
        )
    };
    if rc < 0 {
        return Err(io::Error::last_os_error());
    }
    let n = usize::try_from(rc).map_or(0, |n| n.min(cap));
    Ok(RawRecv {
        buf,
        n,
        sinfo,
        msg_flags,
    })
}

// ===========================================================================
// Expected-PPID check
// ===========================================================================

/// The PPID one association is meant to carry, and how many received messages
/// carried another.
///
/// An association runs one application protocol: NGAP (PPID 60, TS 38.412 §7)
/// for a gNB, S1AP (PPID 18, TS 36.412 §7) for an eNB. The caller states which,
/// so a mismatch is reported against the protocol actually in use rather than
/// against an assumed NGAP — the eNB used to log `non-NGAP PPID 18` for every
/// conformant S1AP message. The message is still delivered with the PPID it
/// carried; the warning is the diagnostic, and the count makes it observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PpidCheck {
    expected: u32,
    unexpected: u64,
}

impl PpidCheck {
    fn new(expected: u32) -> Self {
        Self {
            expected,
            unexpected: 0,
        }
    }

    /// Record a received PPID; warn and count it if it is not the expected one.
    /// Returns whether it matched.
    fn observe(&mut self, ppid: u32, stream_id: u16) -> bool {
        if ppid == self.expected {
            return true;
        }
        self.unexpected += 1;
        warn!(
            "Kernel SCTP received PPID {} on stream {} (this association expects {})",
            ppid, stream_id, self.expected
        );
        false
    }
}

// ===========================================================================
// KernelSctpAssociation
// ===========================================================================

/// A one-to-one kernel SCTP association to an AMF.
///
/// Mirrors the surface of `crate::association::SctpAssociation` that the gNB
/// dispatches through: [`connect`](Self::connect), [`send_with_ppid`](Self::send_with_ppid),
/// [`send`](Self::send), [`recv`](Self::recv), [`local_addr`](Self::local_addr),
/// [`remote_addr`](Self::remote_addr), [`is_established`](Self::is_established),
/// and [`close`](Self::close).
pub struct KernelSctpAssociation {
    /// Non-blocking SCTP socket wrapped for the tokio reactor. `AsyncFd` does
    /// not own/close the fd in a way that conflicts with the `Socket`; the
    /// `Socket` below is the owner and closes the fd on drop.
    async_fd: AsyncFd<RawFd>,
    /// Owns the file descriptor lifetime (closes it on drop).
    socket: Socket,
    /// Cached raw fd for the FFI `sctp_sendmsg`/`sctp_recvmsg` calls. Same value
    /// as `socket.as_raw_fd()`; cached to avoid any `AsRawFd` trait ambiguity
    /// between `RawFd` and `AsyncFd<RawFd>`.
    fd: RawFd,
    local_addr: SocketAddr,
    remote_addr: SocketAddr,
    /// Set once `connect` returns, which it does only after the 4-way handshake
    /// completed, so this is true after `connect`.
    established: bool,
    /// Maximum receive buffer size from config (per-recv allocation cap).
    max_recv: usize,
    /// The pieces of a partially delivered message received so far, with
    /// its PPID and stream (RFC 6458 §8.1.20), until the piece with MSG_EOR.
    partial: Option<(Vec<u8>, u32, u16)>,
    /// The PPID this association carries, as stated by the caller.
    ppid_check: PpidCheck,
}

impl KernelSctpAssociation {
    /// Establish a one-to-one kernel SCTP association from `local` to `remote`.
    ///
    /// Steps:
    /// 1. Create a `SOCK_STREAM` SCTP socket (IP proto 132).
    /// 2. Set `SCTP_INITMSG` (stream counts) and `SCTP_RECVRCVINFO` (so recv
    ///    can read the PPID).
    /// 3. Switch to non-blocking, register with the tokio reactor, bind to
    ///    `local` and connect to `remote`, awaiting the 4-way handshake for at
    ///    most `config.connect_timeout` without blocking a worker thread (#311).
    ///
    /// `expected_ppid` is the application protocol this association carries —
    /// [`NGAP_PPID`] for a gNB, [`crate::S1AP_PPID`] for an eNB. A received
    /// message with any other PPID is still delivered, but is warned about and
    /// counted ([`unexpected_ppid_count`](Self::unexpected_ppid_count)).
    pub async fn connect(
        local: SocketAddr,
        remote: SocketAddr,
        config: &SctpConfig,
        expected_ppid: u32,
    ) -> Result<Self> {
        info!(
            "Connecting to kernel SCTP endpoint at {} (proto 132)",
            remote
        );

        // 1. Create the SCTP socket. Protocol::from(132) == IPPROTO_SCTP.
        //    (libc::IPPROTO_SCTP is available on Linux, but we go through
        //    socket2's Protocol::from to keep the socket2 wrapper consistent.)
        let domain = if remote.is_ipv6() {
            Domain::IPV6
        } else {
            Domain::IPV4
        };
        let socket = Socket::new(
            domain,
            Type::STREAM,
            Some(Protocol::from(libc::IPPROTO_SCTP)),
        )
        .map_err(|e| SctpError::ConnectionFailed(format!("socket(SCTP): {e}")))?;

        let fd = socket.as_raw_fd();

        // 2a. SCTP_INITMSG: request the configured number of out/in streams.
        let initmsg = libc::sctp_initmsg {
            sinit_num_ostreams: config.max_outbound_streams,
            sinit_max_instreams: config.max_inbound_streams,
            sinit_max_attempts: 0,   // 0 = kernel default
            sinit_max_init_timeo: 0, // 0 = kernel default
        };
        // SAFETY: `fd` is a valid open socket; `&initmsg` points to a properly
        // sized `sctp_initmsg`; we pass its exact byte length.
        let rc = unsafe {
            libc::setsockopt(
                fd,
                SOL_SCTP,
                SCTP_INITMSG,
                std::ptr::addr_of!(initmsg).cast::<c_void>(),
                std::mem::size_of::<libc::sctp_initmsg>() as socklen_t,
            )
        };
        if rc != 0 {
            return Err(SctpError::ConnectionFailed(format!(
                "setsockopt(SCTP_INITMSG): {}",
                io::Error::last_os_error()
            )));
        }

        // 2b. SCTP_EVENTS: enable the legacy SCTP_SNDRCV ancillary data
        //     (sctp_data_io_event) so lksctp's sctp_recvmsg fills sinfo —
        //     including sinfo_ppid and sinfo_stream — and subscribe to the
        //     association + shutdown notifications we act on. Mirrors Open5GS's
        //     ogs_sctp_socket. (The newer SCTP_RECVRCVINFO delivers a cmsg that
        //     sctp_recvmsg ignores, which made the received PPID read as 0.)
        let events = SctpEventSubscribe {
            sctp_data_io_event: 1,
            sctp_association_event: 1,
            sctp_shutdown_event: 1,
            ..Default::default()
        };
        // SAFETY: `fd` is a valid open socket; `&events` points to a properly
        // sized `sctp_event_subscribe`; we pass its exact byte length.
        let rc = unsafe {
            libc::setsockopt(
                fd,
                SOL_SCTP,
                SCTP_EVENTS,
                std::ptr::addr_of!(events).cast::<c_void>(),
                std::mem::size_of::<SctpEventSubscribe>() as socklen_t,
            )
        };
        if rc != 0 {
            // The PPID/stream would be unreadable without the SCTP_SNDRCV cmsg.
            // Surface a clear error so the operator can diagnose kernel/libsctp.
            return Err(SctpError::ConnectionFailed(format!(
                "setsockopt(SCTP_EVENTS): {}",
                io::Error::last_os_error()
            )));
        }

        // 3. Bind, then connect WITHOUT blocking (#311). A blocking connect held
        //    the tokio worker thread that called it for the whole kernel INIT
        //    retransmission sequence (tens of seconds toward a black-holed peer),
        //    starving every other task on that worker. The socket goes
        //    non-blocking first; the 4-way handshake completes when it turns
        //    writable, and SO_ERROR says how it ended (connect(2), EINPROGRESS).
        let bind_addr = SockAddr::from(local);
        socket
            .bind(&bind_addr)
            .map_err(|e| SctpError::ConnectionFailed(format!("bind({local}): {e}")))?;
        socket
            .set_nonblocking(true)
            .map_err(|e| SctpError::ConnectionFailed(format!("set_nonblocking: {e}")))?;

        let async_fd = AsyncFd::new(fd)
            .map_err(|e| SctpError::ConnectionFailed(format!("AsyncFd::new: {e}")))?;

        let remote_sa = SockAddr::from(remote);
        match socket.connect(&remote_sa) {
            Ok(()) => {}
            Err(e) if e.raw_os_error() == Some(libc::EINPROGRESS) => {
                let handshake = async {
                    loop {
                        let mut guard = async_fd.writable().await.map_err(|e| {
                            SctpError::ConnectionFailed(format!("connect({remote}): {e}"))
                        })?;
                        match socket.take_error() {
                            Ok(None) => {}
                            Ok(Some(e)) | Err(e) => {
                                return Err(SctpError::ConnectionFailed(format!(
                                    "connect({remote}): {e}"
                                )))
                            }
                        }
                        // Writable with no error: established once the peer
                        // address is set; otherwise a spurious wake-up.
                        if socket.peer_addr().is_ok() {
                            return Ok(());
                        }
                        guard.clear_ready();
                    }
                };
                tokio::time::timeout(config.connect_timeout, handshake)
                    .await
                    .map_err(|_| {
                        SctpError::ConnectionFailed(format!(
                            "connect({remote}): no association within {:?}",
                            config.connect_timeout
                        ))
                    })??;
            }
            Err(e) => {
                return Err(SctpError::ConnectionFailed(format!(
                    "connect({remote}): {e}"
                )))
            }
        }

        // Read back the OS-assigned local address (port).
        let local_addr = socket
            .local_addr()
            .ok()
            .and_then(|sa| sa.as_socket())
            .unwrap_or(local);

        info!(
            "Kernel SCTP association established: {} -> {}",
            local_addr, remote
        );

        Ok(Self {
            async_fd,
            socket,
            fd,
            local_addr,
            remote_addr: remote,
            established: true,
            max_recv: config.max_receive_buffer_size as usize,
            partial: None,
            ppid_check: PpidCheck::new(expected_ppid),
        })
    }

    /// Send `data` on `stream_id` using the NGAP PPID (60).
    pub async fn send(&mut self, stream_id: u16, data: &[u8]) -> Result<()> {
        self.send_with_ppid(stream_id, data, NGAP_PPID).await
    }

    /// Send `data` on `stream_id` with an explicit `ppid`.
    ///
    /// The PPID is converted to network byte order before being handed to
    /// `sctp_sendmsg`, matching Open5GS (`htobe32(ppid)`), so PPID 60 appears
    /// verbatim on the wire.
    pub async fn send_with_ppid(&mut self, stream_id: u16, data: &[u8], ppid: u32) -> Result<()> {
        if !self.established {
            return Err(SctpError::InvalidState(
                "Cannot send: kernel SCTP association not established".into(),
            ));
        }

        // CRITICAL: network byte order for the PPID, exactly like Open5GS.
        let ppid_be = ppid.to_be();
        let fd = self.fd;

        loop {
            // Wait until the socket is writable, then attempt the send inside
            // the guard so we don't block a tokio worker thread.
            let mut guard = self.async_fd.writable().await.map_err(SctpError::Io)?;

            let result = guard.try_io(|_| {
                // SAFETY: `fd` is a valid connected SCTP socket; `data` is a
                // valid slice of `data.len()` bytes; `to`/`tolen` are NULL/0
                // because the socket is connected (one-to-one).
                let n = unsafe {
                    sctp_sendmsg(
                        fd,
                        data.as_ptr().cast::<c_void>(),
                        data.len() as size_t,
                        std::ptr::null(),
                        0,
                        ppid_be,
                        0, // flags
                        stream_id,
                        0, // timetolive (0 = reliable)
                        0, // context
                    )
                };
                if n < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(n as usize)
                }
            });

            match result {
                Ok(Ok(n)) => {
                    debug!(
                        "Kernel SCTP sent {} bytes on stream {} with PPID {} (wire be: {:#010x})",
                        n, stream_id, ppid, ppid_be
                    );
                    return Ok(());
                }
                Ok(Err(e)) => return Err(SctpError::Io(e)),
                // try_io reported WouldBlock; the guard is cleared, loop and
                // re-await writability.
                Err(_would_block) => continue,
            }
        }
    }

    /// Receive a single SCTP message.
    ///
    /// Returns `Ok(None)` when the peer has shut the association down (recv of
    /// length 0). The returned [`ReceivedMessage`] carries the stream id and
    /// the PPID converted back from network byte order to host order.
    pub async fn recv(&mut self) -> Result<Option<ReceivedMessage>> {
        if !self.established {
            return Err(SctpError::AssociationClosed);
        }
        let (fd, cap) = (self.fd, self.max_recv.max(1));
        loop {
            let mut guard = self.async_fd.readable().await.map_err(SctpError::Io)?;
            // Only an EAGAIN reported through `try_io` clears readiness, so a
            // would-block must never be mistaken for data (#243). While data
            // remains, readiness stays set and `readable()` returns at once.
            let raw = match guard.try_io(|_| raw_recvmsg(fd, cap)) {
                Ok(Ok(raw)) => raw,
                Ok(Err(e)) if e.kind() == io::ErrorKind::Interrupted => continue,
                Ok(Err(e)) => return Err(SctpError::Io(e)),
                Err(_would_block) => continue,
            };
            drop(guard);
            match self.classify(raw) {
                RecvStep::Message(m) => return Ok(Some(m)),
                RecvStep::Closed => return Ok(None),
                RecvStep::Pending => continue,
            }
        }
    }

    /// Non-blocking receive: attempt a single `sctp_recvmsg` and return at once.
    ///
    /// The gNB's SCTP task drives every backend through one 10 ms poll loop
    /// (`poll_connections` → `try_recv`) rather than awaiting [`recv`](Self::recv)
    /// directly. The kernel socket is non-blocking, so this performs exactly one
    /// `sctp_recvmsg`:
    /// * a data PDU → `Ok(Some(..))`;
    /// * nothing pending (`EWOULDBLOCK`) → `Ok(None)`;
    /// * an SCTP notification → drained, `Ok(None)` (a terminal SHUTDOWN /
    ///   COMM_LOST flips `established`, so the next poll reports closure);
    /// * orderly peer shutdown (`recvmsg` returns 0) or an already-closed
    ///   association → `Err(AssociationClosed)`, which the poll loop turns into
    ///   an association-down teardown.
    ///
    /// Unlike [`recv`](Self::recv) this never awaits readiness — the periodic
    /// poll supplies the cadence. PPID handling is identical (network byte order
    /// on the wire, converted with `u32::from_be`).
    pub fn try_recv(&mut self) -> Result<Option<ReceivedMessage>> {
        if !self.established {
            return Err(SctpError::AssociationClosed);
        }
        loop {
            match self.recv_step() {
                Ok(RecvStep::Message(m)) => return Ok(Some(m)),
                Ok(RecvStep::Closed) => return Err(SctpError::AssociationClosed),
                // A notification or a partial delivery was consumed; look for
                // more in this tick rather than waiting for the next poll.
                Ok(RecvStep::Pending) => continue,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(None),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(SctpError::Io(e)),
            }
        }
    }

    /// One `sctp_recvmsg` on the non-blocking socket, classified per the SCTP
    /// sockets API (RFC 6458):
    /// * `Err(WouldBlock)` — nothing to read (EAGAIN). Never data (#243);
    /// * a notification (`MSG_NOTIFICATION`, §6.1) — association state is
    ///   updated; [`RecvStep::Closed`] if it ended the association, else
    ///   [`RecvStep::Pending`];
    /// * a return of 0 — the peer shut the association down (EOF);
    /// * user data without `MSG_EOR` — one piece of a message larger than the
    ///   receive buffer (partial delivery, §8.1.20): buffered, `Pending`;
    /// * user data with `MSG_EOR` — the end of a message: the buffered pieces
    ///   plus this one, with the stream and PPID of the message (§5.3.2;
    ///   the PPID in network byte order).
    fn recv_step(&mut self) -> io::Result<RecvStep> {
        let raw = raw_recvmsg(self.fd, self.max_recv.max(1))?;
        Ok(self.classify(raw))
    }

    /// Classify one completed `sctp_recvmsg` (see [`recv_step`](Self::recv_step)).
    fn classify(&mut self, raw: RawRecv) -> RecvStep {
        let RawRecv {
            mut buf,
            n,
            sinfo,
            msg_flags,
        } = raw;
        if (msg_flags & libc::MSG_NOTIFICATION) != 0 {
            trace!("Kernel SCTP notification received ({} bytes)", n);
            self.handle_notification_state(&buf[..n]);
            return if self.established {
                RecvStep::Pending
            } else {
                RecvStep::Closed
            };
        }

        if n == 0 && self.partial.is_none() {
            debug!("Kernel SCTP peer shut down the association");
            self.established = false;
            return RecvStep::Closed;
        }

        buf.truncate(n);
        let piece_ppid = u32::from_be(sinfo.sinfo_ppid);
        let piece_stream = sinfo.sinfo_stream;
        let (mut data, ppid, stream_id) = match self.partial.take() {
            // Later pieces of one message: the stream and PPID are the
            // message's, taken from its first piece.
            Some((mut acc, ppid, stream)) => {
                acc.extend_from_slice(&buf);
                (acc, ppid, stream)
            }
            None => (buf, piece_ppid, piece_stream),
        };

        if (msg_flags & libc::MSG_EOR) == 0 {
            trace!(
                "Kernel SCTP partial delivery: {} bytes so far on stream {}",
                data.len(),
                stream_id
            );
            self.partial = Some((std::mem::take(&mut data), ppid, stream_id));
            return RecvStep::Pending;
        }

        debug!(
            "Kernel SCTP received {} bytes on stream {} with PPID {}",
            data.len(),
            stream_id,
            ppid
        );
        self.ppid_check.observe(ppid, stream_id);
        RecvStep::Message(ReceivedMessage {
            stream_id,
            data: Bytes::from(data),
            ppid,
        })
    }

    /// Inspect an SCTP notification body and update `established` if it
    /// indicates the association is gone (SHUTDOWN / COMM_LOST).
    ///
    /// The notification body begins with a `sctp_notification` whose first
    /// `u16` is `sn_type`. We only need to distinguish the terminal events.
    fn handle_notification_state(&mut self, body: &[u8]) {
        // sctp_assoc_change == 0x0001; sctp_shutdown_event == 0x0006 (UAPI).
        const SCTP_ASSOC_CHANGE: u16 = 0x0001;
        const SCTP_SHUTDOWN_EVENT: u16 = 0x0006;
        // sac_state for SCTP_ASSOC_CHANGE: SCTP_COMM_LOST == 2, SCTP_SHUTDOWN_COMP == 4.
        const SCTP_COMM_LOST: u16 = 2;
        const SCTP_SHUTDOWN_COMP: u16 = 4;

        if body.len() < 2 {
            return;
        }
        let sn_type = u16::from_ne_bytes([body[0], body[1]]);
        match sn_type {
            SCTP_SHUTDOWN_EVENT => {
                info!("Kernel SCTP SHUTDOWN_EVENT — association closing");
                self.established = false;
            }
            // struct sctp_assoc_change: sn_type(u16), sn_flags(u16),
            // sn_length(u32), sac_state(u16), ...
            SCTP_ASSOC_CHANGE if body.len() >= 10 => {
                let sac_state = u16::from_ne_bytes([body[8], body[9]]);
                if sac_state == SCTP_COMM_LOST || sac_state == SCTP_SHUTDOWN_COMP {
                    info!(
                        "Kernel SCTP ASSOC_CHANGE state={} — association lost",
                        sac_state
                    );
                    self.established = false;
                }
            }
            _ => {}
        }
    }

    /// Returns the local socket address (with the OS-assigned port).
    pub fn local_addr(&self) -> SocketAddr {
        self.local_addr
    }

    /// Returns the remote AMF address.
    pub fn remote_addr(&self) -> SocketAddr {
        self.remote_addr
    }

    /// Returns true while the association is up.
    pub fn is_established(&self) -> bool {
        self.established
    }

    /// The PPID this association was opened to carry.
    pub fn expected_ppid(&self) -> u32 {
        self.ppid_check.expected
    }

    /// How many received messages carried a PPID other than
    /// [`expected_ppid`](Self::expected_ppid).
    pub fn unexpected_ppid_count(&self) -> u64 {
        self.ppid_check.unexpected
    }

    /// Close the association. Performs an SCTP shutdown via the underlying
    /// socket (dropping the `Socket` also closes the fd).
    pub fn close(&mut self) {
        if self.established {
            self.established = false;
            // Best-effort graceful shutdown; the fd is closed when `socket`
            // (and this struct) is dropped.
            if let Err(e) = self.socket.shutdown(std::net::Shutdown::Both) {
                trace!("Kernel SCTP shutdown returned: {}", e);
            }
            debug!("Kernel SCTP association closed");
        }
    }
}

impl Drop for KernelSctpAssociation {
    fn drop(&mut self) {
        if self.established {
            self.close();
        }
    }
}

// NOTE on fd ownership (no double-close):
// `socket2::Socket` owns the file descriptor and closes it in its own `Drop`.
// `AsyncFd<RawFd>` is used purely for tokio reactor readiness registration: a
// bare `RawFd` has no `Drop`, so when `AsyncFd<RawFd>` is dropped it only
// deregisters the fd from the reactor — it does NOT close it. Therefore the fd
// is closed exactly once, by `Socket`'s `Drop`. Both fields are kept alive in
// the struct for the association's lifetime.

#[cfg(test)]
mod tests {
    use super::{PpidCheck, SctpEventSubscribe};
    use crate::association::{SctpConfig, NGAP_PPID};
    use crate::S1AP_PPID;
    use libc::c_int;
    use std::mem;

    #[test]
    fn an_s1ap_association_does_not_flag_ppid_18() {
        let mut check = PpidCheck::new(S1AP_PPID);
        assert!(check.observe(18, 0), "PPID 18 is what S1-MME carries");
        assert!(check.observe(18, 1));
        assert_eq!(check.unexpected, 0, "conformant S1AP is never counted");
    }

    #[test]
    fn an_s1ap_association_flags_any_other_ppid() {
        let mut check = PpidCheck::new(S1AP_PPID);
        assert!(!check.observe(NGAP_PPID, 0), "NGAP on S1-MME is a mismatch");
        assert!(!check.observe(0, 0), "unspecified (0) is a mismatch too");
        assert_eq!(check.unexpected, 2);
    }

    #[test]
    fn an_ngap_association_keeps_flagging_non_ngap_ppids() {
        // The gNB relies on this as its only PPID check (#186 increment 2).
        let mut check = PpidCheck::new(NGAP_PPID);
        assert!(check.observe(60, 0));
        assert!(!check.observe(S1AP_PPID, 0));
        assert_eq!(check.unexpected, 1);
    }

    /// #311: a kernel connect toward a peer that never answers neither blocks the
    /// runtime thread nor runs past `connect_timeout`. Run on the single-threaded
    /// test runtime, where a blocking `connect(2)` would freeze the ticker below
    /// for the whole kernel INIT retransmission sequence. 192.0.2.1 is TEST-NET-1
    /// (RFC 5737): nothing answers there. Skips, saying so, on a host without
    /// kernel SCTP or where the address fails at once (no route).
    #[tokio::test]
    async fn a_kernel_connect_to_a_silent_peer_does_not_block_the_runtime() {
        if socket2::Socket::new(
            socket2::Domain::IPV4,
            socket2::Type::STREAM,
            Some(socket2::Protocol::from(libc::IPPROTO_SCTP)),
        )
        .is_err()
        {
            eprintln!("skipping: no kernel SCTP on this host");
            return;
        }
        let ticks = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let t = ticks.clone();
        let ticker = tokio::spawn(async move {
            loop {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                t.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        });
        let config = SctpConfig {
            connect_timeout: std::time::Duration::from_secs(1),
            ..SctpConfig::default()
        };
        let started = std::time::Instant::now();
        let result = super::KernelSctpAssociation::connect(
            "0.0.0.0:0".parse().unwrap(),
            "192.0.2.1:38412".parse().unwrap(),
            &config,
            NGAP_PPID,
        )
        .await;
        let elapsed = started.elapsed();
        ticker.abort();
        assert!(result.is_err(), "nothing answers at 192.0.2.1");
        assert!(
            elapsed < std::time::Duration::from_secs(5),
            "connect_timeout bounds the attempt: {elapsed:?}"
        );
        if elapsed < std::time::Duration::from_millis(900) {
            eprintln!("skipping the runtime check: the connect failed at once ({:?})", result.err());
            return;
        }
        let n = ticks.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            n >= 40,
            "the runtime kept running during the connect: {n} ticks"
        );
    }

    /// Real kernel SCTP over loopback: a listener sends one S1AP (18) and one
    /// NGAP (60) message to a client opened for S1AP. Both are delivered with
    /// the PPID they carried; only the NGAP one is counted. Skips, saying so,
    /// on a host without kernel SCTP.
    #[tokio::test]
    async fn kernel_loopback_counts_only_the_unexpected_ppid() {
        use socket2::{Domain, Protocol, SockAddr, Socket, Type};
        use std::os::fd::AsRawFd;

        let listener = match Socket::new(
            Domain::IPV4,
            Type::STREAM,
            Some(Protocol::from(libc::IPPROTO_SCTP)),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: no kernel SCTP on this host ({e})");
                return;
            }
        };
        let any: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        listener.bind(&SockAddr::from(any)).expect("bind");
        listener.listen(1).expect("listen");
        let server_addr = listener
            .local_addr()
            .expect("local addr")
            .as_socket()
            .expect("inet");

        let server = std::thread::spawn(move || {
            let (conn, _) = listener.accept().expect("accept");
            for (ppid, msg) in [(S1AP_PPID, &b"s1ap"[..]), (NGAP_PPID, b"ngap")] {
                // SAFETY: `conn` is a connected one-to-one SCTP socket; `msg` is
                // a valid slice; `to`/`tolen` are NULL/0 for a connected socket.
                let n = unsafe {
                    super::sctp_sendmsg(
                        conn.as_raw_fd(),
                        msg.as_ptr().cast(),
                        msg.len(),
                        std::ptr::null(),
                        0,
                        ppid.to_be(),
                        0,
                        0,
                        0,
                        0,
                    )
                };
                assert_eq!(n, msg.len() as c_int, "sctp_sendmsg");
            }
            // Keep the association up until the client has read both.
            std::thread::sleep(std::time::Duration::from_millis(500));
        });

        let mut client = super::KernelSctpAssociation::connect(
            any,
            server_addr,
            &SctpConfig::default(),
            S1AP_PPID,
        )
        .await
        .expect("connect");
        assert_eq!(client.expected_ppid(), S1AP_PPID);

        let first = client.recv().await.expect("recv").expect("open");
        assert_eq!((first.ppid, &first.data[..]), (S1AP_PPID, &b"s1ap"[..]));
        assert_eq!(client.unexpected_ppid_count(), 0, "PPID 18 is expected");

        let second = client.recv().await.expect("recv").expect("open");
        assert_eq!((second.ppid, &second.data[..]), (NGAP_PPID, &b"ngap"[..]));
        assert_eq!(
            client.unexpected_ppid_count(),
            1,
            "an NGAP message on an S1AP association is counted, and still delivered"
        );

        server.join().expect("server thread");
    }

    /// Accept one kernel-SCTP association on 127.0.0.1 and run `serve` on the
    /// server side in a thread. `None` (with a note) on a host without SCTP.
    fn kernel_server(
        serve: impl FnOnce(socket2::Socket) + Send + 'static,
    ) -> Option<(std::net::SocketAddr, std::thread::JoinHandle<()>)> {
        use socket2::{Domain, Protocol, SockAddr, Socket, Type};
        let listener = match Socket::new(
            Domain::IPV4,
            Type::STREAM,
            Some(Protocol::from(libc::IPPROTO_SCTP)),
        ) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: no kernel SCTP on this host ({e})");
                return None;
            }
        };
        let any: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        listener.bind(&SockAddr::from(any)).expect("bind");
        listener.listen(1).expect("listen");
        let addr = listener
            .local_addr()
            .expect("local addr")
            .as_socket()
            .expect("inet");
        let handle = std::thread::spawn(move || {
            let (conn, _) = listener.accept().expect("accept");
            serve(conn);
        });
        Some((addr, handle))
    }

    /// Send `msg` with `ppid` on a connected one-to-one kernel SCTP socket.
    fn server_send(conn: &socket2::Socket, msg: &[u8], ppid: u32) {
        use std::os::fd::AsRawFd;
        // SAFETY: `conn` is a connected one-to-one SCTP socket; `msg` is a
        // valid slice; `to`/`tolen` are NULL/0 for a connected socket.
        let n = unsafe {
            super::sctp_sendmsg(
                conn.as_raw_fd(),
                msg.as_ptr().cast(),
                msg.len(),
                std::ptr::null(),
                0,
                ppid.to_be(),
                0,
                0,
                0,
                0,
            )
        };
        assert_eq!(n, msg.len() as c_int, "sctp_sendmsg");
    }

    /// #243: once the one pending message is read, the socket has nothing
    /// left. `sctp_recvmsg` then fails with EAGAIN, and that must be reported
    /// as "nothing pending" — not as a phantom PPID-0 message the size of the
    /// receive buffer, which is what a `ssize_t`-declared binding of libsctp's
    /// `int`-returning function produced, spinning the eNB at 100% CPU.
    #[tokio::test]
    async fn an_empty_socket_yields_no_phantom_message() {
        let Some((addr, server)) = kernel_server(|conn| {
            server_send(&conn, b"s1-setup-response", S1AP_PPID);
            std::thread::sleep(std::time::Duration::from_millis(800));
        }) else {
            return;
        };
        let any: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let mut client =
            super::KernelSctpAssociation::connect(any, addr, &SctpConfig::default(), S1AP_PPID)
                .await
                .expect("connect");

        let first = client.recv().await.expect("recv").expect("open");
        assert_eq!(
            (first.ppid, &first.data[..]),
            (S1AP_PPID, &b"s1-setup-response"[..])
        );

        // Non-blocking path: nothing pending is Ok(None), every time.
        for _ in 0..1000 {
            assert!(
                client.try_recv().expect("try_recv").is_none(),
                "an empty socket must not produce a message"
            );
        }
        // Async path: recv waits instead of returning a phantom message.
        let waited =
            tokio::time::timeout(std::time::Duration::from_millis(300), client.recv()).await;
        assert!(
            waited.is_err(),
            "recv must wait on an empty socket, got {waited:?}"
        );
        assert_eq!(
            client.unexpected_ppid_count(),
            0,
            "no PPID-0 message was seen"
        );

        server.join().expect("server thread");
    }

    /// A message larger than the receive buffer arrives in several partial
    /// deliveries; only the last carries MSG_EOR (RFC 6458 §8.1.20). They must
    /// be reassembled into one message with its own PPID and stream.
    #[tokio::test]
    async fn a_partially_delivered_message_is_reassembled() {
        let big: Vec<u8> = (0..5000u32).map(|i| (i % 251) as u8).collect();
        let sent = big.clone();
        let Some((addr, server)) = kernel_server(move |conn| {
            server_send(&conn, &sent, S1AP_PPID);
            server_send(&conn, b"after", S1AP_PPID);
            std::thread::sleep(std::time::Duration::from_millis(800));
        }) else {
            return;
        };
        let config = SctpConfig {
            max_receive_buffer_size: 512,
            ..SctpConfig::default()
        };
        let any: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let mut client = super::KernelSctpAssociation::connect(any, addr, &config, S1AP_PPID)
            .await
            .expect("connect");

        let whole = client.recv().await.expect("recv").expect("open");
        assert_eq!(whole.ppid, S1AP_PPID);
        assert_eq!(
            whole.data.len(),
            big.len(),
            "one message, not 512-octet pieces"
        );
        assert_eq!(&whole.data[..], &big[..]);
        let next = client.recv().await.expect("recv").expect("open");
        assert_eq!(&next.data[..], b"after", "message boundaries are kept");
        assert_eq!(client.unexpected_ppid_count(), 0);

        server.join().expect("server thread");
    }

    /// An orderly peer shutdown ends the association: `recv` reports EOF
    /// (`Ok(None)`) instead of looping.
    #[tokio::test]
    async fn a_peer_shutdown_is_end_of_stream() {
        let Some((addr, server)) = kernel_server(|conn| {
            server_send(&conn, b"bye", S1AP_PPID);
            std::thread::sleep(std::time::Duration::from_millis(200));
            drop(conn);
        }) else {
            return;
        };
        let any: std::net::SocketAddr = "127.0.0.1:0".parse().expect("addr");
        let mut client =
            super::KernelSctpAssociation::connect(any, addr, &SctpConfig::default(), S1AP_PPID)
                .await
                .expect("connect");
        let first = client.recv().await.expect("recv").expect("open");
        assert_eq!(&first.data[..], b"bye");
        let end = tokio::time::timeout(std::time::Duration::from_secs(5), client.recv())
            .await
            .expect("recv returns once the peer has shut down");
        assert!(matches!(end, Ok(None)), "EOF, got {end:?}");
        assert!(!client.is_established());
        server.join().expect("server thread");
    }

    /// `SctpEventSubscribe` is the one SCTP struct `libc` lacks, so it is still
    /// transcribed and handed to `setsockopt` by size: a dropped, reordered or
    /// widened member shifts every flag after it and silently subscribes the
    /// wrong events. Pin the size, the alignment and the offsets of the three
    /// members `connect` sets against `<linux/sctp.h>`.
    #[test]
    fn event_subscribe_layout_matches_the_kernel() {
        assert_eq!(mem::size_of::<SctpEventSubscribe>(), 13);
        assert_eq!(mem::align_of::<SctpEventSubscribe>(), 1);
        assert_eq!(mem::offset_of!(SctpEventSubscribe, sctp_data_io_event), 0);
        assert_eq!(
            mem::offset_of!(SctpEventSubscribe, sctp_association_event),
            1
        );
        assert_eq!(mem::offset_of!(SctpEventSubscribe, sctp_shutdown_event), 5);
        assert_eq!(
            mem::offset_of!(SctpEventSubscribe, sctp_stream_change_event),
            12
        );
    }
}
