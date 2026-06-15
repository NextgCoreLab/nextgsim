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
//! convert it with `u32::from_be` before comparing against [`NGAP_PPID`].

use std::io;
use std::net::SocketAddr;
use std::os::fd::{AsRawFd, RawFd};

use bytes::Bytes;
use libc::{c_int, c_void, size_t, sockaddr, socklen_t, ssize_t};
use socket2::{Domain, Protocol, SockAddr, Socket, Type};
use tokio::io::unix::AsyncFd;
use tracing::{debug, info, trace, warn};

use crate::association::{ReceivedMessage, Result, SctpConfig, SctpError, NGAP_PPID};

// ===========================================================================
// libsctp (lksctp-tools) FFI
// ===========================================================================

/// lksctp `struct sctp_sndrcvinfo` (from `<netinet/sctp.h>`).
///
/// Layout taken from the lksctp / Linux kernel UAPI definition. All multi-byte
/// integer fields are in host byte order **except** `sinfo_ppid`, which the
/// kernel carries in network byte order (see the module-level note).
///
/// ```c
/// struct sctp_sndrcvinfo {
///     __u16 sinfo_stream;
///     __u16 sinfo_ssn;
///     __u16 sinfo_flags;
///     __u32 sinfo_ppid;
///     __u32 sinfo_context;
///     __u32 sinfo_timetolive;
///     __u32 sinfo_tsn;
///     __u32 sinfo_cumtsn;
///     sctp_assoc_t sinfo_assoc_id; // sctp_assoc_t == __s32 (i32)
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SctpSndRcvInfo {
    sinfo_stream: u16,
    sinfo_ssn: u16,
    sinfo_flags: u16,
    sinfo_ppid: u32,
    sinfo_context: u32,
    sinfo_timetolive: u32,
    sinfo_tsn: u32,
    sinfo_cumtsn: u32,
    sinfo_assoc_id: i32,
}

/// lksctp `struct sctp_initmsg` (from `<netinet/sctp.h>`), used with the
/// `SCTP_INITMSG` socket option to request the number of streams.
///
/// ```c
/// struct sctp_initmsg {
///     __u16 sinit_num_ostreams;
///     __u16 sinit_max_instreams;
///     __u16 sinit_max_attempts;
///     __u16 sinit_max_init_timeo;
/// };
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct SctpInitMsg {
    sinit_num_ostreams: u16,
    sinit_max_instreams: u16,
    sinit_max_attempts: u16,
    sinit_max_init_timeo: u16,
}

/// lksctp `struct sctp_event_subscribe` (from `<netinet/sctp.h>`), used with
/// the `SCTP_EVENTS` socket option. Every field is a `__u8` flag (0/1). The
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

// SCTP socket-option level and names (from <netinet/sctp.h>). These are stable
// UAPI constants; libc does not always export them, so we define them locally.
/// `IPPROTO_SCTP` socket option level for `setsockopt` (== 132).
const SOL_SCTP: c_int = 132;
/// `SCTP_INITMSG` socket option: request outbound/inbound stream counts.
const SCTP_INITMSG: c_int = 2;
/// `SCTP_EVENTS` socket option: subscribe to SCTP events. Critically, enabling
/// `sctp_data_io_event` makes the kernel attach the legacy `SCTP_SNDRCV`
/// ancillary message that lksctp's `sctp_recvmsg()` parses to fill
/// `sctp_sndrcvinfo` (PPID + stream). This is the same setup Open5GS uses
/// (`ogs_sctp_socket`). The newer `SCTP_RECVRCVINFO` delivers a *different*
/// cmsg (`SCTP_RCVINFO`) that `sctp_recvmsg()` does not read, which left the
/// received PPID/stream zeroed.
const SCTP_EVENTS: c_int = 11;

// libsctp helper functions. We link against `-lsctp` (libsctp-dev). These are
// the classic one-to-one helpers that carry the PPID/stream out-of-band so we
// do not have to assemble `cmsg` ancillary data by hand.
#[link(name = "sctp")]
extern "C" {
    /// `sctp_sendmsg` — send `len` bytes on `sd` with the given `ppid` (network
    /// byte order), `stream_no`, and SCTP flags. `to`/`tolen` may be NULL/0 on
    /// a connected one-to-one socket.
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
    ) -> ssize_t;

    /// `sctp_recvmsg` — receive up to `len` bytes into `msg`, filling `sinfo`
    /// (stream, ppid in network byte order, flags, ...) and `msg_flags`.
    fn sctp_recvmsg(
        sd: c_int,
        msg: *mut c_void,
        len: size_t,
        from: *mut sockaddr,
        fromlen: *mut socklen_t,
        sinfo: *mut SctpSndRcvInfo,
        msg_flags: *mut c_int,
    ) -> ssize_t;
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
    /// Set once `connect` returns; the kernel completes the 4-way handshake
    /// synchronously for a blocking connect, so this is true after `connect`.
    established: bool,
    /// Maximum receive buffer size from config (per-recv allocation cap).
    max_recv: usize,
}

impl KernelSctpAssociation {
    /// Establish a one-to-one kernel SCTP association from `local` to `remote`.
    ///
    /// Steps:
    /// 1. Create a `SOCK_STREAM` SCTP socket (IP proto 132).
    /// 2. Set `SCTP_INITMSG` (stream counts) and `SCTP_RECVRCVINFO` (so recv
    ///    can read the PPID).
    /// 3. Bind to `local`, connect to `remote` (blocking 4-way handshake).
    /// 4. Switch to non-blocking and register with the tokio reactor.
    pub async fn connect(
        local: SocketAddr,
        remote: SocketAddr,
        config: &SctpConfig,
    ) -> Result<Self> {
        info!("Connecting to kernel SCTP endpoint at {} (proto 132)", remote);

        // 1. Create the SCTP socket. Protocol::from(132) == IPPROTO_SCTP.
        //    (libc::IPPROTO_SCTP is available on Linux, but we go through
        //    socket2's Protocol::from to keep the socket2 wrapper consistent.)
        let domain = if remote.is_ipv6() {
            Domain::IPV6
        } else {
            Domain::IPV4
        };
        let socket = Socket::new(domain, Type::STREAM, Some(Protocol::from(libc::IPPROTO_SCTP)))
            .map_err(|e| SctpError::ConnectionFailed(format!("socket(SCTP): {e}")))?;

        let fd = socket.as_raw_fd();

        // 2a. SCTP_INITMSG: request the configured number of out/in streams.
        let initmsg = SctpInitMsg {
            sinit_num_ostreams: config.max_outbound_streams,
            sinit_max_instreams: config.max_inbound_streams,
            sinit_max_attempts: 0, // 0 = kernel default
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
                std::mem::size_of::<SctpInitMsg>() as socklen_t,
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

        // 3. Bind + connect. We do this while the socket is still blocking so
        //    the kernel performs the full 4-way handshake synchronously and we
        //    get a clear error if the AMF is unreachable.
        let bind_addr = SockAddr::from(local);
        socket
            .bind(&bind_addr)
            .map_err(|e| SctpError::ConnectionFailed(format!("bind({local}): {e}")))?;

        let remote_sa = SockAddr::from(remote);
        socket
            .connect(&remote_sa)
            .map_err(|e| SctpError::ConnectionFailed(format!("connect({remote}): {e}")))?;

        // Read back the OS-assigned local address (port).
        let local_addr = socket
            .local_addr()
            .ok()
            .and_then(|sa| sa.as_socket())
            .unwrap_or(local);

        // 4. Non-blocking + register with the tokio reactor.
        socket
            .set_nonblocking(true)
            .map_err(|e| SctpError::ConnectionFailed(format!("set_nonblocking: {e}")))?;

        let async_fd = AsyncFd::new(fd)
            .map_err(|e| SctpError::ConnectionFailed(format!("AsyncFd::new: {e}")))?;

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

        let fd = self.fd;
        let cap = self.max_recv.max(1);

        loop {
            let mut guard = self.async_fd.readable().await.map_err(SctpError::Io)?;

            let mut buf = vec![0u8; cap];
            let mut sinfo = SctpSndRcvInfo::default();
            let mut msg_flags: c_int = 0;

            let result = guard.try_io(|_| {
                // SAFETY: `fd` is a valid SCTP socket; `buf` has `cap` bytes;
                // `sinfo`/`msg_flags` are valid out-params; from/fromlen NULL
                // (we don't need the peer address on a connected socket).
                let n = unsafe {
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
                if n < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(n as usize)
                }
            });

            let n = match result {
                Ok(Ok(n)) => n,
                Ok(Err(e)) => return Err(SctpError::Io(e)),
                Err(_would_block) => continue,
            };

            // MSG_NOTIFICATION: this is an SCTP notification event, not user
            // data. Drain it and keep waiting for an actual NGAP PDU.
            if (msg_flags & libc::MSG_NOTIFICATION) != 0 {
                trace!("Kernel SCTP notification received ({} bytes), skipping", n);
                // A SHUTDOWN/COMM_LOST notification means the association is
                // gone; treat as closed so the gNB tears the connection down.
                self.handle_notification_state(&buf[..n.min(buf.len())]);
                if !self.established {
                    return Ok(None);
                }
                continue;
            }

            if n == 0 {
                // Peer performed an orderly shutdown.
                debug!("Kernel SCTP peer shut down the association");
                self.established = false;
                return Ok(None);
            }

            buf.truncate(n);
            // CRITICAL: sinfo_ppid arrives in network byte order; convert back.
            let ppid = u32::from_be(sinfo.sinfo_ppid);
            let stream_id = sinfo.sinfo_stream;

            debug!(
                "Kernel SCTP received {} bytes on stream {} with PPID {} (wire be: {:#010x})",
                n, stream_id, ppid, sinfo.sinfo_ppid
            );
            if ppid != NGAP_PPID {
                warn!(
                    "Kernel SCTP received non-NGAP PPID {} (expected {})",
                    ppid, NGAP_PPID
                );
            }

            return Ok(Some(ReceivedMessage {
                stream_id,
                data: Bytes::from(buf),
                ppid,
            }));
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

        let fd = self.fd;
        let cap = self.max_recv.max(1);
        let mut buf = vec![0u8; cap];
        let mut sinfo = SctpSndRcvInfo::default();
        let mut msg_flags: c_int = 0;

        // SAFETY: `fd` is a valid SCTP socket; `buf` has `cap` bytes;
        // `sinfo`/`msg_flags` are valid out-params; from/fromlen NULL. The
        // socket is non-blocking, so this returns EWOULDBLOCK instead of
        // blocking when no message is ready.
        let n = unsafe {
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

        if n < 0 {
            let err = io::Error::last_os_error();
            if err.kind() == io::ErrorKind::WouldBlock {
                // No PDU buffered this tick; the next poll will retry.
                return Ok(None);
            }
            return Err(SctpError::Io(err));
        }
        let n = n as usize;

        // MSG_NOTIFICATION: an SCTP event, not user data. Update association
        // state and report "nothing to deliver" for this tick.
        if (msg_flags & libc::MSG_NOTIFICATION) != 0 {
            trace!("Kernel SCTP notification received ({} bytes), skipping", n);
            self.handle_notification_state(&buf[..n.min(buf.len())]);
            if !self.established {
                return Err(SctpError::AssociationClosed);
            }
            return Ok(None);
        }

        if n == 0 {
            // Peer performed an orderly shutdown.
            debug!("Kernel SCTP peer shut down the association");
            self.established = false;
            return Err(SctpError::AssociationClosed);
        }

        buf.truncate(n);
        // CRITICAL: sinfo_ppid arrives in network byte order; convert back.
        let ppid = u32::from_be(sinfo.sinfo_ppid);
        let stream_id = sinfo.sinfo_stream;

        debug!(
            "Kernel SCTP received {} bytes on stream {} with PPID {} (wire be: {:#010x})",
            n, stream_id, ppid, sinfo.sinfo_ppid
        );
        if ppid != NGAP_PPID {
            warn!(
                "Kernel SCTP received non-NGAP PPID {} (expected {})",
                ppid, NGAP_PPID
            );
        }

        Ok(Some(ReceivedMessage {
            stream_id,
            data: Bytes::from(buf),
            ppid,
        }))
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
