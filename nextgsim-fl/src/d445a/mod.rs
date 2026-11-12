//! Demo D445A (nextgsim#480): network-assisted federated learning over real PDU
//! sessions. **Non-normative**: TS 23.501 §5.46.1 has the AF use network
//! monitoring "to assist the AI/ML application server in scheduling available
//! UE(s)", and puts the AI/ML logic itself out of 3GPP scope.
//!
//! Three pieces, all behind the `demo-d445a-fl-t8` cargo feature and the
//! `demo.d445a-fl-t8` config block (both off by default):
//!
//! - [`server`]: the `nr-fl-server` AF. It listens for UE FL clients (one TCP
//!   connection per UE, carried over the UE's PDU session), learns each UE's
//!   identity from the NEF by the source address ([`nef`]), subscribes to T8
//!   `LOSS_OF_CONNECTIVITY` for it, runs the rounds, scores every round on the
//!   held-out set, and scores the local-only and pooled baselines itself.
//! - [`nef`]: the AF's NEF client (UE ID retrieval, T8 monitoring subscribe /
//!   delete) and the T8 callback listener.
//! - [`client`]: the UE side, run inside `nr-ue` and bound to the PDU session's
//!   TUN (`uesimtunN`).
//!
//! The wire is the row-free `NFL1` framing of [`crate::frame`]: Hello, GlobalModel
//! and Update, nothing else. The task, the model and the aggregation rules are the
//! library's ([`crate::synthetic`], [`crate::mlp`], [`crate::sim::aggregate`]), so
//! a network run can be compared bit for bit with the in-process run of the same
//! seed (on the same platform image: libm differs across platforms).

pub mod client;
pub mod nef;
pub mod server;
#[cfg(test)]
pub(crate) mod stub_nef;

/// The demo key, as written under `demo:`.
pub const KEY: &str = "d445a-fl-t8";
/// 3 UEs; `ue3` deregisters, the NEF's T8 report excludes it, then it is
/// readmitted.
pub const MODE_T8_EXCLUSION: &str = "t8-exclusion";
/// 5 UEs; UE 005 sends ×10 sign-flipped deltas, and the AF compares FedAvg with
/// median, trimmed mean and Krum.
pub const MODE_POISONED_UE: &str = "poisoned-ue";
/// The modes, in the order the ticket lists them.
pub const MODES: &[&str] = &[MODE_T8_EXCLUSION, MODE_POISONED_UE];

/// The member that poisons in `poisoned-ue`: UE 005, and only it.
pub const ATTACKER_MEMBER: u32 = 5;
/// The attacker sends `-POISON_SCALE ×` its honest delta (×10 sign-flipped).
pub const POISON_SCALE: f32 = 10.0;

/// The label of member `m` (1-based): `ue1`, `ue2`, ... It is also the AF's sort
/// key, the same as [`crate::sim::member`] of `m - 1`.
pub fn label(member: u32) -> String {
    format!("ue{member}")
}

/// The member number of a label `ueN` (N ≥ 1), or `None`.
pub fn member_of(label: &str) -> Option<u32> {
    let n = label.strip_prefix("ue")?;
    if n.is_empty() || n.starts_with('0') || !n.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    n.parse().ok().filter(|m| *m >= 1)
}

/// Why reading a frame stopped.
#[derive(Debug)]
pub enum ReadError {
    /// The transport failed, or closed in the middle of a frame.
    Io(std::io::Error),
    /// The bytes are not a valid frame. `raw` is what had arrived, so the AF can
    /// still scan it for canary rows before it closes the connection.
    Frame {
        /// What the decoder refused.
        error: crate::frame::FrameError,
        /// The buffered bytes.
        raw: Vec<u8>,
    },
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transport: {e}"),
            Self::Frame { error, .. } => write!(f, "refused frame: {error}"),
        }
    }
}

/// Reads `NFL1` frames off a byte stream. The buffer never holds more than one
/// frame's header plus [`crate::frame::MAX_PAYLOAD`]: the decoder refuses a larger
/// length as soon as the header is in.
#[derive(Debug, Default)]
pub struct FrameReader {
    buf: Vec<u8>,
}

impl FrameReader {
    /// The next frame and its raw bytes; `Ok(None)` on a clean EOF between frames.
    pub async fn next<R: tokio::io::AsyncRead + Unpin>(
        &mut self,
        r: &mut R,
    ) -> Result<Option<(crate::frame::Frame, Vec<u8>)>, ReadError> {
        use tokio::io::AsyncReadExt;
        loop {
            match crate::frame::decode(&self.buf) {
                Ok(Some((frame, used))) => {
                    let raw: Vec<u8> = self.buf.drain(..used).collect();
                    return Ok(Some((frame, raw)));
                }
                Ok(None) => {}
                Err(error) => {
                    return Err(ReadError::Frame {
                        error,
                        raw: std::mem::take(&mut self.buf),
                    })
                }
            }
            let mut chunk = [0u8; 8192];
            let n = r.read(&mut chunk).await.map_err(ReadError::Io)?;
            if n == 0 {
                return if self.buf.is_empty() {
                    Ok(None)
                } else {
                    Err(ReadError::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "closed in the middle of a frame",
                    )))
                };
            }
            self.buf.extend_from_slice(&chunk[..n]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_and_members_round_trip() {
        for m in [1, 3, 5, 12] {
            assert_eq!(member_of(&label(m)), Some(m));
            assert_eq!(label(m), crate::sim::member(m - 1));
        }
        for bad in ["ue0", "ue", "ue01", "UE1", "ue-1", "x1", "ue1a"] {
            assert_eq!(member_of(bad), None, "{bad}");
        }
    }
}
