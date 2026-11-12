//! The FL wire frames between the AF (`nr-fl-server`) and a UE client (nextgsim#480).
//!
//! **Row-free by construction.** No frame type has a field that can hold a
//! training row: the client sends a Hello, then per round an Update carrying its
//! model delta, its sample count and its training loss. The server sends the
//! global model. Training data never has a place to go, and the canary scan
//! ([`crate::synthetic::contains_row`]) checks that it did not.
//!
//! Encoding, all integers big-endian, `f32` little-endian (the same encoding the
//! canary scan searches for):
//!
//! ```text
//! magic "NFL1" | type u8 | payload length u32 | payload
//! Hello        (1): label length u8 | label (UTF-8, ≤ 64 bytes)
//! GlobalModel  (2): round u32 | count u32 | count × f32
//! Update       (3): round u32 | samples u32 | loss f32 | count u32 | count × f32
//! ```
//!
//! The decoder is incremental (a TCP stream delivers partial frames) and strict: an
//! unknown magic or type, a payload whose declared counts do not fill it exactly,
//! or a payload over [`MAX_PAYLOAD`] is an error, and the caller closes the
//! connection.

/// The frame magic.
pub const MAGIC: &[u8; 4] = b"NFL1";
/// The largest payload accepted: 1 MiB, far above the 261-parameter model.
pub const MAX_PAYLOAD: usize = 1 << 20;
/// The longest Hello label.
pub const MAX_LABEL: usize = 64;
const HEADER: usize = 9;

/// One frame.
#[derive(Debug, Clone, PartialEq)]
pub enum Frame {
    /// Client → server, once: a label for the logs. Identity comes from the NEF
    /// (by the client's address), never from this.
    Hello { label: String },
    /// Server → client: the global model for `round`.
    GlobalModel { round: u32, weights: Vec<f32> },
    /// Client → server: the result of local training on the round's model.
    Update {
        round: u32,
        samples: u32,
        loss: f32,
        delta: Vec<f32>,
    },
}

/// Why bytes are not a frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FrameError {
    BadMagic,
    UnknownType(u8),
    TooLarge(usize),
    /// The payload does not match its own declared counts.
    Malformed(&'static str),
}

impl std::fmt::Display for FrameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "not an NFL1 frame"),
            Self::UnknownType(t) => write!(f, "unknown frame type {t}"),
            Self::TooLarge(n) => write!(f, "payload of {n} bytes exceeds {MAX_PAYLOAD}"),
            Self::Malformed(why) => write!(f, "malformed payload: {why}"),
        }
    }
}

impl std::error::Error for FrameError {}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) {
    out.extend_from_slice(&(v.len() as u32).to_be_bytes());
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// Encode `frame`. Errors only for a payload too large to send, or a Hello label
/// over [`MAX_LABEL`].
pub fn encode(frame: &Frame) -> Result<Vec<u8>, FrameError> {
    let (kind, payload) = match frame {
        Frame::Hello { label } => {
            if label.len() > MAX_LABEL {
                return Err(FrameError::Malformed("label too long"));
            }
            let mut p = vec![label.len() as u8];
            p.extend_from_slice(label.as_bytes());
            (1u8, p)
        }
        Frame::GlobalModel { round, weights } => {
            let mut p = round.to_be_bytes().to_vec();
            put_f32s(&mut p, weights);
            (2, p)
        }
        Frame::Update {
            round,
            samples,
            loss,
            delta,
        } => {
            let mut p = round.to_be_bytes().to_vec();
            p.extend_from_slice(&samples.to_be_bytes());
            p.extend_from_slice(&loss.to_le_bytes());
            put_f32s(&mut p, delta);
            (3, p)
        }
    };
    if payload.len() > MAX_PAYLOAD {
        return Err(FrameError::TooLarge(payload.len()));
    }
    let mut out = Vec::with_capacity(HEADER + payload.len());
    out.extend_from_slice(MAGIC);
    out.push(kind);
    out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    out.extend_from_slice(&payload);
    Ok(out)
}

struct Reader<'a> {
    b: &'a [u8],
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], FrameError> {
        if self.b.len() < n {
            return Err(FrameError::Malformed("truncated field"));
        }
        let (head, rest) = self.b.split_at(n);
        self.b = rest;
        Ok(head)
    }

    fn u32(&mut self) -> Result<u32, FrameError> {
        Ok(u32::from_be_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn f32(&mut self) -> Result<f32, FrameError> {
        Ok(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn f32s(&mut self) -> Result<Vec<f32>, FrameError> {
        let n = self.u32()? as usize;
        if n.checked_mul(4).is_none_or(|bytes| bytes != self.b.len()) {
            return Err(FrameError::Malformed("count does not fill the payload"));
        }
        (0..n).map(|_| self.f32()).collect()
    }
}

/// Decode one frame from the front of `buf`: `Ok(None)` when more bytes are
/// needed, `Ok(Some((frame, consumed)))` otherwise.
pub fn decode(buf: &[u8]) -> Result<Option<(Frame, usize)>, FrameError> {
    if buf.len() < HEADER {
        // A wrong magic is reported as soon as it is visible.
        if !MAGIC.starts_with(&buf[..buf.len().min(4)]) {
            return Err(FrameError::BadMagic);
        }
        return Ok(None);
    }
    if &buf[..4] != MAGIC {
        return Err(FrameError::BadMagic);
    }
    let kind = buf[4];
    if !(1..=3).contains(&kind) {
        return Err(FrameError::UnknownType(kind));
    }
    let len = u32::from_be_bytes(buf[5..9].try_into().expect("4 bytes")) as usize;
    if len > MAX_PAYLOAD {
        return Err(FrameError::TooLarge(len));
    }
    if buf.len() < HEADER + len {
        return Ok(None);
    }
    let mut r = Reader {
        b: &buf[HEADER..HEADER + len],
    };
    let frame = match kind {
        1 => {
            let n = r.take(1)?[0] as usize;
            if n > MAX_LABEL || n != r.b.len() {
                return Err(FrameError::Malformed("label length"));
            }
            let label = std::str::from_utf8(r.take(n)?)
                .map_err(|_| FrameError::Malformed("label is not UTF-8"))?
                .to_string();
            Frame::Hello { label }
        }
        2 => Frame::GlobalModel {
            round: r.u32()?,
            weights: r.f32s()?,
        },
        _ => Frame::Update {
            round: r.u32()?,
            samples: r.u32()?,
            loss: r.f32()?,
            delta: r.f32s()?,
        },
    };
    Ok(Some((frame, HEADER + len)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames() -> Vec<Frame> {
        vec![
            Frame::Hello {
                label: "ue1".into(),
            },
            Frame::GlobalModel {
                round: 7,
                weights: vec![0.5, -1.25, f32::MIN_POSITIVE],
            },
            Frame::Update {
                round: 7,
                samples: 200,
                loss: 0.125,
                delta: vec![-0.0, 3.0],
            },
            Frame::GlobalModel {
                round: 0,
                weights: vec![],
            },
        ]
    }

    #[test]
    fn every_frame_round_trips_bit_exactly() {
        for f in frames() {
            let bytes = encode(&f).expect("encode");
            let (back, used) = decode(&bytes).expect("decode").expect("complete");
            assert_eq!(used, bytes.len());
            assert_eq!(encode(&back).expect("re-encode"), bytes, "{f:?}");
        }
    }

    /// A TCP stream delivers frames in pieces and back to back.
    #[test]
    fn the_decoder_is_incremental() {
        let stream: Vec<u8> = frames().iter().flat_map(|f| encode(f).unwrap()).collect();
        let mut got = Vec::new();
        let mut buf = Vec::new();
        for b in stream {
            buf.push(b);
            while let Some((f, used)) = decode(&buf).expect("valid stream") {
                got.push(f);
                buf.drain(..used);
            }
        }
        assert_eq!(got, frames());
        assert!(buf.is_empty());
    }

    #[test]
    fn bad_input_is_refused() {
        assert_eq!(decode(b"HTTP"), Err(FrameError::BadMagic));
        assert_eq!(decode(b"NFL1\x09\0\0\0\0"), Err(FrameError::UnknownType(9)));
        let mut big = b"NFL1\x02".to_vec();
        big.extend_from_slice(&((MAX_PAYLOAD + 1) as u32).to_be_bytes());
        assert_eq!(decode(&big), Err(FrameError::TooLarge(MAX_PAYLOAD + 1)));
        // A GlobalModel claiming 3 weights but carrying 2.
        let mut short = encode(&Frame::GlobalModel {
            round: 1,
            weights: vec![1.0, 2.0],
        })
        .unwrap();
        short[9 + 7] = 3;
        assert!(matches!(decode(&short), Err(FrameError::Malformed(_))));
        // ...and one claiming 1 weight but carrying 2 (trailing bytes).
        let mut long = encode(&Frame::GlobalModel {
            round: 1,
            weights: vec![1.0, 2.0],
        })
        .unwrap();
        long[9 + 7] = 1;
        assert!(matches!(decode(&long), Err(FrameError::Malformed(_))));
        // A count that would overflow is refused, not allocated.
        let mut huge = encode(&Frame::GlobalModel {
            round: 1,
            weights: vec![],
        })
        .unwrap();
        huge[13..17].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(matches!(decode(&huge), Err(FrameError::Malformed(_))));
        assert!(encode(&Frame::Hello {
            label: "x".repeat(65)
        })
        .is_err());
        assert_eq!(decode(b"NF"), Ok(None), "a partial header waits");
    }
}
