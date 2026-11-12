//! The FL wire frames between the AF (`nr-fl-server`) and a UE client (nextgsim#480).
//!
//! **Row-free by construction.** No frame type has a field that can hold a
//! training row: the client sends a Hello, then per round an Update carrying its
//! model delta, its sample count and its training loss. The server sends the
//! global model. The codec enforces that, rather than leaving it to the caller:
//! - a delta or a weight vector is exactly [`mlp::PARAMS`] floats, so a row cannot
//!   be appended to one (a 277-float delta is `Malformed`);
//! - every `f32` must be finite: a NaN or inf is `Malformed`, so it never reaches
//!   an aggregator;
//! - the Hello label is 1 to [`MAX_LABEL`] (16) bytes of `[A-Za-z0-9_-]`, too short
//!   for a 64-byte row and too narrow a charset to carry arbitrary floats.
//!
//! What remains is the delta's values themselves, which the canary scan
//! ([`crate::synthetic::contains_row`]) checks.
//!
//! **For the AF (plan item 6):** `Update.samples` is client-supplied and is the
//! FedAvg weight. A member claiming `u32::MAX` would take all of it, so the AF must
//! cap it or use its own per-member count; the codec does not judge it.
//!
//! Encoding, all integers big-endian, `f32` little-endian (the same encoding the
//! canary scan searches for):
//!
//! ```text
//! magic "NFL1" | type u8 | payload length u32 | payload
//! Hello        (1): label length u8 | label ([A-Za-z0-9_-], 1..=16 bytes)
//! GlobalModel  (2): round u32 | count u32 (= 261) | count × f32
//! Update       (3): round u32 | samples u32 | loss f32 | count u32 (= 261) | count × f32
//! ```
//!
//! The decoder is incremental (a TCP stream delivers partial frames) and strict: an
//! unknown magic or type, a payload whose declared counts do not fill it exactly,
//! a count other than [`mlp::PARAMS`], a non-finite float, a bad label or a payload
//! over [`MAX_PAYLOAD`] is an error, and the caller closes the connection. The
//! encoder refuses the same things, so a frame it produces always decodes.

use crate::mlp;

/// The frame magic.
pub const MAGIC: &[u8; 4] = b"NFL1";
/// The largest payload accepted: 1 MiB, far above the 261-parameter model.
pub const MAX_PAYLOAD: usize = 1 << 20;
/// The longest Hello label: 16 bytes, a quarter of one 64-byte training row.
pub const MAX_LABEL: usize = 16;
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

/// Whether `label` is a valid Hello label: 1 to [`MAX_LABEL`] bytes of
/// `[A-Za-z0-9_-]`.
pub fn valid_label(label: &[u8]) -> bool {
    (1..=MAX_LABEL).contains(&label.len())
        && label
            .iter()
            .all(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
}

fn check_finite(x: f32) -> Result<f32, FrameError> {
    if x.is_finite() {
        Ok(x)
    } else {
        Err(FrameError::Malformed("non-finite float"))
    }
}

fn put_f32s(out: &mut Vec<u8>, v: &[f32]) -> Result<(), FrameError> {
    if v.len() != mlp::PARAMS {
        return Err(FrameError::Malformed("vector is not the model's size"));
    }
    out.extend_from_slice(&(v.len() as u32).to_be_bytes());
    for x in v {
        out.extend_from_slice(&check_finite(*x)?.to_le_bytes());
    }
    Ok(())
}

/// Encode `frame`. Errors for anything [`decode`] would refuse: a bad label, a
/// vector that is not [`mlp::PARAMS`] long, a non-finite float, or a payload too
/// large to send.
pub fn encode(frame: &Frame) -> Result<Vec<u8>, FrameError> {
    let (kind, payload) = match frame {
        Frame::Hello { label } => {
            if !valid_label(label.as_bytes()) {
                return Err(FrameError::Malformed("label"));
            }
            let mut p = vec![label.len() as u8];
            p.extend_from_slice(label.as_bytes());
            (1u8, p)
        }
        Frame::GlobalModel { round, weights } => {
            let mut p = round.to_be_bytes().to_vec();
            put_f32s(&mut p, weights)?;
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
            p.extend_from_slice(&check_finite(*loss)?.to_le_bytes());
            put_f32s(&mut p, delta)?;
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
        check_finite(f32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }

    fn f32s(&mut self) -> Result<Vec<f32>, FrameError> {
        let n = self.u32()? as usize;
        if n.checked_mul(4).is_none_or(|bytes| bytes != self.b.len()) {
            return Err(FrameError::Malformed("count does not fill the payload"));
        }
        if n != mlp::PARAMS {
            return Err(FrameError::Malformed("vector is not the model's size"));
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
            if n != r.b.len() {
                return Err(FrameError::Malformed("label length"));
            }
            let bytes = r.take(n)?;
            if !valid_label(bytes) {
                return Err(FrameError::Malformed("label"));
            }
            // `valid_label` admits ASCII only, so this is always UTF-8.
            let label = String::from_utf8_lossy(bytes).into_owned();
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

    /// A model-sized vector with some awkward values in front.
    fn model(front: &[f32]) -> Vec<f32> {
        let mut v: Vec<f32> = (0..mlp::PARAMS).map(|i| i as f32 * 0.01).collect();
        v[..front.len()].copy_from_slice(front);
        v
    }

    fn frames() -> Vec<Frame> {
        vec![
            Frame::Hello {
                label: "ue1".into(),
            },
            Frame::GlobalModel {
                round: 7,
                weights: model(&[0.5, -1.25, f32::MIN_POSITIVE]),
            },
            Frame::Update {
                round: 7,
                samples: 200,
                loss: 0.125,
                delta: model(&[-0.0, 3.0, f32::MAX]),
            },
            Frame::Hello {
                label: "AF-node_2".into(),
            },
        ]
    }

    fn update(delta: Vec<f32>, loss: f32) -> Frame {
        Frame::Update {
            round: 1,
            samples: 240,
            loss,
            delta,
        }
    }

    /// Build the raw bytes of a frame, bypassing `encode`'s checks, so the
    /// decoder's own checks are what is tested.
    fn raw(kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.push(kind);
        out.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        out.extend_from_slice(payload);
        out
    }

    fn raw_update(delta: &[f32], loss: f32) -> Vec<u8> {
        let mut p = 1u32.to_be_bytes().to_vec();
        p.extend_from_slice(&240u32.to_be_bytes());
        p.extend_from_slice(&loss.to_le_bytes());
        p.extend_from_slice(&(delta.len() as u32).to_be_bytes());
        for x in delta {
            p.extend_from_slice(&x.to_le_bytes());
        }
        raw(3, &p)
    }

    fn raw_global(weights: &[f32]) -> Vec<u8> {
        let mut p = 1u32.to_be_bytes().to_vec();
        p.extend_from_slice(&(weights.len() as u32).to_be_bytes());
        for x in weights {
            p.extend_from_slice(&x.to_le_bytes());
        }
        raw(2, &p)
    }

    fn raw_hello(label: &[u8]) -> Vec<u8> {
        let mut p = vec![label.len() as u8];
        p.extend_from_slice(label);
        raw(1, &p)
    }

    fn malformed(r: Result<Option<(Frame, usize)>, FrameError>) -> bool {
        matches!(r, Err(FrameError::Malformed(_)))
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
        let good = model(&[]);
        // A GlobalModel claiming one weight more than it carries...
        let mut short = raw_global(&good);
        short[9 + 7] += 1;
        assert!(malformed(decode(&short)));
        // ...and one claiming one fewer (trailing bytes).
        let mut long = raw_global(&good);
        long[9 + 7] -= 1;
        assert!(malformed(decode(&long)));
        // A count that would overflow is refused, not allocated.
        let mut huge = raw_global(&[]);
        huge[13..17].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(malformed(decode(&huge)));
        assert_eq!(decode(b"NF"), Ok(None), "a partial header waits");
    }

    /// Review M4: a vector is exactly the model's size, so a training row (16
    /// floats) cannot ride along: the 277-float delta of the `leak-raw` arm, one
    /// float more or less, and an empty vector are all refused, both ways.
    #[test]
    fn a_vector_must_be_the_models_size() {
        let good = model(&[]);
        assert!(decode(&raw_update(&good, 0.1)).unwrap().is_some());
        assert!(decode(&raw_global(&good)).unwrap().is_some());
        for len in [0, 1, mlp::PARAMS - 1, mlp::PARAMS + 1, mlp::PARAMS + 16] {
            let v = vec![0.5f32; len];
            assert!(malformed(decode(&raw_update(&v, 0.1))), "Update of {len}");
            assert!(malformed(decode(&raw_global(&v))), "GlobalModel of {len}");
            assert!(encode(&update(v.clone(), 0.1)).is_err(), "encode {len}");
            assert!(encode(&Frame::GlobalModel {
                round: 1,
                weights: v
            })
            .is_err());
        }
    }

    /// Review M1: NaN and ±inf never leave the decoder, in the delta, the weights
    /// or the loss, whatever the NaN's sign or payload.
    #[test]
    fn non_finite_floats_are_refused() {
        let quiet_neg = f32::from_bits(0xffc0_0001);
        for bad in [
            f32::NAN,
            -f32::NAN,
            quiet_neg,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let mut v = model(&[]);
            v[mlp::PARAMS - 1] = bad;
            assert!(malformed(decode(&raw_update(&v, 0.1))), "delta {bad}");
            assert!(malformed(decode(&raw_global(&v))), "weights {bad}");
            assert!(
                malformed(decode(&raw_update(&model(&[]), bad))),
                "loss {bad}"
            );
            assert!(encode(&update(v.clone(), 0.1)).is_err());
            assert!(encode(&update(model(&[]), bad)).is_err());
        }
        // The largest finite values are fine.
        assert!(decode(&raw_update(&model(&[f32::MAX, f32::MIN]), f32::MAX))
            .unwrap()
            .is_some());
    }

    /// Review M4: the Hello label is 1..=16 bytes of `[A-Za-z0-9_-]`, so it cannot
    /// hold a 64-byte row, even one whose bytes happen to be valid UTF-8.
    #[test]
    fn the_hello_label_is_short_and_plain() {
        for ok in ["u", "ue1", "AF-node_2", "abcdefghijklmnop"] {
            assert!(decode(&raw_hello(ok.as_bytes())).unwrap().is_some(), "{ok}");
            assert!(encode(&Frame::Hello { label: ok.into() }).is_ok());
        }
        // 0x3f414141 ≈ 0.755 is the bytes "AAA?": a row of it is ASCII.
        let row: Vec<u8> = [f32::from_bits(0x3f41_4141); 16]
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect();
        assert!(
            std::str::from_utf8(&row).is_ok(),
            "a row that is valid UTF-8"
        );
        let seventeen = "abcdefghijklmnopq";
        for bad in [
            b"".as_slice(),
            seventeen.as_bytes(),
            &row,
            b"ue 1",
            b"ue1\n",
            b"ue.1",
            "üe1".as_bytes(),
            b"ue\x001",
        ] {
            assert!(malformed(decode(&raw_hello(bad))), "{bad:?}");
            if let Ok(label) = std::str::from_utf8(bad) {
                assert!(encode(&Frame::Hello {
                    label: label.into()
                })
                .is_err());
            }
        }
    }
}
