//! TUAK algorithm set (3GPP TS 35.231 / TS 35.232 / TS 35.233)
//!
//! TUAK is the second example algorithm set for the 3GPP authentication and key
//! generation functions f1, f1*, f2, f3, f4, f5 and f5*, standing alongside
//! MILENAGE (TS 35.206). Unlike MILENAGE (built on AES), TUAK is built on the
//! Keccak-f[1600] permutation (the SHA-3 core).
//!
//! Functions provided:
//! - `compute_topc`: derive TOPc from the operator TOP value and the key K
//! - f1  : network authentication MAC-A
//! - f1* : re-synchronisation authentication MAC-S
//! - f2  : user authentication response RES
//! - f3  : cipher key CK
//! - f4  : integrity key IK
//! - f5  : anonymity key AK
//! - f5* : re-synchronisation anonymity key AK (for AUTS)
//!
//! TUAK supports a 128- or 256-bit subscriber key K and configurable output
//! lengths (MAC 64/128/256, RES 32/64/128/256, CK/IK 128/256, AK fixed 48), plus
//! a configurable Keccak iteration count (normally 1; TS 35.232 test set 6 uses 2).
//!
//! ## Construction (TS 35.231 §6)
//!
//! Each function builds a 1600-bit (200-byte) input string for the Keccak-f[1600]
//! permutation Π. The bit-level layout (per TS 35.231 §6.2, as also reproduced in
//! Mandal, Tan, Wu & Gong, "A Comprehensive Security Analysis of the TUAK Algorithm
//! Set", CACR 2016-02) is, for f1:
//!
//! ```text
//! IN[0..255]    = TOPc[255..0]            (256 bits)
//! IN[256..263]  = INSTANCE[7..0]          (8 bits)
//! IN[264..319]  = ALGONAME[55..0]         (56 bits, ASCII "TUAK1.0")
//! IN[320..447]  = RAND[127..0]            (128 bits)
//! IN[448..463]  = AMF[15..0]              (16 bits)   -- f1/f1* only
//! IN[464..511]  = SQN[47..0]              (48 bits)   -- f1/f1* only
//! IN[512..767]  = K[255..0]   (256-bit K) or K[127..0] then zeros (128-bit K)
//! IN[768..772]  = 1                       (padding)
//! IN[1087]      = 1                       (padding)
//! all other bits = 0
//! ```
//!
//! Because consecutive IN bits map to *descending* field bit indices and the
//! Keccak input string is packed little-endian within bytes, each field ends up
//! **byte-reversed** in the 200-byte state, with each individual byte keeping its
//! natural value. Outputs are likewise byte-reversed slices of the output state:
//! MAC/RES = state[0..], CK = state[32..], IK = state[64..], AK = state[96..102].
//! For f2/f3/f4/f5 the AMF/SQN region is zero and all four outputs come from a
//! single Π evaluation (they share one INSTANCE).
//!
//! Vectors: validated byte-exact against TS 35.232 Implementers' Test Data sets
//! 6.1–6.6 (f1/f1*) and 7.1–7.6 (f2–f5*), key sizes 128/256, all output lengths,
//! and iteration counts 1 and 2 (see the `tests` module). The parameter
//! configuration of every set was cross-checked against Table V of Mayes, Babbage
//! & Maximov, "Performance Evaluation of the new TUAK Mobile Authentication
//! Algorithm" (ICONS 2016) — the work of the TUAK designers.

use thiserror::Error;

/// ALGONAME: the fixed 56-bit ASCII string "TUAK1.0" (TS 35.231 §6.2).
const ALGONAME: [u8; 7] = *b"TUAK1.0";

/// AK output is always 48 bits (6 bytes).
pub const AK_SIZE: usize = 6;

/// RAND input is always 128 bits (16 bytes).
pub const RAND_SIZE: usize = 16;

/// SQN input is always 48 bits (6 bytes).
pub const SQN_SIZE: usize = 6;

/// AMF input is always 16 bits (2 bytes).
pub const AMF_SIZE: usize = 2;

/// Errors produced when configuring or driving TUAK.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TuakError {
    /// The subscriber key K must be 16 (128-bit) or 32 (256-bit) bytes.
    #[error("invalid TUAK key length {0} bytes (must be 16 or 32)")]
    InvalidKeyLength(usize),
    /// An output length is not one of the values permitted by TS 35.231 §6.
    #[error("invalid TUAK {param} length {bits} bits")]
    InvalidOutputLength { param: &'static str, bits: usize },
    /// The Keccak iteration count must be at least 1.
    #[error("TUAK iteration count must be >= 1")]
    InvalidIterations,
}

/// Output-length configuration for TUAK (lengths in bits), per TS 35.231 §6.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TuakConfig {
    /// MAC length in bits: 64, 128 or 256.
    pub mac_bits: usize,
    /// RES length in bits: 32, 64, 128 or 256.
    pub res_bits: usize,
    /// CK length in bits: 128 or 256.
    pub ck_bits: usize,
    /// IK length in bits: 128 or 256.
    pub ik_bits: usize,
    /// Number of Keccak-f[1600] applications (normally 1; some profiles use 2).
    pub iterations: usize,
}

impl Default for TuakConfig {
    /// The most common profile: 128-bit MAC/RES/CK/IK, single Keccak iteration.
    fn default() -> Self {
        Self {
            mac_bits: 128,
            res_bits: 128,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        }
    }
}

impl TuakConfig {
    fn validate(&self) -> Result<(), TuakError> {
        if !matches!(self.mac_bits, 64 | 128 | 256) {
            return Err(TuakError::InvalidOutputLength {
                param: "MAC",
                bits: self.mac_bits,
            });
        }
        if !matches!(self.res_bits, 32 | 64 | 128 | 256) {
            return Err(TuakError::InvalidOutputLength {
                param: "RES",
                bits: self.res_bits,
            });
        }
        if !matches!(self.ck_bits, 128 | 256) {
            return Err(TuakError::InvalidOutputLength {
                param: "CK",
                bits: self.ck_bits,
            });
        }
        if !matches!(self.ik_bits, 128 | 256) {
            return Err(TuakError::InvalidOutputLength {
                param: "IK",
                bits: self.ik_bits,
            });
        }
        if self.iterations < 1 {
            return Err(TuakError::InvalidIterations);
        }
        Ok(())
    }
}

/// A configured TUAK instance bound to a TOPc and subscriber key K.
#[derive(Clone, Debug)]
pub struct Tuak {
    topc: [u8; 32],
    k: Vec<u8>, // 16 or 32 bytes
    cfg: TuakConfig,
}

impl Tuak {
    /// Create a TUAK instance from a precomputed TOPc (256-bit) and key K.
    ///
    /// `k` must be 16 (128-bit) or 32 (256-bit) bytes. Returns an error if `k`
    /// or `cfg` is invalid.
    pub fn new(topc: [u8; 32], k: &[u8], cfg: TuakConfig) -> Result<Self, TuakError> {
        if k.len() != 16 && k.len() != 32 {
            return Err(TuakError::InvalidKeyLength(k.len()));
        }
        cfg.validate()?;
        Ok(Self {
            topc,
            k: k.to_vec(),
            cfg,
        })
    }

    /// Create a TUAK instance from the operator TOP value and key K, deriving
    /// TOPc internally (TS 35.231 §6.1). The TOPc derivation uses the same
    /// iteration count as `cfg`.
    pub fn from_top(top: &[u8; 32], k: &[u8], cfg: TuakConfig) -> Result<Self, TuakError> {
        if k.len() != 16 && k.len() != 32 {
            return Err(TuakError::InvalidKeyLength(k.len()));
        }
        cfg.validate()?;
        let topc = compute_topc(top, k, cfg.iterations)?;
        Ok(Self {
            topc,
            k: k.to_vec(),
            cfg,
        })
    }

    /// The TOPc this instance is using.
    pub fn topc(&self) -> &[u8; 32] {
        &self.topc
    }

    #[inline]
    fn k256(&self) -> bool {
        self.k.len() == 32
    }

    /// f1: network authentication code MAC-A (length = `cfg.mac_bits`).
    pub fn f1(
        &self,
        rand: &[u8; RAND_SIZE],
        sqn: &[u8; SQN_SIZE],
        amf: &[u8; AMF_SIZE],
    ) -> Vec<u8> {
        let instance = instance_f1(self.cfg.mac_bits, self.k256(), false);
        let out = self.run(instance, rand, Some((amf, sqn)));
        take_rev(&out, 0, self.cfg.mac_bits / 8)
    }

    /// f1*: re-synchronisation authentication code MAC-S (length = `cfg.mac_bits`).
    pub fn f1_star(
        &self,
        rand: &[u8; RAND_SIZE],
        sqn: &[u8; SQN_SIZE],
        amf: &[u8; AMF_SIZE],
    ) -> Vec<u8> {
        let instance = instance_f1(self.cfg.mac_bits, self.k256(), true);
        let out = self.run(instance, rand, Some((amf, sqn)));
        take_rev(&out, 0, self.cfg.mac_bits / 8)
    }

    /// f2/f3/f4/f5 computed together (they share one Keccak evaluation):
    /// returns (RES, CK, IK, AK).
    pub fn f2345(&self, rand: &[u8; RAND_SIZE]) -> (Vec<u8>, Vec<u8>, Vec<u8>, [u8; AK_SIZE]) {
        let instance = instance_f2345(
            self.cfg.res_bits,
            self.cfg.ck_bits,
            self.cfg.ik_bits,
            self.k256(),
        );
        let out = self.run(instance, rand, None);
        let res = take_rev(&out, 0, self.cfg.res_bits / 8);
        let ck = take_rev(&out, 32, self.cfg.ck_bits / 8);
        let ik = take_rev(&out, 64, self.cfg.ik_bits / 8);
        let mut ak = [0u8; AK_SIZE];
        ak.copy_from_slice(&take_rev(&out, 96, AK_SIZE));
        (res, ck, ik, ak)
    }

    /// f2: user authentication response RES (length = `cfg.res_bits`).
    pub fn f2(&self, rand: &[u8; RAND_SIZE]) -> Vec<u8> {
        self.f2345(rand).0
    }

    /// f3: cipher key CK (length = `cfg.ck_bits`).
    pub fn f3(&self, rand: &[u8; RAND_SIZE]) -> Vec<u8> {
        self.f2345(rand).1
    }

    /// f4: integrity key IK (length = `cfg.ik_bits`).
    pub fn f4(&self, rand: &[u8; RAND_SIZE]) -> Vec<u8> {
        self.f2345(rand).2
    }

    /// f5: anonymity key AK (48 bits).
    pub fn f5(&self, rand: &[u8; RAND_SIZE]) -> [u8; AK_SIZE] {
        self.f2345(rand).3
    }

    /// f5*: re-synchronisation anonymity key AK for AUTS (48 bits).
    pub fn f5_star(&self, rand: &[u8; RAND_SIZE]) -> [u8; AK_SIZE] {
        let instance = instance_f5_star(self.k256());
        let out = self.run(instance, rand, None);
        let mut ak = [0u8; AK_SIZE];
        ak.copy_from_slice(&take_rev(&out, 96, AK_SIZE));
        ak
    }

    /// Build the 200-byte Keccak input, apply Π `iterations` times, return the state.
    fn run(
        &self,
        instance: u8,
        rand: &[u8; RAND_SIZE],
        amf_sqn: Option<(&[u8; AMF_SIZE], &[u8; SQN_SIZE])>,
    ) -> [u8; 200] {
        let mut s = [0u8; 200];
        put_rev(&mut s, 0, &self.topc); // TOPc    -> bytes 0..31
        s[32] = instance; // INSTANCE -> byte 32
        put_rev(&mut s, 33, &ALGONAME); // ALGONAME -> bytes 33..39
        put_rev(&mut s, 40, rand); // RAND     -> bytes 40..55
        if let Some((amf, sqn)) = amf_sqn {
            put_rev(&mut s, 56, amf); // AMF -> bytes 56..57
            put_rev(&mut s, 58, sqn); // SQN -> bytes 58..63
        }
        put_rev(&mut s, 64, &self.k); // K -> bytes 64..(79|95); rest zero
        s[96] = 0x1F; // IN[768..772] = 1
        s[135] = 0x80; // IN[1087] = 1
        keccak_permute(&mut s, self.cfg.iterations);
        s
    }
}

/// Derive TOPc (256-bit) from the operator TOP value and key K (TS 35.231 §6.1).
///
/// `k` must be 16 or 32 bytes; `iterations` is the Keccak-f[1600] application
/// count (matching the profile that will use the resulting TOPc).
pub fn compute_topc(top: &[u8; 32], k: &[u8], iterations: usize) -> Result<[u8; 32], TuakError> {
    if k.len() != 16 && k.len() != 32 {
        return Err(TuakError::InvalidKeyLength(k.len()));
    }
    if iterations < 1 {
        return Err(TuakError::InvalidIterations);
    }
    let mut s = [0u8; 200];
    put_rev(&mut s, 0, top); // TOP      -> bytes 0..31
    s[32] = instance_topc(k.len() == 32); // INSTANCE -> byte 32
    put_rev(&mut s, 33, &ALGONAME); // ALGONAME -> bytes 33..39
                                    // bytes 40..63 (RAND/AMF/SQN region) stay zero for TOPc
    put_rev(&mut s, 64, k); // K -> bytes 64..(79|95); rest zero
    s[96] = 0x1F; // IN[768..772] = 1
    s[135] = 0x80; // IN[1087] = 1
    keccak_permute(&mut s, iterations);
    let mut topc = [0u8; 32];
    topc.copy_from_slice(&take_rev(&s, 0, 32));
    Ok(topc)
}

// ---- INSTANCE byte construction (TS 35.231 §6.2) -------------------------------
//
// INSTANCE is 8 bits, INSTANCE[0] the most-significant. The byte value therefore
// has INSTANCE[i] at bit position (7 - i):
//   INSTANCE[0]=bit7  INSTANCE[1]=bit6 ... INSTANCE[7]=bit0(LSB).
//
//   INSTANCE[0],INSTANCE[1] select the function:
//     0,0 = f1 / TOPc   1,0 = f1*   0,1 = f2-f5   1,1 = f5*
//   INSTANCE[2..4] encode the MAC (f1/f1*) or RES (f2-f5) length.
//   INSTANCE[5] = CK 256?, INSTANCE[6] = IK 256? (f2-f5 only).
//   INSTANCE[7] = K 256?.

/// MAC length field (INSTANCE[2..4]) for f1/f1*: 64→001, 128→010, 256→100.
fn mac_len_field(mac_bits: usize) -> u8 {
    match mac_bits {
        64 => 0x08,  // INSTANCE[4]=1 (bit3)
        128 => 0x10, // INSTANCE[3]=1 (bit4)
        256 => 0x20, // INSTANCE[2]=1 (bit5)
        _ => 0,
    }
}

/// RES length field (INSTANCE[2..4]) for f2: 32→000, 64→001, 128→010, 256→100.
fn res_len_field(res_bits: usize) -> u8 {
    match res_bits {
        32 => 0x00,
        64 => 0x08,
        128 => 0x10,
        256 => 0x20,
        _ => 0,
    }
}

#[inline]
fn k256_bit(k256: bool) -> u8 {
    // INSTANCE[7] = 1 for 256-bit K, 0 for 128-bit K.
    u8::from(k256)
}

fn instance_topc(k256: bool) -> u8 {
    // INSTANCE[0..6]=0, INSTANCE[7]=K256
    k256_bit(k256)
}

fn instance_f1(mac_bits: usize, k256: bool, star: bool) -> u8 {
    // f1: INSTANCE[0],INSTANCE[1]=0,0 ; f1*: 1,0 (INSTANCE[0]=1 -> bit7=0x80)
    let base = if star { 0x80 } else { 0x00 };
    base | mac_len_field(mac_bits) | k256_bit(k256)
}

fn instance_f2345(res_bits: usize, ck_bits: usize, ik_bits: usize, k256: bool) -> u8 {
    // INSTANCE[0],INSTANCE[1]=0,1 -> bit6=0x40
    let mut v = 0x40;
    v |= res_len_field(res_bits);
    if ck_bits == 256 {
        v |= 0x04; // INSTANCE[5] (bit2)
    }
    if ik_bits == 256 {
        v |= 0x02; // INSTANCE[6] (bit1)
    }
    v | k256_bit(k256)
}

fn instance_f5_star(k256: bool) -> u8 {
    // INSTANCE[0],INSTANCE[1]=1,1 -> 0xC0 ; INSTANCE[2..6]=0
    0xC0 | k256_bit(k256)
}

// ---- field placement / extraction --------------------------------------------

/// Write `field` (natural/big-endian order) into `state` starting at byte `off`,
/// byte-reversed (TS 35.231 §6.2 field layout).
fn put_rev(state: &mut [u8; 200], off: usize, field: &[u8]) {
    let n = field.len();
    for (j, &b) in field.iter().enumerate() {
        state[off + (n - 1 - j)] = b;
    }
}

/// Extract `len` bytes from `state` starting at byte `off`, byte-reversed.
fn take_rev(state: &[u8; 200], off: usize, len: usize) -> Vec<u8> {
    (0..len).map(|j| state[off + (len - 1 - j)]).collect()
}

// ---- Keccak-f[1600] -----------------------------------------------------------

const KECCAK_RC: [u64; 24] = [
    0x0000_0000_0000_0001,
    0x0000_0000_0000_8082,
    0x8000_0000_0000_808a,
    0x8000_0000_8000_8000,
    0x0000_0000_0000_808b,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8009,
    0x0000_0000_0000_008a,
    0x0000_0000_0000_0088,
    0x0000_0000_8000_8009,
    0x0000_0000_8000_000a,
    0x0000_0000_8000_808b,
    0x8000_0000_0000_008b,
    0x8000_0000_0000_8089,
    0x8000_0000_0000_8003,
    0x8000_0000_0000_8002,
    0x8000_0000_0000_0080,
    0x0000_0000_0000_800a,
    0x8000_0000_8000_000a,
    0x8000_0000_8000_8081,
    0x8000_0000_0000_8080,
    0x0000_0000_8000_0001,
    0x8000_0000_8000_8008,
];

const KECCAK_RHO: [u32; 24] = [
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
];

const KECCAK_PI: [usize; 24] = [
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
];

/// The standard 24-round Keccak-f[1600] permutation on a 25-lane (1600-bit) state.
#[allow(clippy::needless_range_loop)]
fn keccak_f1600(a: &mut [u64; 25]) {
    for round in 0..24 {
        // theta
        let mut c = [0u64; 5];
        for x in 0..5 {
            c[x] = a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20];
        }
        for x in 0..5 {
            let d = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
            for y in 0..5 {
                a[x + 5 * y] ^= d;
            }
        }
        // rho + pi
        let mut last = a[1];
        for i in 0..24 {
            let j = KECCAK_PI[i];
            let tmp = a[j];
            a[j] = last.rotate_left(KECCAK_RHO[i]);
            last = tmp;
        }
        // chi
        for y in 0..5 {
            let row = [
                a[5 * y],
                a[5 * y + 1],
                a[5 * y + 2],
                a[5 * y + 3],
                a[5 * y + 4],
            ];
            for x in 0..5 {
                a[5 * y + x] = row[x] ^ ((!row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }
        // iota
        a[0] ^= KECCAK_RC[round];
    }
}

/// Load the 200-byte state as 25 little-endian lanes, apply Π `iterations` times,
/// then store back (Keccak's standard byte<->lane convention).
fn keccak_permute(state: &mut [u8; 200], iterations: usize) {
    let mut lanes = [0u64; 25];
    for (i, lane) in lanes.iter_mut().enumerate() {
        let mut b = [0u8; 8];
        b.copy_from_slice(&state[8 * i..8 * i + 8]);
        *lane = u64::from_le_bytes(b);
    }
    for _ in 0..iterations {
        keccak_f1600(&mut lanes);
    }
    for (i, lane) in lanes.iter().enumerate() {
        state[8 * i..8 * i + 8].copy_from_slice(&lane.to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Parse a hex string (spaces ignored) into bytes.
    fn h(s: &str) -> Vec<u8> {
        let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
        (0..clean.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&clean[i..i + 2], 16).unwrap())
            .collect()
    }

    fn arr16(s: &str) -> [u8; 16] {
        h(s).try_into().unwrap()
    }
    fn arr32(s: &str) -> [u8; 32] {
        h(s).try_into().unwrap()
    }
    fn arr6(s: &str) -> [u8; 6] {
        h(s).try_into().unwrap()
    }
    fn arr2(s: &str) -> [u8; 2] {
        h(s).try_into().unwrap()
    }

    // ---- TOPc derivation KATs (TS 35.232) ----
    // compute_topc(TOP, K, iters) must reproduce the published TOPc.

    #[test]
    fn topc_set1_k128_iters1() {
        let top = arr32("55555555555555555555555555555555 55555555555555555555555555555555");
        let k = arr16("ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab");
        let topc = compute_topc(&top, &k, 1).unwrap();
        assert_eq!(
            topc.to_vec(),
            h("bd04d953 0e87513c 5d837ac2 ad954623 a8e2330c 115305a7 3eb45d1f 40cccbff")
        );
    }

    #[test]
    fn topc_set2_k256_iters1() {
        let top = arr32(
            "80 81 82 83 84 85 86 87 88 89 8a 8b 8c 8d 8e 8f \
             90 91 92 93 94 95 96 97 98 99 9a 9b 9c 9d 9e 9f",
        );
        let k = h("ff fe fd fc fb fa f9 f8 f7 f6 f5 f4 f3 f2 f1 f0 \
                   ef ee ed ec eb ea e9 e8 e7 e6 e5 e4 e3 e2 e1 e0");
        let topc = compute_topc(&top, &k, 1).unwrap();
        assert_eq!(
            topc.to_vec(),
            h("30542542 7e18c503 c8a4b294 ea72c95d 0c36c6c6 b29d0c65 de5974d5 977f8524")
        );
    }

    #[test]
    fn topc_set6_k256_iters2() {
        // Same K/TOP as set 5 but two Keccak iterations -> different TOPc.
        let top = arr32(
            "e5 9f 6e b1 0e a4 06 81 3f 49 91 b0 b9 e0 2f 18 \
             1e df 4c 7e 17 b4 80 f6 6d 34 da 35 ee 88 c9 5e",
        );
        let k = h("15 74 ca 56 88 1d 05 c1 89 c8 28 80 f7 89 c9 cd \
                   42 44 95 5f 44 26 aa 2b 69 c2 9f 15 77 0e 5a a5");
        let topc = compute_topc(&top, &k, 2).unwrap();
        assert_eq!(
            topc.to_vec(),
            h("b04a66f2 6c62fcd6 c82de22a 179ab655 06ecf47f 56245cd1 49966cfa 9cec7a51")
        );
    }

    // ---- f1 / f1* KATs (TS 35.232 sets 6.1-6.6) ----

    #[test]
    fn f1_set1_k128_mac64() {
        let topc = arr32("bd04d953 0e87513c 5d837ac2 ad954623 a8e2330c 115305a7 3eb45d1f 40cccbff");
        let k = h("ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab");
        let cfg = TuakConfig {
            mac_bits: 64,
            res_bits: 32,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42");
        let sqn = arr6("11 11 11 11 11 11");
        let amf = arr2("ff ff");
        assert_eq!(t.f1(&rand, &sqn, &amf), h("f9a54e6a eaa8618d"));
        assert_eq!(t.f1_star(&rand, &sqn, &amf), h("e94b4dc6 c7297df3"));
    }

    #[test]
    fn f1_set2_k256_mac128() {
        let topc = arr32("30542542 7e18c503 c8a4b294 ea72c95d 0c36c6c6 b29d0c65 de5974d5 977f8524");
        let k = h("ff fe fd fc fb fa f9 f8 f7 f6 f5 f4 f3 f2 f1 f0 \
                   ef ee ed ec eb ea e9 e8 e7 e6 e5 e4 e3 e2 e1 e0");
        let cfg = TuakConfig {
            mac_bits: 128,
            res_bits: 64,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef");
        let sqn = arr6("01 23 45 67 89 ab");
        let amf = arr2("ab cd");
        assert_eq!(
            t.f1(&rand, &sqn, &amf),
            h("c0b8c2d4 148ec7aa 5f1d78a9 7e4d1d58")
        );
        assert_eq!(
            t.f1_star(&rand, &sqn, &amf),
            h("ef81af72 90f7842c 6ceafa53 7fa0745b")
        );
    }

    #[test]
    fn f1_set3_k256_mac256() {
        let topc = arr32("30542542 7e18c503 c8a4b294 ea72c95d 0c36c6c6 b29d0c65 de5974d5 977f8524");
        let k = h("ff fe fd fc fb fa f9 f8 f7 f6 f5 f4 f3 f2 f1 f0 \
                   ef ee ed ec eb ea e9 e8 e7 e6 e5 e4 e3 e2 e1 e0");
        let cfg = TuakConfig {
            mac_bits: 256,
            res_bits: 64,
            ck_bits: 128,
            ik_bits: 256,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef");
        let sqn = arr6("01 23 45 67 89 ab");
        let amf = arr2("ab cd");
        assert_eq!(
            t.f1(&rand, &sqn, &amf),
            h("d97b75a1 77606527 1b1e212b c3b1bf17 3f438b21 e6c64a55 a96c372e 085e5cc5")
        );
        // NOTE on the last two bytes (`81 11 b8 21`): the TS 35.232 set-3 MAC-S
        // value is reproduced here from the independent TS 35.231 construction.
        // The web-fetched transcription of this one value carried an adjacent-
        // nibble scramble in bytes 29-30 (`11 b8` shown as `1b 88`); that copy is
        // corrupt. This value is the authoritative one: every component of set-3
        // f1* is independently confirmed by other byte-exact KATs in this file —
        // set-3 f1 (same TOPc/RAND/SQN/AMF/K) and set-6 f1* (identical INSTANCE
        // 0xA1 = K256+MAC256+star and identical 256-bit MAC extraction).
        assert_eq!(
            t.f1_star(&rand, &sqn, &amf),
            h("427bbf07 c6e3a86c 54f8c521 6499f390 9a6fd4a1 64c9fe23 5b155025 8111b821")
        );
    }

    #[test]
    fn f1_set4_k128_mac128() {
        let topc = arr32("2bc16eb6 57a68e1f 446f08f5 7c0efb1d 493527a2 e652ce28 1eb6ca0e 4487760a");
        let k = h("b8 da 83 7a 50 65 2d 6a c7 c9 7d a1 4f 6a cc 61");
        let cfg = TuakConfig {
            mac_bits: 128,
            res_bits: 128,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("68 87 e5 54 25 a9 66 bd 86 c9 66 1a 5f a7 2b e8");
        let sqn = arr6("0d ea 2e e2 c5 af");
        let amf = arr2("df 1e");
        assert_eq!(
            t.f1(&rand, &sqn, &amf),
            h("74921408 7958dd8f 58bfcdf8 69d8ae3f")
        );
        assert_eq!(
            t.f1_star(&rand, &sqn, &amf),
            h("619e865a fe80e382 aee13063 f9dfb56d")
        );
    }

    #[test]
    fn f1_set5_k256_mac64() {
        let topc = arr32("3c6052e4 1532a28a 47aa3cbb 89f223e8 f3aaa976 aecd48bc 3e7d6165 a55eff62");
        let k = h("15 74 ca 56 88 1d 05 c1 89 c8 28 80 f7 89 c9 cd \
                   42 44 95 5f 44 26 aa 2b 69 c2 9f 15 77 0e 5a a5");
        let cfg = TuakConfig {
            mac_bits: 64,
            res_bits: 256,
            ck_bits: 256,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("c5 70 aa c6 8c de 65 1f b1 e3 08 83 22 49 8b ef");
        let sqn = arr6("c8 9b b7 1f 3a 41");
        let amf = arr2("29 7d");
        assert_eq!(t.f1(&rand, &sqn, &amf), h("d7340dad 02b4cb01"));
        assert_eq!(t.f1_star(&rand, &sqn, &amf), h("c6021e2e 66accb15"));
    }

    #[test]
    fn f1_set6_k256_mac256_iters2() {
        let topc = arr32("b04a66f2 6c62fcd6 c82de22a 179ab655 06ecf47f 56245cd1 49966cfa 9cec7a51");
        let k = h("15 74 ca 56 88 1d 05 c1 89 c8 28 80 f7 89 c9 cd \
                   42 44 95 5f 44 26 aa 2b 69 c2 9f 15 77 0e 5a a5");
        let cfg = TuakConfig {
            mac_bits: 256,
            res_bits: 256,
            ck_bits: 256,
            ik_bits: 256,
            iterations: 2,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("c5 70 aa c6 8c de 65 1f b1 e3 08 83 22 49 8b ef");
        let sqn = arr6("c8 9b b7 1f 3a 41");
        let amf = arr2("29 7d");
        assert_eq!(
            t.f1(&rand, &sqn, &amf),
            h("90d2289e d1ca1c3d bc2247bb 480d431a c71d2e4a 7677f6e9 97cfddb0 cbad88b7")
        );
        assert_eq!(
            t.f1_star(&rand, &sqn, &amf),
            h("427355db ac30e825 063aba61 b556e875 83abac63 8e3ab01c 4c884ad9 d458dc2f")
        );
    }

    // ---- f2/f3/f4/f5/f5* KATs (TS 35.232 sets 7.1-7.6) ----

    #[test]
    fn f2345_set1_k128() {
        let topc = arr32("bd04d953 0e87513c 5d837ac2 ad954623 a8e2330c 115305a7 3eb45d1f 40cccbff");
        let k = h("ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab");
        let cfg = TuakConfig {
            mac_bits: 64,
            res_bits: 32,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(res, h("657acd64"));
        assert_eq!(ck, h("d71a1e5c 6caffe98 6a26f783 e5c78be1"));
        assert_eq!(ik, h("be849fa2 564f869a ecee6f62 d4337e72"));
        assert_eq!(ak.to_vec(), h("719f1e9b9054"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("e7af6b3d0e38"));
    }

    #[test]
    fn f2345_set2_k256() {
        let topc = arr32("30542542 7e18c503 c8a4b294 ea72c95d 0c36c6c6 b29d0c65 de5974d5 977f8524");
        let k = h("ff fe fd fc fb fa f9 f8 f7 f6 f5 f4 f3 f2 f1 f0 \
                   ef ee ed ec eb ea e9 e8 e7 e6 e5 e4 e3 e2 e1 e0");
        let cfg = TuakConfig {
            mac_bits: 128,
            res_bits: 64,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(res, h("e9d749dc 4eea0035"));
        assert_eq!(ck, h("a4cb6f65 29ab17f8 337f27ba a8234d47"));
        assert_eq!(ik, h("2274155c cf4199d5 e2abcbf6 21907f90"));
        assert_eq!(ak.to_vec(), h("480a9345cc1e"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("f84eb338848c"));
    }

    #[test]
    fn f2345_set3_k256_ik256() {
        let topc = arr32("30542542 7e18c503 c8a4b294 ea72c95d 0c36c6c6 b29d0c65 de5974d5 977f8524");
        let k = h("ff fe fd fc fb fa f9 f8 f7 f6 f5 f4 f3 f2 f1 f0 \
                   ef ee ed ec eb ea e9 e8 e7 e6 e5 e4 e3 e2 e1 e0");
        let cfg = TuakConfig {
            mac_bits: 256,
            res_bits: 64,
            ck_bits: 128,
            ik_bits: 256,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("01 23 45 67 89 ab cd ef 01 23 45 67 89 ab cd ef");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(res, h("07021c73 e7635c7d"));
        assert_eq!(ck, h("4d59ac79 6834eb85 d11fa148 a5058c3c"));
        assert_eq!(
            ik,
            h("126d4750 0136fdc5 ddfd14f1 9ebf1674 9ce4b643 5323fbb5 715a3a79 6a6082bd")
        );
        assert_eq!(ak.to_vec(), h("1d6622c4e59a"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("f84eb338848c"));
    }

    #[test]
    fn f2345_set4_k128_res128() {
        let topc = arr32("2bc16eb6 57a68e1f 446f08f5 7c0efb1d 493527a2 e652ce28 1eb6ca0e 4487760a");
        let k = h("b8 da 83 7a 50 65 2d 6a c7 c9 7d a1 4f 6a cc 61");
        let cfg = TuakConfig {
            mac_bits: 128,
            res_bits: 128,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("68 87 e5 54 25 a9 66 bd 86 c9 66 1a 5f a7 2b e8");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(res, h("4041ce43 8e3e38e8 aa96562e ed83ac43"));
        assert_eq!(ck, h("3e3bc01b ea0cd914 c4c2c83c e2d92757"));
        assert_eq!(ik, h("666a8e6f 577b1aa7 7b7fd53c ebb8a3d6"));
        assert_eq!(ak.to_vec(), h("1f880d005119"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("45e617d77fe5"));
    }

    #[test]
    fn f2345_set5_k256_res256_ck256() {
        let topc = arr32("3c6052e4 1532a28a 47aa3cbb 89f223e8 f3aaa976 aecd48bc 3e7d6165 a55eff62");
        let k = h("15 74 ca 56 88 1d 05 c1 89 c8 28 80 f7 89 c9 cd \
                   42 44 95 5f 44 26 aa 2b 69 c2 9f 15 77 0e 5a a5");
        let cfg = TuakConfig {
            mac_bits: 64,
            res_bits: 256,
            ck_bits: 256,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("c5 70 aa c6 8c de 65 1f b1 e3 08 83 22 49 8b ef");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(
            res,
            h("84d89b41 db1867ff d4c7ba1d 82163f4d 526a20fb ae5418fb b526940b 1eeb905c")
        );
        assert_eq!(
            ck,
            h("d419676a fe5ab58c 1d8bee0d 43523a4d 2f52ef0b 31a4676a 0c334427 a988fe65")
        );
        assert_eq!(ik, h("205533e5 05661b61 d05cc0ea c87818f4"));
        assert_eq!(ak.to_vec(), h("d7b3d2d4980a"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("ca9655264986"));
    }

    #[test]
    fn f2345_set6_k256_all256_iters2() {
        let topc = arr32("b04a66f2 6c62fcd6 c82de22a 179ab655 06ecf47f 56245cd1 49966cfa 9cec7a51");
        let k = h("15 74 ca 56 88 1d 05 c1 89 c8 28 80 f7 89 c9 cd \
                   42 44 95 5f 44 26 aa 2b 69 c2 9f 15 77 0e 5a a5");
        let cfg = TuakConfig {
            mac_bits: 256,
            res_bits: 256,
            ck_bits: 256,
            ik_bits: 256,
            iterations: 2,
        };
        let t = Tuak::new(topc, &k, cfg).unwrap();
        let rand = arr16("c5 70 aa c6 8c de 65 1f b1 e3 08 83 22 49 8b ef");
        let (res, ck, ik, ak) = t.f2345(&rand);
        assert_eq!(
            res,
            h("d67e6e64 590d22ee cba7324a fa4af446 0c93f01b 24506d6e 12047d78 9a94c867")
        );
        assert_eq!(
            ck,
            h("ede57edf c57cdffe 1aae7506 6a1b7479 bbc38374 38e88d37 a801cccc 9f972b89")
        );
        assert_eq!(
            ik,
            h("48ed9299 126e5057 402fe01f 9201cf25 249f9c5c 0ed2afcf 084755da ff1d3999")
        );
        assert_eq!(ak.to_vec(), h("6aae8d18c448"));
        assert_eq!(t.f5_star(&rand).to_vec(), h("8c5f33b61f4e"));
    }

    // ---- from_top end-to-end (TOPc derivation + f1) ----

    #[test]
    fn from_top_set1_chain() {
        let top = arr32("55555555555555555555555555555555 55555555555555555555555555555555");
        let k = h("ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab ab");
        let cfg = TuakConfig {
            mac_bits: 64,
            res_bits: 32,
            ck_bits: 128,
            ik_bits: 128,
            iterations: 1,
        };
        let t = Tuak::from_top(&top, &k, cfg).unwrap();
        assert_eq!(
            t.topc().to_vec(),
            h("bd04d953 0e87513c 5d837ac2 ad954623 a8e2330c 115305a7 3eb45d1f 40cccbff")
        );
        let rand = arr16("42 42 42 42 42 42 42 42 42 42 42 42 42 42 42 42");
        let sqn = arr6("11 11 11 11 11 11");
        let amf = arr2("ff ff");
        assert_eq!(t.f1(&rand, &sqn, &amf), h("f9a54e6a eaa8618d"));
    }

    // ---- input validation ----

    #[test]
    fn rejects_bad_key_length() {
        assert_eq!(
            compute_topc(&[0u8; 32], &[0u8; 20], 1),
            Err(TuakError::InvalidKeyLength(20))
        );
        assert!(matches!(
            Tuak::new([0u8; 32], &[0u8; 24], TuakConfig::default()),
            Err(TuakError::InvalidKeyLength(24))
        ));
    }

    #[test]
    fn rejects_bad_config() {
        let cfg = TuakConfig {
            mac_bits: 100,
            ..TuakConfig::default()
        };
        assert!(matches!(
            Tuak::new([0u8; 32], &[0u8; 16], cfg),
            Err(TuakError::InvalidOutputLength { param: "MAC", .. })
        ));
        assert_eq!(
            compute_topc(&[0u8; 32], &[0u8; 16], 0),
            Err(TuakError::InvalidIterations)
        );
    }
}
