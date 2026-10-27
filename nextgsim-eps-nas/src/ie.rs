//! EPS NAS information elements (TS 24.301 §9.9) and the octet reader the
//! message codecs share.
//!
//! Only the IEs the UE side of attach, authentication, security mode control
//! and default bearer activation carries or reads are modelled; every other
//! optional IE of a received message is skipped by its format (§9.6 / TS 24.007
//! §11.2.4) so a newer network's additions do not break decoding.

use nextgsim_common::Plmn;

/// Errors decoding a NAS message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum NasDecodeError {
    /// The message ended inside an IE.
    #[error("message truncated at octet {0}")]
    Truncated(usize),
    /// A value outside its IE's range.
    #[error("invalid {0}")]
    Invalid(&'static str),
    /// Not the message the decoder was asked for.
    #[error("unexpected message type 0x{0:02x}")]
    UnexpectedType(u8),
}

/// A cursor over a received message.
#[derive(Debug, Clone)]
pub struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    /// A reader at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    /// Octets left.
    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// The next octet without consuming it.
    pub fn peek(&self) -> Option<u8> {
        self.buf.get(self.pos).copied()
    }

    /// One octet.
    pub fn u8(&mut self) -> Result<u8, NasDecodeError> {
        let v = *self
            .buf
            .get(self.pos)
            .ok_or(NasDecodeError::Truncated(self.pos))?;
        self.pos += 1;
        Ok(v)
    }

    /// `n` octets.
    pub fn bytes(&mut self, n: usize) -> Result<&'a [u8], NasDecodeError> {
        let end = self
            .pos
            .checked_add(n)
            .ok_or(NasDecodeError::Truncated(self.pos))?;
        let s = self
            .buf
            .get(self.pos..end)
            .ok_or(NasDecodeError::Truncated(self.pos))?;
        self.pos = end;
        Ok(s)
    }

    /// An LV value (1-octet length).
    pub fn lv(&mut self) -> Result<&'a [u8], NasDecodeError> {
        let n = self.u8()?;
        self.bytes(usize::from(n))
    }

    /// An LV-E value (2-octet length).
    pub fn lv_e(&mut self) -> Result<&'a [u8], NasDecodeError> {
        let hi = self.u8()?;
        let lo = self.u8()?;
        self.bytes(usize::from(u16::from_be_bytes([hi, lo])))
    }

    /// The optional part of a message: `(IEI, value)` for each IE, where a
    /// type-1 IE (half-octet IEI, TS 24.007 §11.2.4) is returned with its IEI in
    /// the high nibble and its value in the low nibble of a one-octet value.
    /// `tv_lengths` lists the type-3 (TV) IEIs of the message with their total
    /// length in octets including the IEI; every other IEI is TLV, or TLV-E
    /// when its high nibble is 7 (TS 24.007 §11.2.4, as TS 24.301 assigns them).
    pub fn optional_ies(
        &mut self,
        tv_lengths: &[(u8, usize)],
    ) -> Result<Vec<(u8, &'a [u8])>, NasDecodeError> {
        let mut out = Vec::new();
        while self.remaining() > 0 {
            let start = self.pos;
            let iei = self.u8()?;
            if iei & 0x80 != 0 {
                // Type 1 (TV, one octet) or type 2 (T only): IEI in bits 8-5.
                out.push((iei & 0xF0, &self.buf[start..start + 1]));
                continue;
            }
            if let Some(&(_, len)) = tv_lengths.iter().find(|(i, _)| *i == iei) {
                let v = self.bytes(len - 1)?;
                out.push((iei, v));
                continue;
            }
            let v = if iei & 0xF0 == 0x70 {
                self.lv_e()?
            } else {
                self.lv()?
            };
            out.push((iei, v));
        }
        Ok(out)
    }
}

/// Append an LV value.
pub fn put_lv(out: &mut Vec<u8>, v: &[u8]) {
    out.push(u8::try_from(v.len()).expect("LV value of more than 255 octets"));
    out.extend_from_slice(v);
}

/// Append an LV-E value.
pub fn put_lv_e(out: &mut Vec<u8>, v: &[u8]) {
    out.extend_from_slice(
        &u16::try_from(v.len())
            .expect("LV-E value of more than 65535 octets")
            .to_be_bytes(),
    );
    out.extend_from_slice(v);
}

/// Append a TLV IE.
pub fn put_tlv(out: &mut Vec<u8>, iei: u8, v: &[u8]) {
    out.push(iei);
    put_lv(out, v);
}

/// TBCD digits of `s` (TS 24.008 §10.5.1.4 / TS 24.301 §9.9.3.12), first
/// digit in the low nibble.
fn digits(s: &str) -> Option<Vec<u8>> {
    s.bytes()
        .map(|b| b.is_ascii_digit().then_some(b - b'0'))
        .collect()
}

/// A mobile identity with digits: IMSI, IMEI or IMEISV (TS 24.008 §10.5.1.4,
/// which §9.9.2.3 and the IMSI form of §9.9.3.12 use).
fn digit_identity(type_of_identity: u8, s: &str) -> Option<Vec<u8>> {
    let d = digits(s)?;
    let (first, rest) = d.split_first()?;
    let odd = d.len() % 2 == 1;
    let mut out = vec![(first << 4) | (u8::from(odd) << 3) | (type_of_identity & 0x07)];
    for pair in rest.chunks(2) {
        let hi = pair.get(1).copied().unwrap_or(0x0F);
        out.push((hi << 4) | pair[0]);
    }
    Some(out)
}

fn identity_digits(v: &[u8]) -> String {
    let mut s = String::new();
    let odd = v[0] & 0x08 != 0;
    s.push(char::from(b'0' + (v[0] >> 4)));
    for (i, b) in v[1..].iter().enumerate() {
        s.push(char::from(b'0' + (b & 0x0F)));
        let last = i == v.len() - 2;
        if (odd || !last) && (b >> 4) <= 9 {
            s.push(char::from(b'0' + (b >> 4)));
        }
    }
    s
}

/// The GUTI (TS 24.301 §9.9.3.12, TS 23.003 §2.8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Guti {
    /// MCC/MNC.
    pub plmn: Plmn,
    /// MME Group ID.
    pub mme_group_id: u16,
    /// MME Code.
    pub mme_code: u8,
    /// M-TMSI.
    pub m_tmsi: u32,
}

/// The EPS mobile identity (§9.9.3.12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpsMobileIdentity {
    /// IMSI digits.
    Imsi(String),
    /// GUTI.
    Guti(Guti),
    /// IMEI digits.
    Imei(String),
}

const ID_TYPE_IMSI: u8 = 1;
const ID_TYPE_IMEI: u8 = 3;
const ID_TYPE_GUTI: u8 = 6;

impl EpsMobileIdentity {
    /// The value part (without length).
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Imsi(s) => digit_identity(ID_TYPE_IMSI, s).expect("IMSI is digits"),
            Self::Imei(s) => digit_identity(ID_TYPE_IMEI, s).expect("IMEI is digits"),
            Self::Guti(g) => {
                // Octet 3: 1111 | odd/even 0 | type 110.
                let mut out = vec![0xF0 | ID_TYPE_GUTI];
                out.extend_from_slice(&g.plmn.encode());
                out.extend_from_slice(&g.mme_group_id.to_be_bytes());
                out.push(g.mme_code);
                out.extend_from_slice(&g.m_tmsi.to_be_bytes());
                out
            }
        }
    }

    /// Decode the value part.
    pub fn decode(v: &[u8]) -> Result<Self, NasDecodeError> {
        let first = *v
            .first()
            .ok_or(NasDecodeError::Invalid("EPS mobile identity"))?;
        match first & 0x07 {
            ID_TYPE_GUTI => {
                if v.len() != 11 {
                    return Err(NasDecodeError::Invalid("GUTI length"));
                }
                Ok(Self::Guti(Guti {
                    plmn: Plmn::decode([v[1], v[2], v[3]]),
                    mme_group_id: u16::from_be_bytes([v[4], v[5]]),
                    mme_code: v[6],
                    m_tmsi: u32::from_be_bytes([v[7], v[8], v[9], v[10]]),
                }))
            }
            ID_TYPE_IMSI => Ok(Self::Imsi(identity_digits(v))),
            ID_TYPE_IMEI => Ok(Self::Imei(identity_digits(v))),
            _ => Err(NasDecodeError::Invalid("EPS mobile identity type")),
        }
    }
}

/// A mobile identity answered in IDENTITY RESPONSE or SECURITY MODE COMPLETE
/// (TS 24.008 §10.5.1.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MobileIdentity {
    /// IMSI.
    Imsi(String),
    /// IMEI.
    Imei(String),
    /// IMEISV.
    Imeisv(String),
    /// "No identity" (type 000), the answer when the requested one is absent.
    NoIdentity,
}

impl MobileIdentity {
    /// The value part.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            Self::Imsi(s) => digit_identity(1, s).expect("IMSI is digits"),
            Self::Imei(s) => digit_identity(2, s).expect("IMEI is digits"),
            Self::Imeisv(s) => digit_identity(3, s).expect("IMEISV is digits"),
            // §10.5.1.4: "no identity", even number of identity digits, filler.
            Self::NoIdentity => vec![0xF0],
        }
    }
}

/// The UE network capability (§9.9.3.34): which EEA and EIA the UE supports,
/// as octets 3 and 4 with EEA0/EIA0 in bit 8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UeNetworkCapability {
    /// EEA0..EEA7, bit 8 = EEA0.
    pub eea: u8,
    /// EIA0..EIA7, bit 8 = EIA0.
    pub eia: u8,
}

impl UeNetworkCapability {
    /// From algorithm identity lists.
    pub fn from_algorithms(eea: &[u8], eia: &[u8]) -> Self {
        let mask = |algs: &[u8]| {
            algs.iter()
                .filter(|a| **a < 8)
                .fold(0u8, |m, a| m | (0x80 >> a))
        };
        Self {
            eea: mask(eea),
            eia: mask(eia),
        }
    }

    /// Whether `eea` is supported.
    pub fn supports_eea(&self, alg: u8) -> bool {
        alg < 8 && self.eea & (0x80 >> alg) != 0
    }

    /// Whether `eia` is supported.
    pub fn supports_eia(&self, alg: u8) -> bool {
        alg < 8 && self.eia & (0x80 >> alg) != 0
    }

    /// The value part: octets 3 and 4 only. The E-UTRAN-only UE has no UMTS
    /// or GPRS algorithms to state, and §9.9.3.34 lets the IE end after
    /// octet 4.
    pub fn encode(&self) -> Vec<u8> {
        vec![self.eea, self.eia]
    }
}

/// The NAS key set identifier (§9.9.3.21): `value` 0..6, 7 = "no key is
/// available".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NasKsi {
    /// Type of security context flag: false = native.
    pub mapped: bool,
    /// The identifier.
    pub value: u8,
}

impl NasKsi {
    /// "No key is available" (UE to network).
    pub const NO_KEY: Self = Self {
        mapped: false,
        value: 7,
    };

    /// The half-octet.
    pub fn nibble(&self) -> u8 {
        (u8::from(self.mapped) << 3) | (self.value & 0x07)
    }

    /// From the half-octet.
    pub fn from_nibble(n: u8) -> Self {
        Self {
            mapped: n & 0x08 != 0,
            value: n & 0x07,
        }
    }
}

/// A GPRS timer value (§9.9.3.16, TS 24.008 §10.5.7.3) in seconds, `None`
/// when deactivated.
pub fn gprs_timer_seconds(octet: u8) -> Option<u64> {
    let value = u64::from(octet & 0x1F);
    match octet >> 5 {
        0 => Some(value * 2),
        1 => Some(value * 60),
        2 => Some(value * 360),
        7 => None,
        // "Other values shall be interpreted as multiples of 1 minute".
        _ => Some(value * 60),
    }
}

/// A GPRS timer 3 value (TS 24.008 §10.5.7.4a) in seconds, as the T3412
/// extended value IE (§9.9.3.16B) of an integrity-protected message reads it:
/// unit 110 is 320 hours (NOTE 1). `None` when deactivated; unit 111 is not
/// applicable to T3412 and makes the IE count as absent (NOTE 2), which the
/// caller decides.
pub fn gprs_timer3_seconds(octet: u8) -> Option<u64> {
    let value = u64::from(octet & 0x1F);
    match octet >> 5 {
        0 => Some(value * 600),
        1 => Some(value * 3600),
        2 => Some(value * 36_000),
        3 => Some(value * 2),
        4 => Some(value * 30),
        5 => Some(value * 60),
        6 => Some(value * 320 * 3600),
        _ => None,
    }
}

/// EPS bearer context status (§9.9.2.1): bit `n` is EBI(n), "not BEARER
/// CONTEXT-INACTIVE" when set. Octet 3 carries EBI(7)..EBI(0) in bits 8..1,
/// octet 4 EBI(15)..EBI(8); EBI(0) is spare.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EpsBearerContextStatus(u16);

impl EpsBearerContextStatus {
    /// The status with `ebis` active (EBIs outside 1..=15 are ignored).
    pub fn from_ebis(ebis: impl IntoIterator<Item = u8>) -> Self {
        Self(
            ebis.into_iter()
                .filter(|e| (1..=15).contains(e))
                .fold(0, |acc, e| acc | 1 << e),
        )
    }

    /// Whether EBI `ebi` is not BEARER CONTEXT-INACTIVE.
    pub fn is_active(self, ebi: u8) -> bool {
        (1..=15).contains(&ebi) && self.0 & (1 << ebi) != 0
    }

    /// The value part: octets 3 and 4.
    pub fn encode(self) -> [u8; 2] {
        [(self.0 & 0xFF) as u8, (self.0 >> 8) as u8]
    }

    /// Decode the value part. Octets past 4 are ignored, as a longer IE from a
    /// later release would be; the spare EBI(0) bit is not read.
    pub fn decode(v: &[u8]) -> Result<Self, NasDecodeError> {
        let [lo, hi, ..] = v else {
            return Err(NasDecodeError::Invalid("EPS bearer context status length"));
        };
        Ok(Self(u16::from_le_bytes([*lo, *hi]) & !1))
    }
}

/// A tracking area identity (§9.9.3.32).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tai {
    /// PLMN.
    pub plmn: Plmn,
    /// TAC.
    pub tac: u16,
}

/// Decode a tracking area identity list (§9.9.3.33) value.
pub fn decode_tai_list(v: &[u8]) -> Result<Vec<Tai>, NasDecodeError> {
    let mut r = Reader::new(v);
    let mut out = Vec::new();
    while r.remaining() > 0 {
        let head = r.u8()?;
        let n = usize::from(head & 0x1F) + 1;
        match (head >> 5) & 0x03 {
            0 => {
                let p = r.bytes(3)?;
                let plmn = Plmn::decode([p[0], p[1], p[2]]);
                for _ in 0..n {
                    let t = r.bytes(2)?;
                    out.push(Tai {
                        plmn,
                        tac: u16::from_be_bytes([t[0], t[1]]),
                    });
                }
            }
            1 => {
                let p = r.bytes(3)?;
                let plmn = Plmn::decode([p[0], p[1], p[2]]);
                let t = r.bytes(2)?;
                let first = u16::from_be_bytes([t[0], t[1]]);
                for i in 0..n {
                    out.push(Tai {
                        plmn,
                        tac: first.wrapping_add(i as u16),
                    });
                }
            }
            2 => {
                for _ in 0..n {
                    let p = r.bytes(3)?;
                    let t = r.bytes(2)?;
                    out.push(Tai {
                        plmn: Plmn::decode([p[0], p[1], p[2]]),
                        tac: u16::from_be_bytes([t[0], t[1]]),
                    });
                }
            }
            _ => return Err(NasDecodeError::Invalid("type of TAI list")),
        }
    }
    Ok(out)
}

/// The PDN type (§9.9.4.10).
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum PdnType {
    /// IPv4.
    #[default]
    Ipv4,
    /// IPv6.
    Ipv6,
    /// IPv4v6.
    Ipv4v6,
}

impl PdnType {
    /// The 3-bit value.
    pub fn value(self) -> u8 {
        match self {
            Self::Ipv4 => 1,
            Self::Ipv6 => 2,
            Self::Ipv4v6 => 3,
        }
    }
}

/// A PDN address (§9.9.4.9).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdnAddress {
    /// IPv4 address.
    Ipv4(std::net::Ipv4Addr),
    /// IPv6 interface identifier.
    Ipv6([u8; 8]),
    /// Both.
    Ipv4v6([u8; 8], std::net::Ipv4Addr),
}

/// Decode a PDN address value.
pub fn decode_pdn_address(v: &[u8]) -> Result<PdnAddress, NasDecodeError> {
    let (head, rest) = v
        .split_first()
        .ok_or(NasDecodeError::Invalid("PDN address"))?;
    let v4 = |b: &[u8]| std::net::Ipv4Addr::new(b[0], b[1], b[2], b[3]);
    match (head & 0x07, rest.len()) {
        (1, 4) => Ok(PdnAddress::Ipv4(v4(rest))),
        (2, 8) => Ok(PdnAddress::Ipv6(rest.try_into().expect("8 octets"))),
        (3, 12) => Ok(PdnAddress::Ipv4v6(
            rest[..8].try_into().expect("8 octets"),
            v4(&rest[8..]),
        )),
        _ => Err(NasDecodeError::Invalid("PDN address")),
    }
}

/// An APN in the label encoding of TS 23.003 §9.1 (§9.9.4.1 → TS 24.008
/// §10.5.6.1).
pub fn encode_apn(apn: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(apn.len() + 1);
    for label in apn.split('.') {
        out.push(u8::try_from(label.len()).expect("APN label of more than 63 octets"));
        out.extend_from_slice(label.as_bytes());
    }
    out
}

/// Decode a label-encoded APN.
pub fn decode_apn(v: &[u8]) -> String {
    let mut labels = Vec::new();
    let mut r = Reader::new(v);
    while let Ok(n) = r.u8() {
        match r.bytes(usize::from(n)) {
            Ok(l) => labels.push(String::from_utf8_lossy(l).into_owned()),
            Err(_) => break,
        }
    }
    labels.join(".")
}

/// The highest bit rate the EPS QoS IE codes (TS 24.301 Table 9.9.4.3.1,
/// extended-2 octet `11110110`): 10 Gbit/s, in kbit/s.
pub const MAX_EPS_QOS_KBPS: u32 = 10_000_000;

/// A bit rate in kbit/s as the three octets of TS 24.301 Table 9.9.4.3.1:
/// the base octet (TS 24.008 Table 10.5.156 coding), the extended octet and
/// the extended-2 octet. A rate between two coded values is rounded up to the
/// next one; 0 kbit/s is base `11111111`. Above 8640 kbit/s the base octet is
/// `11111110` and the extended octet carries the rate; above 256 Mbit/s the
/// extended octet is `11111010` and the extended-2 octet carries it. `None`
/// above [`MAX_EPS_QOS_KBPS`].
pub fn encode_eps_bit_rate(kbps: u32) -> Option<[u8; 3]> {
    let b = |v: u32| u8::try_from(v).expect("coded value fits an octet");
    Some(match kbps {
        0 => [0xFF, 0, 0],
        1..=63 => [b(kbps), 0, 0],
        64..=568 => [b(0x40 + (kbps - 64).div_ceil(8)), 0, 0],
        569..=8640 => [b(0x80 + (kbps.max(576) - 576).div_ceil(64)), 0, 0],
        8641..=16_000 => [0xFE, b((kbps - 8600).div_ceil(100)), 0],
        16_001..=128_000 => [0xFE, b(0x4A + (kbps - 16_000).div_ceil(1000)), 0],
        128_001..=256_000 => [0xFE, b(0xBA + (kbps - 128_000).div_ceil(2000)), 0],
        256_001..=500_000 => [0xFE, 0xFA, b((kbps - 256_000).div_ceil(4000))],
        500_001..=1_500_000 => [0xFE, 0xFA, b(0x3D + (kbps - 500_000).div_ceil(10_000))],
        1_500_001..=MAX_EPS_QOS_KBPS => {
            [0xFE, 0xFA, b(0xA1 + (kbps - 1_500_000).div_ceil(100_000))]
        }
        _ => return None,
    })
}

/// The bit rate, in kbit/s, the three octets of [`encode_eps_bit_rate`] stand
/// for (TS 24.301 Table 9.9.4.3.1): what the network reads.
pub fn decode_eps_bit_rate([base, ext, ext2]: [u8; 3]) -> u32 {
    let (base, ext, ext2) = (u32::from(base), u32::from(ext), u32::from(ext2));
    match (ext2, ext) {
        (1..=0x3D, _) => 256_000 + ext2 * 4000,
        (0x3E..=0xA1, _) => 500_000 + (ext2 - 0x3D) * 10_000,
        (0xA2.., _) => 1_500_000 + (ext2.min(0xF6) - 0xA1) * 100_000,
        (0, 1..=0x4A) => 8600 + ext * 100,
        (0, 0x4B..=0xBA) => 16_000 + (ext - 0x4A) * 1000,
        (0, 0xBB..) => 128_000 + (ext.min(0xFA) - 0xBA) * 2000,
        (0, 0) => match base {
            0 | 0xFF => 0,
            1..=0x3F => base,
            0x40..=0x7F => 64 + (base - 0x40) * 8,
            _ => 576 + (base - 0x80) * 64,
        },
    }
}

/// The contents of an EPS quality of service IE (TS 24.301 §9.9.4.3, from
/// octet 3) for a GBR request: the QCI, then MBR UL, MBR DL, GBR UL, GBR DL.
/// The extended octets (8-11) follow when a rate is above 8640 kbit/s, the
/// extended-2 octets (12-15) when one is above 256 Mbit/s: 5, 9 or 13 octets.
/// `None` when a rate is above [`MAX_EPS_QOS_KBPS`].
pub fn encode_eps_qos(qci: u8, mbr: (u32, u32), gbr: (u32, u32)) -> Option<Vec<u8>> {
    let rates = [mbr.0, mbr.1, gbr.0, gbr.1]
        .into_iter()
        .map(encode_eps_bit_rate)
        .collect::<Option<Vec<_>>>()?;
    let mut out = vec![qci];
    out.extend(rates.iter().map(|r| r[0]));
    let ext = rates.iter().any(|r| r[1] != 0);
    let ext2 = rates.iter().any(|r| r[2] != 0);
    if ext || ext2 {
        out.extend(rates.iter().map(|r| r[1]));
    }
    if ext2 {
        out.extend(rates.iter().map(|r| r[2]));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imsi_identity_matches_24008_packing() {
        // 15 digits (odd): digit 1 with odd flag and type 001, then pairs with
        // the later digit in the high nibble; no filler.
        let v = EpsMobileIdentity::Imsi("999700000000001".into()).encode();
        assert_eq!(v, [0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10]);
        assert_eq!(
            EpsMobileIdentity::decode(&v).unwrap(),
            EpsMobileIdentity::Imsi("999700000000001".into())
        );
    }

    #[test]
    fn imeisv_has_even_digits_and_a_filler() {
        // 16 digits: even flag, type 011, last octet F|d16.
        let v = MobileIdentity::Imeisv("3534900698733019".into()).encode();
        assert_eq!(v, [0x33, 0x35, 0x94, 0x00, 0x96, 0x78, 0x33, 0x10, 0xF9]);
    }

    #[test]
    fn guti_identity_is_eleven_octets() {
        let g = Guti {
            plmn: Plmn::new(999, 70, false),
            mme_group_id: 2,
            mme_code: 1,
            m_tmsi: 0xC000_0001,
        };
        let v = EpsMobileIdentity::Guti(g).encode();
        assert_eq!(
            v,
            [0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x01]
        );
        assert_eq!(
            EpsMobileIdentity::decode(&v).unwrap(),
            EpsMobileIdentity::Guti(g)
        );
    }

    #[test]
    fn ue_network_capability_bits_put_algorithm_0_in_bit_8() {
        let c = UeNetworkCapability::from_algorithms(&[0, 1, 2], &[0, 1, 2]);
        assert_eq!(c.encode(), [0xE0, 0xE0]);
        assert!(c.supports_eia(2) && !c.supports_eia(3));
    }

    #[test]
    fn tai_lists_of_all_three_types_decode() {
        // Type 00, two TACs of 999-70.
        let t0 = decode_tai_list(&[0x01, 0x99, 0xF9, 0x07, 0x00, 0x01, 0x00, 0x05]).unwrap();
        assert_eq!(t0.iter().map(|t| t.tac).collect::<Vec<_>>(), [1, 5]);
        // Type 01, three consecutive TACs from 7.
        let t1 = decode_tai_list(&[0x22, 0x99, 0xF9, 0x07, 0x00, 0x07]).unwrap();
        assert_eq!(t1.iter().map(|t| t.tac).collect::<Vec<_>>(), [7, 8, 9]);
        // Type 10, one TAI.
        let t2 = decode_tai_list(&[0x40, 0x99, 0xF9, 0x07, 0x00, 0x03]).unwrap();
        assert_eq!(t2[0].plmn, Plmn::new(999, 70, false));
    }

    #[test]
    fn gprs_timer_units() {
        assert_eq!(gprs_timer_seconds(0x09), Some(18));
        assert_eq!(gprs_timer_seconds(0x29), Some(9 * 60));
        assert_eq!(gprs_timer_seconds(0x49), Some(9 * 360));
        assert_eq!(gprs_timer_seconds(0xE0), None);
    }

    #[test]
    fn apn_labels_round_trip() {
        let v = encode_apn("internet.mnc070.mcc999.gprs");
        assert_eq!(&v[..9], b"\x08internet");
        assert_eq!(decode_apn(&v), "internet.mnc070.mcc999.gprs");
    }

    #[test]
    fn optional_ies_honour_type_1_tv_tlv_and_tlv_e() {
        // C- (type 1), 53 TV2, 50 TLV, 78 TLV-E.
        let b = [
            0xC1, 0x53, 0x11, 0x50, 0x01, 0xAA, 0x78, 0x00, 0x02, 0x01, 0x02,
        ];
        let ies = Reader::new(&b).optional_ies(&[(0x53, 2)]).unwrap();
        assert_eq!(
            ies,
            vec![
                (0xC0, &b[0..1]),
                (0x53, &b[2..3]),
                (0x50, &b[5..6]),
                (0x78, &b[9..11]),
            ]
        );
    }

    /// #356, §9.9.2.1: EBI(n) is bit n of the two octets, EBI(7)..EBI(0) in
    /// octet 3 and EBI(15)..EBI(8) in octet 4.
    #[test]
    fn eps_bearer_context_status_matches_its_9_9_2_1_layout() {
        assert_eq!(
            EpsBearerContextStatus::from_ebis([5]).encode(),
            [0x20, 0x00]
        );
        assert_eq!(
            EpsBearerContextStatus::from_ebis([9]).encode(),
            [0x00, 0x02]
        );
        assert_eq!(
            EpsBearerContextStatus::from_ebis([5, 6, 15]).encode(),
            [0x60, 0x80]
        );
        assert_eq!(
            EpsBearerContextStatus::from_ebis([0, 16]).encode(),
            [0, 0],
            "EBI(0) is spare; 16 is no EBI"
        );
        let s = EpsBearerContextStatus::decode(&[0x21, 0x02]).unwrap();
        assert!(s.is_active(5) && s.is_active(9));
        assert!(
            !s.is_active(6) && !s.is_active(0),
            "the spare bit is not read"
        );
        assert!(EpsBearerContextStatus::decode(&[0x20]).is_err());
    }

    /// TS 24.008 §10.5.7.4a: GPRS timer 3 units, 320 hours for 110 in a
    /// protected message (NOTE 1).
    #[test]
    fn gprs_timer_3_units() {
        assert_eq!(gprs_timer3_seconds(0b000_00011), Some(1800));
        assert_eq!(gprs_timer3_seconds(0b001_00010), Some(7200));
        assert_eq!(gprs_timer3_seconds(0b010_00001), Some(36_000));
        assert_eq!(gprs_timer3_seconds(0b011_00101), Some(10));
        assert_eq!(gprs_timer3_seconds(0b100_00010), Some(60));
        assert_eq!(gprs_timer3_seconds(0b101_00011), Some(180));
        assert_eq!(gprs_timer3_seconds(0b110_00001), Some(320 * 3600));
        assert_eq!(gprs_timer3_seconds(0b111_00001), None);
    }

    /// #434, TS 24.301 Table 9.9.4.3.1: each range's first and last coded
    /// value, a rate between two coded values rounded up, and 0 kbit/s as
    /// `11111111`.
    #[test]
    fn eps_bit_rates_code_per_table_9_9_4_3_1() {
        for (kbps, octets, reads) in [
            (0, [0xFF, 0, 0], 0),
            (1, [0x01, 0, 0], 1),
            (63, [0x3F, 0, 0], 63),
            (64, [0x40, 0, 0], 64),
            (65, [0x41, 0, 0], 72),
            (568, [0x7F, 0, 0], 568),
            (569, [0x80, 0, 0], 576),
            (1000, [0x87, 0, 0], 1024),
            (8640, [0xFE, 0, 0], 8640),
            (8641, [0xFE, 0x01, 0], 8700),
            (16_000, [0xFE, 0x4A, 0], 16_000),
            (16_001, [0xFE, 0x4B, 0], 17_000),
            (128_000, [0xFE, 0xBA, 0], 128_000),
            (128_001, [0xFE, 0xBB, 0], 130_000),
            (256_000, [0xFE, 0xFA, 0], 256_000),
            (256_001, [0xFE, 0xFA, 0x01], 260_000),
            (500_000, [0xFE, 0xFA, 0x3D], 500_000),
            (500_001, [0xFE, 0xFA, 0x3E], 510_000),
            (1_500_000, [0xFE, 0xFA, 0xA1], 1_500_000),
            (1_500_001, [0xFE, 0xFA, 0xA2], 1_600_000),
            (10_000_000, [0xFE, 0xFA, 0xF6], 10_000_000),
        ] {
            assert_eq!(encode_eps_bit_rate(kbps), Some(octets), "{kbps} kbit/s");
            assert_eq!(decode_eps_bit_rate(octets), reads, "{kbps} kbit/s");
        }
        assert_eq!(encode_eps_bit_rate(10_000_001), None);
        // Never below the asked rate, and never more than one step above it
        for kbps in (0..10_000_000).step_by(997) {
            let reads = decode_eps_bit_rate(encode_eps_bit_rate(kbps).unwrap());
            assert!(reads >= kbps, "{kbps}");
            assert_eq!(
                encode_eps_bit_rate(reads).map(decode_eps_bit_rate),
                Some(reads)
            );
        }
    }

    /// #434, §9.9.4.3: 5 octets of contents up to 8640 kbit/s, 9 with the
    /// extended octets, 13 with the extended-2 ones; a rate the IE cannot code
    /// is refused.
    #[test]
    fn eps_qos_contents_grow_with_the_rates() {
        assert_eq!(
            encode_eps_qos(1, (64, 64), (64, 64)),
            Some(vec![1, 0x40, 0x40, 0x40, 0x40])
        );
        assert_eq!(
            encode_eps_qos(1, (20_000, 64), (64, 64)),
            Some(vec![1, 0xFE, 0x40, 0x40, 0x40, 0x4E, 0, 0, 0])
        );
        assert_eq!(
            encode_eps_qos(1, (300_000, 300_000), (300_000, 300_000)),
            Some(vec![
                1, 0xFE, 0xFE, 0xFE, 0xFE, 0xFA, 0xFA, 0xFA, 0xFA, 0x0B, 0x0B, 0x0B, 0x0B
            ])
        );
        assert_eq!(encode_eps_qos(1, (64, 64), (64, 10_000_001)), None);
    }
}
