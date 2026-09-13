//! UE policy delivery service (UPDP) codec — TS 24.501 Annex D + TS 24.526 §5.2
//!
//! Wave-6 item E8 (nextgsim-ue side). This is the UE-side half of the PCF↔UE
//! "UE policy delivery service" protocol carried inside the DL/UL NAS TRANSPORT
//! Payload container of type "UE policy container" (0b0101, TS 24.501
//! §9.11.3.39, [`crate::ies::ie1::PayloadContainerType::UePolicyContainer`]):
//!
//! - MANAGE UE POLICY COMMAND (decode) — TS 24.501 Table D.5.1.1.1
//! - MANAGE UE POLICY COMPLETE (encode) — Table D.5.2.1.1 (PTI + type 0x02)
//! - MANAGE UE POLICY COMMAND REJECT (encode) with the UE policy section
//!   management result of D.6.3 (PTI + type 0x03 + LV-E result)
//! - UE policy section management list decode per figures D.6.2.1-D.6.2.7
//! - URSP part-contents decode per TS 24.526 V18.5.0 §5.2 (traffic descriptor /
//!   route selection descriptor component walk)
//!
//! # Independence
//!
//! This is a deliberately INDEPENDENT re-implementation of the nextgcore-nas
//! encoder/decoder (`nextgcore/src/libs/nextgcore-nas/src/fiveg/ue_policy.rs`).
//! No code is shared across the two repositories — the two codecs agreeing
//! byte-for-byte on the hand-derived E1 golden vectors is the cross-stack
//! strict-peer value. The E1 vectors are re-cited verbatim in the crate tests.
//!
//! # Length-field convention (the fiddly part)
//!
//! Every 2-octet length field counts the octets FOLLOWING the length field up
//! to the end of its own structure (the length field itself is NOT included,
//! everything after it IS — including any type octet). Spec evidence:
//! TS 24.501 Table D.6.2.1 NOTE 2 ("The UE policy part contents length
//! indicates the length of the value part of the UE policy part field, i.e.
//! octet q+2 to octet r") — octet q+2 is the part-type octet immediately after
//! the 2-octet length.
//!
//! # Fail-closed policy
//!
//! Decoders reject truncation, length-scope over/under-runs, set spare bits,
//! reserved part types, PTI outside the PCF-initiated range and unknown
//! component identifiers. The UE never COMPLETEs a policy section it could not
//! store: a decode/instruction failure produces a MANAGE UE POLICY COMMAND
//! REJECT carrying a per-instruction D.6.3 result (see the nextgsim-ue handler).

use crate::ies::ie1::{PduSessionType, SscMode};
use thiserror::Error;

// ============================================================================
// Constants (TS 24.501 Annex D)
// ============================================================================

/// MANAGE UE POLICY COMMAND (TS 24.501 Table D.6.1.1).
pub const UPDP_MSG_MANAGE_UE_POLICY_COMMAND: u8 = 0x01;
/// MANAGE UE POLICY COMPLETE (Table D.6.1.1).
pub const UPDP_MSG_MANAGE_UE_POLICY_COMPLETE: u8 = 0x02;
/// MANAGE UE POLICY COMMAND REJECT (Table D.6.1.1).
pub const UPDP_MSG_MANAGE_UE_POLICY_COMMAND_REJECT: u8 = 0x03;
/// UE STATE INDICATION (Table D.6.1.1: bit pattern 0000 0100).
pub const UPDP_MSG_UE_STATE_INDICATION: u8 = 0x04;

/// PCF-initiated procedures use PTI 80H-FEH (TS 24.501 D.1.2).
pub const PCF_PTI_MIN: u8 = 0x80;
/// Upper bound of the PCF PTI range (FFH is reserved).
pub const PCF_PTI_MAX: u8 = 0xFE;
/// UE-initiated procedures use PTI 01H-77H (TS 24.501 D.1.2).
///
/// NOT zero: D.1.2 says "When the UE initiates a procedure, the UE shall use a
/// PTI value in range between 01H and 77H", and D.2.2.2 a) says the UE "shall
/// allocate a PTI value currently not used". PTI 0 means "no procedure
/// transaction identity assigned" and would make the PCF unable to correlate a
/// response.
pub const UE_PTI_MIN: u8 = 0x01;
/// Upper bound of the UE PTI range (TS 24.501 D.1.2).
pub const UE_PTI_MAX: u8 = 0x77;

/// D.6.3 result cause "Protocol error, unspecified" (0110 1111). The receiving
/// entity treats any other value as this one (Table D.6.3.1).
pub const CAUSE_PROTOCOL_ERROR_UNSPECIFIED: u8 = 0x6F;

// Traffic descriptor component type identifiers (TS 24.526 Table 5.2.1).
const TD_ID_MATCH_ALL: u8 = 0b0000_0001;
const TD_ID_OS_ID_OS_APP_ID: u8 = 0b0000_1000;
const TD_ID_IPV4_REMOTE: u8 = 0b0001_0000;
const TD_ID_IPV6_REMOTE: u8 = 0b0010_0001;
const TD_ID_PROTOCOL: u8 = 0b0011_0000;
const TD_ID_SINGLE_REMOTE_PORT: u8 = 0b0101_0000;
const TD_ID_REMOTE_PORT_RANGE: u8 = 0b0101_0001;
const TD_ID_DNN: u8 = 0b1000_1000;
const TD_ID_DEST_FQDN: u8 = 0b1001_0001;

// Route selection descriptor component type identifiers (TS 24.526 Table 5.2.1).
const RSD_ID_SSC_MODE: u8 = 0b0000_0001;
const RSD_ID_SNSSAI: u8 = 0b0000_0010;
const RSD_ID_DNN: u8 = 0b0000_0100;
const RSD_ID_PDU_SESSION_TYPE: u8 = 0b0000_1000;
const RSD_ID_PREFERRED_ACCESS: u8 = 0b0001_0000;

/// DNS label maximum length (TS 23.003 §9.1 / §28.3.2.1).
const MAX_LABEL_LEN: usize = 63;

// ============================================================================
// Error type
// ============================================================================

/// UE policy delivery service codec error (fail-closed decode).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum UePolicyError {
    /// The buffer ended before a field could be read.
    #[error("UPDP buffer too short: needed {needed} more octet(s), {available} available")]
    Truncated {
        /// Octets that were required.
        needed: usize,
        /// Octets that were available.
        available: usize,
    },
    /// A length-scoped structure was not consumed exactly.
    #[error("UPDP length-scope error in {scope}: {msg}")]
    LengthScope {
        /// The structure whose length field was violated.
        scope: &'static str,
        /// What went wrong.
        msg: String,
    },
    /// A field held a value the spec forbids.
    #[error("UPDP invalid value in {scope}: {msg}")]
    InvalidValue {
        /// The structure being decoded.
        scope: &'static str,
        /// What was wrong.
        msg: String,
    },
    /// The message type octet was not the expected one.
    #[error("UPDP unexpected message type {got:#04x} (expected {expected:#04x})")]
    UnexpectedMessageType {
        /// The message type read from the wire.
        got: u8,
        /// The message type expected.
        expected: u8,
    },
    /// The PTI was outside the PCF-initiated 80H-FEH range (D.1.2).
    #[error("UPDP PTI {0:#04x} outside the PCF-initiated range 80H-FEH (TS 24.501 D.1.2)")]
    PtiOutOfRange(u8),
    /// The PTI was outside the UE-initiated 01H-77H range (D.1.2).
    #[error("UPDP PTI {0:#04x} outside the UE-initiated range 01H-77H (TS 24.501 D.1.2)")]
    UePtiOutOfRange(u8),
}

type Result<T> = core::result::Result<T, UePolicyError>;

// ============================================================================
// Strict scoped reader
// ============================================================================

/// Bounds-checked forward cursor. Each length-scoped structure is decoded from
/// its own sub-reader and consumed EXACTLY via [`Reader::finish`], catching
/// both over-runs and under-runs of the D.6.2/§5.2 nested length fields.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn u8(&mut self) -> Result<u8> {
        if self.remaining() < 1 {
            return Err(UePolicyError::Truncated {
                needed: 1,
                available: self.remaining(),
            });
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn u16(&mut self) -> Result<u16> {
        let hi = self.u8()?;
        let lo = self.u8()?;
        Ok(u16::from_be_bytes([hi, lo]))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(UePolicyError::Truncated {
                needed: n,
                available: self.remaining(),
            });
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// A sub-reader over the next `n` octets, advancing this reader past them.
    /// A declared length that overruns the parent scope is a fail-closed error.
    fn sub(&mut self, n: usize, scope: &'static str) -> Result<Reader<'a>> {
        if self.remaining() < n {
            return Err(UePolicyError::LengthScope {
                scope,
                msg: format!(
                    "declared length {n} exceeds the {} remaining octet(s)",
                    self.remaining()
                ),
            });
        }
        let s = self.take(n)?;
        Ok(Reader::new(s))
    }

    fn finish(self, scope: &'static str) -> Result<()> {
        if self.remaining() != 0 {
            return Err(UePolicyError::LengthScope {
                scope,
                msg: format!(
                    "{} unconsumed trailing octet(s) inside length scope",
                    self.remaining()
                ),
            });
        }
        Ok(())
    }
}

// ============================================================================
// PLMN BCD coding (TS 24.501 figure D.6.2.3 + Table D.6.2.1 NOTE 1)
// ============================================================================

/// A PLMN identity decoded from its 3-octet BCD form. Used as (part of) the
/// UE policy section store key, so it is `Hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlmnId {
    /// MCC digits 1..3 (each 0..=9).
    pub mcc: [u8; 3],
    /// MNC digits 1..3 (digit 3 unused when `mnc_len == 2`).
    pub mnc: [u8; 3],
    /// Number of MNC digits (2 or 3).
    pub mnc_len: u8,
}

impl PlmnId {
    /// Decodes the 3-octet BCD PLMN of figure D.6.2.3 (octet d+2 = MCC2<<4 |
    /// MCC1; octet d+3 = MNC3<<4 | MCC3, with MNC3 = 0b1111 for a 2-digit MNC
    /// per Table D.6.2.1 NOTE 1; octet d+4 = MNC2<<4 | MNC1). Non-BCD digits
    /// are rejected.
    fn decode(b: &[u8]) -> Result<Self> {
        debug_assert_eq!(b.len(), 3);
        let mcc = [b[0] & 0x0F, (b[0] >> 4) & 0x0F, b[1] & 0x0F];
        let mnc3 = (b[1] >> 4) & 0x0F;
        let mnc12 = [b[2] & 0x0F, (b[2] >> 4) & 0x0F];
        if mcc.iter().any(|d| *d > 9) || mnc12.iter().any(|d| *d > 9) {
            return Err(UePolicyError::InvalidValue {
                scope: "PLMN",
                msg: "MCC/MNC contains a non-BCD digit".into(),
            });
        }
        let (mnc, mnc_len) = if mnc3 == 0x0F {
            ([mnc12[0], mnc12[1], 0], 2)
        } else if mnc3 <= 9 {
            ([mnc12[0], mnc12[1], mnc3], 3)
        } else {
            return Err(UePolicyError::InvalidValue {
                scope: "PLMN",
                msg: "MNC digit 3 is neither BCD nor 0b1111".into(),
            });
        };
        Ok(Self { mcc, mnc, mnc_len })
    }

    /// Decodes a PLMN from its 3-octet BCD form (public wrapper over the
    /// internal decoder; used to seed a fallback subresult PLMN).
    pub fn from_bcd(b: [u8; 3]) -> Result<Self> {
        Self::decode(&b)
    }

    /// Encodes back to the 3-octet BCD form (used to echo the PLMN in a
    /// MANAGE UE POLICY COMMAND REJECT result).
    fn encode(&self) -> [u8; 3] {
        let mnc3 = if self.mnc_len == 2 { 0x0F } else { self.mnc[2] };
        [
            (self.mcc[1] << 4) | self.mcc[0],
            (mnc3 << 4) | self.mcc[2],
            (self.mnc[1] << 4) | self.mnc[0],
        ]
    }
}

/// Decodes a length-prefixed dotted name (TS 23.003 §9.1: each label = 1-octet
/// length + that many ASCII octets; not zero-terminated).
fn decode_labels(bytes: &[u8], scope: &'static str) -> Result<String> {
    let mut r = Reader::new(bytes);
    let mut labels: Vec<String> = Vec::new();
    while r.remaining() > 0 {
        let len = usize::from(r.u8()?);
        if len == 0 || len > MAX_LABEL_LEN {
            return Err(UePolicyError::InvalidValue {
                scope,
                msg: format!("invalid label length {len}"),
            });
        }
        let label = r.take(len)?;
        if !label.is_ascii() {
            return Err(UePolicyError::InvalidValue {
                scope,
                msg: "non-ASCII label octets".into(),
            });
        }
        labels.push(String::from_utf8_lossy(label).into_owned());
    }
    if labels.is_empty() {
        return Err(UePolicyError::InvalidValue {
            scope,
            msg: "at least one label required".into(),
        });
    }
    Ok(labels.join("."))
}

// ============================================================================
// TS 24.526 §5.2 — URSP part contents (decode side, minimal)
// ============================================================================

/// Preferred access type value (TS 24.501 §9.11.2.1A: 0b01 = 3GPP,
/// 0b10 = non-3GPP).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreferredAccessType {
    /// 3GPP access (0b01).
    ThreeGpp,
    /// Non-3GPP access (0b10).
    NonThreeGpp,
}

/// Traffic descriptor component (TS 24.526 Table 5.2.1). Decode-only walk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrafficDescriptorComponent {
    /// 0b0000_0001 Match-all type (no value field).
    MatchAll,
    /// 0b0000_1000 OS Id + OS App Id type: 16-octet OS Id + 1-octet length +
    /// OS App Id.
    OsIdOsAppId {
        /// 16-octet OS Id (RFC 4122 UUID).
        os_id: [u8; 16],
        /// Application identity octets.
        os_app_id: Vec<u8>,
    },
    /// 0b0001_0000 IPv4 remote address + mask.
    Ipv4RemoteAddress {
        /// 4-octet address.
        addr: [u8; 4],
        /// 4-octet mask.
        mask: [u8; 4],
    },
    /// 0b0010_0001 IPv6 remote address + prefix length.
    Ipv6RemoteAddress {
        /// 16-octet address.
        addr: [u8; 16],
        /// Prefix length (0..=128).
        prefix_len: u8,
    },
    /// 0b0011_0000 Protocol identifier / next header (one octet).
    ProtocolIdentifier(u8),
    /// 0b0101_0000 Single remote port (two octets).
    SingleRemotePort(u16),
    /// 0b0101_0001 Remote port range (low, high).
    RemotePortRange {
        /// Low port limit (inclusive).
        low: u16,
        /// High port limit (inclusive).
        high: u16,
    },
    /// 0b1000_1000 DNN type (APN value).
    Dnn(String),
    /// 0b1001_0001 Destination FQDN type.
    DestinationFqdn(String),
}

impl TrafficDescriptorComponent {
    fn decode(r: &mut Reader) -> Result<Self> {
        let id = r.u8()?;
        Ok(match id {
            TD_ID_MATCH_ALL => Self::MatchAll,
            TD_ID_OS_ID_OS_APP_ID => {
                let mut os_id = [0u8; 16];
                os_id.copy_from_slice(r.take(16)?);
                let len = usize::from(r.u8()?);
                Self::OsIdOsAppId {
                    os_id,
                    os_app_id: r.take(len)?.to_vec(),
                }
            }
            TD_ID_IPV4_REMOTE => {
                let mut addr = [0u8; 4];
                addr.copy_from_slice(r.take(4)?);
                let mut mask = [0u8; 4];
                mask.copy_from_slice(r.take(4)?);
                Self::Ipv4RemoteAddress { addr, mask }
            }
            TD_ID_IPV6_REMOTE => {
                let mut addr = [0u8; 16];
                addr.copy_from_slice(r.take(16)?);
                let prefix_len = r.u8()?;
                if prefix_len > 128 {
                    return Err(UePolicyError::InvalidValue {
                        scope: "traffic descriptor",
                        msg: format!("IPv6 prefix length {prefix_len} exceeds 128"),
                    });
                }
                Self::Ipv6RemoteAddress { addr, prefix_len }
            }
            TD_ID_PROTOCOL => Self::ProtocolIdentifier(r.u8()?),
            TD_ID_SINGLE_REMOTE_PORT => Self::SingleRemotePort(r.u16()?),
            TD_ID_REMOTE_PORT_RANGE => {
                let low = r.u16()?;
                let high = r.u16()?;
                if low > high {
                    return Err(UePolicyError::InvalidValue {
                        scope: "traffic descriptor",
                        msg: format!("remote port range low {low} exceeds high {high}"),
                    });
                }
                Self::RemotePortRange { low, high }
            }
            TD_ID_DNN => {
                let len = usize::from(r.u8()?);
                Self::Dnn(decode_labels(r.take(len)?, "TD DNN")?)
            }
            TD_ID_DEST_FQDN => {
                let len = usize::from(r.u8()?);
                Self::DestinationFqdn(decode_labels(r.take(len)?, "TD destination FQDN")?)
            }
            other => {
                return Err(UePolicyError::InvalidValue {
                    scope: "traffic descriptor",
                    msg: format!(
                        "unknown component type identifier {other:#04x} (TS 24.526 Table 5.2.1)"
                    ),
                });
            }
        })
    }
}

/// Route selection descriptor component (TS 24.526 Table 5.2.1). Decode-only.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteSelectionDescriptorComponent {
    /// 0b0000_0001 SSC mode type (1 octet, bits 3..1).
    SscMode(SscMode),
    /// 0b0000_0010 S-NSSAI type (1-octet length + value, no mapped-HPLMN).
    SNssai {
        /// Slice/service type.
        sst: u8,
        /// Optional 3-octet slice differentiator.
        sd: Option<[u8; 3]>,
    },
    /// 0b0000_0100 DNN type (APN value).
    Dnn(String),
    /// 0b0000_1000 PDU session type (1 octet, bits 3..1).
    PduSessionType(PduSessionType),
    /// 0b0001_0000 Preferred access type (1 octet, bits 2..1).
    PreferredAccessType(PreferredAccessType),
}

impl RouteSelectionDescriptorComponent {
    fn decode(r: &mut Reader) -> Result<Self> {
        let id = r.u8()?;
        Ok(match id {
            RSD_ID_SSC_MODE => {
                let b = r.u8()?;
                if b & 0xF8 != 0 {
                    return Err(UePolicyError::InvalidValue {
                        scope: "RSD SSC mode",
                        msg: "spare bits 8..4 set".into(),
                    });
                }
                let mode =
                    SscMode::try_from(b & 0x07).map_err(|_| UePolicyError::InvalidValue {
                        scope: "RSD SSC mode",
                        msg: format!("invalid SSC mode value {}", b & 0x07),
                    })?;
                Self::SscMode(mode)
            }
            RSD_ID_SNSSAI => {
                let len = usize::from(r.u8()?);
                match len {
                    1 => Self::SNssai {
                        sst: r.u8()?,
                        sd: None,
                    },
                    4 => {
                        let sst = r.u8()?;
                        let mut sd = [0u8; 3];
                        sd.copy_from_slice(r.take(3)?);
                        Self::SNssai { sst, sd: Some(sd) }
                    }
                    other => {
                        return Err(UePolicyError::InvalidValue {
                            scope: "RSD S-NSSAI",
                            msg: format!("invalid length {other} (must be 1 or 4)"),
                        });
                    }
                }
            }
            RSD_ID_DNN => {
                let len = usize::from(r.u8()?);
                Self::Dnn(decode_labels(r.take(len)?, "RSD DNN")?)
            }
            RSD_ID_PDU_SESSION_TYPE => {
                let b = r.u8()?;
                if b & 0xF8 != 0 {
                    return Err(UePolicyError::InvalidValue {
                        scope: "RSD PDU session type",
                        msg: "spare bits 8..4 set".into(),
                    });
                }
                let t = PduSessionType::try_from(b & 0x07).map_err(|_| {
                    UePolicyError::InvalidValue {
                        scope: "RSD PDU session type",
                        msg: format!("invalid value {}", b & 0x07),
                    }
                })?;
                Self::PduSessionType(t)
            }
            RSD_ID_PREFERRED_ACCESS => {
                let b = r.u8()?;
                if b & 0xFC != 0 {
                    return Err(UePolicyError::InvalidValue {
                        scope: "RSD preferred access type",
                        msg: "spare bits 8..3 set".into(),
                    });
                }
                match b & 0x03 {
                    1 => Self::PreferredAccessType(PreferredAccessType::ThreeGpp),
                    2 => Self::PreferredAccessType(PreferredAccessType::NonThreeGpp),
                    v => {
                        return Err(UePolicyError::InvalidValue {
                            scope: "RSD preferred access type",
                            msg: format!("invalid value {v}"),
                        });
                    }
                }
            }
            other => {
                return Err(UePolicyError::InvalidValue {
                    scope: "route selection descriptor",
                    msg: format!(
                        "unknown component type identifier {other:#04x} (TS 24.526 Table 5.2.1)"
                    ),
                });
            }
        })
    }
}

/// Route selection descriptor (TS 24.526 Figure 5.2.4).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteSelectionDescriptor {
    /// RSD precedence (higher value = lower priority).
    pub precedence: u8,
    /// At least one component.
    pub components: Vec<RouteSelectionDescriptorComponent>,
}

impl RouteSelectionDescriptor {
    fn decode(r: &mut Reader) -> Result<Self> {
        let rsd_len = usize::from(r.u16()?);
        let mut rsd = r.sub(rsd_len, "route selection descriptor")?;
        let precedence = rsd.u8()?;
        let contents_len = usize::from(rsd.u16()?);
        let mut contents = rsd.sub(contents_len, "route selection descriptor contents")?;
        rsd.finish("route selection descriptor")?;
        let mut components = Vec::new();
        while contents.remaining() > 0 {
            components.push(RouteSelectionDescriptorComponent::decode(&mut contents)?);
        }
        contents.finish("route selection descriptor contents")?;
        if components.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "route selection descriptor",
                msg: "must contain at least one component".into(),
            });
        }
        Ok(Self {
            precedence,
            components,
        })
    }
}

/// URSP rule (TS 24.526 Figure 5.2.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UrspRule {
    /// Rule precedence (unique within one URSP; higher value = lower priority).
    pub precedence: u8,
    /// Traffic descriptor components (at least one).
    pub traffic_descriptor: Vec<TrafficDescriptorComponent>,
    /// Route selection descriptors (at least one).
    pub route_selection_descriptors: Vec<RouteSelectionDescriptor>,
    /// Optional Figure 5.2.4A URERI additional-indications bit.
    pub ureri: Option<bool>,
}

impl UrspRule {
    fn decode(r: &mut Reader) -> Result<Self> {
        let rule_len = usize::from(r.u16()?);
        let mut rule = r.sub(rule_len, "URSP rule")?;
        let precedence = rule.u8()?;
        let td_len = usize::from(rule.u16()?);
        let mut td = rule.sub(td_len, "traffic descriptor")?;
        let mut traffic_descriptor = Vec::new();
        while td.remaining() > 0 {
            traffic_descriptor.push(TrafficDescriptorComponent::decode(&mut td)?);
        }
        td.finish("traffic descriptor")?;
        if traffic_descriptor.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "URSP rule",
                msg: "traffic descriptor must contain at least one component".into(),
            });
        }
        let rsdl_len = usize::from(rule.u16()?);
        let mut rsdl = rule.sub(rsdl_len, "route selection descriptor list")?;
        let mut route_selection_descriptors = Vec::new();
        while rsdl.remaining() > 0 {
            route_selection_descriptors.push(RouteSelectionDescriptor::decode(&mut rsdl)?);
        }
        rsdl.finish("route selection descriptor list")?;
        if route_selection_descriptors.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "URSP rule",
                msg: "must contain at least one route selection descriptor".into(),
            });
        }
        // Figure 5.2.2: at most one optional trailing "Additional indications"
        // octet (Figure 5.2.4A) may remain in the rule scope.
        let ureri = match rule.remaining() {
            0 => None,
            1 => {
                let b = rule.u8()?;
                if b & 0xFE != 0 {
                    return Err(UePolicyError::InvalidValue {
                        scope: "URSP rule",
                        msg: "additional indications spare bits 8..2 set (Figure 5.2.4A)".into(),
                    });
                }
                Some(b & 0x01 == 0x01)
            }
            n => {
                return Err(UePolicyError::LengthScope {
                    scope: "URSP rule",
                    msg: format!("{n} unexpected octets after the route selection descriptor list"),
                });
            }
        };
        rule.finish("URSP rule")?;
        Ok(Self {
            precedence,
            traffic_descriptor,
            route_selection_descriptors,
            ureri,
        })
    }
}

/// Decodes URSP part contents: one or more concatenated URSP rules
/// (TS 24.526 Figure 5.2.1).
pub fn decode_ursp_rules(bytes: &[u8]) -> Result<Vec<UrspRule>> {
    let mut r = Reader::new(bytes);
    let mut rules = Vec::new();
    while r.remaining() > 0 {
        rules.push(UrspRule::decode(&mut r)?);
    }
    r.finish("URSP part contents")?;
    if rules.is_empty() {
        return Err(UePolicyError::InvalidValue {
            scope: "URSP part contents",
            msg: "must contain at least one URSP rule".into(),
        });
    }
    Ok(rules)
}

// ============================================================================
// TS 24.501 D.6.2 — UE policy section management list (decode)
// ============================================================================

/// UE policy part type (TS 24.501 Table D.6.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UePolicyPartType {
    /// URSP (0001).
    Ursp,
    /// ANDSP (0010).
    Andsp,
    /// V2XP (0011).
    V2xp,
    /// ProSeP (0100).
    ProSeP,
    /// A2XP (0101).
    A2xp,
    /// RSLPP (0110).
    Rslpp,
}

impl UePolicyPartType {
    fn from_bits(v: u8) -> Result<Self> {
        Ok(match v {
            0x01 => Self::Ursp,
            0x02 => Self::Andsp,
            0x03 => Self::V2xp,
            0x04 => Self::ProSeP,
            0x05 => Self::A2xp,
            0x06 => Self::Rslpp,
            other => {
                return Err(UePolicyError::InvalidValue {
                    scope: "UE policy part",
                    msg: format!("reserved UE policy part type {other:#03x} (Table D.6.2.1)"),
                });
            }
        })
    }
}

/// UE policy part (TS 24.501 Figure D.6.2.7): part type + opaque contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UePolicyPart {
    /// The part type (bits 4..1 of octet q+2).
    pub part_type: UePolicyPartType,
    /// The part contents (octets q+3..r); for URSP, the TS 24.526 §5.2 bytes.
    pub contents: Vec<u8>,
}

impl UePolicyPart {
    /// Decodes the URSP rules of a URSP part (fails if the part is not URSP).
    pub fn decode_ursp(&self) -> Result<Vec<UrspRule>> {
        if self.part_type != UePolicyPartType::Ursp {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy part",
                msg: format!("part type {:?} is not URSP", self.part_type),
            });
        }
        decode_ursp_rules(&self.contents)
    }

    fn decode(r: &mut Reader) -> Result<Self> {
        let part_len = usize::from(r.u16()?);
        if part_len == 0 {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy part",
                msg: "contents length must cover at least the part-type octet (NOTE 2)".into(),
            });
        }
        let mut part = r.sub(part_len, "UE policy part")?;
        let type_octet = part.u8()?;
        if type_octet & 0xF0 != 0 {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy part",
                msg: "spare bits 8..5 of the part-type octet set (Table D.6.2.1)".into(),
            });
        }
        let part_type = UePolicyPartType::from_bits(type_octet & 0x0F)?;
        let contents = part.take(part.remaining())?.to_vec();
        part.finish("UE policy part")?;
        Ok(Self {
            part_type,
            contents,
        })
    }
}

/// Instruction (TS 24.501 Figure D.6.2.5): UPSC + zero or more parts. An
/// instruction with no parts orders deletion of the section (D.2.1.3).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// UE policy section code (set by the PCF).
    pub upsc: u16,
    /// UE policy parts (may be empty = delete the section).
    pub parts: Vec<UePolicyPart>,
}

impl Instruction {
    fn decode(r: &mut Reader) -> Result<Self> {
        let instr_len = usize::from(r.u16()?);
        let mut instr = r.sub(instr_len, "instruction")?;
        let upsc = instr.u16()?;
        let mut parts = Vec::new();
        while instr.remaining() > 0 {
            parts.push(UePolicyPart::decode(&mut instr)?);
        }
        instr.finish("instruction")?;
        Ok(Self { upsc, parts })
    }
}

/// UE policy section management sublist (TS 24.501 Figure D.6.2.3): one PLMN +
/// one or more instructions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlmnSublist {
    /// The PLMN this sublist applies to.
    pub plmn_id: PlmnId,
    /// At least one instruction.
    pub instructions: Vec<Instruction>,
}

impl PlmnSublist {
    fn decode(r: &mut Reader) -> Result<Self> {
        let sublist_len = usize::from(r.u16()?);
        let mut sublist = r.sub(sublist_len, "UE policy section management sublist")?;
        let plmn_id = PlmnId::decode(sublist.take(3)?)?;
        let mut instructions = Vec::new();
        while sublist.remaining() > 0 {
            instructions.push(Instruction::decode(&mut sublist)?);
        }
        sublist.finish("UE policy section management sublist")?;
        if instructions.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy section management sublist",
                msg: "must contain at least one instruction (Figure D.6.2.4)".into(),
            });
        }
        Ok(Self {
            plmn_id,
            instructions,
        })
    }
}

// ============================================================================
// TS 24.501 D.5 — messages
// ============================================================================

/// MANAGE UE POLICY COMMAND (TS 24.501 Table D.5.1.1.1): PTI + message type +
/// UE policy section management list (LV-E). Decode only — the UE receives it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManageUePolicyCommand {
    /// PCF-allocated PTI (80H-FEH, D.1.2).
    pub pti: u8,
    /// One or more per-PLMN sublists (Figure D.6.2.2).
    pub sublists: Vec<PlmnSublist>,
}

impl ManageUePolicyCommand {
    /// Strict decode of the full message content carried in the Payload
    /// container ("UE policy container", 0x05). The optional IEs 0x42 / 0x70
    /// are not supported: any trailing octet after the LV-E list is rejected.
    pub fn decode(content: &[u8]) -> Result<Self> {
        let mut r = Reader::new(content);
        let pti = r.u8()?;
        let msg_type = r.u8()?;
        if msg_type != UPDP_MSG_MANAGE_UE_POLICY_COMMAND {
            return Err(UePolicyError::UnexpectedMessageType {
                got: msg_type,
                expected: UPDP_MSG_MANAGE_UE_POLICY_COMMAND,
            });
        }
        if !(PCF_PTI_MIN..=PCF_PTI_MAX).contains(&pti) {
            return Err(UePolicyError::PtiOutOfRange(pti));
        }
        let list_len = usize::from(r.u16()?);
        let mut list = r.sub(list_len, "UE policy section management list")?;
        r.finish("MANAGE UE POLICY COMMAND")?;
        let mut sublists = Vec::new();
        while list.remaining() > 0 {
            sublists.push(PlmnSublist::decode(&mut list)?);
        }
        list.finish("UE policy section management list")?;
        if sublists.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy section management list",
                msg: "must contain at least one sublist (Figure D.6.2.2)".into(),
            });
        }
        Ok(Self { pti, sublists })
    }
}

/// MANAGE UE POLICY COMPLETE (TS 24.501 Table D.5.2.1.1): PTI + message type
/// only. The PTI echoes the command's PTI (D.1.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ManageUePolicyComplete {
    /// Echoed PCF PTI (80H-FEH).
    pub pti: u8,
}

impl ManageUePolicyComplete {
    /// Encodes the 2-octet UE policy delivery service message content (the
    /// bytes that ride in the UL NAS TRANSPORT UE policy container).
    pub fn encode(&self) -> Result<Vec<u8>> {
        if !(PCF_PTI_MIN..=PCF_PTI_MAX).contains(&self.pti) {
            return Err(UePolicyError::PtiOutOfRange(self.pti));
        }
        Ok(vec![self.pti, UPDP_MSG_MANAGE_UE_POLICY_COMPLETE])
    }
}

/// One D.6.3 result (Figure D.6.3.5): UPSC + 1-based failed instruction order +
/// cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UePolicyResult {
    /// UPSC of the failed instruction's section.
    pub upsc: u16,
    /// 1-based order of the failed instruction within the sublist contents.
    pub failed_instruction_order: u16,
    /// Cause octet ([`CAUSE_PROTOCOL_ERROR_UNSPECIFIED`] is the only assigned
    /// value).
    pub cause: u8,
}

/// UE policy section management subresult (Figure D.6.3.3): 1-octet count +
/// PLMN + count × result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UePolicySectionManagementSubresult {
    /// The PLMN the results apply to.
    pub plmn_id: PlmnId,
    /// One or more results.
    pub results: Vec<UePolicyResult>,
}

/// UE policy section management result (TS 24.501 D.6.3): one or more
/// subresults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UePolicySectionManagementResult {
    /// One or more subresults (Figure D.6.3.2).
    pub subresults: Vec<UePolicySectionManagementSubresult>,
}

/// MANAGE UE POLICY COMMAND REJECT (TS 24.501 Table D.5.3.1.1): PTI + message
/// type + UE policy section management result (LV-E, D.6.3). Encode only — the
/// UE sends it when it cannot store one or more requested sections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManageUePolicyCommandReject {
    /// Echoed PCF PTI (80H-FEH).
    pub pti: u8,
    /// The per-instruction failure report.
    pub result: UePolicySectionManagementResult,
}

impl ManageUePolicyCommandReject {
    /// Encodes the full UE policy delivery service message content.
    pub fn encode(&self) -> Result<Vec<u8>> {
        if !(PCF_PTI_MIN..=PCF_PTI_MAX).contains(&self.pti) {
            return Err(UePolicyError::PtiOutOfRange(self.pti));
        }
        if self.result.subresults.is_empty() {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy section management result",
                msg: "must contain at least one subresult (Figure D.6.3.2)".into(),
            });
        }
        let mut contents: Vec<u8> = Vec::new();
        for sub in &self.result.subresults {
            if sub.results.is_empty() {
                return Err(UePolicyError::InvalidValue {
                    scope: "UE policy section management subresult",
                    msg: "must contain at least one result (Table D.6.3.1)".into(),
                });
            }
            let count =
                u8::try_from(sub.results.len()).map_err(|_| UePolicyError::InvalidValue {
                    scope: "UE policy section management subresult",
                    msg: "holds more than 255 results (1-octet count)".into(),
                })?;
            contents.push(count);
            contents.extend_from_slice(&sub.plmn_id.encode());
            for res in &sub.results {
                contents.extend_from_slice(&res.upsc.to_be_bytes());
                contents.extend_from_slice(&res.failed_instruction_order.to_be_bytes());
                contents.push(res.cause);
            }
        }
        let len = u16::try_from(contents.len()).map_err(|_| UePolicyError::LengthScope {
            scope: "UE policy section management result",
            msg: format!("contents {} exceed the 2-octet LV-E length", contents.len()),
        })?;
        let mut out = Vec::with_capacity(4 + contents.len());
        out.push(self.pti);
        out.push(UPDP_MSG_MANAGE_UE_POLICY_COMMAND_REJECT);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&contents);
        Ok(out)
    }
}

// ============================================================================
// UE STATE INDICATION (TS 24.501 D.5.4) — UE-initiated
// ============================================================================

/// One UPSI sublist (TS 24.501 Figure D.6.4.2): a PLMN plus the UPSCs of the
/// sections stored for it.
///
/// A UPSI is the pair (PLMN ID, UPSC) — the PLMN is the sublist key and the
/// UPSC is the per-section code the PCF assigned, so the sublist is how a UE
/// reports "for this PLMN I hold sections 1, 4 and 9".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpsiSublist {
    /// The PLMN whose sections are listed.
    pub plmn_id: PlmnId,
    /// UPSCs of the stored sections for that PLMN.
    pub upscs: Vec<u16>,
}

/// UE policy classmark (TS 24.501 D.6.5 octet 3).
///
/// Only the four defined bits are modelled; bits 5-8 of octet 3 and the whole
/// of the optional octets 4-5 are spare and "shall be coded as zero", so they
/// are not represented — a field that can only ever hold zero is a field that
/// can disagree with the encoder.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UePolicyClassmark {
    /// SupportANDSP (bit 1): ANDSP supported by the UE.
    pub andsp: bool,
    /// EPSURSP (bit 2): URSP provisioning in EPS supported by the UE.
    pub ursp_in_eps: bool,
    /// SVPSU (bit 3): VPS URSP supported by the UE.
    pub vps_ursp: bool,
    /// SupportRURE (bit 4): reporting of URSP rule enforcement supported.
    pub report_ursp_rule_enforcement: bool,
}

impl UePolicyClassmark {
    /// Octet 3 of D.6.5: the four capability bits, spare bits 5-8 zero.
    pub fn encode(&self) -> u8 {
        u8::from(self.andsp)
            | (u8::from(self.ursp_in_eps) << 1)
            | (u8::from(self.vps_ursp) << 2)
            | (u8::from(self.report_ursp_rule_enforcement) << 3)
    }

    /// Decode octet 3 of D.6.5, ignoring the spare bits.
    pub fn decode(octet: u8) -> Self {
        Self {
            andsp: octet & 0x01 != 0,
            ursp_in_eps: octet & 0x02 != 0,
            vps_ursp: octet & 0x04 != 0,
            report_ursp_rule_enforcement: octet & 0x08 != 0,
        }
    }
}

/// UE STATE INDICATION (TS 24.501 Table D.5.4.1.1): the UE reporting which UE
/// policy sections it holds and what policy features it supports.
///
/// Sent inside the Payload container IE of a REGISTRATION REQUEST with the
/// Payload container type set to "UE policy container" (§5.5.1.2.2), which is
/// the only transport this version of the protocol defines for it — there is no
/// UL NAS TRANSPORT path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UeStateIndication {
    /// UE-allocated PTI, 01H-77H (D.1.2).
    pub pti: u8,
    /// The UPSI list: one sublist per PLMN with stored sections. **May be
    /// empty** — D.2.2.2 NOTE 1 says a UE with no stored section sets the
    /// length of UPSI list contents to zero rather than omitting the IE, since
    /// the IE is mandatory (Table D.5.4.1.1).
    pub upsi_list: Vec<UpsiSublist>,
    /// The UE policy classmark (mandatory, D.6.5).
    pub classmark: UePolicyClassmark,
}

impl UeStateIndication {
    /// Encodes the UE policy delivery service message content: PTI, message
    /// type, UPSI list (LV-E) and UE policy classmark (LV).
    ///
    /// The UE OS Id IE (IEI 41H, optional) is not emitted: this simulator has
    /// no OS identity to report, and D.2.2.2 g) makes it a "may".
    pub fn encode(&self) -> Result<Vec<u8>> {
        if !(UE_PTI_MIN..=UE_PTI_MAX).contains(&self.pti) {
            return Err(UePolicyError::UePtiOutOfRange(self.pti));
        }

        // UPSI list contents: the concatenated sublists (Figure D.6.4.1).
        let mut upsi_contents: Vec<u8> = Vec::new();
        for sublist in &self.upsi_list {
            if sublist.upscs.is_empty() {
                // Figure D.6.4.2 shows UPSC 1 as a non-optional octet pair, so
                // a sublist with no UPSC is not representable. Reporting an
                // empty UPSI LIST is how "nothing stored" is said (NOTE 1);
                // an empty SUBLIST would claim a PLMN with no sections.
                return Err(UePolicyError::InvalidValue {
                    scope: "UPSI sublist",
                    msg: "must carry at least one UPSC (Figure D.6.4.2)".into(),
                });
            }
            // Length of UPSI sublist counts what follows it: PLMN (3) + 2 per UPSC.
            let sublist_len = 3 + 2 * sublist.upscs.len();
            let Ok(sublist_len) = u16::try_from(sublist_len) else {
                return Err(UePolicyError::InvalidValue {
                    scope: "UPSI sublist",
                    msg: "sublist length exceeds 65535 octets".into(),
                });
            };
            upsi_contents.extend_from_slice(&sublist_len.to_be_bytes());
            upsi_contents.extend_from_slice(&sublist.plmn_id.encode());
            for upsc in &sublist.upscs {
                upsi_contents.extend_from_slice(&upsc.to_be_bytes());
            }
        }
        let Ok(upsi_len) = u16::try_from(upsi_contents.len()) else {
            return Err(UePolicyError::InvalidValue {
                scope: "UPSI list",
                msg: "contents exceed 65535 octets".into(),
            });
        };

        let mut out: Vec<u8> = Vec::with_capacity(6 + upsi_contents.len());
        out.push(self.pti);
        out.push(UPDP_MSG_UE_STATE_INDICATION);
        // UPSI list: LV-E, so a 2-octet length and no IEI (Table D.5.4.1.1).
        out.extend_from_slice(&upsi_len.to_be_bytes());
        out.extend_from_slice(&upsi_contents);
        // UE policy classmark: LV, so a 1-octet length and no IEI. Length 1
        // because only octet 3 is emitted; octets 4-5 are spare.
        out.push(1);
        out.push(self.classmark.encode());
        Ok(out)
    }

    /// Decodes the message content [`Self::encode`] produces.
    ///
    /// Present for round-trip testing and for a peer implementation to use; the
    /// UE itself never receives this message (Direction: UE to network).
    pub fn decode(content: &[u8]) -> Result<Self> {
        let mut r = Reader::new(content);
        let pti = r.u8()?;
        if !(UE_PTI_MIN..=UE_PTI_MAX).contains(&pti) {
            return Err(UePolicyError::UePtiOutOfRange(pti));
        }
        let msg_type = r.u8()?;
        if msg_type != UPDP_MSG_UE_STATE_INDICATION {
            return Err(UePolicyError::UnexpectedMessageType {
                got: msg_type,
                expected: UPDP_MSG_UE_STATE_INDICATION,
            });
        }

        let upsi_len = usize::from(r.u16()?);
        let mut upsi = r.sub(upsi_len, "UPSI list")?;
        let mut upsi_list = Vec::new();
        while upsi.remaining() > 0 {
            let sublist_len = usize::from(upsi.u16()?);
            let mut sublist = upsi.sub(sublist_len, "UPSI sublist")?;
            let plmn_id = PlmnId::decode(sublist.take(3)?)?;
            let mut upscs = Vec::new();
            while sublist.remaining() > 0 {
                upscs.push(sublist.u16()?);
            }
            sublist.finish("UPSI sublist")?;
            if upscs.is_empty() {
                return Err(UePolicyError::InvalidValue {
                    scope: "UPSI sublist",
                    msg: "carries no UPSC (Figure D.6.4.2)".into(),
                });
            }
            upsi_list.push(UpsiSublist { plmn_id, upscs });
        }
        upsi.finish("UPSI list")?;

        let classmark_len = usize::from(r.u8()?);
        if classmark_len == 0 {
            return Err(UePolicyError::InvalidValue {
                scope: "UE policy classmark",
                msg: "mandatory IE with zero-length contents (D.6.5 needs octet 3)".into(),
            });
        }
        let classmark_bytes = r.take(classmark_len)?;
        // Only octet 3 carries defined bits; any further octets are spare.
        let classmark = UePolicyClassmark::decode(classmark_bytes[0]);
        Ok(Self {
            pti,
            upsi_list,
            classmark,
        })
    }
}

// ============================================================================
// Tests — golden vectors (E1) re-cited verbatim + fail-closed decode
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- E1 golden vectors (hex literals re-cited from
    // nextgcore/src/libs/nextgcore-nas/tests/ue_policy_golden_vectors/data.rs;
    // NOT shared as code — independence is the strict-peer value) -------------

    /// E1 vector (e): UE policy section management list in LV-E form.
    const VEC_E_SECTION_MGMT_LIST_LVE: &[u8] = &[
        0x00, 0x2B, // list contents length = 43
        0x00, 0x29, // sublist length = 41
        0x00, 0xF1, 0x10, // PLMN 001/01 (BCD)
        0x00, 0x24, // instruction contents length = 36
        0x00, 0x01, // UPSC = 0x0001
        0x00, 0x20, // part contents length = 32 (includes type octet)
        0x01, // part type URSP (0001)
        0x00, 0x1D, 0xFF, 0x00, 0x01, 0x01, 0x00, 0x17, // URSP rule (== VEC_A)
        0x00, 0x15, 0xFF, 0x00, 0x12, //
        0x01, 0x01, // SSC mode 1
        0x02, 0x01, 0x01, // S-NSSAI SST=1
        0x04, 0x09, 0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74, // DNN internet
        0x08, 0x03, // PDU session type IPv4v6
    ];

    /// E1 vector (f): complete MANAGE UE POLICY COMMAND content.
    const VEC_F_MANAGE_UE_POLICY_COMMAND: &[u8] = &[
        0x80, // PTI = 0x80
        0x01, // message type = MANAGE UE POLICY COMMAND
        0x00, 0x2B, // list contents length = 43
        0x00, 0x29, 0x00, 0xF1, 0x10, // sublist header
        0x00, 0x24, 0x00, 0x01, // instruction: len 36, UPSC 0x0001
        0x00, 0x20, 0x01, // part: len 32, type URSP
        0x00, 0x1D, 0xFF, 0x00, 0x01, 0x01, 0x00, 0x17, // URSP rule == VEC_A
        0x00, 0x15, 0xFF, 0x00, 0x12, //
        0x01, 0x01, //
        0x02, 0x01, 0x01, //
        0x04, 0x09, 0x08, 0x69, 0x6E, 0x74, 0x65, 0x72, 0x6E, 0x65, 0x74, //
        0x08, 0x03, //
    ];

    /// E1 vector (b): DNN "ims" URSP rule (used to test a non-match-all TD).
    const VEC_B_URSP_RULE_DNN_IMS: &[u8] = &[
        0x00, 0x1F, 0x0A, 0x00, 0x06, //
        0x88, 0x04, 0x03, 0x69, 0x6D, 0x73, // TD: DNN "ims"
        0x00, 0x14, //
        0x00, 0x12, 0x0A, 0x00, 0x0F, //
        0x01, 0x01, // SSC mode 1
        0x02, 0x01, 0x01, // S-NSSAI SST=1
        0x04, 0x04, 0x03, 0x69, 0x6D, 0x73, // DNN "ims"
        0x08, 0x03, // PDU session type IPv4v6
        0x10, 0x01, // preferred access 3GPP
    ];

    fn plmn_001_01() -> PlmnId {
        PlmnId {
            mcc: [0, 0, 1],
            mnc: [0, 1, 0],
            mnc_len: 2,
        }
    }

    #[test]
    fn plmn_bcd_decode_001_01() {
        assert_eq!(PlmnId::decode(&[0x00, 0xF1, 0x10]).unwrap(), plmn_001_01());
        // round-trip
        assert_eq!(plmn_001_01().encode(), [0x00, 0xF1, 0x10]);
    }

    #[test]
    fn decode_vec_f_command_fields() {
        let cmd = ManageUePolicyCommand::decode(VEC_F_MANAGE_UE_POLICY_COMMAND).unwrap();
        assert_eq!(cmd.pti, 0x80);
        assert_eq!(cmd.sublists.len(), 1);
        let sub = &cmd.sublists[0];
        assert_eq!(sub.plmn_id, plmn_001_01());
        assert_eq!(sub.instructions.len(), 1);
        let instr = &sub.instructions[0];
        assert_eq!(instr.upsc, 0x0001);
        assert_eq!(instr.parts.len(), 1);
        let part = &instr.parts[0];
        assert_eq!(part.part_type, UePolicyPartType::Ursp);
        let rules = part.decode_ursp().unwrap();
        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.precedence, 255);
        assert_eq!(
            rule.traffic_descriptor,
            vec![TrafficDescriptorComponent::MatchAll]
        );
        assert_eq!(rule.route_selection_descriptors.len(), 1);
        let rsd = &rule.route_selection_descriptors[0];
        assert_eq!(rsd.precedence, 255);
        assert_eq!(
            rsd.components,
            vec![
                RouteSelectionDescriptorComponent::SscMode(SscMode::SscMode1),
                RouteSelectionDescriptorComponent::SNssai { sst: 1, sd: None },
                RouteSelectionDescriptorComponent::Dnn("internet".to_string()),
                RouteSelectionDescriptorComponent::PduSessionType(PduSessionType::Ipv4v6),
            ]
        );
        assert_eq!(rule.ureri, None);
    }

    #[test]
    fn decode_vec_e_list_matches_vec_f_body() {
        // VEC_E is exactly the LV-E list that follows PTI+type in VEC_F.
        let mut r = Reader::new(VEC_E_SECTION_MGMT_LIST_LVE);
        let len = usize::from(r.u16().unwrap());
        let mut list = r.sub(len, "list").unwrap();
        r.finish("list lv-e").unwrap();
        let mut sublists = Vec::new();
        while list.remaining() > 0 {
            sublists.push(PlmnSublist::decode(&mut list).unwrap());
        }
        let cmd = ManageUePolicyCommand::decode(VEC_F_MANAGE_UE_POLICY_COMMAND).unwrap();
        assert_eq!(sublists, cmd.sublists);
    }

    #[test]
    fn decode_dnn_ims_rule() {
        let rules = decode_ursp_rules(VEC_B_URSP_RULE_DNN_IMS).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].precedence, 10);
        assert_eq!(
            rules[0].traffic_descriptor,
            vec![TrafficDescriptorComponent::Dnn("ims".to_string())]
        );
        assert!(rules[0]
            .route_selection_descriptors
            .iter()
            .any(|r| r.components.contains(
                &RouteSelectionDescriptorComponent::PreferredAccessType(
                    PreferredAccessType::ThreeGpp
                )
            )));
    }

    #[test]
    fn complete_encode_is_two_octets() {
        assert_eq!(
            ManageUePolicyComplete { pti: 0x80 }.encode().unwrap(),
            vec![0x80, 0x02]
        );
        assert_eq!(
            ManageUePolicyComplete { pti: 0xC3 }.encode().unwrap(),
            vec![0xC3, 0x02]
        );
    }

    #[test]
    fn complete_rejects_non_pcf_pti() {
        assert!(ManageUePolicyComplete { pti: 0x00 }.encode().is_err());
        assert!(ManageUePolicyComplete { pti: 0x7F }.encode().is_err());
        assert!(ManageUePolicyComplete { pti: 0xFF }.encode().is_err());
    }

    #[test]
    fn reject_encode_golden() {
        let reject = ManageUePolicyCommandReject {
            pti: 0x80,
            result: UePolicySectionManagementResult {
                subresults: vec![UePolicySectionManagementSubresult {
                    plmn_id: plmn_001_01(),
                    results: vec![UePolicyResult {
                        upsc: 0x0001,
                        failed_instruction_order: 1,
                        cause: CAUSE_PROTOCOL_ERROR_UNSPECIFIED,
                    }],
                }],
            },
        };
        // PTI(0x80) + type(0x03) + LV-E len(0x0009) + count(1) + PLMN(00 F1 10)
        // + UPSC(00 01) + order(00 01) + cause(6F)
        assert_eq!(
            reject.encode().unwrap(),
            vec![0x80, 0x03, 0x00, 0x09, 0x01, 0x00, 0xF1, 0x10, 0x00, 0x01, 0x00, 0x01, 0x6F,]
        );
    }

    #[test]
    fn reject_rejects_empty_result() {
        let reject = ManageUePolicyCommandReject {
            pti: 0x80,
            result: UePolicySectionManagementResult { subresults: vec![] },
        };
        assert!(reject.encode().is_err());
    }

    #[test]
    fn command_rejects_wrong_message_type() {
        let mut bad = VEC_F_MANAGE_UE_POLICY_COMMAND.to_vec();
        bad[1] = 0x02; // COMPLETE type in a COMMAND slot
        assert!(matches!(
            ManageUePolicyCommand::decode(&bad),
            Err(UePolicyError::UnexpectedMessageType { .. })
        ));
    }

    #[test]
    fn command_rejects_non_pcf_pti() {
        let mut bad = VEC_F_MANAGE_UE_POLICY_COMMAND.to_vec();
        bad[0] = 0x10;
        assert!(matches!(
            ManageUePolicyCommand::decode(&bad),
            Err(UePolicyError::PtiOutOfRange(0x10))
        ));
    }

    #[test]
    fn command_rejects_truncation() {
        for cut in 1..VEC_F_MANAGE_UE_POLICY_COMMAND.len() {
            assert!(
                ManageUePolicyCommand::decode(&VEC_F_MANAGE_UE_POLICY_COMMAND[..cut]).is_err(),
                "prefix of {cut} octets should not decode"
            );
        }
    }

    #[test]
    fn command_rejects_trailing_octets() {
        let mut bad = VEC_F_MANAGE_UE_POLICY_COMMAND.to_vec();
        bad.push(0x42); // stray octet after the LV-E list
        assert!(ManageUePolicyCommand::decode(&bad).is_err());
    }

    #[test]
    fn part_rejects_spare_type_bits() {
        let mut bad = VEC_F_MANAGE_UE_POLICY_COMMAND.to_vec();
        // The part-type octet is at index 15 in VEC_F (after part length 00 20).
        assert_eq!(bad[15], 0x01);
        bad[15] = 0x11; // set a spare bit (bits 8..5)
        assert!(ManageUePolicyCommand::decode(&bad).is_err());
    }

    #[test]
    fn reserved_part_type_rejected() {
        let mut bad = VEC_F_MANAGE_UE_POLICY_COMMAND.to_vec();
        bad[15] = 0x07; // reserved part type 0111
        assert!(ManageUePolicyCommand::decode(&bad).is_err());
    }

    #[test]
    fn ssc_mode_spare_bits_rejected() {
        // Isolate the URSP rule and flip the SSC mode value octet's spare bits.
        let mut rule = VEC_B_URSP_RULE_DNN_IMS.to_vec();
        // SSC mode value octet follows the first "01" id inside RSD contents.
        // Find the SSC-mode id/value pair (0x01,0x01) after the RSD contents len.
        // Index walk: rule[0..2]=len, [2]=prec, [3..5]=td len, td(6)=..., etc.
        // Simplest: mutate the known SSC value octet. In VEC_B it is at index 19.
        assert_eq!(rule[18], 0x01); // SSC mode id
        assert_eq!(rule[19], 0x01); // SSC mode value
        rule[19] = 0x09; // spare bit 4 set
        assert!(decode_ursp_rules(&rule).is_err());
    }

    #[test]
    fn delete_section_instruction_has_no_parts() {
        // A sublist with a single instruction carrying UPSC only (no parts).
        // list: len; sublist: len, PLMN, instr(len=2, UPSC only)
        let content = [
            0x80u8, 0x01, // PTI, type
            0x00, 0x09, // list len = 9 (one sublist on the wire)
            0x00, 0x07, // sublist len = 7 (PLMN 3 + instruction 4)
            0x00, 0xF1, 0x10, // PLMN
            0x00, 0x02, // instruction contents len = 2 (UPSC only)
            0x00, 0x07, // UPSC = 7
        ];
        let cmd = ManageUePolicyCommand::decode(&content).unwrap();
        assert_eq!(cmd.sublists[0].instructions[0].upsc, 7);
        assert!(cmd.sublists[0].instructions[0].parts.is_empty());
    }

    // ========================================================================
    // UE STATE INDICATION (TS 24.501 D.5.4) — #48
    // ========================================================================

    /// #48: the encoded UE STATE INDICATION is asserted BYTE FOR BYTE against a
    /// buffer computed by hand from Table D.5.4.1.1 and figures D.6.4.1/D.6.4.2.
    ///
    /// A round trip alone would not catch a wrong layout: a self-consistent
    /// encoder and decoder agree with each other while disagreeing with the PCF.
    #[test]
    fn ue_state_indication_encodes_the_d_5_4_layout_byte_for_byte() {
        let msg = UeStateIndication {
            pti: 0x05,
            upsi_list: vec![UpsiSublist {
                plmn_id: plmn_001_01(),
                upscs: vec![1, 4],
            }],
            classmark: UePolicyClassmark {
                andsp: true,
                ursp_in_eps: false,
                vps_ursp: false,
                report_ursp_rule_enforcement: false,
            },
        };

        assert_eq!(
            msg.encode().expect("encodes"),
            vec![
                0x05, // PTI (V, 1 octet) — UE-initiated range 01H-77H
                0x04, // UE STATE INDICATION message identity (Table D.6.1.1)
                0x00, 0x09, // UPSI list: LV-E length = 9 (one sublist)
                0x00, 0x07, // sublist length = 7 = PLMN 3 + 2 UPSCs x 2
                0x00, 0xF1, 0x10, // PLMN 001/01 (MNC digit 3 = 1111)
                0x00, 0x01, // UPSC 1
                0x00, 0x04, // UPSC 4
                0x01, // UE policy classmark: LV length = 1 (octet 3 only)
                0x01, // SupportANDSP set, EPSURSP/SVPSU/SupportRURE clear
            ]
        );
    }

    /// #48: a UE holding nothing still sends the mandatory UPSI list with a
    /// ZERO length (D.2.2.2 NOTE 1) rather than omitting it.
    ///
    /// The IE is mandatory in Table D.5.4.1.1, so omitting it produces a message
    /// the PCF cannot parse; the length field is what says "nothing stored".
    #[test]
    fn an_empty_upsi_list_is_a_zero_length_not_an_absent_ie() {
        let msg = UeStateIndication {
            pti: 0x77,
            upsi_list: Vec::new(),
            classmark: UePolicyClassmark::default(),
        };
        assert_eq!(
            msg.encode().expect("encodes"),
            vec![0x77, 0x04, 0x00, 0x00, 0x01, 0x00]
        );
    }

    /// #48: every classmark bit lands where D.6.5 Table D.6.5.1 puts it.
    ///
    /// Asserted bit by bit rather than on one combined value: a single "all set"
    /// vector passes against an encoder that has two bits transposed.
    #[test]
    fn classmark_bits_are_at_their_d_6_5_positions() {
        let bit = |cm: UePolicyClassmark| cm.encode();
        assert_eq!(
            bit(UePolicyClassmark {
                andsp: true,
                ..Default::default()
            }),
            0b0000_0001
        );
        assert_eq!(
            bit(UePolicyClassmark {
                ursp_in_eps: true,
                ..Default::default()
            }),
            0b0000_0010
        );
        assert_eq!(
            bit(UePolicyClassmark {
                vps_ursp: true,
                ..Default::default()
            }),
            0b0000_0100
        );
        assert_eq!(
            bit(UePolicyClassmark {
                report_ursp_rule_enforcement: true,
                ..Default::default()
            }),
            0b0000_1000
        );
        // Spare bits 5-8 "shall be coded as zero".
        assert_eq!(
            bit(UePolicyClassmark {
                andsp: true,
                ursp_in_eps: true,
                vps_ursp: true,
                report_ursp_rule_enforcement: true,
            }) & 0xF0,
            0,
        );
        // And every bit survives a decode.
        for octet in 0u8..16 {
            assert_eq!(UePolicyClassmark::decode(octet).encode(), octet);
        }
    }

    /// #48: decode is the inverse of encode over multi-PLMN and multi-UPSC
    /// shapes, so a PCF-side parser reading these bytes recovers the state.
    #[test]
    fn ue_state_indication_round_trips() {
        for msg in [
            UeStateIndication {
                pti: 0x01,
                upsi_list: Vec::new(),
                classmark: UePolicyClassmark::default(),
            },
            UeStateIndication {
                pti: 0x42,
                upsi_list: vec![
                    UpsiSublist {
                        plmn_id: plmn_001_01(),
                        upscs: vec![9],
                    },
                    UpsiSublist {
                        // 3-digit MNC, to exercise the MNC-digit-3 encoding.
                        plmn_id: PlmnId::from_bcd([0x21, 0x63, 0x54]).expect("valid"),
                        upscs: vec![1, 2, 0xFFFF],
                    },
                ],
                classmark: UePolicyClassmark {
                    andsp: true,
                    ursp_in_eps: true,
                    vps_ursp: true,
                    report_ursp_rule_enforcement: true,
                },
            },
        ] {
            let bytes = msg.encode().expect("encodes");
            assert_eq!(UeStateIndication::decode(&bytes).expect("decodes"), msg);
        }
    }

    /// #48: the PTI range is the UE-initiated one, 01H-77H (D.1.2).
    ///
    /// PTI 0 is rejected explicitly, because the issue that asked for this
    /// message suggested PTI 0 and D.1.2 forbids it — 0 means "no procedure
    /// transaction identity assigned".
    #[test]
    fn a_pti_outside_the_ue_range_is_refused() {
        let with_pti = |pti| UeStateIndication {
            pti,
            upsi_list: Vec::new(),
            classmark: UePolicyClassmark::default(),
        };
        assert!(matches!(
            with_pti(0x00).encode(),
            Err(UePolicyError::UePtiOutOfRange(0x00))
        ));
        assert!(matches!(
            with_pti(0x78).encode(),
            Err(UePolicyError::UePtiOutOfRange(0x78))
        ));
        assert!(matches!(
            with_pti(0x80).encode(),
            Err(UePolicyError::UePtiOutOfRange(0x80))
        ));
        assert!(with_pti(0x01).encode().is_ok());
        assert!(with_pti(0x77).encode().is_ok());
    }

    /// #48: a sublist with no UPSC is refused on encode and on decode.
    ///
    /// Figure D.6.4.2 makes UPSC 1 a non-optional octet pair, so such a sublist
    /// is not representable. Saying "nothing stored" is a zero-length LIST, not
    /// an empty sublist naming a PLMN that has no sections.
    #[test]
    fn a_sublist_with_no_upsc_is_refused_both_ways() {
        let msg = UeStateIndication {
            pti: 0x01,
            upsi_list: vec![UpsiSublist {
                plmn_id: plmn_001_01(),
                upscs: Vec::new(),
            }],
            classmark: UePolicyClassmark::default(),
        };
        assert!(matches!(
            msg.encode(),
            Err(UePolicyError::InvalidValue {
                scope: "UPSI sublist",
                ..
            })
        ));

        // Hand-built bytes for the same shape: a 3-octet sublist (PLMN only).
        let bytes = [
            0x01u8, 0x04, 0x00, 0x05, 0x00, 0x03, 0x00, 0xF1, 0x10, 0x01, 0x00,
        ];
        assert!(matches!(
            UeStateIndication::decode(&bytes),
            Err(UePolicyError::InvalidValue {
                scope: "UPSI sublist",
                ..
            })
        ));
    }

    /// #48: a zero-length UE policy classmark is refused — it is a mandatory IE
    /// and D.6.5 needs octet 3, so an empty one carries no capability at all.
    #[test]
    fn a_zero_length_classmark_is_refused() {
        let bytes = [0x01u8, 0x04, 0x00, 0x00, 0x00];
        assert!(matches!(
            UeStateIndication::decode(&bytes),
            Err(UePolicyError::InvalidValue {
                scope: "UE policy classmark",
                ..
            })
        ));
    }

    /// #48: the wrong message type is refused rather than silently accepted as a
    /// UE STATE INDICATION.
    #[test]
    fn a_non_ue_state_indication_message_type_is_refused() {
        let bytes = [
            0x01u8,
            UPDP_MSG_MANAGE_UE_POLICY_COMPLETE,
            0x00,
            0x00,
            0x01,
            0x00,
        ];
        assert!(matches!(
            UeStateIndication::decode(&bytes),
            Err(UePolicyError::UnexpectedMessageType {
                got: 0x02,
                expected: 0x04
            })
        ));
    }
}
