//! Traffic flow template (TS 24.008 §10.5.6.12, used by EPS as TS 24.301
//! §9.9.4.16) and the uplink packet matching a UE does with it (TS 23.401
//! §5.4.1, TS 23.060 §15.3; nextgsim #283).
//!
//! The codec is written from TS 24.008 Table 10.5.162 and is independent of
//! nextgcore's PGW-C encoder on purpose: this UE is what tests it.

use std::net::{Ipv4Addr, Ipv6Addr};

/// TFT operation code (octet 3, bits 8-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TftOperation {
    /// 001 Create new TFT.
    CreateNew,
    /// 010 Delete existing TFT.
    DeleteExisting,
    /// 011 Add packet filters to existing TFT.
    AddFilters,
    /// 100 Replace packet filters in existing TFT.
    ReplaceFilters,
    /// 101 Delete packet filters from existing TFT.
    DeleteFilters,
    /// 110 No TFT operation.
    NoOperation,
}

impl TftOperation {
    fn from_bits(b: u8) -> Option<Self> {
        Some(match b {
            1 => Self::CreateNew,
            2 => Self::DeleteExisting,
            3 => Self::AddFilters,
            4 => Self::ReplaceFilters,
            5 => Self::DeleteFilters,
            6 => Self::NoOperation,
            _ => return None,
        })
    }

    fn bits(self) -> u8 {
        match self {
            Self::CreateNew => 1,
            Self::DeleteExisting => 2,
            Self::AddFilters => 3,
            Self::ReplaceFilters => 4,
            Self::DeleteFilters => 5,
            Self::NoOperation => 6,
        }
    }
}

/// Packet filter direction (bits 6-5 of the identifier octet).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterDirection {
    /// 00 pre-Rel-7 TFT filter.
    PreRel7,
    /// 01 downlink only.
    Downlink,
    /// 10 uplink only.
    Uplink,
    /// 11 bidirectional.
    Bidirectional,
}

impl FilterDirection {
    fn from_bits(b: u8) -> Self {
        match b & 0x03 {
            0 => Self::PreRel7,
            1 => Self::Downlink,
            2 => Self::Uplink,
            _ => Self::Bidirectional,
        }
    }

    fn bits(self) -> u8 {
        match self {
            Self::PreRel7 => 0,
            Self::Downlink => 1,
            Self::Uplink => 2,
            Self::Bidirectional => 3,
        }
    }

    /// Whether the filter classifies uplink packets.
    pub fn applies_uplink(self) -> bool {
        matches!(self, Self::Uplink | Self::Bidirectional)
    }
}

/// One packet filter component (Table 10.5.162). "Local" is the UE, "remote"
/// the external network entity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterComponent {
    /// 0x10 IPv4 remote address and mask.
    Ipv4Remote(Ipv4Addr, Ipv4Addr),
    /// 0x11 IPv4 local address and mask.
    Ipv4Local(Ipv4Addr, Ipv4Addr),
    /// 0x20 IPv6 remote address and mask.
    Ipv6Remote(Ipv6Addr, Ipv6Addr),
    /// 0x21 IPv6 remote address / prefix length.
    Ipv6RemotePrefix(Ipv6Addr, u8),
    /// 0x23 IPv6 local address / prefix length.
    Ipv6LocalPrefix(Ipv6Addr, u8),
    /// 0x30 Protocol identifier / next header.
    Protocol(u8),
    /// 0x40 Single local port.
    LocalPort(u16),
    /// 0x41 Local port range (low, high).
    LocalPortRange(u16, u16),
    /// 0x50 Single remote port.
    RemotePort(u16),
    /// 0x51 Remote port range (low, high).
    RemotePortRange(u16, u16),
    /// 0x60 Security parameter index.
    Spi(u32),
    /// 0x70 Type of service / traffic class and mask.
    TypeOfService(u8, u8),
    /// 0x80 Flow label (20 bits).
    FlowLabel(u32),
    /// 0x81-0x87: an Ethernet component (MAC, VLAN, Ethertype): its type and
    /// value. Never matches an IP packet; kept so a filter round-trips.
    Ethernet(u8, Vec<u8>),
}

impl FilterComponent {
    /// The component value length for a type identifier, or `None` for a
    /// reserved one.
    fn value_len(ty: u8) -> Option<usize> {
        Some(match ty {
            0x10 | 0x11 => 8,
            0x20 => 32,
            0x21 | 0x23 => 17,
            0x30 => 1,
            0x40 | 0x50 => 2,
            0x41 | 0x51 => 4,
            0x60 => 4,
            0x70 => 2,
            0x80 => 3,
            0x81 | 0x82 => 6,
            0x83 | 0x84 | 0x87 => 2,
            0x85 | 0x86 => 1,
            _ => return None,
        })
    }

    fn decode(ty: u8, v: &[u8]) -> Self {
        let v4 = |o: usize| Ipv4Addr::new(v[o], v[o + 1], v[o + 2], v[o + 3]);
        let v6 = |o: usize| {
            let mut a = [0u8; 16];
            a.copy_from_slice(&v[o..o + 16]);
            Ipv6Addr::from(a)
        };
        let u16_at = |o: usize| u16::from_be_bytes([v[o], v[o + 1]]);
        match ty {
            0x10 => Self::Ipv4Remote(v4(0), v4(4)),
            0x11 => Self::Ipv4Local(v4(0), v4(4)),
            0x20 => Self::Ipv6Remote(v6(0), v6(16)),
            0x21 => Self::Ipv6RemotePrefix(v6(0), v[16]),
            0x23 => Self::Ipv6LocalPrefix(v6(0), v[16]),
            0x30 => Self::Protocol(v[0]),
            0x40 => Self::LocalPort(u16_at(0)),
            0x41 => Self::LocalPortRange(u16_at(0), u16_at(2)),
            0x50 => Self::RemotePort(u16_at(0)),
            0x51 => Self::RemotePortRange(u16_at(0), u16_at(2)),
            0x60 => Self::Spi(u32::from_be_bytes([v[0], v[1], v[2], v[3]])),
            0x70 => Self::TypeOfService(v[0], v[1]),
            0x80 => Self::FlowLabel(
                (u32::from(v[0] & 0x0F) << 16) | (u32::from(v[1]) << 8) | u32::from(v[2]),
            ),
            other => Self::Ethernet(other, v.to_vec()),
        }
    }

    fn encode(&self, out: &mut Vec<u8>) {
        match self {
            Self::Ipv4Remote(a, m) | Self::Ipv4Local(a, m) => {
                out.push(if matches!(self, Self::Ipv4Remote(..)) {
                    0x10
                } else {
                    0x11
                });
                out.extend_from_slice(&a.octets());
                out.extend_from_slice(&m.octets());
            }
            Self::Ipv6Remote(a, m) => {
                out.push(0x20);
                out.extend_from_slice(&a.octets());
                out.extend_from_slice(&m.octets());
            }
            Self::Ipv6RemotePrefix(a, p) | Self::Ipv6LocalPrefix(a, p) => {
                out.push(if matches!(self, Self::Ipv6RemotePrefix(..)) {
                    0x21
                } else {
                    0x23
                });
                out.extend_from_slice(&a.octets());
                out.push(*p);
            }
            Self::Protocol(p) => out.extend_from_slice(&[0x30, *p]),
            Self::LocalPort(p) => {
                out.push(0x40);
                out.extend_from_slice(&p.to_be_bytes());
            }
            Self::LocalPortRange(l, h) => {
                out.push(0x41);
                out.extend_from_slice(&l.to_be_bytes());
                out.extend_from_slice(&h.to_be_bytes());
            }
            Self::RemotePort(p) => {
                out.push(0x50);
                out.extend_from_slice(&p.to_be_bytes());
            }
            Self::RemotePortRange(l, h) => {
                out.push(0x51);
                out.extend_from_slice(&l.to_be_bytes());
                out.extend_from_slice(&h.to_be_bytes());
            }
            Self::Spi(s) => {
                out.push(0x60);
                out.extend_from_slice(&s.to_be_bytes());
            }
            Self::TypeOfService(t, m) => out.extend_from_slice(&[0x70, *t, *m]),
            Self::FlowLabel(f) => {
                out.push(0x80);
                out.extend_from_slice(&[((f >> 16) & 0x0F) as u8, (f >> 8) as u8, *f as u8]);
            }
            Self::Ethernet(ty, v) => {
                out.push(*ty);
                out.extend_from_slice(v);
            }
        }
    }
}

/// One packet filter of a TFT.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketFilter {
    /// Packet filter identifier (0-15).
    pub id: u8,
    /// Direction the filter applies to.
    pub direction: FilterDirection,
    /// Evaluation precedence: the lower the value, the earlier it is evaluated.
    pub precedence: u8,
    /// Components; a packet matches when it matches every one.
    pub components: Vec<FilterComponent>,
}

/// A decoded TFT IE value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tft {
    /// The operation the network asks for.
    pub operation: TftOperation,
    /// Packet filters (create / add / replace).
    pub filters: Vec<PacketFilter>,
    /// Packet filter identifiers (delete packet filters).
    pub deleted_ids: Vec<u8>,
}

/// Why a TFT is refused (TS 24.301 §6.4.2.4, §6.4.3.4, Annex B).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TftError {
    /// ESM cause #41 Semantic error in the TFT operation.
    SemanticOperation,
    /// ESM cause #42 Syntactical error in the TFT operation.
    SyntacticalOperation,
    /// ESM cause #44 Semantic errors in packet filter(s).
    SemanticFilter,
    /// ESM cause #45 Syntactical errors in packet filter(s).
    SyntacticalFilter,
}

impl TftError {
    /// The ESM cause value.
    pub fn esm_cause(self) -> u8 {
        match self {
            Self::SemanticOperation => 41,
            Self::SyntacticalOperation => 42,
            Self::SemanticFilter => 44,
            Self::SyntacticalFilter => 45,
        }
    }
}

impl Tft {
    /// Decode a TFT IE value (from octet 3).
    pub fn decode(v: &[u8]) -> Result<Self, TftError> {
        let (&first, mut rest) = v.split_first().ok_or(TftError::SyntacticalOperation)?;
        let operation = TftOperation::from_bits(first >> 5).ok_or(TftError::SemanticOperation)?;
        let e_bit = first & 0x10 != 0;
        let n = usize::from(first & 0x0F);
        let mut filters = Vec::new();
        let mut deleted_ids = Vec::new();
        match operation {
            // §10.5.6.12: these carry an empty packet filter list
            TftOperation::DeleteExisting | TftOperation::NoOperation => {
                if n != 0 {
                    return Err(TftError::SyntacticalOperation);
                }
            }
            TftOperation::DeleteFilters => {
                if n == 0 || rest.len() < n {
                    return Err(TftError::SyntacticalOperation);
                }
                deleted_ids = rest[..n].iter().map(|b| b & 0x0F).collect();
                rest = &rest[n..];
            }
            TftOperation::CreateNew | TftOperation::AddFilters | TftOperation::ReplaceFilters => {
                // §6.4.2.4 b) 1), §6.4.3.4 b) 1): an empty packet filter list
                if n == 0 {
                    return Err(TftError::SyntacticalOperation);
                }
                for _ in 0..n {
                    // §6.4.2.4 b) 2), §6.4.3.4 b) 6): fewer packet filters
                    // than the number announced, #42
                    if rest.is_empty() {
                        return Err(TftError::SyntacticalOperation);
                    }
                    if rest.len() < 3 {
                        return Err(TftError::SyntacticalFilter);
                    }
                    let (id_dir, precedence, len) = (rest[0], rest[1], usize::from(rest[2]));
                    let body = rest.get(3..3 + len).ok_or(TftError::SyntacticalFilter)?;
                    rest = &rest[3 + len..];
                    let mut components = Vec::new();
                    let mut b = body;
                    while let Some((&ty, after)) = b.split_first() {
                        let l =
                            FilterComponent::value_len(ty).ok_or(TftError::SyntacticalFilter)?;
                        let value = after.get(..l).ok_or(TftError::SyntacticalFilter)?;
                        // "there shall not be more than one occurrence of each
                        // packet filter component type": a coding error, #45
                        if components
                            .iter()
                            .any(|c: &FilterComponent| component_type(c) == ty)
                        {
                            return Err(TftError::SyntacticalFilter);
                        }
                        components.push(FilterComponent::decode(ty, value));
                        b = &after[l..];
                    }
                    if components.is_empty() {
                        return Err(TftError::SyntacticalFilter);
                    }
                    filters.push(PacketFilter {
                        id: id_dir & 0x0F,
                        direction: FilterDirection::from_bits(id_dir >> 4),
                        precedence,
                        components,
                    });
                }
                // §6.4.2.4 d) 1): two filters with one identifier, #45
                let mut ids: Vec<u8> = filters.iter().map(|f| f.id).collect();
                ids.sort_unstable();
                ids.dedup();
                if ids.len() != filters.len() {
                    return Err(TftError::SyntacticalFilter);
                }
            }
        }
        // The parameters list (E bit) is accepted and not acted on. Without
        // the E bit, octets after the announced packet filters mean more
        // filters than announced: §6.4.2.4 b) 2), §6.4.3.4 b) 6), #42.
        if !e_bit && !rest.is_empty() {
            return Err(TftError::SyntacticalOperation);
        }
        Ok(Self {
            operation,
            filters,
            deleted_ids,
        })
    }

    /// Encode the TFT IE value (from octet 3).
    pub fn encode(&self) -> Vec<u8> {
        let n = match self.operation {
            TftOperation::DeleteFilters => self.deleted_ids.len(),
            TftOperation::DeleteExisting | TftOperation::NoOperation => 0,
            _ => self.filters.len(),
        };
        let mut out = vec![(self.operation.bits() << 5) | (n as u8 & 0x0F)];
        match self.operation {
            TftOperation::DeleteFilters => {
                out.extend(self.deleted_ids.iter().map(|i| i & 0x0F));
            }
            TftOperation::DeleteExisting | TftOperation::NoOperation => {}
            _ => {
                for f in &self.filters {
                    out.push((f.direction.bits() << 4) | (f.id & 0x0F));
                    out.push(f.precedence);
                    let mut body = Vec::new();
                    for c in &f.components {
                        c.encode(&mut body);
                    }
                    out.push(body.len() as u8);
                    out.extend_from_slice(&body);
                }
            }
        }
        out
    }
}

/// A Traffic flow aggregate description (TS 24.301 §9.9.4.15, coded as the
/// TFT of TS 24.008 §10.5.6.12) for a UE-requested change of the QoS of
/// existing traffic flows (§6.5.4.2, #434). TS 24.301 §6.5.4.2: the UE
/// includes "the packet filter identifier(s) ... in the Packet filter
/// identifier parameter in the parameters list" and sets the TFT operation
/// code to "No TFT operation"; TS 24.008 §10.5.6.12: for that operation the
/// number of packet filters is coded as 0 and the packet filter list is
/// empty. So: op code 110, E bit 1, count 0, then parameter 03H "Packet
/// filter identifier" with one octet per identifier (bits 4-1).
pub fn no_operation_aggregate(ids: &[u8]) -> Vec<u8> {
    let mut out = vec![(TftOperation::NoOperation.bits() << 5) | 0x10];
    out.extend_from_slice(&[PARAMETER_PACKET_FILTER_IDENTIFIER, ids.len() as u8]);
    out.extend(ids.iter().map(|i| i & 0x0F));
    out
}

/// TS 24.008 §10.5.6.12, parameters list: parameter identifier 03H "Packet
/// filter identifier".
const PARAMETER_PACKET_FILTER_IDENTIFIER: u8 = 0x03;

fn component_type(c: &FilterComponent) -> u8 {
    let mut v = Vec::new();
    c.encode(&mut v);
    v[0]
}

/// The fields of an uplink IP packet a packet filter looks at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PacketKey {
    /// Source (local) address.
    pub src: std::net::IpAddr,
    /// Destination (remote) address.
    pub dst: std::net::IpAddr,
    /// IPv4 protocol / IPv6 next header.
    pub protocol: u8,
    /// Source (local) port, for TCP / UDP / SCTP.
    pub src_port: Option<u16>,
    /// Destination (remote) port, for TCP / UDP / SCTP.
    pub dst_port: Option<u16>,
    /// IPsec ESP / AH security parameter index.
    pub spi: Option<u32>,
    /// IPv4 type of service / IPv6 traffic class.
    pub tos: u8,
    /// IPv6 flow label.
    pub flow_label: Option<u32>,
}

impl PacketKey {
    /// Parse an IPv4 or IPv6 packet; `None` for anything else.
    pub fn parse(p: &[u8]) -> Option<Self> {
        match p.first()? >> 4 {
            4 => {
                let ihl = usize::from(p.first()? & 0x0F) * 4;
                if p.len() < ihl.max(20) {
                    return None;
                }
                let protocol = p[9];
                let (src_port, dst_port, spi) = transport(protocol, p.get(ihl..)?);
                Some(Self {
                    src: Ipv4Addr::new(p[12], p[13], p[14], p[15]).into(),
                    dst: Ipv4Addr::new(p[16], p[17], p[18], p[19]).into(),
                    protocol,
                    src_port,
                    dst_port,
                    spi,
                    tos: p[1],
                    flow_label: None,
                })
            }
            6 => {
                if p.len() < 40 {
                    return None;
                }
                let mut s = [0u8; 16];
                s.copy_from_slice(&p[8..24]);
                let mut d = [0u8; 16];
                d.copy_from_slice(&p[24..40]);
                let protocol = p[6];
                let (src_port, dst_port, spi) = transport(protocol, &p[40..]);
                Some(Self {
                    src: Ipv6Addr::from(s).into(),
                    dst: Ipv6Addr::from(d).into(),
                    protocol,
                    src_port,
                    dst_port,
                    spi,
                    tos: (p[0] << 4) | (p[1] >> 4),
                    flow_label: Some(
                        (u32::from(p[1] & 0x0F) << 16) | (u32::from(p[2]) << 8) | u32::from(p[3]),
                    ),
                })
            }
            _ => None,
        }
    }
}

fn transport(protocol: u8, l4: &[u8]) -> (Option<u16>, Option<u16>, Option<u32>) {
    match protocol {
        // TCP, UDP, SCTP
        6 | 17 | 132 if l4.len() >= 4 => (
            Some(u16::from_be_bytes([l4[0], l4[1]])),
            Some(u16::from_be_bytes([l4[2], l4[3]])),
            None,
        ),
        // ESP: SPI first
        50 if l4.len() >= 4 => (
            None,
            None,
            Some(u32::from_be_bytes([l4[0], l4[1], l4[2], l4[3]])),
        ),
        // AH: SPI at octet 4
        51 if l4.len() >= 8 => (
            None,
            None,
            Some(u32::from_be_bytes([l4[4], l4[5], l4[6], l4[7]])),
        ),
        _ => (None, None, None),
    }
}

fn v4_matches(a: std::net::IpAddr, want: Ipv4Addr, mask: Ipv4Addr) -> bool {
    match a {
        std::net::IpAddr::V4(a) => {
            u32::from(a) & u32::from(mask) == u32::from(want) & u32::from(mask)
        }
        std::net::IpAddr::V6(_) => false,
    }
}

fn v6_masked(a: std::net::IpAddr, want: Ipv6Addr, mask: u128) -> bool {
    match a {
        std::net::IpAddr::V6(a) => u128::from(a) & mask == u128::from(want) & mask,
        std::net::IpAddr::V4(_) => false,
    }
}

fn prefix_mask(len: u8) -> u128 {
    match len {
        0 => 0,
        l if l >= 128 => u128::MAX,
        l => u128::MAX << (128 - u32::from(l)),
    }
}

impl PacketFilter {
    /// Whether an uplink packet matches every component (TS 23.060 §15.3).
    pub fn matches_uplink(&self, k: &PacketKey) -> bool {
        self.direction.applies_uplink()
            && self.components.iter().all(|c| match c {
                FilterComponent::Ipv4Remote(a, m) => v4_matches(k.dst, *a, *m),
                FilterComponent::Ipv4Local(a, m) => v4_matches(k.src, *a, *m),
                FilterComponent::Ipv6Remote(a, m) => v6_masked(k.dst, *a, u128::from(*m)),
                FilterComponent::Ipv6RemotePrefix(a, p) => v6_masked(k.dst, *a, prefix_mask(*p)),
                FilterComponent::Ipv6LocalPrefix(a, p) => v6_masked(k.src, *a, prefix_mask(*p)),
                FilterComponent::Protocol(p) => k.protocol == *p,
                FilterComponent::LocalPort(p) => k.src_port == Some(*p),
                FilterComponent::LocalPortRange(l, h) => {
                    k.src_port.is_some_and(|x| (*l..=*h).contains(&x))
                }
                FilterComponent::RemotePort(p) => k.dst_port == Some(*p),
                FilterComponent::RemotePortRange(l, h) => {
                    k.dst_port.is_some_and(|x| (*l..=*h).contains(&x))
                }
                FilterComponent::Spi(s) => k.spi == Some(*s),
                FilterComponent::TypeOfService(t, m) => k.tos & m == t & m,
                FilterComponent::FlowLabel(f) => k.flow_label == Some(*f),
                FilterComponent::Ethernet(..) => false,
            })
    }
}

/// Apply a MODIFY's TFT operation to the filters a bearer holds
/// (TS 24.301 §6.4.3.3, §6.4.3.4). `dedicated`: the bearer is a dedicated one,
/// which cannot be left without a TFT. The inconsistencies §6.4.3.4 resolves
/// rather than diagnoses are resolved as it says:
/// - "Create new TFT" on a bearer with one replaces it (a) 1));
/// - add / replace on a default bearer with no TFT creates one (a) 2));
/// - delete on a default bearer with no TFT, or deleting its last filter, leaves
///   it without one, i.e. match-all (a) 2), a) 3));
/// - replacing a filter that does not exist adds it, deleting one that does not
///   exist is done (b) 3), b) 4));
/// - an added filter whose identifier is in use replaces the old one (d) 1)).
pub fn apply_operation(
    current: &[PacketFilter],
    tft: &Tft,
    dedicated: bool,
) -> Result<Vec<PacketFilter>, TftError> {
    let mut out = current.to_vec();
    match tft.operation {
        TftOperation::CreateNew => out = tft.filters.clone(),
        // a) 4)
        TftOperation::DeleteExisting if dedicated => return Err(TftError::SemanticOperation),
        TftOperation::DeleteExisting => out.clear(),
        // d) 1) / d) 2): an old filter with a new one's identifier, or with
        // its precedence (the clash "among all TFTs" includes this one), is
        // deleted
        TftOperation::AddFilters | TftOperation::ReplaceFilters => {
            out.retain(|x| {
                !tft.filters
                    .iter()
                    .any(|f| f.id == x.id || f.precedence == x.precedence)
            });
            out.extend(tft.filters.iter().cloned());
        }
        TftOperation::DeleteFilters => {
            out.retain(|x| !tft.deleted_ids.contains(&x.id));
            // a) 3): a dedicated bearer's TFT may not be emptied
            if dedicated && out.is_empty() {
                return Err(TftError::SemanticOperation);
            }
        }
        TftOperation::NoOperation => {}
    }
    // c) 2): a dedicated bearer keeps an uplink filter
    if dedicated && !out.iter().any(|f| f.direction.applies_uplink()) {
        return Err(TftError::SemanticFilter);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A create-new TFT: one uplink filter, precedence 0x10, UDP to
    /// 10.45.0.1/32 port 5060. Hand-derived from TS 24.008 Figure 10.5.144 /
    /// Table 10.5.162: 21 | 21 10 0E | 10 0A2D0001 FFFFFFFF | 30 11 | 50 13C4.
    const SIP_UPLINK: [u8; 14 + 3] = [
        0x21, 0x21, 0x10, 0x0E, 0x10, 0x0A, 0x2D, 0x00, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x30, 0x11,
        0x50, 0x13,
    ];

    fn sip_tft() -> Vec<u8> {
        let mut v = SIP_UPLINK.to_vec();
        v.push(0xC4);
        v
    }

    #[test]
    fn a_create_new_tft_decodes_from_its_24008_layout() {
        let tft = Tft::decode(&sip_tft()).expect("decode");
        assert_eq!(tft.operation, TftOperation::CreateNew);
        assert_eq!(
            tft.filters,
            vec![PacketFilter {
                id: 1,
                direction: FilterDirection::Uplink,
                precedence: 0x10,
                components: vec![
                    FilterComponent::Ipv4Remote(
                        Ipv4Addr::new(10, 45, 0, 1),
                        Ipv4Addr::new(255, 255, 255, 255)
                    ),
                    FilterComponent::Protocol(17),
                    FilterComponent::RemotePort(5060),
                ],
            }]
        );
        assert_eq!(tft.encode(), sip_tft(), "round trip");
    }

    /// Every component type round-trips at its Table 10.5.162 length.
    #[test]
    fn every_component_type_round_trips() {
        let comps = vec![
            FilterComponent::Ipv4Local(Ipv4Addr::new(10, 0, 0, 2), Ipv4Addr::new(255, 0, 0, 0)),
            FilterComponent::LocalPortRange(1000, 2000),
            FilterComponent::RemotePortRange(3000, 4000),
            FilterComponent::Spi(0xDEADBEEF),
            FilterComponent::TypeOfService(0xB8, 0xFC),
        ];
        let v6 = vec![
            FilterComponent::Ipv6RemotePrefix("2001:db8::1".parse().unwrap(), 64),
            FilterComponent::Ipv6LocalPrefix("2001:db8::2".parse().unwrap(), 128),
            FilterComponent::LocalPort(7),
            FilterComponent::FlowLabel(0xABCDE),
        ];
        let v6m = vec![FilterComponent::Ipv6Remote(
            "2001:db8::".parse().unwrap(),
            "ffff:ffff::".parse().unwrap(),
        )];
        let eth = vec![FilterComponent::Ethernet(0x87, vec![0x08, 0x00])];
        for components in [comps, v6, v6m, eth] {
            let tft = Tft {
                operation: TftOperation::CreateNew,
                filters: vec![PacketFilter {
                    id: 3,
                    direction: FilterDirection::Bidirectional,
                    precedence: 7,
                    components,
                }],
                deleted_ids: vec![],
            };
            assert_eq!(Tft::decode(&tft.encode()).expect("decode"), tft);
        }
    }

    #[test]
    fn malformed_tfts_map_to_their_esm_causes() {
        // Reserved operation code 000 / 111
        assert_eq!(Tft::decode(&[0x01]), Err(TftError::SemanticOperation));
        // Create with no filters (§6.4.2.4 b) 1))
        assert_eq!(Tft::decode(&[0x20]), Err(TftError::SyntacticalOperation));
        // Delete existing with a filter count
        assert_eq!(Tft::decode(&[0x41]), Err(TftError::SyntacticalOperation));
        // A filter that runs past the IE
        assert_eq!(
            Tft::decode(&SIP_UPLINK[..10]),
            Err(TftError::SyntacticalFilter)
        );
        // A reserved component type
        assert_eq!(
            Tft::decode(&[0x21, 0x21, 0x00, 0x02, 0x99, 0x00]),
            Err(TftError::SyntacticalFilter)
        );
        // The same component type twice
        assert_eq!(
            Tft::decode(&[0x21, 0x21, 0x00, 0x04, 0x30, 0x11, 0x30, 0x06]),
            Err(TftError::SyntacticalFilter)
        );
        // Two filters announced, one present; one announced, two present
        // without the E bit (§6.4.2.4 b) 2), §6.4.3.4 b) 6)); a parameters
        // list after the filters with the E bit set is accepted
        assert_eq!(
            Tft::decode(&[0x22, 0x21, 0x00, 0x02, 0x30, 0x11]),
            Err(TftError::SyntacticalOperation)
        );
        assert_eq!(
            Tft::decode(&[0x21, 0x21, 0x00, 0x02, 0x30, 0x11, 0x22, 0x01, 0x02, 0x30, 0x06]),
            Err(TftError::SyntacticalOperation)
        );
        assert_eq!(
            Tft::decode(&[0x51, 0x01, 0x02]),
            Err(TftError::SyntacticalOperation),
            "delete filters: more ids than announced"
        );
        assert!(Tft::decode(&[0x31, 0x21, 0x00, 0x02, 0x30, 0x11, 0x01, 0x01, 0x00]).is_ok());
        // Two filters with one identifier (§6.4.2.4 d) 1))
        assert_eq!(
            Tft::decode(&[0x22, 0x21, 0x00, 0x02, 0x30, 0x11, 0x21, 0x01, 0x02, 0x30, 0x06]),
            Err(TftError::SyntacticalFilter)
        );
    }

    fn udp4(src: [u8; 4], dst: [u8; 4], sport: u16, dport: u16) -> Vec<u8> {
        let mut p = vec![0x45, 0x00, 0x00, 0x1C, 0, 0, 0, 0, 64, 17, 0, 0];
        p.extend_from_slice(&src);
        p.extend_from_slice(&dst);
        p.extend_from_slice(&sport.to_be_bytes());
        p.extend_from_slice(&dport.to_be_bytes());
        p.extend_from_slice(&[0, 8, 0, 0]);
        p
    }

    #[test]
    fn an_uplink_filter_matches_on_every_component_and_only_then() {
        let f = &Tft::decode(&sip_tft()).unwrap().filters[0];
        let key = |p: Vec<u8>| PacketKey::parse(&p).unwrap();
        assert!(f.matches_uplink(&key(udp4([10, 45, 0, 2], [10, 45, 0, 1], 40000, 5060))));
        assert!(!f.matches_uplink(&key(udp4([10, 45, 0, 2], [10, 45, 0, 1], 40000, 5061))));
        assert!(!f.matches_uplink(&key(udp4([10, 45, 0, 2], [10, 45, 0, 9], 40000, 5060))));
        let mut tcp = udp4([10, 45, 0, 2], [10, 45, 0, 1], 40000, 5060);
        tcp[9] = 6;
        assert!(!f.matches_uplink(&key(tcp)), "protocol");
        let down = PacketFilter {
            direction: FilterDirection::Downlink,
            ..f.clone()
        };
        assert!(
            !down.matches_uplink(&key(udp4([10, 45, 0, 2], [10, 45, 0, 1], 40000, 5060))),
            "a downlink-only filter does not classify uplink"
        );
    }

    #[test]
    fn modify_operations_follow_6_4_3_4() {
        let base = Tft::decode(&sip_tft()).unwrap().filters;
        let filter = |id: u8, proto: u8| PacketFilter {
            id,
            direction: FilterDirection::Uplink,
            precedence: 0x20 + id,
            components: vec![FilterComponent::Protocol(proto)],
        };
        let op = |operation, filters, deleted_ids| Tft {
            operation,
            filters,
            deleted_ids,
        };
        // Add: a new identifier is added; one in use is replaced (d) 1))
        let two = apply_operation(
            &base,
            &op(TftOperation::AddFilters, vec![filter(2, 6)], vec![]),
            true,
        )
        .unwrap();
        assert_eq!(two.iter().map(|f| f.id).collect::<Vec<_>>(), vec![1, 2]);
        let replaced = apply_operation(
            &two,
            &op(TftOperation::AddFilters, vec![filter(2, 17)], vec![]),
            true,
        )
        .unwrap();
        assert_eq!(replaced.len(), 2);
        assert_eq!(replaced[1].components, vec![FilterComponent::Protocol(17)]);
        // Add with the precedence of one of this TFT's own filters: the old
        // one is deleted (d) 2), "among all TFTs")
        let same_precedence = PacketFilter {
            precedence: 0x10,
            ..filter(3, 6)
        };
        assert_eq!(
            apply_operation(
                &replaced,
                &op(TftOperation::AddFilters, vec![same_precedence], vec![]),
                true
            )
            .unwrap()
            .iter()
            .map(|f| (f.id, f.precedence))
            .collect::<Vec<_>>(),
            vec![(2, 0x22), (3, 0x10)]
        );
        // Replace of a filter that does not exist adds it (b) 3))
        assert_eq!(
            apply_operation(
                &base,
                &op(TftOperation::ReplaceFilters, vec![filter(9, 6)], vec![]),
                true
            )
            .unwrap()
            .len(),
            2
        );
        // Delete: one of two; a non-existent one is done (b) 4)); the last one of a
        // dedicated bearer is #41 (a) 3))
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteFilters, vec![], vec![1]),
                true
            )
            .unwrap()
            .len(),
            1
        );
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteFilters, vec![], vec![7]),
                true
            )
            .unwrap()
            .len(),
            2
        );
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteFilters, vec![], vec![1, 2]),
                true
            ),
            Err(TftError::SemanticOperation)
        );
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteFilters, vec![], vec![1, 2]),
                false
            )
            .unwrap(),
            vec![]
        );
        // Delete existing TFT: #41 for a dedicated bearer (a) 4)), match-all for the default
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteExisting, vec![], vec![]),
                true
            ),
            Err(TftError::SemanticOperation)
        );
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::DeleteExisting, vec![], vec![]),
                false
            )
            .unwrap(),
            vec![]
        );
        // Create new on a bearer with a TFT replaces it (a) 1))
        assert_eq!(
            apply_operation(
                &two,
                &op(TftOperation::CreateNew, vec![filter(5, 6)], vec![]),
                true
            )
            .unwrap(),
            vec![filter(5, 6)]
        );
        // Add on a default bearer with no TFT creates one (a) 2))
        assert_eq!(
            apply_operation(
                &[],
                &op(TftOperation::AddFilters, vec![filter(3, 6)], vec![]),
                false
            )
            .unwrap()
            .len(),
            1
        );
        // A dedicated bearer left with only downlink filters: #44 (c) 2))
        let down = PacketFilter {
            direction: FilterDirection::Downlink,
            ..filter(4, 6)
        };
        assert_eq!(
            apply_operation(&[], &op(TftOperation::CreateNew, vec![down], vec![]), true),
            Err(TftError::SemanticFilter)
        );
    }

    /// #434, TS 24.301 §6.5.4.2 and TS 24.008 §10.5.6.12: the UE's "No TFT
    /// operation" aggregate is operation code 110, E bit set, no packet filter,
    /// and the identifiers in parameter 03H of the parameters list.
    #[test]
    fn a_no_operation_aggregate_lists_the_identifiers() {
        assert_eq!(no_operation_aggregate(&[1]), vec![0xD0, 0x03, 0x01, 0x01]);
        assert_eq!(
            no_operation_aggregate(&[1, 2, 15]),
            vec![0xD0, 0x03, 0x03, 0x01, 0x02, 0x0F]
        );
        // The UE's own decoder reads it as op 110 with no packet filter
        let t = Tft::decode(&no_operation_aggregate(&[1, 2])).unwrap();
        assert_eq!(t.operation, TftOperation::NoOperation);
        assert!(t.filters.is_empty() && t.deleted_ids.is_empty());
    }
}
