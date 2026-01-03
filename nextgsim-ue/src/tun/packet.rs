//! IP packet parsing utilities.
//!
//! This module provides utilities for parsing IP packet headers to extract
//! routing information needed for user plane data handling.

use std::net::{Ipv4Addr, Ipv6Addr};

/// IP protocol version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IpVersion {
    /// IPv4
    V4,
    /// IPv6
    V6,
}

/// Parsed IP packet information.
#[derive(Debug, Clone)]
pub struct PacketInfo {
    /// IP version
    pub version: IpVersion,
    /// Source IP address (as bytes)
    pub src_addr: Vec<u8>,
    /// Destination IP address (as bytes)
    pub dst_addr: Vec<u8>,
    /// IP protocol number (e.g., 6 for TCP, 17 for UDP)
    pub protocol: u8,
    /// Total packet length
    pub total_length: u16,
}

/// Represents a parsed IP packet.
#[derive(Debug)]
pub struct IpPacket<'a> {
    /// The raw packet data
    data: &'a [u8],
    /// Parsed packet information
    info: PacketInfo,
}

impl<'a> IpPacket<'a> {
    /// Parses an IP packet from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns `None` if the packet is too short or has an invalid version.
    #[must_use]
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        if data.is_empty() {
            return None;
        }

        let version = (data[0] >> 4) & 0x0F;

        match version {
            4 => Self::parse_ipv4(data),
            6 => Self::parse_ipv6(data),
            _ => None,
        }
    }

    /// Parses an IPv4 packet.
    fn parse_ipv4(data: &'a [u8]) -> Option<Self> {
        // Minimum IPv4 header is 20 bytes
        if data.len() < 20 {
            return None;
        }

        let ihl = (data[0] & 0x0F) as usize * 4;
        if data.len() < ihl {
            return None;
        }

        let total_length = u16::from_be_bytes([data[2], data[3]]);
        let protocol = data[9];
        let src_addr = data[12..16].to_vec();
        let dst_addr = data[16..20].to_vec();

        Some(Self {
            data,
            info: PacketInfo {
                version: IpVersion::V4,
                src_addr,
                dst_addr,
                protocol,
                total_length,
            },
        })
    }

    /// Parses an IPv6 packet.
    fn parse_ipv6(data: &'a [u8]) -> Option<Self> {
        // IPv6 header is 40 bytes
        if data.len() < 40 {
            return None;
        }

        let payload_length = u16::from_be_bytes([data[4], data[5]]);
        let total_length = 40 + payload_length;
        let protocol = data[6]; // Next Header
        let src_addr = data[8..24].to_vec();
        let dst_addr = data[24..40].to_vec();

        Some(Self {
            data,
            info: PacketInfo {
                version: IpVersion::V6,
                src_addr,
                dst_addr,
                protocol,
                total_length,
            },
        })
    }

    /// Returns the IP version.
    #[must_use]
    pub fn version(&self) -> IpVersion {
        self.info.version
    }

    /// Returns the packet information.
    #[must_use]
    pub fn info(&self) -> &PacketInfo {
        &self.info
    }

    /// Returns the raw packet data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        self.data
    }

    /// Returns the source IPv4 address if this is an IPv4 packet.
    #[must_use]
    pub fn src_ipv4(&self) -> Option<Ipv4Addr> {
        if self.info.version == IpVersion::V4 && self.info.src_addr.len() == 4 {
            Some(Ipv4Addr::new(
                self.info.src_addr[0],
                self.info.src_addr[1],
                self.info.src_addr[2],
                self.info.src_addr[3],
            ))
        } else {
            None
        }
    }

    /// Returns the destination IPv4 address if this is an IPv4 packet.
    #[must_use]
    pub fn dst_ipv4(&self) -> Option<Ipv4Addr> {
        if self.info.version == IpVersion::V4 && self.info.dst_addr.len() == 4 {
            Some(Ipv4Addr::new(
                self.info.dst_addr[0],
                self.info.dst_addr[1],
                self.info.dst_addr[2],
                self.info.dst_addr[3],
            ))
        } else {
            None
        }
    }

    /// Returns the source IPv6 address if this is an IPv6 packet.
    #[must_use]
    pub fn src_ipv6(&self) -> Option<Ipv6Addr> {
        if self.info.version == IpVersion::V6 && self.info.src_addr.len() == 16 {
            let bytes: [u8; 16] = self.info.src_addr.clone().try_into().ok()?;
            Some(Ipv6Addr::from(bytes))
        } else {
            None
        }
    }

    /// Returns the destination IPv6 address if this is an IPv6 packet.
    #[must_use]
    pub fn dst_ipv6(&self) -> Option<Ipv6Addr> {
        if self.info.version == IpVersion::V6 && self.info.dst_addr.len() == 16 {
            let bytes: [u8; 16] = self.info.dst_addr.clone().try_into().ok()?;
            Some(Ipv6Addr::from(bytes))
        } else {
            None
        }
    }

    /// Returns the IP protocol number.
    #[must_use]
    pub fn protocol(&self) -> u8 {
        self.info.protocol
    }

    /// Returns the total packet length.
    #[must_use]
    pub fn total_length(&self) -> u16 {
        self.info.total_length
    }
}

/// Determines the IP version from the first byte of a packet.
#[must_use]
pub fn get_ip_version(data: &[u8]) -> Option<IpVersion> {
    if data.is_empty() {
        return None;
    }

    match (data[0] >> 4) & 0x0F {
        4 => Some(IpVersion::V4),
        6 => Some(IpVersion::V6),
        _ => None,
    }
}

/// Extracts the destination IPv4 address from a packet without full parsing.
#[must_use]
pub fn get_dst_ipv4(data: &[u8]) -> Option<Ipv4Addr> {
    if data.len() < 20 {
        return None;
    }

    let version = (data[0] >> 4) & 0x0F;
    if version != 4 {
        return None;
    }

    Some(Ipv4Addr::new(data[16], data[17], data[18], data[19]))
}

/// Extracts the source IPv4 address from a packet without full parsing.
#[must_use]
pub fn get_src_ipv4(data: &[u8]) -> Option<Ipv4Addr> {
    if data.len() < 20 {
        return None;
    }

    let version = (data[0] >> 4) & 0x0F;
    if version != 4 {
        return None;
    }

    Some(Ipv4Addr::new(data[12], data[13], data[14], data[15]))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Sample IPv4 packet (ICMP echo request)
    fn sample_ipv4_packet() -> Vec<u8> {
        vec![
            0x45, 0x00, 0x00, 0x54, // Version, IHL, TOS, Total Length
            0x00, 0x00, 0x40, 0x00, // ID, Flags, Fragment Offset
            0x40, 0x01, 0x00, 0x00, // TTL, Protocol (ICMP=1), Checksum
            0x0a, 0x2d, 0x00, 0x02, // Source IP: 10.45.0.2
            0x08, 0x08, 0x08, 0x08, // Dest IP: 8.8.8.8
            // ... payload would follow
        ]
    }

    // Sample IPv6 packet
    fn sample_ipv6_packet() -> Vec<u8> {
        vec![
            0x60, 0x00, 0x00, 0x00, // Version, Traffic Class, Flow Label
            0x00, 0x10, 0x3a, 0x40, // Payload Length, Next Header (ICMPv6=58), Hop Limit
            // Source address (16 bytes)
            0x20, 0x01, 0x0d, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x01, // Destination address (16 bytes)
            0x20, 0x01, 0x0d, 0xb8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x02,
        ]
    }

    #[test]
    fn test_parse_ipv4() {
        let data = sample_ipv4_packet();
        let packet = IpPacket::parse(&data).expect("should parse IPv4 packet");

        assert_eq!(packet.version(), IpVersion::V4);
        assert_eq!(packet.protocol(), 1); // ICMP
        assert_eq!(packet.total_length(), 84);
        assert_eq!(packet.src_ipv4(), Some(Ipv4Addr::new(10, 45, 0, 2)));
        assert_eq!(packet.dst_ipv4(), Some(Ipv4Addr::new(8, 8, 8, 8)));
    }

    #[test]
    fn test_parse_ipv6() {
        let data = sample_ipv6_packet();
        let packet = IpPacket::parse(&data).expect("should parse IPv6 packet");

        assert_eq!(packet.version(), IpVersion::V6);
        assert_eq!(packet.protocol(), 58); // ICMPv6
        assert_eq!(packet.total_length(), 40 + 16);
    }

    #[test]
    fn test_parse_invalid() {
        // Empty packet
        assert!(IpPacket::parse(&[]).is_none());

        // Too short for IPv4
        assert!(IpPacket::parse(&[0x45, 0x00]).is_none());

        // Invalid version
        assert!(IpPacket::parse(&[0x30, 0x00, 0x00, 0x00]).is_none());
    }

    #[test]
    fn test_get_ip_version() {
        assert_eq!(get_ip_version(&[0x45]), Some(IpVersion::V4));
        assert_eq!(get_ip_version(&[0x60]), Some(IpVersion::V6));
        assert_eq!(get_ip_version(&[0x30]), None);
        assert_eq!(get_ip_version(&[]), None);
    }

    #[test]
    fn test_get_dst_ipv4() {
        let data = sample_ipv4_packet();
        assert_eq!(get_dst_ipv4(&data), Some(Ipv4Addr::new(8, 8, 8, 8)));
    }

    #[test]
    fn test_get_src_ipv4() {
        let data = sample_ipv4_packet();
        assert_eq!(get_src_ipv4(&data), Some(Ipv4Addr::new(10, 45, 0, 2)));
    }
}
