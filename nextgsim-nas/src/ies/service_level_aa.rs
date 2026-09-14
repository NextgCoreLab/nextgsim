//! Service-level-AA container (3GPP TS 24.501 §9.11.2.10)
//!
//! The container carries upper-layer authentication and authorization
//! information between the UE and the network. It appears in two places, and
//! this module is the **single** codec for both — the bytes have to agree or a
//! registration-path container and a NAS-transport one would drift apart:
//!
//! - as a type-6 IE (IEI `0x72`) in REGISTRATION REQUEST (§8.2.6), carrying the
//!   CAA-level UAV ID as the Service-level device ID; and
//! - as the *contents* of the Payload container IE in UL/DL NAS TRANSPORT when
//!   the Payload container type is `Service-level-AA container` (`0b1001`,
//!   §9.11.3.40) — the transport UUAA (TS 23.256 §5.2.2) uses.
//!
//! ## Container framing
//!
//! The contents are a sequence of *Service-level-AA parameters*, each one an IE
//! in its own right whose framing follows its TS 24.007 type (figures
//! 9.11.2.10.3, 9.11.2.10.4 and 9.11.2.10.6):
//!
//! | IEI    | Parameter                             | TS 24.007 type | Framing               |
//! |--------|---------------------------------------|----------------|-----------------------|
//! | `0x10` | Service-level device ID (§9.11.2.11)  | 4              | IEI + 1-octet len     |
//! | `0x20` | Server address (§9.11.2.12)           | 4              | IEI + 1-octet len     |
//! | `0x30` | Response (§9.11.2.14)                 | 4              | IEI + 1-octet len     |
//! | `0x40` | Payload type (§9.11.2.15)             | 4              | IEI + 1-octet len     |
//! | `0x50` | Service status indication (§9.11.2.18)| 4              | IEI + 1-octet len     |
//! | `0x70` | Payload (§9.11.2.13)                  | 6              | IEI + **2**-octet len |
//! | `0xA-` | Pending indication (§9.11.2.17)       | 1              | one octet, IEI in the high nibble |
//!
//! Per table 9.11.2.10.1 a receiver "shall ignore service-level-AA parameter
//! with type of service-level-AA parameter field containing an unknown IEI".
//! Skipping one requires guessing its framing, and this decoder applies the
//! TS 24.007 rule that an IEI with bit 8 set is a single-octet (type 1/2)
//! parameter and any other IEI carries a 1-octet length. That is right for
//! every parameter the table defines today except a *future* type-6 addition,
//! whose 2-octet length would be mis-skipped — recorded here rather than
//! silently assumed, because the alternative (abandoning the parse at the first
//! unknown IEI) loses parameters the spec says to keep reading past.

use std::fmt;

/// Service-level device ID parameter (TS 24.501 §9.11.2.10 table 9.11.2.10.1,
/// §9.11.2.11) — a type-4 parameter whose value is a UTF-8 string. On the
/// registration path this is the CAA-level UAV ID.
pub const SLAA_PARAM_SERVICE_LEVEL_DEVICE_ID: u8 = 0x10;

/// Service-level-AA server address parameter (§9.11.2.12), type 4.
pub const SLAA_PARAM_SERVER_ADDRESS: u8 = 0x20;

/// Service-level-AA response parameter (§9.11.2.14), type 4, 1-octet value.
pub const SLAA_PARAM_RESPONSE: u8 = 0x30;

/// Service-level-AA payload type parameter (§9.11.2.15), type 4, 1-octet value.
pub const SLAA_PARAM_PAYLOAD_TYPE: u8 = 0x40;

/// Service-level-AA service status indication parameter (§9.11.2.18), type 4,
/// 1-octet value.
pub const SLAA_PARAM_SERVICE_STATUS: u8 = 0x50;

/// Service-level-AA payload parameter (§9.11.2.13), type **6** — a 2-octet
/// length, because a UUAA payload can exceed 255 octets.
pub const SLAA_PARAM_PAYLOAD: u8 = 0x70;

/// Service-level-AA pending indication parameter (§9.11.2.17), type 1: the IEI
/// occupies the high nibble and the value the low one, so the whole parameter
/// is a single octet.
pub const SLAA_PARAM_PENDING_INDICATION_IEI: u8 = 0xA0;

/// Server address type octet values (§9.11.2.12 table 9.11.2.12.1). All other
/// values are spare and "shall be ignored" on receipt.
mod address_type {
    pub const IPV4: u8 = 0b0000_0001;
    pub const IPV6: u8 = 0b0000_0010;
    pub const IPV4V6: u8 = 0b0000_0011;
    pub const FQDN: u8 = 0b0000_0100;
}

/// The address of the service-level authentication and authorization server
/// (TS 24.501 §9.11.2.12).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ServiceLevelAaServerAddress {
    /// IPv4 address, octets 4-7 of the parameter value.
    Ipv4([u8; 4]),
    /// IPv6 address, octets 4-19.
    Ipv6([u8; 16]),
    /// IPv4 address followed by an IPv6 address, octets 4-7 then 8-23.
    Ipv4v6([u8; 4], [u8; 16]),
    /// FQDN, encoded per TS 23.003 §19.4.2.1 (length-prefixed labels), carried
    /// here as the raw label bytes so this codec neither imposes nor loses the
    /// label framing.
    Fqdn(Vec<u8>),
}

impl ServiceLevelAaServerAddress {
    /// Encode the parameter *value*: the address-type octet then the address.
    fn encode_value(&self) -> Vec<u8> {
        let mut out = Vec::new();
        match self {
            Self::Ipv4(v4) => {
                out.push(address_type::IPV4);
                out.extend_from_slice(v4);
            }
            Self::Ipv6(v6) => {
                out.push(address_type::IPV6);
                out.extend_from_slice(v6);
            }
            Self::Ipv4v6(v4, v6) => {
                out.push(address_type::IPV4V6);
                out.extend_from_slice(v4);
                out.extend_from_slice(v6);
            }
            Self::Fqdn(labels) => {
                out.push(address_type::FQDN);
                out.extend_from_slice(labels);
            }
        }
        out
    }

    /// Decode a parameter value. `None` for a spare address type or a value too
    /// short for the type it claims — both of which §9.11.2.12 says to ignore,
    /// and ignoring is safer than reading a truncated address as a real one.
    fn decode_value(value: &[u8]) -> Option<Self> {
        let (atype, addr) = value.split_first()?;
        match *atype {
            address_type::IPV4 if addr.len() >= 4 => Some(Self::Ipv4(addr[..4].try_into().ok()?)),
            address_type::IPV6 if addr.len() >= 16 => Some(Self::Ipv6(addr[..16].try_into().ok()?)),
            address_type::IPV4V6 if addr.len() >= 20 => Some(Self::Ipv4v6(
                addr[..4].try_into().ok()?,
                addr[4..20].try_into().ok()?,
            )),
            address_type::FQDN if !addr.is_empty() => Some(Self::Fqdn(addr.to_vec())),
            _ => None,
        }
    }
}

/// Type of payload carried in the Service-level-AA payload parameter
/// (TS 24.501 §9.11.2.15 table 9.11.2.15.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServiceLevelAaPayloadType {
    /// UUAA payload: an application-layer payload for the UUAA procedure
    /// between a UE supporting UAS services and the USS (NOTE 1).
    Uuaa,
    /// C2 authorization payload (NOTE 2).
    C2Authorization,
}

impl ServiceLevelAaPayloadType {
    /// The octet-3 value.
    pub fn value(self) -> u8 {
        match self {
            Self::Uuaa => 0b0000_0001,
            Self::C2Authorization => 0b0000_0010,
        }
    }

    /// Read the octet-3 value. `None` for a spare value, which §9.11.2.15 says
    /// the receiver shall ignore.
    pub fn from_value(value: u8) -> Option<Self> {
        match value {
            0b0000_0001 => Some(Self::Uuaa),
            0b0000_0010 => Some(Self::C2Authorization),
            _ => None,
        }
    }
}

/// A two-bit result field of the Service-level-AA response parameter
/// (TS 24.501 §9.11.2.14): the SLAR (bits 2-1) and the C2AR (bits 4-3) share
/// this shape.
///
/// ## Bit order, and why it is written down
///
/// The tables for SLAR and C2AR are the **only** two in TS 24.501 §9-§10 that
/// print their bit columns low-bit-first ("1 | 2" and "3 | 4"); the other 45-odd
/// multi-bit tables in the same document print MSB-first ("2 | 1", "4 | 3",
/// "6 | 5"). Read literally, "0 1 → successful" would mean bit 1 clear and
/// bit 2 set, i.e. field value `0b10`. Read by the document's dominant
/// convention it means field value `0b01`.
///
/// This codec takes the **MSB-first** reading — `Success` is `0b01`,
/// `NotSuccessfulOrRevoked` is `0b10` — because a two-bit result field written
/// `00/01/10/11` is a binary number everywhere else in the spec, and because
/// treating two adjacent tables as an ordering erratum is a smaller claim than
/// treating the whole document as one. The reading is a single pair of constants
/// here, so an interop capture that disagrees is a two-line correction rather
/// than a rewrite. Tracked for a human ruling as a `decision` issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ServiceLevelAaResult {
    /// No information (`0b00`).
    #[default]
    NoInformation,
    /// The authorization was successful (`0b01`).
    Successful,
    /// The authorization was not successful, or has been revoked (`0b10`).
    NotSuccessfulOrRevoked,
    /// Reserved (`0b11`).
    Reserved,
}

impl ServiceLevelAaResult {
    /// The two-bit field value.
    fn bits(self) -> u8 {
        match self {
            Self::NoInformation => 0b00,
            Self::Successful => 0b01,
            Self::NotSuccessfulOrRevoked => 0b10,
            Self::Reserved => 0b11,
        }
    }

    /// Read a two-bit field value. Total, so a `Reserved` peer value survives a
    /// round trip instead of being flattened into `NoInformation`.
    fn from_bits(bits: u8) -> Self {
        match bits & 0b11 {
            0b00 => Self::NoInformation,
            0b01 => Self::Successful,
            0b10 => Self::NotSuccessfulOrRevoked,
            _ => Self::Reserved,
        }
    }

    /// Whether this result authorizes the service. Only an explicit
    /// `Successful` does: `NoInformation` and `Reserved` are not authorizations,
    /// and defaulting them to "allowed" is how a UE ends up flying on a
    /// response the network never gave.
    pub fn is_authorized(self) -> bool {
        matches!(self, Self::Successful)
    }
}

/// Service-level-AA response parameter (TS 24.501 §9.11.2.14): the
/// service-level result (SLAR) and the C2 authorization result (C2AR).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ServiceLevelAaResponse {
    /// Service-level-AA result field (SLAR), octet 3 bits 2-1.
    pub slar: ServiceLevelAaResult,
    /// C2 authorization result field (C2AR), octet 3 bits 4-3.
    pub c2ar: ServiceLevelAaResult,
}

impl ServiceLevelAaResponse {
    /// Encode octet 3. Bits 5-8 are spare and coded as zero.
    fn encode_value(self) -> u8 {
        self.slar.bits() | (self.c2ar.bits() << 2)
    }

    /// Decode octet 3, ignoring the spare bits.
    fn decode_value(octet: u8) -> Self {
        Self {
            slar: ServiceLevelAaResult::from_bits(octet),
            c2ar: ServiceLevelAaResult::from_bits(octet >> 2),
        }
    }
}

/// A decoded Service-level-AA container (TS 24.501 §9.11.2.10).
///
/// Every parameter is optional because the container is a bag of parameters
/// rather than a fixed record: a registration carries only the device ID, a
/// UUAA request carries a payload type plus payload, and a UUAA result carries
/// a response.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ServiceLevelAaContainer {
    /// Service-level device ID (`0x10`) — the CAA-level UAV ID on the
    /// registration path.
    pub device_id: Option<String>,
    /// Service-level-AA server address (`0x20`).
    pub server_address: Option<ServiceLevelAaServerAddress>,
    /// Service-level-AA response (`0x30`).
    pub response: Option<ServiceLevelAaResponse>,
    /// Service-level-AA payload type (`0x40`). Always accompanied by
    /// [`Self::payload`] on the wire (§9.11.2.10 figure 9.11.2.10.5 NOTE).
    pub payload_type: Option<ServiceLevelAaPayloadType>,
    /// Service-level-AA payload (`0x70`), transparent to NAS.
    pub payload: Option<Vec<u8>>,
    /// Service-level-AA pending indication (`0xA-`): `true` when the
    /// service-level-AA procedure is to be performed (SLAPI = 1).
    pub pending_indication: Option<bool>,
    /// Service-level-AA service status indication (`0x50`): `true` when UAS
    /// services are enabled.
    pub uas_services_enabled: Option<bool>,
    /// Whether decoding stopped early because a parameter's length ran past the
    /// end of the container. The parameters read before that point are kept —
    /// §7.6 error handling keeps what parsed — but a caller that needs to know
    /// the container was malformed can see it here rather than inferring it from
    /// a missing field.
    pub truncated: bool,
}

impl ServiceLevelAaContainer {
    /// A container carrying just the Service-level device ID, which is the
    /// registration path's whole content.
    pub fn with_device_id(device_id: impl Into<String>) -> Self {
        Self {
            device_id: Some(device_id.into()),
            ..Self::default()
        }
    }

    /// A container carrying a UUAA payload: the payload type parameter and the
    /// payload itself, in the order figure 9.11.2.10.5 requires.
    pub fn with_uuaa_payload(payload: Vec<u8>) -> Self {
        Self {
            payload_type: Some(ServiceLevelAaPayloadType::Uuaa),
            payload: Some(payload),
            ..Self::default()
        }
    }

    /// Whether this container carries no parameter at all. An empty container is
    /// legal on the wire (contents are 0 octets) but says nothing, so a sender
    /// can check before spending a NAS message on it.
    pub fn is_empty(&self) -> bool {
        self.device_id.is_none()
            && self.server_address.is_none()
            && self.response.is_none()
            && self.payload_type.is_none()
            && self.payload.is_none()
            && self.pending_indication.is_none()
            && self.uas_services_enabled.is_none()
    }

    /// Encode the container *contents* (figure 9.11.2.10.2) — the bytes that
    /// follow the container IEI and its 2-octet length on the registration path,
    /// and the Payload container contents on the NAS-transport path.
    ///
    /// Parameters go out in ascending IEI order with one deliberate exception:
    /// the payload (`0x70`) is emitted directly after its payload type (`0x40`),
    /// because the NOTE under table 9.11.2.10.1 requires the pair to be
    /// adjacent and the service status indication (`0x50`) would otherwise sort
    /// between them.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();

        if let Some(ref id) = self.device_id {
            push_type4(&mut out, SLAA_PARAM_SERVICE_LEVEL_DEVICE_ID, id.as_bytes());
        }
        if let Some(ref addr) = self.server_address {
            push_type4(&mut out, SLAA_PARAM_SERVER_ADDRESS, &addr.encode_value());
        }
        if let Some(response) = self.response {
            push_type4(&mut out, SLAA_PARAM_RESPONSE, &[response.encode_value()]);
        }
        // The payload type and its payload are adjacent by construction: NOTE
        // under table 9.11.2.10.1 says "a service-level-AA payload type is
        // always followed by the associated service-level-AA payload", so the
        // status indication cannot be allowed to slot between them.
        if let Some(payload_type) = self.payload_type {
            push_type4(&mut out, SLAA_PARAM_PAYLOAD_TYPE, &[payload_type.value()]);
        }
        if let Some(ref payload) = self.payload {
            push_type6(&mut out, SLAA_PARAM_PAYLOAD, payload);
        }
        if let Some(enabled) = self.uas_services_enabled {
            push_type4(&mut out, SLAA_PARAM_SERVICE_STATUS, &[u8::from(enabled)]);
        }
        if let Some(pending) = self.pending_indication {
            out.push(SLAA_PARAM_PENDING_INDICATION_IEI | u8::from(pending));
        }

        out
    }

    /// Decode container contents. Unknown parameters are skipped per
    /// table 9.11.2.10.1; a length that runs past the end sets
    /// [`Self::truncated`] and stops the walk.
    pub fn decode(bytes: &[u8]) -> Self {
        let mut out = Self::default();
        let mut i = 0;

        while i < bytes.len() {
            let iei = bytes[i];

            // Type 1: the IEI is the high nibble and the value the low one, so
            // the parameter is this octet and nothing more (figure 9.11.2.10.6).
            if iei & 0x80 != 0 {
                if iei & 0xF0 == SLAA_PARAM_PENDING_INDICATION_IEI {
                    // SLAPI is bit 1; bits 2-4 are spare.
                    out.pending_indication = Some(iei & 0x01 != 0);
                }
                i += 1;
                continue;
            }

            let is_type6 = iei == SLAA_PARAM_PAYLOAD;
            let len_octets = if is_type6 { 2 } else { 1 };
            if i + 1 + len_octets > bytes.len() {
                out.truncated = true;
                break;
            }
            let plen = if is_type6 {
                usize::from(u16::from_be_bytes([bytes[i + 1], bytes[i + 2]]))
            } else {
                usize::from(bytes[i + 1])
            };
            let start = i + 1 + len_octets;
            let Some(end) = start.checked_add(plen).filter(|end| *end <= bytes.len()) else {
                out.truncated = true;
                break;
            };
            let value = &bytes[start..end];

            match iei {
                SLAA_PARAM_SERVICE_LEVEL_DEVICE_ID => {
                    // A device ID that is not UTF-8 is dropped rather than
                    // lossily converted: it identifies a UAV to a CAA, and a
                    // replacement character in it names some other aircraft.
                    out.device_id = String::from_utf8(value.to_vec()).ok();
                }
                SLAA_PARAM_SERVER_ADDRESS => {
                    out.server_address = ServiceLevelAaServerAddress::decode_value(value);
                }
                SLAA_PARAM_RESPONSE => {
                    if let Some(&octet) = value.first() {
                        out.response = Some(ServiceLevelAaResponse::decode_value(octet));
                    }
                }
                SLAA_PARAM_PAYLOAD_TYPE => {
                    out.payload_type = value
                        .first()
                        .copied()
                        .and_then(ServiceLevelAaPayloadType::from_value);
                }
                SLAA_PARAM_PAYLOAD => out.payload = Some(value.to_vec()),
                SLAA_PARAM_SERVICE_STATUS => {
                    // UAS is bit 1 of octet 3; bits 2-8 are spare.
                    out.uas_services_enabled = value.first().map(|octet| octet & 0x01 != 0);
                }
                _ => {} // Unknown IEI: ignored, per table 9.11.2.10.1.
            }

            i = end;
        }

        out
    }
}

impl fmt::Display for ServiceLevelAaContainer {
    /// A one-line summary for logs. The payload is summarised by length, never
    /// printed: it is opaque upper-layer authentication material.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SLAA[")?;
        if let Some(ref id) = self.device_id {
            write!(f, "device_id={id} ")?;
        }
        if let Some(response) = self.response {
            write!(f, "slar={:?} c2ar={:?} ", response.slar, response.c2ar)?;
        }
        if let Some(payload_type) = self.payload_type {
            write!(f, "payload_type={payload_type:?} ")?;
        }
        if let Some(ref payload) = self.payload {
            write!(f, "payload={}B ", payload.len())?;
        }
        if let Some(pending) = self.pending_indication {
            write!(f, "pending={pending} ")?;
        }
        if let Some(enabled) = self.uas_services_enabled {
            write!(f, "uas_enabled={enabled} ")?;
        }
        if self.truncated {
            write!(f, "TRUNCATED ")?;
        }
        write!(f, "]")
    }
}

/// Append a type-4 parameter: IEI, 1-octet length, value. A value longer than
/// 255 octets is truncated, because the length field cannot describe it and a
/// wrong length desynchronises every parameter after this one.
fn push_type4(out: &mut Vec<u8>, iei: u8, value: &[u8]) {
    let len = value.len().min(u8::MAX as usize);
    out.push(iei);
    out.push(len as u8);
    out.extend_from_slice(&value[..len]);
}

/// Append a type-6 parameter: IEI, 2-octet length, value.
fn push_type6(out: &mut Vec<u8>, iei: u8, value: &[u8]) {
    let len = value.len().min(u16::MAX as usize);
    out.push(iei);
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.extend_from_slice(&value[..len]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_id_only_container_matches_the_registration_wire_form() {
        // The form already on the wire for registration (WAVE-4 T4.3):
        // param-IEI 0x10, 1-octet length, UTF-8 value.
        let bytes = ServiceLevelAaContainer::with_device_id("FAA-N12345").encode();
        assert_eq!(
            bytes,
            vec![0x10, 0x0A, b'F', b'A', b'A', b'-', b'N', b'1', b'2', b'3', b'4', b'5']
        );
        assert_eq!(
            ServiceLevelAaContainer::decode(&bytes).device_id.as_deref(),
            Some("FAA-N12345")
        );
    }

    #[test]
    fn every_parameter_survives_an_encode_decode_round_trip() {
        let container = ServiceLevelAaContainer {
            device_id: Some("FAA-N12345".to_string()),
            server_address: Some(ServiceLevelAaServerAddress::Ipv4([192, 0, 2, 10])),
            response: Some(ServiceLevelAaResponse {
                slar: ServiceLevelAaResult::Successful,
                c2ar: ServiceLevelAaResult::NotSuccessfulOrRevoked,
            }),
            payload_type: Some(ServiceLevelAaPayloadType::Uuaa),
            payload: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            pending_indication: Some(true),
            uas_services_enabled: Some(true),
            truncated: false,
        };
        let decoded = ServiceLevelAaContainer::decode(&container.encode());
        assert_eq!(decoded, container);
    }

    #[test]
    fn a_payload_longer_than_255_octets_needs_the_type6_length() {
        // The whole reason the payload parameter is type 6: a 300-octet UUAA
        // payload cannot be described by a 1-octet length, and a codec that
        // used one would truncate it.
        let payload = vec![0x5A; 300];
        let bytes = ServiceLevelAaContainer::with_uuaa_payload(payload.clone()).encode();
        // 0x40 0x01 0x01 (payload type = UUAA), then 0x70 0x01 0x2C (300).
        assert_eq!(&bytes[..3], &[0x40, 0x01, 0x01]);
        assert_eq!(&bytes[3..6], &[0x70, 0x01, 0x2C]);
        assert_eq!(
            ServiceLevelAaContainer::decode(&bytes).payload.as_deref(),
            Some(payload.as_slice())
        );
    }

    #[test]
    fn payload_type_is_immediately_followed_by_its_payload() {
        // Table 9.11.2.10.1 NOTE. The service status indication (0x50) sorts
        // between 0x40 and 0x70 by IEI, so an encoder ordering purely by IEI
        // would split the pair.
        let bytes = ServiceLevelAaContainer {
            payload_type: Some(ServiceLevelAaPayloadType::Uuaa),
            payload: Some(vec![0x01]),
            uas_services_enabled: Some(true),
            ..Default::default()
        }
        .encode();
        let type_at = bytes.iter().position(|b| *b == SLAA_PARAM_PAYLOAD_TYPE);
        let payload_at = bytes.iter().position(|b| *b == SLAA_PARAM_PAYLOAD);
        assert_eq!(type_at, Some(0));
        assert_eq!(payload_at, Some(3), "payload must follow its type directly");
    }

    #[test]
    fn an_unknown_type4_parameter_is_skipped_and_later_ones_still_parse() {
        // Table 9.11.2.10.1: "shall ignore ... an unknown IEI".
        let bytes = [
            0x60, 0x02, 0xAA, 0xBB, // unknown type-4 parameter
            0x10, 0x03, b'X', b'Y', b'Z', // device ID after it
        ];
        let decoded = ServiceLevelAaContainer::decode(&bytes);
        assert_eq!(decoded.device_id.as_deref(), Some("XYZ"));
        assert!(!decoded.truncated);
    }

    #[test]
    fn an_unknown_type1_parameter_costs_exactly_one_octet() {
        // 0x80 has bit 8 set, so it is a type-1 parameter with no length octet.
        // Skipping it as if it had one would eat the device ID's IEI.
        let bytes = [0x80, 0x10, 0x03, b'X', b'Y', b'Z'];
        assert_eq!(
            ServiceLevelAaContainer::decode(&bytes).device_id.as_deref(),
            Some("XYZ")
        );
    }

    #[test]
    fn a_length_running_past_the_end_is_reported_and_earlier_parameters_kept() {
        let bytes = [
            0x10, 0x03, b'X', b'Y', b'Z', // device ID, complete
            0x30, 0x09, 0x00, // response claiming 9 octets, only 1 present
        ];
        let decoded = ServiceLevelAaContainer::decode(&bytes);
        assert_eq!(decoded.device_id.as_deref(), Some("XYZ"));
        assert!(decoded.truncated);
        assert!(decoded.response.is_none());
    }

    #[test]
    fn response_bits_place_slar_and_c2ar_in_their_own_fields() {
        // SLAR is bits 2-1 and C2AR bits 4-3, so a successful SLAR with no C2AR
        // information is 0b0000_0001 and the reverse is 0b0000_0100.
        let slar_only = ServiceLevelAaResponse {
            slar: ServiceLevelAaResult::Successful,
            c2ar: ServiceLevelAaResult::NoInformation,
        };
        assert_eq!(slar_only.encode_value(), 0b0000_0001);
        let c2ar_only = ServiceLevelAaResponse {
            slar: ServiceLevelAaResult::NoInformation,
            c2ar: ServiceLevelAaResult::Successful,
        };
        assert_eq!(c2ar_only.encode_value(), 0b0000_0100);
        // Spare bits 5-8 on receipt are ignored rather than read as a result.
        assert_eq!(ServiceLevelAaResponse::decode_value(0b1111_0001), slar_only);
    }

    #[test]
    fn only_an_explicit_success_authorizes() {
        assert!(ServiceLevelAaResult::Successful.is_authorized());
        assert!(!ServiceLevelAaResult::NoInformation.is_authorized());
        assert!(!ServiceLevelAaResult::NotSuccessfulOrRevoked.is_authorized());
        assert!(!ServiceLevelAaResult::Reserved.is_authorized());
    }

    #[test]
    fn each_server_address_type_round_trips_and_a_spare_type_is_ignored() {
        for addr in [
            ServiceLevelAaServerAddress::Ipv4([192, 0, 2, 1]),
            ServiceLevelAaServerAddress::Ipv6([0x20; 16]),
            ServiceLevelAaServerAddress::Ipv4v6([198, 51, 100, 7], [0x11; 16]),
            ServiceLevelAaServerAddress::Fqdn(b"\x03uss\x07example".to_vec()),
        ] {
            let container = ServiceLevelAaContainer {
                server_address: Some(addr.clone()),
                ..Default::default()
            };
            assert_eq!(
                ServiceLevelAaContainer::decode(&container.encode()).server_address,
                Some(addr)
            );
        }
        // Address type 0x09 is spare: ignored, not guessed at.
        assert!(ServiceLevelAaServerAddress::decode_value(&[0x09, 1, 2, 3, 4]).is_none());
        // An IPv6 address type with only 4 octets of address is not an IPv4.
        assert!(ServiceLevelAaServerAddress::decode_value(&[0x02, 1, 2, 3, 4]).is_none());
    }

    #[test]
    fn a_spare_payload_type_is_ignored_but_the_payload_is_still_read() {
        // §9.11.2.15: "the receiving entity shall ignore the service-level-AA
        // payload type value set to a spare value". The payload itself is not
        // spare and dropping it would lose the exchange.
        let bytes = [0x40, 0x01, 0x7F, 0x70, 0x00, 0x02, 0xAA, 0xBB];
        let decoded = ServiceLevelAaContainer::decode(&bytes);
        assert!(decoded.payload_type.is_none());
        assert_eq!(decoded.payload.as_deref(), Some(&[0xAA, 0xBB][..]));
    }

    #[test]
    fn a_non_utf8_device_id_is_dropped_rather_than_mangled() {
        let bytes = [0x10, 0x02, 0xFF, 0xFE];
        assert!(ServiceLevelAaContainer::decode(&bytes).device_id.is_none());
    }

    #[test]
    fn pending_indication_is_one_octet_with_the_iei_in_the_high_nibble() {
        let set = ServiceLevelAaContainer {
            pending_indication: Some(true),
            ..Default::default()
        };
        assert_eq!(set.encode(), vec![0xA1]);
        let clear = ServiceLevelAaContainer {
            pending_indication: Some(false),
            ..Default::default()
        };
        assert_eq!(clear.encode(), vec![0xA0]);
        assert_eq!(
            ServiceLevelAaContainer::decode(&[0xA1]).pending_indication,
            Some(true)
        );
        assert_eq!(
            ServiceLevelAaContainer::decode(&[0xA0]).pending_indication,
            Some(false)
        );
    }

    #[test]
    fn an_empty_container_is_legal_and_says_nothing() {
        let empty = ServiceLevelAaContainer::default();
        assert!(empty.is_empty());
        assert!(empty.encode().is_empty());
        assert!(ServiceLevelAaContainer::decode(&[]).is_empty());
        assert!(!ServiceLevelAaContainer::with_device_id("X").is_empty());
    }

    #[test]
    fn display_summarises_the_payload_by_length_and_never_prints_it() {
        let text = ServiceLevelAaContainer::with_uuaa_payload(vec![0xAB; 8]).to_string();
        assert!(text.contains("payload=8B"), "{text}");
        assert!(
            !text.contains("ab"),
            "opaque auth material must not be logged: {text}"
        );
    }
}
