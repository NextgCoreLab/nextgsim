//! Security Mode Messages (3GPP TS 24.501 Section 8.2.25-8.2.27)
//!
//! This module implements the Security Mode procedure messages:
//! - Security Mode Command (network to UE)
//! - Security Mode Complete (UE to network)
//! - Security Mode Reject (UE to network)

use bytes::{Buf, BufMut};
use thiserror::Error;

use crate::enums::MmMessageType;
use crate::header::PlainMmHeader;
use crate::ies::ie1::{IeImeiSvRequest, InformationElement1};
use crate::messages::mm::registration::{Ie5gMmCause, Ie5gsMobileIdentity, MmCause};
use crate::security::NasKeySetIdentifier;

/// Error type for Security Mode message encoding/decoding
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SecurityModeError {
    /// Buffer too short for decoding
    #[error("Buffer too short: expected at least {expected} bytes, got {actual}")]
    BufferTooShort {
        /// Expected minimum bytes
        expected: usize,
        /// Actual bytes available
        actual: usize,
    },
    /// Invalid message type
    #[error("Invalid message type: expected {expected:?}, got {actual:?}")]
    InvalidMessageType {
        /// Expected message type
        expected: MmMessageType,
        /// Actual message type
        actual: MmMessageType,
    },
    /// Invalid IE value
    #[error("Invalid IE value: {0}")]
    InvalidIeValue(String),
    /// Unknown IEI
    #[error("Unknown IEI: 0x{0:02X}")]
    UnknownIei(u8),
}


// ============================================================================
// IEI values for Security Mode messages
// ============================================================================

/// IEI values for Security Mode Command optional IEs
#[allow(dead_code)]
mod security_mode_command_iei {
    /// IMEISV request (Type 1, IEI 0xE)
    pub const IMEISV_REQUEST: u8 = 0xE;
    /// Selected EPS NAS security algorithms (Type 3, IEI 0x57)
    pub const SELECTED_EPS_NAS_SECURITY_ALGORITHMS: u8 = 0x57;
    /// Additional 5G security information (Type 4, IEI 0x36)
    pub const ADDITIONAL_5G_SECURITY_INFORMATION: u8 = 0x36;
    /// EAP message (Type 6, IEI 0x78)
    pub const EAP_MESSAGE: u8 = 0x78;
    /// ABBA (Type 4, IEI 0x38)
    pub const ABBA: u8 = 0x38;
    /// Replayed S1 UE security capabilities (Type 4, IEI 0x19)
    pub const REPLAYED_S1_UE_SECURITY_CAPABILITIES: u8 = 0x19;
}

/// IEI values for Security Mode Complete optional IEs
#[allow(dead_code)]
mod security_mode_complete_iei {
    /// IMEISV (Type 6, IEI 0x77)
    pub const IMEISV: u8 = 0x77;
    /// NAS message container (Type 6, IEI 0x71)
    pub const NAS_MESSAGE_CONTAINER: u8 = 0x71;
    /// Non-IMEISV PEI (Type 6, IEI 0x78)
    pub const NON_IMEISV_PEI: u8 = 0x78;
}

// ============================================================================
// UE Security Capability IE (Type 4)
// ============================================================================

/// UE Security Capability IE (3GPP TS 24.501 Section 9.11.3.54)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IeUeSecurityCapability {
    /// 5G-EA0 supported
    pub ea0: bool,
    /// 128-5G-EA1 supported
    pub ea1_128: bool,
    /// 128-5G-EA2 supported
    pub ea2_128: bool,
    /// 128-5G-EA3 supported
    pub ea3_128: bool,
    /// 5G-EA4 supported
    pub ea4: bool,
    /// 5G-EA5 supported
    pub ea5: bool,
    /// 5G-EA6 supported
    pub ea6: bool,
    /// 5G-EA7 supported
    pub ea7: bool,
    /// 5G-IA0 supported
    pub ia0: bool,
    /// 128-5G-IA1 supported
    pub ia1_128: bool,
    /// 128-5G-IA2 supported
    pub ia2_128: bool,
    /// 128-5G-IA3 supported
    pub ia3_128: bool,
    /// 5G-IA4 supported
    pub ia4: bool,
    /// 5G-IA5 supported
    pub ia5: bool,
    /// 5G-IA6 supported
    pub ia6: bool,
    /// 5G-IA7 supported
    pub ia7: bool,
    /// EPS encryption algorithms (optional, octets 5-6)
    pub eps_ea: Option<u8>,
    /// EPS integrity algorithms (optional, octets 5-6)
    pub eps_ia: Option<u8>,
}

impl IeUeSecurityCapability {
    /// Create a new UE Security Capability IE with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode from bytes (without IEI, with length)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, SecurityModeError> {
        if buf.remaining() < 1 {
            return Err(SecurityModeError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let length = buf.get_u8() as usize;
        if buf.remaining() < length || length < 2 {
            return Err(SecurityModeError::BufferTooShort {
                expected: length.max(2),
                actual: buf.remaining(),
            });
        }

        // Octet 3: 5G-EA algorithms
        let ea_octet = buf.get_u8();
        // Octet 4: 5G-IA algorithms
        let ia_octet = buf.get_u8();

        let mut cap = Self {
            ea0: (ea_octet >> 7) & 0x01 == 1,
            ea1_128: (ea_octet >> 6) & 0x01 == 1,
            ea2_128: (ea_octet >> 5) & 0x01 == 1,
            ea3_128: (ea_octet >> 4) & 0x01 == 1,
            ea4: (ea_octet >> 3) & 0x01 == 1,
            ea5: (ea_octet >> 2) & 0x01 == 1,
            ea6: (ea_octet >> 1) & 0x01 == 1,
            ea7: ea_octet & 0x01 == 1,
            ia0: (ia_octet >> 7) & 0x01 == 1,
            ia1_128: (ia_octet >> 6) & 0x01 == 1,
            ia2_128: (ia_octet >> 5) & 0x01 == 1,
            ia3_128: (ia_octet >> 4) & 0x01 == 1,
            ia4: (ia_octet >> 3) & 0x01 == 1,
            ia5: (ia_octet >> 2) & 0x01 == 1,
            ia6: (ia_octet >> 1) & 0x01 == 1,
            ia7: ia_octet & 0x01 == 1,
            eps_ea: None,
            eps_ia: None,
        };

        // Optional EPS algorithms (octets 5-6)
        if length >= 4 {
            cap.eps_ea = Some(buf.get_u8());
            cap.eps_ia = Some(buf.get_u8());
            // Skip any remaining bytes
            if length > 4 {
                buf.advance(length - 4);
            }
        } else if length > 2 {
            buf.advance(length - 2);
        }

        Ok(cap)
    }

    /// Encode to bytes (without IEI, with length)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        let has_eps = self.eps_ea.is_some() && self.eps_ia.is_some();
        let length = if has_eps { 4 } else { 2 };
        buf.put_u8(length);

        // Octet 3: 5G-EA algorithms
        let ea_octet = ((self.ea0 as u8) << 7)
            | ((self.ea1_128 as u8) << 6)
            | ((self.ea2_128 as u8) << 5)
            | ((self.ea3_128 as u8) << 4)
            | ((self.ea4 as u8) << 3)
            | ((self.ea5 as u8) << 2)
            | ((self.ea6 as u8) << 1)
            | (self.ea7 as u8);
        buf.put_u8(ea_octet);

        // Octet 4: 5G-IA algorithms
        let ia_octet = ((self.ia0 as u8) << 7)
            | ((self.ia1_128 as u8) << 6)
            | ((self.ia2_128 as u8) << 5)
            | ((self.ia3_128 as u8) << 4)
            | ((self.ia4 as u8) << 3)
            | ((self.ia5 as u8) << 2)
            | ((self.ia6 as u8) << 1)
            | (self.ia7 as u8);
        buf.put_u8(ia_octet);

        // Optional EPS algorithms
        if let (Some(eps_ea), Some(eps_ia)) = (self.eps_ea, self.eps_ia) {
            buf.put_u8(eps_ea);
            buf.put_u8(eps_ia);
        }
    }

    /// Get encoded length (including length field)
    pub fn encoded_len(&self) -> usize {
        let has_eps = self.eps_ea.is_some() && self.eps_ia.is_some();
        if has_eps { 5 } else { 3 }
    }
}


// ============================================================================
// NAS Security Algorithms IE (Type 3)
// ============================================================================

/// Selected NAS Security Algorithms IE (3GPP TS 24.501 Section 9.11.3.34)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IeNasSecurityAlgorithms {
    /// Type of ciphering algorithm (5G-EA0 to 5G-EA7)
    pub ciphering: u8,
    /// Type of integrity protection algorithm (5G-IA0 to 5G-IA7)
    pub integrity: u8,
}

impl IeNasSecurityAlgorithms {
    /// Create a new NAS Security Algorithms IE
    pub fn new(ciphering: u8, integrity: u8) -> Self {
        Self {
            ciphering: ciphering & 0x0F,
            integrity: integrity & 0x0F,
        }
    }

    /// Decode from a single octet
    pub fn decode(value: u8) -> Self {
        Self {
            ciphering: (value >> 4) & 0x0F,
            integrity: value & 0x0F,
        }
    }

    /// Encode to a single octet
    pub fn encode(&self) -> u8 {
        ((self.ciphering & 0x0F) << 4) | (self.integrity & 0x0F)
    }
}

// ============================================================================
// Security Mode Command Message (3GPP TS 24.501 Section 8.2.25)
// ============================================================================

/// Security Mode Command message (network to UE)
///
/// 3GPP TS 24.501 Section 8.2.25
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityModeCommand {
    /// Selected NAS security algorithms (mandatory, Type 3)
    pub selected_nas_security_algorithms: IeNasSecurityAlgorithms,
    /// ngKSI - NAS key set identifier (mandatory, Type 1)
    pub ng_ksi: NasKeySetIdentifier,
    /// Replayed UE security capabilities (mandatory, Type 4)
    pub replayed_ue_security_capabilities: IeUeSecurityCapability,
    /// IMEISV request (optional, Type 1, IEI 0xE)
    pub imeisv_request: Option<IeImeiSvRequest>,
    /// Selected EPS NAS security algorithms (optional, Type 3, IEI 0x57)
    pub selected_eps_nas_security_algorithms: Option<u8>,
    /// Additional 5G security information (optional, Type 4, IEI 0x36)
    pub additional_5g_security_information: Option<Vec<u8>>,
    /// EAP message (optional, Type 6, IEI 0x78)
    pub eap_message: Option<Vec<u8>>,
    /// ABBA (optional, Type 4, IEI 0x38)
    pub abba: Option<Vec<u8>>,
    /// Replayed S1 UE security capabilities (optional, Type 4, IEI 0x19)
    pub replayed_s1_ue_security_capabilities: Option<Vec<u8>>,
}

impl Default for SecurityModeCommand {
    fn default() -> Self {
        Self {
            selected_nas_security_algorithms: IeNasSecurityAlgorithms::default(),
            ng_ksi: NasKeySetIdentifier::no_key(),
            replayed_ue_security_capabilities: IeUeSecurityCapability::default(),
            imeisv_request: None,
            selected_eps_nas_security_algorithms: None,
            additional_5g_security_information: None,
            eap_message: None,
            abba: None,
            replayed_s1_ue_security_capabilities: None,
        }
    }
}

impl SecurityModeCommand {
    /// Create a new Security Mode Command with mandatory fields
    pub fn new(
        selected_nas_security_algorithms: IeNasSecurityAlgorithms,
        ng_ksi: NasKeySetIdentifier,
        replayed_ue_security_capabilities: IeUeSecurityCapability,
    ) -> Self {
        Self {
            selected_nas_security_algorithms,
            ng_ksi,
            replayed_ue_security_capabilities,
            ..Default::default()
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, SecurityModeError> {
        if buf.remaining() < 2 {
            return Err(SecurityModeError::BufferTooShort {
                expected: 2,
                actual: buf.remaining(),
            });
        }

        // Octet 4: Selected NAS security algorithms
        let alg_octet = buf.get_u8();
        let selected_nas_security_algorithms = IeNasSecurityAlgorithms::decode(alg_octet);

        // Octet 5: spare (high nibble) + ngKSI (low nibble)
        let ksi_octet = buf.get_u8();
        let ng_ksi = NasKeySetIdentifier::decode(ksi_octet & 0x0F)
            .map_err(|e| SecurityModeError::InvalidIeValue(e.to_string()))?;

        // Replayed UE security capabilities (mandatory, Type 4)
        let replayed_ue_security_capabilities = IeUeSecurityCapability::decode(buf)?;

        let mut msg = Self::new(
            selected_nas_security_algorithms,
            ng_ksi,
            replayed_ue_security_capabilities,
        );

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            // Check for Type 1 IEs (4-bit IEI in high nibble)
            let iei_high = (iei >> 4) & 0x0F;
            if iei_high == security_mode_command_iei::IMEISV_REQUEST {
                buf.advance(1);
                msg.imeisv_request = Some(
                    IeImeiSvRequest::decode(iei & 0x0F)
                        .map_err(|e| SecurityModeError::InvalidIeValue(e.to_string()))?,
                );
                continue;
            }

            // Full octet IEI
            match iei {
                security_mode_command_iei::SELECTED_EPS_NAS_SECURITY_ALGORITHMS => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    msg.selected_eps_nas_security_algorithms = Some(buf.get_u8());
                }
                security_mode_command_iei::ADDITIONAL_5G_SECURITY_INFORMATION => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.additional_5g_security_information = Some(data);
                }
                security_mode_command_iei::EAP_MESSAGE => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.eap_message = Some(data);
                }
                security_mode_command_iei::ABBA => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.abba = Some(data);
                }
                security_mode_command_iei::REPLAYED_S1_UE_SECURITY_CAPABILITIES => {
                    buf.advance(1);
                    if buf.remaining() < 1 {
                        break;
                    }
                    let len = buf.get_u8() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.replayed_s1_ue_security_capabilities = Some(data);
                }
                _ => {
                    // Skip unknown IEs
                    buf.advance(1);
                    if buf.remaining() > 0 {
                        let len = buf.get_u8() as usize;
                        if buf.remaining() >= len {
                            buf.advance(len);
                        }
                    }
                }
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        let header = PlainMmHeader::new(MmMessageType::SecurityModeCommand);
        header.encode(buf);

        // Selected NAS security algorithms (mandatory)
        buf.put_u8(self.selected_nas_security_algorithms.encode());

        // ngKSI (mandatory, spare in high nibble)
        buf.put_u8(self.ng_ksi.encode() & 0x0F);

        // Replayed UE security capabilities (mandatory)
        self.replayed_ue_security_capabilities.encode(buf);

        // Optional IEs
        if let Some(ref imeisv_req) = self.imeisv_request {
            buf.put_u8((security_mode_command_iei::IMEISV_REQUEST << 4) | (imeisv_req.encode() & 0x0F));
        }

        if let Some(eps_alg) = self.selected_eps_nas_security_algorithms {
            buf.put_u8(security_mode_command_iei::SELECTED_EPS_NAS_SECURITY_ALGORITHMS);
            buf.put_u8(eps_alg);
        }

        if let Some(ref info) = self.additional_5g_security_information {
            buf.put_u8(security_mode_command_iei::ADDITIONAL_5G_SECURITY_INFORMATION);
            buf.put_u8(info.len() as u8);
            buf.put_slice(info);
        }

        if let Some(ref eap) = self.eap_message {
            buf.put_u8(security_mode_command_iei::EAP_MESSAGE);
            buf.put_u16(eap.len() as u16);
            buf.put_slice(eap);
        }

        if let Some(ref abba) = self.abba {
            buf.put_u8(security_mode_command_iei::ABBA);
            buf.put_u8(abba.len() as u8);
            buf.put_slice(abba);
        }

        if let Some(ref s1_cap) = self.replayed_s1_ue_security_capabilities {
            buf.put_u8(security_mode_command_iei::REPLAYED_S1_UE_SECURITY_CAPABILITIES);
            buf.put_u8(s1_cap.len() as u8);
            buf.put_slice(s1_cap);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::SecurityModeCommand
    }
}

// ============================================================================
// Security Mode Complete Message (3GPP TS 24.501 Section 8.2.26)
// ============================================================================

/// Security Mode Complete message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.26
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SecurityModeComplete {
    /// IMEISV (optional, Type 6, IEI 0x77)
    pub imeisv: Option<Ie5gsMobileIdentity>,
    /// NAS message container (optional, Type 6, IEI 0x71)
    pub nas_message_container: Option<Vec<u8>>,
    /// Non-IMEISV PEI (optional, Type 6, IEI 0x78)
    pub non_imeisv_pei: Option<Ie5gsMobileIdentity>,
}

impl SecurityModeComplete {
    /// Create a new Security Mode Complete
    pub fn new() -> Self {
        Self::default()
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, SecurityModeError> {
        let mut msg = Self::new();

        // Parse optional IEs
        while buf.remaining() > 0 {
            let iei = buf.chunk()[0];

            match iei {
                security_mode_complete_iei::IMEISV => {
                    buf.advance(1);
                    msg.imeisv = Some(
                        Ie5gsMobileIdentity::decode(buf)
                            .map_err(|e| SecurityModeError::InvalidIeValue(e.to_string()))?,
                    );
                }
                security_mode_complete_iei::NAS_MESSAGE_CONTAINER => {
                    buf.advance(1);
                    if buf.remaining() < 2 {
                        break;
                    }
                    let len = buf.get_u16() as usize;
                    if buf.remaining() < len {
                        break;
                    }
                    let mut data = vec![0u8; len];
                    buf.copy_to_slice(&mut data);
                    msg.nas_message_container = Some(data);
                }
                security_mode_complete_iei::NON_IMEISV_PEI => {
                    buf.advance(1);
                    msg.non_imeisv_pei = Some(
                        Ie5gsMobileIdentity::decode(buf)
                            .map_err(|e| SecurityModeError::InvalidIeValue(e.to_string()))?,
                    );
                }
                _ => {
                    // Skip unknown IEs
                    buf.advance(1);
                    if buf.remaining() > 0 {
                        let len = buf.get_u8() as usize;
                        if buf.remaining() >= len {
                            buf.advance(len);
                        }
                    }
                }
            }
        }

        Ok(msg)
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        let header = PlainMmHeader::new(MmMessageType::SecurityModeComplete);
        header.encode(buf);

        // Optional IEs
        if let Some(ref imeisv) = self.imeisv {
            buf.put_u8(security_mode_complete_iei::IMEISV);
            imeisv.encode(buf);
        }

        if let Some(ref container) = self.nas_message_container {
            buf.put_u8(security_mode_complete_iei::NAS_MESSAGE_CONTAINER);
            buf.put_u16(container.len() as u16);
            buf.put_slice(container);
        }

        if let Some(ref pei) = self.non_imeisv_pei {
            buf.put_u8(security_mode_complete_iei::NON_IMEISV_PEI);
            pei.encode(buf);
        }
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::SecurityModeComplete
    }
}

// ============================================================================
// Security Mode Reject Message (3GPP TS 24.501 Section 8.2.27)
// ============================================================================

/// Security Mode Reject message (UE to network)
///
/// 3GPP TS 24.501 Section 8.2.27
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecurityModeReject {
    /// 5GMM cause (mandatory, Type 3)
    pub mm_cause: Ie5gMmCause,
}

impl Default for SecurityModeReject {
    fn default() -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(MmCause::ProtocolErrorUnspecified),
        }
    }
}

impl SecurityModeReject {
    /// Create a new Security Mode Reject with the specified cause
    pub fn new(cause: MmCause) -> Self {
        Self {
            mm_cause: Ie5gMmCause::new(cause),
        }
    }

    /// Decode from bytes (after header has been parsed)
    pub fn decode<B: Buf>(buf: &mut B) -> Result<Self, SecurityModeError> {
        if buf.remaining() < 1 {
            return Err(SecurityModeError::BufferTooShort {
                expected: 1,
                actual: buf.remaining(),
            });
        }

        let mm_cause = Ie5gMmCause::decode(buf)
            .map_err(|e| SecurityModeError::InvalidIeValue(e.to_string()))?;

        Ok(Self { mm_cause })
    }

    /// Encode to bytes (including header)
    pub fn encode<B: BufMut>(&self, buf: &mut B) {
        // Header
        let header = PlainMmHeader::new(MmMessageType::SecurityModeReject);
        header.encode(buf);

        // 5GMM cause (mandatory)
        self.mm_cause.encode(buf);
    }

    /// Get the message type
    pub fn message_type() -> MmMessageType {
        MmMessageType::SecurityModeReject
    }
}


// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ies::ie1::ImeiSvRequest;

    #[test]
    fn test_ue_security_capability_encode_decode() {
        let mut cap = IeUeSecurityCapability::new();
        cap.ea0 = true;
        cap.ea1_128 = true;
        cap.ea2_128 = true;
        cap.ia0 = true;
        cap.ia1_128 = true;
        cap.ia2_128 = true;

        let mut buf = Vec::new();
        cap.encode(&mut buf);

        let decoded = IeUeSecurityCapability::decode(&mut buf.as_slice()).unwrap();
        assert_eq!(decoded.ea0, true);
        assert_eq!(decoded.ea1_128, true);
        assert_eq!(decoded.ea2_128, true);
        assert_eq!(decoded.ia0, true);
        assert_eq!(decoded.ia1_128, true);
        assert_eq!(decoded.ia2_128, true);
    }

    #[test]
    fn test_nas_security_algorithms_encode_decode() {
        let alg = IeNasSecurityAlgorithms::new(0x02, 0x01); // EA2, IA1
        let encoded = alg.encode();
        assert_eq!(encoded, 0x21);

        let decoded = IeNasSecurityAlgorithms::decode(encoded);
        assert_eq!(decoded.ciphering, 0x02);
        assert_eq!(decoded.integrity, 0x01);
    }

    #[test]
    fn test_security_mode_command_encode_decode() {
        let alg = IeNasSecurityAlgorithms::new(0x02, 0x02);
        let ng_ksi = NasKeySetIdentifier::no_key();
        let mut cap = IeUeSecurityCapability::new();
        cap.ea0 = true;
        cap.ia0 = true;

        let msg = SecurityModeCommand::new(alg, ng_ksi, cap.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded = SecurityModeCommand::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.selected_nas_security_algorithms, alg);
        assert_eq!(decoded.ng_ksi, ng_ksi);
        assert_eq!(decoded.replayed_ue_security_capabilities.ea0, cap.ea0);
        assert_eq!(decoded.replayed_ue_security_capabilities.ia0, cap.ia0);
    }

    #[test]
    fn test_security_mode_command_with_optional_ies() {
        let alg = IeNasSecurityAlgorithms::new(0x01, 0x01);
        let ng_ksi = NasKeySetIdentifier::no_key();
        let cap = IeUeSecurityCapability::new();

        let mut msg = SecurityModeCommand::new(alg, ng_ksi, cap);
        msg.imeisv_request = Some(IeImeiSvRequest::new(ImeiSvRequest::Requested));
        msg.abba = Some(vec![0x00, 0x00]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded = SecurityModeCommand::decode(&mut &buf[3..]).unwrap();

        assert!(decoded.imeisv_request.is_some());
        assert!(decoded.abba.is_some());
        assert_eq!(decoded.abba.unwrap(), vec![0x00, 0x00]);
    }

    #[test]
    fn test_security_mode_complete_encode_decode() {
        let msg = SecurityModeComplete::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Should be header only (3 bytes)
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x5E); // Message type

        // Skip header for decoding
        let decoded = SecurityModeComplete::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn test_security_mode_complete_with_nas_container() {
        let mut msg = SecurityModeComplete::new();
        msg.nas_message_container = Some(vec![0x7E, 0x00, 0x41, 0x01, 0x02]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Skip header (3 bytes) for decoding
        let decoded = SecurityModeComplete::decode(&mut &buf[3..]).unwrap();

        assert!(decoded.nas_message_container.is_some());
        assert_eq!(
            decoded.nas_message_container.unwrap(),
            vec![0x7E, 0x00, 0x41, 0x01, 0x02]
        );
    }

    #[test]
    fn test_security_mode_reject_encode_decode() {
        let msg = SecurityModeReject::new(MmCause::UeSecurityCapabilitiesMismatch);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Header (3 bytes) + cause (1 byte)
        assert_eq!(buf.len(), 4);
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Security header type
        assert_eq!(buf[2], 0x5F); // Message type
        assert_eq!(buf[3], 0x17); // UE security capabilities mismatch

        // Skip header for decoding
        let decoded = SecurityModeReject::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.mm_cause.value, MmCause::UeSecurityCapabilitiesMismatch);
    }

    #[test]
    fn test_message_types() {
        assert_eq!(
            SecurityModeCommand::message_type(),
            MmMessageType::SecurityModeCommand
        );
        assert_eq!(
            SecurityModeComplete::message_type(),
            MmMessageType::SecurityModeComplete
        );
        assert_eq!(
            SecurityModeReject::message_type(),
            MmMessageType::SecurityModeReject
        );
    }
}
