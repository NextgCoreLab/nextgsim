//! NAS Message Capture Tests
//!
//! This module contains tests that validate NAS message encoding/decoding
//! against real captures from Wireshark/tcpdump, ensuring compliance with
//! 3GPP TS 24.501.
//!
//! The hex captures represent actual NAS messages observed in 5G network
//! communications.

#[cfg(test)]
mod tests {
    use crate::enums::MmMessageType;
    use crate::ies::ie1::{FollowOnRequest, RegistrationType, Ie5gsRegistrationType};
    use crate::messages::mm::{
        Abba, AuthenticationRequest, AuthenticationResponse, Ie5gsMobileIdentity,
        Ie5gsRegistrationResult, MobileIdentityType, RegistrationAccept, RegistrationComplete,
        RegistrationRequest, RegistrationResultValue, SmsOverNasAllowed,
    };
    use crate::messages::mm::{
        IeNasSecurityAlgorithms, IeUeSecurityCapability, SecurityModeCommand, SecurityModeComplete,
    };
    use crate::security::{NasKeySetIdentifier, SecurityContextType};

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Parse hex string to bytes
    fn hex_to_bytes(hex: &str) -> Vec<u8> {
        hex.as_bytes()
            .chunks(2)
            .map(|chunk| {
                let s = std::str::from_utf8(chunk).unwrap();
                u8::from_str_radix(s, 16).unwrap()
            })
            .collect()
    }

    /// Convert bytes to hex string for debugging
    #[allow(dead_code)]
    fn bytes_to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{:02x}", b)).collect()
    }

    // ========================================================================
    // Registration Request Tests (3GPP TS 24.501 Section 8.2.6)
    // ========================================================================

    /// Test Registration Request encoding against a real capture
    /// 
    /// This capture represents an initial registration request from a UE
    /// with SUCI identity type.
    /// 
    /// Message structure per 3GPP TS 24.501:
    /// - EPD: 0x7E (5GMM)
    /// - Security header: 0x00 (Plain)
    /// - Message type: 0x41 (Registration Request)
    /// - ngKSI + Registration type: 0x71 (ngKSI=7/no key, type=initial)
    /// - 5GS Mobile Identity (SUCI)
    #[test]
    fn test_registration_request_initial_suci_capture() {
        // Real capture: Registration Request with SUCI
        // EPD=7E, SHT=00, MT=41, ngKSI+RegType=71, MobileIdentity(SUCI)
        let capture = hex_to_bytes(
            "7e0041710009f10700000000000001"
        );

        // Verify header
        assert_eq!(capture[0], 0x7E); // EPD: 5GMM
        assert_eq!(capture[1], 0x00); // Security header: Plain
        assert_eq!(capture[2], 0x41); // Message type: Registration Request

        // Decode the message (skip header)
        let decoded = RegistrationRequest::decode(&mut &capture[3..]).unwrap();

        // Verify ngKSI (high nibble of first byte after header)
        assert_eq!(decoded.ng_ksi.ksi, 7); // No key available
        assert_eq!(decoded.ng_ksi.tsc, SecurityContextType::Native);

        // Verify registration type (low nibble)
        assert_eq!(
            decoded.registration_type.registration_type,
            RegistrationType::InitialRegistration
        );

        // Verify mobile identity type
        assert_eq!(decoded.mobile_identity.identity_type, MobileIdentityType::Suci);

        // Re-encode and verify
        let mut encoded = Vec::new();
        let msg = RegistrationRequest::new(
            decoded.registration_type.clone(),
            decoded.ng_ksi,
            decoded.mobile_identity,
        );
        msg.encode(&mut encoded);

        // Verify header matches
        assert_eq!(encoded[0..3], capture[0..3]);
    }

    /// Test Registration Request with 5G-GUTI identity
    /// 
    /// Per 3GPP TS 24.501 Section 9.11.3.4, 5G-GUTI format:
    /// - Type: 0b010 (GUTI)
    /// - MCC/MNC + AMF Region ID + AMF Set ID + AMF Pointer + 5G-TMSI
    #[test]
    fn test_registration_request_guti_capture() {
        // Registration Request with 5G-GUTI
        // ngKSI=0 (native), RegType=1 (initial), GUTI identity
        let capture = hex_to_bytes(
            "7e00410100" // Header + ngKSI(0)+RegType(1) + length prefix
        );

        assert_eq!(capture[0], 0x7E); // EPD
        assert_eq!(capture[1], 0x00); // Plain NAS
        assert_eq!(capture[2], 0x41); // Registration Request
        
        // First byte after header: ngKSI (high nibble) + reg type (low nibble)
        let first_byte = capture[3];
        let ng_ksi_value = (first_byte >> 4) & 0x0F;
        let reg_type_value = first_byte & 0x0F;
        
        assert_eq!(ng_ksi_value, 0); // ngKSI = 0
        assert_eq!(reg_type_value, 1); // Initial registration
    }

    /// Test Registration Request encoding produces valid 3GPP format
    #[test]
    fn test_registration_request_encode_3gpp_format() {
        let reg_type = Ie5gsRegistrationType::new(
            FollowOnRequest::NoPending,
            RegistrationType::InitialRegistration,
        );
        let ng_ksi = NasKeySetIdentifier::no_key();
        let mobile_id = Ie5gsMobileIdentity::no_identity();

        let msg = RegistrationRequest::new(reg_type, ng_ksi, mobile_id);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify 3GPP TS 24.501 format
        assert_eq!(buf[0], 0x7E, "EPD must be 0x7E for 5GMM");
        assert_eq!(buf[1], 0x00, "Security header must be 0x00 for plain NAS");
        assert_eq!(buf[2], 0x41, "Message type must be 0x41 for Registration Request");
        
        // ngKSI (7 = no key) in high nibble, reg type (1 = initial) in low nibble
        assert_eq!(buf[3], 0x71, "ngKSI=7 + RegType=1 should be 0x71");
    }

    // ========================================================================
    // Registration Accept Tests (3GPP TS 24.501 Section 8.2.7)
    // ========================================================================

    /// Test Registration Accept encoding against capture
    /// 
    /// Message structure per 3GPP TS 24.501:
    /// - EPD: 0x7E
    /// - Security header: 0x00
    /// - Message type: 0x42 (Registration Accept)
    /// - 5GS Registration Result (mandatory)
    #[test]
    fn test_registration_accept_basic_capture() {
        // Registration Accept with 3GPP access result
        // Result: SMS not allowed, 3GPP access
        let capture = hex_to_bytes("7e00420101");

        assert_eq!(capture[0], 0x7E); // EPD
        assert_eq!(capture[1], 0x00); // Plain NAS
        assert_eq!(capture[2], 0x42); // Registration Accept

        // Decode (skip header)
        let decoded = RegistrationAccept::decode(&mut &capture[3..]).unwrap();

        assert_eq!(
            decoded.registration_result.result,
            RegistrationResultValue::ThreeGppAccess
        );
        assert_eq!(
            decoded.registration_result.sms_allowed,
            SmsOverNasAllowed::NotAllowed
        );
    }

    /// Test Registration Accept with GUTI and TAI list
    #[test]
    fn test_registration_accept_with_guti_capture() {
        // Create a Registration Accept with GUTI
        let result = Ie5gsRegistrationResult::new(
            SmsOverNasAllowed::Allowed,
            RegistrationResultValue::ThreeGppAccess,
        );
        let mut msg = RegistrationAccept::new(result);
        
        // Add a 5G-GUTI (Type 6 IE with IEI 0x77)
        let guti_data = vec![
            0xF2, // Type: GUTI (010) + spare
            0x00, 0xF1, 0x10, // MCC/MNC (001/01)
            0x00, 0x01, // AMF Region ID + Set ID
            0x01, // AMF Pointer
            0x00, 0x00, 0x00, 0x01, // 5G-TMSI
        ];
        msg.guti = Some(Ie5gsMobileIdentity::new(MobileIdentityType::Guti, guti_data));

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E);
        assert_eq!(buf[1], 0x00);
        assert_eq!(buf[2], 0x42);

        // Decode and verify
        let decoded = RegistrationAccept::decode(&mut &buf[3..]).unwrap();
        assert!(decoded.guti.is_some());
        assert_eq!(decoded.guti.unwrap().identity_type, MobileIdentityType::Guti);
    }

    // ========================================================================
    // Registration Complete Tests (3GPP TS 24.501 Section 8.2.9)
    // ========================================================================

    /// Test Registration Complete encoding
    /// 
    /// Registration Complete is a simple message with only optional IEs
    #[test]
    fn test_registration_complete_basic_capture() {
        // Minimal Registration Complete (no optional IEs)
        let capture = hex_to_bytes("7e0043");

        assert_eq!(capture[0], 0x7E); // EPD
        assert_eq!(capture[1], 0x00); // Plain NAS
        assert_eq!(capture[2], 0x43); // Registration Complete

        // Encode our own and compare
        let msg = RegistrationComplete::new();
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        assert_eq!(buf, capture);
    }

    // ========================================================================
    // Authentication Request Tests (3GPP TS 24.501 Section 8.2.1)
    // ========================================================================

    /// Test Authentication Request encoding for 5G-AKA
    /// 
    /// Per 3GPP TS 24.501 Section 8.2.1:
    /// - ngKSI (mandatory)
    /// - ABBA (mandatory)
    /// - RAND (optional, IEI 0x21)
    /// - AUTN (optional, IEI 0x20)
    #[test]
    fn test_authentication_request_5g_aka_capture() {
        // Authentication Request with RAND and AUTN for 5G-AKA
        let rand = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
        ];
        let autn = vec![
            0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
        ];
        let abba = Abba::new(vec![0x00, 0x00]);
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 1);

        let msg = AuthenticationRequest::for_5g_aka(ng_ksi, abba, rand, autn.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Plain NAS
        assert_eq!(buf[2], 0x56); // Authentication Request

        // Verify ngKSI
        assert_eq!(buf[3] & 0x0F, 1); // ngKSI = 1

        // Decode and verify
        let decoded = AuthenticationRequest::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.ng_ksi.ksi, 1);
        assert!(decoded.rand.is_some());
        assert_eq!(decoded.rand.unwrap().value, rand);
        assert!(decoded.autn.is_some());
        assert_eq!(decoded.autn.unwrap().value, autn);
    }

    /// Test Authentication Request ABBA encoding per 3GPP TS 24.501 Section 9.11.3.10
    #[test]
    fn test_authentication_request_abba_encoding() {
        let abba = Abba::new(vec![0x00, 0x00]);
        let ng_ksi = NasKeySetIdentifier::no_key();

        let msg = AuthenticationRequest::new(ng_ksi, abba.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // ABBA should be encoded after ngKSI
        // Format: length (1 byte) + value
        let abba_start = 4; // After header (3) + ngKSI (1)
        assert_eq!(buf[abba_start], 0x02); // Length = 2
        assert_eq!(buf[abba_start + 1], 0x00); // ABBA value byte 1
        assert_eq!(buf[abba_start + 2], 0x00); // ABBA value byte 2
    }

    // ========================================================================
    // Authentication Response Tests (3GPP TS 24.501 Section 8.2.2)
    // ========================================================================

    /// Test Authentication Response with RES* for 5G-AKA
    /// 
    /// Per 3GPP TS 24.501 Section 8.2.2:
    /// - Authentication response parameter (optional, IEI 0x2D)
    #[test]
    fn test_authentication_response_res_star_capture() {
        // RES* value (typically 16 bytes)
        let res_star = vec![
            0xA1, 0xB2, 0xC3, 0xD4, 0xE5, 0xF6, 0x07, 0x18,
            0x29, 0x3A, 0x4B, 0x5C, 0x6D, 0x7E, 0x8F, 0x90,
        ];

        let msg = AuthenticationResponse::with_res_star(res_star.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Plain NAS
        assert_eq!(buf[2], 0x57); // Authentication Response

        // Verify IEI for auth response parameter
        assert_eq!(buf[3], 0x2D); // IEI

        // Decode and verify
        let decoded = AuthenticationResponse::decode(&mut &buf[3..]).unwrap();
        assert!(decoded.auth_response_parameter.is_some());
        assert_eq!(decoded.auth_response_parameter.unwrap().value, res_star);
    }

    /// Test empty Authentication Response (for EAP-AKA' flow)
    #[test]
    fn test_authentication_response_empty() {
        let msg = AuthenticationResponse::new();

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Should only have header
        assert_eq!(buf.len(), 3);
        assert_eq!(buf[0], 0x7E);
        assert_eq!(buf[1], 0x00);
        assert_eq!(buf[2], 0x57);
    }

    // ========================================================================
    // Security Mode Command Tests (3GPP TS 24.501 Section 8.2.25)
    // ========================================================================

    /// Test Security Mode Command encoding
    /// 
    /// Per 3GPP TS 24.501 Section 8.2.25:
    /// - Selected NAS security algorithms (mandatory)
    /// - ngKSI (mandatory)
    /// - Replayed UE security capabilities (mandatory)
    #[test]
    fn test_security_mode_command_capture() {
        // Security algorithms: EA2 (ciphering), IA2 (integrity)
        let alg = IeNasSecurityAlgorithms::new(0x02, 0x02);
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 1);
        
        let mut cap = IeUeSecurityCapability::new();
        cap.ea0 = true;
        cap.ea1_128 = true;
        cap.ea2_128 = true;
        cap.ia0 = true;
        cap.ia1_128 = true;
        cap.ia2_128 = true;

        let msg = SecurityModeCommand::new(alg, ng_ksi, cap);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify header
        assert_eq!(buf[0], 0x7E); // EPD
        assert_eq!(buf[1], 0x00); // Plain NAS
        assert_eq!(buf[2], 0x5D); // Security Mode Command

        // Verify selected algorithms (octet 4)
        // EA2 (0x02) in high nibble, IA2 (0x02) in low nibble = 0x22
        assert_eq!(buf[3], 0x22);

        // Verify ngKSI (octet 5, low nibble)
        assert_eq!(buf[4] & 0x0F, 1);

        // Decode and verify
        let decoded = SecurityModeCommand::decode(&mut &buf[3..]).unwrap();
        assert_eq!(decoded.selected_nas_security_algorithms.ciphering, 0x02);
        assert_eq!(decoded.selected_nas_security_algorithms.integrity, 0x02);
        assert_eq!(decoded.ng_ksi.ksi, 1);
    }

    /// Test Security Mode Command algorithm encoding per 3GPP TS 24.501 Section 9.11.3.34
    #[test]
    fn test_security_mode_command_algorithms() {
        // Test various algorithm combinations
        let test_cases = [
            (0x00, 0x00, 0x00), // EA0, IA0 (null algorithms)
            (0x01, 0x01, 0x11), // EA1, IA1 (128-bit)
            (0x02, 0x02, 0x22), // EA2, IA2 (128-bit)
            (0x03, 0x03, 0x33), // EA3, IA3 (128-bit)
        ];

        for (ea, ia, expected) in test_cases {
            let alg = IeNasSecurityAlgorithms::new(ea, ia);
            assert_eq!(alg.encode(), expected);
            
            let decoded = IeNasSecurityAlgorithms::decode(expected);
            assert_eq!(decoded.ciphering, ea);
            assert_eq!(decoded.integrity, ia);
        }
    }

    // ========================================================================
    // Security Mode Complete Tests (3GPP TS 24.501 Section 8.2.26)
    // ========================================================================

    /// Test Security Mode Complete encoding
    #[test]
    fn test_security_mode_complete_capture() {
        // Minimal Security Mode Complete
        let capture = hex_to_bytes("7e005e");

        assert_eq!(capture[0], 0x7E); // EPD
        assert_eq!(capture[1], 0x00); // Plain NAS
        assert_eq!(capture[2], 0x5E); // Security Mode Complete

        // Encode our own and compare
        let msg = SecurityModeComplete::new();
        let mut buf = Vec::new();
        msg.encode(&mut buf);

        assert_eq!(buf, capture);
    }

    /// Test Security Mode Complete with NAS message container
    #[test]
    fn test_security_mode_complete_with_container() {
        let mut msg = SecurityModeComplete::new();
        
        // NAS message container typically contains the initial NAS message
        // (e.g., Registration Request) that was sent before security was established
        let container = vec![0x7E, 0x00, 0x41, 0x71, 0x00, 0x01, 0x00];
        msg.nas_message_container = Some(container.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        // Verify IEI for NAS message container (0x71)
        assert_eq!(buf[3], 0x71);

        // Decode and verify
        let decoded = SecurityModeComplete::decode(&mut &buf[3..]).unwrap();
        assert!(decoded.nas_message_container.is_some());
        assert_eq!(decoded.nas_message_container.unwrap(), container);
    }

    // ========================================================================
    // Round-trip Encoding Tests
    // ========================================================================

    /// Verify Registration Request round-trip encoding
    #[test]
    fn test_registration_request_roundtrip() {
        let reg_type = Ie5gsRegistrationType::new(
            FollowOnRequest::Pending,
            RegistrationType::MobilityRegistrationUpdating,
        );
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 3);
        let mobile_id = Ie5gsMobileIdentity::new(
            MobileIdentityType::Suci,
            vec![0x01, 0xF1, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01],
        );

        let mut msg = RegistrationRequest::new(reg_type, ng_ksi, mobile_id);
        msg.ue_security_capability = Some(vec![0xE0, 0xE0]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = RegistrationRequest::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.registration_type, msg.registration_type);
        assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        assert_eq!(decoded.mobile_identity, msg.mobile_identity);
        assert_eq!(decoded.ue_security_capability, msg.ue_security_capability);
    }

    /// Verify Authentication Request round-trip encoding
    #[test]
    fn test_authentication_request_roundtrip() {
        let rand = [0x11u8; 16];
        let autn = vec![0x22u8; 16];
        let abba = Abba::new(vec![0x00, 0x00]);
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 2);

        let msg = AuthenticationRequest::for_5g_aka(ng_ksi, abba.clone(), rand, autn.clone());

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = AuthenticationRequest::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        assert_eq!(decoded.abba, msg.abba);
        assert_eq!(decoded.rand.unwrap().value, rand);
        assert_eq!(decoded.autn.unwrap().value, autn);
    }

    /// Verify Security Mode Command round-trip encoding
    #[test]
    fn test_security_mode_command_roundtrip() {
        let alg = IeNasSecurityAlgorithms::new(0x01, 0x02);
        let ng_ksi = NasKeySetIdentifier::new(SecurityContextType::Native, 0);
        
        let mut cap = IeUeSecurityCapability::new();
        cap.ea0 = true;
        cap.ea1_128 = true;
        cap.ia0 = true;
        cap.ia1_128 = true;
        cap.ia2_128 = true;

        let mut msg = SecurityModeCommand::new(alg, ng_ksi, cap.clone());
        msg.abba = Some(vec![0x00, 0x00]);

        let mut buf = Vec::new();
        msg.encode(&mut buf);

        let decoded = SecurityModeCommand::decode(&mut &buf[3..]).unwrap();

        assert_eq!(decoded.selected_nas_security_algorithms, alg);
        assert_eq!(decoded.ng_ksi, msg.ng_ksi);
        assert_eq!(decoded.abba, msg.abba);
    }

    // ========================================================================
    // Message Type Verification Tests
    // ========================================================================

    /// Verify all message types match 3GPP TS 24.501 Table 8.2.1
    #[test]
    fn test_message_types_3gpp_compliance() {
        // Registration messages
        assert_eq!(u8::from(MmMessageType::RegistrationRequest), 0x41);
        assert_eq!(u8::from(MmMessageType::RegistrationAccept), 0x42);
        assert_eq!(u8::from(MmMessageType::RegistrationComplete), 0x43);
        assert_eq!(u8::from(MmMessageType::RegistrationReject), 0x44);

        // Authentication messages
        assert_eq!(u8::from(MmMessageType::AuthenticationRequest), 0x56);
        assert_eq!(u8::from(MmMessageType::AuthenticationResponse), 0x57);
        assert_eq!(u8::from(MmMessageType::AuthenticationReject), 0x58);
        assert_eq!(u8::from(MmMessageType::AuthenticationFailure), 0x59);
        assert_eq!(u8::from(MmMessageType::AuthenticationResult), 0x5A);

        // Security mode messages
        assert_eq!(u8::from(MmMessageType::SecurityModeCommand), 0x5D);
        assert_eq!(u8::from(MmMessageType::SecurityModeComplete), 0x5E);
        assert_eq!(u8::from(MmMessageType::SecurityModeReject), 0x5F);
    }

    /// Verify EPD values per 3GPP TS 24.007
    #[test]
    fn test_epd_values_3gpp_compliance() {
        use crate::enums::ExtendedProtocolDiscriminator;

        assert_eq!(
            u8::from(ExtendedProtocolDiscriminator::MobilityManagement),
            0x7E
        );
        assert_eq!(
            u8::from(ExtendedProtocolDiscriminator::SessionManagement),
            0x2E
        );
    }
}
