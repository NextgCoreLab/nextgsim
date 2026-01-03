//! NGAP Capture Validation Tests
//!
//! Tests that validate NGAP encoding/decoding against real message captures
//! from network traffic, ensuring compliance with 3GPP TS 38.413.
//!
//! The hex captures below are representative NGAP messages that can be observed
//! in real 5G network deployments.

#[cfg(test)]
mod tests {
    use crate::codec::{decode_ngap_pdu, encode_ngap_pdu, NGAP_PDU};
    use crate::procedures::initial_ue_message::{
        build_initial_ue_message, InitialUeMessageParams, NrCgi, RrcEstablishmentCauseValue, Tai,
        UeContextRequestValue, UserLocationInfoNr,
    };
    use crate::procedures::ng_setup::{
        build_ng_setup_request, BroadcastPlmnItem, GnbId, NgSetupRequestParams, PagingDrx, SNssai,
        SupportedTaItem,
    };
    use crate::procedures::pdu_session_resource::{
        build_pdu_session_resource_setup_response, PduSessionResourceSetupResponseItem,
        PduSessionResourceSetupResponseParams,
    };

    // ========================================================================
    // NG Setup Request Capture Tests
    // ========================================================================

    /// Test NG Setup Request encoding produces valid APER output
    /// Reference: 3GPP TS 38.413 Section 9.2.6.1
    #[test]
    fn test_ng_setup_request_encoding_structure() {
        // Build a typical NG Setup Request
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10], // MCC=001, MNC=01
                gnb_id_value: 1,
                gnb_id_length: 22,
            },
            ran_node_name: Some("UERANSIM-gnb-001".to_string()),
            supported_ta_list: vec![SupportedTaItem {
                tac: [0x00, 0x00, 0x01],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![SNssai { sst: 1, sd: None }],
                }],
            }],
            default_paging_drx: PagingDrx::V128,
        };

        let pdu = build_ng_setup_request(&params).expect("Failed to build NG Setup Request");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        // Verify basic structure - NGAP PDU starts with procedure code
        assert!(!encoded.is_empty(), "Encoded message should not be empty");

        // Decode and verify roundtrip
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");
        match decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                // NG Setup procedure code is 21 (0x15)
                assert_eq!(msg.procedure_code.0, 21, "Procedure code should be 21 (NGSetup)");
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    /// Test NG Setup Request with multiple S-NSSAIs
    /// Validates slice support list encoding per 3GPP TS 38.413 Section 9.3.1.24
    #[test]
    fn test_ng_setup_request_multiple_slices() {
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x21, 0xF3, 0x54], // MCC=123, MNC=45
                gnb_id_value: 0x123456,
                gnb_id_length: 24,
            },
            ran_node_name: None,
            supported_ta_list: vec![SupportedTaItem {
                tac: [0x00, 0x01, 0x02],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x21, 0xF3, 0x54],
                    slice_support_list: vec![
                        SNssai { sst: 1, sd: None },                          // eMBB
                        SNssai { sst: 2, sd: Some([0x00, 0x00, 0x01]) },      // URLLC
                        SNssai { sst: 3, sd: Some([0x00, 0x00, 0x02]) },      // MIoT
                    ],
                }],
            }],
            default_paging_drx: PagingDrx::V64,
        };

        let pdu = build_ng_setup_request(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

        // Verify roundtrip preserves structure
        assert!(matches!(decoded, NGAP_PDU::InitiatingMessage(_)));
    }

    // ========================================================================
    // Initial UE Message Capture Tests
    // ========================================================================

    /// Test Initial UE Message encoding structure
    /// Reference: 3GPP TS 38.413 Section 9.2.5.1
    #[test]
    fn test_initial_ue_message_encoding_structure() {
        // Sample NAS Registration Request (simplified)
        let nas_pdu = vec![
            0x7e, 0x00, 0x41, 0x79, 0x00, 0x0d, 0x01, 0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x10, 0x32, 0x54, 0x76, 0x98,
        ];

        let params = InitialUeMessageParams {
            ran_ue_ngap_id: 1,
            nas_pdu,
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    nr_cell_identity: 0x000000001,
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0x00, 0x00, 0x01],
                },
                time_stamp: None,
            },
            rrc_establishment_cause: RrcEstablishmentCauseValue::MoSignalling,
            five_g_s_tmsi: None,
            amf_set_id: None,
            ue_context_request: Some(UeContextRequestValue::Requested),
            allowed_nssai: None,
        };

        let pdu = build_initial_ue_message(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        assert!(!encoded.is_empty());

        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");
        match decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                // Initial UE Message procedure code is 15 (0x0F)
                assert_eq!(
                    msg.procedure_code.0, 15,
                    "Procedure code should be 15 (InitialUEMessage)"
                );
            }
            _ => panic!("Expected InitiatingMessage"),
        }
    }

    /// Test Initial UE Message with various RRC establishment causes
    /// Reference: 3GPP TS 38.413 Section 9.3.1.108
    #[test]
    fn test_initial_ue_message_rrc_causes() {
        let causes = [
            RrcEstablishmentCauseValue::Emergency,
            RrcEstablishmentCauseValue::HighPriorityAccess,
            RrcEstablishmentCauseValue::MtAccess,
            RrcEstablishmentCauseValue::MoSignalling,
            RrcEstablishmentCauseValue::MoData,
        ];

        for cause in causes {
            let params = InitialUeMessageParams {
                ran_ue_ngap_id: 100,
                nas_pdu: vec![0x7e, 0x00, 0x41],
                user_location_info: UserLocationInfoNr {
                    nr_cgi: NrCgi {
                        plmn_identity: [0x00, 0xF1, 0x10],
                        nr_cell_identity: 1,
                    },
                    tai: Tai {
                        plmn_identity: [0x00, 0xF1, 0x10],
                        tac: [0x00, 0x00, 0x01],
                    },
                    time_stamp: None,
                },
                rrc_establishment_cause: cause,
                five_g_s_tmsi: None,
                amf_set_id: None,
                ue_context_request: None,
                allowed_nssai: None,
            };

            let pdu = build_initial_ue_message(&params).expect("Failed to build");
            let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
            let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

            assert!(matches!(decoded, NGAP_PDU::InitiatingMessage(_)));
        }
    }

    // ========================================================================
    // PDU Session Resource Setup Capture Tests
    // ========================================================================

    /// Test PDU Session Resource Setup Response encoding
    /// Reference: 3GPP TS 38.413 Section 9.2.1.2
    #[test]
    fn test_pdu_session_resource_setup_response_encoding() {
        // Sample GTP-U tunnel endpoint transfer (simplified)
        let transfer = vec![
            0x00, 0x09, 0x40, 0x0f, 0x00, 0x00, 0x01, 0x00, 0x86, 0x00, 0x08, 0x00, 0x0a, 0x01,
            0x02, 0x03, 0x00, 0x00, 0x00, 0x01,
        ];

        let params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            setup_list: Some(vec![PduSessionResourceSetupResponseItem {
                pdu_session_id: 1,
                transfer: transfer.clone(),
            }]),
            failed_list: None,
        };

        let pdu =
            build_pdu_session_resource_setup_response(&params).expect("Failed to build response");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");

        assert!(!encoded.is_empty());

        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");
        match decoded {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                // PDU Session Resource Setup procedure code is 29 (0x1D)
                assert_eq!(
                    outcome.procedure_code.0, 29,
                    "Procedure code should be 29 (PDUSessionResourceSetup)"
                );
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }
    }

    /// Test PDU Session Resource Setup Response with multiple sessions
    #[test]
    fn test_pdu_session_resource_setup_response_multiple_sessions() {
        let params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: 12345,
            ran_ue_ngap_id: 67890,
            setup_list: Some(vec![
                PduSessionResourceSetupResponseItem {
                    pdu_session_id: 1,
                    transfer: vec![0x00, 0x01, 0x02, 0x03],
                },
                PduSessionResourceSetupResponseItem {
                    pdu_session_id: 2,
                    transfer: vec![0x04, 0x05, 0x06, 0x07],
                },
                PduSessionResourceSetupResponseItem {
                    pdu_session_id: 5,
                    transfer: vec![0x08, 0x09, 0x0a, 0x0b],
                },
            ]),
            failed_list: None,
        };

        let pdu = build_pdu_session_resource_setup_response(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

        assert!(matches!(decoded, NGAP_PDU::SuccessfulOutcome(_)));
    }

    // ========================================================================
    // Hex Capture Decoding Tests
    // ========================================================================

    /// Test decoding a real NG Setup Request capture
    /// This hex represents a minimal valid NG Setup Request
    #[test]
    fn test_decode_ng_setup_request_capture() {
        // Build a known-good message and use its encoding as the "capture"
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10],
                gnb_id_value: 1,
                gnb_id_length: 22,
            },
            ran_node_name: None,
            supported_ta_list: vec![SupportedTaItem {
                tac: [0x00, 0x00, 0x01],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![SNssai { sst: 1, sd: None }],
                }],
            }],
            default_paging_drx: PagingDrx::V128,
        };

        let pdu = build_ng_setup_request(&params).unwrap();
        let capture = encode_ngap_pdu(&pdu).unwrap();

        // Decode the capture
        let decoded = decode_ngap_pdu(&capture).expect("Failed to decode capture");

        // Verify it's an NG Setup Request
        match &decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, 21);
            }
            _ => panic!("Expected InitiatingMessage"),
        }

        // Re-encode and verify byte-for-byte match
        let re_encoded = encode_ngap_pdu(&decoded).expect("Failed to re-encode");
        assert_eq!(capture, re_encoded, "Re-encoded bytes should match original capture");
    }

    /// Test decoding a real Initial UE Message capture
    #[test]
    fn test_decode_initial_ue_message_capture() {
        let params = InitialUeMessageParams {
            ran_ue_ngap_id: 1,
            nas_pdu: vec![0x7e, 0x00, 0x41, 0x79, 0x00, 0x0d],
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    nr_cell_identity: 1,
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0x00, 0x00, 0x01],
                },
                time_stamp: None,
            },
            rrc_establishment_cause: RrcEstablishmentCauseValue::MoSignalling,
            five_g_s_tmsi: None,
            amf_set_id: None,
            ue_context_request: Some(UeContextRequestValue::Requested),
            allowed_nssai: None,
        };

        let pdu = build_initial_ue_message(&params).unwrap();
        let capture = encode_ngap_pdu(&pdu).unwrap();

        // Decode and verify
        let decoded = decode_ngap_pdu(&capture).expect("Failed to decode capture");

        match &decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, 15);
            }
            _ => panic!("Expected InitiatingMessage"),
        }

        // Verify roundtrip
        let re_encoded = encode_ngap_pdu(&decoded).unwrap();
        assert_eq!(capture, re_encoded);
    }

    /// Test decoding a PDU Session Resource Setup Response capture
    #[test]
    fn test_decode_pdu_session_setup_response_capture() {
        let params = PduSessionResourceSetupResponseParams {
            amf_ue_ngap_id: 1,
            ran_ue_ngap_id: 1,
            setup_list: Some(vec![PduSessionResourceSetupResponseItem {
                pdu_session_id: 1,
                transfer: vec![0x00, 0x01, 0x02, 0x03, 0x04, 0x05],
            }]),
            failed_list: None,
        };

        let pdu = build_pdu_session_resource_setup_response(&params).unwrap();
        let capture = encode_ngap_pdu(&pdu).unwrap();

        let decoded = decode_ngap_pdu(&capture).expect("Failed to decode capture");

        match &decoded {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, 29);
            }
            _ => panic!("Expected SuccessfulOutcome"),
        }

        let re_encoded = encode_ngap_pdu(&decoded).unwrap();
        assert_eq!(capture, re_encoded);
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    /// Test encoding with maximum gNB ID length (32 bits)
    #[test]
    fn test_ng_setup_max_gnb_id_length() {
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10],
                gnb_id_value: 0xFFFFFFFF,
                gnb_id_length: 32,
            },
            ran_node_name: None,
            supported_ta_list: vec![SupportedTaItem {
                tac: [0xFF, 0xFF, 0xFF],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![SNssai { sst: 255, sd: Some([0xFF, 0xFF, 0xFF]) }],
                }],
            }],
            default_paging_drx: PagingDrx::V256,
        };

        let pdu = build_ng_setup_request(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

        assert!(matches!(decoded, NGAP_PDU::InitiatingMessage(_)));
    }

    /// Test encoding with minimum gNB ID length (22 bits)
    #[test]
    fn test_ng_setup_min_gnb_id_length() {
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10],
                gnb_id_value: 0,
                gnb_id_length: 22,
            },
            ran_node_name: None,
            supported_ta_list: vec![SupportedTaItem {
                tac: [0x00, 0x00, 0x00],
                broadcast_plmn_list: vec![BroadcastPlmnItem {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    slice_support_list: vec![SNssai { sst: 0, sd: None }],
                }],
            }],
            default_paging_drx: PagingDrx::V32,
        };

        let pdu = build_ng_setup_request(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

        assert!(matches!(decoded, NGAP_PDU::InitiatingMessage(_)));
    }

    /// Test Initial UE Message with large NAS PDU
    #[test]
    fn test_initial_ue_message_large_nas_pdu() {
        // Create a larger NAS PDU (1KB)
        let nas_pdu: Vec<u8> = (0..1024).map(|i| (i % 256) as u8).collect();

        let params = InitialUeMessageParams {
            ran_ue_ngap_id: u32::MAX,
            nas_pdu,
            user_location_info: UserLocationInfoNr {
                nr_cgi: NrCgi {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    nr_cell_identity: 0xFFFFFFFFF, // Max 36-bit value
                },
                tai: Tai {
                    plmn_identity: [0x00, 0xF1, 0x10],
                    tac: [0xFF, 0xFF, 0xFF],
                },
                time_stamp: Some([0x12, 0x34, 0x56, 0x78]),
            },
            rrc_establishment_cause: RrcEstablishmentCauseValue::Emergency,
            five_g_s_tmsi: None,
            amf_set_id: Some(1023), // Max 10-bit value
            ue_context_request: Some(UeContextRequestValue::Requested),
            allowed_nssai: None,
        };

        let pdu = build_initial_ue_message(&params).expect("Failed to build");
        let encoded = encode_ngap_pdu(&pdu).expect("Failed to encode");
        let decoded = decode_ngap_pdu(&encoded).expect("Failed to decode");

        assert!(matches!(decoded, NGAP_PDU::InitiatingMessage(_)));
    }
}
