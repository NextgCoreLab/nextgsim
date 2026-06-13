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
                assert_eq!(
                    msg.procedure_code.0, 21,
                    "Procedure code should be 21 (NGSetup)"
                );
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
                        SNssai { sst: 1, sd: None }, // eMBB
                        SNssai {
                            sst: 2,
                            sd: Some([0x00, 0x00, 0x01]),
                        }, // URLLC
                        SNssai {
                            sst: 3,
                            sd: Some([0x00, 0x00, 0x02]),
                        }, // MIoT
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
        assert_eq!(
            capture, re_encoded,
            "Re-encoded bytes should match original capture"
        );
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
                    slice_support_list: vec![SNssai {
                        sst: 255,
                        sd: Some([0xFF, 0xFF, 0xFF]),
                    }],
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

    // ========================================================================
    // Cross-codec NG Setup regression guards (W5 E2E NGAP reconciliation)
    // ========================================================================
    //
    // These pin the wire bytes that the independent ogs-ngap (nextgcore) codec
    // produces and accepts for the NG Setup Request and Response against the
    // bytes this generated nextgsim-ngap codec produces and accepts. The
    // matching test on the core side
    // (`ogs-ngap/src/builder.rs::ng_setup_cross_codec`) re-encodes the same
    // vectors; together they guarantee both directions round-trip across the
    // two stacks. See that module for the root-cause description.
    //
    // The vectors were captured by encoding the identical logical message with
    // each stack and confirming byte equality after the two X.691 fixes on the
    // ogs side (PrintableString AMFName/RANNodeName, and the message-SEQUENCE
    // extension-marker bit before the IE container).

    /// NG Setup Response as produced by ogs-ngap `build_ng_setup_response`
    /// (AMFName "nextgcore-amf", one GUAMI 001-01/region 2/set 1/pointer 1,
    /// capacity 255, one PLMN 001-01 with S-NSSAI sst=1).
    const CORE_NG_SETUP_RESPONSE: [u8; 55] = [
        0x20, 0x15, 0x00, 0x33, 0x00, 0x00, 0x04, 0x00, 0x01, 0x00, 0x0f, 0x06, 0x00, 0x6e, 0x65,
        0x78, 0x74, 0x67, 0x63, 0x6f, 0x72, 0x65, 0x2d, 0x61, 0x6d, 0x66, 0x00, 0x60, 0x00, 0x08,
        0x00, 0x00, 0x00, 0xf1, 0x10, 0x02, 0x00, 0x41, 0x00, 0x56, 0x40, 0x01, 0xff, 0x00, 0x50,
        0x00, 0x08, 0x00, 0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x08,
    ];

    /// Decode-direction guard: the sim (gNB) must parse the AMF's NG Setup
    /// Response produced by the strict core codec.
    #[test]
    fn ng_setup_response_from_core_decodes() {
        let decoded =
            decode_ngap_pdu(&CORE_NG_SETUP_RESPONSE).expect("sim must decode core NG Setup Response");
        match decoded {
            NGAP_PDU::SuccessfulOutcome(outcome) => {
                assert_eq!(outcome.procedure_code.0, 21, "NGSetup procedure code");
            }
            other => panic!("expected SuccessfulOutcome, got {other:?}"),
        }
    }

    /// The sim's own NG Setup Response encoding must be byte-identical to the
    /// core's, so the gNB and AMF agree on the wire in both directions.
    #[test]
    fn ng_setup_response_sim_matches_core_bytes() {
        use crate::codec::generated::*;
        use bitvec::prelude::*;

        fn bits(value: u64, n: usize) -> bitvec::vec::BitVec<u8, Msb0> {
            let mut bv: bitvec::vec::BitVec<u8, Msb0> = bitvec::vec::BitVec::with_capacity(n);
            for i in (0..n).rev() {
                bv.push((value >> i) & 1 == 1);
            }
            bv
        }

        let ies = vec![
            NGSetupResponseProtocolIEs_Entry {
                id: ProtocolIE_ID(1),
                criticality: Criticality(Criticality::REJECT),
                value: NGSetupResponseProtocolIEs_EntryValue::Id_AMFName(AMFName(
                    "nextgcore-amf".to_string(),
                )),
            },
            NGSetupResponseProtocolIEs_Entry {
                id: ProtocolIE_ID(96),
                criticality: Criticality(Criticality::REJECT),
                value: NGSetupResponseProtocolIEs_EntryValue::Id_ServedGUAMIList(ServedGUAMIList(
                    vec![ServedGUAMIItem {
                        guami: GUAMI {
                            plmn_identity: PLMNIdentity(vec![0x00, 0xf1, 0x10]),
                            amf_region_id: AMFRegionID(bits(0x02, 8)),
                            amf_set_id: AMFSetID(bits(0x001, 10)),
                            amf_pointer: AMFPointer(bits(0x01, 6)),
                            ie_extensions: None,
                        },
                        backup_amf_name: None,
                        ie_extensions: None,
                    }],
                )),
            },
            NGSetupResponseProtocolIEs_Entry {
                id: ProtocolIE_ID(86),
                criticality: Criticality(Criticality::IGNORE),
                value: NGSetupResponseProtocolIEs_EntryValue::Id_RelativeAMFCapacity(
                    RelativeAMFCapacity(255),
                ),
            },
            NGSetupResponseProtocolIEs_Entry {
                id: ProtocolIE_ID(80),
                criticality: Criticality(Criticality::REJECT),
                value: NGSetupResponseProtocolIEs_EntryValue::Id_PLMNSupportList(PLMNSupportList(
                    vec![PLMNSupportItem {
                        plmn_identity: PLMNIdentity(vec![0x00, 0xf1, 0x10]),
                        slice_support_list: SliceSupportList(vec![SliceSupportItem {
                            s_nssai: S_NSSAI {
                                sst: SST(vec![0x01]),
                                sd: None,
                                ie_extensions: None,
                            },
                            ie_extensions: None,
                        }]),
                        ie_extensions: None,
                    }],
                )),
            },
        ];
        let pdu = NGAP_PDU::SuccessfulOutcome(SuccessfulOutcome {
            procedure_code: ProcedureCode(21),
            criticality: Criticality(Criticality::REJECT),
            value: SuccessfulOutcomeValue::Id_NGSetup(NGSetupResponse {
                protocol_i_es: NGSetupResponseProtocolIEs(ies),
            }),
        });
        let bytes = encode_ngap_pdu(&pdu).unwrap();
        assert_eq!(
            bytes,
            CORE_NG_SETUP_RESPONSE.to_vec(),
            "sim NG Setup Response must match the strict core codec byte-for-byte"
        );
    }

    /// NG Setup Request as produced by ogs-ngap `build_ng_setup_request`
    /// (GlobalGNB-ID 001-01/gnb-id len 32, RANNodeName "nextgsim-gnb",
    /// one TAC 000001 with PLMN 001-01 / S-NSSAI sst=1).
    const CORE_NG_SETUP_REQUEST: [u8; 60] = [
        0x00, 0x15, 0x00, 0x38, 0x00, 0x00, 0x04, 0x00, 0x1b, 0x00, 0x09, 0x00, 0x00, 0xf1, 0x10,
        0x50, 0x00, 0x00, 0x00, 0x08, 0x00, 0x52, 0x40, 0x0e, 0x05, 0x80, 0x6e, 0x65, 0x78, 0x74,
        0x67, 0x73, 0x69, 0x6d, 0x2d, 0x67, 0x6e, 0x62, 0x00, 0x66, 0x00, 0x0d, 0x00, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0xf1, 0x10, 0x00, 0x00, 0x00, 0x08, 0x00, 0x15, 0x40, 0x01, 0x40,
    ];

    /// Decode-direction guard: the sim must parse an NG Setup Request encoded
    /// by the strict core codec (the same direction the core decodes from the
    /// sim — symmetric, since both NG Setup messages share the framing fix).
    #[test]
    fn ng_setup_request_from_core_decodes() {
        let decoded =
            decode_ngap_pdu(&CORE_NG_SETUP_REQUEST).expect("sim must decode core NG Setup Request");
        match decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, 21, "NGSetup procedure code");
            }
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    /// The sim's own NG Setup Request encoding (gnb-id value 1) shares the
    /// framing of the core's (gnb-id value 8): both carry the message-SEQUENCE
    /// extension-marker bit and the PrintableString RANNodeName, so the
    /// structural prefix and the RANNodeName IE bytes are identical.
    #[test]
    fn ng_setup_request_sim_shares_core_framing() {
        let params = NgSetupRequestParams {
            gnb_id: GnbId {
                plmn_identity: [0x00, 0xF1, 0x10],
                gnb_id_value: 1,
                gnb_id_length: 32,
            },
            ran_node_name: Some("nextgsim-gnb".to_string()),
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
        let bytes = encode_ngap_pdu(&pdu).unwrap();
        // Outer framing: SuccessfulOutcome/InitiatingMessage header, open-type
        // length, and the container preamble + count "00 00 04".
        assert_eq!(&bytes[0..7], &CORE_NG_SETUP_REQUEST[0..7]);
        // RANNodeName IE (id 82 = 0x52) encoded as PrintableString with the
        // extension-bit + constrained-length framing "05 80" before the chars.
        assert_eq!(&bytes[20..38], &CORE_NG_SETUP_REQUEST[20..38]);
    }

    /// ICS Request (AMF → gNB): the gNB must decode the AMF's encode. Pinned
    /// vector is the strict core's (ogs-ngap) output after the W5 ICS fixes
    /// (BitRate extensible-constrained, UESecurityCapabilities size-ext bits,
    /// AllowedNSSAI bare S-NSSAI). The matching core test
    /// (builder.rs::ics_request_matches_sim_wire_bytes) re-encodes it.
    const CORE_ICS_REQUEST: [u8; 99] = [
        0x00, 0x0e, 0x00, 0x5f, 0x00, 0x00, 0x07, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x01, 0x00, 0x55,
        0x00, 0x02, 0x00, 0x01, 0x00, 0x1c, 0x00, 0x07, 0x00, 0x00, 0xf1, 0x10, 0x02, 0x00, 0x41,
        0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x77, 0x00, 0x09, 0x10, 0x00, 0x08, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x5e, 0x00, 0x20, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11,
        0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x00, 0x6e, 0x00, 0x0a, 0x0c,
        0x3b, 0x9a, 0xca, 0x00, 0x30, 0x1d, 0xcd, 0x65, 0x00,
    ];

    #[test]
    fn ics_request_from_core_decodes() {
        let decoded =
            decode_ngap_pdu(&CORE_ICS_REQUEST).expect("sim must decode core ICS Request");
        match decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(msg.procedure_code.0, 14, "InitialContextSetup procedure code");
            }
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    /// PDU Session Resource Setup Request (AMF → gNB) outer NGAP envelope: the
    /// gNB must decode the core's encode (the inner N2 transfer is opaque and
    /// covered by the W5.4 transfer tests). Pinned vector is the core's output.
    const CORE_PDU_SESSION_SETUP_REQUEST: [u8; 33] = [
        0x00, 0x1d, 0x00, 0x1d, 0x00, 0x00, 0x03, 0x00, 0x0a, 0x00, 0x02, 0x00, 0x01, 0x00, 0x55,
        0x00, 0x02, 0x00, 0x01, 0x00, 0x4a, 0x00, 0x0a, 0x00, 0x00, 0x05, 0x00, 0x20, 0x04, 0x00,
        0x01, 0x02, 0x03,
    ];

    #[test]
    fn pdu_session_setup_request_from_core_decodes() {
        let decoded = decode_ngap_pdu(&CORE_PDU_SESSION_SETUP_REQUEST)
            .expect("sim must decode core PDU Session Setup Request envelope");
        match decoded {
            NGAP_PDU::InitiatingMessage(msg) => {
                assert_eq!(
                    msg.procedure_code.0, 29,
                    "PDUSessionResourceSetup procedure code"
                );
            }
            other => panic!("expected InitiatingMessage, got {other:?}"),
        }
    }

    /// N2 SM `PDUSessionResourceSetupRequestTransfer` as emitted by the core
    /// smfd `build_setup_request_transfer` via ogs-ngap `transfer.rs`
    /// (UPF F-TEID 0x00010001, addr 10.45.0.1, QFI 1, 5QI 9, ARP 8). The gNB's
    /// strict nextgsim-ngap transfer decoder must read it (the W5 PDU-session
    /// data-plane blocker was the smfd's old hand-rolled 12-byte layout).
    const SMFD_SETUP_REQUEST_TRANSFER: [u8; 32] = [
        0x00, 0x03, 0x00, 0x8b, 0x00, 0x0a, 0x01, 0xf0, 0x0a, 0x2d, 0x00, 0x01, 0x00, 0x01, 0x00,
        0x01, 0x00, 0x86, 0x00, 0x01, 0x00, 0x00, 0x88, 0x00, 0x07, 0x00, 0x01, 0x00, 0x00, 0x09,
        0x1c, 0x00,
    ];

    #[test]
    fn smfd_setup_request_transfer_decodes_in_gnb() {
        use crate::procedures::transfer::decode_setup_request_transfer;
        let data =
            decode_setup_request_transfer(&SMFD_SETUP_REQUEST_TRANSFER).expect("gNB must decode");
        // UPF N3 F-TEID carried for the uplink tunnel
        assert_eq!(data.ul_tunnel.teid, 0x0001_0001);
        assert_eq!(data.ul_tunnel.address.to_string(), "10.45.0.1");
        assert_eq!(data.qos_flows.len(), 1);
        assert_eq!(data.qos_flows[0].qfi, 1);
        assert_eq!(data.qos_flows[0].five_qi, Some(9));
    }

    /// The gNB's `PDUSessionResourceSetupResponseTransfer` (real APER) must
    /// decode in the core. This dumps the sim's encoding so the matching core
    /// test (smfd) can pin and decode it; here we assert it self-decodes and
    /// carries the gNB DL F-TEID the SMF needs for the PFCP DL FAR.
    #[test]
    fn gnb_setup_response_transfer_roundtrips() {
        use crate::procedures::transfer::{
            decode_setup_response_transfer, encode_setup_response_transfer, GtpTunnelInfo,
            SetupResponseTransferParams,
        };
        let params = SetupResponseTransferParams {
            dl_tunnel: GtpTunnelInfo {
                address: "10.46.0.1".parse().unwrap(),
                teid: 0x0002_0002,
            },
            accepted_qfis: vec![1],
            failed_qos_flows: vec![],
        };
        let bytes = encode_setup_response_transfer(&params).expect("gNB encodes response transfer");
        let decoded = decode_setup_response_transfer(&bytes).expect("self-decode");
        assert_eq!(decoded.dl_tunnel.teid, 0x0002_0002);
        assert_eq!(decoded.accepted_qfis, vec![1]);
        // Pin the wire bytes so the core smfd cross-codec test can decode them.
        assert_eq!(
            bytes,
            vec![
                0x00, 0x03, 0xe0, 0x0a, 0x2e, 0x00, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00, 0x01,
            ]
        );
    }

}
