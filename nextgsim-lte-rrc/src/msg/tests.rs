use super::*;

fn plmn() -> PlmnId {
    PlmnId {
        mcc: 999,
        mnc: 70,
        long_mnc: false,
    }
}

fn cell() -> CellInfo {
    CellInfo {
        plmns: vec![plmn()],
        tac: 1,
        cell_identity: 0x19B01,
        cell_barred: false,
        q_rx_lev_min_dbm: -140,
        freq_band: 7,
        system_info_value_tag: 0,
    }
}

/// Golden bytes derived from X.691 (UPER, TS 36.331 §8.1) by hand, not
/// captured from the encoder:
///
/// ```text
/// UL-CCCH-MessageType  CHOICE{c1, messageClassExtension}   1 bit  0   c1
/// c1                   CHOICE{reestablishmentReq, req}     1 bit  1   rrcConnectionRequest
/// criticalExtensions   CHOICE{r8, r15}                     1 bit  0   r8
/// ue-Identity          CHOICE{s-TMSI, randomValue}         1 bit  1   randomValue
/// randomValue          BIT STRING (SIZE (40))             40 bits 0x123456789A
/// establishmentCause   ENUMERATED, 8 values, no ext        3 bits 011 mo-Signalling
/// spare                BIT STRING (SIZE (1))               1 bit  0
/// = 0101 | 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 | 011 | 0
/// = 51 23 45 67 89 A6   (48 bits, no padding)
/// ```
const GOLDEN_RRC_CONNECTION_REQUEST: [u8; 6] = [0x51, 0x23, 0x45, 0x67, 0x89, 0xA6];

/// ```text
/// UL-DCCH-MessageType  CHOICE{c1, messageClassExtension}    1 bit  0
/// c1                   CHOICE of 16                         4 bits 0100 rrcConnectionSetupComplete (4)
/// rrc-TransactionIdentifier INTEGER (0..3)                  2 bits 00
/// criticalExtensions   CHOICE{c1, future}                   1 bit  0
/// c1                   CHOICE{r8, spare3, spare2, spare1}   2 bits 00
/// r8-IEs preamble      registeredMME, nonCriticalExt absent 2 bits 00
/// selectedPLMN-Identity INTEGER (1..maxPLMN-r11 = 6)        3 bits 000 (1)
/// dedicatedInfoNAS     OCTET STRING, length 2               8 bits 00000010, then 07 41
/// = 0 0100 00 0 00 00 000 | 00000010 | 00000111 01000001 | pad 0
/// = 20 00 04 0E 82
/// ```
const GOLDEN_RRC_CONNECTION_SETUP_COMPLETE: [u8; 5] = [0x20, 0x00, 0x04, 0x0E, 0x82];

#[test]
fn rrc_connection_request_matches_x691_golden() {
    let bytes = encode_rrc_connection_request(
        InitialUeIdentity::Random(0x12_3456_789A),
        EstablishmentCauseValue::MoSignalling,
    )
    .unwrap();
    assert_eq!(bytes, GOLDEN_RRC_CONNECTION_REQUEST);
    assert_eq!(
        decode_ul_ccch(&GOLDEN_RRC_CONNECTION_REQUEST).unwrap(),
        UlCcch::ConnectionRequest {
            identity: InitialUeIdentity::Random(0x12_3456_789A),
            cause: EstablishmentCauseValue::MoSignalling,
        }
    );
}

#[test]
fn rrc_connection_setup_complete_matches_x691_golden() {
    let bytes = encode_rrc_connection_setup_complete(0, 1, &[0x07, 0x41]).unwrap();
    assert_eq!(bytes, GOLDEN_RRC_CONNECTION_SETUP_COMPLETE);
    assert_eq!(
        decode_ul_dcch(&GOLDEN_RRC_CONNECTION_SETUP_COMPLETE).unwrap(),
        UlDcch::ConnectionSetupComplete {
            transaction_id: 0,
            selected_plmn_index: 1,
            nas: vec![0x07, 0x41],
        }
    );
}

#[test]
fn s_tmsi_request_round_trips() {
    let id = InitialUeIdentity::STmsi {
        mmec: 0x01,
        m_tmsi: 0xC000_0001,
    };
    let bytes = encode_rrc_connection_request(id, EstablishmentCauseValue::MtAccess).unwrap();
    // 0 c1 | 1 req | 0 r8 | 0 s-TMSI | mmec 0000 0001 ... -> first octet 0100 0000.
    assert_eq!(bytes[0], 0x40);
    assert_eq!(
        decode_ul_ccch(&bytes).unwrap(),
        UlCcch::ConnectionRequest {
            identity: id,
            cause: EstablishmentCauseValue::MtAccess
        }
    );
}

#[test]
fn random_value_over_40_bits_is_refused() {
    assert!(encode_rrc_connection_request(
        InitialUeIdentity::Random(1 << 40),
        EstablishmentCauseValue::MoData
    )
    .is_err());
}

#[test]
fn mib_round_trips_and_drops_sfn_lsbs() {
    let info = MibInfo {
        dl_bandwidth: DlBandwidth::N50,
        sfn: 0x3FF,
    };
    let bytes = encode_mib(&info).unwrap();
    // 24 bits on the air (§6.2.2): 3 + 1 + 2 + 8 + 5 + 1 + (1 + 2) + 1 = 24.
    assert_eq!(bytes.len(), 3);
    assert_eq!(
        decode_mib(&bytes).unwrap(),
        MibInfo {
            dl_bandwidth: DlBandwidth::N50,
            sfn: 0x3FC
        }
    );
}

#[test]
fn sib1_round_trips() {
    let c = cell();
    let bytes = encode_sib1(&c).unwrap();
    assert_eq!(decode_bcch_dl_sch(&bytes).unwrap(), BcchDlSch::Sib1(c));
}

#[test]
fn sib1_with_three_digit_mnc_and_six_plmns_round_trips() {
    let mut c = cell();
    c.plmns = (0..6)
        .map(|i| PlmnId {
            mcc: 310,
            mnc: 410 + i,
            long_mnc: true,
        })
        .collect();
    c.cell_barred = true;
    c.q_rx_lev_min_dbm = -44;
    let bytes = encode_sib1(&c).unwrap();
    assert_eq!(decode_bcch_dl_sch(&bytes).unwrap(), BcchDlSch::Sib1(c));
}

#[test]
fn sib1_refuses_values_outside_the_asn1_types() {
    let mut c = cell();
    c.q_rx_lev_min_dbm = -141;
    assert!(encode_sib1(&c).is_err());
    let mut c = cell();
    c.plmns.clear();
    assert!(encode_sib1(&c).is_err());
    let mut c = cell();
    c.cell_identity = 1 << 28;
    assert!(encode_sib1(&c).is_err());
}

#[test]
fn sib2_round_trips_every_t300() {
    for t300 in [100, 200, 300, 400, 600, 1000, 1500, 2000] {
        let s = Sib2Info {
            timers: UeTimers {
                t300_ms: t300,
                ..UeTimers::default()
            },
            default_paging_cycle: PagingCycle::Rf128,
        };
        let bytes = encode_sib2_si(&s).unwrap();
        assert_eq!(
            decode_bcch_dl_sch(&bytes).unwrap(),
            BcchDlSch::SystemInformation { sib2: Some(s) }
        );
    }
    let bad = Sib2Info {
        timers: UeTimers {
            t300_ms: 250,
            ..UeTimers::default()
        },
        default_paging_cycle: PagingCycle::Rf32,
    };
    assert!(encode_sib2_si(&bad).is_err());
}

#[test]
fn rrc_connection_setup_configures_srb1() {
    for tid in 0..=3 {
        let bytes = encode_rrc_connection_setup(tid).unwrap();
        assert_eq!(
            decode_dl_ccch(&bytes).unwrap(),
            DlCcch::ConnectionSetup {
                transaction_id: tid,
                configures_srb1: true
            }
        );
    }
    assert!(encode_rrc_connection_setup(4).is_err());
}

#[test]
fn rrc_connection_reject_round_trips() {
    let bytes = encode_rrc_connection_reject(16).unwrap();
    assert_eq!(
        decode_dl_ccch(&bytes).unwrap(),
        DlCcch::ConnectionReject { wait_time_s: 16 }
    );
    assert!(encode_rrc_connection_reject(0).is_err());
}

#[test]
fn information_transfer_round_trips_both_directions() {
    let nas: Vec<u8> = (0..=255u8).collect();
    let ul = encode_ul_information_transfer(&nas).unwrap();
    assert_eq!(
        decode_ul_dcch(&ul).unwrap(),
        UlDcch::InformationTransfer { nas: nas.clone() }
    );
    let dl = encode_dl_information_transfer(2, &nas).unwrap();
    assert_eq!(
        decode_dl_dcch(&dl).unwrap(),
        DlDcch::InformationTransfer { nas }
    );
}

#[test]
fn rrc_connection_release_round_trips() {
    let bytes = encode_rrc_connection_release(1).unwrap();
    assert_eq!(decode_dl_dcch(&bytes).unwrap(), DlDcch::ConnectionRelease);
    assert!(encode_rrc_connection_release(4).is_err());
}
