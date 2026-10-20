//! The real LTE UE task -- with its real EPS NAS -- against the real eNB radio
//! task over RLS on loopback, with the MME side of S1 played by the test.
//!
//! What it proves end to end, with no shared code path between the two ends
//! other than the protocol: the UE selects the cell (TS 36.304 §5.2.3),
//! establishes an RRC connection (TS 36.331 §5.3.3) and its NAS sends an
//! ATTACH REQUEST it built itself (TS 24.301 §5.5.1.2.2); EPS AKA and the NAS
//! SMC run over DOWNLINK/UPLINK NAS TRANSPORT; INITIAL CONTEXT SETUP REQUEST
//! makes the eNB run the AS Security Mode Command with the `K_eNB` the MME
//! derived (TS 33.401 §7.2.6.2) and an RRCConnectionReconfiguration carrying
//! the ATTACH ACCEPT; the UE's ATTACH COMPLETE comes back over a ciphered,
//! integrity-protected SRB1; and INITIAL CONTEXT SETUP RESPONSE names the
//! eNB's S1-U endpoint.
//!
//! The MME here is written from TS 24.301 / TS 33.401 with the crypto
//! primitives directly -- not with `nextgsim-eps-nas` -- so a symmetric
//! misreading in the UE's NAS security cannot pass.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

use nextgsim_common::Plmn;
use nextgsim_crypto::eps_kdf::{derive_algorithm_key, derive_kasme, derive_kenb, EpsAlgorithmType};
use nextgsim_crypto::milenage::Milenage;
use nextgsim_crypto::nea::nea2_encrypt;
use nextgsim_crypto::nia::nia2_compute_mac;
use nextgsim_enb::radio::{run_operational, RlsRadio};
use nextgsim_enb::s1_task::{S1Message, S1Transport};
use nextgsim_lte_ue::config::{LteUeConfig, SubscriberConfig};
use nextgsim_lte_ue::nas::EpsNas;
use nextgsim_lte_ue::rrc::UeRrcState;
use nextgsim_s1ap::codec::*;
use tokio::sync::{mpsc, watch};

const K: [u8; 16] = [
    0x46, 0x5B, 0x5C, 0xE8, 0xB1, 0x99, 0xB4, 0x9F, 0xAA, 0x5F, 0x0A, 0x2E, 0xE2, 0x38, 0xA6, 0xBC,
];
const OPC: [u8; 16] = [
    0xE8, 0xED, 0x28, 0x9D, 0xEB, 0xA9, 0x52, 0xE4, 0x28, 0x3B, 0x54, 0xE8, 0x8E, 0x6D, 0x83, 0x4D,
];

struct ChannelMme {
    rx: mpsc::UnboundedReceiver<Vec<u8>>,
    tx: mpsc::UnboundedSender<(u16, Vec<u8>)>,
}

impl S1Transport for ChannelMme {
    async fn send(&mut self, stream: u16, data: &[u8], _ppid: u32) -> std::io::Result<()> {
        let _ = self.tx.send((stream, data.to_vec()));
        Ok(())
    }
    async fn recv(&mut self) -> std::io::Result<Option<S1Message>> {
        Ok(self.rx.recv().await.map(|data| S1Message {
            stream: 0,
            ppid: nextgsim_sctp::S1AP_PPID,
            data,
        }))
    }
}

async fn next_s1(rx: &mut mpsc::UnboundedReceiver<(u16, Vec<u8>)>) -> (u16, S1AP_PDU) {
    let (stream, b) = tokio::time::timeout(Duration::from_secs(10), rx.recv())
        .await
        .expect("no S1 message from the eNB within 10 s")
        .expect("eNB task ended");
    (stream, decode_s1ap_pdu(&b).unwrap())
}

fn dl_nas(mme: u32, enb: u32, nas: &[u8]) -> Vec<u8> {
    let ie = |id, v| DownlinkNASTransportProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(Criticality::REJECT),
        value: v,
    };
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_DOWNLINK_NAS_TRANSPORT),
        criticality: Criticality(Criticality::IGNORE),
        value: InitiatingMessageValue::Id_downlinkNASTransport(DownlinkNASTransport {
            protocol_i_es: DownlinkNASTransportProtocolIEs(vec![
                ie(
                    ID_MME_UE_S1AP_ID,
                    DownlinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(
                        mme,
                    )),
                ),
                ie(
                    ID_E_NB_UE_S1AP_ID,
                    DownlinkNASTransportProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(
                        enb,
                    )),
                ),
                ie(
                    ID_NAS_PDU,
                    DownlinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(NAS_PDU(nas.to_vec())),
                ),
            ]),
        }),
    }))
    .unwrap()
}

fn bits(octets: &[u8]) -> bitvec::vec::BitVec<u8, bitvec::order::Msb0> {
    octets
        .iter()
        .flat_map(|o| (0..8).rev().map(move |i| (o >> i) & 1 == 1))
        .collect()
}

/// INITIAL CONTEXT SETUP REQUEST (TS 36.413 §9.1.4.1) with one E-RAB
/// carrying `nas`, UE security capabilities 128-EEA1/2 and 128-EIA1/2.
fn initial_context_setup(mme: u32, enb: u32, kenb: &[u8; 32], nas: &[u8]) -> Vec<u8> {
    use InitialContextSetupRequestProtocolIEs_EntryValue as V;
    let ie = |id, v| InitialContextSetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(Criticality::REJECT),
        value: v,
    };
    let erab = E_RABToBeSetupListCtxtSUReq_Entry {
        id: ProtocolIE_ID(ID_E_RAB_TO_BE_SETUP_ITEM_CTXT_SU_REQ),
        criticality: Criticality(Criticality::REJECT),
        value: E_RABToBeSetupListCtxtSUReq_EntryValue::Id_E_RABToBeSetupItemCtxtSUReq(
            E_RABToBeSetupItemCtxtSUReq {
                e_rab_id: E_RAB_ID(5),
                e_ra_blevel_qo_s_parameters: E_RABLevelQoSParameters {
                    qci: QCI(9),
                    allocation_retention_priority: AllocationAndRetentionPriority {
                        priority_level: PriorityLevel(8),
                        pre_emption_capability: Pre_emptionCapability(1),
                        pre_emption_vulnerability: Pre_emptionVulnerability(1),
                        ie_extensions: None,
                    },
                    gbr_qos_information: None,
                    ie_extensions: None,
                },
                transport_layer_address: TransportLayerAddress(bits(&[127, 0, 0, 2])),
                gtp_teid: GTP_TEID(vec![0, 0, 0x12, 0x34]),
                nas_pdu: Some(NAS_PDU(nas.to_vec())),
                ie_extensions: None,
            },
        ),
    };
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_INITIAL_CONTEXT_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_InitialContextSetup(InitialContextSetupRequest {
            protocol_i_es: InitialContextSetupRequestProtocolIEs(vec![
                ie(ID_MME_UE_S1AP_ID, V::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(mme))),
                ie(
                    ID_E_NB_UE_S1AP_ID,
                    V::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(enb)),
                ),
                ie(
                    ID_U_EAGGREGATE_MAXIMUM_BITRATE,
                    V::Id_uEaggregateMaximumBitrate(UEAggregateMaximumBitrate {
                        u_eaggregate_maximum_bit_rate_dl: BitRate(1_000_000_000),
                        u_eaggregate_maximum_bit_rate_ul: BitRate(1_000_000_000),
                        ie_extensions: None,
                    }),
                ),
                ie(
                    ID_E_RAB_TO_BE_SETUP_LIST_CTXT_SU_REQ,
                    V::Id_E_RABToBeSetupListCtxtSUReq(E_RABToBeSetupListCtxtSUReq(vec![erab])),
                ),
                ie(
                    ID_UE_SECURITY_CAPABILITIES,
                    V::Id_UESecurityCapabilities(UESecurityCapabilities {
                        // §9.2.1.40: bit 1 = 128-EEA1, bit 2 = 128-EEA2.
                        encryption_algorithms: EncryptionAlgorithms(bits(&[0xC0, 0x00])),
                        integrity_protection_algorithms: IntegrityProtectionAlgorithms(bits(&[
                            0xC0, 0x00,
                        ])),
                        ie_extensions: None,
                    }),
                ),
                ie(ID_SECURITY_KEY, V::Id_SecurityKey(SecurityKey(bits(kenb)))),
            ]),
        }),
    }))
    .unwrap()
}

/// The MME's NAS side of one attach: an authentication vector for the test
/// subscriber and the EEA2/EIA2 NAS context it keys.
struct MmeNas {
    rand: [u8; 16],
    autn: [u8; 16],
    xres: [u8; 8],
    kasme: [u8; 32],
    k_int: [u8; 16],
    k_enc: [u8; 16],
    dl: u32,
    ul: u32,
}

impl MmeNas {
    fn new() -> Self {
        let m = Milenage::new(&K, &OPC);
        let rand = [0x5A; 16];
        let sqn = [0, 0, 0, 0, 0, 0x21];
        let ak = m.f5(&rand);
        let mut autn = [0u8; 16];
        for i in 0..6 {
            autn[i] = sqn[i] ^ ak[i];
        }
        autn[6..8].copy_from_slice(&[0x80, 0x00]);
        autn[8..].copy_from_slice(&m.f1(&rand, &sqn, &[0x80, 0x00]));
        let kasme = derive_kasme(
            &m.f3(&rand),
            &m.f4(&rand),
            &Plmn::new(999, 70, false).encode(),
            autn[..6].try_into().unwrap(),
        );
        Self {
            rand,
            autn,
            xres: m.f2(&rand),
            kasme,
            k_int: derive_algorithm_key(&kasme, EpsAlgorithmType::NasInt, 2),
            k_enc: derive_algorithm_key(&kasme, EpsAlgorithmType::NasEnc, 2),
            dl: 0,
            ul: 0,
        }
    }

    fn protect(&mut self, plain: &[u8], ht: u8) -> Vec<u8> {
        let count = self.dl;
        let mut body = plain.to_vec();
        if ht == 2 {
            nea2_encrypt(count, 0, 1, &self.k_enc, &mut body);
        }
        let mut input = vec![(count & 0xFF) as u8];
        input.extend_from_slice(&body);
        let mac = nia2_compute_mac(count, 0, 1, &self.k_int, &input);
        self.dl += 1;
        let mut out = vec![(ht << 4) | 7];
        out.extend_from_slice(&mac);
        out.extend_from_slice(&input);
        out
    }

    fn unprotect(&mut self, pdu: &[u8]) -> Vec<u8> {
        let count = self.ul;
        assert_eq!(
            &pdu[1..5],
            &nia2_compute_mac(count, 0, 0, &self.k_int, &pdu[5..]),
            "uplink NAS MAC at COUNT {count}"
        );
        let mut body = pdu[6..].to_vec();
        if matches!(pdu[0] >> 4, 2 | 4) {
            nea2_encrypt(count, 0, 0, &self.k_enc, &mut body);
        }
        self.ul += 1;
        body
    }
}

fn uplink_nas(pdu: S1AP_PDU) -> (u32, Vec<u8>) {
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_uplinkNASTransport(u),
        ..
    }) = pdu
    else {
        panic!("expected UPLINK NAS TRANSPORT, got {pdu:?}");
    };
    let mut got = (None, None);
    for ie in u.protocol_i_es.0 {
        match ie.value {
            UplinkNASTransportProtocolIEs_EntryValue::Id_MME_UE_S1AP_ID(v) => got.0 = Some(v.0),
            UplinkNASTransportProtocolIEs_EntryValue::Id_NAS_PDU(v) => got.1 = Some(v.0),
            _ => {}
        }
    }
    (got.0.unwrap(), got.1.unwrap())
}

async fn wait_state(rx: &mut watch::Receiver<UeRrcState>, f: impl Fn(UeRrcState) -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !f(*rx.borrow_and_update()) {
            rx.changed().await.unwrap();
        }
    })
    .await
    .expect("UE did not reach the expected RRC state");
}

struct Harness {
    to_enb: mpsc::UnboundedSender<Vec<u8>>,
    mme_rx: mpsc::UnboundedReceiver<(u16, Vec<u8>)>,
    state_rx: watch::Receiver<UeRrcState>,
    enb_ue_id: u32,
    _stop: watch::Sender<bool>,
}

/// eNB and UE up; returns once INITIAL UE MESSAGE has reached the MME,
/// checking it carries the ATTACH REQUEST the UE's NAS built.
async fn start() -> Harness {
    let mut cfg = nextgsim_enb::load_and_validate_enb_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../config/enb.yaml"
    ))
    .unwrap();
    cfg.radio.link_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
    cfg.radio.rls_port = 0;
    cfg.radio.si_broadcast_period_ms = 40;
    cfg.s1ap_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
    let mut radio = RlsRadio::bind(&cfg).await.unwrap();
    let enb_addr: SocketAddr = radio.local_addr().unwrap();
    let (to_enb, from_mme) = mpsc::unbounded_channel();
    let (to_mme, mut mme_rx) = mpsc::unbounded_channel();
    let mut s1 = ChannelMme {
        rx: from_mme,
        tx: to_mme,
    };
    let enb_cfg = cfg.clone();
    tokio::spawn(async move { run_operational(&enb_cfg, &mut radio, &mut s1).await });

    let ue_cfg = LteUeConfig {
        plmn: Plmn::new(999, 70, false),
        enb_search_list: vec![enb_addr.to_string()],
        sti: Some(0xABCD),
        position: [0, 0, 0],
        subscriber: Some(SubscriberConfig {
            imsi: "999700000000001".into(),
            key: K,
            op: OPC,
            op_type: nextgsim_common::OpType::Opc,
            sqn: 0,
            imei_sv: Some("3534900698733019".into()),
            eea: vec![0, 1, 2],
            eia: vec![1, 2],
            apn: None,
            pdn_type: Default::default(),
        }),
    };
    nextgsim_lte_ue::config::validate(&ue_cfg).unwrap();
    let nas = Box::new(EpsNas::new(ue_cfg.eps_nas().unwrap()));
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (state_tx, state_rx) = watch::channel(UeRrcState::NoCell);
    let (stop_tx, stop_rx) = watch::channel(false);
    tokio::spawn(async move {
        nextgsim_lte_ue::task::run(&ue_cfg, nas, socket, state_tx, stop_rx).await
    });

    let (stream, pdu) = next_s1(&mut mme_rx).await;
    assert_eq!(
        stream, 1,
        "TS 36.412 §7: UE-associated signalling is not on stream 0"
    );
    let S1AP_PDU::InitiatingMessage(InitiatingMessage {
        value: InitiatingMessageValue::Id_initialUEMessage(m),
        ..
    }) = pdu
    else {
        panic!("expected INITIAL UE MESSAGE, got {pdu:?}");
    };
    let mut enb_ue_id = None;
    let mut nas = None;
    for ie in m.protocol_i_es.0 {
        match ie.value {
            InitialUEMessageProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb_ue_id = Some(v.0),
            InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(v) => nas = Some(v.0),
            _ => {}
        }
    }
    let nas = nas.unwrap();
    // The UE's own ATTACH REQUEST: plain, KSI 7 / EPS attach, IMSI identity,
    // UE network capability EEA0-2 / EIA1-2 (TS 24.301 §8.2.4).
    assert_eq!(&nas[..3], &[0x07, 0x41, 0x71]);
    assert_eq!(
        &nas[3..12],
        &[0x08, 0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10]
    );
    assert_eq!(&nas[12..15], &[0x02, 0xE0, 0x60]);
    Harness {
        to_enb,
        mme_rx,
        state_rx,
        enb_ue_id: enb_ue_id.unwrap(),
        _stop: stop_tx,
    }
}

/// AKA and the NAS SMC over NAS transport; returns the MME's NAS context.
async fn authenticate_and_secure(h: &mut Harness) -> MmeNas {
    let mut mme = MmeNas::new();
    let mut auth = vec![0x07, 0x52, 0x00];
    auth.extend_from_slice(&mme.rand);
    auth.push(16);
    auth.extend_from_slice(&mme.autn);
    h.to_enb.send(dl_nas(4242, h.enb_ue_id, &auth)).unwrap();
    let (mme_id, resp) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(mme_id, 4242);
    assert_eq!(&resp[..3], &[0x07, 0x53, 8]);
    assert_eq!(
        &resp[3..],
        &mme.xres,
        "RES = XRES: the USIM computed f2 over our RAND"
    );

    // NAS SMC: EEA2/EIA2, KSI 0, replayed EEA0-2/EIA1-2, header type 3.
    let smc = mme.protect(&[0x07, 0x5D, 0x22, 0x00, 0x02, 0xE0, 0x60], 3);
    h.to_enb.send(dl_nas(4242, h.enb_ue_id, &smc)).unwrap();
    let (_, complete) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(
        complete[0], 0x47,
        "integrity protected and ciphered, new context"
    );
    assert_eq!(&mme.unprotect(&complete)[..2], &[0x07, 0x5E]);
    mme
}

fn attach_accept(mme: &mut MmeNas) -> Vec<u8> {
    let mut esm = vec![0x52, 0x01, 0xC1, 0x01, 0x09];
    let apn = b"\x08internet";
    esm.push(apn.len() as u8);
    esm.extend_from_slice(apn);
    esm.extend_from_slice(&[0x05, 0x01, 10, 45, 0, 2]);
    let mut m = vec![
        0x07, 0x42, 0x01, 0x49, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01,
    ];
    m.extend_from_slice(&(esm.len() as u16).to_be_bytes());
    m.extend_from_slice(&esm);
    mme.protect(&m, 2)
}

#[tokio::test]
async fn the_ue_attaches_through_the_enb_with_as_security_and_initial_context_setup() {
    let mut h = start().await;
    wait_state(&mut h.state_rx, |s| {
        matches!(s, UeRrcState::Connected { .. })
    })
    .await;
    let mut mme = authenticate_and_secure(&mut h).await;

    // TS 33.401 §7.2.6.2: K_eNB from K_ASME and the SMC complete's UL COUNT 0.
    let kenb = derive_kenb(&mme.kasme, 0);
    let accept = attach_accept(&mut mme);
    h.to_enb
        .send(initial_context_setup(4242, h.enb_ue_id, &kenb, &accept))
        .unwrap();

    // The eNB answers only after the AS SMC verified and the reconfiguration
    // completed: INITIAL CONTEXT SETUP RESPONSE with the eNB's S1-U endpoint.
    let (_, pdu) = next_s1(&mut h.mme_rx).await;
    let S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
        value: SuccessfulOutcomeValue::Id_InitialContextSetup(resp),
        ..
    }) = pdu
    else {
        panic!("expected INITIAL CONTEXT SETUP RESPONSE, got {pdu:?}");
    };
    let mut erab = None;
    for ie in resp.protocol_i_es.0 {
        if let InitialContextSetupResponseProtocolIEs_EntryValue::Id_E_RABSetupListCtxtSURes(l) =
            ie.value
        {
            let E_RABSetupListCtxtSURes_EntryValue::Id_E_RABSetupItemCtxtSURes(i) = &l.0[0].value;
            erab = Some((
                i.e_rab_id.0,
                i.transport_layer_address.0.len(),
                i.gtp_teid.0.clone(),
            ));
        }
    }
    let (id, addr_bits, teid) = erab.expect("E-RAB Setup List");
    assert_eq!((id, addr_bits), (5, 32));
    assert_ne!(teid, vec![0, 0, 0, 0]);

    // ATTACH COMPLETE, sent by the UE's NAS after the RRCConnectionReconfiguration
    // delivered the ATTACH ACCEPT, over SRB1 with AS ciphering on.
    let (_, ul) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    let complete = mme.unprotect(&ul);
    assert_eq!(&complete[..2], &[0x07, 0x43]);
    assert_eq!(
        &complete[4..7],
        &[0x52, 0x01, 0xC2],
        "default EPS bearer 5 accepted"
    );
}

#[tokio::test]
async fn a_wrong_kenb_fails_the_as_smc_and_initial_context_setup() {
    let mut h = start().await;
    wait_state(&mut h.state_rx, |s| {
        matches!(s, UeRrcState::Connected { .. })
    })
    .await;
    let mut mme = authenticate_and_secure(&mut h).await;
    // A K_eNB the UE cannot have derived (what an MME that never derived one
    // would send): the UE's SecurityModeCommand integrity check fails, it
    // answers SecurityModeFailure, and the eNB fails the procedure.
    let accept = attach_accept(&mut mme);
    h.to_enb
        .send(initial_context_setup(4242, h.enb_ue_id, &[0; 32], &accept))
        .unwrap();
    let (_, pdu) = next_s1(&mut h.mme_rx).await;
    let S1AP_PDU::UnsuccessfulOutcome(UnsuccessfulOutcome {
        value: UnsuccessfulOutcomeValue::Id_InitialContextSetup(f),
        ..
    }) = pdu
    else {
        panic!("expected INITIAL CONTEXT SETUP FAILURE, got {pdu:?}");
    };
    let cause = f.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        InitialContextSetupFailureProtocolIEs_EntryValue::Id_Cause(Cause::RadioNetwork(c)) => {
            Some(c.0)
        }
        _ => None,
    });
    assert_eq!(
        cause,
        Some(CauseRadioNetwork::FAILURE_IN_RADIO_INTERFACE_PROCEDURE)
    );
}
