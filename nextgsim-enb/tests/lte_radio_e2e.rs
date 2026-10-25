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
//! integrity-protected SRB2 -- the reconfiguration added it -- and INITIAL
//! CONTEXT SETUP RESPONSE names the eNB's S1-U endpoint. Then the user plane:
//! an IP packet the UE sends leaves the eNB as a G-PDU on the SGW's TEID, and
//! a G-PDU on the eNB's TEID reaches the UE's IP stack, through PDCP
//! (ciphered with `K_UPenc`), RLC UM and RLS in both directions.
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
use nextgsim_enb::radio::{run_operational, RlsRadio, S1uEndpoint};
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
fn initial_context_setup(
    mme: u32,
    enb: u32,
    kenb: &[u8; 32],
    nas: &[u8],
    sgw: Ipv4Addr,
) -> Vec<u8> {
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
                transport_layer_address: TransportLayerAddress(bits(&sgw.octets())),
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
    /// IP packets into the UE's uplink, and out of its downlink.
    ip_ul: mpsc::Sender<Vec<u8>>,
    ip_dl: mpsc::Receiver<Vec<u8>>,
    /// The eNB's S1-U socket address.
    enb_s1u: SocketAddr,
    _stop: watch::Sender<bool>,
}

/// The eNB UE S1AP ID and the NAS-PDU of an INITIAL UE MESSAGE.
fn initial_ue_message(pdu: S1AP_PDU) -> (u32, Vec<u8>) {
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
    (enb_ue_id.unwrap(), nas.unwrap())
}

/// UE CONTEXT RELEASE COMMAND (TS 36.413 §9.1.4.6), cause NAS normal release.
fn release_command(mme: u32, enb: u32) -> Vec<u8> {
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_UE_CONTEXT_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_UEContextRelease(UEContextReleaseCommand {
            protocol_i_es: UEContextReleaseCommandProtocolIEs(vec![
                UEContextReleaseCommandProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_UE_S1AP_I_DS),
                    criticality: Criticality(Criticality::REJECT),
                    value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_UE_S1AP_IDs(
                        UE_S1AP_IDs::UE_S1AP_ID_pair(UE_S1AP_ID_pair {
                            mme_ue_s1ap_id: MME_UE_S1AP_ID(mme),
                            enb_ue_s1ap_id: ENB_UE_S1AP_ID(enb),
                            ie_extensions: None,
                        }),
                    ),
                },
                UEContextReleaseCommandProtocolIEs_Entry {
                    id: ProtocolIE_ID(ID_CAUSE),
                    criticality: Criticality(Criticality::IGNORE),
                    value: UEContextReleaseCommandProtocolIEs_EntryValue::Id_Cause(Cause::Nas(
                        CauseNas(CauseNas::NORMAL_RELEASE),
                    )),
                },
            ]),
        }),
    }))
    .unwrap()
}

/// Wait for the UE CONTEXT RELEASE COMPLETE that answers a release command.
async fn expect_release_complete(rx: &mut mpsc::UnboundedReceiver<(u16, Vec<u8>)>) {
    let (_, pdu) = next_s1(rx).await;
    assert!(
        matches!(
            pdu,
            S1AP_PDU::SuccessfulOutcome(SuccessfulOutcome {
                value: SuccessfulOutcomeValue::Id_UEContextRelease(_),
                ..
            })
        ),
        "expected UE CONTEXT RELEASE COMPLETE, got {pdu:?}"
    );
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
    let mut s1u = S1uEndpoint::bind(&cfg, Some(0)).await.unwrap();
    let enb_s1u = s1u.local_addr().unwrap();
    let (to_enb, from_mme) = mpsc::unbounded_channel();
    let (to_mme, mut mme_rx) = mpsc::unbounded_channel();
    let mut s1 = ChannelMme {
        rx: from_mme,
        tx: to_mme,
    };
    let enb_cfg = cfg.clone();
    tokio::spawn(async move { run_operational(&enb_cfg, &mut radio, &mut s1u, &mut s1).await });

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
        tun: None,
    };
    nextgsim_lte_ue::config::validate(&ue_cfg).unwrap();
    let nas = Box::new(EpsNas::new(ue_cfg.eps_nas().unwrap()));
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (state_tx, state_rx) = watch::channel(UeRrcState::NoCell);
    let (stop_tx, stop_rx) = watch::channel(false);
    let (ip, ip_ul, ip_dl) = nextgsim_lte_ue::task::IpIo::channel(16);
    tokio::spawn(async move {
        nextgsim_lte_ue::task::run(&ue_cfg, nas, socket, state_tx, stop_rx, ip).await
    });

    let (stream, pdu) = next_s1(&mut mme_rx).await;
    assert_eq!(
        stream, 1,
        "TS 36.412 §7: UE-associated signalling is not on stream 0"
    );
    let (enb_ue_id, nas) = initial_ue_message(pdu);
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
        enb_ue_id,
        ip_ul,
        ip_dl,
        enb_s1u,
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
    attach_accept_with(mme, 0x49, &[])
}

/// ATTACH ACCEPT (TS 24.301 §8.2.1) with T3412 octet `t3412` (§9.9.3.16) and
/// the optional IEs `ies` after the ESM container.
fn attach_accept_with(mme: &mut MmeNas, t3412: u8, ies: &[u8]) -> Vec<u8> {
    let mut esm = vec![0x52, 0x01, 0xC1, 0x01, 0x09];
    let apn = b"\x08internet";
    esm.push(apn.len() as u8);
    esm.extend_from_slice(apn);
    esm.extend_from_slice(&[0x05, 0x01, 10, 45, 0, 2]);
    let mut m = vec![
        0x07, 0x42, 0x01, t3412, 0x06, 0x00, 0x99, 0xF9, 0x07, 0x00, 0x01,
    ];
    m.extend_from_slice(&(esm.len() as u16).to_be_bytes());
    m.extend_from_slice(&esm);
    m.extend_from_slice(ies);
    mme.protect(&m, 2)
}

/// Run the attach to ATTACH COMPLETE with the SGW's S1-U address `sgw`;
/// returns the eNB's S1-U TEID for E-RAB 5 and the MME's NAS context.
async fn attach(h: &mut Harness, sgw: Ipv4Addr) -> (u32, MmeNas) {
    attach_with(h, sgw, attach_accept).await
}

/// [`attach`] with the ATTACH ACCEPT `accept` builds.
async fn attach_with(
    h: &mut Harness,
    sgw: Ipv4Addr,
    accept: impl FnOnce(&mut MmeNas) -> Vec<u8>,
) -> (u32, MmeNas) {
    wait_state(&mut h.state_rx, |s| {
        matches!(s, UeRrcState::Connected { .. })
    })
    .await;
    let mut mme = authenticate_and_secure(h).await;

    // TS 33.401 §7.2.6.2: K_eNB from K_ASME and the SMC complete's UL COUNT 0.
    let kenb = derive_kenb(&mme.kasme, 0);
    let accept = accept(&mut mme);
    h.to_enb
        .send(initial_context_setup(
            4242,
            h.enb_ue_id,
            &kenb,
            &accept,
            sgw,
        ))
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
    // delivered the ATTACH ACCEPT, over SRB2 (which that reconfiguration added)
    // with AS ciphering on.
    let (_, ul) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    let complete = mme.unprotect(&ul);
    assert_eq!(&complete[..2], &[0x07, 0x43]);
    assert_eq!(
        &complete[4..7],
        &[0x52, 0x01, 0xC2],
        "default EPS bearer 5 accepted"
    );
    (u32::from_be_bytes(teid.try_into().unwrap()), mme)
}

#[tokio::test]
async fn the_ue_attaches_through_the_enb_with_as_security_and_initial_context_setup() {
    let mut h = start().await;
    attach(&mut h, Ipv4Addr::new(127, 0, 0, 2)).await;
}

/// The user plane end to end: UE IP stack → PDCP (EEA2) → RLC UM → RLS →
/// eNB → G-PDU to the SGW's S1-U TEID, and back on the eNB's TEID.
#[tokio::test]
async fn ip_packets_cross_the_drb_and_s1u_in_both_directions() {
    // A stand-in SGW-U on its own loopback address, on the GTP-U port the eNB
    // derives from the E-RAB's Transport Layer Address.
    let sgw_ip = Ipv4Addr::new(127, 0, 0, 77);
    let sgw = tokio::net::UdpSocket::bind((sgw_ip, 2152)).await.unwrap();
    let mut h = start().await;
    let (enb_teid, _) = attach(&mut h, sgw_ip).await;

    let mut buf = vec![0u8; 2048];
    for i in 0..5u8 {
        // Uplink: an IPv4 header's worth of octets from the UE address.
        let packet = vec![
            0x45, 0, 0, 24, 0, i, 0, 0, 64, 1, 0, 0, 10, 45, 0, 2, 8, 8, 8, 8, 0, 0, 0, i,
        ];
        h.ip_ul.send(packet.clone()).await.unwrap();
        let (n, from) = tokio::time::timeout(Duration::from_secs(5), sgw.recv_from(&mut buf))
            .await
            .expect("no G-PDU at the SGW")
            .unwrap();
        // TS 29.281 §5.1: version 1, PT 1, no optional fields; G-PDU; the
        // SGW's TEID from the E-RAB item (0x1234); the packet unchanged.
        let mut expect = vec![0x30, 0xFF, 0x00, packet.len() as u8, 0x00, 0x00, 0x12, 0x34];
        expect.extend_from_slice(&packet);
        assert_eq!(&buf[..n], expect.as_slice(), "uplink G-PDU {i}");
        assert_eq!(from, h.enb_s1u);

        // Downlink: a G-PDU on the eNB's TEID reaches the UE's IP stack.
        let reply = vec![
            0x45, 0, 0, 24, 0, i, 0, 0, 64, 1, 0, 0, 8, 8, 8, 8, 10, 45, 0, 2, 0, 0, 0, i,
        ];
        let mut g = vec![0x30, 0xFF, 0x00, reply.len() as u8];
        g.extend_from_slice(&enb_teid.to_be_bytes());
        g.extend_from_slice(&reply);
        sgw.send_to(&g, h.enb_s1u).await.unwrap();
        let got = tokio::time::timeout(Duration::from_secs(5), h.ip_dl.recv())
            .await
            .expect("no downlink packet at the UE")
            .unwrap();
        assert_eq!(got, reply, "downlink packet {i}");
    }
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
        .send(initial_context_setup(
            4242,
            h.enb_ue_id,
            &[0; 32],
            &accept,
            Ipv4Addr::new(127, 0, 0, 2),
        ))
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

/// nextgsim #300: an attached UE whose connection is released (as when an
/// MME restart takes S1 down) is EMM-REGISTERED in EMM-IDLE. Uplink data
/// then makes it send a SERVICE REQUEST (TS 24.301 §5.6.1.1 b) -- in an
/// INITIAL UE MESSAGE with the S-TMSI, verified here with the short MAC
/// (§9.9.3.28) -- and a SERVICE REJECT #9 from an MME that does not know it
/// (§5.6.1.5) makes it attach again, with its IMSI, on a new connection.
#[tokio::test]
async fn an_idle_ue_sends_a_service_request_and_reattaches_after_reject_9() {
    let mut h = start().await;
    attach(&mut h, Ipv4Addr::new(127, 0, 0, 2)).await;
    h.to_enb.send(release_command(4242, h.enb_ue_id)).unwrap();
    expect_release_complete(&mut h.mme_rx).await;
    wait_state(&mut h.state_rx, |s| matches!(s, UeRrcState::Camped { .. })).await;

    // Uplink data in RRC_IDLE.
    let packet = vec![
        0x45, 0, 0, 24, 0, 1, 0, 0, 64, 1, 0, 0, 10, 45, 0, 2, 8, 8, 8, 8, 0, 0, 0, 1,
    ];
    h.ip_ul.send(packet).await.unwrap();
    let (enb_ue_id, sr) = initial_ue_message(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(sr.len(), 4, "SERVICE REQUEST: {sr:02x?}");
    assert_eq!(sr[0], 0xC7, "security header type 12");
    // KSI 0 (the SMC's), sequence number: UL NAS COUNT 2 after SECURITY MODE
    // COMPLETE (0) and ATTACH COMPLETE (1).
    assert_eq!(sr[1], 0x02);
    let k_int = derive_algorithm_key(&MmeNas::new().kasme, EpsAlgorithmType::NasInt, 2);
    let mac = nia2_compute_mac(2, 0, 0, &k_int, &sr[..2]);
    assert_eq!(&sr[2..], &mac[2..], "short MAC");

    // A restarted MME: no context, SERVICE REJECT #9, then the release.
    h.to_enb
        .send(dl_nas(5151, enb_ue_id, &[0x07, 0x4E, 9]))
        .unwrap();
    h.to_enb.send(release_command(5151, enb_ue_id)).unwrap();
    expect_release_complete(&mut h.mme_rx).await;

    let (_, nas) = initial_ue_message(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(
        &nas[..3],
        &[0x07, 0x41, 0x71],
        "a plain ATTACH REQUEST with no KSI: GUTI and KSI were deleted"
    );
    assert_eq!(
        &nas[3..12],
        &[0x08, 0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10],
        "IMSI identity"
    );
}

/// nextgsim #301: the MME's network-initiated detach with "re-attach
/// required" (TS 24.301 §5.5.2.3), as nextgcore mmed sends it when the UE's
/// PDN connection is lost at a restarted PGW-U or SGW. The UE answers DETACH
/// ACCEPT over SRB2 under its NAS context, and once the network releases the
/// connection attaches again -- integrity protected with the kept context.
#[tokio::test]
async fn a_network_detach_with_reattach_required_is_accepted_and_the_ue_reattaches() {
    let mut h = start().await;
    let (_, mut mme) = attach(&mut h, Ipv4Addr::new(127, 0, 0, 2)).await;
    // DETACH REQUEST: re-attach required, EMM cause #10, ciphered.
    let detach = mme.protect(&[0x07, 0x45, 0x01, 0x53, 10], 2);
    h.to_enb.send(dl_nas(4242, h.enb_ue_id, &detach)).unwrap();
    let (_, ul) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(mme.unprotect(&ul), vec![0x07, 0x46], "DETACH ACCEPT");

    h.to_enb.send(release_command(4242, h.enb_ue_id)).unwrap();
    expect_release_complete(&mut h.mme_rx).await;
    let (_, nas) = initial_ue_message(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(
        nas[0], 0x17,
        "integrity protected ATTACH REQUEST: {nas:02x?}"
    );
    assert_eq!(&mme.unprotect(&nas)[..2], &[0x07, 0x41]);
}

/// E-RAB Level QoS Parameters (TS 36.413 §9.2.1.15) for a non-GBR `qci`.
fn erab_qos(qci: u8) -> E_RABLevelQoSParameters {
    E_RABLevelQoSParameters {
        qci: QCI(qci),
        allocation_retention_priority: AllocationAndRetentionPriority {
            priority_level: PriorityLevel(8),
            pre_emption_capability: Pre_emptionCapability(1),
            pre_emption_vulnerability: Pre_emptionVulnerability(1),
            ie_extensions: None,
        },
        gbr_qos_information: None,
        ie_extensions: None,
    }
}

/// E-RAB SETUP REQUEST (TS 36.413 §9.1.3.1) for one E-RAB `id` carrying `nas`.
fn erab_setup_request(
    mme: u32,
    enb: u32,
    id: u8,
    sgw: Ipv4Addr,
    sgw_teid: u32,
    nas: &[u8],
) -> Vec<u8> {
    use E_RABSetupRequestProtocolIEs_EntryValue as V;
    let ie = |id, v| E_RABSetupRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(Criticality::REJECT),
        value: v,
    };
    let erab = E_RABToBeSetupListBearerSUReq_Entry {
        id: ProtocolIE_ID(ID_E_RAB_TO_BE_SETUP_ITEM_BEARER_SU_REQ),
        criticality: Criticality(Criticality::REJECT),
        value: E_RABToBeSetupListBearerSUReq_EntryValue::Id_E_RABToBeSetupItemBearerSUReq(
            E_RABToBeSetupItemBearerSUReq {
                e_rab_id: E_RAB_ID(id),
                e_ra_blevel_qo_s_parameters: erab_qos(5),
                transport_layer_address: TransportLayerAddress(bits(&sgw.octets())),
                gtp_teid: GTP_TEID(sgw_teid.to_be_bytes().to_vec()),
                nas_pdu: NAS_PDU(nas.to_vec()),
                ie_extensions: None,
            },
        ),
    };
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_E_RAB_SETUP),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_E_RABSetup(E_RABSetupRequest {
            protocol_i_es: E_RABSetupRequestProtocolIEs(vec![
                ie(ID_MME_UE_S1AP_ID, V::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(mme))),
                ie(
                    ID_E_NB_UE_S1AP_ID,
                    V::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(enb)),
                ),
                ie(
                    ID_E_RAB_TO_BE_SETUP_LIST_BEARER_SU_REQ,
                    V::Id_E_RABToBeSetupListBearerSUReq(E_RABToBeSetupListBearerSUReq(vec![erab])),
                ),
            ]),
        }),
    }))
    .unwrap()
}

/// E-RAB MODIFY REQUEST (TS 36.413 §9.1.3.3) for one E-RAB `id` carrying `nas`.
fn erab_modify_request(mme: u32, enb: u32, id: u8, nas: &[u8]) -> Vec<u8> {
    use E_RABModifyRequestProtocolIEs_EntryValue as V;
    let ie = |id, v| E_RABModifyRequestProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(Criticality::REJECT),
        value: v,
    };
    let erab = E_RABToBeModifiedListBearerModReq_Entry {
        id: ProtocolIE_ID(ID_E_RAB_TO_BE_MODIFIED_ITEM_BEARER_MOD_REQ),
        criticality: Criticality(Criticality::REJECT),
        value: E_RABToBeModifiedListBearerModReq_EntryValue::Id_E_RABToBeModifiedItemBearerModReq(
            E_RABToBeModifiedItemBearerModReq {
                e_rab_id: E_RAB_ID(id),
                e_rab_level_qo_s_parameters: erab_qos(5),
                nas_pdu: NAS_PDU(nas.to_vec()),
                ie_extensions: None,
            },
        ),
    };
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_E_RAB_MODIFY),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_E_RABModify(E_RABModifyRequest {
            protocol_i_es: E_RABModifyRequestProtocolIEs(vec![
                ie(ID_MME_UE_S1AP_ID, V::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(mme))),
                ie(
                    ID_E_NB_UE_S1AP_ID,
                    V::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(enb)),
                ),
                ie(
                    ID_E_RAB_TO_BE_MODIFIED_LIST_BEARER_MOD_REQ,
                    V::Id_E_RABToBeModifiedListBearerModReq(E_RABToBeModifiedListBearerModReq(
                        vec![erab],
                    )),
                ),
            ]),
        }),
    }))
    .unwrap()
}

/// E-RAB RELEASE COMMAND (TS 36.413 §9.1.3.5) for E-RAB `id` carrying `nas`.
fn erab_release_command(mme: u32, enb: u32, id: u8, nas: &[u8]) -> Vec<u8> {
    use E_RABReleaseCommandProtocolIEs_EntryValue as V;
    let ie = |id, criticality, v| E_RABReleaseCommandProtocolIEs_Entry {
        id: ProtocolIE_ID(id),
        criticality: Criticality(criticality),
        value: v,
    };
    let erab = E_RABList_Entry {
        id: ProtocolIE_ID(ID_E_RAB_ITEM),
        criticality: Criticality(Criticality::IGNORE),
        value: E_RABList_EntryValue::Id_E_RABItem(E_RABItem {
            e_rab_id: E_RAB_ID(id),
            cause: Cause::Nas(CauseNas(CauseNas::NORMAL_RELEASE)),
            ie_extensions: None,
        }),
    };
    encode_s1ap_pdu(&S1AP_PDU::InitiatingMessage(InitiatingMessage {
        procedure_code: ProcedureCode(ID_E_RAB_RELEASE),
        criticality: Criticality(Criticality::REJECT),
        value: InitiatingMessageValue::Id_E_RABRelease(E_RABReleaseCommand {
            protocol_i_es: E_RABReleaseCommandProtocolIEs(vec![
                ie(
                    ID_MME_UE_S1AP_ID,
                    Criticality::REJECT,
                    V::Id_MME_UE_S1AP_ID(MME_UE_S1AP_ID(mme)),
                ),
                ie(
                    ID_E_NB_UE_S1AP_ID,
                    Criticality::REJECT,
                    V::Id_eNB_UE_S1AP_ID(ENB_UE_S1AP_ID(enb)),
                ),
                ie(
                    ID_E_RAB_TO_BE_RELEASED_LIST,
                    Criticality::IGNORE,
                    V::Id_E_RABToBeReleasedList(E_RABList(vec![erab])),
                ),
                ie(
                    ID_NAS_PDU,
                    Criticality::IGNORE,
                    V::Id_NAS_PDU(NAS_PDU(nas.to_vec())),
                ),
            ]),
        }),
    }))
    .unwrap()
}

/// The eNB's answer to an E-RAB procedure and the UE's NAS reply to the ESM
/// request it carried, in whichever order they reach the MME.
async fn erab_answer_and_nas(
    h: &mut Harness,
    mme: &mut MmeNas,
) -> (SuccessfulOutcomeValue, Vec<u8>) {
    let (mut answer, mut nas) = (None, None);
    while answer.is_none() || nas.is_none() {
        match next_s1(&mut h.mme_rx).await.1 {
            S1AP_PDU::SuccessfulOutcome(o) => answer = Some(o.value),
            pdu => nas = Some(mme.unprotect(&uplink_nas(pdu).1)),
        }
    }
    (answer.unwrap(), nas.unwrap())
}

/// A UDP/IPv4 packet from the UE's address to 8.8.8.8 on `dport`.
fn udp_to(dport: u16, tag: u8) -> Vec<u8> {
    let mut p = vec![
        0x45, 0, 0, 29, 0, tag, 0, 0, 64, 17, 0, 0, 10, 45, 0, 2, 8, 8, 8, 8,
    ];
    p.extend_from_slice(&40000u16.to_be_bytes());
    p.extend_from_slice(&dport.to_be_bytes());
    p.extend_from_slice(&[0, 9, 0, 0, tag]);
    p
}

/// Send `packet` up from the UE and return the SGW TEID of the G-PDU it
/// leaves the eNB in, checking the payload is the packet unchanged.
async fn uplink_teid(h: &mut Harness, sgw: &tokio::net::UdpSocket, packet: &[u8]) -> u32 {
    h.ip_ul.send(packet.to_vec()).await.unwrap();
    let mut buf = vec![0u8; 2048];
    let (n, from) = tokio::time::timeout(Duration::from_secs(5), sgw.recv_from(&mut buf))
        .await
        .expect("no G-PDU at the SGW")
        .unwrap();
    assert_eq!(from, h.enb_s1u);
    assert_eq!(&buf[..2], &[0x30, 0xFF], "G-PDU");
    assert_eq!(&buf[8..n], packet, "the packet unchanged");
    u32::from_be_bytes(buf[4..8].try_into().unwrap())
}

/// Send a G-PDU on the eNB's `teid` and check it reaches the UE's IP stack.
async fn downlink_on(h: &mut Harness, sgw: &tokio::net::UdpSocket, teid: u32, tag: u8) {
    let reply = vec![
        0x45, 0, 0, 24, 0, tag, 0, 0, 64, 1, 0, 0, 8, 8, 8, 8, 10, 45, 0, 2, 0, 0, 0, tag,
    ];
    let mut g = vec![0x30, 0xFF, 0x00, reply.len() as u8];
    g.extend_from_slice(&teid.to_be_bytes());
    g.extend_from_slice(&reply);
    sgw.send_to(&g, h.enb_s1u).await.unwrap();
    let got = tokio::time::timeout(Duration::from_secs(5), h.ip_dl.recv())
        .await
        .expect("no downlink packet at the UE")
        .unwrap();
    assert_eq!(got, reply, "downlink on TEID {teid:#x}");
}

/// #283: a dedicated EPS bearer end to end. E-RAB SETUP REQUEST carries
/// ACTIVATE DEDICATED EPS BEARER CONTEXT REQUEST with a TFT for UDP to port
/// 5060; the eNB adds the DRB (TS 36.413 §8.2.1.2) and answers with its S1-U
/// endpoint, the UE accepts (TS 24.301 §6.4.2.3) and from then on sends what
/// the TFT matches on the dedicated bearer -- so it lands on the dedicated
/// SGW TEID -- and everything else on the default one. E-RAB MODIFY with
/// MODIFY EPS BEARER CONTEXT REQUEST replaces the filter (§6.4.3.3, port
/// 5061); E-RAB RELEASE with DEACTIVATE EPS BEARER CONTEXT REQUEST releases
/// the DRB (TS 36.331 §5.3.10.2) and the bearer (§6.4.4.3), and port 5061
/// traffic falls back to the default bearer.
#[tokio::test]
async fn a_dedicated_bearer_carries_the_uplink_its_tft_matches() {
    const DEFAULT_SGW_TEID: u32 = 0x1234;
    const DEDICATED_SGW_TEID: u32 = 0x5678;
    let sgw_ip = Ipv4Addr::new(127, 0, 0, 78);
    let sgw = tokio::net::UdpSocket::bind((sgw_ip, 2152)).await.unwrap();
    let mut h = start().await;
    let (default_teid, mut mme) = attach(&mut h, sgw_ip).await;
    assert_eq!(
        uplink_teid(&mut h, &sgw, &udp_to(5060, 0)).await,
        DEFAULT_SGW_TEID
    );

    // TS 24.008 §10.5.6.12: create new TFT, one filter, id 1, uplink only,
    // precedence 0x10: protocol UDP (0x30), single remote port 5060 (0x50).
    let tft = [0x21, 0x21, 0x10, 0x05, 0x30, 0x11, 0x50, 0x13, 0xC4];
    // TS 24.301 Table 8.3.3.1: EBI 6, PTI 0, linked EBI 5, EPS QoS QCI 5, TFT.
    let mut activate = vec![0x62, 0x00, 0xC5, 0x05, 0x01, 0x05, tft.len() as u8];
    activate.extend_from_slice(&tft);
    let nas = mme.protect(&activate, 2);
    h.to_enb
        .send(erab_setup_request(
            4242,
            h.enb_ue_id,
            6,
            sgw_ip,
            DEDICATED_SGW_TEID,
            &nas,
        ))
        .unwrap();
    let (answer, accept) = erab_answer_and_nas(&mut h, &mut mme).await;
    let SuccessfulOutcomeValue::Id_E_RABSetup(resp) = answer else {
        panic!("expected E-RAB SETUP RESPONSE, got {answer:?}");
    };
    let mut setup = None;
    for ie in resp.protocol_i_es.0 {
        match ie.value {
            E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABSetupListBearerSURes(l) => {
                let E_RABSetupListBearerSURes_EntryValue::Id_E_RABSetupItemBearerSURes(i) =
                    &l.0[0].value;
                setup = Some((
                    i.e_rab_id.0,
                    i.transport_layer_address.0.len(),
                    i.gtp_teid.0.clone(),
                ));
            }
            E_RABSetupResponseProtocolIEs_EntryValue::Id_E_RABFailedToSetupListBearerSURes(l) => {
                panic!("E-RAB failed to set up: {l:?}")
            }
            _ => {}
        }
    }
    let (id, addr_bits, teid) = setup.expect("E-RAB Setup List");
    assert_eq!((id, addr_bits), (6, 32));
    let dedicated_teid = u32::from_be_bytes(teid.try_into().unwrap());
    assert_ne!(dedicated_teid, 0);
    assert_ne!(dedicated_teid, default_teid, "a TEID of its own");
    assert_eq!(
        &accept[..3],
        &[0x62, 0x00, 0xC6],
        "ACTIVATE DEDICATED EPS BEARER CONTEXT ACCEPT for EBI 6"
    );

    // The TFT picks the bearer: UDP to 5060 rides EBI 6, the rest EBI 5.
    for i in 1..4u8 {
        assert_eq!(
            uplink_teid(&mut h, &sgw, &udp_to(5060, i)).await,
            DEDICATED_SGW_TEID
        );
        assert_eq!(
            uplink_teid(&mut h, &sgw, &udp_to(5061, i)).await,
            DEFAULT_SGW_TEID
        );
    }
    downlink_on(&mut h, &sgw, dedicated_teid, 1).await;
    downlink_on(&mut h, &sgw, default_teid, 2).await;

    // §6.4.3: replace packet filter 1 with one for port 5061.
    let tft = [0x81, 0x21, 0x10, 0x05, 0x30, 0x11, 0x50, 0x13, 0xC5];
    let mut modify = vec![0x62, 0x00, 0xC9, 0x36, tft.len() as u8];
    modify.extend_from_slice(&tft);
    let nas = mme.protect(&modify, 2);
    h.to_enb
        .send(erab_modify_request(4242, h.enb_ue_id, 6, &nas))
        .unwrap();
    let (answer, accept) = erab_answer_and_nas(&mut h, &mut mme).await;
    let SuccessfulOutcomeValue::Id_E_RABModify(resp) = answer else {
        panic!("expected E-RAB MODIFY RESPONSE, got {answer:?}");
    };
    assert!(
        resp.protocol_i_es.0.iter().any(|ie| matches!(
            &ie.value,
            E_RABModifyResponseProtocolIEs_EntryValue::Id_E_RABModifyListBearerModRes(l)
                if matches!(&l.0[0].value,
                    E_RABModifyListBearerModRes_EntryValue::Id_E_RABModifyItemBearerModRes(i)
                        if i.e_rab_id.0 == 6)
        )),
        "E-RAB 6 modified: {resp:?}"
    );
    assert_eq!(
        &accept[..3],
        &[0x62, 0x00, 0xCA],
        "MODIFY EPS BEARER CONTEXT ACCEPT"
    );
    assert_eq!(
        uplink_teid(&mut h, &sgw, &udp_to(5061, 4)).await,
        DEDICATED_SGW_TEID
    );
    assert_eq!(
        uplink_teid(&mut h, &sgw, &udp_to(5060, 4)).await,
        DEFAULT_SGW_TEID
    );

    // §6.4.4: deactivate EBI 6 (cause #36 regular deactivation).
    let nas = mme.protect(&[0x62, 0x00, 0xCD, 36], 2);
    h.to_enb
        .send(erab_release_command(4242, h.enb_ue_id, 6, &nas))
        .unwrap();
    let (answer, accept) = erab_answer_and_nas(&mut h, &mut mme).await;
    let SuccessfulOutcomeValue::Id_E_RABRelease(resp) = answer else {
        panic!("expected E-RAB RELEASE RESPONSE, got {answer:?}");
    };
    assert!(
        resp.protocol_i_es.0.iter().any(|ie| matches!(
            &ie.value,
            E_RABReleaseResponseProtocolIEs_EntryValue::Id_E_RABReleaseListBearerRelComp(l)
                if matches!(&l.0[0].value,
                    E_RABReleaseListBearerRelComp_EntryValue::Id_E_RABReleaseItemBearerRelComp(i)
                        if i.e_rab_id.0 == 6)
        )),
        "E-RAB 6 released: {resp:?}"
    );
    assert_eq!(
        &accept[..3],
        &[0x62, 0x00, 0xCE],
        "DEACTIVATE EPS BEARER CONTEXT ACCEPT"
    );
    for i in 5..7u8 {
        assert_eq!(
            uplink_teid(&mut h, &sgw, &udp_to(5061, i)).await,
            DEFAULT_SGW_TEID
        );
    }
    // The dedicated tunnel is gone: a G-PDU on its TEID reaches nobody, so
    // the next packet at the UE is the one on the default bearer's.
    let mut g = vec![0x30, 0xFF, 0x00, 4];
    g.extend_from_slice(&dedicated_teid.to_be_bytes());
    g.extend_from_slice(&[0x45, 0, 0, 4]);
    sgw.send_to(&g, h.enb_s1u).await.unwrap();
    downlink_on(&mut h, &sgw, default_teid, 3).await;
}

/// #356: the periodic tracking area update end to end. The ATTACH ACCEPT
/// sets T3412 to 2 s and a GUTI; released to RRC_IDLE, the UE's NAS starts
/// T3412 (TS 24.301 §5.3.5), and on expiry RRC establishes a connection for a
/// TRACKING AREA UPDATE REQUEST it built itself: integrity protected, not
/// ciphered, "periodic updating", the Old GUTI, the last visited TAI and the
/// EPS bearer context status of bearer 5 (`20 00`). The TAU ACCEPT with a new
/// GUTI is acknowledged with TRACKING AREA UPDATE COMPLETE, ciphered.
#[tokio::test]
async fn t3412_expiry_sends_a_periodic_tracking_area_update() {
    // GUTI (IEI 50): 999/70, MME group 0x0002, MME code 0x01, M-TMSI 0xC0000001.
    const GUTI: [u8; 13] = [
        0x50, 0x0B, 0xF6, 0x99, 0xF9, 0x07, 0x00, 0x02, 0x01, 0xC0, 0x00, 0x00, 0x01,
    ];
    let mut h = start().await;
    // T3412 0x01: unit 000 (2 seconds) x 1.
    let (_, mut mme) = attach_with(&mut h, Ipv4Addr::new(127, 0, 0, 2), |m| {
        attach_accept_with(m, 0x01, &GUTI)
    })
    .await;
    h.to_enb.send(release_command(4242, h.enb_ue_id)).unwrap();
    expect_release_complete(&mut h.mme_rx).await;

    let (enb_ue_id, tau) = initial_ue_message(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(tau[0], 0x17, "integrity protected, not ciphered (§4.4.5)");
    let plain = mme.unprotect(&tau);
    let mut expect = vec![0x07, 0x48, 0x03];
    expect.extend_from_slice(&GUTI[1..]);
    expect.extend_from_slice(&[0x52, 0x99, 0xF9, 0x07, 0x00, 0x01]);
    expect.extend_from_slice(&[0x57, 0x02, 0x20, 0x00, 0xE0]);
    assert_eq!(plain, expect, "KSI 0, periodic updating, Old GUTI, TAI, EBCS");

    // TRACKING AREA UPDATE ACCEPT, "TA updated", new GUTI M-TMSI 0xC0000002.
    let mut accept = vec![0x07, 0x49, 0x00];
    accept.extend_from_slice(&GUTI[..12]);
    accept.push(0x02);
    let accept = mme.protect(&accept, 2);
    h.to_enb.send(dl_nas(6262, enb_ue_id, &accept)).unwrap();
    let (mme_id, complete) = uplink_nas(next_s1(&mut h.mme_rx).await.1);
    assert_eq!(mme_id, 6262);
    assert_eq!(complete[0] >> 4, 2, "integrity protected and ciphered");
    assert_eq!(mme.unprotect(&complete), vec![0x07, 0x4A]);
}
