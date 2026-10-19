//! The real LTE UE task against the real eNB radio task over RLS on loopback,
//! with the MME side of S1 played by the test.
//!
//! What it proves end to end, with no shared code path between the two ends
//! other than the protocol: the UE selects the cell from the eNB's broadcast
//! system information (TS 36.304 §5.2.3), establishes an RRC connection
//! (TS 36.331 §5.3.3) whose SetupComplete arrives over SRB1 PDCP, and its NAS
//! reaches the MME in INITIAL UE MESSAGE (TS 36.413 §8.6.2.1); NAS then flows
//! both ways (§8.6.2.2/§8.6.2.3 ↔ §5.6.1/§5.6.2), and UE CONTEXT RELEASE
//! COMMAND returns the UE to RRC_IDLE (§8.3.3, §5.3.8).
//!
//! The NAS message is an opaque, correctly-encoded EPS ATTACH REQUEST used as
//! a TEST INPUT only: the product UE has no EMM until increment 3.

use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use nextgsim_common::Plmn;
use nextgsim_enb::radio::{run_operational, RlsRadio};
use nextgsim_enb::s1_task::{S1Message, S1Transport};
use nextgsim_lte_rrc::msg::EstablishmentCauseValue;
use nextgsim_lte_ue::config::LteUeConfig;
use nextgsim_lte_ue::nas::{InitialNas, NasLayer, RrcIndication};
use nextgsim_lte_ue::rrc::UeRrcState;
use nextgsim_s1ap::codec::*;
use tokio::sync::{mpsc, watch};

/// TS 24.301 §8.2.4 ATTACH REQUEST, plain NAS (security header type 0):
/// EMM PD 0x07, type 0x41, EPS attach type 1 (EPS attach) with NAS KSI 7
/// (no key), EPS mobile identity = IMSI 999700000000001, UE network
/// capability (EEA0-2, EIA0-2), and an ESM message container holding a PDN
/// CONNECTIVITY REQUEST (PTI 1, request type initial, PDN type IPv4).
const ATTACH_REQUEST: &[u8] = &[
    0x07, 0x41, // EMM, ATTACH REQUEST
    0x71, // NAS KSI 7 | EPS attach type 1
    // EPS mobile identity, LV: odd-length IMSI 999700000000001, digit 1 in the
    // first octet with odd/even = 1 and type 001, then digit pairs low nibble
    // first (TS 24.301 §9.9.3.12, TS 24.008 §10.5.1.4).
    0x08, 0x99, 0x99, 0x07, 0x00, 0x00, 0x00, 0x00, 0x10, 0x02, 0xE0,
    0xE0, // UE network capability: EEA0-2, EIA0-2
    0x00, 0x04, 0x02, 0x01, 0xD0, 0x11, // ESM container: PDN CONNECTIVITY REQUEST
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

#[derive(Default)]
struct NasLog {
    downlink: Vec<Vec<u8>>,
    indications: Vec<RrcIndication>,
    asked: bool,
}

struct TestNas(Arc<Mutex<NasLog>>);

impl NasLayer for TestNas {
    fn initial_nas(&mut self) -> Option<InitialNas> {
        let mut l = self.0.lock().unwrap();
        if l.asked {
            return None; // one attach attempt
        }
        l.asked = true;
        Some(InitialNas {
            nas: ATTACH_REQUEST.to_vec(),
            cause: EstablishmentCauseValue::MoSignalling,
            s_tmsi: None,
        })
    }
    fn on_downlink_nas(&mut self, nas: &[u8]) -> Vec<Vec<u8>> {
        self.0.lock().unwrap().downlink.push(nas.to_vec());
        vec![vec![0x07, 0x53, 0x08]] // an AUTHENTICATION RESPONSE-shaped reply
    }
    fn on_rrc_indication(&mut self, ind: RrcIndication) {
        self.0.lock().unwrap().indications.push(ind);
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

fn release(mme: u32, enb: u32) -> Vec<u8> {
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

async fn wait_state(rx: &mut watch::Receiver<UeRrcState>, f: impl Fn(UeRrcState) -> bool) {
    tokio::time::timeout(Duration::from_secs(10), async {
        while !f(*rx.borrow_and_update()) {
            rx.changed().await.unwrap();
        }
    })
    .await
    .expect("UE did not reach the expected RRC state");
}

#[tokio::test]
async fn ue_attach_request_reaches_the_mme_and_nas_flows_both_ways() {
    let mut cfg = nextgsim_enb::load_and_validate_enb_config(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../config/enb.yaml"
    ))
    .unwrap();
    cfg.radio.link_ip = IpAddr::V4(Ipv4Addr::LOCALHOST);
    cfg.radio.rls_port = 0; // any free port
    cfg.radio.si_broadcast_period_ms = 40;
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
    };
    let log = Arc::new(Mutex::new(NasLog::default()));
    let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let (state_tx, mut state_rx) = watch::channel(UeRrcState::NoCell);
    let (_stop_tx, stop_rx) = watch::channel(false);
    let nas = Box::new(TestNas(Arc::clone(&log)));
    tokio::spawn(async move {
        nextgsim_lte_ue::task::run(&ue_cfg, nas, socket, state_tx, stop_rx).await
    });

    // 1. INITIAL UE MESSAGE carrying the UE's ATTACH REQUEST, on a UE stream.
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
    let mut cgi = None;
    for ie in m.protocol_i_es.0 {
        match ie.value {
            InitialUEMessageProtocolIEs_EntryValue::Id_eNB_UE_S1AP_ID(v) => enb_ue_id = Some(v.0),
            InitialUEMessageProtocolIEs_EntryValue::Id_NAS_PDU(v) => nas = Some(v.0),
            InitialUEMessageProtocolIEs_EntryValue::Id_EUTRAN_CGI(c) => {
                cgi = Some(
                    c.cell_id
                        .0
                        .iter()
                        .fold(0u32, |a, b| (a << 1) | u32::from(*b)),
                )
            }
            _ => {}
        }
    }
    assert_eq!(nas.as_deref(), Some(ATTACH_REQUEST));
    assert_eq!(cgi, Some(cfg.cell_identity()));
    let enb_ue_id = enb_ue_id.unwrap();
    wait_state(&mut state_rx, |s| matches!(s, UeRrcState::Connected { .. })).await;

    // 2. DOWNLINK NAS TRANSPORT reaches the UE's NAS; its reply comes back as
    //    UPLINK NAS TRANSPORT naming the MME UE S1AP ID the MME chose.
    to_enb
        .send(dl_nas(4242, enb_ue_id, &[0x07, 0x52, 0x00]))
        .unwrap();
    let (_, pdu) = next_s1(&mut mme_rx).await;
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
    assert_eq!(got, (Some(4242), Some(vec![0x07, 0x53, 0x08])));
    assert_eq!(log.lock().unwrap().downlink, vec![vec![0x07, 0x52, 0x00]]);

    // 3. UE CONTEXT RELEASE COMMAND: COMPLETE to the MME, RRC_IDLE at the UE.
    to_enb.send(release(4242, enb_ue_id)).unwrap();
    let (_, pdu) = next_s1(&mut mme_rx).await;
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
    wait_state(&mut state_rx, |s| matches!(s, UeRrcState::Camped { .. })).await;
    assert_eq!(
        log.lock().unwrap().indications,
        vec![RrcIndication::Connected, RrcIndication::Released]
    );
}
