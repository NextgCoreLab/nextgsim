//! The eNB's user plane: per-E-RAB DRBs (TS 36.323 PDCP over TS 36.322 RLC UM,
//! on RLS) bridged to S1-U GTP-U tunnels (TS 29.281, TS 36.414).
//!
//! Pure, like [`crate::rrc`]: RLS data PDUs and GTP-U datagrams in, datagrams
//! out. [`crate::radio`] owns the sockets.
//!
//! - **Uplink**: an RLS data PDU on a DRB's LCID is an RLC UMD PDU; RLC UM
//!   reassembles SDUs ([`LteUmRx`]), PDCP deciphers them ([`LteDrbPdcp`]), and
//!   each IP packet goes to the SGW as a G-PDU on the SGW's S1-U TEID from the
//!   E-RAB's INITIAL CONTEXT SETUP REQUEST item (TS 36.413 §9.1.4.1).
//! - **Downlink**: a G-PDU on one of this eNB's TEIDs (the ones INITIAL
//!   CONTEXT SETUP RESPONSE named, §9.1.4.2) is ciphered by the DRB's PDCP and
//!   framed by RLC UM for the UE.
//! - **Path management** (TS 29.281 §7.2): an Echo Request is answered with
//!   an Echo Response carrying the Recovery IE with restart counter 0 (§8.2:
//!   "shall be set to 0 by the sending entity").
//! - **Error handling** (§7.3.1): a G-PDU for a TEID with no tunnel is
//!   answered with an Error Indication naming that TEID and this eNB's
//!   address, so the SGW can release the stale tunnel.
//!
//! G-PDUs are sent without a sequence number: S1-U carries no in-order
//! delivery requirement (TS 29.281 §5.1, the S flag is optional for G-PDUs).

use std::collections::HashMap;
use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use bytes::Bytes;
use nextgsim_gtp::codec::{GtpHeader, GtpMessageType};
use nextgsim_gtp::tunnel::GTP_U_PORT;
use nextgsim_pdcp::lte_drb::drb_bearer;
use nextgsim_pdcp::lte_srb::DIRECTION_DOWNLINK;
use nextgsim_pdcp::LteDrbPdcp;
use nextgsim_rlc::lte_um::{LteUmRx, LteUmTx};

/// `t-Reordering` of the DRBs, as configured in RRCConnectionReconfiguration
/// (`nextgsim_lte_rrc::msg`'s DRB-ToAddMod: ms35).
pub const DRB_T_REORDERING: Duration = Duration::from_millis(35);

/// One E-RAB's user plane, as RRC hands it over once the UE has completed the
/// RRCConnectionReconfiguration that set up its DRB.
#[derive(Clone, PartialEq, Eq)]
pub struct UpBearer {
    /// E-RAB ID.
    pub e_rab_id: u8,
    /// The DRB's logical channel identity (3..=10).
    pub lcid: u8,
    /// This eNB's S1-U TEID for the E-RAB.
    pub enb_teid: u32,
    /// The SGW's S1-U endpoint.
    pub sgw: SocketAddr,
    /// The SGW's S1-U TEID.
    pub sgw_teid: u32,
    /// The ciphering algorithm (EEA0..3).
    pub eea: u8,
    /// `K_UPenc`. Never printed.
    pub k_up_enc: [u8; 16],
}

impl std::fmt::Debug for UpBearer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpBearer")
            .field("e_rab_id", &self.e_rab_id)
            .field("lcid", &self.lcid)
            .field("enb_teid", &self.enb_teid)
            .field("sgw", &self.sgw)
            .field("sgw_teid", &self.sgw_teid)
            .field("eea", &self.eea)
            .finish_non_exhaustive()
    }
}

/// The SGW's S1-U endpoint from the E-RAB item's Transport Layer Address
/// (TS 36.413 §9.2.2.1: 32 bits IPv4, 128 bits IPv6, or 160 bits both, IPv4
/// first) on the GTP-U port.
pub fn sgw_endpoint(address: &[u8]) -> Option<SocketAddr> {
    let ip = match address.len() {
        4 | 20 => IpAddr::from(<[u8; 4]>::try_from(&address[..4]).ok()?),
        16 => IpAddr::from(<[u8; 16]>::try_from(address).ok()?),
        _ => return None,
    };
    Some(SocketAddr::new(ip, GTP_U_PORT))
}

#[derive(Debug)]
struct Drb {
    e_rab_id: u8,
    pdcp: LteDrbPdcp,
    rlc_tx: LteUmTx,
    rlc_rx: LteUmRx,
    sgw: SocketAddr,
    sgw_teid: u32,
}

/// What a received S1-U datagram becomes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Downlink {
    /// An RLC UMD PDU for `ue` on the DRB `lcid`.
    ToUe {
        /// RLS UE id.
        ue: u32,
        /// The DRB's LCID.
        lcid: u8,
        /// The RLC PDU.
        pdu: Vec<u8>,
    },
    /// A GTP-U reply (Echo Response, Error Indication) to `to`.
    Reply {
        /// Where.
        to: SocketAddr,
        /// The datagram.
        datagram: Vec<u8>,
    },
    /// Nothing to do.
    None,
}

/// The eNB's user-plane entities, keyed by RLS UE id and LCID.
#[derive(Debug)]
pub struct EnbUserPlane {
    drbs: HashMap<(u32, u8), Drb>,
    by_teid: HashMap<u32, (u32, u8)>,
    local_ip: IpAddr,
}

impl EnbUserPlane {
    /// An empty user plane whose S1-U address (for Error Indications) is
    /// `local_ip`.
    pub fn new(local_ip: IpAddr) -> Self {
        Self {
            drbs: HashMap::new(),
            by_teid: HashMap::new(),
            local_ip,
        }
    }

    /// Number of DRBs established.
    pub fn len(&self) -> usize {
        self.drbs.len()
    }

    /// Whether no DRB is established.
    pub fn is_empty(&self) -> bool {
        self.drbs.is_empty()
    }

    /// Establish the DRBs of `ue`.
    pub fn install(&mut self, ue: u32, bearers: &[UpBearer]) {
        for b in bearers {
            let pdcp =
                match LteDrbPdcp::new(drb_bearer(b.lcid), DIRECTION_DOWNLINK, b.eea, b.k_up_enc) {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("UE {ue}: E-RAB {}: {e}", b.e_rab_id);
                        continue;
                    }
                };
            log::info!(
                "UE {ue}: E-RAB {} on LCID {}: S1-U eNB TEID {:#x} <-> SGW {} TEID {:#x}",
                b.e_rab_id,
                b.lcid,
                b.enb_teid,
                b.sgw,
                b.sgw_teid
            );
            self.by_teid.insert(b.enb_teid, (ue, b.lcid));
            self.drbs.insert(
                (ue, b.lcid),
                Drb {
                    e_rab_id: b.e_rab_id,
                    pdcp,
                    rlc_tx: LteUmTx::new(),
                    rlc_rx: LteUmRx::new(DRB_T_REORDERING),
                    sgw: b.sgw,
                    sgw_teid: b.sgw_teid,
                },
            );
        }
    }

    /// Release every DRB of `ue` (its RRC context is gone).
    pub fn release(&mut self, ue: u32) {
        let before = self.drbs.len();
        self.drbs.retain(|(u, _), _| *u != ue);
        self.by_teid.retain(|_, (u, _)| *u != ue);
        if self.drbs.len() != before {
            log::info!("UE {ue}: user plane released");
        }
    }

    /// Release the DRBs of `ue` on `lcids` and their S1-U tunnels (E-RAB
    /// RELEASE, TS 36.413 §8.2.3; #283). The UE's other DRBs stay.
    pub fn release_drbs(&mut self, ue: u32, lcids: &[u8]) {
        for lcid in lcids {
            if let Some(d) = self.drbs.remove(&(ue, *lcid)) {
                log::info!(
                    "UE {ue}: E-RAB {} on LCID {lcid}: user plane released",
                    d.e_rab_id
                );
            }
        }
        self.by_teid
            .retain(|_, (u, l)| !(*u == ue && lcids.contains(l)));
    }

    fn to_sgw(drb: &mut Drb, sdus: Vec<Vec<u8>>, ue: u32, out: &mut Vec<(SocketAddr, Vec<u8>)>) {
        for sdu in sdus {
            match drb.pdcp.receive(&sdu) {
                Ok((ip, _)) => {
                    let g = GtpHeader::g_pdu(drb.sgw_teid, Bytes::from(ip));
                    out.push((drb.sgw, g.encode().to_vec()));
                }
                Err(e) => log::warn!("UE {ue}: E-RAB {}: uplink PDCP: {e}", drb.e_rab_id),
            }
        }
    }

    /// An RLS data PDU from `ue` on `lcid`: the G-PDUs it produces.
    pub fn uplink(
        &mut self,
        ue: u32,
        lcid: u8,
        rlc_pdu: &[u8],
        now: Instant,
    ) -> Vec<(SocketAddr, Vec<u8>)> {
        let Some(drb) = self.drbs.get_mut(&(ue, lcid)) else {
            log::debug!("UE {ue}: data PDU on LCID {lcid} with no DRB, discarded");
            return Vec::new();
        };
        let sdus = match drb.rlc_rx.receive(rlc_pdu, now) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("UE {ue}: LCID {lcid}: RLC UM: {e}");
                return Vec::new();
            }
        };
        let mut out = Vec::new();
        Self::to_sgw(drb, sdus, ue, &mut out);
        out
    }

    /// The earliest RLC t-Reordering expiry.
    pub fn next_deadline(&self) -> Option<Instant> {
        self.drbs
            .values()
            .filter_map(|d| d.rlc_rx.next_deadline())
            .min()
    }

    /// Run due t-Reordering timers; the G-PDUs released.
    pub fn on_timer(&mut self, now: Instant) -> Vec<(SocketAddr, Vec<u8>)> {
        let mut out = Vec::new();
        for ((ue, _), drb) in &mut self.drbs {
            let sdus = drb.rlc_rx.on_timer(now);
            Self::to_sgw(drb, sdus, *ue, &mut out);
        }
        out
    }

    /// A datagram received on S1-U from `from`.
    pub fn downlink(&mut self, datagram: &[u8], from: SocketAddr) -> Downlink {
        let h = match GtpHeader::decode(datagram) {
            Ok(h) => h,
            Err(e) => {
                log::debug!("undecodable S1-U datagram from {from}: {e}");
                return Downlink::None;
            }
        };
        match h.message_type {
            GtpMessageType::GPdu => {
                let Some(&(ue, lcid)) = self.by_teid.get(&h.teid) else {
                    log::warn!(
                        "G-PDU for unknown TEID {:#x} from {from}; Error Indication",
                        h.teid
                    );
                    return Downlink::Reply {
                        to: from,
                        datagram: GtpHeader::error_indication(h.teid, self.local_ip)
                            .encode()
                            .to_vec(),
                    };
                };
                let drb = self
                    .drbs
                    .get_mut(&(ue, lcid))
                    .expect("by_teid and drbs agree");
                let (pdcp, _) = drb.pdcp.transmit(&h.payload);
                Downlink::ToUe {
                    ue,
                    lcid,
                    pdu: drb.rlc_tx.transmit(&pdcp),
                }
            }
            GtpMessageType::EchoRequest => {
                let mut r = GtpHeader::echo_response(0);
                if let Some(seq) = h.sequence_number {
                    r = r.with_sequence_number(seq);
                }
                Downlink::Reply {
                    to: from,
                    datagram: r.with_recovery(0).encode().to_vec(),
                }
            }
            GtpMessageType::ErrorIndication => {
                log::warn!(
                    "S1-U Error Indication from {from} for TEID {:?}",
                    h.error_indication_teid()
                );
                Downlink::None
            }
            other => {
                log::debug!("S1-U {other:?} from {from} ignored");
                Downlink::None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_pdcp::lte_srb::DIRECTION_UPLINK;
    use std::net::Ipv4Addr;

    const KEY: [u8; 16] = [7; 16];

    fn bearer() -> UpBearer {
        UpBearer {
            e_rab_id: 5,
            lcid: 3,
            enb_teid: 0x0000_0001,
            sgw: "10.0.0.9:2152".parse().unwrap(),
            sgw_teid: 0xABCD_0001,
            eea: 2,
            k_up_enc: KEY,
        }
    }

    fn ue_side() -> (LteDrbPdcp, LteUmTx, LteUmRx) {
        (
            LteDrbPdcp::new(drb_bearer(3), DIRECTION_UPLINK, 2, KEY).unwrap(),
            LteUmTx::new(),
            LteUmRx::new(DRB_T_REORDERING),
        )
    }

    #[test]
    fn sgw_endpoint_reads_ipv4_ipv6_and_both() {
        assert_eq!(
            sgw_endpoint(&[10, 0, 0, 9]),
            Some("10.0.0.9:2152".parse().unwrap())
        );
        let mut both = vec![10, 0, 0, 9];
        both.extend([0u8; 16]);
        assert_eq!(sgw_endpoint(&both), Some("10.0.0.9:2152".parse().unwrap()));
        assert!(sgw_endpoint(&[0; 16]).unwrap().is_ipv6());
        assert_eq!(sgw_endpoint(&[1, 2, 3]), None);
    }

    #[test]
    fn uplink_ip_packet_becomes_a_g_pdu_on_the_sgw_teid() {
        let now = Instant::now();
        let mut up = EnbUserPlane::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        up.install(1, &[bearer()]);
        let (mut pdcp, mut tx, _) = ue_side();
        let ip = vec![0x45, 0, 0, 20, 1, 2, 3, 4];
        let out = up.uplink(1, 3, &tx.transmit(&pdcp.transmit(&ip).0), now);
        // TS 29.281 §5.1, by hand: flags 0x30 (version 1, PT 1, no E/S/PN),
        // type 0xFF (G-PDU), length 8, TEID 0xABCD0001, then the packet.
        let mut g = vec![0x30, 0xFF, 0x00, 0x08, 0xAB, 0xCD, 0x00, 0x01];
        g.extend_from_slice(&ip);
        assert_eq!(out, vec![("10.0.0.9:2152".parse().unwrap(), g)]);
    }

    #[test]
    fn downlink_g_pdu_reaches_the_ue_deciphered() {
        let now = Instant::now();
        let mut up = EnbUserPlane::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        up.install(1, &[bearer()]);
        let (mut pdcp, _, mut rx) = ue_side();
        let ip = vec![0x45, 9, 9, 9];
        let g = GtpHeader::g_pdu(1, Bytes::from(ip.clone())).encode();
        let Downlink::ToUe { ue, lcid, pdu } = up.downlink(&g, "10.0.0.9:2152".parse().unwrap())
        else {
            panic!("not delivered");
        };
        assert_eq!((ue, lcid), (1, 3));
        let sdus = rx.receive(&pdu, now).unwrap();
        assert_eq!(pdcp.receive(&sdus[0]).unwrap().0, ip);
    }

    #[test]
    fn unknown_teid_gets_an_error_indication_and_echo_gets_a_response() {
        let mut up = EnbUserPlane::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let from: SocketAddr = "10.0.0.9:2152".parse().unwrap();
        let g = GtpHeader::g_pdu(0x99, Bytes::from_static(b"x")).encode();
        let Downlink::Reply { to, datagram } = up.downlink(&g, from) else {
            panic!("no Error Indication");
        };
        assert_eq!(to, from);
        let ei = GtpHeader::decode(&datagram).unwrap();
        assert_eq!(ei.message_type, GtpMessageType::ErrorIndication);
        assert_eq!(ei.error_indication_teid(), Some(0x99));
        assert_eq!(
            ei.error_indication_peer(),
            Some(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)))
        );
        let echo = GtpHeader::echo_request(0).with_sequence_number(7).encode();
        let Downlink::Reply { datagram, .. } = up.downlink(&echo, from) else {
            panic!("no Echo Response");
        };
        let r = GtpHeader::decode(&datagram).unwrap();
        assert_eq!(r.message_type, GtpMessageType::EchoResponse);
        assert_eq!(r.sequence_number, Some(7));
        assert_eq!(r.recovery_restart_counter(), Some(0));
    }

    #[test]
    fn release_removes_the_tunnels() {
        let mut up = EnbUserPlane::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        up.install(1, &[bearer()]);
        up.release(1);
        assert!(up.is_empty());
        let g = GtpHeader::g_pdu(1, Bytes::from_static(b"x")).encode();
        assert!(matches!(
            up.downlink(&g, "10.0.0.9:2152".parse().unwrap()),
            Downlink::Reply { .. }
        ));
    }

    /// #283: releasing one E-RAB's DRB removes its tunnel only.
    #[test]
    fn release_drbs_keeps_the_other_bearers() {
        let mut up = EnbUserPlane::new(IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)));
        let dedicated = UpBearer {
            e_rab_id: 6,
            lcid: 4,
            enb_teid: 2,
            ..bearer()
        };
        up.install(1, &[bearer()]);
        up.install(1, &[dedicated]);
        up.release_drbs(1, &[4]);
        let from: SocketAddr = "10.0.0.9:2152".parse().unwrap();
        let g = |teid| GtpHeader::g_pdu(teid, Bytes::from_static(b"x")).encode();
        assert!(matches!(up.downlink(&g(2), from), Downlink::Reply { .. }));
        assert!(matches!(
            up.downlink(&g(1), from),
            Downlink::ToUe { ue: 1, lcid: 3, .. }
        ));
        assert!(!up.is_empty());
    }
}
