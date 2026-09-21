//! #56 — NTN SIB19 end to end: the gNB broadcasts its ephemeris, and the UE
//! **applies** a non-zero autonomous timing advance derived from it.
//!
//! Runs the REAL gNB RRC task, the REAL UE RRC task and the REAL UE RLS task in one
//! process, carrying a real UPER-encoded SIB19 across the seam between them:
//!
//! ```text
//! GnbConfig.ntn_config (ephemeris, Common TA, K_offset)
//!         │  sib19_params
//!         ▼
//! RrcTask::ntn_config  ──► broadcast_system_information
//!         │                  SystemInformation { sib19-v1700 }, real UPER
//!         ▼  RlsMessage::BroadcastRrc (BCCH-DL-SCH)
//! UE RrcTask::handle_downlink_rrc
//!         │  decode SIB19 → derive T_TA from ephemeris + GNSS position
//!         ▼  RlsMessage::ApplyNtnPrecompensation
//! UE RlsTask::ntn_precompensation  ──► apply_ntn_uplink_timing on every uplink
//!         ▼
//! last_uplink_timing_offset_us == -T_TA
//! ```
//!
//! # What this covers, and why each criterion needs an end-to-end test
//!
//! Issue #56's criteria 2, 3, 5 and 6 are all statements about state **crossing
//! tasks and being consumed**, which is exactly what the defect was: the NTN
//! parameters were stored on both ends and read by neither, so every unit test
//! about them passed while nothing was applied.
//!
//! * **Criterion 1 / 6** — the gNB encodes SIB19 from `self.ntn_config` and puts it
//!   on BCCH. `the_gnb_broadcasts_its_ephemeris_in_sib19` asserts the broadcast
//!   bytes decode to the configured ephemeris. The SIB19 *codec* is unit-tested in
//!   `nextgsim_rrc::procedures::system_information`; what needs this harness is that
//!   the stored configuration reaches the air rather than sitting in a field.
//! * **Criterion 2 / 3** — the UE reads the ephemeris and derives a **non-zero** TA
//!   from it. `the_ue_applies_a_non_zero_ta_derived_from_the_broadcast_ephemeris`
//!   asserts the applied value against a number computed **by hand from the
//!   geometry**, not against whatever the code produced.
//! * **Criterion 5** — the assertion is on the **applied** value: the UE RLS task's
//!   `last_uplink_timing_offset_us`, which is written by the transmit path itself.
//!   Deleting the read site in `apply_ntn_uplink_timing` leaves it `None` and the
//!   test fails; no log line is involved.
//!
//! # Why the geometry is what it is
//!
//! The satellite is at 600 km altitude directly above the UE, on the equator at
//! longitude 0, receding radially at 1 km/s. Directly overhead so the slant range is
//! exactly the altitude and the TA can be computed on paper; radially so the whole
//! velocity projects onto the line of sight and the Doppler can be too. An oblique
//! geometry would hide a dropped axis or a sign error behind a plausible-looking
//! number.
//!
//! The Common TA is deliberately NOT zero and deliberately NOT equal to the
//! service-link RTT. If it were zero, a UE that ignored the ephemeris and applied
//! only the broadcast Common TA would produce 0 and be caught — but so would a UE
//! that applied nothing at all, and the two are different defects. With
//! `common_ta_us = 4072` against a service-link RTT of ~4003 us, "applied only
//! ta-Common", "applied only the ephemeris" and "applied both" are three
//! distinguishable numbers, and the test pins the third.

use std::net::SocketAddr;
use std::time::Duration;

use nextgsim_common::config::{GnbConfig, NtnConfig, NtnEphemerisConfig, UeConfig};
use nextgsim_common::{OctetString, Plmn};
use nextgsim_gnb::tasks::{
    GnbTaskBase, RlsMessage as GnbRlsMessage, TaskMessage as GnbTaskMessage,
};
use nextgsim_gnb::RlsTask as GnbRlsTask;
use nextgsim_gnb::RrcTask as GnbRrcTask;
use nextgsim_gnb::Task as GnbTask;
use nextgsim_rls::RrcChannel;
use nextgsim_rrc::procedures::system_information::decode_system_information;
use nextgsim_ue::rls::{RlsTask as UeRlsTask, RlsTaskConfig};
use nextgsim_ue::rrc::RrcTask as UeRrcTask;
use nextgsim_ue::{
    RlsMessage as UeRlsMessage, Task as UeTask, TaskMessage as UeTaskMessage, UeTaskBase,
};
use tokio::sync::mpsc;

/// WGS84 equatorial radius, m. Only used to place the UE and the satellite at
/// plausible ECEF coordinates.
const EARTH_RADIUS_M: f64 = 6_378_137.0;

/// The satellite's altitude above the UE, m. 600 km: a representative LEO.
const ALTITUDE_M: f64 = 600_000.0;

/// The satellite's radial speed, m/s. Positive is receding.
const RANGE_RATE_M_S: f64 = 1000.0;

/// `common_ta_us` this cell broadcasts. See the module docs for why this value and
/// not zero.
const COMMON_TA_US: u64 = 4072;

/// `cellSpecificKoffset` this cell broadcasts, in slots.
const K_OFFSET: u16 = 478;

/// The UE's uplink carrier, Hz. S-band, matching the config default.
const UPLINK_CARRIER_HZ: f64 = 2e9;

/// Speed of light, m/s — the same constant `nextgsim_rrc`'s NTN maths uses.
const C_M_S: f64 = 299_792_458.0;

/// How long to wait for the loopback cell discovery both directions need.
///
/// Generous: this ecosystem has a recorded incident where socket binds under a
/// `tokio::time::timeout` starved on a loaded shared host, and the symptom was a
/// timeout rather than an assertion failure.
const DISCOVERY_TIMEOUT: Duration = Duration::from_secs(10);

/// The UE's GNSS position: on the equator at longitude 0, directly under the
/// satellite.
fn ue_gnss_position() -> [f64; 3] {
    [EARTH_RADIUS_M, 0.0, 0.0]
}

/// The **hand-computed** timing advance this configuration must produce, in
/// microseconds.
///
/// `T_TA = 2 x (600 km / c) + ta_common` (TS 38.300 §16.14.2.2 with §16.14.2.1's
/// "Common TA is ... the RTT between the RP and the NTN payload"):
/// 2 x 600000/299792458 s = 4002.77 us, plus 4072 us = **8074.77 us**.
///
/// Computed from the constants rather than written as a literal so the arithmetic is
/// auditable, but every input is fixed above — this is not a restatement of the
/// production expression, which also converts the ephemeris out of its 1.3 m wire
/// steps and reads `ta-Common` out of its 4.072 ns ones.
fn expected_ta_us() -> f64 {
    2.0 * ALTITUDE_M / C_M_S * 1e6 + COMMON_TA_US as f64
}

/// The **hand-computed** uplink Doppler pre-compensation, in Hz.
///
/// `f_d = -(range rate / c) x f_c = -(1000/299792458) x 2e9 = -6671.3 Hz`. Negative
/// because the satellite is receding: the link red-shifts the uplink, so the UE must
/// transmit higher for the signal to arrive on the nominal frequency.
fn expected_doppler_hz() -> f64 {
    -(RANGE_RATE_M_S / C_M_S) * UPLINK_CARRIER_HZ
}

/// An NTN cell configured with the geometry above.
fn ntn_gnb_config() -> GnbConfig {
    GnbConfig {
        nci: 0x0000_0001_0,
        gnb_id_length: 32,
        plmn: Plmn::new(1, 1, false),
        tac: 1,
        ntn_config: Some(NtnConfig {
            satellite_type: "LEO".to_string(),
            satellite_id: 1,
            // Not read by the SIB19 path -- the UE derives its own delay from the
            // ephemeris, which is the whole point -- but part of a coherent NTN block.
            propagation_delay_us: (ALTITUDE_M / C_M_S * 1e6) as u64,
            common_ta_us: COMMON_TA_US,
            k_offset: K_OFFSET,
            cell_center_lat: 0.0,
            cell_center_lon: 0.0,
            cell_radius_km: 500.0,
            earth_fixed: true,
            autonomous_ta: true,
            max_doppler_hz: 50_000.0,
            ephemeris: NtnEphemerisConfig {
                position_m: [EARTH_RADIUS_M + ALTITUDE_M, 0.0, 0.0],
                velocity_m_s: [RANGE_RATE_M_S, 0.0, 0.0],
            },
            ul_sync_validity_s: 30,
        }),
        ..Default::default()
    }
}

/// A UE with a valid GNSS position, as TS 38.300 §16.14.2.2 requires of a UE
/// connecting to an NTN cell.
fn ntn_ue_config() -> UeConfig {
    UeConfig {
        gnss_position_ecef_m: Some(ue_gnss_position()),
        ntn_uplink_carrier_hz: UPLINK_CARRIER_HZ,
        ..Default::default()
    }
}

/// A loopback address with a port the OS has just confirmed free.
async fn free_loopback_addr() -> SocketAddr {
    let probe = tokio::net::UdpSocket::bind("127.0.0.1:0")
        .await
        .expect("probe socket for a free port");
    probe.local_addr().expect("probe local address")
}

/// Takes the next broadcast the gNB's RRC task handed its RLS task, as
/// `(channel, pdu)`.
fn take_broadcast(
    rx: &mut mpsc::Receiver<GnbTaskMessage<GnbRlsMessage>>,
) -> Option<(RrcChannel, OctetString)> {
    while let Ok(msg) = rx.try_recv() {
        if let GnbTaskMessage::Message(GnbRlsMessage::BroadcastRrc {
            rrc_channel, data, ..
        }) = msg
        {
            return Some((rrc_channel, data));
        }
    }
    None
}

/// Drives the REAL gNB RRC task's broadcast and returns the `SystemInformation` PDU
/// it put on BCCH-DL-SCH.
///
/// `with_sib1 = true` so the SIB1 goes out too, which is the path a real cell takes —
/// the SI is the *second* BCCH-DL-SCH broadcast of the pair, and taking the second
/// rather than assuming the first also guards against SIB19 displacing SIB1.
async fn broadcast_system_information(config: GnbConfig) -> OctetString {
    let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, mut rls_rx, _sctp_rx) =
        GnbTaskBase::new(config, 32);
    let mut task = GnbRrcTask::new(task_base);
    task.broadcast_system_information(true).await;

    let (mib_channel, _) = take_broadcast(&mut rls_rx).expect("the MIB comes first");
    assert_eq!(mib_channel, RrcChannel::BcchBch);
    let (sib1_channel, _) = take_broadcast(&mut rls_rx).expect("then SIB1");
    assert_eq!(sib1_channel, RrcChannel::BcchDlSch);
    let (si_channel, si) = take_broadcast(&mut rls_rx)
        .expect("then the SystemInformation carrying SIB19; an NTN cell must broadcast one");
    assert_eq!(si_channel, RrcChannel::BcchDlSch);
    si
}

/// **Criteria 1 and 6.** The gNB reads its stored NTN configuration on the broadcast
/// path and puts the configured ephemeris, Common TA and validity on the air as real
/// UPER.
#[tokio::test]
async fn the_gnb_broadcasts_its_ephemeris_in_sib19() {
    let si = broadcast_system_information(ntn_gnb_config()).await;

    let decoded = decode_system_information(si.data()).expect("the broadcast SI must decode");
    let sib19 = decoded
        .sib19
        .expect("an NTN cell's SystemInformation must carry SIB19 (TS 38.300 §16.4)");
    let cfg = sib19
        .ntn_config
        .expect("and SIB19 must carry the serving cell's ntn-Config");

    // The ephemeris, read back in metres, must be where the config put the satellite
    // -- within the 1.3 m wire step.
    let position = cfg.ephemeris.position_m();
    assert!(
        (position[0] - (EARTH_RADIUS_M + ALTITUDE_M)).abs() <= 1.3,
        "broadcast satellite X {} m must be the configured {} m",
        position[0],
        EARTH_RADIUS_M + ALTITUDE_M
    );
    assert_eq!(position[1], 0.0);
    assert_eq!(position[2], 0.0);

    let velocity = cfg.ephemeris.velocity_m_s();
    assert!(
        (velocity[0] - RANGE_RATE_M_S).abs() <= 0.06,
        "broadcast satellite VX {} m/s must be the configured {RANGE_RATE_M_S} m/s",
        velocity[0]
    );

    // The Common TA, read back in microseconds through its 4.072 ns granularity.
    assert!(
        (cfg.ta_common_us() - COMMON_TA_US as f64).abs() < 0.01,
        "broadcast ta-Common {} us must be the configured {COMMON_TA_US} us; a \
         mis-scaled conversion here is wrong by a factor of ~245 and every round-trip \
         test would still pass",
        cfg.ta_common_us()
    );

    assert_eq!(cfg.cell_specific_k_offset, K_OFFSET);
    assert_eq!(
        cfg.ul_sync_validity_duration_s(),
        Some(30),
        "ntn-UlSyncValidityDuration must be the configured 30 s"
    );

    // And the broadcast is USABLE, not merely well-formed: the TA a UE would derive
    // from these exact bytes is the hand-computed one.
    assert!(
        (cfg.autonomous_ta_us(ue_gnss_position()) - expected_ta_us()).abs() < 0.1,
        "the TA derivable from the broadcast ephemeris ({} us) must be the \
         hand-computed {} us",
        cfg.autonomous_ta_us(ue_gnss_position()),
        expected_ta_us()
    );
}

/// A terrestrial cell must broadcast NO SIB19. An NTN IE on a TN cell would tell a
/// UE to advance its uplink for a satellite that does not exist.
#[tokio::test]
async fn a_terrestrial_cell_broadcasts_no_sib19() {
    let tn = GnbConfig {
        ntn_config: None,
        ..ntn_gnb_config()
    };
    let si = broadcast_system_information(tn).await;
    let decoded = decode_system_information(si.data()).expect("the SI must decode");
    assert!(
        decoded.sib19.is_none(),
        "a cell with no ntn_config must broadcast no SIB19"
    );
    assert!(
        decoded.sib2.is_some(),
        "and its SIB2/3/4 reselection broadcast must be unaffected"
    );
}

/// Runs the UE's RLS task against a live gNB RLS task over loopback UDP until the UE
/// has a real serving cell, then sends one uplink RRC PDU and stops.
///
/// Returns the stopped task, so the caller can assert what it APPLIED.
///
/// A real serving cell is required and not a shortcut: `handle_rrc_pdu_delivery`
/// returns early when there is none, so a UE that never discovered a cell sends no
/// uplink and records no timing — which would make this test pass vacuously whether
/// the pre-compensation was applied or not.
///
/// `precompensation` is installed before the uplink when `Some`, in the same order
/// the RRC task produces it: the install message is queued ahead of the uplink, and
/// the channel guarantees the order.
async fn send_one_uplink(ue_config: UeConfig, precompensation: Option<UeRlsMessage>) -> UeRlsTask {
    let gnb_addr = free_loopback_addr().await;

    // A live gNB RLS task, so the UE's cell discovery is real: the UE learns the
    // cell's address from a heartbeat ack, which is what populates the
    // `cell_addresses` map its uplink needs.
    let (gnb_base, _app_rx, _ngap_rx, mut gnb_rrc_rx, _gtp_rx, gnb_rls_rx, _sctp_rx) =
        GnbTaskBase::new(ntn_gnb_config(), 64);
    let mut gnb_rls = GnbRlsTask::with_bind_address(gnb_base, gnb_addr);
    tokio::spawn(async move { gnb_rls.run(gnb_rls_rx).await });

    let (ue_base, _ue_app_rx, _ue_nas_rx, mut ue_rrc_rx, ue_rls_rx) =
        UeTaskBase::new(ue_config, 64);
    let rls_tx = ue_base.rls_tx.clone();
    let mut ue_rls = UeRlsTask::new(
        ue_base,
        RlsTaskConfig {
            gnb_search_list: vec![gnb_addr],
            bind_address: Some("127.0.0.1:0".parse().expect("UE bind address")),
            heartbeat_interval: Duration::from_millis(100),
            heartbeat_threshold: Duration::from_millis(2000),
        },
    );

    // Run the UE RLS task in the background while discovery happens.
    let handle = tokio::spawn(async move {
        ue_rls.run(ue_rls_rx).await;
        ue_rls
    });

    // Wait for both halves of discovery, then make the found cell the serving one --
    // the step the UE's RRC task performs in a full run.
    tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = gnb_rrc_rx.recv().await {
            if let GnbTaskMessage::Message(nextgsim_gnb::tasks::RrcMessage::SignalDetected {
                ..
            }) = msg
            {
                return;
            }
        }
        panic!("gNB RRC channel closed before the UE was detected");
    })
    .await
    .expect("the gNB must discover the UE");

    let cell_id = tokio::time::timeout(DISCOVERY_TIMEOUT, async {
        while let Some(msg) = ue_rrc_rx.recv().await {
            if let UeTaskMessage::Message(nextgsim_ue::RrcMessage::SignalChanged {
                cell_id, ..
            }) = msg
            {
                return cell_id;
            }
        }
        panic!("UE RRC channel closed before a cell was found");
    })
    .await
    .expect("the UE must discover the cell");

    rls_tx
        .send(UeRlsMessage::AssignCurrentCell { cell_id })
        .await
        .expect("UE RLS task alive");

    if let Some(install) = precompensation {
        rls_tx.send(install).await.expect("UE RLS task alive");
    }
    rls_tx
        .send(UeRlsMessage::RrcPduDelivery {
            channel: RrcChannel::UlDcch,
            pdu_id: 0,
            pdu: OctetString::from_slice(&[0x01, 0x02, 0x03]),
        })
        .await
        .expect("UE RLS task alive");
    rls_tx.shutdown().await.expect("UE RLS task alive");

    handle.await.expect("the UE RLS task must stop cleanly")
}

/// **Criteria 2, 3 and 5 — the load-bearing test of this issue.**
///
/// The UE decodes the broadcast SIB19, derives a non-zero autonomous TA from the
/// ephemeris, and the value is **APPLIED** on the uplink transmit path: the
/// assertion is on `last_uplink_timing_offset_us`, which only the transmit path
/// writes.
#[tokio::test]
async fn the_ue_applies_a_non_zero_ta_derived_from_the_broadcast_ephemeris() {
    // A real gNB RRC task produces the real broadcast bytes.
    let si = broadcast_system_information(ntn_gnb_config()).await;

    // A real UE RRC task decodes them. This is the production entry point the RLS
    // task calls on a BCCH-DL-SCH delivery.
    let (rrc_base, _app_rx, _nas_rx, _rrc_rx, mut rrc_rls_rx) =
        UeTaskBase::new(ntn_ue_config(), 64);
    let mut ue_rrc = UeRrcTask::new(rrc_base);
    ue_rrc
        .handle_downlink_rrc(1, RrcChannel::BcchDlSch, si)
        .await;

    // The RRC task derived the pre-compensation and it matches the hand-computed
    // value.
    let timing = ue_rrc
        .ntn_timing()
        .expect("the UE must have derived a pre-compensation from the broadcast SIB19");
    assert!(
        (timing.applied_ta_us - expected_ta_us()).abs() < 0.1,
        "the derived T_TA {} us must be the hand-computed {} us (2 x 600 km one-way \
         + ta-Common {COMMON_TA_US} us)",
        timing.applied_ta_us,
        expected_ta_us()
    );
    assert!(
        (timing.applied_doppler_hz - expected_doppler_hz()).abs() < 1.0,
        "the derived Doppler pre-compensation {} Hz must be the hand-computed {} Hz",
        timing.applied_doppler_hz,
        expected_doppler_hz()
    );
    assert!(
        timing.applied_ta_us > 8000.0,
        "and it must be NON-ZERO: {} us",
        timing.applied_ta_us
    );
    // The ephemeris CONTRIBUTED. A UE that applied only the broadcast ta-Common
    // would land at 4072 us; the difference is the service-link RTT it computed from
    // the satellite's position and its own.
    assert!(
        timing.applied_ta_us - COMMON_TA_US as f64 > 4000.0,
        "the ephemeris must contribute ~4003 us on top of ta-Common; a T_TA of \
         {COMMON_TA_US} us would mean the ephemeris was never read"
    );

    // The RRC task SENT the derived values to the task that owns the transmitter.
    // Taking the real message off the channel is what makes the rest of this test
    // end-to-end: the value the transmitter applies below is the one the RRC task
    // put on the wire between them, not a copy the test constructed.
    let install = match rrc_rls_rx.try_recv() {
        Ok(UeTaskMessage::Message(msg @ UeRlsMessage::ApplyNtnPrecompensation { .. })) => msg,
        other => panic!(
            "the UE's RRC task must send ApplyNtnPrecompensation to the RLS task after \
             decoding SIB19; a derivation that never crosses this seam is the \
             stored-then-never-read defect issue #56 is about. Got {other:?}"
        ),
    };
    let UeRlsMessage::ApplyNtnPrecompensation {
        ta_us: sent_ta_us, ..
    } = &install
    else {
        unreachable!("matched above")
    };
    assert!(
        (sent_ta_us - expected_ta_us()).abs() < 0.1,
        "the T_TA sent across the seam ({sent_ta_us} us) must be the hand-computed {} us",
        expected_ta_us()
    );

    // Now run the REAL uplink path with that exact message installed ahead of an
    // uplink PDU, against a live gNB so the UE has a real serving cell.
    let ue_rls = send_one_uplink(ntn_ue_config(), Some(install)).await;

    // THE APPLIED VALUE. Written by the transmit path itself, not restated from the
    // configuration: if the read in `apply_ntn_uplink_timing` were deleted this would
    // be `None`.
    let applied_offset = ue_rls
        .last_uplink_timing_offset_us()
        .expect("the UE must have sent an uplink and recorded its transmit timing");
    assert!(
        (applied_offset + expected_ta_us()).abs() < 0.1,
        "the APPLIED uplink transmit offset {applied_offset} us must be \
         -{} us: a timing ADVANCE moves the transmission EARLIER than the downlink \
         frame boundary (TS 38.300 §16.14.2.1), so the offset is negative",
        expected_ta_us()
    );
    assert!(
        applied_offset < 0.0,
        "an advance must be negative; {applied_offset} us is a DELAY, which is the \
         opposite pre-compensation"
    );

    let installed = ue_rls
        .ntn_precompensation_applied()
        .expect("the pre-compensation must be installed on the transmit path");
    assert!((installed.ta_us - expected_ta_us()).abs() < 0.1);
    assert!((installed.doppler_hz - expected_doppler_hz()).abs() < 1.0);
    assert_eq!(installed.k_offset, K_OFFSET);
}

/// A terrestrial UE's uplink timing must be exactly unchanged: offset `0.0`, no
/// pre-compensation installed. This is what keeps every non-NTN scenario timed as it
/// was before issue #56.
#[tokio::test]
async fn a_ue_on_a_terrestrial_cell_applies_no_uplink_shift() {
    // No pre-compensation installed: exactly the state a UE on a cell that
    // broadcasts no SIB19 is in.
    let ue_rls = send_one_uplink(UeConfig::default(), None).await;

    assert_eq!(
        ue_rls.last_uplink_timing_offset_us(),
        Some(0.0),
        "a terrestrial UE must transmit on the downlink frame boundary"
    );
    assert!(
        ue_rls.ntn_precompensation_applied().is_none(),
        "and must have no NTN pre-compensation installed"
    );
}

/// **The §16.14.2.2 "shall not transmit" rule.** A UE with no GNSS position cannot
/// compute an RTT, so it must apply NOTHING rather than substituting the origin —
/// which would compute a slant range from the centre of the Earth.
#[tokio::test]
async fn a_ue_without_a_gnss_position_derives_no_precompensation() {
    let si = broadcast_system_information(ntn_gnb_config()).await;

    let no_gnss = UeConfig {
        gnss_position_ecef_m: None,
        ..ntn_ue_config()
    };
    let (base, _app_rx, _nas_rx, _rrc_rx, _rls_rx) = UeTaskBase::new(no_gnss, 64);
    let mut ue_rrc = UeRrcTask::new(base);
    ue_rrc
        .handle_downlink_rrc(1, RrcChannel::BcchDlSch, si)
        .await;

    assert!(
        ue_rrc.ntn_timing().is_none(),
        "TS 38.300 §16.14.2.2: without a valid GNSS position the UE cannot compute \
         the RTT, so it must derive NO pre-compensation rather than one referenced \
         to the centre of the Earth"
    );
}
