//! System information broadcast (BCCH) — TS 38.331 §5.2.1
//!
//! Builds the MIB and SIB1 this cell broadcasts from its own configuration, so a
//! UE learns the cell's PLMN, TAC and cell identity from the air interface
//! instead of assuming them.
//!
//! # What is real and what is a stand-in
//!
//! The messages are real ASN.1/UPER (`nextgsim-rrc`'s `encode_mib` / `encode_sib1`
//! over the Rel-15.6.0 schema) carrying this cell's configured identity. The
//! physical-layer parameters have no counterpart in a simulator with no PHY, so
//! they are fixed at the values a 30 kHz FR1 cell would use and are documented as
//! such below rather than being made configurable knobs nothing reads.

use nextgsim_common::config::GnbConfig;
use nextgsim_common::frame_clock;
use nextgsim_rrc::procedures::rrc_release::{
    CellReselectionPrioritiesParams, FreqPriorityNrParams,
};
use nextgsim_rrc::procedures::system_information::{
    encode_mib, encode_sib1, encode_system_information, CellBarredStatus, CellSelectionInfo,
    DmrsTypeAPosition, InterFreqCarrier, IntraFreqNeighbour, IntraFreqReselection, MibParams,
    PdcchConfigSib1Params, PlmnIdentity, PlmnIdentityInfo, Sib1Params, Sib2Params, Sib3Params,
    Sib4Params, SubCarrierSpacingCommon, SystemInformationError, SystemInformationParams,
    UeTimersAndConstantsParams,
};

/// CORESET#0 index broadcast in the MIB's `pdcch-ConfigSIB1`.
///
/// A PHY parameter with no meaning in this simulator (there is no PDCCH); index 0
/// is a valid choice for a 30 kHz FR1 cell (TS 38.213 Table 13-1).
const CORESET_ZERO: u8 = 0;

/// Search-space#0 index broadcast in the MIB's `pdcch-ConfigSIB1`; see
/// [`CORESET_ZERO`].
const SEARCH_SPACE_ZERO: u8 = 0;

/// `ssb-SubcarrierOffset` broadcast in the MIB. Zero: there is no SSB raster
/// here to be offset from.
const SSB_SUBCARRIER_OFFSET: u8 = 0;

/// `q-RxLevMin` broadcast in SIB1.
///
/// The IE is `INTEGER (-70..-22)` in units of **2 dBm** (TS 38.331 §6.3.2), so
/// -70 is -140 dBm: the most permissive value the range allows. Deliberately so —
/// cell suitability here should be decided by the simulator's own signal model,
/// not by a receive-level floor picked in this file.
///
/// Note the UE side compares this value against a level in plain dBm
/// (`nextgsim-ue`'s `Sib1Info::q_rx_lev_min`), a pre-existing simulator
/// convention rather than the spec's 2 dBm unit; the most permissive value is
/// permissive under either reading, which is why the mismatch is harmless here.
const Q_RX_LEV_MIN: i8 = -70;

/// Builds the MIB this cell broadcasts.
///
/// Takes no cell configuration: every field the MIB carries is a physical-layer
/// parameter this simulator does not model, so there is nothing cell-specific to
/// read. SIB1 is where this cell's identity goes.
///
/// SYSTEM FRAME NUMBER: the 6 most significant bits of the live SFN, which is
/// what TS 38.331 gives the MIB (the 4 LSBs travel on the PBCH payload, which this
/// simulator does not model). It used to be a constant 0 because no frame clock
/// existed; `nextgsim_common::frame_clock` now derives one from the wall clock on
/// both sides (issue #99), so this field carries real timing.
///
/// It is deliberately NOT widened to the full 10 bits. A UE cannot recover the
/// exact frame from the MIB alone -- the granularity is 16 frames -- so the MIB is
/// a CROSS-CHECK on the shared clock rather than the clock itself. Putting 10 bits
/// in a 6-bit field would make a decoder that trusts the field wrong about a real
/// network.
pub fn mib_params() -> MibParams {
    MibParams {
        system_frame_number: frame_clock::sfn_msb6(frame_clock::current_sfn()),
        sub_carrier_spacing_common: SubCarrierSpacingCommon::Scs30Or120,
        ssb_subcarrier_offset: SSB_SUBCARRIER_OFFSET,
        dmrs_type_a_position: DmrsTypeAPosition::Pos2,
        pdcch_config_sib1: PdcchConfigSib1Params {
            coreset_zero: CORESET_ZERO,
            search_space_zero: SEARCH_SPACE_ZERO,
        },
        cell_barred: CellBarredStatus::NotBarred,
        intra_freq_reselection: IntraFreqReselection::Allowed,
    }
}

/// Builds the SIB1 this cell broadcasts, carrying the configured PLMN, TAC and
/// NR Cell Identity (TS 38.331 §6.3.2 `CellAccessRelatedInfo`).
pub fn sib1_params(config: &GnbConfig) -> Sib1Params {
    let plmn = PlmnIdentity {
        mcc: Some(mcc_digits(config.plmn.mcc)),
        mnc: mnc_digits(config.plmn.mnc, config.plmn.long_mnc),
    };
    Sib1Params {
        cell_selection_info: Some(CellSelectionInfo {
            q_rx_lev_min: Q_RX_LEV_MIN,
            q_rx_lev_min_offset: None,
            q_rx_lev_min_sul: None,
            q_qual_min: None,
            q_qual_min_offset: None,
        }),
        plmn_identity_info_list: vec![PlmnIdentityInfo {
            plmn_identity_list: vec![plmn],
            tracking_area_code: Some(config.tac),
            cell_identity: config.nci,
        }],
        ims_emergency_support: false,
        ecall_over_ims_support: false,
        // The UE timers a cell may impose (TS 38.331 §7.1.1). T300 is the one
        // that matters here: the UE's RRCSetupRequest guard uses it.
        ue_timers_and_constants: Some(UeTimersAndConstantsParams {
            t300_ms: 1000,
            t301_ms: 1000,
            t310_ms: 1000,
            n310: 1,
            t311_ms: 1000,
            n311: 1,
            t319_ms: 1000,
        }),
        // Rel-17 RedCap reselection is not signalled: it has no conformant home
        // in the Rel-15 schema this tree compiles (see issue #105), and the
        // private marker the field would otherwise use is not on the wire for
        // any real peer.
        intra_freq_reselection_redcap: false,
    }
}

/// The encoded MIB (`BCCH-BCH-Message`) broadcast by this cell.
pub fn encode_cell_mib() -> Result<Vec<u8>, SystemInformationError> {
    encode_mib(&mib_params())
}

/// The encoded SIB1 (`BCCH-DL-SCH-Message`) for this cell.
pub fn encode_cell_sib1(config: &GnbConfig) -> Result<Vec<u8>, SystemInformationError> {
    encode_sib1(&sib1_params(config))
}

/// MCC as its three decimal digits, most significant first.
fn mcc_digits(mcc: u16) -> [u8; 3] {
    [
        ((mcc / 100) % 10) as u8,
        ((mcc / 10) % 10) as u8,
        (mcc % 10) as u8,
    ]
}

/// MNC as its two or three decimal digits, most significant first.
///
/// A two-digit MNC must stay two digits: 001-01 and 001-001 are different PLMNs,
/// so padding a short MNC would advertise a network this cell does not serve.
fn mnc_digits(mnc: u16, long_mnc: bool) -> Vec<u8> {
    if long_mnc {
        vec![
            ((mnc / 100) % 10) as u8,
            ((mnc / 10) % 10) as u8,
            (mnc % 10) as u8,
        ]
    } else {
        vec![((mnc / 10) % 10) as u8, (mnc % 10) as u8]
    }
}

/// Builds the SIB2/SIB3/SIB4 this cell broadcasts, carrying the idle-mode
/// reselection parameters of TS 38.304 §5.2.4.6 (issue #50).
///
/// SIB1 tells the UE what this cell is; these tell it when to leave. Before this
/// the UE took `Q_hyst` and `Treselection` from compile-time constants and had no
/// per-cell `Qoffset` or reselection priority at all, so a cell could not
/// influence its own mobility.
///
/// Returns `None` when the operator turned the broadcast off
/// (`reselection.broadcast = false`), which is the pre-#50 behaviour: the UE then
/// falls back to its constants and says so.
///
/// SIB4 is omitted rather than emptied when no other carrier is configured:
/// `interFreqCarrierFreqList` is `SIZE (1..maxFreq)`, so a zero-length list is
/// not encodable.
pub fn system_information_params(config: &GnbConfig) -> Option<SystemInformationParams> {
    let r = &config.reselection;
    if !r.broadcast {
        return None;
    }

    Some(SystemInformationParams {
        sib2: Some(Sib2Params {
            q_hyst_db: r.q_hyst_db,
            t_reselection_s: r.t_reselection_s,
            cell_reselection_priority: r.cell_reselection_priority,
            // The same value SIB1 broadcasts, and for the same reason: cell
            // suitability here should be decided by the simulator's signal model
            // rather than by a receive-level floor picked in this file. Two
            // different floors on one cell would also be a conformance oddity.
            q_rx_lev_min: Q_RX_LEV_MIN,
            s_intra_search_p: r.s_intra_search_p,
            thresh_serving_low_p: r.thresh_serving_low_p,
        }),
        sib3: Some(Sib3Params {
            intra_freq_neighbours: r
                .intra_freq_neighbours
                .iter()
                .map(|n| IntraFreqNeighbour {
                    phys_cell_id: n.phys_cell_id,
                    q_offset_db: n.q_offset_db,
                })
                .collect(),
        }),
        sib4: if r.inter_freq_carriers.is_empty() {
            None
        } else {
            Some(Sib4Params {
                inter_freq_carriers: r
                    .inter_freq_carriers
                    .iter()
                    .map(|c| InterFreqCarrier {
                        dl_carrier_freq: c.dl_carrier_freq,
                        cell_reselection_priority: c.cell_reselection_priority,
                        thresh_x_high_p: c.thresh_x_high_p,
                        thresh_x_low_p: c.thresh_x_low_p,
                        q_rx_lev_min: c.q_rx_lev_min,
                        t_reselection_s: c.t_reselection_s,
                    })
                    .collect(),
            })
        },
    })
}

/// The encoded `SystemInformation` (`BCCH-DL-SCH-Message`) carrying this cell's
/// SIB2/SIB3/SIB4, or `None` when the broadcast is turned off.
pub fn encode_cell_system_information(
    config: &GnbConfig,
) -> Option<Result<Vec<u8>, SystemInformationError>> {
    system_information_params(config).map(|params| encode_system_information(&params))
}

/// The DEDICATED `cellReselectionPriorities` this cell hands a UE in RRCRelease
/// (TS 38.331 §6.3.2, issue #50).
///
/// Built from the same `reselection` block SIB4 comes from, so a released UE gets
/// the priorities this cell broadcasts rather than a second, divergent set.
///
/// Returns `None` when there is nothing dedicated to say — no configured other
/// carriers — rather than an empty list. `freqPriorityListNR` is
/// `SIZE (1..maxFreq)`, and more importantly an empty dedicated list is not the
/// same message as an absent one: TS 38.304 §5.2.4.1 has the UE use the
/// broadcast priorities when no dedicated ones were provided, and *delete* its
/// stored dedicated priorities when it receives an empty list. Sending an empty
/// list on every release would therefore keep wiping state the UE should keep.
pub fn release_cell_reselection_priorities(
    config: &GnbConfig,
) -> Option<CellReselectionPrioritiesParams> {
    let r = &config.reselection;
    if r.inter_freq_carriers.is_empty() {
        return None;
    }
    Some(CellReselectionPrioritiesParams {
        freq_priority_list_nr: r
            .inter_freq_carriers
            .iter()
            .map(|c| FreqPriorityNrParams {
                carrier_freq: c.dl_carrier_freq,
                priority: c.cell_reselection_priority,
            })
            .collect(),
        // T320 bounds how long the dedicated priorities stay valid. Left unset:
        // this simulator has no timer driving their expiry, and advertising a
        // validity the UE would honour while the network forgot it would make the
        // two disagree about which priorities are in force.
        t320: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nextgsim_common::Plmn;
    use nextgsim_rrc::procedures::system_information::{decode_mib, decode_sib1};

    fn config() -> GnbConfig {
        GnbConfig {
            nci: 0x0000_0001_2,
            plmn: Plmn::new(1, 1, false),
            tac: 7,
            ..Default::default()
        }
    }

    #[test]
    fn the_broadcast_sib1_carries_this_cells_identity() {
        let encoded = encode_cell_sib1(&config()).expect("SIB1 must encode");
        let decoded = decode_sib1(&encoded).expect("and decode");

        let info = &decoded.plmn_identity_info_list[0];
        assert_eq!(info.tracking_area_code, Some(7), "the configured TAC");
        assert_eq!(info.cell_identity, 0x0000_0001_2, "the configured NCI");
        assert_eq!(
            info.plmn_identity_list[0],
            PlmnIdentity {
                mcc: Some([0, 0, 1]),
                mnc: vec![0, 1],
            },
            "PLMN 001-01 with a TWO-digit MNC"
        );
    }

    /// A three-digit MNC must not be truncated and a two-digit one must not be
    /// padded: 001-01 and 001-001 are different networks.
    #[test]
    fn a_long_mnc_is_broadcast_with_three_digits() {
        let config = GnbConfig {
            plmn: Plmn::new(310, 260, true),
            ..config()
        };
        let decoded = decode_sib1(&encode_cell_sib1(&config).unwrap()).unwrap();
        assert_eq!(
            decoded.plmn_identity_info_list[0].plmn_identity_list[0],
            PlmnIdentity {
                mcc: Some([3, 1, 0]),
                mnc: vec![2, 6, 0],
            }
        );
    }

    #[test]
    fn the_broadcast_mib_is_decodable_and_advertises_an_unbarred_cell() {
        let decoded = decode_mib(&encode_cell_mib().unwrap()).expect("MIB must decode");
        assert_eq!(decoded.cell_barred, CellBarredStatus::NotBarred);
        assert_eq!(
            decoded.intra_freq_reselection,
            IntraFreqReselection::Allowed
        );
        // FLIPPED by issue #99. This asserted `== 0, "no frame clock exists"`,
        // which pinned the absence of the clock. A clock now exists, so the MIB
        // must carry the live SFN's 6 most significant bits -- and the assertion
        // has to be against the clock rather than a constant, or it would only be
        // pinning whatever number happened to be there.
        let expected = frame_clock::sfn_msb6(frame_clock::current_sfn());
        assert!(
            decoded.system_frame_number == expected
                || decoded.system_frame_number == expected.wrapping_sub(1) % 64,
            "MIB SFN {} must be the live SFN's MSB6 ({expected}); a 16-frame \
             boundary may fall between encode and assert, which is the only \
             tolerated difference",
            decoded.system_frame_number
        );
        assert!(
            decoded.system_frame_number <= 63,
            "the MIB field is 6 bits wide"
        );
    }
    // ========================================================================
    // SIB2/SIB3/SIB4 reselection broadcast (issue #50)
    // ========================================================================

    /// The `reselection` YAML key must actually be READ.
    ///
    /// Nothing in this config sets `deny_unknown_fields`, so a key that no field
    /// claims is silently DROPPED rather than rejected -- the recorded failure
    /// mode where a config block is shipped, looks configured, and is inert. This
    /// asserts NON-DEFAULT values arrive, because a test using the defaults would
    /// pass whether the key was parsed or ignored.
    #[test]
    fn the_reselection_config_key_is_deserialised_rather_than_dropped() {
        let yaml = r#"
nci: 1
gnb_id_length: 24
plmn:
  mcc: 1
  mnc: 1
  long_mnc: false
tac: 7
nssai: []
amf_configs: []
link_ip: 127.0.0.1
ngap_ip: 127.0.0.1
gtp_ip: 127.0.0.1
gtp_advertise_ip: null
ignore_stream_ids: false
reselection:
  broadcast: true
  q_hyst_db: 10
  t_reselection_s: 3
  cell_reselection_priority: 6
  s_intra_search_p: 20
  thresh_serving_low_p: 9
  intra_freq_neighbours:
    - phys_cell_id: 128
      q_offset_db: -3
  inter_freq_carriers:
    - dl_carrier_freq: 632628
      cell_reselection_priority: 2
"#;
        let parsed: GnbConfig = serde_yaml::from_str(yaml).expect("the sample must parse");
        let r = &parsed.reselection;
        assert_eq!(r.q_hyst_db, 10, "q_hyst_db was dropped (default is 4)");
        assert_eq!(
            r.t_reselection_s, 3,
            "t_reselection_s was dropped (default is 1)"
        );
        assert_eq!(r.cell_reselection_priority, 6);
        assert_eq!(r.s_intra_search_p, 20);
        assert_eq!(r.thresh_serving_low_p, 9);
        assert_eq!(r.intra_freq_neighbours.len(), 1);
        assert_eq!(r.intra_freq_neighbours[0].phys_cell_id, 128);
        assert_eq!(r.intra_freq_neighbours[0].q_offset_db, -3);
        assert_eq!(r.inter_freq_carriers.len(), 1);
        assert_eq!(r.inter_freq_carriers[0].dl_carrier_freq, 632628);
        // Omitted per-carrier fields take their documented defaults rather than
        // failing the parse.
        assert_eq!(r.inter_freq_carriers[0].q_rx_lev_min, -70);
    }

    /// The shipped `config/gnb.yaml` must parse, and its `reselection` block must
    /// be the one the parser reads. A sample config that no longer deserialises is
    /// a defect an operator hits before any test does.
    #[test]
    fn the_shipped_gnb_sample_config_parses_with_its_reselection_block() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/gnb.yaml");
        let text = std::fs::read_to_string(path).expect("the shipped sample must be readable");
        let parsed: GnbConfig = serde_yaml::from_str(&text).expect("and must parse");
        assert!(
            parsed.reselection.broadcast,
            "the shipped sample must broadcast SIB2/3/4, or the documented default \
             and the shipped default disagree"
        );
        assert_eq!(parsed.reselection.q_hyst_db, 4);
        assert_eq!(parsed.reselection.t_reselection_s, 1);
    }

    /// What the cell broadcasts is what the config says, asserted by DECODING the
    /// emitted bytes rather than by reading the params struct back.
    #[test]
    fn the_broadcast_sib2_carries_the_configured_reselection_parameters() {
        use nextgsim_common::config::{CellReselectionBroadcastConfig, IntraFreqNeighbourConfig};
        use nextgsim_rrc::procedures::system_information::decode_system_information;

        let cfg = GnbConfig {
            reselection: CellReselectionBroadcastConfig {
                q_hyst_db: 12,
                t_reselection_s: 4,
                cell_reselection_priority: 5,
                intra_freq_neighbours: vec![IntraFreqNeighbourConfig {
                    phys_cell_id: 300,
                    q_offset_db: 6,
                }],
                ..Default::default()
            },
            ..config()
        };

        let encoded = encode_cell_system_information(&cfg)
            .expect("the broadcast is on")
            .expect("and must encode");
        let decoded = decode_system_information(&encoded).expect("and decode");

        let sib2 = decoded.sib2.expect("SIB2 must be broadcast");
        assert_eq!(sib2.q_hyst_db, 12);
        assert_eq!(sib2.t_reselection_s, 4);
        assert_eq!(sib2.cell_reselection_priority, 5);
        assert_eq!(
            sib2.q_rx_lev_min, Q_RX_LEV_MIN,
            "SIB2 must carry the SAME q-RxLevMin as SIB1: two different receive \
             floors on one cell is a conformance oddity"
        );

        let sib3 = decoded.sib3.expect("SIB3 must be broadcast");
        assert_eq!(sib3.intra_freq_neighbours.len(), 1);
        assert_eq!(sib3.intra_freq_neighbours[0].phys_cell_id, 300);
        assert_eq!(sib3.intra_freq_neighbours[0].q_offset_db, 6);

        assert!(
            decoded.sib4.is_none(),
            "with no configured other carrier, SIB4 must be OMITTED rather than \
             carrying an unencodable empty list"
        );
    }

    /// `broadcast: false` restores the pre-#50 behaviour: nothing on the air.
    #[test]
    fn turning_the_broadcast_off_emits_no_system_information() {
        use nextgsim_common::config::CellReselectionBroadcastConfig;

        let cfg = GnbConfig {
            reselection: CellReselectionBroadcastConfig {
                broadcast: false,
                ..Default::default()
            },
            ..config()
        };
        assert!(
            encode_cell_system_information(&cfg).is_none(),
            "an operator who turned the broadcast off must get no SI, and not an \
             encode error either"
        );
        // And the SIB1/MIB broadcast is untouched by that switch.
        assert!(encode_cell_sib1(&cfg).is_ok());
    }

    /// The DEDICATED priorities handed out in RRCRelease come from the same
    /// configured carrier list as SIB4, so a released UE is not given a second,
    /// divergent set.
    #[test]
    fn the_dedicated_release_priorities_mirror_the_configured_carriers() {
        use nextgsim_common::config::{CellReselectionBroadcastConfig, InterFreqCarrierConfig};

        // No carriers: nothing dedicated to say, and an ABSENT IE rather than an
        // empty list -- an empty one would tell the UE to delete its stored
        // dedicated priorities on every release.
        assert!(release_cell_reselection_priorities(&config()).is_none());

        let cfg = GnbConfig {
            reselection: CellReselectionBroadcastConfig {
                inter_freq_carriers: vec![InterFreqCarrierConfig {
                    dl_carrier_freq: 632628,
                    cell_reselection_priority: 3,
                    thresh_x_high_p: 8,
                    thresh_x_low_p: 4,
                    q_rx_lev_min: -70,
                    t_reselection_s: 1,
                }],
                ..Default::default()
            },
            ..config()
        };
        let dedicated =
            release_cell_reselection_priorities(&cfg).expect("a configured carrier is dedicated");
        assert_eq!(dedicated.freq_priority_list_nr.len(), 1);
        assert_eq!(dedicated.freq_priority_list_nr[0].carrier_freq, 632628);
        assert_eq!(dedicated.freq_priority_list_nr[0].priority, 3);
        assert!(
            dedicated.t320.is_none(),
            "T320 must stay unset: nothing here expires the dedicated priorities, \
             and advertising a validity the UE honours while the network forgets \
             makes the two disagree"
        );
    }
}
