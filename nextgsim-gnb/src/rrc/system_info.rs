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
use nextgsim_rrc::procedures::system_information::{
    encode_mib, encode_sib1, CellBarredStatus, CellSelectionInfo, DmrsTypeAPosition,
    IntraFreqReselection, MibParams, PdcchConfigSib1Params, PlmnIdentity, PlmnIdentityInfo,
    Sib1Params, SubCarrierSpacingCommon, SystemInformationError, UeTimersAndConstantsParams,
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
/// SYSTEM FRAME NUMBER: always 0. The MIB carries the 6 most significant bits of
/// the SFN, and this simulator maintains no frame clock at all (see the paging
/// occasion discussion in issue #99), so a counter here would be a number that
/// looks like frame timing while corresponding to nothing. Zero at least does not
/// pretend otherwise.
pub fn mib_params() -> MibParams {
    MibParams {
        system_frame_number: 0,
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
        assert_eq!(decoded.system_frame_number, 0, "no frame clock exists");
    }
}
