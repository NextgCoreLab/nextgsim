//! Secondary cell configuration (TS 38.331 §5.3.5.5.9)
//!
//! An RRCReconfiguration adds and releases secondary cells through the
//! `sCellToAddModList` and `sCellToReleaseList` of a `CellGroupConfig`. This
//! module builds and reads that list.
//!
//! # Why this is real UPER and the CHO container is not
//!
//! `CellGroupConfig` carries `sCellToAddModList SEQUENCE (SIZE (1..maxNrofSCells))
//! OF SCellConfig` and `sCellToReleaseList SEQUENCE (SIZE (1..maxNrofSCells)) OF
//! SCellIndex` in **Rel-15**, so the vendored `rrc-15.6.0.asn1` schema already
//! models both and the encoding here is the conformant one. That is the
//! difference from `conditionalReconfiguration`, which is Rel-16 and therefore
//! unreachable until issue #105 upgrades the schema — the reason that container
//! is the simulator's own byte format.
//!
//! The transport is still the simulator's hand-rolled DL-DCCH envelope (issue
//! #107), so what travels is `[0x0E][transaction id][UPER CellGroupConfig]` —
//! the same shape as the UE capability transfer, whose payload is likewise a
//! real UPER encoding inside a hand-rolled envelope.
//!
//! # What an SCell means here
//!
//! Carrier aggregation is not modelled: there is no per-SCell BWP, no MAC/PHY
//! aggregation, no `sCellState` activation and no PUCCH SCell. The one thing an
//! SCell is used for is that measurement event A6 is defined against it
//! (§5.5.4.7, "the (secondary) cell corresponding to the measObjectNR associated
//! to this event"), so the UE needs to know which cell that is.
//!
//! # Reference
//! - 3GPP TS 38.331 §5.3.5.5.9 (`sCellToAddModList`), §6.3.2 (`SCellConfig`,
//!   `ServingCellConfigCommon`), §5.5.4.7 (event A6)

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// `SCellIndex ::= INTEGER (1..31)` — TS 38.331 §6.3.2.
pub const SCELL_INDEX_MIN: u8 = 1;
/// `maxNrofSCells`, the upper bound of `SCellIndex`.
pub const SCELL_INDEX_MAX: u8 = 31;

/// `PhysCellId ::= INTEGER (0..1007)` — TS 38.331 §6.3.2.
pub const PHYS_CELL_ID_MAX: u16 = 1007;

/// `dmrs-TypeA-Position` and `ss-PBCH-BlockPower` are mandatory in
/// `ServingCellConfigCommon`, so a value has to be chosen even for a simulator
/// that models neither. `pos2` and 0 dBm are the neutral pair: they say nothing
/// about the cell beyond making the encoding well-formed.
const DEFAULT_DMRS_TYPE_A_POSITION: u8 = ServingCellConfigCommonDmrs_TypeA_Position::POS2;
/// See [`DEFAULT_DMRS_TYPE_A_POSITION`].
const DEFAULT_SS_PBCH_BLOCK_POWER_DBM: i8 = 0;

/// Errors from the secondary-cell configuration codec.
#[derive(Debug, Error)]
pub enum ScellConfigError {
    /// An `sCellIndex` outside `INTEGER (1..31)`
    #[error("sCellIndex {0} is outside the SCellIndex range 1..=31 (TS 38.331 §6.3.2)")]
    InvalidScellIndex(u8),
    /// A `physCellId` outside `INTEGER (0..1007)`
    #[error("physCellId {0} is outside the PhysCellId range 0..=1007 (TS 38.331 §6.3.2)")]
    InvalidPhysCellId(u16),
    /// Neither list carried anything: the ASN.1 lists have a lower bound of 1,
    /// so an empty configuration has no encoding.
    #[error("an SCell configuration must add or release at least one cell")]
    Empty,
    /// Too many cells for `SEQUENCE (SIZE (1..31))`
    #[error("{0} SCells exceeds maxNrofSCells (31)")]
    TooManyScells(usize),
    /// The UPER codec rejected the message
    #[error("RRC codec error: {0}")]
    Codec(#[from] RrcCodecError),
}

/// One secondary cell to add or modify.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScellAddition {
    /// `sCellIndex` (1..31), the identity the release list refers to
    pub scell_index: u8,
    /// `sCellConfigCommon.physCellId` — which cell this is
    pub phys_cell_id: u16,
    /// `sCellConfigCommon.ssb-SubcarrierSpacing`, in kHz, when known. Only the
    /// values `SubcarrierSpacing` can express survive a round trip; anything else
    /// is dropped, since the IE is optional.
    pub ssb_subcarrier_spacing_khz: Option<u32>,
}

impl ScellAddition {
    /// An SCell at `scell_index` on `phys_cell_id`, with no subcarrier spacing.
    pub fn new(scell_index: u8, phys_cell_id: u16) -> Self {
        Self {
            scell_index,
            phys_cell_id,
            ssb_subcarrier_spacing_khz: None,
        }
    }

    fn validate(&self) -> Result<(), ScellConfigError> {
        if !(SCELL_INDEX_MIN..=SCELL_INDEX_MAX).contains(&self.scell_index) {
            return Err(ScellConfigError::InvalidScellIndex(self.scell_index));
        }
        if self.phys_cell_id > PHYS_CELL_ID_MAX {
            return Err(ScellConfigError::InvalidPhysCellId(self.phys_cell_id));
        }
        Ok(())
    }
}

/// The secondary-cell part of a `CellGroupConfig`: what to add and what to
/// release.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScellConfig {
    /// `sCellToAddModList`
    pub to_add: Vec<ScellAddition>,
    /// `sCellToReleaseList`, by `sCellIndex`
    pub to_release: Vec<u8>,
}

impl ScellConfig {
    /// A configuration that adds one secondary cell.
    pub fn add_one(scell_index: u8, phys_cell_id: u16) -> Self {
        Self {
            to_add: vec![ScellAddition::new(scell_index, phys_cell_id)],
            to_release: Vec::new(),
        }
    }

    /// A configuration that releases one secondary cell.
    pub fn release_one(scell_index: u8) -> Self {
        Self {
            to_add: Vec::new(),
            to_release: vec![scell_index],
        }
    }
}

/// The kHz value a `SubcarrierSpacing` enumerator denotes, for the values NR
/// defines; the three `spare` code points have no numeric meaning.
fn subcarrier_spacing_khz(spacing: &SubcarrierSpacing) -> Option<u32> {
    match spacing.0 {
        SubcarrierSpacing::K_HZ15 => Some(15),
        SubcarrierSpacing::K_HZ30 => Some(30),
        SubcarrierSpacing::K_HZ60 => Some(60),
        SubcarrierSpacing::K_HZ120 => Some(120),
        SubcarrierSpacing::K_HZ240 => Some(240),
        _ => None,
    }
}

/// The `SubcarrierSpacing` enumerator for a kHz value, or `None` when NR has no
/// enumerator for it.
fn subcarrier_spacing_from_khz(khz: u32) -> Option<SubcarrierSpacing> {
    let code = match khz {
        15 => SubcarrierSpacing::K_HZ15,
        30 => SubcarrierSpacing::K_HZ30,
        60 => SubcarrierSpacing::K_HZ60,
        120 => SubcarrierSpacing::K_HZ120,
        240 => SubcarrierSpacing::K_HZ240,
        _ => return None,
    };
    Some(SubcarrierSpacing(code))
}

/// Build the `CellGroupConfig` that carries `config`.
///
/// The master cell group (`cellGroupId` 0) is used: the simulator has no
/// secondary cell *group*, only secondary cells within the one group it has.
pub fn build_scell_cell_group_config(
    config: &ScellConfig,
) -> Result<CellGroupConfig, ScellConfigError> {
    if config.to_add.is_empty() && config.to_release.is_empty() {
        return Err(ScellConfigError::Empty);
    }
    if config.to_add.len() > usize::from(SCELL_INDEX_MAX) {
        return Err(ScellConfigError::TooManyScells(config.to_add.len()));
    }
    if config.to_release.len() > usize::from(SCELL_INDEX_MAX) {
        return Err(ScellConfigError::TooManyScells(config.to_release.len()));
    }

    let mut additions = Vec::with_capacity(config.to_add.len());
    for addition in &config.to_add {
        addition.validate()?;
        additions.push(SCellConfig {
            s_cell_index: SCellIndex(addition.scell_index),
            s_cell_config_common: Some(ServingCellConfigCommon {
                phys_cell_id: Some(PhysCellId(addition.phys_cell_id)),
                downlink_config_common: None,
                uplink_config_common: None,
                supplementary_uplink_config: None,
                n_timing_advance_offset: None,
                ssb_positions_in_burst: None,
                ssb_periodicity_serving_cell: None,
                dmrs_type_a_position: ServingCellConfigCommonDmrs_TypeA_Position(
                    DEFAULT_DMRS_TYPE_A_POSITION,
                ),
                lte_crs_to_match_around: None,
                rate_match_pattern_to_add_mod_list: None,
                rate_match_pattern_to_release_list: None,
                ssb_subcarrier_spacing: addition
                    .ssb_subcarrier_spacing_khz
                    .and_then(subcarrier_spacing_from_khz),
                tdd_ul_dl_configuration_common: None,
                ss_pbch_block_power: ServingCellConfigCommonSs_PBCH_BlockPower(
                    DEFAULT_SS_PBCH_BLOCK_POWER_DBM,
                ),
            }),
            // No dedicated configuration: there is no per-SCell BWP or PDSCH
            // configuration to carry while carrier aggregation is unmodelled.
            s_cell_config_dedicated: None,
        });
    }

    let mut releases = Vec::with_capacity(config.to_release.len());
    for &index in &config.to_release {
        if !(SCELL_INDEX_MIN..=SCELL_INDEX_MAX).contains(&index) {
            return Err(ScellConfigError::InvalidScellIndex(index));
        }
        releases.push(SCellIndex(index));
    }

    Ok(CellGroupConfig {
        cell_group_id: CellGroupId(0),
        rlc_bearer_to_add_mod_list: None,
        rlc_bearer_to_release_list: None,
        mac_cell_group_config: None,
        physical_cell_group_config: None,
        sp_cell_config: None,
        s_cell_to_add_mod_list: (!additions.is_empty())
            .then_some(CellGroupConfigSCellToAddModList(additions)),
        s_cell_to_release_list: (!releases.is_empty())
            .then_some(CellGroupConfigSCellToReleaseList(releases)),
    })
}

/// UPER-encode an SCell configuration as a `CellGroupConfig`.
pub fn encode_scell_config(config: &ScellConfig) -> Result<Vec<u8>, ScellConfigError> {
    let cell_group = build_scell_cell_group_config(config)?;
    Ok(encode_rrc(&cell_group)?)
}

/// Read the SCell configuration out of a UPER `CellGroupConfig`.
///
/// An addition whose `sCellConfigCommon` omits `physCellId` is skipped rather
/// than reported with a placeholder: the whole point of the IE here is to say
/// which cell the SCell is, and a configuration that does not say is not one the
/// UE can act on.
pub fn decode_scell_config(bytes: &[u8]) -> Result<ScellConfig, ScellConfigError> {
    let cell_group: CellGroupConfig = decode_rrc(bytes)?;
    Ok(scell_config_of(&cell_group))
}

/// The SCell configuration a decoded `CellGroupConfig` carries, which may be
/// empty: an RRCReconfiguration is free to reconfigure a cell group without
/// touching its secondary cells.
pub fn scell_config_of(cell_group: &CellGroupConfig) -> ScellConfig {
    let to_add = cell_group
        .s_cell_to_add_mod_list
        .as_ref()
        .map(|list| {
            list.0
                .iter()
                .filter_map(|scell| {
                    let common = scell.s_cell_config_common.as_ref()?;
                    let phys_cell_id = common.phys_cell_id.as_ref()?;
                    Some(ScellAddition {
                        scell_index: scell.s_cell_index.0,
                        phys_cell_id: phys_cell_id.0,
                        ssb_subcarrier_spacing_khz: common
                            .ssb_subcarrier_spacing
                            .as_ref()
                            .and_then(subcarrier_spacing_khz),
                    })
                })
                .collect()
        })
        .unwrap_or_default();

    let to_release = cell_group
        .s_cell_to_release_list
        .as_ref()
        .map(|list| list.0.iter().map(|index| index.0).collect())
        .unwrap_or_default();

    ScellConfig { to_add, to_release }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_scell_addition_round_trips_through_uper() {
        let config = ScellConfig {
            to_add: vec![
                ScellAddition {
                    scell_index: 1,
                    phys_cell_id: 42,
                    ssb_subcarrier_spacing_khz: Some(30),
                },
                ScellAddition::new(7, 1007),
            ],
            to_release: vec![3, 31],
        };

        let bytes = encode_scell_config(&config).expect("encode");
        let decoded = decode_scell_config(&bytes).expect("decode");

        assert_eq!(decoded, config);
    }

    #[test]
    fn a_release_only_configuration_round_trips() {
        let config = ScellConfig::release_one(5);
        let decoded =
            decode_scell_config(&encode_scell_config(&config).expect("encode")).expect("decode");
        assert_eq!(decoded, config);
        assert!(decoded.to_add.is_empty());
    }

    /// The subcarrier spacing is an ENUMERATED, so only the values NR defines
    /// survive; an unrepresentable one is dropped rather than mis-encoded.
    #[test]
    fn an_unrepresentable_subcarrier_spacing_is_dropped_not_faked() {
        let config = ScellConfig {
            to_add: vec![ScellAddition {
                scell_index: 2,
                phys_cell_id: 100,
                ssb_subcarrier_spacing_khz: Some(45), // no SubcarrierSpacing enumerator
            }],
            to_release: Vec::new(),
        };

        let decoded =
            decode_scell_config(&encode_scell_config(&config).expect("encode")).expect("decode");

        assert_eq!(decoded.to_add[0].phys_cell_id, 100);
        assert_eq!(decoded.to_add[0].ssb_subcarrier_spacing_khz, None);
    }

    #[test]
    fn every_defined_subcarrier_spacing_round_trips() {
        for khz in [15, 30, 60, 120, 240] {
            let config = ScellConfig {
                to_add: vec![ScellAddition {
                    scell_index: 1,
                    phys_cell_id: 1,
                    ssb_subcarrier_spacing_khz: Some(khz),
                }],
                to_release: Vec::new(),
            };
            let decoded = decode_scell_config(&encode_scell_config(&config).expect("encode"))
                .expect("decode");
            assert_eq!(
                decoded.to_add[0].ssb_subcarrier_spacing_khz,
                Some(khz),
                "{khz} kHz must survive the round trip"
            );
        }
    }

    #[test]
    fn an_out_of_range_scell_index_is_rejected() {
        for index in [0, 32] {
            let err = encode_scell_config(&ScellConfig::add_one(index, 1))
                .expect_err("sCellIndex is INTEGER (1..31)");
            assert!(matches!(err, ScellConfigError::InvalidScellIndex(i) if i == index));
        }
        let err = encode_scell_config(&ScellConfig::release_one(0))
            .expect_err("a release index is an sCellIndex too");
        assert!(matches!(err, ScellConfigError::InvalidScellIndex(0)));
    }

    #[test]
    fn an_out_of_range_phys_cell_id_is_rejected() {
        let err = encode_scell_config(&ScellConfig::add_one(1, 1008))
            .expect_err("physCellId is INTEGER (0..1007)");
        assert!(matches!(err, ScellConfigError::InvalidPhysCellId(1008)));
    }

    /// Both ASN.1 lists have a lower bound of 1, so there is no encoding for a
    /// configuration that changes nothing.
    #[test]
    fn an_empty_configuration_is_rejected() {
        let err = encode_scell_config(&ScellConfig::default()).expect_err("nothing to encode");
        assert!(matches!(err, ScellConfigError::Empty));
    }

    /// A cell group that says nothing about secondary cells reads as no change,
    /// not as an error: the same IE carries DRB reconfiguration.
    #[test]
    fn a_cell_group_without_scells_reads_as_no_change() {
        let cell_group = CellGroupConfig {
            cell_group_id: CellGroupId(0),
            rlc_bearer_to_add_mod_list: None,
            rlc_bearer_to_release_list: None,
            mac_cell_group_config: None,
            physical_cell_group_config: None,
            sp_cell_config: None,
            s_cell_to_add_mod_list: None,
            s_cell_to_release_list: None,
        };
        let bytes = encode_rrc(&cell_group).expect("encode");

        let decoded = decode_scell_config(&bytes).expect("decode");

        assert_eq!(decoded, ScellConfig::default());
    }

    /// A `ServingCellConfigCommon` that says nothing beyond the two mandatory
    /// fields. Its `physCellId` is absent, which is legal ASN.1 and useless here.
    fn common_without_a_phys_cell_id() -> ServingCellConfigCommon {
        ServingCellConfigCommon {
            phys_cell_id: None,
            downlink_config_common: None,
            uplink_config_common: None,
            supplementary_uplink_config: None,
            n_timing_advance_offset: None,
            ssb_positions_in_burst: None,
            ssb_periodicity_serving_cell: None,
            dmrs_type_a_position: ServingCellConfigCommonDmrs_TypeA_Position(
                DEFAULT_DMRS_TYPE_A_POSITION,
            ),
            lte_crs_to_match_around: None,
            rate_match_pattern_to_add_mod_list: None,
            rate_match_pattern_to_release_list: None,
            ssb_subcarrier_spacing: None,
            tdd_ul_dl_configuration_common: None,
            ss_pbch_block_power: ServingCellConfigCommonSs_PBCH_BlockPower(
                DEFAULT_SS_PBCH_BLOCK_POWER_DBM,
            ),
        }
    }

    /// An addition that does not say which cell it is cannot be acted on, so it is
    /// not reported as an addition at all — not reported with `physCellId` 0,
    /// which is a real NR cell identity and would configure the wrong cell.
    ///
    /// Both ways of not saying are covered: no `sCellConfigCommon` and an
    /// `sCellConfigCommon` whose `physCellId` is absent. The first alone does not
    /// exercise the `physCellId` check, because the decoder gives up on the
    /// enclosing IE first.
    #[test]
    fn an_addition_without_a_phys_cell_id_is_skipped() {
        for (label, common) in [
            ("no sCellConfigCommon", None),
            (
                "sCellConfigCommon without physCellId",
                Some(common_without_a_phys_cell_id()),
            ),
        ] {
            let cell_group = CellGroupConfig {
                cell_group_id: CellGroupId(0),
                rlc_bearer_to_add_mod_list: None,
                rlc_bearer_to_release_list: None,
                mac_cell_group_config: None,
                physical_cell_group_config: None,
                sp_cell_config: None,
                s_cell_to_add_mod_list: Some(CellGroupConfigSCellToAddModList(vec![SCellConfig {
                    s_cell_index: SCellIndex(4),
                    s_cell_config_common: common,
                    s_cell_config_dedicated: None,
                }])),
                s_cell_to_release_list: None,
            };
            let bytes = encode_rrc(&cell_group).expect("encode");

            let decoded = decode_scell_config(&bytes).expect("decode");

            assert!(
                decoded.to_add.is_empty(),
                "an addition with {label} must not be reported"
            );
        }
    }

    #[test]
    fn a_truncated_container_is_an_error_not_a_default() {
        let bytes = encode_scell_config(&ScellConfig::add_one(1, 42)).expect("encode");
        let err = decode_scell_config(&bytes[..1]).expect_err("truncated must not decode");
        assert!(matches!(err, ScellConfigError::Codec(_)));
    }
}
