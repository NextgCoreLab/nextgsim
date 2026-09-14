//! `SuspendConfig`: the IE that makes RRC_INACTIVE reachable
//! (TS 38.331 §5.3.8.3, §6.3.2; TS 38.304 §5.5).
//!
//! An `RRCRelease` **with** a `suspendConfig` moves the UE to RRC_INACTIVE; the
//! same message **without** one moves it to RRC_IDLE (§5.3.8.3). So this IE is the
//! whole difference between suspending a UE and releasing it, and both ends have to
//! read it the same way — which is why it lives here rather than in either endpoint
//! (issue #38).
//!
//! # What the IE carries, and why each field matters to the other end
//!
//! ```text
//! fullI-RNTI              40 bits  the key the gNB stores the context under
//! shortI-RNTI             24 bits  the same UE, for RRCResumeRequest on UL-CCCH
//! ran-PagingCycle                  how often the UE listens for RAN paging
//! ran-NotificationAreaInfo OPT     the cells the UE may roam without an RNAU
//! t380                     OPT     the periodic RNAU timer
//! nextHopChainingCount             the NCC the resumed connection re-keys from
//! ```
//!
//! Both I-RNTIs are mandatory and they identify the **same** UE: which one goes on
//! the wire depends on whether SIB1 signals `useFullResumeID`, not on the network's
//! preference, so a gNB that allocated them independently would fail to find its own
//! context half the time. [`SuspendConfigParams`] therefore takes one identity and
//! derives the short form from it.

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use bitvec::prelude::*;
use thiserror::Error;

/// Width of a full I-RNTI in bits (TS 38.331 `I-RNTI-Value`).
pub const FULL_I_RNTI_BITS: usize = 40;

/// Width of a short I-RNTI in bits (TS 38.331 `ShortI-RNTI-Value`).
pub const SHORT_I_RNTI_BITS: usize = 24;

/// Width of an NR Cell Identity in bits (TS 38.331 `CellIdentity`).
pub const CELL_IDENTITY_BITS: usize = 36;

/// The most cells a RAN Notification Area cell list can hold
/// (`PLMN-RAN-AreaCell.ran-AreaCells`, SIZE (1..32) per PLMN).
pub const MAX_RNA_CELLS: usize = 32;

/// `t380` values in minutes, paired with their enumeration index
/// (TS 38.331 `PeriodicRNAU-TimerValue`).
///
/// A table rather than arithmetic because the steps are not uniform — 5, 10, 20, 30,
/// 60, 120, 360, 720 — so a value outside the set is **refused**, not rounded. T380
/// decides when the UE wakes to do an RNAU; a silently adjusted one makes a periodic
/// update happen at a time nobody configured, and the two ends would still agree,
/// so nothing would notice.
pub const T380_MINUTES_TO_INDEX: &[(u16, u8)] = &[
    (5, PeriodicRNAU_TimerValue::MIN5),
    (10, PeriodicRNAU_TimerValue::MIN10),
    (20, PeriodicRNAU_TimerValue::MIN20),
    (30, PeriodicRNAU_TimerValue::MIN30),
    (60, PeriodicRNAU_TimerValue::MIN60),
    (120, PeriodicRNAU_TimerValue::MIN120),
    (360, PeriodicRNAU_TimerValue::MIN360),
    (720, PeriodicRNAU_TimerValue::MIN720),
];

/// `ran-PagingCycle` values in radio frames, paired with their enumeration index
/// (TS 38.331 `PagingCycle`). Refused rather than rounded, for the same reason.
pub const PAGING_CYCLE_RF_TO_INDEX: &[(u16, u8)] = &[
    (32, PagingCycle::RF32),
    (64, PagingCycle::RF64),
    (128, PagingCycle::RF128),
    (256, PagingCycle::RF256),
];

/// Errors from building or reading a `SuspendConfig`.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SuspendConfigError {
    /// Codec error.
    #[error("Codec error: {0}")]
    CodecError(String),

    /// An I-RNTI wider than its field.
    #[error("I-RNTI {0:#x} does not fit in {1} bits")]
    IRntiTooWide(u64, usize),

    /// A `t380` outside `PeriodicRNAU-TimerValue`.
    #[error("t380 of {0} minutes is not an enumerated PeriodicRNAU-TimerValue; refusing rather than rounding")]
    UnsupportedT380(u16),

    /// A `ran-PagingCycle` outside `PagingCycle`.
    #[error("ran-PagingCycle of {0} radio frames is not an enumerated PagingCycle")]
    UnsupportedPagingCycle(u16),

    /// An NCC outside 0..=7.
    #[error("nextHopChainingCount {0} is outside 0..=7")]
    InvalidNcc(u8),

    /// An empty or oversized RAN Notification Area cell list.
    #[error("a RAN Notification Area cell list must hold 1..={MAX_RNA_CELLS} cells, got {0}")]
    InvalidRnaCellCount(usize),
}

impl From<RrcCodecError> for SuspendConfigError {
    fn from(e: RrcCodecError) -> Self {
        Self::CodecError(e.to_string())
    }
}

/// The RAN Notification Area the UE may move within without doing an RNAU
/// (TS 38.331 `RAN-NotificationAreaInfo`, TS 38.304 §5.5).
///
/// Only the `cellList` arm is modelled. The `ran-AreaConfigList` arm names RAN areas
/// by `trackingAreaCode` plus an optional `ran-AreaCodeList`, and this simulator has
/// no RAN-area concept to map those onto — a gNB that emitted them would be naming
/// areas it could not later decide the UE had left, which is worse than naming cells
/// it can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RanNotificationArea {
    /// The 36-bit NR Cell Identities in the area.
    CellList(Vec<u64>),
}

impl RanNotificationArea {
    /// Whether `cell_identity` is inside this area — i.e. whether the UE may camp
    /// there without an RNAU.
    ///
    /// A cell **not** in the area is an RNA crossing (TS 38.304 §5.5), so this is the
    /// predicate the UE's RNAU trigger is built on.
    pub fn contains(&self, cell_identity: u64) -> bool {
        match self {
            Self::CellList(cells) => cells.contains(&cell_identity),
        }
    }

    /// How many cells the area names.
    pub fn len(&self) -> usize {
        match self {
            Self::CellList(cells) => cells.len(),
        }
    }

    /// Whether the area names no cells at all, which is not encodable.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Parameters for building a `SuspendConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuspendConfigParams {
    /// The full 40-bit I-RNTI. The short form is derived from its low 24 bits, so
    /// the two always name the same UE — see the module docs.
    pub full_i_rnti: u64,
    /// `ran-PagingCycle` in radio frames: 32, 64, 128 or 256.
    pub ran_paging_cycle_rf: u16,
    /// The RAN Notification Area, when the network configures one. `None` means the
    /// UE does an RNAU on leaving its **current** cell (TS 38.304 §5.5: with no
    /// area configured the serving cell is the area).
    pub ran_notification_area: Option<RanNotificationArea>,
    /// `t380` in minutes, when a periodic RNAU is configured. `None` disables the
    /// periodic update, leaving only the RNA-crossing trigger.
    pub t380_minutes: Option<u16>,
    /// `nextHopChainingCount` (0..=7) the resumed connection re-keys from.
    pub next_hop_chaining_count: u8,
}

/// A decoded `SuspendConfig`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuspendConfigData {
    /// Full 40-bit I-RNTI.
    pub full_i_rnti: u64,
    /// Short 24-bit I-RNTI.
    pub short_i_rnti: u32,
    /// `ran-PagingCycle` in radio frames.
    pub ran_paging_cycle_rf: u16,
    /// The RAN Notification Area, if configured.
    pub ran_notification_area: Option<RanNotificationArea>,
    /// `t380` in minutes, if configured.
    pub t380_minutes: Option<u16>,
    /// `nextHopChainingCount`.
    pub next_hop_chaining_count: u8,
}

/// The short I-RNTI derived from a full one: its low 24 bits.
///
/// Named because **both ends** depend on the relation. The gNB stores the context
/// under the full identity and receives whichever form the UE sends; if the two were
/// unrelated, a `RRCResumeRequest` carrying the short form could not be resolved to a
/// stored context at all.
pub fn short_i_rnti_of(full_i_rnti: u64) -> u32 {
    (full_i_rnti & 0x00FF_FFFF) as u32
}

fn bits_of(value: u64, width: usize) -> BitVec<u8, Msb0> {
    let mut bv: BitVec<u8, Msb0> = BitVec::with_capacity(width);
    for i in (0..width).rev() {
        bv.push((value >> i) & 1 == 1);
    }
    bv
}

fn value_of(bits: &BitSlice<u8, Msb0>) -> u64 {
    bits.iter().fold(0u64, |acc, b| (acc << 1) | u64::from(*b))
}

/// Build a structured `SuspendConfig`.
pub fn build_suspend_config(
    params: &SuspendConfigParams,
) -> Result<SuspendConfig, SuspendConfigError> {
    if params.full_i_rnti > 0xFF_FFFF_FFFF {
        return Err(SuspendConfigError::IRntiTooWide(
            params.full_i_rnti,
            FULL_I_RNTI_BITS,
        ));
    }
    if params.next_hop_chaining_count > 7 {
        return Err(SuspendConfigError::InvalidNcc(
            params.next_hop_chaining_count,
        ));
    }
    let paging_cycle = PAGING_CYCLE_RF_TO_INDEX
        .iter()
        .find(|(rf, _)| *rf == params.ran_paging_cycle_rf)
        .map(|(_, idx)| *idx)
        .ok_or(SuspendConfigError::UnsupportedPagingCycle(
            params.ran_paging_cycle_rf,
        ))?;
    let t380 = match params.t380_minutes {
        None => None,
        Some(minutes) => Some(
            T380_MINUTES_TO_INDEX
                .iter()
                .find(|(m, _)| *m == minutes)
                .map(|(_, idx)| *idx)
                .ok_or(SuspendConfigError::UnsupportedT380(minutes))?,
        ),
    };
    let ran_notification_area_info = match &params.ran_notification_area {
        None => None,
        Some(RanNotificationArea::CellList(cells)) => {
            if cells.is_empty() || cells.len() > MAX_RNA_CELLS {
                return Err(SuspendConfigError::InvalidRnaCellCount(cells.len()));
            }
            Some(RAN_NotificationAreaInfo::CellList(PLMN_RAN_AreaCellList(
                vec![PLMN_RAN_AreaCell {
                    // The PLMN is omitted, which `PLMN-RAN-AreaCell` allows: the
                    // cells are in the UE's serving PLMN, and naming it here would
                    // let a gNB claim an area in a PLMN it does not serve.
                    plmn_identity: None,
                    ran_area_cells: PLMN_RAN_AreaCellRan_AreaCells(
                        cells
                            .iter()
                            .map(|nci| CellIdentity(bits_of(*nci, CELL_IDENTITY_BITS)))
                            .collect(),
                    ),
                }],
            )))
        }
    };

    Ok(SuspendConfig {
        full_i_rnti: I_RNTI_Value(bits_of(params.full_i_rnti, FULL_I_RNTI_BITS)),
        short_i_rnti: ShortI_RNTI_Value(bits_of(
            u64::from(short_i_rnti_of(params.full_i_rnti)),
            SHORT_I_RNTI_BITS,
        )),
        ran_paging_cycle: PagingCycle(paging_cycle),
        ran_notification_area_info,
        t380: t380.map(PeriodicRNAU_TimerValue),
        next_hop_chaining_count: NextHopChainingCount(params.next_hop_chaining_count),
    })
}

/// Read a structured `SuspendConfig`.
///
/// Enumerated values past this schema's set fall back to the widest legal value
/// rather than failing: a `SuspendConfig` the UE cannot fully parse still names an
/// I-RNTI it must use, and refusing the whole IE would leave the UE in IDLE
/// believing it had been released.
pub fn parse_suspend_config(config: &SuspendConfig) -> SuspendConfigData {
    let ran_paging_cycle_rf = PAGING_CYCLE_RF_TO_INDEX
        .iter()
        .find(|(_, idx)| *idx == config.ran_paging_cycle.0)
        .map(|(rf, _)| *rf)
        // The longest cycle: a UE that listened *more* often than configured would
        // spend power it was told not to, and one that listened less would miss pages.
        // Neither is right, and this at least matches the widest legal value.
        .unwrap_or(256);
    let t380_minutes = config.t380.as_ref().and_then(|t| {
        T380_MINUTES_TO_INDEX
            .iter()
            .find(|(_, idx)| *idx == t.0)
            .map(|(m, _)| *m)
    });
    let ran_notification_area =
        config
            .ran_notification_area_info
            .as_ref()
            .and_then(|info| match info {
                RAN_NotificationAreaInfo::CellList(list) => Some(RanNotificationArea::CellList(
                    list.0
                        .iter()
                        .flat_map(|per_plmn| {
                            per_plmn
                                .ran_area_cells
                                .0
                                .iter()
                                .map(|cell| value_of(&cell.0))
                        })
                        .collect(),
                )),
                // A RAN-area-coded area this simulator cannot map onto cells. `None`
                // rather than an empty list, because an empty area would make EVERY cell
                // a crossing and trigger an RNAU on the spot.
                RAN_NotificationAreaInfo::Ran_AreaConfigList(_) => None,
            });

    SuspendConfigData {
        full_i_rnti: value_of(&config.full_i_rnti.0),
        short_i_rnti: value_of(&config.short_i_rnti.0) as u32,
        ran_paging_cycle_rf,
        ran_notification_area,
        t380_minutes,
        next_hop_chaining_count: config.next_hop_chaining_count.0,
    }
}

/// Build and UPER-encode a `SuspendConfig` to the bytes
/// [`crate::procedures::rrc_release::RrcReleaseParams::suspend_config`] carries.
pub fn encode_suspend_config(params: &SuspendConfigParams) -> Result<Vec<u8>, SuspendConfigError> {
    Ok(encode_rrc(&build_suspend_config(params)?)?)
}

/// Decode and read a `SuspendConfig` from those bytes.
pub fn decode_suspend_config(bytes: &[u8]) -> Result<SuspendConfigData, SuspendConfigError> {
    let config: SuspendConfig = decode_rrc(bytes)?;
    Ok(parse_suspend_config(&config))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> SuspendConfigParams {
        SuspendConfigParams {
            full_i_rnti: 0x12_3456_789A,
            ran_paging_cycle_rf: 64,
            ran_notification_area: Some(RanNotificationArea::CellList(vec![
                0x0_0000_0010,
                0x0_0000_0011,
            ])),
            t380_minutes: Some(30),
            next_hop_chaining_count: 3,
        }
    }

    /// #38, criterion 1: every field of a `SuspendConfig` survives the wire.
    #[test]
    fn a_suspend_config_round_trips() {
        let p = params();
        let bytes = encode_suspend_config(&p).expect("encode");
        let data = decode_suspend_config(&bytes).expect("decode");
        assert_eq!(data.full_i_rnti, p.full_i_rnti);
        assert_eq!(data.short_i_rnti, short_i_rnti_of(p.full_i_rnti));
        assert_eq!(data.ran_paging_cycle_rf, 64);
        assert_eq!(data.t380_minutes, Some(30));
        assert_eq!(data.next_hop_chaining_count, 3);
        assert_eq!(
            data.ran_notification_area,
            Some(RanNotificationArea::CellList(vec![
                0x0_0000_0010,
                0x0_0000_0011
            ]))
        );
    }

    /// The two I-RNTIs name the same UE. Pinned because a gNB stores its context
    /// under the full identity and may receive either form.
    #[test]
    fn the_short_i_rnti_is_the_low_bits_of_the_full_one() {
        assert_eq!(short_i_rnti_of(0x12_3456_789A), 0x0056_789A);
        assert_eq!(short_i_rnti_of(0xFF_FFFF_FFFF), 0x00FF_FFFF);
        assert_eq!(short_i_rnti_of(0), 0);
        // And the encoded IE agrees with the helper, so the wire cannot disagree with
        // the lookup the gNB does.
        let data = decode_suspend_config(&encode_suspend_config(&params()).unwrap()).unwrap();
        assert_eq!(data.short_i_rnti, short_i_rnti_of(data.full_i_rnti));
    }

    /// Every enumerated `t380` and `ran-PagingCycle` round trips, and a value outside
    /// the set is refused rather than rounded.
    #[test]
    fn the_enumerated_timers_round_trip_and_others_are_refused() {
        for (minutes, _) in T380_MINUTES_TO_INDEX {
            let mut p = params();
            p.t380_minutes = Some(*minutes);
            let data = decode_suspend_config(&encode_suspend_config(&p).unwrap()).unwrap();
            assert_eq!(data.t380_minutes, Some(*minutes), "t380 {minutes} min");
        }
        for (rf, _) in PAGING_CYCLE_RF_TO_INDEX {
            let mut p = params();
            p.ran_paging_cycle_rf = *rf;
            let data = decode_suspend_config(&encode_suspend_config(&p).unwrap()).unwrap();
            assert_eq!(data.ran_paging_cycle_rf, *rf, "paging cycle rf{rf}");
        }
        let mut p = params();
        p.t380_minutes = Some(45);
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::UnsupportedT380(45)),
            "a non-enumerated t380 must be refused, not rounded: it decides when the \
             UE wakes for an RNAU"
        );
        let mut p = params();
        p.ran_paging_cycle_rf = 100;
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::UnsupportedPagingCycle(100))
        );
    }

    /// An absent `t380` disables the periodic RNAU and is distinguishable from a
    /// configured one.
    #[test]
    fn an_absent_t380_stays_absent() {
        let mut p = params();
        p.t380_minutes = None;
        let data = decode_suspend_config(&encode_suspend_config(&p).unwrap()).unwrap();
        assert_eq!(
            data.t380_minutes, None,
            "no periodic RNAU was configured, and inventing one would wake the UE"
        );
        // And the encodings differ, or the IE is signalled nowhere.
        assert_ne!(
            encode_suspend_config(&p).unwrap(),
            encode_suspend_config(&params()).unwrap()
        );
    }

    /// `contains` is the RNA-crossing predicate, so it must answer about the cells
    /// actually configured.
    #[test]
    fn the_notification_area_answers_which_cells_need_no_rnau() {
        let area = RanNotificationArea::CellList(vec![0x10, 0x11, 0x12]);
        assert!(area.contains(0x10));
        assert!(area.contains(0x12));
        assert!(
            !area.contains(0x13),
            "a cell outside the area is an RNA crossing (TS 38.304 §5.5)"
        );
        assert_eq!(area.len(), 3);
        assert!(!area.is_empty());
    }

    /// An I-RNTI wider than 40 bits, an NCC above 7, and an empty or oversized cell
    /// list are all refused at build time rather than silently truncated.
    #[test]
    fn out_of_range_values_are_refused_at_build_time() {
        let mut p = params();
        p.full_i_rnti = 0x0100_0000_0000;
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::IRntiTooWide(
                0x0100_0000_0000,
                FULL_I_RNTI_BITS
            )),
            "a truncated I-RNTI would name a different UE, or none"
        );
        let mut p = params();
        p.next_hop_chaining_count = 8;
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::InvalidNcc(8))
        );
        let mut p = params();
        p.ran_notification_area = Some(RanNotificationArea::CellList(vec![]));
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::InvalidRnaCellCount(0)),
            "an empty area would make every cell a crossing and trigger an RNAU at once"
        );
        let mut p = params();
        p.ran_notification_area = Some(RanNotificationArea::CellList(
            (0..MAX_RNA_CELLS as u64 + 1).collect(),
        ));
        assert_eq!(
            build_suspend_config(&p).err(),
            Some(SuspendConfigError::InvalidRnaCellCount(MAX_RNA_CELLS + 1))
        );
    }

    /// A 36-bit cell identity survives at both ends of its range, so the RNA cannot
    /// silently drop the high bits of an NCI.
    #[test]
    fn cell_identities_survive_the_full_36_bit_range() {
        for nci in [0u64, 1, 0xF_FFFF_FFFF, 0x8_0000_0001] {
            let mut p = params();
            p.ran_notification_area = Some(RanNotificationArea::CellList(vec![nci]));
            let data = decode_suspend_config(&encode_suspend_config(&p).unwrap()).unwrap();
            assert_eq!(
                data.ran_notification_area,
                Some(RanNotificationArea::CellList(vec![nci])),
                "NCI {nci:#x} must survive"
            );
        }
    }

    /// A RAN-area-coded notification area yields `None`, not an empty area.
    #[test]
    fn a_ran_area_coded_notification_area_is_none_and_not_empty() {
        let config = SuspendConfig {
            full_i_rnti: I_RNTI_Value(bits_of(1, FULL_I_RNTI_BITS)),
            short_i_rnti: ShortI_RNTI_Value(bits_of(1, SHORT_I_RNTI_BITS)),
            ran_paging_cycle: PagingCycle(PagingCycle::RF64),
            ran_notification_area_info: Some(RAN_NotificationAreaInfo::Ran_AreaConfigList(
                PLMN_RAN_AreaConfigList(vec![PLMN_RAN_AreaConfig {
                    plmn_identity: None,
                    ran_area: PLMN_RAN_AreaConfigRan_Area(vec![RAN_AreaConfig {
                        tracking_area_code: TrackingAreaCode(bits_of(1, 24)),
                        ran_area_code_list: None,
                    }]),
                }]),
            )),
            t380: None,
            next_hop_chaining_count: NextHopChainingCount(0),
        };
        let data = parse_suspend_config(&config);
        assert_eq!(
            data.ran_notification_area, None,
            "an area this simulator cannot map onto cells must be absent, not empty -- \
             an empty area makes every cell a crossing"
        );
        // The I-RNTI is still usable, which is the point of not refusing the IE.
        assert_eq!(data.full_i_rnti, 1);
    }
}
