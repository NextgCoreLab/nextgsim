//! The E-UTRA RRC messages the nextgsim eNB and LTE UE exchange, built from and
//! parsed into plain Rust values over the generated TS 36.331 types.
//!
//! Scope is RRC connection establishment and NAS transfer (nextgsim #186,
//! increment 2): system information (§5.2), RRC connection establishment
//! (§5.3.3) and DL/UL information transfer (§5.6.1, §5.6.2); and from
//! increment 3, initial security activation (§5.3.4) and the RRC connection
//! reconfiguration that sets up the default bearer's DRB and carries the
//! ATTACH ACCEPT (§5.3.5). Every builder
//! encodes the message's *logical channel* PDU (`BCCH-BCH-Message`,
//! `DL-CCCH-Message`, ...), which is what TS 36.331 §6.2.1 defines as the unit
//! RRC hands to the lower layers.

use asn1_codecs::uper::UperCodec;
use bitvec::prelude::*;

use crate::codec::{self, *};

/// Errors building or parsing an RRC message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RrcMsgError {
    /// UPER encode or decode failed.
    #[error("{0}")]
    Codec(String),
    /// A value does not fit the ASN.1 type it is encoded into.
    #[error("invalid value: {0}")]
    Invalid(String),
}

impl From<LteRrcCodecError> for RrcMsgError {
    fn from(e: LteRrcCodecError) -> Self {
        Self::Codec(e.to_string())
    }
}

fn enc<T: UperCodec>(v: &T) -> Result<Vec<u8>, RrcMsgError> {
    Ok(codec::encode(v)?)
}

fn dec<T: UperCodec<Output = T>>(b: &[u8]) -> Result<T, RrcMsgError> {
    Ok(codec::decode::<T>(b)?)
}

/// `bits` low-order bits of `value`, most significant first.
fn bits_of(value: u64, bits: usize) -> BitVec<u8, Msb0> {
    let mut v = BitVec::<u8, Msb0>::with_capacity(bits);
    for i in (0..bits).rev() {
        v.push((value >> i) & 1 == 1);
    }
    v
}

fn value_of(bits: &BitSlice<u8, Msb0>) -> u64 {
    bits.iter().fold(0u64, |acc, b| (acc << 1) | u64::from(*b))
}

/// A PLMN identity as MCC/MNC numbers (TS 23.003 §2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlmnId {
    /// Mobile Country Code, 0..=999.
    pub mcc: u16,
    /// Mobile Network Code, 0..=999.
    pub mnc: u16,
    /// True for a three-digit MNC.
    pub long_mnc: bool,
}

fn digits(value: u16, n: usize) -> Vec<MCC_MNC_Digit> {
    let s = format!("{value:0n$}");
    s.bytes().map(|b| MCC_MNC_Digit(b - b'0')).collect()
}

fn plmn_to_asn(p: PlmnId) -> Result<PLMN_Identity, RrcMsgError> {
    if p.mcc > 999 || p.mnc > 999 || (!p.long_mnc && p.mnc > 99) {
        return Err(RrcMsgError::Invalid(format!(
            "PLMN {}-{} (long_mnc={})",
            p.mcc, p.mnc, p.long_mnc
        )));
    }
    // §6.3.4 PLMN-Identity: the MCC is present whenever it differs from the
    // preceding entry's; carrying it always is valid and self-contained.
    Ok(PLMN_Identity {
        mcc: Some(MCC(digits(p.mcc, 3))),
        mnc: MNC(digits(p.mnc, if p.long_mnc { 3 } else { 2 })),
    })
}

fn digits_value(d: &[MCC_MNC_Digit]) -> u16 {
    d.iter().fold(0u16, |acc, x| acc * 10 + u16::from(x.0))
}

fn plmn_from_asn(p: &PLMN_Identity, previous_mcc: Option<u16>) -> Option<PlmnId> {
    let mcc = match &p.mcc {
        Some(m) => digits_value(&m.0),
        // §6.3.4: an absent mcc takes the value of the preceding entry's.
        None => previous_mcc?,
    };
    Some(PlmnId {
        mcc,
        mnc: digits_value(&p.mnc.0),
        long_mnc: p.mnc.0.len() == 3,
    })
}

// ---------------------------------------------------------------------------
// System information (TS 36.331 §5.2, §6.2.2)
// ---------------------------------------------------------------------------

/// `dl-Bandwidth` in resource blocks (§6.2.2 MasterInformationBlock).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlBandwidth {
    /// 6 RBs (1.4 MHz).
    N6,
    /// 15 RBs (3 MHz).
    N15,
    /// 25 RBs (5 MHz).
    N25,
    /// 50 RBs (10 MHz).
    N50,
    /// 75 RBs (15 MHz).
    N75,
    /// 100 RBs (20 MHz).
    N100,
}

impl DlBandwidth {
    fn to_asn(self) -> u8 {
        match self {
            Self::N6 => MasterInformationBlockDl_Bandwidth::N6,
            Self::N15 => MasterInformationBlockDl_Bandwidth::N15,
            Self::N25 => MasterInformationBlockDl_Bandwidth::N25,
            Self::N50 => MasterInformationBlockDl_Bandwidth::N50,
            Self::N75 => MasterInformationBlockDl_Bandwidth::N75,
            Self::N100 => MasterInformationBlockDl_Bandwidth::N100,
        }
    }
    fn from_asn(v: u8) -> Option<Self> {
        Some(match v {
            0 => Self::N6,
            1 => Self::N15,
            2 => Self::N25,
            3 => Self::N50,
            4 => Self::N75,
            5 => Self::N100,
            _ => return None,
        })
    }
}

/// The MIB content this crate models (§6.2.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MibInfo {
    /// Downlink transmission bandwidth.
    pub dl_bandwidth: DlBandwidth,
    /// The 8 most significant bits of the 10-bit SFN (§6.2.2
    /// `systemFrameNumber`: "the two least significant bits of the SFN are
    /// acquired implicitly"). Stored as the full SFN; only `sfn >> 2` is sent.
    pub sfn: u16,
}

/// Encode a `BCCH-BCH-Message` (MIB) for `info` (§5.2.1.2, §6.2.2).
///
/// PHICH configuration is fixed at normal duration / Ng = 1: RLS replaces the
/// physical layer, so the UE has no PHICH to decode, but the field is
/// mandatory and must carry a valid value. `schedulingInfoSIB1-BR-r13` = 0 means
/// "SIB1-BR is not scheduled" (this is not a BL/CE cell); the r15/r17 fields are
/// set to their "no information" values.
pub fn encode_mib(info: &MibInfo) -> Result<Vec<u8>, RrcMsgError> {
    if info.sfn > 1023 {
        return Err(RrcMsgError::Invalid(format!("SFN {} > 1023", info.sfn)));
    }
    let mib = MasterInformationBlock {
        dl_bandwidth: MasterInformationBlockDl_Bandwidth(info.dl_bandwidth.to_asn()),
        phich_config: PHICH_Config {
            phich_duration: PHICH_ConfigPhich_Duration(PHICH_ConfigPhich_Duration::NORMAL),
            phich_resource: PHICH_ConfigPhich_Resource(PHICH_ConfigPhich_Resource::ONE),
        },
        system_frame_number: MasterInformationBlockSystemFrameNumber(bits_of(
            u64::from(info.sfn >> 2),
            8,
        )),
        scheduling_info_sib1_br_r13: MasterInformationBlockSchedulingInfoSIB1_BR_r13(0),
        system_info_unchanged_br_r15: MasterInformationBlockSystemInfoUnchanged_BR_r15(false),
        part_earfcn_r17: MasterInformationBlockPartEARFCN_r17::Spare(
            MasterInformationBlockPartEARFCN_r17_spare(bits_of(0, 2)),
        ),
        spare: MasterInformationBlockSpare(bits_of(0, 1)),
    };
    enc(&BCCH_BCH_Message { message: mib })
}

/// Decode a `BCCH-BCH-Message`. The returned SFN has its two least significant
/// bits zero, since the MIB does not carry them (§6.2.2).
pub fn decode_mib(bytes: &[u8]) -> Result<MibInfo, RrcMsgError> {
    let msg: BCCH_BCH_Message = dec(bytes)?;
    let mib = msg.message;
    let dl_bandwidth = DlBandwidth::from_asn(mib.dl_bandwidth.0)
        .ok_or_else(|| RrcMsgError::Invalid("dl-Bandwidth".into()))?;
    Ok(MibInfo {
        dl_bandwidth,
        sfn: (value_of(&mib.system_frame_number.0) as u16) << 2,
    })
}

/// The cell parameters SIB1 and SIB2 broadcast.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellInfo {
    /// `plmn-IdentityList`, 1..=6 entries (§6.2.2 SystemInformationBlockType1).
    pub plmns: Vec<PlmnId>,
    /// `trackingAreaCode`.
    pub tac: u16,
    /// `cellIdentity`, 28 bits (the E-UTRAN Cell Identity, TS 36.300 §8.2).
    pub cell_identity: u32,
    /// `cellBarred`.
    pub cell_barred: bool,
    /// `q-RxLevMin` in dBm, -140..=-44 in steps of 2 (§6.3.4 Q-RxLevMin: the
    /// IE is the value divided by 2).
    pub q_rx_lev_min_dbm: i16,
    /// `freqBandIndicator`, 1..=64.
    pub freq_band: u8,
    /// `systemInfoValueTag`, 0..=31 (§5.2.1.3: changes when SI changes).
    pub system_info_value_tag: u8,
}

/// `ue-TimersAndConstants` as durations and counts (§6.3.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UeTimers {
    /// T300, ms: RRC connection establishment supervision (§7.3).
    pub t300_ms: u16,
    /// T301, ms.
    pub t301_ms: u16,
    /// T310, ms.
    pub t310_ms: u16,
    /// N310.
    pub n310: u8,
    /// T311, ms.
    pub t311_ms: u16,
    /// N311.
    pub n311: u8,
}

impl Default for UeTimers {
    /// T300 1000 ms, T301 1000 ms, T310 1000 ms, N310 1, T311 10000 ms, N311 1:
    /// common operator values within the §6.3.6 ranges.
    fn default() -> Self {
        Self {
            t300_ms: 1000,
            t301_ms: 1000,
            t310_ms: 1000,
            n310: 1,
            t311_ms: 10000,
            n311: 1,
        }
    }
}

const T300_T301_MS: [u16; 8] = [100, 200, 300, 400, 600, 1000, 1500, 2000];
const T310_MS: [u16; 7] = [0, 50, 100, 200, 500, 1000, 2000];
const N310: [u8; 8] = [1, 2, 3, 4, 6, 8, 10, 20];
const T311_MS: [u16; 7] = [1000, 3000, 5000, 10000, 15000, 20000, 30000];
const N311: [u8; 8] = [1, 2, 3, 4, 5, 6, 8, 10];

fn index_of<T: PartialEq + std::fmt::Display>(
    table: &[T],
    v: T,
    what: &str,
) -> Result<u8, RrcMsgError> {
    table
        .iter()
        .position(|x| *x == v)
        .map(|i| i as u8)
        .ok_or_else(|| RrcMsgError::Invalid(format!("{what} {v} is not an allowed value")))
}

fn entry<T: Copy>(table: &[T], i: u8, what: &str) -> Result<T, RrcMsgError> {
    table
        .get(usize::from(i))
        .copied()
        .ok_or_else(|| RrcMsgError::Invalid(format!("{what} index {i}")))
}

/// Default paging cycle in radio frames (§6.3.2 PCCH-Config; TS 36.304 §7.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PagingCycle {
    /// 32 radio frames.
    Rf32,
    /// 64 radio frames.
    Rf64,
    /// 128 radio frames.
    Rf128,
    /// 256 radio frames.
    Rf256,
}

/// The SIB2 content this crate models (§6.3.1 SystemInformationBlockType2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sib2Info {
    /// `ue-TimersAndConstants`.
    pub timers: UeTimers,
    /// `pcch-Config.defaultPagingCycle`.
    pub default_paging_cycle: PagingCycle,
}

fn sib1_asn(cell: &CellInfo) -> Result<SystemInformationBlockType1, RrcMsgError> {
    if cell.plmns.is_empty() || cell.plmns.len() > 6 {
        return Err(RrcMsgError::Invalid(format!(
            "{} PLMNs (plmn-IdentityList is 1..6)",
            cell.plmns.len()
        )));
    }
    if cell.cell_identity >= 1 << 28 {
        return Err(RrcMsgError::Invalid("cellIdentity exceeds 28 bits".into()));
    }
    if cell.q_rx_lev_min_dbm % 2 != 0 || !(-140..=-44).contains(&cell.q_rx_lev_min_dbm) {
        return Err(RrcMsgError::Invalid(format!(
            "q-RxLevMin {} dBm (must be even, -140..-44)",
            cell.q_rx_lev_min_dbm
        )));
    }
    if !(1..=64).contains(&cell.freq_band) || cell.system_info_value_tag > 31 {
        return Err(RrcMsgError::Invalid(
            "freqBandIndicator or systemInfoValueTag".into(),
        ));
    }
    let plmn_identity_list = cell
        .plmns
        .iter()
        .map(|p| {
            Ok(PLMN_IdentityInfo {
                plmn_identity: plmn_to_asn(*p)?,
                cell_reserved_for_operator_use: PLMN_IdentityInfoCellReservedForOperatorUse(
                    PLMN_IdentityInfoCellReservedForOperatorUse::NOT_RESERVED,
                ),
            })
        })
        .collect::<Result<Vec<_>, RrcMsgError>>()?;
    Ok(SystemInformationBlockType1 {
        cell_access_related_info: SystemInformationBlockType1CellAccessRelatedInfo {
            plmn_identity_list: PLMN_IdentityList(plmn_identity_list),
            tracking_area_code: TrackingAreaCode(bits_of(u64::from(cell.tac), 16)),
            cell_identity: CellIdentity(bits_of(u64::from(cell.cell_identity), 28)),
            cell_barred: SystemInformationBlockType1CellAccessRelatedInfoCellBarred(
                if cell.cell_barred {
                    SystemInformationBlockType1CellAccessRelatedInfoCellBarred::BARRED
                } else {
                    SystemInformationBlockType1CellAccessRelatedInfoCellBarred::NOT_BARRED
                },
            ),
            intra_freq_reselection:
                SystemInformationBlockType1CellAccessRelatedInfoIntraFreqReselection(
                    SystemInformationBlockType1CellAccessRelatedInfoIntraFreqReselection::ALLOWED,
                ),
            csg_indication: SystemInformationBlockType1CellAccessRelatedInfoCsg_Indication(false),
            csg_identity: None,
        },
        cell_selection_info: SystemInformationBlockType1CellSelectionInfo {
            q_rx_lev_min: Q_RxLevMin((cell.q_rx_lev_min_dbm / 2) as i8),
            q_rx_lev_min_offset: None,
        },
        p_max: None,
        freq_band_indicator: FreqBandIndicator(cell.freq_band),
        // §6.2.2 schedulingInfoList: "SystemInformationBlockType2 is always
        // present in the first SystemInformation message listed", so the first
        // entry's mapping is empty. One SI message, period 16 radio frames.
        scheduling_info_list: SchedulingInfoList(vec![SchedulingInfo {
            si_periodicity: SI_Periodicity_r12(SI_Periodicity_r12::RF16),
            sib_mapping_info: SIB_MappingInfo(vec![]),
        }]),
        tdd_config: None,
        si_window_length: SystemInformationBlockType1Si_WindowLength(
            SystemInformationBlockType1Si_WindowLength::MS20,
        ),
        system_info_value_tag: SystemInformationBlockType1SystemInfoValueTag(
            cell.system_info_value_tag,
        ),
        non_critical_extension: None,
    })
}

/// Encode a `BCCH-DL-SCH-Message` carrying SIB1 (§5.2.1.2, §6.2.2).
///
/// FDD (no `tdd-Config`), no CSG, intra-frequency reselection allowed, no
/// `p-Max` (the UE then uses its maximum, §6.3.4).
pub fn encode_sib1(cell: &CellInfo) -> Result<Vec<u8>, RrcMsgError> {
    enc(&BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(
            BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(sib1_asn(cell)?),
        ),
    })
}

fn sib2_asn(sib2: &Sib2Info) -> Result<SystemInformationBlockType2, RrcMsgError> {
    let t = sib2.timers;
    let paging = match sib2.default_paging_cycle {
        PagingCycle::Rf32 => PCCH_ConfigDefaultPagingCycle::RF32,
        PagingCycle::Rf64 => PCCH_ConfigDefaultPagingCycle::RF64,
        PagingCycle::Rf128 => PCCH_ConfigDefaultPagingCycle::RF128,
        PagingCycle::Rf256 => PCCH_ConfigDefaultPagingCycle::RF256,
    };
    // The physical-layer common configuration below is mandatory in
    // RadioResourceConfigCommonSIB, so it must carry valid values. RLS replaces
    // the physical layer and MAC, so the UE does not act on them; the values are
    // the ones a typical 10 MHz FDD cell broadcasts.
    let rrc_common = RadioResourceConfigCommonSIB {
        extensions: Default::default(),
        rach_config_common: RACH_ConfigCommon {
            extensions: Default::default(),
            preamble_info: RACH_ConfigCommonPreambleInfo {
                number_of_ra_preambles: RACH_ConfigCommonPreambleInfoNumberOfRA_Preambles(
                    RACH_ConfigCommonPreambleInfoNumberOfRA_Preambles::N52,
                ),
                preambles_group_a_config: None,
            },
            power_ramping_parameters: PowerRampingParameters {
                power_ramping_step: PowerRampingParametersPowerRampingStep(
                    PowerRampingParametersPowerRampingStep::D_B2,
                ),
                preamble_initial_received_target_power:
                    PowerRampingParametersPreambleInitialReceivedTargetPower(
                        PowerRampingParametersPreambleInitialReceivedTargetPower::D_BM_104,
                    ),
            },
            ra_supervision_info: RACH_ConfigCommonRa_SupervisionInfo {
                preamble_trans_max: PreambleTransMax(PreambleTransMax::N10),
                ra_response_window_size: RACH_ConfigCommonRa_SupervisionInfoRa_ResponseWindowSize(
                    RACH_ConfigCommonRa_SupervisionInfoRa_ResponseWindowSize::SF10,
                ),
                mac_contention_resolution_timer:
                    RACH_ConfigCommonRa_SupervisionInfoMac_ContentionResolutionTimer(
                        RACH_ConfigCommonRa_SupervisionInfoMac_ContentionResolutionTimer::SF64,
                    ),
            },
            max_harq_msg3_tx: RACH_ConfigCommonMaxHARQ_Msg3Tx(4),
        },
        bcch_config: BCCH_Config {
            modification_period_coeff: BCCH_ConfigModificationPeriodCoeff(
                BCCH_ConfigModificationPeriodCoeff::N4,
            ),
        },
        pcch_config: PCCH_Config {
            default_paging_cycle: PCCH_ConfigDefaultPagingCycle(paging),
            nb: PCCH_ConfigNB(PCCH_ConfigNB::ONE_T),
        },
        prach_config: PRACH_ConfigSIB {
            root_sequence_index: PRACH_ConfigSIBRootSequenceIndex(128),
            prach_config_info: PRACH_ConfigInfo {
                prach_config_index: PRACH_ConfigInfoPrach_ConfigIndex(3),
                high_speed_flag: PRACH_ConfigInfoHighSpeedFlag(false),
                zero_correlation_zone_config: PRACH_ConfigInfoZeroCorrelationZoneConfig(5),
                prach_freq_offset: PRACH_ConfigInfoPrach_FreqOffset(4),
            },
        },
        pdsch_config_common: PDSCH_ConfigCommon {
            reference_signal_power: PDSCH_ConfigCommonReferenceSignalPower(18),
            p_b: PDSCH_ConfigCommonP_b(0),
        },
        pusch_config_common: PUSCH_ConfigCommon {
            pusch_config_basic: PUSCH_ConfigCommonPusch_ConfigBasic {
                n_sb: PUSCH_ConfigCommonPusch_ConfigBasicN_SB(1),
                hopping_mode: PUSCH_ConfigCommonPusch_ConfigBasicHoppingMode(
                    PUSCH_ConfigCommonPusch_ConfigBasicHoppingMode::INTER_SUB_FRAME,
                ),
                pusch_hopping_offset: PUSCH_ConfigCommonPusch_ConfigBasicPusch_HoppingOffset(2),
                enable64_qam: PUSCH_ConfigCommonPusch_ConfigBasicEnable64QAM(false),
            },
            ul_reference_signals_pusch: UL_ReferenceSignalsPUSCH {
                group_hopping_enabled: UL_ReferenceSignalsPUSCHGroupHoppingEnabled(false),
                group_assignment_pusch: UL_ReferenceSignalsPUSCHGroupAssignmentPUSCH(0),
                sequence_hopping_enabled: UL_ReferenceSignalsPUSCHSequenceHoppingEnabled(false),
                cyclic_shift: UL_ReferenceSignalsPUSCHCyclicShift(0),
            },
        },
        pucch_config_common: PUCCH_ConfigCommon {
            delta_pucch_shift: PUCCH_ConfigCommonDeltaPUCCH_Shift(
                PUCCH_ConfigCommonDeltaPUCCH_Shift::DS1,
            ),
            nrb_cqi: PUCCH_ConfigCommonNRB_CQI(1),
            ncs_an: PUCCH_ConfigCommonNCS_AN(0),
            n1pucch_an: PUCCH_ConfigCommonN1PUCCH_AN(12),
        },
        sounding_rs_ul_config_common: SoundingRS_UL_ConfigCommon::Release(
            SoundingRS_UL_ConfigCommon_release,
        ),
        uplink_power_control_common: UplinkPowerControlCommon {
            p0_nominal_pusch: UplinkPowerControlCommonP0_NominalPUSCH(-85),
            alpha: Alpha_r12(Alpha_r12::AL07),
            p0_nominal_pucch: UplinkPowerControlCommonP0_NominalPUCCH(-107),
            delta_f_list_pucch: DeltaFList_PUCCH {
                delta_f_pucch_format1: DeltaFList_PUCCHDeltaF_PUCCH_Format1(
                    DeltaFList_PUCCHDeltaF_PUCCH_Format1::DELTA_F0,
                ),
                delta_f_pucch_format1b: DeltaFList_PUCCHDeltaF_PUCCH_Format1b(
                    DeltaFList_PUCCHDeltaF_PUCCH_Format1b::DELTA_F3,
                ),
                delta_f_pucch_format2: DeltaFList_PUCCHDeltaF_PUCCH_Format2(
                    DeltaFList_PUCCHDeltaF_PUCCH_Format2::DELTA_F0,
                ),
                delta_f_pucch_format2a: DeltaFList_PUCCHDeltaF_PUCCH_Format2a(
                    DeltaFList_PUCCHDeltaF_PUCCH_Format2a::DELTA_F0,
                ),
                delta_f_pucch_format2b: DeltaFList_PUCCHDeltaF_PUCCH_Format2b(
                    DeltaFList_PUCCHDeltaF_PUCCH_Format2b::DELTA_F0,
                ),
            },
            delta_preamble_msg3: UplinkPowerControlCommonDeltaPreambleMsg3(4),
        },
        ul_cyclic_prefix_length: UL_CyclicPrefixLength(UL_CyclicPrefixLength::LEN1),
    };
    Ok(SystemInformationBlockType2 {
        extensions: Default::default(),
        ac_barring_info: None,
        radio_resource_config_common: rrc_common,
        ue_timers_and_constants: UE_TimersAndConstants {
            extensions: Default::default(),
            t300: UE_TimersAndConstantsT300(index_of(&T300_T301_MS, t.t300_ms, "T300")?),
            t301: UE_TimersAndConstantsT301(index_of(&T300_T301_MS, t.t301_ms, "T301")?),
            t310: UE_TimersAndConstantsT310(index_of(&T310_MS, t.t310_ms, "T310")?),
            n310: UE_TimersAndConstantsN310(index_of(&N310, t.n310, "N310")?),
            t311: UE_TimersAndConstantsT311(index_of(&T311_MS, t.t311_ms, "T311")?),
            n311: UE_TimersAndConstantsN311(index_of(&N311, t.n311, "N311")?),
        },
        // §6.3.1: absent ul-CarrierFreq means the default TX-RX separation of
        // the band (TS 36.101); absent ul-Bandwidth means equal to DL.
        freq_info: SystemInformationBlockType2FreqInfo {
            ul_carrier_freq: None,
            ul_bandwidth: None,
            additional_spectrum_emission: AdditionalSpectrumEmission(1),
        },
        mbsfn_subframe_config_list: None,
        time_alignment_timer_common: TimeAlignmentTimer(TimeAlignmentTimer::INFINITY),
    })
}

/// Encode a `BCCH-DL-SCH-Message` carrying the SystemInformation message that
/// holds SIB2 (§5.2.1.2, §6.2.2 SystemInformation).
pub fn encode_sib2_si(sib2: &Sib2Info) -> Result<Vec<u8>, RrcMsgError> {
    let si = SystemInformation {
        critical_extensions: SystemInformationCriticalExtensions::SystemInformation_r8(
            SystemInformation_r8_IEs {
                sib_type_and_info: SystemInformation_r8_IEsSib_TypeAndInfo(vec![
                    SystemInformation_r8_IEsSib_TypeAndInfo_Entry::Sib2(sib2_asn(sib2)?),
                ]),
                non_critical_extension: None,
            },
        ),
    };
    enc(&BCCH_DL_SCH_Message {
        message: BCCH_DL_SCH_MessageType::C1(BCCH_DL_SCH_MessageType_c1::SystemInformation(si)),
    })
}

/// A decoded `BCCH-DL-SCH-Message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BcchDlSch {
    /// SIB1.
    Sib1(CellInfo),
    /// A SystemInformation message; `sib2` is `Some` when it carries SIB2.
    SystemInformation {
        /// The SIB2 it carries, if any.
        sib2: Option<Sib2Info>,
    },
    /// A message class extension this build does not interpret.
    Other,
}

/// Decode a `BCCH-DL-SCH-Message`.
pub fn decode_bcch_dl_sch(bytes: &[u8]) -> Result<BcchDlSch, RrcMsgError> {
    let msg: BCCH_DL_SCH_Message = dec(bytes)?;
    let BCCH_DL_SCH_MessageType::C1(c1) = msg.message else {
        return Ok(BcchDlSch::Other);
    };
    match c1 {
        BCCH_DL_SCH_MessageType_c1::SystemInformationBlockType1(s) => {
            let cari = &s.cell_access_related_info;
            let mut plmns = Vec::new();
            let mut prev = None;
            for info in &cari.plmn_identity_list.0 {
                let p = plmn_from_asn(&info.plmn_identity, prev)
                    .ok_or_else(|| RrcMsgError::Invalid("first PLMN has no MCC".into()))?;
                prev = Some(p.mcc);
                plmns.push(p);
            }
            Ok(BcchDlSch::Sib1(CellInfo {
                plmns,
                tac: value_of(&cari.tracking_area_code.0) as u16,
                cell_identity: value_of(&cari.cell_identity.0) as u32,
                cell_barred: cari.cell_barred.0
                    == SystemInformationBlockType1CellAccessRelatedInfoCellBarred::BARRED,
                q_rx_lev_min_dbm: i16::from(s.cell_selection_info.q_rx_lev_min.0) * 2,
                freq_band: s.freq_band_indicator.0,
                system_info_value_tag: s.system_info_value_tag.0,
            }))
        }
        BCCH_DL_SCH_MessageType_c1::SystemInformation(si) => {
            let SystemInformationCriticalExtensions::SystemInformation_r8(r8) =
                si.critical_extensions
            else {
                return Ok(BcchDlSch::SystemInformation { sib2: None });
            };
            let mut sib2 = None;
            for e in r8.sib_type_and_info.0 {
                if let SystemInformation_r8_IEsSib_TypeAndInfo_Entry::Sib2(s) = e {
                    let u = &s.ue_timers_and_constants;
                    let paging = match s
                        .radio_resource_config_common
                        .pcch_config
                        .default_paging_cycle
                        .0
                    {
                        0 => PagingCycle::Rf32,
                        1 => PagingCycle::Rf64,
                        2 => PagingCycle::Rf128,
                        _ => PagingCycle::Rf256,
                    };
                    sib2 = Some(Sib2Info {
                        timers: UeTimers {
                            t300_ms: entry(&T300_T301_MS, u.t300.0, "T300")?,
                            t301_ms: entry(&T300_T301_MS, u.t301.0, "T301")?,
                            t310_ms: entry(&T310_MS, u.t310.0, "T310")?,
                            n310: entry(&N310, u.n310.0, "N310")?,
                            t311_ms: entry(&T311_MS, u.t311.0, "T311")?,
                            n311: entry(&N311, u.n311.0, "N311")?,
                        },
                        default_paging_cycle: paging,
                    });
                }
            }
            Ok(BcchDlSch::SystemInformation { sib2 })
        }
    }
}

// ---------------------------------------------------------------------------
// RRC connection establishment (TS 36.331 §5.3.3)
// ---------------------------------------------------------------------------

/// `ue-Identity` of RRCConnectionRequest (§6.2.2 InitialUE-Identity).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitialUeIdentity {
    /// `s-TMSI`: MMEC (8 bits) and M-TMSI (32 bits), when upper layers provide
    /// one (§5.3.3.3).
    STmsi {
        /// MME Code.
        mmec: u8,
        /// M-TMSI.
        m_tmsi: u32,
    },
    /// `randomValue`, 40 bits, otherwise (§5.3.3.3: "draw a random value in the
    /// range 0 .. 2^40-1").
    Random(u64),
}

/// `establishmentCause` (§6.2.2 EstablishmentCause). The S1AP RRC
/// Establishment Cause has the same values in the same order
/// (TS 36.413 §9.2.1.3a).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstablishmentCauseValue {
    /// emergency.
    Emergency,
    /// highPriorityAccess.
    HighPriorityAccess,
    /// mt-Access.
    MtAccess,
    /// mo-Signalling.
    MoSignalling,
    /// mo-Data.
    MoData,
    /// delayTolerantAccess-v1020.
    DelayTolerantAccess,
    /// mo-VoiceCall-v1280.
    MoVoiceCall,
}

impl EstablishmentCauseValue {
    /// The enumeration index (§6.2.2 and TS 36.413 §9.2.1.3a share it).
    pub fn index(self) -> u8 {
        match self {
            Self::Emergency => EstablishmentCause::EMERGENCY,
            Self::HighPriorityAccess => EstablishmentCause::HIGH_PRIORITY_ACCESS,
            Self::MtAccess => EstablishmentCause::MT_ACCESS,
            Self::MoSignalling => EstablishmentCause::MO_SIGNALLING,
            Self::MoData => EstablishmentCause::MO_DATA,
            Self::DelayTolerantAccess => EstablishmentCause::DELAY_TOLERANT_ACCESS_V1020,
            Self::MoVoiceCall => EstablishmentCause::MO_VOICE_CALL_V1280,
        }
    }
    fn from_index(i: u8) -> Option<Self> {
        Some(match i {
            0 => Self::Emergency,
            1 => Self::HighPriorityAccess,
            2 => Self::MtAccess,
            3 => Self::MoSignalling,
            4 => Self::MoData,
            5 => Self::DelayTolerantAccess,
            6 => Self::MoVoiceCall,
            _ => return None,
        })
    }
}

/// Encode a `UL-CCCH-Message` carrying RRCConnectionRequest (§5.3.3.3).
pub fn encode_rrc_connection_request(
    identity: InitialUeIdentity,
    cause: EstablishmentCauseValue,
) -> Result<Vec<u8>, RrcMsgError> {
    let ue_identity = match identity {
        InitialUeIdentity::STmsi { mmec, m_tmsi } => InitialUE_Identity::S_TMSI(S_TMSI {
            mmec: MMEC(bits_of(u64::from(mmec), 8)),
            m_tmsi: S_TMSIM_TMSI(bits_of(u64::from(m_tmsi), 32)),
        }),
        InitialUeIdentity::Random(v) => {
            if v >= 1 << 40 {
                return Err(RrcMsgError::Invalid("randomValue exceeds 40 bits".into()));
            }
            InitialUE_Identity::RandomValue(InitialUE_Identity_randomValue(bits_of(v, 40)))
        }
    };
    enc(&UL_CCCH_Message {
        message: UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcConnectionRequest(
            RRCConnectionRequest {
                critical_extensions:
                    RRCConnectionRequestCriticalExtensions::RrcConnectionRequest_r8(
                        RRCConnectionRequest_r8_IEs {
                            ue_identity,
                            establishment_cause: EstablishmentCause(cause.index()),
                            spare: RRCConnectionRequest_r8_IEsSpare(bits_of(0, 1)),
                        },
                    ),
            },
        )),
    })
}

/// A decoded `UL-CCCH-Message`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UlCcch {
    /// RRCConnectionRequest.
    ConnectionRequest {
        /// `ue-Identity`.
        identity: InitialUeIdentity,
        /// `establishmentCause`.
        cause: EstablishmentCauseValue,
    },
    /// A message this build does not handle (re-establishment, the 5GC
    /// variant, or a message class extension); the eNB ignores it.
    Unsupported,
}

/// Decode a `UL-CCCH-Message`.
pub fn decode_ul_ccch(bytes: &[u8]) -> Result<UlCcch, RrcMsgError> {
    let msg: UL_CCCH_Message = dec(bytes)?;
    let UL_CCCH_MessageType::C1(UL_CCCH_MessageType_c1::RrcConnectionRequest(req)) = msg.message
    else {
        return Ok(UlCcch::Unsupported);
    };
    let RRCConnectionRequestCriticalExtensions::RrcConnectionRequest_r8(r8) =
        req.critical_extensions
    else {
        return Ok(UlCcch::Unsupported);
    };
    let identity = match r8.ue_identity {
        InitialUE_Identity::S_TMSI(s) => InitialUeIdentity::STmsi {
            mmec: value_of(&s.mmec.0) as u8,
            m_tmsi: value_of(&s.m_tmsi.0) as u32,
        },
        InitialUE_Identity::RandomValue(r) => InitialUeIdentity::Random(value_of(&r.0)),
    };
    let cause = EstablishmentCauseValue::from_index(r8.establishment_cause.0)
        .ok_or_else(|| RrcMsgError::Invalid("establishmentCause spare".into()))?;
    Ok(UlCcch::ConnectionRequest { identity, cause })
}

/// Encode a `DL-CCCH-Message` carrying RRCConnectionSetup (§5.3.3.4).
///
/// It configures SRB1 with the default RLC and logical-channel configuration
/// (§9.2.1.1) and the default MAC main configuration (§9.2.2). No
/// `physicalConfigDedicated` is sent: the field is optional ("Need ON") and RLS
/// replaces the physical layer, so the UE keeps the default physical channel
/// configuration (§9.2.4).
pub fn encode_rrc_connection_setup(transaction_id: u8) -> Result<Vec<u8>, RrcMsgError> {
    if transaction_id > 3 {
        return Err(RrcMsgError::Invalid("rrc-TransactionIdentifier > 3".into()));
    }
    let setup = RRCConnectionSetup {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: RRCConnectionSetupCriticalExtensions::C1(
            RRCConnectionSetupCriticalExtensions_c1::RrcConnectionSetup_r8(
                RRCConnectionSetup_r8_IEs {
                    radio_resource_config_dedicated: RadioResourceConfigDedicated {
                        extensions: Default::default(),
                        srb_to_add_mod_list: Some(SRB_ToAddModList(vec![SRB_ToAddMod {
                            extensions: Default::default(),
                            srb_identity: SRB_ToAddModSrb_Identity(1),
                            rlc_config: Some(SRB_ToAddModRlc_Config::DefaultValue(
                                SRB_ToAddModRlc_Config_defaultValue,
                            )),
                            logical_channel_config: Some(
                                SRB_ToAddModLogicalChannelConfig::DefaultValue(
                                    SRB_ToAddModLogicalChannelConfig_defaultValue,
                                ),
                            ),
                        }])),
                        drb_to_add_mod_list: None,
                        drb_to_release_list: None,
                        mac_main_config: Some(
                            RadioResourceConfigDedicatedMac_MainConfig::DefaultValue(
                                RadioResourceConfigDedicatedMac_MainConfig_defaultValue,
                            ),
                        ),
                        sps_config: None,
                        physical_config_dedicated: None,
                    },
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&DL_CCCH_Message {
        message: DL_CCCH_MessageType::C1(DL_CCCH_MessageType_c1::RrcConnectionSetup(setup)),
    })
}

/// Encode a `DL-CCCH-Message` carrying RRCConnectionReject (§5.3.3.8).
/// `wait_time_s` is 1..=16 seconds (§6.2.2 RRCConnectionReject `waitTime`).
pub fn encode_rrc_connection_reject(wait_time_s: u8) -> Result<Vec<u8>, RrcMsgError> {
    if !(1..=16).contains(&wait_time_s) {
        return Err(RrcMsgError::Invalid(format!("waitTime {wait_time_s} s")));
    }
    let reject = RRCConnectionReject {
        critical_extensions: RRCConnectionRejectCriticalExtensions::C1(
            RRCConnectionRejectCriticalExtensions_c1::RrcConnectionReject_r8(
                RRCConnectionReject_r8_IEs {
                    wait_time: RRCConnectionReject_r8_IEsWaitTime(wait_time_s),
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&DL_CCCH_Message {
        message: DL_CCCH_MessageType::C1(DL_CCCH_MessageType_c1::RrcConnectionReject(reject)),
    })
}

/// A decoded `DL-CCCH-Message`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlCcch {
    /// RRCConnectionSetup.
    ConnectionSetup {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
        /// True when `srb-ToAddModList` configures SRB1, which §5.3.3.4 requires
        /// the UE to establish before sending RRCConnectionSetupComplete.
        configures_srb1: bool,
    },
    /// RRCConnectionReject.
    ConnectionReject {
        /// `waitTime`, seconds.
        wait_time_s: u8,
    },
    /// A message this build does not handle.
    Unsupported,
}

/// Decode a `DL-CCCH-Message`.
pub fn decode_dl_ccch(bytes: &[u8]) -> Result<DlCcch, RrcMsgError> {
    let msg: DL_CCCH_Message = dec(bytes)?;
    let DL_CCCH_MessageType::C1(c1) = msg.message else {
        return Ok(DlCcch::Unsupported);
    };
    Ok(match c1 {
        DL_CCCH_MessageType_c1::RrcConnectionSetup(s) => {
            let configures_srb1 = match &s.critical_extensions {
                RRCConnectionSetupCriticalExtensions::C1(
                    RRCConnectionSetupCriticalExtensions_c1::RrcConnectionSetup_r8(r8),
                ) => r8
                    .radio_resource_config_dedicated
                    .srb_to_add_mod_list
                    .as_ref()
                    .is_some_and(|l| l.0.iter().any(|m| m.srb_identity.0 == 1)),
                _ => false,
            };
            DlCcch::ConnectionSetup {
                transaction_id: s.rrc_transaction_identifier.0,
                configures_srb1,
            }
        }
        DL_CCCH_MessageType_c1::RrcConnectionReject(r) => match r.critical_extensions {
            RRCConnectionRejectCriticalExtensions::C1(
                RRCConnectionRejectCriticalExtensions_c1::RrcConnectionReject_r8(r8),
            ) => DlCcch::ConnectionReject {
                wait_time_s: r8.wait_time.0,
            },
            _ => DlCcch::Unsupported,
        },
        _ => DlCcch::Unsupported,
    })
}

// ---------------------------------------------------------------------------
// DCCH: setup complete and NAS transfer (§5.3.3.4, §5.6.1, §5.6.2)
// ---------------------------------------------------------------------------

/// Encode a `UL-DCCH-Message` carrying RRCConnectionSetupComplete (§5.3.3.4).
///
/// `selected_plmn_index` is the 1-based index of the selected PLMN in SIB1's
/// `plmn-IdentityList` (§6.2.2 `selectedPLMN-Identity`, 1..maxPLMN-r11 = 6).
pub fn encode_rrc_connection_setup_complete(
    transaction_id: u8,
    selected_plmn_index: u8,
    nas: &[u8],
) -> Result<Vec<u8>, RrcMsgError> {
    if transaction_id > 3 || !(1..=6).contains(&selected_plmn_index) {
        return Err(RrcMsgError::Invalid(
            "rrc-TransactionIdentifier or selectedPLMN-Identity".into(),
        ));
    }
    let complete = RRCConnectionSetupComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: RRCConnectionSetupCompleteCriticalExtensions::C1(
            RRCConnectionSetupCompleteCriticalExtensions_c1::RrcConnectionSetupComplete_r8(
                RRCConnectionSetupComplete_r8_IEs {
                    selected_plmn_identity: RRCConnectionSetupComplete_r8_IEsSelectedPLMN_Identity(
                        selected_plmn_index,
                    ),
                    registered_mme: None,
                    dedicated_info_nas: DedicatedInfoNAS(nas.to_vec()),
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::RrcConnectionSetupComplete(
            complete,
        )),
    })
}

/// Encode a `UL-DCCH-Message` carrying ULInformationTransfer with
/// `dedicatedInfoNAS` (§5.6.2).
pub fn encode_ul_information_transfer(nas: &[u8]) -> Result<Vec<u8>, RrcMsgError> {
    let ul = ULInformationTransfer {
        critical_extensions: ULInformationTransferCriticalExtensions::C1(
            ULInformationTransferCriticalExtensions_c1::UlInformationTransfer_r8(
                ULInformationTransfer_r8_IEs {
                    dedicated_info_type:
                        ULInformationTransfer_r8_IEsDedicatedInfoType::DedicatedInfoNAS(
                            DedicatedInfoNAS(nas.to_vec()),
                        ),
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::UlInformationTransfer(ul)),
    })
}

/// A decoded `UL-DCCH-Message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UlDcch {
    /// RRCConnectionSetupComplete.
    ConnectionSetupComplete {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
        /// `selectedPLMN-Identity` (1-based index into SIB1's list).
        selected_plmn_index: u8,
        /// `dedicatedInfoNAS`.
        nas: Vec<u8>,
    },
    /// ULInformationTransfer carrying `dedicatedInfoNAS`.
    InformationTransfer {
        /// `dedicatedInfoNAS`.
        nas: Vec<u8>,
    },
    /// SecurityModeComplete (§5.3.4.3).
    SecurityModeComplete {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
    },
    /// SecurityModeFailure (§5.3.4.3).
    SecurityModeFailure {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
    },
    /// RRCConnectionReconfigurationComplete (§5.3.5.3).
    ReconfigurationComplete {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
    },
    /// A message this build does not handle.
    Unsupported,
}

/// Decode a `UL-DCCH-Message`.
pub fn decode_ul_dcch(bytes: &[u8]) -> Result<UlDcch, RrcMsgError> {
    let msg: UL_DCCH_Message = dec(bytes)?;
    let UL_DCCH_MessageType::C1(c1) = msg.message else {
        return Ok(UlDcch::Unsupported);
    };
    Ok(match c1 {
        UL_DCCH_MessageType_c1::RrcConnectionSetupComplete(c) => match c.critical_extensions {
            RRCConnectionSetupCompleteCriticalExtensions::C1(
                RRCConnectionSetupCompleteCriticalExtensions_c1::RrcConnectionSetupComplete_r8(r8),
            ) => UlDcch::ConnectionSetupComplete {
                transaction_id: c.rrc_transaction_identifier.0,
                selected_plmn_index: r8.selected_plmn_identity.0,
                nas: r8.dedicated_info_nas.0,
            },
            _ => UlDcch::Unsupported,
        },
        UL_DCCH_MessageType_c1::UlInformationTransfer(u) => match u.critical_extensions {
            ULInformationTransferCriticalExtensions::C1(
                ULInformationTransferCriticalExtensions_c1::UlInformationTransfer_r8(r8),
            ) => match r8.dedicated_info_type {
                ULInformationTransfer_r8_IEsDedicatedInfoType::DedicatedInfoNAS(n) => {
                    UlDcch::InformationTransfer { nas: n.0 }
                }
                _ => UlDcch::Unsupported,
            },
            _ => UlDcch::Unsupported,
        },
        UL_DCCH_MessageType_c1::SecurityModeComplete(c) => UlDcch::SecurityModeComplete {
            transaction_id: c.rrc_transaction_identifier.0,
        },
        UL_DCCH_MessageType_c1::SecurityModeFailure(f) => UlDcch::SecurityModeFailure {
            transaction_id: f.rrc_transaction_identifier.0,
        },
        UL_DCCH_MessageType_c1::RrcConnectionReconfigurationComplete(c) => {
            UlDcch::ReconfigurationComplete {
                transaction_id: c.rrc_transaction_identifier.0,
            }
        }
        _ => UlDcch::Unsupported,
    })
}

/// Encode a `DL-DCCH-Message` carrying DLInformationTransfer with
/// `dedicatedInfoNAS` (§5.6.1).
pub fn encode_dl_information_transfer(
    transaction_id: u8,
    nas: &[u8],
) -> Result<Vec<u8>, RrcMsgError> {
    if transaction_id > 3 {
        return Err(RrcMsgError::Invalid("rrc-TransactionIdentifier > 3".into()));
    }
    let dl = DLInformationTransfer {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: DLInformationTransferCriticalExtensions::C1(
            DLInformationTransferCriticalExtensions_c1::DlInformationTransfer_r8(
                DLInformationTransfer_r8_IEs {
                    dedicated_info_type:
                        DLInformationTransfer_r8_IEsDedicatedInfoType::DedicatedInfoNAS(
                            DedicatedInfoNAS(nas.to_vec()),
                        ),
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::DlInformationTransfer(dl)),
    })
}

/// Encode a `DL-DCCH-Message` carrying RRCConnectionRelease (§5.3.8) with
/// `releaseCause` other, no redirection and no idle-mode mobility control.
pub fn encode_rrc_connection_release(transaction_id: u8) -> Result<Vec<u8>, RrcMsgError> {
    if transaction_id > 3 {
        return Err(RrcMsgError::Invalid("rrc-TransactionIdentifier > 3".into()));
    }
    let release = RRCConnectionRelease {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: RRCConnectionReleaseCriticalExtensions::C1(
            RRCConnectionReleaseCriticalExtensions_c1::RrcConnectionRelease_r8(
                RRCConnectionRelease_r8_IEs {
                    release_cause: ReleaseCause(ReleaseCause::OTHER),
                    redirected_carrier_info: None,
                    idle_mode_mobility_control_info: None,
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcConnectionRelease(release)),
    })
}

/// A decoded `DL-DCCH-Message`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DlDcch {
    /// DLInformationTransfer carrying `dedicatedInfoNAS`.
    InformationTransfer {
        /// `dedicatedInfoNAS`.
        nas: Vec<u8>,
    },
    /// RRCConnectionRelease.
    ConnectionRelease,
    /// SecurityModeCommand (§5.3.4).
    SecurityModeCommand {
        /// `rrc-TransactionIdentifier`.
        transaction_id: u8,
        /// `cipheringAlgorithm` (EEA identity).
        eea: u8,
        /// `integrityProtAlgorithm` (EIA identity).
        eia: u8,
    },
    /// RRCConnectionReconfiguration (§5.3.5).
    ConnectionReconfiguration(Reconfiguration),
    /// A message this build does not handle.
    Unsupported,
}

/// Decode a `DL-DCCH-Message`.
pub fn decode_dl_dcch(bytes: &[u8]) -> Result<DlDcch, RrcMsgError> {
    let msg: DL_DCCH_Message = dec(bytes)?;
    Ok(match msg.message {
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::DlInformationTransfer(d)) => {
            match d.critical_extensions {
                DLInformationTransferCriticalExtensions::C1(
                    DLInformationTransferCriticalExtensions_c1::DlInformationTransfer_r8(r8),
                ) => match r8.dedicated_info_type {
                    DLInformationTransfer_r8_IEsDedicatedInfoType::DedicatedInfoNAS(n) => {
                        DlDcch::InformationTransfer { nas: n.0 }
                    }
                    _ => DlDcch::Unsupported,
                },
                _ => DlDcch::Unsupported,
            }
        }
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcConnectionRelease(_)) => {
            DlDcch::ConnectionRelease
        }
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::SecurityModeCommand(c)) => {
            match c.critical_extensions {
                SecurityModeCommandCriticalExtensions::C1(
                    SecurityModeCommandCriticalExtensions_c1::SecurityModeCommand_r8(r8),
                ) => {
                    let a = r8.security_config_smc.security_algorithm_config;
                    DlDcch::SecurityModeCommand {
                        transaction_id: c.rrc_transaction_identifier.0,
                        eea: a.ciphering_algorithm.0,
                        eia: a.integrity_prot_algorithm.0,
                    }
                }
                _ => DlDcch::Unsupported,
            }
        }
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcConnectionReconfiguration(r)) => match r
            .critical_extensions
        {
            RRCConnectionReconfigurationCriticalExtensions::C1(
                RRCConnectionReconfigurationCriticalExtensions_c1::RrcConnectionReconfiguration_r8(
                    r8,
                ),
            ) => DlDcch::ConnectionReconfiguration(Reconfiguration {
                transaction_id: r.rrc_transaction_identifier.0,
                srb2: r8
                    .radio_resource_config_dedicated
                    .as_ref()
                    .and_then(|d| d.srb_to_add_mod_list.as_ref())
                    .is_some_and(|l| l.0.iter().any(|s| s.srb_identity.0 == 2)),
                drbs: r8
                    .radio_resource_config_dedicated
                    .and_then(|d| d.drb_to_add_mod_list)
                    .map(|l| {
                        l.0.into_iter()
                            .map(|d| DrbConfig {
                                rlc: decoded_drb_rlc(&d),
                                eps_bearer_identity: d.eps_bearer_identity.map_or(0, |e| e.0),
                                drb_identity: d.drb_identity.0,
                                logical_channel_identity: d
                                    .logical_channel_identity
                                    .map_or(0, |l| l.0),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                nas: r8
                    .dedicated_info_nas_list
                    .map(|l| l.0.into_iter().map(|n| n.0).collect())
                    .unwrap_or_default(),
                mobility_control: r8.mobility_control_info.is_some(),
            }),
            _ => DlDcch::Unsupported,
        },
        _ => DlDcch::Unsupported,
    })
}

// ---------------------------------------------------------------------------
// Security activation and RRC connection reconfiguration (§5.3.4, §5.3.5)
// ---------------------------------------------------------------------------

fn check_tid(transaction_id: u8) -> Result<(), RrcMsgError> {
    if transaction_id > 3 {
        return Err(RrcMsgError::Invalid("rrc-TransactionIdentifier > 3".into()));
    }
    Ok(())
}

/// Encode a `DL-DCCH-Message` carrying SecurityModeCommand (§5.3.4.2) with
/// the selected EEA (`cipheringAlgorithm`) and EIA (`integrityProtAlgorithm`)
/// identities, 0..3.
pub fn encode_security_mode_command(
    transaction_id: u8,
    eea: u8,
    eia: u8,
) -> Result<Vec<u8>, RrcMsgError> {
    check_tid(transaction_id)?;
    if eea > 3 || eia > 3 {
        return Err(RrcMsgError::Invalid(format!("EEA{eea} / EIA{eia}")));
    }
    let smc = SecurityModeCommand {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: SecurityModeCommandCriticalExtensions::C1(
            SecurityModeCommandCriticalExtensions_c1::SecurityModeCommand_r8(
                SecurityModeCommand_r8_IEs {
                    security_config_smc: SecurityConfigSMC {
                        security_algorithm_config: SecurityAlgorithmConfig {
                            ciphering_algorithm: CipheringAlgorithm_r12(eea),
                            integrity_prot_algorithm: SecurityAlgorithmConfigIntegrityProtAlgorithm(
                                eia,
                            ),
                        },
                    },
                    non_critical_extension: None,
                },
            ),
        ),
    };
    enc(&DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::SecurityModeCommand(smc)),
    })
}

/// Encode a `UL-DCCH-Message` carrying SecurityModeComplete (§5.3.4.3).
pub fn encode_security_mode_complete(transaction_id: u8) -> Result<Vec<u8>, RrcMsgError> {
    check_tid(transaction_id)?;
    let c = SecurityModeComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: SecurityModeCompleteCriticalExtensions::SecurityModeComplete_r8(
            SecurityModeComplete_r8_IEs {
                non_critical_extension: None,
            },
        ),
    };
    enc(&UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::SecurityModeComplete(c)),
    })
}

/// Encode a `UL-DCCH-Message` carrying SecurityModeFailure (§5.3.4.3).
pub fn encode_security_mode_failure(transaction_id: u8) -> Result<Vec<u8>, RrcMsgError> {
    check_tid(transaction_id)?;
    let f = SecurityModeFailure {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: SecurityModeFailureCriticalExtensions::SecurityModeFailure_r8(
            SecurityModeFailure_r8_IEs {
                non_critical_extension: None,
            },
        ),
    };
    enc(&UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::SecurityModeFailure(f)),
    })
}

/// One DRB of an RRCConnectionReconfiguration (§6.3.2 DRB-ToAddMod).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DrbConfig {
    /// `eps-BearerIdentity` (0..15): the E-RAB ID of the EPS bearer it carries.
    pub eps_bearer_identity: u8,
    /// `drb-Identity` (1..32).
    pub drb_identity: u8,
    /// `logicalChannelIdentity` (3..10).
    pub logical_channel_identity: u8,
    /// The DRB's RLC mode and, with it, its PDCP SN length (§6.3.2
    /// `RLC-Config`, `PDCP-Config`).
    pub rlc: DrbRlc,
}

/// The RLC mode of a DRB and the PDCP SN length that goes with it.
///
/// TS 36.323 §6.2.3/§6.2.4: a DRB on RLC AM uses the 12-bit PDCP SN; on RLC
/// UM, `pdcp-SN-Size` chooses 7 or 12 bits. This eNB configures
/// [`DrbRlc::UmSn10`] (see [`drb_to_add_mod`]); the decoder reports whatever
/// the network sent so a UE can refuse a configuration it does not support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrbRlc {
    /// RLC UM bi-directional, 10-bit RLC SN, 12-bit PDCP SN.
    UmSn10,
    /// RLC UM bi-directional with any other SN pairing (5-bit RLC SN or a
    /// 7-bit PDCP SN) -- decoded, not configured by this eNB.
    UmOther,
    /// RLC AM (12-bit PDCP SN) -- decoded, not configured by this eNB.
    Am,
    /// Uni-directional UM, or no `rlc-Config` at all.
    Other,
}

/// A decoded RRCConnectionReconfiguration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reconfiguration {
    /// `rrc-TransactionIdentifier`.
    pub transaction_id: u8,
    /// `srb-ToAddModList` adds SRB2.
    pub srb2: bool,
    /// DRBs to add (`drb-ToAddModList`).
    pub drbs: Vec<DrbConfig>,
    /// `dedicatedInfoNASList`, in order.
    pub nas: Vec<Vec<u8>>,
    /// Whether `mobilityControlInfo` is present (a handover).
    pub mobility_control: bool,
}

/// The DRB-ToAddMod for one EPS bearer: RLC UM bi-directional with a 10-bit
/// SN, PDCP with a 12-bit SN and no header compression -- `nextgsim`
/// advertises no ROHC profile.
///
/// **Why UM, for a QCI 9 default bearer.** TS 36.331 leaves the RLC mode of a
/// DRB to E-UTRAN (DRBs have no default configuration, §9.2.1) and TS 23.203
/// specifies a QCI by its packet error loss rate, not by an RLC mode. RLC AM
/// exists to recover HARQ residual losses over the air; RLS replaces the
/// physical layer and MAC and carries no residual radio loss model, so ARQ
/// would add status traffic and nothing else. The NR side of `nextgsim` runs
/// its DRBs on RLC UM for the same reason. A 10-bit RLC SN (UM window 512,
/// TS 36.322 §7.2) and a 12-bit PDCP SN (TS 36.323 §6.2.3, the length a DRB on
/// AM would also use) keep both sequence spaces far larger than anything in
/// flight between two RLS endpoints.
///
/// The remaining values are this eNB's choices: `t-Reordering` 35 ms,
/// PDCP `discardTimer` infinity, logical channel priority 13 in group 3,
/// PBR 8 kbit/s with a 100 ms bucket.
fn drb_to_add_mod(d: &DrbConfig) -> DRB_ToAddMod {
    DRB_ToAddMod {
        extensions: Default::default(),
        eps_bearer_identity: Some(DRB_ToAddModEps_BearerIdentity(d.eps_bearer_identity)),
        drb_identity: DRB_Identity(d.drb_identity),
        pdcp_config: Some(PDCP_Config {
            extensions: Default::default(),
            discard_timer: Some(PDCP_ConfigDiscardTimer(PDCP_ConfigDiscardTimer::INFINITY)),
            rlc_am: None,
            rlc_um: Some(PDCP_ConfigRlc_UM {
                pdcp_sn_size: PDCP_ConfigRlc_UMPdcp_SN_Size(
                    PDCP_ConfigRlc_UMPdcp_SN_Size::LEN12BITS,
                ),
            }),
            header_compression: PDCP_ConfigHeaderCompression::NotUsed(
                PDCP_ConfigHeaderCompression_notUsed,
            ),
        }),
        rlc_config: Some(RLC_Config::Um_Bi_Directional(
            RLC_Config_um_Bi_Directional {
                ul_um_rlc: UL_UM_RLC {
                    sn_field_length: SN_FieldLength(SN_FieldLength::SIZE10),
                },
                dl_um_rlc: DL_UM_RLC {
                    sn_field_length: SN_FieldLength(SN_FieldLength::SIZE10),
                    t_reordering: T_Reordering(T_Reordering::MS35),
                },
            },
        )),
        logical_channel_identity: Some(DRB_ToAddModLogicalChannelIdentity(
            d.logical_channel_identity,
        )),
        logical_channel_config: Some(LogicalChannelConfig {
            extensions: Default::default(),
            ul_specific_parameters: Some(LogicalChannelConfigUl_SpecificParameters {
                priority: LogicalChannelConfigUl_SpecificParametersPriority(13),
                prioritised_bit_rate: LogicalChannelConfigUl_SpecificParametersPrioritisedBitRate(
                    LogicalChannelConfigUl_SpecificParametersPrioritisedBitRate::K_BPS8,
                ),
                bucket_size_duration: LogicalChannelConfigUl_SpecificParametersBucketSizeDuration(
                    LogicalChannelConfigUl_SpecificParametersBucketSizeDuration::MS100,
                ),
                logical_channel_group: Some(
                    LogicalChannelConfigUl_SpecificParametersLogicalChannelGroup(3),
                ),
            }),
        }),
    }
}

/// What a received DRB-ToAddMod configures, as a [`DrbRlc`].
fn decoded_drb_rlc(d: &DRB_ToAddMod) -> DrbRlc {
    let pdcp12 = d
        .pdcp_config
        .as_ref()
        .and_then(|p| p.rlc_um.as_ref())
        .is_some_and(|u| u.pdcp_sn_size.0 == PDCP_ConfigRlc_UMPdcp_SN_Size::LEN12BITS);
    match &d.rlc_config {
        Some(RLC_Config::Am(_)) => DrbRlc::Am,
        Some(RLC_Config::Um_Bi_Directional(u))
            if pdcp12
                && u.ul_um_rlc.sn_field_length.0 == SN_FieldLength::SIZE10
                && u.dl_um_rlc.sn_field_length.0 == SN_FieldLength::SIZE10 =>
        {
            DrbRlc::UmSn10
        }
        Some(RLC_Config::Um_Bi_Directional(_)) => DrbRlc::UmOther,
        _ => DrbRlc::Other,
    }
}

/// SRB2 with the default RLC and logical channel configuration (§9.2.1.2):
/// "E-UTRAN ... establishes SRB2 ... after having activated security" (§4.2.2).
fn srb2_to_add_mod() -> SRB_ToAddMod {
    SRB_ToAddMod {
        extensions: Default::default(),
        srb_identity: SRB_ToAddModSrb_Identity(2),
        rlc_config: Some(SRB_ToAddModRlc_Config::DefaultValue(
            SRB_ToAddModRlc_Config_defaultValue,
        )),
        logical_channel_config: Some(SRB_ToAddModLogicalChannelConfig::DefaultValue(
            SRB_ToAddModLogicalChannelConfig_defaultValue,
        )),
    }
}

/// Encode a `DL-DCCH-Message` carrying RRCConnectionReconfiguration
/// (§5.3.5.2) that adds SRB2 when `add_srb2`, adds `drbs`, and carries `nas`
/// in `dedicatedInfoNASList`
/// (1..maxDRB = 11 entries, §6.2.2).
pub fn encode_rrc_connection_reconfiguration(
    transaction_id: u8,
    add_srb2: bool,
    drbs: &[DrbConfig],
    nas: &[Vec<u8>],
) -> Result<Vec<u8>, RrcMsgError> {
    check_tid(transaction_id)?;
    if drbs.len() > 11 || nas.len() > 11 {
        return Err(RrcMsgError::Invalid("more than maxDRB (11) entries".into()));
    }
    for d in drbs {
        if d.eps_bearer_identity > 15
            || !(1..=32).contains(&d.drb_identity)
            || !(3..=10).contains(&d.logical_channel_identity)
        {
            return Err(RrcMsgError::Invalid(format!("{d:?}")));
        }
    }
    let r8 = RRCConnectionReconfiguration_r8_IEs {
        meas_config: None,
        mobility_control_info: None,
        dedicated_info_nas_list: (!nas.is_empty()).then(|| {
            RRCConnectionReconfiguration_r8_IEsDedicatedInfoNASList(
                nas.iter().map(|n| DedicatedInfoNAS(n.clone())).collect(),
            )
        }),
        radio_resource_config_dedicated: (add_srb2 || !drbs.is_empty()).then(|| {
            RadioResourceConfigDedicated {
                extensions: Default::default(),
                srb_to_add_mod_list: add_srb2.then(|| SRB_ToAddModList(vec![srb2_to_add_mod()])),
                drb_to_add_mod_list: (!drbs.is_empty())
                    .then(|| DRB_ToAddModList(drbs.iter().map(drb_to_add_mod).collect())),
                drb_to_release_list: None,
                mac_main_config: None,
                sps_config: None,
                physical_config_dedicated: None,
            }
        }),
        security_config_ho: None,
        non_critical_extension: None,
    };
    let r = RRCConnectionReconfiguration {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions: RRCConnectionReconfigurationCriticalExtensions::C1(
            RRCConnectionReconfigurationCriticalExtensions_c1::RrcConnectionReconfiguration_r8(r8),
        ),
    };
    enc(&DL_DCCH_Message {
        message: DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcConnectionReconfiguration(r)),
    })
}

/// Encode a `UL-DCCH-Message` carrying RRCConnectionReconfigurationComplete
/// (§5.3.5.3).
pub fn encode_rrc_connection_reconfiguration_complete(
    transaction_id: u8,
) -> Result<Vec<u8>, RrcMsgError> {
    check_tid(transaction_id)?;
    let c = RRCConnectionReconfigurationComplete {
        rrc_transaction_identifier: RRC_TransactionIdentifier(transaction_id),
        critical_extensions:
            RRCConnectionReconfigurationCompleteCriticalExtensions::RrcConnectionReconfigurationComplete_r8(
                RRCConnectionReconfigurationComplete_r8_IEs {
                    non_critical_extension: None,
                },
            ),
    };
    enc(&UL_DCCH_Message {
        message: UL_DCCH_MessageType::C1(
            UL_DCCH_MessageType_c1::RrcConnectionReconfigurationComplete(c),
        ),
    })
}

#[cfg(test)]
mod tests;
