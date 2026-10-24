//! The NG-RAN node's half of Multi-RTT positioning (TS 38.305 §8.10.3.2) over
//! NRPPa (TS 38.455), nextgsim #347:
//!
//! | NRPPa procedure code | Procedure | Clause | Transport |
//! |---|---|---|---|
//! | 9  `id-positioningInformationExchange` | the SRS Configuration in the answer | §8.2.6 | UE-associated |
//! | 17 `id-positioningActivation`          | Positioning Activation              | §8.2.9 | UE-associated |
//! | 18 `id-positioningDeactivation`        | Positioning Deactivation            | §8.2.10 | UE-associated |
//! | 11 `id-Measurement`                    | Measurement (gNB Rx-Tx)             | §8.5.1 | non-UE-associated |
//!
//! This module only parses and builds. Which UE an SRS belongs to, whether it
//! is active and what the TRP measures are the gNB's state, so the caller
//! decides and hands the outcome to a builder.

use super::{nrppa, CRITICALITY_IGNORE, CRITICALITY_REJECT};

/// `id-Measurement` — Measurement (§8.5.1).
pub const PROC_MEASUREMENT: u8 = 11;
/// `id-positioningActivation` — Positioning Activation (§8.2.9).
pub const PROC_POSITIONING_ACTIVATION: u8 = 17;
/// `id-positioningDeactivation` — Positioning Deactivation (§8.2.10).
pub const PROC_POSITIONING_DEACTIVATION: u8 = 18;

/// `id-Cause`.
pub const IE_CAUSE: u16 = 0;
/// `id-SRSConfiguration`.
pub const IE_SRS_CONFIGURATION: u16 = 26;
/// `id-LMF-Measurement-ID`.
pub const IE_LMF_MEASUREMENT_ID: u16 = 39;
/// `id-RAN-Measurement-ID`.
pub const IE_RAN_MEASUREMENT_ID: u16 = 40;
/// `id-TRP-MeasurementResponseList`.
pub const IE_TRP_MEASUREMENT_RESPONSE_LIST: u16 = 42;
/// `id-SystemFrameNumber`.
pub const IE_SYSTEM_FRAME_NUMBER: u16 = 49;
/// `id-SlotNumber`.
pub const IE_SLOT_NUMBER: u16 = 50;

/// `TRPMeasurementType` value `gNB-RxTxTimeDiff` (the first enumeration).
pub const TRP_MEASUREMENT_GNB_RX_TX_TIME_DIFF: u8 = 0;

/// `ReportCharacteristics` value `onDemand`.
const REPORT_ON_DEMAND: u8 = 0;

/// `CauseRadioNetwork` values (TS 38.455 §9.2.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NrppaRadioCause {
    /// `unspecified`.
    Unspecified = 0,
    /// `requested-item-not-supported`.
    RequestedItemNotSupported = 1,
    /// `requested-item-temporarily-not-available`.
    RequestedItemTemporarilyNotAvailable = 2,
}

impl NrppaRadioCause {
    fn to_cause(self) -> nrppa::Cause {
        nrppa::Cause::RadioNetwork(nrppa::CauseRadioNetwork(self as u8))
    }
}

/// The SRS resource type an LMF asks for in `RequestedSRSTransmissionCharacteristics`,
/// and the type of the positioning SRS resource set this node configures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrsResourceType {
    /// `periodic`: transmitted from configuration, no activation.
    Periodic,
    /// `semi-persistent`: transmitted between activation and deactivation.
    SemiPersistent,
    /// `aperiodic`: transmitted when triggered.
    Aperiodic,
}

/// What a POSITIONING ACTIVATION REQUEST asks for (`SRSType`, §9.2.29).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrsActivation {
    /// `semipersistentSRS`: activate the named SRS resource set.
    SemiPersistent {
        /// `SRSResourceSetID ::= INTEGER (0..15, ...)`.
        resource_set_id: u8,
    },
    /// `aperiodicSRS`: trigger the aperiodic SRS.
    Aperiodic,
}

/// What a POSITIONING DEACTIVATION stops (`AbortTransmission`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SrsDeactivation {
    /// `deactivateSRSResourceSetID`.
    ResourceSet(u8),
    /// `releaseALL`.
    ReleaseAll,
}

/// The parts of a MEASUREMENT REQUEST (§9.1.1.x) this node acts on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrpMeasurementRequest {
    /// `LMF-Measurement-ID`, echoed in the answer.
    pub lmf_measurement_id: u32,
    /// The `tRP-ID`s of `TRP-MeasurementRequestList`.
    pub trp_ids: Vec<u16>,
    /// `ReportCharacteristics` is `onDemand`.
    pub on_demand: bool,
    /// The `TRPMeasurementType`s of `TRPMeasurementQuantities`.
    pub quantities: Vec<u8>,
    /// The positioning SRS resource set ids the request's `SRSConfiguration`
    /// names, with the carrier's `pCI-NR` when it has one. Empty when the
    /// request carries no SRS Configuration.
    pub srs_resource_sets: Vec<(u8, Option<u16>)>,
}

/// The positioning SRS this node configures for one UE, which goes to the LMF
/// as the Positioning Information Response's `SRSConfiguration`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SrsAllocation {
    /// `possrsResourceSetID ::= INTEGER (0..15)`, unique among this node's UEs.
    pub resource_set_id: u8,
    /// The resource set's type.
    pub resource_type: SrsResourceType,
    /// The SRS `sequenceId ::= INTEGER (0..65535)`.
    pub sequence_id: u16,
    /// The serving cell's `pCI-NR`.
    pub pci: u16,
    /// The carrier's `pointA`, an NR-ARFCN.
    pub point_a: u32,
}

/// The carrier the SRS is configured on: 51 PRBs at 30 kHz, the whole UL BWP.
const SRS_SCS_30KHZ: u8 = 1;
const SRS_CARRIER_PRBS: u16 = 51;
/// `locationAndBandwidth` is a RIV over 275 PRBs (TS 38.214 §5.1.2.2.2 with
/// N = 275): starting at PRB 0 with length L it is `275 * (L - 1)`.
const SRS_BWP_RIV: u16 = 275 * (SRS_CARRIER_PRBS - 1);
/// `SRSPeriodicity` value `slot20`, 10 ms at 30 kHz.
const SRS_PERIODICITY_SLOT20: u8 = 7;

/// The `pointA` of a [`SRS_CARRIER_PRBS`]-PRB, 30 kHz carrier centred on the
/// NR-ARFCN `centre`. The NR-ARFCN step is 5, 15 or 60 kHz by frequency range
/// (TS 38.104 §5.4.2.1: below 3 GHz, 3 to 24.25 GHz, above).
#[must_use]
pub fn srs_point_a(centre: u32) -> u32 {
    let step_khz = match centre {
        0..=599_999 => 5,
        600_000..=2_016_666 => 15,
        _ => 60,
    };
    let half_bandwidth_khz = u32::from(SRS_CARRIER_PRBS) * 12 * 30 / 2;
    centre.saturating_sub(half_bandwidth_khz / step_khz)
}

/// The `SRSConfiguration` (§9.2.28) for `srs`: one carrier, one UL BWP, one
/// positioning SRS resource in one positioning SRS resource set.
#[must_use]
pub fn build_srs_configuration(srs: &SrsAllocation) -> nrppa::SRSConfiguration {
    let (resource_type_pos, set_type) = match srs.resource_type {
        SrsResourceType::Periodic => (
            nrppa::ResourceTypePos::Periodic(nrppa::ResourceTypePeriodicPos {
                srs_periodicity: nrppa::SRSPeriodicity(SRS_PERIODICITY_SLOT20),
                offset: nrppa::ResourceTypePeriodicPosOffset(0),
                ie_extensions: None,
            }),
            nrppa::PosResourceSetType::Periodic(nrppa::PosResourceSetTypePeriodic {
                posperiodic_set: nrppa::PosResourceSetTypePeriodicPosperiodicSet(0),
                ie_extensions: None,
            }),
        ),
        SrsResourceType::SemiPersistent => (
            nrppa::ResourceTypePos::Semi_persistent(nrppa::ResourceTypeSemi_persistentPos {
                srs_periodicity: nrppa::SRSPeriodicity(SRS_PERIODICITY_SLOT20),
                offset: nrppa::ResourceTypeSemi_persistentPosOffset(0),
                ie_extensions: None,
            }),
            nrppa::PosResourceSetType::Semi_persistent(nrppa::PosResourceSetTypeSemi_persistent {
                possemi_persistent_set:
                    nrppa::PosResourceSetTypeSemi_persistentPossemi_persistentSet(0),
                ie_extensions: None,
            }),
        ),
        SrsResourceType::Aperiodic => (
            nrppa::ResourceTypePos::Aperiodic(nrppa::ResourceTypeAperiodicPos {
                slot_offset: nrppa::ResourceTypeAperiodicPosSlotOffset(0),
                ie_extensions: None,
            }),
            nrppa::PosResourceSetType::Aperiodic(nrppa::PosResourceSetTypeAperiodic {
                srs_resource_trigger: nrppa::PosResourceSetTypeAperiodicSRSResourceTrigger(1),
                ie_extensions: None,
            }),
        ),
    };

    let resource = nrppa::PosSRSResource_Item {
        srs_pos_resource_id: nrppa::SRSPosResourceID(0),
        transmission_comb_pos: nrppa::TransmissionCombPos::N2(nrppa::TransmissionCombPos_n2 {
            comb_offset_n2: nrppa::TransmissionCombPos_n2CombOffset_n2(0),
            cyclic_shift_n2: nrppa::TransmissionCombPos_n2CyclicShift_n2(0),
        }),
        start_position: nrppa::PosSRSResource_ItemStartPosition(0),
        nrof_symbols: nrppa::PosSRSResource_ItemNrofSymbols(0),
        freq_domain_shift: nrppa::PosSRSResource_ItemFreqDomainShift(0),
        c_srs: nrppa::PosSRSResource_ItemC_SRS(0),
        group_or_sequence_hopping: nrppa::PosSRSResource_ItemGroupOrSequenceHopping(0),
        resource_type_pos,
        sequence_id: nrppa::PosSRSResource_ItemSequenceId(srs.sequence_id),
        spatial_relation_pos: None,
        ie_extensions: None,
    };
    let set = nrppa::PosSRSResourceSet_Item {
        possrs_resource_set_id: nrppa::PosSRSResourceSet_ItemPossrsResourceSetID(
            srs.resource_set_id,
        ),
        poss_rs_resource_id_per_set_list: nrppa::PosSRSResourceIDPerSet_List(vec![
            nrppa::SRSPosResourceID(0),
        ]),
        posresource_set_type: set_type,
        ie_extensions: None,
    };

    nrppa::SRSConfiguration {
        srs_carrier_list: nrppa::SRSCarrier_List(vec![nrppa::SRSCarrier_List_Item {
            point_a: nrppa::SRSCarrier_List_ItemPointA(srs.point_a),
            uplink_channel_bw_per_scs_list: nrppa::UplinkChannelBW_PerSCS_List(vec![
                nrppa::SCS_SpecificCarrier {
                    offset_to_carrier: nrppa::SCS_SpecificCarrierOffsetToCarrier(0),
                    subcarrier_spacing: nrppa::SCS_SpecificCarrierSubcarrierSpacing(SRS_SCS_30KHZ),
                    carrier_bandwidth: nrppa::SCS_SpecificCarrierCarrierBandwidth(SRS_CARRIER_PRBS),
                    ie_extensions: None,
                },
            ]),
            active_ulbwp: nrppa::ActiveULBWP {
                location_and_bandwidth: nrppa::ActiveULBWPLocationAndBandwidth(SRS_BWP_RIV),
                subcarrier_spacing: nrppa::ActiveULBWPSubcarrierSpacing(SRS_SCS_30KHZ),
                cyclic_prefix: nrppa::ActiveULBWPCyclicPrefix(0),
                tx_direct_current_location: nrppa::ActiveULBWPTxDirectCurrentLocation(0),
                shift7dot5k_hz: None,
                srs_config: nrppa::SRSConfig {
                    srs_resource_list: None,
                    pos_srs_resource_list: Some(nrppa::PosSRSResource_List(vec![resource])),
                    srs_resource_set_list: None,
                    pos_srs_resource_set_list: Some(nrppa::PosSRSResourceSet_List(vec![set])),
                    ie_extensions: None,
                },
                ie_extensions: None,
            },
            pci_nr: Some(nrppa::SRSCarrier_List_ItemPCI_NR(srs.pci)),
            ie_extensions: None,
        }]),
        ie_extensions: None,
    }
}

/// The SRS resource type `RequestedSRSTransmissionCharacteristics` asks for.
pub(super) fn requested_srs_resource_type(
    req: &nrppa::PositioningInformationRequest,
) -> Option<SrsResourceType> {
    use nrppa::PositioningInformationRequestProtocolIEs_EntryValue as V;
    req.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        V::Id_RequestedSRSTransmissionCharacteristics(chars) => Some(match chars.resource_type.0 {
            0 => SrsResourceType::Periodic,
            2 => SrsResourceType::Aperiodic,
            _ => SrsResourceType::SemiPersistent,
        }),
        _ => None,
    })
}

/// The `SRSType` of a POSITIONING ACTIVATION REQUEST; `None` when it is
/// missing or a choice extension this node does not know.
pub(super) fn parse_positioning_activation(
    req: &nrppa::PositioningActivationRequest,
) -> Option<SrsActivation> {
    use nrppa::PositioningActivationRequestProtocolIEs_EntryValue as V;
    req.protocol_i_es.0.iter().find_map(|ie| match &ie.value {
        V::Id_SRSType(nrppa::SRSType::SemipersistentSRS(sp)) => {
            Some(SrsActivation::SemiPersistent {
                resource_set_id: sp.srs_resource_set_id.0,
            })
        }
        V::Id_SRSType(nrppa::SRSType::AperiodicSRS(_)) => Some(SrsActivation::Aperiodic),
        _ => None,
    })
}

/// The `AbortTransmission` of a POSITIONING DEACTIVATION.
pub(super) fn parse_positioning_deactivation(
    msg: &nrppa::PositioningDeactivation,
) -> Option<SrsDeactivation> {
    use nrppa::PositioningDeactivationProtocolIEs_EntryValue as V;
    msg.protocol_i_es.0.iter().find_map(|ie| {
        let V::Id_AbortTransmission(abort) = &ie.value;
        match abort {
            nrppa::AbortTransmission::DeactivateSRSResourceSetID(id) => {
                Some(SrsDeactivation::ResourceSet(id.0))
            }
            nrppa::AbortTransmission::ReleaseALL(_) => Some(SrsDeactivation::ReleaseAll),
            nrppa::AbortTransmission::Choice_extension(_) => None,
        }
    })
}

/// The positioning SRS resource set ids an `SRSConfiguration` names, each with
/// its carrier's `pCI-NR`.
#[must_use]
pub fn srs_configuration_resource_sets(config: &nrppa::SRSConfiguration) -> Vec<(u8, Option<u16>)> {
    let mut sets = Vec::new();
    for carrier in &config.srs_carrier_list.0 {
        let pci = carrier.pci_nr.as_ref().map(|p| p.0);
        if let Some(list) = &carrier.active_ulbwp.srs_config.pos_srs_resource_set_list {
            sets.extend(list.0.iter().map(|set| (set.possrs_resource_set_id.0, pci)));
        }
    }
    sets
}

/// A MEASUREMENT REQUEST's [`TrpMeasurementRequest`]; `Err` names a missing
/// mandatory IE.
pub(super) fn parse_measurement_request(
    req: &nrppa::MeasurementRequest,
) -> Result<TrpMeasurementRequest, &'static str> {
    use nrppa::MeasurementRequestProtocolIEs_EntryValue as V;
    let mut lmf_measurement_id = None;
    let mut trp_ids = None;
    let mut on_demand = None;
    let mut quantities = None;
    let mut srs_resource_sets = Vec::new();
    for ie in &req.protocol_i_es.0 {
        match &ie.value {
            V::Id_LMF_Measurement_ID(id) => lmf_measurement_id = Some(id.0),
            V::Id_TRP_MeasurementRequestList(list) => {
                trp_ids = Some(list.0.iter().map(|item| item.trp_id.0).collect());
            }
            V::Id_ReportCharacteristics(rc) => on_demand = Some(rc.0 == REPORT_ON_DEMAND),
            V::Id_TRPMeasurementQuantities(q) => {
                quantities = Some(q.0.iter().map(|item| item.trp_measurement_type.0).collect());
            }
            V::Id_SRSConfiguration(config) => {
                srs_resource_sets = srs_configuration_resource_sets(config);
            }
            _ => {}
        }
    }
    Ok(TrpMeasurementRequest {
        lmf_measurement_id: lmf_measurement_id.ok_or("LMF-Measurement-ID")?,
        trp_ids: trp_ids.ok_or("TRP-MeasurementRequestList")?,
        on_demand: on_demand.ok_or("ReportCharacteristics")?,
        quantities: quantities.ok_or("TRPMeasurementQuantities")?,
        srs_resource_sets,
    })
}

/// A cell time as a TimeStamp gives it: SFN and the 30 kHz slot in its frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellTime {
    /// `SystemFrameNumber ::= INTEGER (0..1023)`.
    pub sfn: u16,
    /// The 30 kHz slot, `0..=19`.
    pub slot: u8,
}

impl CellTime {
    /// The SFN and 30 kHz slot `micros` microseconds after the instant SFN 0
    /// began (the Positioning Information Response's SFN Initialisation Time).
    #[must_use]
    pub fn from_micros_since_sfn0(micros: u128) -> Self {
        // A frame is 10 ms; a 30 kHz slot is 0.5 ms.
        let sfn = ((micros / 10_000) % 1024) as u16;
        let slot = ((micros % 10_000) / 500) as u8;
        Self { sfn, slot }
    }
}

/// One TRP's gNB Rx-Tx time difference measurement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrpRxTxMeasurement {
    /// `TRP-ID`.
    pub trp_id: u16,
    /// The gNB Rx-Tx time difference, in seconds.
    pub rx_tx_seconds: f64,
    /// When it was measured.
    pub time: CellTime,
}

/// `T_c`, the NR basic time unit (TS 38.211 §4.1): 1 / (480 000 * 4096) s.
pub const T_C_SECONDS: f64 = 1.0 / (480_000.0 * 4096.0);

/// The `k0` report of a gNB Rx-Tx time difference: `985024 + t / T_c`,
/// rounded, kept inside `1..=1970048` (0 and 1970049 are the open-ended ends
/// of the range). This is the mapping nextgcore lmfd decodes.
#[must_use]
pub fn gnb_rx_tx_k0_report(seconds: f64) -> u32 {
    let v = (seconds / T_C_SECONDS).round() + 985_024.0;
    v.clamp(1.0, 1_970_048.0) as u32
}

fn successful(
    procedure: u8,
    transaction_id: u16,
    value: nrppa::SuccessfulOutcomeValue,
) -> nrppa::NRPPA_PDU {
    nrppa::NRPPA_PDU::SuccessfulOutcome(nrppa::SuccessfulOutcome {
        procedure_code: nrppa::ProcedureCode(procedure),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value,
    })
}

fn unsuccessful(
    procedure: u8,
    transaction_id: u16,
    value: nrppa::UnsuccessfulOutcomeValue,
) -> nrppa::NRPPA_PDU {
    nrppa::NRPPA_PDU::UnsuccessfulOutcome(nrppa::UnsuccessfulOutcome {
        procedure_code: nrppa::ProcedureCode(procedure),
        criticality: nrppa::Criticality(CRITICALITY_REJECT),
        nrppatransaction_id: nrppa::NRPPATransactionID(transaction_id),
        value,
    })
}

/// POSITIONING INFORMATION FAILURE (§8.2.6.3) with `cause`.
#[must_use]
pub fn build_positioning_information_failure(
    transaction_id: u16,
    cause: NrppaRadioCause,
) -> nrppa::NRPPA_PDU {
    use nrppa::PositioningInformationFailureProtocolIEs_EntryValue as V;
    unsuccessful(
        super::PROC_POSITIONING_INFORMATION_EXCHANGE,
        transaction_id,
        nrppa::UnsuccessfulOutcomeValue::Id_positioningInformationExchange(
            nrppa::PositioningInformationFailure {
                protocol_i_es: nrppa::PositioningInformationFailureProtocolIEs(vec![
                    nrppa::PositioningInformationFailureProtocolIEs_Entry {
                        id: nrppa::ProtocolIE_ID(IE_CAUSE),
                        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                        value: V::Id_Cause(cause.to_cause()),
                    },
                ]),
            },
        ),
    )
}

/// POSITIONING ACTIVATION RESPONSE (§8.2.9.2), with the SFN and slot of the
/// activation.
#[must_use]
pub fn build_positioning_activation_response(
    transaction_id: u16,
    at: CellTime,
) -> nrppa::NRPPA_PDU {
    use nrppa::PositioningActivationResponseProtocolIEs_EntryValue as V;
    successful(
        PROC_POSITIONING_ACTIVATION,
        transaction_id,
        nrppa::SuccessfulOutcomeValue::Id_positioningActivation(
            nrppa::PositioningActivationResponse {
                protocol_i_es: nrppa::PositioningActivationResponseProtocolIEs(vec![
                    nrppa::PositioningActivationResponseProtocolIEs_Entry {
                        id: nrppa::ProtocolIE_ID(IE_SYSTEM_FRAME_NUMBER),
                        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                        value: V::Id_SystemFrameNumber(nrppa::SystemFrameNumber(at.sfn)),
                    },
                    nrppa::PositioningActivationResponseProtocolIEs_Entry {
                        id: nrppa::ProtocolIE_ID(IE_SLOT_NUMBER),
                        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                        value: V::Id_SlotNumber(nrppa::SlotNumber(at.slot)),
                    },
                ]),
            },
        ),
    )
}

/// POSITIONING ACTIVATION FAILURE (§8.2.9.3) with `cause`.
#[must_use]
pub fn build_positioning_activation_failure(
    transaction_id: u16,
    cause: NrppaRadioCause,
) -> nrppa::NRPPA_PDU {
    use nrppa::PositioningActivationFailureProtocolIEs_EntryValue as V;
    unsuccessful(
        PROC_POSITIONING_ACTIVATION,
        transaction_id,
        nrppa::UnsuccessfulOutcomeValue::Id_positioningActivation(
            nrppa::PositioningActivationFailure {
                protocol_i_es: nrppa::PositioningActivationFailureProtocolIEs(vec![
                    nrppa::PositioningActivationFailureProtocolIEs_Entry {
                        id: nrppa::ProtocolIE_ID(IE_CAUSE),
                        criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                        value: V::Id_Cause(cause.to_cause()),
                    },
                ]),
            },
        ),
    )
}

/// MEASUREMENT RESPONSE (§8.5.1.2) carrying one gNB Rx-Tx time difference per
/// TRP in `measurements`, which must not be empty.
#[must_use]
pub fn build_measurement_response(
    transaction_id: u16,
    lmf_measurement_id: u32,
    ran_measurement_id: u32,
    measurements: &[TrpRxTxMeasurement],
) -> nrppa::NRPPA_PDU {
    use nrppa::MeasurementResponseProtocolIEs_EntryValue as V;
    let items = measurements
        .iter()
        .map(|m| nrppa::TRP_MeasurementResponseItem {
            trp_id: nrppa::TRP_ID(m.trp_id),
            measurement_result: nrppa::TrpMeasurementResult(vec![
                nrppa::TrpMeasurementResultItem {
                    measured_results_value: nrppa::TrpMeasuredResultsValue::GNB_RxTxTimeDiff(
                        nrppa::GNB_RxTxTimeDiff {
                            rx_tx_time_diff: nrppa::GNBRxTxTimeDiffMeas::K0(
                                nrppa::GNBRxTxTimeDiffMeas_k0(gnb_rx_tx_k0_report(m.rx_tx_seconds)),
                            ),
                            additional_path_list: None,
                            ie_extensions: None,
                        },
                    ),
                    time_stamp: nrppa::TimeStamp {
                        system_frame_number: nrppa::SystemFrameNumber(m.time.sfn),
                        slot_index: nrppa::TimeStampSlotIndex::SCS_30(
                            nrppa::TimeStampSlotIndex_sCS_30(m.time.slot),
                        ),
                        measurement_time: None,
                        ie_extension: None,
                    },
                    measurement_quality: None,
                    measurement_beam_info: None,
                    ie_extensions: None,
                },
            ]),
            ie_extensions: None,
        })
        .collect();
    successful(
        PROC_MEASUREMENT,
        transaction_id,
        nrppa::SuccessfulOutcomeValue::Id_Measurement(nrppa::MeasurementResponse {
            protocol_i_es: nrppa::MeasurementResponseProtocolIEs(vec![
                nrppa::MeasurementResponseProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(IE_LMF_MEASUREMENT_ID),
                    criticality: nrppa::Criticality(CRITICALITY_REJECT),
                    value: V::Id_LMF_Measurement_ID(nrppa::Measurement_ID(lmf_measurement_id)),
                },
                nrppa::MeasurementResponseProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(IE_RAN_MEASUREMENT_ID),
                    criticality: nrppa::Criticality(CRITICALITY_REJECT),
                    value: V::Id_RAN_Measurement_ID(nrppa::Measurement_ID(ran_measurement_id)),
                },
                nrppa::MeasurementResponseProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(IE_TRP_MEASUREMENT_RESPONSE_LIST),
                    criticality: nrppa::Criticality(CRITICALITY_REJECT),
                    value: V::Id_TRP_MeasurementResponseList(nrppa::TRP_MeasurementResponseList(
                        items,
                    )),
                },
            ]),
        }),
    )
}

/// MEASUREMENT FAILURE (§8.5.1.3) with `cause`.
#[must_use]
pub fn build_measurement_failure(
    transaction_id: u16,
    lmf_measurement_id: u32,
    cause: NrppaRadioCause,
) -> nrppa::NRPPA_PDU {
    use nrppa::MeasurementFailureProtocolIEs_EntryValue as V;
    unsuccessful(
        PROC_MEASUREMENT,
        transaction_id,
        nrppa::UnsuccessfulOutcomeValue::Id_Measurement(nrppa::MeasurementFailure {
            protocol_i_es: nrppa::MeasurementFailureProtocolIEs(vec![
                nrppa::MeasurementFailureProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(IE_LMF_MEASUREMENT_ID),
                    criticality: nrppa::Criticality(CRITICALITY_REJECT),
                    value: V::Id_LMF_Measurement_ID(nrppa::Measurement_ID(lmf_measurement_id)),
                },
                nrppa::MeasurementFailureProtocolIEs_Entry {
                    id: nrppa::ProtocolIE_ID(IE_CAUSE),
                    criticality: nrppa::Criticality(CRITICALITY_IGNORE),
                    value: V::Id_Cause(cause.to_cause()),
                },
            ]),
        }),
    )
}
