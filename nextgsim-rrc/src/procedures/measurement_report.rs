//! Enhanced Measurement Report
//!
//! Implements enhanced Measurement Report handling as defined in 3GPP TS 38.331
//! Section 5.5.5, with extensions for AI/ML-based measurements and 6G metrics.
//!
//! This module implements:
//! - `MeasurementReportData` - Parsed measurement report with NR measurement results
//! - `MeasResult2NR` - Measurement results for NR cells
//! - `MeasResultServFreqListNR` - Serving frequency measurement results
//! - 6G: Enhanced measurement quantities (beam quality, AI-predicted values)

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during Measurement Report procedures
#[derive(Debug, Error)]
pub enum MeasurementReportError {
    /// Codec error during encoding/decoding
    #[error("Codec error: {0}")]
    CodecError(#[from] RrcCodecError),

    /// Invalid message type received
    #[error("Invalid message type: expected {expected}, got {actual}")]
    InvalidMessageType {
        /// Expected message type
        expected: String,
        /// Actual message type received
        actual: String,
    },

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Measurement quantity type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasQuantityType {
    /// Reference Signal Received Power (RSRP) in dBm
    Rsrp,
    /// Reference Signal Received Quality (RSRQ) in dB
    Rsrq,
    /// Signal to Interference plus Noise Ratio (SINR) in dB
    Sinr,
}

/// Measurement result value with quantity type
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasQuantityResult {
    /// Measurement quantity type
    pub quantity_type: MeasQuantityType,
    /// Measured value (RSRP: -156 to -31 dBm, RSRQ: -43 to 20 dB, SINR: -23 to 40 dB)
    pub value: f64,
}

/// Measurement result for a single NR cell
#[derive(Debug, Clone)]
pub struct MeasResultNr {
    /// Physical Cell ID (0-1007)
    pub phys_cell_id: Option<u16>,
    /// Cell-level measurement results
    pub cell_results: MeasResultCellNr,
    /// RS index results (SSB or CSI-RS)
    pub rs_index_results: Option<MeasResultRsIndexNr>,
}

/// Cell-level measurement results for NR
#[derive(Debug, Clone)]
pub struct MeasResultCellNr {
    /// SSB-based results
    pub ssb_results: Option<MeasCellResults>,
    /// CSI-RS-based results
    pub csi_rs_results: Option<MeasCellResults>,
}

/// Measurement results for a single RS type (SSB or CSI-RS)
#[derive(Debug, Clone)]
pub struct MeasCellResults {
    /// RSRP value (0-127, maps to -156 to -31 dBm)
    pub rsrp: Option<u8>,
    /// RSRQ value (0-127, maps to -43 to 20 dB)
    pub rsrq: Option<u8>,
    /// SINR value (0-127, maps to -23 to 40 dB)
    pub sinr: Option<u8>,
}

/// RS index measurement results
#[derive(Debug, Clone)]
pub struct MeasResultRsIndexNr {
    /// SSB index results
    pub ssb_results: Vec<MeasResultPerSsbIndex>,
    /// CSI-RS index results
    pub csi_rs_results: Vec<MeasResultPerCsiRsIndex>,
}

/// Per-SSB index measurement result
#[derive(Debug, Clone)]
pub struct MeasResultPerSsbIndex {
    /// SSB index (0-63)
    pub ssb_index: u8,
    /// Measurement results for this SSB
    pub results: MeasCellResults,
}

/// Per-CSI-RS index measurement result
#[derive(Debug, Clone)]
pub struct MeasResultPerCsiRsIndex {
    /// CSI-RS index (0-95)
    pub csi_rs_index: u8,
    /// Measurement results for this CSI-RS
    pub results: MeasCellResults,
}

/// Measurement result for NR frequencies (`MeasResult2NR` in 3GPP)
#[derive(Debug, Clone)]
pub struct MeasResult2Nr {
    /// SSB frequency in ARFCN
    pub ssb_frequency_arfcn: Option<u32>,
    /// Reference frequency for CSI-RS
    pub ref_freq_csi_rs: Option<u32>,
    /// List of measured cells on this frequency
    pub meas_result_list: Vec<MeasResultNr>,
}

/// Serving cell measurement results
#[derive(Debug, Clone)]
pub struct MeasResultServFreqNr {
    /// Serving cell index (0-31)
    pub serv_cell_index: u8,
    /// Measurement results for the serving cell
    pub meas_result_serving_cell: MeasResultNr,
    /// Best neighbour measurement results (if available)
    pub meas_result_best_neigh_cell: Option<MeasResultNr>,
}

/// Measurement result for one inter-RAT (E-UTRA) cell (`MeasResultEUTRA` in
/// TS 38.331 §5.5.5), as reported for an event B1 or B2 measurement.
///
/// The quantities are the E-UTRA **ranges** of TS 36.133, not dBm:
/// `RSRP-RangeEUTRA` is `INTEGER (0..97)`, `RSRQ-RangeEUTRA` `(0..34)` and
/// `SINR-RangeEUTRA` `(0..127)`. Converting a level to a range is the caller's
/// job, because the mapping is per quantity and this codec does not model
/// measurement quantities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MeasResultEutra {
    /// `eutra-PhysCellId` (0-1007)
    pub eutra_phys_cell_id: u16,
    /// `rsrpResultEUTRA`, an `RSRP-RangeEUTRA` (0-97)
    pub rsrp: Option<u8>,
    /// `rsrqResultEUTRA`, an `RSRQ-RangeEUTRA` (0-34)
    pub rsrq: Option<u8>,
    /// `sinr-ResultEUTRA`, a `SINR-RangeEUTRA` (0-127)
    pub sinr: Option<u8>,
}

impl MeasResultEutra {
    /// An E-UTRA result carrying RSRP only, which is what B1/B2 are configured
    /// on in this simulator.
    pub fn with_rsrp(eutra_phys_cell_id: u16, rsrp: u8) -> Self {
        Self {
            eutra_phys_cell_id,
            rsrp: Some(rsrp),
            rsrq: None,
            sinr: None,
        }
    }
}

/// Measurement ID for identifying measurement configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeasIdValue(pub u8);

/// 6G: Enhanced measurement quantities
#[derive(Debug, Clone)]
pub struct EnhancedMeasQuantities {
    /// Beam quality indicator (0.0 to 1.0)
    pub beam_quality: Option<f64>,
    /// AI-predicted RSRP trend (positive = improving, negative = degrading)
    pub predicted_rsrp_trend: Option<f64>,
    /// Sensing-assisted measurement quality
    pub sensing_quality: Option<f64>,
    /// Sub-THz beam alignment metric (0.0 to 1.0)
    pub beam_alignment_metric: Option<f64>,
    /// NTN-specific Doppler compensation quality
    pub doppler_compensation_quality: Option<f64>,
}

/// Full measurement report data
#[derive(Debug, Clone)]
pub struct MeasurementReportData {
    /// Measurement ID
    pub meas_id: MeasIdValue,
    /// Serving cell measurements
    pub serv_freq_results: Vec<MeasResultServFreqNr>,
    /// Neighbour cell measurements (per frequency)
    pub neigh_freq_results: Vec<MeasResult2Nr>,
    /// Inter-RAT (E-UTRA) neighbour measurements, from the `measResultListEUTRA`
    /// arm of `measResultNeighCells`. Empty for an intra-NR report: the IE is a
    /// CHOICE, so a report carries NR results or E-UTRA results, never both.
    pub eutra_neigh_results: Vec<MeasResultEutra>,
    /// 6G: Enhanced measurement quantities
    pub enhanced_quantities: Option<EnhancedMeasQuantities>,
}

/// Parameters for building a Measurement Report
#[derive(Debug, Clone)]
pub struct MeasurementReportParams {
    /// Measurement ID (1-64)
    pub meas_id: u8,
    /// Serving cell measurements
    pub serv_freq_results: Vec<MeasResultServFreqNr>,
    /// Neighbour cell measurements
    pub neigh_freq_results: Vec<MeasResult2Nr>,
    /// Inter-RAT (E-UTRA) neighbour measurements, for a B1/B2 report.
    ///
    /// `measResultNeighCells` is a CHOICE, so this and `neigh_freq_results` are
    /// mutually exclusive: giving both is an error rather than a silent drop of
    /// one of them.
    pub eutra_neigh_results: Vec<MeasResultEutra>,
    /// 6G: Enhanced measurement quantities
    pub enhanced_quantities: Option<EnhancedMeasQuantities>,
}

/// Build a Measurement Report UL-DCCH message
pub fn build_measurement_report(
    params: &MeasurementReportParams,
) -> Result<UL_DCCH_Message, MeasurementReportError> {
    if params.meas_id < 1 || params.meas_id > 64 {
        return Err(MeasurementReportError::InvalidFieldValue(
            "Measurement ID must be 1-64".to_string(),
        ));
    }

    // Build MeasResults
    // The generated type uses MeasId and MeasResults structs
    let meas_id = MeasId(params.meas_id);

    // Build serving cell measurement results
    // We build a simplified MeasResults with the measId and measResultServingMOList
    let mut serv_mo_list_entries = Vec::new();
    for serv in &params.serv_freq_results {
        let meas_result_nr = build_meas_result_nr_value(&serv.meas_result_serving_cell);
        let entry = MeasResultServMO {
            serv_cell_id: ServCellIndex(serv.serv_cell_index),
            meas_result_serving_cell: meas_result_nr,
            meas_result_best_neigh_cell: serv
                .meas_result_best_neigh_cell
                .as_ref()
                .map(build_meas_result_nr_value),
        };
        serv_mo_list_entries.push(entry);
    }

    let meas_result_serv_mo_list = MeasResultServMOList(serv_mo_list_entries);

    let meas_result_neigh_cells =
        build_neigh_cell_results(&params.neigh_freq_results, &params.eutra_neigh_results)?;

    let meas_results = MeasResults {
        meas_id,
        meas_result_serving_mo_list: meas_result_serv_mo_list,
        meas_result_neigh_cells,
    };

    let measurement_report_ies = MeasurementReport_IEs {
        meas_results,
        late_non_critical_extension: None,
        non_critical_extension: None,
    };

    let measurement_report = MeasurementReport {
        critical_extensions: MeasurementReportCriticalExtensions::MeasurementReport(
            measurement_report_ies,
        ),
    };

    let message_type = UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::MeasurementReport(
        measurement_report,
    ));

    Ok(UL_DCCH_Message {
        message: message_type,
    })
}

/// Build a `MeasResultNR` value for the generated types
fn build_meas_result_nr_value(nr: &MeasResultNr) -> MeasResultNR {
    let cell_results =
        MeasResultNRMeasResultCellResults {
            results_ssb_cell: nr
                .cell_results
                .ssb_results
                .as_ref()
                .map(|r| MeasQuantityResults {
                    rsrp: r.rsrp.map(RSRP_Range),
                    rsrq: r.rsrq.map(RSRQ_Range),
                    sinr: r.sinr.map(SINR_Range),
                }),
            results_csi_rs_cell: nr.cell_results.csi_rs_results.as_ref().map(|r| {
                MeasQuantityResults {
                    rsrp: r.rsrp.map(RSRP_Range),
                    rsrq: r.rsrq.map(RSRQ_Range),
                    sinr: r.sinr.map(SINR_Range),
                }
            }),
        };

    // Build RS index results if provided
    let rs_index_results = nr.rs_index_results.as_ref().map(|ri| {
        let ssb_indexes = if ri.ssb_results.is_empty() {
            None
        } else {
            Some(ResultsPerSSB_IndexList(
                ri.ssb_results
                    .iter()
                    .map(|s| ResultsPerSSB_Index {
                        ssb_index: SSB_Index(s.ssb_index),
                        ssb_results: Some(MeasQuantityResults {
                            rsrp: s.results.rsrp.map(RSRP_Range),
                            rsrq: s.results.rsrq.map(RSRQ_Range),
                            sinr: s.results.sinr.map(SINR_Range),
                        }),
                    })
                    .collect(),
            ))
        };
        let csi_rs_indexes = if ri.csi_rs_results.is_empty() {
            None
        } else {
            Some(ResultsPerCSI_RS_IndexList(
                ri.csi_rs_results
                    .iter()
                    .map(|c| ResultsPerCSI_RS_Index {
                        csi_rs_index: CSI_RS_Index(c.csi_rs_index),
                        csi_rs_results: Some(MeasQuantityResults {
                            rsrp: c.results.rsrp.map(RSRP_Range),
                            rsrq: c.results.rsrq.map(RSRQ_Range),
                            sinr: c.results.sinr.map(SINR_Range),
                        }),
                    })
                    .collect(),
            ))
        };
        MeasResultNRMeasResultRsIndexResults {
            results_ssb_indexes: ssb_indexes,
            results_csi_rs_indexes: csi_rs_indexes,
        }
    });

    let meas_result = MeasResultNRMeasResult {
        cell_results,
        rs_index_results,
    };

    MeasResultNR {
        phys_cell_id: nr.phys_cell_id.map(PhysCellId),
        meas_result,
    }
}

/// Build a list of neighbor cell results for the generated types
/// Build the `measResultNeighCells` CHOICE.
///
/// `MeasResults.measResultNeighCells` is
/// `CHOICE { measResultListNR, ..., measResultListEUTRA }`, so exactly one arm
/// can be present. A caller supplying both lists is rejected: dropping one
/// silently is how an inter-RAT report would go out looking intra-NR.
///
/// The E-UTRA arm is an **extension** of the CHOICE (`..., measResultListEUTRA`),
/// which is why `an_inter_rat_report_round_trips_through_the_choice_extension_arm`
/// pins its encoding rather than assuming the generated codec handles the
/// extension marker.
fn build_neigh_cell_results(
    neigh: &[MeasResult2Nr],
    eutra: &[MeasResultEutra],
) -> Result<Option<MeasResultsMeasResultNeighCells>, MeasurementReportError> {
    let mut nr_list = Vec::new();
    for freq in neigh {
        for cell in &freq.meas_result_list {
            nr_list.push(build_meas_result_nr_value(cell));
        }
    }

    if !nr_list.is_empty() && !eutra.is_empty() {
        return Err(MeasurementReportError::InvalidFieldValue(format!(
            "measResultNeighCells is a CHOICE: {} NR and {} E-UTRA neighbour \
             results cannot both be reported for measId in one report",
            nr_list.len(),
            eutra.len()
        )));
    }

    if !nr_list.is_empty() {
        return Ok(Some(MeasResultsMeasResultNeighCells::MeasResultListNR(
            MeasResultListNR(nr_list),
        )));
    }

    if eutra.is_empty() {
        return Ok(None);
    }

    let mut eutra_list = Vec::with_capacity(eutra.len());
    for cell in eutra {
        eutra_list.push(build_meas_result_eutra_value(cell)?);
    }
    Ok(Some(MeasResultsMeasResultNeighCells::MeasResultListEUTRA(
        MeasResultListEUTRA(eutra_list),
    )))
}

/// `eutra-PhysCellId` is `PhysCellId ::= INTEGER (0..1007)`.
const EUTRA_PHYS_CELL_ID_MAX: u16 = 1007;
/// `RSRP-RangeEUTRA ::= INTEGER (0..97)` (TS 36.133).
const RSRP_RANGE_EUTRA_MAX: u8 = 97;
/// `RSRQ-RangeEUTRA ::= INTEGER (0..34)` (TS 36.133).
const RSRQ_RANGE_EUTRA_MAX: u8 = 34;
/// `SINR-RangeEUTRA ::= INTEGER (0..127)` (TS 36.133).
const SINR_RANGE_EUTRA_MAX: u8 = 127;

/// Build a `MeasResultEUTRA` value, rejecting anything outside the ASN.1
/// constraints rather than letting the encoder truncate it onto the wire.
fn build_meas_result_eutra_value(
    cell: &MeasResultEutra,
) -> Result<MeasResultEUTRA, MeasurementReportError> {
    if cell.eutra_phys_cell_id > EUTRA_PHYS_CELL_ID_MAX {
        return Err(MeasurementReportError::InvalidFieldValue(format!(
            "eutra-PhysCellId {} is outside INTEGER (0..{EUTRA_PHYS_CELL_ID_MAX})",
            cell.eutra_phys_cell_id
        )));
    }
    for (name, value, max) in [
        ("RSRP-RangeEUTRA", cell.rsrp, RSRP_RANGE_EUTRA_MAX),
        ("RSRQ-RangeEUTRA", cell.rsrq, RSRQ_RANGE_EUTRA_MAX),
        ("SINR-RangeEUTRA", cell.sinr, SINR_RANGE_EUTRA_MAX),
    ] {
        if let Some(value) = value {
            if value > max {
                return Err(MeasurementReportError::InvalidFieldValue(format!(
                    "{name} {value} is outside INTEGER (0..{max})"
                )));
            }
        }
    }
    if cell.rsrp.is_none() && cell.rsrq.is_none() && cell.sinr.is_none() {
        return Err(MeasurementReportError::MissingMandatoryField(format!(
            "MeasResultEUTRA for eutra-PhysCellId {} carries no measurement \
             quantity at all",
            cell.eutra_phys_cell_id
        )));
    }

    Ok(MeasResultEUTRA {
        eutra_phys_cell_id: PhysCellId(cell.eutra_phys_cell_id),
        meas_result: MeasQuantityResultsEUTRA {
            rsrp: cell.rsrp.map(RSRP_RangeEUTRA),
            rsrq: cell.rsrq.map(RSRQ_RangeEUTRA),
            sinr: cell.sinr.map(SINR_RangeEUTRA),
        },
        // cgi-Info needs a whole CGI_InfoEUTRA (PLMN, cell identity, tracking
        // area code); nothing in this simulator reads an E-UTRA CGI, and the IE
        // is optional.
        cgi_info: None,
    })
}

/// Read a `MeasResultEUTRA` back into the domain type.
fn parse_meas_result_eutra(cell: &MeasResultEUTRA) -> MeasResultEutra {
    MeasResultEutra {
        eutra_phys_cell_id: cell.eutra_phys_cell_id.0,
        rsrp: cell.meas_result.rsrp.as_ref().map(|r| r.0),
        rsrq: cell.meas_result.rsrq.as_ref().map(|r| r.0),
        sinr: cell.meas_result.sinr.as_ref().map(|r| r.0),
    }
}

/// Parse a Measurement Report from a UL-DCCH message
pub fn parse_measurement_report(
    msg: &UL_DCCH_Message,
) -> Result<MeasurementReportData, MeasurementReportError> {
    let report = match &msg.message {
        UL_DCCH_MessageType::C1(c1) => match c1 {
            UL_DCCH_MessageType_c1::MeasurementReport(r) => r,
            _ => {
                return Err(MeasurementReportError::InvalidMessageType {
                    expected: "MeasurementReport".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(MeasurementReportError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &report.critical_extensions {
        MeasurementReportCriticalExtensions::MeasurementReport(ies) => ies,
        MeasurementReportCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(MeasurementReportError::InvalidMessageType {
                expected: "measurementReport".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    let meas_id = MeasIdValue(ies.meas_results.meas_id.0);

    // Parse serving cell results
    let mut serv_freq_results = Vec::new();
    for serv_mo in &ies.meas_results.meas_result_serving_mo_list.0 {
        let serving_cell = parse_meas_result_nr(&serv_mo.meas_result_serving_cell);
        let best_neigh = serv_mo
            .meas_result_best_neigh_cell
            .as_ref()
            .map(parse_meas_result_nr);

        serv_freq_results.push(MeasResultServFreqNr {
            serv_cell_index: serv_mo.serv_cell_id.0,
            meas_result_serving_cell: serving_cell,
            meas_result_best_neigh_cell: best_neigh,
        });
    }

    // Parse neighbor cell results. The IE is a CHOICE, so at most one of the two
    // lists below is non-empty.
    let mut neigh_freq_results = Vec::new();
    let mut eutra_neigh_results = Vec::new();
    match &ies.meas_results.meas_result_neigh_cells {
        Some(MeasResultsMeasResultNeighCells::MeasResultListNR(list)) => {
            // All neighbor cells reported as a flat list; group into single frequency entry
            let nr_results: Vec<MeasResultNr> = list.0.iter().map(parse_meas_result_nr).collect();
            if !nr_results.is_empty() {
                neigh_freq_results.push(MeasResult2Nr {
                    ssb_frequency_arfcn: None,
                    ref_freq_csi_rs: None,
                    meas_result_list: nr_results,
                });
            }
        }
        Some(MeasResultsMeasResultNeighCells::MeasResultListEUTRA(list)) => {
            eutra_neigh_results = list.0.iter().map(parse_meas_result_eutra).collect();
        }
        None => {}
    }

    Ok(MeasurementReportData {
        meas_id,
        serv_freq_results,
        neigh_freq_results,
        eutra_neigh_results,
        enhanced_quantities: None,
    })
}

/// Parse a `MeasResultNR` generated type into our domain type
fn parse_meas_result_nr(nr: &MeasResultNR) -> MeasResultNr {
    let ssb_results = nr
        .meas_result
        .cell_results
        .results_ssb_cell
        .as_ref()
        .map(|r| MeasCellResults {
            rsrp: r.rsrp.as_ref().map(|v| v.0),
            rsrq: r.rsrq.as_ref().map(|v| v.0),
            sinr: r.sinr.as_ref().map(|v| v.0),
        });

    let csi_rs_results = nr
        .meas_result
        .cell_results
        .results_csi_rs_cell
        .as_ref()
        .map(|r| MeasCellResults {
            rsrp: r.rsrp.as_ref().map(|v| v.0),
            rsrq: r.rsrq.as_ref().map(|v| v.0),
            sinr: r.sinr.as_ref().map(|v| v.0),
        });

    // Parse RS index results
    let rs_index_results = nr.meas_result.rs_index_results.as_ref().map(|ri| {
        let ssb_results = ri
            .results_ssb_indexes
            .as_ref()
            .map(|list| {
                list.0
                    .iter()
                    .map(|s| MeasResultPerSsbIndex {
                        ssb_index: s.ssb_index.0,
                        results: s
                            .ssb_results
                            .as_ref()
                            .map(|r| MeasCellResults {
                                rsrp: r.rsrp.as_ref().map(|v| v.0),
                                rsrq: r.rsrq.as_ref().map(|v| v.0),
                                sinr: r.sinr.as_ref().map(|v| v.0),
                            })
                            .unwrap_or(MeasCellResults {
                                rsrp: None,
                                rsrq: None,
                                sinr: None,
                            }),
                    })
                    .collect()
            })
            .expect("value expected");

        let csi_rs_results = ri
            .results_csi_rs_indexes
            .as_ref()
            .map(|list| {
                list.0
                    .iter()
                    .map(|c| MeasResultPerCsiRsIndex {
                        csi_rs_index: c.csi_rs_index.0,
                        results: c
                            .csi_rs_results
                            .as_ref()
                            .map(|r| MeasCellResults {
                                rsrp: r.rsrp.as_ref().map(|v| v.0),
                                rsrq: r.rsrq.as_ref().map(|v| v.0),
                                sinr: r.sinr.as_ref().map(|v| v.0),
                            })
                            .unwrap_or(MeasCellResults {
                                rsrp: None,
                                rsrq: None,
                                sinr: None,
                            }),
                    })
                    .collect()
            })
            .expect("value expected");

        MeasResultRsIndexNr {
            ssb_results,
            csi_rs_results,
        }
    });

    MeasResultNr {
        phys_cell_id: nr.phys_cell_id.as_ref().map(|pci| pci.0),
        cell_results: MeasResultCellNr {
            ssb_results,
            csi_rs_results,
        },
        rs_index_results,
    }
}

/// Build and encode a Measurement Report to bytes
pub fn encode_measurement_report(
    params: &MeasurementReportParams,
) -> Result<Vec<u8>, MeasurementReportError> {
    let msg = build_measurement_report(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse a Measurement Report from bytes
pub fn decode_measurement_report(
    bytes: &[u8],
) -> Result<MeasurementReportData, MeasurementReportError> {
    let msg: UL_DCCH_Message = decode_rrc(bytes)?;
    parse_measurement_report(&msg)
}

/// Check if a UL-DCCH message is a Measurement Report
pub fn is_measurement_report(msg: &UL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        UL_DCCH_MessageType::C1(UL_DCCH_MessageType_c1::MeasurementReport(_))
    )
}

// ============================================================================
// Helper functions for measurement value conversion
// ============================================================================

/// Convert RSRP range value (0-127) to dBm (-156 to -31)
pub fn rsrp_range_to_dbm(range: u8) -> f64 {
    -156.0 + range as f64
}

/// Convert dBm to RSRP range value (0-127)
pub fn dbm_to_rsrp_range(dbm: f64) -> u8 {
    let range = (dbm + 156.0).round() as i16;
    range.clamp(0, 127) as u8
}

/// Convert RSRQ range value (0-127) to dB (-43 to 20)
pub fn rsrq_range_to_db(range: u8) -> f64 {
    -43.0 + (range as f64 * 0.5)
}

/// Convert dB to RSRQ range value (0-127)
pub fn db_to_rsrq_range(db: f64) -> u8 {
    let range = ((db + 43.0) / 0.5).round() as i16;
    range.clamp(0, 127) as u8
}

/// Convert SINR range value (0-127) to dB (-23 to 40)
pub fn sinr_range_to_db(range: u8) -> f64 {
    -23.0 + (range as f64 * 0.5)
}

/// Convert dB to SINR range value (0-127)
pub fn db_to_sinr_range(db: f64) -> u8 {
    let range = ((db + 23.0) / 0.5).round() as i16;
    range.clamp(0, 127) as u8
}

impl MeasurementReportData {
    /// Validate the measurement report data
    pub fn validate(&self) -> Result<(), MeasurementReportError> {
        if self.meas_id.0 < 1 || self.meas_id.0 > 64 {
            return Err(MeasurementReportError::InvalidFieldValue(
                "Measurement ID must be 1-64".to_string(),
            ));
        }
        if self.serv_freq_results.is_empty() {
            return Err(MeasurementReportError::MissingMandatoryField(
                "serv_freq_results (at least one serving cell measurement required)".to_string(),
            ));
        }
        for serv in &self.serv_freq_results {
            if serv.serv_cell_index > 31 {
                return Err(MeasurementReportError::InvalidFieldValue(format!(
                    "Serving cell index {} exceeds maximum 31",
                    serv.serv_cell_index
                )));
            }
        }
        Ok(())
    }
}

impl EnhancedMeasQuantities {
    /// Validate enhanced measurement quantities
    pub fn validate(&self) -> Result<(), MeasurementReportError> {
        if let Some(bq) = self.beam_quality {
            if !(0.0..=1.0).contains(&bq) {
                return Err(MeasurementReportError::InvalidFieldValue(
                    "Beam quality must be in range [0.0, 1.0]".to_string(),
                ));
            }
        }
        if let Some(ba) = self.beam_alignment_metric {
            if !(0.0..=1.0).contains(&ba) {
                return Err(MeasurementReportError::InvalidFieldValue(
                    "Beam alignment metric must be in range [0.0, 1.0]".to_string(),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_meas_result_nr() -> MeasResultNr {
        MeasResultNr {
            phys_cell_id: Some(100),
            cell_results: MeasResultCellNr {
                ssb_results: Some(MeasCellResults {
                    rsrp: Some(80), // -76 dBm
                    rsrq: Some(40), // -23 dB
                    sinr: Some(60), //  7 dB
                }),
                csi_rs_results: None,
            },
            rs_index_results: None,
        }
    }

    fn create_test_params() -> MeasurementReportParams {
        MeasurementReportParams {
            meas_id: 1,
            serv_freq_results: vec![MeasResultServFreqNr {
                serv_cell_index: 0,
                meas_result_serving_cell: create_test_meas_result_nr(),
                meas_result_best_neigh_cell: Some(MeasResultNr {
                    phys_cell_id: Some(200),
                    cell_results: MeasResultCellNr {
                        ssb_results: Some(MeasCellResults {
                            rsrp: Some(70),
                            rsrq: Some(35),
                            sinr: Some(50),
                        }),
                        csi_rs_results: None,
                    },
                    rs_index_results: None,
                }),
            }],
            eutra_neigh_results: Vec::new(),
            neigh_freq_results: vec![],
            enhanced_quantities: None,
        }
    }

    #[test]
    fn test_build_measurement_report() {
        let params = create_test_params();
        let result = build_measurement_report(&params);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parse_measurement_report() {
        let params = create_test_params();
        let msg = build_measurement_report(&params).unwrap();
        let data = parse_measurement_report(&msg).unwrap();

        assert_eq!(data.meas_id.0, 1);
        assert_eq!(data.serv_freq_results.len(), 1);
        assert_eq!(data.serv_freq_results[0].serv_cell_index, 0);

        let serving = &data.serv_freq_results[0].meas_result_serving_cell;
        assert_eq!(serving.phys_cell_id, Some(100));
        let ssb = serving.cell_results.ssb_results.as_ref().unwrap();
        assert_eq!(ssb.rsrp, Some(80));
        assert_eq!(ssb.rsrq, Some(40));
        assert_eq!(ssb.sinr, Some(60));
    }

    #[test]
    fn test_encode_decode_measurement_report() {
        let params = create_test_params();
        let encoded = encode_measurement_report(&params).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let decoded = decode_measurement_report(&encoded).expect("Failed to decode");
        assert_eq!(decoded.meas_id.0, params.meas_id);
    }

    #[test]
    fn test_is_measurement_report() {
        let params = create_test_params();
        let msg = build_measurement_report(&params).unwrap();
        assert!(is_measurement_report(&msg));
    }

    #[test]
    fn test_invalid_meas_id() {
        let mut params = create_test_params();
        params.meas_id = 0;
        assert!(build_measurement_report(&params).is_err());

        params.meas_id = 65;
        assert!(build_measurement_report(&params).is_err());
    }

    #[test]
    fn test_rsrp_conversion() {
        assert_eq!(rsrp_range_to_dbm(0), -156.0);
        assert_eq!(rsrp_range_to_dbm(127), -29.0);
        assert_eq!(rsrp_range_to_dbm(80), -76.0);

        assert_eq!(dbm_to_rsrp_range(-156.0), 0);
        assert_eq!(dbm_to_rsrp_range(-76.0), 80);
        assert_eq!(dbm_to_rsrp_range(-200.0), 0); // clamped
    }

    #[test]
    fn test_rsrq_conversion() {
        assert_eq!(rsrq_range_to_db(0), -43.0);
        assert_eq!(rsrq_range_to_db(40), -23.0);
    }

    #[test]
    fn test_sinr_conversion() {
        assert_eq!(sinr_range_to_db(0), -23.0);
        assert_eq!(sinr_range_to_db(60), 7.0);
    }

    #[test]
    fn test_measurement_report_data_validate() {
        let data = MeasurementReportData {
            meas_id: MeasIdValue(1),
            serv_freq_results: vec![MeasResultServFreqNr {
                serv_cell_index: 0,
                meas_result_serving_cell: create_test_meas_result_nr(),
                meas_result_best_neigh_cell: None,
            }],
            eutra_neigh_results: Vec::new(),
            neigh_freq_results: vec![],
            enhanced_quantities: None,
        };
        assert!(data.validate().is_ok());
    }

    #[test]
    fn test_measurement_report_data_validate_empty() {
        let data = MeasurementReportData {
            meas_id: MeasIdValue(1),
            serv_freq_results: vec![],
            eutra_neigh_results: Vec::new(),
            neigh_freq_results: vec![],
            enhanced_quantities: None,
        };
        assert!(data.validate().is_err());
    }

    #[test]
    fn test_enhanced_quantities_validate() {
        let eq = EnhancedMeasQuantities {
            beam_quality: Some(0.9),
            predicted_rsrp_trend: Some(1.5),
            sensing_quality: Some(0.8),
            beam_alignment_metric: Some(0.95),
            doppler_compensation_quality: None,
        };
        assert!(eq.validate().is_ok());

        let eq_invalid = EnhancedMeasQuantities {
            beam_quality: Some(1.5), // invalid
            predicted_rsrp_trend: None,
            sensing_quality: None,
            beam_alignment_metric: None,
            doppler_compensation_quality: None,
        };
        assert!(eq_invalid.validate().is_err());
    }

    #[test]
    fn test_multiple_serving_cells() {
        let params = MeasurementReportParams {
            meas_id: 2,
            serv_freq_results: vec![
                MeasResultServFreqNr {
                    serv_cell_index: 0,
                    meas_result_serving_cell: create_test_meas_result_nr(),
                    meas_result_best_neigh_cell: None,
                },
                MeasResultServFreqNr {
                    serv_cell_index: 1,
                    meas_result_serving_cell: MeasResultNr {
                        phys_cell_id: Some(300),
                        cell_results: MeasResultCellNr {
                            ssb_results: Some(MeasCellResults {
                                rsrp: Some(90),
                                rsrq: None,
                                sinr: None,
                            }),
                            csi_rs_results: None,
                        },
                        rs_index_results: None,
                    },
                    meas_result_best_neigh_cell: None,
                },
            ],
            eutra_neigh_results: Vec::new(),
            neigh_freq_results: vec![],
            enhanced_quantities: None,
        };

        let msg = build_measurement_report(&params).unwrap();
        let data = parse_measurement_report(&msg).unwrap();
        assert_eq!(data.serv_freq_results.len(), 2);
    }

    #[test]
    fn test_csi_rs_results() {
        let nr = MeasResultNr {
            phys_cell_id: Some(100),
            cell_results: MeasResultCellNr {
                ssb_results: None,
                csi_rs_results: Some(MeasCellResults {
                    rsrp: Some(85),
                    rsrq: Some(45),
                    sinr: Some(55),
                }),
            },
            rs_index_results: None,
        };
        assert!(nr.cell_results.ssb_results.is_none());
        assert!(nr.cell_results.csi_rs_results.is_some());
    }

    // ========================================================================
    // Inter-RAT reporting (issue #113, TS 38.331 §5.5.5 measResultListEUTRA)
    // ========================================================================

    /// Params for a B1/B2 report: a serving cell and inter-RAT neighbours, with
    /// no NR neighbours, because `measResultNeighCells` is a CHOICE.
    fn inter_rat_params(eutra: Vec<MeasResultEutra>) -> MeasurementReportParams {
        MeasurementReportParams {
            eutra_neigh_results: eutra,
            neigh_freq_results: Vec::new(),
            ..create_test_params()
        }
    }

    /// Params with one NR neighbour, since `create_test_params` has none.
    fn nr_neighbour_params() -> MeasurementReportParams {
        MeasurementReportParams {
            neigh_freq_results: vec![MeasResult2Nr {
                ssb_frequency_arfcn: Some(620_000),
                ref_freq_csi_rs: None,
                meas_result_list: vec![create_test_meas_result_nr()],
            }],
            ..create_test_params()
        }
    }

    /// The whole `MeasResultEUTRA` structure survives a build → parse round trip,
    /// which is the half of the encoding that this crate owns: every quantity,
    /// both bounds, and the CHOICE arm being the E-UTRA one.
    ///
    /// This is a **structure** round trip, not a byte one, because the UPER
    /// encoder cannot reach the extension arm at all — see
    /// `the_eutra_choice_arm_cannot_be_uper_encoded_by_this_codec`.
    #[test]
    fn an_inter_rat_report_round_trips_at_the_structure_level() {
        let cells = vec![
            MeasResultEutra {
                eutra_phys_cell_id: 42,
                rsrp: Some(50),
                rsrq: Some(20),
                sinr: Some(70),
            },
            MeasResultEutra::with_rsrp(1007, 97),
        ];
        let params = inter_rat_params(cells.clone());

        let msg = build_measurement_report(&params).expect("build");
        let decoded = parse_measurement_report(&msg).expect("parse");

        assert_eq!(decoded.meas_id.0, params.meas_id);
        assert_eq!(
            decoded.eutra_neigh_results, cells,
            "every E-UTRA quantity survives the CHOICE arm"
        );
        assert!(
            decoded.neigh_freq_results.is_empty(),
            "an inter-RAT report carries no NR neighbour list"
        );
    }

    /// **The ceiling #113 asks to be recorded.** `measResultListEUTRA` is an
    /// *extension* arm of the `measResultNeighCells` CHOICE
    /// (`CHOICE { measResultListNR, ..., measResultListEUTRA }`), and
    /// `asn1-codecs` 0.7 refuses outright to encode an extended choice index:
    /// `per/common/encode/mod.rs` returns `EncodeNotSupported` with
    /// "Encode of extended choice not yet implemented" whenever the selected arm
    /// is past the extension marker. Decoding one *is* implemented, so this is an
    /// encoder-only ceiling.
    ///
    /// So a UE cannot put an inter-RAT measurement result on the wire in real
    /// UPER, whatever #107 does about the hand-rolled report — the block is one
    /// layer below, in the codec crate.
    ///
    /// This test **fails when the limitation is lifted**, which is the signal to
    /// replace `an_inter_rat_report_round_trips_at_the_structure_level` with a
    /// byte round trip. It is pinned as a ceiling, not asserted as correct.
    #[test]
    fn the_eutra_choice_arm_cannot_be_uper_encoded_by_this_codec() {
        let params = inter_rat_params(vec![MeasResultEutra::with_rsrp(42, 50)]);

        // The structure builds fine: the refusal is in the encoder, not here.
        build_measurement_report(&params).expect("the message structure is valid");

        let err = encode_measurement_report(&params)
            .expect_err("asn1-codecs 0.7 cannot encode an extended choice index");
        let message = err.to_string();
        assert!(
            message.contains("extended choice"),
            "expected the extended-choice refusal, got: {message}"
        );

        // An intra-NR report over the same code path encodes, so the failure is
        // specific to the extension arm and not to measurement reports at large.
        encode_measurement_report(&nr_neighbour_params()).expect("the non-extended NR arm encodes");
    }

    /// The two arms must not be confusable: an intra-NR report still encodes,
    /// decodes as NR, and carries no E-UTRA list — so adding the extension arm
    /// did not disturb the arm that was already there.
    #[test]
    fn an_intra_nr_report_still_round_trips_as_the_nr_arm() {
        let encoded = encode_measurement_report(&nr_neighbour_params()).expect("encode");
        let decoded = decode_measurement_report(&encoded).expect("decode");

        assert!(
            !decoded.neigh_freq_results.is_empty(),
            "the NR arm is unaffected by the E-UTRA arm existing"
        );
        assert!(decoded.eutra_neigh_results.is_empty());
    }

    /// A CHOICE can carry one arm. Supplying both lists is rejected, because
    /// dropping one silently is how an inter-RAT report would go out looking
    /// intra-NR.
    #[test]
    fn nr_and_eutra_neighbours_in_one_report_are_rejected() {
        let params = MeasurementReportParams {
            eutra_neigh_results: vec![MeasResultEutra::with_rsrp(42, 50)],
            ..nr_neighbour_params()
        };

        let err = build_measurement_report(&params).expect_err("the CHOICE cannot carry both");
        assert!(
            matches!(err, MeasurementReportError::InvalidFieldValue(ref m) if m.contains("CHOICE")),
            "unexpected error: {err}"
        );
    }

    /// The E-UTRA quantities are the TS 36.133 ranges, with tighter bounds than
    /// the NR ones — RSRQ is `INTEGER (0..34)`, not `(0..127)`. An out-of-range
    /// value is an error rather than a field truncated on the wire.
    #[test]
    fn out_of_range_eutra_quantities_are_rejected() {
        let cases = [
            ("eutra-PhysCellId", MeasResultEutra::with_rsrp(1008, 50)),
            ("RSRP-RangeEUTRA", MeasResultEutra::with_rsrp(1, 98)),
            (
                "RSRQ-RangeEUTRA",
                MeasResultEutra {
                    eutra_phys_cell_id: 1,
                    rsrp: None,
                    rsrq: Some(35),
                    sinr: None,
                },
            ),
            (
                "SINR-RangeEUTRA",
                MeasResultEutra {
                    eutra_phys_cell_id: 1,
                    rsrp: None,
                    rsrq: None,
                    sinr: Some(128),
                },
            ),
        ];

        for (constraint, cell) in cases {
            let err = build_measurement_report(&inter_rat_params(vec![cell]))
                .expect_err("an out-of-range value must not be encoded");
            let message = err.to_string();
            assert!(
                message.contains(constraint),
                "the error must name the constraint it violated; got: {message}"
            );
        }

        // The bounds themselves are legal: the check rejects what is past them,
        // not the edge.
        build_measurement_report(&inter_rat_params(vec![MeasResultEutra {
            eutra_phys_cell_id: 1007,
            rsrp: Some(97),
            rsrq: Some(34),
            sinr: Some(127),
        }]))
        .expect("the upper bound of every range is encodable");
    }

    /// A `MeasResultEUTRA` with no quantity at all reports nothing: every member
    /// of `MeasQuantityResultsEUTRA` is optional, so the encoding is legal and
    /// useless.
    #[test]
    fn an_eutra_result_with_no_quantity_is_rejected() {
        let params = inter_rat_params(vec![MeasResultEutra {
            eutra_phys_cell_id: 42,
            rsrp: None,
            rsrq: None,
            sinr: None,
        }]);

        let err = build_measurement_report(&params).expect_err("nothing measured");
        assert!(matches!(
            err,
            MeasurementReportError::MissingMandatoryField(_)
        ));
    }

    /// The `measResultListEUTRA` bound is `SEQUENCE (SIZE (1..8))`, so eight fit.
    /// Structure-level, for the reason in
    /// `the_eutra_choice_arm_cannot_be_uper_encoded_by_this_codec`.
    #[test]
    fn a_full_eutra_neighbour_list_round_trips_at_the_structure_level() {
        let cells: Vec<_> = (0..8)
            .map(|i| MeasResultEutra::with_rsrp(100 + i, 40 + i as u8))
            .collect();

        let msg = build_measurement_report(&inter_rat_params(cells.clone())).expect("build");
        let decoded = parse_measurement_report(&msg).expect("parse");

        assert_eq!(decoded.eutra_neigh_results, cells);
    }
}
