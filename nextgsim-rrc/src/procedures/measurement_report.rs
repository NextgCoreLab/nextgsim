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

/// Measurement result for NR frequencies (MeasResult2NR in 3GPP)
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
            meas_result_best_neigh_cell: serv.meas_result_best_neigh_cell
                .as_ref()
                .map(build_meas_result_nr_value),
        };
        serv_mo_list_entries.push(entry);
    }

    let meas_result_serv_mo_list = MeasResultServMOList(serv_mo_list_entries);

    let meas_result_neigh_cells = build_neigh_cell_results(&params.neigh_freq_results);

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

    let message_type = UL_DCCH_MessageType::C1(
        UL_DCCH_MessageType_c1::MeasurementReport(measurement_report),
    );

    Ok(UL_DCCH_Message { message: message_type })
}

/// Build a MeasResultNR value for the generated types
fn build_meas_result_nr_value(nr: &MeasResultNr) -> MeasResultNR {
    let cell_results = MeasResultNRMeasResultCellResults {
        results_ssb_cell: nr.cell_results.ssb_results.as_ref().map(|r| {
            MeasQuantityResults {
                rsrp: r.rsrp.map(RSRP_Range),
                rsrq: r.rsrq.map(RSRQ_Range),
                sinr: r.sinr.map(SINR_Range),
            }
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
                ri.ssb_results.iter().map(|s| {
                    ResultsPerSSB_Index {
                        ssb_index: SSB_Index(s.ssb_index),
                        ssb_results: Some(MeasQuantityResults {
                            rsrp: s.results.rsrp.map(RSRP_Range),
                            rsrq: s.results.rsrq.map(RSRQ_Range),
                            sinr: s.results.sinr.map(SINR_Range),
                        }),
                    }
                }).collect(),
            ))
        };
        let csi_rs_indexes = if ri.csi_rs_results.is_empty() {
            None
        } else {
            Some(ResultsPerCSI_RS_IndexList(
                ri.csi_rs_results.iter().map(|c| {
                    ResultsPerCSI_RS_Index {
                        csi_rs_index: CSI_RS_Index(c.csi_rs_index),
                        csi_rs_results: Some(MeasQuantityResults {
                            rsrp: c.results.rsrp.map(RSRP_Range),
                            rsrq: c.results.rsrq.map(RSRQ_Range),
                            sinr: c.results.sinr.map(SINR_Range),
                        }),
                    }
                }).collect(),
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
fn build_neigh_cell_results(neigh: &[MeasResult2Nr]) -> Option<MeasResultsMeasResultNeighCells> {
    if neigh.is_empty() {
        return None;
    }
    let mut nr_list = Vec::new();
    for freq in neigh {
        for cell in &freq.meas_result_list {
            nr_list.push(build_meas_result_nr_value(cell));
        }
    }
    if nr_list.is_empty() {
        return None;
    }
    Some(MeasResultsMeasResultNeighCells::MeasResultListNR(
        MeasResultListNR(nr_list),
    ))
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

    // Parse neighbor cell results
    let neigh_freq_results = match &ies.meas_results.meas_result_neigh_cells {
        Some(MeasResultsMeasResultNeighCells::MeasResultListNR(list)) => {
            // All neighbor cells reported as a flat list; group into single frequency entry
            let nr_results: Vec<MeasResultNr> = list.0.iter().map(parse_meas_result_nr).collect();
            if nr_results.is_empty() {
                Vec::new()
            } else {
                vec![MeasResult2Nr {
                    ssb_frequency_arfcn: None,
                    ref_freq_csi_rs: None,
                    meas_result_list: nr_results,
                }]
            }
        }
        _ => Vec::new(),
    };

    Ok(MeasurementReportData {
        meas_id,
        serv_freq_results,
        neigh_freq_results,
        enhanced_quantities: None,
    })
}

/// Parse a MeasResultNR generated type into our domain type
fn parse_meas_result_nr(nr: &MeasResultNR) -> MeasResultNr {
    let ssb_results = nr.meas_result.cell_results.results_ssb_cell.as_ref().map(|r| {
        MeasCellResults {
            rsrp: r.rsrp.as_ref().map(|v| v.0),
            rsrq: r.rsrq.as_ref().map(|v| v.0),
            sinr: r.sinr.as_ref().map(|v| v.0),
        }
    });

    let csi_rs_results = nr.meas_result.cell_results.results_csi_rs_cell.as_ref().map(|r| {
        MeasCellResults {
            rsrp: r.rsrp.as_ref().map(|v| v.0),
            rsrq: r.rsrq.as_ref().map(|v| v.0),
            sinr: r.sinr.as_ref().map(|v| v.0),
        }
    });

    // Parse RS index results
    let rs_index_results = nr.meas_result.rs_index_results.as_ref().map(|ri| {
        let ssb_results = ri.results_ssb_indexes.as_ref()
            .map(|list| {
                list.0.iter().map(|s| {
                    MeasResultPerSsbIndex {
                        ssb_index: s.ssb_index.0,
                        results: s.ssb_results.as_ref().map(|r| MeasCellResults {
                            rsrp: r.rsrp.as_ref().map(|v| v.0),
                            rsrq: r.rsrq.as_ref().map(|v| v.0),
                            sinr: r.sinr.as_ref().map(|v| v.0),
                        }).unwrap_or(MeasCellResults { rsrp: None, rsrq: None, sinr: None }),
                    }
                }).collect()
            })
            .unwrap_or_default();

        let csi_rs_results = ri.results_csi_rs_indexes.as_ref()
            .map(|list| {
                list.0.iter().map(|c| {
                    MeasResultPerCsiRsIndex {
                        csi_rs_index: c.csi_rs_index.0,
                        results: c.csi_rs_results.as_ref().map(|r| MeasCellResults {
                            rsrp: r.rsrp.as_ref().map(|v| v.0),
                            rsrq: r.rsrq.as_ref().map(|v| v.0),
                            sinr: r.sinr.as_ref().map(|v| v.0),
                        }).unwrap_or(MeasCellResults { rsrp: None, rsrq: None, sinr: None }),
                    }
                }).collect()
            })
            .unwrap_or_default();

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
}
