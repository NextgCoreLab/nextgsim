//! Conditional Handover (CHO) Configuration
//!
//! 6G extension: RRC procedure for Conditional Handover configuration
//! as defined in 3GPP TS 38.331 Section 5.3.5.8 with enhancements for
//! predictive and AI-assisted handover decisions.
//!
//! This module implements:
//! - `ChoConfig` - Conditional Handover configuration with conditions and target cell configs
//! - Condition types: event-based (A3, A5), timer-based, and 6G predictive

use thiserror::Error;

/// Errors that can occur during Conditional Handover procedures
#[derive(Debug, Error)]
pub enum ConditionalHandoverError {
    /// Invalid CHO configuration
    #[error("Invalid CHO configuration: {0}")]
    InvalidConfig(String),

    /// Missing mandatory field
    #[error("Missing mandatory field: {0}")]
    MissingMandatoryField(String),

    /// Encoding/decoding error
    #[error("Codec error: {0}")]
    CodecError(String),
}

/// CHO condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChoConditionType {
    /// Event A3: Neighbour becomes offset better than `SpCell`
    EventA3,
    /// Event A5: `SpCell` becomes worse than threshold1 AND neighbour becomes better than threshold2
    EventA5,
    /// Timer-based: handover after timer expiry
    TimerBased,
    /// 6G: Predictive handover based on UE trajectory
    Predictive,
    /// 6G: AI-assisted condition evaluation
    AiAssisted,
}

/// Time-to-trigger values for event-based conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeToTrigger {
    /// 0 ms
    Ms0,
    /// 40 ms
    Ms40,
    /// 64 ms
    Ms64,
    /// 80 ms
    Ms80,
    /// 100 ms
    Ms100,
    /// 128 ms
    Ms128,
    /// 160 ms
    Ms160,
    /// 256 ms
    Ms256,
    /// 320 ms
    Ms320,
    /// 480 ms
    Ms480,
    /// 512 ms
    Ms512,
    /// 640 ms
    Ms640,
    /// 1024 ms
    Ms1024,
    /// 1280 ms
    Ms1280,
    /// 2560 ms
    Ms2560,
    /// 5120 ms
    Ms5120,
}

impl TimeToTrigger {
    /// Get the time-to-trigger value in milliseconds
    pub fn to_ms(&self) -> u32 {
        match self {
            TimeToTrigger::Ms0 => 0,
            TimeToTrigger::Ms40 => 40,
            TimeToTrigger::Ms64 => 64,
            TimeToTrigger::Ms80 => 80,
            TimeToTrigger::Ms100 => 100,
            TimeToTrigger::Ms128 => 128,
            TimeToTrigger::Ms160 => 160,
            TimeToTrigger::Ms256 => 256,
            TimeToTrigger::Ms320 => 320,
            TimeToTrigger::Ms480 => 480,
            TimeToTrigger::Ms512 => 512,
            TimeToTrigger::Ms640 => 640,
            TimeToTrigger::Ms1024 => 1024,
            TimeToTrigger::Ms1280 => 1280,
            TimeToTrigger::Ms2560 => 2560,
            TimeToTrigger::Ms5120 => 5120,
        }
    }
}

/// Hysteresis value in dB (0.0 to 30.0, step 0.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hysteresis(pub f64);

impl Hysteresis {
    /// Create a new Hysteresis value, clamped to valid range
    pub fn new(db: f64) -> Result<Self, ConditionalHandoverError> {
        if !(0.0..=30.0).contains(&db) {
            return Err(ConditionalHandoverError::InvalidConfig(
                "Hysteresis must be in range [0.0, 30.0] dB".to_string(),
            ));
        }
        Ok(Hysteresis(db))
    }
}

/// RSRP threshold value in dBm (-156 to -31)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RsrpThreshold(pub i16);

impl RsrpThreshold {
    /// Create a new RSRP threshold
    pub fn new(dbm: i16) -> Result<Self, ConditionalHandoverError> {
        if !(-156..=-31).contains(&dbm) {
            return Err(ConditionalHandoverError::InvalidConfig(
                "RSRP threshold must be in range [-156, -31] dBm".to_string(),
            ));
        }
        Ok(RsrpThreshold(dbm))
    }
}

/// RSRQ threshold value in dB (-43 to 20, step 0.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RsrqThreshold(pub f64);

impl RsrqThreshold {
    /// Create a new RSRQ threshold
    pub fn new(db: f64) -> Result<Self, ConditionalHandoverError> {
        if !(-43.0..=20.0).contains(&db) {
            return Err(ConditionalHandoverError::InvalidConfig(
                "RSRQ threshold must be in range [-43.0, 20.0] dB".to_string(),
            ));
        }
        Ok(RsrqThreshold(db))
    }
}

/// Offset value for A3 event in dB (-30.0 to 30.0, step 0.5)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct A3Offset(pub f64);

impl A3Offset {
    /// Create a new A3 offset value
    pub fn new(db: f64) -> Result<Self, ConditionalHandoverError> {
        if !(-30.0..=30.0).contains(&db) {
            return Err(ConditionalHandoverError::InvalidConfig(
                "A3 offset must be in range [-30.0, 30.0] dB".to_string(),
            ));
        }
        Ok(A3Offset(db))
    }
}

/// Event A3 condition parameters
///
/// Condition: Neighbour becomes amount of offset better than `SpCell`
#[derive(Debug, Clone)]
pub struct EventA3Condition {
    /// A3 offset value in dB
    pub a3_offset: A3Offset,
    /// Hysteresis in dB
    pub hysteresis: Hysteresis,
    /// Time to trigger
    pub time_to_trigger: TimeToTrigger,
    /// Whether to use RSRP (true) or RSRQ (false)
    pub use_rsrp: bool,
}

/// Event A5 condition parameters
///
/// Condition: `SpCell` RSRP < threshold1 AND Neighbour RSRP > threshold2
#[derive(Debug, Clone)]
pub struct EventA5Condition {
    /// Threshold 1 (for serving cell becoming worse)
    pub threshold1: RsrpThreshold,
    /// Threshold 2 (for neighbour cell becoming better)
    pub threshold2: RsrpThreshold,
    /// Hysteresis in dB
    pub hysteresis: Hysteresis,
    /// Time to trigger
    pub time_to_trigger: TimeToTrigger,
}

/// Timer-based condition parameters
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimerBasedCondition {
    /// Timer duration in milliseconds
    pub timer_ms: u32,
    /// Whether to restart timer on better measurement
    pub restart_on_improvement: bool,
}

/// 6G: Predictive handover condition parameters
#[derive(Debug, Clone)]
pub struct PredictiveCondition {
    /// Predicted time until handover is needed (ms)
    pub predicted_handover_time_ms: u32,
    /// Confidence level of prediction (0.0 to 1.0)
    pub confidence_level: f64,
    /// Minimum confidence required to trigger CHO
    pub min_confidence_threshold: f64,
    /// UE speed estimate in m/s (if available)
    pub ue_speed_ms: Option<f64>,
    /// UE heading in degrees (0-360, if available)
    pub ue_heading_deg: Option<f64>,
}

/// 6G: AI-assisted condition evaluation parameters
#[derive(Debug, Clone)]
pub struct AiAssistedCondition {
    /// AI model ID used for evaluation
    pub model_id: String,
    /// Model version
    pub model_version: String,
    /// Feature vector for AI model input
    pub feature_vector: Vec<f64>,
    /// Decision threshold (0.0 to 1.0)
    pub decision_threshold: f64,
    /// Fallback to event-based if AI unavailable
    pub fallback_condition: Option<Box<ChoCondition>>,
}

/// CHO execution condition
#[derive(Debug, Clone)]
pub enum ChoCondition {
    /// Event A3 based condition
    EventA3(EventA3Condition),
    /// Event A5 based condition
    EventA5(EventA5Condition),
    /// Timer-based condition
    Timer(TimerBasedCondition),
    /// 6G: Predictive condition
    Predictive(PredictiveCondition),
    /// 6G: AI-assisted condition
    AiAssisted(AiAssistedCondition),
}

impl ChoCondition {
    /// Get the condition type
    pub fn condition_type(&self) -> ChoConditionType {
        match self {
            ChoCondition::EventA3(_) => ChoConditionType::EventA3,
            ChoCondition::EventA5(_) => ChoConditionType::EventA5,
            ChoCondition::Timer(_) => ChoConditionType::TimerBased,
            ChoCondition::Predictive(_) => ChoConditionType::Predictive,
            ChoCondition::AiAssisted(_) => ChoConditionType::AiAssisted,
        }
    }
}

/// Target cell configuration for CHO
#[derive(Debug, Clone)]
pub struct ChoTargetCellConfig {
    /// Physical Cell ID (0-1007)
    pub phys_cell_id: u16,
    /// SSB frequency in ARFCN
    pub ssb_frequency_arfcn: u32,
    /// SSB subcarrier spacing (15, 30, 120, or 240 kHz)
    pub ssb_subcarrier_spacing_khz: u16,
    /// NR Cell Identity (36 bits)
    pub nr_cell_identity: Option<u64>,
    /// PLMN Identity (3 bytes)
    pub plmn_identity: Option<[u8; 3]>,
    /// Reconfiguration data for the target cell (opaque)
    pub rrc_reconfiguration: Option<Vec<u8>>,
}

/// Conditional Handover configuration
///
/// Contains one or more candidate cells with their execution conditions.
/// When a condition is met, the UE performs handover to the corresponding target cell.
#[derive(Debug, Clone)]
pub struct ChoConfig {
    /// CHO configuration ID
    pub config_id: u8,
    /// List of candidate cells with conditions
    pub candidate_cells: Vec<ChoCandidateCell>,
    /// Maximum number of candidate cells the UE can maintain
    pub max_candidate_cells: Option<u8>,
    /// Whether to report CHO execution to the network
    pub report_cho_execution: bool,
    /// 6G: Enable predictive handover features
    pub predictive_ho_enabled: bool,
    /// 6G: AI model ID for assisted handover decisions
    pub ai_model_id: Option<String>,
}

/// A candidate cell for conditional handover
#[derive(Debug, Clone)]
pub struct ChoCandidateCell {
    /// Candidate cell index (unique within the CHO config)
    pub candidate_index: u8,
    /// Execution condition for this candidate
    pub condition: ChoCondition,
    /// Target cell configuration
    pub target_cell: ChoTargetCellConfig,
    /// Priority (lower = higher priority, used when multiple conditions are met)
    pub priority: u8,
}

/// Parsed CHO execution report data
#[derive(Debug, Clone)]
pub struct ChoExecutionReport {
    /// CHO config ID that was executed
    pub config_id: u8,
    /// Candidate cell index that was selected
    pub selected_candidate_index: u8,
    /// Physical Cell ID of the target cell
    pub target_phys_cell_id: u16,
    /// Condition type that triggered the execution
    pub triggered_condition_type: ChoConditionType,
    /// Timestamp of execution in ms
    pub execution_time_ms: u64,
    /// 6G: AI confidence score if AI-assisted
    pub ai_confidence: Option<f64>,
}

impl ChoConfig {
    /// Validate the CHO configuration
    pub fn validate(&self) -> Result<(), ConditionalHandoverError> {
        if self.candidate_cells.is_empty() {
            return Err(ConditionalHandoverError::MissingMandatoryField(
                "candidate_cells (at least one required)".to_string(),
            ));
        }

        if let Some(max) = self.max_candidate_cells {
            if self.candidate_cells.len() > max as usize {
                return Err(ConditionalHandoverError::InvalidConfig(format!(
                    "Number of candidate cells ({}) exceeds maximum ({})",
                    self.candidate_cells.len(),
                    max
                )));
            }
        }

        // Validate each candidate cell
        for candidate in &self.candidate_cells {
            candidate.validate()?;
        }

        // Check for duplicate candidate indices
        let mut seen_indices = std::collections::HashSet::new();
        for candidate in &self.candidate_cells {
            if !seen_indices.insert(candidate.candidate_index) {
                return Err(ConditionalHandoverError::InvalidConfig(format!(
                    "Duplicate candidate index: {}",
                    candidate.candidate_index
                )));
            }
        }

        Ok(())
    }
}

impl ChoCandidateCell {
    /// Validate the candidate cell configuration
    pub fn validate(&self) -> Result<(), ConditionalHandoverError> {
        if self.target_cell.phys_cell_id > 1007 {
            return Err(ConditionalHandoverError::InvalidConfig(format!(
                "PhysCellId {} exceeds maximum value 1007",
                self.target_cell.phys_cell_id
            )));
        }

        // Validate SSB subcarrier spacing
        match self.target_cell.ssb_subcarrier_spacing_khz {
            15 | 30 | 120 | 240 => {}
            other => {
                return Err(ConditionalHandoverError::InvalidConfig(format!(
                    "Invalid SSB subcarrier spacing: {other} kHz (must be 15, 30, 120, or 240)"
                )))
            }
        }

        // Validate NR Cell Identity (36 bits)
        if let Some(nci) = self.target_cell.nr_cell_identity {
            if nci >= (1u64 << 36) {
                return Err(ConditionalHandoverError::InvalidConfig(
                    "NR Cell Identity exceeds 36 bits".to_string(),
                ));
            }
        }

        // Validate condition-specific parameters
        match &self.condition {
            ChoCondition::Predictive(pred) => {
                if pred.confidence_level < 0.0 || pred.confidence_level > 1.0 {
                    return Err(ConditionalHandoverError::InvalidConfig(
                        "Confidence level must be in range [0.0, 1.0]".to_string(),
                    ));
                }
                if pred.min_confidence_threshold < 0.0 || pred.min_confidence_threshold > 1.0 {
                    return Err(ConditionalHandoverError::InvalidConfig(
                        "Minimum confidence threshold must be in range [0.0, 1.0]".to_string(),
                    ));
                }
            }
            ChoCondition::AiAssisted(ai) => {
                if ai.decision_threshold < 0.0 || ai.decision_threshold > 1.0 {
                    return Err(ConditionalHandoverError::InvalidConfig(
                        "Decision threshold must be in range [0.0, 1.0]".to_string(),
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }
}

/// Encode a CHO configuration to bytes (simplified serialization)
pub fn encode_cho_config(config: &ChoConfig) -> Result<Vec<u8>, ConditionalHandoverError> {
    config.validate()?;
    let mut bytes = Vec::with_capacity(64);

    // config_id (1 byte)
    bytes.push(config.config_id);
    // number of candidates (1 byte)
    bytes.push(config.candidate_cells.len() as u8);
    // flags (1 byte): bit 0 = report_cho_execution, bit 1 = predictive_ho_enabled
    let mut flags: u8 = 0;
    if config.report_cho_execution {
        flags |= 0x01;
    }
    if config.predictive_ho_enabled {
        flags |= 0x02;
    }
    bytes.push(flags);

    // Encode each candidate cell (simplified: just condition type + phys_cell_id + priority)
    for candidate in &config.candidate_cells {
        bytes.push(candidate.candidate_index);
        bytes.push(match candidate.condition.condition_type() {
            ChoConditionType::EventA3 => 0,
            ChoConditionType::EventA5 => 1,
            ChoConditionType::TimerBased => 2,
            ChoConditionType::Predictive => 3,
            ChoConditionType::AiAssisted => 4,
        });
        bytes.extend_from_slice(&candidate.target_cell.phys_cell_id.to_be_bytes());
        bytes.push(candidate.priority);
    }

    Ok(bytes)
}

/// Decode a CHO configuration from bytes (simplified deserialization)
///
/// Note: This only decodes the simplified header fields. Full condition
/// parameters require the complete RRC reconfiguration message.
pub fn decode_cho_config_header(bytes: &[u8]) -> Result<(u8, u8, bool, bool), ConditionalHandoverError> {
    if bytes.len() < 3 {
        return Err(ConditionalHandoverError::CodecError(
            "Insufficient bytes for CHO config header".to_string(),
        ));
    }
    let config_id = bytes[0];
    let num_candidates = bytes[1];
    let report_cho_execution = (bytes[2] & 0x01) != 0;
    let predictive_ho_enabled = (bytes[2] & 0x02) != 0;

    Ok((config_id, num_candidates, report_cho_execution, predictive_ho_enabled))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_a3_condition() -> ChoCondition {
        ChoCondition::EventA3(EventA3Condition {
            a3_offset: A3Offset::new(3.0).unwrap(),
            hysteresis: Hysteresis::new(1.0).unwrap(),
            time_to_trigger: TimeToTrigger::Ms256,
            use_rsrp: true,
        })
    }

    fn create_test_a5_condition() -> ChoCondition {
        ChoCondition::EventA5(EventA5Condition {
            threshold1: RsrpThreshold::new(-110).unwrap(),
            threshold2: RsrpThreshold::new(-100).unwrap(),
            hysteresis: Hysteresis::new(2.0).unwrap(),
            time_to_trigger: TimeToTrigger::Ms320,
        })
    }

    fn create_test_target_cell(pci: u16) -> ChoTargetCellConfig {
        ChoTargetCellConfig {
            phys_cell_id: pci,
            ssb_frequency_arfcn: 620000,
            ssb_subcarrier_spacing_khz: 30,
            nr_cell_identity: Some(0x123456789),
            plmn_identity: Some([0x00, 0xF1, 0x10]),
            rrc_reconfiguration: None,
        }
    }

    fn create_test_cho_config() -> ChoConfig {
        ChoConfig {
            config_id: 1,
            candidate_cells: vec![
                ChoCandidateCell {
                    candidate_index: 0,
                    condition: create_test_a3_condition(),
                    target_cell: create_test_target_cell(100),
                    priority: 0,
                },
                ChoCandidateCell {
                    candidate_index: 1,
                    condition: create_test_a5_condition(),
                    target_cell: create_test_target_cell(200),
                    priority: 1,
                },
            ],
            max_candidate_cells: Some(4),
            report_cho_execution: true,
            predictive_ho_enabled: false,
            ai_model_id: None,
        }
    }

    #[test]
    fn test_cho_config_validate() {
        let config = create_test_cho_config();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cho_config_empty_candidates() {
        let config = ChoConfig {
            config_id: 1,
            candidate_cells: vec![],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cho_config_exceeds_max_candidates() {
        let config = ChoConfig {
            config_id: 1,
            candidate_cells: vec![
                ChoCandidateCell {
                    candidate_index: 0,
                    condition: create_test_a3_condition(),
                    target_cell: create_test_target_cell(100),
                    priority: 0,
                },
                ChoCandidateCell {
                    candidate_index: 1,
                    condition: create_test_a3_condition(),
                    target_cell: create_test_target_cell(200),
                    priority: 1,
                },
            ],
            max_candidate_cells: Some(1),
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cho_config_duplicate_indices() {
        let config = ChoConfig {
            config_id: 1,
            candidate_cells: vec![
                ChoCandidateCell {
                    candidate_index: 0,
                    condition: create_test_a3_condition(),
                    target_cell: create_test_target_cell(100),
                    priority: 0,
                },
                ChoCandidateCell {
                    candidate_index: 0, // duplicate!
                    condition: create_test_a5_condition(),
                    target_cell: create_test_target_cell(200),
                    priority: 1,
                },
            ],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cho_invalid_phys_cell_id() {
        let mut cell = ChoCandidateCell {
            candidate_index: 0,
            condition: create_test_a3_condition(),
            target_cell: create_test_target_cell(1008), // invalid
            priority: 0,
        };
        assert!(cell.validate().is_err());

        cell.target_cell.phys_cell_id = 1007;
        assert!(cell.validate().is_ok());
    }

    #[test]
    fn test_cho_invalid_ssb_scs() {
        let mut cell = ChoCandidateCell {
            candidate_index: 0,
            condition: create_test_a3_condition(),
            target_cell: create_test_target_cell(100),
            priority: 0,
        };
        cell.target_cell.ssb_subcarrier_spacing_khz = 60; // invalid
        assert!(cell.validate().is_err());
    }

    #[test]
    fn test_cho_condition_types() {
        let a3 = create_test_a3_condition();
        assert_eq!(a3.condition_type(), ChoConditionType::EventA3);

        let a5 = create_test_a5_condition();
        assert_eq!(a5.condition_type(), ChoConditionType::EventA5);

        let timer = ChoCondition::Timer(TimerBasedCondition {
            timer_ms: 5000,
            restart_on_improvement: true,
        });
        assert_eq!(timer.condition_type(), ChoConditionType::TimerBased);
    }

    #[test]
    fn test_cho_predictive_condition() {
        let config = ChoConfig {
            config_id: 2,
            candidate_cells: vec![ChoCandidateCell {
                candidate_index: 0,
                condition: ChoCondition::Predictive(PredictiveCondition {
                    predicted_handover_time_ms: 5000,
                    confidence_level: 0.85,
                    min_confidence_threshold: 0.7,
                    ue_speed_ms: Some(30.0),
                    ue_heading_deg: Some(90.0),
                }),
                target_cell: create_test_target_cell(300),
                priority: 0,
            }],
            max_candidate_cells: None,
            report_cho_execution: true,
            predictive_ho_enabled: true,
            ai_model_id: None,
        };
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cho_predictive_invalid_confidence() {
        let cell = ChoCandidateCell {
            candidate_index: 0,
            condition: ChoCondition::Predictive(PredictiveCondition {
                predicted_handover_time_ms: 5000,
                confidence_level: 1.5, // invalid
                min_confidence_threshold: 0.7,
                ue_speed_ms: None,
                ue_heading_deg: None,
            }),
            target_cell: create_test_target_cell(300),
            priority: 0,
        };
        assert!(cell.validate().is_err());
    }

    #[test]
    fn test_cho_ai_assisted_condition() {
        let cell = ChoCandidateCell {
            candidate_index: 0,
            condition: ChoCondition::AiAssisted(AiAssistedCondition {
                model_id: "beam-predict-v3".to_string(),
                model_version: "1.0.0".to_string(),
                feature_vector: vec![0.5, 0.3, -0.1, 0.8],
                decision_threshold: 0.75,
                fallback_condition: Some(Box::new(create_test_a3_condition())),
            }),
            target_cell: create_test_target_cell(400),
            priority: 0,
        };
        assert!(cell.validate().is_ok());
    }

    #[test]
    fn test_hysteresis_valid() {
        assert!(Hysteresis::new(0.0).is_ok());
        assert!(Hysteresis::new(15.0).is_ok());
        assert!(Hysteresis::new(30.0).is_ok());
    }

    #[test]
    fn test_hysteresis_invalid() {
        assert!(Hysteresis::new(-1.0).is_err());
        assert!(Hysteresis::new(31.0).is_err());
    }

    #[test]
    fn test_rsrp_threshold_valid() {
        assert!(RsrpThreshold::new(-156).is_ok());
        assert!(RsrpThreshold::new(-100).is_ok());
        assert!(RsrpThreshold::new(-31).is_ok());
    }

    #[test]
    fn test_rsrp_threshold_invalid() {
        assert!(RsrpThreshold::new(-157).is_err());
        assert!(RsrpThreshold::new(-30).is_err());
    }

    #[test]
    fn test_a3_offset_valid() {
        assert!(A3Offset::new(-30.0).is_ok());
        assert!(A3Offset::new(0.0).is_ok());
        assert!(A3Offset::new(30.0).is_ok());
    }

    #[test]
    fn test_a3_offset_invalid() {
        assert!(A3Offset::new(-31.0).is_err());
        assert!(A3Offset::new(31.0).is_err());
    }

    #[test]
    fn test_time_to_trigger_values() {
        assert_eq!(TimeToTrigger::Ms0.to_ms(), 0);
        assert_eq!(TimeToTrigger::Ms256.to_ms(), 256);
        assert_eq!(TimeToTrigger::Ms5120.to_ms(), 5120);
    }

    #[test]
    fn test_encode_decode_cho_config() {
        let config = create_test_cho_config();
        let encoded = encode_cho_config(&config).expect("Failed to encode");
        assert!(!encoded.is_empty());

        let (config_id, num_candidates, report, predictive) =
            decode_cho_config_header(&encoded).expect("Failed to decode");
        assert_eq!(config_id, 1);
        assert_eq!(num_candidates, 2);
        assert!(report);
        assert!(!predictive);
    }

    #[test]
    fn test_cho_execution_report() {
        let report = ChoExecutionReport {
            config_id: 1,
            selected_candidate_index: 0,
            target_phys_cell_id: 100,
            triggered_condition_type: ChoConditionType::EventA3,
            execution_time_ms: 1700000000000,
            ai_confidence: None,
        };
        assert_eq!(report.config_id, 1);
        assert_eq!(report.triggered_condition_type, ChoConditionType::EventA3);
    }

    #[test]
    fn test_rsrq_threshold_valid() {
        assert!(RsrqThreshold::new(-43.0).is_ok());
        assert!(RsrqThreshold::new(0.0).is_ok());
        assert!(RsrqThreshold::new(20.0).is_ok());
    }

    #[test]
    fn test_rsrq_threshold_invalid() {
        assert!(RsrqThreshold::new(-44.0).is_err());
        assert!(RsrqThreshold::new(21.0).is_err());
    }
}
