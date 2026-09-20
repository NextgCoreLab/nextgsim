//! Conditional Handover (CHO) Configuration
//!
//! Conditional reconfiguration is TS 38.331 **Section 5.3.5.13** (§5.3.5.13.4
//! evaluation, §5.3.5.13.5 execution); §5.3.5.8 is the reconfiguration-failure
//! clause and was a misattribution. Rel-16 defines the execution conditions
//! `condEventA3` and `condEventA5`; this module adds three non-normative
//! research conditions (timer-based, predictive, AI-assisted) on top.
//!
//! This module implements:
//! - `ChoConfig` - Conditional Handover configuration with conditions and target cell configs
//! - Condition types: event-based (A3, A5), timer-based, and 6G predictive
//! - The container codec: [`encode_cho_config`] / [`decode_cho_config`]
//!
//! The container is the simulator's own byte format, not UPER
//! `ConditionalReconfiguration`. That IE arrived in Rel-16 and was unreachable
//! while the vendored schema was Rel-15; #105 has since upgraded the schema to
//! Rel-19, so `ConditionalReconfiguration` now exists in the generated tree and
//! this container is a migration candidate rather than a necessity. Migrating it
//! is separate work: it changes a wire format both ends read, and the
//! research-only condition types above (timer-based, predictive, AI-assisted) have
//! no normative home, so the container cannot simply be deleted. The UE-side
//! runtime that stores candidates and evaluates their execution conditions lives
//! in `nextgsim-ue/src/rrc/conditional_handover.rs`.

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
    /// The wire code for this value: its index in the `TimeToTrigger` ENUMERATED
    /// of TS 38.331 (`timeToTrigger`), which is the order declared above.
    pub fn to_code(&self) -> u8 {
        match self {
            TimeToTrigger::Ms0 => 0,
            TimeToTrigger::Ms40 => 1,
            TimeToTrigger::Ms64 => 2,
            TimeToTrigger::Ms80 => 3,
            TimeToTrigger::Ms100 => 4,
            TimeToTrigger::Ms128 => 5,
            TimeToTrigger::Ms160 => 6,
            TimeToTrigger::Ms256 => 7,
            TimeToTrigger::Ms320 => 8,
            TimeToTrigger::Ms480 => 9,
            TimeToTrigger::Ms512 => 10,
            TimeToTrigger::Ms640 => 11,
            TimeToTrigger::Ms1024 => 12,
            TimeToTrigger::Ms1280 => 13,
            TimeToTrigger::Ms2560 => 14,
            TimeToTrigger::Ms5120 => 15,
        }
    }

    /// The value for a wire code, or `None` if the code is not one of the 16
    /// `timeToTrigger` values.
    pub fn from_code(code: u8) -> Option<Self> {
        Some(match code {
            0 => TimeToTrigger::Ms0,
            1 => TimeToTrigger::Ms40,
            2 => TimeToTrigger::Ms64,
            3 => TimeToTrigger::Ms80,
            4 => TimeToTrigger::Ms100,
            5 => TimeToTrigger::Ms128,
            6 => TimeToTrigger::Ms160,
            7 => TimeToTrigger::Ms256,
            8 => TimeToTrigger::Ms320,
            9 => TimeToTrigger::Ms480,
            10 => TimeToTrigger::Ms512,
            11 => TimeToTrigger::Ms640,
            12 => TimeToTrigger::Ms1024,
            13 => TimeToTrigger::Ms1280,
            14 => TimeToTrigger::Ms2560,
            15 => TimeToTrigger::Ms5120,
            _ => return None,
        })
    }

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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
#[derive(Debug, Clone, PartialEq)]
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
            ChoCondition::AiAssisted(ai)
                if (ai.decision_threshold < 0.0 || ai.decision_threshold > 1.0) =>
            {
                return Err(ConditionalHandoverError::InvalidConfig(
                    "Decision threshold must be in range [0.0, 1.0]".to_string(),
                ));
            }
            _ => {}
        }

        Ok(())
    }
}

/// A dB value on the RRC 0.5 dB grid, as the half-dB integer the wire carries.
///
/// `a3-Offset` and `hysteresis` are `INTEGER (0..30)` in units of 0.5 dB in
/// TS 38.331, so a value off that grid cannot be signalled and is rounded to it.
fn to_half_db(db: f64) -> i16 {
    (db * 2.0).round() as i16
}

/// The dB value a half-dB wire integer denotes.
fn from_half_db(half_db: i16) -> f64 {
    f64::from(half_db) / 2.0
}

/// Encode a CHO configuration to bytes.
///
/// # Wire layout
///
/// This is the simulator's own container, not UPER `ConditionalReconfiguration`.
/// That IE needed a Rel-16 schema, which #105 has now supplied (Rel-19), so the
/// container is retained for compatibility and for the non-normative condition
/// types rather than out of necessity — see the module docs.
/// The three-byte header is unchanged from the original encoder, so
/// [`decode_cho_config_header`] reads any version of this container:
///
/// ```text
/// [0]      config_id
/// [1]      candidate count
/// [2]      flags: bit 0 report_cho_execution, bit 1 predictive_ho_enabled
/// then per candidate:
/// [0]      candidate_index
/// [1]      condition type code (0 = A3, 1 = A5, 2 = timer)
/// [2..4]   phys_cell_id (big endian)
/// [4]      priority
/// [5..7]   detail length (big endian) -- condition detail then target detail
/// [7..]    detail
/// ```
///
/// The per-candidate detail block is what the original encoder dropped: it wrote
/// only index, condition type, PCI and priority, so a decoder could not
/// reconstruct a candidate's execution condition or target configuration. The
/// UE-side conditional-reconfiguration runtime (`nextgsim-ue`, issue #20) needs
/// both, hence the completion. The length prefix keeps a decoder able to skip a
/// candidate whose detail it does not understand.
///
/// # Conditions that cannot be carried
///
/// [`ChoCondition::Predictive`] and [`ChoCondition::AiAssisted`] are this
/// simulator's research extensions (model identifiers, feature vectors, a boxed
/// fallback condition); no wire encoding is defined for them, so encoding one is
/// an error rather than a silent loss of the parameters.
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

        let mut detail = encode_condition(&candidate.condition)?;
        detail.extend_from_slice(&encode_target_cell(&candidate.target_cell));
        let detail_len = u16::try_from(detail.len()).map_err(|_| {
            ConditionalHandoverError::CodecError(format!(
                "candidate {} detail is {} bytes, which does not fit the 16-bit length prefix",
                candidate.candidate_index,
                detail.len()
            ))
        })?;
        bytes.extend_from_slice(&detail_len.to_be_bytes());
        bytes.extend_from_slice(&detail);
    }

    Ok(bytes)
}

/// Encode an execution condition's parameters.
fn encode_condition(condition: &ChoCondition) -> Result<Vec<u8>, ConditionalHandoverError> {
    let mut out = Vec::with_capacity(8);
    match condition {
        ChoCondition::EventA3(a3) => {
            out.extend_from_slice(&to_half_db(a3.a3_offset.0).to_be_bytes());
            out.push(to_half_db(a3.hysteresis.0) as u8);
            out.push(a3.time_to_trigger.to_code());
            out.push(u8::from(a3.use_rsrp));
        }
        ChoCondition::EventA5(a5) => {
            out.extend_from_slice(&a5.threshold1.0.to_be_bytes());
            out.extend_from_slice(&a5.threshold2.0.to_be_bytes());
            out.push(to_half_db(a5.hysteresis.0) as u8);
            out.push(a5.time_to_trigger.to_code());
        }
        ChoCondition::Timer(timer) => {
            out.extend_from_slice(&timer.timer_ms.to_be_bytes());
            out.push(u8::from(timer.restart_on_improvement));
        }
        ChoCondition::Predictive(_) | ChoCondition::AiAssisted(_) => {
            return Err(ConditionalHandoverError::CodecError(format!(
                "{:?} is a research extension with no wire encoding: its parameters \
                 (model identifiers, feature vectors, fallback condition) cannot be \
                 carried in the CHO container",
                condition.condition_type()
            )));
        }
    }
    Ok(out)
}

/// Encode a target cell configuration.
fn encode_target_cell(target: &ChoTargetCellConfig) -> Vec<u8> {
    let mut out = Vec::with_capacity(16);
    out.extend_from_slice(&target.ssb_frequency_arfcn.to_be_bytes());
    out.extend_from_slice(&target.ssb_subcarrier_spacing_khz.to_be_bytes());

    let mut flags: u8 = 0;
    if target.nr_cell_identity.is_some() {
        flags |= 0x01;
    }
    if target.plmn_identity.is_some() {
        flags |= 0x02;
    }
    if target.rrc_reconfiguration.is_some() {
        flags |= 0x04;
    }
    out.push(flags);

    if let Some(nci) = target.nr_cell_identity {
        // 36 bits in 5 octets, high nibble of the first octet unused
        out.extend_from_slice(&nci.to_be_bytes()[3..8]);
    }
    if let Some(plmn) = target.plmn_identity {
        out.extend_from_slice(&plmn);
    }
    if let Some(ref reconfig) = target.rrc_reconfiguration {
        // Truncating at 65535 is not reachable: an RRCReconfiguration that long
        // exceeds every PDCP SDU size the simulator uses.
        let len = u16::try_from(reconfig.len()).unwrap_or(u16::MAX);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&reconfig[..usize::from(len)]);
    }
    out
}

/// A cursor over a candidate's detail block that reports a short read rather
/// than panicking on a slice out of range.
struct DetailReader<'a> {
    bytes: &'a [u8],
    pos: usize,
    candidate_index: u8,
}

impl<'a> DetailReader<'a> {
    fn new(bytes: &'a [u8], candidate_index: u8) -> Self {
        Self {
            bytes,
            pos: 0,
            candidate_index,
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], ConditionalHandoverError> {
        if self.pos + n > self.bytes.len() {
            return Err(ConditionalHandoverError::CodecError(format!(
                "candidate {}: detail block ends after {} bytes, {} more needed",
                self.candidate_index,
                self.bytes.len(),
                self.pos + n - self.bytes.len()
            )));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, ConditionalHandoverError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ConditionalHandoverError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn i16(&mut self) -> Result<i16, ConditionalHandoverError> {
        Ok(self.u16()? as i16)
    }

    fn u32(&mut self) -> Result<u32, ConditionalHandoverError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Decode a CHO configuration from bytes.
///
/// The inverse of [`encode_cho_config`], including each candidate's execution
/// condition and target cell configuration. The container is the simulator's own
/// (see that function for the layout and for why the two research conditions
/// cannot be carried).
///
/// Values on the RRC 0.5 dB grid come back quantised: an `a3-Offset` of 3.3 dB
/// encodes as 3.5 dB, because that is the resolution `INTEGER (0..30)` in units
/// of 0.5 dB has.
pub fn decode_cho_config(bytes: &[u8]) -> Result<ChoConfig, ConditionalHandoverError> {
    let (config_id, num_candidates, report_cho_execution, predictive_ho_enabled) =
        decode_cho_config_header(bytes)?;

    let mut pos = 3;
    let mut candidate_cells = Vec::with_capacity(usize::from(num_candidates));

    for nth in 0..num_candidates {
        if pos + 7 > bytes.len() {
            return Err(ConditionalHandoverError::CodecError(format!(
                "candidate {nth} of {num_candidates}: header needs 7 bytes, {} available",
                bytes.len().saturating_sub(pos)
            )));
        }
        let candidate_index = bytes[pos];
        let condition_code = bytes[pos + 1];
        let phys_cell_id = u16::from_be_bytes([bytes[pos + 2], bytes[pos + 3]]);
        let priority = bytes[pos + 4];
        let detail_len = usize::from(u16::from_be_bytes([bytes[pos + 5], bytes[pos + 6]]));
        pos += 7;

        if pos + detail_len > bytes.len() {
            return Err(ConditionalHandoverError::CodecError(format!(
                "candidate {candidate_index}: detail claims {detail_len} bytes, {} available",
                bytes.len() - pos
            )));
        }
        let mut reader = DetailReader::new(&bytes[pos..pos + detail_len], candidate_index);
        pos += detail_len;

        let condition = decode_condition(condition_code, candidate_index, &mut reader)?;
        let target_cell = decode_target_cell(phys_cell_id, &mut reader)?;

        candidate_cells.push(ChoCandidateCell {
            candidate_index,
            condition,
            target_cell,
            priority,
        });
    }

    let config = ChoConfig {
        config_id,
        candidate_cells,
        // The container carries no maximum: the count itself bounds the list, and
        // an absent maximum is what `validate` treats as unbounded.
        max_candidate_cells: None,
        report_cho_execution,
        predictive_ho_enabled,
        // ai_model_id is a String with no encoding in this container; a decoded
        // config therefore never claims an AI model.
        ai_model_id: None,
    };
    config.validate()?;
    Ok(config)
}

/// Decode an execution condition from its type code and parameter bytes.
fn decode_condition(
    code: u8,
    candidate_index: u8,
    reader: &mut DetailReader<'_>,
) -> Result<ChoCondition, ConditionalHandoverError> {
    let time_to_trigger = |code: u8| {
        TimeToTrigger::from_code(code).ok_or_else(|| {
            ConditionalHandoverError::CodecError(format!(
                "candidate {candidate_index}: {code} is not a timeToTrigger value"
            ))
        })
    };

    match code {
        0 => {
            let a3_offset = A3Offset::new(from_half_db(reader.i16()?))?;
            let hysteresis = Hysteresis::new(from_half_db(i16::from(reader.u8()?)))?;
            let ttt = time_to_trigger(reader.u8()?)?;
            let use_rsrp = reader.u8()? != 0;
            Ok(ChoCondition::EventA3(EventA3Condition {
                a3_offset,
                hysteresis,
                time_to_trigger: ttt,
                use_rsrp,
            }))
        }
        1 => {
            let threshold1 = RsrpThreshold::new(reader.i16()?)?;
            let threshold2 = RsrpThreshold::new(reader.i16()?)?;
            let hysteresis = Hysteresis::new(from_half_db(i16::from(reader.u8()?)))?;
            let ttt = time_to_trigger(reader.u8()?)?;
            Ok(ChoCondition::EventA5(EventA5Condition {
                threshold1,
                threshold2,
                hysteresis,
                time_to_trigger: ttt,
            }))
        }
        2 => Ok(ChoCondition::Timer(TimerBasedCondition {
            timer_ms: reader.u32()?,
            restart_on_improvement: reader.u8()? != 0,
        })),
        3 | 4 => Err(ConditionalHandoverError::CodecError(format!(
            "candidate {candidate_index}: condition type {code} (predictive / AI-assisted) \
             has no wire encoding, so no container can carry one"
        ))),
        other => Err(ConditionalHandoverError::CodecError(format!(
            "candidate {candidate_index}: unknown condition type {other}"
        ))),
    }
}

/// Decode a target cell configuration; `phys_cell_id` comes from the candidate
/// header rather than the detail block.
fn decode_target_cell(
    phys_cell_id: u16,
    reader: &mut DetailReader<'_>,
) -> Result<ChoTargetCellConfig, ConditionalHandoverError> {
    let ssb_frequency_arfcn = reader.u32()?;
    let ssb_subcarrier_spacing_khz = reader.u16()?;
    let flags = reader.u8()?;

    let nr_cell_identity = if flags & 0x01 != 0 {
        let b = reader.take(5)?;
        Some(u64::from_be_bytes([0, 0, 0, b[0], b[1], b[2], b[3], b[4]]))
    } else {
        None
    };
    let plmn_identity = if flags & 0x02 != 0 {
        let b = reader.take(3)?;
        Some([b[0], b[1], b[2]])
    } else {
        None
    };
    let rrc_reconfiguration = if flags & 0x04 != 0 {
        let len = usize::from(reader.u16()?);
        Some(reader.take(len)?.to_vec())
    } else {
        None
    };

    Ok(ChoTargetCellConfig {
        phys_cell_id,
        ssb_frequency_arfcn,
        ssb_subcarrier_spacing_khz,
        nr_cell_identity,
        plmn_identity,
        rrc_reconfiguration,
    })
}

/// Decode only the header of a CHO configuration
///
/// Reads `config_id`, the candidate count and the two flags — the three bytes
/// every version of this container starts with. Use [`decode_cho_config`] for
/// the candidates themselves.
pub fn decode_cho_config_header(
    bytes: &[u8],
) -> Result<(u8, u8, bool, bool), ConditionalHandoverError> {
    if bytes.len() < 3 {
        return Err(ConditionalHandoverError::CodecError(
            "Insufficient bytes for CHO config header".to_string(),
        ));
    }
    let config_id = bytes[0];
    let num_candidates = bytes[1];
    let report_cho_execution = (bytes[2] & 0x01) != 0;
    let predictive_ho_enabled = (bytes[2] & 0x02) != 0;

    Ok((
        config_id,
        num_candidates,
        report_cho_execution,
        predictive_ho_enabled,
    ))
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

    /// The completed container round-trips every candidate: conditions and
    /// target cells, not just the header. This is what the UE-side runtime
    /// (issue #20) needs, and what the original encoder dropped.
    #[test]
    fn a_cho_config_survives_a_container_round_trip() {
        let config = create_test_cho_config();
        let encoded = encode_cho_config(&config).expect("encode");
        let decoded = decode_cho_config(&encoded).expect("decode");

        assert_eq!(decoded.config_id, config.config_id);
        assert_eq!(decoded.report_cho_execution, config.report_cho_execution);
        assert_eq!(decoded.predictive_ho_enabled, config.predictive_ho_enabled);
        assert_eq!(decoded.candidate_cells, config.candidate_cells);
        // The container carries neither the maximum nor the AI model name.
        assert_eq!(decoded.max_candidate_cells, None);
        assert_eq!(decoded.ai_model_id, None);
    }

    #[test]
    fn a_timer_condition_and_an_opaque_reconfiguration_round_trip() {
        let config = ChoConfig {
            config_id: 9,
            candidate_cells: vec![ChoCandidateCell {
                candidate_index: 3,
                condition: ChoCondition::Timer(TimerBasedCondition {
                    timer_ms: 5000,
                    restart_on_improvement: true,
                }),
                target_cell: ChoTargetCellConfig {
                    phys_cell_id: 1007,
                    ssb_frequency_arfcn: 632628,
                    ssb_subcarrier_spacing_khz: 120,
                    nr_cell_identity: Some((1u64 << 36) - 1),
                    plmn_identity: None,
                    rrc_reconfiguration: Some(vec![0xDE, 0xAD, 0xBE, 0xEF]),
                },
                priority: 7,
            }],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: true,
            ai_model_id: None,
        };

        let decoded =
            decode_cho_config(&encode_cho_config(&config).expect("encode")).expect("decode");
        assert_eq!(decoded.candidate_cells, config.candidate_cells);
        assert!(decoded.predictive_ho_enabled);
    }

    /// dB values are quantised onto the 0.5 dB grid `INTEGER (0..30)` in units of
    /// 0.5 dB gives, so an off-grid offset comes back rounded rather than exact.
    #[test]
    fn an_off_grid_offset_comes_back_quantised_to_half_a_db() {
        let config = ChoConfig {
            config_id: 1,
            candidate_cells: vec![ChoCandidateCell {
                candidate_index: 0,
                condition: ChoCondition::EventA3(EventA3Condition {
                    a3_offset: A3Offset::new(3.3).unwrap(),
                    hysteresis: Hysteresis::new(1.2).unwrap(),
                    time_to_trigger: TimeToTrigger::Ms640,
                    use_rsrp: true,
                }),
                target_cell: create_test_target_cell(100),
                priority: 0,
            }],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: false,
            ai_model_id: None,
        };

        let decoded =
            decode_cho_config(&encode_cho_config(&config).expect("encode")).expect("decode");
        match &decoded.candidate_cells[0].condition {
            ChoCondition::EventA3(a3) => {
                assert_eq!(a3.a3_offset, A3Offset(3.5));
                assert_eq!(a3.hysteresis, Hysteresis(1.0));
                assert_eq!(a3.time_to_trigger, TimeToTrigger::Ms640);
            }
            other => panic!("expected an A3 condition, got {other:?}"),
        }
    }

    /// The header layout is unchanged by the completion, so the header decoder
    /// still reads a container written by the extended encoder.
    #[test]
    fn the_header_decoder_still_reads_a_completed_container() {
        let encoded = encode_cho_config(&create_test_cho_config()).expect("encode");
        assert_eq!(encoded[0], 1, "config_id is still the first byte");
        assert_eq!(encoded[1], 2, "candidate count is still the second");

        let (config_id, num_candidates, report, predictive) =
            decode_cho_config_header(&encoded).expect("decode header");
        assert_eq!((config_id, num_candidates), (1, 2));
        assert!(report);
        assert!(!predictive);
    }

    /// The two research conditions carry model identifiers and feature vectors
    /// with no wire encoding, so encoding one fails loudly instead of silently
    /// dropping the parameters (which is what the original encoder did).
    #[test]
    fn a_research_condition_cannot_be_encoded() {
        let config = ChoConfig {
            config_id: 2,
            candidate_cells: vec![ChoCandidateCell {
                candidate_index: 0,
                condition: ChoCondition::Predictive(PredictiveCondition {
                    predicted_handover_time_ms: 5000,
                    confidence_level: 0.85,
                    min_confidence_threshold: 0.7,
                    ue_speed_ms: None,
                    ue_heading_deg: None,
                }),
                target_cell: create_test_target_cell(300),
                priority: 0,
            }],
            max_candidate_cells: None,
            report_cho_execution: false,
            predictive_ho_enabled: true,
            ai_model_id: None,
        };

        let err = encode_cho_config(&config).expect_err("predictive must not encode");
        assert!(
            format!("{err}").contains("no wire encoding"),
            "the error must say why: {err}"
        );
    }

    #[test]
    fn a_truncated_container_is_rejected_rather_than_panicking() {
        let encoded = encode_cho_config(&create_test_cho_config()).expect("encode");

        for cut in 3..encoded.len() {
            assert!(
                decode_cho_config(&encoded[..cut]).is_err(),
                "a container cut to {cut} of {} bytes must not decode",
                encoded.len()
            );
        }
        assert!(
            decode_cho_config(&encoded).is_ok(),
            "the whole container does"
        );
    }

    /// A candidate whose detail length runs past the container is rejected with
    /// the mismatch named, rather than being read against a clamped slice and
    /// failing later with a less useful message.
    #[test]
    fn a_detail_length_past_the_end_of_the_container_is_named() {
        let mut encoded = encode_cho_config(&create_test_cho_config()).expect("encode");
        // The first candidate's detail length prefix sits at 3 + 5.
        encoded[8] = 0xFF;
        encoded[9] = 0xFF;

        let err = decode_cho_config(&encoded).expect_err("an over-long detail must not decode");
        let text = format!("{err}");
        assert!(
            text.contains("detail claims 65535 bytes"),
            "the error must name the mismatch: {text}"
        );
    }

    /// The length prefix is what lets an older decoder read a container a newer
    /// encoder wrote: detail bytes beyond the ones it knows are skipped, and the
    /// next candidate is still found.
    #[test]
    fn detail_bytes_a_decoder_does_not_know_are_skipped() {
        let original = create_test_cho_config();
        let encoded = encode_cho_config(&original).expect("encode");

        // Grow the first candidate's detail by three bytes of "future" fields.
        let detail_len_at = 3 + 5;
        let detail_len = usize::from(u16::from_be_bytes([encoded[8], encoded[9]]));
        let detail_end = detail_len_at + 2 + detail_len;
        let mut grown = encoded[..detail_end].to_vec();
        grown.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        grown.extend_from_slice(&encoded[detail_end..]);
        let grown_len = u16::try_from(detail_len + 3).unwrap().to_be_bytes();
        grown[detail_len_at] = grown_len[0];
        grown[detail_len_at + 1] = grown_len[1];

        let decoded = decode_cho_config(&grown).expect("decode a grown container");
        assert_eq!(
            decoded.candidate_cells, original.candidate_cells,
            "both candidates decode, the unknown bytes are skipped"
        );
    }

    #[test]
    fn an_unknown_condition_type_is_rejected() {
        let mut encoded = encode_cho_config(&create_test_cho_config()).expect("encode");
        encoded[4] = 9; // the first candidate's condition type code
        let err = decode_cho_config(&encoded).expect_err("unknown type must not decode");
        assert!(format!("{err}").contains("unknown condition type"), "{err}");
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
