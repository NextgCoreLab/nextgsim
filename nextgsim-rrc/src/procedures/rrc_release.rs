//! RRC Release Procedure
//!
//! Implements the RRC Release procedure as defined in 3GPP TS 38.331 Section 5.3.8.
//! This procedure is used to release an RRC connection between the UE and the network.
//!
//! The procedure consists of one message:
//! 1. `RRCRelease` - gNB → UE: Network command to release the RRC connection

use crate::codec::generated::*;
use crate::codec::{decode_rrc, encode_rrc, RrcCodecError};
use thiserror::Error;

/// Errors that can occur during RRC Release procedures
#[derive(Debug, Error)]
pub enum RrcReleaseError {
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

    /// Invalid field value
    #[error("Invalid field value: {0}")]
    InvalidFieldValue(String),
}

/// Deprioritisation type for RRC Release
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprioritisationType {
    /// Deprioritise frequency
    Frequency,
    /// Deprioritise NR
    Nr,
}

/// Deprioritisation timer values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeprioritisationTimer {
    /// 5 minutes
    Min5,
    /// 10 minutes
    Min10,
    /// 15 minutes
    Min15,
    /// 30 minutes
    Min30,
}

/// Deprioritisation request for RRC Release
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeprioritisationReq {
    /// Type of deprioritisation
    pub deprioritisation_type: DeprioritisationType,
    /// Timer duration
    pub deprioritisation_timer: DeprioritisationTimer,
}

/// T320 timer values for cell reselection priorities (TS 38.331 §6.3.2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum T320Value {
    /// 5 minutes
    Min5,
    /// 10 minutes
    Min10,
    /// 20 minutes
    Min20,
    /// 30 minutes
    Min30,
    /// 60 minutes
    Min60,
    /// 120 minutes
    Min120,
    /// 180 minutes
    Min180,
}

/// Dedicated NR frequency reselection priority
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FreqPriorityNrParams {
    /// NR ARFCN of the carrier
    pub carrier_freq: u32,
    /// Cell reselection priority (0-7)
    pub priority: u8,
}

/// CellReselectionPriorities IE for RRC Release (TS 38.331 §6.3.2)
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct CellReselectionPrioritiesParams {
    /// Dedicated NR frequency priorities
    pub freq_priority_list_nr: Vec<FreqPriorityNrParams>,
    /// T320 validity timer (optional)
    pub t320: Option<T320Value>,
}

// ============================================================================
// RRC Release
// ============================================================================

/// Parameters for building an RRC Release message
#[derive(Debug, Clone)]
pub struct RrcReleaseParams {
    /// RRC Transaction Identifier (0-3)
    pub rrc_transaction_id: u8,
    /// Redirected carrier info (encoded as bytes, optional)
    pub redirected_carrier_info: Option<Vec<u8>>,
    /// Cell reselection priorities (optional)
    pub cell_reselection_priorities: Option<CellReselectionPrioritiesParams>,
    /// Suspend configuration (encoded as bytes, optional)
    pub suspend_config: Option<Vec<u8>>,
    /// Deprioritisation request (optional)
    pub deprioritisation_req: Option<DeprioritisationReq>,
    /// Wait time in seconds (optional, from v1540 extension)
    pub wait_time: Option<u8>,
}

/// Parsed RRC Release data
#[derive(Debug, Clone)]
pub struct RrcReleaseData {
    /// RRC Transaction Identifier
    pub rrc_transaction_id: u8,
    /// Redirected carrier info (raw bytes)
    pub redirected_carrier_info: Option<Vec<u8>>,
    /// Cell reselection priorities
    pub cell_reselection_priorities: Option<CellReselectionPrioritiesParams>,
    /// Suspend configuration (raw bytes)
    pub suspend_config: Option<Vec<u8>>,
    /// Deprioritisation request
    pub deprioritisation_req: Option<DeprioritisationReq>,
    /// Wait time in seconds
    pub wait_time: Option<u8>,
}

/// Build an RRC Release message
pub fn build_rrc_release(params: &RrcReleaseParams) -> Result<DL_DCCH_Message, RrcReleaseError> {
    if params.rrc_transaction_id > 3 {
        return Err(RrcReleaseError::InvalidFieldValue(
            "RRC Transaction ID must be 0-3".to_string(),
        ));
    }

    // Decode redirected carrier info if provided
    let redirected_carrier_info = if let Some(ref bytes) = params.redirected_carrier_info {
        Some(decode_rrc::<RedirectedCarrierInfo>(bytes)?)
    } else {
        None
    };

    // Decode suspend config if provided
    let suspend_config = if let Some(ref bytes) = params.suspend_config {
        Some(decode_rrc::<SuspendConfig>(bytes)?)
    } else {
        None
    };

    // Build deprioritisation request if provided
    let deprioritisation_req = params.deprioritisation_req.map(|req| {
        let deprioritisation_type = match req.deprioritisation_type {
            DeprioritisationType::Frequency => {
                RRCRelease_IEsDeprioritisationReqDeprioritisationType(
                    RRCRelease_IEsDeprioritisationReqDeprioritisationType::FREQUENCY,
                )
            }
            DeprioritisationType::Nr => RRCRelease_IEsDeprioritisationReqDeprioritisationType(
                RRCRelease_IEsDeprioritisationReqDeprioritisationType::NR,
            ),
        };

        let deprioritisation_timer = match req.deprioritisation_timer {
            DeprioritisationTimer::Min5 => RRCRelease_IEsDeprioritisationReqDeprioritisationTimer(
                RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN5,
            ),
            DeprioritisationTimer::Min10 => RRCRelease_IEsDeprioritisationReqDeprioritisationTimer(
                RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN10,
            ),
            DeprioritisationTimer::Min15 => RRCRelease_IEsDeprioritisationReqDeprioritisationTimer(
                RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN15,
            ),
            DeprioritisationTimer::Min30 => RRCRelease_IEsDeprioritisationReqDeprioritisationTimer(
                RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN30,
            ),
        };

        RRCRelease_IEsDeprioritisationReq {
            deprioritisation_type,
            deprioritisation_timer,
        }
    });

    // Build v1540 extension if wait_time is provided
    let non_critical_extension = params.wait_time.map(|wt| RRCRelease_v1540_IEs {
        wait_time: Some(RejectWaitTime(wt)),
        non_critical_extension: None,
    });

    // Build cell reselection priorities if provided
    let cell_reselection_priorities = params
        .cell_reselection_priorities
        .as_ref()
        .map(build_cell_reselection_priorities)
        .transpose()?;

    let rrc_release_ies = RRCRelease_IEs {
        redirected_carrier_info,
        cell_reselection_priorities,
        suspend_config,
        deprioritisation_req,
        late_non_critical_extension: None,
        non_critical_extension,
    };

    let rrc_release = RRCRelease {
        rrc_transaction_identifier: RRC_TransactionIdentifier(params.rrc_transaction_id),
        critical_extensions: RRCReleaseCriticalExtensions::RrcRelease(rrc_release_ies),
    };

    let message_type = DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcRelease(rrc_release));

    Ok(DL_DCCH_Message {
        message: message_type,
    })
}

/// Parse an RRC Release from a DL-DCCH message
pub fn parse_rrc_release(msg: &DL_DCCH_Message) -> Result<RrcReleaseData, RrcReleaseError> {
    let rrc_release = match &msg.message {
        DL_DCCH_MessageType::C1(c1) => match c1 {
            DL_DCCH_MessageType_c1::RrcRelease(release) => release,
            _ => {
                return Err(RrcReleaseError::InvalidMessageType {
                    expected: "RRCRelease".to_string(),
                    actual: "other c1 message".to_string(),
                })
            }
        },
        _ => {
            return Err(RrcReleaseError::InvalidMessageType {
                expected: "c1".to_string(),
                actual: "messageClassExtension".to_string(),
            })
        }
    };

    let ies = match &rrc_release.critical_extensions {
        RRCReleaseCriticalExtensions::RrcRelease(ies) => ies,
        RRCReleaseCriticalExtensions::CriticalExtensionsFuture(_) => {
            return Err(RrcReleaseError::InvalidMessageType {
                expected: "rrcRelease".to_string(),
                actual: "criticalExtensionsFuture".to_string(),
            })
        }
    };

    // Encode redirected carrier info back to bytes if present
    let redirected_carrier_info = if let Some(ref info) = ies.redirected_carrier_info {
        Some(encode_rrc(info)?)
    } else {
        None
    };

    // Encode suspend config back to bytes if present
    let suspend_config = if let Some(ref config) = ies.suspend_config {
        Some(encode_rrc(config)?)
    } else {
        None
    };

    // Parse deprioritisation request
    let deprioritisation_req = ies.deprioritisation_req.as_ref().map(|req| {
        let deprioritisation_type = match req.deprioritisation_type.0 {
            RRCRelease_IEsDeprioritisationReqDeprioritisationType::FREQUENCY => {
                DeprioritisationType::Frequency
            }
            RRCRelease_IEsDeprioritisationReqDeprioritisationType::NR => DeprioritisationType::Nr,
            _ => DeprioritisationType::Frequency, // Default fallback
        };

        let deprioritisation_timer = match req.deprioritisation_timer.0 {
            RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN5 => {
                DeprioritisationTimer::Min5
            }
            RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN10 => {
                DeprioritisationTimer::Min10
            }
            RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN15 => {
                DeprioritisationTimer::Min15
            }
            RRCRelease_IEsDeprioritisationReqDeprioritisationTimer::MIN30 => {
                DeprioritisationTimer::Min30
            }
            _ => DeprioritisationTimer::Min5, // Default fallback
        };

        DeprioritisationReq {
            deprioritisation_type,
            deprioritisation_timer,
        }
    });

    // Extract wait_time from v1540 extension
    let wait_time = ies
        .non_critical_extension
        .as_ref()
        .and_then(|ext| ext.wait_time.as_ref().map(|wt| wt.0));

    // Parse cell reselection priorities
    let cell_reselection_priorities = ies
        .cell_reselection_priorities
        .as_ref()
        .map(parse_cell_reselection_priorities);

    Ok(RrcReleaseData {
        rrc_transaction_id: rrc_release.rrc_transaction_identifier.0,
        redirected_carrier_info,
        cell_reselection_priorities,
        suspend_config,
        deprioritisation_req,
        wait_time,
    })
}

/// Build the generated `CellReselectionPriorities` from typed parameters
fn build_cell_reselection_priorities(
    params: &CellReselectionPrioritiesParams,
) -> Result<CellReselectionPriorities, RrcReleaseError> {
    for entry in &params.freq_priority_list_nr {
        if entry.priority > 7 {
            return Err(RrcReleaseError::InvalidFieldValue(
                "Cell reselection priority must be 0-7".to_string(),
            ));
        }
    }

    let freq_priority_list_nr = if params.freq_priority_list_nr.is_empty() {
        None
    } else {
        Some(FreqPriorityListNR(
            params
                .freq_priority_list_nr
                .iter()
                .map(|entry| FreqPriorityNR {
                    carrier_freq: ARFCN_ValueNR(entry.carrier_freq),
                    cell_reselection_priority: CellReselectionPriority(entry.priority),
                    cell_reselection_sub_priority: None,
                })
                .collect(),
        ))
    };

    let t320 = params.t320.map(|t| {
        CellReselectionPrioritiesT320(match t {
            T320Value::Min5 => CellReselectionPrioritiesT320::MIN5,
            T320Value::Min10 => CellReselectionPrioritiesT320::MIN10,
            T320Value::Min20 => CellReselectionPrioritiesT320::MIN20,
            T320Value::Min30 => CellReselectionPrioritiesT320::MIN30,
            T320Value::Min60 => CellReselectionPrioritiesT320::MIN60,
            T320Value::Min120 => CellReselectionPrioritiesT320::MIN120,
            T320Value::Min180 => CellReselectionPrioritiesT320::MIN180,
        })
    });

    Ok(CellReselectionPriorities {
        freq_priority_list_eutra: None,
        freq_priority_list_nr,
        t320,
    })
}

/// Parse the generated `CellReselectionPriorities` into typed parameters
fn parse_cell_reselection_priorities(
    priorities: &CellReselectionPriorities,
) -> CellReselectionPrioritiesParams {
    let freq_priority_list_nr = priorities
        .freq_priority_list_nr
        .as_ref()
        .map(|list| {
            list.0
                .iter()
                .map(|entry| FreqPriorityNrParams {
                    carrier_freq: entry.carrier_freq.0,
                    priority: entry.cell_reselection_priority.0,
                })
                .collect()
        })
        .unwrap_or_default();

    let t320 = priorities.t320.as_ref().and_then(|t| match t.0 {
        CellReselectionPrioritiesT320::MIN5 => Some(T320Value::Min5),
        CellReselectionPrioritiesT320::MIN10 => Some(T320Value::Min10),
        CellReselectionPrioritiesT320::MIN20 => Some(T320Value::Min20),
        CellReselectionPrioritiesT320::MIN30 => Some(T320Value::Min30),
        CellReselectionPrioritiesT320::MIN60 => Some(T320Value::Min60),
        CellReselectionPrioritiesT320::MIN120 => Some(T320Value::Min120),
        CellReselectionPrioritiesT320::MIN180 => Some(T320Value::Min180),
        _ => None,
    });

    CellReselectionPrioritiesParams {
        freq_priority_list_nr,
        t320,
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Build and encode an RRC Release to bytes
pub fn encode_rrc_release(params: &RrcReleaseParams) -> Result<Vec<u8>, RrcReleaseError> {
    let msg = build_rrc_release(params)?;
    Ok(encode_rrc(&msg)?)
}

/// Decode and parse an RRC Release from bytes
pub fn decode_rrc_release(bytes: &[u8]) -> Result<RrcReleaseData, RrcReleaseError> {
    let msg: DL_DCCH_Message = decode_rrc(bytes)?;
    parse_rrc_release(&msg)
}

/// Check if a DL-DCCH message is an RRC Release
pub fn is_rrc_release(msg: &DL_DCCH_Message) -> bool {
    matches!(
        &msg.message,
        DL_DCCH_MessageType::C1(DL_DCCH_MessageType_c1::RrcRelease(_))
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // RRC Release Tests
    // ========================================================================

    fn create_test_release_params() -> RrcReleaseParams {
        RrcReleaseParams {
            rrc_transaction_id: 0,
            cell_reselection_priorities: None,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: None,
            wait_time: None,
        }
    }

    #[test]
    fn test_build_rrc_release() {
        let params = create_test_release_params();
        let result = build_rrc_release(&params);
        assert!(result.is_ok());

        let msg = result.unwrap();
        assert!(is_rrc_release(&msg));
    }

    #[test]
    fn test_parse_rrc_release() {
        let params = create_test_release_params();
        let msg = build_rrc_release(&params).unwrap();
        let result = parse_rrc_release(&msg);
        assert!(result.is_ok());

        let data = result.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
        assert_eq!(data.deprioritisation_req, params.deprioritisation_req);
        assert_eq!(data.wait_time, params.wait_time);
    }

    #[test]
    fn test_rrc_release_with_deprioritisation() {
        let params = RrcReleaseParams {
            rrc_transaction_id: 1,
            cell_reselection_priorities: None,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: Some(DeprioritisationReq {
                deprioritisation_type: DeprioritisationType::Frequency,
                deprioritisation_timer: DeprioritisationTimer::Min15,
            }),
            wait_time: None,
        };

        let msg = build_rrc_release(&params).unwrap();
        let data = parse_rrc_release(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 1);
        assert!(data.deprioritisation_req.is_some());
        let req = data.deprioritisation_req.unwrap();
        assert_eq!(req.deprioritisation_type, DeprioritisationType::Frequency);
        assert_eq!(req.deprioritisation_timer, DeprioritisationTimer::Min15);
    }

    #[test]
    fn test_rrc_release_with_wait_time() {
        let params = RrcReleaseParams {
            rrc_transaction_id: 2,
            cell_reselection_priorities: None,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: None,
            wait_time: Some(16),
        };

        let msg = build_rrc_release(&params).unwrap();
        let data = parse_rrc_release(&msg).unwrap();

        assert_eq!(data.rrc_transaction_id, 2);
        assert_eq!(data.wait_time, Some(16));
    }

    #[test]
    fn test_encode_decode_rrc_release() {
        let params = create_test_release_params();
        let encoded = encode_rrc_release(&params);
        assert!(encoded.is_ok());

        let bytes = encoded.unwrap();
        assert!(!bytes.is_empty());

        let decoded = decode_rrc_release(&bytes);
        assert!(decoded.is_ok());

        let data = decoded.unwrap();
        assert_eq!(data.rrc_transaction_id, params.rrc_transaction_id);
    }

    #[test]
    fn test_invalid_rrc_transaction_id() {
        let params = RrcReleaseParams {
            rrc_transaction_id: 5, // Invalid: must be 0-3
            cell_reselection_priorities: None,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: None,
            wait_time: None,
        };

        let result = build_rrc_release(&params);
        assert!(result.is_err());
    }

    #[test]
    fn test_rrc_release_all_transaction_ids() {
        // Test all valid transaction IDs (0-3)
        for id in 0..=3 {
            let params = RrcReleaseParams {
                rrc_transaction_id: id,
                cell_reselection_priorities: None,
                redirected_carrier_info: None,
                suspend_config: None,
                deprioritisation_req: None,
                wait_time: None,
            };
            let msg = build_rrc_release(&params).unwrap();
            let data = parse_rrc_release(&msg).unwrap();
            assert_eq!(data.rrc_transaction_id, id);
        }
    }

    #[test]
    fn test_deprioritisation_type_nr() {
        let params = RrcReleaseParams {
            rrc_transaction_id: 0,
            cell_reselection_priorities: None,
            redirected_carrier_info: None,
            suspend_config: None,
            deprioritisation_req: Some(DeprioritisationReq {
                deprioritisation_type: DeprioritisationType::Nr,
                deprioritisation_timer: DeprioritisationTimer::Min30,
            }),
            wait_time: None,
        };

        let msg = build_rrc_release(&params).unwrap();
        let data = parse_rrc_release(&msg).unwrap();

        let req = data.deprioritisation_req.unwrap();
        assert_eq!(req.deprioritisation_type, DeprioritisationType::Nr);
        assert_eq!(req.deprioritisation_timer, DeprioritisationTimer::Min30);
    }

    #[test]
    fn test_all_deprioritisation_timers() {
        let timers = [
            DeprioritisationTimer::Min5,
            DeprioritisationTimer::Min10,
            DeprioritisationTimer::Min15,
            DeprioritisationTimer::Min30,
        ];

        for timer in timers {
            let params = RrcReleaseParams {
                rrc_transaction_id: 0,
                cell_reselection_priorities: None,
                redirected_carrier_info: None,
                suspend_config: None,
                deprioritisation_req: Some(DeprioritisationReq {
                    deprioritisation_type: DeprioritisationType::Frequency,
                    deprioritisation_timer: timer,
                }),
                wait_time: None,
            };

            let msg = build_rrc_release(&params).unwrap();
            let data = parse_rrc_release(&msg).unwrap();
            assert_eq!(
                data.deprioritisation_req.unwrap().deprioritisation_timer,
                timer
            );
        }
    }

    #[test]
    fn test_release_with_cell_reselection_priorities_roundtrip() {
        let priorities = CellReselectionPrioritiesParams {
            freq_priority_list_nr: vec![
                FreqPriorityNrParams {
                    carrier_freq: 632628,
                    priority: 5,
                },
                FreqPriorityNrParams {
                    carrier_freq: 423170,
                    priority: 3,
                },
            ],
            t320: Some(T320Value::Min10),
        };
        let params = RrcReleaseParams {
            cell_reselection_priorities: Some(priorities.clone()),
            ..create_test_release_params()
        };
        let msg = build_rrc_release(&params).unwrap();
        let data = parse_rrc_release(&msg).unwrap();
        assert_eq!(data.cell_reselection_priorities, Some(priorities));
    }

    #[test]
    fn test_cell_reselection_priorities_rejects_invalid_priority() {
        let params = RrcReleaseParams {
            cell_reselection_priorities: Some(CellReselectionPrioritiesParams {
                freq_priority_list_nr: vec![FreqPriorityNrParams {
                    carrier_freq: 1,
                    priority: 8, // out of range 0-7
                }],
                t320: None,
            }),
            ..create_test_release_params()
        };
        assert!(build_rrc_release(&params).is_err());
    }
}
