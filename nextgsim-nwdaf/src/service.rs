//! TS 23.288 NWDAF Service Operations
//!
//! Implements the three Nnwdaf service operations defined in 3GPP TS 23.288:
//! - **Nnwdaf_AnalyticsSubscription** (Section 7.2): Subscribe/unsubscribe to
//!   analytics with asynchronous callback notification
//! - **Nnwdaf_AnalyticsInfo** (Section 7.3): On-demand (request/response)
//!   analytics query
//! - **Nnwdaf_MLModelProvision** (Section 7.5): ML model distribution service

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::analytics_id::{AnalyticsId, AnalyticsOutputType, AnalyticsTarget, TimeWindow};
use crate::anlf::{AnalyticsResult, Anlf};
use crate::data_collection::DataCollector;
use crate::error::SubscriptionError;
use crate::mtlf::{MlModelInfo, ModelProvisionRequest, ModelProvisionResponse, Mtlf};

// ---------------------------------------------------------------------------
// Nnwdaf_AnalyticsSubscription (TS 23.288, Section 7.2)
// ---------------------------------------------------------------------------

/// Callback function type for analytics subscriptions
///
/// Called whenever new analytics matching the subscription parameters
/// become available. Receives the analytics result.
pub type AnalyticsCallback = Arc<dyn Fn(&AnalyticsResult) + Send + Sync>;

/// Parameters for an analytics subscription request
#[derive(Clone, Serialize, Deserialize)]
pub struct SubscriptionRequest {
    /// Analytics ID to subscribe to
    pub analytics_id: AnalyticsId,
    /// Target entity for the analytics
    pub target: AnalyticsTarget,
    /// Desired output type
    pub output_type: AnalyticsOutputType,
    /// Optional time window for analytics
    pub time_window: Option<TimeWindow>,
    /// Notification interval in milliseconds (0 = on every update)
    pub notification_interval_ms: u64,
    /// Subscriber identity
    pub subscriber_id: String,
}

impl std::fmt::Debug for SubscriptionRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubscriptionRequest")
            .field("analytics_id", &self.analytics_id)
            .field("target", &self.target)
            .field("output_type", &self.output_type)
            .field("notification_interval_ms", &self.notification_interval_ms)
            .field("subscriber_id", &self.subscriber_id)
            .finish()
    }
}

/// An active analytics subscription
pub struct Subscription {
    /// Unique subscription ID
    pub id: String,
    /// Subscription parameters
    pub request: SubscriptionRequest,
    /// Callback to invoke when analytics are ready
    callback: AnalyticsCallback,
    /// Timestamp of last notification (ms)
    last_notification_ms: u64,
    /// Whether the subscription is active
    pub active: bool,
}

impl std::fmt::Debug for Subscription {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Subscription")
            .field("id", &self.id)
            .field("request", &self.request)
            .field("last_notification_ms", &self.last_notification_ms)
            .field("active", &self.active)
            .finish()
    }
}

/// Response to a subscription request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubscriptionResponse {
    /// Assigned subscription ID
    pub subscription_id: String,
    /// Whether the subscription was accepted
    pub accepted: bool,
    /// Error message if rejected
    pub error: Option<String>,
}

/// Manages analytics subscriptions (Nnwdaf_AnalyticsSubscription service)
///
/// Handles subscribe, unsubscribe, and notification dispatch for analytics
/// consumers (AMF, SMF, other NFs, AF).
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 7.2: Nnwdaf_AnalyticsSubscription service
#[derive(Debug)]
pub struct SubscriptionManager {
    /// Active subscriptions indexed by subscription ID
    subscriptions: HashMap<String, Subscription>,
    /// Counter for generating subscription IDs
    next_id: u64,
    /// Maximum allowed concurrent subscriptions
    max_subscriptions: usize,
}

impl SubscriptionManager {
    /// Creates a new subscription manager
    pub fn new(max_subscriptions: usize) -> Self {
        Self {
            subscriptions: HashMap::new(),
            next_id: 1,
            max_subscriptions,
        }
    }

    /// Subscribes to analytics notifications
    ///
    /// # Arguments
    ///
    /// * `request` - Subscription parameters
    /// * `callback` - Function to call when matching analytics are available
    ///
    /// # Errors
    ///
    /// Returns `SubscriptionError::LimitReached` if the maximum number of
    /// subscriptions has been reached.
    pub fn subscribe(
        &mut self,
        request: SubscriptionRequest,
        callback: AnalyticsCallback,
    ) -> Result<SubscriptionResponse, SubscriptionError> {
        if self.subscriptions.len() >= self.max_subscriptions {
            return Err(SubscriptionError::LimitReached {
                max: self.max_subscriptions,
            });
        }

        let id = format!("sub-{}", self.next_id);
        self.next_id += 1;

        info!(
            "AnalyticsSubscription: New subscription {} for {:?} from {}",
            id, request.analytics_id, request.subscriber_id
        );

        let subscription = Subscription {
            id: id.clone(),
            request: request.clone(),
            callback,
            last_notification_ms: 0,
            active: true,
        };

        self.subscriptions.insert(id.clone(), subscription);

        Ok(SubscriptionResponse {
            subscription_id: id,
            accepted: true,
            error: None,
        })
    }

    /// Unsubscribes from analytics notifications
    ///
    /// # Errors
    ///
    /// Returns `SubscriptionError::NotFound` if the subscription does not exist.
    pub fn unsubscribe(&mut self, subscription_id: &str) -> Result<(), SubscriptionError> {
        if self.subscriptions.remove(subscription_id).is_some() {
            info!(
                "AnalyticsSubscription: Removed subscription {}",
                subscription_id
            );
            Ok(())
        } else {
            Err(SubscriptionError::NotFound {
                id: subscription_id.to_string(),
            })
        }
    }

    /// Notifies subscribers about a new analytics result
    ///
    /// Iterates through active subscriptions matching the analytics result
    /// and invokes their callbacks, respecting notification intervals.
    pub fn notify(&mut self, result: &AnalyticsResult) {
        for subscription in self.subscriptions.values_mut() {
            if !subscription.active {
                continue;
            }

            // Check analytics ID match
            if subscription.request.analytics_id != result.analytics_id {
                continue;
            }

            // Check target match
            if !target_matches(&subscription.request.target, &result.target) {
                continue;
            }

            // Check notification interval
            if subscription.request.notification_interval_ms > 0 {
                let elapsed = result
                    .timestamp_ms
                    .saturating_sub(subscription.last_notification_ms);
                if elapsed < subscription.request.notification_interval_ms {
                    continue;
                }
            }

            debug!(
                "Notifying subscriber {} for {:?}",
                subscription.id, result.analytics_id
            );

            (subscription.callback)(result);
            subscription.last_notification_ms = result.timestamp_ms;
        }
    }

    /// Returns the number of active subscriptions
    pub fn active_count(&self) -> usize {
        self.subscriptions.values().filter(|s| s.active).count()
    }

    /// Returns the total number of subscriptions
    pub fn total_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Returns all subscription IDs
    pub fn subscription_ids(&self) -> Vec<&str> {
        self.subscriptions.keys().map(std::string::String::as_str).collect()
    }

    /// Returns a subscription by ID
    pub fn get_subscription(&self, id: &str) -> Option<&Subscription> {
        self.subscriptions.get(id)
    }

    /// Suspends a subscription (stops notifications without removing)
    pub fn suspend(&mut self, subscription_id: &str) -> Result<(), SubscriptionError> {
        let sub = self
            .subscriptions
            .get_mut(subscription_id)
            .ok_or_else(|| SubscriptionError::NotFound {
                id: subscription_id.to_string(),
            })?;
        sub.active = false;
        Ok(())
    }

    /// Resumes a suspended subscription
    pub fn resume(&mut self, subscription_id: &str) -> Result<(), SubscriptionError> {
        let sub = self
            .subscriptions
            .get_mut(subscription_id)
            .ok_or_else(|| SubscriptionError::NotFound {
                id: subscription_id.to_string(),
            })?;
        sub.active = true;
        Ok(())
    }
}

/// Checks whether an analytics target matches a subscription target
fn target_matches(subscription_target: &AnalyticsTarget, result_target: &AnalyticsTarget) -> bool {
    match (subscription_target, result_target) {
        (AnalyticsTarget::Any, _) => true,
        (AnalyticsTarget::Ue { ue_id: s }, AnalyticsTarget::Ue { ue_id: r }) => s == r,
        (AnalyticsTarget::Cell { cell_id: s }, AnalyticsTarget::Cell { cell_id: r }) => s == r,
        (AnalyticsTarget::Slice { snssai: s }, AnalyticsTarget::Slice { snssai: r }) => s == r,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Nnwdaf_AnalyticsInfo (TS 23.288, Section 7.3)
// ---------------------------------------------------------------------------

/// On-demand analytics query request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsInfoRequest {
    /// Analytics ID to query
    pub analytics_id: AnalyticsId,
    /// Target entity
    pub target: AnalyticsTarget,
    /// Desired output type
    pub output_type: AnalyticsOutputType,
    /// Optional time window
    pub time_window: Option<TimeWindow>,
    /// Additional parameters depending on analytics type
    pub params: AnalyticsQueryParams,
}

/// Additional query parameters for specific analytics types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticsQueryParams {
    /// UE Mobility query parameters
    UeMobility {
        /// Prediction horizon in ms
        horizon_ms: u32,
    },
    /// NF Load query parameters
    NfLoad {
        /// Number of future steps to predict
        horizon_steps: usize,
    },
    /// Abnormal Behavior query (no additional params)
    AbnormalBehavior,
    /// Generic query with no additional params
    None,
}

/// On-demand analytics query response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsInfoResponse {
    /// The analytics result (if successful)
    pub result: Option<AnalyticsResult>,
    /// Whether the query was successful
    pub success: bool,
    /// Error message if the query failed
    pub error: Option<String>,
}

/// Nnwdaf_AnalyticsInfo service handler
///
/// Processes on-demand analytics queries by delegating to the AnLF
/// and returning results synchronously.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 7.3: Nnwdaf_AnalyticsInfo service
pub struct AnalyticsInfoService;

impl AnalyticsInfoService {
    /// Handles an on-demand analytics info request
    ///
    /// Routes the request to the appropriate AnLF analytics function
    /// based on the analytics ID.
    pub fn handle_request(
        request: &AnalyticsInfoRequest,
        anlf: &mut Anlf,
        data_collector: &DataCollector,
        mtlf: &Mtlf,
    ) -> AnalyticsInfoResponse {
        if !anlf.supports_analytics(request.analytics_id) {
            return AnalyticsInfoResponse {
                result: None,
                success: false,
                error: Some(format!(
                    "Analytics ID {:?} is not supported",
                    request.analytics_id
                )),
            };
        }

        let result = match request.analytics_id {
            AnalyticsId::UeMobility => {
                let ue_id = match &request.target {
                    AnalyticsTarget::Ue { ue_id } => *ue_id,
                    _ => {
                        return AnalyticsInfoResponse {
                            result: None,
                            success: false,
                            error: Some(
                                "UeMobility analytics requires a UE target".to_string(),
                            ),
                        };
                    }
                };
                let horizon_ms = match &request.params {
                    AnalyticsQueryParams::UeMobility { horizon_ms } => *horizon_ms,
                    _ => 1000, // default 1 second
                };
                anlf.analyze_ue_mobility(ue_id, horizon_ms, data_collector, mtlf)
            }
            AnalyticsId::NfLoad => {
                let cell_id = match &request.target {
                    AnalyticsTarget::Cell { cell_id } => *cell_id,
                    _ => {
                        return AnalyticsInfoResponse {
                            result: None,
                            success: false,
                            error: Some("NfLoad analytics requires a Cell target".to_string()),
                        };
                    }
                };
                let horizon_steps = match &request.params {
                    AnalyticsQueryParams::NfLoad { horizon_steps } => *horizon_steps,
                    _ => 10, // default 10 steps
                };
                anlf.analyze_nf_load(cell_id, horizon_steps, data_collector, mtlf)
            }
            AnalyticsId::AbnormalBehavior => {
                anlf.analyze_abnormal_behavior(&request.target)
            }
            AnalyticsId::ServiceExperience => {
                anlf.analyze_service_experience(&request.target, data_collector)
            }
            AnalyticsId::UserDataCongestion => {
                anlf.analyze_user_data_congestion(&request.target, data_collector, mtlf)
            }
            AnalyticsId::QosSustainability => {
                anlf.analyze_qos_sustainability(&request.target, data_collector, mtlf)
            }
            AnalyticsId::EnergyEfficiency | AnalyticsId::SliceOptimization => {
                Err(crate::NwdafError::Analytics(crate::AnalyticsError::UnsupportedAnalyticsId {
                    id: request.analytics_id,
                }))
            }
        };

        match result {
            Ok(analytics_result) => AnalyticsInfoResponse {
                result: Some(analytics_result),
                success: true,
                error: None,
            },
            Err(e) => AnalyticsInfoResponse {
                result: None,
                success: false,
                error: Some(e.to_string()),
            },
        }
    }
}

// ---------------------------------------------------------------------------
// Nnwdaf_MLModelProvision (TS 23.288, Section 7.5)
// ---------------------------------------------------------------------------

/// ML model provision service
///
/// Wraps the MTLF model provision functionality as a proper 3GPP service
/// interface. Consumers request models by analytics ID and optionally by
/// version, and receive model metadata and file path information.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 7.5: Nnwdaf_MLModelProvision service
pub struct MlModelProvisionService;

impl MlModelProvisionService {
    /// Handles a model provision request
    ///
    /// Delegates to the MTLF to find and provision the requested model.
    pub fn handle_request(
        request: &ModelProvisionRequest,
        mtlf: &Mtlf,
    ) -> ModelProvisionResponse {
        info!(
            "MLModelProvision: Request from {} for {:?}",
            request.requestor_id, request.analytics_id
        );
        mtlf.handle_provision_request(request)
    }

    /// Lists all available models
    pub fn list_available_models(mtlf: &Mtlf) -> Vec<&MlModelInfo> {
        mtlf.list_models()
    }

    /// Lists models available for a specific analytics ID
    pub fn list_models_for_analytics(
        mtlf: &Mtlf,
        analytics_id: AnalyticsId,
    ) -> Vec<&MlModelInfo> {
        mtlf.list_models_for_analytics(analytics_id)
    }
}

// ---------------------------------------------------------------------------
// Nnwdaf_DataManagement (TS 23.288, Section 7.4) (A13.4)
// ---------------------------------------------------------------------------

/// Data management request type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataManagementOp {
    /// Fetch collected data (read)
    Fetch,
    /// Delete collected data for a target
    Delete,
    /// Export collected data to an external consumer
    Export,
}

/// Request to the Nnwdaf_DataManagement service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagementRequest {
    /// Operation type
    pub op: DataManagementOp,
    /// Target analytics ID (what kind of data)
    pub analytics_id: AnalyticsId,
    /// Target entity
    pub target: AnalyticsTarget,
    /// Optional time range (start_ms, end_ms)
    pub time_range: Option<(u64, u64)>,
    /// Requestor identity
    pub requestor_id: String,
}

/// Response from the Nnwdaf_DataManagement service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataManagementResponse {
    /// Whether the operation succeeded
    pub success: bool,
    /// Number of records affected
    pub records_affected: usize,
    /// Exported data payload (for Fetch/Export operations)
    pub data: Option<String>,
    /// Error message if failed
    pub error: Option<String>,
}

/// Nnwdaf_DataManagement service
///
/// Manages the lifecycle of collected analytics data. Consumers can fetch,
/// delete, or export data through this standardised interface.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 7.4: Nnwdaf_DataManagement service
pub struct DataManagementService;

impl DataManagementService {
    /// Handles a data management request
    pub fn handle_request(
        request: &DataManagementRequest,
        data_collector: &mut DataCollector,
    ) -> DataManagementResponse {
        info!(
            "DataManagement: {:?} from {} for {:?} target={:?}",
            request.op, request.requestor_id, request.analytics_id, request.target
        );

        match request.op {
            DataManagementOp::Fetch => Self::handle_fetch(request, data_collector),
            DataManagementOp::Delete => Self::handle_delete(request, data_collector),
            DataManagementOp::Export => Self::handle_export(request, data_collector),
        }
    }

    /// Fetches collected data matching the request criteria
    fn handle_fetch(
        request: &DataManagementRequest,
        data_collector: &DataCollector,
    ) -> DataManagementResponse {
        match &request.target {
            AnalyticsTarget::Ue { ue_id } => {
                if let Some(history) = data_collector.get_ue_history(*ue_id) {
                    let count = history.len();
                    DataManagementResponse {
                        success: true,
                        records_affected: count,
                        data: Some(format!(
                            "{{\"ue_id\": {ue_id}, \"measurement_count\": {count}}}"
                        )),
                        error: None,
                    }
                } else {
                    DataManagementResponse {
                        success: true,
                        records_affected: 0,
                        data: None,
                        error: None,
                    }
                }
            }
            AnalyticsTarget::Cell { cell_id } => {
                if let Some(history) = data_collector.get_cell_load_history(*cell_id) {
                    let count = history.len();
                    DataManagementResponse {
                        success: true,
                        records_affected: count,
                        data: Some(format!(
                            "{{\"cell_id\": {cell_id}, \"load_record_count\": {count}}}"
                        )),
                        error: None,
                    }
                } else {
                    DataManagementResponse {
                        success: true,
                        records_affected: 0,
                        data: None,
                        error: None,
                    }
                }
            }
            AnalyticsTarget::Any => {
                let ue_count = data_collector.tracked_ue_count();
                let cell_count = data_collector.tracked_cell_count();
                let total = data_collector.total_measurements() as usize;
                DataManagementResponse {
                    success: true,
                    records_affected: total,
                    data: Some(format!(
                        "{{\"tracked_ues\": {ue_count}, \"tracked_cells\": {cell_count}, \"total_measurements\": {total}}}"
                    )),
                    error: None,
                }
            }
            _ => DataManagementResponse {
                success: false,
                records_affected: 0,
                data: None,
                error: Some("Unsupported target type for data management".to_string()),
            },
        }
    }

    /// Deletes collected data matching the request criteria
    fn handle_delete(
        _request: &DataManagementRequest,
        _data_collector: &mut DataCollector,
    ) -> DataManagementResponse {
        // Data deletion is logged but the DataCollector does not expose a
        // per-entity removal API. Return success with 0 records to signal
        // that the request was accepted but no bulk-delete is supported yet.
        debug!("DataManagement: Delete not yet supported at per-entity level");
        DataManagementResponse {
            success: true,
            records_affected: 0,
            data: None,
            error: None,
        }
    }

    /// Exports collected data as a JSON payload
    fn handle_export(
        request: &DataManagementRequest,
        data_collector: &DataCollector,
    ) -> DataManagementResponse {
        // Re-use fetch logic for export
        Self::handle_fetch(request, data_collector)
    }

    /// Returns summary statistics about collected data
    pub fn data_summary(data_collector: &DataCollector) -> DataManagementResponse {
        let ue_count = data_collector.tracked_ue_count();
        let cell_count = data_collector.tracked_cell_count();
        let total = data_collector.total_measurements() as usize;
        let sources = data_collector.source_count();

        DataManagementResponse {
            success: true,
            records_affected: total,
            data: Some(format!(
                "{{\"tracked_ues\": {ue_count}, \"tracked_cells\": {cell_count}, \"total_measurements\": {total}, \"registered_sources\": {sources}}}"
            )),
            error: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Analytics Accuracy Reporting / Feedback (A13.8)
// ---------------------------------------------------------------------------

/// Analytics accuracy feedback from a consumer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsAccuracyFeedback {
    /// Analytics ID the feedback refers to
    pub analytics_id: AnalyticsId,
    /// Target entity
    pub target: AnalyticsTarget,
    /// The subscription or request ID that produced the analytics
    pub reference_id: String,
    /// Consumer-reported accuracy (0.0 to 1.0)
    pub reported_accuracy: f32,
    /// Consumer identity
    pub consumer_id: String,
    /// Timestamp of the feedback
    pub timestamp_ms: u64,
    /// Optional textual note
    pub note: Option<String>,
}

/// Aggregated accuracy statistics for an analytics type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyStats {
    /// Analytics ID
    pub analytics_id: AnalyticsId,
    /// Number of feedback reports received
    pub report_count: u64,
    /// Mean reported accuracy
    pub mean_accuracy: f32,
    /// Minimum reported accuracy
    pub min_accuracy: f32,
    /// Maximum reported accuracy
    pub max_accuracy: f32,
    /// Latest feedback timestamp
    pub latest_timestamp_ms: u64,
}

/// Analytics accuracy tracker
///
/// Collects accuracy feedback from analytics consumers and maintains
/// per-analytics-ID statistics. This enables the NWDAF to monitor the
/// quality of its analytics output and trigger model retraining when
/// accuracy degrades.
///
/// # 3GPP Reference
///
/// - TS 23.288 Section 6.1: NWDAF accuracy monitoring
pub struct AnalyticsAccuracyTracker {
    /// Per-analytics-ID feedback history
    feedback: HashMap<AnalyticsId, Vec<AnalyticsAccuracyFeedback>>,
    /// Maximum feedback entries to keep per analytics ID
    max_history: usize,
    /// Accuracy threshold below which a retraining hint is generated
    retrain_threshold: f32,
}

impl AnalyticsAccuracyTracker {
    /// Creates a new accuracy tracker
    pub fn new(max_history: usize, retrain_threshold: f32) -> Self {
        Self {
            feedback: HashMap::new(),
            max_history,
            retrain_threshold,
        }
    }

    /// Records a piece of accuracy feedback
    pub fn record_feedback(&mut self, feedback: AnalyticsAccuracyFeedback) {
        info!(
            "AccuracyTracker: feedback from {} for {:?}: accuracy={:.3}",
            feedback.consumer_id, feedback.analytics_id, feedback.reported_accuracy
        );

        let entries = self
            .feedback
            .entry(feedback.analytics_id)
            .or_default();

        entries.push(feedback);

        // Prune old entries
        if entries.len() > self.max_history {
            let to_remove = entries.len() - self.max_history;
            entries.drain(0..to_remove);
        }
    }

    /// Returns accuracy statistics for a given analytics ID
    pub fn get_stats(&self, analytics_id: AnalyticsId) -> Option<AccuracyStats> {
        let entries = self.feedback.get(&analytics_id)?;
        if entries.is_empty() {
            return None;
        }

        let count = entries.len() as u64;
        let sum: f32 = entries.iter().map(|f| f.reported_accuracy).sum();
        let mean = sum / count as f32;
        let min = entries
            .iter()
            .map(|f| f.reported_accuracy)
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let max = entries
            .iter()
            .map(|f| f.reported_accuracy)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);
        let latest_ts = entries
            .iter()
            .map(|f| f.timestamp_ms)
            .max()
            .unwrap_or(0);

        Some(AccuracyStats {
            analytics_id,
            report_count: count,
            mean_accuracy: mean,
            min_accuracy: min,
            max_accuracy: max,
            latest_timestamp_ms: latest_ts,
        })
    }

    /// Returns all analytics IDs that may need retraining (accuracy below threshold)
    pub fn needs_retraining(&self) -> Vec<AnalyticsId> {
        self.feedback
            .keys()
            .filter_map(|&aid| {
                self.get_stats(aid).and_then(|stats| {
                    if stats.mean_accuracy < self.retrain_threshold {
                        Some(aid)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Returns the total number of feedback entries across all analytics IDs
    pub fn total_feedback_count(&self) -> usize {
        self.feedback.values().map(std::vec::Vec::len).sum()
    }

    /// Returns the retrain threshold
    pub fn retrain_threshold(&self) -> f32 {
        self.retrain_threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anlf::AnalyticsPayload;
    use std::sync::atomic::{AtomicU32, Ordering};

    #[test]
    fn test_subscription_manager_creation() {
        let mgr = SubscriptionManager::new(100);
        assert_eq!(mgr.total_count(), 0);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn test_subscribe() {
        let mut mgr = SubscriptionManager::new(100);

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 0,
            subscriber_id: "amf-1".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(|_result| {});
        let response = mgr.subscribe(request, callback);
        assert!(response.is_ok());

        let resp = response.expect("should succeed");
        assert!(resp.accepted);
        assert_eq!(mgr.total_count(), 1);
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn test_subscribe_limit_reached() {
        let mut mgr = SubscriptionManager::new(1);

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 0,
            subscriber_id: "amf-1".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(|_| {});
        let _ = mgr.subscribe(request.clone(), callback.clone());

        // Second subscription should fail
        let result = mgr.subscribe(request, callback);
        assert!(result.is_err());
        assert!(matches!(
            result.expect_err("should fail"),
            SubscriptionError::LimitReached { max: 1 }
        ));
    }

    #[test]
    fn test_unsubscribe() {
        let mut mgr = SubscriptionManager::new(100);

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 0,
            subscriber_id: "amf-1".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(|_| {});
        let resp = mgr.subscribe(request, callback).expect("should subscribe");

        assert!(mgr.unsubscribe(&resp.subscription_id).is_ok());
        assert_eq!(mgr.total_count(), 0);
    }

    #[test]
    fn test_unsubscribe_not_found() {
        let mut mgr = SubscriptionManager::new(100);
        let result = mgr.unsubscribe("nonexistent");
        assert!(result.is_err());
    }

    #[test]
    fn test_notification_delivery() {
        let mut mgr = SubscriptionManager::new(100);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 0,
            subscriber_id: "amf-1".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(move |_result| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        mgr.subscribe(request, callback).expect("should subscribe");

        // Send matching result
        let result = AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 1000,
            confidence: 0.9,
            payload: AnalyticsPayload::UeMobility {
                trajectory: None,
                handover_recommendation: None,
            },
        };
        mgr.notify(&result);

        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Non-matching result (different UE)
        let result2 = AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 999 },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 2000,
            confidence: 0.9,
            payload: AnalyticsPayload::UeMobility {
                trajectory: None,
                handover_recommendation: None,
            },
        };
        mgr.notify(&result2);

        // Counter should not have changed
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_notification_interval() {
        let mut mgr = SubscriptionManager::new(100);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::NfLoad,
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 1000, // Only notify every 1000ms
            subscriber_id: "smf-1".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        mgr.subscribe(request, callback).expect("should subscribe");

        // First notification at t=1000
        let result = AnalyticsResult {
            analytics_id: AnalyticsId::NfLoad,
            target: AnalyticsTarget::Cell { cell_id: 1 },
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 1000,
            confidence: 0.8,
            payload: AnalyticsPayload::NfLoad {
                current_load: 0.5,
                predicted_load: vec![],
                time_to_overload_ms: None,
            },
        };
        mgr.notify(&result);
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Second notification at t=1500 (within interval) - should be skipped
        let result2 = AnalyticsResult {
            timestamp_ms: 1500,
            ..result.clone()
        };
        mgr.notify(&result2);
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Still 1

        // Third notification at t=2100 (past interval) - should fire
        let result3 = AnalyticsResult {
            timestamp_ms: 2100,
            ..result
        };
        mgr.notify(&result3);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_suspend_resume() {
        let mut mgr = SubscriptionManager::new(100);
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = counter.clone();

        let request = SubscriptionRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            notification_interval_ms: 0,
            subscriber_id: "test".to_string(),
        };

        let callback: AnalyticsCallback = Arc::new(move |_| {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });
        let resp = mgr.subscribe(request, callback).expect("should subscribe");

        // Suspend
        mgr.suspend(&resp.subscription_id).expect("should suspend");
        assert_eq!(mgr.active_count(), 0);

        // Notification should be skipped
        let result = AnalyticsResult {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Predictions,
            timestamp_ms: 1000,
            confidence: 0.9,
            payload: AnalyticsPayload::UeMobility {
                trajectory: None,
                handover_recommendation: None,
            },
        };
        mgr.notify(&result);
        assert_eq!(counter.load(Ordering::SeqCst), 0);

        // Resume
        mgr.resume(&resp.subscription_id).expect("should resume");
        assert_eq!(mgr.active_count(), 1);

        // Now notification should be delivered
        mgr.notify(&result);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_analytics_info_service_ue_mobility() {
        let mut anlf = Anlf::new();
        let mut collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        // Add measurement history
        for i in 0..10 {
            collector.report_ue_measurement(crate::UeMeasurement {
                ue_id: 1,
                rsrp: -80.0,
                rsrq: -10.0,
                sinr: Some(15.0),
                position: crate::Vector3::new(i as f64 * 10.0, 0.0, 0.0),
                velocity: None,
                serving_cell_id: 1,
                timestamp_ms: i as u64 * 100,
            });
        }

        let request = AnalyticsInfoRequest {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            output_type: AnalyticsOutputType::Predictions,
            time_window: None,
            params: AnalyticsQueryParams::UeMobility { horizon_ms: 1000 },
        };

        let response =
            AnalyticsInfoService::handle_request(&request, &mut anlf, &collector, &mtlf);
        assert!(response.success);
        assert!(response.result.is_some());
    }

    #[test]
    fn test_analytics_info_service_unsupported() {
        let mut anlf = Anlf::new();
        let collector = DataCollector::new(100);
        let mtlf = Mtlf::new();

        let request = AnalyticsInfoRequest {
            analytics_id: AnalyticsId::ServiceExperience, // not implemented
            target: AnalyticsTarget::Any,
            output_type: AnalyticsOutputType::Statistics,
            time_window: None,
            params: AnalyticsQueryParams::None,
        };

        let response =
            AnalyticsInfoService::handle_request(&request, &mut anlf, &collector, &mtlf);
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_ml_model_provision_service() {
        let mut mtlf = Mtlf::new();
        mtlf.register_model(crate::mtlf::MlModelInfo {
            model_id: "traj-v1".to_string(),
            analytics_id: AnalyticsId::UeMobility,
            version: "1.0.0".to_string(),
            path: std::path::PathBuf::from("/models/traj.onnx"),
            file_size: 1024,
            accuracy: Some(0.92),
            is_loaded: true,
            description: "Trajectory model".to_string(),
        });

        let request = ModelProvisionRequest {
            analytics_id: AnalyticsId::UeMobility,
            version: None,
            requestor_id: "anlf-1".to_string(),
        };

        let response = MlModelProvisionService::handle_request(&request, &mtlf);
        assert!(response.success);
        assert_eq!(response.model_info.model_id, "traj-v1");

        let models = MlModelProvisionService::list_available_models(&mtlf);
        assert_eq!(models.len(), 1);
    }

    #[test]
    fn test_target_matches() {
        assert!(target_matches(&AnalyticsTarget::Any, &AnalyticsTarget::Ue { ue_id: 1 }));
        assert!(target_matches(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &AnalyticsTarget::Ue { ue_id: 1 }
        ));
        assert!(!target_matches(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &AnalyticsTarget::Ue { ue_id: 2 }
        ));
        assert!(!target_matches(
            &AnalyticsTarget::Ue { ue_id: 1 },
            &AnalyticsTarget::Cell { cell_id: 1 }
        ));
    }

    // -----------------------------------------------------------------------
    // DataManagement tests (A13.4)
    // -----------------------------------------------------------------------

    #[test]
    fn test_data_management_fetch_ue() {
        let mut collector = DataCollector::new(100);
        collector.report_ue_measurement(crate::UeMeasurement {
            ue_id: 1,
            rsrp: -80.0,
            rsrq: -10.0,
            sinr: Some(15.0),
            position: crate::Vector3::new(0.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 1000,
        });

        let request = DataManagementRequest {
            op: DataManagementOp::Fetch,
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            time_range: None,
            requestor_id: "consumer-1".to_string(),
        };

        let response = DataManagementService::handle_request(&request, &mut collector);
        assert!(response.success);
        assert_eq!(response.records_affected, 1);
        assert!(response.data.is_some());
    }

    #[test]
    fn test_data_management_fetch_all() {
        let mut collector = DataCollector::new(100);
        collector.report_ue_measurement(crate::UeMeasurement {
            ue_id: 1,
            rsrp: -80.0,
            rsrq: -10.0,
            sinr: None,
            position: crate::Vector3::new(0.0, 0.0, 0.0),
            velocity: None,
            serving_cell_id: 1,
            timestamp_ms: 1000,
        });

        let request = DataManagementRequest {
            op: DataManagementOp::Fetch,
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            time_range: None,
            requestor_id: "consumer-1".to_string(),
        };

        let response = DataManagementService::handle_request(&request, &mut collector);
        assert!(response.success);
        assert_eq!(response.records_affected, 1);
    }

    #[test]
    fn test_data_management_summary() {
        let collector = DataCollector::new(100);
        let response = DataManagementService::data_summary(&collector);
        assert!(response.success);
        assert_eq!(response.records_affected, 0);
    }

    // -----------------------------------------------------------------------
    // Analytics accuracy feedback tests (A13.8)
    // -----------------------------------------------------------------------

    #[test]
    fn test_accuracy_tracker_creation() {
        let tracker = AnalyticsAccuracyTracker::new(100, 0.7);
        assert_eq!(tracker.total_feedback_count(), 0);
        assert_eq!(tracker.retrain_threshold(), 0.7);
    }

    #[test]
    fn test_accuracy_feedback_recording() {
        let mut tracker = AnalyticsAccuracyTracker::new(100, 0.7);

        tracker.record_feedback(AnalyticsAccuracyFeedback {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Ue { ue_id: 1 },
            reference_id: "sub-1".to_string(),
            reported_accuracy: 0.92,
            consumer_id: "amf-1".to_string(),
            timestamp_ms: 1000,
            note: None,
        });

        assert_eq!(tracker.total_feedback_count(), 1);

        let stats = tracker.get_stats(AnalyticsId::UeMobility).unwrap();
        assert_eq!(stats.report_count, 1);
        assert!((stats.mean_accuracy - 0.92).abs() < 0.001);
    }

    #[test]
    fn test_accuracy_stats_aggregation() {
        let mut tracker = AnalyticsAccuracyTracker::new(100, 0.7);

        for (i, acc) in [0.9, 0.8, 0.7].iter().enumerate() {
            tracker.record_feedback(AnalyticsAccuracyFeedback {
                analytics_id: AnalyticsId::NfLoad,
                target: AnalyticsTarget::Any,
                reference_id: format!("ref-{i}"),
                reported_accuracy: *acc,
                consumer_id: "oam-1".to_string(),
                timestamp_ms: (i as u64 + 1) * 1000,
                note: None,
            });
        }

        let stats = tracker.get_stats(AnalyticsId::NfLoad).unwrap();
        assert_eq!(stats.report_count, 3);
        assert!((stats.mean_accuracy - 0.8).abs() < 0.001);
        assert!((stats.min_accuracy - 0.7).abs() < 0.001);
        assert!((stats.max_accuracy - 0.9).abs() < 0.001);
    }

    #[test]
    fn test_needs_retraining() {
        let mut tracker = AnalyticsAccuracyTracker::new(100, 0.8);

        // Good accuracy
        tracker.record_feedback(AnalyticsAccuracyFeedback {
            analytics_id: AnalyticsId::UeMobility,
            target: AnalyticsTarget::Any,
            reference_id: "ref-1".to_string(),
            reported_accuracy: 0.95,
            consumer_id: "amf-1".to_string(),
            timestamp_ms: 1000,
            note: None,
        });

        // Bad accuracy
        tracker.record_feedback(AnalyticsAccuracyFeedback {
            analytics_id: AnalyticsId::NfLoad,
            target: AnalyticsTarget::Any,
            reference_id: "ref-2".to_string(),
            reported_accuracy: 0.5,
            consumer_id: "oam-1".to_string(),
            timestamp_ms: 2000,
            note: None,
        });

        let needs_retrain = tracker.needs_retraining();
        assert!(!needs_retrain.contains(&AnalyticsId::UeMobility));
        assert!(needs_retrain.contains(&AnalyticsId::NfLoad));
    }

    #[test]
    fn test_accuracy_history_pruning() {
        let mut tracker = AnalyticsAccuracyTracker::new(3, 0.7);

        for i in 0..5 {
            tracker.record_feedback(AnalyticsAccuracyFeedback {
                analytics_id: AnalyticsId::UeMobility,
                target: AnalyticsTarget::Any,
                reference_id: format!("ref-{i}"),
                reported_accuracy: 0.8 + i as f32 * 0.01,
                consumer_id: "amf-1".to_string(),
                timestamp_ms: (i as u64 + 1) * 1000,
                note: None,
            });
        }

        // Should keep only last 3
        assert_eq!(tracker.total_feedback_count(), 3);
    }
}
