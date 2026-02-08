//! Error types for NWDAF operations
//!
//! Defines the error hierarchy for analytics, subscription,
//! prediction, and data collection operations.

use std::path::PathBuf;
use thiserror::Error;

use crate::analytics_id::AnalyticsId;

/// Top-level NWDAF error type
#[derive(Error, Debug)]
pub enum NwdafError {
    /// Analytics operation failed
    #[error("Analytics error: {0}")]
    Analytics(#[from] AnalyticsError),

    /// Prediction operation failed
    #[error("Prediction error: {0}")]
    Prediction(#[from] PredictionError),

    /// Subscription operation failed
    #[error("Subscription error: {0}")]
    Subscription(#[from] SubscriptionError),

    /// Data collection error
    #[error("Data collection error: {0}")]
    DataCollection(#[from] DataCollectionError),

    /// Model-related error (from nextgsim-ai)
    #[error("Model error: {0}")]
    Model(#[from] nextgsim_ai::ModelError),

    /// Inference error (from nextgsim-ai)
    #[error("Inference error: {0}")]
    Inference(#[from] nextgsim_ai::InferenceError),
}

/// Errors related to analytics operations
#[derive(Error, Debug)]
pub enum AnalyticsError {
    /// Requested analytics ID is not supported
    #[error("Unsupported analytics ID: {id:?}")]
    UnsupportedAnalyticsId {
        /// The unsupported analytics identifier
        id: AnalyticsId,
    },

    /// Insufficient data to produce analytics
    #[error("Insufficient data for analytics: need at least {required} samples, have {available}")]
    InsufficientData {
        /// Minimum samples required
        required: usize,
        /// Samples currently available
        available: usize,
    },

    /// Analytics computation failed
    #[error("Analytics computation failed: {reason}")]
    ComputationFailed {
        /// Description of the failure
        reason: String,
    },

    /// Target entity (UE, cell, etc.) not found in collected data
    #[error("Target not found: {target}")]
    TargetNotFound {
        /// Identifier of the missing target
        target: String,
    },
}

/// Errors related to prediction operations
#[derive(Error, Debug)]
pub enum PredictionError {
    /// No model loaded for the requested prediction type
    #[error("No model loaded for prediction; using fallback")]
    NoModelLoaded,

    /// Model file not found
    #[error("Model file not found: {path}")]
    ModelNotFound {
        /// Path that was searched
        path: PathBuf,
    },

    /// Model inference failed
    #[error("Model inference failed: {reason}")]
    InferenceFailed {
        /// Description of the failure
        reason: String,
    },

    /// Insufficient history for prediction
    #[error("Insufficient history: need {required} points, have {available}")]
    InsufficientHistory {
        /// Minimum points required
        required: usize,
        /// Points currently available
        available: usize,
    },

    /// Invalid prediction horizon
    #[error("Invalid prediction horizon: {horizon_ms}ms")]
    InvalidHorizon {
        /// The invalid horizon value
        horizon_ms: u32,
    },
}

/// Errors related to subscription operations
#[derive(Error, Debug)]
pub enum SubscriptionError {
    /// Subscription not found
    #[error("Subscription not found: {id}")]
    NotFound {
        /// The missing subscription ID
        id: String,
    },

    /// Maximum number of subscriptions reached
    #[error("Maximum subscriptions reached: {max}")]
    LimitReached {
        /// Maximum allowed subscriptions
        max: usize,
    },

    /// Subscription callback failed
    #[error("Subscription callback failed: {reason}")]
    CallbackFailed {
        /// Description of the failure
        reason: String,
    },

    /// Invalid subscription parameters
    #[error("Invalid subscription parameters: {reason}")]
    InvalidParameters {
        /// Description of the invalid parameters
        reason: String,
    },
}

/// Errors related to data collection
#[derive(Error, Debug)]
pub enum DataCollectionError {
    /// Data source already registered
    #[error("Data source already registered: {source_id}")]
    AlreadyRegistered {
        /// The duplicate source ID
        source_id: String,
    },

    /// Data source not found
    #[error("Data source not found: {source_id}")]
    SourceNotFound {
        /// The missing source ID
        source_id: String,
    },

    /// Invalid measurement data
    #[error("Invalid measurement data: {reason}")]
    InvalidData {
        /// Description of the data issue
        reason: String,
    },
}
