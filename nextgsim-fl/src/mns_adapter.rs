//! Conceptual adapter onto the TS 28.105 AI/ML training-management model.
//!
//! Maps this engine's training round + per-round metrics onto the **field
//! concepts** of the TS 28.105 AI/ML training IOCs:
//! - `MLTrainingFunction`  (§7.3a.1.2.1)
//! - `MLTrainingRequest`   (§7.3a.1.2.2)
//! - `MLTrainingProcess`   (§7.3a.1.2.3)
//! - `MLTrainingReport`    (§7.3a.1.2.4)
//!
//! SCOPE: **conceptual alignment only.** These are plain serde structs, NOT a
//! conformant YANG/JSON MnS schema. This crate is the FL training *process* that
//! would run *under* an `MLTrainingFunction`; it is not itself a TS 28.105 MnS
//! producer. Per TS 28.105 §6.2b.2.15.1 NOTE 2 the FL algorithm is outside the
//! scope of standardization, so this adapter only documents how the engine's
//! outputs would surface in the management information model.

use crate::metrics::RoundMetrics;
use crate::{RoundStatus, TrainingRound};
use serde::{Deserialize, Serialize};

/// Conceptual mirror of the TS 28.105 `MLTrainingRequest` `requestStatus`
/// attribute values (e.g. NOT_STARTED / TRAINING / TRAINING_COMPLETED /
/// CANCELLED / FAILED). Naming follows the spec's conceptual states; this is
/// not a wire-conformant enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MlRequestStatus {
    /// No training round has started yet.
    NotStarted,
    /// A round is collecting updates or aggregating.
    Training,
    /// The round completed successfully.
    TrainingCompleted,
    /// The round was cancelled by a consumer.
    Cancelled,
    /// The round failed.
    Failed,
}

impl MlRequestStatus {
    /// Maps the engine's [`RoundStatus`] onto the conceptual request status.
    pub fn from_round_status(status: RoundStatus) -> Self {
        match status {
            RoundStatus::WaitingForParticipants => Self::NotStarted,
            RoundStatus::Collecting | RoundStatus::Aggregating => Self::Training,
            RoundStatus::Complete => Self::TrainingCompleted,
            RoundStatus::Failed => Self::Failed,
        }
    }
}

/// Conceptual subset of the TS 28.105 `MLTrainingRequest` IOC attributes.
///
/// Conceptual mapping only; not a conformant 28.105 datatype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlTrainingRequest {
    /// `mLTrainingType` — the kind of model/analytics to train.
    pub ml_training_type: String,
    /// Consumer / source identity that requested the training.
    pub consumer_id: String,
    /// `mLModelRef` — reference to the model to be (re)trained, if any.
    pub ml_model_ref: Option<String>,
    /// Consumer-requested participants for the round.
    pub expected_participants: Vec<String>,
}

impl MlTrainingRequest {
    /// Maps a conceptual `MLTrainingRequest` onto the engine's expected
    /// participant list (what a producer would instantiate the round with).
    ///
    /// Conceptual mapping only.
    pub fn to_engine_config(&self) -> Vec<String> {
        self.expected_participants.clone()
    }
}

/// Per-participant contribution/performance entry within an
/// [`MlTrainingReport`] (conceptual `MLTrainingReport` contribution list).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantContributionReport {
    /// Participant identity.
    pub participant_id: String,
    /// Number of local training samples this participant contributed.
    pub num_samples: u64,
}

/// Conceptual subset of the TS 28.105 `MLTrainingReport` IOC attributes.
///
/// Conceptual mapping only; not a conformant 28.105 datatype.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MlTrainingReport {
    /// Round number this report covers.
    pub round: u64,
    /// `requestStatus` — conceptual training-request state.
    pub request_status: MlRequestStatus,
    /// `mLModelRef` — reference to the produced model, if the round completed.
    pub ml_model_ref: Option<String>,
    /// `trainingDataQualityScore` — None when the engine did not compute a
    /// training-data quality score (it currently does not), so consumers do not
    /// mistake an absent score for a measured one.
    pub training_data_quality_score: Option<f32>,
    /// Average training loss across participants for the round, if known.
    pub avg_loss: Option<f32>,
    /// Total samples contributed across all participants in the round.
    pub total_samples: u64,
    /// Per-participant contribution/performance list.
    pub contributions: Vec<ParticipantContributionReport>,
}

impl MlTrainingReport {
    /// Builds a conceptual `MLTrainingReport` from a training round and its
    /// optional per-round metrics.
    ///
    /// Conceptual mapping only; not a conformant 28.105 report. The contribution
    /// list and `requestStatus` are derived from the round; `avg_loss` is taken
    /// from `metrics` when available; `training_data_quality_score` is left
    /// `None` because the engine does not compute one.
    pub fn from_round(round: &TrainingRound, metrics: Option<&RoundMetrics>) -> Self {
        let mut contributions: Vec<ParticipantContributionReport> = round
            .received_updates
            .values()
            .map(|u| ParticipantContributionReport {
                participant_id: u.participant_id.clone(),
                num_samples: u.num_samples,
            })
            .collect();
        // Deterministic order (HashMap iteration order is unspecified).
        contributions.sort_by(|a, b| a.participant_id.cmp(&b.participant_id));

        let total_samples = round.received_updates.values().map(|u| u.num_samples).sum();

        Self {
            round: round.round,
            request_status: MlRequestStatus::from_round_status(round.status),
            ml_model_ref: round
                .result
                .as_ref()
                .map(|m| format!("model-v{}", m.version)),
            training_data_quality_score: None,
            avg_loss: metrics.map(|m| m.avg_loss),
            total_samples,
            contributions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{timestamp_now, AggregatedModel, ModelUpdate};

    fn round_with_two_updates(status: RoundStatus, with_result: bool) -> TrainingRound {
        let mut round = TrainingRound::new(7, vec!["ue-1".into(), "ue-2".into()], 60);
        round.status = status;
        round.received_updates.insert(
            "ue-1".into(),
            ModelUpdate {
                participant_id: "ue-1".into(),
                base_version: 1,
                gradients: vec![0.1, 0.2],
                num_samples: 100,
                loss: 0.5,
                timestamp_ms: timestamp_now(),
            },
        );
        round.received_updates.insert(
            "ue-2".into(),
            ModelUpdate {
                participant_id: "ue-2".into(),
                base_version: 1,
                gradients: vec![0.3, 0.4],
                num_samples: 200,
                loss: 0.4,
                timestamp_ms: timestamp_now(),
            },
        );
        if with_result {
            round.result = Some(AggregatedModel {
                version: 3,
                weights: vec![0.2, 0.3],
                num_participants: 2,
                total_samples: 300,
                avg_loss: 0.45,
                timestamp_ms: timestamp_now(),
            });
        }
        round
    }

    #[test]
    fn report_from_completed_round_populates_fields() {
        let round = round_with_two_updates(RoundStatus::Complete, true);
        let metrics = RoundMetrics {
            round: 7,
            avg_loss: 0.45,
            min_loss: 0.4,
            max_loss: 0.5,
            num_participants: 2,
            total_samples: 300,
            duration_ms: 10,
            timestamp_ms: timestamp_now(),
        };

        let report = MlTrainingReport::from_round(&round, Some(&metrics));

        assert_eq!(report.round, 7);
        assert_eq!(report.request_status, MlRequestStatus::TrainingCompleted);
        assert_eq!(report.ml_model_ref.as_deref(), Some("model-v3"));
        assert_eq!(report.total_samples, 300);
        assert_eq!(report.avg_loss, Some(0.45));
        // Not measured -> None, distinct from a fabricated high score.
        assert_eq!(report.training_data_quality_score, None);
        assert_eq!(report.contributions.len(), 2);
        assert_eq!(report.contributions[0].participant_id, "ue-1");
        assert_eq!(report.contributions[0].num_samples, 100);
        assert_eq!(report.contributions[1].num_samples, 200);
    }

    #[test]
    fn status_maps_and_request_round_trips() {
        let waiting = round_with_two_updates(RoundStatus::WaitingForParticipants, false);
        let report = MlTrainingReport::from_round(&waiting, None);
        assert_eq!(report.request_status, MlRequestStatus::NotStarted);
        assert_eq!(report.ml_model_ref, None);
        assert_eq!(report.avg_loss, None);

        let collecting = round_with_two_updates(RoundStatus::Collecting, false);
        assert_eq!(
            MlTrainingReport::from_round(&collecting, None).request_status,
            MlRequestStatus::Training
        );

        let req = MlTrainingRequest {
            ml_training_type: "qoe-prediction".into(),
            consumer_id: "nwdaf-1".into(),
            ml_model_ref: Some("model-v3".into()),
            expected_participants: vec!["ue-1".into(), "ue-2".into()],
        };
        assert_eq!(req.to_engine_config(), vec!["ue-1", "ue-2"]);
    }
}
