//! AI Agent Framework Task for gNB

use crate::tasks::{AgentMessage, GnbTaskBase, RrcMessage, Task, TaskMessage};
use nextgsim_agent::{AgentCapabilities, AgentCoordinator, AgentId, AgentType, Intent, IntentType};
use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub struct AgentTask {
    task_base: GnbTaskBase,
    coordinator: AgentCoordinator,
    agents: HashMap<String, Vec<String>>,
    /// Cell observations received (#459).
    observations: u64,
}

impl AgentTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            task_base,
            coordinator: AgentCoordinator::new(),
            agents: HashMap::new(),
            observations: 0,
        }
    }

    /// Records a cell observation in the coordinator's state store, under
    /// `cell/<id>/connected_ues` and `cell/<id>/throughput_mbps`, so intent
    /// executors read the cell's measured state. A throughput the producer could
    /// not measure is left out rather than written as a number.
    fn record_cell_observation(
        &mut self,
        cell_id: i32,
        connected_ues: u32,
        throughput_mbps: Option<f32>,
    ) {
        if self.observations == 0 {
            info!(
                "Agent: first cell observation, cell {} with {} connected UE(s)",
                cell_id, connected_ues
            );
        }
        self.observations += 1;
        let state = self.coordinator.state_provider();
        state.set(
            &format!("cell/{cell_id}/connected_ues"),
            &connected_ues.to_string(),
        );
        if let Some(mbps) = throughput_mbps {
            state.set(
                &format!("cell/{cell_id}/throughput_mbps"),
                &mbps.to_string(),
            );
        }
    }

    /// Cell observations received so far.
    pub fn observations(&self) -> u64 {
        self.observations
    }

    /// The coordinator, for reading the state the observations wrote.
    pub fn coordinator(&self) -> &AgentCoordinator {
        &self.coordinator
    }
}

#[async_trait::async_trait]
impl Task for AgentTask {
    type Message = AgentMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("Agent task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        AgentMessage::RegisterAgent {
                            agent_id,
                            agent_type,
                            capabilities,
                        } => {
                            debug!(
                                "Agent: Registering {} (type={}, caps={})",
                                agent_id,
                                agent_type,
                                capabilities.len()
                            );
                            let a_id = AgentId::new(&agent_id);
                            let a_type = match agent_type.as_str() {
                                "mobility" => AgentType::Mobility,
                                "resource" => AgentType::Resource,
                                "qos" => AgentType::Qos,
                                "security" => AgentType::Security,
                                "slicing" => AgentType::Slicing,
                                _ => AgentType::Custom,
                            };
                            let caps = AgentCapabilities {
                                allowed_intents: capabilities.clone(),
                                ..Default::default()
                            };
                            self.coordinator.register_agent(a_id, a_type, caps);
                            self.agents.insert(agent_id, capabilities);
                        }
                        AgentMessage::SubmitIntent {
                            agent_id,
                            intent_type,
                            parameters,
                        } => {
                            debug!("Agent: Intent '{}' from {}", intent_type, agent_id);
                            let a_id = AgentId::new(&agent_id);
                            let i_type = match intent_type.as_str() {
                                "query" => IntentType::Query,
                                "optimize_resources" => IntentType::OptimizeResources,
                                "trigger_handover" => IntentType::TriggerHandover,
                                "adjust_qos" => IntentType::AdjustQos,
                                "create_slice" => IntentType::CreateSlice,
                                "modify_slice" => IntentType::ModifySlice,
                                other => IntentType::Custom(other.to_string()),
                            };
                            let mut intent = Intent::new(a_id, i_type);
                            for (k, v) in parameters {
                                intent = intent.with_param(k, v);
                            }
                            if let Err(e) = self.coordinator.submit_intent(intent) {
                                warn!("Agent: Intent submission failed: {}", e);
                            }
                        }
                        AgentMessage::CellObservation {
                            cell_id,
                            connected_ues,
                            throughput_mbps,
                        } => {
                            self.record_cell_observation(cell_id, connected_ues, throughput_mbps);
                        }
                        AgentMessage::CoordinationEvent {
                            event_type,
                            agent_ids,
                        } => {
                            debug!(
                                "Agent: Coordination event '{}' with {} agents",
                                event_type,
                                agent_ids.len()
                            );
                            // Process any pending intents queued during this coordination cycle
                            let results = self.coordinator.process_intents();
                            if !results.is_empty() {
                                debug!(
                                    "Agent: Processed {} intents after coordination event '{}'",
                                    results.len(),
                                    event_type
                                );
                                // Forward any handover actions produced by intent processing to RRC
                                for result in &results {
                                    if let Some(ue_id_str) = result.data.get("ue_id") {
                                        if let Some(target_cell_str) =
                                            result.data.get("target_cell")
                                        {
                                            if let (Ok(ue_id), Ok(target_cell)) = (
                                                ue_id_str.parse::<i32>(),
                                                target_cell_str.parse::<i32>(),
                                            ) {
                                                let confidence = result
                                                    .data
                                                    .get("confidence")
                                                    .and_then(|s| s.parse::<f32>().ok())
                                                    .unwrap_or(0.8);
                                                debug!(
                                                    "Agent: Forwarding handover action: ue_id={}, target_cell={}, confidence={}",
                                                    ue_id, target_cell, confidence
                                                );
                                                let msg = RrcMessage::NwdafHandoverRecommendation {
                                                    ue_id,
                                                    target_cell,
                                                    confidence,
                                                };
                                                if let Err(e) =
                                                    self.task_base.rrc_tx.send(msg).await
                                                {
                                                    warn!("Agent: Failed to send handover action to RRC: {}", e);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!(
            "Agent task stopped, {} registered agents",
            self.agents.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::DEFAULT_CHANNEL_CAPACITY;
    use nextgsim_common::config::GnbConfig;

    /// #459: a cell observation lands in the coordinator's state store, where
    /// intent executors read it; an unmeasured throughput is left out.
    #[tokio::test]
    async fn a_cell_observation_is_written_to_the_agent_state() {
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(GnbConfig::default(), DEFAULT_CHANNEL_CAPACITY);
        let mut task = AgentTask::new(task_base);
        let (tx, rx) = mpsc::channel(4);
        tx.send(TaskMessage::Message(AgentMessage::CellObservation {
            cell_id: 7,
            connected_ues: 3,
            throughput_mbps: Some(1.5),
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Message(AgentMessage::CellObservation {
            cell_id: 8,
            connected_ues: 0,
            throughput_mbps: None,
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Shutdown).await.unwrap();
        task.run(rx).await;

        let state = task.coordinator().state_provider();
        assert_eq!(task.observations(), 2);
        assert_eq!(state.get("cell/7/connected_ues").as_deref(), Some("3"));
        assert_eq!(state.get("cell/7/throughput_mbps").as_deref(), Some("1.5"));
        assert_eq!(state.get("cell/8/connected_ues").as_deref(), Some("0"));
        assert_eq!(state.get("cell/8/throughput_mbps"), None, "not invented");
    }
}
