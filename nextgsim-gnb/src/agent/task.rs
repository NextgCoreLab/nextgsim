//! AI Agent Framework Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use nextgsim_agent::{AgentCapabilities, AgentCoordinator, AgentId, AgentType, Intent, IntentType};
use crate::tasks::{GnbTaskBase, AgentMessage, Task, TaskMessage};

pub struct AgentTask {
    _task_base: GnbTaskBase,
    coordinator: AgentCoordinator,
    agents: HashMap<String, Vec<String>>,
}

impl AgentTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            coordinator: AgentCoordinator::new(),
            agents: HashMap::new(),
        }
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
                        AgentMessage::RegisterAgent { agent_id, agent_type, capabilities } => {
                            debug!("Agent: Registering {} (type={}, caps={})", agent_id, agent_type, capabilities.len());
                            let a_id = AgentId::new(&agent_id);
                            let a_type = match agent_type.as_str() {
                                "mobility" => AgentType::Mobility,
                                "resource" => AgentType::Resource,
                                "qos" => AgentType::Qos,
                                "security" => AgentType::Security,
                                "slicing" => AgentType::Slicing,
                                _ => AgentType::Custom,
                            };
                            let mut caps = AgentCapabilities::default();
                            caps.allowed_intents = capabilities.clone();
                            self.coordinator.register_agent(a_id, a_type, caps);
                            self.agents.insert(agent_id, capabilities);
                        }
                        AgentMessage::SubmitIntent { agent_id, intent_type, parameters } => {
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
                        AgentMessage::CoordinationEvent { event_type, agent_ids } => {
                            debug!("Agent: Coordination event '{}' with {} agents", event_type, agent_ids.len());
                            // Process any pending intents queued during this coordination cycle
                            let results = self.coordinator.process_intents();
                            if !results.is_empty() {
                                debug!("Agent: Processed {} intents after coordination event '{}'", results.len(), event_type);
                            }
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("Agent task stopped, {} registered agents", self.agents.len());
    }
}
