//! AI Agent Framework Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info};
use nextgsim_agent::AgentCoordinator;
use crate::tasks::{GnbTaskBase, AgentMessage, Task, TaskMessage};

pub struct AgentTask {
    _task_base: GnbTaskBase,
    _coordinator: AgentCoordinator,
    agents: HashMap<String, Vec<String>>,
}

impl AgentTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            _coordinator: AgentCoordinator::new(),
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
                            self.agents.insert(agent_id, capabilities);
                        }
                        AgentMessage::SubmitIntent { agent_id, intent_type, parameters: _ } => {
                            debug!("Agent: Intent '{}' from {}", intent_type, agent_id);
                        }
                        AgentMessage::CoordinationEvent { event_type, agent_ids } => {
                            debug!("Agent: Coordination event '{}' with {} agents", event_type, agent_ids.len());
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
