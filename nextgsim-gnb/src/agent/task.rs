//! AI Agent Framework Task for gNB

use crate::tasks::{
    AgentMessage, GnbTaskBase, InboundHandoverVerdict, RrcMessage, Task, TaskMessage,
};
#[cfg(feature = "demo-d446a-agent-conflict")]
use crate::tasks::{HandoverTrigger, RlsMessage};
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
    /// Demo D446A's agents (nextgsim#483), when the config names any.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    d446a: Option<super::d446a::D446a>,
    /// Where D446A's audit lives.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    d446a_audit: super::d446a::Audit,
    /// Reads D446A's current mode; `None` when the demo is not active.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    d446a_mode: fn() -> Option<String>,
    /// The UEs with an RRC context on the cell, as the RRC task last reported
    /// them (`AgentMessage::ConnectedUes`): the energy guardrail's input.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    d446a_connected: Vec<i32>,
}

#[cfg(feature = "demo-d446a-agent-conflict")]
fn installed_d446a_mode() -> Option<String> {
    nextgsim_demo::active_mode(crate::demo::D446A_KEY)
}

impl AgentTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        #[cfg(feature = "demo-d446a-agent-conflict")]
        let d446a_audit = super::d446a::Audit::default();
        #[allow(unused_mut)]
        let mut task = Self {
            #[cfg(feature = "demo-d446a-agent-conflict")]
            d446a: super::d446a::D446a::from_config(&task_base.config.agents, d446a_audit.clone()),
            #[cfg(feature = "demo-d446a-agent-conflict")]
            d446a_audit,
            #[cfg(feature = "demo-d446a-agent-conflict")]
            d446a_mode: installed_d446a_mode,
            #[cfg(feature = "demo-d446a-agent-conflict")]
            d446a_connected: Vec::new(),
            task_base,
            coordinator: AgentCoordinator::new(),
            agents: HashMap::new(),
            observations: 0,
        };
        #[cfg(feature = "demo-d446a-agent-conflict")]
        {
            let regs = task
                .d446a
                .as_ref()
                .map(|d| d.registrations())
                .unwrap_or_default();
            for (id, caps) in regs {
                let kinds: Vec<String> = Vec::new();
                task.coordinator
                    .register_agent(AgentId::new(&id), AgentType::Resource, caps);
                task.agents.insert(id, kinds);
            }
        }
        task
    }

    /// Makes this task's D446A audit the one `nr-cli agent-audit` shows
    /// (production wiring only; nextgsim#483).
    pub fn install_cli(&self) {
        #[cfg(feature = "demo-d446a-agent-conflict")]
        super::d446a::D446a::install_cli_audit(&self.d446a_audit);
    }

    /// Overrides where D446A's mode is read (tests).
    #[cfg(all(test, feature = "demo-d446a-agent-conflict"))]
    fn with_d446a_mode(mut self, mode: fn() -> Option<String>) -> Self {
        self.d446a_mode = mode;
        self
    }

    /// D446A's tick: the decisions of this gNB's demo agents, then their
    /// actuation (nextgsim#483).
    #[cfg(feature = "demo-d446a-agent-conflict")]
    async fn d446a_tick(&mut self, served_ue_ids: &[i32]) {
        let mode = (self.d446a_mode)();
        let Some(d) = self.d446a.as_mut() else {
            return;
        };
        let actions = d.on_tick(
            std::time::Instant::now(),
            served_ue_ids,
            &self.d446a_connected,
            mode.as_deref(),
        );
        self.d446a_act(actions).await;
    }

    #[cfg(feature = "demo-d446a-agent-conflict")]
    async fn d446a_act(&mut self, actions: Vec<super::d446a::Action>) {
        use super::d446a::Action;
        for action in actions {
            match action {
                Action::SwitchOff => {
                    if let Err(e) = self.task_base.rls_tx.send(RlsMessage::SetOffAir).await {
                        warn!("Agent: switch-off not delivered to RLS: {e}");
                    }
                    if let Err(e) = self
                        .task_base
                        .rrc_tx
                        .send(RrcMessage::EnergySwitchOff)
                        .await
                    {
                        warn!("Agent: switch-off not delivered to RRC: {e}");
                    }
                }
                Action::Handover(intent) => {
                    let id = intent.id.clone();
                    let ue = intent.target.clone().unwrap_or_default();
                    if let Err(e) = self.coordinator.submit_intent(intent) {
                        if let Some(d) = self.d446a.as_mut() {
                            d.on_actuation(&id, ue.parse().unwrap_or(-1), false, false, &e);
                        }
                        continue;
                    }
                    // Executed only from the conflict-checked queue (#483).
                    for result in self.coordinator.process_intents_full() {
                        self.forward_agent_handover(&result).await;
                    }
                }
            }
        }
    }

    /// Hands an executed handover intent to RRC with its cause and intent id.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    async fn forward_agent_handover(&mut self, result: &nextgsim_agent::IntentExecutionResult) {
        if result.status != nextgsim_agent::IntentStatus::Success {
            if let Some(d) = self.d446a.as_mut() {
                d.on_actuation(
                    &result.intent_id,
                    -1,
                    false,
                    false,
                    result.message.as_deref().unwrap_or("not executed"),
                );
            }
            return;
        }
        let out = &result.output;
        let (Some(ue_id), Some(target_cell)) = (
            out.get("ue_id").and_then(|v| v.parse::<i32>().ok()),
            out.get("target_cell").and_then(|v| v.parse::<i32>().ok()),
        ) else {
            return;
        };
        let mut trigger = HandoverTrigger {
            agent_intent_id: Some(result.intent_id.clone()),
            ..Default::default()
        };
        if out.get("cause").map(String::as_str) == Some(super::d446a::CAUSE_REDUCE_LOAD) {
            trigger.cause = nextgsim_ngap::procedures::handover::HandoverCause::RadioNetwork(
                nextgsim_ngap::procedures::ng_setup::RadioNetworkCause::ReduceLoadInServingCell,
            );
        }
        let msg = RrcMessage::AgentHandover {
            ue_id,
            target_cell,
            trigger,
        };
        if let Err(e) = self.task_base.rrc_tx.send(msg).await {
            warn!("Agent: handover action not delivered to RRC: {e}");
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
                            served_ue_ids,
                        } => {
                            self.record_cell_observation(cell_id, connected_ues, throughput_mbps);
                            #[cfg(feature = "demo-d446a-agent-conflict")]
                            self.d446a_tick(&served_ue_ids).await;
                            #[cfg(not(feature = "demo-d446a-agent-conflict"))]
                            let _ = served_ue_ids;
                        }
                        AgentMessage::ConnectedUes { ue_ids } => {
                            #[cfg(feature = "demo-d446a-agent-conflict")]
                            {
                                self.d446a_connected = ue_ids;
                            }
                            #[cfg(not(feature = "demo-d446a-agent-conflict"))]
                            let _ = ue_ids;
                        }
                        AgentMessage::InboundHandover {
                            cause,
                            load_balancing,
                            source_cell,
                            source_gnb_id,
                            target_cell,
                            reply,
                        } => {
                            #[cfg(feature = "demo-d446a-agent-conflict")]
                            {
                                let mode = (self.d446a_mode)();
                                let req = super::d446a::Inbound {
                                    cause,
                                    load_balancing,
                                    source_cell,
                                    source_gnb_id,
                                    target_cell,
                                };
                                let (verdict, actions) = match self.d446a.as_mut() {
                                    Some(d) => d.arbitrate(
                                        std::time::Instant::now(),
                                        &req,
                                        mode.as_deref(),
                                    ),
                                    None => (
                                        InboundHandoverVerdict::Admit {
                                            handback_hold: None,
                                        },
                                        Vec::new(),
                                    ),
                                };
                                let _ = reply.send(verdict);
                                self.d446a_act(actions).await;
                            }
                            #[cfg(not(feature = "demo-d446a-agent-conflict"))]
                            {
                                let _ = (
                                    cause,
                                    load_balancing,
                                    source_cell,
                                    source_gnb_id,
                                    target_cell,
                                );
                                let _ = reply.send(InboundHandoverVerdict::Admit {
                                    handback_hold: None,
                                });
                            }
                        }
                        AgentMessage::ActuationResult {
                            intent_id,
                            ue_id,
                            success,
                            signalled,
                            detail,
                        } => {
                            #[cfg(feature = "demo-d446a-agent-conflict")]
                            if let Some(d) = self.d446a.as_mut() {
                                // Nothing reached the network: lb-1 may try the UE
                                // again at a later tick.
                                let retry_ue = !success && !signalled;
                                d.on_actuation(&intent_id, ue_id, success, retry_ue, &detail);
                            }
                            #[cfg(not(feature = "demo-d446a-agent-conflict"))]
                            let _ = (intent_id, ue_id, success, signalled, detail);
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
            served_ue_ids: vec![],
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Message(AgentMessage::CellObservation {
            cell_id: 8,
            connected_ues: 0,
            throughput_mbps: None,
            served_ue_ids: vec![],
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

    /// #483: on a loaded cell, lb-1's intent is executed from the queue and
    /// reaches RRC as an AgentHandover with the load-balancing cause and the
    /// intent id, so NGAP can carry the cause and report the outcome.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    #[tokio::test]
    async fn a_loaded_cell_actuates_lb_1_with_the_load_balancing_cause() {
        use nextgsim_common::config::{AgentRuleConfig, AgentRuleKind};
        let config = GnbConfig {
            agents: vec![AgentRuleConfig {
                id: "lb-1".into(),
                kind: AgentRuleKind::LoadBalancing,
                cell: 0x10,
                priority: None,
                load_threshold_ues: 1,
                grace_s: 20,
                handback_hold_s: 60,
                peers: vec![],
            }],
            ..GnbConfig::default()
        };
        let (task_base, _app_rx, _ngap_rx, mut rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);
        let mut task =
            AgentTask::new(task_base).with_d446a_mode(|| Some("load-balancing".to_string()));
        let (tx, rx) = mpsc::channel(4);
        tx.send(TaskMessage::Message(AgentMessage::CellObservation {
            cell_id: 0,
            connected_ues: 1,
            throughput_mbps: None,
            served_ue_ids: vec![1],
        }))
        .await
        .unwrap();
        tx.send(TaskMessage::Shutdown).await.unwrap();
        task.run(rx).await;

        let Some(TaskMessage::Message(RrcMessage::AgentHandover {
            ue_id,
            target_cell,
            trigger,
        })) = rrc_rx.try_recv().ok()
        else {
            panic!("lb-1's handover must reach RRC");
        };
        assert_eq!((ue_id, target_cell), (1, 16));
        assert_eq!(
            trigger.cause,
            nextgsim_ngap::procedures::handover::HandoverCause::RadioNetwork(
                nextgsim_ngap::procedures::ng_setup::RadioNetworkCause::ReduceLoadInServingCell
            )
        );
        assert!(trigger.agent_intent_id.is_some());
    }
}
