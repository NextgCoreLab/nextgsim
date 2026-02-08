//! Network Knowledge Exposure Function (NKEF) Task for gNB

use std::collections::HashMap;
use tokio::sync::mpsc;
use tracing::{debug, info};

use nextgsim_nkef::{Entity, EntityType, KnowledgeGraph};

use crate::tasks::{GnbTaskBase, NkefMessage, Task, TaskMessage};

/// NKEF Task for gNB
pub struct NkefTask {
    _task_base: GnbTaskBase,
    knowledge_graph: KnowledgeGraph,
    entity_count: HashMap<String, usize>,
}

impl NkefTask {
    pub fn new(task_base: GnbTaskBase) -> Self {
        Self {
            _task_base: task_base,
            knowledge_graph: KnowledgeGraph::new(),
            entity_count: HashMap::new(),
        }
    }

    fn handle_update_knowledge(
        &mut self,
        entity_type: String,
        entity_id: String,
        properties: Vec<(String, String)>,
    ) {
        debug!(
            "NKEF: Updating knowledge for {} '{}' with {} properties",
            entity_type, entity_id, properties.len()
        );

        let etype = match entity_type.as_str() {
            "cell" => EntityType::Cell,
            "ue" => EntityType::Ue,
            "gnb" => EntityType::Gnb,
            "slice" => EntityType::Slice,
            "amf" => EntityType::Amf,
            _ => EntityType::Service,
        };

        let mut entity = Entity::new(entity_id.clone(), etype);
        for (k, v) in &properties {
            entity = entity.with_property(k, v);
        }
        self.knowledge_graph.add_entity(entity);

        *self.entity_count.entry(entity_type.clone()).or_insert(0) += 1;

        info!(
            "NKEF: Updated {} '{}' (total {} entities of this type)",
            entity_type,
            entity_id,
            self.entity_count.get(&entity_type).unwrap_or(&0)
        );
    }

    fn handle_semantic_query(&mut self, query: String, max_results: u32) {
        debug!("NKEF: Semantic query '{}' (max_results={})", query, max_results);

        let results = self.knowledge_graph.search(&query, max_results as usize);

        info!("NKEF: Query '{}' returned {} results", query, results.len());
    }

    fn handle_retrieve_context(&mut self, prompt: String, max_tokens: u32) {
        debug!(
            "NKEF: RAG context retrieval for prompt '{}' (max_tokens={})",
            prompt, max_tokens
        );

        let context = self.knowledge_graph.generate_context(&prompt, max_tokens);

        info!(
            "NKEF: Retrieved context with {} sources for prompt",
            context.sources.len()
        );
    }
}

#[async_trait::async_trait]
impl Task for NkefTask {
    type Message = NkefMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("NKEF task started");

        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        NkefMessage::UpdateKnowledge {
                            entity_type, entity_id, properties,
                        } => {
                            self.handle_update_knowledge(entity_type, entity_id, properties);
                        }
                        NkefMessage::SemanticQuery { query, max_results } => {
                            self.handle_semantic_query(query, max_results);
                        }
                        NkefMessage::RetrieveContext { prompt, max_tokens } => {
                            self.handle_retrieve_context(prompt, max_tokens);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => {
                    info!("NKEF task received shutdown signal");
                    break;
                }
                None => {
                    info!("NKEF task channel closed");
                    break;
                }
            }
        }

        let total_entities: usize = self.entity_count.values().sum();
        info!("NKEF task stopped, {} total entities in knowledge graph", total_entities);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::{GnbTaskBase, DEFAULT_CHANNEL_CAPACITY};
    use nextgsim_common::config::GnbConfig;
    use nextgsim_common::Plmn;

    fn test_config() -> GnbConfig {
        GnbConfig {
            nci: 0x000000010,
            gnb_id_length: 32,
            plmn: Plmn::new(001, 01, false),
            tac: 1,
            nssai: vec![],
            amf_configs: vec![],
            link_ip: "127.0.0.1".parse().unwrap(),
            ngap_ip: "127.0.0.1".parse().unwrap(),
            gtp_ip: "127.0.0.1".parse().unwrap(),
            gtp_advertise_ip: None,
            ignore_stream_ids: false,
            upf_addr: None,
            upf_port: 2152,
            pqc_config: nextgsim_common::config::PqcConfig::default(),
        }
    }

    #[tokio::test]
    async fn test_nkef_task_creation() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let task = NkefTask::new(task_base);
        assert_eq!(task.entity_count.len(), 0);
    }

    #[tokio::test]
    async fn test_nkef_task_update_knowledge() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NkefTask::new(task_base);
        let properties = vec![
            ("name".to_string(), "cell-1".to_string()),
            ("load".to_string(), "0.5".to_string()),
        ];
        task.handle_update_knowledge("cell".to_string(), "1".to_string(), properties);
        assert_eq!(*task.entity_count.get("cell").unwrap(), 1);
    }

    #[tokio::test]
    async fn test_nkef_task_semantic_query() {
        let config = test_config();
        let (task_base, _app_rx, _ngap_rx, _rrc_rx, _gtp_rx, _rls_rx, _sctp_rx) =
            GnbTaskBase::new(config, DEFAULT_CHANNEL_CAPACITY);

        let mut task = NkefTask::new(task_base);
        task.handle_semantic_query("high load cells".to_string(), 10);
    }
}
