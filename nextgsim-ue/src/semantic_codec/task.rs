//! Semantic Codec Task for UE

use crate::tasks::{SemanticCodecMessage, Task, TaskMessage, UeTaskBase};
use tokio::sync::mpsc;
use tracing::{debug, info};

pub struct SemanticCodecTask {
    _task_base: UeTaskBase,
}

impl SemanticCodecTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self {
            _task_base: task_base,
        }
    }
}

#[async_trait::async_trait]
impl Task for SemanticCodecTask {
    type Message = SemanticCodecMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("Semantic Codec task started");
        // Demo D444A (#477): the UL feature stream, only when the demo is
        // compiled in, enabled and configured.
        #[cfg(feature = "demo-d444a-ambr-adapt")]
        let d444a = super::d444a::start(self._task_base.config.semantic_stream.as_ref());
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    SemanticCodecMessage::Encode {
                        task_type,
                        data: _,
                        dimensions: _,
                        channel_quality: _,
                        response_tx: _,
                    } => {
                        debug!("Semantic Codec: Encode {:?}", task_type);
                    }
                    SemanticCodecMessage::Decode {
                        task_type,
                        features: _,
                        importance: _,
                        original_dims: _,
                        response_tx: _,
                    } => {
                        debug!("Semantic Codec: Decode {:?}", task_type);
                    }
                    SemanticCodecMessage::UpdateEncoder { model_id } => {
                        debug!("Semantic Codec: Update encoder {}", model_id);
                    }
                    SemanticCodecMessage::UpdateDecoder { model_id } => {
                        debug!("Semantic Codec: Update decoder {}", model_id);
                    }
                    SemanticCodecMessage::SessionAmbrChanged { psi, session_ambr } => {
                        debug!("Semantic Codec: PDU session {psi} Session-AMBR {session_ambr:?}");
                        #[cfg(feature = "demo-d444a-ambr-adapt")]
                        super::d444a::on_session_ambr(d444a.as_ref(), psi, &session_ambr);
                    }
                    SemanticCodecMessage::SetAdaptiveCompression {
                        enabled,
                        min_quality: _,
                        target_compression: _,
                    } => {
                        debug!("Semantic Codec: Adaptive compression={}", enabled);
                    }
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("Semantic Codec task stopped");
    }
}
