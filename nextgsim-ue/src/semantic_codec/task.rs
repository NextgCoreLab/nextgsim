//! Semantic Codec Task for UE

use tokio::sync::mpsc;
use tracing::{debug, info};
use crate::tasks::{UeTaskBase, SemanticCodecMessage, Task, TaskMessage};

pub struct SemanticCodecTask {
    _task_base: UeTaskBase,
}

impl SemanticCodecTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        Self { _task_base: task_base }
    }
}

#[async_trait::async_trait]
impl Task for SemanticCodecTask {
    type Message = SemanticCodecMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!("Semantic Codec task started");
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => {
                    match msg {
                        SemanticCodecMessage::Encode { task_type, data: _, dimensions: _, channel_quality: _, response_tx: _ } => {
                            debug!("Semantic Codec: Encode {:?}", task_type);
                        }
                        SemanticCodecMessage::Decode { task_type, features: _, importance: _, original_dims: _, response_tx: _ } => {
                            debug!("Semantic Codec: Decode {:?}", task_type);
                        }
                        SemanticCodecMessage::UpdateEncoder { model_id } => {
                            debug!("Semantic Codec: Update encoder {}", model_id);
                        }
                        SemanticCodecMessage::UpdateDecoder { model_id } => {
                            debug!("Semantic Codec: Update decoder {}", model_id);
                        }
                        SemanticCodecMessage::SetAdaptiveCompression { enabled, min_quality: _, target_compression: _ } => {
                            debug!("Semantic Codec: Adaptive compression={}", enabled);
                        }
                    }
                }
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("Semantic Codec task stopped");
    }
}
