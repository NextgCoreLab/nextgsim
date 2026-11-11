//! Task wiring for the UE: build the task graph, then spawn it (#452).
//!
//! `nr-ue` and the integration tests both build the task graph here, so a test
//! exercises the wiring the binary runs rather than a copy of it.
//!
//! # Why the order matters
//!
//! Every task gets its own clone of [`UeTaskBase`], and the optional 6G and
//! Rel-18 handles in it are a by-value `Option`. A clone taken before
//! `init_6g_tasks` keeps `sixg: None` for the life of the task. The binary used
//! to clone the RRC base first and initialize the 6G handles afterwards, on
//! another clone, so nothing the RRC task routed ever reached the SHE client,
//! ISAC sensor or semantic codec, and the [`TaskManager`]'s `shutdown` never
//! reached a 6G task.
//!
//! [`UeWiring::new`] initializes every optional handle on the manager's own
//! base **before** anything can clone it.
//!
//! # What stays in the binary
//!
//! The NAS task loop and the TUN forwarding are binary code (`main.rs`), so
//! [`UeWiring::spawn`] returns the NAS receiver for the binary to run. The task
//! base the binary gives them comes from [`UeWiring::task_base`], which is only
//! reachable after every handle is in place.

use tokio::sync::mpsc;
use tracing::{info, warn};

use nextgsim_common::config::UeConfig;

use crate::app::AppTask;
use crate::mint::MintTask;
use crate::ranging::RangingTask;
use crate::rls::RlsTask;
use crate::rrc::RrcTask;
#[cfg(any(
    feature = "nextgsim-she",
    feature = "nextgsim-nwdaf",
    feature = "nextgsim-isac",
    feature = "nextgsim-fl",
    feature = "nextgsim-semantic",
))]
use crate::tasks::UeSixgReceivers;
use crate::tasks::{
    AppMessage, NasMessage, RlsMessage, RrcMessage, Task, TaskManager, TaskMessage,
    UeRel18Receivers, UeTaskBase,
};

/// Receivers for the four core UE tasks.
pub struct UeCoreReceivers {
    pub app_rx: mpsc::Receiver<TaskMessage<AppMessage>>,
    pub nas_rx: mpsc::Receiver<TaskMessage<NasMessage>>,
    pub rrc_rx: mpsc::Receiver<TaskMessage<RrcMessage>>,
    pub rls_rx: mpsc::Receiver<TaskMessage<RlsMessage>>,
}

/// The UE task graph, built but not yet running.
///
/// The fields are public so a test can put a probe in front of a receiver before
/// [`UeWiring::spawn`] hands it to its task. The binary does not touch them.
pub struct UeWiring {
    /// The manager; its task base already carries every optional handle.
    pub task_manager: TaskManager,
    /// Receivers for the core tasks.
    pub core: UeCoreReceivers,
    /// Rel-18 receivers (Ranging, MINT, and Sidelink when compiled in).
    pub rel18: UeRel18Receivers,
    /// 6G receivers, `Some` when at least one compiled-in 6G task is enabled.
    #[cfg(any(
        feature = "nextgsim-she",
        feature = "nextgsim-nwdaf",
        feature = "nextgsim-isac",
        feature = "nextgsim-fl",
        feature = "nextgsim-semantic",
    ))]
    pub sixg: Option<UeSixgReceivers>,
}

/// Whether any 6G task is both compiled in and enabled by `config`. A flag
/// without its feature is a no-op.
#[cfg(any(
    feature = "nextgsim-she",
    feature = "nextgsim-nwdaf",
    feature = "nextgsim-isac",
    feature = "nextgsim-fl",
    feature = "nextgsim-semantic",
))]
fn any_6g_enabled(config: &UeConfig) -> bool {
    cfg!(feature = "nextgsim-she") && config.she_enabled
        || cfg!(feature = "nextgsim-nwdaf") && config.ai_ml_enabled
        || cfg!(feature = "nextgsim-isac") && config.isac_enabled
        || cfg!(feature = "nextgsim-fl") && config.federated_learning_enabled
        || cfg!(feature = "nextgsim-semantic") && config.semantic_comm_enabled
}

impl UeWiring {
    /// Builds the task graph for `config`: the manager, the core and Rel-18
    /// channels, and the 6G channels the config and features ask for.
    pub fn new(config: UeConfig, channel_capacity: usize) -> Self {
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        let sixg = any_6g_enabled(&config);

        let (mut task_manager, app_rx, nas_rx, rrc_rx, rls_rx) =
            TaskManager::new(config, channel_capacity);

        // All before any clone of the task base exists (see the module docs).
        let rel18 = task_manager.init_rel18_tasks(channel_capacity);
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        let sixg = sixg.then(|| task_manager.init_6g_tasks(channel_capacity));

        Self {
            task_manager,
            core: UeCoreReceivers {
                app_rx,
                nas_rx,
                rrc_rx,
                rls_rx,
            },
            rel18,
            #[cfg(any(
                feature = "nextgsim-she",
                feature = "nextgsim-nwdaf",
                feature = "nextgsim-isac",
                feature = "nextgsim-fl",
                feature = "nextgsim-semantic",
            ))]
            sixg,
        }
    }

    /// A clone of the fully initialized task base.
    pub fn task_base(&self) -> UeTaskBase {
        self.task_manager.task_base()
    }

    /// Spawns the App, RRC, RLS, Rel-18 and 6G tasks, and returns the manager
    /// and the NAS receiver, whose loop the binary runs.
    ///
    /// `cli_node_name` is the name the CLI server registers; `None` skips the
    /// CLI server, as `nr-ue -l` does. Must be called from inside a Tokio
    /// runtime.
    pub fn spawn(
        self,
        cli_node_name: Option<String>,
    ) -> (TaskManager, mpsc::Receiver<TaskMessage<NasMessage>>) {
        let Self {
            task_manager,
            core,
            rel18,
            #[cfg(any(
                feature = "nextgsim-she",
                feature = "nextgsim-nwdaf",
                feature = "nextgsim-isac",
                feature = "nextgsim-fl",
                feature = "nextgsim-semantic",
            ))]
            sixg,
        } = self;
        let task_base = task_manager.task_base();

        // App task (status updates and CLI commands)
        let app_task_base = task_base.clone();
        let app_rx = core.app_rx;
        tokio::spawn(async move {
            let mut app_task = match cli_node_name {
                None => AppTask::new_without_cli(app_task_base),
                Some(node_name) => {
                    let mut task = AppTask::new(app_task_base);
                    match task.init_cli_server(vec![node_name]).await {
                        Ok(port) => {
                            if port > 0 {
                                info!("CLI server listening on port {}", port);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to initialize CLI server: {}", e);
                        }
                    }
                    task
                }
            };
            app_task.run(app_rx).await;
        });
        info!("App task spawned");

        // RRC task: the conformant library RrcTask (issue #30). It runs real RRC
        // connection establishment (RRCSetupRequest on SRB0/CCCH → RRCSetup →
        // RRCSetupComplete on SRB1/DCCH, TS 38.331 §5.3.3), cell selection, UAC,
        // re-establishment, resume and AS-security.
        spawn_task(RrcTask::new(task_base.clone()), core.rrc_rx);
        info!("RRC task spawned (library RrcTask)");

        // Rel-18 5G-Advanced tasks (Ranging TS 23.586, MINT TS 23.761, Sidelink).
        let UeRel18Receivers {
            ranging_rx,
            mint_rx,
            #[cfg(feature = "sidelink")]
            sidelink_rx,
        } = rel18;
        spawn_task(RangingTask::new(task_base.clone()), ranging_rx);
        info!("{}", crate::ranging::SPAWN_LOG);
        spawn_task(MintTask::new(task_base.clone()), mint_rx);
        info!("MINT task spawned (Rel-18, TS 23.761)");
        // Issue #54: a default build neither spawns the sidelink task nor logs
        // anything about it.
        #[cfg(feature = "sidelink")]
        {
            spawn_task(
                crate::sidelink::SidelinkTask::new(task_base.clone()),
                sidelink_rx,
            );
            info!("{}", crate::sidelink::SPAWN_LOG);
        }

        // 6G AI-native network function tasks (Rel-20)
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        if let Some(sixg) = sixg {
            spawn_6g_tasks(&task_base, sixg);
        }

        // RLS task (cell search and gNB communication)
        spawn_task(RlsTask::from_ue_config(task_base), core.rls_rx);
        info!("RLS task spawned");

        (task_manager, core.nas_rx)
    }
}

/// Spawns `task` on its receiver.
fn spawn_task<T>(mut task: T, rx: mpsc::Receiver<TaskMessage<T::Message>>)
where
    T: Task,
    T::Message: 'static,
{
    tokio::spawn(async move { task.run(rx).await });
}

/// Spawns the enabled, compiled-in 6G tasks.
#[cfg(any(
    feature = "nextgsim-she",
    feature = "nextgsim-nwdaf",
    feature = "nextgsim-isac",
    feature = "nextgsim-fl",
    feature = "nextgsim-semantic",
))]
fn spawn_6g_tasks(task_base: &UeTaskBase, sixg: UeSixgReceivers) {
    let config = &task_base.config;

    #[cfg(feature = "nextgsim-she")]
    if config.she_enabled {
        spawn_task(
            crate::she_client::SheClientTask::new(task_base.clone()),
            sixg.she_client_rx,
        );
        info!("SHE Client task spawned");
    }

    #[cfg(feature = "nextgsim-nwdaf")]
    if config.ai_ml_enabled {
        spawn_task(
            crate::nwdaf_reporter::NwdafReporterTask::new(task_base.clone()),
            sixg.nwdaf_reporter_rx,
        );
        info!("NWDAF Reporter task spawned");
    }

    #[cfg(feature = "nextgsim-isac")]
    if config.isac_enabled {
        spawn_task(
            crate::isac_sensor::IsacSensorTask::new(task_base.clone()),
            sixg.isac_sensor_rx,
        );
        info!("ISAC Sensor task spawned");
    }

    #[cfg(feature = "nextgsim-fl")]
    if config.federated_learning_enabled {
        spawn_task(
            crate::fl_participant::FlParticipantTask::new(task_base.clone()),
            sixg.fl_participant_rx,
        );
        info!("FL Participant task spawned");
    }

    #[cfg(feature = "nextgsim-semantic")]
    if config.semantic_comm_enabled {
        spawn_task(
            crate::semantic_codec::SemanticCodecTask::new(task_base.clone()),
            sixg.semantic_codec_rx,
        );
        info!("Semantic Codec task spawned");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::DEFAULT_CHANNEL_CAPACITY;

    /// A default config builds the core and Rel-18 channels and no 6G ones, and
    /// every handed-out base carries the Rel-18 handles (#452).
    #[test]
    fn a_default_config_builds_no_6g_channels() {
        let wiring = UeWiring::new(UeConfig::default(), DEFAULT_CHANNEL_CAPACITY);
        let base = wiring.task_base();
        assert!(base.rel18.is_some(), "Rel-18 handles are always installed");
        #[cfg(any(
            feature = "nextgsim-she",
            feature = "nextgsim-nwdaf",
            feature = "nextgsim-isac",
            feature = "nextgsim-fl",
            feature = "nextgsim-semantic",
        ))]
        {
            assert!(wiring.sixg.is_none());
            assert!(base.sixg.is_none());
        }
    }

    /// With a 6G task enabled, every handed-out base carries the 6G handles.
    #[cfg(feature = "nextgsim-she")]
    #[test]
    fn the_task_base_carries_the_6g_handles() {
        let config = UeConfig {
            she_enabled: true,
            ..UeConfig::default()
        };
        let wiring = UeWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
        assert!(wiring.sixg.is_some());
        assert!(wiring.task_base().sixg.is_some());
    }
}
