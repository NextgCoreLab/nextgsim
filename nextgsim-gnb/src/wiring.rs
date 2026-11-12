//! Task wiring for the gNB: build the task graph, then spawn it (#452).
//!
//! This is the one place the task graph is assembled. `nr-gnb` calls it, and so
//! do the integration tests, so a test exercises the wiring the binary runs
//! rather than a copy of it.
//!
//! # Why the order matters
//!
//! Every task gets its own clone of [`GnbTaskBase`], and the optional 6G and
//! Rel-18 handles in it are a by-value `Option`. A clone taken before
//! `init_6g_tasks` / `init_rel18_tasks` keeps `None` for the life of the task.
//! The binary used to clone the RRC and RLS bases first and initialize the
//! handles afterwards, on yet another clone, so:
//!
//! - nothing the RLS or RRC task sent ever reached a 6G task;
//! - the [`TaskManager`]'s own base never had the handles either, so
//!   `shutdown` never reached the 6G or Energy tasks.
//!
//! [`GnbWiring::new`] initializes every optional handle on the manager's own
//! base **before** anything can clone it; the only way to get a base out of a
//! `GnbWiring` is after that.

use tokio::sync::mpsc;
use tracing::{info, warn};

use nextgsim_common::config::GnbConfig;

use crate::app::AppTask;
use crate::energy::EnergyTask;
use crate::gtp::GtpTask;
use crate::ngap::NgapTask;
use crate::rls::RlsTask;
use crate::rrc::RrcTask;
use crate::sctp::SctpTask;
use crate::tasks::{
    AppMessage, GnbRel18Receivers, GnbSixgReceivers, GnbTaskBase, GtpMessage, NgapMessage,
    RlsMessage, RrcMessage, SctpMessage, Task, TaskError, TaskManager, TaskMessage,
};

/// Receivers for the six core gNB tasks.
pub struct GnbCoreReceivers {
    pub app_rx: mpsc::Receiver<TaskMessage<AppMessage>>,
    pub ngap_rx: mpsc::Receiver<TaskMessage<NgapMessage>>,
    pub rrc_rx: mpsc::Receiver<TaskMessage<RrcMessage>>,
    pub gtp_rx: mpsc::Receiver<TaskMessage<GtpMessage>>,
    pub rls_rx: mpsc::Receiver<TaskMessage<RlsMessage>>,
    pub sctp_rx: mpsc::Receiver<TaskMessage<SctpMessage>>,
}

/// The gNB task graph, built but not yet running.
///
/// The fields are public so a test can put a probe in front of a receiver before
/// [`GnbWiring::spawn`] hands it to its task. The binary does not touch them.
pub struct GnbWiring {
    /// The manager; its task base already carries every optional handle.
    pub task_manager: TaskManager,
    /// Receivers for the core tasks.
    pub core: GnbCoreReceivers,
    /// Rel-18 receivers, `Some` when `energy_saving_enabled`.
    pub rel18: Option<GnbRel18Receivers>,
    /// 6G receivers, `Some` when at least one compiled-in 6G task is enabled.
    pub sixg: Option<GnbSixgReceivers>,
}

/// Whether any 6G task is both compiled in and enabled by `config`. A `*_enabled`
/// flag without its feature is a no-op, mirroring the UE side.
fn any_6g_enabled(config: &GnbConfig) -> bool {
    cfg!(feature = "nextgsim-she") && config.she_enabled
        || cfg!(feature = "nextgsim-nwdaf") && config.nwdaf_enabled
        || cfg!(feature = "nextgsim-nkef") && config.nkef_enabled
        || cfg!(feature = "nextgsim-isac") && config.isac_enabled
        || cfg!(feature = "nextgsim-agent") && config.agent_enabled
}

impl GnbWiring {
    /// Builds the task graph for `config`: the manager, the core channels, and
    /// the Rel-18 and 6G channels the config and features ask for.
    ///
    /// With every 6G flag and `energy_saving_enabled` off, as in the default
    /// config, this builds exactly the six core channels it always did.
    pub fn new(config: GnbConfig, channel_capacity: usize) -> Self {
        let energy_saving = config.energy_saving_enabled;
        let sixg = any_6g_enabled(&config);

        let (mut task_manager, app_rx, ngap_rx, rrc_rx, gtp_rx, rls_rx, sctp_rx) =
            TaskManager::new(config, channel_capacity);

        // Both before any clone of the task base exists (see the module docs).
        let rel18 = energy_saving.then(|| task_manager.init_rel18_tasks(channel_capacity));
        let sixg = sixg.then(|| task_manager.init_6g_tasks(channel_capacity));

        Self {
            task_manager,
            core: GnbCoreReceivers {
                app_rx,
                ngap_rx,
                rrc_rx,
                gtp_rx,
                rls_rx,
                sctp_rx,
            },
            rel18,
            sixg,
        }
    }

    /// A clone of the fully initialized task base.
    pub fn task_base(&self) -> GnbTaskBase {
        self.task_manager.task_base()
    }

    /// Spawns every task and returns the manager that shuts them down.
    ///
    /// `disable_cmd` skips the CLI server, as `nr-gnb -l` does. Must be called
    /// from inside a Tokio runtime.
    pub fn spawn(self, disable_cmd: bool) -> TaskManager {
        let Self {
            task_manager,
            core,
            rel18,
            sixg,
        } = self;
        let task_base = task_manager.task_base();

        // App task (status updates and CLI commands)
        let app_task_base = task_base.clone();
        let app_rx = core.app_rx;
        tokio::spawn(async move {
            let mut app_task = if disable_cmd {
                AppTask::new_without_cli(app_task_base)
            } else {
                let mut task = AppTask::new(app_task_base);
                match task.init_cli_server("gnb".to_string()).await {
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
            };
            app_task.run(app_rx).await;
            Ok::<(), TaskError>(())
        });
        info!("App task spawned");

        spawn_task(NgapTask::new(task_base.clone()), core.ngap_rx);
        info!("NGAP task spawned");
        spawn_task(RrcTask::new(task_base.clone()), core.rrc_rx);
        info!("RRC task spawned");
        spawn_task(GtpTask::new(task_base.clone()), core.gtp_rx);
        info!("GTP task spawned");
        spawn_task(RlsTask::new(task_base.clone()), core.rls_rx);
        info!("RLS task spawned");
        spawn_task(SctpTask::new(task_base.clone()), core.sctp_rx);
        info!("SCTP task spawned");

        // Rel-18 5G-Advanced tasks
        if let Some(rel18) = rel18 {
            spawn_task(EnergyTask::new(task_base.clone()), rel18.energy_rx);
            info!("Energy task spawned");
        }

        // 6G AI-native network function tasks (Rel-20). A receiver whose task is
        // not compiled in or not enabled is dropped here, which closes its
        // channel; producers check for that rather than logging a send failure.
        if let Some(sixg) = sixg {
            spawn_6g_tasks(&task_base, sixg);
        }

        task_manager
    }
}

/// Spawns `task` on its receiver.
fn spawn_task<T>(mut task: T, rx: mpsc::Receiver<TaskMessage<T::Message>>)
where
    T: Task,
    T::Message: 'static,
{
    tokio::spawn(async move {
        task.run(rx).await;
        Ok::<(), TaskError>(())
    });
}

/// Spawns the enabled, compiled-in 6G tasks.
#[allow(unused_variables)]
fn spawn_6g_tasks(task_base: &GnbTaskBase, sixg: GnbSixgReceivers) {
    let GnbSixgReceivers {
        she_rx,
        nwdaf_rx,
        nkef_rx,
        isac_rx,
        agent_rx,
    } = sixg;
    let config = &task_base.config;

    #[cfg(feature = "nextgsim-she")]
    if config.she_enabled {
        spawn_task(crate::she::SheTask::new(task_base.clone()), she_rx);
        info!("SHE task spawned");
    }

    #[cfg(feature = "nextgsim-nwdaf")]
    if config.nwdaf_enabled {
        spawn_task(crate::nwdaf::NwdafTask::new(task_base.clone()), nwdaf_rx);
        info!("NWDAF task spawned");
    }

    #[cfg(feature = "nextgsim-nkef")]
    if config.nkef_enabled {
        spawn_task(crate::nkef::NkefTask::new(task_base.clone()), nkef_rx);
        info!("NKEF task spawned");
    }

    #[cfg(feature = "nextgsim-isac")]
    if config.isac_enabled {
        spawn_task(crate::isac::IsacTask::new(task_base.clone()), isac_rx);
        info!("ISAC task spawned");
    }

    #[cfg(feature = "nextgsim-agent")]
    if config.agent_enabled {
        let agent = crate::agent::AgentTask::new(task_base.clone());
        agent.install_cli();
        spawn_task(agent, agent_rx);
        info!("Agent task spawned");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::DEFAULT_CHANNEL_CAPACITY;

    /// A default config builds the six core channels and nothing else, so the
    /// default build's task graph is the one it always was.
    #[test]
    fn a_default_config_builds_no_optional_channels() {
        let wiring = GnbWiring::new(GnbConfig::default(), DEFAULT_CHANNEL_CAPACITY);
        assert!(wiring.rel18.is_none());
        assert!(wiring.sixg.is_none());
        let base = wiring.task_base();
        assert!(base.rel18.is_none());
        assert!(base.sixg.is_none());
    }

    /// Every clone handed out carries the Rel-18 handles once energy saving is on.
    #[test]
    fn the_task_base_carries_the_rel18_handles() {
        let config = GnbConfig {
            energy_saving_enabled: true,
            ..GnbConfig::default()
        };
        let wiring = GnbWiring::new(config, DEFAULT_CHANNEL_CAPACITY);
        assert!(wiring.rel18.is_some());
        assert!(wiring.task_base().rel18.is_some());
    }
}
