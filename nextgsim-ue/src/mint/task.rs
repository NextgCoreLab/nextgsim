//! MINT Task for UE - Multi-IMSI terminal support
//!
//! Implements Rel-18 MINT per TS 23.761:
//! - Multiple IMSI/SUPI per UE (dual-SIM / multi-USIM)
//! - Per-IMSI NAS context management
//! - IMSI selection for outgoing calls/sessions
//! - Simultaneous registration on multiple PLMNs

use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::tasks::{MintMessage, Task, TaskMessage, UeTaskBase};

/// NAS context state for a single subscription.
#[derive(Debug, Clone)]
struct SubscriptionNasContext {
    /// Subscription index (0 = primary)
    index: u8,
    /// SUPI for this subscription
    supi: String,
    /// Whether this subscription is registered
    registered: bool,
    /// Serving PLMN MCC
    serving_mcc: Option<u16>,
    /// Serving PLMN MNC
    serving_mnc: Option<u16>,
    /// 5G-GUTI if assigned
    guti: Option<String>,
    /// Active PDU session IDs on this subscription
    active_sessions: Vec<u8>,
}

impl SubscriptionNasContext {
    fn new(index: u8, supi: &str) -> Self {
        Self {
            index,
            supi: supi.to_string(),
            registered: false,
            serving_mcc: None,
            serving_mnc: None,
            guti: None,
            active_sessions: Vec::new(),
        }
    }
}

pub struct MintTask {
    _task_base: UeTaskBase,
    /// Per-subscription NAS contexts
    subscriptions: Vec<SubscriptionNasContext>,
    /// Index of the currently active subscription
    active_index: u8,
    /// Whether simultaneous registration is allowed
    simultaneous_registration: bool,
}

impl MintTask {
    pub fn new(task_base: UeTaskBase) -> Self {
        // Initialize primary subscription from config
        let primary_supi = task_base
            .config
            .supi
            .as_ref()
            .map(std::string::ToString::to_string)
            .unwrap_or_else(|| "imsi-000000000000000".to_string());

        let mut subscriptions = vec![SubscriptionNasContext::new(0, &primary_supi)];

        // Add secondary subscriptions from MINT config
        if let Some(ref mint_config) = task_base.config.mint_config {
            for (i, secondary_supi) in mint_config.secondary_supis.iter().enumerate() {
                subscriptions.push(SubscriptionNasContext::new(
                    (i + 1) as u8,
                    secondary_supi,
                ));
            }
        }

        let simultaneous = task_base
            .config
            .mint_config
            .as_ref()
            .is_some_and(|c| c.simultaneous_registration);

        Self {
            _task_base: task_base,
            subscriptions,
            active_index: 0,
            simultaneous_registration: simultaneous,
        }
    }

    fn _active_subscription(&self) -> Option<&SubscriptionNasContext> {
        self.subscriptions
            .iter()
            .find(|s| s.index == self.active_index)
    }

    fn select_subscription_for_dnn(&self, dnn: &str) -> u8 {
        // Simple DNN-based selection: if any subscription is registered
        // and has fewer active sessions, prefer it
        let mut best_index = self.active_index;
        let mut min_sessions = usize::MAX;

        for sub in &self.subscriptions {
            if sub.registered && sub.active_sessions.len() < min_sessions {
                min_sessions = sub.active_sessions.len();
                best_index = sub.index;
            }
        }

        debug!(
            "MINT: Selected subscription {} for DNN '{}'",
            best_index, dnn
        );
        best_index
    }
}

#[async_trait::async_trait]
impl Task for MintTask {
    type Message = MintMessage;

    async fn run(&mut self, mut rx: mpsc::Receiver<TaskMessage<Self::Message>>) {
        info!(
            "MINT task started with {} subscriptions (simultaneous={})",
            self.subscriptions.len(),
            self.simultaneous_registration
        );
        loop {
            match rx.recv().await {
                Some(TaskMessage::Message(msg)) => match msg {
                    MintMessage::SwitchSubscription { index } => {
                        if (index as usize) < self.subscriptions.len() {
                            let old = self.active_index;
                            self.active_index = index;
                            debug!("MINT: Switched active subscription {} -> {}", old, index);
                        } else {
                            warn!(
                                "MINT: Invalid subscription index {} (max={})",
                                index,
                                self.subscriptions.len() - 1
                            );
                        }
                    }
                    MintMessage::RegistrationUpdate {
                        subscription_index,
                        registered,
                        serving_plmn,
                        guti,
                    } => {
                        if let Some(sub) = self
                            .subscriptions
                            .iter_mut()
                            .find(|s| s.index == subscription_index)
                        {
                            sub.registered = registered;
                            if let Some((mcc, mnc)) = serving_plmn {
                                sub.serving_mcc = Some(mcc);
                                sub.serving_mnc = Some(mnc);
                            }
                            sub.guti = guti;
                            debug!(
                                "MINT: Subscription {} registration={} SUPI={}",
                                subscription_index, registered, sub.supi
                            );
                        }
                    }
                    MintMessage::SelectForSession { dnn, response_tx } => {
                        let selected = self.select_subscription_for_dnn(&dnn);
                        let supi = self
                            .subscriptions
                            .iter()
                            .find(|s| s.index == selected)
                            .map(|s| s.supi.clone())
                            .unwrap_or_default();
                        if let Some(tx) = response_tx {
                            let _ = tx.send((selected, supi));
                        }
                    }
                    MintMessage::SessionUpdate {
                        subscription_index,
                        psi,
                        active,
                    } => {
                        if let Some(sub) = self
                            .subscriptions
                            .iter_mut()
                            .find(|s| s.index == subscription_index)
                        {
                            if active {
                                if !sub.active_sessions.contains(&psi) {
                                    sub.active_sessions.push(psi);
                                }
                            } else {
                                sub.active_sessions.retain(|&s| s != psi);
                            }
                            debug!(
                                "MINT: Subscription {} session PSI={} active={} total={}",
                                subscription_index,
                                psi,
                                active,
                                sub.active_sessions.len()
                            );
                        }
                    }
                    MintMessage::GetStatus { response_tx } => {
                        let status: Vec<(u8, String, bool, usize)> = self
                            .subscriptions
                            .iter()
                            .map(|s| {
                                (
                                    s.index,
                                    s.supi.clone(),
                                    s.registered,
                                    s.active_sessions.len(),
                                )
                            })
                            .collect();
                        if let Some(tx) = response_tx {
                            let _ = tx.send(status);
                        }
                    }
                },
                Some(TaskMessage::Shutdown) => break,
                None => break,
            }
        }
        info!("MINT task stopped");
    }
}
