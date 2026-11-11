//! Demo D446A (nextgsim#483): a load-balancing agent and an energy-saving agent
//! with opposing intents on one booster cell, arbitrated by the booster into
//! exactly one N2 handover, or none.
//!
//! - **Coverage gNB (`load-balancing` agent, e.g. `lb-1`):** when the cell it
//!   serves carries at least `load_threshold_ues` UEs (load measured as a UE
//!   count), it submits a `TriggerHandover` intent for one served UE toward the
//!   booster `cell`, cause "Reduce load in serving cell" (TS 38.413 §9.3.1.2).
//!   The coordinator executes it from its conflict-checked queue and the result
//!   becomes an RRC `AgentHandover`; NGAP reports the outcome back
//!   (`ActuationSucceeded` / `ActuationFailed`).
//! - **Booster gNB (`energy-saving` agent, e.g. `energy-1`):** when its cell
//!   serves no UE it queues `OptimizeResources target=<cell> status=inactive`,
//!   Pending for `grace_s`. At the deadline it switches the cell off the air,
//!   unless the `SwitchOffWithConnectedUes` guardrail holds it (a UE is served).
//! - **Admission control at the booster** (TS 38.300 §15.5.1.3): a
//!   load-balancing HANDOVER REQUEST becomes an inbound `TriggerHandover` intent
//!   on `Cell:<target>`. It contests the Pending switch-off on the same
//!   resource, found by [`ConflictDetector::check_conflict`]. The demo **mode**
//!   picks the winner: `load-balancing` admits the UE and defers the switch-off;
//!   `energy-saving` refuses the request (HANDOVER FAILURE "Cell not
//!   available") and switches off now. Refusing during a Pending state is this
//!   demo's policy, **non-normative**: TS 38.300 §15.4.2.1 lets a booster refuse
//!   incoming handovers only during the switch-off time period.
//!
//! Every step is audited ([`AuditRecord`], `nr-cli <gnb> agent-audit [--json]`)
//! and logged with the `[demo:d446a-agent-conflict]` prefix. Nothing here runs
//! unless the demo is active: [`D446a::on_tick`] and [`D446a::arbitrate`] take
//! the current mode, and do nothing without one.

use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use nextgsim_agent::{AgentId, ConflictDetector, Intent, IntentType};
use nextgsim_common::config::{AgentRuleConfig, AgentRuleKind};
use serde::Serialize;

use crate::demo::D446A_KEY;
use crate::tasks::InboundHandoverVerdict;

/// The `cause` parameter value of a load-balancing intent.
pub const CAUSE_REDUCE_LOAD: &str = "reduce-load-in-serving-cell";

/// One audit entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AuditRecord {
    /// Position in the audit, from 1.
    pub seq: u64,
    /// `IntentSubmitted`, `InboundIntent`, `ConflictDetected`, `Resolved`,
    /// `Deferred`, `GuardrailHold`, `PolicyChanged`, `Executed`,
    /// `ActuationSucceeded` or `ActuationFailed`.
    pub event: String,
    /// The agent (or `lb-1@gnb-2` for an inbound one) the entry is about.
    pub agent: String,
    /// What happened.
    pub detail: String,
}

/// A shared, append-only audit.
pub type Audit = Arc<Mutex<Vec<AuditRecord>>>;

/// The audit `nr-cli agent-audit` reads: the production Agent task's, installed
/// once. Tests use their own [`Audit`] and never touch it.
static CLI_AUDIT: OnceLock<Audit> = OnceLock::new();

/// What the Agent task has to do after a decision.
#[derive(Debug, Clone)]
pub enum Action {
    /// Submit this handover intent to the coordinator and actuate its result.
    Handover(Intent),
    /// Switch this cell off the air.
    SwitchOff,
}

/// An inbound handover to arbitrate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Inbound {
    /// Its cause, rendered.
    pub cause: String,
    /// Whether that cause is "Reduce load in serving cell".
    pub load_balancing: bool,
    /// The source cell's NR Cell Identity (UE History Information).
    pub source_cell: Option<u64>,
    /// The source gNB ID.
    pub source_gnb_id: Option<u32>,
    /// The cell asked for.
    pub target_cell: u64,
}

#[derive(Debug, Clone)]
struct Pending {
    intent: Intent,
    deadline: Instant,
}

/// The demo's agents on one gNB.
#[derive(Debug)]
pub struct D446a {
    lb: Option<AgentRuleConfig>,
    energy: Option<AgentRuleConfig>,
    audit: Audit,
    pending: Option<Pending>,
    off: bool,
    lb_tried: HashSet<i32>,
    last_policy: Option<String>,
}

impl D446a {
    /// The demo's agents from the gNB's `agents:` list (the first of each kind),
    /// or `None` when it configures none.
    pub fn from_config(agents: &[AgentRuleConfig], audit: Audit) -> Option<Self> {
        let pick = |k| agents.iter().find(|a| a.kind == k).cloned();
        let (lb, energy) = (
            pick(AgentRuleKind::LoadBalancing),
            pick(AgentRuleKind::EnergySaving),
        );
        if lb.is_none() && energy.is_none() {
            return None;
        }
        Some(Self {
            lb,
            energy,
            audit,
            pending: None,
            off: false,
            lb_tried: HashSet::new(),
            last_policy: None,
        })
    }

    /// Makes `audit` the one `nr-cli agent-audit` shows. Production only.
    pub fn install_cli_audit(audit: &Audit) {
        let _ = CLI_AUDIT.set(audit.clone());
    }

    /// The configured agents, as `nextgsim_agent` registrations: id, type and
    /// the capabilities the kind needs.
    pub fn registrations(&self) -> Vec<(String, nextgsim_agent::AgentCapabilities)> {
        let mut out = Vec::new();
        if let Some(lb) = &self.lb {
            out.push((
                lb.id.clone(),
                nextgsim_agent::AgentCapabilities {
                    read_state: true,
                    trigger_actions: true,
                    ..Default::default()
                },
            ));
        }
        if let Some(e) = &self.energy {
            out.push((
                e.id.clone(),
                nextgsim_agent::AgentCapabilities {
                    read_state: true,
                    modify_config: true,
                    ..Default::default()
                },
            ));
        }
        out
    }

    /// Whether the cell was switched off.
    pub fn is_off(&self) -> bool {
        self.off
    }

    fn record(&self, event: &str, agent: &str, detail: String) {
        let mut audit = self.audit.lock().unwrap_or_else(|p| p.into_inner());
        let seq = audit.len() as u64 + 1;
        nextgsim_demo::demo_log!(info, D446A_KEY, "audit #{seq}: {event} ({agent}) {detail}");
        audit.push(AuditRecord {
            seq,
            event: event.to_string(),
            agent: agent.to_string(),
            detail,
        });
    }

    /// One Agent tick: the cell's served UEs, at `now`, in demo `mode` (`None`
    /// when the demo is not active: nothing happens).
    pub fn on_tick(
        &mut self,
        now: Instant,
        served_ue_ids: &[i32],
        mode: Option<&str>,
    ) -> Vec<Action> {
        let mut actions = Vec::new();
        if mode.is_none() {
            return actions;
        }
        let served = served_ue_ids.len() as u32;
        // Forget UEs that left, so a returning one may be moved again.
        self.lb_tried.retain(|ue| served_ue_ids.contains(ue));
        if let Some(lb) = &self.lb {
            if served >= lb.load_threshold_ues {
                if let Some(&ue) = served_ue_ids.iter().find(|ue| !self.lb_tried.contains(ue)) {
                    let target_cell = lb.cell & 0xF_FFFF_FFFF;
                    let intent = Intent::new(AgentId::new(&lb.id), IntentType::TriggerHandover)
                        .with_target(ue.to_string())
                        .with_param("target_cell", target_cell.to_string())
                        .with_param("cause", CAUSE_REDUCE_LOAD)
                        .with_priority(lb.priority());
                    let id = lb.id.clone();
                    self.lb_tried.insert(ue);
                    self.record(
                        "IntentSubmitted",
                        &id,
                        format!(
                            "TriggerHandover ue={ue} target_cell={target_cell} \
                             cause={CAUSE_REDUCE_LOAD} (served UEs {served} >= {})",
                            self.lb.as_ref().map_or(0, |l| l.load_threshold_ues)
                        ),
                    );
                    actions.push(Action::Handover(intent));
                }
            }
        }
        if let Some(energy) = self.energy.clone() {
            if self.off {
                return actions;
            }
            let cell = energy.cell & 0xF_FFFF_FFFF;
            match &self.pending {
                Some(p) if now >= p.deadline => {
                    if served > 0 {
                        self.pending = None;
                        self.record(
                            "GuardrailHold",
                            &energy.id,
                            format!(
                                "SwitchOffWithConnectedUes: ues={served}, cell {cell} stays on"
                            ),
                        );
                    } else {
                        actions.push(self.switch_off(&energy.id, cell, "grace window over"));
                    }
                }
                Some(_) => {}
                None if served == 0 => {
                    let intent =
                        Intent::new(AgentId::new(&energy.id), IntentType::OptimizeResources)
                            .with_target(cell.to_string())
                            .with_param("status", "inactive")
                            .with_priority(energy.priority());
                    self.pending = Some(Pending {
                        intent,
                        deadline: now + Duration::from_secs(energy.grace_s),
                    });
                    self.record(
                        "IntentSubmitted",
                        &energy.id,
                        format!(
                            "OptimizeResources target={cell} status=inactive: switch-off Pending \
                             (grace {} s)",
                            energy.grace_s
                        ),
                    );
                }
                None => {}
            }
        }
        actions
    }

    fn switch_off(&mut self, agent: &str, cell: u64, why: &str) -> Action {
        self.pending = None;
        self.off = true;
        self.record(
            "Executed",
            agent,
            format!("switch-off of cell {cell} ({why})"),
        );
        nextgsim_demo::demo_log!(warn, D446A_KEY, "Agent switch-off: cell {cell} off the air");
        Action::SwitchOff
    }

    /// The booster's admission decision for an inbound handover, in demo `mode`.
    pub fn arbitrate(
        &mut self,
        now: Instant,
        req: &Inbound,
        mode: Option<&str>,
    ) -> (InboundHandoverVerdict, Vec<Action>) {
        let admit = InboundHandoverVerdict::Admit {
            handback_hold: None,
        };
        let Some(mode) = mode else {
            return (admit, Vec::new());
        };
        let Some(energy) = self.energy.clone() else {
            return (admit, Vec::new());
        };
        if !req.load_balancing {
            return (admit, Vec::new());
        }
        let target_cell = req.target_cell & 0xF_FFFF_FFFF;
        let label = match req.source_gnb_id {
            Some(g) => match energy.peers.iter().find(|p| p.gnb_id == g) {
                Some(p) => format!("{}@gnb-{g}", p.agent),
                None => format!("load-balancing@gnb-{g}"),
            },
            None => "load-balancing@unknown".to_string(),
        };
        self.record(
            "InboundIntent",
            &label,
            format!(
                "TriggerHandover target_cell={target_cell} cause={}",
                req.cause
            ),
        );
        if self.off {
            self.record(
                "Resolved",
                &label,
                format!("refused: cell {target_cell} is off the air"),
            );
            return (InboundHandoverVerdict::Refuse, Vec::new());
        }
        let inbound = Intent::new(AgentId::new(&label), IntentType::TriggerHandover)
            .with_param("target_cell", target_cell.to_string())
            .with_param("cause", CAUSE_REDUCE_LOAD)
            // The HANDOVER REQUEST carries no priority; 8 is the load-balancing
            // default (AgentRuleConfig::priority). It decides nothing here: the
            // mode breaks a Cell:<booster> tie (`inbound_wins`).
            .with_priority(8);
        let hold = req.source_cell.map(|nci| {
            (
                (nci % 1008) as i32,
                Duration::from_secs(energy.handback_hold_s),
            )
        });
        let Some(pending) = self.pending.clone() else {
            return (
                InboundHandoverVerdict::Admit {
                    handback_hold: hold,
                },
                Vec::new(),
            );
        };
        let Some(conflict) = ConflictDetector::check_conflict(&inbound, &pending.intent) else {
            return (
                InboundHandoverVerdict::Admit {
                    handback_hold: hold,
                },
                Vec::new(),
            );
        };
        let contested: Vec<String> = conflict
            .contested_resources
            .iter()
            .map(|r| format!("{:?}:{}", r.kind, r.id))
            .collect();
        self.record(
            "ConflictDetected",
            &label,
            format!("{} vs {} on {}", label, energy.id, contested.join(",")),
        );
        // PolicyChanged is recorded at the first contest after a mode change, which
        // is when the change applies. The `set` itself is audited at once by
        // nextgsim-demo (`[demo:...] audit: mode A -> B from nr-cli`); `by=cli` is
        // that source, the only way a mode changes at runtime.
        if let Some(previous) = self.last_policy.replace(mode.to_string()) {
            if previous != mode {
                self.record(
                    "PolicyChanged",
                    D446A_KEY,
                    format!("from={previous} to={mode} by=cli"),
                );
            }
        }
        if self.inbound_wins(mode, &inbound, &pending.intent) {
            self.record(
                "Resolved",
                &label,
                format!("winner {label} (policy {mode})"),
            );
            if let Some(p) = self.pending.as_mut() {
                p.deadline = now + Duration::from_secs(energy.grace_s);
            }
            self.record(
                "Deferred",
                &energy.id,
                format!(
                    "switch-off of cell {target_cell} deferred {} s",
                    energy.grace_s
                ),
            );
            self.record(
                "Executed",
                &label,
                format!("admitted on cell {target_cell}"),
            );
            (
                InboundHandoverVerdict::Admit {
                    handback_hold: hold,
                },
                Vec::new(),
            )
        } else {
            self.record(
                "Resolved",
                &label,
                format!(
                    "winner {} (policy {mode}): refused, CellNotAvailable",
                    energy.id
                ),
            );
            let action = self.switch_off(&energy.id, target_cell, "won the contest");
            (InboundHandoverVerdict::Refuse, vec![action])
        }
    }

    /// The tie-breaker for a `Cell:<booster>` load-balancing vs energy contest:
    /// the operator's policy (the demo mode), not the priorities.
    fn inbound_wins(&self, mode: &str, _inbound: &Intent, _energy: &Intent) -> bool {
        mode == "load-balancing"
    }

    /// Audits an actuation outcome from NGAP.
    pub fn on_actuation(&mut self, intent_id: &str, ue_id: i32, success: bool, detail: &str) {
        let agent = self.lb.as_ref().map_or("-".to_string(), |l| l.id.clone());
        let event = if success {
            "ActuationSucceeded"
        } else {
            "ActuationFailed"
        };
        self.record(
            event,
            &agent,
            format!("intent {intent_id} ue={ue_id}: {detail}"),
        );
    }
}

/// `nr-cli <gnb> agent-audit [--json]`: the demo's audit, or `None` when the
/// command is not one of these.
pub fn route_cli(command: &str) -> Option<Result<String, String>> {
    let mut words = command.split_whitespace();
    if words.next() != Some("agent-audit") {
        return None;
    }
    let json = match words.next() {
        None => false,
        Some("--json") => true,
        Some(other) => return Some(Err(format!("agent-audit: unknown argument {other:?}"))),
    };
    let Some(audit) = CLI_AUDIT.get() else {
        return Some(Err(
            "agent-audit: no agent demo is running on this gNB".to_string()
        ));
    };
    let audit = audit.lock().unwrap_or_else(|p| p.into_inner());
    Some(Ok(render(&audit, json)))
}

/// The audit as JSON, or one line per entry.
pub fn render(audit: &[AuditRecord], json: bool) -> String {
    if json {
        return serde_json::to_string_pretty(audit).unwrap_or_else(|_| "[]".to_string());
    }
    audit
        .iter()
        .map(|r| format!("{:>3} {} ({}) {}", r.seq, r.event, r.agent, r.detail))
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests;
