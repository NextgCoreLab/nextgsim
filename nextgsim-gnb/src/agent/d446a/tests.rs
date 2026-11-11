//! D446A decision tests (nextgsim#483). Each drives [`D446a`] with the inputs the
//! Agent task feeds it and asserts the audit chain the acceptance criteria name.

use super::*;
use nextgsim_common::config::AgentPeerConfig;

const LB: &str = "load-balancing";
const ES: &str = "energy-saving";

fn energy_rule() -> AgentRuleConfig {
    AgentRuleConfig {
        id: "energy-1".into(),
        kind: AgentRuleKind::EnergySaving,
        cell: 0x10,
        priority: None,
        load_threshold_ues: 1,
        grace_s: 20,
        handback_hold_s: 60,
        peers: vec![AgentPeerConfig {
            gnb_id: 2,
            agent: "lb-1".into(),
        }],
    }
}

fn lb_rule() -> AgentRuleConfig {
    AgentRuleConfig {
        id: "lb-1".into(),
        kind: AgentRuleKind::LoadBalancing,
        cell: 0x10,
        priority: None,
        load_threshold_ues: 1,
        grace_s: 20,
        handback_hold_s: 60,
        peers: vec![],
    }
}

fn booster() -> (D446a, Audit) {
    let audit = Audit::default();
    (
        D446a::from_config(&[energy_rule()], audit.clone()).unwrap(),
        audit,
    )
}

fn inbound_lb() -> Inbound {
    Inbound {
        cause: "RadioNetwork(ReduceLoadInServingCell)".into(),
        load_balancing: true,
        source_cell: Some(0x20),
        source_gnb_id: Some(2),
        target_cell: 0x10,
    }
}

fn events(audit: &Audit) -> Vec<String> {
    audit
        .lock()
        .unwrap()
        .iter()
        .map(|r| format!("{} ({})", r.event, r.agent))
        .collect()
}

/// Main arm: lb-1 wins on Cell:16, energy-1 is Deferred and then held by the
/// guardrail because the admitted UE is served; one execution, a handback hold
/// on the source PCI 32.
#[test]
fn load_balancing_mode_admits_defers_then_holds() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    assert!(
        b.on_tick(t0, &[], Some(LB)).is_empty(),
        "Pending, not executed"
    );
    let (verdict, actions) = b.arbitrate(t0 + Duration::from_secs(5), &inbound_lb(), Some(LB));
    assert_eq!(
        verdict,
        InboundHandoverVerdict::Admit {
            handback_hold: Some((32, Duration::from_secs(60)))
        }
    );
    assert!(actions.is_empty());
    // The original deadline (t0+20) passes with the switch-off deferred to t0+25.
    assert!(b
        .on_tick(t0 + Duration::from_secs(21), &[1001], Some(LB))
        .is_empty());
    assert!(b
        .on_tick(t0 + Duration::from_secs(26), &[1001], Some(LB))
        .is_empty());
    assert!(!b.is_off());
    assert_eq!(
        events(&audit),
        vec![
            "IntentSubmitted (energy-1)",
            "InboundIntent (lb-1@gnb-2)",
            "ConflictDetected (lb-1@gnb-2)",
            "Resolved (lb-1@gnb-2)",
            "Deferred (energy-1)",
            "Executed (lb-1@gnb-2)",
            "GuardrailHold (energy-1)",
        ]
    );
    let a = audit.lock().unwrap();
    assert!(a[2].detail.contains("Cell:16"), "{}", a[2].detail);
    assert!(
        a[3].detail.starts_with("winner lb-1@gnb-2"),
        "{}",
        a[3].detail
    );
    assert!(a[6].detail.contains("ues=1"), "{}", a[6].detail);
    assert_eq!(a.iter().filter(|r| r.event == "Executed").count(), 1);
}

/// Swap arm: energy-1 wins, the request is refused and the cell goes off the air;
/// a later request is refused because the cell is off.
#[test]
fn energy_saving_mode_refuses_and_switches_off() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], Some(ES));
    let (verdict, actions) = b.arbitrate(t0, &inbound_lb(), Some(ES));
    assert_eq!(verdict, InboundHandoverVerdict::Refuse);
    assert!(matches!(actions.as_slice(), [Action::SwitchOff]));
    assert!(b.is_off());
    let a = audit.lock().unwrap().clone();
    assert_eq!(a[3].event, "Resolved");
    assert!(
        a[3].detail.starts_with("winner energy-1"),
        "{}",
        a[3].detail
    );
    assert_eq!(a[4].event, "Executed");
    assert_eq!(a[4].agent, "energy-1");
    let (again, _) = b.arbitrate(t0, &inbound_lb(), Some(ES));
    assert_eq!(again, InboundHandoverVerdict::Refuse);
    assert!(b
        .on_tick(t0 + Duration::from_secs(60), &[], Some(ES))
        .is_empty());
}

/// Only the mode differs between the two arms; a mode change between two
/// contests flips the second only, with PolicyChanged between the two Resolved.
#[test]
fn a_mode_change_applies_to_the_next_contest_only() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], Some(LB));
    let (first, _) = b.arbitrate(t0, &inbound_lb(), Some(LB));
    assert!(matches!(first, InboundHandoverVerdict::Admit { .. }));
    let (second, _) = b.arbitrate(t0, &inbound_lb(), Some(ES));
    assert_eq!(second, InboundHandoverVerdict::Refuse);
    let ev = events(&audit);
    let resolved: Vec<usize> = ev
        .iter()
        .enumerate()
        .filter(|(_, e)| e.starts_with("Resolved"))
        .map(|(i, _)| i)
        .collect();
    let changed = ev
        .iter()
        .position(|e| e.starts_with("PolicyChanged"))
        .expect("audited");
    assert!(resolved[0] < changed && changed < resolved[1], "{ev:?}");
    let a = audit.lock().unwrap();
    assert_eq!(
        a[changed].detail,
        "from=load-balancing to=energy-saving by=cli"
    );
}

/// The fixed priorities (lb 8 vs energy 5) would always admit; the energy-saving
/// mode must refuse even so (the revert arm `446-policy-toggle`).
#[test]
fn the_mode_not_the_priority_decides() {
    let (mut b, _) = booster();
    b.on_tick(Instant::now(), &[], Some(ES));
    let (v, _) = b.arbitrate(Instant::now(), &inbound_lb(), Some(ES));
    assert_eq!(v, InboundHandoverVerdict::Refuse);
}

/// With no Pending switch-off there is no contest: admitted, nothing audited
/// beyond the inbound intent. A non-load-balancing handover is not arbitrated.
#[test]
fn no_contest_without_a_pending_switch_off() {
    let (mut b, audit) = booster();
    let (v, _) = b.arbitrate(Instant::now(), &inbound_lb(), Some(LB));
    assert!(matches!(
        v,
        InboundHandoverVerdict::Admit {
            handback_hold: Some(_)
        }
    ));
    let radio = Inbound {
        load_balancing: false,
        ..inbound_lb()
    };
    let (v, _) = b.arbitrate(Instant::now(), &radio, Some(LB));
    assert_eq!(
        v,
        InboundHandoverVerdict::Admit {
            handback_hold: None
        }
    );
    assert_eq!(events(&audit), vec!["InboundIntent (lb-1@gnb-2)"]);
}

/// The grace window ends with no UE served: the switch-off executes.
#[test]
fn an_empty_booster_switches_off_after_the_grace() {
    let (mut b, _) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], Some(LB));
    assert!(b
        .on_tick(t0 + Duration::from_secs(19), &[], Some(LB))
        .is_empty());
    assert!(matches!(
        b.on_tick(t0 + Duration::from_secs(20), &[], Some(LB))
            .as_slice(),
        [Action::SwitchOff]
    ));
}

/// Kill switch: with no active mode nothing is queued, arbitrated or audited.
#[test]
fn an_inactive_demo_does_nothing() {
    let audit = Audit::default();
    let mut b = D446a::from_config(&[energy_rule(), lb_rule()], audit.clone()).unwrap();
    assert!(b.on_tick(Instant::now(), &[], None).is_empty());
    assert!(b.on_tick(Instant::now(), &[1], None).is_empty());
    let (v, a) = b.arbitrate(Instant::now(), &inbound_lb(), None);
    assert_eq!(
        v,
        InboundHandoverVerdict::Admit {
            handback_hold: None
        }
    );
    assert!(a.is_empty());
    assert!(audit.lock().unwrap().is_empty());
    assert!(D446a::from_config(&[], audit).is_none());
}

/// lb-1 moves one served UE toward the booster with the load-balancing cause,
/// once per UE.
#[test]
fn a_loaded_cell_hands_one_ue_to_the_booster_once() {
    let audit = Audit::default();
    let mut a = D446a::from_config(&[lb_rule()], audit.clone()).unwrap();
    let now = Instant::now();
    let actions = a.on_tick(now, &[1], Some(LB));
    let [Action::Handover(intent)] = actions.as_slice() else {
        panic!("{actions:?}");
    };
    assert_eq!(intent.intent_type, IntentType::TriggerHandover);
    assert_eq!(intent.target.as_deref(), Some("1"));
    assert_eq!(
        intent.parameters.get("target_cell").map(String::as_str),
        Some("16")
    );
    assert_eq!(
        intent.parameters.get("cause").map(String::as_str),
        Some(CAUSE_REDUCE_LOAD)
    );
    assert!(a.on_tick(now, &[1], Some(LB)).is_empty(), "not repeated");
    a.on_actuation(&intent.id, 1, false, "HANDOVER PREPARATION FAILURE");
    assert_eq!(
        events(&audit),
        vec!["IntentSubmitted (lb-1)", "ActuationFailed (lb-1)"]
    );
}

#[test]
fn the_audit_renders_as_json() {
    let (mut b, audit) = booster();
    b.on_tick(Instant::now(), &[], Some(LB));
    let json = render(&audit.lock().unwrap(), true);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v[0]["event"], "IntentSubmitted");
    assert_eq!(v[0]["agent"], "energy-1");
    assert!(route_cli("ue-list").is_none());
}
