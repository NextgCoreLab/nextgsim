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
        b.on_tick(t0, &[], &[], Some(LB)).is_empty(),
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
        .on_tick(t0 + Duration::from_secs(21), &[1001], &[1001], Some(LB))
        .is_empty());
    assert!(b
        .on_tick(t0 + Duration::from_secs(26), &[1001], &[1001], Some(LB))
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
    b.on_tick(t0, &[], &[], Some(ES));
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
        .on_tick(t0 + Duration::from_secs(60), &[], &[], Some(ES))
        .is_empty());
}

/// Only the mode differs between the two arms; a mode change between two
/// contests flips the second only, with PolicyChanged between the two Resolved.
#[test]
fn a_mode_change_applies_to_the_next_contest_only() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], &[], Some(LB));
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
    b.on_tick(Instant::now(), &[], &[], Some(ES));
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
    b.on_tick(t0, &[], &[], Some(LB));
    assert!(b
        .on_tick(t0 + Duration::from_secs(19), &[], &[], Some(LB))
        .is_empty());
    assert!(matches!(
        b.on_tick(t0 + Duration::from_secs(20), &[], &[], Some(LB))
            .as_slice(),
        [Action::SwitchOff]
    ));
}

/// Kill switch: with no active mode nothing is queued, arbitrated or audited.
#[test]
fn an_inactive_demo_does_nothing() {
    let audit = Audit::default();
    let mut b = D446a::from_config(&[energy_rule(), lb_rule()], audit.clone()).unwrap();
    assert!(b.on_tick(Instant::now(), &[], &[], None).is_empty());
    assert!(b.on_tick(Instant::now(), &[1], &[1], None).is_empty());
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
    let actions = a.on_tick(now, &[1], &[1], Some(LB));
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
    assert!(
        a.on_tick(now, &[1], &[1], Some(LB)).is_empty(),
        "not repeated"
    );
    a.on_actuation(&intent.id, 1, false, false, "HANDOVER PREPARATION FAILURE");
    assert_eq!(
        events(&audit),
        vec!["IntentSubmitted (lb-1)", "ActuationFailed (lb-1)"]
    );
}

#[test]
fn the_audit_renders_as_json() {
    let (mut b, audit) = booster();
    b.on_tick(Instant::now(), &[], &[], Some(LB));
    let json = render(&audit.lock().unwrap(), true);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v[0]["event"], "IntentSubmitted");
    assert_eq!(v[0]["agent"], "energy-1");
    assert!(route_cli("ue-list").is_none());
}

/// The guardrail counts connected UEs, not UEs whose bearer has carried data:
/// a UE admitted by the handover that then stays silent (no RLC entity, so not
/// "served") must still hold the switch-off at the deadline (the revert arm
/// `446-connected-guardrail`).
#[test]
fn a_connected_but_silent_ue_holds_the_switch_off() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], &[], Some(LB));
    let (verdict, _) = b.arbitrate(t0, &inbound_lb(), Some(LB));
    assert!(matches!(verdict, InboundHandoverVerdict::Admit { .. }));
    // Past the deferred deadline (t0+20): connected, but nothing on the bearer.
    let actions = b.on_tick(t0 + Duration::from_secs(21), &[], &[1001], Some(LB));
    assert!(actions.is_empty(), "{actions:?}");
    assert!(!b.is_off(), "a connected UE would have been dropped");
    let a = audit.lock().unwrap();
    let hold = a.last().expect("audited");
    assert_eq!(hold.event, "GuardrailHold");
    assert!(hold.detail.contains("ues=1"), "{}", hold.detail);
}

/// After a runtime switch to energy-saving, the second contest is refused, but
/// the UE the first contest admitted is still connected: the guardrail keeps
/// the cell on, so the flip changes the second contest only.
#[test]
fn an_energy_saving_win_does_not_drop_a_connected_ue() {
    let (mut b, audit) = booster();
    let t0 = Instant::now();
    b.on_tick(t0, &[], &[], Some(LB));
    let (first, _) = b.arbitrate(t0, &inbound_lb(), Some(LB));
    assert!(matches!(first, InboundHandoverVerdict::Admit { .. }));
    b.on_tick(t0 + Duration::from_secs(2), &[], &[1001], Some(ES));
    let (second, actions) = b.arbitrate(t0 + Duration::from_secs(5), &inbound_lb(), Some(ES));
    assert_eq!(second, InboundHandoverVerdict::Refuse);
    assert!(
        actions.is_empty(),
        "no switch-off with a UE connected: {actions:?}"
    );
    assert!(!b.is_off());
    let ev = events(&audit);
    assert_eq!(
        &ev[ev.len() - 3..],
        &[
            "PolicyChanged (d446a-agent-conflict)",
            "Resolved (lb-1@gnb-2)",
            "GuardrailHold (energy-1)",
        ],
        "{ev:?}"
    );
}

/// A UE whose first packet makes it served before its PDU session resources
/// are set up: lb-1's handover is refused locally (no HANDOVER REQUIRED was
/// sent). That UE must be tried again at a later tick, not dropped for good
/// (the revert arm `446-retry-unsignalled`); a network refusal stays final.
#[test]
fn an_unsignalled_handover_failure_is_retried() {
    let audit = Audit::default();
    let mut a = D446a::from_config(&[lb_rule()], audit.clone()).unwrap();
    let now = Instant::now();
    let actions = a.on_tick(now, &[1], &[1], Some(LB));
    let [Action::Handover(first)] = actions.as_slice() else {
        panic!("lb-1 must fire");
    };
    let first = first.id.clone();
    a.on_actuation(&first, 1, false, true, "no HANDOVER REQUIRED was sent");
    let again = a.on_tick(now + Duration::from_secs(1), &[1], &[1], Some(LB));
    let [Action::Handover(second)] = again.as_slice() else {
        panic!("the UE must be tried again: {again:?}");
    };
    assert_eq!(second.target.as_deref(), Some("1"));
    let second = second.id.clone();
    // Refused by the network this time: final.
    a.on_actuation(&second, 1, false, false, "HANDOVER PREPARATION FAILURE");
    assert!(a
        .on_tick(now + Duration::from_secs(2), &[1], &[1], Some(LB))
        .is_empty());
    let a = audit.lock().unwrap();
    let failed: Vec<&str> = a
        .iter()
        .filter(|r| r.event == "ActuationFailed")
        .map(|r| r.detail.as_str())
        .collect();
    assert!(
        failed[0].ends_with("not signalled, retry 1/3"),
        "{failed:?}"
    );
}

/// A UE that can never be handed over (a persistent local failure) is tried
/// at most MAX_UNSIGNALLED_RETRIES + 1 times, not at every tick.
#[test]
fn a_persistently_unsignalled_ue_is_retried_a_bounded_number_of_times() {
    let audit = Audit::default();
    let mut a = D446a::from_config(&[lb_rule()], audit.clone()).unwrap();
    let now = Instant::now();
    let mut tries = 0;
    for t in 0..20 {
        for action in a.on_tick(now + Duration::from_secs(t), &[1], &[1], Some(LB)) {
            let Action::Handover(intent) = action else {
                continue;
            };
            tries += 1;
            a.on_actuation(&intent.id, 1, false, true, "no HANDOVER REQUIRED was sent");
        }
    }
    assert_eq!(tries, MAX_UNSIGNALLED_RETRIES + 1);
    let a = audit.lock().unwrap();
    let last = a
        .iter()
        .rev()
        .find(|r| r.event == "ActuationFailed")
        .unwrap();
    assert!(
        last.detail.ends_with("retries exhausted"),
        "{}",
        last.detail
    );
}
