//! A deterministic in-process FL run over the demo model and task (nextgsim#480).
//!
//! This is the library half of the demo. It holds no network and no clock. It
//! runs rounds of local training and aggregation, scores every round on the
//! held-out set, and computes the two honest baselines: local-only (each client
//! alone) and pooled (all data in one place). The AF (`nr-fl-server`) and the UE
//! client do the same steps over real PDU sessions. Here they run in one process,
//! so the scorecard thresholds of #480 can be checked, and reruns compared bit for
//! bit, without a network.
//!
//! A client's contribution is its model DELTA (trained − global). The server
//! applies an aggregate of the deltas, which is what lets the robust rules
//! (median, trimmed mean, Krum) reject a poisoned delta.

use crate::mlp::{self, TrainConfig};
use crate::synthetic::{Dataset, Task};
use crate::{fedavg_sorted, krum_select, median_aggregate, trimmed_mean_aggregate};
use std::collections::BTreeMap;

/// How the server combines the round's deltas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Rule {
    /// Sample-weighted mean (FedAvg), in sorted-member order.
    FedAvg,
    /// Coordinate-wise median.
    Median,
    /// Coordinate-wise mean after trimming: `ratio` is the TOTAL fraction trimmed,
    /// half from each end ([`trimmed_mean_aggregate`]), so 0.4 of 5 is one each.
    TrimmedMean(f32),
    /// Krum with `f` tolerated Byzantine members: the one delta closest to its
    /// neighbours.
    Krum(usize),
}

/// One run's settings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunConfig {
    pub seed: u64,
    pub clients: u32,
    pub samples_per_client: usize,
    pub held_out_per_client: usize,
    pub rounds: u32,
    pub train: TrainConfig,
    /// A client whose delta is replaced by `-scale ×` its honest delta (the
    /// poisoned-UE arm: `-10`), or `None`.
    pub poisoned: Option<(u32, f32)>,
}

impl RunConfig {
    /// The #480 `t8-exclusion` shape: 3 clients, 10 rounds.
    pub fn demo(seed: u64) -> Self {
        Self {
            seed,
            clients: 3,
            samples_per_client: 240,
            held_out_per_client: 200,
            rounds: 10,
            train: TrainConfig::default(),
            poisoned: None,
        }
    }
}

/// What one round produced.
#[derive(Debug, Clone, PartialEq)]
pub struct Round {
    pub round: u32,
    /// Held-out MSE of the global model AFTER this round.
    pub mse: f32,
    pub participants: Vec<String>,
    /// For [`Rule::Krum`]: the member whose delta was selected, and the member with
    /// the worst (largest) score, the one the AF logs as `excluded=`.
    pub krum: Option<(String, String)>,
}

/// A whole run.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    pub rounds: Vec<Round>,
    /// The final global model.
    pub model: Vec<f32>,
}

/// The member name of client `c` (`ue1`, `ue2`, ...), the AF's sort key.
pub fn member(c: u32) -> String {
    format!("ue{}", c + 1)
}

/// The run's data: per-client training sets and the held-out set.
pub struct Data {
    pub clients: Vec<Dataset>,
    pub held_out: Dataset,
}

impl Data {
    pub fn new(cfg: &RunConfig) -> Self {
        let task = Task::new(cfg.seed, cfg.clients);
        Self {
            clients: (0..cfg.clients)
                .map(|c| task.client_data(c, cfg.samples_per_client))
                .collect(),
            held_out: task.held_out(cfg.held_out_per_client),
        }
    }
}

/// Client `c`'s delta for `round`, trained from `global`.
pub fn local_delta(
    global: &[f32],
    data: &Dataset,
    cfg: &RunConfig,
    c: u32,
    round: u32,
) -> Vec<f32> {
    let mut p = global.to_vec();
    mlp::train(&mut p, &data.xs, &data.ys, cfg.train, cfg.seed, c, round);
    p.iter().zip(global).map(|(t, g)| t - g).collect()
}

/// Run `cfg` with `rule`, `members(round)` naming who takes part in each round
/// (the AF's member selection: a departed UE is simply absent).
pub fn run(cfg: &RunConfig, rule: Rule, members: impl Fn(u32) -> Vec<u32>) -> Outcome {
    let data = Data::new(cfg);
    let mut global = mlp::init(cfg.seed);
    let mut rounds = Vec::new();
    for r in 1..=cfg.rounds {
        let mut deltas: BTreeMap<String, (Vec<f32>, u64)> = BTreeMap::new();
        for c in members(r) {
            let honest = local_delta(&global, &data.clients[c as usize], cfg, c, r);
            let delta = match cfg.poisoned {
                Some((bad, scale)) if bad == c => honest.iter().map(|d| -scale * d).collect(),
                _ => honest,
            };
            deltas.insert(member(c), (delta, data.clients[c as usize].len() as u64));
        }
        let (agg, krum) = aggregate(rule, &deltas);
        for (w, d) in global.iter_mut().zip(&agg) {
            *w += d;
        }
        rounds.push(Round {
            round: r,
            mse: mlp::mse(&global, &data.held_out.xs, &data.held_out.ys),
            participants: deltas.keys().cloned().collect(),
            krum,
        });
    }
    Outcome {
        rounds,
        model: global,
    }
}

fn aggregate(
    rule: Rule,
    deltas: &BTreeMap<String, (Vec<f32>, u64)>,
) -> (Vec<f32>, Option<(String, String)>) {
    // BTreeMap order: sorted by member, so every rule sees the same order.
    let names: Vec<&String> = deltas.keys().collect();
    let vecs: Vec<Vec<f32>> = deltas.values().map(|(d, _)| d.clone()).collect();
    match rule {
        Rule::FedAvg => (fedavg_sorted(deltas), None),
        Rule::Median => (median_aggregate(&vecs), None),
        Rule::TrimmedMean(ratio) => (trimmed_mean_aggregate(&vecs, ratio), None),
        Rule::Krum(f) => {
            let k = krum_select(&vecs, f);
            let worst = k.worst().map(|i| names[i].clone()).unwrap_or_default();
            let picked = k.selected.map(|i| names[i].clone()).unwrap_or_default();
            (k.weights, Some((picked, worst)))
        }
    }
}

/// The local-only baseline: the mean held-out MSE of each client trained alone
/// for as many epochs as the federation ran.
pub fn local_only_mse(cfg: &RunConfig) -> f32 {
    let data = Data::new(cfg);
    let mut sum = 0.0f32;
    for c in 0..cfg.clients {
        let mut p = mlp::init(cfg.seed);
        let d = &data.clients[c as usize];
        for r in 1..=cfg.rounds {
            mlp::train(&mut p, &d.xs, &d.ys, cfg.train, cfg.seed, c, r);
        }
        sum += mlp::mse(&p, &data.held_out.xs, &data.held_out.ys);
    }
    sum / cfg.clients as f32
}

/// The pooled baseline: one model trained on every client's data together, for as
/// many epochs as the federation ran.
pub fn pooled_mse(cfg: &RunConfig) -> f32 {
    let data = Data::new(cfg);
    let all = Dataset::concat(&data.clients);
    let mut p = mlp::init(cfg.seed);
    for r in 1..=cfg.rounds {
        mlp::train(&mut p, &all.xs, &all.ys, cfg.train, cfg.seed, u32::MAX, r);
    }
    mlp::mse(&p, &data.held_out.xs, &data.held_out.ys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{self, Frame};

    fn everyone(cfg: &RunConfig) -> impl Fn(u32) -> Vec<u32> {
        let n = cfg.clients;
        move |_| (0..n).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// #480 scorecard, at library level, on the demo shape: round 10 ≤ 0.5× round
    /// 1, ≤ 0.8× mean local-only, ≤ 1.2× pooled.
    #[test]
    fn the_federation_meets_the_scorecard() {
        let cfg = RunConfig::demo(480);
        let out = run(&cfg, Rule::FedAvg, everyone(&cfg));
        let first = out.rounds[0].mse;
        let last = out.rounds.last().unwrap().mse;
        let local = local_only_mse(&cfg);
        let pooled = pooled_mse(&cfg);
        eprintln!("round1 {first} round10 {last} local-only {local} pooled {pooled}");
        assert!(last <= 0.5 * first, "round 10 {last} vs round 1 {first}");
        assert!(last <= 0.8 * local, "round 10 {last} vs local-only {local}");
        assert!(last <= 1.2 * pooled, "round 10 {last} vs pooled {pooled}");
    }

    /// "Reruns are bit-identical": the same seed gives the same model, bit for bit,
    /// and the same per-round MSE; another seed does not.
    #[test]
    fn reruns_are_bit_identical() {
        let cfg = RunConfig::demo(480);
        let a = run(&cfg, Rule::FedAvg, everyone(&cfg));
        let b = run(&cfg, Rule::FedAvg, everyone(&cfg));
        assert_eq!(bits(&a.model), bits(&b.model));
        assert_eq!(
            a.rounds.iter().map(|r| r.mse.to_bits()).collect::<Vec<_>>(),
            b.rounds.iter().map(|r| r.mse.to_bits()).collect::<Vec<_>>()
        );
        let other = run(&RunConfig::demo(481), Rule::FedAvg, everyone(&cfg));
        assert_ne!(bits(&a.model), bits(&other.model));
    }

    /// A departed member is simply absent from the next rounds, and the run still
    /// learns.
    #[test]
    fn a_member_can_leave_and_return() {
        let cfg = RunConfig::demo(480);
        let out = run(&cfg, Rule::FedAvg, |r| {
            if (4..7).contains(&r) {
                vec![0, 1]
            } else {
                vec![0, 1, 2]
            }
        });
        assert_eq!(out.rounds[3].participants, vec!["ue1", "ue2"]);
        assert_eq!(out.rounds[6].participants, vec!["ue1", "ue2", "ue3"]);
        assert!(out.rounds[9].mse <= 0.5 * out.rounds[0].mse);
    }

    /// #480 poisoned-UE arm at library level: 5 clients, client 5 (`ue5`) sends
    /// ×10 sign-flipped deltas.
    ///
    /// "The clean run" is read per rule: the same rule, same seed, with no
    /// poisoned member. That measures what the ATTACK costs each rule, apart from
    /// what the rule costs without one (on this seed the clean median and trimmed
    /// mean runs are about 1.1× clean FedAvg). So: median and trimmed mean within
    /// 1.1× of their clean runs, FedAvg ≥ 10× its clean run, and Krum scores
    /// `ue5` worst in every round (the AF's `excluded=`) and never selects it.
    #[test]
    fn robust_rules_survive_the_poisoned_member() {
        let clean_cfg = RunConfig {
            clients: 5,
            ..RunConfig::demo(480)
        };
        let poisoned = RunConfig {
            poisoned: Some((4, 10.0)),
            ..clean_cfg
        };
        let last = |cfg: &RunConfig, rule| run(cfg, rule, everyone(cfg)).rounds[9].mse;
        // `TrimmedMean`'s ratio is the TOTAL fraction trimmed, split over both
        // ends: 0.4 of 5 members is one from each end.
        for (name, rule, bound) in [
            ("median", Rule::Median, 1.1f32),
            ("trimmed mean", Rule::TrimmedMean(0.4), 1.1),
        ] {
            let (clean, attacked) = (last(&clean_cfg, rule), last(&poisoned, rule));
            eprintln!("{name}: clean {clean} poisoned {attacked}");
            assert!(
                attacked <= bound * clean,
                "{name}: {attacked} vs its clean run {clean}"
            );
        }
        let (clean, attacked) = (
            last(&clean_cfg, Rule::FedAvg),
            last(&poisoned, Rule::FedAvg),
        );
        eprintln!("fedavg: clean {clean} poisoned {attacked}");
        // FedAvg may diverge to inf or NaN, which is the failure this arm shows;
        // NaN counts as "≥ 10×".
        assert!(
            attacked.is_nan() || attacked >= 10.0 * clean,
            "fedavg {attacked} vs its clean run {clean}"
        );
        let krum = run(&poisoned, Rule::Krum(1), everyone(&poisoned));
        for r in &krum.rounds {
            let (picked, worst) = r.krum.clone().expect("krum");
            assert_eq!(
                worst, "ue5",
                "round {}: the poisoned member scores worst",
                r.round
            );
            assert_ne!(picked, "ue5");
        }
    }

    /// Row-free: every frame of a whole run (Hello, the global model each round,
    /// every Update) contains none of any client's canary rows. The positive
    /// control (a frame that DOES carry a row is caught) is in `synthetic`.
    #[test]
    fn no_training_row_crosses_the_wire() {
        let cfg = RunConfig::demo(480);
        let data = Data::new(&cfg);
        let mut wire = Vec::new();
        let mut global = mlp::init(cfg.seed);
        for c in 0..cfg.clients {
            wire.extend(frame::encode(&Frame::Hello { label: member(c) }).unwrap());
        }
        for r in 1..=3 {
            wire.extend(
                frame::encode(&Frame::GlobalModel {
                    round: r,
                    weights: global.clone(),
                })
                .unwrap(),
            );
            let mut deltas = BTreeMap::new();
            for c in 0..cfg.clients {
                let d = local_delta(&global, &data.clients[c as usize], &cfg, c, r);
                wire.extend(
                    frame::encode(&Frame::Update {
                        round: r,
                        samples: 240,
                        loss: 0.1,
                        delta: d.clone(),
                    })
                    .unwrap(),
                );
                deltas.insert(member(c), (d, 240));
            }
            for (w, d) in global.iter_mut().zip(fedavg_sorted(&deltas)) {
                *w += d;
            }
        }
        for (c, d) in data.clients.iter().enumerate() {
            for canary in d.canaries() {
                assert!(
                    !crate::synthetic::contains_row(&wire, canary),
                    "client {c}'s canary on the wire"
                );
            }
        }
    }
}
