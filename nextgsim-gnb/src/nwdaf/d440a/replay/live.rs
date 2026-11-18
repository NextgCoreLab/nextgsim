//! The D440A live harness's generated configs (nextgsim#462 PR 3c).
//!
//! The six-gNB compose run (`scripts/demo/d440a-check.sh <mode> <seed>`) runs
//! the committed scenario's `live` seeds through the real stack. Its configs are
//! generated here from the same [`ScenarioFile`] and [`ScenarioFile::run`] the
//! replay uses, so a live seed's route is the replay's route for that seed and
//! cannot drift from it:
//! - `config/demo-ai-paging/gnb-{1..6}.yaml`: `config/gnb.yaml` with gNB N's
//!   cell, address and grid position, the other five as handover neighbours
//!   and as the paging grid, the NWDAF task on, and the D440A block last (the
//!   compose entrypoint replaces it with the run's mode);
//! - `config/demo-ai-paging/ue-seed-<seed>.yaml`: `config/ue-d440a-m0.yaml`
//!   hearing all six gNBs, starting where the seed's route starts and following
//!   it;
//! - `config/demo-ai-paging/d440a-live-seeds.yaml`: each live seed's release
//!   and page times on the route's clock, and what the replay predicts for it
//!   in every model-free mode. The harness reads the times. The replay lines
//!   are recorded next to the live result and are not asserted against it,
//!   because the live run has heartbeat jitter, connected-mode handovers and
//!   real NGAP timing that the replay does not model.
//!
//! The configs are checked in, and
//! `the_live_harness_configs_are_generated_from_the_scenario` fails when one
//! differs from what this module generates. To regenerate them:
//! `D440A_WRITE_LIVE=1 cargo test -p nextgsim-gnb --features demo-d440a-ai-paging --lib d440a::replay::live`.

use super::*;

const DEMO_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/demo-ai-paging");
const CONFIG_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../config");

/// The UE's route starts this long after start-up. By then the UE has registered
/// and holds its PDU session, so it is CM-CONNECTED for the route's first
/// `connected_s`. The harness fails, naming the reason, if the route starts
/// before the PDU session is up.
const START_AFTER_S: f64 = 60.0;

/// gNB N's address on the n2ho stack's network. gNBs 1 and 2 keep the
/// addresses of that stack's `gnb-west` and `gnb-east` (172.23.0.101 is its UE).
fn gnb_address(index: usize) -> String {
    let last = [100, 102, 103, 104, 105, 106][index];
    format!("172.23.0.{last}")
}

/// `text` with the one occurrence of `from` replaced. Panics on zero or several
/// occurrences, so a changed base config cannot be half-substituted.
fn replace_once(text: &str, from: &str, to: &str) -> String {
    assert_eq!(
        text.matches(from).count(),
        1,
        "the base config must hold exactly one {from:?}"
    );
    text.replacen(from, to, 1)
}

fn nci_hex(nci: u64) -> String {
    format!("{nci:#011x}")
}

fn position(p: SimPosition) -> String {
    format!("{{ x: {}, y: {}, z: {} }}", p.x, p.y, p.z)
}

/// gNB `index` (0-based) of the grid.
fn gnb_config(file: &ScenarioFile, index: usize, base: &str) -> String {
    let own = &file.grid[index];
    let others: Vec<&GridEntry> = file
        .grid
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(_, e)| e)
        .collect();
    let addr = gnb_address(index);
    let n = index + 1;
    let mut text = base.to_string();
    text = replace_once(
        &text,
        "\nnci: 0x000000010\n",
        &format!("\nnci: {}\n", nci_hex(own.nci)),
    );
    for key in ["link_ip", "ngap_ip", "gtp_ip"] {
        text = replace_once(
            &text,
            &format!("\n{key}: 172.23.0.100\n"),
            &format!("\n{key}: {addr}\n"),
        );
    }
    text = replace_once(&text, "\nnwdaf_enabled: false ", "\nnwdaf_enabled: true  ");
    text = replace_once(
        &text,
        "\nsim_position: { x: 0, y: 0, z: 0 }\n",
        &format!("\nsim_position: {}\n", position(own.position)),
    );
    let neighbours: String = others
        .iter()
        .map(|e| format!("  - nci: {}\n    tac: 1\n", nci_hex(e.nci)))
        .collect();
    text = replace_once(
        &text,
        "\nhandover_neighbours:\n  - nci: 0x000000020\n    tac: 1\n",
        &format!("\nhandover_neighbours:\n{neighbours}"),
    );
    let grid: String = others
        .iter()
        .map(|e| {
            format!(
                "  - nci: {}\n    position: {}\n",
                nci_hex(e.nci),
                position(e.position)
            )
        })
        .collect();
    format!(
        "# D440A (nextgsim#462 PR 3c) gNB {n} of the six-gNB grid. GENERATED from\n\
         # config/gnb.yaml and config/demo-ai-paging/d440a-scenario.yaml by\n\
         # nextgsim-gnb nwdaf::d440a::replay::live; do not edit by hand.\n\
         # Cell {nci} (gNB ID {n}), address {addr}, at {pos}. The other five gNBs\n\
         # are its handover neighbours and its paging grid, and the NWDAF task is\n\
         # on (the predictor modes need its heartbeat history). One TAC for all\n\
         # six, so the registration area holds every gNB. Run by\n\
         # scripts/demo/d440a-check.sh with a gNB built with --features\n\
         # demo-d440a-ai-paging. Its entrypoint (scripts/demo/d440a.compose.yml)\n\
         # replaces the trailing demo block with the run's mode, or drops it for\n\
         # the `off` arm.\n\
         #\n\
         {text}\n\
         # D440A: the cells a recommendation resolves against (the other five).\n\
         paging_grid:\n\
         {grid}\n\
         # D440A: the demo toggle (convention #458). Keep this block last: the\n\
         # compose entrypoint cuts the file at `demo:`.\n\
         demo:\n  enabled: true\n  d440a-ai-paging:\n    enabled: true\n    mode: ct\n",
        nci = nci_hex(own.nci),
        pos = position(own.position),
        text = text.trim_end(),
    )
}

/// The UE of live seed `run.seed`.
fn ue_config(file: &ScenarioFile, run: &Run, base: &str) -> String {
    let head_end = base
        .find("# UE Configuration for NextGSim\n")
        .expect("the M0 UE config's own header ends before its base header");
    let block_start = base.find("gnb_search_list:\n").expect("gnb_search_list");
    let block_end = base[block_start..]
        .find("\n\n")
        .map(|i| block_start + i)
        .expect("the motion block ends with a blank line");
    let addresses: String = (0..file.grid.len())
        .map(|i| format!("  - \"{}\"\n", gnb_address(i)))
        .collect();
    let waypoints: String = run
        .route
        .waypoints
        .iter()
        .map(|w| format!("    - {}\n", position(*w)))
        .collect();
    format!(
        "# UE of the D440A live harness (nextgsim#462 PR 3c), seed {seed}. GENERATED\n\
         # from config/ue-d440a-m0.yaml and config/demo-ai-paging/d440a-scenario.yaml\n\
         # by nextgsim-gnb nwdaf::d440a::replay::live; do not edit by hand.\n\
         # The UE hears all six gNBs of the grid and, {start_after} s after start-up,\n\
         # follows the replay's constant-turn route for seed {seed}. It is CM-CONNECTED\n\
         # for its first {release} s, released then by the harness, and paged {page} s\n\
         # into the route (config/demo-ai-paging/d440a-live-seeds.yaml).\n\
         {head}\
         gnb_search_list:\n\
         {addresses}\
         sim_position: {start}\n\
         sim_route:\n  waypoints:\n{waypoints}  speed_mps: {speed}\n  start_after_secs: {start_after}\n\
         {tail}",
        seed = run.seed,
        start_after = START_AFTER_S,
        release = run.release_at_s,
        page = run.page_at_s,
        head = &base[head_end..block_start],
        start = position(run.start),
        speed = run.route.speed_mps,
        tail = &base[block_end..],
    )
}

fn gnb_list(ids: &[u32]) -> String {
    let ids: Vec<String> = ids.iter().map(|id| format!("{id:#x}")).collect();
    format!("[{}]", ids.join(", "))
}

/// The live seeds' clock and the replay's prediction for each.
fn live_seeds(file: &ScenarioFile) -> String {
    let mut out = String::from(
        "# D440A live seeds (nextgsim#462 PR 3c). GENERATED from\n\
         # config/demo-ai-paging/d440a-scenario.yaml by nextgsim-gnb\n\
         # nwdaf::d440a::replay::live; do not edit by hand.\n\
         #\n\
         # For each live seed: its UE config, and the route's clock in seconds after\n\
         # the UE logs `Simulated motion started`: the harness releases the UE at\n\
         # release_at_s and sends it downlink data at page_at_s. `replay` is what the\n\
         # in-process replay gives the same route in each model-free mode (gNB IDs).\n\
         # It is recorded next to the live result and is NOT asserted against it:\n\
         # the live run has heartbeat jitter, connected-mode handovers and real NGAP\n\
         # timing, none of which the replay models.\n",
    );
    out.push_str(&format!("start_after_secs: {START_AFTER_S}\nseeds:\n"));
    for &seed in &file.seeds.live {
        let run = file.run(seed);
        out.push_str(&format!(
            "  - seed: {seed}\n    ue_config: ue-seed-{seed}.yaml\n    \
             release_at_s: {}\n    page_at_s: {}\n    replay:\n",
            run.release_at_s, run.page_at_s
        ));
        for (name, mode) in super::tests::MODES {
            let o = replay(file, &run, mode, &|_| {});
            out.push_str(&format!(
                "      {name}: {{ serving: {:#x}, recommended: {}, answering: {:#x}, \
                 first_attempt_hit: {}, pagings: {} }}\n",
                o.serving_gnb,
                gnb_list(&o.recommended),
                o.answering_gnb,
                o.first_attempt_hit,
                o.pagings
            ));
        }
    }
    out
}

/// Every generated file, by name in `config/demo-ai-paging/`.
fn generated() -> Vec<(String, String)> {
    let file = scenario_file();
    let read = |name: &str| {
        std::fs::read_to_string(format!("{CONFIG_DIR}/{name}")).expect("read a base config")
    };
    let (gnb, ue) = (read("gnb.yaml"), read("ue-d440a-m0.yaml"));
    let mut out: Vec<(String, String)> = (0..file.grid.len())
        .map(|i| (format!("gnb-{}.yaml", i + 1), gnb_config(&file, i, &gnb)))
        .collect();
    for &seed in &file.seeds.live {
        out.push((
            format!("ue-seed-{seed}.yaml"),
            ue_config(&file, &file.run(seed), &ue),
        ));
    }
    out.push(("d440a-live-seeds.yaml".to_string(), live_seeds(&file)));
    out
}

/// #462 PR 3c: every live-harness config is exactly what the scenario
/// generates, so the live run cannot drift from the replay.
#[test]
fn the_live_harness_configs_are_generated_from_the_scenario() {
    let write = std::env::var_os("D440A_WRITE_LIVE").is_some();
    for (name, want) in generated() {
        let path = format!("{DEMO_DIR}/{name}");
        if write {
            std::fs::write(&path, &want).expect("write a generated config");
            continue;
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(
            have == want,
            "config/demo-ai-paging/{name} is not what the scenario generates; regenerate with \
             D440A_WRITE_LIVE=1 cargo test -p nextgsim-gnb --features demo-d440a-ai-paging --lib \
             d440a::replay::live"
        );
    }
}

/// #462 PR 3c: the generated configs load and validate as a gNB and a UE, and
/// each gNB is its grid cell, naming the other five as its paging grid and its
/// handover neighbours. Each UE starts where its seed's route starts and
/// follows that route.
#[test]
fn the_live_harness_configs_load_as_the_scenario_says() {
    use crate::app::load_and_validate_gnb_config;
    let file = scenario_file();
    let cells = file.cells();
    for (i, cell) in cells.iter().enumerate() {
        let config = load_and_validate_gnb_config(format!("{DEMO_DIR}/gnb-{}.yaml", i + 1))
            .unwrap_or_else(|e| panic!("gnb-{}: {e:?}", i + 1));
        assert_eq!(config.nci, cell.nci, "gnb-{}", i + 1);
        assert_eq!(config.sim_position, file.grid[i].position);
        assert!(
            config.nwdaf_enabled,
            "gnb-{}: the predictors need NWDAF",
            i + 1
        );
        let others: Vec<u64> = cells
            .iter()
            .filter(|c| c.nci != cell.nci)
            .map(|c| c.nci)
            .collect();
        let grid: Vec<u64> = config.paging_grid.iter().map(|e| e.nci).collect();
        assert_eq!(grid, others, "gnb-{}: paging grid", i + 1);
        for e in &config.paging_grid {
            let at = file
                .grid
                .iter()
                .find(|g| g.nci == e.nci)
                .expect("a grid cell");
            assert_eq!(e.position, at.position);
        }
        let neighbours: Vec<u64> = config.handover_neighbours.iter().map(|n| n.nci).collect();
        assert_eq!(neighbours, others, "gnb-{}: handover neighbours", i + 1);
    }
    for &seed in &file.seeds.live {
        let path = format!("{DEMO_DIR}/ue-seed-{seed}.yaml");
        let ue: nextgsim_common::config::UeConfig =
            serde_yaml::from_str(&std::fs::read_to_string(&path).expect("read"))
                .unwrap_or_else(|e| panic!("{path}: {e}"));
        ue.validate_sim_motion()
            .unwrap_or_else(|e| panic!("{path}: {e}"));
        let run = file.run(seed);
        assert_eq!(ue.sim_position, run.start, "seed {seed}");
        let route = ue.sim_route.as_ref().expect("a route");
        assert_eq!(route.waypoints, run.route.waypoints, "seed {seed}");
        assert_eq!(route.speed_mps, run.route.speed_mps, "seed {seed}");
        assert_eq!(ue.gnb_search_list.len(), 6, "seed {seed}");
    }
}
