//! The demos this gNB binary knows (the toggle convention, nextgsim#458).
//!
//! A demo registers its [`DemoSpec`] here **only under its own cargo feature**
//! (`demo-<key>`), so a default build registers nothing: its registry is empty,
//! any `demo.<key>` block in the config fails startup with
//! `DemoConfigError::UnknownKey`, and no demo state is ever active.
//!
//! ```rust,ignore
//! #[cfg(feature = "demo-d441a-congestion-loop")]
//! specs.push(DemoSpec { key: D441A_KEY, modes: D441A_MODES, requires: &[] });
//! ```

use nextgsim_demo::{ActiveDemos, DemoConfigError, DemoRegistry, DemoRuntime, DemoSpec};

/// D446A, load-balancing vs energy-saving agents over a booster cell (nextgsim#483).
pub const D446A_KEY: &str = "d446a-agent-conflict";
/// D446A's modes: which agent wins a `Cell:<booster>` contest.
pub const D446A_MODES: &[&str] = &["load-balancing", "energy-saving"];

/// The handover-triggering demos (#467 D441D, #483 D446A, #486 D446D, #487
/// D446E). At most one of them may be active on one gNB: two would each drive
/// handovers of the same UEs and neither demo's chain would mean anything.
pub const HANDOVER_DEMOS: &[&str] = &[
    D441D_KEY,
    D446A_KEY,
    "d446d-agent-mobility",
    "d446e-agent-actuation",
];

/// D447A, live-state knowledge exposure (nextgsim#488). Registered under the
/// `nkef-sba` cargo feature.
pub const D447A_KEY: &str = "d447a-nkef-live";

/// D447A's modes. `on`: the RLS, N2 and SBA tees feed NKEF and a query answers
/// from them. `baseline`: today's NKEF, tees off, so nothing live is returned.
/// `internal` / `external` are per-query consumers, not modes.
pub const D447A_MODES: &[&str] = &["on", "baseline"];

/// Held by every test that reads or switches D447A's process-global mode, so a
/// test that switches it to `baseline` cannot race one that expects `on`.
#[cfg(all(test, feature = "nkef-sba"))]
pub static D447A_TEST_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

/// D440B, verified OAM-style model loading with serving counters (nextgsim#463).
/// Registered under the `model-registry` cargo feature.
pub const D440B_KEY: &str = "d440b-model-verify";

/// D440B's modes. `verify`: the model is checked and activated only on a pass.
/// `fallback`: no model is activated and every prediction uses the fallback,
/// reproducing the silent-fallback state a consumer's check must catch.
pub const D440B_MODES: &[&str] = &["verify", "fallback"];

/// D440A, model-predicted paging (nextgsim#462). Registered under the
/// `demo-d440a-ai-paging` cargo feature.
pub const D440A_KEY: &str = "d440a-ai-paging";

/// D440A's modes, so far. `page-all` is the predictor-off arm: no
/// recommendation, so the AMF pages the registration area, exactly as with no
/// demo. The predictor arms (`last-served`, `ct`, `lsq`, `onnx`) are registered
/// by the PR that implements them (#462 PR 3b), so no mode is accepted before
/// it does what it names.
pub const D440A_MODES: &[&str] = &["page-all"];

/// D441A, the RAN-fed User Data Congestion loop (nextgsim#465). Registered under
/// the `demo-d441a-congestion-loop` cargo feature.
pub const D441A_KEY: &str = "d441a-congestion-loop";

/// D441A's modes. `statistics`: export the measured cell occupancy every
/// window. `prediction`: the same, plus the 8-sample OLS prediction.
/// `no-loop`: no export (the baseline), the capacity-C queue still in place.
pub const D441A_MODES: &[&str] = &["statistics", "prediction", "no-loop"];

/// D441D, predictive N2 handover from the gNB NWDAF vs A3 (nextgsim#467).
/// Registered under the `demo-d441d-predictive-ho` cargo feature.
pub const D441D_KEY: &str = "d441d-predictive-ho";

/// D441D's modes, all on the same route and shadowing seeds:
/// - `a3`: the baseline. No predictor; the gNB acts on the UE's A3
///   MeasurementReport exactly as with no demo.
/// - `lsq`: the least-squares trajectory predictor triggers the handover; A3 is
///   evaluated in shadow and not acted on.
/// - `onnx`: the same with a d440b-verified ONNX model (needs
///   `d440b-model-verify` in mode `verify`); a prediction no verified model
///   served is labelled `predictor=fallback`.
pub const D441D_MODES: &[&str] = &["a3", "lsq", "onnx"];

/// D440C, NWDAF UE-mobility analytics steer AMF paging (nextgsim#464).
/// Registered under the `demo-d440c-nwdaf-paging` cargo feature.
pub const D440C_KEY: &str = "d440c-nwdaf-paging";

/// D440C's modes, the ones amfd registers for the same key (nextgcore#1469), so
/// one `demo:` block reads the same on every node. On the gNB each mode only
/// turns the PAGING assistance log on: what is sent is amfd's decision.
/// `ran-recommendation` is not registered on amfd until #462's amfd demo lands,
/// so it is not here either (no mode is accepted before it does what it names).
pub const D440C_MODES: &[&str] = &["nwdaf-mobility", "registration-area"];

/// The demos compiled into this binary.
#[allow(clippy::vec_init_then_push)] // one push per demo feature
pub fn registry() -> DemoRegistry {
    #[allow(unused_mut)]
    let mut specs: Vec<DemoSpec> = Vec::new();
    #[cfg(feature = "demo-d446a-agent-conflict")]
    specs.push(DemoSpec {
        key: D446A_KEY,
        modes: D446A_MODES,
        requires: &[],
    });
    #[cfg(feature = "demo-d440a-ai-paging")]
    specs.push(DemoSpec {
        key: D440A_KEY,
        modes: D440A_MODES,
        requires: &[],
    });
    #[cfg(feature = "nkef-sba")]
    specs.push(DemoSpec {
        key: D447A_KEY,
        modes: D447A_MODES,
        requires: &[],
    });
    #[cfg(feature = "model-registry")]
    specs.push(DemoSpec {
        key: D440B_KEY,
        modes: D440B_MODES,
        requires: &[],
    });
    #[cfg(feature = "demo-d441a-congestion-loop")]
    specs.push(DemoSpec {
        key: D441A_KEY,
        modes: D441A_MODES,
        requires: &[],
    });
    // Its `onnx` mode alone needs d440b, which a per-demo `requires` cannot
    // say: [`check_d441d_dependencies`] checks it.
    #[cfg(feature = "demo-d441d-predictive-ho")]
    specs.push(DemoSpec {
        key: D441D_KEY,
        modes: D441D_MODES,
        requires: &[],
    });
    #[cfg(feature = "demo-d440c-nwdaf-paging")]
    specs.push(DemoSpec {
        key: D440C_KEY,
        modes: D440C_MODES,
        requires: &[],
    });
    DemoRegistry::new(&specs)
}

/// A gNB demo block that cannot start.
#[derive(Debug, thiserror::Error)]
pub enum GnbDemoError {
    /// The block itself is invalid (unknown key or mode, malformed).
    #[error(transparent)]
    Config(#[from] DemoConfigError),
    /// Two handover-triggering demos are active on this gNB.
    #[error("HandoverDemoConflict: demos {first} and {second} both trigger handovers; enable one per gNB")]
    HandoverDemoConflict {
        /// One of them.
        first: String,
        /// The other.
        second: String,
    },
    /// D441D is active but the gNB config cannot run it.
    #[error("demo d441d-predictive-ho cannot start: {0}")]
    D441d(String),
}

/// D441D's `onnx` mode needs `d440b-model-verify` active in mode `verify` on
/// the same gNB (nextgsim#467): the model it predicts with must be one the
/// d440b checks verified. Startup fails otherwise.
pub fn check_d441d_dependencies(active: &ActiveDemos) -> Result<(), GnbDemoError> {
    if active.mode(D441D_KEY) == Some("onnx") && active.mode(D440B_KEY) != Some("verify") {
        return Err(GnbDemoError::Config(DemoConfigError::MissingDependency {
            demo: D441D_KEY.to_string(),
            requires: format!("{D440B_KEY} (mode verify)"),
        }));
    }
    Ok(())
}

/// Fails when more than one of [`HANDOVER_DEMOS`] is active.
pub fn check_handover_demo_exclusive(active: &ActiveDemos) -> Result<(), GnbDemoError> {
    let mut on = HANDOVER_DEMOS.iter().filter(|k| active.is_active(k));
    match (on.next(), on.next()) {
        (Some(a), Some(b)) => Err(GnbDemoError::HandoverDemoConflict {
            first: (*a).to_string(),
            second: (*b).to_string(),
        }),
        _ => Ok(()),
    }
}

/// Read and install the `demo:` block of the gNB config text at startup. An
/// error must fail startup: the convention allows no silent fallback.
pub fn init(config_yaml: &str) -> Result<&'static DemoRuntime, GnbDemoError> {
    let root: serde_yaml::Value =
        serde_yaml::from_str(config_yaml).map_err(|e| DemoConfigError::Malformed {
            detail: format!("config is not valid YAML: {e}"),
        })?;
    let runtime = DemoRuntime::from_root(&root, registry())?;
    check_handover_demo_exclusive(&runtime.snapshot())?;
    check_d441d_dependencies(&runtime.snapshot())?;
    nextgsim_demo::install(runtime)
        .map_err(|_| GnbDemoError::Config(DemoConfigError::AlreadyInstalled))
}

/// Whether D440A is active in this process (its feature compiled in, and both
/// gates on). Always false in a build without `demo-d440a-ai-paging`.
pub fn d440a_active() -> bool {
    cfg!(feature = "demo-d440a-ai-paging") && nextgsim_demo::active_mode(D440A_KEY).is_some()
}

/// What D440A reads of the gNB config at startup, logged once when the demo is
/// active (nextgsim#462): the paging grid this gNB resolves recommendations
/// against. Silent when the demo is off, so a default run logs nothing new.
pub fn d440a_startup(config: &nextgsim_common::config::GnbConfig) {
    if !d440a_active() {
        return;
    }
    let cells: Vec<String> = config
        .paging_grid
        .iter()
        .map(|e| {
            format!(
                "gNB {:#x} cell {:#x} at ({}, {}, {})",
                e.gnb_id(config.gnb_id_length),
                e.nci,
                e.position.x,
                e.position.y,
                e.position.z
            )
        })
        .collect();
    nextgsim_demo::demo_log!(
        info,
        D440A_KEY,
        "paging grid: this gNB {:#x} at ({}, {}, {}), {} neighbour cell(s): [{}]",
        config.gnb_id(),
        config.sim_position.x,
        config.sim_position.y,
        config.sim_position.z,
        cells.len(),
        cells.join(", ")
    );
}

/// Whether D441D is active in this process (its feature compiled in, and both
/// gates on). Always false in a build without `demo-d441d-predictive-ho`.
pub fn d441d_active() -> bool {
    cfg!(feature = "demo-d441d-predictive-ho") && nextgsim_demo::active_mode(D441D_KEY).is_some()
}

/// What D441D needs of the gNB config, and the startup line it logs
/// (nextgsim#467): the NWDAF task on, and at least one positioned handover
/// neighbour. `Ok(None)` when the demo is not active.
#[cfg(feature = "demo-d441d-predictive-ho")]
pub fn d441d_check_config(
    config: &nextgsim_common::config::GnbConfig,
    mode: Option<&str>,
) -> Result<Option<String>, GnbDemoError> {
    use crate::nwdaf::d441d;
    let Some(mode) = mode else {
        return Ok(None);
    };
    if !config.nwdaf_enabled {
        return Err(GnbDemoError::D441d(
            "nwdaf_enabled is false: the predictor and the shadow A3 run in the NWDAF task".into(),
        ));
    }
    d441d::Engine::from_config(config).map_err(GnbDemoError::D441d)?;
    let cells: Vec<String> = config
        .handover_neighbours
        .iter()
        .filter_map(|n| {
            n.position
                .map(|p| format!("cell {:#x} at ({}, {}, {})", n.nci, p.x, p.y, p.z))
        })
        .collect();
    let s = config.shadowing;
    Ok(Some(format!(
        "predictive handover mode={mode}: this cell {:#x} at ({}, {}, {}), {} positioned \
         neighbour(s): [{}]; A3 margin a3-Offset {} dB + hysteresis {} dB for {} ms; \
         horizon {} ms, dwell {} ms, at most {} attempt(s) {} ms apart; shadowing sigma {} \
         dB seed {} decorrelation {} m",
        config.nci,
        config.sim_position.x,
        config.sim_position.y,
        config.sim_position.z,
        cells.len(),
        cells.join(", "),
        config.cho_a3_offset_db,
        config.cho_hysteresis_db,
        d441d::A3_TTT_MS,
        d441d::HORIZON_MS,
        d441d::MIN_DWELL_MS,
        d441d::MAX_ATTEMPTS,
        d441d::RETRY_AFTER_MS,
        s.sigma_db,
        s.seed,
        s.decorrelation_m
    )))
}

/// D441D's startup check and line, against the installed demo state. An error
/// fails startup. Silent and `Ok` when the demo is off.
pub fn d441d_startup(config: &nextgsim_common::config::GnbConfig) -> Result<(), GnbDemoError> {
    #[cfg(feature = "demo-d441d-predictive-ho")]
    if let Some(line) =
        d441d_check_config(config, nextgsim_demo::active_mode(D441D_KEY).as_deref())?
    {
        nextgsim_demo::demo_log!(info, D441D_KEY, "{line}");
    }
    #[cfg(not(feature = "demo-d441d-predictive-ho"))]
    let _ = config;
    Ok(())
}

/// D440C's gNB line for a PAGING's Assistance Data for Paging (nextgsim#464):
/// `PAGING assistance: cells=[0x20,0x10] attempt 1/5 next-scope=changed`.
///
/// Cells are NR Cell Identities in the order received (the most recently
/// visited first, TS 38.413 §9.3.1.71), an E-UTRA cell as `eutra:0x..`; `cells=[]`
/// when the IE carries only the Paging Attempt Information. `attempt -` when it
/// carries none, and no `next-scope=` when the AMF signalled none (its last
/// attempt). The PLMN is not printed: amfd lists only cells of its own PLMN.
#[cfg(feature = "demo-d440c-nwdaf-paging")]
pub fn d440c_assistance_line(
    assistance: &nextgsim_ngap::procedures::paging::AssistanceDataForPagingValue,
) -> String {
    use nextgsim_ngap::procedures::paging::{NextPagingAreaScopeValue, RecommendedCellCgi};
    let cells: Vec<String> = assistance
        .recommended_cells
        .iter()
        .map(|c| match &c.cgi {
            RecommendedCellCgi::Nr {
                nr_cell_identity, ..
            } => format!("{nr_cell_identity:#x}"),
            RecommendedCellCgi::Eutra {
                eutra_cell_identity,
                ..
            } => format!("eutra:{eutra_cell_identity:#x}"),
        })
        .collect();
    let attempt = match &assistance.paging_attempt_information {
        None => "attempt -".to_string(),
        Some(pai) => {
            let scope = match pai.next_paging_area_scope {
                None => "",
                Some(NextPagingAreaScopeValue::Same) => " next-scope=same",
                Some(NextPagingAreaScopeValue::Changed) => " next-scope=changed",
            };
            format!(
                "attempt {}/{}{scope}",
                pai.paging_attempt_count, pai.intended_number_of_paging_attempts
            )
        }
    };
    format!("PAGING assistance: cells=[{}] {attempt}", cells.join(","))
}

/// Whether D446A is active in this process (its feature compiled in, and both
/// gates on). Always false in a build without `demo-d446a-agent-conflict`.
pub fn d446a_active() -> bool {
    cfg!(feature = "demo-d446a-agent-conflict") && nextgsim_demo::active_mode(D446A_KEY).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// nextgsim#488: an `nkef-sba` build knows d447a-nkef-live with its two
    /// modes, and an unknown mode fails startup with no fallback.
    #[test]
    #[cfg(feature = "nkef-sba")]
    fn an_nkef_sba_build_registers_d447a_and_rejects_an_unknown_mode() {
        assert!(registry().keys().contains(&D447A_KEY));
        let root: serde_yaml::Value = serde_yaml::from_str(
            "demo:\n  enabled: true\n  d447a-nkef-live:\n    enabled: true\n    mode: bogus\n",
        )
        .unwrap();
        match DemoRuntime::from_root(&root, registry()) {
            Err(DemoConfigError::UnknownMode { demo, mode, .. }) => {
                assert_eq!((demo.as_str(), mode.as_str()), (D447A_KEY, "bogus"));
            }
            other => panic!("expected UnknownMode, got {other:?}"),
        }
        // The kill switch off: nothing is active even with the key enabled.
        let root: serde_yaml::Value = serde_yaml::from_str(
            "demo:\n  enabled: false\n  d447a-nkef-live:\n    enabled: true\n    mode: \"on\"\n",
        )
        .unwrap();
        let rt = DemoRuntime::from_root(&root, registry()).unwrap();
        assert!(rt.snapshot().is_empty());
        assert_eq!(rt.mode(D447A_KEY), None);
    }

    /// A default build knows no demo, so a `demo.<key>` block cannot switch
    /// anything on: it fails startup instead.
    #[cfg(not(any(
        feature = "demo-d441d-predictive-ho",
        feature = "demo-d440a-ai-paging",
        feature = "demo-d446a-agent-conflict",
        feature = "model-registry",
        feature = "nkef-sba",
        feature = "demo-d441a-congestion-loop",
        feature = "demo-d440c-nwdaf-paging"
    )))]
    #[test]
    fn a_default_build_rejects_every_demo_key() {
        assert!(registry().is_empty());
        let root: serde_yaml::Value = serde_yaml::from_str(
            "demo:\n  enabled: true\n  d441a-congestion-loop:\n    enabled: true\n    mode: baseline\n",
        )
        .unwrap();
        assert!(matches!(
            DemoRuntime::from_root(&root, registry()),
            Err(DemoConfigError::UnknownKey { .. })
        ));
        // A config without the block, or with only the kill switch, is unchanged.
        for yaml in ["nci: 16\n", "demo:\n  enabled: false\n"] {
            let root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            let rt = DemoRuntime::from_root(&root, registry()).unwrap();
            assert!(rt.snapshot().is_empty());
        }
    }

    /// #462: a D440A build registers its key with `page-all` only. A predictor
    /// mode is refused until the PR that implements it registers it, and so is
    /// `mode: bogus`; the kill switch leaves the demo inactive.
    #[cfg(feature = "demo-d440a-ai-paging")]
    #[test]
    fn the_d440a_build_registers_page_all_and_no_unimplemented_mode() {
        let parse = |y: &str| serde_yaml::from_str::<serde_yaml::Value>(y).unwrap();
        let block = |on: bool, mode: &str| {
            format!(
                "demo:\n  enabled: {on}\n  d440a-ai-paging:\n    enabled: true\n    mode: {mode}\n"
            )
        };
        let rt = DemoRuntime::from_root(&parse(&block(true, "page-all")), registry()).unwrap();
        assert_eq!(rt.mode(D440A_KEY).as_deref(), Some("page-all"));
        for mode in ["ct", "onnx", "bogus"] {
            let err = DemoRuntime::from_root(&parse(&block(true, mode)), registry()).unwrap_err();
            assert!(
                matches!(&err, DemoConfigError::UnknownMode { demo, mode: m, .. }
                    if demo == D440A_KEY && m == mode),
                "{mode}: {err:?}"
            );
        }
        let off = DemoRuntime::from_root(&parse(&block(false, "page-all")), registry()).unwrap();
        assert_eq!(off.mode(D440A_KEY), None, "the kill switch wins");
    }

    /// #462: the shipped D440A configs' demo blocks are accepted by a D440A
    /// build, in mode page-all.
    #[cfg(feature = "demo-d440a-ai-paging")]
    #[test]
    fn the_d440a_demo_configs_enable_page_all() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/demo-ai-paging");
        for name in ["gnb-west.yaml", "gnb-east.yaml"] {
            let text = std::fs::read_to_string(format!("{dir}/{name}")).expect("read");
            let root: serde_yaml::Value = serde_yaml::from_str(&text).expect("yaml");
            let rt = DemoRuntime::from_root(&root, registry()).expect(name);
            assert_eq!(rt.mode(D440A_KEY).as_deref(), Some("page-all"), "{name}");
        }
    }

    /// #483: the D446A build registers its key and two modes; `mode: bogus`
    /// fails with `UnknownMode`, and the kill switch leaves it inactive.
    #[cfg(feature = "demo-d446a-agent-conflict")]
    #[test]
    fn the_d446a_build_registers_its_key_and_modes() {
        let parse = |y: &str| serde_yaml::from_str::<serde_yaml::Value>(y).unwrap();
        let block = |on: bool, mode: &str| {
            format!("demo:\n  enabled: {on}\n  d446a-agent-conflict:\n    enabled: true\n    mode: {mode}\n")
        };
        for mode in D446A_MODES {
            let rt = DemoRuntime::from_root(&parse(&block(true, mode)), registry()).unwrap();
            assert_eq!(rt.mode(D446A_KEY).as_deref(), Some(*mode));
        }
        let err = DemoRuntime::from_root(&parse(&block(true, "bogus")), registry()).unwrap_err();
        assert!(
            matches!(&err, DemoConfigError::UnknownMode { demo, mode, .. }
                if demo == D446A_KEY && mode == "bogus"),
            "{err:?}"
        );
        let off =
            DemoRuntime::from_root(&parse(&block(false, "load-balancing")), registry()).unwrap();
        assert_eq!(off.mode(D446A_KEY), None, "kill switch");
    }

    /// #483: two handover-triggering demos on one gNB fail startup by name.
    #[test]
    fn two_handover_demos_on_one_gnb_conflict() {
        let reg = DemoRegistry::new(&[
            DemoSpec {
                key: HANDOVER_DEMOS[1],
                modes: &["a"],
                requires: &[],
            },
            DemoSpec {
                key: HANDOVER_DEMOS[2],
                modes: &["a"],
                requires: &[],
            },
        ]);
        let yaml = format!(
            "demo:\n  enabled: true\n  {}:\n    enabled: true\n    mode: a\n  {}:\n    enabled: true\n    mode: a\n",
            HANDOVER_DEMOS[1], HANDOVER_DEMOS[2]
        );
        let rt = DemoRuntime::from_root(&serde_yaml::from_str(&yaml).unwrap(), reg).unwrap();
        let err = check_handover_demo_exclusive(&rt.snapshot()).unwrap_err();
        assert!(err.to_string().starts_with("HandoverDemoConflict"), "{err}");
        // One alone is fine.
        assert!(check_handover_demo_exclusive(
            &DemoRuntime::from_root(&serde_yaml::from_str("nci: 1\n").unwrap(), registry())
                .unwrap()
                .snapshot()
        )
        .is_ok());
    }

    fn d440b(yaml: &str) -> Result<DemoRuntime, DemoConfigError> {
        let root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        DemoRuntime::from_root(&root, registry())
    }

    /// Without `model-registry` the d440b block is an unknown key: a block whose
    /// cargo feature is not built fails startup.
    #[cfg(not(feature = "model-registry"))]
    #[test]
    fn without_model_registry_the_d440b_block_fails_startup() {
        let err = d440b(
            "demo:\n  enabled: true\n  d440b-model-verify:\n    enabled: true\n    mode: verify\n",
        )
        .unwrap_err();
        assert!(
            matches!(&err, DemoConfigError::UnknownKey { key, .. } if key == D440B_KEY),
            "{err}"
        );
    }

    #[cfg(feature = "model-registry")]
    mod d440b {
        use super::*;

        #[test]
        fn each_mode_is_accepted_and_active() {
            for mode in ["verify", "fallback"] {
                let rt = d440b(&format!(
                    "demo:\n  enabled: true\n  d440b-model-verify:\n    enabled: true\n    mode: {mode}\n"
                ))
                .unwrap();
                assert_eq!(rt.mode(D440B_KEY).as_deref(), Some(mode));
                assert_eq!(
                    rt.snapshot().banner().unwrap(),
                    format!("DEMO MODE ACTIVE: d440b-model-verify={mode}")
                );
            }
        }

        /// `mode: bogus` fails startup with the named error.
        #[test]
        fn an_unknown_mode_fails_startup() {
            let err = d440b(
                "demo:\n  enabled: true\n  d440b-model-verify:\n    enabled: true\n    mode: bogus\n",
            )
            .unwrap_err();
            assert_eq!(
                err,
                DemoConfigError::UnknownMode {
                    demo: D440B_KEY.into(),
                    mode: "bogus".into(),
                    known: vec!["verify", "fallback"],
                }
            );
        }

        /// The three demo-off shapes leave nothing active, and `demo show` says
        /// so.
        #[test]
        fn every_demo_off_shape_is_inactive() {
            for yaml in [
                "nci: 16\n",
                "demo:\n  enabled: false\n",
                "demo:\n  enabled: false\n  d440b-model-verify:\n    enabled: true\n    mode: verify\n",
            ] {
                let rt = d440b(yaml).unwrap();
                assert!(rt.snapshot().is_empty(), "{yaml}");
                assert_eq!(rt.mode(D440B_KEY), None);
                let show = rt.show(D440B_KEY).unwrap();
                assert!(show.contains("active=false mode=-"), "{show}");
                let mut metrics = String::new();
                rt.render_metrics(&mut metrics);
                assert!(metrics.is_empty(), "no demo series when off: {metrics}");
            }
        }

        /// `set mode=bogus` is rejected and the mode stays; a valid switch is
        /// applied (and both are audited by the demo crate).
        #[test]
        fn a_runtime_switch_is_checked() {
            let rt = d440b(
                "demo:\n  enabled: true\n  d440b-model-verify:\n    enabled: true\n    mode: verify\n",
            )
            .unwrap();
            let err = rt
                .handle_cli("d440b-model-verify set mode=bogus", "test")
                .unwrap_err();
            assert!(err.to_string().contains("unknown mode \"bogus\""), "{err}");
            assert_eq!(rt.mode(D440B_KEY).as_deref(), Some("verify"));
            rt.handle_cli("d440b-model-verify set mode=fallback", "test")
                .unwrap();
            assert_eq!(rt.mode(D440B_KEY).as_deref(), Some("fallback"));
            let mut metrics = String::new();
            rt.render_metrics(&mut metrics);
            assert!(metrics
                .contains("nextg_demo_enabled{demo=\"d440b-model-verify\",mode=\"fallback\"} 1"));
        }
    }

    fn d441d_block(on: bool, mode: &str) -> String {
        format!(
            "demo:\n  enabled: {on}\n  d441d-predictive-ho:\n    enabled: true\n    mode: {mode}\n"
        )
    }

    /// Without its feature the d441d block is an unknown key.
    #[cfg(not(feature = "demo-d441d-predictive-ho"))]
    #[test]
    fn without_its_feature_the_d441d_block_fails_startup() {
        let root: serde_yaml::Value = serde_yaml::from_str(&d441d_block(true, "lsq")).unwrap();
        let err = DemoRuntime::from_root(&root, registry()).unwrap_err();
        assert!(
            matches!(&err, DemoConfigError::UnknownKey { key, .. } if key == D441D_KEY),
            "{err}"
        );
    }

    /// #467: the handover-demo list names D441D by the ticket's key, so the
    /// conflict check covers it.
    #[test]
    fn the_handover_demos_name_d441d_by_its_key() {
        assert!(HANDOVER_DEMOS.contains(&"d441d-predictive-ho"));
        assert!(!HANDOVER_DEMOS.contains(&"d441d-ue-mobility-handover"));
    }

    /// #467: each config/demo-predictive-ho file is its base config with only
    /// the D441D changes: the NWDAF task on, the neighbour's position, and the
    /// shadowing and demo blocks (the UE: only its motion).
    #[test]
    fn the_d441d_configs_change_only_what_the_demo_needs() {
        use serde_yaml::Value;
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../config");
        let load = |f: &str| -> Value {
            serde_yaml::from_str(&std::fs::read_to_string(format!("{dir}/{f}")).unwrap()).unwrap()
        };
        for (demo, base) in [
            ("gnb-west.yaml", "gnb.yaml"),
            ("gnb-east.yaml", "gnb-n2ho-target.yaml"),
        ] {
            let mut d = load(&format!("demo-predictive-ho/{demo}"));
            let m = d.as_mapping_mut().unwrap();
            assert_eq!(m.get("nwdaf_enabled"), Some(&Value::Bool(true)), "{demo}");
            m.insert("nwdaf_enabled".into(), Value::Bool(false));
            assert!(
                m.remove("demo").is_some() && m.remove("shadowing").is_some(),
                "{demo}"
            );
            let nb = m["handover_neighbours"][0].as_mapping_mut().unwrap();
            assert!(nb.remove("position").is_some(), "{demo}");
            assert_eq!(
                d,
                load(base),
                "{demo} must be {base} plus the D441D changes"
            );
        }
        let mut ue = load("demo-predictive-ho/ue.yaml");
        let mut base = load("ue-n2ho.yaml");
        for key in ["sim_position", "sim_movement", "sim_route"] {
            ue.as_mapping_mut().unwrap().remove(key);
            base.as_mapping_mut().unwrap().remove(key);
        }
        assert_eq!(
            ue, base,
            "the D441D UE differs from ue-n2ho.yaml only in its motion"
        );
    }

    #[cfg(feature = "demo-d441d-predictive-ho")]
    mod d441d {
        use super::*;
        use nextgsim_common::config::GnbConfig;

        fn rt(yaml: &str) -> Result<DemoRuntime, DemoConfigError> {
            DemoRuntime::from_root(&serde_yaml::from_str(yaml).unwrap(), registry())
        }

        /// Each of the three modes is accepted and active, with its banner.
        #[test]
        fn each_mode_is_accepted_and_active() {
            for mode in ["a3", "lsq"] {
                let rt = rt(&d441d_block(true, mode)).unwrap();
                assert_eq!(rt.mode(D441D_KEY).as_deref(), Some(mode));
                assert_eq!(
                    rt.snapshot().banner().unwrap(),
                    format!("DEMO MODE ACTIVE: d441d-predictive-ho={mode}")
                );
                assert!(check_d441d_dependencies(&rt.snapshot()).is_ok());
            }
            let onnx = format!(
                "{}  d440b-model-verify:\n    enabled: true\n    mode: verify\n",
                d441d_block(true, "onnx")
            );
            let rt = rt(&onnx).unwrap();
            assert_eq!(rt.mode(D441D_KEY).as_deref(), Some("onnx"));
            assert!(check_d441d_dependencies(&rt.snapshot()).is_ok());
        }

        /// `mode: bogus` fails startup with the named error.
        #[test]
        fn an_unknown_mode_fails_startup() {
            assert_eq!(
                rt(&d441d_block(true, "bogus")).unwrap_err(),
                DemoConfigError::UnknownMode {
                    demo: D441D_KEY.into(),
                    mode: "bogus".into(),
                    known: D441D_MODES.to_vec(),
                }
            );
        }

        /// `onnx` without d440b in mode `verify` (absent, or in `fallback`)
        /// fails startup with `MissingDependency`.
        #[test]
        fn onnx_needs_d440b_in_verify() {
            for extra in [
                "",
                "  d440b-model-verify:\n    enabled: true\n    mode: fallback\n",
            ] {
                let rt = rt(&format!("{}{extra}", d441d_block(true, "onnx"))).unwrap();
                let err = check_d441d_dependencies(&rt.snapshot()).unwrap_err();
                assert!(
                    matches!(&err, GnbDemoError::Config(DemoConfigError::MissingDependency { demo, requires })
                        if demo == D441D_KEY && requires.starts_with(D440B_KEY)),
                    "{err}"
                );
            }
        }

        /// No block, the kill switch off, and the kill switch off with the key
        /// enabled: nothing active, `demo show` says so, no metric series.
        #[test]
        fn every_demo_off_shape_is_inactive() {
            for yaml in [
                "nci: 16\n".to_string(),
                "demo:\n  enabled: false\n".to_string(),
                d441d_block(false, "lsq"),
            ] {
                let rt = rt(&yaml).unwrap();
                assert!(rt.snapshot().is_empty(), "{yaml}");
                assert_eq!(rt.mode(D441D_KEY), None);
                assert!(rt.show(D441D_KEY).unwrap().contains("active=false mode=-"));
                let mut metrics = String::new();
                rt.render_metrics(&mut metrics);
                assert!(metrics.is_empty(), "{metrics}");
            }
        }

        /// `set mode=bogus` is refused and the mode stays; a valid switch applies
        /// and shows in the metric.
        #[test]
        fn a_runtime_switch_is_checked() {
            let rt = rt(&d441d_block(true, "lsq")).unwrap();
            let err = rt
                .handle_cli("d441d-predictive-ho set mode=bogus", "test")
                .unwrap_err();
            assert!(err.to_string().contains("unknown mode \"bogus\""), "{err}");
            assert_eq!(rt.mode(D441D_KEY).as_deref(), Some("lsq"));
            rt.handle_cli("d441d-predictive-ho set mode=a3", "test")
                .unwrap();
            assert_eq!(rt.mode(D441D_KEY).as_deref(), Some("a3"));
            let mut metrics = String::new();
            rt.render_metrics(&mut metrics);
            assert!(
                metrics.contains("nextg_demo_enabled{demo=\"d441d-predictive-ho\",mode=\"a3\"} 1")
            );
        }

        /// The demo with #483's D446A on one gNB fails startup with the
        /// conflict error (both call `execute_handover`).
        #[cfg(feature = "demo-d446a-agent-conflict")]
        #[test]
        fn d441d_with_d446a_on_one_gnb_conflicts() {
            let yaml = format!(
                "{}  d446a-agent-conflict:\n    enabled: true\n    mode: load-balancing\n",
                d441d_block(true, "lsq")
            );
            let err = check_handover_demo_exclusive(&rt(&yaml).unwrap().snapshot()).unwrap_err();
            assert!(
                matches!(&err, GnbDemoError::HandoverDemoConflict { first, second }
                    if first == D441D_KEY && second == D446A_KEY),
                "{err}"
            );
        }

        fn shipped(name: &str) -> (String, GnbConfig) {
            let path = format!(
                "{}/../config/demo-predictive-ho/{name}",
                env!("CARGO_MANIFEST_DIR")
            );
            let text = std::fs::read_to_string(&path).expect("read");
            let config: GnbConfig = serde_yaml::from_str(&text).expect(name);
            (text, config)
        }

        /// The shipped gNB configs are accepted, in mode lsq, and pass the
        /// startup check, which logs the neighbour each one names.
        #[test]
        fn the_shipped_configs_start_in_lsq() {
            for (name, other) in [("gnb-west.yaml", "0x20"), ("gnb-east.yaml", "0x10")] {
                let (text, config) = shipped(name);
                let rt = DemoRuntime::from_root(&serde_yaml::from_str(&text).unwrap(), registry())
                    .expect(name);
                assert_eq!(rt.mode(D441D_KEY).as_deref(), Some("lsq"), "{name}");
                let line = d441d_check_config(&config, Some("lsq")).unwrap().unwrap();
                assert!(
                    line.contains(&format!("1 positioned neighbour(s): [cell {other} at")),
                    "{line}"
                );
                assert!(config.shadowing.is_enabled(), "{name}");
            }
        }

        /// The startup check refuses a config the demo cannot run, and is silent
        /// with the demo off.
        #[test]
        fn the_startup_check_needs_the_nwdaf_and_a_positioned_neighbour() {
            let (_, config) = shipped("gnb-west.yaml");
            assert!(d441d_check_config(&config, None).unwrap().is_none());
            let mut no_nwdaf = config.clone();
            no_nwdaf.nwdaf_enabled = false;
            let err = d441d_check_config(&no_nwdaf, Some("lsq")).unwrap_err();
            assert!(err.to_string().contains("nwdaf_enabled"), "{err}");
            let mut no_pos = config;
            no_pos.handover_neighbours[0].position = None;
            let err = d441d_check_config(&no_pos, Some("a3")).unwrap_err();
            assert!(err.to_string().contains("position"), "{err}");
        }
    }

    fn d441a(yaml: &str) -> Result<DemoRuntime, DemoConfigError> {
        let root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        DemoRuntime::from_root(&root, registry())
    }

    fn d441a_block(on: bool, mode: &str) -> String {
        format!("demo:\n  enabled: {on}\n  d441a-congestion-loop:\n    enabled: true\n    mode: {mode}\n")
    }

    /// Without the feature the d441a block is an unknown key.
    #[cfg(not(feature = "demo-d441a-congestion-loop"))]
    #[test]
    fn without_its_feature_the_d441a_block_fails_startup() {
        let err = d441a(&d441a_block(true, "statistics")).unwrap_err();
        assert!(
            matches!(&err, DemoConfigError::UnknownKey { key, .. } if key == D441A_KEY),
            "{err}"
        );
    }

    #[cfg(feature = "demo-d441a-congestion-loop")]
    mod d441a {
        use super::*;

        /// nextgsim#465: each of the three modes is accepted and active.
        #[test]
        fn each_mode_is_accepted_and_active() {
            for mode in D441A_MODES {
                let rt = d441a(&d441a_block(true, mode)).unwrap();
                assert_eq!(rt.mode(D441A_KEY).as_deref(), Some(*mode));
                assert_eq!(
                    rt.snapshot().banner().unwrap(),
                    format!("DEMO MODE ACTIVE: d441a-congestion-loop={mode}")
                );
            }
        }

        /// `mode: bogus` fails startup with the named error.
        #[test]
        fn an_unknown_mode_fails_startup() {
            assert_eq!(
                d441a(&d441a_block(true, "bogus")).unwrap_err(),
                DemoConfigError::UnknownMode {
                    demo: D441A_KEY.into(),
                    mode: "bogus".into(),
                    known: D441A_MODES.to_vec(),
                }
            );
        }

        /// No block, the kill switch off, and the kill switch off with the key
        /// enabled: nothing active, no metric series.
        #[test]
        fn every_demo_off_shape_is_inactive() {
            for yaml in [
                "nci: 16\n".to_string(),
                "demo:\n  enabled: false\n".to_string(),
                d441a_block(false, "statistics"),
            ] {
                let rt = d441a(&yaml).unwrap();
                assert!(rt.snapshot().is_empty(), "{yaml}");
                assert_eq!(rt.mode(D441A_KEY), None);
                let mut metrics = String::new();
                rt.render_metrics(&mut metrics);
                assert!(metrics.is_empty(), "{metrics}");
            }
        }

        /// `set mode=bogus` is refused and the mode stays; a valid switch applies.
        #[test]
        fn a_runtime_switch_is_checked() {
            let rt = d441a(&d441a_block(true, "statistics")).unwrap();
            let err = rt
                .handle_cli("d441a-congestion-loop set mode=bogus", "test")
                .unwrap_err();
            assert!(err.to_string().contains("unknown mode \"bogus\""), "{err}");
            assert_eq!(rt.mode(D441A_KEY).as_deref(), Some("statistics"));
            rt.handle_cli("d441a-congestion-loop set mode=prediction", "test")
                .unwrap();
            assert_eq!(rt.mode(D441A_KEY).as_deref(), Some("prediction"));
            let mut metrics = String::new();
            rt.render_metrics(&mut metrics);
            assert!(metrics.contains(
                "nextg_demo_enabled{demo=\"d441a-congestion-loop\",mode=\"prediction\"} 1"
            ));
        }
    }

    fn d440c(yaml: &str) -> Result<DemoRuntime, DemoConfigError> {
        let root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
        DemoRuntime::from_root(&root, registry())
    }

    fn d440c_block(on: bool, mode: &str) -> String {
        format!(
            "demo:\n  enabled: {on}\n  d440c-nwdaf-paging:\n    enabled: true\n    mode: {mode}\n"
        )
    }

    /// nextgsim#464: without its feature the d440c block is an unknown key, so a
    /// gNB built without the demo refuses a config that asks for it.
    #[cfg(not(feature = "demo-d440c-nwdaf-paging"))]
    #[test]
    fn without_its_feature_the_d440c_block_fails_startup() {
        let err = d440c(&d440c_block(true, "nwdaf-mobility")).unwrap_err();
        assert!(
            matches!(&err, DemoConfigError::UnknownKey { key, .. } if key == D440C_KEY),
            "{err}"
        );
    }

    #[cfg(feature = "demo-d440c-nwdaf-paging")]
    mod d440c {
        use super::*;

        /// nextgsim#464: amfd's two modes are accepted and active, with the
        /// banner; `bogus` and the not-yet-implemented `ran-recommendation` fail
        /// startup with `UnknownMode`, as on amfd.
        #[test]
        fn the_amfd_modes_are_accepted_and_no_other() {
            for mode in D440C_MODES {
                let rt = d440c(&d440c_block(true, mode)).unwrap();
                assert_eq!(rt.mode(D440C_KEY).as_deref(), Some(*mode));
                assert_eq!(
                    rt.snapshot().banner().unwrap(),
                    format!("DEMO MODE ACTIVE: d440c-nwdaf-paging={mode}")
                );
            }
            for mode in ["bogus", "ran-recommendation"] {
                assert_eq!(
                    d440c(&d440c_block(true, mode)).unwrap_err(),
                    DemoConfigError::UnknownMode {
                        demo: D440C_KEY.into(),
                        mode: mode.into(),
                        known: D440C_MODES.to_vec(),
                    }
                );
            }
        }

        /// No block, the kill switch off, and the kill switch off with the key
        /// enabled: nothing active, no metric series.
        #[test]
        fn every_demo_off_shape_is_inactive() {
            for yaml in [
                "nci: 16\n".to_string(),
                "demo:\n  enabled: false\n".to_string(),
                d440c_block(false, "nwdaf-mobility"),
            ] {
                let rt = d440c(&yaml).unwrap();
                assert!(rt.snapshot().is_empty(), "{yaml}");
                assert_eq!(rt.mode(D440C_KEY), None);
                let mut metrics = String::new();
                rt.render_metrics(&mut metrics);
                assert!(metrics.is_empty(), "{metrics}");
            }
        }

        /// The shipped config/demo-nwdaf-paging/ gNB configs are accepted, in
        /// mode nwdaf-mobility.
        #[test]
        fn the_d440c_demo_configs_enable_nwdaf_mobility() {
            let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/../config/demo-nwdaf-paging");
            for name in ["gnb-a.yaml", "gnb-b.yaml", "gnb-c.yaml"] {
                let text = std::fs::read_to_string(format!("{dir}/{name}")).expect("read");
                let rt = d440c(&text).expect(name);
                assert_eq!(
                    rt.mode(D440C_KEY).as_deref(),
                    Some("nwdaf-mobility"),
                    "{name}"
                );
            }
        }
    }

    /// nextgsim#464: the D440C line names the cells in received order, the
    /// attempt and the next scope, and says when either part is absent.
    #[cfg(feature = "demo-d440c-nwdaf-paging")]
    #[test]
    fn the_d440c_assistance_line_names_cells_attempt_and_scope() {
        use nextgsim_ngap::procedures::paging::{
            AssistanceDataForPagingValue as A, NextPagingAreaScopeValue as S,
            PagingAttemptInfoValue as P, RecommendedCellCgi as C, RecommendedCellValue as R,
        };
        let nr = |nci| R {
            cgi: C::Nr {
                plmn_identity: [0x99, 0xF9, 0x07],
                nr_cell_identity: nci,
            },
            time_stayed_in_cell: None,
        };
        let pai = |k, n, s| {
            Some(P {
                paging_attempt_count: k,
                intended_number_of_paging_attempts: n,
                next_paging_area_scope: s,
            })
        };
        let line = |a: A| d440c_assistance_line(&a);
        assert_eq!(
            line(A {
                recommended_cells: vec![nr(0x20), nr(0x10)],
                paging_attempt_information: pai(1, 5, Some(S::Changed)),
            }),
            "PAGING assistance: cells=[0x20,0x10] attempt 1/5 next-scope=changed"
        );
        assert_eq!(
            line(A {
                recommended_cells: vec![],
                paging_attempt_information: pai(2, 5, Some(S::Same)),
            }),
            "PAGING assistance: cells=[] attempt 2/5 next-scope=same"
        );
        assert_eq!(
            line(A {
                recommended_cells: vec![R {
                    cgi: C::Eutra {
                        plmn_identity: [0x99, 0xF9, 0x07],
                        eutra_cell_identity: 0xABC,
                    },
                    time_stayed_in_cell: Some(3),
                }],
                paging_attempt_information: None,
            }),
            "PAGING assistance: cells=[eutra:0xabc] attempt -"
        );
        assert_eq!(
            line(A {
                recommended_cells: vec![nr(0x10)],
                paging_attempt_information: pai(5, 5, None),
            }),
            "PAGING assistance: cells=[0x10] attempt 5/5"
        );
    }
}
