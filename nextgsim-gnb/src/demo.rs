//! The demos this gNB binary knows (the toggle convention, nextgsim#458).
//!
//! A demo registers its [`DemoSpec`] here **only under its own cargo feature**
//! (`demo-<key>`), so a default build registers nothing: its registry is empty,
//! any `demo.<key>` block in the config fails startup with
//! `DemoConfigError::UnknownKey`, and no demo state is ever active.
//!
//! ```rust,ignore
//! #[cfg(feature = "demo-d441a-congestion-loop")]
//! specs.push(DemoSpec { key: "d441a-congestion-loop", modes: &["baseline", "closed-loop"], requires: &[] });
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
    "d441d-ue-mobility-handover",
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
    nextgsim_demo::install(runtime)
        .map_err(|_| GnbDemoError::Config(DemoConfigError::AlreadyInstalled))
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
        feature = "demo-d446a-agent-conflict",
        feature = "model-registry",
        feature = "nkef-sba"
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
}
