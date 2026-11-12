//! The demos this UE binary knows (the toggle convention, nextgsim#458).
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

use nextgsim_demo::{DemoConfigError, DemoRegistry, DemoRuntime, DemoSpec};

/// The demos compiled into this binary.
#[allow(clippy::vec_init_then_push)] // one push per demo feature
pub fn registry() -> DemoRegistry {
    #[allow(unused_mut)]
    let mut specs: Vec<DemoSpec> = Vec::new();
    // D444A, app adaptation to a network-signalled Session-AMBR (nextgsim#477).
    #[cfg(feature = "demo-d444a-ambr-adapt")]
    specs.push(DemoSpec {
        key: nextgsim_semantic::rate_adapt::D444A_KEY,
        modes: nextgsim_semantic::rate_adapt::D444A_MODES,
        requires: &[],
    });
    // D445A, network-assisted federated learning over PDU sessions (nextgsim#480).
    #[cfg(feature = "demo-d445a-fl-t8")]
    specs.push(DemoSpec {
        key: nextgsim_fl::d445a::KEY,
        modes: nextgsim_fl::d445a::MODES,
        requires: &[],
    });
    DemoRegistry::new(&specs)
}

/// Read and install the `demo:` block of the UE config text at startup. An
/// error must fail startup: the convention allows no silent fallback.
pub fn init(config_yaml: &str) -> Result<&'static DemoRuntime, DemoConfigError> {
    nextgsim_demo::init_from_yaml(config_yaml, registry())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A default build knows no demo, so a `demo.<key>` block cannot switch
    /// anything on: it fails startup instead.
    #[test]
    #[cfg(not(any(feature = "demo-d444a-ambr-adapt", feature = "demo-d445a-fl-t8")))]
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

    /// nextgsim#477: a `demo-d444a-ambr-adapt` build knows d444a-ambr-adapt with
    /// modes `semantic` and `raw`; `mode: bogus` fails startup with no fallback.
    #[test]
    #[cfg(feature = "demo-d444a-ambr-adapt")]
    fn a_d444a_build_registers_its_two_modes_and_rejects_an_unknown_one() {
        let parse = |mode: &str| {
            let root: serde_yaml::Value = serde_yaml::from_str(&format!(
                "demo:\n  enabled: true\n  d444a-ambr-adapt:\n    enabled: true\n    mode: {mode}\n"
            ))
            .unwrap();
            DemoRuntime::from_root(&root, registry())
        };
        for mode in ["semantic", "raw"] {
            assert_eq!(
                parse(mode).unwrap().mode("d444a-ambr-adapt").as_deref(),
                Some(mode)
            );
        }
        match parse("bogus") {
            Err(DemoConfigError::UnknownMode { demo, mode, .. }) => {
                assert_eq!(
                    (demo.as_str(), mode.as_str()),
                    ("d444a-ambr-adapt", "bogus")
                );
            }
            other => panic!("expected UnknownMode, got {other:?}"),
        }
        // Off: no block, or the kill switch off, activates nothing.
        for yaml in [
            "nci: 16\n",
            "demo:\n  enabled: false\n  d444a-ambr-adapt:\n    enabled: true\n    mode: semantic\n",
        ] {
            let root: serde_yaml::Value = serde_yaml::from_str(yaml).unwrap();
            assert!(DemoRuntime::from_root(&root, registry())
                .unwrap()
                .snapshot()
                .is_empty());
        }
    }

    /// nextgsim#480: a `demo-d445a-fl-t8` build knows d445a-fl-t8 with modes
    /// `t8-exclusion` and `poisoned-ue`; `mode: bogus` fails startup with
    /// `UnknownMode { demo: "d445a-fl-t8", mode: "bogus" }`, and the kill switch
    /// off activates nothing.
    #[test]
    #[cfg(feature = "demo-d445a-fl-t8")]
    fn a_d445a_build_registers_its_two_modes_and_rejects_an_unknown_one() {
        let parse = |kill: bool, on: bool, mode: &str| {
            let root: serde_yaml::Value = serde_yaml::from_str(&format!(
                "demo:\n  enabled: {kill}\n  d445a-fl-t8:\n    enabled: {on}\n    mode: {mode}\n"
            ))
            .unwrap();
            DemoRuntime::from_root(&root, registry())
        };
        for mode in ["t8-exclusion", "poisoned-ue"] {
            assert_eq!(
                parse(true, true, mode)
                    .unwrap()
                    .mode("d445a-fl-t8")
                    .as_deref(),
                Some(mode)
            );
        }
        match parse(true, true, "bogus") {
            Err(DemoConfigError::UnknownMode { demo, mode, .. }) => {
                assert_eq!((demo.as_str(), mode.as_str()), ("d445a-fl-t8", "bogus"));
            }
            other => panic!("expected UnknownMode, got {other:?}"),
        }
        assert!(parse(false, true, "t8-exclusion")
            .unwrap()
            .snapshot()
            .is_empty());
        assert!(parse(true, false, "t8-exclusion")
            .unwrap()
            .snapshot()
            .is_empty());
    }
}
