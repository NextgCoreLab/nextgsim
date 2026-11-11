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

use nextgsim_demo::{DemoConfigError, DemoRegistry, DemoRuntime, DemoSpec};

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

/// The demos compiled into this binary.
#[allow(clippy::vec_init_then_push)]
pub fn registry() -> DemoRegistry {
    #[allow(unused_mut)]
    let mut specs: Vec<DemoSpec> = Vec::new();
    #[cfg(feature = "nkef-sba")]
    specs.push(DemoSpec {
        key: D447A_KEY,
        modes: D447A_MODES,
        requires: &[],
    });
    DemoRegistry::new(&specs)
}

/// Read and install the `demo:` block of the gNB config text at startup. An
/// error must fail startup: the convention allows no silent fallback.
pub fn init(config_yaml: &str) -> Result<&'static DemoRuntime, DemoConfigError> {
    nextgsim_demo::init_from_yaml(config_yaml, registry())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// nextgsim#488: an `nkef-sba` build knows d447a-nkef-live with its two
    /// modes, and an unknown mode fails startup with no fallback.
    #[test]
    #[cfg(feature = "nkef-sba")]
    fn an_nkef_sba_build_registers_d447a_and_rejects_an_unknown_mode() {
        assert_eq!(registry().keys(), vec![D447A_KEY]);
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
    #[test]
    #[cfg(not(feature = "nkef-sba"))]
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
}
